"""
High-Performance Numba-accelerated Vectorized Fast Backtest Engine (S1 and S2 Funnel Stages).
Features pessimistic intra-candle SL/TP resolution, causal quote-to-USD conversion,
exact spread and slippage cost multipliers (1.5x fitness), and zero-copy memory layouts.
Returns authentic trade durations, pnls, directions, and equity curves.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


# Module-level cache for auxiliary NumPy vectors: (opens, highs, lows, closes, spreads_price, atrs_price, sma_20, ny_hours, bpd)
_AUX_VECTORS_CACHE: dict[tuple[str, int, float, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}


def clear_vector_cache() -> None:
    """Clears the auxiliary vectors cache."""
    _AUX_VECTORS_CACHE.clear()


@njit(fastmath=True)
def run_fast_backtest_kernel(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    spreads_price: np.ndarray,
    atrs_price: np.ndarray,
    quote_to_usd: np.ndarray,
    entry_longs: np.ndarray,
    entry_shorts: np.ndarray,
    sl_atr_mult: float,
    tp_atr_mult: float,
    max_holding_bars: int,
    cost_multiplier: float,
    account_equity: float,
    vol_target: float,
    commission_per_lot: float,
    exit_type_code: int = 0,
    trailing_atr: float = 1.2,
    sma_20: np.ndarray | None = None,
    ny_hours: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates trades with position management using Numba.
    Supports 4 exit archetypes (exit_type_code):
    0: sl_tp
    1: breakeven_trail
    2: dynamic_mean_reversion
    3: pre_rollover_flat
    """
    n_bars = len(closes)
    max_trades = n_bars
    trade_pnls = np.zeros(max_trades, dtype=np.float64)
    trade_durations = np.zeros(max_trades, dtype=np.int64)
    trade_directions = np.zeros(max_trades, dtype=np.int32)
    trade_entry_bars = np.zeros(max_trades, dtype=np.int64)
    trade_exit_bars = np.zeros(max_trades, dtype=np.int64)
    trade_entry_prices = np.zeros(max_trades, dtype=np.float64)
    trade_exit_prices = np.zeros(max_trades, dtype=np.float64)
    trade_exit_reasons = np.zeros(max_trades, dtype=np.int8)  # 0: SL, 1: TP, 2: TIME, 3: ROLLOVER
    equity_curve = np.zeros(n_bars, dtype=np.float64)

    current_equity = account_equity
    equity_curve[0] = current_equity

    in_pos = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    active_sl_distance = 0.0
    entry_bar = 0
    trade_count = 0
    pos_units = 0.0

    for i in range(1, n_bars):
        bar_open = opens[i]
        bar_high = highs[i]
        bar_low = lows[i]
        # Strictly causal ATR known at the start of bar i
        bar_atr = atrs_price[i - 1]
        exited_this_bar = False

        # 1. Manage active position
        if in_pos != 0:
            holding_bars = i - entry_bar
            half_spread = 0.5 * spreads_price[i] * cost_multiplier
            q_usd = quote_to_usd[i]

            sl_hit = False
            tp_hit = False
            rollover_hit = False
            exit_price = 0.0
            exit_reason_code = 0  # 0: SL, 1: TP, 2: TIME, 3: ROLLOVER

            # Archetype 3: pre_rollover_flat (check at 16:00 NY)
            if exit_type_code == 3 and ny_hours is not None:
                if ny_hours[i] == 16:
                    rollover_hit = True

            if in_pos == 1:
                # Long
                # Open-bar exits (TIME expiry or pre-rollover flat) happen immediately at bar_open
                # without intra-candle high/low lookahead
                if rollover_hit or holding_bars >= max_holding_bars:
                    exit_price = bar_open - half_spread
                    exit_reason_code = 0 if bar_open <= sl_price else (3 if rollover_hit else 2)
                    in_pos = 0
                    exited_this_bar = True
                else:
                    # Causal SL check first (prevents intra-candle peak ratcheting from stopping out on the same bar)
                    if bar_low <= sl_price:
                        sl_hit = True

                    # Archetype 1: breakeven_trail (ratchet up if SL not hit)
                    if not sl_hit and exit_type_code == 1:
                        gain = bar_high - entry_price
                        if gain >= 1.0 * active_sl_distance:
                            sl_price = max(sl_price, entry_price + half_spread)
                        if gain >= 1.5 * active_sl_distance:
                            sl_price = max(sl_price, bar_high - trailing_atr * bar_atr)

                    # Archetype 2: dynamic_mean_reversion (uses strictly causal sma_20[i - 1])
                    if exit_type_code == 2 and sma_20 is not None:
                        sma_val = sma_20[i - 1]
                        if bar_high >= sma_val > entry_price:
                            tp_hit = True
                            tp_price = max(bar_open, sma_val)

                    if not tp_hit and bar_high >= tp_price:
                        tp_hit = True

                    # Pessimistic resolution: SL hit wins if both hit
                    if sl_hit and tp_hit or sl_hit:
                        exit_price = min(sl_price, bar_open) - half_spread
                        exit_reason_code = 0
                        in_pos = 0
                        exited_this_bar = True
                    elif tp_hit:
                        exit_price = tp_price - half_spread
                        exit_reason_code = 1
                        in_pos = 0
                        exited_this_bar = True

                if in_pos == 0:
                    pnl_price = exit_price - entry_price
                    commission_usd = (pos_units / 100000.0) * commission_per_lot * cost_multiplier
                    realized_pnl = (pnl_price * pos_units * q_usd) - commission_usd

                    trade_pnls[trade_count] = realized_pnl
                    trade_durations[trade_count] = holding_bars
                    trade_directions[trade_count] = 1
                    trade_entry_bars[trade_count] = entry_bar
                    trade_exit_bars[trade_count] = i
                    trade_entry_prices[trade_count] = entry_price
                    trade_exit_prices[trade_count] = exit_price
                    trade_exit_reasons[trade_count] = exit_reason_code
                    trade_count += 1
                    current_equity += realized_pnl

            elif in_pos == -1:
                # Short
                # Open-bar exits (TIME expiry or pre-rollover flat) happen immediately at bar_open
                # without intra-candle high/low lookahead
                if rollover_hit or holding_bars >= max_holding_bars:
                    exit_price = bar_open + half_spread
                    exit_reason_code = 0 if bar_open >= sl_price else (3 if rollover_hit else 2)
                    in_pos = 0
                    exited_this_bar = True
                else:
                    # Causal SL check first
                    if bar_high >= sl_price:
                        sl_hit = True

                    # Archetype 1: breakeven_trail (ratchet down if SL not hit)
                    if not sl_hit and exit_type_code == 1:
                        gain = entry_price - bar_low
                        if gain >= 1.0 * active_sl_distance:
                            sl_price = min(sl_price, entry_price - half_spread)
                        if gain >= 1.5 * active_sl_distance:
                            sl_price = min(sl_price, bar_low + trailing_atr * bar_atr)

                    # Archetype 2: dynamic_mean_reversion (uses strictly causal sma_20[i - 1])
                    if exit_type_code == 2 and sma_20 is not None:
                        sma_val = sma_20[i - 1]
                        if bar_low <= sma_val < entry_price:
                            tp_hit = True
                            tp_price = min(bar_open, sma_val)

                    if not tp_hit and bar_low <= tp_price:
                        tp_hit = True

                    # Pessimistic resolution
                    if sl_hit and tp_hit or sl_hit:
                        exit_price = max(sl_price, bar_open) + half_spread
                        exit_reason_code = 0
                        in_pos = 0
                        exited_this_bar = True
                    elif tp_hit:
                        exit_price = tp_price + half_spread
                        exit_reason_code = 1
                        in_pos = 0
                        exited_this_bar = True

                if in_pos == 0:
                    pnl_price = entry_price - exit_price
                    commission_usd = (pos_units / 100000.0) * commission_per_lot * cost_multiplier
                    realized_pnl = (pnl_price * pos_units * q_usd) - commission_usd

                    trade_pnls[trade_count] = realized_pnl
                    trade_durations[trade_count] = holding_bars
                    trade_directions[trade_count] = -1
                    trade_entry_bars[trade_count] = entry_bar
                    trade_exit_bars[trade_count] = i
                    trade_entry_prices[trade_count] = entry_price
                    trade_exit_prices[trade_count] = exit_price
                    trade_exit_reasons[trade_count] = exit_reason_code
                    trade_count += 1
                    current_equity += realized_pnl

        # 2. Check for new entry if flat AND did not exit during this bar (signal generated at bar i-1)
        if in_pos == 0 and not exited_this_bar and i < n_bars - 1:
            sig_idx = i - 1
            long_sig = entry_longs[sig_idx]
            short_sig = entry_shorts[sig_idx]

            # Rollover / spread protection check
            cur_spread = spreads_price[i]
            spread_ok = True
            if ny_hours is not None and (ny_hours[i] == 17 or (exit_type_code == 3 and ny_hours[i] == 16)):
                spread_ok = False
            elif bar_atr > 0 and cur_spread > (0.25 * bar_atr):
                spread_ok = False

            if spread_ok:
                half_spread = 0.5 * spreads_price[i] * cost_multiplier
                q_usd = quote_to_usd[i]

                # Position sizing: 1% risk baseline scaled with vol_target
                risk_pct = 0.01
                if vol_target > 0.0:
                    risk_pct = min(0.025, max(0.0025, 0.01 * (vol_target / 0.10)))
                risk_cash = current_equity * risk_pct
                sl_distance = max(sl_atr_mult * bar_atr, half_spread * 2.0)
                active_sl_distance = sl_distance
                sl_risk_per_unit = sl_distance * q_usd
                base_unit_usd = max(bar_open * q_usd, 1e-4)
                if sl_risk_per_unit > 0:
                    pos_units = min(risk_cash / sl_risk_per_unit, (current_equity * 10.0) / base_unit_usd)
                else:
                    pos_units = 10000.0

                if long_sig and not short_sig:
                    in_pos = 1
                    entry_price = opens[i] + half_spread
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + (tp_atr_mult * bar_atr)
                    entry_bar = i
                elif short_sig and not long_sig:
                    in_pos = -1
                    entry_price = opens[i] - half_spread
                    sl_price = entry_price + sl_distance
                    tp_price = entry_price - (tp_atr_mult * bar_atr)
                    entry_bar = i

        # 3. Mark-to-market equity curve
        if in_pos == 1:
            unrealized = (closes[i] - entry_price) * pos_units * quote_to_usd[i]
            equity_curve[i] = current_equity + unrealized
        elif in_pos == -1:
            unrealized = (entry_price - closes[i]) * pos_units * quote_to_usd[i]
            equity_curve[i] = current_equity + unrealized
        else:
            equity_curve[i] = current_equity

    return (
        trade_pnls[:trade_count],
        trade_durations[:trade_count],
        trade_directions[:trade_count],
        equity_curve,
        trade_count,
        trade_entry_bars[:trade_count],
        trade_exit_bars[:trade_count],
        trade_entry_prices[:trade_count],
        trade_exit_prices[:trade_count],
        trade_exit_reasons[:trade_count],
    )


def calculate_performance_metrics(
    trade_pnls: np.ndarray,
    equity_curve: np.ndarray,
    initial_equity: float = 100000.0,
    bars_per_day: int = 24,
) -> dict[str, float]:
    """
    Computes standard institutional performance metrics from realized trades and equity curve.
    """
    n_trades = len(trade_pnls)
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "total_return_pct": 0.0,
            "max_dd_pct": 0.0,
            "net_pnl_usd": 0.0,
        }

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    win_rate = float(len(wins) / n_trades)

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Standard Annualized Sharpe from daily returns
    eq = equity_curve[equity_curve > 0]
    if len(eq) > 1:
        bar_returns = np.diff(eq) / eq[:-1]
        n_bars = len(bar_returns)
        bpd = max(1, int(bars_per_day))
        if bpd > 1 and n_bars >= bpd:
            indices = np.arange(0, n_bars, bpd)
            daily_returns = np.add.reduceat(bar_returns, indices)
            mean_daily = float(np.mean(daily_returns))
            std_daily = float(np.std(daily_returns))
            daily_sharpe = (mean_daily / (std_daily + 1e-9)) * math.sqrt(252)
        else:
            mean_ret = float(np.mean(bar_returns))
            std_ret = float(np.std(bar_returns))
            annual_factor = math.sqrt(252 * bpd)
            daily_sharpe = (mean_ret / (std_ret + 1e-9)) * annual_factor
    else:
        daily_sharpe = 0.0

    # Transactional Sharpe from trade PnLs (retained for reference)
    trade_std = float(np.std(trade_pnls))
    trade_mean = float(np.mean(trade_pnls))
    scale_factor = math.sqrt(max(10, min(n_trades, 100)))
    trade_sharpe = (trade_mean / (trade_std + 1e-9)) * scale_factor

    # Standard Annualized Daily Sharpe (bounded between -5.0 and +10.0)
    sharpe = float(np.clip(daily_sharpe, -5.0, 10.0))

    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / np.where(peaks > 0, peaks, 1.0)
    max_dd_pct = float(np.max(drawdowns)) * 100.0

    net_pnl_usd = float(np.sum(trade_pnls))
    total_return_pct = float((equity_curve[-1] - initial_equity) / initial_equity) * 100.0

    return {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "net_pnl_usd": round(net_pnl_usd, 2),
    }


def run_vectorized_backtest(
    df: pd.DataFrame,
    pair: str,
    entry_longs: np.ndarray,
    entry_shorts: np.ndarray,
    quote_to_usd: np.ndarray | None = None,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    max_holding_bars: int = 24,
    cost_multiplier: float = 1.5,
    account_equity: float = 100000.0,
    vol_target: float = 0.10,
    commission_per_lot: float = 7.0,
    exit_type: str = "sl_tp",
    trailing_atr: float = 1.2,
) -> dict[str, Any]:
    """
    High-level entry point for vectorized backtesting with 4 exit archetypes.
    Uses high-performance cached NumPy auxiliary arrays to eliminate Pandas overhead.
    """
    n_bars = len(df)
    c0 = float(df["close"].iloc[0]) if n_bars > 0 else 0.0
    c_end = float(df["close"].iloc[-1]) if n_bars > 0 else 0.0
    sum_c = round(float(np.sum(df["close"].values)), 4) if n_bars > 0 else 0.0
    sum_h = round(float(np.sum(df["high"].values)), 4) if n_bars > 0 else 0.0
    cache_key = (str(pair), n_bars, c0, c_end, sum_c, sum_h)

    cached = _AUX_VECTORS_CACHE.get(cache_key)
    if cached is not None:
        (opens, highs, lows, closes, spreads_price, atrs_price, sma_20, ny_hours, bpd) = cached
    else:
        opens = np.ascontiguousarray(df["open"].values, dtype=np.float64)
        highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
        lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
        closes = np.ascontiguousarray(df["close"].values, dtype=np.float64)

        pip_unit = 0.01 if "JPY" in pair else 0.0001
        spreads_pips = df["spread"].values.astype(np.float64) if "spread" in df.columns else np.full(len(closes), 1.5)
        spreads_price = np.ascontiguousarray(spreads_pips * pip_unit, dtype=np.float64)

        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        atrs_price = np.ascontiguousarray(pd.Series(tr).rolling(20, min_periods=1).mean().values, dtype=np.float64)

        sma_20 = np.ascontiguousarray(pd.Series(closes).rolling(20, min_periods=1).mean().values, dtype=np.float64)

        bpd = 24
        if "time" in df.columns and n_bars > 1:
            times = pd.to_datetime(df["time"], utc=True)
            med_sec = float(times.diff().dt.total_seconds().median())
            if med_sec > 0:
                bpd = max(1, int(round(86400.0 / med_sec)))
            if med_sec < 20.0 * 3600.0:
                ny_hours = np.ascontiguousarray(times.dt.tz_convert("America/New_York").dt.hour.values, dtype=np.int32)
                rollover_mask = (ny_hours == 17)
                diurnal_spread_mult = np.where(rollover_mask, 3.5, 1.0)
                spreads_price = np.ascontiguousarray(spreads_price * diurnal_spread_mult, dtype=np.float64)
            else:
                ny_hours = np.zeros(n_bars, dtype=np.int32)
        else:
            ny_hours = np.zeros(n_bars, dtype=np.int32)

        cached = (opens, highs, lows, closes, spreads_price, atrs_price, sma_20, ny_hours, bpd)
        _AUX_VECTORS_CACHE[cache_key] = cached

    if quote_to_usd is None:
        quote_to_usd = np.ones(len(closes), dtype=np.float64)

    # 4 exit archetypes mapping
    exit_type_map = {
        "sl_tp": 0,
        "breakeven_trail": 1,
        "dynamic_mean_reversion": 2,
        "pre_rollover_flat": 3,
    }
    exit_type_code = exit_type_map.get(str(exit_type).lower(), 0)

    (
        pnls,
        durations,
        directions,
        equity_curve,
        _n_trades,
        entry_bars,
        exit_bars,
        entry_prices,
        exit_prices,
        exit_reasons,
    ) = run_fast_backtest_kernel(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        spreads_price=spreads_price,
        atrs_price=atrs_price,
        quote_to_usd=quote_to_usd,
        entry_longs=entry_longs.astype(np.bool_),
        entry_shorts=entry_shorts.astype(np.bool_),
        sl_atr_mult=float(sl_atr_mult),
        tp_atr_mult=float(tp_atr_mult),
        max_holding_bars=int(max_holding_bars),
        cost_multiplier=float(cost_multiplier),
        account_equity=float(account_equity),
        vol_target=float(vol_target),
        commission_per_lot=float(commission_per_lot),
        exit_type_code=int(exit_type_code),
        trailing_atr=float(trailing_atr),
        sma_20=sma_20,
        ny_hours=ny_hours,
    )

    metrics = calculate_performance_metrics(pnls, equity_curve, initial_equity=account_equity, bars_per_day=bpd)
    metrics["holding_bars_median"] = float(np.median(durations)) if len(durations) > 0 else 0.0
    metrics["durations"] = durations
    metrics["trade_pnls"] = pnls
    metrics["pnls"] = pnls
    metrics["trade_directions"] = directions
    metrics["equity_curve"] = equity_curve
    metrics["entry_bars"] = entry_bars
    metrics["exit_bars"] = exit_bars
    metrics["entry_prices"] = entry_prices
    metrics["exit_prices"] = exit_prices
    metrics["exit_reasons"] = exit_reasons

    return metrics
