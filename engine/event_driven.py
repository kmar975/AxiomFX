"""
Institutional S3 Event-Driven Verification Engine.
Simulates order-by-order execution, generates full trade logs (trades.parquet),
calculates MAE/MFE, Wednesday 3x swap, slippage, rollover protection,
and verifies agreement with vectorized fast engine (compare_engines).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.costs import (
    get_diurnal_spread_multiplier,
    get_pair_specs,
    get_quote_to_usd_rate,
    is_market_entry_allowed,
    is_rollover_hour,
    resolve_intracandle_conflict,
)
from engine.vectorized import run_vectorized_backtest


class TradeRecord:
    def __init__(
        self,
        trade_id: int,
        pair: str,
        direction: int,  # 1 for long, -1 for short
        entry_idx: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        pos_units: float,
        sl_price: float,
        tp_price: float,
        quote_to_usd: float,
    ):
        self.trade_id = trade_id
        self.pair = pair
        self.direction = direction
        self.entry_idx = entry_idx
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.pos_units = pos_units
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.quote_to_usd_entry = quote_to_usd

        self.exit_idx: int = -1
        self.exit_time: pd.Timestamp | None = None
        self.exit_price: float = 0.0
        self.exit_reason: str = ""
        self.pnl_pips: float = 0.0
        self.pnl_usd: float = 0.0
        self.holding_bars: int = 0
        self.mae_pips: float = 0.0
        self.mfe_pips: float = 0.0
        self.swap_usd: float = 0.0
        self.commission_usd: float = 0.0


class EventDrivenEngine:
    def __init__(
        self,
        account_equity: float = 100000.0,
        cost_multiplier: float = 1.5,
        commission_per_lot_usd: float = 7.0,
        swap_long_annual_pct: float = -0.015,
        swap_short_annual_pct: float = -0.005,
    ):
        self.initial_equity = account_equity
        self.equity = account_equity
        self.cost_multiplier = cost_multiplier
        self.commission_per_lot = commission_per_lot_usd
        self.swap_long_pct = swap_long_annual_pct
        self.swap_short_pct = swap_short_annual_pct

    def run_backtest(
        self,
        df: pd.DataFrame,
        pair: str,
        entry_longs: np.ndarray,
        entry_shorts: np.ndarray,
        market_closes: dict[str, np.ndarray],
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        max_holding_bars: int = 24,
        vol_target: float = 0.10,
        exit_type: str = "sl_tp",
        trailing_atr: float = 1.2,
        sma_20: np.ndarray | None = None,
        ny_hours: np.ndarray | None = None,
    ) -> tuple[list[TradeRecord], pd.DataFrame, dict[str, Any]]:
        self.equity = self.initial_equity
        spec = get_pair_specs(pair)
        pip_unit = float(spec["pip_unit"])
        pip_mult = float(spec["pip_mult"])

        n_bars = len(df)
        times = pd.to_datetime(df["time"], utc=True)
        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        spreads_pips = df["spread"].values.astype(np.float64) if "spread" in df.columns else np.full(n_bars, 1.5)
        spreads_price = spreads_pips * pip_unit

        # Precompute ATR20
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        atrs_price = pd.Series(tr).rolling(20, min_periods=1).mean().values.astype(np.float64)

        if sma_20 is None:
            sma_20 = pd.Series(closes).rolling(20, min_periods=1).mean().values.astype(np.float64)

        is_intraday = True
        if len(times) > 1:
            med_sec = float(times.diff().dt.total_seconds().median())
            is_intraday = (med_sec < 20.0 * 3600.0)

        if ny_hours is None:
            if is_intraday:
                try:
                    ny_hours = times.dt.tz_convert("America/New_York").dt.hour.values.astype(np.int32)
                except Exception:
                    ny_hours = np.zeros(n_bars, dtype=np.int32)
            else:
                ny_hours = np.zeros(n_bars, dtype=np.int32)

        trades: list[TradeRecord] = []
        active_trade: TradeRecord | None = None
        active_sl_distance = 0.0
        equity_history = np.zeros(n_bars, dtype=np.float64)
        equity_history[0] = self.equity

        for i in range(1, n_bars):
            cur_time = times.iloc[i]
            cur_open = opens[i]
            cur_high = highs[i]
            cur_low = lows[i]
            cur_close = closes[i]
            is_rollover = is_rollover_hour(cur_time) if is_intraday else False
            diurnal_mult = get_diurnal_spread_multiplier(cur_time) if is_intraday else 1.0
            cur_spread = spreads_price[i] * diurnal_mult
            half_spread = 0.5 * cur_spread * self.cost_multiplier
            q_usd = get_quote_to_usd_rate(pair, i, market_closes)
            bar_atr = atrs_price[i - 1]
            exited_this_bar = False

            # 1. Manage active trade
            if active_trade is not None:
                holding_bars = i - active_trade.entry_idx
                is_long = (active_trade.direction == 1)

                # Track MAE / MFE
                if is_long:
                    fav_price = cur_high - active_trade.entry_price
                    adv_price = active_trade.entry_price - cur_low
                else:
                    fav_price = active_trade.entry_price - cur_low
                    adv_price = cur_high - active_trade.entry_price

                active_trade.mfe_pips = max(active_trade.mfe_pips, max(0.0, fav_price * pip_mult))
                active_trade.mae_pips = max(active_trade.mae_pips, max(0.0, adv_price * pip_mult))

                # Wednesday swap check (3x rollover at 17:00 NY / 22:00 UTC)
                if cur_time.dayofweek == 2 and cur_time.hour in (21, 22):
                    swap_rate = (self.swap_long_pct if is_long else self.swap_short_pct) / 365.0 * 3.0
                    active_trade.swap_usd += (active_trade.pos_units * cur_open * q_usd) * swap_rate

                # Archetype 3: pre_rollover_flat
                rollover_hit = False
                if exit_type == "pre_rollover_flat" and ny_hours is not None:
                    if ny_hours[i] == 16:
                        rollover_hit = True

                exit_price = 0.0
                exit_reason = ""

                # Open-bar exits (TIME expiry or pre-rollover flat) happen immediately at cur_open
                # without intra-candle high/low lookahead
                if rollover_hit or holding_bars >= max_holding_bars:
                    exit_reason = "ROLLOVER" if rollover_hit else "TIME"
                    if is_long:
                        exit_price = cur_open - half_spread
                        if cur_open <= active_trade.sl_price:
                            exit_reason = "SL"
                    else:
                        exit_price = cur_open + half_spread
                        if cur_open >= active_trade.sl_price:
                            exit_reason = "SL"
                else:
                    # Causal SL check first (prevents intra-candle peak ratcheting from stopping out on the same bar)
                    sl_hit = False
                    if is_long:
                        if cur_low <= active_trade.sl_price:
                            sl_hit = True
                    else:
                        if cur_high >= active_trade.sl_price:
                            sl_hit = True

                    # Archetype 1: breakeven_trail ratcheting (only ratchet if SL was not hit on this bar)
                    if not sl_hit and exit_type == "breakeven_trail" and active_sl_distance > 0:
                        if is_long:
                            gain = cur_high - active_trade.entry_price
                            if gain >= 1.0 * active_sl_distance:
                                active_trade.sl_price = max(active_trade.sl_price, active_trade.entry_price + half_spread)
                            if gain >= 1.5 * active_sl_distance:
                                active_trade.sl_price = max(active_trade.sl_price, cur_high - trailing_atr * bar_atr)
                        else:
                            gain = active_trade.entry_price - cur_low
                            if gain >= 1.0 * active_sl_distance:
                                active_trade.sl_price = min(active_trade.sl_price, active_trade.entry_price - half_spread)
                            if gain >= 1.5 * active_sl_distance:
                                active_trade.sl_price = min(active_trade.sl_price, cur_low + trailing_atr * bar_atr)

                    # Archetype 2: dynamic_mean_reversion
                    tp_hit = False
                    if exit_type == "dynamic_mean_reversion" and sma_20 is not None:
                        sma_val = sma_20[i - 1]
                        if is_long and cur_high >= sma_val and sma_val > active_trade.entry_price:
                            tp_hit = True
                            active_trade.tp_price = max(cur_open, sma_val)
                        elif not is_long and cur_low <= sma_val and sma_val < active_trade.entry_price:
                            tp_hit = True
                            active_trade.tp_price = min(cur_open, sma_val)

                    if not tp_hit:
                        if is_long and cur_high >= active_trade.tp_price:
                            tp_hit = True
                        elif not is_long and cur_low <= active_trade.tp_price:
                            tp_hit = True

                    # Pessimistic resolution: SL hit wins if both hit
                    if sl_hit and tp_hit or sl_hit:
                        exit_reason = "SL"
                        if is_long:
                            exit_price = min(active_trade.sl_price, cur_open) - half_spread
                        else:
                            exit_price = max(active_trade.sl_price, cur_open) + half_spread
                    elif tp_hit:
                        exit_reason = "TP"
                        if is_long:
                            exit_price = active_trade.tp_price - half_spread
                        else:
                            exit_price = active_trade.tp_price + half_spread

                if exit_reason != "":
                    active_trade.exit_idx = i
                    active_trade.exit_time = cur_time
                    active_trade.exit_price = exit_price
                    active_trade.exit_reason = exit_reason
                    active_trade.holding_bars = holding_bars

                    # Realized PnL
                    if is_long:
                        pnl_price = exit_price - active_trade.entry_price
                    else:
                        pnl_price = active_trade.entry_price - exit_price

                    active_trade.pnl_pips = pnl_price * pip_mult
                    commission = (active_trade.pos_units / 100000.0) * self.commission_per_lot * self.cost_multiplier
                    active_trade.commission_usd = commission
                    active_trade.pnl_usd = (pnl_price * active_trade.pos_units * q_usd) - commission + active_trade.swap_usd

                    self.equity += active_trade.pnl_usd
                    trades.append(active_trade)
                    active_trade = None
                    exited_this_bar = True

            # 2. Check for entry if flat AND did not exit this bar
            if active_trade is None and not exited_this_bar and i < n_bars - 1:
                sig_idx = i - 1
                long_sig = bool(entry_longs[sig_idx])
                short_sig = bool(entry_shorts[sig_idx])

                bar_atr = atrs_price[i - 1]
                is_pre_rollover_hour = (exit_type == "pre_rollover_flat" and ny_hours is not None and ny_hours[i] == 16)
                if not is_rollover and not is_pre_rollover_hour and is_market_entry_allowed(cur_spread, bar_atr, max_spread_atr_frac=0.25):
                    # Volatility targeting position sizing
                    risk_pct = 0.01
                    if vol_target > 0.0:
                        risk_pct = min(0.025, max(0.0025, 0.01 * (vol_target / 0.10)))
                    risk_cash = self.equity * risk_pct
                    sl_dist = max(sl_atr_mult * bar_atr, half_spread * 2.0)
                    active_sl_distance = sl_dist
                    sl_risk_unit = sl_dist * q_usd
                    base_unit_usd = max(cur_open * q_usd, 1e-4)
                    pos_units = min(risk_cash / sl_risk_unit, (self.equity * 10.0) / base_unit_usd) if sl_risk_unit > 0 else 10000.0

                    if long_sig and not short_sig:
                        e_price = cur_open + half_spread
                        active_trade = TradeRecord(
                            trade_id=len(trades) + 1,
                            pair=pair,
                            direction=1,
                            entry_idx=i,
                            entry_time=cur_time,
                            entry_price=e_price,
                            pos_units=pos_units,
                            sl_price=e_price - sl_dist,
                            tp_price=e_price + (tp_atr_mult * bar_atr),
                            quote_to_usd=q_usd,
                        )
                    elif short_sig and not long_sig:
                        e_price = cur_open - half_spread
                        active_trade = TradeRecord(
                            trade_id=len(trades) + 1,
                            pair=pair,
                            direction=-1,
                            entry_idx=i,
                            entry_time=cur_time,
                            entry_price=e_price,
                            pos_units=pos_units,
                            sl_price=e_price + sl_dist,
                            tp_price=e_price - (tp_atr_mult * bar_atr),
                            quote_to_usd=q_usd,
                        )

            # 3. Mark-to-market equity curve
            if active_trade is not None:
                is_long = (active_trade.direction == 1)
                unrealized_pnl = (cur_close - active_trade.entry_price if is_long else active_trade.entry_price - cur_close) * active_trade.pos_units * q_usd
                equity_history[i] = self.equity + unrealized_pnl
            else:
                equity_history[i] = self.equity

        # Build trades dataframe
        records = []
        for t in trades:
            records.append({
                "trade_id": t.trade_id,
                "pair": t.pair,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_idx": t.entry_idx,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_idx": t.exit_idx,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pos_units": t.pos_units,
                "pnl_pips": round(t.pnl_pips, 2),
                "pnl_usd": round(t.pnl_usd, 2),
                "holding_bars": t.holding_bars,
                "mae_pips": round(t.mae_pips, 2),
                "mfe_pips": round(t.mfe_pips, 2),
                "commission_usd": round(t.commission_usd, 2),
                "swap_usd": round(t.swap_usd, 2),
            })
        trades_df = pd.DataFrame(records)

        # Performance summary
        n_tr = len(trades)
        wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
        losses = [t.pnl_usd for t in trades if t.pnl_usd < 0]
        win_rate = float(len(wins) / n_tr) if n_tr > 0 else 0.0
        gross_profit = float(sum(wins)) if wins else 0.0
        gross_loss = float(abs(sum(losses))) if losses else 0.0
        pf = float(gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        tot_ret = float((self.equity - self.initial_equity) / self.initial_equity) * 100.0

        peaks = np.maximum.accumulate(equity_history)
        drawdowns = (peaks - equity_history) / np.where(peaks > 0, peaks, 1.0)
        max_dd = float(np.max(drawdowns)) * 100.0

        durations = [t.holding_bars for t in trades]
        median_hold = float(np.median(durations)) if durations else 0.0

        metrics = {
            "n_trades": n_tr,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(pf, 3),
            "total_return_pct": round(tot_ret, 2),
            "max_dd_pct": round(max_dd, 2),
            "net_pnl_usd": round(self.equity - self.initial_equity, 2),
            "holding_bars_median": median_hold,
            "equity_history": equity_history,
        }
        return trades, trades_df, metrics


def compare_engines(
    df: pd.DataFrame,
    pair: str,
    entry_longs: np.ndarray,
    entry_shorts: np.ndarray,
    market_closes: dict[str, np.ndarray],
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    max_holding_bars: int = 24,
    cost_multiplier: float = 1.5,
    tolerance_pct: float = 0.05,
    exit_type: str = "sl_tp",
    trailing_atr: float = 1.2,
) -> tuple[bool, dict[str, Any]]:
    """
    Verifies that Fast Vectorized Engine matches Event-Driven Engine within tight tolerance.
    """
    # 1. Run Vectorized
    from engine.costs import compute_quote_to_usd_vector
    q_vec = compute_quote_to_usd_vector(pair, len(df), market_closes)
    vec_res = run_vectorized_backtest(
        df=df,
        pair=pair,
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        quote_to_usd=q_vec,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
        max_holding_bars=max_holding_bars,
        cost_multiplier=cost_multiplier,
        exit_type=exit_type,
        trailing_atr=trailing_atr,
    )

    # 2. Run Event-Driven (without swap to isolate pure execution logic)
    ev_engine = EventDrivenEngine(cost_multiplier=cost_multiplier, swap_long_annual_pct=0.0, swap_short_annual_pct=0.0)
    _, _, ev_res = ev_engine.run_backtest(
        df=df,
        pair=pair,
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        market_closes=market_closes,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
        max_holding_bars=max_holding_bars,
        exit_type=exit_type,
        trailing_atr=trailing_atr,
    )

    n_diff = abs(vec_res["n_trades"] - ev_res["n_trades"])
    pnl_diff = abs(vec_res["net_pnl_usd"] - ev_res["net_pnl_usd"])
    base_pnl = max(abs(vec_res["net_pnl_usd"]), abs(ev_res["net_pnl_usd"]), 100.0)
    rel_pnl_diff = pnl_diff / base_pnl

    matches = bool((n_diff <= 1) and (rel_pnl_diff <= tolerance_pct))
    comparison = {
        "matches": matches,
        "vec_trades": vec_res["n_trades"],
        "ev_trades": ev_res["n_trades"],
        "vec_pnl": vec_res["net_pnl_usd"],
        "ev_pnl": ev_res["net_pnl_usd"],
        "pnl_diff": pnl_diff,
        "rel_pnl_diff": rel_pnl_diff,
    }
    return matches, comparison
