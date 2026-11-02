"""
Institutional Live Scanner Engine (§ 5.2, § 10.2).
Evaluates strategies in real time on the latest candle across 4 distinct states:
1. BUY (+1)
2. SELL (-1)
3. ARMED (0: Context and Setup met within M-bar validity, awaiting Trigger)
4. NEUTRAL (0)
Computes causal entry, stop-loss, and take-profit price levels with exact pip precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.costs import get_pair_specs
from grammar.compiler import compile_live_state
from grammar.strategy import Strategy


def scan_strategy_on_live_candle(
    strategy: Strategy,
    features_df: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    pair: str,
    timeframe: str = "H1",
    meta_filter: Any = None,
) -> dict[str, Any]:
    """
    Evaluates strategy on the latest closed candle and generates execution specs.
    Optionally applies trained secondary meta-model filter if provided.
    """
    spec = get_pair_specs(pair)
    dec = int(spec["dec"])
    pip_unit = float(spec["pip_unit"])
    pip_mult = float(spec["pip_mult"])

    last_idx = len(df_ohlcv) - 1
    state, armed_dir = compile_live_state(strategy, features_df, bar_idx=last_idx, return_direction=True)

    # Apply secondary ML meta-filter if present
    if meta_filter is not None and getattr(meta_filter, "model", None) is not None:
        if state in ("BUY", "SELL"):
            valid_cols = [c for c in meta_filter.feature_cols if c in features_df.columns]
            if valid_cols:
                x_sig = features_df[valid_cols].iloc[[last_idx]].values.astype(np.float32)
                x_sig = np.nan_to_num(x_sig, nan=50.0, posinf=100.0, neginf=0.0)
                prob = float(meta_filter.model.predict_proba(x_sig)[0, 1])
                if prob < meta_filter.threshold:
                    state = "NEUTRAL"

    last_close = df_ohlcv["close"].iloc[last_idx]
    last_spread_pips = df_ohlcv["spread"].iloc[last_idx] if "spread" in df_ohlcv.columns else 1.5
    last_spread_price = last_spread_pips * pip_unit

    # 20-period True Range ATR consistent with vectorized and event-driven backtesting
    highs = df_ohlcv["high"].values.astype(float)
    lows = df_ohlcv["low"].values.astype(float)
    closes = df_ohlcv["close"].values.astype(float)
    if len(closes) >= 2:
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        atr20_val = float(pd.Series(tr).rolling(20, min_periods=1).mean().iloc[-1])
        atr_price = max(atr20_val, pip_unit * 10.0)
    else:
        bar_range = df_ohlcv["high"].iloc[last_idx] - df_ohlcv["low"].iloc[last_idx]
        atr_price = max(bar_range, pip_unit * 10.0)

    sl_dist = strategy.exit.sl_atr * atr_price
    tp_dist = strategy.exit.tp_atr * atr_price

    if state == "BUY":
        entry_price = round(last_close + 0.5 * last_spread_price, dec)
        sl_price = round(entry_price - sl_dist, dec)
        tp_price = round(entry_price + tp_dist, dec)
        alert_desc = f"BUY SIGNAL: {pair} {timeframe} | {strategy.strategy_id} triggered on {strategy.trigger.feature}"
    elif state == "SELL":
        entry_price = round(last_close - 0.5 * last_spread_price, dec)
        sl_price = round(entry_price + sl_dist, dec)
        tp_price = round(entry_price - tp_dist, dec)
        alert_desc = f"SELL SIGNAL: {pair} {timeframe} | {strategy.strategy_id} triggered on {strategy.trigger.feature}"
    elif state == "ARMED":
        entry_price = round(last_close, dec)
        if armed_dir == -1:
            sl_price = round(entry_price + sl_dist, dec)
            tp_price = round(entry_price - tp_dist, dec)
            alert_desc = f"ARMED SHORT: {pair} {timeframe} | Setup active, awaiting short trigger {strategy.trigger.feature}"
        else:
            sl_price = round(entry_price - sl_dist, dec)
            tp_price = round(entry_price + tp_dist, dec)
            alert_desc = f"ARMED LONG: {pair} {timeframe} | Setup active, awaiting long trigger {strategy.trigger.feature}"
    else:
        entry_price = round(last_close, dec)
        sl_price = round(entry_price - sl_dist, dec)
        tp_price = round(entry_price + tp_dist, dec)
        alert_desc = f"NEUTRAL: {pair} {timeframe} | Inactive"

    return {
        "strategy_id": strategy.strategy_id,
        "family": strategy.family,
        "pair": pair,
        "timeframe": timeframe,
        "state": state,  # BUY, SELL, ARMED, NEUTRAL
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pips": round(sl_dist * pip_mult, 1),
        "tp_pips": round(tp_dist * pip_mult, 1),
        "alert_desc": alert_desc,
        "last_candle_time": str(df_ohlcv["time"].iloc[last_idx]),
    }
