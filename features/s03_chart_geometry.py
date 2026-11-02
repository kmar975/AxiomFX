"""
S3: Chart Geometry, Causal Swings, BOS/CHoCH and Multi-Period Liquidity Sweeps.
Strict zero look-ahead: a swing at cand is confirmed and registered ONLY at t = cand + L.
Includes multi-period liquidity sweeps (s3_sweep_20, s3_sweep_60) from institutional bank setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import calc_atr, rolling_percentile_fast


def compute_s03_features(df: pd.DataFrame, pair: str, l_confirm: int = 3) -> pd.DataFrame:
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    # State variables
    confirmed_swing_highs = []  # list of (confirmed_time_t, cand_idx, price)
    confirmed_swing_lows = []

    structure_state = np.zeros(n, dtype=np.float64)  # +1 for HH+HL, -1 for LH+LL, 0 neutral
    bos = np.zeros(n, dtype=np.float64)              # +1 bull BOS, -1 bear BOS
    choch = np.zeros(n, dtype=np.float64)            # +1 bull CHoCH, -1 bear CHoCH
    retrace_frac = np.zeros(n, dtype=np.float64)
    consol_frac = np.zeros(n, dtype=np.float64)
    liquidity_sweep = np.zeros(n, dtype=np.float64)

    last_sh_price = np.nan
    last_sl_price = np.nan
    prev_sh_price = np.nan
    prev_sl_price = np.nan

    for t in range(n):
        # 1. Check if bar (cand = t - l_confirm) is a confirmed swing high or low
        cand = t - l_confirm
        if cand >= l_confirm:
            # Check swing high
            window_h = highs[cand - l_confirm : t + 1]
            if highs[cand] == np.max(window_h) and np.sum(window_h == highs[cand]) == 1:
                prev_sh_price = last_sh_price
                last_sh_price = highs[cand]
                confirmed_swing_highs.append((t, cand, last_sh_price))

            # Check swing low
            window_l = lows[cand - l_confirm : t + 1]
            if lows[cand] == np.min(window_l) and np.sum(window_l == lows[cand]) == 1:
                prev_sl_price = last_sl_price
                last_sl_price = lows[cand]
                confirmed_swing_lows.append((t, cand, last_sl_price))

        # 2. Determine structure state (HH+HL vs LH+LL)
        if not np.isnan(last_sh_price) and not np.isnan(prev_sh_price):
            if not np.isnan(last_sl_price) and not np.isnan(prev_sl_price):
                if last_sh_price > prev_sh_price and last_sl_price > prev_sl_price:
                    structure_state[t] = 1.0
                elif last_sh_price < prev_sh_price and last_sl_price < prev_sl_price:
                    structure_state[t] = -1.0
                else:
                    structure_state[t] = structure_state[t - 1] if t > 0 else 0.0
            else:
                structure_state[t] = structure_state[t - 1] if t > 0 else 0.0
        else:
            structure_state[t] = structure_state[t - 1] if t > 0 else 0.0

        # 3. BOS and CHoCH
        c = closes[t]
        if not np.isnan(last_sh_price) and not np.isnan(last_sl_price):
            if structure_state[t] == 1.0:
                # In uptrend: break above SH is continuation BOS
                if c > last_sh_price and (t == 0 or closes[t - 1] <= last_sh_price):
                    bos[t] = 1.0
                # Break below SL is reversal CHoCH
                elif c < last_sl_price and (t == 0 or closes[t - 1] >= last_sl_price):
                    choch[t] = -1.0
            elif structure_state[t] == -1.0:
                # In downtrend: break below SL is continuation BOS
                if c < last_sl_price and (t == 0 or closes[t - 1] >= last_sl_price):
                    bos[t] = -1.0
                # Break above SH is reversal CHoCH
                elif c > last_sh_price and (t == 0 or closes[t - 1] <= last_sh_price):
                    choch[t] = 1.0

            # 4. Swing Liquidity sweep: wick passes swing high/low but close returns inside
            if highs[t] > last_sh_price and c < last_sh_price:
                liquidity_sweep[t] = -1.0  # Bearish sweep of highs
            elif lows[t] < last_sl_price and c > last_sl_price:
                liquidity_sweep[t] = 1.0   # Bullish sweep of lows

            # 5. Retracement fraction of last impulse
            impulse_span = abs(last_sh_price - last_sl_price)
            if impulse_span > 1e-8:
                if last_sh_price > last_sl_price:
                    retrace_frac[t] = np.clip((last_sh_price - c) / impulse_span, 0.0, 2.0)
                else:
                    retrace_frac[t] = np.clip((c - last_sh_price) / impulse_span, 0.0, 2.0)

        # 6. Consolidation fraction (Range of last 10 bars / (10 * ATR))
        if t >= 10:
            local_range = np.max(highs[t - 10 : t + 1]) - np.min(lows[t - 10 : t + 1])
            consol_frac[t] = local_range / (atr_safe[t] * 3.0)

    # 7. Multi-Period Liquidity Sweeps (Stop Hunts: N=20 and N=60)
    # Strictly causal: shift(1) ensures extremes are from bars [t-N .. t-1]
    series_highs = pd.Series(highs)
    series_lows = pd.Series(lows)

    prior_low_20 = series_lows.shift(1).rolling(20, min_periods=20).min().values
    prior_high_20 = series_highs.shift(1).rolling(20, min_periods=20).max().values
    sweep_low_20 = (lows < prior_low_20) & (closes > prior_low_20)
    sweep_high_20 = (highs > prior_high_20) & (closes < prior_high_20)
    s3_sweep_20 = np.where(sweep_low_20, 1.0, np.where(sweep_high_20, -1.0, 0.0)).astype(np.float64)

    prior_low_60 = series_lows.shift(1).rolling(60, min_periods=60).min().values
    prior_high_60 = series_highs.shift(1).rolling(60, min_periods=60).max().values
    sweep_low_60 = (lows < prior_low_60) & (closes > prior_low_60)
    sweep_high_60 = (highs > prior_high_60) & (closes < prior_high_60)
    s3_sweep_60 = np.where(sweep_low_60, 1.0, np.where(sweep_high_60, -1.0, 0.0)).astype(np.float64)

    feats = pd.DataFrame({
        "s3_structure_state": structure_state,
        "s3_bos": bos,
        "s3_choch": choch,
        "s3_liquidity_sweep": liquidity_sweep,
        "s3_retrace_frac_pct": rolling_percentile_fast(retrace_frac),
        "s3_consol_frac_pct": rolling_percentile_fast(consol_frac),
        "s3_sweep_20": s3_sweep_20,
        "s3_sweep_60": s3_sweep_60,
    })
    return feats
