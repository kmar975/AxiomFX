"""
S5: Volatility, Compression, and Squeeze Features.
Calculates causal ATR percentiles, term-structure ratios, Squeeze metrics, and Garman-Klass volatility.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from features.base import calc_atr, calc_sma, rolling_percentile_fast


def compute_s05_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    # ATRs
    atr_5 = calc_atr(highs, lows, closes, period=5)
    atr_10 = calc_atr(highs, lows, closes, period=10)
    atr_20 = calc_atr(highs, lows, closes, period=20)
    atr_40 = calc_atr(highs, lows, closes, period=40)
    atr_80 = calc_atr(highs, lows, closes, period=80)

    atr_20_safe = np.where(atr_20 > 1e-8, atr_20, 1e-4)

    # Term structure ratios
    ratio_5_20 = atr_5 / atr_20_safe
    ratio_10_40 = atr_10 / np.where(atr_40 > 1e-8, atr_40, 1e-4)
    ratio_20_80 = atr_20 / np.where(atr_80 > 1e-8, atr_80, 1e-4)

    # Squeeze: Bollinger Bands (20, 2.0) vs Keltner Channels (20, 1.5)
    sma_20 = calc_sma(closes, window=20)
    rolling_std_20 = pd.Series(closes).rolling(20, min_periods=1).std().fillna(1e-4).values.astype(np.float64)
    bb_upper = sma_20 + 2.0 * rolling_std_20
    bb_lower = sma_20 - 2.0 * rolling_std_20
    bb_width = (bb_upper - bb_lower) / atr_20_safe

    kc_upper = sma_20 + 1.5 * atr_20
    kc_lower = sma_20 - 1.5 * atr_20
    kc_width = (kc_upper - kc_lower) / atr_20_safe

    # Squeeze is on when BB is inside KC
    squeeze_on = np.where((bb_upper <= kc_upper) & (bb_lower >= kc_lower), 1.0, 0.0)
    squeeze_ratio = bb_width / np.where(kc_width > 1e-8, kc_width, 1e-4)

    # Garman-Klass Volatility Estimator
    # GK = 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    gk_vol = np.zeros(n, dtype=np.float64)
    safe_lows = np.where(lows > 1e-8, lows, 1.0)
    safe_opens = np.where(opens > 1e-8, opens, 1.0)
    log_hl = np.log(np.maximum(highs / safe_lows, 1.0))
    log_co = np.log(np.maximum(closes / safe_opens, 1e-6))
    gk_instant = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
    gk_vol = pd.Series(gk_instant).rolling(20, min_periods=1).mean().values.astype(np.float64)

    # ADX 14 calculation for BB + ADX grid
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    smooth_pdm = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    smooth_mdm = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    di_sum = smooth_pdm + smooth_mdm
    safe_di_sum = np.where(di_sum > 1e-8, di_sum, 1.0)
    dx = np.where(di_sum > 1e-8, 100.0 * np.abs(smooth_pdm - smooth_mdm) / safe_di_sum, 0.0)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False).mean().values

    # Bollinger Bands + ADX grid: periods [14, 20] x deviations [1.5, 1.8, 2.0, 2.2, 2.5]
    bb_adx_feats: dict[str, np.ndarray] = {}
    for p in [14, 20]:
        sma_p = pd.Series(closes).rolling(p, min_periods=1).mean().values
        std_p = pd.Series(closes).rolling(p, min_periods=1).std(ddof=0).fillna(1e-4).values
        for sig in [1.5, 1.8, 2.0, 2.2, 2.5]:
            sig_str = str(sig).replace('.', 'p')
            upper = sma_p + sig * std_p
            lower = sma_p - sig * std_p
            # Mean reversion: Close < Lower and ADX < 20
            bb_adx_feats[f"s5_bb_{p}_{sig_str}_mr_lower"] = ((closes < lower) & (adx_14 < 20.0)).astype(np.float64)
            bb_adx_feats[f"s5_bb_{p}_{sig_str}_mr_upper"] = ((closes > upper) & (adx_14 < 20.0)).astype(np.float64)
            # Breakout: Close > Upper and ADX > 22
            bb_adx_feats[f"s5_bb_{p}_{sig_str}_bo_upper"] = ((closes > upper) & (adx_14 > 22.0)).astype(np.float64)
            bb_adx_feats[f"s5_bb_{p}_{sig_str}_bo_lower"] = ((closes < lower) & (adx_14 > 22.0)).astype(np.float64)

    res_dict: dict[str, Any] = {
        "s5_atr20_pct": rolling_percentile_fast(atr_20),
        "s5_atr_ratio_5_20_pct": rolling_percentile_fast(ratio_5_20),
        "s5_atr_ratio_10_40_pct": rolling_percentile_fast(ratio_10_40),
        "s5_atr_ratio_20_80_pct": rolling_percentile_fast(ratio_20_80),
        "s5_bb_width_pct": rolling_percentile_fast(bb_width),
        "s5_squeeze_on": squeeze_on,
        "s5_squeeze_ratio_pct": rolling_percentile_fast(squeeze_ratio),
        "s5_gk_vol_pct": rolling_percentile_fast(gk_vol),
    }
    res_dict.update(bb_adx_feats)
    feats = pd.DataFrame(res_dict)
    return feats
