"""
S1: Price Action and Trend Features.
Dimensionless, normalized in units of causal ATR or as percentiles [0..100].
Zero look-ahead guaranteed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from features.base import (
    calc_atr,
    calc_ema,
    calc_kalman_filter,
    calc_kama,
    calc_rsi,
    calc_sma,
    rolling_percentile_fast,
)


@njit(fastmath=True)
def compute_supertrend_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes SuperTrend direction (+1 for Bullish, -1 for Bearish) and value.
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.int32)
    supertrend = np.zeros(n, dtype=np.float64)

    upper_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    direction[0] = 1
    supertrend[0] = basic_lower[0]
    upper_band[0] = basic_upper[0]
    lower_band[0] = basic_lower[0]

    for i in range(1, n):
        if basic_upper[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            upper_band[i] = basic_upper[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if basic_lower[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            lower_band[i] = basic_lower[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if close[i] > upper_band[i - 1]:
            direction[i] = 1
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]

    return direction, supertrend


@njit(fastmath=True)
def compute_stochastic_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
    s_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Slow %K and %D Stochastic.
    """
    n = len(closes)
    fast_k = np.full(n, 50.0, dtype=np.float64)

    for i in range(k_period - 1, n):
        highest_h = highs[i - k_period + 1 : i + 1].max()
        lowest_l = lows[i - k_period + 1 : i + 1].min()
        denom = highest_h - lowest_l
        if denom > 1e-9:
            fast_k[i] = ((closes[i] - lowest_l) / denom) * 100.0
        else:
            fast_k[i] = 50.0

    slow_k = np.full(n, 50.0, dtype=np.float64)
    for i in range(k_period + s_period - 2, n):
        slow_k[i] = np.mean(fast_k[i - s_period + 1 : i + 1])

    slow_d = np.full(n, 50.0, dtype=np.float64)
    for i in range(k_period + s_period + d_period - 3, n):
        slow_d[i] = np.mean(slow_k[i - d_period + 1 : i + 1])

    return slow_k, slow_d


@njit(fastmath=True)
def compute_cci_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Computes Commodity Channel Index (CCI)."""
    n = len(closes)
    tp = (highs + lows + closes) / 3.0
    cci = np.zeros(n, dtype=np.float64)

    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean_tp = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean_tp))
        if mean_dev > 1e-9:
            cci[i] = (tp[i] - mean_tp) / (0.015 * mean_dev)
        else:
            cci[i] = 0.0

    return cci


@njit(fastmath=True)
def compute_williams_r_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Computes Williams %R."""
    n = len(closes)
    wr = np.full(n, -50.0, dtype=np.float64)

    for i in range(period - 1, n):
        highest_h = highs[i - period + 1 : i + 1].max()
        lowest_l = lows[i - period + 1 : i + 1].min()
        denom = highest_h - lowest_l
        if denom > 1e-9:
            wr[i] = ((highest_h - closes[i]) / denom) * -100.0
        else:
            wr[i] = -50.0

    return wr


def compute_s01_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    # 1. Scaled returns
    ret_5 = np.zeros(n, dtype=np.float64)
    ret_20 = np.zeros(n, dtype=np.float64)
    for i in range(5, n):
        ret_5[i] = (closes[i] - closes[i - 5]) / atr_safe[i]
    for i in range(20, n):
        ret_20[i] = (closes[i] - closes[i - 20]) / atr_safe[i]

    # 2. Distances to Averages in ATR
    sma_20 = calc_sma(closes, window=20)
    ema_20 = calc_ema(closes, span=20)
    kama_10 = calc_kama(closes, er_period=10)
    kalman = calc_kalman_filter(closes)

    sma_dist_atr = (closes - sma_20) / atr_safe
    ema_dist_atr = (closes - ema_20) / atr_safe
    kama_dist_atr = (closes - kama_10) / atr_safe
    kalman_dist_atr = (closes - kalman) / atr_safe

    # 3. Kaufman Efficiency Ratio
    er_10 = np.zeros(n, dtype=np.float64)
    for i in range(10, n):
        change = abs(closes[i] - closes[i - 10])
        path = np.sum(np.abs(np.diff(closes[i - 10 : i + 1])))
        er_10[i] = (change / path) if path > 1e-12 else 0.0

    # 4. Donchian Position (20 and 55)
    donchian_pos = np.full(n, 0.5, dtype=np.float64)
    for i in range(20, n):
        dh = np.max(highs[i - 20 : i + 1])
        dl = np.min(lows[i - 20 : i + 1])
        donchian_pos[i] = (closes[i] - dl) / (dh - dl) if (dh - dl) > 1e-8 else 0.5

    # 5. MACD normalized by ATR
    ema_12 = calc_ema(closes, span=12)
    ema_26 = calc_ema(closes, span=26)
    macd_line = ema_12 - ema_26
    macd_norm_atr = macd_line / atr_safe

    # 6. Dense RSI Grid [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 21, 28]
    rsi_periods = [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 21, 28]
    rsi_dict: dict[str, np.ndarray] = {}
    for p in rsi_periods:
        rsi_dict[f"s1_rsi_{p}"] = calc_rsi(closes, period=p)

    # 7. Dense Stochastic Grid [5, 8, 14, 21]
    stoch_k_periods = [5, 8, 14, 21]
    stoch_dict: dict[str, np.ndarray] = {}
    for kp in stoch_k_periods:
        sk, sd = compute_stochastic_numba(highs, lows, closes, k_period=kp, d_period=3, s_period=3)
        stoch_dict[f"s1_stoch_{kp}"] = sk
        stoch_dict[f"s1_stoch_d_{kp}"] = sd

        sk_prev = np.roll(sk, 1); sk_prev[0] = sk[0]
        sd_prev = np.roll(sd, 1); sd_prev[0] = sd[0]
        stoch_dict[f"s1_stoch_{kp}_cross_up"] = ((sk > sd) & (sk_prev <= sd_prev)).astype(np.float64)
        stoch_dict[f"s1_stoch_{kp}_cross_dn"] = ((sk < sd) & (sk_prev >= sd_prev)).astype(np.float64)

    # 8. Dense CCI [7, 10, 14, 20, 30] & Williams %R [5, 9, 14, 21]
    cci_periods = [7, 10, 14, 20, 30]
    cci_dict: dict[str, np.ndarray] = {}
    for cp in cci_periods:
        c_val = compute_cci_numba(highs, lows, closes, period=cp)
        cci_dict[f"s1_cci_{cp}_pct"] = rolling_percentile_fast(c_val)
        c_prev = np.roll(c_val, 1); c_prev[0] = c_val[0]
        cci_dict[f"s1_cci_{cp}_cross_up_m100"] = ((c_val > -100.0) & (c_prev <= -100.0)).astype(np.float64)
        cci_dict[f"s1_cci_{cp}_cross_dn_100"] = ((c_val < 100.0) & (c_prev >= 100.0)).astype(np.float64)

    wr_periods = [5, 9, 14, 21]
    wr_dict: dict[str, np.ndarray] = {}
    for wp in wr_periods:
        wr_val = compute_williams_r_numba(highs, lows, closes, period=wp)
        wr_dict[f"s1_williams_r_{wp}"] = wr_val + 100.0  # shifted to [0..100]

    # 9. SuperTrend Numba (10, 3.0) and (14, 2.0)
    st_dir_10_3, st_val_10_3 = compute_supertrend_numba(highs, lows, closes, atr_safe, multiplier=3.0)
    st_dir_14_2, st_val_14_2 = compute_supertrend_numba(highs, lows, closes, atr_safe, multiplier=2.0)
    st_dist_10_3 = (closes - st_val_10_3) / atr_safe
    st_dist_14_2 = (closes - st_val_14_2) / atr_safe

    # 10. EMA Stacks Grid [8, 13, 20, 34, 50, 89, 100, 144, 200]
    ema_spans = [8, 13, 20, 34, 50, 89, 100, 144, 200]
    emas: dict[int, np.ndarray] = {}
    ema_feats: dict[str, np.ndarray] = {}
    for s in ema_spans:
        e_val = calc_ema(closes, span=s)
        emas[s] = e_val
        ema_feats[f"s1_ema_dist_{s}_pct"] = rolling_percentile_fast((closes - e_val) / atr_safe)

    ema_feats["s1_ema_stack_8_20"] = (emas[8] > emas[20]).astype(np.float64)
    ema_feats["s1_ema_stack_13_34"] = (emas[13] > emas[34]).astype(np.float64)
    bull_stack_20_50_200 = (emas[20] > emas[50]) & (emas[50] > emas[200])
    bear_stack_20_50_200 = (emas[20] < emas[50]) & (emas[50] < emas[200])
    ema_feats["s1_ema_stack_20_50_200"] = np.where(
        bull_stack_20_50_200, 1.0, np.where(bear_stack_20_50_200, -1.0, 0.0)
    ).astype(np.float64)

    # 11. Donchian Channel Breakouts [10, 20, 40, 60] (with causal shift 1)
    donch_feats: dict[str, np.ndarray] = {}
    for lb in [10, 20, 40, 60]:
        d_high = pd.Series(highs).shift(1).rolling(lb, min_periods=lb).max().values
        d_low = pd.Series(lows).shift(1).rolling(lb, min_periods=lb).min().values
        donch_feats[f"s1_donchian_breakout_high_{lb}"] = (closes > d_high).astype(np.float64)
        donch_feats[f"s1_donchian_breakout_low_{lb}"] = (closes < d_low).astype(np.float64)

    # 12. ADX (Wilder)
    adx_14 = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    smooth_pdm = calc_ema(plus_dm, span=14)
    smooth_mdm = calc_ema(minus_dm, span=14)
    di_sum = smooth_pdm + smooth_mdm
    safe_di_sum = np.where(di_sum > 1e-8, di_sum, 1.0)
    dx = np.where(di_sum > 1e-8, 100.0 * np.abs(smooth_pdm - smooth_mdm) / safe_di_sum, 0.0)
    adx_14 = calc_ema(dx, span=14)

    # Assemble complete S1 feature dictionary
    res_dict: dict[str, Any] = {
        "s1_ret_5_pct": rolling_percentile_fast(ret_5),
        "s1_ret_20_pct": rolling_percentile_fast(ret_20),
        "s1_sma_dist_pct": rolling_percentile_fast(sma_dist_atr),
        "s1_ema_dist_pct": rolling_percentile_fast(ema_dist_atr),
        "s1_kama_dist_pct": rolling_percentile_fast(kama_dist_atr),
        "s1_kalman_dist_pct": rolling_percentile_fast(kalman_dist_atr),
        "s1_er_10_pct": rolling_percentile_fast(er_10),
        "s1_donchian_pos": donchian_pos * 100.0,
        "s1_macd_norm_pct": rolling_percentile_fast(macd_norm_atr),
        "s1_adx_14": adx_14,
        "s1_supertrend_10_3_dir": st_dir_10_3.astype(np.float64),
        "s1_supertrend_10_3_dist_pct": rolling_percentile_fast(st_dist_10_3),
        "s1_supertrend_14_2_dir": st_dir_14_2.astype(np.float64),
        "s1_supertrend_14_2_dist_pct": rolling_percentile_fast(st_dist_14_2),
    }
    res_dict.update(rsi_dict)
    res_dict.update(stoch_dict)
    res_dict.update(cci_dict)
    res_dict.update(wr_dict)
    res_dict.update(ema_feats)
    res_dict.update(donch_feats)

    feats = pd.DataFrame(res_dict)
    return feats
