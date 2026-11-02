"""
S10: Analogs, Candlestick N-Grams, and Causal Volatility Regimes.
Strictly causal forward matching: no backward lookahead smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from features.base import calc_atr, rolling_percentile_fast


@njit(fastmath=True)
def _fast_ngram_prob(symbol: np.ndarray, closes: np.ndarray, n: int) -> np.ndarray:
    ngram_prob = np.full(n, 50.0, dtype=np.float64)
    for i in range(50, n):
        s0 = symbol[i - 1]
        s1 = symbol[i]
        limit = i - 5
        up_count = 0
        total_count = 0
        for j in range(1, limit - 1):
            if symbol[j - 1] == s0 and symbol[j] == s1:
                total_count += 1
                if closes[j + 1] > closes[j]:
                    up_count += 1
        if total_count >= 5:
            prior_weight = 5.0
            p = ((up_count + 0.5 * prior_weight) / (total_count + prior_weight)) * 100.0
            ngram_prob[i] = p
    return ngram_prob


@njit(fastmath=True)
def _fast_knn_exp_ret(closes: np.ndarray, atr_safe: np.ndarray, n: int, p_len: int = 10) -> np.ndarray:
    knn_exp_ret = np.full(n, 0.0, dtype=np.float64)
    for i in range(100, n):
        t_start = i - p_len + 1
        t_base = closes[t_start]
        target_norm = (closes[t_start : i + 1] - t_base) / atr_safe[i]
        best_sim = -1e9
        best_future_ret = 0.0
        limit = i - p_len - 5
        for j in range(p_len, limit, 10):
            c_start = j - p_len + 1
            c_base = closes[c_start]
            cand_norm = (closes[c_start : j + 1] - c_base) / atr_safe[j]
            diff = target_norm - cand_norm
            dist = np.mean(diff * diff)
            sim = -dist
            if sim > best_sim:
                best_sim = sim
                best_future_ret = (closes[j + 5] - closes[j]) / atr_safe[j]
        knn_exp_ret[i] = best_future_ret
    return knn_exp_ret


def compute_s10_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    # 1. Causal Volatility Regime Probability (Forward Sigmoid Z-Score)
    # Strictly causal: uses only past bars <= t
    atr_series = pd.Series(atr)
    atr_mean_60 = atr_series.rolling(60, min_periods=20).mean().values
    atr_std_60 = atr_series.rolling(60, min_periods=20).std().fillna(1e-4).values
    z_vol = (atr - atr_mean_60) / np.where(atr_std_60 > 1e-8, atr_std_60, 1e-4)
    # Sigmoid mapping to [0..100]
    regime_p_high_vol = 100.0 / (1.0 + np.exp(-z_vol))

    # 2. Fast Causal N-Gram / Shape Sequence
    # Discretize candle into 4 states: Bull strong, Bull weak, Bear weak, Bear strong
    candle_dir = np.where(closes >= opens, 1, -1)
    body = np.abs(closes - opens)
    is_strong = np.where(body >= 0.5 * atr_safe, 2, 1)
    symbol = (candle_dir * is_strong).astype(np.int64)  # values in {-2, -1, 1, 2}

    ngram_prob = _fast_ngram_prob(symbol, closes, n)

    # 3. Fast Causal k-NN Analog Return
    knn_exp_ret = _fast_knn_exp_ret(closes, atr_safe, n, p_len=10)

    feats = pd.DataFrame({
        "s10_regime_p_high_vol": regime_p_high_vol,
        "s10_ngram_bayes_prob": ngram_prob,
        "s10_knn_exp_ret_pct": rolling_percentile_fast(knn_exp_ret),
    })
    return feats
