"""
S7: Statistical and Series Properties.
Rolling autocorrelation, Variance Ratio, Hurst exponent proxy, skewness, kurtosis, and price z-score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from features.base import rolling_percentile_fast


@njit(fastmath=True)
def _calc_rolling_return_entropy(closes: np.ndarray, window: int = 40) -> np.ndarray:
    n = len(closes)
    entropy = np.full(n, 50.0, dtype=np.float64)
    rets = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        rets[i] = closes[i] - closes[i - 1]
    for i in range(window, n):
        up_count = 0
        for j in range(i - window + 1, i + 1):
            if rets[j] > 0.0:
                up_count += 1
        p_up = up_count / float(window)
        p_dn = 1.0 - p_up
        if 0.0 < p_up < 1.0:
            h = -(p_up * np.log2(p_up) + p_dn * np.log2(p_dn))
            val = h * 100.0
            if val < 0.0:
                val = 0.0
            elif val > 100.0:
                val = 100.0
            entropy[i] = val
        else:
            entropy[i] = 0.0
    return entropy


def compute_s07_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    # Returns
    rets = np.zeros(n, dtype=np.float64)
    rets[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-4)

    rets_s = pd.Series(rets)

    # 1. Rolling Autocorrelation (Lag 1)
    autocorr_lag1 = np.zeros(n, dtype=np.float64)
    for i in range(25, n):
        w = rets[i - 20 : i + 1]
        std = np.std(w)
        if std > 1e-8:
            r0 = w[1:] - np.mean(w[1:])
            r1 = w[:-1] - np.mean(w[:-1])
            cov = np.mean(r0 * r1)
            autocorr_lag1[i] = np.clip(cov / (std ** 2), -1.0, 1.0)

    # 2. Variance Ratio (VR_5)
    vr_5 = np.ones(n, dtype=np.float64)
    rets_5 = np.zeros(n, dtype=np.float64)
    rets_5[5:] = (closes[5:] - closes[:-5]) / np.maximum(closes[:-5], 1e-4)
    for i in range(30, n):
        v1 = np.var(rets[i - 25 : i + 1])
        v5 = np.var(rets_5[i - 25 : i + 1])
        if v1 > 1e-10:
            vr_5[i] = v5 / (5.0 * v1)

    # 3. Hurst Exponent (R/S proxy over 40 bars)
    hurst_rs = np.full(n, 0.5, dtype=np.float64)
    for i in range(40, n):
        w = rets[i - 40 : i + 1]
        mean_w = np.mean(w)
        s = np.std(w)
        if s > 1e-8:
            cum_dev = np.cumsum(w - mean_w)
            r = np.max(cum_dev) - np.min(cum_dev)
            if r > 1e-8:
                h = np.log(r / s) / np.log(40.0)
                hurst_rs[i] = np.clip(h, 0.0, 1.0)

    # 4. Skewness and Kurtosis (20 bars)
    skew_20 = rets_s.rolling(20, min_periods=5).skew().fillna(0.0).values.astype(np.float64)
    kurt_20 = rets_s.rolling(20, min_periods=5).kurt().fillna(0.0).values.astype(np.float64)

    # 5. Price Z-Score (20 bars)
    c_s = pd.Series(closes)
    mean_20 = c_s.rolling(20, min_periods=5).mean().values
    std_20 = c_s.rolling(20, min_periods=5).std().fillna(1e-4).values
    zscore_20 = (closes - mean_20) / np.where(std_20 > 1e-8, std_20, 1e-4)

    # 6. Shannon Entropy of returns
    entropy_40 = _calc_rolling_return_entropy(closes, window=40)

    feats = pd.DataFrame({
        "s7_autocorr_lag1_pct": rolling_percentile_fast(autocorr_lag1),
        "s7_variance_ratio_5_pct": rolling_percentile_fast(vr_5),
        "s7_hurst_rs": hurst_rs * 100.0,
        "s7_skew_20_pct": rolling_percentile_fast(skew_20),
        "s7_kurt_20_pct": rolling_percentile_fast(kurt_20),
        "s7_price_zscore_pct": rolling_percentile_fast(zscore_20),
        "s7_return_entropy": entropy_40,
    })
    return feats
