"""
Labeling Engine for Quantitative Evaluation.
Generates h-bar future volatility-scaled returns, triple-barrier labels,
and future volatility targets across precomputed float32 matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import calc_atr


def compute_future_returns(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Computes forward return scaled by causal ATR: (close[t+h] - close[t]) / ATR[t].
    Returns dict mapping horizon h -> float32 array of length N.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20, 60]
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    labels = {}
    for h in horizons:
        ret_h = np.zeros(n, dtype=np.float32)
        if n > h:
            ret_h[:-h] = ((closes[h:] - closes[:-h]) / atr_safe[:-h]).astype(np.float32)
        labels[h] = ret_h

    return labels


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    barrier_atr_mult: float = 1.5,
    max_horizon_bars: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triple-barrier method: upper barrier (+k*ATR), lower barrier (-k*ATR), vertical barrier (H bars).
    Returns:
    - label: +1 (upper barrier first), -1 (lower barrier first), 0 (vertical barrier first)
    - holding_bars: duration until first barrier touch
    """
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    labels = np.zeros(n, dtype=np.int8)
    durations = np.zeros(n, dtype=np.int16)

    for i in range(n - max_horizon_bars):
        entry_p = closes[i]
        barrier = barrier_atr_mult * atr_safe[i]
        up_b = entry_p + barrier
        dn_b = entry_p - barrier

        hit = 0
        hit_bar = max_horizon_bars

        for k in range(1, max_horizon_bars + 1):
            idx = i + k
            h = highs[idx]
            l = lows[idx]

            up_hit = h >= up_b
            dn_hit = l <= dn_b

            if up_hit and dn_hit:
                # Pessimistic conflict: treat as loss (-1)
                hit = -1
                hit_bar = k
                break
            elif up_hit:
                hit = 1
                hit_bar = k
                break
            elif dn_hit:
                hit = -1
                hit_bar = k
                break

        labels[i] = hit
        durations[i] = hit_bar

    return labels, durations
