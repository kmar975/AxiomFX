"""
Core Mathematical and Causal Primitives for Feature Engineering.
All algorithms strictly use past data (<= t) with zero future look-ahead.
Includes fast Numba rolling percentiles, causal ATR, Kalman filter, KAMA, and EMA.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(fastmath=True)
def calc_tr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))
    return tr


@njit(fastmath=True)
def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Wilder's ATR computed causally.
    """
    n = len(closes)
    tr = calc_tr(highs, lows, closes)
    atr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return atr

    # Initial SMA for first 'period' bars
    init_bars = min(n, period)
    init_sum = 0.0
    for i in range(init_bars):
        init_sum += tr[i]
        atr[i] = init_sum / (i + 1)

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@njit(fastmath=True)
def rolling_percentile_fast(
    values: np.ndarray,
    window: int = 250,
    min_periods: int = 50,
) -> np.ndarray:
    """
    Computes rolling empirical percentile (0.0 to 100.0) strictly using past observations.
    O(N * window) Numba kernel running in milliseconds for typical bar lengths.
    """
    n = len(values)
    pcts = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start_idx = max(0, i - window + 1)
        k = i - start_idx + 1
        val_i = values[i]

        if np.isnan(val_i):
            pcts[i] = 50.0
            continue

        if k < min_periods:
            pcts[i] = 50.0
            continue

        count_less = 0.0
        valid_k = 0
        for j in range(start_idx, i + 1):
            val_j = values[j]
            if not np.isnan(val_j):
                valid_k += 1
                diff = val_j - val_i
                if abs(diff) <= 1e-9:
                    count_less += 0.5
                elif diff < 0.0:
                    count_less += 1.0

        if valid_k > 0:
            pcts[i] = (count_less / valid_k) * 100.0
        else:
            pcts[i] = 50.0

    return pcts


@njit(fastmath=True)
def calc_ema(values: np.ndarray, span: int) -> np.ndarray:
    n = len(values)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (span + 1.0)
    ema[0] = values[0]
    for i in range(1, n):
        v = values[i]
        if np.isnan(v):
            ema[i] = ema[i - 1]
        else:
            ema[i] = alpha * v + (1.0 - alpha) * ema[i - 1]
    return ema


@njit(fastmath=True)
def calc_sma(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    sma = np.zeros(n, dtype=np.float64)
    if n == 0:
        return sma
    running_sum = 0.0
    for i in range(n):
        running_sum += values[i]
        if i >= window:
            running_sum -= values[i - window]
            sma[i] = running_sum / window
        else:
            sma[i] = running_sum / (i + 1)
    return sma


@njit(fastmath=True)
def calc_kama(
    values: np.ndarray,
    er_period: int = 10,
    fast_span: int = 2,
    slow_span: int = 30,
) -> np.ndarray:
    """
    Kaufman Adaptive Moving Average (KAMA) computed causally.
    """
    n = len(values)
    kama = np.zeros(n, dtype=np.float64)
    if n == 0:
        return kama

    fast_sc = 2.0 / (fast_span + 1.0)
    slow_sc = 2.0 / (slow_span + 1.0)

    kama[0] = values[0]
    for i in range(1, n):
        if i < er_period:
            kama[i] = values[i]
            continue

        direction = abs(values[i] - values[i - er_period])
        volatility = 0.0
        for j in range(i - er_period + 1, i + 1):
            volatility += abs(values[j] - values[j - 1])

        er = (direction / volatility) if volatility > 1e-12 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (values[i] - kama[i - 1])

    return kama


@njit(fastmath=True)
def calc_kalman_filter(
    values: np.ndarray,
    process_variance: float = 1e-5,
    measurement_variance: float = 1e-2,
) -> np.ndarray:
    """
    1D Constant Velocity Kalman Filter running strictly forward (causal).
    """
    n = len(values)
    filtered = np.zeros(n, dtype=np.float64)
    if n == 0:
        return filtered

    x_est = values[0]
    p_est = 1.0

    filtered[0] = x_est
    for i in range(1, n):
        # Predict
        x_pred = x_est
        p_pred = p_est + process_variance

        # Update
        val = values[i]
        if np.isnan(val):
            filtered[i] = x_pred
            continue

        k_gain = p_pred / (p_pred + measurement_variance)
        x_est = x_pred + k_gain * (val - x_pred)
        p_est = (1.0 - k_gain) * p_pred
        filtered[i] = x_est

    return filtered


@njit(fastmath=True)
def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Causal Wilder's Relative Strength Index (RSI).
    """
    n = len(closes)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n <= period:
        return rsi

    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
