"""
S2: Candlestick Structure and Seed Patterns.
Continuous mathematical descriptors + Tier 1 seed pattern detectors.
Includes Fair Value Gaps (FVG) and Prior Consecutive Bar sequences from institutional bank setup.
All zero look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import calc_atr, rolling_percentile_fast


def compute_s02_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    # Continuous descriptors
    body = closes - opens
    body_atr = body / atr_safe
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    candle_range = highs - lows
    range_safe = np.where(candle_range > 1e-8, candle_range, 1e-4)

    upper_wick_atr = upper_wick / atr_safe
    lower_wick_atr = lower_wick / atr_safe
    range_atr = candle_range / atr_safe

    # CLV (0 to 100)
    clv = np.clip((closes - lows) / range_safe * 100.0, 0.0, 100.0)

    # Gap / ATR
    gap_atr = np.zeros(n, dtype=np.float64)
    gap_atr[1:] = (opens[1:] - closes[:-1]) / atr_safe[1:]

    # Relational patterns
    inside_bar = np.zeros(n, dtype=np.float64)
    outside_bar = np.zeros(n, dtype=np.float64)
    engulf_bull = np.zeros(n, dtype=np.float64)
    engulf_bear = np.zeros(n, dtype=np.float64)
    nr4 = np.zeros(n, dtype=np.float64)
    nr7 = np.zeros(n, dtype=np.float64)
    wr7 = np.zeros(n, dtype=np.float64)
    pin_bar_bull = np.zeros(n, dtype=np.float64)
    pin_bar_bear = np.zeros(n, dtype=np.float64)
    doji = np.zeros(n, dtype=np.float64)
    fakey = np.zeros(n, dtype=np.float64)
    three_bar_rev = np.zeros(n, dtype=np.float64)

    # Fair Value Gaps (FVG) and Prior Consecutive Bars
    bull_fvg = np.zeros(n, dtype=np.float64)
    bear_fvg = np.zeros(n, dtype=np.float64)
    two_bear_prior = np.zeros(n, dtype=np.float64)
    three_bear_prior = np.zeros(n, dtype=np.float64)
    two_bull_prior = np.zeros(n, dtype=np.float64)
    three_bull_prior = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        # Inside & Outside
        if highs[i] <= highs[i - 1] and lows[i] >= lows[i - 1]:
            inside_bar[i] = 1.0
        if highs[i] >= highs[i - 1] and lows[i] <= lows[i - 1]:
            outside_bar[i] = 1.0

        # Engulfing
        if closes[i] > opens[i] and closes[i - 1] < opens[i - 1] and closes[i] >= opens[i - 1] and opens[i] <= closes[i - 1]:
            engulf_bull[i] = 1.0
        if closes[i] < opens[i] and closes[i - 1] > opens[i - 1] and closes[i] <= opens[i - 1] and opens[i] >= closes[i - 1]:
            engulf_bear[i] = 1.0

        # Doji
        if abs(body[i]) <= 0.10 * candle_range[i]:
            doji[i] = 1.0

        # Pin Bar
        if lower_wick[i] >= 2.0 * abs(body[i]) and clv[i] >= 65.0:
            pin_bar_bull[i] = 1.0
        if upper_wick[i] >= 2.0 * abs(body[i]) and clv[i] <= 35.0:
            pin_bar_bear[i] = 1.0

        # NR4, NR7, WR7
        if i >= 4 and candle_range[i] <= np.min(candle_range[i - 3 : i]):
            nr4[i] = 1.0
        if i >= 7:
            if candle_range[i] <= np.min(candle_range[i - 6 : i]):
                nr7[i] = 1.0
            if candle_range[i] >= np.max(candle_range[i - 6 : i]):
                wr7[i] = 1.0

        # Fakey (Inside bar at i-1, false breakout at i)
        # Bullish Fakey (+1.0): false break of mother low with close back inside
        # Bearish Fakey (-1.0): false break of mother high with close back inside
        if i >= 2 and inside_bar[i - 1] == 1.0:
            if lows[i] < lows[i - 1] and closes[i] > lows[i - 1]:
                fakey[i] = 1.0
            elif highs[i] > highs[i - 1] and closes[i] < highs[i - 1]:
                fakey[i] = -1.0

        # Three-bar reversal
        if i >= 2:
            if lows[i - 1] < lows[i - 2] and lows[i - 1] < lows[i] and closes[i] > highs[i - 1]:
                three_bar_rev[i] = 1.0
            elif highs[i - 1] > highs[i - 2] and highs[i - 1] > highs[i] and closes[i] < lows[i - 1]:
                three_bar_rev[i] = -1.0

        # Fair Value Gap (FVG) across [i-2, i]
        if i >= 2:
            if lows[i] > highs[i - 2]:
                bull_fvg[i] = 1.0
            elif highs[i] < lows[i - 2]:
                bear_fvg[i] = 1.0

        # Consecutive bars prior to i (strictly causal: checks i-1, i-2, i-3)
        if i >= 2:
            is_two_bear = (closes[i - 1] < opens[i - 1]) and (closes[i - 2] < opens[i - 2])
            is_two_bull = (closes[i - 1] > opens[i - 1]) and (closes[i - 2] > opens[i - 2])
            if is_two_bear:
                two_bear_prior[i] = 1.0
                if i >= 3 and (closes[i - 3] < opens[i - 3]):
                    three_bear_prior[i] = 1.0
            if is_two_bull:
                two_bull_prior[i] = 1.0
                if i >= 3 and (closes[i - 3] > opens[i - 3]):
                    three_bull_prior[i] = 1.0

    # Marubozu (body > 70% and > 80% with candle_range > 1.2 * atr)
    marubozu_bull_70 = ((closes > opens) & (np.abs(body) >= 0.70 * candle_range) & (candle_range >= 1.2 * atr_safe)).astype(np.float64)
    marubozu_bear_70 = ((closes < opens) & (np.abs(body) >= 0.70 * candle_range) & (candle_range >= 1.2 * atr_safe)).astype(np.float64)
    marubozu_bull_80 = ((closes > opens) & (np.abs(body) >= 0.80 * candle_range) & (candle_range >= 1.2 * atr_safe)).astype(np.float64)
    marubozu_bear_80 = ((closes < opens) & (np.abs(body) >= 0.80 * candle_range) & (candle_range >= 1.2 * atr_safe)).astype(np.float64)

    # Double Inside Bar (prior to i: bar i-1 inside i-2, and bar i-2 inside i-3)
    double_inside_bar = np.zeros(n, dtype=np.float64)
    for i in range(3, n):
        if highs[i - 1] < highs[i - 2] and lows[i - 1] > lows[i - 2] and highs[i - 2] < highs[i - 3] and lows[i - 2] > lows[i - 3]:
            double_inside_bar[i] = 1.0

    feats = pd.DataFrame({
        "s2_body_atr_pct": rolling_percentile_fast(body_atr),
        "s2_upper_wick_pct": rolling_percentile_fast(upper_wick_atr),
        "s2_lower_wick_pct": rolling_percentile_fast(lower_wick_atr),
        "s2_range_atr_pct": rolling_percentile_fast(range_atr),
        "s2_clv": clv,
        "s2_gap_atr_pct": rolling_percentile_fast(gap_atr),
        "s2_inside_bar": inside_bar,
        "s2_outside_bar": outside_bar,
        "s2_double_inside_bar": double_inside_bar,
        "s2_marubozu_bull_70": marubozu_bull_70,
        "s2_marubozu_bear_70": marubozu_bear_70,
        "s2_marubozu_bull_80": marubozu_bull_80,
        "s2_marubozu_bear_80": marubozu_bear_80,
        "s2_engulf_bull": engulf_bull,
        "s2_engulf_bear": engulf_bear,
        "s2_nr4": nr4,
        "s2_nr7": nr7,
        "s2_wr7": wr7,
        "s2_seed_pin_bull": pin_bar_bull,
        "s2_seed_pin_bear": pin_bar_bear,
        "s2_seed_doji": doji,
        "s2_seed_fakey": fakey,
        "s2_seed_three_bar_rev": three_bar_rev,
        "s2_bull_fvg": bull_fvg,
        "s2_bear_fvg": bear_fvg,
        "s2_two_consecutive_bear_prior": two_bear_prior,
        "s2_three_consecutive_bear_prior": three_bear_prior,
        "s2_two_consecutive_bull_prior": two_bull_prior,
        "s2_three_consecutive_bull_prior": three_bull_prior,
    })
    return feats
