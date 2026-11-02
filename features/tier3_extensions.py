"""
Tier 3 Experimental & Subjective Extensions (§ 4 & config/library_tiers.yaml).
Includes Harmonic Patterns (Gartley, Bat, Butterfly), Cup & Handle,
and Smart Money Concepts (SMC: Order Blocks, Fair Value Gaps, Premium/Discount).
Strictly causal with zero look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import calc_atr, rolling_percentile_fast


def compute_tier3_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Computes Tier 3 experimental features when enabled by --enable-tier3 flag.
    """
    n = len(df)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    atr20 = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr20 > 1e-8, atr20, 1e-4)

    # 1. SMC: Fair Value Gaps (FVG)
    # Bullish FVG: Low of candle i > High of candle i-2 by at least 0.2 ATR
    # Bearish FVG: High of candle i < Low of candle i-2 by at least 0.2 ATR
    smc_fvg = np.zeros(n, dtype=np.float64)
    for i in range(2, n):
        if lows[i] > highs[i - 2] + 0.2 * atr_safe[i]:
            smc_fvg[i] = 1.0  # Bullish FVG imbalance
        elif highs[i] < lows[i - 2] - 0.2 * atr_safe[i]:
            smc_fvg[i] = -1.0 # Bearish FVG imbalance

    # 2. SMC: Order Blocks (OB)
    # Bullish OB: last down-candle before strong 2-bar bullish impulse (> 1.5 ATR)
    # Bearish OB: last up-candle before strong 2-bar bearish impulse (> 1.5 ATR)
    smc_order_blocks = np.zeros(n, dtype=np.float64)
    for i in range(3, n):
        impulse_up = (closes[i] - opens[i - 1]) / atr_safe[i]
        impulse_down = (opens[i - 1] - closes[i]) / atr_safe[i]
        if impulse_up > 1.5 and closes[i - 2] < opens[i - 2]:
            smc_order_blocks[i] = 1.0
        elif impulse_down > 1.5 and closes[i - 2] > opens[i - 2]:
            smc_order_blocks[i] = -1.0

    # 3. SMC: Premium / Discount Equilibrium
    # Rolling 40-bar range: 0-100% position (50% = Equilibrium, >70% = Premium, <30% = Discount)
    smc_premium_discount = np.full(n, 50.0, dtype=np.float64)
    for i in range(40, n):
        w_h = np.max(highs[i - 40 : i + 1])
        w_l = np.min(lows[i - 40 : i + 1])
        rng = max(w_h - w_l, 1e-5)
        smc_premium_discount[i] = np.clip((closes[i] - w_l) / rng * 100.0, 0.0, 100.0)

    # 4. Harmonic Patterns: Gartley, Bat, Butterfly
    # Causal swing detection on 5-bar pivots
    harmonic_gartley = np.zeros(n, dtype=np.float64)
    harmonic_bat = np.zeros(n, dtype=np.float64)
    harmonic_butterfly = np.zeros(n, dtype=np.float64)

    swings: list[tuple[int, float, int]] = []  # (bar_idx, price, +1 for high, -1 for low)
    l_piv = 4
    for t in range(l_piv * 2, n):
        cand = t - l_piv
        w_h = highs[cand - l_piv : t + 1]
        w_l = lows[cand - l_piv : t + 1]
        if highs[cand] == np.max(w_h) and (not swings or swings[-1][2] != 1):
            swings.append((cand, highs[cand], 1))
        elif lows[cand] == np.min(w_l) and (not swings or swings[-1][2] != -1):
            swings.append((cand, lows[cand], -1))

        # Check last 4 swings: X, A, B, C and current close as candidate D
        if len(swings) >= 4:
            s_x, p_x, dir_x = swings[-4]
            s_a, p_a, dir_a = swings[-3]
            s_b, p_b, dir_b = swings[-2]
            s_c, p_c, dir_c = swings[-1]

            xa = abs(p_a - p_x)
            if xa > 1e-5:
                ab_retrace = abs(p_b - p_a) / xa
                cd_retrace = abs(closes[t] - p_c) / max(1e-5, abs(p_c - p_b))

                # Gartley: AB ~ 0.618 of XA (0.55..0.68)
                if 0.55 <= ab_retrace <= 0.68 and 0.70 <= cd_retrace <= 0.88:
                    harmonic_gartley[t] = 1.0 if dir_a == -1 else -1.0

                # Bat: AB ~ 0.382 - 0.50 of XA
                if 0.35 <= ab_retrace <= 0.52 and 0.80 <= cd_retrace <= 0.95:
                    harmonic_bat[t] = 1.0 if dir_a == -1 else -1.0

                # Butterfly: AB ~ 0.786 of XA, CD extension 1.27
                if 0.72 <= ab_retrace <= 0.85 and 1.20 <= cd_retrace <= 1.35:
                    harmonic_butterfly[t] = 1.0 if dir_a == -1 else -1.0

    # 5. Cup and Handle
    # U-shaped base (20 bars) followed by 5-bar tight handle consolidation
    cup_and_handle = np.zeros(n, dtype=np.float64)
    for i in range(25, n):
        cup_high_left = highs[i - 25]
        cup_low = np.min(lows[i - 20 : i - 5])
        cup_high_right = highs[i - 5]
        handle_low = np.min(lows[i - 5 : i + 1])
        handle_high = np.max(highs[i - 5 : i + 1])

        depth = cup_high_left - cup_low
        if depth > 1.0 * atr_safe[i] and abs(cup_high_left - cup_high_right) < 0.3 * depth:
            if handle_low > cup_low + 0.5 * depth and handle_high <= cup_high_right + 0.1 * atr_safe[i]:
                if closes[i] > cup_high_right:
                    cup_and_handle[i] = 1.0  # Breakout from handle

    feats = pd.DataFrame({
        "tier3_smc_fair_value_gaps": smc_fvg,
        "tier3_smc_order_blocks": smc_order_blocks,
        "tier3_smc_premium_discount": smc_premium_discount,
        "tier3_harmonic_gartley": harmonic_gartley,
        "tier3_harmonic_bat": harmonic_bat,
        "tier3_harmonic_butterfly": harmonic_butterfly,
        "tier3_cup_and_handle": cup_and_handle,
    })
    return feats
