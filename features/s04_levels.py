"""
S4: Horizontal Levels, Prior Day High/Low (PDH/PDL), Swing Clusters, and Round Numbers.
All levels are causally anchored and lagged so that future levels are never visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.costs import get_pair_specs
from features.base import calc_atr, rolling_percentile_fast


def compute_s04_features(df: pd.DataFrame, pair: str, l_confirm: int = 3) -> pd.DataFrame:
    times = pd.to_datetime(df["time"], utc=True)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    atr = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr > 1e-8, atr, 1e-4)

    spec = get_pair_specs(pair)
    pip_unit = float(spec["pip_unit"])
    round_100_step = pip_unit * 100.0
    round_50_step = pip_unit * 50.0

    # 1. Round number distances / ATR
    dist_round100_atr = np.zeros(n, dtype=np.float64)
    dist_round50_atr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = closes[i]
        rem100 = c % round_100_step
        d100 = min(rem100, round_100_step - rem100)
        dist_round100_atr[i] = d100 / atr_safe[i]

        rem50 = c % round_50_step
        d50 = min(rem50, round_50_step - rem50)
        dist_round50_atr[i] = d50 / atr_safe[i]

    # 2. Daily PDH / PDL / PDC (Anchored to 17:00 NY)
    # Group by NY day and compute prior day levels with shift(1)
    df_ny = pd.DataFrame({"time": times, "high": highs, "low": lows, "close": closes})
    ny_times = times.dt.tz_convert("America/New_York")
    # Date adjusted for 17:00 cutoff: if hour >= 17, belongs to next trading date
    adj_dates = (ny_times + pd.Timedelta(hours=7)).dt.date
    df_ny["fx_date"] = adj_dates

    # Daily aggregates
    daily = df_ny.groupby("fx_date").agg({
        "high": "max",
        "low": "min",
        "close": "last",
    }).shift(1)  # Strictly shift(1) so prior day is used!

    daily.columns = ["pdh", "pdl", "pdc"]
    df_merged = df_ny.merge(daily, on="fx_date", how="left")

    pdh = df_merged["pdh"].values
    pdl = df_merged["pdl"].values
    pdc = df_merged["pdc"].values

    dist_pdh_atr = np.where(np.isnan(pdh), 0.0, (pdh - closes) / atr_safe)
    dist_pdl_atr = np.where(np.isnan(pdl), 0.0, (closes - pdl) / atr_safe)

    # Classic Pivots
    pp = (pdh + pdl + pdc) / 3.0
    r1 = 2.0 * pp - pdl
    s1 = 2.0 * pp - pdh
    dist_r1_atr = np.where(np.isnan(r1), 0.0, (r1 - closes) / atr_safe)
    dist_s1_atr = np.where(np.isnan(s1), 0.0, (closes - s1) / atr_safe)

    # 3. Causal Swing Clusters
    # Swings confirmed at t = cand + L
    # We maintain swings within a stationary moving window (max_swing_age = 500 bars)
    # storing (confirmation_bar, price) tuples so older historical levels naturally expire.
    confirmed_swings: list[tuple[int, float]] = []
    dist_cluster_above_atr = np.full(n, 5.0, dtype=np.float64)
    dist_cluster_below_atr = np.full(n, 5.0, dtype=np.float64)
    max_swing_age = 500

    for t in range(n):
        cand = t - l_confirm
        if cand >= l_confirm:
            w_h = highs[cand - l_confirm : t + 1]
            if highs[cand] == np.max(w_h) and np.sum(w_h == highs[cand]) == 1:
                confirmed_swings.append((t, highs[cand]))
            w_l = lows[cand - l_confirm : t + 1]
            if lows[cand] == np.min(w_l) and np.sum(w_l == lows[cand]) == 1:
                confirmed_swings.append((t, lows[cand]))

        # Prune swings older than max_swing_age to eliminate non-stationary drift
        while confirmed_swings and (t - confirmed_swings[0][0] > max_swing_age):
            confirmed_swings.pop(0)

        c = closes[t]
        sw_above = [s for (t_c, s) in confirmed_swings if s > c]
        sw_below = [s for (t_c, s) in confirmed_swings if s < c]

        if sw_above:
            dist_cluster_above_atr[t] = (min(sw_above) - c) / atr_safe[t]
        if sw_below:
            dist_cluster_below_atr[t] = (c - max(sw_below)) / atr_safe[t]

    feats = pd.DataFrame({
        "s4_dist_pdh_atr_pct": rolling_percentile_fast(dist_pdh_atr),
        "s4_dist_pdl_atr_pct": rolling_percentile_fast(dist_pdl_atr),
        "s4_dist_r1_atr_pct": rolling_percentile_fast(dist_r1_atr),
        "s4_dist_s1_atr_pct": rolling_percentile_fast(dist_s1_atr),
        "s4_dist_round100_pct": rolling_percentile_fast(dist_round100_atr),
        "s4_dist_cluster_above_pct": rolling_percentile_fast(dist_cluster_above_atr),
        "s4_dist_cluster_below_pct": rolling_percentile_fast(dist_cluster_below_atr),
    })
    return feats
