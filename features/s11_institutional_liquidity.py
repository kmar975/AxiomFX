"""
S11: Institutional Liquidity, ADR Exhaustion, ICT Anchors, PWH/PWL, and Equilibrium.
Strictly causal: FX day anchored at 17:00 America/New_York.
Zero lookahead guaranteed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import calc_atr, rolling_percentile_fast


def compute_s11_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    n = len(df)
    times = pd.to_datetime(df["time"], utc=True)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    atr20 = calc_atr(highs, lows, closes, period=20)
    atr_safe = np.where(atr20 > 1e-8, atr20, 1e-4)

    # Convert timestamps to America/New_York
    ny_times = times.dt.tz_convert("America/New_York")
    ny_hours = ny_times.dt.hour.values

    # Shift by 7 hours so that 17:00 NY anchors the new trading day and Sunday 17:00 NY begins the new FX trading week
    ny_adj_times = ny_times + pd.Timedelta(hours=7)
    fx_dates = ny_adj_times.dt.date.values

    # Pre-calculate daily High, Low, Range for each unique FX date
    daily_records: dict[object, dict[str, float]] = {}
    unique_dates = []
    seen = set()
    for d in fx_dates:
        if d not in seen:
            seen.add(d)
            unique_dates.append(d)

    for d in unique_dates:
        mask = (fx_dates == d)
        daily_records[d] = {
            "high": float(np.max(highs[mask])),
            "low": float(np.min(lows[mask])),
            "range": float(np.max(highs[mask]) - np.min(lows[mask])),
        }

    # Weekly mapping adjusted for 17:00 NY cutoff (Sunday 17:00 NY starts the new FX week)
    iso_cal = ny_adj_times.dt.isocalendar()
    week_keys = (iso_cal["year"].values.astype(np.int64) * 100 + iso_cal["week"].values.astype(np.int64))

    unique_weeks: list[int] = []
    seen_weeks = set()
    for w in week_keys:
        if w not in seen_weeks:
            seen_weeks.add(w)
            unique_weeks.append(int(w))

    weekly_records: dict[int, dict[str, float]] = {}
    for w in unique_weeks:
        mask = (week_keys == w)
        weekly_records[w] = {
            "high": float(np.max(highs[mask])),
            "low": float(np.min(lows[mask])),
            "open": float(opens[mask][0]),
        }

    # Build causal daily and weekly historical lookups
    # Compute ADR14 of completed past days
    completed_days_ranges: list[float] = []
    adr_14_by_date: dict[object, float] = {}
    pdl_by_date: dict[object, float] = {}
    pdh_by_date: dict[object, float] = {}
    prev_range_by_date: dict[object, float] = {}

    for d in unique_dates:
        if len(completed_days_ranges) >= 1:
            prev_range = completed_days_ranges[-1]
            prev_range_by_date[d] = prev_range
            pdl_by_date[d] = daily_records[unique_dates[len(completed_days_ranges) - 1]]["low"]
            pdh_by_date[d] = daily_records[unique_dates[len(completed_days_ranges) - 1]]["high"]
        else:
            # First day has no completed prior day: strictly causal NaN (zero look-ahead)
            prev_range_by_date[d] = np.nan
            pdl_by_date[d] = np.nan
            pdh_by_date[d] = np.nan

        if len(completed_days_ranges) >= 14:
            adr14 = float(np.mean(completed_days_ranges[-14:]))
        elif len(completed_days_ranges) > 0:
            adr14 = float(np.mean(completed_days_ranges))
        else:
            # Initial causal proxy strictly from first bar ATR (zero future daily knowledge)
            adr14 = float(atr_safe[0] * 5.0)
        adr_14_by_date[d] = adr14

        completed_days_ranges.append(daily_records[d]["range"])

    # Output feature arrays
    adr_exhaustion = np.zeros(n, dtype=np.float64)
    prior_day_comp = np.zeros(n, dtype=np.float64)
    dist_ny_midnight_atr = np.zeros(n, dtype=np.float64)
    dist_weekly_open_atr = np.zeros(n, dtype=np.float64)
    pwh_pwl_sweep = np.zeros(n, dtype=np.float64)
    pd_equilibrium = np.full(n, 50.0, dtype=np.float64)

    # Running daily high/low state
    cur_date = None
    running_h = -1e9
    running_l = 1e9
    ny_midnight_open = opens[0]
    cur_week = None
    prev_week_h = np.nan
    prev_week_l = np.nan
    weekly_open = opens[0]

    for i in range(n):
        d = fx_dates[i]
        w = week_keys[i]

        if d != cur_date:
            cur_date = d
            running_h = highs[i]
            running_l = lows[i]
        else:
            running_h = max(running_h, highs[i])
            running_l = min(running_l, lows[i])

        # Track 00:00 NY open
        if ny_hours[i] == 0:
            ny_midnight_open = opens[i]

        # Track weekly boundary
        if w != cur_week:
            # New week started
            w_idx = unique_weeks.index(w) if w in unique_weeks else 0
            if w_idx > 0:
                prev_w = unique_weeks[w_idx - 1]
                prev_week_h = weekly_records[prev_w]["high"]
                prev_week_l = weekly_records[prev_w]["low"]
            cur_week = w
            weekly_open = opens[i]

        adr14_val = adr_14_by_date.get(d, 0.01)
        running_range = running_h - running_l

        # 1. ADR exhaustion
        adr_exhaustion[i] = (running_range / max(adr14_val, 1e-4)) * 100.0

        # 2. Prior day compression (< 55% ADR14)
        p_range = prev_range_by_date.get(d, np.nan)
        if np.isnan(p_range):
            prior_day_comp[i] = 0.0
        else:
            prior_day_comp[i] = 1.0 if (p_range < 0.55 * adr14_val) else 0.0

        # 3. Distance from NY Midnight Open
        dist_ny_midnight_atr[i] = (closes[i] - ny_midnight_open) / atr_safe[i]

        # 4. Distance from Weekly Open
        dist_weekly_open_atr[i] = (closes[i] - weekly_open) / atr_safe[i]

        # 5. PWH / PWL Sweep
        if not np.isnan(prev_week_h) and not np.isnan(prev_week_l):
            if lows[i] < prev_week_l and closes[i] > prev_week_l:
                pwh_pwl_sweep[i] = 1.0  # Bullish sweep of PWL
            elif highs[i] > prev_week_h and closes[i] < prev_week_h:
                pwh_pwl_sweep[i] = -1.0  # Bearish sweep of PWH

        # 6. Prior Day Equilibrium
        pdh = pdh_by_date.get(d, np.nan)
        pdl = pdl_by_date.get(d, np.nan)
        if np.isnan(pdh) or np.isnan(pdl):
            pd_equilibrium[i] = 50.0  # Neutral equilibrium on first day (no prior completed day)
        else:
            pd_range = max(pdh - pdl, 1e-5)
            pd_equilibrium[i] = np.clip((closes[i] - pdl) / pd_range * 100.0, 0.0, 100.0)

    feats = pd.DataFrame({
        "s11_adr14_exhaustion_pct": np.clip(adr_exhaustion, 0.0, 200.0),
        "s11_prior_day_compression": prior_day_comp,
        "s11_dist_ny_midnight_open_atr_pct": rolling_percentile_fast(dist_ny_midnight_atr),
        "s11_dist_weekly_open_atr_pct": rolling_percentile_fast(dist_weekly_open_atr),
        "s11_pwh_pwl_sweep": pwh_pwl_sweep,
        "s11_pd_equilibrium_pct": pd_equilibrium,
    })
    return feats
