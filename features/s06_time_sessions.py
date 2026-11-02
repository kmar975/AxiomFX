"""
S6: Time, Sessions, Asian Range Breakouts, and London Judas Swings.
Cyclical cyclical hour/dow transforms and institutional session boundary indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_s06_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    times = pd.to_datetime(df["time"], utc=True)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    hours = times.dt.hour.values
    minutes = times.dt.minute.values
    dows = times.dt.dayofweek.values

    # Cyclical representations rescaled to [0.0 .. 100.0] for standard percentile & threshold compatibility
    hour_sin = (np.sin(2.0 * np.pi * hours / 24.0) + 1.0) * 50.0
    hour_cos = (np.cos(2.0 * np.pi * hours / 24.0) + 1.0) * 50.0
    dow_sin = (np.sin(2.0 * np.pi * dows / 5.0) + 1.0) * 50.0
    dow_cos = (np.cos(2.0 * np.pi * dows / 5.0) + 1.0) * 50.0

    # Determine if timeframe is D1, W1, MN (median bar step >= 20 hours)
    is_daily_or_higher = False
    if n > 1:
        median_step_sec = float(pd.Series(times).diff().dt.total_seconds().median())
        is_daily_or_higher = median_step_sec >= 20.0 * 3600.0

    if is_daily_or_higher:
        # Daily and higher bars encompass all global trading sessions
        session_asia = np.ones(n, dtype=np.float64)
        session_london = np.ones(n, dtype=np.float64)
        session_ny = np.ones(n, dtype=np.float64)
        session_london_fix = np.ones(n, dtype=np.float64)

        asian_breakout = np.zeros(n, dtype=np.float64)
        judas_swing = np.zeros(n, dtype=np.float64)

        # On daily+ bars, breakout & sweep reflect prior period range expansion & liquidity sweep
        for i in range(1, n):
            prev_h = highs[i - 1]
            prev_l = lows[i - 1]
            if closes[i] > prev_h:
                asian_breakout[i] = 1.0
            elif closes[i] < prev_l:
                asian_breakout[i] = -1.0

            if highs[i] > prev_h and closes[i] < prev_h:
                judas_swing[i] = -1.0  # Bearish sweep fade
            elif lows[i] < prev_l and closes[i] > prev_l:
                judas_swing[i] = 1.0   # Bullish sweep fade
    else:
        # Session indicators for intraday bars (M1..H4)
        # Asian: 00:00 - 08:00 UTC
        # London: 07:00 - 16:00 UTC
        # New York: 12:00 - 21:00 UTC
        # London Fix: 15:45 - 16:30 UTC
        session_asia = np.where((hours >= 0) & (hours < 8), 1.0, 0.0)
        session_london = np.where((hours >= 7) & (hours < 16), 1.0, 0.0)
        session_ny = np.where((hours >= 12) & (hours < 21), 1.0, 0.0)
        session_london_fix = np.where(
            ((hours == 15) & (minutes >= 45)) | ((hours == 16) & (minutes <= 30)), 1.0, 0.0
        )

        # Asian Range Breakout & London Judas Swing
        asian_breakout = np.zeros(n, dtype=np.float64)
        judas_swing = np.zeros(n, dtype=np.float64)

        # Track Asian High and Low for each day
        current_day = None
        asia_h = -1e9
        asia_l = 1e9

        for i in range(n):
            day_val = times.iloc[i].date()
            h_val = hours[i]

            if day_val != current_day:
                current_day = day_val
                asia_h = -1e9
                asia_l = 1e9

            # Accumulate Asian range (00:00 - 07:00)
            if 0 <= h_val < 7:
                asia_h = max(asia_h, highs[i])
                asia_l = min(asia_l, lows[i])

            # In London session (07:00 - 12:00), test for breakout / Judas
            if 7 <= h_val <= 12 and asia_h > 0 and asia_l < 1e8:
                # Asian breakout
                if closes[i] > asia_h:
                    asian_breakout[i] = 1.0
                elif closes[i] < asia_l:
                    asian_breakout[i] = -1.0

                # Judas swing: in early London (07:00 - 09:00), price pierces Asian H or L then closes back inside
                if 7 <= h_val <= 9:
                    if highs[i] > asia_h and closes[i] < asia_h:
                        judas_swing[i] = -1.0  # Bearish Judas fade
                    elif lows[i] < asia_l and closes[i] > asia_l:
                        judas_swing[i] = 1.0   # Bullish Judas fade

    feats = pd.DataFrame({
        "s6_hour_sin": hour_sin,
        "s6_hour_cos": hour_cos,
        "s6_dow_sin": dow_sin,
        "s6_dow_cos": dow_cos,
        "s6_session_asia": session_asia,
        "s6_session_london": session_london,
        "s6_session_ny": session_ny,
        "s6_session_london_fix": session_london_fix,
        "s6_asian_breakout": asian_breakout,
        "s6_judas_swing": judas_swing,
    })
    return feats
