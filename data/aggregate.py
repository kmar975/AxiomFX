"""
Resampling engine for Forex candlesticks.
Supports 8 timeframes: M1, M5, M15, H1, H4, D1, W1, MN.
Daily D1 bars anchor at 17:00 America/New_York (5:00 PM NY).
"""

from __future__ import annotations

import pandas as pd

from data.process_and_clean_timeframes import clean_and_enforce_continuity

TF_RESAMPLE_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "24h",
    "W1": "W-FRI",
    "MN": "MS",
}


def resample_ohlcv(
    df: pd.DataFrame,
    target_tf: str,
    pair: str,
    anchor_hour_ny: int = 17,
) -> pd.DataFrame:
    """
    Resamples dataframe to target timeframe.
    For D1, anchors to 17:00 NY (offset of 22:00 or 21:00 UTC depending on DST).
    """
    if target_tf not in TF_RESAMPLE_RULES:
        raise ValueError(f"Unsupported timeframe: {target_tf}")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    elif df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    df = df.sort_values("time").reset_index(drop=True)
    df.set_index("time", inplace=True)

    if target_tf == "D1":
        # Anchor at 17:00 America/New_York (24h FX day boundary with robust DST handling)
        df_ny = df.tz_convert("America/New_York")
        df_shifted = df_ny.copy()
        df_shifted.index = df_shifted.index - pd.Timedelta(hours=anchor_hour_ny)
        resampled = df_shifted.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "spread": "mean",
            "tick_volume": "sum",
        }).dropna(subset=["close"])
        resampled.index = (resampled.index + pd.Timedelta(hours=anchor_hour_ny)).tz_convert("UTC")
    elif target_tf == "W1":
        resampled = df.resample("W-FRI").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "spread": "mean",
            "tick_volume": "sum",
        }).dropna(subset=["close"])
    elif target_tf == "MN":
        resampled = df.resample("MS").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "spread": "mean",
            "tick_volume": "sum",
        }).dropna(subset=["close"])
    else:
        rule = TF_RESAMPLE_RULES[target_tf]
        resampled = df.resample(rule, closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "spread": "mean",
            "tick_volume": "sum",
        }).dropna(subset=["close"])

    resampled = resampled.reset_index()
    resampled = clean_and_enforce_continuity(resampled, pair)
    return resampled
