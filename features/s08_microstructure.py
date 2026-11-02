"""
S8: Microstructure Proxies, Spread/Volume Deviations, Directional Efficiency, and Choppiness.
Calculates diurnal-adjusted deviations and intra-bar efficiency metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features.base import calc_atr, rolling_percentile_fast


def compute_s08_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    times = pd.to_datetime(df["time"], utc=True)
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    spreads = df["spread"].values.astype(np.float64) if "spread" in df.columns else np.full(n, 1.5)
    tick_volumes = df["tick_volume"].values.astype(np.float64) if "tick_volume" in df.columns else np.full(n, 500.0)

    # Directional efficiency
    candle_range = highs - lows
    safe_range = np.where(candle_range > 1e-8, candle_range, 1e-4)
    dir_efficiency = np.abs(closes - opens) / safe_range * 100.0

    # Diurnal time-of-day adjustment for spread (or diurnal range expansion if spread is constant) and volume
    hours = times.dt.hour.values
    spread_dev = np.zeros(n, dtype=np.float64)
    tvol_dev = np.zeros(n, dtype=np.float64)
    spread_is_constant = bool(np.std(spreads) < 1e-5)

    # Rolling hour baseline: compare current spread/range to 20-day mean for that specific hour
    for i in range(n):
        cur_h = hours[i]
        # Lookback up to 480 bars (~20 days for H1)
        start_idx = max(0, i - 480)
        mask = (hours[start_idx : i + 1] == cur_h)
        h_spreads = spreads[start_idx : i + 1][mask]
        h_tvols = tick_volumes[start_idx : i + 1][mask]
        h_ranges = candle_range[start_idx : i + 1][mask]

        if len(h_spreads) > 3:
            if spread_is_constant:
                # Diurnal range expansion proxy: compare current candle range to 20-day hourly mean
                m_cr = np.mean(h_ranges)
                spread_dev[i] = candle_range[i] - m_cr
            else:
                m_sp = np.mean(h_spreads)
                spread_dev[i] = spreads[i] - m_sp

            m_tv = np.mean(h_tvols)
            tvol_dev[i] = (tick_volumes[i] - m_tv) / (m_tv + 1e-4)

    # Choppiness Index (20 bars)
    # CI = 100 * LOG10( SUM(ATR(1), n) / (MaxHigh(n) - MinLow(n)) ) / LOG10(n)
    tr1 = calc_atr(highs, lows, closes, period=1)
    choppiness_20 = np.full(n, 50.0, dtype=np.float64)
    log_20 = math.log10(20.0)

    for i in range(20, n):
        sum_tr = np.sum(tr1[i - 20 + 1 : i + 1])
        hh = np.max(highs[i - 20 + 1 : i + 1])
        ll = np.min(lows[i - 20 + 1 : i + 1])
        if (hh - ll) > 1e-8 and sum_tr > 1e-8:
            ci = 100.0 * (math.log10(sum_tr / (hh - ll)) / log_20)
            choppiness_20[i] = np.clip(ci, 0.0, 100.0)

    # VSA Tick Volume Spikes and Patterns (vs SMA50)
    vol_sma50 = pd.Series(tick_volumes).rolling(50, min_periods=1).mean().values
    vol_sma50_safe = np.where(vol_sma50 > 1e-4, vol_sma50, 1e-4)

    vol_spike_1p5 = (tick_volumes > 1.5 * vol_sma50_safe).astype(np.float64)
    vol_spike_2p0 = (tick_volumes > 2.0 * vol_sma50_safe).astype(np.float64)
    vol_spike_2p5 = (tick_volumes > 2.5 * vol_sma50_safe).astype(np.float64)
    vol_spike_3p0 = (tick_volumes > 3.0 * vol_sma50_safe).astype(np.float64)

    high_vol_bull = ((tick_volumes > 1.5 * vol_sma50_safe) & (closes > opens)).astype(np.float64)
    high_vol_bear = ((tick_volumes > 1.5 * vol_sma50_safe) & (closes < opens)).astype(np.float64)
    low_vol_pullback = (tick_volumes < 0.7 * vol_sma50_safe).astype(np.float64)

    feats = pd.DataFrame({
        "s8_spread_dev_pct": rolling_percentile_fast(spread_dev),
        "s8_tvol_dev_pct": rolling_percentile_fast(tvol_dev),
        "s8_dir_efficiency": dir_efficiency,
        "s8_choppiness_20": choppiness_20,
        "s8_volume_spike_1p5x": vol_spike_1p5,
        "s8_volume_spike_2p0x": vol_spike_2p0,
        "s8_volume_spike_2p5x": vol_spike_2p5,
        "s8_volume_spike_3p0x": vol_spike_3p0,
        "s8_high_volume_bull_bar": high_vol_bull,
        "s8_high_volume_bear_bar": high_vol_bear,
        "s8_low_volume_pullback": low_vol_pullback,
    })
    return feats
