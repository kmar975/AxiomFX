"""
S9: Cross-Market Context, Triangular Parity Deviations, and Causal Currency Strength.
Decomposes currency strength (EUR, USD, GBP, JPY, AUD, CHF) causally from the 7-pair basket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import rolling_percentile_fast


def _get_aligned_closes(target_df: pd.DataFrame, source_df: pd.DataFrame) -> np.ndarray:
    """Aligns source pair close prices strictly by time to target_df."""
    n = len(target_df)
    if "time" in target_df.columns and "time" in source_df.columns:
        m = pd.merge(target_df[["time"]], source_df[["time", "close"]], on="time", how="left")
        arr = m["close"].ffill().bfill().values.astype(np.float64)
        return arr if len(arr) == n else arr[:n]
    c = source_df["close"].values.astype(np.float64)
    if len(c) >= n:
        return c[:n]
    padded = np.full(n, c[-1] if len(c) > 0 else 1.0, dtype=np.float64)
    padded[:len(c)] = c
    return padded


def compute_s09_features(
    df: pd.DataFrame,
    pair: str,
    basket_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    n = len(df)

    # 1. Triangular Parity Deviation
    # EURGBP * GBPUSD / EURUSD - 1.0
    tri_dev = np.zeros(n, dtype=np.float64)
    if "EURUSD" in basket_data and "GBPUSD" in basket_data and "EURGBP" in basket_data:
        c_eurusd = _get_aligned_closes(df, basket_data["EURUSD"])
        c_gbpusd = _get_aligned_closes(df, basket_data["GBPUSD"])
        c_eurgbp = _get_aligned_closes(df, basket_data["EURGBP"])
        syn_eurusd = c_eurgbp * c_gbpusd
        tri_dev = (syn_eurusd / np.maximum(c_eurusd, 1e-4)) - 1.0

    # 2. Causal Relative Currency Strength
    # Calculate 20-bar percentage return for each pair
    pair_ret_20 = {}
    for p, p_df in basket_data.items():
        c_arr = _get_aligned_closes(df, p_df)
        r20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            r20[i] = (c_arr[i] - c_arr[i - 20]) / c_arr[i - 20]
        pair_ret_20[p] = r20

    # Currencies: EUR, USD, GBP, JPY, AUD, CHF
    # USD strength = average of -EURUSD, -GBPUSD, +USDJPY, -AUDUSD, +USDCHF
    usd_strength = np.zeros(n, dtype=np.float64)
    eur_strength = np.zeros(n, dtype=np.float64)
    gbp_strength = np.zeros(n, dtype=np.float64)
    jpy_strength = np.zeros(n, dtype=np.float64)
    aud_strength = np.zeros(n, dtype=np.float64)
    chf_strength = np.zeros(n, dtype=np.float64)

    if "EURUSD" in pair_ret_20:
        usd_strength -= pair_ret_20["EURUSD"]
        eur_strength += pair_ret_20["EURUSD"]
    if "GBPUSD" in pair_ret_20:
        usd_strength -= pair_ret_20["GBPUSD"]
        gbp_strength += pair_ret_20["GBPUSD"]
    if "USDJPY" in pair_ret_20:
        usd_strength += pair_ret_20["USDJPY"]
        jpy_strength -= pair_ret_20["USDJPY"]
    if "AUDUSD" in pair_ret_20:
        usd_strength -= pair_ret_20["AUDUSD"]
        aud_strength += pair_ret_20["AUDUSD"]
    if "USDCHF" in pair_ret_20:
        usd_strength += pair_ret_20["USDCHF"]
        chf_strength -= pair_ret_20["USDCHF"]

    curr_map = {
        "EUR": eur_strength,
        "USD": usd_strength,
        "GBP": gbp_strength,
        "JPY": jpy_strength,
        "AUD": aud_strength,
        "CHF": chf_strength,
    }

    base_curr = pair[:3]
    quote_curr = pair[3:]
    base_s = curr_map.get(base_curr, np.zeros(n))
    quote_s = curr_map.get(quote_curr, np.zeros(n))

    rel_strength_diff = base_s - quote_s

    feats = pd.DataFrame({
        "s9_rel_strength_diff_pct": rolling_percentile_fast(rel_strength_diff),
        "s9_tri_dev_pct": rolling_percentile_fast(tri_dev),
    })
    return feats
