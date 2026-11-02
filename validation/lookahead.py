"""
V1 Validation Test: Strict Zero Look-Ahead Verification.
Asserts that truncating timeseries at any bar T produces identical feature values
for all bars <= T compared to computing features on the full timeseries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.store import FeatureStore


def verify_zero_lookahead_for_pair(
    df: pd.DataFrame,
    pair: str,
    cut_idx: int,
    basket_data: dict[str, pd.DataFrame],
    tolerance: float = 1e-4,
) -> tuple[bool, dict[str, float]]:
    """
    Computes features on full dataframe, then on truncated dataframe df[:cut_idx+1],
    and compares every single feature column for all rows <= cut_idx.
    """
    feature_store = FeatureStore()

    # 1. Full features
    full_feats = feature_store.compute_all_features_for_pair(df, pair, basket_data)

    # 2. Truncated basket
    truncated_basket = {p: p_df.iloc[: cut_idx + 1].copy() for p, p_df in basket_data.items()}
    trunc_df = df.iloc[: cut_idx + 1].copy()

    trunc_feats = feature_store.compute_all_features_for_pair(trunc_df, pair, truncated_basket)

    max_diffs = {}
    all_matched = True

    for col in full_feats.columns:
        full_vals = full_feats[col].iloc[: cut_idx + 1].values
        trunc_vals = trunc_feats[col].values

        # Filter out NaN comparisons
        valid_mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)
        if np.sum(valid_mask) > 0:
            diff = float(np.max(np.abs(full_vals[valid_mask] - trunc_vals[valid_mask])))
        else:
            diff = 0.0

        max_diffs[col] = diff
        if diff > tolerance:
            all_matched = False

    return all_matched, max_diffs
