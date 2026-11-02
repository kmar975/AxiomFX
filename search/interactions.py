"""
Phase B2: Feature Interactions, 5x5 Conditional Grids, and Seed Rule Mining.
Identifies synergistic orthogonal feature pairs and extracts candidate seed rules.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from grammar.strategy import (
    Condition,
    ExitSpec,
    Strategy,
    is_binary_feature,
    is_signed_directional_feature,
)


def _bin_feature(f: np.ndarray, feat_name: str = "", n_bins: int = 5) -> np.ndarray:
    """Bins a feature into 0..n_bins-1 adapting to binary, signed, or continuous percentiles."""
    if feat_name:
        if is_signed_directional_feature(feat_name):
            mid = n_bins // 2
            return np.where(f < -0.1, 0, np.where(f > 0.1, n_bins - 1, mid)).astype(int)
        elif is_binary_feature(feat_name):
            return np.where(f > 0.5, n_bins - 1, 0).astype(int)

    # Auto-detect from unique values if name not given or unknown
    valid_f = f[~np.isnan(f)]
    if len(valid_f) > 0:
        unique_vals = np.unique(valid_f)
        if len(unique_vals) <= 3 and np.all(np.isin(unique_vals, [-1.0, 0.0, 1.0])):
            mid = n_bins // 2
            return np.where(f < -0.1, 0, np.where(f > 0.1, n_bins - 1, mid)).astype(int)
        if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0])):
            return np.where(f > 0.5, n_bins - 1, 0).astype(int)

    min_val = np.nanmin(f) if len(f) > 0 else 0.0
    max_val = np.nanmax(f) if len(f) > 0 else 100.0
    if 0.0 <= min_val and max_val <= 100.0 and max_val > 1.0:
        return np.clip((f / 20.0).astype(int), 0, n_bins - 1)
    else:
        span = max(max_val - min_val, 1e-6)
        scaled = (f - min_val) / span
        return np.clip((scaled * n_bins).astype(int), 0, n_bins - 1)


def _make_condition(feat: str, src: str, bin_idx: int, n_bins: int = 5) -> Condition:
    """Generates an appropriate Condition (binary, signed, or continuous) for a low/high bin."""
    is_high = (bin_idx >= n_bins // 2)
    if is_signed_directional_feature(feat):
        thresh = 1.0 if is_high else -1.0
        return Condition(feat, src, "event", thresh)
    elif is_binary_feature(feat):
        op = "gt" if is_high else "lt"
        return Condition(feat, src, op, 0.5)
    else:
        op = "gt" if is_high else "lt"
        thresh = 70.0 if is_high else 30.0
        return Condition(feat, src, op, thresh)


def compute_conditional_2d_grid(
    feat1: np.ndarray,
    feat2: np.ndarray,
    target_returns: np.ndarray,
    n_bins: int = 5,
    feat1_name: str = "",
    feat2_name: str = "",
) -> np.ndarray:
    """
    Computes 5x5 conditional mean future return grid for two features.
    """
    grid = np.zeros((n_bins, n_bins), dtype=np.float64)
    counts = np.zeros((n_bins, n_bins), dtype=np.int64)

    # Bin features into quantiles (0..4)
    valid = ~np.isnan(feat1) & ~np.isnan(feat2) & ~np.isnan(target_returns)
    f1 = feat1[valid]
    f2 = feat2[valid]
    ret = target_returns[valid]

    if len(f1) < 50:
        return grid

    b1 = _bin_feature(f1, feat1_name, n_bins)
    b2 = _bin_feature(f2, feat2_name, n_bins)

    for i in range(len(f1)):
        grid[b1[i], b2[i]] += ret[i]
        counts[b1[i], b2[i]] += 1

    mean_grid = np.where(counts > 5, grid / np.maximum(counts, 1), 0.0)
    return mean_grid


def mine_interaction_seed_rules(
    features_df: pd.DataFrame,
    forward_returns: np.ndarray,
    top_features: list[str],
    max_rules: int = 60,
) -> list[Strategy]:
    """
    Scans pairs of orthogonal top features for extreme quadrants (win rate / mean return)
    and extracts seed rules. Uses expanded top_features pool (up to 25) and relaxed edge threshold (0.05).
    """
    seed_rules: list[Strategy] = []
    rule_idx = 0

    def _extract_src(feat: str) -> str:
        if feat.startswith("htf_"):
            parts = feat.split("_")
            return f"{parts[0]}_{parts[1]}"
        return feat.split("_")[0]

    # Group features by source
    pairs = list(itertools.combinations(top_features[:25], 2))

    for f1, f2 in pairs:
        src1 = _extract_src(f1)
        src2 = _extract_src(f2)
        if src1 == src2:
            continue  # Enforce distinct orthogonal sources

        if f1 not in features_df.columns or f2 not in features_df.columns:
            continue

        arr1 = features_df[f1].values
        arr2 = features_df[f2].values
        grid = compute_conditional_2d_grid(arr1, arr2, forward_returns, feat1_name=f1, feat2_name=f2)

        # Check corners: (0,0), (0,4), (4,0), (4,4)
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]

        for b1, b2 in corners:
            val = grid[b1, b2]
            if abs(val) > 0.05:  # Significant directional edge
                rule_idx += 1
                cond1 = _make_condition(f1, src1, b1, 5)
                cond2 = _make_condition(f2, src2, b2, 5)
                cond2.direction_mode = "fade" if val < -0.05 else "continuation"

                strat = Strategy(
                    strategy_id=f"B2_SEED_{rule_idx}",
                    family="F15_Ensemble",
                    context=cond1,
                    trigger=cond2,
                    exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
                )
                if strat.validate_rules():
                    seed_rules.append(strat)
                    if len(seed_rules) >= max_rules:
                        return seed_rules

    return seed_rules
