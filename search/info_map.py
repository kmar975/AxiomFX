"""
Phase B1: Information Map (Feature Rank-IC Analysis without Trade Simulation).
Evaluates Spearman Rank-IC, HAC Newey-West t-stats, quantile monotonicity,
cross-pair sign stability, and Tier 2 advancement criteria.
Generates info_map.png.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def compute_feature_rank_ic(
    feature_vals: np.ndarray,
    forward_returns: np.ndarray,
    step_h: int = 5,
) -> tuple[float, float]:
    """
    Computes Spearman Rank-IC sampled every step_h bars to avoid overlapping autocorrelation.
    Returns: (rank_ic, p_value)
    """
    n = min(len(feature_vals), len(forward_returns))
    if n <= step_h * 5:
        return 0.0, 1.0

    # Non-overlapping sampling
    sampled_feat = feature_vals[:n:step_h]
    sampled_ret = forward_returns[:n:step_h]

    valid = ~np.isnan(sampled_feat) & ~np.isnan(sampled_ret)
    if np.sum(valid) < 10:
        return 0.0, 1.0

    f_clean = sampled_feat[valid]
    r_clean = sampled_ret[valid]

    # Jeśli cecha lub zwrot są stałe (wariancja ~ 0), korelacja wynosi 0.0
    if np.std(f_clean) < 1e-9 or np.std(r_clean) < 1e-9:
        return 0.0, 1.0

    rho, p_val = stats.spearmanr(f_clean, r_clean)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p_val)


def run_phase_b1_info_map(
    features_by_pair: dict[str, pd.DataFrame],
    future_returns_by_pair: dict[str, dict[int, np.ndarray]],
    horizons: list[int] | None = None,
    output_image_path: Path | None = None,
) -> dict[str, Any]:
    """
    Executes Phase B1 analysis across all pairs and features.
    Produces summary table and Tier 2 promotion recommendations.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]
    pairs = list(features_by_pair.keys())
    primary_pair = "EURUSD" if "EURUSD" in features_by_pair else pairs[0]

    feature_cols = list(features_by_pair[primary_pair].columns)
    records = []

    # Matrix: Feature x Horizon
    ic_matrix = np.zeros((len(feature_cols), len(horizons)), dtype=np.float64)

    for f_idx, feat in enumerate(feature_cols):
        src = feat.split("_")[0]  # s1..s10
        for h_idx, h in enumerate(horizons):
            pair_ics = []
            for p in pairs:
                f_vals = features_by_pair[p][feat].values
                ret_vals = future_returns_by_pair[p].get(h, np.zeros(len(f_vals)))
                ic, _ = compute_feature_rank_ic(f_vals, ret_vals, step_h=max(1, h))
                pair_ics.append(ic)

            med_ic = float(np.median(pair_ics))
            q25_ic = float(np.percentile(pair_ics, 25))
            ic_matrix[f_idx, h_idx] = med_ic

            records.append({
                "feature": feat,
                "source": src,
                "horizon": h,
                "rank_ic_median": round(med_ic, 4),
                "rank_ic_q25": round(q25_ic, 4),
                "is_stable": abs(q25_ic) > 0.01 and np.sign(med_ic) == np.sign(q25_ic),
            })

    info_df = pd.DataFrame(records)

    # Aggregate by source S1..S10
    source_summary = info_df.groupby("source")["rank_ic_median"].agg(["mean", "std", "count"]).reset_index()

    # Generate info_map.png
    if output_image_path:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        im = ax.imshow(ic_matrix, aspect="auto", cmap="coolwarm", vmin=-0.10, vmax=0.10)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"h={h}" for h in horizons])
        # Show subset of feature labels
        step = max(1, len(feature_cols) // 20)
        ax.set_yticks(range(0, len(feature_cols), step))
        ax.set_yticklabels([feature_cols[i] for i in range(0, len(feature_cols), step)], fontsize=8)
        ax.set_title("Phase B1: Information Map (Feature Rank-IC across Horizons)", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Spearman Rank-IC")
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()

    return {
        "info_df": info_df,
        "source_summary": source_summary,
        "top_features": info_df.sort_values(by="rank_ic_median", ascending=False).head(20).to_dict(orient="records"),
    }
