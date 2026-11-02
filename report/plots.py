"""
Visualization and Diagnostic Plotting Engine.
Generates publication-grade Matplotlib PNG artifacts:
- v2_null_histogram.png
- multi_pair_heatmap.png
- discovery_curve.png
- best_strategy_live.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_v2_null_histogram(
    real_fitness: float,
    null_fitnesses: list[float],
    p_value: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

    if null_fitnesses:
        ax.hist(null_fitnesses, bins=15, color="#546e7a", alpha=0.7, edgecolor="#263238", label="Null Worlds (V2)")
    ax.axvline(real_fitness, color="#ff1744", linewidth=2.5, linestyle="--", label=f"Real Fitness = {real_fitness:.3f}")

    ax.set_title(f"V2 Synchronous Day-Shuffled Null Test (p = {p_value:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pooled Multi-Pair Fitness")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_multi_pair_heatmap(
    pair_metrics: dict[str, dict[str, float]],
    candidate_ids: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(pair_metrics.keys())
    if not pairs or not candidate_ids:
        return

    # Build matrix of Sharpe ratios
    matrix = np.zeros((len(pairs), len(candidate_ids)), dtype=np.float64)
    for i, p in enumerate(pairs):
        for j, c_id in enumerate(candidate_ids):
            # Lookup Sharpe
            matrix[i, j] = pair_metrics[p].get(c_id, 0.5)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1.0, vmax=2.5)

    ax.set_xticks(range(len(candidate_ids)))
    ax.set_xticklabels(candidate_ids, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=10, fontweight="bold")

    # Annotate numbers
    for i in range(len(pairs)):
        for j in range(len(candidate_ids)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Multi-Pair Cross-Sectional Sharpe Matrix (V6)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Sharpe Ratio (1.5x Costs)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_discovery_curve(
    eval_indices: list[int],
    max_fitness_history: list[float],
    archive_coverage_pct: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)

    color_fit = "#2962ff"
    ax1.set_xlabel("Evaluations (Funnel S2)")
    ax1.set_ylabel("Max Fitness Found", color=color_fit)
    ax1.plot(eval_indices, max_fitness_history, color=color_fit, linewidth=2.0)
    ax1.tick_params(axis="y", labelcolor=color_fit)

    ax2 = ax1.twinx()
    color_cov = "#00c853"
    ax2.set_ylabel("MAP-Elites Archive Coverage (%)", color=color_cov)
    ax2.plot(eval_indices, archive_coverage_pct, color=color_cov, linestyle="--", linewidth=1.8)
    ax2.tick_params(axis="y", labelcolor=color_cov)

    plt.title("QD Quality-Diversity Discovery Trajectory", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_best_strategy_live(
    times: pd.Series,
    equity_curve: np.ndarray,
    strategy_id: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)

    ax.plot(times, equity_curve, color="#00e676", linewidth=2.0, label=f"Live Top: {strategy_id}")
    ax.axhline(100000.0, color="#787b86", linestyle="--", alpha=0.7)
    ax.set_title(f"Best Strategy Performance: {strategy_id}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
