"""
Portfolio Construction Layer: Hierarchical Risk Parity (HRP) and Cluster Diversification.
Groups refined elites by return correlation (< 0.40), selects one champion per cluster,
allocates weights via Hierarchical Risk Parity, and exports output/candidates.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import yaml

from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy


def get_cluster_var(cov_matrix: np.ndarray, cluster_indices: list[int]) -> float:
    """
    Computes portfolio variance of an inverse-variance weighted sub-cluster (de Prado HRP).
    """
    sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
    inv_diag = 1.0 / np.maximum(np.diag(sub_cov), 1e-8)
    w = inv_diag / np.sum(inv_diag)
    return float(np.dot(w.T, np.dot(sub_cov, w)))


def compute_hrp_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Computes Hierarchical Risk Parity (HRP) portfolio weights from covariance matrix.
    Uses distance d = sqrt(0.5 * (1 - corr)), quasi-diagonalization, and recursive bisection.
    """
    n = cov_matrix.shape[0]
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    # Ridge regularization to prevent zero-variance singularities
    cov_matrix = cov_matrix + np.eye(n) * 1e-7

    # 1. Correlation and Distance Matrix
    std = np.sqrt(np.diag(cov_matrix))
    std_inv = np.where(std > 1e-8, 1.0 / std, 1.0)
    corr = cov_matrix * np.outer(std_inv, std_inv)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))

    # Condense distance matrix for linkage
    condensed_dist = ssd.squareform(dist, checks=False)
    link = sch.linkage(condensed_dist, method="single")

    # 2. Quasi-Diagonalization (leaf order)
    def get_quasi_diag(link_mat):
        root = link_mat[-1, :2].astype(int)
        order = [root[0], root[1]]
        num_items = n
        while any(i >= num_items for i in order):
            new_order = []
            for item in order:
                if item < num_items:
                    new_order.append(item)
                else:
                    cluster_idx = item - num_items
                    new_order.extend(link_mat[cluster_idx, :2].astype(int))
            order = new_order
        return order

    sorted_indices = get_quasi_diag(link)

    # 3. Recursive Bisection
    weights = pd.Series(1.0, index=sorted_indices)
    clusters = [sorted_indices]

    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                mid = len(cluster) // 2
                c1 = cluster[:mid]
                c2 = cluster[mid:]

                # Cluster variance (de Prado inverse-variance portfolio variance)
                v1 = get_cluster_var(cov_matrix, c1)
                v2 = get_cluster_var(cov_matrix, c2)

                alpha1 = 1.0 - (v1 / (v1 + v2))
                alpha2 = 1.0 - alpha1

                weights.loc[c1] *= alpha1
                weights.loc[c2] *= alpha2

                new_clusters.append(c1)
                new_clusters.append(c2)
        clusters = new_clusters

    w_arr = np.zeros(n, dtype=np.float64)
    for idx, w in weights.items():
        w_arr[idx] = w

    # Normalize
    sum_w = np.sum(w_arr)
    return w_arr / (sum_w if sum_w > 0 else 1.0)


def _to_native(obj: Any) -> Any:
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def build_hrp_candidate_portfolio(
    elites: list[tuple[Strategy, float, dict[str, Any]]],
    dev_data: dict[str, pd.DataFrame],
    features_by_pair: dict[str, pd.DataFrame],
    max_correlation_cluster: float = 0.40,
    min_candidates: int = 5,
    max_candidates: int = 15,
    output_yaml_path: Path = Path("output/candidates.yaml"),
    require_profitable: bool = False,
    cost_multiplier: float = 1.5,
) -> tuple[list[Strategy], np.ndarray, dict[str, Any]]:
    """
    Selects 5-15 diverse cluster champions, computes HRP weights, and writes output/candidates.yaml.
    """
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    eur_df = dev_data["EURUSD"]
    n_bars = len(eur_df)
    q_vec = compute_quote_to_usd_vector("EURUSD", n_bars, {p: d["close"].values for p, d in dev_data.items()})

    # 1. Extract daily return vectors on EURUSD
    valid_elites = []

    for strat, fit, details in elites:
        # Strictly enforce positive fitness
        if fit <= 0.0:
            continue

        l_mask, s_mask = compile_strategy(strat, features_by_pair["EURUSD"])
        res = run_vectorized_backtest(
            df=eur_df,
            pair="EURUSD",
            entry_longs=l_mask,
            entry_shorts=s_mask,
            quote_to_usd=q_vec,
            sl_atr_mult=strat.exit.sl_atr,
            tp_atr_mult=strat.exit.tp_atr,
            max_holding_bars=strat.exit.max_holding_bars,
            cost_multiplier=cost_multiplier,
            exit_type=getattr(strat.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(strat.exit, "trailing_atr", 1.2),
        )
        eq = res.get("equity_curve", np.full(n_bars, 100000.0))
        daily_ret = np.diff(eq) / np.maximum(eq[:-1], 1.0)
        valid_elites.append((strat, fit, details, res, daily_ret))

    if not valid_elites:
        # Fallback empty specification
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump({"portfolio_candidates": []}, f)
        return [], np.array([]), {}

    # Strictly require profitable elites (PF > 1.0 and net_pnl > 0 and total_return_pct > 0)
    profitable = [
        item for item in valid_elites
        if item[3].get("profit_factor", 0.0) > 1.0
        and item[3].get("net_pnl_usd", 0.0) > 0.0
        and item[3].get("total_return_pct", 0.0) > 0.0
    ]
    if require_profitable and not profitable:
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump({"portfolio_candidates": []}, f)
        return [], np.array([]), {}

    final_pool = profitable if profitable else valid_elites

    valid_elites = [(item[0], item[1], item[2], item[3]) for item in final_pool]
    strategy_returns = [item[4] for item in final_pool]

    ret_matrix = np.column_stack(strategy_returns)
    corr_matrix = np.corrcoef(ret_matrix, rowvar=False)
    if corr_matrix.ndim == 0 or np.isnan(corr_matrix).all():
        corr_matrix = np.eye(len(valid_elites))
    else:
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 2. Cluster strategies by correlation
    dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
    condensed = ssd.squareform(dist_mat, checks=False)

    if len(valid_elites) > 1:
        link = sch.linkage(condensed, method="average")
        cluster_labels = sch.fcluster(link, t=max_correlation_cluster, criterion="distance")
    else:
        cluster_labels = np.array([1])

    # 3. Select champion per cluster
    clusters: dict[int, list[int]] = {}
    for idx, c_id in enumerate(cluster_labels):
        clusters.setdefault(c_id, []).append(idx)

    chosen_indices = []
    for c_id, members in clusters.items():
        # Pick member with highest fitness
        best_member = max(members, key=lambda i: valid_elites[i][1])
        chosen_indices.append(best_member)

    # Sort champions by fitness descending and clamp to min_candidates..max_candidates
    chosen_indices.sort(key=lambda i: valid_elites[i][1], reverse=True)
    if len(chosen_indices) < min_candidates:
        # Add next best runners-up if available, strictly requiring correlation < 0.85 with all already chosen champions
        all_sorted = sorted(range(len(valid_elites)), key=lambda i: valid_elites[i][1], reverse=True)
        for idx in all_sorted:
            if idx not in chosen_indices:
                if all(corr_matrix[idx, chosen] < 0.85 for chosen in chosen_indices):
                    chosen_indices.append(idx)
            if len(chosen_indices) >= min_candidates:
                break

    chosen_indices = chosen_indices[:max_candidates]

    # 4. HRP Weights on chosen champions
    chosen_cov = np.cov(ret_matrix[:, chosen_indices], rowvar=False)
    if chosen_cov.ndim == 0:
        hrp_weights = np.array([1.0])
    else:
        hrp_weights = compute_hrp_weights(chosen_cov)

    champions = [valid_elites[i][0] for i in chosen_indices]

    # 5. Export candidates.yaml
    candidates_list = []
    for rank, idx in enumerate(chosen_indices):
        strat, fit, details, b_res = valid_elites[idx]
        w = float(hrp_weights[rank])
        wr = float(b_res.get("win_rate", details.get("win_rate_eur", 0.0)))
        pf = float(b_res.get("profit_factor", details.get("profit_factor_eur", 0.0)))
        tr = float(b_res.get("total_return_pct", details.get("total_return_pct", 0.0)))
        mdd = float(b_res.get("max_dd_pct", details.get("max_dd_pct", 0.0)))
        nt = int(b_res.get("n_trades", details.get("n_trades_eur", 0)))
        hold = float(b_res.get("holding_bars_median", details.get("median_hold_hours", 0.0)))

        candidates_list.append({
            "rank": rank + 1,
            "strategy_id": strat.strategy_id,
            "family": strat.family,
            "hrp_weight": round(w, 4),
            "fitness": round(fit, 3),
            "win_rate": wr,
            "win_rate_eur": wr,
            "profit_factor": pf,
            "profit_factor_eur": pf,
            "total_return_pct": tr,
            "max_dd_pct": mdd,
            "n_trades": nt,
            "n_trades_eur": nt,
            "total_trades": nt,
            "avg_holding_bars": hold,
            "median_hold_hours": hold,
            "holding_bars_median": hold,
            "genotype": strat.genotype_dict(),
        })

    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(_to_native({"portfolio_candidates": candidates_list}), f, sort_keys=False)

    portfolio_meta = {
        "n_candidates": len(champions),
        "weights": hrp_weights.tolist(),
        "candidates": candidates_list,
    }
    return champions, hrp_weights, portfolio_meta
