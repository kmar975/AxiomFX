"""
Multi-Pair Pooled Fitness Function with Session Basket Matching and Dollar Correlation Correction (§ 8).
Splits DEV into calendar blocks, applies 1.5x fitness costs, computes correlation-discounted weights (w_p),
enforces the 2-hour anti-scalping filter, and applies complexity and drawdown penalties.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy


def get_matched_subbasket(family: str, all_pairs: list[str]) -> list[str]:
    """
    Returns matched sub-basket P_strat for strategy family (§ 8.2).
    - London/European session: EURUSD, GBPUSD, USDCHF, EURGBP
    - Asian session: EURUSD, USDJPY, AUDUSD, EURJPY
    - All other families: full 7 pairs
    """
    if "London" in family or "Judas" in family or family == "F8_Sessions":
        candidates = ["EURUSD", "GBPUSD", "USDCHF", "EURGBP"]
        return [p for p in candidates if p in all_pairs]
    elif "Asian" in family:
        candidates = ["EURUSD", "USDJPY", "AUDUSD", "EURJPY"]
        return [p for p in candidates if p in all_pairs]
    return all_pairs


def compute_basket_correlation_weights(
    dev_data: dict[str, pd.DataFrame],
    basket_pairs: list[str],
) -> dict[str, float]:
    """
    Computes correlation-adjusted weights w_p proportional to (1 + rho_p)^(-1).
    """
    if len(basket_pairs) <= 1:
        return {p: 1.0 for p in basket_pairs}

    # Build matrix of daily/hourly returns
    ret_series = {}
    for p in basket_pairs:
        c = dev_data[p]["close"].values
        ret = np.diff(c) / np.maximum(c[:-1], 1e-4)
        ret_series[p] = ret

    # Common length
    min_len = min(len(r) for r in ret_series.values())
    df_rets = pd.DataFrame({p: ret_series[p][:min_len] for p in basket_pairs})
    corr_mat = df_rets.corr().fillna(0.0).values

    raw_weights = np.zeros(len(basket_pairs), dtype=np.float64)
    for i in range(len(basket_pairs)):
        # Mean correlation with all other pairs
        other_corrs = [corr_mat[i, j] for j in range(len(basket_pairs)) if i != j]
        avg_rho = float(np.mean(other_corrs)) if other_corrs else 0.0
        avg_rho = max(-0.90, min(0.99, avg_rho))
        raw_weights[i] = 1.0 / (1.0 + max(0.0, avg_rho))

    sum_w = np.sum(raw_weights)
    norm_weights = raw_weights / (sum_w if sum_w > 0 else 1.0)
    return {basket_pairs[i]: float(norm_weights[i]) for i in range(len(basket_pairs))}


def split_into_time_blocks(
    df: pd.DataFrame,
    n_blocks: int = 4,
) -> list[tuple[int, int]]:
    """
    Divides dataframe into calendar/chronological slices.
    """
    n = len(df)
    block_size = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else n
        if end - start > 10:
            blocks.append((start, end))
    return blocks


def evaluate_multipair_fitness(
    strategy: Strategy,
    dev_data: dict[str, pd.DataFrame],
    features_by_pair: dict[str, pd.DataFrame],
    cost_multiplier: float = 1.5,
    min_median_hold_hours: float = 1.0,
    min_trades_primary: int = 10,
    min_trades_basket: int = 30,
) -> tuple[float, dict[str, Any]]:
    """
    Calculates institutional pooled fitness (§ 8) across DEV blocks.
    Applies anti-scalping hard filter, complexity penalty, and consistency bonuses.
    """
    all_pairs = list(dev_data.keys())
    p_strat = get_matched_subbasket(strategy.family, all_pairs)
    weights = compute_basket_correlation_weights(dev_data, p_strat)

    eur_df = dev_data["EURUSD"]
    eur_feats = features_by_pair["EURUSD"]
    l_mask_eur, s_mask_eur = compile_strategy(strategy, eur_feats)

    # 1. Primary pair block evaluation
    eur_blocks = split_into_time_blocks(eur_df, n_blocks=4)
    block_sharpes_eur = []
    block_results_eur = []
    total_trades_eur = 0
    all_durations_eur = []

    q_vec_eur = compute_quote_to_usd_vector("EURUSD", len(eur_df), {p: d["close"].values for p, d in dev_data.items()})

    # Full backtest on EURUSD
    full_eur_res = run_vectorized_backtest(
        df=eur_df,
        pair="EURUSD",
        entry_longs=l_mask_eur,
        entry_shorts=s_mask_eur,
        quote_to_usd=q_vec_eur,
        sl_atr_mult=strategy.exit.sl_atr,
        tp_atr_mult=strategy.exit.tp_atr,
        max_holding_bars=strategy.exit.max_holding_bars,
        cost_multiplier=cost_multiplier,
        exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
        trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
    )
    total_trades_eur = full_eur_res["n_trades"]
    all_durations_eur = full_eur_res.get("durations", [])

    for s_idx, e_idx in eur_blocks:
        sub_df = eur_df.iloc[s_idx:e_idx].copy()
        sub_l = l_mask_eur[s_idx:e_idx]
        sub_s = s_mask_eur[s_idx:e_idx]
        sub_q = q_vec_eur[s_idx:e_idx]
        res = run_vectorized_backtest(
            df=sub_df,
            pair="EURUSD",
            entry_longs=sub_l,
            entry_shorts=sub_s,
            quote_to_usd=sub_q,
            sl_atr_mult=strategy.exit.sl_atr,
            tp_atr_mult=strategy.exit.tp_atr,
            max_holding_bars=strategy.exit.max_holding_bars,
            cost_multiplier=cost_multiplier,
            exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
        )
        block_results_eur.append(res)
        if res["n_trades"] < 3:
            block_sharpes_eur.append(full_eur_res["sharpe"] * 0.5)
        else:
            block_sharpes_eur.append(res["sharpe"])

    # Q25 (lower quartile) Sharpe on EURUSD
    q25_eur = float(np.percentile(block_sharpes_eur, 25)) if block_sharpes_eur else 0.0

    # 2. Hard Anti-Scalping Filter
    median_hold_bars = float(np.median(all_durations_eur)) if len(all_durations_eur) > 0 else 0.0
    # Assuming H1 timeframe: 1 bar = 1 hour
    median_hold_hours = median_hold_bars * (1.0 if strategy.timeframe == "H1" else 0.25)
    if median_hold_hours < min_median_hold_hours:
        return -999.0, {
            "fitness": -999.0,
            "failed_filter": f"Anti-scalping: median hold {median_hold_hours:.2f}h < {min_median_hold_hours}h",
            "recency_adj": 0.0,
            "recent_block_pnl": 0.0,
            "recent_block_pf": 0.0,
        }

    # 3. Hard Activity Filter
    if total_trades_eur < min_trades_primary:
        return -999.0, {
            "fitness": -999.0,
            "failed_filter": f"Insufficient trades EURUSD: {total_trades_eur} < {min_trades_primary}",
            "recency_adj": 0.0,
            "recent_block_pnl": 0.0,
            "recent_block_pf": 0.0,
        }

    # 4. Multi-Pair Basket Evaluation
    basket_q25_by_pair = {}
    total_basket_trades = total_trades_eur
    pos_weighted_mass = 0.0

    for p in p_strat:
        if p == "EURUSD":
            basket_q25_by_pair[p] = q25_eur
            if full_eur_res["net_pnl_usd"] > 0:
                pos_weighted_mass += weights[p]
            continue

        p_df = dev_data[p]
        p_feats = features_by_pair[p]
        l_p, s_p = compile_strategy(strategy, p_feats)
        q_p = compute_quote_to_usd_vector(p, len(p_df), {k: d["close"].values for k, d in dev_data.items()})

        # Full run on pair p
        p_res = run_vectorized_backtest(
            df=p_df,
            pair=p,
            entry_longs=l_p,
            entry_shorts=s_p,
            quote_to_usd=q_p,
            sl_atr_mult=strategy.exit.sl_atr,
            tp_atr_mult=strategy.exit.tp_atr,
            max_holding_bars=strategy.exit.max_holding_bars,
            cost_multiplier=cost_multiplier,
            exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
        )
        total_basket_trades += p_res["n_trades"]
        if p_res["net_pnl_usd"] > 0:
            pos_weighted_mass += weights[p]

        # Blocks on pair p
        p_blocks = split_into_time_blocks(p_df, n_blocks=4)
        p_block_sharpes = []
        for s_idx, e_idx in p_blocks:
            sub_res = run_vectorized_backtest(
                df=p_df.iloc[s_idx:e_idx].copy(),
                pair=p,
                entry_longs=l_p[s_idx:e_idx],
                entry_shorts=s_p[s_idx:e_idx],
                quote_to_usd=q_p[s_idx:e_idx],
                sl_atr_mult=strategy.exit.sl_atr,
                tp_atr_mult=strategy.exit.tp_atr,
                max_holding_bars=strategy.exit.max_holding_bars,
                cost_multiplier=cost_multiplier,
                exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
                trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
            )
            if sub_res["n_trades"] < 3:
                p_block_sharpes.append(p_res["sharpe"] * 0.5)
            else:
                p_block_sharpes.append(sub_res["sharpe"])

        q25_p = float(np.percentile(p_block_sharpes, 25)) if p_block_sharpes else 0.0
        basket_q25_by_pair[p] = q25_p

    # Enforce minimum total basket trades requirement
    effective_min_basket = min_trades_primary if len(p_strat) <= 1 else min_trades_basket
    if total_basket_trades < effective_min_basket:
        return -999.0, {
            "fitness": -999.0,
            "failed_filter": f"Insufficient total basket trades: {total_basket_trades} < {effective_min_basket}",
            "recency_adj": 0.0,
            "recent_block_pnl": 0.0,
            "recent_block_pf": 0.0,
        }

    # Basket Fitness = sum(w_p * Q25_p)
    fitness_basket = sum(weights[p] * basket_q25_by_pair[p] for p in p_strat)

    # Consistency bonus / penalty
    bonus_consistency = 0.20 if (pos_weighted_mass >= 0.50 and full_eur_res["net_pnl_usd"] > 0) else 0.0

    # Profit factor bonus on primary pair EURUSD
    pf_bonus = 0.25 * min(2.0, max(0.0, full_eur_res["profit_factor"] - 1.0))

    # Recency gate (OOS-1 test): 4th block of EURUSD (last 25% of DEV, adjacent to HOLDOUT)
    recent_block_res = block_results_eur[-1] if block_results_eur else full_eur_res
    pos_eur_blocks = sum(1 for r in block_results_eur if r["net_pnl_usd"] > 0)

    recency_adj = 0.0
    if recent_block_res["net_pnl_usd"] < 0 and pos_eur_blocks < 2:
        recency_adj = -0.20
    elif recent_block_res["profit_factor"] >= 1.10 and pos_eur_blocks >= 3:
        recency_adj = 0.30

    # Complexity penalty across all 5 slots
    n_conds = 1 + (1 if strategy.context else 0) + (1 if getattr(strategy, "context_2", None) else 0) + (1 if strategy.setup else 0) + (1 if getattr(strategy, "setup_2", None) else 0)
    complexity_penalty = 0.02 * (n_conds - 1)

    # Drawdown penalty
    dd_penalty = 0.01 * max(0.0, full_eur_res["max_dd_pct"] - 15.0)

    # Core EUR score combining full Sharpe and block lower quartile
    core_eur_score = 0.65 * full_eur_res["sharpe"] + 0.35 * q25_eur

    # Final Combined Fitness
    raw_fitness = (
        0.60 * core_eur_score
        + 0.40 * fitness_basket
        + bonus_consistency
        + pf_bonus
        + recency_adj
        - complexity_penalty
        - dd_penalty
    )

    details = {
        "fitness": round(raw_fitness, 3),
        "q25_eur": round(q25_eur, 3),
        "fitness_basket": round(fitness_basket, 3),
        "pos_weighted_mass": round(pos_weighted_mass, 2),
        "n_trades_eur": total_trades_eur,
        "n_trades_basket": total_basket_trades,
        "median_hold_hours": round(median_hold_hours, 2),
        "win_rate_eur": full_eur_res["win_rate"],
        "profit_factor_eur": full_eur_res["profit_factor"],
        "max_dd_pct": full_eur_res["max_dd_pct"],
        "total_return_pct": full_eur_res["total_return_pct"],
        "recent_block_pnl": recent_block_res["net_pnl_usd"],
        "recent_block_pf": recent_block_res["profit_factor"],
        "recency_adj": recency_adj,
        "durations": full_eur_res.get("durations", np.array([])),
        "trade_pnls": full_eur_res.get("trade_pnls", np.array([])),
        "trade_directions": full_eur_res.get("trade_directions", np.array([])),
    }
    return raw_fitness, details
