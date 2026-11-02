"""
High-Performance Multi-Core Exhaustive Layer on the Orthogonal Core (Phase E0).
Deterministically screens 1-condition, sequential 2-condition [SETUP (window M) -> TRIGGER],
and 3-condition [CONTEXT -> SETUP (window M) -> TRIGGER] rules from the full Tier 1 feature bank
and lookahead-free HTF features with strict source orthogonality.
Uses Numba parallel batch kernel and uint64 bitsets for maximum CPU throughput.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

from grammar.compiler import compile_strategy, expand_window_mask, invert_condition_for_short
from grammar.operators import eval_operator
from grammar.strategy import (
    Condition,
    ExitSpec,
    Strategy,
    is_binary_feature,
    is_signed_directional_feature,
)
from search.funnel import FunnelExecutor


@njit(parallel=True, fastmath=True)
def evaluate_batch_s0_signals_numba(
    signals_matrix: np.ndarray,      # (n_candidates, n_bars) int8 or uint8 (+1 long, -1/255 short, 0 none)
    forward_returns: np.ndarray,     # (n_bars,) float32
    min_signals: int = 10,
    cost_proxy: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates thousands of bidirectional signal combinations in parallel with zero heap allocations.
    For LONG signals (+1), adds forward_returns; for SHORT signals (-1 or 255), subtracts forward_returns.
    Returns:
    - scores: float32[:] (mean directional forward return minus cost proxy)
    - signal_counts: int32[:]
    """
    n_candidates = signals_matrix.shape[0]
    n_bars = signals_matrix.shape[1]

    scores = np.full(n_candidates, -999.0, dtype=np.float32)
    signal_counts = np.zeros(n_candidates, dtype=np.int32)

    for c in prange(n_candidates):
        count = 0
        sum_ret = 0.0
        for t in range(n_bars):
            sig = signals_matrix[c, t]
            if sig == 1:
                count += 1
                sum_ret += forward_returns[t]
            elif sig == -1 or sig == 255:
                count += 1
                sum_ret -= forward_returns[t]

        signal_counts[c] = count
        if count >= min_signals:
            mean_ret = sum_ret / count
            scores[c] = mean_ret - cost_proxy

    return scores, signal_counts


def _extract_source_prefix(feature_name: str) -> str:
    """Extracts orthogonal information source prefix (s1..s10 or htf_s1..htf_s5)."""
    if feature_name.startswith("htf_"):
        parts = feature_name.split("_")
        return f"{parts[0]}_{parts[1]}"
    return feature_name.split("_")[0]


def _rank_features_by_variance(features_df: pd.DataFrame, feats: list[str]) -> list[str]:
    """Ranks features by standard deviation on EURUSD DEV to prioritize informative signals."""
    stds = {f: float(np.nanstd(features_df[f].values)) if f in features_df.columns else 0.0 for f in feats}
    return sorted(feats, key=lambda f: stds[f], reverse=True)


def run_phase_e0_exhaustive(
    tier1_features: list[str],
    features_by_pair: dict[str, pd.DataFrame],
    future_returns_by_pair: dict[str, dict[int, np.ndarray]],
    funnel: FunnelExecutor,
    max_candidates: int = 500,
) -> list[tuple[Strategy, float, dict[str, Any]]]:
    """
    Exhaustively searches 1-condition, sequential 2-condition [SETUP (window M) -> TRIGGER],
    and 3-condition [CONTEXT -> SETUP -> TRIGGER] rules across Tier 1 + HTF features.
    Uses ultra-fast atomic mask precompilation and multi-horizon parallel Numba batch evaluation.
    """
    passing_candidates: list[tuple[Strategy, float, dict[str, Any]]] = []

    # 1. Map features to their orthogonal source
    feature_source_map = {f: _extract_source_prefix(f) for f in tier1_features}
    unique_sources = sorted(set(feature_source_map.values()))

    setup_windows = [3, 5]
    exit_specs_e0 = [
        ExitSpec(exit_type="sl_tp", sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
        ExitSpec(exit_type="breakeven_trail", sl_atr=1.5, tp_atr=3.0, max_holding_bars=36, trailing_atr=1.2),
        ExitSpec(exit_type="dynamic_mean_reversion", sl_atr=2.0, tp_atr=3.0, max_holding_bars=36),
        ExitSpec(exit_type="pre_rollover_flat", sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    ]

    eur_feats = features_by_pair["EURUSD"]
    n_bars = len(eur_feats)

    # Multi-horizon weighted forward return on EURUSD: 0.4 * 5 + 0.4 * 10 + 0.2 * 20
    ret_5 = future_returns_by_pair["EURUSD"].get(5, np.zeros(n_bars, dtype=np.float32)).astype(np.float32)
    ret_10 = future_returns_by_pair["EURUSD"].get(10, ret_5).astype(np.float32)
    ret_20 = future_returns_by_pair["EURUSD"].get(20, ret_10).astype(np.float32)
    eur_fwd_ret = (0.4 * ret_5 + 0.4 * ret_10 + 0.2 * ret_20).astype(np.float32)

    def _get_feature_candidates(feat: str) -> list[tuple[str, float]]:
        if is_signed_directional_feature(feat):
            return [("event", 1.0), ("event", -1.0)]
        elif is_binary_feature(feat):
            return [("gt", 0.5)]
        else:
            return [
                ("gt", 65.0),
                ("gt", 75.0),
                ("gt", 85.0),
                ("lt", 35.0),
                ("lt", 25.0),
                ("lt", 15.0),
                ("cross_above", 30.0),
                ("cross_below", 70.0),
            ]

    def _get_setup_candidates(feat: str) -> list[tuple[str, float]]:
        if is_signed_directional_feature(feat):
            return [("event", 1.0), ("event", -1.0)]
        elif is_binary_feature(feat):
            return [("gt", 0.5)]
        else:
            return [("gt", 75.0), ("lt", 25.0)]

    def _get_trigger_candidates(feat: str) -> list[tuple[str, float]]:
        if is_signed_directional_feature(feat):
            return [("event", 1.0), ("event", -1.0)]
        elif is_binary_feature(feat):
            return [("gt", 0.5)]
        else:
            return [("gt", 75.0), ("lt", 25.0), ("cross_above", 30.0), ("cross_below", 70.0)]

    # Precompile 1D boolean atomic masks and window-expanded setup masks
    atom_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
    for feat in tier1_features:
        if feat not in eur_feats.columns:
            continue
        f_vals = eur_feats[feat].values
        # Union of all possible candidates across L1, L2, L3
        all_cands = set(_get_feature_candidates(feat) + _get_setup_candidates(feat) + _get_trigger_candidates(feat))
        for op, thresh in all_cands:
            key = (feat, op, thresh)
            if key in atom_cache:
                continue
            cond_long = Condition(feature=feat, source=feature_source_map[feat], operator=op, threshold=thresh)
            cond_short = invert_condition_for_short(cond_long)
            l_m = eval_operator(f_vals, op, thresh)
            if cond_short.feature in eur_feats.columns:
                s_vals = eur_feats[cond_short.feature].values
                s_m = eval_operator(s_vals, cond_short.operator, cond_short.threshold, cond_short.threshold_high)
            else:
                s_m = np.zeros(n_bars, dtype=bool)
            atom_cache[key] = (l_m, s_m)

    setup_window_cache: dict[tuple[str, str, float, int], tuple[np.ndarray, np.ndarray]] = {}
    for (feat, op, thresh), (l_m, s_m) in atom_cache.items():
        for m_win in (3, 4, 5):
            exp_l = expand_window_mask(l_m, m_win)
            exp_s = expand_window_mask(s_m, m_win)
            setup_window_cache[(feat, op, thresh, m_win)] = (exp_l, exp_s)

    hypotheses: list[Strategy] = []
    signals_list: list[np.ndarray] = []
    rule_id = 0
    max_e0_hypotheses = 150000

    # -------------------------------------------------------------
    # Layer 1: Single condition rules across all Tier 1 + HTF features
    # Both continuation and fade, all 4 exit archetypes
    # -------------------------------------------------------------
    for feat in tier1_features:
        if feat not in eur_feats.columns:
            continue
        src = feature_source_map[feat]

        for op, thresh in _get_feature_candidates(feat):
            atom_tuple = atom_cache.get((feat, op, thresh))
            if atom_tuple is None:
                continue
            raw_l, raw_s = atom_tuple

            cond_check = Condition(feature=feat, source=src, operator=op, threshold=thresh)
            if cond_check.is_regime_ceiling:
                continue

            for d_mode in ("continuation", "fade"):
                rule_id += 1
                cond = Condition(
                    feature=feat,
                    source=src,
                    operator=op,
                    threshold=thresh,
                    direction_mode=d_mode,
                )
                fam = "F1_Trend" if "s1" in src else ("F2_Breakout" if "s3" in src else "F4_Pullback")
                exit_spec = exit_specs_e0[rule_id % len(exit_specs_e0)]
                strat = Strategy(
                    strategy_id=f"E0_1_{rule_id}",
                    family=fam,
                    trigger=cond,
                    exit=ExitSpec(
                        exit_type=exit_spec.exit_type,
                        sl_atr=exit_spec.sl_atr,
                        tp_atr=exit_spec.tp_atr,
                        max_holding_bars=exit_spec.max_holding_bars,
                        trailing_atr=exit_spec.trailing_atr,
                    ),
                )
                hypotheses.append(strat)

                l_sig, s_sig = (raw_s, raw_l) if d_mode == "fade" else (raw_l, raw_s)
                both = l_sig & s_sig
                sig = np.zeros(n_bars, dtype=np.int8)
                sig[l_sig & ~both] = 1
                sig[s_sig & ~both] = -1
                signals_list.append(sig)

    # -------------------------------------------------------------
    # Layer 2: Sequential 2-Condition [SETUP (M=3..5) -> TRIGGER]
    # Source orthogonality strictly enforced: src(Setup) != src(Trigger)
    # Top 6 features per source, continuation & fade
    # -------------------------------------------------------------
    for src_setup, src_trig in itertools.combinations(unique_sources, 2):
        if len(hypotheses) >= max_e0_hypotheses:
            break
        raw_setup = [f for f in tier1_features if feature_source_map[f] == src_setup and f in eur_feats.columns]
        raw_trig = [f for f in tier1_features if feature_source_map[f] == src_trig and f in eur_feats.columns]
        feats_setup = _rank_features_by_variance(eur_feats, raw_setup)[:6]
        feats_trig = _rank_features_by_variance(eur_feats, raw_trig)[:6]

        for fs in feats_setup:
            cands_s = _get_setup_candidates(fs)
            for ft in feats_trig:
                cands_t = _get_trigger_candidates(ft)

                for op_s, th_s in cands_s:
                    for op_t, th_t in cands_t:
                        trig_atom = atom_cache.get((ft, op_t, th_t))
                        if trig_atom is None:
                            continue
                        trig_l, trig_s = trig_atom

                        for m_win in setup_windows:
                            setup_atom = setup_window_cache.get((fs, op_s, th_s, m_win))
                            if setup_atom is None:
                                continue
                            exp_l, exp_s = setup_atom

                            raw_comb_l = exp_l & trig_l
                            raw_comb_s = exp_s & trig_s

                            for d_mode in ("continuation", "fade"):
                                c_setup = Condition(feature=fs, source=src_setup, operator=op_s, threshold=th_s)
                                c_trig = Condition(feature=ft, source=src_trig, operator=op_t, threshold=th_t, direction_mode=d_mode)
                                if c_trig.is_regime_ceiling:
                                    continue

                                rule_id += 1
                                exit_spec = exit_specs_e0[rule_id % len(exit_specs_e0)]
                                strat = Strategy(
                                    strategy_id=f"E0_2_{rule_id}",
                                    family="F4_Pullback" if op_t == "lt" else "F1_Trend",
                                    setup=c_setup,
                                    setup_expiry=m_win,
                                    trigger=c_trig,
                                    exit=ExitSpec(
                                        exit_type=exit_spec.exit_type,
                                        sl_atr=exit_spec.sl_atr,
                                        tp_atr=exit_spec.tp_atr,
                                        max_holding_bars=exit_spec.max_holding_bars,
                                        trailing_atr=exit_spec.trailing_atr,
                                    ),
                                )
                                hypotheses.append(strat)

                                l_sig, s_sig = (raw_comb_s, raw_comb_l) if d_mode == "fade" else (raw_comb_l, raw_comb_s)
                                both = l_sig & s_sig
                                sig = np.zeros(n_bars, dtype=np.int8)
                                sig[l_sig & ~both] = 1
                                sig[s_sig & ~both] = -1
                                signals_list.append(sig)
                                if len(hypotheses) >= max_e0_hypotheses:
                                    break

    # -------------------------------------------------------------
    # Layer 3: 3-Condition [CONTEXT -> SETUP (M=4) -> TRIGGER]
    # Source orthogonality: src(Context) != src(Setup) != src(Trigger)
    # 4 best features per source, continuation & fade
    # -------------------------------------------------------------
    if len(unique_sources) >= 3:
        for s_ctx, s_set, s_trg in itertools.combinations(unique_sources, 3):
            raw_c = [f for f in tier1_features if feature_source_map[f] == s_ctx and f in eur_feats.columns]
            raw_s = [f for f in tier1_features if feature_source_map[f] == s_set and f in eur_feats.columns]
            raw_t = [f for f in tier1_features if feature_source_map[f] == s_trg and f in eur_feats.columns]

            top_ctx = _rank_features_by_variance(eur_feats, raw_c)[:4]
            top_set = _rank_features_by_variance(eur_feats, raw_s)[:4]
            top_trg = _rank_features_by_variance(eur_feats, raw_t)[:4]

            for f_ctx in top_ctx:
                cand_c = _get_feature_candidates(f_ctx)
                op_c, th_c = cand_c[0]
                ctx_atom = atom_cache.get((f_ctx, op_c, th_c))
                if ctx_atom is None:
                    continue
                ctx_l, ctx_s = ctx_atom

                for f_set in top_set:
                    cand_s = _get_feature_candidates(f_set)
                    op_s, th_s = cand_s[0]
                    setup_atom = setup_window_cache.get((f_set, op_s, th_s, 4))
                    if setup_atom is None:
                        continue
                    exp_l, exp_s = setup_atom

                    for f_trg in top_trg:
                        cand_t = _get_feature_candidates(f_trg)
                        op_t, th_t = cand_t[-1]  # Contrasting threshold / trigger
                        trig_atom = atom_cache.get((f_trg, op_t, th_t))
                        if trig_atom is None:
                            continue
                        trig_l, trig_s = trig_atom

                        raw_l = ctx_l & exp_l & trig_l
                        raw_s = ctx_s & exp_s & trig_s

                        for d_mode in ("continuation", "fade"):
                            c_ctx = Condition(feature=f_ctx, source=s_ctx, operator=op_c, threshold=th_c)
                            c_set = Condition(feature=f_set, source=s_set, operator=op_s, threshold=th_s)
                            c_trg = Condition(feature=f_trg, source=s_trg, operator=op_t, threshold=th_t, direction_mode=d_mode)
                            if c_trg.is_regime_ceiling:
                                continue

                            rule_id += 1
                            exit_spec = exit_specs_e0[rule_id % len(exit_specs_e0)]
                            strat = Strategy(
                                strategy_id=f"E0_3_{rule_id}",
                                family="F4_Pullback" if d_mode == "fade" else "F1_Trend",
                                context=c_ctx,
                                setup=c_set,
                                setup_expiry=4,
                                trigger=c_trg,
                                exit=ExitSpec(
                                    exit_type=exit_spec.exit_type,
                                    sl_atr=exit_spec.sl_atr,
                                    tp_atr=exit_spec.tp_atr,
                                    max_holding_bars=exit_spec.max_holding_bars,
                                    trailing_atr=exit_spec.trailing_atr,
                                ),
                            )
                            hypotheses.append(strat)

                            l_sig, s_sig = (raw_s, raw_l) if d_mode == "fade" else (raw_l, raw_s)
                            both = l_sig & s_sig
                            sig = np.zeros(n_bars, dtype=np.int8)
                            sig[l_sig & ~both] = 1
                            sig[s_sig & ~both] = -1
                            signals_list.append(sig)
                            if len(hypotheses) >= max_e0_hypotheses:
                                break
                        if len(hypotheses) >= max_e0_hypotheses:
                            break
                    if len(hypotheses) >= max_e0_hypotheses:
                        break
                if len(hypotheses) >= max_e0_hypotheses:
                    break
            if len(hypotheses) >= max_e0_hypotheses:
                break

    # -------------------------------------------------------------
    # Multi-Core Numba Screening of All Hypotheses Simultaneously
    # -------------------------------------------------------------
    print(f"[Phase E0] Compiled {len(hypotheses):,} orthogonal hypotheses. Running parallel Numba screening...")
    signals_matrix = np.vstack(signals_list)

    scores, counts = evaluate_batch_s0_signals_numba(
        signals_matrix=signals_matrix,
        forward_returns=eur_fwd_ret,
        min_signals=10,
        cost_proxy=0.02,
    )

    # Rank by score
    valid_idx = np.where(scores > -900.0)[0]
    sorted_idx = valid_idx[np.argsort(-scores[valid_idx])]

    # -------------------------------------------------------------
    # Pass Top Filtered Candidates into Multi-Pair Funnel S0/S1
    # -------------------------------------------------------------
    top_candidates = [hypotheses[i] for i in sorted_idx[:max_candidates * 5]]
    print(f"[Phase E0] Screened {len(sorted_idx):,} promising signals. Evaluating top {len(top_candidates):,} in multi-pair funnel...")

    for strat in top_candidates:
        passed, score, details = funnel.evaluate_s0_screening(
            strategy=strat,
            features_by_pair=features_by_pair,
            future_returns_by_pair=future_returns_by_pair,
            min_trades=10,
        )
        if passed:
            passing_candidates.append((strat, score, details))
            if len(passing_candidates) >= max_candidates:
                break

    print(f"[Phase E0] Exhaustive search complete: {len(passing_candidates)} robust orthogonal candidates passed to Phase C.")
    return passing_candidates
