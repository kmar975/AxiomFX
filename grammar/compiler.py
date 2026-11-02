"""
DSL Strategy Compiler.
Compiles Strategy trees and conditions into boolean signal execution arrays.
Enforces the critical is_regime_ceiling non-inversion rule and evaluates M-bar setup validity windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grammar.operators import eval_operator
from grammar.strategy import Condition, Strategy


def mirror_directional_feature(feature_name: str) -> str | None:
    """
    Swaps directional pair members between bull and bear / high and low / up and dn / pdh and pdl.
    Preserves exact prefixes (e.g. htf_).
    """
    # Specific asymmetric cross naming: _cross_up_m100 <-> _cross_dn_100
    if "_cross_up_m100" in feature_name:
        return feature_name.replace("_cross_up_m100", "_cross_dn_100")
    if "_cross_dn_100" in feature_name:
        return feature_name.replace("_cross_dn_100", "_cross_up_m100")

    if "_bull" in feature_name:
        return feature_name.replace("_bull", "_bear")
    if "_bear" in feature_name:
        return feature_name.replace("_bear", "_bull")
    if "bull_prior" in feature_name:
        return feature_name.replace("bull_prior", "bear_prior")
    if "bear_prior" in feature_name:
        return feature_name.replace("bear_prior", "bull_prior")
    if "_high_" in feature_name:
        return feature_name.replace("_high_", "_low_")
    if "_low_" in feature_name:
        return feature_name.replace("_low_", "_high_")
    if "_mr_lower" in feature_name:
        return feature_name.replace("_mr_lower", "_mr_upper")
    if "_mr_upper" in feature_name:
        return feature_name.replace("_mr_upper", "_mr_lower")
    if "_bo_upper" in feature_name:
        return feature_name.replace("_bo_upper", "_bo_lower")
    if "_bo_lower" in feature_name:
        return feature_name.replace("_bo_lower", "_bo_upper")
    if "_cross_up" in feature_name:
        return feature_name.replace("_cross_up", "_cross_dn")
    if "_cross_dn" in feature_name:
        return feature_name.replace("_cross_dn", "_cross_up")

    # Levels mirroring: pdh <-> pdl, r1 <-> s1, cluster_above <-> cluster_below
    if "_pdh_" in feature_name:
        return feature_name.replace("_pdh_", "_pdl_")
    if "_pdl_" in feature_name:
        return feature_name.replace("_pdl_", "_pdh_")
    if "_r1_" in feature_name:
        return feature_name.replace("_r1_", "_s1_")
    if "_s1_" in feature_name:
        return feature_name.replace("_s1_", "_r1_")
    if "_cluster_above_" in feature_name:
        return feature_name.replace("_cluster_above_", "_cluster_below_")
    if "_cluster_below_" in feature_name:
        return feature_name.replace("_cluster_below_", "_cluster_above_")

    return None


def invert_condition_for_short(cond: Condition) -> Condition:
    """
    Inverts a condition for Short positions under symmetry.
    1. If is_regime_ceiling is True, returns condition UNCHANGED.
    2. If cond.feature is a directional pair member (_bull <-> _bear, etc.),
       swaps to the mirrored feature while keeping operator and threshold.
    3. If operator is 'event', inverts threshold sign (-cond.threshold).
    4. For signed directional features (e.g. s3_structure_state), inverts comparison direction.
    5. For percentile features [0..100], inverts threshold (100.0 - cond.threshold) and swaps gt <-> lt.
    """
    if cond.is_regime_ceiling:
        return cond

    clean_feat = cond.feature.removeprefix("htf_")

    # 1. Mirrored directional pairs (bull/bear, high/low, cross_up/dn, bb mr/bo, pdh/pdl)
    mirrored = mirror_directional_feature(cond.feature)
    if mirrored is not None:
        return Condition(
            feature=mirrored,
            source=cond.source,
            operator=cond.operator,
            threshold=cond.threshold,
            direction_mode=cond.direction_mode,
            threshold_high=cond.threshold_high,
            is_regime_ceiling=False,
        )

    # 2. Signed ternary features (+1.0 / -1.0)
    if cond.operator == "event":
        return Condition(
            feature=cond.feature,
            source=cond.source,
            operator="event",
            threshold=-cond.threshold,
            direction_mode=cond.direction_mode,
            threshold_high=cond.threshold_high,
            is_regime_ceiling=False,
        )

    # 3. Signed directional features with inequality comparisons (e.g. s3_structure_state > 0)
    from grammar.strategy import SIGNED_DIRECTIONAL_FEATURES, is_binary_feature
    if cond.feature in SIGNED_DIRECTIONAL_FEATURES or clean_feat in SIGNED_DIRECTIONAL_FEATURES:
        inv_op = cond.operator
        if cond.operator == "gt":
            inv_op = "lt"
        elif cond.operator == "lt":
            inv_op = "gt"
        return Condition(
            feature=cond.feature,
            source=cond.source,
            operator=inv_op,
            threshold=-cond.threshold,
            direction_mode=cond.direction_mode,
            threshold_high=None,
            is_regime_ceiling=False,
        )

    # 3b. Binary directional features {0.0, 1.0} with threshold near 0.5 (e.g. s1_ema_stack_*)
    if is_binary_feature(cond.feature) or abs(cond.threshold - 0.5) < 1e-3:
        inv_op = "lt" if cond.operator == "gt" else ("gt" if cond.operator == "lt" else cond.operator)
        return Condition(
            feature=cond.feature,
            source=cond.source,
            operator=inv_op,
            threshold=0.5,
            direction_mode=cond.direction_mode,
            threshold_high=None,
            is_regime_ceiling=False,
        )

    # 4. Continuous percentile features [0..100]
    inv_thresh = 100.0 - cond.threshold
    inv_thresh_high = (100.0 - cond.threshold) if cond.threshold_high is None else (100.0 - cond.threshold_high)
    # Ensure inv_thresh <= inv_thresh_high if operator is 'between'
    actual_low = min(inv_thresh, inv_thresh_high)
    actual_high = max(inv_thresh, inv_thresh_high)

    inv_op = cond.operator
    if cond.operator == "gt":
        inv_op = "lt"
    elif cond.operator == "lt":
        inv_op = "gt"
    elif cond.operator == "cross_above":
        inv_op = "cross_below"
    elif cond.operator == "cross_below":
        inv_op = "cross_above"

    return Condition(
        feature=cond.feature,
        source=cond.source,
        operator=inv_op,
        threshold=actual_low if cond.operator == "between" else inv_thresh,
        direction_mode=cond.direction_mode,
        threshold_high=actual_high if cond.operator == "between" else None,
        is_regime_ceiling=False,
    )


def expand_window_mask(mask: np.ndarray, window_m: int) -> np.ndarray:
    """
    Returns boolean array where arr[t] is True if mask was True at any point in [t - window_m, t].
    """
    n = len(mask)
    expanded = np.zeros(n, dtype=bool)
    if window_m <= 1:
        return mask.copy()

    # Rolling max of boolean array
    s = pd.Series(mask.astype(int))
    expanded = (s.rolling(window_m, min_periods=1).max().values > 0)
    return expanded


def compile_strategy(
    strategy: Strategy,
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compiles a strategy against a feature dataframe.
    Returns: (entry_long_mask, entry_short_mask)
    """
    if isinstance(features_df, dict):
        features_df = pd.DataFrame([features_df])
    n = len(features_df)
    entry_long = np.ones(n, dtype=bool)
    entry_short = np.ones(n, dtype=bool)

    # 1. Context (1 and 2)
    for ctx in [strategy.context, getattr(strategy, "context_2", None)]:
        if ctx is not None:
            c_long = ctx
            c_short = invert_condition_for_short(c_long) if strategy.symmetric else c_long

            if c_long.feature in features_df.columns:
                feat_arr = features_df[c_long.feature].values
                mask_long = eval_operator(feat_arr, c_long.operator, c_long.threshold, c_long.threshold_high)
                entry_long &= mask_long
            else:
                entry_long[:] = False

            if c_short.feature in features_df.columns:
                feat_arr = features_df[c_short.feature].values
                mask_short = eval_operator(feat_arr, c_short.operator, c_short.threshold, c_short.threshold_high)
                entry_short &= mask_short
            else:
                entry_short[:] = False

    # 2. Setup (1 and 2, each with its M-bar validity window)
    setups = [
        (strategy.setup, strategy.setup_expiry),
        (getattr(strategy, "setup_2", None), getattr(strategy, "setup_2_expiry", 5)),
    ]
    for stp, expiry in setups:
        if stp is not None:
            s_long = stp
            s_short = invert_condition_for_short(s_long) if strategy.symmetric else s_long
            m_win = max(1, expiry)

            if s_long.feature in features_df.columns:
                feat_arr = features_df[s_long.feature].values
                raw_s_long = eval_operator(feat_arr, s_long.operator, s_long.threshold, s_long.threshold_high)
                entry_long &= expand_window_mask(raw_s_long, m_win)
            else:
                entry_long[:] = False

            if s_short.feature in features_df.columns:
                feat_arr = features_df[s_short.feature].values
                raw_s_short = eval_operator(feat_arr, s_short.operator, s_short.threshold, s_short.threshold_high)
                entry_short &= expand_window_mask(raw_s_short, m_win)
            else:
                entry_short[:] = False

    # 3. Trigger (Mandatory)
    t_long = strategy.trigger
    t_short = invert_condition_for_short(t_long) if strategy.symmetric else t_long

    if t_long.feature in features_df.columns:
        feat_arr = features_df[t_long.feature].values
        mask_t_long = eval_operator(feat_arr, t_long.operator, t_long.threshold, t_long.threshold_high)
    else:
        mask_t_long = np.zeros(n, dtype=bool)

    if t_short.feature in features_df.columns:
        feat_arr = features_df[t_short.feature].values
        mask_t_short = eval_operator(feat_arr, t_short.operator, t_short.threshold, t_short.threshold_high)
    else:
        mask_t_short = np.zeros(n, dtype=bool)

    entry_long &= mask_t_long
    entry_short &= mask_t_short

    # 4. Direction mode: fade inverts entire strategy signals (counter-trend mean reversion)
    if getattr(strategy.trigger, "direction_mode", "continuation") == "fade":
        entry_long, entry_short = entry_short.copy(), entry_long.copy()

    # 5. Cancel conflicting simultaneous triggers
    both = entry_long & entry_short
    if np.any(both):
        entry_long[both] = False
        entry_short[both] = False

    return entry_long, entry_short


def compile_live_state(
    strategy: Strategy,
    features_df: pd.DataFrame,
    bar_idx: int = -1,
    return_direction: bool = False,
) -> str | tuple[str, int]:
    """
    Evaluates strategy at a specific bar (default: last bar) into 4 live states:
    - BUY (+1)
    - SELL (-1)
    - ARMED (0: Context and Setup met within validity window, awaiting Trigger)
    - NEUTRAL (0)
    Note: ARMED requires an explicit SETUP condition to be active. Pure context rules
    do not arm without a setup event.
    """
    if isinstance(features_df, dict):
        features_df = pd.DataFrame([features_df])
    long_mask, short_mask = compile_strategy(strategy, features_df)
    idx = bar_idx if bar_idx >= 0 else (len(features_df) + bar_idx)

    if long_mask[idx]:
        return ("BUY", 1) if return_direction else "BUY"
    elif short_mask[idx]:
        return ("SELL", -1) if return_direction else "SELL"

    # Only strategies with an explicit SETUP condition can enter ARMED state
    has_setup = (strategy.setup is not None) or (getattr(strategy, "setup_2", None) is not None)
    if not has_setup:
        return ("NEUTRAL", 0) if return_direction else "NEUTRAL"

    # Check ARMED condition for Long: all defined contexts and setups are met
    preconditions_ok_long = True

    for ctx in [strategy.context, getattr(strategy, "context_2", None)]:
        if ctx is not None:
            if ctx.feature in features_df.columns:
                c_mask = eval_operator(features_df[ctx.feature].values, ctx.operator, ctx.threshold, ctx.threshold_high)
                if not c_mask[idx]:
                    preconditions_ok_long = False
            else:
                preconditions_ok_long = False

    for stp, expiry in [
        (strategy.setup, strategy.setup_expiry),
        (getattr(strategy, "setup_2", None), getattr(strategy, "setup_2_expiry", 5)),
    ]:
        if stp is not None:
            if stp.feature in features_df.columns:
                raw_s = eval_operator(features_df[stp.feature].values, stp.operator, stp.threshold, stp.threshold_high)
                exp_s = expand_window_mask(raw_s, max(1, expiry))
                if not exp_s[idx]:
                    preconditions_ok_long = False
            else:
                preconditions_ok_long = False

    is_fade = (getattr(strategy.trigger, "direction_mode", "continuation") == "fade")

    if preconditions_ok_long:
        armed_dir = -1 if is_fade else 1
        return ("ARMED", armed_dir) if return_direction else "ARMED"

    # Check ARMED condition for Short
    preconditions_ok_short = True

    for ctx in [strategy.context, getattr(strategy, "context_2", None)]:
        if ctx is not None:
            c_short = invert_condition_for_short(ctx) if strategy.symmetric else ctx
            if c_short.feature in features_df.columns:
                c_mask = eval_operator(features_df[c_short.feature].values, c_short.operator, c_short.threshold, c_short.threshold_high)
                if not c_mask[idx]:
                    preconditions_ok_short = False
            else:
                preconditions_ok_short = False

    for stp, expiry in [
        (strategy.setup, strategy.setup_expiry),
        (getattr(strategy, "setup_2", None), getattr(strategy, "setup_2_expiry", 5)),
    ]:
        if stp is not None:
            s_short = invert_condition_for_short(stp) if strategy.symmetric else stp
            if s_short.feature in features_df.columns:
                raw_s = eval_operator(features_df[s_short.feature].values, s_short.operator, s_short.threshold, s_short.threshold_high)
                exp_s = expand_window_mask(raw_s, max(1, expiry))
                if not exp_s[idx]:
                    preconditions_ok_short = False
            else:
                preconditions_ok_short = False

    if preconditions_ok_short:
        armed_dir = 1 if is_fade else -1
        return ("ARMED", armed_dir) if return_direction else "ARMED"

    return ("NEUTRAL", 0) if return_direction else "NEUTRAL"
