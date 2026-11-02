"""
Institutional Genetic Algorithm Operators for Strategy Evolution (Phase C).
Implements tree structure mutation, setup expiry mutation, threshold/operator mutation,
exit spec parameter mutation, and elite crossover.
Guarantees source orthogonality and strict genotype deduplication.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from grammar.strategy import (
    NON_INVERTIBLE_BINARY_FEATURES,
    SIGNED_DIRECTIONAL_FEATURES,
    Condition,
    ExitSpec,
    Strategy,
    is_binary_feature,
    is_signed_directional_feature,
)

THRESHOLD_GRID = [10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0]
SL_ATR_OPTIONS = [1.0, 1.2, 1.5, 2.0, 2.5]
TP_ATR_OPTIONS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
HOLDING_BARS_OPTIONS = [12, 24, 36, 48, 72]
SETUP_EXPIRY_OPTIONS = [1, 2, 3, 5, 8]
OPERATOR_OPTIONS = ["gt", "lt", "cross_above", "cross_below"]
EXIT_TYPE_OPTIONS = ["sl_tp", "breakeven_trail", "dynamic_mean_reversion", "pre_rollover_flat"]
TRAILING_ATR_OPTIONS = [0.8, 1.0, 1.2, 1.5, 2.0]


def _get_source_from_feature(feat: str) -> str:
    if feat.startswith("htf_"):
        parts = feat.split("_")
        return f"{parts[0]}_{parts[1]}"
    return feat.split("_")[0]


def create_random_condition(feat: str, rng: np.random.RandomState) -> Condition:
    src = _get_source_from_feature(feat)
    if is_signed_directional_feature(feat):
        sign = float(rng.choice([1.0, -1.0]))
        return Condition(feature=feat, source=src, operator="event", threshold=sign)
    if is_binary_feature(feat):
        return Condition(feature=feat, source=src, operator="gt", threshold=0.5)
    op = rng.choice(OPERATOR_OPTIONS)
    thresh = float(rng.choice(THRESHOLD_GRID))
    return Condition(feature=feat, source=src, operator=op, threshold=thresh)


def mutate_and_crossover_strategy(
    parent_a: Strategy,
    parent_b: Strategy | None,
    tier1_features: Sequence[str],
    rng: np.random.RandomState,
    crossover_prob: float = 0.35,
    mutation_type: str | None = None,
) -> Strategy:
    """
    Executes deep genetic recombination and multi-operator mutation:
    1. Crossover: Combines context/setup from parent A and trigger/exit from parent B.
    2. Tree Structure Mutation: Adds, removes, or swaps context / setup conditions.
    3. Memory Window Mutation: Mutates setup_expiry M in {1, 2, 3, 5, 8}.
    4. Condition Mutation: Mutates feature, operator, or threshold on percentiles.
    5. Exit Spec Mutation: Mutates SL/TP/holding with tp_atr >= sl_atr invariant.
    """
    # 1. Crossover
    if parent_b is not None and rng.uniform(0.0, 1.0) < crossover_prob:
        child = copy.deepcopy(parent_a)
        # Take trigger and exit from parent B
        child.trigger = copy.deepcopy(parent_b.trigger)
        child.exit = copy.deepcopy(parent_b.exit)
        if not child.validate_rules():
            # If orthogonality violated, fallback to parent_a clone
            child = copy.deepcopy(parent_a)
    else:
        child = copy.deepcopy(parent_a)

    all_conds = [child.trigger, child.context, getattr(child, "context_2", None), child.setup, getattr(child, "setup_2", None)]
    active_sources = {c.source for c in all_conds if c is not None}

    # 2. Tree Structure Mutation (depth k=1..5: add, remove, swap context/context_2 or setup/setup_2)
    if mutation_type is None:
        mutation_type = rng.choice(["structure", "expiry", "condition", "exit", "combined"])

    if mutation_type in ("structure", "combined"):
        action = rng.choice(["toggle_context", "toggle_context_2", "toggle_setup", "toggle_setup_2", "swap_condition"])

        if action in ("toggle_context", "toggle_context_2"):
            attr_name = "context" if action == "toggle_context" else "context_2"
            curr_c = getattr(child, attr_name, None)
            if curr_c is None and len(active_sources) < 5:
                avail = [f for f in tier1_features if _get_source_from_feature(f) not in active_sources]
                if avail:
                    setattr(child, attr_name, create_random_condition(rng.choice(avail), rng))
            elif curr_c is not None:
                if rng.uniform(0.0, 1.0) < 0.40:
                    setattr(child, attr_name, None)
                else:
                    curr_src = curr_c.source
                    avail = [f for f in tier1_features if _get_source_from_feature(f) not in active_sources or _get_source_from_feature(f) == curr_src]
                    if avail:
                        setattr(child, attr_name, create_random_condition(rng.choice(avail), rng))

        elif action in ("toggle_setup", "toggle_setup_2"):
            attr_name = "setup" if action == "toggle_setup" else "setup_2"
            exp_attr = "setup_expiry" if action == "toggle_setup" else "setup_2_expiry"
            curr_s = getattr(child, attr_name, None)
            if curr_s is None and len(active_sources) < 5:
                avail = [f for f in tier1_features if _get_source_from_feature(f) not in active_sources]
                if avail:
                    setattr(child, attr_name, create_random_condition(rng.choice(avail), rng))
                    setattr(child, exp_attr, int(rng.choice(SETUP_EXPIRY_OPTIONS)))
            elif curr_s is not None:
                if rng.uniform(0.0, 1.0) < 0.40:
                    setattr(child, attr_name, None)
                else:
                    curr_src = curr_s.source
                    avail = [f for f in tier1_features if _get_source_from_feature(f) not in active_sources or _get_source_from_feature(f) == curr_src]
                    if avail:
                        setattr(child, attr_name, create_random_condition(rng.choice(avail), rng))

        elif action == "swap_condition":
            swap_pair = rng.choice(["context_setup", "setup_trigger", "context_trigger"])
            if swap_pair == "context_setup":
                if child.context is not None and child.setup is not None:
                    child.context, child.setup = child.setup, child.context
            elif swap_pair == "setup_trigger":
                if child.setup is not None and child.trigger is not None:
                    child.setup, child.trigger = child.trigger, child.setup
            elif swap_pair == "context_trigger":
                if child.context is not None and child.trigger is not None:
                    child.context, child.trigger = child.trigger, child.context

            # Reset direction_mode to "continuation" on non-trigger slots
            if child.setup is not None:
                child.setup.direction_mode = "continuation"
            if getattr(child, "setup_2", None) is not None:
                child.setup_2.direction_mode = "continuation"
            if child.context is not None:
                child.context.direction_mode = "continuation"
            if getattr(child, "context_2", None) is not None:
                child.context_2.direction_mode = "continuation"

    # 3. Setup Memory Window (Expiry) Mutation (M in {1, 2, 3, 5, 8})
    if child.setup is not None and (mutation_type in ("expiry", "combined") or rng.uniform(0.0, 1.0) < 0.30):
        child.setup_expiry = int(rng.choice(SETUP_EXPIRY_OPTIONS))
    if getattr(child, "setup_2", None) is not None and (mutation_type in ("expiry", "combined") or rng.uniform(0.0, 1.0) < 0.30):
        child.setup_2_expiry = int(rng.choice(SETUP_EXPIRY_OPTIONS))

    # 4. Feature, Operator, and Threshold Mutation
    if mutation_type in ("condition", "combined") or rng.uniform(0.0, 1.0) < 0.40:
        target_cond_name = rng.choice(["trigger", "context", "context_2", "setup", "setup_2"])
        cond = getattr(child, target_cond_name, None)
        if cond is not None:
            sub_mut = rng.choice(["thresh", "op", "feat", "dir"])
            if sub_mut == "thresh":
                if is_signed_directional_feature(cond.feature):
                    cond.threshold = -cond.threshold
                elif not is_binary_feature(cond.feature):
                    curr = cond.threshold
                    cond.threshold = float(rng.choice([t for t in THRESHOLD_GRID if t != curr] or THRESHOLD_GRID))
            elif sub_mut == "op":
                if is_signed_directional_feature(cond.feature):
                    cond.operator = "event"
                elif not is_binary_feature(cond.feature):
                    cond.operator = rng.choice(OPERATOR_OPTIONS)
            elif sub_mut == "feat":
                other_sources = {
                    c.source
                    for c in [child.trigger, child.context, getattr(child, "context_2", None), child.setup, getattr(child, "setup_2", None)]
                    if c is not None and c is not cond
                }
                avail = [f for f in tier1_features if _get_source_from_feature(f) not in other_sources]
                if cond is child.trigger:
                    avail = [f for f in avail if not Condition(f, _get_source_from_feature(f), "gt", 50.0).is_regime_ceiling]
                if avail:
                    new_f = rng.choice(avail)
                    cond.feature = new_f
                    cond.source = _get_source_from_feature(new_f)
                    if is_signed_directional_feature(new_f):
                        cond.operator = "event"
                        cond.threshold = float(rng.choice([1.0, -1.0]))
                    elif is_binary_feature(new_f):
                        cond.operator = "gt"
                        cond.threshold = 0.5
                    else:
                        cond.operator = rng.choice(OPERATOR_OPTIONS)
                        cond.threshold = float(rng.choice(THRESHOLD_GRID))
            elif sub_mut == "dir":
                curr_dir = getattr(child.trigger, "direction_mode", "continuation")
                child.trigger.direction_mode = "fade" if curr_dir == "continuation" else "continuation"

            # 25% chance to toggle direction mode on trigger
            if rng.uniform(0.0, 1.0) < 0.25 and target_cond_name == "trigger":
                child.trigger.direction_mode = "fade" if getattr(child.trigger, "direction_mode", "continuation") == "continuation" else "continuation"

            # Always reset and re-evaluate is_regime_ceiling on mutated condition
            cond.is_regime_ceiling = False
            cond.__post_init__()

    # 5. Exit Spec Mutation (SL, TP, Holding, 4 Archetypes, Trailing ATR)
    if mutation_type in ("exit", "combined") or rng.uniform(0.0, 1.0) < 0.35:
        new_sl = float(rng.choice(SL_ATR_OPTIONS))
        valid_tps = [tp for tp in TP_ATR_OPTIONS if tp >= new_sl]
        new_tp = float(rng.choice(valid_tps if valid_tps else [new_sl]))
        new_mh = int(rng.choice(HOLDING_BARS_OPTIONS))
        new_exit_type = str(rng.choice(EXIT_TYPE_OPTIONS))
        new_trailing_atr = float(rng.choice(TRAILING_ATR_OPTIONS))
        child.exit = ExitSpec(
            exit_type=new_exit_type,
            sl_atr=new_sl,
            tp_atr=new_tp,
            max_holding_bars=new_mh,
            trailing_atr=new_trailing_atr,
        )

    # Invariant checks
    child.exit.tp_atr = max(child.exit.tp_atr, child.exit.sl_atr)

    if not child.validate_rules():
        # Fallback to pristine parent if rules violated
        return copy.deepcopy(parent_a)

    return child


def create_random_strategy(
    strategy_id: str,
    family: str,
    tier1_features: Sequence[str],
    rng: np.random.RandomState,
) -> Strategy:
    """
    Generates a fresh, valid orthogonal strategy of depth k in {2, 3, 4} conditions
    with random ExitSpec, random setup memory windows, and random direction_mode.
    """
    feature_source_map = {f: _get_source_from_feature(f) for f in tier1_features}
    unique_sources = sorted(set(feature_source_map.values()))
    dir_feats = [f for f in tier1_features if not Condition(f, feature_source_map[f], "gt", 50.0).is_regime_ceiling]
    dir_sources = sorted(set(feature_source_map[f] for f in dir_feats))
    trig_src = rng.choice(dir_sources)
    trig_feats = [f for f in dir_feats if feature_source_map[f] == trig_src]
    trig_feat = rng.choice(trig_feats)
    trig_cond = create_random_condition(trig_feat, rng)
    trig_cond.direction_mode = str(rng.choice(["continuation", "fade"]))

    other_sources = [s for s in unique_sources if s != trig_src]
    k = int(rng.choice([1, 2, 3]))
    selected_sources = list(rng.choice(other_sources, size=min(k, len(other_sources)), replace=False))

    context_cond = None
    setup_cond = None
    setup_exp = 3
    context_2_cond = None

    if len(selected_sources) >= 1:
        if rng.uniform(0.0, 1.0) < 0.60:
            setup_feats = [f for f in tier1_features if feature_source_map[f] == selected_sources[0]]
            setup_cond = create_random_condition(rng.choice(setup_feats), rng)
            setup_exp = int(rng.choice(SETUP_EXPIRY_OPTIONS))
        else:
            ctx_feats = [f for f in tier1_features if feature_source_map[f] == selected_sources[0]]
            context_cond = create_random_condition(rng.choice(ctx_feats), rng)

    if len(selected_sources) >= 2:
        if setup_cond is None:
            setup_feats = [f for f in tier1_features if feature_source_map[f] == selected_sources[1]]
            setup_cond = create_random_condition(rng.choice(setup_feats), rng)
            setup_exp = int(rng.choice(SETUP_EXPIRY_OPTIONS))
        elif context_cond is None:
            ctx_feats = [f for f in tier1_features if feature_source_map[f] == selected_sources[1]]
            context_cond = create_random_condition(rng.choice(ctx_feats), rng)
        else:
            c2_feats = [f for f in tier1_features if feature_source_map[f] == selected_sources[1]]
            context_2_cond = create_random_condition(rng.choice(c2_feats), rng)

    if len(selected_sources) >= 3 and context_2_cond is None:
        c2_feats = [f for f in tier1_features if feature_source_map[f] == selected_sources[2]]
        context_2_cond = create_random_condition(rng.choice(c2_feats), rng)

    sl = float(rng.choice(SL_ATR_OPTIONS))
    valid_tps = [tp for tp in TP_ATR_OPTIONS if tp >= sl]
    tp = float(rng.choice(valid_tps if valid_tps else [sl]))
    exit_type = str(rng.choice(EXIT_TYPE_OPTIONS))
    max_h = int(rng.choice(HOLDING_BARS_OPTIONS))
    trail = float(rng.choice(TRAILING_ATR_OPTIONS))

    strat = Strategy(
        strategy_id=strategy_id,
        family=family,
        trigger=trig_cond,
        setup=setup_cond,
        setup_expiry=setup_exp,
        context=context_cond,
        context_2=context_2_cond,
        exit=ExitSpec(
            exit_type=exit_type,
            sl_atr=sl,
            tp_atr=tp,
            max_holding_bars=max_h,
            trailing_atr=trail,
        ),
    )
    if not strat.validate_rules():
        strat.setup = None
        strat.context = None
        strat.context_2 = None

    return strat

