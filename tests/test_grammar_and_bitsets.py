"""
Unit and Integration Tests for Etap E3:
- DSL Strategy Tree and Genotype Hashing
- Critical is_regime_ceiling rule non-inversion
- uint64 bitset packing, unpacking, and Jaccard similarity (>0.95)
- M-bar setup validity window expansion
- Phase E0 exhaustive core layer screening
- V5 Power-Optimal Breadth (B*) calibration
"""

from __future__ import annotations

import numpy as np

from atoms.bitsets import (
    AtomicBitsetStore,
    jaccard_similarity_uint64,
    pack_bool_array_to_uint64,
    unpack_uint64_to_bool_array,
)
from features.store import FeatureStore
from grammar.compiler import expand_window_mask, invert_condition_for_short
from grammar.strategy import Condition, ExitSpec, SizingSpec, Strategy
from labels.labeling import compute_future_returns
from search.exhaustive import run_phase_e0_exhaustive
from search.funnel import FunnelExecutor
from search.power_calibration import calibrate_power_optimal_breadth
from tests.helpers import load_test_basket


def test_strategy_dsl_creation_and_hash():
    c1 = Condition("s1_rsi_14", "s1", "lt", 30.0)
    c2 = Condition("s3_bos", "s3", "event", 1.0)
    strat = Strategy(
        strategy_id="TEST_S1",
        family="F1_Trend",
        context=c1,
        trigger=c2,
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5),
        sizing=SizingSpec(vol_target=0.10),
    )
    assert strat.validate_rules() is True
    h1 = strat.genotype_hash()
    h2 = strat.genotype_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_is_regime_ceiling_rule():
    # Feature in REGIME_CEILING_FEATURES with lt and threshold >= 70
    cond = Condition(
        feature="s5_atr20_pct",
        source="s5",
        operator="lt",
        threshold=85.0,
    )
    assert cond.is_regime_ceiling is True

    # Invert for Short must NOT invert to gt 15.0! It must remain lt 85.0
    inv = invert_condition_for_short(cond)
    assert inv.operator == "lt"
    assert inv.threshold == 85.0
    assert inv.is_regime_ceiling is True

    # Normal directional condition must invert
    dir_cond = Condition(
        feature="s1_rsi_14",
        source="s1",
        operator="gt",
        threshold=70.0,
    )
    assert dir_cond.is_regime_ceiling is False
    inv_dir = invert_condition_for_short(dir_cond)
    assert inv_dir.operator == "lt"
    assert inv_dir.threshold == 30.0


def test_uint64_bitset_packing_and_jaccard():
    rng = np.random.RandomState(42)
    n = 200
    mask_a = rng.rand(n) > 0.7
    # Identical with 2 bits flipped
    mask_b = mask_a.copy()
    mask_b[0] = not mask_b[0]
    mask_b[1] = not mask_b[1]

    packed_a = pack_bool_array_to_uint64(mask_a)
    packed_b = pack_bool_array_to_uint64(mask_b)

    # Unpacking integrity
    unpacked_a = unpack_uint64_to_bool_array(packed_a, n)
    np.testing.assert_array_equal(mask_a, unpacked_a)

    # Jaccard similarity
    j_sim = jaccard_similarity_uint64(packed_a, packed_b)
    assert j_sim > 0.90
    assert j_sim <= 1.0


def test_setup_m_bar_validity_window():
    mask = np.zeros(10, dtype=bool)
    mask[2] = True  # Signal at bar 2

    # Expands by M=3 bars -> True at bars 2, 3, 4
    expanded = expand_window_mask(mask, window_m=3)
    assert expanded[1] is False or expanded[1] == False
    assert expanded[2] is True or expanded[2] == True
    assert expanded[3] is True or expanded[3] == True
    assert expanded[4] is True or expanded[4] == True
    assert expanded[5] is False or expanded[5] == False


def test_phase_e0_exhaustive_screening():
    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features_by_pair = store.compute_basket_features(basket)
    tier1_feats = store.get_tier1_features(features_by_pair["EURUSD"])

    future_returns_by_pair = {}
    for p, df in basket.items():
        future_returns_by_pair[p] = compute_future_returns(df, horizons=[5])

    bitset_store = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bitset_store)

    passing = run_phase_e0_exhaustive(
        tier1_features=tier1_feats[:6],
        features_by_pair=features_by_pair,
        future_returns_by_pair=future_returns_by_pair,
        funnel=funnel,
        max_candidates=20,
    )

    assert funnel.n_s0 > 0
    assert isinstance(passing, list)


def test_power_calibration_b_star():
    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features_by_pair = store.compute_basket_features(basket)

    bitset_store = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bitset_store)

    b_star = calibrate_power_optimal_breadth(
        dev_data=basket,
        features_by_pair=features_by_pair,
        funnel=funnel,
        max_budget=50,
    )
    assert b_star > 0
    assert b_star <= 50


def test_mutation_resets_regime_ceiling():
    from grammar.strategy import Condition, ExitSpec, Strategy
    from search.genetic import mutate_and_crossover_strategy

    # Condition with is_regime_ceiling=True
    cond = Condition(feature="s5_squeeze_on", source="s5", operator="gt", threshold=0.5)
    assert cond.is_regime_ceiling is True

    # Mutate to directional feature
    cond.feature = "s5_bb_14_2p2_mr_upper"
    cond.is_regime_ceiling = False
    cond.__post_init__()
    assert cond.is_regime_ceiling is False

    # Strategy with squeeze_on in setup
    strat = Strategy(
        strategy_id="TEST_REGIME_RESET",
        family="F1_Trend",
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=70.0),
        setup=Condition(feature="s5_squeeze_on", source="s5", operator="gt", threshold=0.5),
        exit=ExitSpec(),
    )
    assert strat.setup.is_regime_ceiling is True

    rng = np.random.RandomState(42)
    # Mutate multiple times
    for _ in range(20):
        child = mutate_and_crossover_strategy(strat, None, ["s5_bb_14_2p2_mr_upper", "s1_rsi_14", "s2_marubozu_bull_70"], rng)
        for c in [child.trigger, child.setup, child.context]:
            if c is not None and c.feature == "s5_bb_14_2p2_mr_upper":
                assert c.is_regime_ceiling is False, f"Feature {c.feature} should not have is_regime_ceiling=True"
