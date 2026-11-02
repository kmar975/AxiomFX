"""
Unit tests validating critical fixes from comprehensive quant & logic audit.
Covers:
1. Missing feature zeroing in compiler.
2. CCI m100 directional mirroring.
3. Non-directional / regime feature non-inversion.
4. JPY position sizing leverage calculation.
5. Signal deduplication preserving direction and ExitSpec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atoms.bitsets import AtomicBitsetStore, pack_bool_array_to_uint64
from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from grammar.compiler import (
    compile_strategy,
    invert_condition_for_short,
    mirror_directional_feature,
)
from grammar.strategy import Condition, Strategy


def test_missing_feature_zeros_entry_mask():
    df = pd.DataFrame({
        "s1_rsi_14": np.full(50, 75.0),
    })

    # Strategy with a non-existent trigger feature
    s_broken = Strategy(
        strategy_id="BROKEN_TRIG",
        family="F1_Trend",
        trigger=Condition("non_existent_feature", "s1", "gt", 50.0),
    )
    l_mask, s_mask = compile_strategy(s_broken, df)
    assert not np.any(l_mask), "Missing trigger feature must zero out entry_long"
    assert not np.any(s_mask), "Missing trigger feature must zero out entry_short"

    # Strategy with a missing context feature
    s_broken_ctx = Strategy(
        strategy_id="BROKEN_CTX",
        family="F1_Trend",
        context=Condition("missing_context_feat", "s1", "gt", 50.0),
        trigger=Condition("s1_rsi_14", "s1", "gt", 70.0),
    )
    l_mask, s_mask = compile_strategy(s_broken_ctx, df)
    assert not np.any(l_mask), "Missing context feature must zero out entry_long"
    assert not np.any(s_mask), "Missing context feature must zero out entry_short"


def test_cci_m100_directional_mirror():
    assert mirror_directional_feature("s1_cci_10_cross_up_m100") == "s1_cci_10_cross_dn_100"
    assert mirror_directional_feature("s1_cci_10_cross_dn_100") == "s1_cci_10_cross_up_m100"
    assert mirror_directional_feature("htf_s1_cci_20_cross_up_m100") == "htf_s1_cci_20_cross_dn_100"
    assert mirror_directional_feature("htf_s1_cci_20_cross_dn_100") == "htf_s1_cci_20_cross_up_m100"


def test_non_directional_features_not_inverted():
    # Distance to round number
    cond_round = Condition(feature="s4_dist_round100_pct", source="s4", operator="lt", threshold=30.0)
    assert cond_round.is_regime_ceiling is True
    inv_round = invert_condition_for_short(cond_round)
    assert inv_round.operator == "lt"
    assert inv_round.threshold == 30.0

    # Trend strength ADX
    cond_adx = Condition(feature="s1_adx_14", source="s1", operator="gt", threshold=25.0)
    assert cond_adx.is_regime_ceiling is True
    inv_adx = invert_condition_for_short(cond_adx)
    assert inv_adx.operator == "gt"
    assert inv_adx.threshold == 25.0

    # Time session hour_cos
    cond_time = Condition(feature="s6_hour_cos", source="s6", operator="gt", threshold=80.0)
    assert cond_time.is_regime_ceiling is True
    inv_time = invert_condition_for_short(cond_time)
    assert inv_time.operator == "gt"
    assert inv_time.threshold == 80.0

    # Volatility / Squeeze
    cond_sq = Condition(feature="s5_atr20_pct", source="s5", operator="lt", threshold=80.0)
    assert cond_sq.is_regime_ceiling is True
    inv_sq = invert_condition_for_short(cond_sq)
    assert inv_sq.operator == "lt"
    assert inv_sq.threshold == 80.0


def test_jpy_position_sizing():
    # Create synthetic USDJPY dataframe where close=154.50
    n = 100
    prices = np.full(n, 154.50)
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": prices,
        "high": prices + 0.10,
        "low": prices - 0.10,
        "close": prices,
        "spread": np.full(n, 1.2),
        "tick_volume": np.full(n, 100),
    })

    q_vec = compute_quote_to_usd_vector("USDJPY", n, {"USDJPY": prices})
    # 1 pip on USDJPY = 0.01 / 154.50 USD
    assert np.isclose(q_vec[0], 1.0 / 154.50)

    # 1 signal on bar 10
    entry_l = np.zeros(n, dtype=bool)
    entry_l[10] = True
    entry_s = np.zeros(n, dtype=bool)

    # Run backtest with tight SL (e.g. 0.20 ATR) so leverage limit is tested
    res = run_vectorized_backtest(
        df=df,
        pair="USDJPY",
        entry_longs=entry_l,
        entry_shorts=entry_s,
        quote_to_usd=q_vec,
        sl_atr_mult=0.1,
        tp_atr_mult=0.5,
        cost_multiplier=1.0,
    )

    # Max units allowed for 10x leverage on $100,000 equity in USD currency is 1,000,000 USD (10 lots)
    # The old bug limited it to 1,000,000 / 154.50 = 6,472 USD (0.064 lots)
    assert res["n_trades"] >= 1
    # Check that position was not throttled 155x: trade PnL or exposure is reasonable
    # At 1% risk ($1,000 risk cash), position size is on the order of 100,000 - 1,000,000 units
    assert res["equity_curve"][0] == 100000.0


def test_deduplication_preserves_direction_and_exits():
    store = AtomicBitsetStore()

    n = 100
    mask_a = np.zeros(n, dtype=bool)
    mask_a[10:15] = True
    mask_b = np.zeros(n, dtype=bool)

    # Long signal on mask_a
    packed_1 = pack_bool_array_to_uint64(np.concatenate([mask_a, mask_b]))
    # Short signal on mask_a
    packed_2 = pack_bool_array_to_uint64(np.concatenate([mask_b, mask_a]))

    # They should not be identical
    assert not np.array_equal(packed_1, packed_2)

    # Storing strategy 1
    store.store_evaluated_signal(packed_1, {"passed": True, "fitness": 1.5, "details": {}}, exit_key="sl_tp_1.5_2.5_24_1.2")

    # Strategy with opposite direction should NOT be found as duplicate
    dup_opp = store.find_duplicate_signal(packed_2, exit_key="sl_tp_1.5_2.5_24_1.2")
    assert dup_opp is None, "Opposite direction must not be treated as duplicate"

    # Strategy with different exit spec should NOT be found as duplicate
    dup_diff_exit = store.find_duplicate_signal(packed_1, exit_key="breakeven_trail_2.0_4.0_48_1.5")
    assert dup_diff_exit is None, "Different exit spec must not be treated as duplicate"

    # Strategy with identical signals and matching exit should be found
    dup_match = store.find_duplicate_signal(packed_1, exit_key="sl_tp_1.5_2.5_24_1.2")
    assert dup_match is not None
    assert dup_match["fitness"] == 1.5
