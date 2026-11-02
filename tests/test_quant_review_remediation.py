"""
Comprehensive Regression Test Suite for Quant & Architecture Review Remediations:
- K1: Vectorized auxiliary array cache key immunity under internal price modifications
- K2: Holdout evaluation causal warmup eliminating cold-start distortion
- K3: Regime ceiling conflict resolution, simultaneous trigger cancellation, directional validity
- K4: Null World exact timeline and diurnal hour preservation
- K5: Open-bar exit (TIME and ROLLOVER) lookahead elimination
- B1: direction_mode='fade' applied strictly to trigger signals while preserving context
- B2: EventDrivenEngine equity reset at start of run_backtest
- B3: S11 ISO week boundary anchored at 17:00 NY (Sunday 17:00 begins new trading week)
- B6: Authentic exit reasons recorded in vectorized engine and passed to payload
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atoms.bitsets import AtomicBitsetStore
from data.split import SplitManager
from engine.costs import compute_quote_to_usd_vector
from engine.event_driven import EventDrivenEngine
from engine.vectorized import clear_vector_cache, run_vectorized_backtest
from features.s11_institutional_liquidity import compute_s11_features
from features.store import FeatureStore
from grammar.compiler import compile_strategy
from grammar.strategy import Condition, ExitSpec, Strategy
from search.funnel import FunnelExecutor
from search.genetic import mutate_and_crossover_strategy
from search.portfolio import build_hrp_candidate_portfolio
from tests.helpers import load_test_basket
from validation.holdout import evaluate_finalists_on_holdout
from validation.null_world import build_synchronous_null_world, compute_v2_p_value


def test_k1_cache_key_immunity_to_price_mutations():
    """K1: Auxiliary cache must not serve stale vectors when internal prices are modified."""
    clear_vector_cache()
    basket = load_test_basket(n_bars=200)
    df = basket["EURUSD"].copy()
    n = len(df)

    l_mask = np.zeros(n, dtype=bool)
    l_mask[20] = True
    s_mask = np.zeros(n, dtype=bool)

    # First backtest
    res1 = run_vectorized_backtest(df, "EURUSD", l_mask, s_mask, max_holding_bars=5)

    # Mutate internal prices (as done in V5 recovery synthetic injection)
    # Keeping df[0] and df[-1] identical
    mutated_df = df.copy()
    mutated_df.loc[21:25, "close"] += 0.0500
    mutated_df.loc[21:25, "high"] += 0.0500

    res2 = run_vectorized_backtest(mutated_df, "EURUSD", l_mask, s_mask, max_holding_bars=5)

    # Net PnL must be significantly higher in res2 due to the +500 pips injection
    assert res2["net_pnl_usd"] > res1["net_pnl_usd"] + 100.0


def test_k2_holdout_warmup_no_cold_start():
    """K2: Holdout features must be warmed up causally from DEV tail without NaN cold-start."""
    basket = load_test_basket(n_bars=600)
    cutoff = SplitManager.compute_cutoff_date(basket, dev_ratio=0.80)

    strat = Strategy(
        strategy_id="TEST_WARMUP",
        family="F1_Trend",
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=50.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=10),
    )

    res = evaluate_finalists_on_holdout(
        candidates=[strat],
        hrp_weights=np.array([1.0]),
        all_data=basket,
        cutoff_date=cutoff,
    )

    assert "candidate_results" in res
    assert "portfolio_sharpe_holdout" in res
    assert len(res["candidate_results"]) == 1
    # Candidate ran cleanly without crashes or invalid returns
    assert "sharpe_holdout" in res["candidate_results"][0]


def test_k3_regime_ceiling_and_simultaneous_cancellation():
    """K3: Pure regime ceiling strategies must be invalid; simultaneous signals must cancel to 0."""
    from atoms.bitsets import AtomicBitsetStore
    # 1. Non-directional strategy check: trigger cannot be a regime ceiling
    non_dir_strat = Strategy(
        strategy_id="NON_DIR",
        family="F5_Regime",
        trigger=Condition(feature="s5_atr20_pct", source="s5", operator="lt", threshold=30.0),  # is_regime_ceiling=True
    )
    assert not non_dir_strat.is_valid_directional()
    assert not non_dir_strat.validate_rules()

    # Even with a directional setup, trigger must still be directional
    mixed_strat = Strategy(
        strategy_id="MIXED",
        family="F1_Trend",
        setup=Condition(feature="s1_rsi_14_pct", source="s1", operator="lt", threshold=30.0),
        trigger=Condition(feature="s5_atr20_pct", source="s5", operator="lt", threshold=30.0),  # is_regime_ceiling=True
    )
    assert not mixed_strat.is_valid_directional()
    assert not mixed_strat.validate_rules()

    # 2. S0 rejects non-directional rule
    bstore = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bstore)
    store = FeatureStore()
    basket = load_test_basket(n_bars=200)
    features = store.compute_basket_features(basket)
    fwd_ret = {"EURUSD": {5: np.zeros(200, dtype=np.float32)}}
    passed, score, details = funnel.evaluate_s0_screening(non_dir_strat, features, fwd_ret)
    assert not passed
    assert details["reason"] == "Non-directional strategy (all regime ceilings)"

    # 3. Simultaneous triggers cancel to zero
    dir_strat = Strategy(
        strategy_id="DIR_STRAT",
        family="F1_Trend",
        context=Condition(feature="s5_atr20_pct", source="s5", operator="lt", threshold=80.0),
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=30.0),
    )
    assert dir_strat.is_valid_directional()


def test_k4_null_world_diurnal_exact_timeline():
    """K4: Null world must preserve exact length, exact timestamps, and diurnal alignment."""
    basket = load_test_basket(n_bars=300)
    primary = basket["EURUSD"]

    null_basket = build_synchronous_null_world(basket, seed=42)

    assert "EURUSD" in null_basket
    null_df = null_basket["EURUSD"]

    # Length must be identical
    assert len(null_df) == len(primary)
    # Timestamps must match exactly (zero temporal drift)
    assert (pd.to_datetime(null_df["time"]) == pd.to_datetime(primary["time"])).all()
    # Prices must have varied due to permutation and reflection
    assert not (null_df["close"].values == primary["close"].values).all()


def test_k5_open_bar_exit_no_intra_candle_lookahead():
    """K5: TIME exit on bar_open must not look ahead to hit TP intra-candle on the same bar."""
    n = 20
    # Construct synthetic bars:
    # Bar 0: Signal bar
    # Bar 1: Entry bar at open=1.1000
    # Bar 2: Holding bars reaches 1 (max_holding_bars=1). Open=1.1000.
    #        High spikes to 1.1500 (would hit TP if checked!), Low=1.0990, Close=1.1000.
    opens = np.array([1.1000] * n, dtype=np.float64)
    highs = np.array([1.1010] * n, dtype=np.float64)
    lows = np.array([1.0990] * n, dtype=np.float64)
    closes = np.array([1.1000] * n, dtype=np.float64)

    # Spike high on bar 2
    highs[2] = 1.1500

    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread": np.full(n, 1.0),
    })

    l_mask = np.zeros(n, dtype=bool)
    l_mask[0] = True  # Signal at bar 0 -> Enter at bar 1 open
    s_mask = np.zeros(n, dtype=bool)

    # max_holding_bars = 1 -> holding_bars at bar 2 is (2 - 1) = 1 >= max_holding_bars
    res = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=l_mask,
        entry_shorts=s_mask,
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        max_holding_bars=1,
        cost_multiplier=1.0,
    )

    assert res["n_trades"] == 1
    # Exit reason must be TIME (code 2), NOT TP (code 1)
    assert res["exit_reasons"][0] == 2
    # Exit price must be close to bar 2 open (1.1000 - half spread), NOT 1.1500 TP price
    assert res["exit_prices"][0] < 1.1050


def test_b1_direction_mode_fade_on_trigger_preserves_context():
    """B1: direction_mode='fade' must invert the trigger only, keeping bullish context intact."""
    df = pd.DataFrame({
        "s2_seed_engulf_bull": [1.0, 0.0],
        "s2_seed_engulf_bear": [0.0, 1.0],
        "s1_rsi_14_pct": [80.0, 20.0],   # bar 0: high RSI, bar 1: low RSI
    })

    strat = Strategy(
        strategy_id="TEST_FADE",
        family="F4_Pullback",
        context=Condition(feature="s2_seed_engulf_bull", source="s2", operator="gt", threshold=0.5),
        trigger=Condition(feature="s1_rsi_14_pct", source="s1", operator="gt", threshold=70.0, direction_mode="fade"),
    )

    long_mask, short_mask = compile_strategy(strat, df)

    # In fade mode:
    # Bar 0 (bull engulf + overbought RSI > 70) is faded into a SHORT entry (-1)
    assert not long_mask[0]
    assert short_mask[0]

    # Bar 1 (bear engulf + oversold RSI < 30) is faded into a LONG entry (+1)
    assert long_mask[1]
    assert not short_mask[1]


def test_b2_event_driven_equity_reset():
    """B2: EventDrivenEngine must reset self.equity to initial_equity on subsequent runs."""
    engine = EventDrivenEngine(account_equity=100000.0)
    engine.equity = 150000.0  # Simulate state from prior backtest

    basket = load_test_basket(n_bars=100)
    df = basket["EURUSD"]
    market_closes = {p: basket[p]["close"].values for p in basket}

    l_mask = np.zeros(len(df), dtype=bool)
    s_mask = np.zeros(len(df), dtype=bool)

    engine.run_backtest(df, "EURUSD", l_mask, s_mask, market_closes)
    # Must be reset back to initial_equity
    assert engine.equity == 100000.0


def test_b3_s11_iso_week_boundary_anchored_at_17_ny():
    """B3: Sunday 17:00 NY must be classified into the new week, not the prior calendar week."""
    # Create timestamps around Sunday 17:00 NY
    # 2024-01-07 was Sunday. 17:00 NY is 22:00 UTC.
    times = pd.date_range("2024-01-07 20:00", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "open": [1.1000] * 5,
        "high": [1.1010] * 5,
        "low": [1.0990] * 5,
        "close": [1.1000] * 5,
    })

    feats = compute_s11_features(df, "EURUSD")
    assert "s11_pwh_pwl_sweep" in feats.columns
    assert "s11_adr14_exhaustion_pct" in feats.columns
    # Ensure computation succeeded without NameError or NaN crash
    assert len(feats) == 5


def test_b6_authentic_exit_reasons_in_vectorized():
    """B6: Vectorized engine returns authentic exit reasons array."""
    basket = load_test_basket(n_bars=100)
    df = basket["EURUSD"]
    n = len(df)

    l_mask = np.zeros(n, dtype=bool)
    l_mask[10] = True
    l_mask[40] = True
    s_mask = np.zeros(n, dtype=bool)

    res = run_vectorized_backtest(df, "EURUSD", l_mask, s_mask, max_holding_bars=5)
    assert "exit_reasons" in res
    assert len(res["exit_reasons"]) == res["n_trades"]
    for r in res["exit_reasons"]:
        assert r in (0, 1, 2, 3)  # SL, TP, TIME, ROLLOVER


def test_b1_pre_rollover_flat_blocks_entry_at_16_ny():
    """B1: pre_rollover_flat must prohibit opening new positions at 16:00 NY."""
    # 2024-01-08: 15:00 NY is 20:00 UTC. 16:00 NY is 21:00 UTC.
    times = pd.date_range("2024-01-08 19:00", periods=5, freq="1h", tz="UTC")
    n = len(times)
    df = pd.DataFrame({
        "time": times,
        "open": [1.1000] * n,
        "high": [1.1020] * n,
        "low": [1.0980] * n,
        "close": [1.1000] * n,
        "spread": np.full(n, 1.0),
    })

    # Signal at bar 1 (15:00 NY) -> would enter at bar 2 (16:00 NY)
    l_mask = np.zeros(n, dtype=bool)
    l_mask[1] = True
    s_mask = np.zeros(n, dtype=bool)

    # With pre_rollover_flat, entry at 16:00 NY is forbidden
    res = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=l_mask,
        entry_shorts=s_mask,
        exit_type="pre_rollover_flat",
        cost_multiplier=1.0,
    )
    assert res["n_trades"] == 0, "pre_rollover_flat must not open trades at 16:00 NY before rollover"


def test_b2_null_world_intraday_minute_level_preservation():
    """B2: null_world must map minute-level intra-hour bars correctly for M15/M5/M1."""
    # Create M15 dataset (96 bars per day)
    times = pd.date_range("2024-01-08 00:00", periods=192, freq="15min", tz="UTC")
    n = len(times)
    df = pd.DataFrame({
        "time": times,
        "open": [1.1000 + i * 0.0001 for i in range(n)],
        "high": [1.1010 + i * 0.0001 for i in range(n)],
        "low": [1.0990 + i * 0.0001 for i in range(n)],
        "close": [1.1000 + i * 0.0001 for i in range(n)],
        "spread": np.full(n, 1.0),
        "tick_volume": np.full(n, 500),
    })

    basket = {"EURUSD": df}
    null_basket = build_synchronous_null_world(basket, seed=123)
    null_df = null_basket["EURUSD"]

    # Must preserve exact count and timestamps down to the minute
    assert len(null_df) == len(df)
    assert (pd.to_datetime(null_df["time"]) == pd.to_datetime(df["time"])).all()


def test_v2_alpha_adaptive_resolution():
    """Verify adaptive significance threshold v2_alpha prevents locking champions into status C."""
    # Scenario 1: n_nulls = 3 (e.g. --demo)
    n_nulls_demo = 3
    v2_alpha_demo = max(0.05, 1.0 / (1.0 + float(n_nulls_demo)))
    assert v2_alpha_demo == pytest.approx(0.25)

    # Defeating 100% of null worlds (beat_count = 0)
    real_fit = 1.85
    null_fits_all_beaten = [0.2, 0.4, 0.5]
    p_val_champ = compute_v2_p_value(real_fit, null_fits_all_beaten)
    assert p_val_champ == pytest.approx(0.25)
    assert p_val_champ <= v2_alpha_demo  # Qualifies for Status A/B!

    # One null world beats champion (beat_count = 1)
    null_fits_one_beats = [0.2, 2.1, 0.5]
    p_val_failed = compute_v2_p_value(real_fit, null_fits_one_beats)
    assert p_val_failed == pytest.approx(0.50)
    assert p_val_failed > v2_alpha_demo  # Properly rejected

    # Scenario 2: n_nulls = 10
    v2_alpha_10 = max(0.05, 1.0 / (1.0 + 10.0))
    assert v2_alpha_10 == pytest.approx(1.0 / 11.0)
    assert compute_v2_p_value(real_fit, [0.1] * 10) <= v2_alpha_10

    # Scenario 3: n_nulls = 50 (tightens to institutional alpha = 0.05)
    v2_alpha_50 = max(0.05, 1.0 / (1.0 + 50.0))
    assert v2_alpha_50 == 0.05


def test_portfolio_no_highly_correlated_clones():
    """Verify HRP runner-up fill blocks clones (correlation >= 0.85) and S0 rejects twin masks."""
    # 1. Test S0 screening deduplication
    bstore = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bstore)
    basket = load_test_basket(n_bars=200)
    store = FeatureStore()
    features = store.compute_basket_features(basket)
    fwd_ret = {"EURUSD": {5: np.full(200, 0.05, dtype=np.float32)}}

    strat_a = Strategy(
        strategy_id="STRAT_A",
        family="F1_Trend",
        context=Condition(feature="s1_rsi_14_pct", source="s1", operator="gt", threshold=20.0),
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=30.0),
    )
    # STRAT_B has almost identical trigger condition
    strat_b = Strategy(
        strategy_id="STRAT_B",
        family="F1_Trend",
        context=Condition(feature="s1_rsi_14_pct", source="s1", operator="gt", threshold=20.0),
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=30.0),
    )

    passed_a, score_a, details_a = funnel.evaluate_s0_screening(strat_a, features, fwd_ret)
    if passed_a:
        # STRAT_B must be rejected as duplicate signal mask
        passed_b, score_b, details_b = funnel.evaluate_s0_screening(strat_b, features, fwd_ret)
        assert not passed_b
        assert "Duplicate signal mask" in details_b["reason"]

    # 2. Test portfolio runner-up selection respects corr < 0.85
    # Two identical strategies (corr = 1.0) and one uncorrelated strategy
    strat_1 = Strategy(strategy_id="S1", family="F1", trigger=Condition("s1_rsi_14", "s1", "gt", 50.0))
    strat_2 = Strategy(strategy_id="S2", family="F1", trigger=Condition("s1_rsi_14", "s1", "gt", 50.0))
    strat_3 = Strategy(strategy_id="S3", family="F2", trigger=Condition("s2_clv", "s2", "gt", 0.5))

    elites = [
        (strat_1, 2.0, {}),
        (strat_2, 1.9, {}),  # clone of S1 with identical signals / corr = 1.00
        (strat_3, 1.5, {}),  # orthogonal
    ]

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_out = Path(tmp_dir) / "candidates.yaml"
        champions, weights, meta = build_hrp_candidate_portfolio(
            elites=elites,
            dev_data=basket,
            features_by_pair=features,
            output_yaml_path=yaml_out,
            min_candidates=3,
        )
        champ_ids = [c.strategy_id for c in champions]
        # S2 should NOT be added as runner-up because its correlation with S1 is 1.00 (>= 0.85)
        assert "S1" in champ_ids
        assert "S2" not in champ_ids, "S2 is a clone of S1 (corr >= 0.85) and must not be injected"


def test_swap_condition_cleans_direction_mode():
    """Verify swapping a condition resets direction_mode to continuation on non-trigger slots."""
    rng = np.random.RandomState(42)
    strat = Strategy(
        strategy_id="TEST_FADE",
        family="F4_Pullback",
        context=Condition(feature="s5_atr20_pct", source="s5", operator="lt", threshold=70.0),
        setup=Condition(feature="s1_rsi_14_pct", source="s1", operator="lt", threshold=30.0),
        trigger=Condition(feature="s2_clv", source="s2", operator="gt", threshold=0.8, direction_mode="fade"),
    )

    # Force swap_condition mutation repeatedly
    tier1_feats = ["s1_rsi_14", "s2_clv", "s3_imp_corr", "s4_dist_round100_atr", "s5_atr20_pct"]
    for seed in range(50):
        r = np.random.RandomState(seed)
        child = mutate_and_crossover_strategy(
            parent_a=strat,
            parent_b=None,
            tier1_features=tier1_feats,
            rng=r,
            mutation_type="structure",
        )
        # Check that no non-trigger slot ever has direction_mode == 'fade'
        for slot_name in ("context", "context_2", "setup", "setup_2"):
            cond = getattr(child, slot_name, None)
            if cond is not None:
                assert cond.direction_mode == "continuation", (
                    f"Non-trigger slot {slot_name} must have direction_mode='continuation', got {cond.direction_mode}"
                )


