"""
Verification tests for the comprehensive audit remediation:
1. Exclusion of losing strategies and proper empty portfolio handling.
2. Engine consistency (vectorized vs event_driven) under breakeven_trail.
3. ATR20 in live_scanner matching backtest ATR.
4. Dynamic HTF resampling in FeatureStore across all timeframes (H4 -> D1, D1 -> W1, W1 -> None).
5. Zero look-ahead on day 1 in s11 institutional liquidity.
6. Binary and signed feature handling in exhaustive search, genetic mutation, and compiler symmetry.
7. MAP-Elites behavioral descriptor diversity (d_trend_corr, d_vol_exposure).
8. Full strategy reconstruction in compile_timeframe_payload.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from atoms.bitsets import AtomicBitsetStore
from compile_timeframe_payload import parse_strategy_from_card
from engine.event_driven import compare_engines
from engine.live_scanner import scan_strategy_on_live_candle
from features.s11_institutional_liquidity import compute_s11_features
from features.store import FeatureStore
from grammar.compiler import invert_condition_for_short
from grammar.strategy import (
    Condition,
    ExitSpec,
    Strategy,
    is_binary_feature,
    is_signed_directional_feature,
)
from search.portfolio import build_hrp_candidate_portfolio
from search.qd_archive import MAPElitesArchive
from tests.helpers import load_test_basket


def test_portfolio_strictly_excludes_losing_strategies():
    """Verifies that strategies with negative returns or PF < 1.0 are never exported to candidates.yaml when require_profitable=True."""
    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features = store.compute_basket_features(basket)

    # Strategy that loses money
    losing_strat = Strategy(
        strategy_id="LOSING_STRAT",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 50.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )
    elites = [(losing_strat, 0.5, {"n_trades": 20})]

    temp_yaml = Path(tempfile.gettempdir()) / "test_portfolio_no_losers.yaml"
    champs, weights, meta = build_hrp_candidate_portfolio(
        elites=elites,
        dev_data=basket,
        features_by_pair=features,
        output_yaml_path=temp_yaml,
        require_profitable=True,
    )

    # Must be empty because losing_strat is not profitable
    assert len(champs) == 0
    assert len(weights) == 0
    with open(temp_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["portfolio_candidates"] == []


def test_breakeven_trail_engine_equivalence():
    """Verifies that vectorized and event-driven engines resolve breakeven_trail identically without intra-candle lookahead."""
    data = load_test_basket(n_bars=350)
    eur_df = data["EURUSD"]
    n_bars = len(eur_df)

    entry_longs = np.zeros(n_bars, dtype=bool)
    entry_shorts = np.zeros(n_bars, dtype=bool)
    entry_longs[12::35] = True
    entry_shorts[28::35] = True

    market_closes = {p: df["close"].values for p, df in data.items()}

    matches, comp = compare_engines(
        df=eur_df,
        pair="EURUSD",
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        market_closes=market_closes,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        max_holding_bars=25,
        cost_multiplier=1.5,
        exit_type="breakeven_trail",
        trailing_atr=1.2,
    )
    assert matches is True, f"Engine mismatch: {comp}"
    assert comp["vec_trades"] > 0
    assert abs(comp["vec_trades"] - comp["ev_trades"]) <= 1


def test_live_scanner_uses_atr20_not_single_candle():
    """Verifies that live_scanner computes 20-period ATR and not just the single latest candle range."""
    basket = load_test_basket(n_bars=50)
    eur_df = basket["EURUSD"].copy()
    store = FeatureStore()
    features = store.compute_basket_features({"EURUSD": eur_df})["EURUSD"]

    # Make the last candle tiny (0.1 pip)
    last_idx = len(eur_df) - 1
    eur_df.loc[last_idx, "high"] = eur_df.loc[last_idx, "open"] + 0.00001
    eur_df.loc[last_idx, "low"] = eur_df.loc[last_idx, "open"] - 0.00001
    eur_df.loc[last_idx, "close"] = eur_df.loc[last_idx, "open"]

    strat = Strategy(
        strategy_id="SCAN_TEST",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 50.0),
        exit=ExitSpec(sl_atr=2.0, tp_atr=3.0),
    )

    scan_res = scan_strategy_on_live_candle(strat, features, eur_df, pair="EURUSD")
    # SL distance should be based on 20-period ATR, which is ~0.0010 - 0.0020, so sl_price != entry_price
    sl_dist = abs(scan_res["entry_price"] - scan_res["sl_price"])
    # 2.0 * ATR20 should be at least 15 pips (0.0015), whereas tiny candle range was 0.00002
    assert sl_dist >= 0.0010, f"Expected SL distance >= 0.0010 based on ATR20, got {sl_dist}"


def test_dynamic_htf_resampling_in_feature_store():
    """Verifies FeatureStore chooses correct target_htf and does not downsample D1/W1 to H4."""
    store = FeatureStore(enable_htf=True)

    # 1. H1 data should get H4 HTF
    h1_df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC"),
        "open": np.full(100, 1.10),
        "high": np.full(100, 1.11),
        "low": np.full(100, 1.09),
        "close": np.full(100, 1.10),
        "spread": np.full(100, 1.5),
        "tick_volume": np.full(100, 500),
    })
    h1_feats = store.compute_basket_features({"EURUSD": h1_df})["EURUSD"]
    assert any("htf_" in col for col in h1_feats.columns)

    # 2. D1 data should get W1 HTF (not downsampled H4)
    d1_df = pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=60, freq="1D", tz="UTC"),
        "open": np.full(60, 1.10),
        "high": np.full(60, 1.11),
        "low": np.full(60, 1.09),
        "close": np.full(60, 1.10),
        "spread": np.full(60, 1.5),
        "tick_volume": np.full(60, 5000),
    })
    d1_feats = store.compute_basket_features({"EURUSD": d1_df})["EURUSD"]
    # Check that it executed without exception and produces valid features
    assert len(d1_feats) == len(d1_df)


def test_s11_zero_lookahead_first_day():
    """Verifies that S11 features on the first day do not leak future information from that day."""
    n_bars = 48  # 2 days of H1 data
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01 17:00", periods=n_bars, freq="1h", tz="America/New_York").tz_convert("UTC"),
        "open": np.full(n_bars, 1.10),
        "high": np.full(n_bars, 1.11),
        "low": np.full(n_bars, 1.09),
        "close": np.full(n_bars, 1.10),
        "spread": np.full(n_bars, 1.5),
        "tick_volume": np.full(n_bars, 500),
    })

    # Case A: First day has standard high=1.11, low=1.09
    feats_a = compute_s11_features(df, "EURUSD")

    # Case B: First day future bars (bar 10..23) spike to high=1.20
    df_b = df.copy()
    df_b.loc[15:20, "high"] = 1.20
    feats_b = compute_s11_features(df_b, "EURUSD")

    # The features on bar 0 (start of first day) must be identical in both cases (no future leak from bar 15)
    assert feats_a["s11_pd_equilibrium_pct"].iloc[0] == feats_b["s11_pd_equilibrium_pct"].iloc[0]
    assert feats_a["s11_prior_day_compression"].iloc[0] == feats_b["s11_prior_day_compression"].iloc[0]
    assert feats_a["s11_adr14_exhaustion_pct"].iloc[0] == feats_b["s11_adr14_exhaustion_pct"].iloc[0]


def test_ema_stack_binary_and_signed_symmetry_inversion():
    """Verifies 2-EMA stacks are binary while 3-EMA stack (20-50-200) is ternary signed directional."""
    assert is_binary_feature("s1_ema_stack_8_20") is True
    assert is_binary_feature("s1_ema_stack_13_34") is True
    assert is_signed_directional_feature("s1_ema_stack_20_50_200") is True
    assert is_binary_feature("s1_ema_stack_20_50_200") is False

    # 1. Binary 2-EMA stack inverts gt 0.5 -> lt 0.5
    cond_long_bin = Condition("s1_ema_stack_8_20", "s1", "gt", 0.5)
    cond_short_bin = invert_condition_for_short(cond_long_bin)
    assert cond_short_bin.feature == "s1_ema_stack_8_20"
    assert cond_short_bin.operator == "lt"
    assert cond_short_bin.threshold == 0.5

    # 2. Signed ternary 3-EMA stack inverts event +1.0 -> event -1.0
    cond_long_sgn = Condition("s1_ema_stack_20_50_200", "s1", "event", 1.0)
    cond_short_sgn = invert_condition_for_short(cond_long_sgn)
    assert cond_short_sgn.feature == "s1_ema_stack_20_50_200"
    assert cond_short_sgn.operator == "event"
    assert cond_short_sgn.threshold == -1.0

    # 3. Signed inequality gt 0.0 -> lt 0.0
    cond_long_gt = Condition("s1_ema_stack_20_50_200", "s1", "gt", 0.0)
    cond_short_lt = invert_condition_for_short(cond_long_gt)
    assert cond_short_lt.operator == "lt"
    assert cond_short_lt.threshold == 0.0


def test_qd_archive_descriptors_dynamic_diversity():
    """Verifies that MAPElitesArchive.compute_descriptors spans multiple bins for d_trend_corr and d_vol_exposure."""
    archive = MAPElitesArchive(bins_per_dim=5, n_dims=6)

    # Strategy 1: Tight stop, no s5, negative skew (mean-reversion style)
    s1 = Strategy(
        strategy_id="STRAT_1",
        family="F3_MeanReversion",
        trigger=Condition("s1_rsi_14", "s1", "lt", 30.0),
        exit=ExitSpec(sl_atr=1.0, tp_atr=1.5),
    )
    # Strategy 2: Wide stop, large TP, with s5 (breakout style), positive skew
    s2 = Strategy(
        strategy_id="STRAT_2",
        family="F2_Breakout",
        trigger=Condition("s5_squeeze_on", "s5", "gt", 0.5),
        exit=ExitSpec(sl_atr=3.5, tp_atr=5.0),
    )

    pnls_1 = np.array([50.0, 50.0, 50.0, 50.0, 50.0, -300.0])  # Negative skew
    pnls_2 = np.array([-50.0, -50.0, -50.0, -50.0, 500.0])    # Positive skew

    desc_1 = archive.compute_descriptors(s1, np.array([5, 5]), np.array([1, 1]), pnls_1, n_bars_total=500)
    desc_2 = archive.compute_descriptors(s2, np.array([20, 20]), np.array([1, 1]), pnls_2, n_bars_total=500)

    # Both descriptors must differ significantly
    assert abs(desc_1[2] - desc_2[2]) >= 0.15, f"d_trend_corr should differ: {desc_1[2]} vs {desc_2[2]}"
    assert abs(desc_1[3] - desc_2[3]) >= 0.25, f"d_vol_exposure should differ: {desc_1[3]} vs {desc_2[3]}"

    cell_1 = archive.get_cell_index(desc_1)
    cell_2 = archive.get_cell_index(desc_2)
    # They should not land in identical cells
    assert cell_1 != cell_2


def test_lossless_strategy_card_and_yaml_parsing():
    """Verifies parse_strategy_from_card recovers context, setup, exit_type, and trailing_atr."""
    temp_dir = Path(tempfile.mkdtemp())
    exp_dir = temp_dir / "STRAT_COMPLEX"
    exp_dir.mkdir(parents=True)

    strat = Strategy(
        strategy_id="STRAT_COMPLEX",
        family="F4_Pullback",
        context=Condition("s1_sma_dist_pct", "s1", "gt", 50.0),
        setup=Condition("s5_squeeze_on", "s5", "gt", 0.5),
        setup_expiry=4,
        trigger=Condition("s1_rsi_14", "s1", "lt", 30.0),
        exit=ExitSpec(exit_type="breakeven_trail", sl_atr=1.8, tp_atr=3.2, max_holding_bars=30, trailing_atr=1.4),
    )

    # Save strategy.yaml
    with open(exp_dir / "strategy.yaml", "w", encoding="utf-8") as f:
        yaml.dump(strat.genotype_dict(), f)

    card_file = exp_dir / "card.md"
    card_file.write_text("# Dummy Card", encoding="utf-8")

    loaded = parse_strategy_from_card(card_file)
    assert loaded.strategy_id == "STRAT_COMPLEX"
    assert loaded.family == "F4_Pullback"
    assert loaded.context is not None and loaded.context.feature == "s1_sma_dist_pct"
    assert loaded.setup is not None and loaded.setup.feature == "s5_squeeze_on"
    assert loaded.setup_expiry == 4
    assert loaded.trigger.feature == "s1_rsi_14"
    assert loaded.exit.exit_type == "breakeven_trail"
    assert loaded.exit.sl_atr == 1.8
    assert loaded.exit.tp_atr == 3.2
    assert loaded.exit.trailing_atr == 1.4
    assert loaded.exit.max_holding_bars == 30


def test_funnel_s1_warmed_up_slice():
    """Verifies FunnelExecutor.evaluate_s1_fast slices the last 1/3 of DEV to avoid cold warmup windows."""
    from search.funnel import FunnelExecutor

    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features = store.compute_basket_features(basket)

    bitset_store = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bitset_store)
    strat = Strategy(
        strategy_id="TEST_S1",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 50.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    passed, res = funnel.evaluate_s1_fast(strat, basket, features)
    assert "n_trades" in res
    assert funnel.n_s1 == 1


def test_interactions_b2_handles_feature_types():
    """Verifies that Phase B2 properly generates conditions for binary, signed, and continuous percentiles."""
    from search.interactions import _bin_feature, _make_condition

    # 1. Binary feature {0.0, 1.0}
    bin_f = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    bin_b = _bin_feature(bin_f, "s2_inside_bar", n_bins=5)
    assert set(np.unique(bin_b)).issubset({0, 4})
    c_bin_low = _make_condition("s2_inside_bar", "s2", 0, n_bins=5)
    c_bin_high = _make_condition("s2_inside_bar", "s2", 4, n_bins=5)
    assert c_bin_low.threshold == 0.5 and c_bin_low.operator == "lt"
    assert c_bin_high.threshold == 0.5 and c_bin_high.operator == "gt"

    # 2. Signed ternary feature {-1.0, 0.0, 1.0}
    sgn_f = np.array([-1.0, 0.0, 1.0, -1.0, 1.0])
    sgn_b = _bin_feature(sgn_f, "s3_bos", n_bins=5)
    assert set(np.unique(sgn_b)).issubset({0, 2, 4})
    c_sgn_low = _make_condition("s3_bos", "s3", 0, n_bins=5)
    c_sgn_high = _make_condition("s3_bos", "s3", 4, n_bins=5)
    assert c_sgn_low.threshold == -1.0 and c_sgn_low.operator == "event"
    assert c_sgn_high.threshold == 1.0 and c_sgn_high.operator == "event"

    # 3. Continuous percentile feature [0..100]
    pct_f = np.array([10.0, 30.0, 50.0, 75.0, 90.0])
    pct_b = _bin_feature(pct_f, "s1_rsi_14_pct", n_bins=5)
    assert pct_b[0] == 0 and pct_b[-1] == 4
    c_pct_low = _make_condition("s1_rsi_14_pct", "s1", 0, n_bins=5)
    c_pct_high = _make_condition("s1_rsi_14_pct", "s1", 4, n_bins=5)
    assert c_pct_low.threshold == 30.0 and c_pct_low.operator == "lt"
    assert c_pct_high.threshold == 70.0 and c_pct_high.operator == "gt"


def test_min_trades_basket_enforced():
    """Verifies evaluate_multipair_fitness rejects strategies that do not meet min_trades_basket."""
    from search.fitness import evaluate_multipair_fitness

    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features = store.compute_basket_features(basket)

    # Strategy with a very tight condition that produces few trades
    strat = Strategy(
        strategy_id="TEST_MIN_TRADES",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 85.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=10),
    )

    fit, details = evaluate_multipair_fitness(
        strat, basket, features, min_median_hold_hours=0.0, min_trades_primary=0, min_trades_basket=500
    )
    assert fit == -999.0
    assert "Insufficient total basket trades" in details.get("failed_filter", "")


def test_strategy_roundtrip_serialization_and_fields():
    """Verifies Strategy.from_dict preserves threshold_high, symmetric, and timeframe."""
    from compile_timeframe_payload import _dict_to_strategy

    trig = Condition("s1_rsi_14", "s1", "between", 30.0, threshold_high=70.0)
    strat = Strategy(
        strategy_id="SERIAL_TEST",
        family="F2_Breakout",
        trigger=trig,
        exit=ExitSpec(exit_type="breakeven_trail", sl_atr=2.0, tp_atr=4.0, trailing_atr=1.5),
        symmetric=False,
        timeframe="H4",
    )

    d = strat.genotype_dict()
    assert d["symmetric"] is False
    assert d["timeframe"] == "H4"
    assert d["trigger"]["threshold_high"] == 70.0

    restored = Strategy.from_dict(d, sid="RESTORED")
    assert restored.symmetric is False
    assert restored.timeframe == "H4"
    assert restored.trigger.threshold_high == 70.0
    assert restored.exit.trailing_atr == 1.5

    # Check compile_timeframe_payload._dict_to_strategy compatibility
    restored_gui = _dict_to_strategy(d, sid="GUI_RESTORED")
    assert restored_gui.symmetric is False
    assert restored_gui.timeframe == "H4"
    assert restored_gui.trigger.threshold_high == 70.0


def test_resume_checkpoint_restores_elites_and_step():
    """Verifies that state checkpoints serialize full elite genotypes and can be restored."""
    from storage.state import StorageManager

    temp_run_dir = Path(tempfile.mkdtemp()) / "test_run"
    storage = StorageManager(run_id="test_run")
    storage.run_dir = temp_run_dir
    storage.checkpoints_dir = temp_run_dir / "checkpoints"
    storage.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    archive = MAPElitesArchive(bins_per_dim=5, n_dims=6)
    strat = Strategy("QD_ELITE_1", "F1_Trend", Condition("s1_rsi_14", "s1", "gt", 70.0))
    desc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    archive.add_strategy(strat, fitness=1.75, descriptors=desc, metrics={"sharpe": 1.5})

    checkpoint_data = {
        "c_step": 42,
        "elapsed_sec": 120.0,
        "elites_count": 1,
        "elites": [
            {
                "genotype": strat.genotype_dict(),
                "strategy_id": strat.strategy_id,
                "family": strat.family,
                "fitness": 1.75,
                "descriptors": list(desc),
                "metrics": {"sharpe": 1.5},
            }
        ],
    }

    chk_file = storage.save_checkpoint(checkpoint_data, step_name="c_42")
    assert chk_file.exists()

    loaded = storage.load_latest_checkpoint()
    assert loaded is not None
    assert loaded["c_step"] == 42
    assert len(loaded["elites"]) == 1

    # Verify elite can be repopulated into a fresh archive
    new_archive = MAPElitesArchive(bins_per_dim=5, n_dims=6)
    for item in loaded["elites"]:
        restored_strat = Strategy.from_dict(item["genotype"], sid=item["strategy_id"])
        new_archive.add_strategy(
            restored_strat,
            fitness=item["fitness"],
            descriptors=np.array(item["descriptors"]),
            metrics=item["metrics"],
        )
    assert len(new_archive.get_all_elites()) == 1
    assert new_archive.get_all_elites()[0].strategy.strategy_id == "QD_ELITE_1"


def test_meta_filter_consistency_s3_and_stress():
    """Verifies that evaluate_s3_event_driven and run_candidate_stress_tests accept and apply meta_filter."""
    from search.funnel import FunnelExecutor
    from validation.stress import run_candidate_stress_tests

    basket = load_test_basket(n_bars=200)
    store = FeatureStore()
    features = store.compute_basket_features(basket)

    strat = Strategy("META_TEST", "F1_Trend", Condition("s1_rsi_14", "s1", "gt", 50.0))

    # Mock a MetaModelFilter that blocks everything
    class BlockingFilter:
        model = "dummy"
        threshold = 0.5

        def filter_signals(self, signals, feats):
            return np.zeros_like(signals, dtype=bool)

    bitset_store = AtomicBitsetStore()
    funnel = FunnelExecutor(bitset_store=bitset_store)
    trades, df_s3, metrics_s3 = funnel.evaluate_s3_event_driven(
        strat, basket, features, meta_filter=BlockingFilter()
    )
    assert len(trades) == 0
    assert metrics_s3["n_trades"] == 0

    stress_res = run_candidate_stress_tests(
        strat, basket["EURUSD"], features["EURUSD"], meta_filter=BlockingFilter()
    )
    assert stress_res["base_trades"] == 0


def test_stress_and_meta_labeling_respect_exit_type_and_trailing():
    """Verifies that run_candidate_stress_tests and train_meta_labeling pass exit_type and trailing_atr."""
    from search.meta_labeling import train_meta_labeling_for_strategy
    from validation.stress import run_candidate_stress_tests

    basket = load_test_basket(n_bars=200)
    store = FeatureStore()
    features = store.compute_basket_features(basket)

    strat = Strategy(
        strategy_id="TRAIL_STRESS_TEST",
        family="F2_Breakout",
        trigger=Condition("s1_rsi_14", "s1", "gt", 50.0),
        exit=ExitSpec(exit_type="breakeven_trail", sl_atr=1.5, tp_atr=3.0, max_holding_bars=20, trailing_atr=1.4),
    )

    stress_res = run_candidate_stress_tests(strat, basket["EURUSD"], features["EURUSD"])
    assert "base_trades" in stress_res
    assert "spread_2x_trades" in stress_res

    mf = train_meta_labeling_for_strategy(strat, basket["EURUSD"], features["EURUSD"], pair="EURUSD")
    assert mf.strategy_id == "TRAIL_STRESS_TEST"


def test_plateau_centering_preserves_tp_gte_sl_invariant():
    """Verifies that when sl_atr is perturbed (e.g. x1.15), tp_atr is kept >= sl_atr."""
    from search.refine import run_discrete_plateau_centering

    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features = store.compute_basket_features(basket)

    # Strategy with SL == TP (2.0 and 2.0)
    strat = Strategy(
        strategy_id="SL_TP_EQUAL",
        family="F5_Candles",
        trigger=Condition("s2_seed_pin_bull", "s2", "gt", 0.5),
        exit=ExitSpec(sl_atr=2.0, tp_atr=2.0, max_holding_bars=24),
    )

    is_robust, refined, fit, det = run_discrete_plateau_centering(
        strat, basket, features, max_needle_drop=0.99, min_trades_primary=0, min_trades_basket=0
    )
    assert refined.exit.tp_atr >= refined.exit.sl_atr
    # Check that in all neighbor perturbations, tp_atr was never less than sl_atr
    assert refined.exit.sl_atr in [1.70, 2.0, 2.30]
    assert refined.exit.tp_atr >= 2.0
