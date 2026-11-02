"""
Unit and Integration Tests for Etap E5:
- V2 Synchronous day-shuffled null world generation (17:00 NY anchor, epsilon_d, cross-pair alignment)
- V2 empirical p-value computation
- V4 strict HOLDOUT gatekeeper protection (HoldoutAccessViolation)
- V5 synthetic signal recovery verification
- Institutional stress testing suite
- Strategy card and final report generation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data.split import HoldoutAccessViolation, SplitManager
from features.store import FeatureStore
from grammar.strategy import Condition, ExitSpec, Strategy
from report.cards import generate_strategy_card_md
from report.export import generate_final_report_md
from tests.helpers import load_test_basket
from validation.null_world import build_synchronous_null_world, compute_v2_p_value
from validation.recovery import run_v5_recovery_test
from validation.stress import run_candidate_stress_tests


def test_synchronous_null_world_properties():
    basket = load_test_basket(n_bars=300)
    null_basket = build_synchronous_null_world(basket, seed=999)

    # 1. All pairs present
    assert set(null_basket.keys()) == set(basket.keys())

    # 2. Invariants preserved
    for df in null_basket.values():
        assert len(df) > 0
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        # OHLC integrity
        assert (h >= np.maximum(o, c) - 1e-6).all()
        assert (l <= np.minimum(o, c) + 1e-6).all()
        assert (h - l >= 0.0).all()


def test_v2_p_value_calculation():
    # If 0 nulls beat real score of 1.5 out of 5 nulls -> p = (1 + 0) / (1 + 5) = 1/6
    null_scores = [0.2, 0.6, 0.1, 0.8, -0.2]
    p_val = compute_v2_p_value(real_fitness=1.5, null_world_fitnesses=null_scores)
    assert p_val == (1 + 0) / (1 + 5)

    # If 2 nulls (0.6, 0.8) beat real score of 0.5 -> p = (1 + 2) / (1 + 5) = 3/6 = 0.50
    p_val2 = compute_v2_p_value(real_fitness=0.5, null_world_fitnesses=null_scores)
    assert p_val2 == 0.5


def test_v4_holdout_access_protection():
    basket = load_test_basket(n_bars=300)
    cutoff = SplitManager.compute_cutoff_date(basket, dev_ratio=0.80)

    # Access without unlock must raise HoldoutAccessViolation
    with pytest.raises(HoldoutAccessViolation):
        _ = SplitManager.get_holdout_data(basket, cutoff)

    # Access inside context must succeed
    with SplitManager.unlock_holdout_for_finalists():
        h_data = SplitManager.get_holdout_data(basket, cutoff)
        assert len(h_data) == len(basket)
        assert len(h_data["EURUSD"]) > 0


def test_v5_recovery_and_stress():
    basket = load_test_basket(n_bars=300)
    rec_res = run_v5_recovery_test(basket, snr_levels=[0.6, 1.0])
    assert "recalls" in rec_res
    assert len(rec_res["recalls"]) == 2

    # Stress tests
    strat = Strategy(
        strategy_id="TEST_STRESS",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 70.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5),
    )
    store = FeatureStore()
    eur_feats = store.compute_all_features_for_pair(basket["EURUSD"], "EURUSD")

    stress_res = run_candidate_stress_tests(strat, basket["EURUSD"], eur_feats, pair="EURUSD", n_bootstrap=20)
    assert "spread_2x_sharpe" in stress_res
    assert "latency_sharpe" in stress_res
    assert "jitter_sharpe" in stress_res


def test_reports_and_cards():
    temp_dir = Path(tempfile.gettempdir()) / "test_report_dir"
    temp_dir.mkdir(parents=True, exist_ok=True)

    strat = Strategy(
        strategy_id="TEST_CARD",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 70.0),
    )

    card_path = temp_dir / "card.md"
    card_md = generate_strategy_card_md(
        strategy=strat,
        metrics_dev={"fitness": 1.45, "win_rate_eur": 0.58, "profit_factor_eur": 1.7},
        metrics_holdout={"sharpe_holdout": 1.20, "win_rate": 0.55, "profit_factor": 1.5},
        status="A",
        v2_p_value=0.035,
        multi_pair_metrics={"EURUSD": {"sharpe": 1.4, "n_trades": 45}},
        stress_results={"spread_2x_sharpe": 1.05, "latency_sharpe": 1.10},
        output_path=card_path,
    )
    assert "STRATEGY CARD: TEST_CARD" in card_md
    assert card_path.exists()

    # Final report
    rep_path = temp_dir / "final_report.md"
    rep_md = generate_final_report_md(
        run_id="test_run",
        throughput_stats={"N_generated": 100, "N_S2": 50},
        b_star=60,
        candidates_meta=[{"rank": 1, "strategy_id": "TEST_CARD", "family": "F1_Trend", "fitness": 1.45, "status": "A"}],
        holdout_meta={"portfolio_sharpe_holdout": 1.35, "portfolio_total_return_pct": 12.5},
        output_path=rep_path,
    )
    assert "FINAL DISCOVERY REPORT" in rep_md
    assert rep_path.exists()
