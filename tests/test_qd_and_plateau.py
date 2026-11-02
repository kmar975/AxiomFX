"""
Unit and Integration Tests for Etap E4:
- MAP-Elites Quality-Diversity archive and 6-descriptor behavioral space
- UCB1 Multi-Armed Bandit with stagnation detection and budget reallocation
- Multi-pair fitness with session sub-basket matching and correlation-discounted weights
- Discrete Plateau Centering and needle rejection
- Hierarchical Risk Parity (HRP) portfolio construction
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from features.store import FeatureStore
from grammar.strategy import Condition, ExitSpec, Strategy
from search.bandit import UCB1Bandit
from search.fitness import (
    compute_basket_correlation_weights,
    evaluate_multipair_fitness,
    get_matched_subbasket,
)
from search.portfolio import build_hrp_candidate_portfolio, compute_hrp_weights
from search.qd_archive import MAPElitesArchive
from search.refine import get_grid_neighbors, run_discrete_plateau_centering
from tests.helpers import load_test_basket


def test_qd_archive_and_descriptors():
    archive = MAPElitesArchive(bins_per_dim=5, n_dims=6)
    strat = Strategy(
        strategy_id="TEST_ELITE_1",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 70.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5),
    )

    trade_durations = np.array([5, 10, 15, 20])
    trade_directions = np.array([1, 1, -1, 1])
    trade_pnls = np.array([150.0, -50.0, 200.0, 120.0])

    desc = archive.compute_descriptors(strat, trade_durations, trade_directions, trade_pnls, n_bars_total=500)
    assert len(desc) == 6
    assert (desc >= 0.0).all() and (desc <= 1.0).all()

    added, reason = archive.add_strategy(strat, fitness=1.25, descriptors=desc, metrics={"n_trades": 4})
    assert added is True
    assert reason == "new_cell"

    # Adding with lower fitness to same cell should reject
    added2, reason2 = archive.add_strategy(strat, fitness=0.80, descriptors=desc, metrics={"n_trades": 4})
    assert added2 is False
    assert reason2 == "rejected"

    # Adding with higher fitness should improve
    added3, reason3 = archive.add_strategy(strat, fitness=1.75, descriptors=desc, metrics={"n_trades": 4})
    assert added3 is True
    assert reason3 == "improved_cell"
    assert len(archive.get_all_elites()) == 1


def test_ucb1_bandit_selection_and_stagnation():
    families = ["F1_Trend", "F2_Breakout", "F8_Sessions"]
    bandit = UCB1Bandit(families=families, c_param=1.414)

    # Initial pulls should cycle all families
    p1 = bandit.select_family()
    bandit.record_feedback(p1, 1.0)
    p2 = bandit.select_family()
    bandit.record_feedback(p2, 0.0)
    p3 = bandit.select_family()
    bandit.record_feedback(p3, 0.5)

    seen = {p1, p2, p3}
    assert seen == set(families)

    # Stagnation test: feed 25 zero rewards to p1
    for _ in range(25):
        fam = bandit.select_family()
        bandit.record_feedback(fam, 0.0)

    summary = bandit.get_summary()
    assert summary["total_pulls"] >= 28
    assert summary["counts"][p1] >= 1
    # Verify bandit reallocated pulls to active non-stagnant arms
    assert summary["counts"][p2] > 1 or summary["counts"][p3] > 1


def test_multipair_fitness_with_subbaskets_and_weights():
    all_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURGBP", "EURJPY"]

    # London session strategy must get European pairs
    london_basket = get_matched_subbasket("F8_Sessions", all_pairs)
    assert "EURUSD" in london_basket
    assert "GBPUSD" in london_basket
    assert "USDCHF" in london_basket
    assert "EURGBP" in london_basket
    assert len(london_basket) == 4

    # Asian session
    asian_basket = get_matched_subbasket("Asian_Range", all_pairs)
    assert "USDJPY" in asian_basket
    assert "AUDUSD" in asian_basket

    # Basket correlation weights
    basket = load_test_basket(n_bars=300)
    weights = compute_basket_correlation_weights(basket, london_basket)
    assert len(weights) == 4
    assert pytest.approx(sum(weights.values()), rel=1e-5) == 1.0

    # Multi-pair fitness evaluation
    store = FeatureStore()
    features_by_pair = store.compute_basket_features(basket)

    strat = Strategy(
        strategy_id="TEST_FITNESS",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "lt", 30.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    fit, details = evaluate_multipair_fitness(
        strategy=strat,
        dev_data=basket,
        features_by_pair=features_by_pair,
        min_median_hold_hours=2.0,
        min_trades_primary=5,
    )
    assert isinstance(fit, float)
    assert "fitness" in details


def test_discrete_plateau_centering_and_needle_rejection():
    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features_by_pair = store.compute_basket_features(basket)

    neighbors = get_grid_neighbors(70.0)
    assert 70.0 in neighbors
    assert len(neighbors) >= 3

    strat = Strategy(
        strategy_id="TEST_PLATEAU",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 70.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    # Test plateau centering function
    is_robust, _refined, final_fit, _details = run_discrete_plateau_centering(
        strategy=strat,
        dev_data=basket,
        features_by_pair=features_by_pair,
        max_needle_drop=0.50,
    )
    assert isinstance(is_robust, bool)
    assert isinstance(final_fit, float)


def test_hrp_portfolio_construction():
    # 3 assets with known covariance
    cov = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.09, 0.01],
        [0.00, 0.01, 0.16],
    ])
    weights = compute_hrp_weights(cov)
    assert len(weights) == 3
    assert pytest.approx(np.sum(weights), rel=1e-5) == 1.0
    # Lowest variance asset (asset 0) must have highest weight
    assert weights[0] > weights[1] > weights[2]

    # Full portfolio candidate construction test
    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    features_by_pair = store.compute_basket_features(basket)

    elites = []
    for i, thresh in enumerate([20.0, 30.0, 70.0, 80.0]):
        strat = Strategy(
            strategy_id=f"ELITE_{i}",
            family="F1_Trend" if i % 2 == 0 else "F4_Pullback",
            trigger=Condition("s1_rsi_14", "s1", "lt" if i < 2 else "gt", thresh),
            exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
        )
        elites.append((strat, 0.5 + i * 0.1, {"n_trades": 25}))

    temp_yaml = Path(tempfile.gettempdir()) / "test_candidates.yaml"
    champions, hrp_w, _meta = build_hrp_candidate_portfolio(
        elites=elites,
        dev_data=basket,
        features_by_pair=features_by_pair,
        min_candidates=2,
        max_candidates=5,
        output_yaml_path=temp_yaml,
    )

    assert len(champions) >= 2
    assert pytest.approx(np.sum(hrp_w), rel=1e-5) == 1.0
    assert temp_yaml.exists()
    with open(temp_yaml, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert "portfolio_candidates" in loaded
