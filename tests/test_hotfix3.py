"""
Unit and Integration Tests for HOTFIX #3:
- Exact trade entry/exit bars returned by Numba backtest kernel
- Plateau centering preserves binary/event conditions and perturbs exit.sl_atr
- Trigger threshold is never set to 50.0
- Secondary Meta-Labeling degrades to None when not improving PF by >= 5%
- Recency OOS-1 penalty (-0.75) and bonus (+0.30) in multi-pair fitness
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.vectorized import run_vectorized_backtest
from grammar.strategy import Condition, ExitSpec, Strategy
from search.fitness import evaluate_multipair_fitness
from search.meta_labeling import MetaModelFilter, train_meta_labeling_for_strategy
from search.refine import GRID_PERCENTILES, run_discrete_plateau_centering
from tests.helpers import load_test_basket


def test_numba_returns_entry_exit_bars():
    basket = load_test_basket(n_bars=200)
    df = basket["EURUSD"]
    n_bars = len(df)

    # Alternate buy signals
    l_mask = np.zeros(n_bars, dtype=bool)
    l_mask[10] = True
    l_mask[50] = True
    l_mask[100] = True
    s_mask = np.zeros(n_bars, dtype=bool)

    res = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=l_mask,
        entry_shorts=s_mask,
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        max_holding_bars=10,
    )

    assert "entry_bars" in res
    assert "exit_bars" in res
    entry_bars = res["entry_bars"]
    exit_bars = res["exit_bars"]

    assert len(entry_bars) == res["n_trades"]
    assert len(exit_bars) == res["n_trades"]
    if len(entry_bars) > 0:
        # Trade entry happens at bar i after signal at i-1
        assert entry_bars[0] >= 11
        assert (exit_bars >= entry_bars).all()


def test_plateau_centering_binary_and_event():
    # Verify 50.0 is completely removed from GRID_PERCENTILES
    assert 50.0 not in GRID_PERCENTILES

    basket = load_test_basket(n_bars=300)
    from features.store import FeatureStore
    fs = FeatureStore()
    features_by_pair = fs.compute_basket_features(basket)

    # Strategy with purely binary/event conditions
    strat = Strategy(
        strategy_id="TEST_BINARY_SMC",
        family="F5_Candles",
        trigger=Condition(feature="s2_seed_pin_bull", source="s2", operator="gt", threshold=0.5),
        setup=Condition(feature="s5_squeeze_on", source="s5", operator="event", threshold=1.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    is_robust, refined, fit, details = run_discrete_plateau_centering(
        strategy=strat,
        dev_data=basket,
        features_by_pair=features_by_pair,
        max_needle_drop=0.50,
    )

    # Verify binary condition was NOT overwritten with a percentile
    assert refined.trigger.threshold == 0.5
    assert refined.setup.threshold == 1.0
    # Verify plateau centered on exit.sl_atr
    assert details["plateau_centered_slot"] == "exit.sl_atr"
    assert refined.exit.sl_atr in [round(1.5 * 0.85, 2), 1.5, round(1.5 * 1.15, 2)]


def test_plateau_centering_continuous_no_50():
    basket = load_test_basket(n_bars=300)
    from features.store import FeatureStore
    fs = FeatureStore()
    features_by_pair = fs.compute_basket_features(basket)

    # Continuous trigger with threshold 50.0 originally
    strat = Strategy(
        strategy_id="TEST_CONTINUOUS",
        family="F1_Trend",
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=50.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    is_robust, refined, fit, details = run_discrete_plateau_centering(
        strategy=strat,
        dev_data=basket,
        features_by_pair=features_by_pair,
        max_needle_drop=0.50,
    )

    # Centered notch must be one of GRID_PERCENTILES and NEVER 50.0
    assert refined.trigger.threshold != 50.0
    assert refined.trigger.threshold in GRID_PERCENTILES


def test_meta_labeling_degradation_protection():
    # If model does not improve PF by >= 5%, it must return model=None
    strat = Strategy(
        strategy_id="TEST_META_DEGRADATION",
        family="F1_Trend",
        trigger=Condition(feature="s1_rsi_14", source="s1", operator="gt", threshold=30.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=12),
    )

    basket = load_test_basket(n_bars=400)
    df = basket["EURUSD"]
    # Mock features
    feats_df = pd.DataFrame({
        "time": df["time"],
        "s1_rsi_14": np.random.uniform(20.0, 80.0, len(df)),
        "s5_atr20_pct": np.random.uniform(20.0, 80.0, len(df)),
    })

    mf = train_meta_labeling_for_strategy(
        strategy=strat,
        df=df,
        features_df=feats_df,
        pair="EURUSD",
    )

    assert isinstance(mf, MetaModelFilter)
    if mf.model is not None:
        assert mf.train_pf_meta >= mf.train_pf_base * 1.05


def test_recency_oos1_adjustment_applied():
    from features.store import FeatureStore
    basket = load_test_basket(n_bars=400)
    store = FeatureStore()
    features_by_pair = store.compute_basket_features(basket)

    strat = Strategy(
        strategy_id="TEST_RECENCY",
        family="F1_Trend",
        trigger=Condition(feature="s1_rsi_14_pct", source="s1", operator="lt", threshold=50.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    fit, details = evaluate_multipair_fitness(
        strategy=strat,
        dev_data=basket,
        features_by_pair=features_by_pair,
        min_trades_primary=5,
    )

    assert "recency_adj" in details
    assert "recent_block_pnl" in details
    assert "recent_block_pf" in details
    assert details["recency_adj"] in [-0.75, -0.20, 0.0, 0.30]
