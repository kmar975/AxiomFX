"""
Tests for Etap 2 extensions:
1. Zero lookahead on S11, S12, S7 return entropy, SuperTrend, BB+ADX, Donchian, etc.
2. 4 Exit Archetypes in Numba engine: sl_tp, breakeven_trail, dynamic_mean_reversion, pre_rollover_flat.
3. Statistical inference from trejding: DSR, Benjamini-Hochberg FDR, White Reality Check, Monte Carlo drawdown.
4. Depth k=1..5 grammar compilation and validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.vectorized import run_vectorized_backtest
from grammar.compiler import compile_strategy, invert_condition_for_short
from grammar.strategy import Condition, ExitSpec, Strategy
from tests.helpers import load_test_basket
from validation.lookahead import verify_zero_lookahead_for_pair
from validation.statistics import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    monte_carlo_trade_order,
    white_reality_check,
)


def test_new_alpha_zero_lookahead():
    """Verify zero lookahead bias across S11, S12, and S7 return entropy."""
    basket = load_test_basket(n_bars=350)
    eur_df = basket["EURUSD"]
    cut_idx = len(eur_df) - 40

    matched, diffs = verify_zero_lookahead_for_pair(
        df=eur_df,
        pair="EURUSD",
        cut_idx=cut_idx,
        basket_data=basket,
        tolerance=1e-4,
    )

    new_alpha_cols = [
        c for c in diffs
        if c.startswith("s11_") or c.startswith("s12_") or c == "s7_return_entropy" or "supertrend" in c
    ]
    assert len(new_alpha_cols) > 0, "No new alpha columns found in computed features!"

    failed_cols = {col: diffs[col] for col in new_alpha_cols if diffs[col] > 1e-4}
    assert len(failed_cols) == 0, f"Lookahead in new alpha features: {failed_cols}"


def test_four_exit_archetypes():
    """Verify that all 4 exit archetypes execute correctly in Numba backtest using real Dukascopy data."""
    basket = load_test_basket(n_bars=250)
    df = basket["EURUSD"].copy()
    n = len(df)

    long_signals = np.zeros(n, dtype=bool)
    long_signals[30] = True
    long_signals[100] = True
    short_signals = np.zeros(n, dtype=bool)

    # 1. sl_tp
    res_sl_tp = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=long_signals,
        entry_shorts=short_signals,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        exit_type="sl_tp",
    )
    assert res_sl_tp["n_trades"] >= 1

    # 2. breakeven_trail
    res_be_trail = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=long_signals,
        entry_shorts=short_signals,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        exit_type="breakeven_trail",
        trailing_atr=1.0,
    )
    assert res_be_trail["n_trades"] >= 1

    # 3. dynamic_mean_reversion
    res_mr = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=long_signals,
        entry_shorts=short_signals,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        exit_type="dynamic_mean_reversion",
    )
    assert res_mr["n_trades"] >= 1

    # 4. pre_rollover_flat
    res_flat = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=long_signals,
        entry_shorts=short_signals,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        exit_type="pre_rollover_flat",
    )
    assert res_flat["n_trades"] >= 1


def test_trejding_statistical_inference():
    """Verify statistical inference algorithms: DSR, Benjamini-Hochberg, White RC, MC Drawdown."""
    # 1. Deflated Sharpe Ratio
    rets = np.array([0.01, -0.005, 0.012, 0.008, -0.002, 0.015, -0.004, 0.009, 0.011, -0.003] * 10)
    dsr_few_trials = deflated_sharpe_ratio(rets, tested_count=5).probability
    dsr_many_trials = deflated_sharpe_ratio(rets, tested_count=5000).probability
    assert 0.0 <= dsr_few_trials <= 1.0
    assert 0.0 <= dsr_many_trials <= 1.0
    assert dsr_few_trials >= dsr_many_trials, "DSR should haircut significance when tested_count is large"

    # 2. Benjamini-Hochberg FDR
    raw_p_values = [0.001, 0.01, 0.03, 0.08, 0.25]
    fdr_res = benjamini_hochberg(raw_p_values, alpha=0.05)
    assert len(fdr_res.adjusted_p_values) == len(raw_p_values)
    assert bool(fdr_res.rejected.iloc[0]) is True  # 0.001 should be significant

    # 3. White Reality Check
    bench_rets = np.zeros_like(rets)
    w_res = white_reality_check(rets, benchmark_returns=bench_rets, bootstrap_repetitions=200)
    assert 0.0 <= w_res.p_value <= 1.0

    # 4. Monte Carlo Trade Order
    trade_pnls = np.array([120.0, -80.0, 150.0, -90.0, 200.0, -110.0, 50.0, -40.0, 180.0, -70.0])
    mc_res = monte_carlo_trade_order(trade_pnls, repetitions=500)
    assert mc_res.percentile_95_drawdown >= 0.0


def test_k1_to_k5_grammar_compiler():
    """Verify depth k=1..5 grammar compilation and rule validation."""
    strat = Strategy(
        strategy_id="strat_test_k5",
        family="F1_TrendMomentum",
        trigger=Condition(feature="s1_rsi_14_pct", source="s1", operator="lt", threshold=25.0),
        setup=Condition(feature="s2_bull_engulfing", source="s2", operator="gt", threshold=50.0),
        setup_expiry=4,
        setup_2=Condition(feature="s5_atr_pct", source="s5", operator="lt", threshold=30.0),
        setup_2_expiry=6,
        context=Condition(feature="s6_session_london", source="s6", operator="gt", threshold=50.0),
        context_2=Condition(feature="s7_return_entropy", source="s7", operator="lt", threshold=80.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, exit_type="breakeven_trail", trailing_atr=1.2),
    )

    # Rule validation should pass because all 5 conditions come from distinct sources:
    # s1, s2, s5, s6, s7
    strat.validate_rules()

    # Compile signals
    n = 50
    feats_df = pd.DataFrame({
        "s1_rsi_14_pct": np.full(n, 20.0),
        "s2_bull_engulfing": np.full(n, 80.0),
        "s5_atr_pct": np.full(n, 20.0),
        "s6_session_london": np.full(n, 100.0),
        "s7_return_entropy": np.full(n, 40.0),
    })

    long_sig, short_sig = compile_strategy(strat, feats_df)
    assert long_sig.any(), "Signals should trigger when all 5 conditions match"
    assert not short_sig.any()


def test_condition_inversion_rules():
    """
    Verify symmetry inversion rules for Short positions:
    1. s2_bull_fvg (gt 0.5) mirrors to s2_bear_fvg with same operator and threshold 0.5 (not lt 99.5).
    2. s5_squeeze_on is non-invertible binary regime flag (is_regime_ceiling=True), remains unchanged.
    3. s3_sweep_20 (event +1.0) flips event sign to -1.0.
    """
    # 1. Mirrored directional pair
    c_bull_fvg = Condition(feature="s2_bull_fvg", source="s2", operator="gt", threshold=0.5)
    c_short_fvg = invert_condition_for_short(c_bull_fvg)
    assert c_short_fvg.feature == "s2_bear_fvg"
    assert c_short_fvg.operator == "gt"
    assert c_short_fvg.threshold == 0.5

    # 2. Non-invertible binary flag
    c_squeeze = Condition(feature="s5_squeeze_on", source="s5", operator="gt", threshold=0.5)
    assert c_squeeze.is_regime_ceiling is True
    c_short_squeeze = invert_condition_for_short(c_squeeze)
    assert c_short_squeeze.feature == "s5_squeeze_on"
    assert c_short_squeeze.operator == "gt"
    assert c_short_squeeze.threshold == 0.5

    # 3. Signed ternary feature
    c_sweep = Condition(feature="s3_sweep_20", source="s3", operator="event", threshold=1.0)
    c_short_sweep = invert_condition_for_short(c_sweep)
    assert c_short_sweep.feature == "s3_sweep_20"
    assert c_short_sweep.operator == "event"
    assert c_short_sweep.threshold == -1.0
