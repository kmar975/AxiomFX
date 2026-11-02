"""
Targeted Verification Tests for Critical & Architectural Audit Fixes:
1. Same-bar intra-candle time-travel prevention in backtest engines.
2. Mark-to-market equity curve and exact execution exit prices.
3. V2 Null World permutation preservation (monotonic timestamps).
4. Continuous feature genetic mutation (not stuck at 0.5 threshold).
5. Regime ceiling feature detection for ATR ratios.
6. S02 duplicate columns removal.
7. QD Archive rejection of non-positive fitness.
8. Portfolio construction exclusion of losing strategies.
9. Direction mode 'fade' signal inversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.vectorized import run_vectorized_backtest
from features.s02_candle_structure import compute_s02_features
from grammar.compiler import compile_strategy
from grammar.strategy import Condition, ExitSpec, Strategy
from search.genetic import mutate_and_crossover_strategy
from search.qd_archive import MAPElitesArchive
from tests.helpers import load_test_basket
from validation.null_world import build_synchronous_null_world


def test_intra_candle_time_travel_prevention():
    """Verifies that an exit on bar i strictly prevents a new entry on opens[i] of the same bar."""
    n_bars = 20
    opens = np.full(n_bars, 1.1000)
    highs = np.full(n_bars, 1.1050)
    lows = np.full(n_bars, 1.0950)
    closes = np.full(n_bars, 1.1000)

    # Force immediate stop out on bar 2 by dropping low
    lows[2] = 1.0800

    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread": np.full(n_bars, 1.0),
    })

    # Continuous entry signals on bars 0..5
    entry_longs = np.zeros(n_bars, dtype=bool)
    entry_longs[0:6] = True
    entry_shorts = np.zeros(n_bars, dtype=bool)

    q_vec = np.ones(n_bars, dtype=np.float64)

    res = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        quote_to_usd=q_vec,
        sl_atr_mult=1.0,
        tp_atr_mult=3.0,
        max_holding_bars=10,
    )

    entry_bars = res["entry_bars"]
    exit_bars = res["exit_bars"]

    # If first trade entered at bar 1 and stopped out at bar 2,
    # the second trade MUST NOT enter on bar 2 (which would be same-bar time travel)
    assert len(entry_bars) >= 1
    assert entry_bars[0] == 1
    assert exit_bars[0] == 2
    if len(entry_bars) > 1:
        assert entry_bars[1] >= 3, f"Time travel detected: second trade entered at bar {entry_bars[1]} immediately on exit bar 2!"


def test_mark_to_market_equity_and_exit_prices():
    """Verifies equity curve fluctuates mark-to-market while in position and returns exact exit prices."""
    n_bars = 10
    opens = np.array([1.10, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.15, 1.15, 1.15], dtype=float)
    highs = opens + 0.005
    lows = opens - 0.005
    closes = opens + 0.002

    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "spread": np.full(n_bars, 1.0),
    })

    entry_longs = np.zeros(n_bars, dtype=bool)
    entry_longs[0] = True  # Entry on bar 1
    entry_shorts = np.zeros(n_bars, dtype=bool)

    res = run_vectorized_backtest(
        df=df,
        pair="EURUSD",
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        quote_to_usd=np.ones(n_bars),
        sl_atr_mult=2.0,
        tp_atr_mult=5.0,
        max_holding_bars=4,
    )

    eq = res["equity_curve"]
    # Bars 1, 2, 3 should have different unrealized equities as price moved
    assert eq[1] != eq[2] or eq[2] != eq[3], "Equity curve was flat (not mark-to-market)!"
    assert "exit_prices" in res
    assert len(res["exit_prices"]) == len(res["entry_bars"])


def test_null_world_monotonic_time_and_permutation():
    """Verifies that null world timestamps remain strictly increasing and do not reverse the day permutation."""
    basket = load_test_basket(n_bars=400)
    null_basket = build_synchronous_null_world(basket, seed=42)

    for pair, ndf in null_basket.items():
        times = pd.to_datetime(ndf["time"])
        # Timestamps must be strictly monotonic
        assert times.is_monotonic_increasing, f"Null world {pair} timestamps are not monotonic!"
        # Price must not be identical to original basket
        assert not np.array_equal(ndf["close"].values, basket[pair]["close"].values[:len(ndf)])


def test_genetic_mutation_continuous_feature():
    """Verifies that mutating to a continuous feature assigns valid percentile threshold instead of 0.5."""
    rng = np.random.default_rng(123)
    strat = Strategy(
        strategy_id="TEST_BIN",
        family="F2_Breakout",
        trigger=Condition("s2_seed_pin_bull", "s2", "event", 1.0),
        context=Condition("s2_inside_bar", "s2", "gt", 0.5),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    tier1 = ["s2_inside_bar", "s2_body_atr_pct", "s2_range_atr_pct"]
    mutated = mutate_and_crossover_strategy(strat, None, tier1_features=tier1, rng=rng, mutation_type="condition")

    for cond in [mutated.trigger, mutated.context, getattr(mutated, "context_2", None)]:
        if cond is not None and cond.feature in ("s2_body_atr_pct", "s2_range_atr_pct"):
            assert cond.threshold != 0.5, f"Continuous feature {cond.feature} left with threshold 0.5!"
            assert cond.operator != "event", f"Continuous feature {cond.feature} left with operator 'event'!"


def test_regime_ceiling_atr_ratio_features():
    """Verifies that s5_atr_ratio_*_pct features are auto-detected as is_regime_ceiling."""
    for feat in ["s5_atr_ratio_5_20_pct", "s5_atr_ratio_10_40_pct", "htf_s5_atr_ratio_10_40_pct"]:
        c = Condition(feature=feat, source="s5", operator="gt", threshold=70.0)
        assert c.is_regime_ceiling is True, f"{feat} was not detected as regime ceiling!"


def test_s02_no_unprefixed_duplicate_columns():
    """Verifies features/s02_candle_structure produces only s2_ prefixed columns."""
    n = 50
    df = pd.DataFrame({
        "open": np.linspace(1.10, 1.12, n),
        "high": np.linspace(1.11, 1.13, n),
        "low": np.linspace(1.09, 1.11, n),
        "close": np.linspace(1.10, 1.12, n),
        "spread": np.full(n, 1.0),
        "tick_volume": np.full(n, 100),
    })
    feats = compute_s02_features(df, pair="EURUSD")
    for col in feats.columns:
        assert col.startswith("s2_"), f"Column {col} missing s2_ prefix in s02 features!"


def test_qd_archive_rejects_negative_fitness():
    """Verifies MAPElitesArchive strictly rejects strategies with fitness <= 0.0."""
    archive = MAPElitesArchive()
    strat = Strategy(
        strategy_id="LOSING_STRAT",
        family="F1_Trend",
        trigger=Condition("s2_seed_pin_bull", "s2", "event", 1.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )
    desc = np.full(6, 0.5)

    added, reason = archive.add_strategy(strat, fitness=-0.25, descriptors=desc, metrics={})
    assert added is False
    assert reason == "rejected"
    assert len(archive.get_all_elites()) == 0


def test_direction_mode_fade_inversion():
    """Verifies that direction_mode='fade' correctly inverts Long and Short entry masks."""
    strat_cont = Strategy(
        strategy_id="TEST_CONT",
        family="F2_Breakout",
        trigger=Condition("s2_seed_pin_bull", "s2", "event", 1.0, direction_mode="continuation"),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )
    strat_fade = Strategy(
        strategy_id="TEST_FADE",
        family="F2_Breakout",
        trigger=Condition("s2_seed_pin_bull", "s2", "event", 1.0, direction_mode="fade"),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )

    feats = pd.DataFrame({
        "s2_seed_pin_bull": np.array([0, 1, 0, 0, 0] * 4, dtype=float),
        "s2_seed_pin_bear": np.array([0, 0, 0, 1, 0] * 4, dtype=float),
    })

    l_cont, s_cont = compile_strategy(strat_cont, feats)
    l_fade, s_fade = compile_strategy(strat_fade, feats)

    # In fade mode, Long entries match Short continuation and vice versa
    assert np.array_equal(l_fade, s_cont)
    assert np.array_equal(s_fade, l_cont)
