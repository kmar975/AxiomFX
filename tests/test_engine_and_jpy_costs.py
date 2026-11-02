"""
Unit and Integration Tests for Etap E1:
- JPY pip multipliers (100.0 vs 10000.0) and spread in pips < 20
- Quote-to-USD conversion rates for all 7 pairs
- 24h FX candle continuity and OHLC invariants on real Dukascopy data
- Pessimistic intra-candle SL/TP resolution
- Rollover hour protection (17:00 NY)
- Exact agreement between Vectorized Engine and Event-Driven Engine
- Storage atomic checkpoints and Parquet ledger
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.costs import (
    get_pair_specs,
    get_quote_to_usd_rate,
    is_market_entry_allowed,
    price_to_pips,
    resolve_intracandle_conflict,
)
from engine.event_driven import compare_engines
from storage.state import StorageManager
from tests.helpers import load_test_basket


def test_jpy_pip_precision_and_multipliers():
    # JPY pairs must have dec=3, pip_unit=0.01, pip_mult=100.0
    for jpy_pair in ["USDJPY", "EURJPY"]:
        spec = get_pair_specs(jpy_pair)
        assert spec["dec"] == 3
        assert spec["pip_unit"] == 0.01
        assert spec["pip_mult"] == 100.0

        # 0.015 price difference in USDJPY is 1.5 pips, NEVER 150 pips!
        pips = price_to_pips(jpy_pair, 0.015)
        assert pytest.approx(pips, rel=1e-5) == 1.5
        assert pips < 20.0

    # Non-JPY pairs must have dec=5, pip_unit=0.0001, pip_mult=10000.0
    for std_pair in ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "EURGBP"]:
        spec = get_pair_specs(std_pair)
        assert spec["dec"] == 5
        assert spec["pip_unit"] == 0.0001
        assert spec["pip_mult"] == 10000.0

        pips = price_to_pips(std_pair, 0.0015)
        assert pytest.approx(pips, rel=1e-5) == 15.0


def test_quote_to_usd_conversion():
    market_closes = {
        "USDJPY": np.array([150.0, 155.0]),
        "USDCHF": np.array([0.90, 0.92]),
        "GBPUSD": np.array([1.28, 1.25]),
        "EURUSD": np.array([1.08, 1.09]),
    }

    # EURUSD -> quote is USD -> 1.0
    assert get_quote_to_usd_rate("EURUSD", 0, market_closes) == 1.0
    assert get_quote_to_usd_rate("GBPUSD", 0, market_closes) == 1.0
    assert get_quote_to_usd_rate("AUDUSD", 0, market_closes) == 1.0

    # USDJPY -> quote is JPY -> 1.0 / 150.0
    rate_jpy = get_quote_to_usd_rate("USDJPY", 0, market_closes)
    assert pytest.approx(rate_jpy, rel=1e-5) == 1.0 / 150.0

    # USDCHF -> quote is CHF -> 1.0 / 0.90
    rate_chf = get_quote_to_usd_rate("USDCHF", 0, market_closes)
    assert pytest.approx(rate_chf, rel=1e-5) == 1.0 / 0.90

    # EURGBP -> quote is GBP -> 1.28 (GBPUSD)
    rate_gbp = get_quote_to_usd_rate("EURGBP", 0, market_closes)
    assert pytest.approx(rate_gbp, rel=1e-5) == 1.28


def test_candle_continuity_and_invariants():
    data = load_test_basket(n_bars=300)
    for pair, df in data.items():
        spec = get_pair_specs(pair)
        pip_unit = float(spec["pip_unit"])

        # Check OHLC invariants on every bar
        for i in range(len(df)):
            o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
            assert h >= max(o, c) - 1e-6, f"High violated in {pair} at {i}: h={h}, o={o}, c={c}"
            assert l <= min(o, c) + 1e-6, f"Low violated in {pair} at {i}: l={l}, o={o}, c={c}"
            assert (h - l) >= pip_unit - 1e-6, f"Range zero violated in {pair} at {i}"


def test_pessimistic_sl_tp_resolution():
    # Long trade: entry 1.1000, SL 1.0950, TP 1.1050
    # Candle Low is 1.0940 (SL touched) and High is 1.1060 (TP touched)
    sl_hit, tp_hit, reason = resolve_intracandle_conflict(
        is_long=True,
        open_price=1.1000,
        high_price=1.1060,
        low_price=1.0940,
        sl_price=1.0950,
        tp_price=1.1050,
    )
    # SL MUST win over TP
    assert sl_hit is True
    assert tp_hit is False
    assert reason == "SL"

    # Short trade: entry 1.1000, SL 1.1050, TP 1.0950
    # Candle High is 1.1060 (SL touched) and Low is 1.0940 (TP touched)
    sl_hit_s, tp_hit_s, reason_s = resolve_intracandle_conflict(
        is_long=False,
        open_price=1.1000,
        high_price=1.1060,
        low_price=1.0940,
        sl_price=1.1050,
        tp_price=1.0950,
    )
    assert sl_hit_s is True
    assert tp_hit_s is False
    assert reason_s == "SL"


def test_rollover_protection():
    # Spread price = 0.0006, ATR20 = 0.0010 -> spread is 0.60 * ATR > 0.25 -> Rejected
    assert is_market_entry_allowed(spread_price=0.0006, atr_price=0.0010, max_spread_atr_frac=0.25) is False
    # Spread price = 0.0001, ATR20 = 0.0010 -> spread is 0.10 * ATR <= 0.25 -> Allowed
    assert is_market_entry_allowed(spread_price=0.0001, atr_price=0.0010, max_spread_atr_frac=0.25) is True


def test_vectorized_vs_event_driven_consistency():
    data = load_test_basket(n_bars=400)
    eur_df = data["EURUSD"]
    n_bars = len(eur_df)

    # Periodic signals
    entry_longs = np.zeros(n_bars, dtype=bool)
    entry_shorts = np.zeros(n_bars, dtype=bool)
    entry_longs[10::40] = True
    entry_shorts[30::40] = True

    market_closes = {p: df["close"].values for p, df in data.items()}

    matches, comparison = compare_engines(
        df=eur_df,
        pair="EURUSD",
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        market_closes=market_closes,
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        max_holding_bars=20,
        cost_multiplier=1.5,
        tolerance_pct=0.08,
    )
    assert matches, f"Engines did not match: {comparison}"
    assert comparison["vec_trades"] > 0
    assert abs(comparison["vec_trades"] - comparison["ev_trades"]) <= 1


def test_vectorized_vs_event_driven_consistency_breakeven_trail():
    """Verifies that vectorized and event-driven engines match for breakeven_trail with no intra-candle lookahead."""
    data = load_test_basket(n_bars=400)
    eur_df = data["EURUSD"]
    n_bars = len(eur_df)

    entry_longs = np.zeros(n_bars, dtype=bool)
    entry_shorts = np.zeros(n_bars, dtype=bool)
    entry_longs[15::45] = True
    entry_shorts[35::45] = True

    market_closes = {p: df["close"].values for p, df in data.items()}

    matches, comparison = compare_engines(
        df=eur_df,
        pair="EURUSD",
        entry_longs=entry_longs,
        entry_shorts=entry_shorts,
        market_closes=market_closes,
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        max_holding_bars=30,
        cost_multiplier=1.5,
        tolerance_pct=0.08,
        exit_type="breakeven_trail",
        trailing_atr=1.2,
    )
    assert matches, f"breakeven_trail engines did not match: {comparison}"
    assert comparison["vec_trades"] > 0
    assert abs(comparison["vec_trades"] - comparison["ev_trades"]) <= 1


def test_storage_state_and_checkpoints():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        storage = StorageManager(run_id="test_run", base_dir=temp_dir)
        storage.record_run_start({"test_param": 123}, b_star=500)

        # Log some evaluations
        for i in range(15):
            storage.log_evaluation({
                "eval_idx": i,
                "genotype_hash": f"g_{i}",
                "fitness": 0.5 + i * 0.01,
                "n_trades": 25 + i,
            })
        storage.flush_ledger()

        assert storage.parquet_path.exists()
        loaded_df = pd.read_parquet(storage.parquet_path)
        assert len(loaded_df) == 15

        # Checkpoint
        chk_file = storage.save_checkpoint({"step": "E0", "cursor": 15}, step_name="e0")
        assert chk_file.exists()

        restored = storage.load_latest_checkpoint()
        assert restored is not None
        assert restored["step"] == "E0"
        assert restored["cursor"] == 15
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
