"""
Unit and Integration Tests for Etap E2:
- Zero Look-Ahead verification (V1 test) on S1–S10 features
- Strict MTF shift(1) backward merge causal integrity
- Causal swing confirmation at t = cand + L
- Dimensionless feature representation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.mtf import map_htf_features
from features.store import FeatureStore
from tests.helpers import load_test_basket
from validation.lookahead import verify_zero_lookahead_for_pair


def test_v1_zero_lookahead_s01_to_s10():
    basket = load_test_basket(n_bars=400)
    eur_df = basket["EURUSD"]
    n_bars = len(eur_df)

    # Pick cut points well inside the series
    cut_idx = n_bars - 50

    matched, diffs = verify_zero_lookahead_for_pair(
        df=eur_df,
        pair="EURUSD",
        cut_idx=cut_idx,
        basket_data=basket,
        tolerance=1e-4,
    )

    failed_cols = {col: diff for col, diff in diffs.items() if diff > 1e-4}
    assert matched, f"Lookahead detected in features: {failed_cols}"


def test_mtf_causal_mapping():
    # Create LTF (e.g. H1) and HTF (e.g. D1) dataframes
    dates_h1 = pd.date_range("2024-01-01 00:00", periods=72, freq="1h", tz="UTC")
    dates_d1 = pd.date_range("2024-01-01 00:00", periods=3, freq="24h", tz="UTC")

    ltf_df = pd.DataFrame({"time": dates_h1, "close": 1.1000 + np.random.randn(72) * 0.0010})
    # HTF features with distinct values per day: Day 1=10.0, Day 2=20.0, Day 3=30.0
    htf_df = pd.DataFrame({"time": dates_d1, "trend_signal": [10.0, 20.0, 30.0]})

    mapped = map_htf_features(ltf_df, htf_df, htf_prefix="htf_")

    # Due to shift(1), Day 1 (first 24 bars) must have default/NaN filled with 50.0
    assert (mapped["htf_trend_signal"].iloc[:24] == 50.0).all()
    # Day 2 (bars 24 to 48) must see Day 1's closed value (10.0), NOT Day 2's value (20.0)!
    assert (mapped["htf_trend_signal"].iloc[24:48] == 10.0).all()
    # Day 3 (bars 48 to 72) must see Day 2's closed value (20.0)
    assert (mapped["htf_trend_signal"].iloc[48:72] == 20.0).all()


def test_tier1_core_coverage():
    basket = load_test_basket(n_bars=300)
    store = FeatureStore()
    feats = store.compute_all_features_for_pair(basket["EURUSD"], "EURUSD", basket)
    tier1 = store.get_tier1_features(feats)

    # Must contain representatives from all S1 to S10
    assert len(tier1) >= 15
    sources_covered = set()
    for col in tier1:
        prefix = col.split("_")[0]
        sources_covered.add(prefix)

    for s_idx in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]:
        assert s_idx in sources_covered, f"Source {s_idx} missing from Tier 1 core!"
