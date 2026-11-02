"""
Unified Feature Store and Orchestrator.
Computes and aggregates all 10 orthogonal information sources (S1–S10) for any pair or basket.
Includes automated zero-lookahead Multi-Timeframe (MTF) higher timeframe feature mapping.
Manages Tier 1 core descriptors vs Tier 2 extended patterns vs Tier 3 extensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.aggregate import resample_ohlcv
from features.mtf import map_htf_features
from features.s01_price_trend import compute_s01_features
from features.s02_candle_structure import compute_s02_features
from features.s03_chart_geometry import compute_s03_features
from features.s04_levels import compute_s04_features
from features.s05_volatility import compute_s05_features
from features.s06_time_sessions import compute_s06_features
from features.s07_series_stats import compute_s07_features
from features.s08_microstructure import compute_s08_features
from features.s09_cross_market import compute_s09_features
from features.s10_analogs_regimes import compute_s10_features
from features.s11_institutional_liquidity import compute_s11_features
from features.s12_vwap_statarb import compute_s12_features
from features.tier3_extensions import compute_tier3_features


class FeatureStore:
    def __init__(self, enable_tier3: bool = False, enable_htf: bool = True):
        self.enable_tier3 = enable_tier3
        self.enable_htf = enable_htf
        self._cache: dict[str, pd.DataFrame] = {}

    def compute_all_features_for_pair(
        self,
        df: pd.DataFrame,
        pair: str,
        basket_data: dict[str, pd.DataFrame] | None = None,
        htf_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Computes all orthogonal features across S1–S10 for a single pair,
        and automatically attaches lookahead-free HTF context features (prefix `htf_`).
        """
        if basket_data is None:
            basket_data = {pair: df}

        s1 = compute_s01_features(df, pair)
        s2 = compute_s02_features(df, pair)
        s3 = compute_s03_features(df, pair)
        s4 = compute_s04_features(df, pair)
        s5 = compute_s05_features(df, pair)
        s6 = compute_s06_features(df, pair)
        s7 = compute_s07_features(df, pair)
        s8 = compute_s08_features(df, pair)
        s9 = compute_s09_features(df, pair, basket_data)
        s10 = compute_s10_features(df, pair)
        s11 = compute_s11_features(df, pair)
        s12 = compute_s12_features(df, pair, basket_data)

        feature_blocks = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]

        if self.enable_tier3:
            s_tier3 = compute_tier3_features(df, pair)
            feature_blocks.append(s_tier3)

        all_feats = pd.concat(feature_blocks, axis=1)

        # Automated Zero-Lookahead Higher Timeframe (HTF) context feature attachment
        if self.enable_htf and "time" in df.columns and len(df) > 30:
            try:
                times = pd.to_datetime(df["time"], utc=True)
                median_delta_sec = float(times.diff().dt.total_seconds().median())

                # Dynamic Zero-Lookahead Higher Timeframe (HTF) matching:
                # M1, M5 -> M15; M15 -> H1; H1 -> H4; H4 -> D1; D1 -> W1; W1/MN -> None
                if median_delta_sec <= 600.0:
                    target_htf = "M15"
                elif median_delta_sec <= 1800.0:
                    target_htf = "H1"
                elif median_delta_sec <= 7200.0:
                    target_htf = "H4"
                elif median_delta_sec <= 28800.0:
                    target_htf = "D1"
                elif median_delta_sec <= 100000.0:
                    target_htf = "W1"
                else:
                    target_htf = None

                if target_htf is not None or htf_df is not None:
                    if htf_df is None and target_htf is not None:
                        htf_bars = resample_ohlcv(df, target_htf, pair)
                    else:
                        htf_bars = htf_df.copy()

                    if len(htf_bars) > 10:
                        htf_s1 = compute_s01_features(htf_bars, pair)
                        htf_s3 = compute_s03_features(htf_bars, pair)
                        htf_s5 = compute_s05_features(htf_bars, pair)
                        htf_combined = pd.concat([htf_s1, htf_s3, htf_s5], axis=1)
                        htf_combined["time"] = htf_bars["time"].values

                        mapped_htf = map_htf_features(
                            ltf_df=df,
                            htf_features_df=htf_combined,
                            htf_prefix="htf_",
                        )
                        all_feats = pd.concat([all_feats, mapped_htf], axis=1)
            except Exception as ex:
                print(f"[FeatureStore] Notice: automated HTF feature attachment skipped ({ex})")

        return all_feats

    def compute_basket_features(
        self,
        basket_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Computes features for all pairs in the basket.
        """
        features_dict = {}
        for pair, df in basket_data.items():
            feats = self.compute_all_features_for_pair(df, pair, basket_data)
            features_dict[pair] = feats
        return features_dict

    @staticmethod
    def get_tier1_features(features_df: pd.DataFrame) -> list[str]:
        """
        Returns all feature column names from features_df (excluding 'time' and constant features).
        Unlocks the full pool of ~120+ orthogonal features across S1–S12 and HTF context.
        """
        exclude = {"time"}
        if "s8_spread_dev_pct" in features_df.columns:
            vals = features_df["s8_spread_dev_pct"].values
            if np.nanstd(vals) < 1e-4 or np.all(vals == 50.0):
                exclude.add("s8_spread_dev_pct")

        tier1_cols = [c for c in features_df.columns if c not in exclude]
        return tier1_cols
