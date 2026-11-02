"""
Multi-Timeframe (MTF) Feature Mapping with Strict Zero Look-Ahead.
Enforces shift(1) on closed higher timeframe bars before merge_asof(direction='backward').
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_htf_features(
    ltf_df: pd.DataFrame,
    htf_features_df: pd.DataFrame,
    htf_prefix: str = "htf_",
) -> pd.DataFrame:
    """
    Merges higher-timeframe features onto lower-timeframe dataframe.
    CRITICAL RULE: Higher-timeframe features are shifted by 1 full period
    prior to merging, guaranteeing zero look-ahead from incomplete bars.
    """
    ltf_df = ltf_df.copy()
    htf_features_df = htf_features_df.copy()

    if "time" not in ltf_df.columns or "time" not in htf_features_df.columns:
        raise ValueError("Both dataframes must contain 'time' column.")

    ltf_df["time"] = pd.to_datetime(ltf_df["time"], utc=True)
    htf_features_df["time"] = pd.to_datetime(htf_features_df["time"], utc=True)

    ltf_df = ltf_df.sort_values("time").reset_index(drop=True)
    htf_features_df = htf_features_df.sort_values("time").reset_index(drop=True)

    # Separate time and feature columns
    feature_cols = [c for c in htf_features_df.columns if c != "time"]
    
    # Strictly shift features by 1 period before merging!
    shifted_htf = htf_features_df[feature_cols].shift(1)
    shifted_htf["time"] = htf_features_df["time"]
    
    # Rename feature columns with prefix
    rename_map = {c: f"{htf_prefix}{c}" for c in feature_cols}
    shifted_htf.rename(columns=rename_map, inplace=True)

    # Perform causal backward merge_asof
    merged = pd.merge_asof(
        ltf_df[["time"]],
        shifted_htf,
        on="time",
        direction="backward",
    )

    merged_features = merged.drop(columns=["time"]).fillna(50.0)
    return merged_features


def compute_mtf_confluence_signals(
    base_df: pd.DataFrame,
    features_by_tf: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Computes Multi-Timeframe (MTF) Confluence Voting from trejding architecture.
    Combines trend/momentum directional signals across closed timeframes (M15, H1, H4, D1)
    with strict shift(1) backward alignment to guarantee zero lookahead.
    Outputs:
    - mtf_net_vote_ratio: [-100..+100]
    - mtf_unanimous: +1.0 (all long), -1.0 (all short), 0.0
    - mtf_hierarchical: +1.0 (higher TF long bias + lower TF timing), -1.0, 0.0
    """
    if "time" not in base_df.columns:
        raise ValueError("base_df must contain 'time' column.")

    base_times = pd.to_datetime(base_df["time"], utc=True)
    aligned_votes = {}

    for tf_name, tf_df in features_by_tf.items():
        if "time" not in tf_df.columns or len(tf_df) == 0:
            continue
        cur_tf = tf_df.copy()
        cur_tf["time"] = pd.to_datetime(cur_tf["time"], utc=True)
        cur_tf = cur_tf.sort_values("time").reset_index(drop=True)

        # Directional trend signal: +1, -1, 0
        trend_sig = pd.Series(0.0, index=cur_tf.index)
        if "s1_sma_dist_pct" in cur_tf.columns:
            trend_sig = np.where(cur_tf["s1_sma_dist_pct"] > 60.0, 1.0, np.where(cur_tf["s1_sma_dist_pct"] < 40.0, -1.0, 0.0))
        elif "s1_rsi_14" in cur_tf.columns:
            trend_sig = np.where(cur_tf["s1_rsi_14"] > 55.0, 1.0, np.where(cur_tf["s1_rsi_14"] < 45.0, -1.0, 0.0))

        # Strictly shift(1) to avoid lookahead on open bars
        shifted_series = pd.DataFrame({
            "time": cur_tf["time"],
            "vote": pd.Series(trend_sig).shift(1).fillna(0.0),
        })

        merged = pd.merge_asof(
            pd.DataFrame({"time": base_times}),
            shifted_series,
            on="time",
            direction="backward",
        )
        aligned_votes[tf_name] = merged["vote"].fillna(0.0).values

    if not aligned_votes:
        n = len(base_df)
        return pd.DataFrame({
            "mtf_net_vote_ratio": np.zeros(n, dtype=float),
            "mtf_unanimous": np.zeros(n, dtype=float),
            "mtf_hierarchical": np.zeros(n, dtype=float),
        })

    votes_df = pd.DataFrame(aligned_votes)
    long_count = (votes_df == 1.0).sum(axis=1)
    short_count = (votes_df == -1.0).sum(axis=1)
    total_votes = float(votes_df.shape[1])

    # Net vote ratio [-100..+100]
    net_vote_ratio = ((long_count - short_count) / max(total_votes, 1.0)) * 100.0

    # Unanimous signal
    unanimous = np.where(
        long_count == total_votes, 1.0,
        np.where(short_count == total_votes, -1.0, 0.0)
    )

    # Hierarchical: higher TF (H4, D1) bias + lower TF (M15, H1) confirmation
    higher_cols = [c for c in votes_df.columns if c in ["H4", "D1"]]
    lower_cols = [c for c in votes_df.columns if c in ["M15", "H1"]]

    if higher_cols and lower_cols:
        h_bias = votes_df[higher_cols].mean(axis=1)
        l_timing = votes_df[lower_cols].mean(axis=1)
        hierarchical = np.where(
            (h_bias > 0.0) & (l_timing > 0.0), 1.0,
            np.where((h_bias < 0.0) & (l_timing < 0.0), -1.0, 0.0)
        )
    else:
        hierarchical = unanimous

    return pd.DataFrame({
        "mtf_net_vote_ratio": net_vote_ratio.values,
        "mtf_unanimous": unanimous,
        "mtf_hierarchical": hierarchical,
    })
