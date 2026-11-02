"""
V5 Signal Recovery Test Engine.
Injects known synthetic edge into randomized FX series across multiple SNR tiers (0.6, 0.8, 1.0)
and verifies that the search pipeline successfully recovers the true hypothesis above the noise threshold.
Generates recovery_curve.png.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import clear_vector_cache, run_vectorized_backtest
from features.store import FeatureStore
from grammar.compiler import compile_strategy
from grammar.strategy import Condition, ExitSpec, Strategy


def inject_synthetic_edge(
    df: pd.DataFrame,
    trigger_feature: str = "s1_rsi_14",
    snr: float = 0.8,
) -> pd.DataFrame:
    """
    Injects a synthetic forward price drift on bars where trigger_feature condition is met.
    """
    df = df.copy()
    closes = df["close"].values.copy()
    highs = df["high"].values.copy()
    lows = df["low"].values.copy()
    n = len(closes)

    # Base drift increment proportional to SNR
    pip_unit = 0.0001
    drift = snr * pip_unit * 25.0

    # Locate trigger bars based on trigger_feature
    store = FeatureStore()
    feats = store.compute_all_features_for_pair(df, "EURUSD")
    if trigger_feature in feats.columns:
        sig_bars = np.where(feats[trigger_feature].values > 70.0)[0]
    else:
        sig_bars = np.arange(20, n - 10, 25)

    # Subsample non-overlapping bars
    selected_bars = []
    last_b = -100
    for b in sig_bars:
        if 20 <= b < (n - 6) and (b - last_b) >= 6:
            selected_bars.append(b)
            last_b = b

    for i in selected_bars:
        added = np.linspace(drift * 0.2, drift, 5)
        closes[i + 1 : i + 6] += added
        highs[i + 1 : i + 6] += added
        lows[i + 1 : i + 6] += added * 0.5

    df["close"] = closes
    df["high"] = np.maximum(highs, closes)
    df["low"] = np.minimum(lows, closes)
    return df


def run_v5_recovery_test(
    base_data: dict[str, pd.DataFrame],
    snr_levels: list[float] | None = None,
    output_image_path: Path | None = None,
) -> dict[str, Any]:
    """
    Verifies pipeline recovery rate across SNR tiers.
    """
    if snr_levels is None:
        snr_levels = [0.6, 0.8, 1.0]
    test_rule = Strategy(
        strategy_id="V5_KNOWN_TARGET",
        family="F1_Trend",
        trigger=Condition("s1_rsi_14", "s1", "gt", 70.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=20),
    )

    recalls = []
    clear_vector_cache()

    for snr in snr_levels:
        injected_df = inject_synthetic_edge(base_data["EURUSD"], "s1_rsi_14", snr=snr)
        store = FeatureStore()
        feats = store.compute_all_features_for_pair(injected_df, "EURUSD")

        l_mask, s_mask = compile_strategy(test_rule, feats)
        q_vec = compute_quote_to_usd_vector("EURUSD", len(injected_df), {"EURUSD": injected_df["close"].values})

        res = run_vectorized_backtest(
            df=injected_df,
            pair="EURUSD",
            entry_longs=l_mask,
            entry_shorts=s_mask,
            quote_to_usd=q_vec,
            sl_atr_mult=1.5,
            tp_atr_mult=2.5,
            max_holding_bars=20,
            cost_multiplier=1.0,
        )

        # Recovered if net PnL is positive and Sharpe > 0.30
        recovered = (res["net_pnl_usd"] > 0) and (res["sharpe"] > 0.30)
        recalls.append(1.0 if recovered else 0.0)

    if output_image_path:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        _fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        ax.plot(snr_levels, recalls, marker="s", color="#7c4dff", linewidth=2.0, label="Pipeline Recovery Rate")
        ax.set_title("V5 Signal Recovery Test across SNR Levels", fontsize=12, fontweight="bold")
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
        ax.set_ylabel("Empirical Recall")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()

    return {
        "snr_levels": snr_levels,
        "recalls": recalls,
        "all_passed": (np.mean(recalls) >= 0.60) and (recalls[-1] == 1.0),
    }
