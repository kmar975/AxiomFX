"""
V4 Holdout Evaluation Engine.
Evaluates frozen finalists and HRP portfolio strictly once on the unseen 20% HOLDOUT dataset.
Supports secondary LightGBM meta-labeling filter for finalists.
Requires SplitManager.unlock_holdout_for_finalists().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.split import SplitManager
from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from features.store import FeatureStore
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy


def evaluate_finalists_on_holdout(
    candidates: list[Strategy],
    hrp_weights: np.ndarray,
    all_data: dict[str, pd.DataFrame],
    cutoff_date: pd.Timestamp,
    meta_filters: dict[str, Any] | None = None,
    output_image_path: Path | None = None,
) -> dict[str, Any]:
    """
    Evaluates candidates and HRP portfolio on HOLDOUT split under strict unlock context.
    Optionally applies trained LightGBM meta-labeling filter.
    """
    with SplitManager.unlock_holdout_for_finalists():
        # Prepend causal warmup buffer (up to 1000 bars from DEV tail) before Holdout
        # to eliminate the cold-start warmup distortion on rolling indicators, HTF, and percentiles
        warmup_data: dict[str, pd.DataFrame] = {}
        warmup_offsets: dict[str, int] = {}
        for pair, df in all_data.items():
            df_time = pd.to_datetime(df["time"], utc=True)
            dev_slice = df[df_time < cutoff_date]
            holdout_slice = df[df_time >= cutoff_date]
            warmup_len = min(1000, len(dev_slice))
            warmup_tail = dev_slice.iloc[-warmup_len:] if warmup_len > 0 else dev_slice
            combined_df = pd.concat([warmup_tail, holdout_slice], ignore_index=True)
            warmup_data[pair] = combined_df
            warmup_offsets[pair] = len(warmup_tail)

    has_tier3 = any(
        (c is not None and getattr(c, "feature", "").startswith("tier3_"))
        for s in candidates
        for c in [s.trigger, s.context, getattr(s, "context_2", None), s.setup, getattr(s, "setup_2", None)]
    )
    store = FeatureStore(enable_tier3=has_tier3)
    warmed_features = store.compute_basket_features(warmup_data)

    # Slice strictly to holdout rows (zero look-ahead, full indicator warmup)
    holdout_data: dict[str, pd.DataFrame] = {}
    holdout_features: dict[str, pd.DataFrame] = {}
    for pair in all_data:
        offset = warmup_offsets[pair]
        holdout_data[pair] = warmup_data[pair].iloc[offset:].reset_index(drop=True)
        holdout_features[pair] = warmed_features[pair].iloc[offset:].reset_index(drop=True)

    eur_df = holdout_data["EURUSD"]
    n_bars = len(eur_df)
    q_vec = compute_quote_to_usd_vector("EURUSD", n_bars, {p: d["close"].values for p, d in holdout_data.items()})

    candidate_results = []
    equity_curves = []

    for idx, strat in enumerate(candidates):
        l_mask, s_mask = compile_strategy(strat, holdout_features["EURUSD"])

        # Apply secondary meta-labeling filter if available
        meta_filter = meta_filters.get(strat.strategy_id) if meta_filters else None
        if meta_filter is not None and meta_filter.model is not None:
            l_mask = meta_filter.filter_signals(l_mask, holdout_features["EURUSD"], is_dev=False)
            s_mask = meta_filter.filter_signals(s_mask, holdout_features["EURUSD"], is_dev=False)

        res = run_vectorized_backtest(
            df=eur_df,
            pair="EURUSD",
            entry_longs=l_mask,
            entry_shorts=s_mask,
            quote_to_usd=q_vec,
            sl_atr_mult=strat.exit.sl_atr,
            tp_atr_mult=strat.exit.tp_atr,
            max_holding_bars=strat.exit.max_holding_bars,
            cost_multiplier=1.0,  # Base 1.0x costs on holdout
            exit_type=getattr(strat.exit, "exit_type", "sl_tp"),
            trailing_atr=getattr(strat.exit, "trailing_atr", 1.2),
        )
        eq = res.get("equity_curve", np.full(n_bars, 100000.0))
        equity_curves.append(eq)

        is_pos = res["net_pnl_usd"] > 0
        candidate_results.append({
            "strategy_id": strat.strategy_id,
            "n_trades": res["n_trades"],
            "win_rate": res["win_rate"],
            "profit_factor": res["profit_factor"],
            "sharpe_holdout": res["sharpe"],
            "total_return_pct": res["total_return_pct"],
            "max_dd_pct": res["max_dd_pct"],
            "is_positive": is_pos,
            "meta_labeled": meta_filter is not None and meta_filter.model is not None,
            "meta_threshold": meta_filter.threshold if meta_filter else 0.50,
        })

    # Combined HRP Portfolio Equity
    if equity_curves:
        norm_curves = np.array([eq / eq[0] for eq in equity_curves])
        w = hrp_weights if len(hrp_weights) == len(candidates) else np.full(len(candidates), 1.0 / len(candidates))
        portfolio_eq = np.dot(w, norm_curves) * 100000.0

        port_ret = np.diff(portfolio_eq) / portfolio_eq[:-1]
        n_p_bars = len(port_ret)
        bpd = 24  # H1 discovery default
        if n_p_bars >= bpd:
            indices = np.arange(0, n_p_bars, bpd)
            daily_port_ret = np.add.reduceat(port_ret, indices)
            mean_ret = float(np.mean(daily_port_ret))
            std_ret = float(np.std(daily_port_ret))
            port_sharpe = float((mean_ret / (std_ret + 1e-9)) * np.sqrt(252))
        else:
            mean_ret = float(np.mean(port_ret))
            std_ret = float(np.std(port_ret))
            port_sharpe = float((mean_ret / (std_ret + 1e-9)) * np.sqrt(252 * 24))
        tot_ret = float((portfolio_eq[-1] - portfolio_eq[0]) / portfolio_eq[0]) * 100.0
    else:
        portfolio_eq = np.full(n_bars, 100000.0)
        port_sharpe = 0.0
        tot_ret = 0.0

    # Plot portfolio equity
    if output_image_path and len(portfolio_eq) > 1:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        _fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.plot(eur_df["time"], portfolio_eq, color="#00e676", linewidth=2.0, label="HRP Portfolio (Holdout)")
        ax.axhline(100000.0, color="#787b86", linestyle="--", alpha=0.7)
        ax.set_title("V4 Holdout Out-of-Sample Performance", fontsize=12, fontweight="bold")
        ax.set_ylabel("Account Equity (USD)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()

    return {
        "candidate_results": candidate_results,
        "portfolio_equity": portfolio_eq,
        "portfolio_sharpe_holdout": round(port_sharpe, 2),
        "portfolio_total_return_pct": round(tot_ret, 2),
        "times": eur_df["time"].values,
    }
