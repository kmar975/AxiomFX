"""
Institutional Stress Testing Suite (§ 9).
Applies spread doubling (2.0x), 1-bar execution latency, +-20% parameter perturbation,
and block bootstrap confidence intervals to evaluate tail fragility of top finalists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.costs import compute_quote_to_usd_vector
from engine.vectorized import run_vectorized_backtest
from grammar.compiler import compile_strategy
from grammar.strategy import Strategy


def run_candidate_stress_tests(
    strategy: Strategy,
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    pair: str = "EURUSD",
    n_bootstrap: int = 100,
    meta_filter: Any = None,
) -> dict[str, Any]:
    """
    Executes 4 stress dimensions on strategy:
    1. Base performance (1.0x spread)
    2. Double spread (2.0x spread)
    3. 1-bar latency (entry delayed by 1 bar)
    4. Parameter jitter (+-20% SL/TP)
    5. Block bootstrap 95% CI on Sharpe
    """
    n_bars = len(df)
    q_vec = compute_quote_to_usd_vector(pair, n_bars, {pair: df["close"].values})
    l_mask, s_mask = compile_strategy(strategy, features_df)

    if meta_filter is not None and getattr(meta_filter, "model", None) is not None:
        try:
            l_mask = meta_filter.filter_signals(l_mask, features_df, is_dev=True)
            s_mask = meta_filter.filter_signals(s_mask, features_df, is_dev=True)
        except TypeError:
            l_mask = meta_filter.filter_signals(l_mask, features_df)
            s_mask = meta_filter.filter_signals(s_mask, features_df)

    # 1. Base
    base_res = run_vectorized_backtest(
        df=df,
        pair=pair,
        entry_longs=l_mask,
        entry_shorts=s_mask,
        quote_to_usd=q_vec,
        sl_atr_mult=strategy.exit.sl_atr,
        tp_atr_mult=strategy.exit.tp_atr,
        max_holding_bars=strategy.exit.max_holding_bars,
        cost_multiplier=1.0,
        exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
        trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
    )

    # 2. Spread x 2.0
    spread_2x_res = run_vectorized_backtest(
        df=df,
        pair=pair,
        entry_longs=l_mask,
        entry_shorts=s_mask,
        quote_to_usd=q_vec,
        sl_atr_mult=strategy.exit.sl_atr,
        tp_atr_mult=strategy.exit.tp_atr,
        max_holding_bars=strategy.exit.max_holding_bars,
        cost_multiplier=2.0,
        exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
        trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
    )

    # 3. 1-bar Latency Delay
    delayed_l = np.roll(l_mask, 1)
    delayed_s = np.roll(s_mask, 1)
    delayed_l[0] = False
    delayed_s[0] = False
    latency_res = run_vectorized_backtest(
        df=df,
        pair=pair,
        entry_longs=delayed_l,
        entry_shorts=delayed_s,
        quote_to_usd=q_vec,
        sl_atr_mult=strategy.exit.sl_atr,
        tp_atr_mult=strategy.exit.tp_atr,
        max_holding_bars=strategy.exit.max_holding_bars,
        cost_multiplier=1.0,
        exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
        trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
    )

    # 4. Parameter Jitter (+-20%)
    jitter_res = run_vectorized_backtest(
        df=df,
        pair=pair,
        entry_longs=l_mask,
        entry_shorts=s_mask,
        quote_to_usd=q_vec,
        sl_atr_mult=strategy.exit.sl_atr * 1.20,
        tp_atr_mult=strategy.exit.tp_atr * 0.80,
        max_holding_bars=strategy.exit.max_holding_bars,
        cost_multiplier=1.0,
        exit_type=getattr(strategy.exit, "exit_type", "sl_tp"),
        trailing_atr=getattr(strategy.exit, "trailing_atr", 1.2),
    )

    # 5. Stationary Circular Block bootstrap confidence interval
    pnls = base_res.get("pnls", np.array([]))
    bootstrap_sharpes = []
    if len(pnls) >= 10:
        rng = np.random.RandomState(42)
        n_p = len(pnls)
        block_len = max(2, min(8, n_p // 4))
        n_blocks = int(np.ceil(n_p / block_len))
        annual_factor = np.sqrt(min(252.0, max(12.0, float(n_p))))

        for _ in range(n_bootstrap):
            sample_blocks = []
            for _ in range(n_blocks):
                start = rng.randint(0, n_p)
                for b_i in range(block_len):
                    sample_blocks.append(pnls[(start + b_i) % n_p])
            sample = np.array(sample_blocks[:n_p])
            s_std = float(np.std(sample))
            if s_std > 1e-6:
                sh = float(np.mean(sample) / s_std * annual_factor)
                bootstrap_sharpes.append(sh)

    ci_lower = float(np.percentile(bootstrap_sharpes, 2.5)) if bootstrap_sharpes else 0.0
    ci_upper = float(np.percentile(bootstrap_sharpes, 97.5)) if bootstrap_sharpes else 0.0

    return {
        "base_sharpe": base_res["sharpe"],
        "base_trades": base_res["n_trades"],
        "spread_2x_sharpe": spread_2x_res["sharpe"],
        "spread_2x_trades": spread_2x_res["n_trades"],
        "spread_2x_pnl": spread_2x_res["net_pnl_usd"],
        "latency_sharpe": latency_res["sharpe"],
        "jitter_sharpe": jitter_res["sharpe"],
        "sharpe_ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
        "survives_spread_2x": spread_2x_res["net_pnl_usd"] > 0,
    }
