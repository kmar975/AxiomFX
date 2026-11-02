"""
Phase A: Classical FX Baseline Strategies (Appendix D).
Implements standard institutional templates (Moving Average crossovers, RSI zones,
Bollinger Mean-Reversion, Squeeze Breakouts, Donchian Turtles, and Session Breakouts).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from grammar.strategy import Condition, ExitSpec, Strategy
from search.funnel import FunnelExecutor


def generate_baseline_strategies() -> list[Strategy]:
    """
    Generates standard classical benchmarks from Appendix D.
    """
    baselines = []

    # 1. MA Crossover / Trend
    b1 = Strategy(
        strategy_id="BASE_MA_CROSS",
        family="F1_Trend",
        trigger=Condition("s1_ema_dist_pct", "s1", "gt", 60.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )
    baselines.append(b1)

    # 2. RSI Mean-Reversion / Fade
    b2 = Strategy(
        strategy_id="BASE_RSI_FADE",
        family="F3_MeanReversion",
        trigger=Condition("s1_rsi_14", "s1", "lt", 30.0, direction_mode="fade"),
        exit=ExitSpec(sl_atr=1.0, tp_atr=1.5, max_holding_bars=12),
    )
    baselines.append(b2)

    # 3. Bollinger / Squeeze Breakout
    b3 = Strategy(
        strategy_id="BASE_SQUEEZE_BO",
        family="F2_Breakout",
        setup=Condition("s5_squeeze_on", "s5", "event", 1.0),
        setup_expiry=5,
        trigger=Condition("s1_donchian_pos", "s1", "gt", 80.0),
        exit=ExitSpec(sl_atr=2.0, tp_atr=3.0, max_holding_bars=36),
    )
    baselines.append(b3)

    # 4. Donchian / Turtle Breakout (20-bar)
    b4 = Strategy(
        strategy_id="BASE_DONCHIAN_20",
        family="F1_Trend",
        trigger=Condition("s1_donchian_pos", "s1", "gt", 90.0),
        exit=ExitSpec(sl_atr=2.0, tp_atr=4.0, max_holding_bars=48),
    )
    baselines.append(b4)

    # 5. Pin Bar on Key Level
    b5 = Strategy(
        strategy_id="BASE_PIN_LEVEL",
        family="F5_Candles",
        context=Condition("s4_dist_round100_pct", "s4", "lt", 30.0),
        trigger=Condition("s2_seed_pin_bull", "s2", "event", 1.0),
        exit=ExitSpec(sl_atr=1.2, tp_atr=2.4, max_holding_bars=18),
    )
    baselines.append(b5)

    # 6. Asian Range Breakout
    b6 = Strategy(
        strategy_id="BASE_ASIAN_BO",
        family="F8_Sessions",
        context=Condition("s6_session_london", "s6", "event", 1.0),
        trigger=Condition("s6_asian_breakout", "s6", "event", 1.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=20),
    )
    baselines.append(b6)

    # 7. London Judas Swing Fade
    b7 = Strategy(
        strategy_id="BASE_JUDAS_FADE",
        family="F8_Sessions",
        context=Condition("s6_session_london", "s6", "event", 1.0),
        trigger=Condition("s6_judas_swing", "s6", "event", 1.0),
        exit=ExitSpec(sl_atr=1.0, tp_atr=2.0, max_holding_bars=12),
    )
    baselines.append(b7)

    # 8. Relative Currency Strength Basket Momentum
    b8 = Strategy(
        strategy_id="BASE_CURR_MOM",
        family="F12_CurrencyStrength",
        trigger=Condition("s9_rel_strength_diff_pct", "s9", "gt", 70.0),
        exit=ExitSpec(sl_atr=1.5, tp_atr=2.5, max_holding_bars=24),
    )
    baselines.append(b8)

    return baselines


def run_phase_a_baselines(
    dev_data: dict[str, pd.DataFrame],
    features_by_pair: dict[str, pd.DataFrame],
    funnel: FunnelExecutor,
) -> list[tuple[Strategy, float, dict[str, Any]]]:
    """
    Evaluates Phase A baselines across DEV dataset.
    """
    baselines = generate_baseline_strategies()
    results = []
    for strat in baselines:
        _passed, fitness, details = funnel.evaluate_s2_full_dev(strat, dev_data, features_by_pair)
        results.append((strat, fitness, details))
    return results
