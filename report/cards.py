"""
Strategy Fact Sheet and Card Generator (card.md).
Produces structured Markdown cards documenting hypothesis origin, formula,
DEV and HOLDOUT metrics, multi-pair performance, and final A/B/C status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grammar.strategy import Strategy


def generate_strategy_card_md(
    strategy: Strategy,
    metrics_dev: dict[str, Any],
    metrics_holdout: dict[str, Any],
    status: str,
    v2_p_value: float,
    multi_pair_metrics: dict[str, dict[str, float]],
    stress_results: dict[str, Any],
    output_path: Path,
    stats_results: dict[str, Any] | None = None,
) -> str:
    """
    Renders strategy card Markdown document and saves to output_path.
    Includes advanced statistical inference metrics from trejding architecture.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx_str = f"`{strategy.context.feature}` {strategy.context.operator} {strategy.context.threshold}" if strategy.context else "None"
    ctx2_str = f"`{strategy.context_2.feature}` {strategy.context_2.operator} {strategy.context_2.threshold}" if getattr(strategy, "context_2", None) else "None"
    setup_str = f"`{strategy.setup.feature}` {strategy.setup.operator} {strategy.setup.threshold} (window M={strategy.setup_expiry})" if strategy.setup else "None"
    setup2_str = f"`{strategy.setup_2.feature}` {strategy.setup_2.operator} {strategy.setup_2.threshold} (window M={strategy.setup_2_expiry})" if getattr(strategy, "setup_2", None) else "None"
    trig_str = f"`{strategy.trigger.feature}` {strategy.trigger.operator} {strategy.trigger.threshold}"

    stats = stats_results or {}
    dsr_prob = stats.get("dsr_probability", 0.0)
    fdr_adj_p = stats.get("fdr_adjusted_p_value", 1.0)
    white_p = stats.get("white_rc_p_value", 1.0)
    mc_dd95 = stats.get("mc_dd_95", 0.0)

    dev_wr = float(metrics_dev.get("win_rate_eur", metrics_dev.get("win_rate", 0.0)))
    dev_pf = float(metrics_dev.get("profit_factor_eur", metrics_dev.get("profit_factor", 0.0)))
    dev_trades = int(metrics_dev.get("n_trades_eur", metrics_dev.get("n_trades", metrics_dev.get("total_trades", 0))))
    dev_hold = float(metrics_dev.get("median_hold_hours", metrics_dev.get("avg_holding_bars", metrics_dev.get("holding_bars_median", 0.0))))

    hold_wr = float(metrics_holdout.get("win_rate", metrics_holdout.get("win_rate_eur", 0.0)))
    hold_pf = float(metrics_holdout.get("profit_factor", metrics_holdout.get("profit_factor_eur", 0.0)))
    hold_trades = int(metrics_holdout.get("n_trades", metrics_holdout.get("n_trades_eur", metrics_holdout.get("total_trades", 0))))
    hold_hold = float(metrics_holdout.get("median_hold_hours", metrics_holdout.get("avg_holding_bars", metrics_holdout.get("holding_bars_median", 0.0))))

    card = f"""# STRATEGY CARD: {strategy.strategy_id}

**Status:** `{status}` | **Family:** `{strategy.family}` | **V2 p-value:** `{v2_p_value:.4f}`

---

## 1. Specification & Grammar (Tree depth k=1..5)
- **Context 1:** {ctx_str}
- **Context 2:** {ctx2_str}
- **Setup 1:** {setup_str}
- **Setup 2:** {setup2_str}
- **Trigger:** {trig_str}
- **Exit:** Type `{strategy.exit.exit_type}`, SL = `{strategy.exit.sl_atr} ATR`, TP = `{strategy.exit.tp_atr} ATR`, Trailing = `{getattr(strategy.exit, 'trailing_atr', 1.2)} ATR`, Max Holding = `{strategy.exit.max_holding_bars} bars`
- **Sizing:** Vol Target `{strategy.sizing.vol_target * 100:.1f}%`
- **Symmetry:** `{strategy.symmetric}`

---

## 2. In-Sample DEV vs Out-of-Sample HOLDOUT Performance

| Metric | DEV (In-Sample, 1.5x Costs) | HOLDOUT (Out-of-Sample, 1.0x Costs) |
|---|---|---|
| **Sharpe Ratio** | {metrics_dev.get('fitness', 0.0):.2f} | {metrics_holdout.get('sharpe_holdout', 0.0):.2f} |
| **Win Rate** | {dev_wr * 100:.1f}% | {hold_wr * 100:.1f}% |
| **Profit Factor** | {dev_pf:.2f} | {hold_pf:.2f} |
| **Max Drawdown** | {metrics_dev.get('max_dd_pct', 0.0):.1f}% | {metrics_holdout.get('max_dd_pct', 0.0):.1f}% |
| **Trades Count** | {dev_trades} | {hold_trades} |
| **Median Hold** | {dev_hold:.1f} hours | {hold_hold:.1f} hours |

---

## 3. Advanced Statistical Inference (trejding module)
- **Deflated Sharpe Ratio (DSR):** `{dsr_prob * 100:.1f}%` (selection-bias adjusted)
- **FDR Adjusted p-value (Benjamini-Hochberg):** `{fdr_adj_p:.4f}`
- **White's Reality Check p-value:** `{white_p:.4f}`
- **Monte Carlo 95% Permutation Max Drawdown:** `${mc_dd95:.2f}`

---

## 4. Multi-Pair Performance (Cross-Sectional Pooled Evaluation)

| Pair | Trades | Win Rate % | Profit Factor | Sharpe | Net Return % |
|---|---|---|---|---|---|
"""
    for p, pm in multi_pair_metrics.items():
        card += f"| **{p}** | {pm.get('n_trades', 0)} | {pm.get('win_rate', 0.0)*100:.1f}% | {pm.get('profit_factor', 0.0):.2f} | {pm.get('sharpe', 0.0):.2f} | {pm.get('total_return_pct', 0.0):.1f}% |\n"

    card += f"""
---

## 5. Institutional Stress Tests
- **Spread x 2.0:** Sharpe `{stress_results.get('spread_2x_sharpe', 0.0):.2f}` (Net USD PnL: `${stress_results.get('spread_2x_pnl', 0.0):.2f}`)
- **1-Bar Execution Latency:** Sharpe `{stress_results.get('latency_sharpe', 0.0):.2f}`
- **Parameter Perturbation (+-20%):** Sharpe `{stress_results.get('jitter_sharpe', 0.0):.2f}`
- **Bootstrap 95% CI Sharpe:** `{stress_results.get('sharpe_ci_95', [0, 0])}`
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)

    return card
