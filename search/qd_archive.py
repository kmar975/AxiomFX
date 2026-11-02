"""
Quality-Diversity (MAP-Elites) Behavioral Archive (§ 7).
Maintains a diverse archive of high-performing strategies across 6 behavioral descriptors:
log_holding, long_frac, trend_corr, vol_exposure, trades_per_year, and payoff_profile.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from grammar.strategy import Strategy


class QDCell:
    def __init__(
        self,
        cell_index: tuple[int, ...],
        strategy: Strategy,
        fitness: float,
        descriptors: np.ndarray,
        metrics: dict[str, Any],
    ):
        self.cell_index = cell_index
        self.strategy = strategy
        self.fitness = fitness
        self.descriptors = descriptors
        self.metrics = metrics


class MAPElitesArchive:
    def __init__(
        self,
        bins_per_dim: int = 5,
        n_dims: int = 6,
    ):
        self.bins_per_dim = bins_per_dim
        self.n_dims = n_dims
        self.archive: dict[tuple[int, ...], QDCell] = {}
        self.n_evaluations = 0
        self.n_discoveries = 0
        self.n_improvements = 0

    def compute_descriptors(
        self,
        strategy: Strategy,
        trade_durations: np.ndarray,
        trade_directions: np.ndarray,
        trade_pnls: np.ndarray,
        n_bars_total: int,
    ) -> np.ndarray:
        """
        Calculates 6-dimensional normalized descriptor vector [0.0..1.0]:
        1) log_holding
        2) long_frac
        3) trend_corr
        4) vol_exposure
        5) trades_per_year
        6) payoff_profile
        """
        n_trades = len(trade_pnls)
        if n_trades == 0:
            return np.full(6, 0.5, dtype=np.float64)

        # 1. log_holding (normalized between 1 bar and 100 bars)
        med_hold = float(np.median(trade_durations)) if len(trade_durations) > 0 else 10.0
        d_log_holding = np.clip(math.log(1.0 + med_hold) / math.log(101.0), 0.0, 1.0)

        # 2. long_frac
        n_longs = int(np.sum(trade_directions == 1))
        d_long_frac = float(n_longs / n_trades)

        # 3. trend_corr (empirical PnL skewness & trend family alignment: [0.0..1.0])
        # Trend-following strategies exhibit positive right-tail skew; mean-reversion exhibits negative skew
        if n_trades >= 4 and float(np.std(trade_pnls)) > 1e-6:
            diff = trade_pnls - np.mean(trade_pnls)
            m3 = float(np.mean(diff ** 3))
            m2 = float(np.var(trade_pnls))
            skew = m3 / (m2 ** 1.5 + 1e-9)
            skew_norm = float(np.clip((skew + 2.5) / 5.0, 0.05, 0.95))
        else:
            skew_norm = 0.50

        fam_bias = 0.65 if ("Trend" in strategy.family or "Breakout" in strategy.family) else (0.35 if ("Mean" in strategy.family or "Pullback" in strategy.family) else 0.50)
        d_trend_corr = float(np.clip(0.70 * skew_norm + 0.30 * fam_bias, 0.0, 1.0))

        # 4. vol_exposure (causal ATR risk tolerance & volatility regime dependence: [0.0..1.0])
        sl_norm = float(np.clip((strategy.exit.sl_atr - 1.0) / 3.0, 0.0, 1.0))
        tp_norm = float(np.clip((strategy.exit.tp_atr - 1.5) / 4.0, 0.0, 1.0))
        conds = [strategy.trigger, strategy.context, getattr(strategy, "context_2", None), strategy.setup, getattr(strategy, "setup_2", None)]
        s5_active = 1.0 if any(c is not None and "s5" in c.source for c in conds) else 0.0
        d_vol_exposure = float(np.clip(0.45 * sl_norm + 0.35 * tp_norm + 0.20 * s5_active, 0.0, 1.0))

        # 5. trades_per_year (normalized 10 to 300 trades)
        years = max(0.5, n_bars_total / (252.0 * 24.0))
        tpy = n_trades / years
        d_tpy = np.clip(tpy / 300.0, 0.0, 1.0)

        # 6. payoff_profile (Win Rate adjusted by profit factor)
        wins = trade_pnls[trade_pnls > 0]
        wr = len(wins) / n_trades
        d_payoff = np.clip(wr, 0.0, 1.0)

        return np.array([d_log_holding, d_long_frac, d_trend_corr, d_vol_exposure, d_tpy, d_payoff], dtype=np.float64)

    def get_cell_index(self, descriptors: np.ndarray) -> tuple[int, ...]:
        indices = []
        for val in descriptors:
            idx = int(np.clip(val * self.bins_per_dim, 0, self.bins_per_dim - 1))
            indices.append(idx)
        return tuple(indices)

    def add_strategy(
        self,
        strategy: Strategy,
        fitness: float,
        descriptors: np.ndarray,
        metrics: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Attempts to add candidate to archive.
        Returns: (added, reason: 'new_cell', 'improved_cell', 'rejected')
        """
        self.n_evaluations += 1
        # Strictly reject non-positive fitness strategies from the archive
        if fitness <= 0.0:
            return False, "rejected"

        cell_idx = self.get_cell_index(descriptors)

        if cell_idx not in self.archive:
            # New cell discovery!
            self.archive[cell_idx] = QDCell(cell_idx, strategy, fitness, descriptors, metrics)
            self.n_discoveries += 1
            return True, "new_cell"

        existing = self.archive[cell_idx]
        if fitness > existing.fitness:
            # Elite replacement / improvement!
            self.archive[cell_idx] = QDCell(cell_idx, strategy, fitness, descriptors, metrics)
            self.n_improvements += 1
            return True, "improved_cell"

        return False, "rejected"

    def get_all_elites(self) -> list[QDCell]:
        """Returns all currently stored diverse elites."""
        return list(self.archive.values())

    def get_coverage(self) -> float:
        total_cells = self.bins_per_dim ** self.n_dims
        return float(len(self.archive) / total_cells)
