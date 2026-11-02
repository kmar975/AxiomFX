"""
UCB1 Multi-Armed Bandit for Family Budget Allocation (§ 7).
Distributes search evaluations across strategy families F1..F15 based on QD discovery rewards.
Automatically detects stagnation and reallocates budget from stagnant to active families.
"""

from __future__ import annotations

import math
from typing import Any


class UCB1Bandit:
    def __init__(
        self,
        families: list[str],
        c_param: float = 1.414,
        stagnation_window_pct: float = 0.20,
        stagnation_thresh: float = 0.05,
    ):
        self.families = families
        self.c_param = c_param
        self.stagnation_window = stagnation_window_pct
        self.stagnation_thresh = stagnation_thresh

        self.counts: dict[str, int] = {f: 0 for f in families}
        self.rewards: dict[str, float] = {f: 0.0 for f in families}
        self.recent_rewards: dict[str, list[float]] = {f: [] for f in families}
        self.total_pulls = 0

    def select_family(self) -> str:
        """
        Picks next family using Upper Confidence Bound (UCB1) formula.
        Ensures all families are pulled at least once initially.
        """
        self.total_pulls += 1

        # Pull unvisited families first
        for f in self.families:
            if self.counts[f] == 0:
                return f

        best_score = -1e9
        selected_family = self.families[0]

        log_total = math.log(max(1, self.total_pulls))

        for f in self.families:
            n_k = self.counts[f]
            mean_reward = self.rewards[f] / n_k

            # Stagnation penalty check
            is_stagnant = False
            rec = self.recent_rewards[f]
            if len(rec) >= 20:
                recent_rate = sum(rec[-20:]) / 20.0
                if recent_rate < self.stagnation_thresh:
                    is_stagnant = True

            exploration_bonus = self.c_param * math.sqrt(log_total / n_k)
            score = mean_reward + exploration_bonus
            if is_stagnant:
                score *= 0.20  # Heavily discount stagnant family

            if score > best_score:
                best_score = score
                selected_family = f

        return selected_family

    def record_feedback(self, family: str, reward: float) -> None:
        """
        Updates bandit statistics.
        reward is 1.0 for new cell discovery, 0.5 for improvement, 0.0 for reject.
        """
        if family in self.counts:
            self.counts[family] += 1
            self.rewards[family] += reward
            self.recent_rewards[family].append(reward)
            if len(self.recent_rewards[family]) > 50:
                self.recent_rewards[family].pop(0)

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_pulls": self.total_pulls,
            "counts": self.counts,
            "mean_rewards": {
                f: round(self.rewards[f] / max(1, self.counts[f]), 3)
                for f in self.families
            },
        }
