"""
Phase D: Discrete Parameter Plateau Centering (Zero Continuous Overfitting).
Tests discrete threshold neighborhoods on the existing grammar grid (+-1, +-2 notches)
for continuous percentile features, or SL ATR perturbations (+-15%) for binary/event strategies.
Rejects isolated needle elites (neighbor drop > 50%) and centers on the most stable plateau notch.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from grammar.strategy import Strategy
from search.fitness import evaluate_multipair_fitness

GRID_PERCENTILES = [10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0]


def get_grid_neighbors(current_threshold: float) -> list[float]:
    """
    Returns discrete neighborhood notches (+-1, +-2 steps) on the grammar grid.
    Never includes 50.0.
    """
    if current_threshold not in GRID_PERCENTILES:
        # Find closest grid notch
        closest_idx = int(np.argmin(np.abs(np.array(GRID_PERCENTILES) - current_threshold)))
    else:
        closest_idx = GRID_PERCENTILES.index(current_threshold)

    neighbor_indices = [
        max(0, closest_idx - 2),
        max(0, closest_idx - 1),
        closest_idx,
        min(len(GRID_PERCENTILES) - 1, closest_idx + 1),
        min(len(GRID_PERCENTILES) - 1, closest_idx + 2),
    ]
    neighbor_indices = sorted(set(neighbor_indices))
    return [GRID_PERCENTILES[i] for i in neighbor_indices]


def run_discrete_plateau_centering(
    strategy: Strategy,
    dev_data: dict[str, pd.DataFrame],
    features_by_pair: dict[str, pd.DataFrame],
    max_needle_drop: float = 0.50,
    min_trades_primary: int = 5,
    min_trades_basket: int = 50,
) -> tuple[bool, Strategy, float, dict[str, Any]]:
    """
    Evaluates discrete neighborhood of strategy thresholds or exit parameters.
    Protects binary/event features and rejects needle strategies whose neighbor fitness drops by > 50%.
    Returns: (is_robust, plateau_centered_strategy, plateau_fitness, details)
    """
    # Prohibit threshold 50.0 on triggers unconditionally
    if strategy.trigger is not None and strategy.trigger.operator != "event" and strategy.trigger.threshold == 50.0:
        strategy.trigger.threshold = 60.0

    # Find all conditions that are continuous percentile conditions
    # Order: trigger, setup, context, setup_2, context_2
    cond_slots = [
        ("trigger", strategy.trigger),
        ("setup", strategy.setup),
        ("context", strategy.context),
        ("setup_2", getattr(strategy, "setup_2", None)),
        ("context_2", getattr(strategy, "context_2", None)),
    ]

    continuous_slots: list[tuple[str, Condition]] = []
    for slot_name, cond in cond_slots:
        if cond is not None:
            # Must be continuous percentile (not event, not binary 0.5, and threshold >= 10.0)
            if cond.operator != "event" and cond.threshold >= 10.0 and cond.threshold != 0.5:
                continuous_slots.append((slot_name, cond))

    base_fitness, _base_details = evaluate_multipair_fitness(
        strategy, dev_data, features_by_pair, min_trades_primary=min_trades_primary, min_trades_basket=min_trades_basket
    )
    if base_fitness <= 0.0:
        first_slot = continuous_slots[0][0] if continuous_slots else "exit.sl_atr"
        return False, strategy, base_fitness, {
            "reason": "Base fitness non-positive",
            "plateau_centered_slot": first_slot,
        }

    if continuous_slots:
        # Case A: Continuous percentile conditions plateau centering (across all continuous slots)
        refined_strategy = copy.deepcopy(strategy)
        current_fitness = base_fitness
        all_neighbor_scores: dict[str, dict[float, float]] = {}
        last_details: dict[str, Any] = {}
        centered_slots: list[str] = []

        for target_slot, target_cond in continuous_slots:
            cond_in_refined = getattr(refined_strategy, target_slot)
            if cond_in_refined is None:
                continue

            if cond_in_refined.threshold not in GRID_PERCENTILES:
                closest_idx = int(np.argmin(np.abs(np.array(GRID_PERCENTILES) - cond_in_refined.threshold)))
                cond_in_refined.threshold = GRID_PERCENTILES[closest_idx]

            neighbors = get_grid_neighbors(cond_in_refined.threshold)
            neighbor_scores: dict[float, float] = {}

            for thresh in neighbors:
                test_strat = copy.deepcopy(refined_strategy)
                c = getattr(test_strat, target_slot)
                c.threshold = thresh
                fit, _ = evaluate_multipair_fitness(
                    test_strat, dev_data, features_by_pair, min_trades_primary=min_trades_primary, min_trades_basket=min_trades_basket
                )
                neighbor_scores[thresh] = fit

            center_fit = neighbor_scores.get(cond_in_refined.threshold, current_fitness)
            valid_other_fits = [v for k, v in neighbor_scores.items() if k != cond_in_refined.threshold and v > -900.0]
            if valid_other_fits:
                avg_neighbor_fit = float(np.mean(valid_other_fits))
                if avg_neighbor_fit < center_fit * (1.0 - max_needle_drop):
                    return False, strategy, base_fitness, {
                        "reason": f"Needle rejected on {target_slot}: neighbor avg {avg_neighbor_fit:.3f} < {center_fit * (1.0 - max_needle_drop):.3f}",
                        "plateau_centered_slot": target_slot,
                        "neighbor_scores": neighbor_scores,
                    }

            # Pick the notch with the highest smoothed plateau score (average of self and direct neighbors)
            best_notch = cond_in_refined.threshold
            best_plateau_avg = -1e9

            for thresh in neighbors:
                notch_idx = GRID_PERCENTILES.index(thresh)
                sub_neighbors = [
                    GRID_PERCENTILES[i]
                    for i in [max(0, notch_idx - 1), notch_idx, min(len(GRID_PERCENTILES) - 1, notch_idx + 1)]
                ]
                sub_fits = [neighbor_scores[t] for t in sub_neighbors if neighbor_scores.get(t, -999.0) > -900.0]
                sub_avg = float(np.mean(sub_fits)) if sub_fits else -999.0
                if sub_avg > best_plateau_avg:
                    best_plateau_avg = sub_avg
                    best_notch = thresh

            cond_in_refined.threshold = best_notch
            current_fitness, last_details = evaluate_multipair_fitness(
                refined_strategy, dev_data, features_by_pair, min_trades_primary=min_trades_primary, min_trades_basket=min_trades_basket
            )
            all_neighbor_scores[target_slot] = neighbor_scores
            centered_slots.append(target_slot)

        last_details["plateau_centered_slot"] = centered_slots[0] if centered_slots else "trigger"
        last_details["plateau_centered_slots"] = centered_slots
        last_details["neighbor_scores"] = all_neighbor_scores.get(last_details["plateau_centered_slot"], {})
        last_details["all_neighbor_scores"] = all_neighbor_scores

        return True, refined_strategy, current_fitness, last_details

    else:
        # Case B: All conditions are binary (0.5) or event (+-1.0)
        # Test plateau stability via perturbation of strategy.exit.sl_atr (x0.85, x1.0, x1.15)
        base_sl = strategy.exit.sl_atr
        sl_multipliers = [0.85, 1.0, 1.15]
        neighbor_scores = {}

        for mult in sl_multipliers:
            test_sl = round(base_sl * mult, 2)
            test_strat = copy.deepcopy(strategy)
            test_strat.exit.sl_atr = test_sl
            test_strat.exit.tp_atr = max(test_strat.exit.tp_atr, test_sl)
            fit, _ = evaluate_multipair_fitness(
                test_strat, dev_data, features_by_pair, min_trades_primary=min_trades_primary, min_trades_basket=min_trades_basket
            )
            neighbor_scores[test_sl] = fit

        center_fit = neighbor_scores.get(base_sl, base_fitness)
        valid_other_fits = [v for k, v in neighbor_scores.items() if k != base_sl and v > -900.0]
        if valid_other_fits:
            avg_neighbor_fit = float(np.mean(valid_other_fits))
            if avg_neighbor_fit < center_fit * (1.0 - max_needle_drop):
                return False, strategy, base_fitness, {
                    "reason": f"Needle rejected on SL perturbation: neighbor avg {avg_neighbor_fit:.3f} < {center_fit * (1.0 - max_needle_drop):.3f}",
                    "plateau_centered_slot": "exit.sl_atr",
                    "neighbor_scores": neighbor_scores,
                }

        best_sl = max(neighbor_scores, key=lambda k: neighbor_scores[k])
        refined_strategy = copy.deepcopy(strategy)
        refined_strategy.exit.sl_atr = best_sl
        refined_strategy.exit.tp_atr = max(refined_strategy.exit.tp_atr, best_sl)
        final_fitness, final_details = evaluate_multipair_fitness(
            refined_strategy, dev_data, features_by_pair, min_trades_primary=min_trades_primary, min_trades_basket=min_trades_basket
        )

        final_details["plateau_centered_slot"] = "exit.sl_atr"
        final_details["plateau_centered_sl_atr"] = best_sl
        final_details["plateau_avg_fitness"] = float(np.mean(list(neighbor_scores.values())))
        final_details["neighbor_scores"] = neighbor_scores

        return True, refined_strategy, final_fitness, final_details
