"""
Operator evaluation primitives for DSL conditions.
Evaluates gt, lt, between, cross_above, cross_below, and discrete event predicates.
"""

from __future__ import annotations

import numpy as np


def eval_operator(
    feature_arr: np.ndarray,
    op: str,
    threshold: float,
    threshold_high: float | None = None,
) -> np.ndarray:
    n = len(feature_arr)
    mask = np.zeros(n, dtype=bool)

    if op == "gt":
        mask = feature_arr > threshold
    elif op == "lt":
        mask = feature_arr < threshold
    elif op == "between":
        high = threshold_high if threshold_high is not None else (threshold + 20.0)
        mask = (feature_arr >= threshold) & (feature_arr <= high)
    elif op == "cross_above":
        if n > 1:
            mask[1:] = (feature_arr[1:] > threshold) & (feature_arr[:-1] <= threshold)
    elif op == "cross_below":
        if n > 1:
            mask[1:] = (feature_arr[1:] < threshold) & (feature_arr[:-1] >= threshold)
    elif op == "event":
        mask = np.isclose(feature_arr, threshold, atol=1e-3)
    else:
        raise ValueError(f"Unknown operator: {op}")

    return mask
