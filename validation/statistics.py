"""
Institutional Statistical Inference & Multiple Testing Corrections.
Ported from trejding architecture (Bailey & Lopez de Prado 2014, White 2000, Benjamini-Hochberg 1995).
Provides:
- Deflated Sharpe Ratio (DSR) correcting for skewness, kurtosis, and selection bias.
- False Discovery Rate (FDR) Benjamini-Hochberg adjustment for candidate p-values.
- White's Reality Check with circular moving-block bootstrap against candidate data snooping.
- Monte Carlo Trade-Order Permutations (10,000 runs) for 95th/99th percentile Max Drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FDRResult:
    adjusted_p_values: pd.Series
    rejected: pd.Series
    alpha: float


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe: float
    selection_threshold_sharpe: float
    probability: float
    observations: int
    tested_count: int
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class RealityCheckResult:
    observed_statistic: float
    p_value: float
    observations: int
    candidate_count: int
    bootstrap_repetitions: int
    block_size: int


@dataclass(frozen=True)
class TradeOrderMonteCarloResult:
    max_drawdowns: pd.Series
    median_drawdown: float
    percentile_95_drawdown: float
    percentile_99_drawdown: float
    repetitions: int


def _finite_vector(values: pd.Series | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        vector = values.to_numpy(dtype=float)
    else:
        vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return vector


def _expected_maximum_sharpe(tested_count: int, observations: int) -> float:
    if tested_count <= 1:
        return 0.0
    normal = NormalDist()
    euler_mascheroni = 0.5772156649015329
    # Limit extreme z-scores to avoid numerical infinity
    inv_cdf_1 = normal.inv_cdf(min(0.999999999, 1.0 - 1.0 / tested_count))
    inv_cdf_2 = normal.inv_cdf(min(0.999999999, 1.0 - 1.0 / (tested_count * np.e)))
    expected_standard_max = (
        (1.0 - euler_mascheroni) * inv_cdf_1
        + euler_mascheroni * inv_cdf_2
    )
    return float(expected_standard_max / sqrt(max(1, observations - 1)))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    tested_count: int,
    risk_free_per_period: float = 0.0,
) -> DeflatedSharpeResult:
    """
    Estimates Deflated Sharpe Ratio (DSR) probability using number of trials and return moments.
    Corrects for data mining / multiple testing bias and non-normal return distributions.
    """
    values = _finite_vector(returns, name="returns")
    if len(values) < 3:
        raise ValueError("At least three return observations are required for DSR.")
    tested_count = max(1, int(tested_count))

    excess = values - risk_free_per_period
    mean = float(excess.mean())
    sample_std = float(excess.std(ddof=1))
    if sample_std <= 1e-12:
        return DeflatedSharpeResult(
            sharpe=0.0,
            selection_threshold_sharpe=0.0,
            probability=0.0,
            observations=len(values),
            tested_count=tested_count,
            skewness=0.0,
            kurtosis=3.0,
        )

    sharpe = mean / sample_std
    centered = excess - mean
    population_variance = float(np.mean(centered ** 2))
    skewness = float(np.mean(centered ** 3) / (population_variance ** 1.5 + 1e-12))
    kurtosis = float(np.mean(centered ** 4) / (population_variance ** 2.0 + 1e-12))

    selection_threshold = _expected_maximum_sharpe(tested_count, len(values))
    variance_term = (
        1.0
        - skewness * sharpe
        + ((kurtosis - 1.0) / 4.0) * (sharpe ** 2)
    )
    if variance_term <= 0 or not isfinite(variance_term):
        variance_term = 1.0

    z_score = (
        (sharpe - selection_threshold)
        * sqrt(len(values) - 1)
        / sqrt(variance_term)
    )
    z_score = np.clip(z_score, -8.0, 8.0)
    probability = float(NormalDist().cdf(z_score))

    return DeflatedSharpeResult(
        sharpe=sharpe,
        selection_threshold_sharpe=selection_threshold,
        probability=probability,
        observations=len(values),
        tested_count=tested_count,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def benjamini_hochberg(
    p_values: pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
) -> FDRResult:
    """
    Calculates Benjamini-Hochberg False Discovery Rate (FDR) adjusted p-values and rejection decisions.
    """
    if isinstance(p_values, pd.Series):
        p_series = p_values
    else:
        p_series = pd.Series(p_values)

    if p_series.empty:
        raise ValueError("At least one p-value is required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    values = p_series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        values = np.clip(np.nan_to_num(values, nan=1.0), 0.0, 1.0)

    count = len(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.arange(1, count + 1, dtype=float)
    sorted_adjusted = sorted_values * count / ranks
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)

    adjusted = np.empty(count, dtype=float)
    adjusted[order] = sorted_adjusted
    adjusted_series = pd.Series(adjusted, index=p_series.index, name="adjusted_p_value")
    rejected = adjusted_series.le(alpha).rename("reject_null")

    return FDRResult(adjusted_series, rejected, alpha)


def white_reality_check(
    candidate_returns: pd.DataFrame | np.ndarray,
    *,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    bootstrap_repetitions: int = 1000,
    block_size: int | None = None,
    random_seed: int = 42,
) -> RealityCheckResult:
    """
    White's Reality Check (White 2000) with stationary circular block resampling.
    Imposes the joint null hypothesis that no candidate strategy beats the benchmark.
    """
    if isinstance(candidate_returns, pd.DataFrame):
        values = candidate_returns.to_numpy(dtype=float)
    else:
        values = np.asarray(candidate_returns, dtype=float)

    if values.ndim == 1:
        values = values[:, None]

    sample_count, candidate_count = values.shape
    if sample_count < 2 or candidate_count == 0:
        return RealityCheckResult(0.0, 1.0, sample_count, candidate_count, bootstrap_repetitions, 1)

    if benchmark_returns is not None:
        bench_vec = _finite_vector(benchmark_returns, name="benchmark_returns")
        if len(bench_vec) != sample_count:
            raise ValueError("Benchmark returns length must match candidate returns")
    else:
        bench_vec = np.zeros(sample_count, dtype=float)

    excess = values - bench_vec[:, None]
    selected_block_size = max(1, int(sample_count ** (1.0 / 3.0))) if block_size is None else block_size
    selected_block_size = min(max(1, selected_block_size), sample_count)

    observed = sqrt(sample_count) * float(excess.mean(axis=0).max())
    centered = excess - excess.mean(axis=0, keepdims=True)
    block_count = (sample_count + selected_block_size - 1) // selected_block_size
    offsets = np.arange(selected_block_size)

    rng = np.random.default_rng(random_seed)
    exceedances = 0

    for _ in range(bootstrap_repetitions):
        starts = rng.integers(0, sample_count, size=block_count)
        indices = ((starts[:, None] + offsets) % sample_count).ravel()[:sample_count]
        bootstrap_statistic = sqrt(sample_count) * float(centered[indices].mean(axis=0).max())
        if bootstrap_statistic >= observed:
            exceedances += 1

    p_value = float((exceedances + 1.0) / (bootstrap_repetitions + 1.0))
    return RealityCheckResult(
        observed_statistic=observed,
        p_value=p_value,
        observations=sample_count,
        candidate_count=candidate_count,
        bootstrap_repetitions=bootstrap_repetitions,
        block_size=selected_block_size,
    )


def monte_carlo_trade_order(
    trade_pnls: pd.Series | np.ndarray,
    *,
    repetitions: int = 10_000,
    random_seed: int = 42,
) -> TradeOrderMonteCarloResult:
    """
    Estimates maximum drawdown distribution by shuffling the realized trade order 10,000 times.
    Computes median, 95th, and 99th percentile maximum drawdown.
    """
    values = _finite_vector(trade_pnls, name="trade_pnls")
    if values.size < 2:
        return TradeOrderMonteCarloResult(
            max_drawdowns=pd.Series([0.0]),
            median_drawdown=0.0,
            percentile_95_drawdown=0.0,
            percentile_99_drawdown=0.0,
            repetitions=repetitions,
        )

    rng = np.random.default_rng(random_seed)
    drawdowns = np.empty(repetitions, dtype=float)

    for r in range(repetitions):
        perm_pnls = rng.permutation(values)
        equity = np.concatenate(([0.0], np.cumsum(perm_pnls)))
        peaks = np.maximum.accumulate(equity)
        drawdowns[r] = float((peaks - equity).max())

    drawdown_series = pd.Series(drawdowns, name="max_drawdown")
    return TradeOrderMonteCarloResult(
        max_drawdowns=drawdown_series,
        median_drawdown=float(drawdown_series.quantile(0.50)),
        percentile_95_drawdown=float(drawdown_series.quantile(0.95)),
        percentile_99_drawdown=float(drawdown_series.quantile(0.99)),
        repetitions=repetitions,
    )
