"""Observed-volatility comparisons and paired historical uncertainty estimates."""

import numpy as np
import pandas as pd

from .metrics import validate_periods_per_year


def _returns(net_returns):
    if not isinstance(net_returns, pd.DataFrame) or net_returns.empty or not net_returns.columns.is_unique:
        raise ValueError("Net returns must be a nonempty DataFrame with unique strategy names.")
    if (not isinstance(net_returns.index, pd.DatetimeIndex) or net_returns.index.hasnans
            or not net_returns.index.is_unique or not net_returns.index.is_monotonic_increasing):
        raise ValueError("Return dates must be unique, increasing, and nonmissing.")
    if len(net_returns) < 2:
        raise ValueError("At least two synchronized return observations are required.")
    if any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
           for dtype in net_returns.dtypes):
        raise ValueError("Boolean or complex returns are invalid.")
    try:
        values = net_returns.to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Net returns must be numeric.") from exc
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("All strategies need finite returns above -100% on exactly the same dates.")
    if (np.ptp(values, axis=0) == 0).any():
        raise ValueError("Zero-volatility paths cannot enter a positive common-volatility comparison.")
    return values


def _path_metrics(values, periods):
    logs = np.log1p(values)
    wealth = np.vstack([np.zeros(values.shape[1]), np.cumsum(logs, axis=0)])
    drawdown = np.expm1(wealth - np.maximum.accumulate(wealth, axis=0))
    return {
        "mean": values.mean(axis=0) * periods,
        "cagr": np.expm1(logs.mean(axis=0) * periods),
        "volatility": values.std(axis=0, ddof=1) * np.sqrt(periods),
        "max_drawdown": drawdown.min(axis=0),
    }


def normalized_metrics(net_returns: pd.DataFrame, periods=252) -> pd.DataFrame:
    """Compare net paths at one observed volatility without leverage.

    Each return is multiplied by min(all volatilities) / its own volatility.
    This uses the full evaluation period and assumes zero-interest cash.
    It is a statistical normalization, not a tradable cash sleeve: extra trades
    to maintain the scaled exposure are not charged. The initial wealth is one.
    """
    values = _returns(net_returns)
    periods = validate_periods_per_year(periods)
    raw = _path_metrics(values, periods)
    common = float(raw["volatility"].min())
    scale = common / raw["volatility"]
    normalized = _path_metrics(values * scale, periods)
    return pd.DataFrame({
        **{f"raw_{name}": result for name, result in raw.items()},
        **{f"normalized_{name}": result for name, result in normalized.items()},
        "scale": scale, "common_volatility": common,
    }, index=net_returns.columns.rename("strategy"))


def _stationary_indices(rng, count, observations, block_length):
    """Draw circular blocks with geometric lengths, shared by all columns."""
    starts = rng.integers(observations, size=(count, observations))
    restart = rng.random((count, observations)) < 1 / block_length
    restart[:, 0] = True
    positions = np.arange(observations)
    previous = np.maximum.accumulate(np.where(restart, positions, 0), axis=1)
    return (np.take_along_axis(starts, previous, axis=1) + positions - previous) % observations


def paired_normalized_bootstrap(
    net_returns: pd.DataFrame,
    benchmarks=("SPY", "QQQ"),
    *,
    periods=252,
    repetitions=10_000,
    block_lengths=(10, 21, 63),
    seed=20261018,
    confidence=.95,
) -> pd.DataFrame:
    """Give simultaneous one-sided lower bounds for all candidate mean gaps.

    The stationary bootstrap uses identical date indices for every column.
    Each resample recalculates one common volatility across every input path.
    Its maximum centered error covers every candidate-benchmark pair together.
    These approximate historical bounds assume weak temporal dependence and
    stationarity. They do not measure the probability of future outperformance
    or correct unrecorded searches, survivor bias, or prior review of these dates.
    """
    values = _returns(net_returns)
    periods = validate_periods_per_year(periods)
    benchmarks = tuple(benchmarks)
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or not set(benchmarks) <= set(net_returns.columns):
        raise ValueError("Supply unique benchmark names present in the return columns.")
    candidates = [name for name in net_returns if name not in benchmarks]
    if not candidates:
        raise ValueError("At least one candidate and one benchmark are required.")
    for name, value, minimum in (("repetitions", repetitions, 2), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    block_lengths = tuple(block_lengths)
    if (not block_lengths or len(set(block_lengths)) != len(block_lengths)
            or any(isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer))
                   or length < 1 for length in block_lengths)):
        raise ValueError("Block lengths must be unique positive integers.")
    if isinstance(confidence, (bool, np.bool_)) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    candidate_positions = net_returns.columns.get_indexer(candidates)
    benchmark_positions = net_returns.columns.get_indexer(benchmarks)
    metrics = normalized_metrics(net_returns, periods)
    means = metrics.normalized_mean.to_numpy()
    estimate = means[candidate_positions, None] - means[benchmark_positions]
    rows = []
    for block_length in block_lengths:
        rng = np.random.default_rng(np.random.SeedSequence([seed, block_length]))
        maximum_errors = np.empty(repetitions)
        for start in range(0, repetitions, 64):
            stop = min(start + 64, repetitions)
            indices = _stationary_indices(rng, stop - start, len(values), block_length)
            sample = values[indices]
            volatility = sample.std(axis=1, ddof=1)
            if (volatility <= 0).any() or (np.ptp(sample, axis=1) == 0).any():
                raise ValueError("A bootstrap resample has zero volatility; the comparison is undefined.")
            scale = volatility.min(axis=1, keepdims=True) / volatility
            sample_means = sample.mean(axis=1) * scale * periods
            gaps = sample_means[:, candidate_positions, None] - sample_means[:, None, benchmark_positions]
            maximum_errors[start:stop] = (gaps - estimate).max(axis=(1, 2))
        critical = max(0.0, float(np.quantile(maximum_errors, confidence)))
        for i, candidate in enumerate(candidates):
            for j, benchmark in enumerate(benchmarks):
                rows.append({
                    "candidate": candidate, "benchmark": benchmark, "block_length": block_length,
                    "estimate": float(estimate[i, j]), "lower_bound": float(estimate[i, j] - critical),
                    "simultaneous_critical": critical, "confidence": confidence,
                    "family_size": estimate.size, "repetitions": repetitions, "seed": seed,
                    "confidence_kind": "Simultaneous one-sided historical lower bound; not a future probability.",
                })
    return pd.DataFrame(rows)
