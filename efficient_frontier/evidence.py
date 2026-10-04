"""Paired historical comparisons with serial-dependence and multiple-test controls."""

from dataclasses import dataclass
from math import erfc, sqrt
from numbers import Real

import numpy as np
import pandas as pd

from .metrics import validate_periods_per_year


@dataclass
class EvidenceAnalysis:
    summary: pd.DataFrame
    windows: pd.DataFrame
    relative_equity: pd.DataFrame
    settings: dict
    warnings: list[str]


def _validate_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(equity, pd.DataFrame) or equity.empty or not equity.columns.is_unique:
        raise ValueError("Equity must be a nonempty DataFrame with unique column names.")
    if (
        not isinstance(equity.index, pd.DatetimeIndex) or equity.index.hasnans
        or not equity.index.is_unique or not equity.index.is_monotonic_increasing
    ):
        raise ValueError("Equity dates must be unique, increasing, and nonmissing.")
    if len(equity) < 2:
        raise ValueError("Equity requires an initial value and at least one later value.")
    raw_values = equity.to_numpy()
    if (
        any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in equity.dtypes)
        or raw_values.dtype.kind == "O" and any(
            isinstance(value, (bool, np.bool_, complex, np.complexfloating)) for value in raw_values.flat
        )
    ):
        raise ValueError("Boolean or complex equity values are invalid.")
    try:
        values = equity.astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Equity values must be numeric.") from exc
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise ValueError("Equity values must be finite and strictly positive.")
    return values


def _fit_hac(values: np.ndarray, benchmark: np.ndarray | None, lags: int):
    """Fit an intercept, with an optional benchmark slope, and Newey-West errors."""
    count, terms = len(values), 1 if benchmark is None else 2
    undefined = np.full(terms, np.nan)
    if count == 0:
        return undefined, undefined, "No observations remain after the entry interval."
    design = np.ones((count, terms))
    transform = np.eye(terms)
    if benchmark is not None:
        center = float(benchmark.mean())
        scale = float(np.std(benchmark))
        if scale <= 32 * np.finfo(float).eps * max(1, float(np.max(np.abs(benchmark)))):
            return undefined, undefined, "Benchmark variation is too small to identify alpha and beta."
        design[:, 1] = (benchmark - center) / scale
        transform = np.array([[1, -center / scale], [0, 1 / scale]])
    fitted = np.linalg.lstsq(design, values, rcond=None)[0]
    coefficients = transform @ fitted
    if count < 60:
        return coefficients, undefined, "At least 60 paired observations are required for inference."
    residuals = values - design @ fitted
    if np.max(np.abs(residuals)) <= 32 * np.finfo(float).eps * max(1, float(np.max(np.abs(values)))):
        return coefficients, undefined, "Residual variation is too small for uncertainty estimates."
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    inverse = np.linalg.inv(design.T @ design)
    covariance = transform @ inverse @ meat @ inverse @ transform.T * count / (count - terms)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    errors[errors <= 0] = np.nan
    note = "" if np.isfinite(errors[0]) else "The uncertainty estimate is undefined."
    return coefficients, errors, note


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Adjust one fixed family. Undefined tests retain their place in the family."""
    missing = ~np.isfinite(pvalues)
    values = np.where(missing, 1.0, pvalues)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    adjusted[missing] = np.nan
    return adjusted


def analyze_evidence(
    strategy_equity: pd.DataFrame,
    benchmark_equity: pd.DataFrame,
    *,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252,
    delayed_entry: bool = True,
) -> EvidenceAnalysis:
    """Compare net equity paths on identical dates without a future forecast.

    Full-path results and nonoverlap windows retain entry costs. Statistical
    estimates exclude the first cash and entry interval when delayed_entry is
    true. HAC uses Bartlett weights and a finite-sample covariance correction.
    Holm adjustment covers all strategy-benchmark pairs and both return tests.
    Individual confidence intervals do not include a multiple-test adjustment.
    """
    strategy_equity = _validate_equity(strategy_equity)
    benchmark_equity = _validate_equity(benchmark_equity)
    if not strategy_equity.index.equals(benchmark_equity.index):
        raise ValueError("Strategy and benchmark equity must have exactly the same dates.")
    periods_per_year = validate_periods_per_year(periods_per_year)
    if isinstance(risk_free_rate, (bool, np.bool_)) or not isinstance(risk_free_rate, Real) or not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be a finite number.")
    if not isinstance(delayed_entry, (bool, np.bool_)):
        raise ValueError("delayed_entry must be a boolean.")
    strategy_returns = strategy_equity.pct_change(fill_method=None).iloc[1:]
    benchmark_returns = benchmark_equity.pct_change(fill_method=None).iloc[1:]
    if not np.isfinite(strategy_returns.to_numpy()).all() or not np.isfinite(benchmark_returns.to_numpy()).all():
        raise ValueError("Equity paths must produce finite returns.")
    offset = int(delayed_entry)
    inference_dates = strategy_returns.index[offset:]
    count, full_count = len(inference_dates), len(strategy_returns)
    lags = min(max(count - 1, 0), int(np.floor(4 * (count / 100) ** (2 / 9))))
    chunks = np.array_split(np.arange(full_count), min(3, full_count))
    strategy_normalized = strategy_equity.div(strategy_equity.iloc[0])
    benchmark_normalized = benchmark_equity.div(benchmark_equity.iloc[0])
    rows, windows, relative = [], [], {}
    for strategy in strategy_equity.columns:
        for benchmark in benchmark_equity.columns:
            full_strategy = strategy_returns[strategy].to_numpy()
            full_benchmark = benchmark_returns[benchmark].to_numpy()
            active = full_strategy[offset:] - full_benchmark[offset:]
            advantage, advantage_error, advantage_note = _fit_hac(active, None, lags)
            coefficients, alpha_error, alpha_note = _fit_hac(
                full_strategy[offset:] - risk_free_rate / periods_per_year,
                full_benchmark[offset:] - risk_free_rate / periods_per_year,
                lags,
            )
            key = (strategy, benchmark)
            relative[key] = strategy_normalized[strategy] / benchmark_normalized[benchmark]
            strategy_growth = float(strategy_normalized[strategy].iloc[-1])
            benchmark_growth = float(benchmark_normalized[benchmark].iloc[-1])
            tracking_error = float(np.std(active, ddof=1) * np.sqrt(periods_per_year)) if count > 1 else np.nan
            if count and np.ptp(active) <= 32 * np.finfo(float).eps * max(1, float(np.max(np.abs(active)))):
                tracking_error = 0.0
            row = {
                "strategy": strategy,
                "benchmark": benchmark,
                "total_return_difference": strategy_growth - benchmark_growth,
                "cagr_difference": float(
                    np.expm1(np.log(strategy_growth) * periods_per_year / full_count)
                    - np.expm1(np.log(benchmark_growth) * periods_per_year / full_count)
                ),
                "annual_advantage": float(advantage[0] * periods_per_year),
                "beta": float(coefficients[1]),
                "alpha": float(coefficients[0] * periods_per_year),
                "tracking_error": tracking_error,
                "information_ratio": float(advantage[0] * periods_per_year / tracking_error) if tracking_error > 0 else np.nan,
                "observations": count,
                "advantage_note": advantage_note,
                "alpha_note": alpha_note,
            }
            for name, estimate, error in (
                ("advantage", advantage[0], advantage_error[0]),
                ("alpha", coefficients[0], alpha_error[0]),
            ):
                row[f"{name}_ci_low"] = float((estimate - 1.959963984540054 * error) * periods_per_year)
                row[f"{name}_ci_high"] = float((estimate + 1.959963984540054 * error) * periods_per_year)
                row[f"{name}_pvalue"] = erfc(abs(estimate / error) / sqrt(2)) if np.isfinite(error) else np.nan
            ahead = 0
            for number, positions in enumerate(chunks, start=1):
                first, last = positions[0], positions[-1]
                relative_return = float(relative[key].iloc[last + 1] / relative[key].iloc[first] - 1)
                ahead += relative_return > 0
                windows.append({
                    "strategy": strategy, "benchmark": benchmark, "window": number,
                    "start": strategy_returns.index[first].date().isoformat(),
                    "end": strategy_returns.index[last].date().isoformat(),
                    "observations": len(positions),
                    "relative_return": relative_return,
                    "annual_advantage": float((full_strategy[positions] - full_benchmark[positions]).mean() * periods_per_year),
                })
            row["windows_ahead"] = int(ahead)
            rows.append(row)
    summary = pd.DataFrame(rows).set_index(["strategy", "benchmark"])
    adjusted = _holm_adjust(summary[["advantage_pvalue", "alpha_pvalue"]].to_numpy().reshape(-1)).reshape(-1, 2)
    summary[["advantage_pvalue_adjusted", "alpha_pvalue_adjusted"]] = adjusted
    statuses = []
    for row in summary.itertuples():
        advantage_passes = row.advantage_pvalue_adjusted <= 0.05
        alpha_passes = row.alpha_pvalue_adjusted <= 0.05 and row.alpha > 0
        if count < 60:
            status = "Insufficient history"
        elif not np.isfinite(row.advantage_pvalue) and not np.isfinite(row.alpha_pvalue):
            status = "Uncertainty unavailable"
        elif advantage_passes and row.annual_advantage > 0:
            status = "Historical mean advantage and alpha" if alpha_passes else "Historical mean advantage"
        elif advantage_passes and row.annual_advantage < 0:
            status = "Historical mean underperformance"
        else:
            status = "Historical alpha" if alpha_passes else "No clear statistical advantage"
        statuses.append(status)
    summary["status"] = statuses
    relative_equity = pd.DataFrame(relative)
    relative_equity.columns = pd.MultiIndex.from_tuples(relative_equity.columns, names=["strategy", "benchmark"])
    if not np.isfinite(relative_equity.to_numpy()).all():
        raise ValueError("Equity paths must produce finite relative values.")
    warnings = [
        "These tests describe historical paths. They do not estimate the probability of future outperformance.",
        "Holm adjustment covers this report only. It does not account for prior strategy searches or repeated changes after review.",
        "HAC inference assumes consistently spaced observations and stable enough return relationships. Structural changes can invalidate the estimates.",
    ]
    if delayed_entry:
        warnings.append("Statistical estimates exclude the first cash and entry interval. Full-path results and windows retain entry costs.")
    if count < 60:
        warnings.append("Fewer than 60 paired observations remain. The report shows descriptive results without statistical inference.")
    if any(len(positions) < periods_per_year for positions in chunks):
        warnings.append("At least one comparison window has less than one year of observations. Window results can be unstable.")
    return EvidenceAnalysis(
        summary, pd.DataFrame(windows), relative_equity,
        {
            "periods_per_year": periods_per_year, "risk_free_rate": float(risk_free_rate),
            "delayed_entry": bool(delayed_entry), "inference_min_observations": 60,
            "inference_observations": count, "evaluation_observations": full_count,
            "hac_lags": lags, "confidence_level": 0.95, "multiple_testing_method": "Holm",
            "multiple_testing_tests": len(summary) * 2, "pvalue_sides": "two-sided",
            "inference_start": inference_dates[0].date().isoformat() if count else None,
            "inference_end": inference_dates[-1].date().isoformat() if count else None,
            "window_count": len(chunks),
        },
        warnings,
    )
