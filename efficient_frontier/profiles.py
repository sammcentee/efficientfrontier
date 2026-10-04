"""Relative risk profiles on a frontier of worst historical window returns."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .core import Portfolio, TRADING_DAYS, _solve, _validate_price_frame
from .metrics import validate_periods_per_year


@dataclass
class ProfileAnalysis:
    portfolios: dict[str, Portfolio]
    frontier: pd.DataFrame
    frontier_weights: pd.DataFrame
    windows: pd.DataFrame
    window_returns: pd.DataFrame
    summary: pd.DataFrame
    as_of: str
    observations: int
    settings: dict
    warnings: list[str]


def build_profiles(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
    shrinkage: float = 0.1,
    periods_per_year: float = TRADING_DAYS,
    frontier_points: int = 21,
) -> ProfileAnalysis:
    """Use all supplied returns to select Low, Medium, and Extreme profiles.

    Three chronological windows supply annual arithmetic return estimates.
    The frontier minimizes full-sample variance at each worst-window target.
    Medium uses the midpoint target. Extreme maximizes the worst-window return,
    with minimum variance as a tie-break. The labels express relative risk.
    Return targets allow a numerical tolerance of 1e-8 times the largest
    absolute window return estimate.
    """
    prices = _validate_price_frame(prices)
    periods_per_year = validate_periods_per_year(periods_per_year)
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be greater than zero and at most one.")
    if len(prices.columns) * max_weight < 1 - 1e-12:
        raise ValueError("The position cap is infeasible: asset count * max_weight < 1.")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one.")
    if (
        isinstance(frontier_points, bool) or not isinstance(frontier_points, (int, np.integer))
        or frontier_points < 3 or frontier_points % 2 != 1
    ):
        raise ValueError("frontier_points must be an odd integer of at least three.")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if len(returns) < 6:
        raise ValueError("Profiles require at least six returns, with two returns per historical window.")
    chunks = [returns.iloc[positions] for positions in np.array_split(np.arange(len(returns)), 3)]
    window_means = np.asarray([chunk.mean().to_numpy() for chunk in chunks]) * periods_per_year
    mean_returns = returns.mean().to_numpy() * periods_per_year
    sample_covariance = returns.cov().to_numpy() * periods_per_year
    covariance = (1 - shrinkage) * sample_covariance + shrinkage * np.diag(np.diag(sample_covariance))
    if not np.isfinite(window_means).all() or not np.isfinite(covariance).all():
        raise ValueError("Return estimates and covariance must be finite.")

    covariance_scale = float(np.max(np.abs(covariance)))
    solver_covariance = covariance / covariance_scale if covariance_scale else covariance
    return_scale = float(np.max(np.abs(window_means))) or 1.0
    solver_means = window_means / return_scale
    return_tolerance = 1e-8
    labels = prices.columns
    warnings = []

    def solve(problem):
        _solve(problem)
        message = (
            "The profile solver reported limited numerical accuracy. "
            "Weight and return checks passed, but the optimum can be approximate."
        )
        if problem.status == cp.OPTIMAL_INACCURATE and message not in warnings:
            warnings.append(message)

    def portfolio(name, values):
        allocation = np.asarray(values, dtype=float).reshape(-1)
        if (
            not np.isfinite(allocation).all() or abs(allocation.sum() - 1) > 1e-6
            or allocation.min() < -1e-6 or allocation.max() > max_weight + 1e-6
        ):
            raise RuntimeError(f"Solver returned infeasible weights for {name}.")
        allocation = np.clip(allocation, 0, max_weight)
        allocation /= allocation.sum()
        expected = float(mean_returns @ allocation)
        volatility = float(np.sqrt(max(float(allocation @ covariance @ allocation), 0)))
        sharpe = (expected - risk_free_rate) / volatility if volatility > 0 else float("nan")
        return Portfolio(name, pd.Series(allocation, index=labels, name=name), expected, volatility, sharpe)

    weights = cp.Variable(len(labels))
    constraints = [weights >= 0, weights <= max_weight, cp.sum(weights) == 1]
    # The standard half-variance scale leaves the frontier unchanged.
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(solver_covariance)) / 2)
    solve(cp.Problem(objective, constraints))
    minimum = portfolio("Low", weights.value)
    low_score = float(np.min(solver_means @ minimum.weights.to_numpy()))

    score = cp.Variable()
    solve(cp.Problem(cp.Maximize(score), constraints + [solver_means @ weights >= score]))
    endpoint = portfolio("Extreme", weights.value)
    high_score = max(low_score, float(np.min(solver_means @ endpoint.weights.to_numpy())))
    frontier_portfolios = [minimum]
    if high_score - low_score <= return_tolerance:
        frontier_portfolios *= frontier_points
    else:
        target = cp.Parameter()
        # Price ratios can break an exact return tie at floating-point precision.
        problem = cp.Problem(objective, constraints + [solver_means @ weights >= target - return_tolerance])
        for required in np.linspace(low_score, high_score, frontier_points)[1:]:
            target.value = required
            solve(problem)
            point = portfolio("Frontier", weights.value)
            if np.min(solver_means @ point.weights.to_numpy()) < required - 2 * return_tolerance:
                raise RuntimeError("Profile solution missed its required worst-window return.")
            frontier_portfolios.append(point)

    frontier = pd.DataFrame([
        {
            "expected_return": point.expected_return,
            "volatility": point.volatility,
            "sharpe": point.sharpe,
            "worst_window_return": float(np.min(window_means @ point.weights.to_numpy())),
        }
        for point in frontier_portfolios
    ])
    frontier_weights = pd.DataFrame([point.weights.to_numpy() for point in frontier_portfolios], columns=labels)
    selected = {"Low": 0, "Medium": frontier_points // 2, "Extreme": frontier_points - 1}
    portfolios = {
        name: portfolio(name, frontier_portfolios[position].weights.to_numpy())
        for name, position in selected.items()
    }
    summary = frontier.iloc[list(selected.values())].copy()
    summary.index = pd.Index(selected, name="profile")
    summary["max_weight"] = [point.weights.max() for point in portfolios.values()]
    summary["effective_holdings"] = [1 / (point.weights @ point.weights) for point in portfolios.values()]
    windows = pd.DataFrame([
        {
            "window": number,
            "start": chunk.index[0].date().isoformat(),
            "end": chunk.index[-1].date().isoformat(),
            "observations": len(chunk),
            "years": len(chunk) / periods_per_year,
        }
        for number, chunk in enumerate(chunks, start=1)
    ]).set_index("window")
    window_returns = pd.DataFrame(
        {name: window_means @ point.weights.to_numpy() for name, point in portfolios.items()},
        index=windows.index,
    )
    if (windows.years < 1).any():
        warnings.append("At least one historical window has less than one year of observations. Estimates can be unstable.")
    if high_score <= 0:
        warnings.append("No feasible portfolio has a positive estimated return in all three historical windows.")
    if high_score - low_score <= return_tolerance or (
        frontier.volatility.iloc[-1] - frontier.volatility.iloc[0]
        <= 1e-8 * np.sqrt(covariance_scale)
    ):
        warnings.append("The profiles have the same or nearly the same estimated risk. The data provide no distinct risk levels.")
    return ProfileAnalysis(
        portfolios=portfolios,
        frontier=frontier,
        frontier_weights=frontier_weights,
        windows=windows,
        window_returns=window_returns,
        summary=summary,
        as_of=prices.index[-1].date().isoformat(),
        observations=len(returns),
        settings={
            "periods_per_year": periods_per_year,
            "risk_free_rate": risk_free_rate,
            "max_weight": max_weight,
            "shrinkage": shrinkage,
            "window_count": 3,
            "selection_method": "worst_window_return_frontier",
        },
        warnings=warnings,
    )
