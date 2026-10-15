"""Relative risk profiles from historical means, with optional window research."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .core import Portfolio, TRADING_DAYS, _solve, _validate_price_frame, optimize
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
    return_objective: str = "historical_mean",
) -> ProfileAnalysis:
    """Use all supplied returns to select Low, Medium, and Extreme profiles.

    By default, minimize variance at each full-sample arithmetic mean target.
    Low has minimum volatility. Medium is the sampled point nearest halfway
    between the endpoint volatilities. Extreme has maximum feasible mean return.
    Three chronological windows supply diagnostics for these same allocations.

    The optional worst_window objective maximizes the weakest window return.
    Its Medium uses the midpoint return target, with minimum variance as a
    tie-break. Targets allow 1e-8 times the largest absolute window estimate.
    All profile labels express relative risk, not a return guarantee.
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
    if return_objective not in ("historical_mean", "worst_window"):
        raise ValueError("return_objective must be historical_mean or worst_window.")
    if (
        isinstance(frontier_points, bool) or not isinstance(frontier_points, (int, np.integer))
        or frontier_points < 3
    ):
        raise ValueError("frontier_points must be an integer of at least three.")
    if return_objective == "worst_window" and frontier_points % 2 != 1:
        raise ValueError("worst_window frontier_points must be an odd integer of at least three.")
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

    if return_objective == "historical_mean":
        _, frontier, frontier_weights, warnings = optimize(
            pd.Series(mean_returns, index=labels), pd.DataFrame(covariance, index=labels, columns=labels),
            risk_free_rate=risk_free_rate, max_weight=max_weight, frontier_points=frontier_points,
        )
        midpoint_volatility = (frontier.volatility.iloc[0] + frontier.volatility.iloc[-1]) / 2
        middle = int(np.argmin(np.abs(frontier.volatility.to_numpy() - midpoint_volatility)))
        if frontier.expected_return.iloc[-1] <= 0:
            warnings.append("No feasible portfolio has a positive estimated historical mean return.")
    else:
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
            {"expected_return": point.expected_return, "volatility": point.volatility, "sharpe": point.sharpe}
            for point in frontier_portfolios
        ])
        frontier_weights = pd.DataFrame([point.weights.to_numpy() for point in frontier_portfolios], columns=labels)
        middle = frontier_points // 2
        if high_score <= 0:
            warnings.append("No feasible portfolio has a positive estimated return in all three historical windows.")

    frontier["worst_window_return"] = np.min(window_means @ frontier_weights.to_numpy().T, axis=0)
    selected = {"Low": 0, "Medium": middle, "Extreme": len(frontier) - 1}
    portfolios = {
        name: Portfolio(
            name, frontier_weights.iloc[position].rename(name),
            float(frontier.expected_return.iloc[position]), float(frontier.volatility.iloc[position]),
            float(frontier.sharpe.iloc[position]),
        )
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
    if len(frontier) == 1 or (
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
            "return_objective": return_objective,
            "selection_method": (
                "historical_mean_frontier" if return_objective == "historical_mean"
                else "worst_window_return_frontier"
            ),
        },
        warnings=warnings,
    )
