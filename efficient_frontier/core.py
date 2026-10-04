"""Long-only mean-variance optimization and a chronological holdout experiment."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .metrics import performance_metrics, validate_periods_per_year

TRADING_DAYS = 252


@dataclass
class Portfolio:
    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


@dataclass
class Analysis:
    portfolios: dict[str, Portfolio]
    frontier: pd.DataFrame
    frontier_weights: pd.DataFrame
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    equity: pd.DataFrame
    holdout_metrics: pd.DataFrame
    mean_returns: pd.Series
    covariance: pd.DataFrame
    warnings: list[str]
    periods_per_year: float = TRADING_DAYS


def _validate_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty or not prices.columns.is_unique:
        raise ValueError("prices must be a nonempty DataFrame with unique asset names.")
    if (
        not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
        or not prices.index.is_unique or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Prices must have unique, increasing, nonmissing dates.")
    values = prices.to_numpy()
    if (
        any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in prices.dtypes)
        or values.dtype.kind == "O" and any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
                                           for value in values.flat)
    ):
        raise ValueError("Boolean or complex prices are invalid.")
    try:
        prices = prices.astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Prices must be numeric values.") from exc
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and strictly positive; resolve missing data first.")
    return prices


def _solve(problem: cp.Problem) -> None:
    try:
        problem.solve(
            solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9,
            tol_feas=1e-9, max_iter=200,
        )
    except cp.error.SolverError as exc:
        raise RuntimeError(f"Portfolio optimization failed: {exc}") from exc
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Portfolio optimization failed: {problem.status}")


def optimize(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
    frontier_points: int = 40,
) -> tuple[dict[str, Portfolio], pd.DataFrame, pd.DataFrame, list[str]]:
    """Optimize annual arithmetic returns and annual covariance, in decimal units.

    Sharpe uses the supplied annual risk-free rate. Its convex formulation requires
    a feasible positive excess return; otherwise that portfolio is omitted.
    """
    if not isinstance(mean_returns, pd.Series) or mean_returns.empty:
        raise ValueError("mean_returns must be a nonempty pandas Series.")
    labels = mean_returns.index
    if not labels.is_unique:
        raise ValueError("Asset names must be unique.")
    if (
        not isinstance(covariance, pd.DataFrame)
        or not covariance.index.is_unique or not covariance.columns.is_unique
        or set(covariance.index) != set(labels) or set(covariance.columns) != set(labels)
    ):
        raise ValueError("Covariance rows and columns must match the asset names.")
    mu = mean_returns.to_numpy(dtype=float)
    sigma = covariance.loc[labels, labels].to_numpy(dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Expected returns and covariance must be finite.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be greater than zero and at most one.")
    if len(mu) * max_weight < 1 - 1e-12:
        raise ValueError("The position cap is infeasible: asset count * max_weight < 1.")
    if isinstance(frontier_points, bool) or not isinstance(frontier_points, (int, np.integer)) or frontier_points < 2:
        raise ValueError("frontier_points must be an integer of at least two.")
    if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance must be symmetric.")
    sigma = (sigma + sigma.T) / 2
    covariance_scale = float(np.max(np.abs(sigma)))
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        # Cholesky verifies the usual positive-definite case quickly. Singular
        # covariance is also valid, allowing tiny negative roundoff as before.
        if np.linalg.eigvalsh(sigma).min() < -1e-10 * covariance_scale:
            raise ValueError("Covariance must be positive semidefinite.")
    # Positive rescaling leaves all optimum weights unchanged and keeps solver
    # tolerances meaningful for very low-volatility inputs.
    solver_covariance = sigma / covariance_scale if covariance_scale else sigma

    warnings: list[str] = []

    def portfolio(name: str, values: np.ndarray) -> Portfolio:
        weights = np.asarray(values, dtype=float).reshape(-1)
        if (
            not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-6
            or weights.min() < -1e-6 or weights.max() > max_weight + 1e-6
        ):
            raise RuntimeError(f"Solver returned infeasible weights for {name}.")
        weights = np.clip(weights, 0, max_weight)
        weights /= weights.sum()
        expected = float(mu @ weights)
        volatility = float(np.sqrt(max(float(weights @ sigma @ weights), 0)))
        sharpe = (expected - risk_free_rate) / volatility if volatility > 1e-12 else float("nan")
        return Portfolio(name, pd.Series(weights, index=labels, name=name), expected, volatility, sharpe)

    weights = cp.Variable(len(mu))
    constraints = [weights >= 0, cp.sum(weights) == 1]
    if max_weight < 1:
        constraints.append(weights <= max_weight)
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(solver_covariance)))
    _solve(cp.Problem(objective, constraints))
    minimum = portfolio("Minimum volatility", weights.value)
    portfolios = {minimum.name: minimum}

    # The maximum feasible return is a linear program with a greedy exact solution.
    highest_weights = np.zeros(len(mu))
    remaining = 1.0
    for index in np.argsort(-mu, kind="stable"):
        highest_weights[index] = min(max_weight, remaining)
        remaining -= highest_weights[index]
        if remaining <= 1e-12:
            break
    highest_return = float(mu @ highest_weights)

    if highest_return - risk_free_rate > 1e-10:
        # Substitute y = w / ((mu-rf)'w); minimizing variance gives maximum Sharpe.
        scaled = cp.Variable(len(mu))
        excess = mu - risk_free_rate
        excess /= np.max(np.abs(excess))
        sharpe_constraints = [scaled >= 0, excess @ scaled == 1]
        if max_weight < 1:
            # A shared scalar keeps the cap constraints sparse as assets grow.
            normalizer = cp.Variable()
            sharpe_constraints.extend([normalizer == cp.sum(scaled), scaled <= max_weight * normalizer])
        sharpe_problem = cp.Problem(
            cp.Minimize(cp.quad_form(scaled, cp.psd_wrap(solver_covariance))),
            sharpe_constraints,
        )
        _solve(sharpe_problem)
        raw = np.asarray(scaled.value).reshape(-1)
        maximum = portfolio("Maximum Sharpe", raw / raw.sum())
        portfolios[maximum.name] = maximum
    else:
        warnings.append(
            "Maximum Sharpe omitted: no feasible portfolio has positive expected excess return "
            "above the risk-free rate."
        )
    equal = portfolio("Equal weight", np.full(len(mu), 1 / len(mu)))
    portfolios[equal.name] = equal
    if any(not np.isfinite(p.sharpe) for p in portfolios.values()):
        warnings.append("Sharpe is undefined for portfolios with zero estimated volatility.")

    frontier_portfolios = [minimum]
    if highest_return > minimum.expected_return + 1e-10:
        target = cp.Parameter()
        frontier_problem = cp.Problem(objective, constraints + [mu @ weights >= target])
        for required in np.linspace(minimum.expected_return, highest_return, frontier_points)[1:]:
            target.value = required
            _solve(frontier_problem)
            point = portfolio("Frontier", weights.value)
            if point.expected_return < required - 1e-6:
                raise RuntimeError("Frontier solution missed its required return.")
            frontier_portfolios.append(point)
    frontier = pd.DataFrame(
        [{"expected_return": p.expected_return, "volatility": p.volatility, "sharpe": p.sharpe}
         for p in frontier_portfolios]
    )
    frontier_weights = pd.DataFrame([p.weights.to_numpy() for p in frontier_portfolios], columns=labels)
    return portfolios, frontier, frontier_weights, warnings


def analyze(
    prices: pd.DataFrame,
    train_fraction: float = 0.7,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
    shrinkage: float = 0.1,
    frontier_points: int = 40,
    periods_per_year: float = TRADING_DAYS,
) -> Analysis:
    """Fit earlier observations, then hold fixed adjusted-price exposures.

    Means and sample covariance use periods_per_year for annualization.
    Covariance shrinks toward its diagonal. Holdout Sharpe and Sortino use annual
    arithmetic excess returns. Calmar uses CAGR. No fees, tax or FX.
    """
    prices = _validate_price_frame(prices)
    periods_per_year = validate_periods_per_year(periods_per_year)
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one.")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    split = int(np.floor(len(returns) * train_fraction))
    if split < 2 or len(returns) - split < 2:
        raise ValueError("At least two training returns and two holdout returns are required.")
    train_returns, test_returns = returns.iloc[:split], returns.iloc[split:]
    mean_returns = train_returns.mean() * periods_per_year
    sample_covariance = train_returns.cov() * periods_per_year
    diagonal = pd.DataFrame(np.diag(np.diag(sample_covariance)), index=prices.columns, columns=prices.columns)
    covariance = (1 - shrinkage) * sample_covariance + shrinkage * diagonal
    portfolios, frontier, frontier_weights, warnings = optimize(
        mean_returns, covariance, risk_free_rate, max_weight, frontier_points,
    )
    if split < 30:
        warnings.append(
            f"Only {split} training returns are available (fewer than 30). Return and risk estimates can be unstable."
        )
    if len(prices.columns) >= split:
        warnings.append(
            f"The training sample has {split} returns for {len(prices.columns)} assets. "
            "Sample covariance is singular before shrinkage. Use more history or fewer assets."
        )

    # Hold each asset's adjusted-price exposure from the final training close.
    relative_prices = prices.iloc[split:].div(prices.iloc[split])
    equity = pd.DataFrame({name: relative_prices @ p.weights for name, p in portfolios.items()})
    metrics = [{"portfolio": name, **performance_metrics(equity[name], risk_free_rate, periods_per_year)}
               for name in equity.columns]
    holdout_metrics = pd.DataFrame(metrics).set_index("portfolio")
    return Analysis(
        portfolios, frontier, frontier_weights, train_returns, test_returns,
        equity, holdout_metrics, mean_returns, covariance, warnings, periods_per_year,
    )
