"""Chronological portfolio backtests with delayed execution and explicit costs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import TRADING_DAYS, _validate_price_frame, optimize
from .metrics import performance_metrics, validate_periods_per_year

METHOD_NAMES = ("Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window")
PORTFOLIO_NAMES = ("Minimum volatility", "Maximum Sharpe", "Equal weight")


@dataclass
class BacktestStudy:
    equity: pd.DataFrame
    metrics: pd.DataFrame
    allocations: dict[str, pd.DataFrame]
    holdings: dict[str, pd.DataFrame]
    trades: dict[str, pd.DataFrame]
    warnings: list[str]
    settings: dict


def _after_cost(nav: float, positions: np.ndarray, target: np.ndarray, rate: float) -> float:
    """Solve the self-financing rebalance, charging both buys and sells."""
    if rate == 0:
        return nav
    low, high = 0.0, nav
    for _ in range(64):
        value = (low + high) / 2
        if value + rate * np.abs(value * target - positions).sum() > nav:
            high = value
        else:
            low = value
    return (low + high) / 2


def run_backtests(
    prices: pd.DataFrame,
    train_fraction: float = 0.7,
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
    shrinkage: float = 0.1,
    rebalance_every: int = 21,
    rolling_window: int | None = None,
    cost_bps: float = 10.0,
    periods_per_year: float = TRADING_DAYS,
) -> BacktestStudy:
    """Compare four execution methods for three long-only portfolio targets.

    Fit through one close and trade at the following close. The first out-of-
    sample session is cash earning zero interest. Refit windows exclude execution-
    day returns. Costs are basis points per absolute dollar traded, including
    entry and both legs of later trades. There is no terminal liquidation.

    Holding contribution is dollar profit in initial-capital units before costs;
    contributions minus total cost reconcile to the portfolio's total return.
    """
    prices = _validate_price_frame(prices)
    periods_per_year = validate_periods_per_year(periods_per_year)
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be greater than zero and at most one.")
    if len(prices.columns) * max_weight < 1 - 1e-12:
        raise ValueError("The position cap is infeasible: asset count * max_weight < 1.")
    if (
        isinstance(rebalance_every, bool) or not isinstance(rebalance_every, (int, np.integer))
        or rebalance_every < 1
    ):
        raise ValueError("rebalance_every must be a positive integer.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite, nonnegative and less than 10000.")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    split = int(np.floor(len(returns) * train_fraction))
    if split < 2 or len(returns) - split < 2:
        raise ValueError("At least two training returns and two holdout returns are required.")
    if rolling_window is None:
        rolling_window = split
    if (
        isinstance(rolling_window, bool) or not isinstance(rolling_window, (int, np.integer))
        or not 2 <= rolling_window <= split
    ):
        raise ValueError("rolling_window must be an integer between two and the initial training length.")

    labels = prices.columns
    dates = prices.index
    execution_rows = range(split + 1, len(prices) - 1, rebalance_every)
    fits = {}

    def fit(start, end):
        # Only weights and messages are retained, not each dense covariance.
        if (start, end) not in fits:
            sample = returns.iloc[start:end]
            covariance = sample.cov() * periods_per_year
            diagonal = pd.DataFrame(np.diag(np.diag(covariance)), index=labels, columns=labels)
            covariance = (1 - shrinkage) * covariance + shrinkage * diagonal
            portfolios, _, _, messages = optimize(
                sample.mean() * periods_per_year, covariance, risk_free_rate, max_weight,
                frontier_points=2,
            )
            fallback = "Maximum Sharpe" not in portfolios
            targets = {
                name: portfolios[name if name in portfolios else "Minimum volatility"].weights.to_numpy()
                for name in PORTFOLIO_NAMES
            }
            fits[start, end] = (targets, fallback, messages)
        return fits[start, end]

    equity, allocations, holdings, trades, metrics, warnings = {}, {}, {}, {}, [], []
    daily_returns = returns.to_numpy()
    cost_rate = cost_bps / 10_000
    for method in METHOD_NAMES:
        schedule = {}
        rows = [split + 1] if method == "Buy and hold" else execution_rows
        for i in rows:
            end = split if method in ("Buy and hold", "Fixed rebalance") else i - 1
            start = max(0, end - rolling_window) if method == "Rolling window" else 0
            targets, fallback, messages = fit(start, end)
            schedule[i] = (targets, fallback, start, end)
            if fallback:
                warnings.append(
                    f"{dates[i].date()} · {method}: Maximum Sharpe uses Minimum volatility "
                    "because no feasible portfolio has positive expected excess return."
                )
            for message in messages:
                if not message.startswith("Maximum Sharpe omitted:"):
                    warning = f"{dates[i].date()} · {method}: {message}"
                    warnings.append(warning)

        for portfolio in PORTFOLIO_NAMES:
            key = f"{method} · {portfolio}"
            positions = np.zeros(len(labels))
            contribution = np.zeros(len(labels))
            exposure = np.zeros(len(labels))
            max_realized_weight = np.zeros(len(labels))
            cash, nav = 1.0, 1.0
            path, targets_used, trade_rows, trade_dates = [nav], [], [], []
            fallbacks = 0
            for i in range(split + 1, len(prices)):
                exposure += positions / nav
                profit = positions * daily_returns[i - 1]
                contribution += profit
                positions += profit
                nav = float(positions.sum() + cash)
                max_realized_weight = np.maximum(max_realized_weight, positions / nav)
                if i in schedule:
                    targets, fallback, start, end = schedule[i]
                    target = targets[portfolio]
                    after = _after_cost(nav, positions, target, cost_rate)
                    traded = float(np.abs(after * target - positions).sum())
                    trade_rows.append({
                        "turnover": traded / nav, "cost": nav - after,
                        "nav_before": nav, "nav_after": after,
                        "train_start": str(returns.index[start].date()),
                        "train_end": str(returns.index[end - 1].date()),
                    })
                    trade_dates.append(dates[i])
                    targets_used.append(target)
                    fallbacks += int(fallback and portfolio == "Maximum Sharpe")
                    positions, cash, nav = after * target, 0.0, after
                    max_realized_weight = np.maximum(max_realized_weight, positions / nav)
                path.append(nav)

            equity[key] = path
            allocations[key] = pd.DataFrame(targets_used, index=pd.DatetimeIndex(trade_dates), columns=labels)
            trades[key] = pd.DataFrame(trade_rows, index=allocations[key].index)
            holdings[key] = pd.DataFrame({
                "average_weight": exposure / (len(prices) - split - 1),
                "end_weight": positions / nav,
                "selection_frequency": (allocations[key] > 1e-6).mean().to_numpy(),
                "max_target_weight": allocations[key].max().to_numpy(),
                "max_realized_weight": max_realized_weight,
                "pnl_contribution": contribution,
            }, index=labels)
            values = pd.Series(path, dtype=float)
            metrics.append({
                "strategy": key, **performance_metrics(values, risk_free_rate, periods_per_year),
                "total_turnover": float(trades[key].turnover.sum()),
                "total_cost": float(trades[key].cost.sum()),
                "rebalance_count": len(trade_rows), "fallback_count": fallbacks,
            })

    settings = {
        "train_fraction": train_fraction, "risk_free_rate": risk_free_rate,
        "max_weight": max_weight, "shrinkage": shrinkage, "rebalance_every": int(rebalance_every),
        "rolling_window": int(rolling_window), "cost_bps": cost_bps, "split": split,
        "periods_per_year": periods_per_year,
        "trading_days_per_year": periods_per_year, "cash_interest_rate": 0.0,
        "train_start": str(returns.index[0].date()), "train_end": str(dates[split].date()),
        "test_start": str(dates[split + 1].date()), "test_end": str(dates[-1].date()),
        "initial_execution": str(dates[split + 1].date()),
        "execution": "Prior-close information; execute at the following close; first session in cash.",
        "cost_convention": "Basis points per absolute dollar bought or sold; entry included; no final liquidation.",
    }
    return BacktestStudy(
        pd.DataFrame(equity, index=dates[split:]), pd.DataFrame(metrics).set_index("strategy"),
        allocations, holdings, trades, warnings, settings,
    )
