"""One historical-mean stock portfolio with a later chronological replay."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .backtest import BacktestStudy, _after_cost
from .core import Portfolio, TRADING_DAYS, _solve, _validate_price_frame, optimize
from .metrics import performance_metrics, validate_periods_per_year


SELECTED = "Historical-mean portfolio"
BASELINE = "Equal-weight stocks"
MODEL_REVISION = "stock_mean_v2"
BENCHMARK_NAMES = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)"}
STALE_INTERVALS = 20


@dataclass
class SelectionStudy:
    test: BacktestStudy
    latest: dict[str, Portfolio]
    selected: str
    supported: bool
    reasons: list[str]
    settings: dict
    warnings: list[str]
    benchmark_prices: pd.DataFrame | None
    eligibility: pd.DataFrame


def split_stock_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Separate known benchmark ETFs from the stock candidates without a download."""
    prices = _validate_price_frame(prices)
    names = {column: str(column).strip().upper() for column in prices.columns}
    reference_columns = [column for column in prices if names[column] in BENCHMARK_NAMES]
    stocks = prices.drop(columns=reference_columns).copy()
    if stocks.empty:
        raise ValueError("Provide stock price columns as well as SPY or QQQ. The ETFs are comparison benchmarks, not portfolio holdings.")
    references = None
    if reference_columns:
        references = validate_selection_benchmarks(prices[reference_columns].rename(columns=names), prices.index)
    return stocks, references


def validate_selection_benchmarks(prices: pd.DataFrame, expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Accept either or both comparison ETFs, with exactly the stock price dates."""
    prices = _validate_price_frame(prices)
    prices.columns = [str(column).strip().upper() for column in prices.columns]
    if (not prices.columns.is_unique or not prices.index.equals(expected_dates)
            or not set(prices.columns).issubset(BENCHMARK_NAMES)):
        raise ValueError("Benchmark prices must contain SPY and/or QQQ with exactly the stock price dates.")
    return prices.loc[:, [symbol for symbol in BENCHMARK_NAMES if symbol in prices]].copy()


def selection_frontier(study: SelectionStudy, prices: pd.DataFrame, frontier_points: int = 30):
    """Return the latest fitted frontier, its stock weights, and same-date references.

    All rows use annual arithmetic means. The main model uses sample covariance.
    Explicit research shrinkage changes stock model risk, not ETF sample risk.
    This describes the fit period, not the returns of the later historical test.
    """
    prices = _validate_price_frame(prices)
    latest_date = study.settings["latest_as_of"]
    if str(prices.index[-1].date()) != latest_date:
        raise ValueError("Frontier prices must end on the saved latest fit date.")
    coverage = study.eligibility.loc[study.eligibility.fit_date == latest_date]
    labels = coverage.loc[coverage.eligible, "symbol"].tolist()
    returns = prices.loc[:, labels].pct_change(fill_method=None).iloc[1:]
    periods = study.settings["periods_per_year"]
    mean = returns.mean() * periods
    sample = returns.cov() * periods
    shrinkage = study.settings["shrinkage"]
    covariance = (1 - shrinkage) * sample + shrinkage * pd.DataFrame(
        np.diag(np.diag(sample)), index=labels, columns=labels,
    )
    _, frontier, weights, warnings = optimize(
        mean, covariance, risk_free_rate=study.settings["risk_free_rate"],
        max_weight=study.settings["max_weight"], frontier_points=frontier_points,
    )
    frontier = frontier.rename(columns={"expected_return": "fitted_mean", "volatility": "model_volatility"})
    frontier["historical_volatility"] = np.sqrt(np.maximum(
        np.einsum("ij,jk,ik->i", weights.to_numpy(), sample.to_numpy(), weights.to_numpy()), 0,
    ))
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)
    fitted = selection_fit_estimates(study, prices).iloc[-1]
    rows = [{"portfolio": SELECTED, **fitted[["fitted_mean", "model_volatility", "historical_volatility"]].to_dict()}]
    if study.benchmark_prices is not None:
        benchmarks = validate_selection_benchmarks(study.benchmark_prices, prices.index)
        benchmark_returns = benchmarks.pct_change(fill_method=None).iloc[1:]
        for symbol in benchmarks:
            volatility = float(benchmark_returns[symbol].std(ddof=1) * np.sqrt(periods))
            rows.append({"portfolio": BENCHMARK_NAMES[symbol],
                         "fitted_mean": float(benchmark_returns[symbol].mean() * periods),
                         "model_volatility": volatility, "historical_volatility": volatility})
    return frontier, weights, pd.DataFrame(rows).set_index("portfolio"), warnings


def selection_fit_estimates(study: SelectionStudy, prices: pd.DataFrame) -> pd.DataFrame:
    """Recover each saved fit's estimates from its weights and past prices only.

    Historical volatility uses the sample covariance before shrinkage. Neither
    volatility describes the later replay. No portfolio is solved again.
    """
    allocations = study.test.allocations[SELECTED]
    trades = study.test.trades[SELECTED]
    targets = [
        (pd.Timestamp(trades.loc[execution, "train_end"]), execution,
         "First historical fit" if number == 0 else "Historical refit", weights)
        for number, (execution, weights) in enumerate(allocations.iterrows())
    ]
    targets.append((prices.index[-1], pd.NaT, "Latest fit", study.latest[study.selected].weights))
    periods = study.settings["periods_per_year"]
    shrinkage = study.settings["shrinkage"]
    rows = []
    for fit_date, execution, phase, target in targets:
        returns = prices.loc[:fit_date].pct_change(fill_method=None).iloc[1:]
        weights = target.reindex(prices.columns, fill_value=0.0)
        portfolio_returns = returns @ weights
        raw_variance = float(portfolio_returns.var(ddof=1) * periods)
        diagonal_variance = float((returns.var(ddof=1) * periods) @ weights.pow(2))
        model_variance = (1 - shrinkage) * raw_variance + shrinkage * diagonal_variance
        rows.append({
            "fit_date": returns.index[-1], "phase": phase, "execution_date": execution,
            "observations": len(returns), "fitted_mean": float(portfolio_returns.mean() * periods),
            "model_volatility": float(np.sqrt(max(model_variance, 0))),
            "historical_volatility": float(np.sqrt(max(raw_variance, 0))),
            "max_weight": float(weights.max()),
        })
    return pd.DataFrame(rows).set_index("fit_date")


def _longest_unchanged(prices):
    values = prices.to_numpy()
    run = np.zeros(values.shape[1], dtype=int)
    longest = run.copy()
    for unchanged in values[1:] == values[:-1]:
        run = np.where(unchanged, run + 1, 0)
        longest = np.maximum(longest, run)
    return longest


def _fit_targets(prices, target_volatility, max_weight, shrinkage, periods_per_year):
    """Maximize historical mean within the cap and risk limit, using past prices.

    At a tied maximum mean, minimize variance. The secondary solve permits a
    normalized mean shortfall of 1e-8, checked within 2e-8 for numerical error.
    """
    longest = _longest_unchanged(prices)
    eligible = longest < STALE_INTERVALS
    labels = prices.columns[eligible]
    if len(labels) * max_weight < 1 - 1e-12:
        raise ValueError(
            f"{prices.index[-1].date()}: the {max_weight:.1%} position cap is infeasible for "
            f"{len(labels)} eligible stocks after the long unchanged-price screen. "
            "Supply more eligible stocks or explicitly change the cap."
        )
    returns = prices.loc[:, labels].pct_change(fill_method=None).iloc[1:]
    mean = returns.mean().to_numpy() * periods_per_year
    sample = returns.cov().to_numpy() * periods_per_year
    covariance = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    scale = float(np.max(np.abs(covariance))) or 1.0
    weights = cp.Variable(len(labels))
    budget = [weights >= 0, weights <= max_weight, cp.sum(weights) == 1]
    variance = cp.quad_form(weights, cp.psd_wrap(covariance / scale))
    constraints = budget.copy()
    # A long-only portfolio's variance cannot exceed the largest asset variance.
    # Omit this redundant bound to avoid huge ratios for near-constant returns.
    if target_volatility ** 2 < float(np.diag(covariance).max()):
        constraints.append(variance <= target_volatility ** 2 / scale)
    warnings = []

    def solve(problem):
        try:
            _solve(problem)
        except RuntimeError as exc:
            if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
                minimum = cp.Problem(cp.Minimize(variance / 2), budget)
                _solve(minimum)
                floor = float(np.sqrt(max(weights.value @ covariance @ weights.value, 0)))
                raise ValueError(
                    f"{prices.index[-1].date()}: requested annual volatility {target_volatility:.2%} "
                    f"is below the feasible minimum {floor:.2%} with the {max_weight:.1%} position cap."
                ) from exc
            raise
        if problem.status == cp.OPTIMAL_INACCURATE:
            warnings.append(f"{prices.index[-1].date()}: the selection solver reported limited numerical accuracy.")

    def checked():
        value = np.asarray(weights.value, dtype=float).reshape(-1)
        if (not np.isfinite(value).all() or abs(value.sum() - 1) > 1e-7
                or value.min() < -1e-7 or value.max() > max_weight + 1e-7):
            raise RuntimeError("Selection solver returned infeasible stock weights.")
        value = np.clip(value, 0, max_weight)
        value /= value.sum()
        volatility = float(np.sqrt(max(value @ covariance @ value, 0)))
        if volatility > target_volatility + 1e-7:
            raise RuntimeError("Selection solver exceeded the requested estimated volatility.")
        return pd.Series(value, index=labels).reindex(prices.columns, fill_value=0.0)

    scores = mean / (float(np.max(np.abs(mean))) or 1.0)
    objective = cp.Maximize(scores @ weights)
    solve(cp.Problem(objective, constraints))
    selected = checked()
    # Risk can only improve at the same maximum mean when the risk ceiling does
    # not prevent the capped, risk-unconstrained maximum. Check that face first.
    ordered = np.sort(scores)[::-1]
    greedy = np.clip(1 - max_weight * np.arange(len(labels)), 0, max_weight)
    highest_score = float(ordered @ greedy)
    tolerance = 1e-8
    marginal = ordered[np.flatnonzero(greedy > tolerance)[-1]]
    tied = np.abs(scores - marginal) <= tolerance
    tied_weight = 1 - max_weight * np.count_nonzero(scores > marginal + tolerance)
    if (tied.sum() > 1 and tolerance < tied_weight < max_weight * tied.sum() - tolerance
            and scores @ selected.loc[labels].to_numpy() >= highest_score - tolerance):
        # Minimum variance itself enforces the existing risk ceiling. Omitting
        # the redundant ceiling avoids a second solve at a tangential boundary.
        solve(cp.Problem(cp.Minimize(variance / 2), budget + [scores @ weights >= highest_score - tolerance]))
        selected = checked()
        if scores @ selected.loc[labels].to_numpy() < highest_score - 2 * tolerance:
            raise RuntimeError("Selection risk tie-break missed the maximum historical mean.")
    equal = pd.Series(1 / len(labels), index=labels).reindex(prices.columns, fill_value=0.0)
    full_mean = pd.Series(mean, index=labels).reindex(prices.columns, fill_value=0.0)
    full_covariance = pd.DataFrame(covariance, index=labels, columns=labels).reindex(
        index=prices.columns, columns=prices.columns, fill_value=0.0,
    )
    coverage = pd.DataFrame({
        "fit_date": str(prices.index[-1].date()), "symbol": prices.columns,
        "eligible": eligible, "max_unchanged_intervals": longest,
        "reason": np.where(eligible, "", f"At least {STALE_INTERVALS} consecutive unchanged price intervals in past data."),
    })
    if not eligible.all():
        warnings.append(
            f"{prices.index[-1].date()}: the long unchanged-price screen excluded "
            + ", ".join(prices.columns[~eligible]) + ". Repeated prices do not prove an invalid quote."
        )
    return selected, equal, full_mean, full_covariance, coverage, warnings


def _simulate(prices, benchmark_prices, start, end, fit, settings):
    """Replay the stock rule and comparators with the same execution and cost rules."""
    dates = prices.index
    executions = list(range(start + 1, end, settings["rebalance_every"]))
    schedules = {i: fit(i - 1) for i in executions}
    specifications = [(SELECTED, prices, 0), (BASELINE, prices, 1)]
    if benchmark_prices is not None:
        for symbol in benchmark_prices:
            specifications.append((BENCHMARK_NAMES[symbol], benchmark_prices[[symbol]], None))
    equity, allocations, trades, holdings, metrics = {}, {}, {}, {}, []
    for name, asset_prices, target_index in specifications:
        labels = asset_prices.columns
        returns = asset_prices.pct_change(fill_method=None).iloc[1:]
        daily = returns.to_numpy()
        positions = np.zeros(len(labels))
        cash, nav = 1.0, 1.0
        contribution = np.zeros(len(labels) + 1)
        exposure = np.zeros(len(labels) + 1)
        maximum_weight = np.zeros(len(labels) + 1)
        path, target_rows, trade_rows, trade_dates = [nav], [], [], []
        trade_at = executions if target_index is not None else executions[:1]
        for i in range(start + 1, end + 1):
            exposure += np.r_[positions, cash] / nav
            profit = positions * daily[i - 1]
            contribution[:-1] += profit
            positions += profit
            nav = float(positions.sum() + cash)
            maximum_weight = np.maximum(maximum_weight, np.r_[positions, cash] / nav)
            if i in trade_at:
                if target_index is not None:
                    coverage = schedules[i][-2]
                    excluded = ~coverage.eligible.to_numpy()
                    if np.any((positions > 1e-8 * nav) & excluded):
                        affected = ", ".join(labels[(positions > 1e-8 * nav) & excluded])
                        raise ValueError(
                            f"{dates[i].date()}: held stocks {affected} triggered the long unchanged-price screen. "
                            "A sale at the repeated quote is not verified. Supply verified prices before continuing."
                        )
                    target = schedules[i][target_index].to_numpy()
                else:
                    target = np.ones(1)
                before = nav
                after = _after_cost(before, positions, target, settings["cost_bps"] / 10_000)
                traded = float(np.abs(after * target - positions).sum())
                trade_rows.append({
                    "turnover": traded / before, "cost": before - after,
                    "nav_before": before, "nav_after": after,
                    "train_start": str(dates[1].date()), "train_end": str(dates[i - 1].date()),
                })
                trade_dates.append(dates[i])
                target_rows.append(np.r_[target, max(0.0, 1 - target.sum())])
                positions, cash, nav = after * target, after * max(0.0, 1 - target.sum()), after
                maximum_weight = np.maximum(maximum_weight, np.r_[positions, cash] / nav)
            path.append(nav)
        columns = list(labels) + ["CASH"]
        equity[name] = path
        allocations[name] = pd.DataFrame(target_rows, index=pd.DatetimeIndex(trade_dates), columns=columns)
        trades[name] = pd.DataFrame(trade_rows, index=allocations[name].index)
        holdings[name] = pd.DataFrame({
            "average_weight": exposure / (end - start), "end_weight": np.r_[positions, cash] / nav,
            "selection_frequency": (allocations[name] > 1e-6).mean().to_numpy(),
            "max_target_weight": allocations[name].max().to_numpy(), "max_realized_weight": maximum_weight,
            "pnl_contribution": contribution,
        }, index=columns)
        values = pd.Series(path, dtype=float)
        metrics.append({
            "strategy": name, **performance_metrics(values, settings["risk_free_rate"], settings["periods_per_year"]),
            "annual_mean": float(values.pct_change(fill_method=None).iloc[1:].mean() * settings["periods_per_year"]),
            "total_turnover": float(trades[name].turnover.sum()), "total_cost": float(trades[name].cost.sum()),
            "rebalance_count": len(trades[name]), "fallback_count": 0,
        })
    return BacktestStudy(
        pd.DataFrame(equity, index=dates[start:end + 1]), pd.DataFrame(metrics).set_index("strategy"),
        allocations, holdings, trades,
        list(dict.fromkeys(message for result in schedules.values() for message in result[-1])), {
            **settings, "split": start, "train_start": str(dates[1].date()),
            "train_end": str(dates[start].date()), "test_start": str(dates[start + 1].date()),
            "test_end": str(dates[end].date()), "initial_execution": str(dates[start + 1].date()),
        },
    )


def run_selection(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
    target_volatility: float = 0.2,
    max_weight: float = 0.1,
    train_fraction: float = 0.7,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
    shrinkage: float = 0.0,
    periods_per_year: float = TRADING_DAYS,
    risk_free_rate: float = 0.02,
) -> SelectionStudy:
    """Fit one historical-mean rule and replay it on later dates.

    Every fit uses prior closes. The first interval is zero-interest cash and
    execution occurs at the next close. Each scheduled fit expands the history.
    Equal-weight stocks and raw SPY/QQQ funds provide comparison paths.

    The volatility target constrains estimated risk, not realized losses or risk.
    This historical replay is not a claim of prospective validation.
    """
    prices = _validate_price_frame(prices)
    if any(str(column).strip().upper() in BENCHMARK_NAMES for column in prices.columns):
        raise ValueError("SPY and QQQ are comparison benchmarks, not stock holdings. Use split_stock_prices before run_selection.")
    if "CASH" in prices.columns:
        raise ValueError("CASH is reserved for the zero-interest cash allocation.")
    periods_per_year = validate_periods_per_year(periods_per_year)
    for name, value in (("target_volatility", target_volatility), ("max_weight", max_weight)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")
    if max_weight > 1:
        raise ValueError("max_weight must be at most one.")
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one.")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite, nonnegative and less than 10000.")
    if isinstance(rebalance_every, (bool, np.bool_)) or not isinstance(rebalance_every, (int, np.integer)) or rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer.")
    if benchmark_prices is not None:
        benchmark_prices = validate_selection_benchmarks(benchmark_prices, prices.index)
    count = len(prices) - 1
    initial_end = int(np.floor(count * train_fraction))
    if min(initial_end, count - initial_end) < 2:
        raise ValueError("Selection needs at least two returns in each initial training and later test period.")
    settings = {
        "model_revision": MODEL_REVISION,
        "target_volatility": float(target_volatility), "risk_target": float(target_volatility),
        "max_weight": float(max_weight), "train_fraction": float(train_fraction),
        "rebalance_every": int(rebalance_every), "cost_bps": float(cost_bps),
        "shrinkage": float(shrinkage), "periods_per_year": periods_per_year,
        "risk_free_rate": float(risk_free_rate), "cash_interest_rate": 0.0,
        "model_objective": "Maximum historical arithmetic mean return within the position cap and estimated volatility ceiling.",
        "evidence_floor_returns": int(np.ceil(periods_per_year)),
        "evidence_floor_policy": "At least one declared year of later test returns. This descriptive minimum does not measure statistical significance.",
        "price_start": str(prices.index[0].date()), "price_end": str(prices.index[-1].date()),
        "initial_training_start": str(prices.index[1].date()),
        "initial_training_end": str(prices.index[initial_end].date()),
        "test_start": str(prices.index[initial_end + 1].date()), "test_end": str(prices.index[-1].date()),
        "initial_training_returns": initial_end, "test_returns": count - initial_end,
        "execution": "Prior-close information; execute at the following close; first interval in zero-interest cash.",
        "cost_convention": "Basis points per absolute dollar bought or sold; cash transfers excluded; no final liquidation.",
        "risk_constraint": "Annual estimated stock volatility ceiling; realized volatility can exceed the target.",
        "baseline": "Equal weights across eligible stocks, reset on the same schedule without a volatility constraint.",
        "comparisons": "Equal-weight stocks and raw SPY/QQQ buy-and-hold funds. Each uses the same initial delay and trading costs.",
        "screen": f"Exclude stocks with at least {STALE_INTERVALS} consecutive unchanged price intervals in past fit data only.",
        "test_role": "Later historical replay with expanding past-data fits. The researcher can already know these dates.",
    }
    fits = {}

    def fit(end):
        if end not in fits:
            fits[end] = _fit_targets(prices.iloc[:end + 1], target_volatility, max_weight, shrinkage, periods_per_year)
        return fits[end]

    test = _simulate(prices, benchmark_prices, initial_end, count, fit, settings)
    reasons = []
    missing_markets = [symbol for symbol in BENCHMARK_NAMES if benchmark_prices is None or symbol not in benchmark_prices]
    if missing_markets:
        reasons.append(
            "Market benchmark prices are unavailable for " + ", ".join(missing_markets)
            + ". The historical check needs both SPY and QQQ."
        )
    if count - initial_end < settings["evidence_floor_returns"]:
        reasons.append("The later test has less than one declared year of return intervals. The evidence floor is not met.")
    candidate = test.metrics.loc[SELECTED]
    if candidate.volatility > target_volatility + 1e-10:
        reasons.append("Realized annual volatility in the later test exceeded the requested target.")
    for name in test.metrics.index.drop(SELECTED):
        comparator = test.metrics.loc[name]
        if comparator.cagr > candidate.cagr + 1e-10 and comparator.volatility <= candidate.volatility + 1e-10:
            reasons.append(f"{name} had higher compound growth and no more realized volatility in the later test.")
    weights, _, mean, covariance, _, _ = fit(count)
    expected = float(mean @ weights)
    volatility = float(np.sqrt(max(weights @ covariance @ weights, 0)))
    latest = {SELECTED: Portfolio(
        SELECTED, weights.rename(SELECTED), expected, volatility,
        (expected - risk_free_rate) / volatility if volatility > 1e-12 else float("nan"),
    )}
    warnings = list(dict.fromkeys(message for result in fits.values() for message in result[-1]))
    warnings.append("Historical means and estimated volatility can change. A passed historical check does not guarantee future outperformance.")
    eligibility = pd.concat([result[-2] for _, result in sorted(fits.items())], ignore_index=True)
    settings.update({"fit_dates": [str(prices.index[end].date()) for end in sorted(fits)],
                     "latest_as_of": str(prices.index[-1].date())})
    test.settings.update(settings)
    return SelectionStudy(test, latest, SELECTED, not reasons, reasons, settings, warnings, benchmark_prices, eligibility)
