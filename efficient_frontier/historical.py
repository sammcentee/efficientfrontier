"""Rolling portfolio tests with dated membership and explicit price gaps."""

import csv
from dataclasses import dataclass
from io import StringIO
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestStudy, PORTFOLIO_NAMES, _after_cost
from .core import TRADING_DAYS, optimize
from .data import _normalize_price_dates
from .membership import MembershipHistory, load_membership_csv
from .metrics import performance_metrics, validate_periods_per_year


@dataclass
class HistoricalStudy(BacktestStudy):
    coverage: pd.DataFrame
    cash_weights: pd.DataFrame


def validate_historical_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep missing observations. Validate each supplied price without filling gaps."""
    if not isinstance(prices, pd.DataFrame) or len(prices) < 5 or not len(prices.columns):
        raise ValueError("Provide a price table with at least five dates and one asset.")
    result = prices.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if any(not name for name in result.columns) or len({name.casefold() for name in result.columns}) != len(result.columns):
        raise ValueError("Price column names must be nonempty and unique.")
    result.index = _normalize_price_dates(result.index)
    try:
        if (
            any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in result.dtypes)
            or any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
                   for value in result.select_dtypes(include="object", exclude="str").to_numpy().flat)
        ):
            raise ValueError("Boolean or complex prices are invalid.")
        result = result.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Prices must be numeric values or blank cells.") from exc
    values = result.to_numpy()
    supplied = values[~np.isnan(values)]
    if not np.isfinite(supplied).all() or (supplied <= 0).any():
        raise ValueError("Each supplied price must be finite and strictly positive.")
    return result.sort_index()


def load_historical_prices(source) -> pd.DataFrame:
    """Read Date,ASSET1,ASSET2,... with blank cells for unavailable prices."""
    try:
        content = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        reader = csv.reader(StringIO(content.lstrip("\ufeff")), strict=True)
        headers = [name.strip() for name in next(reader)]
        rows = list(reader)
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError("Provide a nonempty UTF-8 price CSV with Date as the first column.") from exc
    if not headers or headers[0].casefold() != "date":
        raise ValueError("The first price CSV column must be named Date.")
    if any(not name for name in headers) or len({name.casefold() for name in headers}) != len(headers):
        raise ValueError("Price CSV column headers must be nonempty and unique.")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Every price CSV row must have the same number of columns as its header.")
    table = pd.DataFrame(rows, columns=headers).set_index(headers[0]).replace(r"^\s*$", np.nan, regex=True)
    return validate_historical_prices(table)


def run_historical_backtests(
    prices: pd.DataFrame,
    membership: MembershipHistory,
    *,
    train_fraction: float = .7,
    rolling_window: int | None = 126,
    rebalance_every: int = 21,
    risk_free_rate: float = .02,
    max_weight: float = 1.,
    shrinkage: float = .1,
    cost_bps: float = 10.,
    periods_per_year: float = TRADING_DAYS,
) -> HistoricalStudy:
    """Use the execution-date roster and a complete window through the prior close.

    Former members leave at the next scheduled trade. Missing prices for held
    assets or new targets stop the test. An empty or cap-infeasible universe
    moves to cash. Cash earns zero interest. No final liquidation is assumed.
    """
    prices = validate_historical_prices(prices)
    periods_per_year = validate_periods_per_year(periods_per_year)
    for name, value in (("train_fraction", train_fraction), ("risk_free_rate", risk_free_rate),
                        ("max_weight", max_weight), ("shrinkage", shrinkage), ("cost_bps", cost_bps)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be greater than zero and at most one.")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one.")
    if not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be nonnegative and less than 10000.")
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, (int, np.integer)) or rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer.")
    split = int(np.floor((len(prices) - 1) * train_fraction))
    if split < 2 or len(prices) - 1 - split < 2:
        raise ValueError("At least two training returns and two holdout returns are required.")
    if rolling_window is None:
        rolling_window = split
    if (isinstance(rolling_window, bool) or not isinstance(rolling_window, (int, np.integer))
            or not 2 <= rolling_window <= split):
        raise ValueError("rolling_window must be an integer between two and the initial training length.")
    dates = prices.index
    membership.at(dates[split])
    membership.at(dates[-1])
    # Include missing members in the exported holdings and target tables too.
    labels = pd.Index(list(dict.fromkeys([*prices.columns, *membership.symbols])))
    values = prices.reindex(columns=labels).to_numpy()
    strategies = [f"Rolling window · {name}" for name in PORTFOLIO_NAMES]
    positions = np.zeros((len(strategies), len(labels)))
    contributions = np.zeros_like(positions)
    exposure = np.zeros_like(positions)
    maximum_weights = np.zeros_like(positions)
    cash, nav = np.ones(len(strategies)), np.ones(len(strategies))
    paths, cash_paths = [nav.copy()], [cash.copy()]
    allocation_rows, trade_rows, trade_dates, coverage = [], [], [], []
    warnings = list(membership.limitations)
    fallbacks = np.zeros(len(strategies), dtype=int)
    schedule = range(split + 1, len(prices) - 1, rebalance_every)
    for i in range(split + 1, len(prices)):
        exposure += positions / nav[:, None]
        held = (positions > 0).any(axis=0)
        unavailable = held & (~np.isfinite(values[i]) | ~np.isfinite(values[i - 1]))
        if unavailable.any():
            raise ValueError(f"Missing price for held asset on {dates[i].date()}: {', '.join(labels[unavailable])}. "
                             "Supply verified prices or corporate-action proceeds; prices are never filled.")
        profit = np.zeros_like(positions)
        profit[:, held] = positions[:, held] * (values[i, held] / values[i - 1, held] - 1)
        contributions += profit
        positions += profit
        nav = positions.sum(axis=1) + cash
        maximum_weights = np.maximum(maximum_weights, positions / nav[:, None])
        if i in schedule:
            sample_prices = prices.iloc[i - rolling_window - 1:i]
            roster = membership.at(dates[i])
            hazards = membership.identity_exclusions(sample_prices.index[0], through=dates[i])
            eligible = []
            for symbol in roster:
                if symbol in hazards:
                    reason = hazards[symbol]
                elif symbol not in prices:
                    reason = "No price column for this member."
                elif sample_prices[symbol].isna().any():
                    reason = f"Need {rolling_window + 1} complete prior prices; no gap filling."
                else:
                    reason = ""
                    eligible.append(symbol)
                coverage.append({"date": str(dates[i].date()), "symbol": symbol,
                                 "status": "excluded" if reason else "eligible", "reason": reason})
            targets = np.zeros_like(positions)
            fallback = False
            if not eligible or len(eligible) * max_weight < 1 - 1e-12:
                warnings.append(f"{dates[i].date()}: {len(eligible)} eligible members cannot meet the holding cap. "
                                "All strategies move to cash until the next scheduled trade.")
            else:
                sample = sample_prices[eligible].pct_change(fill_method=None).iloc[1:]
                covariance = sample.cov() * periods_per_year
                diagonal = pd.DataFrame(np.diag(np.diag(covariance)), index=eligible, columns=eligible)
                covariance = (1 - shrinkage) * covariance + shrinkage * diagonal
                portfolios, _, _, messages = optimize(sample.mean() * periods_per_year, covariance,
                                                      risk_free_rate, max_weight, frontier_points=1)
                fallback = "Maximum Sharpe" not in portfolios
                for row, name in enumerate(PORTFOLIO_NAMES):
                    chosen = portfolios[name if name in portfolios else "Minimum volatility"]
                    targets[row] = chosen.weights.reindex(labels, fill_value=0).to_numpy()
                if fallback:
                    warnings.append(f"{dates[i].date()}: Maximum Sharpe uses Minimum volatility because no feasible "
                                    "portfolio has positive expected excess return.")
                warnings.extend(f"{dates[i].date()}: {message}" for message in messages
                                if not message.startswith("Maximum Sharpe omitted:"))
            required = (targets > 0).any(axis=0)
            missing = required & ~np.isfinite(values[i])
            if missing.any():
                raise ValueError(f"Missing execution price on {dates[i].date()}: {', '.join(labels[missing])}. "
                                 "Targets use prior prices; a missing execution price cannot remove a selected member.")
            event = []
            for row in range(len(strategies)):
                after = _after_cost(nav[row], positions[row], targets[row], cost_bps / 10_000)
                traded = float(np.abs(after * targets[row] - positions[row]).sum())
                cash[row] = after if not targets[row].any() else 0.
                event.append({"turnover": traded / nav[row], "cost": nav[row] - after,
                              "nav_before": nav[row], "nav_after": after,
                              "train_start": str(sample_prices.index[1].date()),
                              "train_end": str(sample_prices.index[-1].date()),
                              "expected_members": len(roster), "eligible_members": len(eligible),
                              "cash_weight": cash[row] / after})
                positions[row] = after * targets[row]
                nav[row] = after
            fallbacks[1] += int(fallback)
            maximum_weights = np.maximum(maximum_weights, positions / nav[:, None])
            allocation_rows.append(targets)
            trade_rows.append(event)
            trade_dates.append(dates[i])
        paths.append(nav.copy())
        cash_paths.append(cash / nav)

    equity = pd.DataFrame(paths, index=dates[split:], columns=strategies)
    cash_weights = pd.DataFrame(cash_paths, index=equity.index, columns=strategies)
    allocations, trades, holdings, metrics = {}, {}, {}, []
    for row, key in enumerate(strategies):
        allocations[key] = pd.DataFrame([event[row] for event in allocation_rows],
                                        index=pd.DatetimeIndex(trade_dates, name="Date"), columns=labels)
        trades[key] = pd.DataFrame([event[row] for event in trade_rows], index=allocations[key].index)
        holdings[key] = pd.DataFrame({
            "average_weight": exposure[row] / (len(prices) - split - 1),
            "end_weight": positions[row] / nav[row],
            "selection_frequency": (allocations[key] > 1e-6).mean().to_numpy(),
            "max_target_weight": allocations[key].max().to_numpy(),
            "max_realized_weight": maximum_weights[row], "pnl_contribution": contributions[row],
        }, index=labels)
        metrics.append({"strategy": key, **performance_metrics(equity[key], risk_free_rate, periods_per_year),
                        "total_turnover": float(trades[key].turnover.sum()), "total_cost": float(trades[key].cost.sum()),
                        "rebalance_count": len(trade_dates), "fallback_count": int(fallbacks[row])})
    coverage = pd.DataFrame(coverage)
    excluded_dates = coverage.loc[coverage.status == "excluded", "date"].nunique()
    if excluded_dates:
        warnings.append(f"{excluded_dates} allocation dates exclude at least one historical member. "
                        "Review coverage.csv. Incomplete prices and membership records can still cause survivorship bias.")
    settings = {
        "historical_membership": True, "train_fraction": train_fraction, "split": split,
        "rolling_window": int(rolling_window), "rebalance_every": int(rebalance_every),
        "risk_free_rate": risk_free_rate, "max_weight": max_weight, "shrinkage": shrinkage,
        "cost_bps": cost_bps, "periods_per_year": periods_per_year, "cash_interest_rate": 0.,
        "train_start": str(dates[split - rolling_window + 1].date()), "train_end": str(dates[split].date()),
        "test_start": str(dates[split + 1].date()), "test_end": str(dates[-1].date()),
        "initial_execution": str(dates[split + 1].date()), "include_profiles": False,
        "execution": "Use the effective roster on each trade date; fit through the prior close; execute at the next close. "
                     "The first evaluation session is cash. Membership publication timestamps are not verified.",
        "departure_policy": "Sell former members at the next scheduled trade, not on the membership change date.",
        "eligibility": "Require a complete prior price window and no known identity ambiguity. Never inspect future prices for eligibility.",
        "cash_policy": "Move fully to zero-interest cash when eligible members cannot meet the holding cap.",
        "cost_convention": "Basis points per absolute dollar bought or sold; entry included; no final liquidation.",
        "membership": membership.metadata,
    }
    return HistoricalStudy(equity, pd.DataFrame(metrics).set_index("strategy"), allocations, holdings, trades,
                           warnings, settings, coverage, cash_weights)


def historical_demo() -> tuple[pd.DataFrame, MembershipHistory]:
    """A reproducible synthetic study with entry, departure, reentry and a short price history."""
    dates = pd.bdate_range("2020-01-02", periods=360, name="Date")
    rng = np.random.default_rng(1942)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0004, .012, (len(dates), 4)), axis=0)),
                          index=dates, columns=["ALPHA", "BETA", "GAMMA", "NEWCO"])
    prices.loc[dates[:170], "NEWCO"] = np.nan
    snapshots = pd.DataFrame({"date": [dates[0], dates[265], dates[286], dates[310]],
                              "tickers": ["ALPHA,BETA,GAMMA", "ALPHA,GAMMA,NEWCO", "ALPHA,BETA,GAMMA,NEWCO",
                                          "ALPHA,BETA,NEWCO"]})
    membership = load_membership_csv(StringIO(snapshots.to_csv(index=False)), coverage_end=dates[-1].date().isoformat(),
                                     source="Synthetic demo with dated membership changes")
    return prices, membership
