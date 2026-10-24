"""Fixed index-holding hypotheses with monthly signals and next-close trades."""

import numpy as np
import pandas as pd

from .backtest import BacktestStudy, _after_cost
from .core import TRADING_DAYS, _validate_price_frame
from .metrics import performance_metrics
from .selection import STALE_INTERVALS, _longest_unchanged


STOCK_RULES = ("equal_weight", "inverse_volatility", "low_volatility_100",
               "momentum_50", "risk_adjusted_momentum_50")
BENCHMARKS = ("SPY", "QQQ")
FUNDS = (*BENCHMARKS, "SPMO", "MTUM", "SPLV", "USMV", "RSP", "QUAL")
TREND_RULE = "qqq_trend10month"
MODEL_REVISION = "index_research_v1"


def _signals(prices, start):
    """A completed month needs an observed next-month execution date."""
    months = prices.index.to_period("M")
    ends = np.flatnonzero(months[:-1] != months[1:])
    rows = [(int(row), position) for position, row in enumerate(ends)
            if position >= 12 and row >= 252 and row + 1 < len(prices) - 1
            and (start is None or prices.index[row + 1] >= pd.Timestamp(start))]
    if not rows:
        raise ValueError("Need 252 prior returns, 13 month ends, and a return after the first execution.")
    return ends, rows


def _settings(cost_bps, target_volatility, start):
    if isinstance(cost_bps, (bool, np.bool_)) or not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite, nonnegative, and less than 10000.")
    if target_volatility is not None and (
        isinstance(target_volatility, (bool, np.bool_)) or not np.isfinite(target_volatility) or target_volatility <= 0
    ):
        raise ValueError("target_volatility must be finite and positive, or None.")
    return {
        "model_revision": MODEL_REVISION, "cost_bps": float(cost_bps),
        "target_volatility": target_volatility, "periods_per_year": TRADING_DAYS,
        "cash_interest_rate": 0.0, "risk_free_rate": 0.0,
        "requested_start": None if start is None else str(pd.Timestamp(start).date()),
        "execution": "Calendar month-end signal; next observed close execution; initial interval in cash.",
        "cost_convention": "Fee per amount bought or sold; entry included; no terminal sale or unused final trade.",
        "warmup": "252 prior returns and 13 completed month ends for every path.",
        "volatility_overlay": "126 prior returns of the desired risky weights; sample volatility; exposure capped at one.",
        "momentum": "Month-end m-1 divided by m-12 minus one; skip the latest month.",
        "risk_adjusted_momentum": "Momentum divided by sample daily volatility over the same m-12 to m-1 window.",
        "stock_screen": "Exclude 20 consecutive unchanged price intervals anywhere in the available past.",
        "selection": "Fixed research hypotheses, not an automatic recommendation or future-return guarantee.",
    }


def _capped_inverse_volatility(volatility):
    """Redistribute inverse-volatility weights proportionally below a 10% cap."""
    if len(volatility) < 10 or not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Capped inverse volatility needs at least ten finite, positive volatility estimates.")
    scores = 1 / volatility.to_numpy()
    weights = np.zeros(len(scores))
    free = np.ones(len(scores), dtype=bool)
    while free.any():
        proposed = (1 - weights.sum()) * scores[free] / scores[free].sum()
        capped = proposed > .1
        if not capped.any():
            weights[free] = proposed
            break
        indices = np.flatnonzero(free)[capped]
        weights[indices] = .1
        free[indices] = False
    return pd.Series(weights, index=volatility.index)


def _stock_targets(prices, signal, ends, month):
    past = prices.iloc[:signal + 1]
    eligible = _longest_unchanged(past) < STALE_INTERVALS
    labels = prices.columns[eligible].sort_values()
    if len(labels) < 100:
        raise ValueError(f"{past.index[-1].date()}: stock rules need at least 100 eligible stocks.")
    returns = past[labels].pct_change(fill_method=None).iloc[1:]
    volatility = returns.iloc[-252:].std(ddof=1)
    begin, end = int(ends[month - 12]), int(ends[month - 1])
    momentum = prices.iloc[end][labels] / prices.iloc[begin][labels] - 1
    signal_volatility = past[labels].iloc[begin:end + 1].pct_change(fill_method=None).iloc[1:].std(ddof=1)
    if (signal_volatility <= 0).any() or not np.isfinite(signal_volatility).all():
        raise ValueError("Risk-adjusted momentum needs finite, positive volatility over its signal window.")
    low = volatility.sort_values(kind="stable").index[:100]
    leaders = momentum.sort_values(ascending=False, kind="stable").index[:50]
    adjusted = (momentum / signal_volatility).sort_values(ascending=False, kind="stable").index[:50]
    targets = {
        "equal_weight": pd.Series(1 / len(labels), index=labels),
        "inverse_volatility": _capped_inverse_volatility(volatility),
        "low_volatility_100": _capped_inverse_volatility(volatility[low]),
        "momentum_50": pd.Series(.02, index=leaders),
        "risk_adjusted_momentum_50": pd.Series(.02, index=adjusted),
    }
    return {name: weights.reindex(prices.columns, fill_value=0.).to_numpy()
            for name, weights in targets.items()}, eligible


def _overlay(prices, signal, weights, target_volatility):
    if target_volatility is None or not weights.any():
        return weights.copy()
    returns = prices.iloc[signal - 126:signal + 1].pct_change(fill_method=None).iloc[1:]
    volatility = float((returns.to_numpy() @ weights).std(ddof=1) * np.sqrt(TRADING_DAYS))
    fraction = min(1., target_volatility / volatility) if volatility > 0 else 1.
    return weights * fraction


def _replay(sources, schedules, settings):
    dates = next(iter(sources.values())).index
    first = min(min(schedule) for schedule in schedules.values()) - 1
    equity, allocations, trades, holdings, metrics = {}, {}, {}, {}, []
    for name, prices in sources.items():
        schedule = schedules[name]
        returns = prices.pct_change(fill_method=None).iloc[1:].to_numpy()
        positions = np.zeros(len(prices.columns))
        cash, nav = 1., 1.
        contribution = np.zeros(len(positions) + 1)
        exposure = contribution.copy()
        maximum = contribution.copy()
        path, targets, records, executed = [nav], [], [], []
        for row in range(first + 1, len(prices)):
            exposure += np.r_[positions, cash] / nav
            profit = positions * returns[row - 1]
            contribution[:-1] += profit
            positions += profit
            nav = float(positions.sum() + cash)
            maximum = np.maximum(maximum, np.r_[positions, cash] / nav)
            if row in schedule:
                signal, target, eligible = schedule[row]
                if eligible is not None and np.any((positions > 1e-8 * nav) & ~eligible):
                    raise ValueError(f"{dates[row].date()}: a held stock triggered the unchanged-price screen. "
                                     "Its sale price is unverified.")
                after = _after_cost(nav, positions, target, settings["cost_bps"] / 10_000)
                amount = float(np.abs(after * target - positions).sum())
                records.append({"turnover": amount / nav, "cost": nav - after,
                                "nav_before": nav, "nav_after": after,
                                "train_start": str(dates[0].date()), "train_end": str(dates[signal].date()),
                                "signal_date": dates[signal]})
                executed.append(dates[row])
                targets.append(np.r_[target, max(0., 1 - target.sum())])
                positions, cash, nav = after * target, after * max(0., 1 - target.sum()), after
                maximum = np.maximum(maximum, np.r_[positions, cash] / nav)
            path.append(nav)
        columns = [*prices.columns, "CASH"]
        equity[name] = path
        allocations[name] = pd.DataFrame(targets, index=pd.DatetimeIndex(executed), columns=columns)
        trades[name] = pd.DataFrame(records, index=allocations[name].index)
        holdings[name] = pd.DataFrame({
            "average_weight": exposure / (len(prices) - first - 1),
            "end_weight": np.r_[positions, cash] / nav,
            "selection_frequency": (allocations[name] > 1e-6).mean().to_numpy(),
            "max_target_weight": allocations[name].max().to_numpy(),
            "max_realized_weight": maximum, "pnl_contribution": contribution,
        }, index=columns)
        values = pd.Series(path)
        metrics.append({"strategy": name, **performance_metrics(values, 0., TRADING_DAYS),
                        "annual_mean": float(values.pct_change().iloc[1:].mean() * TRADING_DAYS),
                        "total_turnover": float(trades[name].turnover.sum()),
                        "total_cost": float(trades[name].cost.sum()), "rebalance_count": len(executed),
                        "fallback_count": 0})
    settings.update({"baseline_date": str(dates[first].date()), "test_start": str(dates[first + 1].date()),
                     "test_end": str(dates[-1].date()), "initial_execution": str(dates[first + 1].date())})
    return BacktestStudy(pd.DataFrame(equity, index=dates[first:]), pd.DataFrame(metrics).set_index("strategy"),
                         allocations, holdings, trades, [], settings)


def run_index_rules(prices, benchmarks, cost_bps=10., target_volatility=None, start=None):
    """Replay five fixed stock rules and SPY/QQQ on the same complete calendar.

    start is the earliest allowed execution date. All paths start in cash at
    the first eligible preceding month-end. The stock universe is fixed input,
    not reconstructed historical membership. At least 100 stocks must qualify.
    """
    prices, benchmarks = _validate_price_frame(prices), _validate_price_frame(benchmarks)
    if "CASH" in prices or not all(isinstance(label, str) for label in prices):
        raise ValueError("Stock names must be strings and cannot use the reserved CASH name.")
    if set(benchmarks) != set(BENCHMARKS) or not prices.index.equals(benchmarks.index):
        raise ValueError("Benchmarks must contain SPY and QQQ on exactly the stock dates.")
    settings = _settings(cost_bps, target_volatility, start)
    ends, signals = _signals(prices, start)
    sources = {name: prices for name in STOCK_RULES}
    sources.update({name: benchmarks[[name]] for name in BENCHMARKS})
    schedules = {name: {} for name in sources}
    for number, (signal, month) in enumerate(signals):
        targets, eligible = _stock_targets(prices, signal, ends, month)
        for name in STOCK_RULES:
            schedules[name][signal + 1] = (signal, _overlay(prices, signal, targets[name], target_volatility), eligible)
        if number == 0 or target_volatility is not None:
            for name in BENCHMARKS:
                schedules[name][signal + 1] = (signal, _overlay(sources[name], signal, np.ones(1), target_volatility), None)
    settings["universe"] = "Fixed complete-history stock universe; no historical membership reconstruction."
    return _replay(sources, schedules, settings)


def run_fund_rules(prices, cost_bps=10., target_volatility=None, start=None):
    """Replay the fixed eight funds and QQQ above its ten-month moving average.

    Raw fund paths buy once. The trend rule and volatility overlays reset at
    monthly execution dates. Cash earns zero. Fund expenses stay in the input
    adjusted-price returns, with the specified wrapper trade fees added.
    """
    prices = _validate_price_frame(prices)
    if set(prices) != set(FUNDS):
        raise ValueError("Fund prices must contain SPY, QQQ, SPMO, MTUM, SPLV, USMV, RSP, and QUAL.")
    settings = _settings(cost_bps, target_volatility, start)
    ends, signals = _signals(prices, start)
    sources = {name: prices[[name]] for name in FUNDS}
    sources[TREND_RULE] = prices[["QQQ"]]
    schedules = {name: {} for name in sources}
    for number, (signal, month) in enumerate(signals):
        if number == 0 or target_volatility is not None:
            for name in FUNDS:
                schedules[name][signal + 1] = (signal, _overlay(sources[name], signal, np.ones(1), target_volatility), None)
        months = prices.QQQ.iloc[ends[month - 9:month + 1]]
        target = np.array([float(prices.QQQ.iloc[signal] > months.mean())])
        schedules[TREND_RULE][signal + 1] = (signal, _overlay(sources[TREND_RULE], signal, target, target_volatility), None)
    settings["trend_rule"] = "QQQ only when its completed month-end price exceeds the average of the last ten month ends."
    return _replay(sources, schedules, settings)
