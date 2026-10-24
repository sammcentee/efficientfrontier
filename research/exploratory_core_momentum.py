"""One fixed 75% SPY / 25% momentum hypothesis, declared after earlier results."""

import numpy as np
import pandas as pd

from efficient_frontier.backtest import BacktestStudy
from efficient_frontier.core import _validate_price_frame
from efficient_frontier.index_rules import BENCHMARKS, _overlay, _replay, _settings, _signals
from efficient_frontier.selection import STALE_INTERVALS, _longest_unchanged


STOCK_CORE = "spy75_momentum50_25"
FUND_CORE = "spy75_spmo25"


def run_core_momentum(prices, benchmarks, momentum_study=None, cost_bps=10, target_volatility=None) -> BacktestStudy:
    """Replay a fixed core and momentum sleeve against actual buy-and-hold funds.

    Supply a saved raw stock study to reuse its momentum_50 targets and dates.
    Otherwise, prices must contain SPY, SPMO and QQQ for the separate fund proxy.
    The optional volatility wrapper applies only to the combined candidate.
    """
    prices, benchmarks = _validate_price_frame(prices), _validate_price_frame(benchmarks)
    if set(benchmarks) != set(BENCHMARKS) or not prices.index.equals(benchmarks.index):
        raise ValueError("Benchmarks must contain SPY and QQQ on exactly the candidate price dates.")
    settings = _settings(cost_bps, target_volatility, None)
    _, signals = _signals(prices, None)
    allowed = {signal + 1: signal for signal, _ in signals}
    dates = prices.index
    schedule = {}
    if momentum_study is None:
        if not {"SPY", "SPMO", "QQQ"}.issubset(prices.columns):
            raise ValueError("Fund proxy prices must contain SPY, SPMO and QQQ.")
        if not prices[list(BENCHMARKS)].equals(benchmarks[list(BENCHMARKS)]):
            raise ValueError("Fund proxy and benchmark prices must use the same SPY and QQQ observations.")
        name, combined = FUND_CORE, prices[["SPY", "SPMO"]]
        for execution, signal in allowed.items():
            target = _overlay(combined, signal, np.array([.75, .25]), target_volatility)
            schedule[execution] = (signal, target, None)
        settings["momentum_source"] = "SPMO fund proxy. This differs from the custom top-50 stock sleeve."
    else:
        if any(symbol in prices for symbol in ("SPY", "QQQ", "CASH")):
            raise ValueError("Stock input must exclude the reserved SPY, QQQ and CASH names.")
        if momentum_study.settings.get("target_volatility") is not None:
            raise ValueError("Supply the raw momentum study before any volatility wrapper.")
        allocations = momentum_study.allocations["momentum_50"]
        trades = momentum_study.trades["momentum_50"]
        if (not allocations.index.equals(trades.index) or not allocations.index.is_unique
                or not allocations.index.is_monotonic_increasing or allocations.empty):
            raise ValueError("Saved momentum targets and trades need matching, unique, increasing execution dates.")
        if set(allocations.columns) != {*prices.columns, "CASH"}:
            raise ValueError("Saved momentum target columns must match the stock prices and CASH.")
        weights = allocations.loc[:, prices.columns]
        values = weights.to_numpy()
        if (not np.isfinite(values).all() or not allocations.CASH.eq(0).all()
                or not np.all(((np.isclose(values, .02, atol=1e-12, rtol=0)) | (values == 0)))
                or not weights.gt(0).sum(axis=1).eq(50).all()):
            raise ValueError("Saved raw momentum targets must hold exactly 50 stocks at 2% each, with no cash.")
        name = STOCK_CORE
        combined = pd.concat([prices, benchmarks[["SPY"]]], axis=1)
        signal_column = "signal_date" if "signal_date" in trades else "train_end"
        for execution_date, row in weights.iterrows():
            execution = dates.get_indexer([execution_date])[0]
            signal_date = pd.Timestamp(trades.loc[execution_date, signal_column])
            if execution not in allowed or dates[allowed[execution]] != signal_date:
                raise ValueError("Saved momentum targets must use a completed month-end signal and the next close.")
            signal = allowed[execution]
            desired = np.r_[.25 * row.to_numpy(), .75]
            target = _overlay(combined, signal, desired, target_volatility)
            eligible = np.r_[_longest_unchanged(prices.iloc[:signal + 1]) < STALE_INTERVALS, True]
            if np.any((desired > 0) & ~eligible):
                raise ValueError("Saved momentum targets include a stock excluded by the past unchanged-price screen.")
            schedule[execution] = (signal, target, eligible)
        settings["momentum_source"] = "Saved raw momentum_50 targets and prior-close signal dates. No stock re-ranking."
    sources = {name: combined, **{symbol: benchmarks[[symbol]] for symbol in BENCHMARKS}}
    first = min(schedule)
    schedules = {name: schedule, **{
        symbol: {first: (schedule[first][0], np.ones(1), None)} for symbol in BENCHMARKS
    }}
    settings.update({
        "model_revision": "exploratory_core_momentum_v1", "exploratory_after_prior_results": True,
        "core_weight": .75, "momentum_weight": .25, "candidate": name,
        "allocation": "Fixed 75% SPY and 25% momentum sleeve, reset monthly at the next close.",
        "benchmark_policy": "Actual SPY and QQQ buy once at the initial execution and hold. No dynamic volatility wrapper.",
        "overlay_scope": "Candidate only. The 75/25 ratio applies within the risky allocation before any cash scale.",
    })
    return _replay(sources, schedules, settings)
