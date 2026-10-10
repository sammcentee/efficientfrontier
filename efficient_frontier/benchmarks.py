"""Independent market references with the same evaluation dates and costs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestStudy
from .core import Analysis, _validate_price_frame
from .data import validate_benchmark_prices
from .metrics import performance_metrics
from .profiles import ProfileAnalysis


BENCHMARK_NAMES = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)"}


@dataclass
class BenchmarkComparison:
    prices: pd.DataFrame
    holdout_equity: pd.DataFrame
    holdout_metrics: pd.DataFrame
    backtest_equity: pd.DataFrame | None
    backtest_metrics: pd.DataFrame | None
    training_estimates: pd.DataFrame
    latest_estimates: pd.DataFrame
    settings: dict


def _benchmark_buy_and_hold(prices, evaluation_dates, *, delayed_entry=False, cost_bps=0.0):
    """Use exact observation dates, with either original or delayed entry."""
    selected = prices.loc[evaluation_dates].rename(columns=BENCHMARK_NAMES)
    if delayed_entry:
        equity = selected.div(selected.iloc[1]) / (1 + cost_bps / 10_000)
        equity.iloc[0] = 1.0
        return equity
    if cost_bps != 0:
        raise ValueError("Original holdout benchmarks exclude fees.")
    return selected.div(selected.iloc[0])


def _estimates(returns, risk_free_rate, periods_per_year):
    mean = returns.mean() * periods_per_year
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return pd.DataFrame({
        "expected_return": mean,
        "volatility": volatility,
        "sharpe": (mean - risk_free_rate) / volatility.where(volatility > 1e-12),
    }).rename(index=BENCHMARK_NAMES).rename_axis("benchmark")


def _metrics(equity, risk_free_rate, periods_per_year):
    return pd.DataFrame.from_dict(
        {name: performance_metrics(equity[name], risk_free_rate, periods_per_year) for name in equity},
        orient="index",
    ).rename_axis("benchmark")


def compare_benchmarks(
    benchmark_prices: pd.DataFrame,
    asset_prices: pd.DataFrame,
    analysis: Analysis,
    study: BacktestStudy | None = None,
    latest_profiles: ProfileAnalysis | None = None,
    *,
    risk_free_rate: float,
) -> BenchmarkComparison:
    """Compare SPY and QQQ without any change to the portfolio universe.

    The original holdout buys at the split close and excludes fees. Backtest
    references buy at the next close with the study's entry fee. They earn zero
    interest before entry and exclude terminal liquidation. Estimates use the
    same observations as the corresponding portfolio estimates.
    """
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if analysis.risk_free_rate != risk_free_rate:
        raise ValueError("Analysis and benchmarks must use the same risk_free_rate.")
    asset_prices = _validate_price_frame(asset_prices)
    expected_returns = analysis.train_returns.index.append(analysis.test_returns.index)
    if (
        not asset_prices.index[1:].equals(expected_returns)
        or not asset_prices.index[len(analysis.train_returns):].equals(analysis.equity.index)
    ):
        raise ValueError("Asset prices and analysis must use the same observation dates.")
    prices = validate_benchmark_prices(benchmark_prices, expected_dates=asset_prices.index)
    periods = analysis.periods_per_year
    returns = prices.pct_change(fill_method=None).iloc[1:]
    training_estimates = _estimates(returns.loc[analysis.train_returns.index], risk_free_rate, periods)
    latest_estimates = _estimates(returns, risk_free_rate, periods)
    latest_estimates["worst_window_return"] = np.nan
    if latest_profiles is not None:
        if latest_profiles.settings["periods_per_year"] != periods:
            raise ValueError("Latest profiles and benchmarks must use the same periods_per_year.")
        if latest_profiles.settings["risk_free_rate"] != risk_free_rate:
            raise ValueError("Latest profiles and benchmarks must use the same risk_free_rate.")
        if latest_profiles.observations != len(returns) or latest_profiles.as_of != str(prices.index[-1].date()):
            raise ValueError("Latest profiles and benchmarks must use the same observation dates.")
        window_means = []
        for window in latest_profiles.windows.itertuples():
            sample = returns.loc[window.start:window.end]
            if len(sample) != window.observations:
                raise ValueError("Latest profiles and benchmarks must use the same historical windows.")
            window_means.append(sample.mean() * periods)
        latest_estimates["worst_window_return"] = (
            pd.concat(window_means, axis=1).min(axis=1).rename(index=BENCHMARK_NAMES)
        )

    holdout_equity = _benchmark_buy_and_hold(prices, analysis.equity.index)
    holdout_metrics = _metrics(holdout_equity, risk_free_rate, periods)
    backtest_equity = backtest_metrics = None
    cost_bps = None
    if study is not None:
        if not study.equity.index.equals(analysis.equity.index):
            raise ValueError("Backtests and benchmarks must use the same evaluation dates.")
        if study.settings["periods_per_year"] != periods:
            raise ValueError("Backtests and benchmarks must use the same periods_per_year.")
        if study.settings["risk_free_rate"] != risk_free_rate:
            raise ValueError("Backtests and benchmarks must use the same risk_free_rate.")
        cost_bps = study.settings["cost_bps"]
        backtest_equity = _benchmark_buy_and_hold(prices, study.equity.index, delayed_entry=True, cost_bps=cost_bps)
        backtest_metrics = _metrics(backtest_equity, risk_free_rate, periods)
        entry_value = 1 / (1 + cost_bps / 10_000)
        backtest_metrics["total_turnover"] = entry_value
        backtest_metrics["total_cost"] = 1 - entry_value
        backtest_metrics["rebalance_count"] = 1
        backtest_metrics["fallback_count"] = 0

    return BenchmarkComparison(
        prices=prices,
        holdout_equity=holdout_equity,
        holdout_metrics=holdout_metrics,
        backtest_equity=backtest_equity,
        backtest_metrics=backtest_metrics,
        training_estimates=training_estimates,
        latest_estimates=latest_estimates,
        settings={
            "symbols": list(prices.columns),
            "display_names": BENCHMARK_NAMES.copy(),
            "periods_per_year": periods,
            "risk_free_rate": risk_free_rate,
            "price_start": str(prices.index[0].date()),
            "price_end": str(prices.index[-1].date()),
            "price_observations": len(prices),
            "holdout_entry": str(holdout_equity.index[0].date()),
            "holdout_cost_bps": 0.0,
            "backtest_entry": str(backtest_equity.index[1].date()) if study is not None else None,
            "backtest_cost_bps": cost_bps,
            "alignment": "Exact matches for every asset observation date. No fills or dropped portfolio dates.",
            "role": "Independent buy-and-hold references. Portfolio position caps do not apply to benchmarks.",
            "backtest_execution": "First interval in zero-interest cash. Entry at the next close. No terminal sale.",
        },
    )
