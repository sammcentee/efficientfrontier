"""Golden results of the Python release for the legacy research parity tests.

Run from a checkout of the Python release (commit 6c6df6e) with its
requirements installed:

    python tests/fixtures/golden/generate_legacy.py

The script writes synthetic input prices as CSV and every result as JSON beside
itself. It never uses market data. It also writes legacy_test_data.json: the
random inputs of the Python tests, so that the Rust ports use the same numbers.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from efficient_frontier.backtest import run_backtests  # noqa: E402
from efficient_frontier.benchmarks import compare_benchmarks  # noqa: E402
from efficient_frontier.core import analyze  # noqa: E402
from efficient_frontier.data import demo_prices  # noqa: E402
from efficient_frontier.evidence import analyze_evidence  # noqa: E402
from efficient_frontier.profiles import build_profiles  # noqa: E402

HERE = Path(__file__).resolve().parent


def value(item):
    if isinstance(item, pd.Timestamp):
        return None if pd.isna(item) else item.date().isoformat()
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, np.integer):
        return int(item)
    if isinstance(item, (float, np.floating)):
        return None if math.isnan(item) else float(item)
    if item is None or item is pd.NaT:
        return None
    if isinstance(item, (list, tuple)):
        return [value(x) for x in item]
    if isinstance(item, dict):
        return {str(k): value(v) for k, v in item.items()}
    return item


def split(frame):
    """A DataFrame in pandas split form, with ISO dates and NaN as null."""
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        labels = [[value(x) for x in row] for row in index]
    else:
        labels = [value(x) for x in index]
    return {
        "index_names": [value(name) for name in index.names],
        "index": labels,
        "columns": [value(c) if isinstance(c, tuple) else str(c) for c in frame.columns],
        "data": [[value(x) for x in row] for row in frame.itertuples(index=False, name=None)],
    }


def synthetic(count, assets, seed, start="2018-01-01"):
    """Daily one-factor prices for the assets, with related SPY and QQQ paths."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, (count, 1))
    betas = np.linspace(0.5, 1.4, assets)
    noise = rng.normal(0, 1, (count, assets)) * np.linspace(0.006, 0.02, assets)
    drift = np.linspace(-0.0002, 0.0009, assets)
    returns = market * betas + noise + drift
    dates = pd.bdate_range(start, periods=count + 1, name="Date")
    prices = pd.DataFrame(100 * np.vstack([np.ones(assets), np.cumprod(1 + returns, axis=0)]),
                          index=dates, columns=[f"S{i:02d}" for i in range(assets)])
    spy = 100 * np.cumprod(np.r_[1, 1 + market[:, 0] + rng.normal(0, 0.002, count)])
    qqq = 100 * np.cumprod(np.r_[1, 1 + 1.2 * market[:, 0] + rng.normal(0.0002, 0.004, count)])
    return prices, pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=dates)


def save_prices(name, prices):
    if name != "demo":
        prices.to_csv(HERE / f"{name}.csv", lineterminator="\n")
    return name


def portfolios(items):
    return {name: {"weights": value(list(p.weights)), "labels": [str(x) for x in p.weights.index],
                   "expected_return": p.expected_return, "volatility": p.volatility, "sharpe": value(p.sharpe)}
            for name, p in items.items()}


def profiles_result(result):
    return {
        "portfolios": portfolios(result.portfolios),
        "frontier": split(result.frontier), "frontier_weights": split(result.frontier_weights),
        "windows": split(result.windows), "window_returns": split(result.window_returns),
        "summary": split(result.summary), "as_of": result.as_of, "observations": result.observations,
        "settings": value(result.settings), "warnings": result.warnings,
    }


def study_result(study):
    return {
        "equity": split(study.equity), "metrics": split(study.metrics),
        "allocations": {k: split(v) for k, v in study.allocations.items()},
        "holdings": {k: split(v) for k, v in study.holdings.items()},
        "trades": {k: split(v) for k, v in study.trades.items()},
        "warnings": study.warnings, "settings": value(study.settings),
    }


def benchmarks_result(result):
    optional = lambda frame: None if frame is None else split(frame)  # noqa: E731
    return {
        "prices": split(result.prices), "holdout_equity": split(result.holdout_equity),
        "holdout_metrics": split(result.holdout_metrics), "backtest_equity": optional(result.backtest_equity),
        "backtest_metrics": optional(result.backtest_metrics), "training_estimates": split(result.training_estimates),
        "latest_estimates": split(result.latest_estimates), "settings": value(result.settings),
    }


def evidence_result(strategy, benchmark, **options):
    result = analyze_evidence(strategy, benchmark, **options)
    relative = result.relative_equity
    return {
        "options": options, "strategy_equity": split(strategy), "benchmark_equity": split(benchmark),
        "summary": split(result.summary), "windows": split(result.windows),
        "relative_equity": {"index": [value(x) for x in relative.index], "columns": [list(c) for c in relative.columns],
                            "data": [[value(x) for x in row] for row in relative.to_numpy()]},
        "settings": value(result.settings), "warnings": result.warnings,
    }


def test_data():
    """The NumPy random inputs of tests/test_evidence.py, test_risk.py and test_backtest.py."""
    data = {}
    for count in (19, 59, 60, 180):
        rng = np.random.default_rng(100)
        benchmark = rng.normal(0.0004, 0.008, count)
        residual = rng.normal(0, 0.002, count)
        for i in range(1, count):
            residual[i] += 0.65 * residual[i - 1]
        data[f"evidence_sample_{count}"] = {"strategy": list(0.0005 + 1.2 * benchmark + residual),
                                            "benchmark": list(benchmark)}
    rng = np.random.default_rng(42)
    data["evidence_signal"] = {"benchmark": list(rng.normal(0.0004, 0.008, 180)),
                               "noise": list(rng.normal(0, 0.0003, 180))}
    data["evidence_exposure"] = list(np.random.default_rng(42).normal(0.003, 0.003, 180))
    rng = np.random.default_rng(42)
    data["risk_returns"] = rng.normal([0.0004, 0.0002, 0.0003], [0.015, 0.006, 0.009], (120, 3)).T.tolist()
    rng = np.random.default_rng(304)
    data["backtest_profile_returns"] = rng.normal([0.01, 0.006], [0.03, 0.015], size=(20, 2)).T.tolist()
    (HERE / "legacy_test_data.json").write_text(json.dumps(data, indent=1))


def main():
    test_data()
    demo = demo_prices()
    capped, _ = synthetic(260, 12, 31)
    weekly = synthetic(900, 6, 41)[0].iloc[::5]
    market, market_benchmarks = synthetic(320, 8, 51)
    files = {"demo": demo, "legacy_capped_prices": capped, "legacy_weekly_prices": weekly,
             "legacy_market_prices": market, "legacy_market_benchmark_prices": market_benchmarks}
    for name, prices in files.items():
        save_prices(name, prices)

    profile_cases = [
        ("demo", None, {}),
        ("demo", None, {"return_objective": "worst_window", "frontier_points": 11, "max_weight": 0.5}),
        ("legacy_capped_prices", None, {"max_weight": 0.15, "shrinkage": 0.3, "frontier_points": 15}),
        ("legacy_capped_prices", None, {"max_weight": 0.15, "shrinkage": 0.3, "frontier_points": 15,
                                        "return_objective": "worst_window"}),
        ("legacy_weekly_prices", None, {"periods_per_year": 52.0, "risk_free_rate": 0.03, "max_weight": 0.4,
                                        "return_objective": "worst_window", "frontier_points": 9}),
        # More assets than returns: singular covariance and the tied-risk score problem.
        ("legacy_capped_prices", 10, {"shrinkage": 0.0, "max_weight": 0.2, "frontier_points": 5,
                                      "return_objective": "worst_window"}),
    ]
    cases = []
    for name, rows, options in profile_cases:
        prices = files[name] if rows is None else files[name].iloc[:rows]
        cases.append({"prices": name, "rows": rows, "options": options,
                      "result": profiles_result(build_profiles(prices, **options))})
    (HERE / "legacy_profiles.json").write_text(json.dumps({"cases": cases}, indent=1))

    backtest_cases = [
        ("demo", None, None, {}),
        ("legacy_capped_prices", None, None, {"train_fraction": 0.6, "rebalance_every": 15, "rolling_window": 60,
                                              "cost_bps": 25.0, "max_weight": 0.3, "shrinkage": 0.2,
                                              "include_profiles": True}),
        ("legacy_weekly_prices", None, None, {"train_fraction": 0.5, "rebalance_every": 8, "rolling_window": 30,
                                              "cost_bps": 0.0, "risk_free_rate": 0.03, "periods_per_year": 52.0,
                                              "include_profiles": True}),
        # No asset beats the risk-free rate: Maximum Sharpe falls back with warnings.
        ("legacy_capped_prices", 120, 5, {"train_fraction": 0.5, "rebalance_every": 10, "cost_bps": 5.0,
                                          "risk_free_rate": 2.0, "include_profiles": True}),
    ]
    cases = []
    for name, rows, columns, options in backtest_cases:
        prices = files[name] if rows is None else files[name].iloc[:rows, :columns]
        cases.append({"prices": name, "rows": rows, "columns": columns, "options": options,
                      "result": study_result(run_backtests(prices, **options))})
    (HERE / "legacy_backtests.json").write_text(json.dumps({"cases": cases}, indent=1))

    settings = {"train_fraction": 0.6, "risk_free_rate": 0.03, "max_weight": 0.4}
    analysis = analyze(market, **settings)
    study = run_backtests(market, **settings, rebalance_every=20, cost_bps=15.0, include_profiles=True)
    latest = build_profiles(market, risk_free_rate=0.03, max_weight=0.4)
    full = compare_benchmarks(market_benchmarks, market, analysis, study, latest, risk_free_rate=0.03)
    plain = compare_benchmarks(market_benchmarks, market, analysis, risk_free_rate=0.03)
    benchmark_cases = {
        "settings": settings, "backtest_options": {"rebalance_every": 20, "cost_bps": 15.0, "include_profiles": True},
        "profile_options": {"risk_free_rate": 0.03, "max_weight": 0.4},
        "full": benchmarks_result(full), "plain": benchmarks_result(plain),
    }
    (HERE / "legacy_benchmarks.json").write_text(json.dumps(benchmark_cases, indent=1))

    weekly_study = run_backtests(weekly, train_fraction=0.4, periods_per_year=52.0, rebalance_every=8, cost_bps=20.0)
    held = weekly.iloc[len(weekly) - len(weekly_study.equity):]
    weekly_benchmarks = pd.DataFrame({"Fund": held.iloc[:, 0], "Index": held.mean(axis=1)})
    evidence_cases = [
        # Every fourth path keeps all four methods with a mix of portfolios and profiles.
        evidence_result(study.equity.iloc[:, ::4], full.backtest_equity, risk_free_rate=0.03, periods_per_year=252.0,
                        delayed_entry=True),
        evidence_result(analysis.equity, full.holdout_equity, risk_free_rate=0.03, periods_per_year=252.0,
                        delayed_entry=False),
        evidence_result(weekly_study.equity.iloc[:, :6], weekly_benchmarks,
                        risk_free_rate=0.01, periods_per_year=52.0, delayed_entry=True),
    ]
    (HERE / "legacy_evidence.json").write_text(json.dumps({"cases": evidence_cases}, indent=1))


if __name__ == "__main__":
    main()
