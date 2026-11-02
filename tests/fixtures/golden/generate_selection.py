"""Golden results of the Python release for the Rust parity tests.

Run from a checkout of the Python release (commit 6c6df6e) with its
requirements installed:

    python tests/fixtures/golden/generate_selection.py

The script writes input prices as CSV and every result as JSON beside itself.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from efficient_frontier.core import analyze  # noqa: E402
from efficient_frontier.data import demo_prices  # noqa: E402
from efficient_frontier.out_of_sample import run_out_of_sample  # noqa: E402
from efficient_frontier.selection import (  # noqa: E402
    run_selection, selection_fit_estimates, selection_frontier,
)

HERE = Path(__file__).resolve().parent


def value(item):
    if isinstance(item, (pd.Timestamp,)):
        return None if pd.isna(item) else item.date().isoformat()
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, (np.integer,)):
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
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        labels = [[value(x) for x in row] for row in index]
    else:
        labels = [value(x) for x in index]
    return {
        "index_names": [value(name) for name in index.names],
        "index": labels,
        "columns": [str(c) for c in frame.columns],
        "data": [[value(x) for x in row] for row in frame.itertuples(index=False, name=None)],
    }


def synthetic(count, assets, seed, start="2019-01-01"):
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
    benchmarks = pd.DataFrame({"SPY": spy, "QQQ": qqq}, index=dates)
    return prices, benchmarks


def selection_case(name, prices, benchmarks, **options):
    prices.to_csv(HERE / f"{name}_prices.csv", lineterminator="\n")
    if benchmarks is not None:
        benchmarks.to_csv(HERE / f"{name}_benchmarks.csv", lineterminator="\n")
    result = {"options": options, "benchmarks": benchmarks is not None}
    try:
        study = run_selection(prices, benchmarks, **options)
    except (ValueError, RuntimeError) as exc:
        result["error"] = str(exc)
        (HERE / f"{name}.json").write_text(json.dumps(result, indent=1))
        return
    test = study.test
    result["selection"] = {
        "settings": value(study.settings), "test_settings": value(test.settings),
        "supported": study.supported, "reasons": study.reasons, "warnings": study.warnings,
        "test_warnings": test.warnings,
        "equity": split(test.equity), "metrics": split(test.metrics),
        "allocations": {k: split(v) for k, v in test.allocations.items()},
        "holdings": {k: split(v) for k, v in test.holdings.items()},
        "trades": {k: split(v) for k, v in test.trades.items()},
        "latest_weights": value(study.latest[study.selected].weights.to_dict()),
        "latest_expected": study.latest[study.selected].expected_return,
        "latest_volatility": study.latest[study.selected].volatility,
        "latest_sharpe": value(study.latest[study.selected].sharpe),
        "eligibility": split(study.eligibility),
    }
    result["fit_estimates"] = split(selection_fit_estimates(study, prices))
    frontier, weights, comparisons, warnings = selection_frontier(study, prices)
    result["frontier"] = {"frontier": split(frontier), "weights": split(weights),
                          "comparisons": split(comparisons), "warnings": warnings}
    try:
        oos = run_out_of_sample(study, prices)
        result["out_of_sample"] = {
            "windows": split(oos.windows), "metrics": split(oos.metrics), "weights": split(oos.weights),
            "eligibility": split(oos.eligibility), "notes": oos.notes, "settings": value(oos.settings),
        }
    except ValueError as exc:
        result["out_of_sample"] = {"error": str(exc)}
    (HERE / f"{name}.json").write_text(json.dumps(result, indent=1))


def analyze_case(name, prices, **options):
    prices.to_csv(HERE / f"{name}_prices.csv", lineterminator="\n")
    analysis = analyze(prices, **options)
    result = {
        "options": options,
        "portfolios": {name: {"weights": value(list(p.weights)), "expected_return": p.expected_return,
                              "volatility": p.volatility, "sharpe": value(p.sharpe)}
                       for name, p in analysis.portfolios.items()},
        "frontier": split(analysis.frontier), "frontier_weights": split(analysis.frontier_weights),
        "holdout_metrics": split(analysis.holdout_metrics), "equity": split(analysis.equity),
        "warnings": analysis.warnings,
    }
    (HERE / f"{name}.json").write_text(json.dumps(result, indent=1))


def main():
    demo = demo_prices()
    selection_case("selection_demo", demo, None, target_volatility=.2, max_weight=.2)
    prices, benchmarks = synthetic(700, 40, 11)
    selection_case("selection_market", prices, benchmarks, target_volatility=.18, max_weight=.1)
    stale = prices.iloc[:300, :15].copy()
    stale.iloc[100:130, 3] = stale.iloc[100, 3]
    stale.iloc[270:, 7] = stale.iloc[270, 7]
    selection_case("selection_stale", stale, benchmarks.iloc[:300], target_volatility=.25, max_weight=.15,
                   shrinkage=.2, cost_bps=25.0)
    selection_case("selection_weekly", prices.iloc[::5, :12], None, target_volatility=.3, max_weight=.25,
                   periods_per_year=52.0, train_fraction=.6)
    selection_case("selection_infeasible", prices.iloc[:200, :12], None, target_volatility=.001, max_weight=.1)
    analyze_case("analyze_demo", demo)
    analyze_case("analyze_capped", prices.iloc[:400, :20], max_weight=.1, shrinkage=0.0, frontier_points=12)
    analyze_case("analyze_singular", prices.iloc[:40, :45 if prices.shape[1] >= 45 else 40], max_weight=.05,
                 frontier_points=8)


if __name__ == "__main__":
    main()
