"""Walk-forward tests of the historical-mean stock model.

Each window fits the model on earlier returns only. It buys that allocation
once and holds it through the next window, which the fit did not see. The same
allocation is also bought and held over its own fit dates. That is the
in-sample result. Both sides use the same holding rule.

Both sides are measured from the purchase close. The first interval of each
path is cash and pays the one entry fee. A short test window would give that
one interval far more weight than a long fit window, so it is left out of every
walk-forward measure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import performance_metrics
from .selection import SELECTED, _fit_targets, _simulate


FIRST_FIT_FRACTION = 0.5
WINDOW_COUNT = 5
MINIMUM_RETURNS = 3
METRICS = ["annual_mean", "volatility", "cagr", "total_return", "max_drawdown"]


@dataclass
class OutOfSampleStudy:
    windows: pd.DataFrame
    metrics: pd.DataFrame
    weights: pd.DataFrame
    eligibility: pd.DataFrame
    notes: list[str]
    settings: dict


def window_bounds(count: int, windows: int = WINDOW_COUNT, first_fit: float = FIRST_FIT_FRACTION) -> list[tuple[int, int]]:
    """Return (last fit return, last test return) for each consecutive window.

    The float arithmetic matches the replay split, so the 70% window uses the
    same fit as the replay.
    """
    edges = [int(np.floor(count * (first_fit + (1 - first_fit) * k / windows))) for k in range(windows)] + [count]
    bounds = list(zip(edges[:-1], edges[1:]))
    if edges[0] < MINIMUM_RETURNS or any(end - split < MINIMUM_RETURNS for split, end in bounds):
        raise ValueError(
            f"Out-of-sample tests need at least {MINIMUM_RETURNS} fit returns and {MINIMUM_RETURNS} test returns "
            f"in each of {windows} windows. This history has {count} returns."
        )
    return bounds


def _held_metrics(path, settings) -> dict:
    """Measure each path from its purchase close, after the cash interval and entry fee."""
    rows = {}
    for strategy, values in path.equity.items():
        held = values.iloc[1:]
        metrics = performance_metrics(held, settings["risk_free_rate"], settings["periods_per_year"])
        returns = held.pct_change(fill_method=None).iloc[1:]
        rows[strategy] = {**{key: metrics[key] for key in METRICS if key != "annual_mean"},
                          "annual_mean": float(returns.mean() * settings["periods_per_year"])}
    return rows


def run_out_of_sample(study, prices: pd.DataFrame, windows: int = WINDOW_COUNT,
                      first_fit: float = FIRST_FIT_FRACTION) -> OutOfSampleStudy:
    """Fit on returns before each window, then hold through that window only.

    Window k fits on returns 1..s_k and keeps interval s_k + 1 in cash. It buys
    at close s_k + 1 and holds to close e_k. The in-sample side keeps interval 1
    in cash, buys the same weights at close 1, and holds to close s_k. The
    windows do not overlap and cover the returns after the first fit. A window
    whose fit is infeasible keeps its dates and reason.
    """
    settings = study.settings
    benchmarks = study.benchmark_prices
    simulation = {key: settings[key] for key in ("cost_bps", "risk_free_rate", "periods_per_year")}
    dates = prices.index
    rows, metrics, weights, eligibility, notes = [], [], {}, [], []
    for number, (split, end) in enumerate(window_bounds(len(prices) - 1, windows, first_fit), 1):
        row = {"window": number, "fit_start": dates[1], "fit_end": dates[split],
               "test_start": dates[split + 1], "test_end": dates[end],
               "fit_returns": split, "test_returns": end - split, "status": ""}
        rows.append(row)
        fit_prices = prices.iloc[:split + 1]
        try:
            fit = _fit_targets(fit_prices, settings["target_volatility"], settings["max_weight"],
                               settings["shrinkage"], settings["periods_per_year"])
        except (ValueError, RuntimeError) as exc:
            row["status"] = str(exc)
            continue
        weights[number] = fit[0]
        eligibility.append(fit[-2].assign(window=number))
        notes.extend(f"Window {number}: {message}" for message in fit[-1])
        paths = {
            "in_sample": _simulate(fit_prices, None if benchmarks is None else benchmarks.iloc[:split + 1],
                                   0, split, fit, simulation),
            "out_of_sample": _simulate(prices.iloc[:end + 1], None if benchmarks is None else benchmarks.iloc[:end + 1],
                                       split, end, fit, simulation),
        }
        for sample, path in paths.items():
            for strategy, values in _held_metrics(path, simulation).items():
                metrics.append({"window": number, "sample": sample, "strategy": strategy, **values})
    index = pd.MultiIndex.from_tuples([], names=["window", "sample", "strategy"])
    metric_frame = (pd.DataFrame(metrics).set_index(["window", "sample", "strategy"])[METRICS] if metrics
                    else pd.DataFrame(columns=METRICS, index=index, dtype=float))
    weight_frame = pd.DataFrame(weights).T.rename_axis("window") if weights else pd.DataFrame(columns=prices.columns)
    eligibility_frame = pd.concat(eligibility, ignore_index=True) if eligibility else pd.DataFrame()
    return OutOfSampleStudy(
        pd.DataFrame(rows).set_index("window"), metric_frame, weight_frame, eligibility_frame,
        list(dict.fromkeys(notes)),
        {"windows": windows, "first_fit_fraction": first_fit, "holding_policy": "buy_and_hold",
         "measured_from": "purchase close; the cash interval and entry fee are excluded", **simulation},
    )


def sample_values(result: OutOfSampleStudy, sample: str, measure: str = "annual_mean") -> pd.DataFrame:
    """One row per completed window and one column per path, for one side of the split, in engine path order."""
    order = list(dict.fromkeys(result.metrics.index.get_level_values("strategy")))
    return result.metrics[measure].xs(sample, level="sample").unstack("strategy")[order]


def portfolio_comparison(result: OutOfSampleStudy, measure: str = "annual_mean") -> pd.DataFrame:
    """One row per completed window: the stock portfolio in and out of sample, then each reference out of sample."""
    if result.metrics.empty:
        return pd.DataFrame(columns=["in_sample", "out_of_sample", "change"])
    inside, outside = sample_values(result, "in_sample", measure), sample_values(result, "out_of_sample", measure)
    frame = pd.DataFrame({"in_sample": inside[SELECTED], "out_of_sample": outside[SELECTED]})
    frame["change"] = frame.out_of_sample - frame.in_sample
    for strategy in outside.columns.drop(SELECTED):
        frame[strategy] = outside[strategy]
    return frame
