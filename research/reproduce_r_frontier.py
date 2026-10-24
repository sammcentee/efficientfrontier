"""Compare a corrected stock frontier with the original R return calculation.

This is an independent Python calculation, not an execution of R. All results
use the fitting dates, without costs, covariance shrinkage, or a later test.
"""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


def r_reference_estimates(prices):
    """Return annual arithmetic means, sample covariance, and R line 39 scores."""
    if (len(prices) < 3 or prices.empty or not prices.columns.is_unique
            or not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or not prices.index.is_unique or not prices.index.is_monotonic_increasing):
        raise ValueError("Prices need unique assets and at least three increasing, unique dates.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and positive, with no missing observations.")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    legacy = 252 * prices.iloc[-1] / prices.iloc[0] / len(prices)
    return returns.mean() * 252, returns.cov() * 252, legacy


def fit_at_risk(mean, covariance, risk_ceiling, max_weight):
    """Maximize the mean and certify it with a global quadratic tangent bound."""
    if not 0 < max_weight <= 1 or len(mean) * max_weight < 1 - 1e-12:
        raise ValueError("The holding cap is infeasible for this asset count.")
    if not np.isfinite(risk_ceiling) or risk_ceiling < 0:
        raise ValueError("The volatility ceiling must be finite and nonnegative.")
    mu, sigma = mean.to_numpy(), covariance.loc[mean.index, mean.index].to_numpy()
    mean_scale, variance_scale = max(float(np.max(np.abs(mu))), 1e-12), float(np.max(np.abs(sigma))) or 1.0
    weights = cp.Variable(len(mu))
    risk = cp.quad_form(weights, cp.psd_wrap(sigma / variance_scale)) <= risk_ceiling ** 2 / variance_scale
    problem = cp.Problem(cp.Maximize(mu @ weights / mean_scale),
                         [weights >= 0, weights <= max_weight, cp.sum(weights) == 1, risk])
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=200)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Reference optimization failed: {problem.status}")
    raw = np.asarray(weights.value)
    if not np.isfinite(raw).all() or abs(raw.sum() - 1) > 1e-7 or raw.min() < -1e-7 or raw.max() > max_weight + 1e-7:
        raise RuntimeError("Reference solver returned infeasible weights.")
    chosen = np.clip(raw, 0, max_weight)
    chosen /= chosen.sum()
    variance = float(chosen @ sigma @ chosen)
    volatility, achieved = float(np.sqrt(max(variance, 0))), float(mu @ chosen)
    if volatility > risk_ceiling + 1e-7 or chosen.max() > max_weight + 1e-7:
        raise RuntimeError("Reference portfolio exceeded its risk or holding limit.")
    multiplier = max(0, np.asarray(risk.dual_value).item() * mean_scale / variance_scale)
    # For feasible x: mu'x <= lambda*(ceiling²+w'Sw) + (mu-2*lambda*Sw)'x.
    # The final linear maximum on the capped simplex has this exact greedy solution.
    gradient = mu - 2 * multiplier * sigma @ chosen
    remaining, linear_max = 1.0, 0.0
    for value in np.sort(gradient)[::-1]:
        amount = min(max_weight, remaining)
        linear_max += amount * value
        remaining -= amount
        if remaining < 1e-14:
            break
    upper = multiplier * (risk_ceiling ** 2 + variance) + linear_max
    return pd.Series(chosen, index=mean.index), {
        "fitted_stock_mean": achieved, "fitted_stock_risk": volatility,
        "largest_weight": float(chosen.max()), "holdings_above_0_1pct": int((chosen > .001).sum()),
        "global_upper_bound_mean": upper, "global_bound_gap": upper - achieved, "solver_status": problem.status,
    }


def compare_r_frontier(prices, benchmarks):
    """Make only the SPY/QQQ risk comparisons, with 100% and 10% holding caps."""
    if not prices.index.equals(benchmarks.index) or not {"SPY", "QQQ"} <= set(benchmarks):
        raise ValueError("SPY and QQQ need exactly the stock price dates.")
    if {"SPY", "QQQ"} & set(prices):
        raise ValueError("Benchmarks are references and cannot be eligible stock holdings.")
    mean, covariance, legacy = r_reference_estimates(prices)
    benchmark_mean, benchmark_covariance, _ = r_reference_estimates(benchmarks[["SPY", "QQQ"]])
    rows, holdings = [], {}
    for cap in (1.0, .10):
        for symbol in ("SPY", "QQQ"):
            ceiling = float(np.sqrt(benchmark_covariance.loc[symbol, symbol]))
            weights, result = fit_at_risk(mean, covariance, ceiling, cap)
            label = f"{symbol}_risk_cap_{int(cap * 100)}pct"
            holdings[label] = weights
            rows.append({"comparison": label, "benchmark": symbol, "max_weight_limit": cap,
                         "annual_benchmark_mean": float(benchmark_mean[symbol]), "benchmark_risk": ceiling,
                         **result, "annual_mean_advantage": result["fitted_stock_mean"] - benchmark_mean[symbol],
                         "legacy_annual_score_for_these_weights": float(legacy @ weights)})
    return pd.DataFrame(rows), pd.DataFrame(holdings).rename_axis("symbol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prices", "benchmarks", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    prices, benchmarks = (pd.read_csv(path, index_col=0, parse_dates=True) for path in (args.prices, args.benchmarks))
    metrics, weights = compare_r_frontier(prices, benchmarks)
    root = Path(__file__).resolve().parents[1]
    sources = {"prices": args.prices, "benchmarks": args.benchmarks, "reproduction_source": Path(__file__),
               "original_R": root / "Efficient Frontier v1.12.R", "original_holdings": root / "spy_holdings.ods"}
    metadata = {"method": __doc__.strip(), "price_start": str(prices.index[0].date()),
                "price_end": str(prices.index[-1].date()), "price_rows": len(prices), "return_rows": len(prices) - 1,
                "stocks": list(prices.columns), "annualization": 252, "covariance_shrinkage": 0, "risk_free_rate": 0,
                "holding_caps": [1.0, .10], "legacy_score_formula": "252 * (last_price / first_price) / price_rows",
                "source_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
                "versions": {name: version(name) for name in ("numpy", "pandas", "cvxpy", "clarabel")}}
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    weights.to_csv(args.output / "weights.csv")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
