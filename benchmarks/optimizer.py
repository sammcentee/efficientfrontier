"""Bounded synthetic optimizer benchmark; results are not investment evidence.

Run: python -m benchmarks.optimizer --sizes 100 500 1000 --output results/benchmarks.json
An optional --baseline core.py compares an archived implementation on identical data.
Each case uses a fresh process, one BLAS/OpenMP thread, and Clarabel's default
thread setting. Timeout covers imports, data generation, optimization and checks;
optimizer_seconds measures only optimize(). Peak RSS includes the whole process.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def run_case(args):
    import numpy as np
    import pandas as pd

    if args.implementation:
        spec = importlib.util.spec_from_file_location("benchmark_core", args.implementation)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        optimize = module.optimize
    else:
        from efficient_frontier.core import optimize

    count = args.sizes[0]
    rng = np.random.default_rng(20261003 + count)
    factors = min(24, count)
    loadings = rng.normal(size=(count, factors)) * 0.05 / np.sqrt(factors)
    sigma = loadings @ loadings.T + np.diag(rng.uniform(0.01, 0.04, count) ** 2)
    labels = [f"SYNTHETIC_{i:05d}" for i in range(count)]
    mu = pd.Series(np.linspace(0.03, 0.20, count), index=labels)
    covariance = pd.DataFrame(sigma, index=labels, columns=labels)
    start = time.perf_counter()
    portfolios, frontier, weights, warnings = optimize(
        mu, covariance, max_weight=args.max_weight, frontier_points=args.frontier_points,
    )
    elapsed = time.perf_counter() - start
    all_weights = np.vstack([*(p.weights.to_numpy() for p in portfolios.values()), weights.to_numpy()])
    tolerance = 1e-6
    checks = {
        "finite_weights": bool(np.isfinite(all_weights).all()),
        "sum_to_one": bool(np.all(np.abs(all_weights.sum(axis=1) - 1) <= tolerance)),
        "nonnegative": bool(np.all(all_weights >= -tolerance)),
        "position_cap": bool(np.all(all_weights <= args.max_weight + tolerance)),
        "frontier_returns": bool(np.allclose(weights.to_numpy() @ mu, frontier.expected_return, atol=tolerance)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Invalid optimizer output: {checks}")
    return {
        "status": "ok", "optimizer_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "actual_frontier_points": len(frontier), "checks": checks, "warnings": warnings,
        "minimum_variance": float(portfolios["Minimum volatility"].volatility ** 2),
        "maximum_sharpe": float(portfolios["Maximum Sharpe"].sharpe),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=positive_int, default=[100, 500, 1000])
    parser.add_argument("--frontier-points", type=positive_int, default=40)
    parser.add_argument("--max-weight", type=positive_float, default=1.0)
    parser.add_argument("--timeout", type=positive_float, default=60)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frontier_points < 2 or args.max_weight > 1:
        parser.error("frontier-points must be at least 2 and max-weight at most 1")
    if any(size * args.max_weight < 1 - 1e-12 for size in args.sizes):
        parser.error("every size * max-weight must be at least 1")
    if args.baseline and not args.baseline.is_file():
        parser.error("baseline must point to an existing Python file")
    if args.worker:
        print(json.dumps(run_case(args), allow_nan=False))
        return 0

    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    report = {
        "model": "Synthetic annual means and dense, full-rank covariance from up to 24 factors plus positive diagonal; seed 20261003 + assets. No market data or investment claims.",
        "metadata": {
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "cvxpy", "clarabel")},
            "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {key: environment.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "clarabel_max_threads": "solver default", "timeout_seconds": args.timeout,
        },
        "cases": [],
    }
    implementations = [("current", None)]
    if args.baseline:
        implementations.insert(0, ("baseline", str(args.baseline.resolve())))
    for size in args.sizes:
        for name, source in implementations:
            command = [sys.executable, "-m", "benchmarks.optimizer", "--worker", "--sizes", str(size),
                       "--frontier-points", str(args.frontier_points), "--max-weight", str(args.max_weight)]
            if source:
                command.extend(["--implementation", source])
            case = {"implementation": name, "assets": size, "requested_frontier_points": args.frontier_points, "max_weight": args.max_weight}
            start = time.perf_counter()
            try:
                result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=args.timeout)
                if result.returncode:
                    case.update(status="error", stderr=result.stderr, stdout=result.stdout)
                else:
                    case.update(json.loads(result.stdout))
                    if result.stderr:
                        case["stderr"] = result.stderr
            except subprocess.TimeoutExpired:
                case["status"] = "timeout"
            case["process_seconds"] = time.perf_counter() - start
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not args.output:
        print(json.dumps(report, indent=2, allow_nan=False))
    return int(any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    sys.exit(main())
