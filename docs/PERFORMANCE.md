# Optimizer capacity and reproducible benchmarks

The active implementation is Python plus NumPy/CVXPY and the compiled Rust Clarabel solver. R is retained only as historical source. No hardcoded maximum ticker count or holdings count exists. Allocations can use any nonempty subset of the supplied universe, subject to the long-only, fully invested model and any optional position cap.

## Measured comparison

Measured on 2026-10-03 with Python 3.14.7, NumPy 2.5.3, pandas 3.0.6, CVXPY 1.9.3 and Clarabel 0.11.1, Linux/WSL2, 14 available logical CPUs. Each case runs in a fresh subprocess with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and Clarabel's default thread setting. These are single-run measurements, not averages or service guarantees.

The baseline is `efficient_frontier/core.py` at commit `277b4ed`. Both versions use identical synthetic annual means and a dense, full-rank covariance matrix generated from up to 24 factors plus a positive diagonal. The optimizer receives the complete dense matrix; this benchmark does not substitute an approximate risk model. A fixed seed makes each size reproducible. Synthetic Sharpe values are test outputs, not evidence of investment returns.

Each case requests 40 frontier points and uses no additional position cap:

| Assets | Previous Python optimizer | Updated optimizer | Previous peak RSS | Updated peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.333 s | 0.161 s | 199.7 MiB | 197.7 MiB |
| 500 | 9.644 s | 4.286 s | 303.1 MiB | 268.3 MiB |
| 1,000 | 29.769 s | 20.757 s | 601.3 MiB | 461.1 MiB |

The current 2,000-asset, 40-point case did not finish within the 60-second subprocess budget and was terminated. No successful timing or peak-memory result is claimed for that case; 2,000 is not an application cap. A separate 500-asset comparison with a 0.5% position cap (requiring at least 200 holdings) completed all 40 points: 7.604 → 4.396 seconds and 304.0 → 271.5 MiB peak RSS, with all allocation checks passing.

All six cases produced 40 points. Every portfolio and frontier allocation was checked for finite values, nonnegative weights, total weight one, and compliance with the selected cap. Frontier expected returns were checked against their actual weights. Separate analytic unit tests verify correct minimum-volatility and maximum-Sharpe solutions, including a 128-asset case in which all 128 holdings are used.

Optimizer time covers `optimize()`, including input validation, problem construction and solving. It excludes model generation, price downloading, historical estimation, charts and report export. Peak RSS is the Linux maximum resident memory of the entire worker process, including Python imports and model generation. It is not covariance-array size alone.

## What changed

- No extra position cap is applied by default in the app or CLI. Single-ticker inputs are supported, and optional caps can be smaller than the old interface's 5% minimum.
- At a 100% cap, nonnegativity and full investment already imply the upper bound; redundant solver constraints are removed.
- A capped maximum-Sharpe problem uses one shared normalization variable, avoiding a dense constraint matrix built by broadcasting a sum across every asset. At 128 assets the constraint matrix has 641 nonzeros with a 5% cap, compared with 16,640 previously; uncapped it has 256 rather than 16,512.
- Positive-definite covariance validation uses Cholesky first, with eigenvalue validation retained for singular cases. This changes the constant cost, not the cubic complexity class of dense validation.
- Large correlation charts start with a display subset. This does not change optimization inputs or exported weights.

The full covariance is still used. Dense covariance storage is quadratic in asset count; factorizing dense systems is generally cubic. Memory, solver convergence, available history and data-provider limits remain practical limits. There is no language switch that makes an arbitrarily large dense problem free to compute.

## Reproduce on Linux

From the repository root after installing the requirements:

```bash
.venv/bin/python -m benchmarks.optimizer \
  --sizes 100 500 1000 --frontier-points 40 \
  --timeout 60 --output results/benchmarks.json
```

The timeout is a benchmark safety budget per subprocess, not a limit in the app. It includes imports, model generation and verification as well as optimization. A timed-out or failed case is recorded and gives the benchmark a nonzero exit code. Sizes, point counts and the time budget can be changed explicitly.

Compare with the previous implementation on the same machine:

```bash
git show 277b4ed:efficient_frontier/core.py > /tmp/efficientfrontier-core-before.py
.venv/bin/python -m benchmarks.optimizer \
  --sizes 100 500 1000 --frontier-points 40 \
  --baseline /tmp/efficientfrontier-core-before.py \
  --output results/benchmarks-comparison.json
```

Add `--max-weight .01` to benchmark an optional 1% position cap; all selected sizes must make that cap feasible. Benchmark JSON is generated locally under ignored `results/` and includes the environment, timings, peak RSS and validation results.

References: [Clarabel Python interface](https://clarabel.org/stable/python/getting_started_py/), [Clarabel solver settings](https://clarabel.org/stable/api_settings/), and [CVXPY solver features and repeated solves](https://www.cvxpy.org/tutorial/solvers/index.html).
