# Optimizer capacity and reproducible benchmarks

The active implementation is Rust. The Clarabel 0.11.1 crate solves the convex programs, with its `faer-sparse` feature. R remains only as historical source. No hardcoded maximum ticker count or holdings count exists. Allocations can use any nonempty subset of the supplied universe, subject to the long-only, fully invested model and any optional position cap.

This page measures computation time and memory. For investment performance against SPY and QQQ, see the [benchmark guide](BENCHMARKS.md).

## Rust and Python releases compared

We measured both releases on 2026-10-30 on one machine, with Linux/WSL2 and 16 logical CPUs. The Python release used NumPy 2.5.3, CVXPY 1.9.3, and Clarabel 0.11.1. The Rust release uses the Clarabel 0.11.1 crate with the `faer-sparse` feature, as the Python wheel does. Each value comes from a single run. The values are not averages or service guarantees.

Both benchmarks use the same design. Each case has synthetic annual means and a dense, full-rank covariance matrix from up to 24 factors plus a positive diagonal. The optimizer receives the complete dense matrix. The benchmark does not use an approximate risk model. Each case requests 40 frontier points and has no additional position cap.

| Assets | Python optimizer | Rust optimizer | Python peak RSS | Rust peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.169 s | 0.120 s | 197.3 MiB | 5.3 MiB |
| 500 | 4.498 s | 4.034 s | 271.6 MiB | 31.4 MiB |
| 1,000 | 22.236 s | 21.386 s | 475.4 MiB | 108.3 MiB |

The Rust optimizer took 29%, 10%, and 4% less time at 100, 500, and 1,000 assets. The Rust peak RSS was 3% to 23% of the Python value. The Python value includes the Python interpreter and its imported libraries.

The Rust benchmark makes its synthetic data with its own seeded generator. The construction is the same, but the random numbers are different. Thus the two benchmarks do not solve identical problems. Synthetic Sharpe values are test outputs, not evidence of investment returns.

Optimizer time covers `optimize()`. This includes input validation, problem construction, and the solver run. It does not include model generation, price downloads, historical estimation, charts, or report export. Peak RSS is the maximum resident memory of the whole worker process in both benchmarks. It is not the size of the covariance array alone.

Complete CLI runs took 0.06–0.63 s in Rust and 1.1–3.5 s in Python on the same machine. These runs were the demo, a CSV study, the backtests, and the historical demo. Most of the Python time was import time.

## Capacity limits

The model always uses the full covariance. Dense covariance storage is quadratic in the asset count. The factorization of dense systems is generally cubic. Memory, solver convergence, available history, and data-provider limits remain practical limits. The change to Rust did not remove these limits.

## Reproduce on Linux

Run this command from the repository root:

```bash
cargo run --release --example optimizer_benchmark -- \
  --sizes 100 500 1000 --frontier-points 40 \
  --timeout 60 --output results/benchmarks.json
```

Each case runs in a new worker process. The timeout is a safety budget for each worker process, not a limit in the app. It includes model generation and checks, as well as optimization. The benchmark records a timed-out or failed case. Such a case gives the benchmark a nonzero exit code. You can change the sizes, the point count, and the time budget.

Each case checks that all weights are finite and nonnegative, sum to one, and obey the cap. It also compares each frontier return with the return of its weights.

Add `--max-weight .01` to benchmark an optional 1% position cap. For each size, the size times the cap must be at least one. The benchmark writes JSON to the ignored `results/` folder. The JSON includes the environment, timings, peak RSS, and the results of the checks. Peak RSS is available on Linux only.

## Compare with the Python release

The Python benchmark is `benchmarks/optimizer.py` at commit `2e5f21c`. Check out that commit and install its packages, as the [research archive](../research/README.md#run-the-archived-scripts) shows. Then run this command from the root of that checkout:

```bash
.venv/bin/python -m benchmarks.optimizer \
  --sizes 100 500 1000 --frontier-points 40 \
  --timeout 60 --output results/benchmarks-python.json
```

The Python benchmark has a `--baseline` option for an archived `core.py`. The Rust benchmark does not have this option.

## Earlier Python measurement: 3 October 2026

This record describes the Python release. It compared the Python optimizer before and after a change for speed. The baseline was `efficient_frontier/core.py` at commit `277b4ed`. The environment was Python 3.14.7, NumPy 2.5.3, pandas 3.0.6, CVXPY 1.9.3, and Clarabel 0.11.1. The machine had Linux/WSL2 and 14 logical CPUs. Each case used one BLAS thread and one OpenMP thread.

| Assets | Previous Python optimizer | Updated optimizer | Previous peak RSS | Updated peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.333 s | 0.161 s | 199.7 MiB | 197.7 MiB |
| 500 | 9.644 s | 4.286 s | 303.1 MiB | 268.3 MiB |
| 1,000 | 29.769 s | 20.757 s | 601.3 MiB | 461.1 MiB |

A 2,000-asset case with 40 points did not complete in the 60-second budget. The value 2,000 is not an application cap. `docs/PERFORMANCE.md` at commit `2e5f21c` keeps the full record and the list of changes.

References: [Clarabel Rust interface](https://clarabel.org/stable/rust/getting_started_rs/), [Clarabel API documentation](https://docs.rs/clarabel/0.11.1/clarabel/), and [Clarabel solver settings](https://clarabel.org/stable/api_settings/).
