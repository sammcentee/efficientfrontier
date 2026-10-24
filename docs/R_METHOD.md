# Comparison with the original R script

The corrected R-style calculation finds stock portfolios above SPY and QQQ on the original fit period. No index fund enters those portfolios. This supports the original historical result, but does not establish later outperformance.

## What v1.12 calculates

`Efficient Frontier v1.12.R` loads the 60 names in `spy_holdings.ods` and requests adjusted prices from 1 January 2020 to 1 January 2024. It drops stocks with missing returns, then samples ten million long-only portfolios. Each draw selects 5–40 stocks. Weights sum to one. There is no per-stock weight cap or covariance shrinkage.

The scatter contains feasible portfolios. Its upper edge approximates the efficient frontier, but random samples can miss better allocations. The Python solver directly maximizes mean return within a risk limit, or minimizes risk at each return target. For the same inputs and constraints, those are two ways to trace the same mean–variance frontier. Python does not impose R's 40-stock count limit. The four comparison portfolios below have 8–15 weights above 0.00001%, so that limit does not bind these results.

The R script registers parallel workers and imports `quadprog`, but its portfolio loop uses neither. Its sparse Dirichlet distribution strongly favors concentrated holdings. In a bounded reproduction with 10,000 draws, about 89% put at least 90% into one stock. This reproduction uses NumPy draws from the same distribution, not the identical R random sequence.

## Return formula correction

Line 38 calculates the correct daily arithmetic means:

```r
mean_returns <- colMeans(returns)
```

Line 39 replaces them with:

```text
(last adjusted price / first adjusted price) / number of price rows
```

The subtraction sits inside `length(...)`, so it does not reduce the row count. The numerator is gross growth, without subtraction of one. With flat prices over roughly four years, the displayed annual score is about 25% instead of zero.

Remove that overwrite to retain the daily arithmetic means. The corrected calculations are:

```text
daily return       = price[t] / price[t−1] − 1
annual mean        = 252 × average daily return
annual covariance  = 252 × sample covariance of daily returns
portfolio mean     = weightsᵀ × annual means
portfolio risk     = sqrt(weightsᵀ × annual covariance × weights)
```

The missing subtraction adds a common constant to every fully invested portfolio's score. It does not alone change the best portfolio at a fixed risk. However, that constant distorts Sharpe rankings. Replacing daily means with endpoint compounded growth can also change the selected stocks. The error invalidates the displayed return units. It does not prove that the stocks failed to beat an index.

## Reproduction on the original dates

The new download contains all 60 stocks and both ETF references, with 1,006 prices from **2 January 2020 to 29 December 2023**. It provides 1,005 daily returns. The historical ticker MMC maps to its current symbol MRSH. No stocks were excluded.

At each benchmark's measured volatility, the solver maximizes the corrected stock mean. All rows use sample covariance, full investment, no short sales, and no fees. These are fitted constant-weight statistics, not buy-and-hold CAGR.

| Reference risk | Benchmark annual mean | Stock mean, no holding cap | Stock mean, 10% holding cap |
| --- | ---: | ---: | ---: |
| SPY: 22.6271% | 13.6939% | 37.7464% | 34.1486% |
| QQQ: 27.2369% | 20.3686% | 48.0814% | 40.3875% |

The independent solver checks weights and actual portfolio-return variance. Its numerical upper-bound gaps are below 0.00005 percentage points of annual mean. The app's allocation routine independently matches all four results within 0.0001 percentage points.

The [result tables and complete stock weights](../research/r_reference/) retain both holding limits. The uncapped solutions put about 34% and 48% in LLY, respectively. The capped solutions obey the 10% limit. Neither includes SPY or QQQ.

The prices were downloaded again in October 2026. They can differ from the vendor's 2024 snapshot. The stock list was saved after the fit period. That introduces stock-selection hindsight. All weights use the same historical observations as their reported fit statistics. No result in this table is an unseen performance test.

## What changed in the app

- The main model uses ordinary sample covariance, as in R. The previous fixed 10% covariance adjustment is removed.
- The latest stock frontier and its holdings appear before the later historical test.
- SPY and QQQ are reference points on the same fit dates. Embedded ETF columns become benchmark inputs, with an explicit notice.
- The single-model API rejects SPY and QQQ as allocation candidates. Custom symbols are not a general security-type classifier.
- The later test retains prior-close decisions, trade fees, and its own realized mean, CAGR, and volatility.

The same model can have a strong fitted frontier and weaker later returns. For the current 500-stock input at a 23% budget and 30% cap, the latest fit gives a 54.10% annual mean. The later rule replay gives 30.61% mean at 23.35% volatility, while QQQ gives 33.47% at 18.52%. The app now shows those different questions separately.

## Reproduce the reference comparison

From the repository root:

```bash
.venv/bin/python research/reproduce_r_frontier.py \
  --prices /path/to/original-period/prices.csv \
  --benchmarks /path/to/original-period/benchmark_prices.csv \
  --output /tmp/r-frontier-reference
```

The reference script uses independent CVXPY problems, not the app's optimizer. It records hashes and writes the four comparisons and complete weights. Input prices remain local. `Rscript` was unavailable during this review, so this is a mathematical reproduction, not a native R execution. The archived R file remains unchanged for comparison.

Sources: [PerformanceAnalytics return calculation](https://search.r-project.org/CRAN/refmans/PerformanceAnalytics/html/Return.calculate.html) and [CVX portfolio optimization](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html).
