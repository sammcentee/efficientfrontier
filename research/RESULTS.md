# Equal-volatility index research: 18 October 2026

A simple 75% SPY / 25% stock-momentum rule beat both benchmarks in the recent historical comparison. The evidence does not establish reliable outperformance. The stock data have survivor bias, the rule follows earlier result review, and every simultaneous lower bound is negative.

## Proposed rule

1. Hold 75% in SPY.
2. At each completed month-end, rank eligible stocks by their prior 11-month adjusted return, excluding the latest month. The endpoints are month-end m−12 and m−1.
3. Put 25% equally into the top 50 stocks: 0.5% of the total portfolio per selected stock.
4. Execute at the next trading close. Hold those shares until the next monthly rebalance.

The test uses the existing 500-stock universe from the S&P 500 and Nasdaq-100. SPY can also contain the selected stocks, so their total economic exposure exceeds the direct 0.5% weight. A production index needs historical constituent records and reliable delisting returns. The current data do not provide that evidence.

The rationale is diversification: the broad core can reduce the concentrated momentum sleeve's volatility. The 75/25 weight is a fixed hypothesis, not a solved maximum-return portfolio. No result proves that it lies on the true future efficient frontier.

## Recent stock result

Dates: **17 April 2025 to 16 October 2026**, 377 daily returns. Costs: 10 basis points per amount bought or sold. Means and volatility use 252 observations per year. CAGR measures compound growth.

| Portfolio | Actual volatility | Mean at equal volatility | CAGR at equal volatility |
| --- | ---: | ---: | ---: |
| 75% SPY / 25% stock momentum | 15.87% | 27.35% | 30.38% |
| SPY | 12.75% | 26.85% | 29.73% |
| QQQ | 18.52% | 23.07% | 24.93% |

All comparison paths have **12.7457% measured annual volatility**. The normalized annual mean advantage is **0.5013 percentage points over SPY** and **4.2769 points over QQQ**. At 25-basis-point trading costs, the advantages fall to **0.2854** and **4.0610** points.

This is retrospective normalization. The diagnostic multiplies each completed net return path by the lowest observed volatility divided by its own volatility. No path uses leverage. The scale uses the completed evaluation period, so it is not a forecast or an executable guarantee. The diagnostic excludes additional trades to maintain those scaled exposures. The actual fully invested rule had 15.87% volatility.

## Longer fund result

A separate rule holds 75% SPY and 25% SPMO, reset monthly. SPMO uses a different momentum signal, selection count, weighting method, and review schedule. It is an imperfect fund proxy, not validation of the custom stock rule.

Dates: **3 January 2023 to 16 October 2026**, 951 daily returns. At 10-basis-point costs and **14.8744% equal measured volatility**:

| Portfolio | Actual volatility | Mean at equal volatility | CAGR at equal volatility |
| --- | ---: | ---: | ---: |
| 75% SPY / 25% SPMO | 15.70% | 21.13% | 22.16% |
| SPY | 14.87% | 20.67% | 21.61% |
| QQQ | 20.08% | 21.15% | 22.20% |

The fund rule beats SPY by **0.4543 percentage points** but trails QQQ by **0.0283 points**. Higher costs increase the QQQ shortfall to **0.0431 points**. It fails against both in 2023, wins against both in 2024 and 2025, and loses to SPY in partial 2026. That variation limits any persistence claim.

## Retain the failed original experiment

The first protocol fixed five stock rules, six index funds, and one QQQ trend rule before their new results. It selected a stock-momentum rule and a QQQ trend rule from development data. Selection and code hashes remain in `selection_lock.json`.

The original comparison applied a monthly 10% volatility target to every candidate and both benchmarks. Momentum beat those controlled benchmarks in the stock evaluation. However, the benchmark control reduced their performance. It did not establish a win over actual passive SPY and QQQ.

After correction, **none of the original raw or controlled candidates beats both passive benchmarks**. The 75/25 rule was declared after that finding. The fully invested version was fixed as primary before its own results. Its secondary 10% volatility-control variant loses to both passive benchmarks in both datasets. No further weights or variants were tested.

## Uncertainty and data limits

The paired stationary bootstrap uses 10,000 repetitions, seed 20261018, a primary 21-day mean block length, and 10/63-day sensitivities. Each sample retains shared dates and recomputes risk normalization. The final families include 12 stock implementations and 16 fund implementations, each against two passive benchmarks.

Simultaneous 97.5% bounds per family target approximate 95% coverage across both datasets, subject to bootstrap assumptions. **Every lower bound is negative**, including the core rule at every block length. This does not establish a positive edge. These bounds also do not remove adaptive rule design, survivor bias, or earlier research searches.

The uncertainty tables use one common risk across all trials: 8.7323% for stocks and 9.3406% for funds. The headline tables use a separate common risk across their three displayed paths. Those positive common scale changes preserve the sign and ranking of annual-mean gaps, but change their percentage-point size.

The recent stock universe contains current constituents with complete price histories. Those filters can favor survivors. Unresolved member quotes and corporate actions in the long local stock corpus prevent its use for confirmation. Actual fund prices cover October 2015 onward, with development through 2022 and later evaluation. Fund selection and historical index rule changes remain limitations. See `fund_methodologies.json` for primary sources and material changes.

Fund prices come from Yahoo Finance through yfinance. Vendor adjustments are not independently certified. `fund_prices_provenance.json` records the request, data checks, and hashes.

Cash earns zero. Fund expenses are already reflected in adjusted prices. The tests exclude taxes and currency conversion consistently. Costs cover recorded strategy trades, not the extra trades implied by retrospective risk scaling.

## Files and validation

- `results/core/`: every follow-up cost and risk variant, annual tables, latest target weights, all-trial passive comparisons, uncertainty, and code hashes.
- `results/original/`: original development/evaluation summaries, including the benchmark definition that required correction.
- `results/audit_original.json` and `results/audit_core.json`: independent replay results with artifact hashes.
- `results.html`: interactive comparison using the same derived tables.

Independent replay checks reconstruct past-only signals, target weights, next-close trades, fees, cash, and daily NAV. Independent statistical checks reproduce the displayed tables and primary bootstrap. The test suite also checks future-data isolation and input/code locks. The research does not change the app's model.

The next useful evidence is a test on a complete historical membership dataset or an unseen period, with this rule fixed in advance. More optimization of these same dates cannot supply that evidence.
