# SPY and QQQ benchmark comparisons

A benchmark gives a reference for historical portfolio results. The app compares one historical-mean portfolio with equal-weight stocks, SPY, and QQQ in one later replay and in five walk-forward test windows. The result status describes the replay comparison. It does not establish future outperformance. Separate legacy CLI and API tools retain the statistical analysis below.

The stock model and all three references buy once and hold. Equal-weight stocks start with equal weights across initially eligible stocks, then let those weights drift. In the replay, all rows use the same test dates, entry delay, and purchase fee rate. The walk-forward windows use the same rule but are measured from the purchase close, without the entry interval and fee. There are no later trades or terminal sales. None of the three references is scaled to the model's volatility budget.

Read each row's return and realized volatility together. Higher return with lower volatility means that reference beat the portfolio rule in this test. These later outcomes do not measure whether the optimizer found the best portfolio under its earlier estimates. The fitted frontier, fit table, and `fit_estimates.csv` show the estimates separately from the later test. SPY and QQQ frontier markers use the same historical dates as the stock fit.

## What the benchmarks represent

| Symbol | Reference market | Provider |
| --- | --- | --- |
| SPY | S&P 500 | [State Street SPDR S&P 500 ETF Trust](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy) |
| QQQ | Nasdaq-100 | [Invesco QQQ ETF](https://www.invesco.com/qqq-etf/en/home.html) |

The app uses adjusted ETF prices, rather than raw index levels. ETF returns reflect fund expenses and can differ from index returns through tracking differences. Provider price adjustments represent distributions and corporate actions. The app does not independently reconstruct those adjustments. The [SPY provider](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy) and [QQQ provider](https://www.invesco.com/qqq-etf/en/home.html) describe their funds and costs.

SPY and QQQ are comparison instruments in the app and its stock-model CLI path. Input columns with either symbol remain outside the stock allocation, with an explicit note. Their prices can supply the comparisons. The stock holding cap does not restrict these benchmark paths. The separate legacy optimizer can still analyze fund portfolios.

## Price coverage and currency

Benchmark prices must cover every supplied asset date. The app does not fill missing benchmark prices or shorten the asset history to make a comparison possible. This requirement also applies to weekly or monthly input. The annualization setting does not change the supplied dates.

| Asset source | Benchmark source |
| --- | --- |
| Combined SPY/Nasdaq-100 stocks or Nasdaq-100 | Reuse the SPY/QQQ prices that define the complete market calendar for constituent eligibility. |
| Other Yahoo inputs | Reuse any SPY/QQQ input columns. If neither is present, download both references separately. |
| CSV | Use a separate benchmark file, or reuse any SPY/QQQ input columns. |
| CSV with an explicit download choice | Download both references when no supplied benchmark file or input column is available. |
| Synthetic demo | No comparison with actual market benchmarks. |

A separate benchmark CSV must contain exactly the asset dates and SPY, QQQ, or both. Extra dates or asset columns are invalid. One reference can appear on its own, but both are required for the historical comparison checks to pass. The separate legacy CLI requires both columns. Yahoo downloads select each exact asset date from the downloaded market dates.

CSV analysis remains offline unless you explicitly request a download. A separate benchmark file does not add its columns to the investment universe. The benchmark labels identify the supplied series. They do not verify that a file contains genuine SPY or QQQ prices.

These comparisons require asset and benchmark prices in USD. The currency declaration records your assumption. It does not verify the data or perform currency conversion. A portfolio in another currency needs consistent conversion before this comparison. Otherwise, exchange-rate changes can distort the result.

If benchmark data is unavailable or invalid, the report gives the reason. Other portfolio analysis remains available. A non-USD declaration disables the benchmark comparison without currency conversion.

Both stock-universe loaders also need SPY and QQQ dates to check constituent history. They download these series even when you disable benchmark comparisons. Failure to obtain this calendar stops the stock-universe analysis. The [universe coverage rules](USER_GUIDE.md#stock-universe) explain exclusions and the current-membership limitation.

## App use

1. Select **Settings**, then open **Data options**.
2. Enable **Compare with the S&P 500 and the Nasdaq-100**.
3. Set **Price currency** to **USD** only when every asset price uses USD. Otherwise, select **Other currency**.
4. Select **Build portfolio**.
5. Read **Historical replay · buy and hold**, or press **2** when no text field has focus.
6. Open **Data & method**, or press **4**, for assumptions, coverage, and exact trades.

For an offline CSV comparison, use **Market prices (optional)** to supply a separate benchmark file. Otherwise, the app reuses SPY/QQQ columns from the stock file. To download references when neither source is supplied, select **Download SPY and QQQ from Yahoo Finance for this file**.

## Command-line use

Use `--target-volatility` for the app's stock model. Set a feasible holding limit for the supplied asset count.

```bash
# Current SPY and Nasdaq-100 stocks, with automatic benchmark prices
.venv/bin/python -m efficient_frontier \
  --sp500-nasdaq100 --target-volatility .20 --currency USD

# Offline stock and benchmark files on identical dates
.venv/bin/python -m efficient_frontier --csv data/prices.csv \
  --target-volatility .20 --benchmark-csv data/benchmarks.csv --currency USD

# Explicit Yahoo benchmark download for a CSV portfolio
.venv/bin/python -m efficient_frontier --csv data/prices.csv \
  --target-volatility .20 --download-benchmarks --currency USD

# Stock study without ETF comparisons
.venv/bin/python -m efficient_frontier --csv data/prices.csv \
  --target-volatility .20 --no-benchmarks
```

`--currency` defaults to `USD`. It declares the common price currency without verification or conversion. `--benchmark-csv`, `--download-benchmarks`, and `--no-benchmarks` are mutually exclusive. A supplied benchmark file or explicit download takes precedence over SPY/QQQ asset columns. The CLI rejects explicit market benchmarks for synthetic demo data.

## The later test and its result status

The app fits the model once on the first 70% of return observations and tests it on the remaining 30%. It buys at the next close, then holds without rebalancing. The fit uses prices only through the preceding close.

The test compares **Stock portfolio · buy and hold**, **Equal-weight stocks**, **S&P 500 (SPY)**, and **Nasdaq-100 (QQQ)**. All paths keep their initial adjusted-price units. Their weights drift with prices. The three reference paths have no volatility budget. All paths use the same dates and fee rate.

A pass requires both ETFs, at least one declared year of test returns, and model realized volatility within the budget. No reference can have higher net CAGR with equal or lower realized volatility. A tie or a lower-return, lower-risk result can pass. These checks do not establish a statistical advantage and do not change the model rule.

Latest holdings use all supplied prices, including the replay dates. They are a new fit of the same model, separate from the initial allocation held in the replay. Current membership, complete-history requirements, and prior review of the test dates can bias the comparison.

## Matching entry dates and costs

The comparison uses the same capital base and evaluation dates for each strategy and benchmark.

| Evaluation | Passive benchmark accounting |
| --- | --- |
| App buy-and-hold replay | Hold cash through the first test interval. Buy at its close with the same 10-basis-point fee rate as the model. Hold without further trades. |
| Legacy original holdout | Allocate at the final training close. Hold through the evaluation period without trade fees. |
| Legacy four-method backtest | Hold cash through the first test interval. Buy at its close with the selected fee rate. |

For the stock-model test and legacy backtests, initial capital is one. With `c = cost_bps / 10000`, entry leaves `1 / (1 + c)` invested in the benchmark. This is a self-financing purchase: the fee comes from the same initial capital. The benchmark then follows adjusted-price growth without further trades. No final sale occurs. The app applies the same accounting to its stock model and equal-weight reference.

The selected trading fee is separate from fund expenses already reflected in ETF prices. The app does not subtract the ETF expense ratio again. These calculations exclude taxes, currency conversion, and a separate market-impact model.

## Exports

In the app, select **Export**, or press **E** when no text field has focus. The bundle contains the later replay, its descriptive status, and the walk-forward windows.

| File | Contents |
| --- | --- |
| `benchmark_prices.csv` | Benchmark prices on every asset date, when present |
| `test/metrics.csv` | Performance of the model and all available references |
| `test/equity.csv` | Corresponding paths through the later test |
| `test/NN_allocations.csv` | Each path's initial target weights |
| `test/NN_holdings.csv` | Each path's asset exposure and profit contributions |
| `test/NN_trades.csv` | Each path's initial purchase date, turnover, and entry fee |
| `out_of_sample/metrics.csv` | Walk-forward results for the model and all available references, when at least one window has a feasible fit |
| `metadata.json` | Settings, model revision, sources, test reasons, walk-forward settings, and numbered file map |

The report also retains exact stock prices, latest fitted weights, and eligibility records. See the [complete app export list](BACKTESTING.md#app-exports). The statistical files below belong to the separate legacy tools.

## Legacy statistical research

The remaining sections describe the earlier CLI and Python API. These calculations do not run in the stock-model app or its `--target-volatility` CLI path.

```bash
# Separate legacy strategy comparison with statistical evidence
.venv/bin/python -m efficient_frontier --csv data/prices.csv \
  --benchmark-csv data/benchmarks.csv --backtests --currency USD
```

Omit `--backtests` for the legacy cost-free holdout. Both commands omit `--target-volatility`. The legacy analysis compares several strategies and computes uncertainty estimates. These choices do not change the app's model.

### Estimates and later evaluation

The original efficient frontier uses the training sample. Latest model holdings use all supplied history. Both are fitted estimates. Benchmark markers use the same fit period as the corresponding frontier. Neither supplies evidence of future benchmark outperformance.

Historical backtests fit each strategy with information available before its execution date. Benchmark evidence uses their later evaluation returns. The legacy CLI uses the original cost-free holdout when you omit `--backtests`. The report identifies which evaluation supplies the evidence.

The comparison also divides the evaluation returns into up to three chronological windows without overlap. Fewer than three returns give fewer windows. These evaluation windows differ from the three estimation windows in the profile model. A result that changes sign across evaluation windows shows period sensitivity. The windows do not form three independent experiments.

Each window reports relative wealth growth: `product(1 + r_s) / product(1 + r_b) - 1`. This ratio compares compound growth factors. It differs from the subtraction in the full-period total-return difference. The first evaluation window retains the entry effect. The report flags windows with less than one model year of observations.

### Return differences and alpha

Let `r_s` be a strategy return, `r_b` a benchmark return, `p` observations per year, and `rf` the annual risk-free assumption.

| Measure | Meaning |
| --- | --- |
| Total-return difference | Strategy compound return minus benchmark compound return over the full evaluation |
| CAGR difference | Strategy annualized compound growth minus benchmark annualized compound growth |
| Annual mean advantage | `mean(r_s - r_b) * p` |
| Benchmark alpha | Annualized intercept from the excess-return regression below |
| Benchmark beta | Estimated sensitivity to benchmark excess returns in that regression |
| Tracking error | Sample standard deviation of `r_s - r_b`, multiplied by `sqrt(p)` |
| Information ratio | Annual mean advantage divided by tracking error |

Tracking error and the information ratio use the same paired observations as inference. They are descriptive measures and do not add hypothesis tests. Zero tracking error makes the information ratio undefined.

The mean advantage is an arithmetic return difference. It does not equal the CAGR difference. A strategy can have a positive arithmetic advantage and a weaker compound result because volatility affects compound growth.

The alpha regression is:

```text
r_s - rf / p = alpha_per_period + beta * (r_b - rf / p) + error
annual_alpha = alpha_per_period * p
```

Alpha measures the average return unexplained by this fitted benchmark relationship. This is a single-factor model. It does not control for every risk exposure or establish a causal source of returns. The intercept approach follows the performance-measurement framework in [Jensen's original paper](https://doi.org/10.1111/j.1540-6261.1968.tb00815.x).

Full-path performance and the three evaluation windows retain the initial entry effect. Backtest inference excludes the first cash-and-entry return because that interval contains no asset exposure before execution. Original holdout inference includes every holdout return because it has no delayed cash interval.

### Uncertainty estimates

Inference requires at least **60 paired returns after the backtest entry interval**. For the original holdout, it requires at least 60 paired holdout returns. This minimum is a product guardrail, not proof that a sample is sufficient. Small samples, changes in market conditions, or weak benchmark variation can still limit the estimates.

The legacy analysis fits a constant to paired return differences for the mean advantage. It fits an intercept and a benchmark excess-return coefficient for alpha. Both use Newey–West heteroskedasticity and autocorrelation consistent (HAC) standard errors. This method accounts for unequal error variance and serial dependence under its assumptions. It needs consistently spaced observations and sufficiently stable return relationships. [Newey and West](https://www.nber.org/papers/t0055)

For `n` paired observations and `k` fitted coefficients, the implementation uses:

```text
maximum_lag = floor(4 * (n / 100) ** (2 / 9))
Bartlett_weight(lag) = 1 - lag / (maximum_lag + 1)
finite_sample_correction = n / (n - k)
```

Here, `k = 1` for the mean advantage and `k = 2` for alpha. Lags count observations, not calendar days. The default lag rule and Bartlett weights follow the [statsmodels HAC convention](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html). The legacy analysis uses a normal approximation for two-sided p-values and individual 95% confidence intervals.

A p-value tests a zero mean advantage or zero alpha under the specified assumptions. It is not the probability that the strategy will win, or that its success was due to chance. The confidence interval describes uncertainty in an estimated historical relationship. It is not a range of guaranteed future returns.

Undefined uncertainty estimates and p-values remain blank. Descriptive point estimates can still appear. Examples include insufficient observations or a regression without enough independent variation. The legacy analysis does not convert an undefined test into evidence of skill.

### The full comparison family

The Holm adjustment covers **all strategies, both benchmarks, and both statistical measures in the current evaluation**. With 24 strategy combinations, the family contains `24 * 2 * 2 = 96` tests. With the original 12 combinations, it contains 48 tests. A holdout-only report uses its available holdout portfolios.

The adjustment orders the raw p-values and applies the Holm step-down correction. Undefined tests count as p-values of one for the correction, but remain undefined in the displayed results. Chart and table selections do not reduce this family. [Holm's original paper](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf)

The 95% confidence intervals are individual intervals. They do not include the Holm correction and are not simultaneous intervals for all 96 tests. A positive estimate alone does not establish a statistically supported advantage. The direction of the estimate and the adjusted p-value both matter.

The family covers only this run. It does not account for earlier searches across dates, assets, costs, strategies, or unpublished experiments. Repeated selection can create an attractive backtest even without a persistent advantage. [Bailey and coauthors on backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

### Statistical evidence status

Statuses use Holm-adjusted p-values at the 5% level:

| Status | Meaning |
| --- | --- |
| Historical mean advantage and alpha | Positive mean advantage and positive alpha both pass the adjusted tests |
| Historical mean advantage | Positive mean advantage passes, but positive alpha does not |
| Historical alpha | Positive alpha passes, but a positive mean advantage does not |
| Historical mean underperformance | Negative mean advantage passes the adjusted test |
| No clear statistical advantage | Neither positive measure passes the adjusted test |
| Insufficient history | Fewer than 60 paired observations remain for inference |
| Uncertainty unavailable | Neither measure has a usable uncertainty estimate |

These labels summarize statistical results. They do not rank future prospects. A negative mean advantage that passes its test takes precedence over a positive alpha label.

### Legacy exports

Legacy command-line reports contain the following files. These are separate from the app exports above.

| File | Contents |
| --- | --- |
| `benchmark_prices.csv` | Benchmark prices on every asset date |
| `benchmark_holdout_metrics.csv` | Cost-free holdout performance |
| `benchmark_holdout_curve.csv` | Cost-free holdout equity paths |
| `benchmark_training_estimates.csv` | Benchmark estimates from the original training sample |
| `benchmark_latest_estimates.csv` | Benchmark estimates from the complete input history |
| `benchmark_backtest_metrics.csv` | Passive results with the backtest entry delay and costs, when backtests are present |
| `benchmark_backtest_curve.csv` | Corresponding passive backtest equity paths |
| `evidence_summary.csv` | Every strategy/benchmark pair, estimates, intervals, p-values, and status |
| `evidence_windows.csv` | Dates, relative growth, and mean advantage for each evaluation window |
| `evidence_relative_curve.csv` | Strategy wealth divided by benchmark wealth, with both paths initially equal to one |

`metadata.json` records the benchmark assumptions, evidence settings, sample counts, test-family size, and warnings. Chart selections do not remove rows from these exports.

### Review the legacy evidence

1. Check the source, USD declaration, coverage, evaluation dates, and entry costs.
2. Compare the strategy with both SPY and QQQ over the full evaluation.
3. Compare the three evaluation windows for changes in the return advantage.
4. Inspect the mean advantage, alpha, confidence intervals, and Holm-adjusted p-values together.
5. Review concentration, turnover, drawdown, and benchmark beta alongside returns.
6. Freeze the assets, rules, costs, and evaluation plan before the next unseen period.

The legacy report initially highlights Expanding window · Medium when profiles are available. This choice does not use the highest observed return. A stronger case for persistence needs fixed rules and later data that did not influence those rules. These statistical results do not promise future outperformance.
