# Model, historical test, and accounting

The app fits stock weights from historical means and covariance, then tests earlier fits on later prices. The CLI runs the same model with `--target-volatility .20`. Earlier frontier and backtest APIs remain separate research tools.

## Portfolio model

Set an annual volatility budget and a per-stock holding limit before the calculation. The app defaults to 20% volatility and 10% per stock. Stock weights are long-only and sum to one. An infeasible limit stops the calculation.

Each fit estimates annual arithmetic means `mu` and covariance `Sigma` from the available earlier returns. The app uses the ordinary sample covariance, annualized by the supplied price frequency. The model solves:

```text
maximize: mu @ w
subject to: sum(w) = 1
            0 <= w <= holding_limit
            w.T @ Sigma @ w <= volatility_budget ** 2
```

Historical arithmetic mean is the only return objective. A higher fitted mean does not establish higher future returns. The volatility budget constrains the estimate, not future realized volatility or losses.

This is a convex mean-variance problem for the supplied estimates and feasible stock weights. The objective is the fitted mean at a volatility ceiling. Sharpe and CAGR are separate metrics. See the [MOSEK mean-variance formulation](https://docs.mosek.com/portfolio-cookbook/markowitz.html). Estimation error can make optimal fitted weights perform poorly later. See [MOSEK's estimation-error discussion](https://docs.mosek.com/portfolio-cookbook/estimationerror.html).

The budget constrains `sqrt(w.T @ Sigma @ w)`. With ordinary sample covariance, this equals the annual sample volatility of the same fixed-weight return series. The later replay uses different dates and changing holdings, so its realized volatility can differ.

The latest fitted frontier uses all supplied stock history and the holding cap. Its selected point applies the requested volatility budget. SPY and QQQ markers use historical means and volatility on those same dates. These markers are separate comparisons, not feasible stock portfolios.

The CLI accepts an explicit `--shrinkage` value for research. With value `a`, it uses `(1 - a) * S + a * diag(diag(S))`, where `S` is sample covariance. A nonzero value can raise or lower fitted portfolio risk relative to ordinary historical volatility. The app and stock-model CLI default to zero. The legacy CLI defaults to 0.1.

With `T` return intervals, the first `floor(0.7 × T)` form the initial training sample. The remaining 30% form one later test. Every 21 observations, the model refits with all returns through the close before execution. The same rule applies throughout. Latest weights use all supplied history and never enter earlier trades.

The test has four paths:

| Path | Rule |
| --- | --- |
| Historical-mean portfolio | Refit stock weights every 21 observations. |
| Equal-weight stocks | Hold equal target weights across eligible stocks, with the same eligibility checks and trade schedule. |
| S&P 500 (SPY) | Buy once and hold. |
| Nasdaq-100 (QQQ) | Buy once and hold. |

The three references have no volatility budget. Equal weights satisfy the holding cap when the stock model is feasible. ETF weights remain separate from the stock holding cap. Missing benchmark data leaves only the available paths and prevents a pass.

All paths share the test dates, initial cash interval, and fee rate. The app charges 10 basis points on bought and sold amounts. It does not charge a terminal sale. See [trade timing](#timing-and-information) and [fee accounting](#costs-and-constraints).

The result status is a descriptive check of that test period. A pass requires both ETFs, at least `ceil(periods_per_year)` test intervals, and model volatility within the budget. It also requires that no reference has strictly higher net CAGR with no more realized volatility. Equal returns can pass. A lower-return, lower-risk model can also pass. This status does not mean the model beats every reference, and it does not establish statistical significance or future skill.

At every fit, the engine excludes stocks with at least 20 consecutive unchanged price intervals in the past sample. It preserves the original prices and exports each dated eligibility decision. Repeated quotes do not prove false data. If a held stock later triggers the screen, the study stops. It does not assume a verified sale at that quote.

Current membership and complete-history requirements create selection and survivorship bias. Prior review of the test dates also creates research hindsight. Correct time order alone does not make those dates fresh prospective evidence.

## App exports

The ZIP contains an offline HTML report and exact CSV data. `metadata.json` records the model revision, settings, initial training dates, test dates, sources, warnings, and test reasons.

| File | Contents |
| --- | --- |
| `prices.csv` | All original stock prices, including excluded columns |
| `benchmark_prices.csv` | Supplied benchmark prices, when present |
| `eligibility.csv` | Dated stock eligibility decisions |
| `latest_weights.csv` | Latest stock weights |
| `latest_estimates.csv` | Latest fitted estimates |
| `latest/frontier.csv` | Fitted return and risk at each frontier point |
| `latest/frontier_weights.csv` | Stock weights at each frontier point |
| `latest/fitted_comparison.csv` | Selected stock portfolio and SPY/QQQ estimates on the same latest fit dates |
| `fit_estimates.csv` | Each historical and latest fit's dates, sample size, mean, model risk, ordinary historical risk, and largest target weight |
| `test/metrics.csv` | Net performance for each path |
| `test/equity.csv` | Exact portfolio values through the later test |
| `test/NN_allocations.csv` | Target weights for one path |
| `test/NN_holdings.csv` | Asset exposures and profit contributions |
| `test/NN_trades.csv` | Fits, trade dates, turnover, and fees |
| `universe_coverage.csv` | Source coverage and exclusions, when supplied |

`metadata.json` maps each numbered file prefix to its path name. Keep the input CSVs and all recorded settings to reproduce the calculation. The CLI accepts explicit assumptions even though the app fixes its training share, refit schedule, fees, and shrinkage.

## Timing and information

1. Split the observed returns in time order. Use the first `floor(train_fraction × T)` for the initial fit.
2. Set initial wealth to one at the final training close. Hold cash until the next supplied close. Deduct entry fees there.
3. At each execution close, value existing holdings first. Fit new targets with returns only through the preceding close. Deduct fees and establish the new holdings.
4. Let new targets earn returns from the next interval. Refit every `rebalance_every` observations. Do not add an unused trade at the final close.

This one-interval delay prevents a signal from receiving a return already in its fit. Execution still assumes the next supplied adjusted close, fractional holdings, and unlimited capacity. Fees approximate spreads, slippage, and commissions. The model has no market-impact, tax, or currency-conversion calculation.

The schedule counts supplied rows. Twenty-one observations approximate one month only for daily market data. With weekly data, a CLI interval of four means four weekly observations. Missing dates can make these intervals longer.

`--periods-per-year` controls annualization and defaults to 252. It does not resample prices or change trade dates. The app uses daily market prices and 252 observations per year for Yahoo input. CSV presets are 252, 365, 52, and 12. Custom app values must be finite and at least one. The CLI accepts any positive finite value.

The model needs at least two initial training returns and two later returns. Five price rows meet that minimum only with a suitable split. The app's fixed 70/30 split needs at least six rows. Short histories and large asset counts make estimates less reliable.

## Legacy strategy comparison

The CLI's `--backtests` option compares four execution methods. Its original targets are minimum volatility, maximum Sharpe, and equal weight. The legacy CLI can also include Low, Medium, and Extreme profiles. These choices are available only in the separate legacy tools.

Every asset in the accepted legacy price dataset remains eligible. The stock-universe loaders first record members without complete prices. The [coverage rules](USER_GUIDE.md#stock-universe) explain this step and its historical selection bias.

| Method | Target weights | Estimation sample |
| --- | --- | --- |
| Buy and hold | Allocate once, then let weights drift | Initial training returns |
| Fixed rebalance | Restore the initial targets every N observations | Initial training returns |
| Expanding window | Estimate new targets every N observations | All returns available before the execution close |
| Rolling window | Estimate new targets every N observations | Latest L returns available before the execution close |

The original targets give 12 method/portfolio combinations. With the three profiles, the comparison contains 24 combinations. Equal-weight targets do not depend on estimated returns, so the fixed, expanding and rolling equal-weight paths intentionally coincide. This is not three independent pieces of evidence. The original **holdout** remains a separate, cost-free calculation with its original entry timing. The legacy report retains this original holdout separately.

The legacy `--rolling-window L` setting counts observed returns. It needs at least two returns and cannot exceed the initial training length. Its default is the initial training length. A shorter window changes both the initial allocation and later allocations. Profile comparisons require at least six returns per fit.

<a id="latest-model-holdings-and-profiles"></a>

## Legacy model holdings and profiles

The legacy frontier model uses historical arithmetic mean return. Latest holdings fit all supplied returns through the final price date. For weights `w`, the fitted annual mean is `mu @ w`, where `mu` is the annualized arithmetic mean of each asset over the complete fit period. This estimate is not compound growth (CAGR) or a forecast.

Covariance uses the same fit period and the selected shrinkage. Weights are long-only, sum to one, and obey the chosen per-asset cap. For each arithmetic mean return target, the frontier minimizes portfolio variance:

```text
minimize: w.T @ covariance @ w
subject to: mu @ w >= target
            sum(w) = 1
            0 <= w_i <= max_weight
```

The profiles use the same selection rule as the classic mean-return frontier, with their own fit dates:

| Profile | Position on the frontier |
| --- | --- |
| Low | Least fitted volatility |
| Medium | Sampled point nearest the midpoint of Low and Extreme volatility |
| Extreme | Endpoint with the highest fitted arithmetic mean return |

Earlier app versions called Extreme **Highest**. These labels are relative to the supplied assets and model. Highest does not add leverage or seek the largest possible variance. When the frontier has no volatility range, all three profiles use the same highest-return mix. Numerical ties can produce several equivalent allocations.

The three chronological, nonoverlapping windows remain diagnostic checks. Their observation counts are nearly equal. Each needs at least two returns, so profiles need at least six returns in total. Window dates identify the first and last return observations.

Each annualized window mean is `mu_window @ w`. `worst_window_return` is the smallest of those three portfolio means. These diagnostics do not select primary weights. They are not confidence bounds or forecasts.

Reports warn when a window has fewer observations than the selected periods per year. They also identify profiles without distinct risk levels. Solver warnings remain visible, and tables report the actual estimates of the computed weights.

**Latest model holdings use an in-sample fit.** Their `as_of` date is the final supplied price date, not a guarantee of current market quotes. The classic frontier uses the same historical-mean objective on the initial training sample. Its holdout evaluates that initial allocation on later dates.

Historical profile backtests use the same historical-mean objective, but each fit uses only returns through the close before execution. Fixed rebalance retains the initial profile targets. Expanding and rolling methods rebuild their profiles at each scheduled fit. Latest full-history weights never enter those earlier trades. Three-window diagnostics also use only each fit's history.

The initial sample and rolling window each need at least six returns for the complete profile comparison. If either is too short, the legacy CLI retains the original 12 combinations and explains why it omitted the profiles. A latest full-history fit can still be available when the earlier backtest fit is too short.

The Python API keeps `include_profiles=False` as its compatibility default. `run_backtests(..., include_profiles=True)` requests the additional profiles. Direct API calls reject shorter fits. The legacy CLI uses the fallback above.

### Optional weakest-window comparison

The Python API retains a separate weakest-window comparison through `build_profiles(..., return_objective="worst_window")`. This is separate from the app's stock model.

This objective maximizes the lowest of the three portfolio window means. At each frontier target, it minimizes variance subject to `mu_window @ w >= target` in every window. Low uses the minimum-variance endpoint. Medium uses the midpoint of the endpoint return targets. Highest uses the maximum feasible weakest-window target, with minimum variance among tied solutions. Medium therefore uses a return midpoint in this optional model.

For this optional objective, the solver allows return-target slack of `1e-8 * S`. Here, `S` is the largest absolute annual asset mean across the three windows, or one if all means are zero. The final check rejects target shortfalls greater than `2e-8 * S`. This preserves the minimum-variance tie-break near numerical ties. A warning identifies cases where no feasible portfolio has positive estimated means in all three windows.

The legacy report labels this optional chart **Weakest-window frontier**. Its ETF markers use the same lowest-window-mean measure. The legacy **Historical mean frontier** and its ETF markers use full-history arithmetic means. Neither chart guarantees future returns or minimum drawdown.

## Costs and constraints

All targets are long-only and fully invested after execution. A weight cap applies at each trade. Price movements can take actual weights above it. The risk-free rate is a constant annual assumption for Sharpe and Sortino. The app fixes it at 2% for exported metrics. It does not affect the stock-model weights or create an investable cash strategy.

Let `V` be wealth before a trade, `h` the current dollar holdings, `w` the target weights, and `c = cost_bps / 10000`. Post-trade wealth `V_after` solves:

```text
V_after + c × sum(abs(V_after × w - h)) = V
```

Fees charge both bought and sold notional. Entry from cash leaves `1 / (1 + c)` invested. A complete switch between two assets leaves `(1 - c) / (1 + c)` of pretrade wealth. There is no hidden division by two in turnover.

**Gross turnover** sums each trade's absolute notional divided by its own pretrade wealth. **Total fees** sum fee payments in units of initial capital. These denominators differ as wealth changes. Fees are also different from the full performance gap against a zero-fee counterfactual, which includes foregone compounding.

In legacy backtests, when no feasible maximum-Sharpe portfolio has positive estimated excess return, that strategy uses minimum-volatility weights for that execution. The warning and fallback count remain in the results. The engine retains the interval. Fixed rebalancing continues to restore its original target, including any original fallback.

## Passive benchmarks and evidence

SPY and QQQ comparisons use the same evaluation dates as each strategy. The backtest benchmarks stay in cash through the first evaluation interval, then buy at its close. The selected entry cost leaves `1 / (1 + cost_bps / 10000)` invested. The original holdout benchmarks enter at the final training close without costs. Neither benchmark path includes a final sale.

Benchmarks remain separate from the eligible assets and portfolio weight cap. The separate legacy statistical comparison uses all strategies in its run, even when chart selections display fewer curves. Latest model holdings remain an in-sample fit. [Benchmark methods and evidence](BENCHMARKS.md) explains the USD assumption, data coverage, HAC estimates, Holm correction, and limits on inference.

## Return and risk metrics

The app test and legacy backtests use the same metric definitions. Backtest returns include the selected fees. Let `p` denote observations per year, `r` the observed portfolio returns, and `rf` the annual risk-free rate.

| Metric | Definition |
| --- | --- |
| CAGR | Compound growth, annualized with `p` and the number of observed returns |
| Volatility | Sample standard deviation of `r`, multiplied by `sqrt(p)` |
| Sharpe | `(mean(r) * p - rf) / annual_volatility` |
| Sortino | `(mean(r) * p - rf) / (sqrt(mean(min(r - rf / p, 0)^2)) * sqrt(p))` |
| Calmar | `CAGR / abs(max_drawdown)` |

Sortino measures downside relative to the risk-free rate per observation. Its denominator includes all observations, with zero contributions from returns at or above that threshold. Calmar compares compound growth with the largest observed fall from a previous peak. A zero denominator makes these ratios undefined. The exports contain no finite ratio in that case.

Annualized metrics use observation counts. They do not infer elapsed calendar years from dates. All input prices must use a common currency. The app does not perform currency conversion.

## Legacy portfolio risk and diversification

The legacy risk analysis describes the initial portfolio, with weights `w` and the annual covariance estimate `Sigma` from its training period. It uses the same covariance shrinkage as the optimizer.

| Metric | Definition |
| --- | --- |
| Largest weight (`max_weight`) | Largest initial asset weight |
| Effective holdings (`effective_holdings`) | `1 / sum(w^2)`. Equal weights across four assets give four effective holdings. |
| Diversification ratio (`diversification_ratio`) | `sum(w_i * sqrt(Sigma_ii)) / sqrt(w.T * Sigma * w)` |
| Risk contribution | `w_i * (Sigma * w)_i / (w.T * Sigma * w)` for each asset |

Risk contributions show each asset's share of portfolio variance. Their sum is one when portfolio variance is positive. Negative contributions are valid when an asset reduces portfolio variance through covariance with other assets. Zero portfolio variance makes the diversification ratio and variance shares undefined. Effective holdings measures allocation concentration. It does not account for correlation between assets.

The variance shares use the [MOSEK portfolio cookbook's risk-contribution definition](https://docs.mosek.com/portfolio-cookbook/risk_parity.html#risk-budgeting-with-variance). The report exports `risk_summary.csv` and `risk_contributions.csv`. These initial estimates differ from the realized holdings statistics below.

## Holdings and interpretation

- **P&L contribution:** asset profit/loss per observation, added in initial-capital units. Sum of all asset contributions minus fees equals the strategy's net total return.
- **Average weight:** average exposure before each observed return, with the initial cash interval and subsequent drift.
- **End weight:** final holding value divided by final portfolio wealth.
- **Selection frequency:** fraction of allocation events with target weight above `1e-6` (0.0001%). It is not the fraction of observations with a position.
- **Maximum target weight / maximum realized weight:** largest requested allocation and largest observed closing allocation, respectively. Realized weight includes the value just before a rebalance.

The legacy report highlights the highest net return, the smallest maximum drawdown, holding contributions and concentration. Its leading strategy is selected **after** observing the results. This is descriptive evidence, not a forecast or a recommended stock portfolio. Contributor and detractor headlines exclude tiny solver residuals. Full numeric holdings remain in the CSVs.

A fixed list chosen later in history creates universe-selection and survivorship bias even when every fit respects time order. Changing settings after seeing results also contaminates the evaluation period. The methods do not repair either problem. Different start dates, fee assumptions and market regimes can change the comparison.

The legacy overview keeps the profile, maximum-Sharpe, and equal-weight comparisons in a fixed order. It does not use the test winner to select the next allocation. A sample-mean optimum can concentrate in past winners and perform poorly later. Estimation error is a known limitation of optimized portfolios. See the [DeMiguel, Garlappi, and Uppal research paper](https://warwick.ac.uk/fac/soc/wbs/subjects/finance/events/recentevents/pastevents/behaviouralfinance1day/uppal_paper_06.pdf). Equal weight is a comparison rule, not a claim of future outperformance.

## Legacy exports and reproduction

Run the separate legacy comparison from the command line:

```bash
.venv/bin/python -m efficient_frontier --csv data/holdings.csv \
  --backtests --train-fraction .5 --rolling-window 252 \
  --rebalance-every 21 --cost-bps 10 --output results/holdings-study
```

Latest profile exports contain:

| File | Contents |
| --- | --- |
| `latest_profile_summary.csv` | Full-history arithmetic means, volatility, window diagnostics, and concentration measures |
| `latest_profile_weights.csv` | Complete asset weights for Low, Medium, and Extreme |
| `latest_profile_windows.csv` | Dates and observation counts for the three windows |
| `latest_profile_window_returns.csv` | Annual arithmetic mean estimates for each profile and window |
| `latest_profile_frontier.csv` | Historical-mean frontier estimates, including the lowest-window-mean diagnostic |
| `latest_profile_frontier_weights.csv` | Complete asset weights at each frontier point |

`metadata.json` includes the latest fit date, observation count, window details, settings, and warnings. `return_objective` records the latest fit objective, and `profile_return_objective` records the objective for profile backtests. In profile CSVs, `expected_return` is the annualized arithmetic mean over the full fit period. `worst_window_return` retains the lowest annual window mean. The optional weakest-window model retains these field definitions. These exports describe the latest model, separately from historical strategy allocations.

The backtest report includes `findings.md`, `backtest_metrics.csv`, `backtest_curve.csv` and one set of allocation, holdings and trade CSVs per combination under `backtests/`. `metadata.json` maps numbered file prefixes to strategy names. Trade logs record the exact estimation start/end dates as well as wealth, fees and turnover. All exports include the full input universe.

Legacy exports retain every calculated strategy and asset.

Open `report.html` and use **Print / save PDF** to create a static report. The HTML charts remain interactive and work offline. Keep source data and generated reports local unless their redistribution is authorized.

Method references: [time-series cross-validation](https://otexts.com/fpp3/tscv.html), [self-financing portfolios and transaction costs](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf), and [portfolio simulation conventions](https://www.cvxportfolio.com/en/stable/simulator.html).
