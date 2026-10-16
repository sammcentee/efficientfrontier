# Backtesting methods and accounting

In the app, **Show portfolios** calculates four methods as part of the study. The CLI enables this comparison with `--backtests`. The original targets are minimum volatility, maximum Sharpe, and equal weight. With sufficient history, the app and CLI also include Low, Medium, and Extreme profiles. The app's **Risk level** control calls Extreme **Highest**.

Every asset in the accepted price dataset remains eligible. Nasdaq-100 input first excludes members without complete prices for the selected period. The [coverage rules](../README.md#the-full-nasdaq-100-universe) explain this step and its historical selection bias. There is no fixed ticker-count or holdings-count cap.

| Method | Target weights | Estimation sample |
| --- | --- | --- |
| Buy and hold | Allocate once, then let weights drift | Initial training returns |
| Fixed rebalance | Restore the initial targets every N observations | Initial training returns |
| Expanding window | Estimate new targets every N observations | All returns available before the execution close |
| Rolling window | Estimate new targets every N observations | Latest L returns available before the execution close |

The original targets give 12 method/portfolio combinations. With the three profiles, the comparison contains 24 combinations. Equal-weight targets do not depend on estimated returns, so the fixed, expanding and rolling equal-weight paths intentionally coincide. This is not three independent pieces of evidence. The original **holdout** remains a separate, cost-free calculation with its original entry timing. Find it under **Research → Frontier → Classic holdout, no trading costs**.

## Timing and information

1. With `T` observed returns, the first `floor(train_fraction × T)` form the initial estimation period. The remaining dates form the common evaluation period.
2. Set initial wealth to one at the last initial-training close. Hold cash at zero interest until the next observed close. Deduct entry fees at that close. Establish positions there. The first evaluation interval therefore earns no asset return.
3. At each later execution close, value the old holdings with the latest observed price changes. Fit new targets with observations **only through the preceding close**. Deduct trading fees. Establish the new holdings. The new targets earn returns from the next observed interval.
4. Rebalance every `N` observed intervals. The default is 21. This is not an exact calendar-month schedule. Do not rebalance at the final close merely to report an unused target. The engine does not charge terminal liquidation fees.

The engine delays execution by one observed interval. A signal cannot receive a return that the fit already includes. It still assumes execution at the next supplied adjusted close, fractional holdings, and unlimited capacity. Spreads, slippage and commissions can be approximated by the fee setting; there is no volume-dependent market impact, tax or FX model. Input adjustments are accepted as supplied rather than independently reconstructing corporate actions.

`--rolling-window L` counts observed returns. The window must contain at least two returns and cannot exceed the initial training length. By default it equals that initial length. A shorter window changes both the initial rolling allocation and later allocations. A comparison with the expanding method therefore measures the full configuration and its different initial sample.

All schedules use the supplied price rows. With weekly data, `--rebalance-every 4` means four observed weekly intervals. With monthly data, it means four observed monthly intervals. Missing dates can make these intervals longer than the named calendar period.

The `--periods-per-year` option controls annualization. Its default is 252, and it accepts positive, finite numbers. For CSV input, **Change → Model assumptions → Price frequency** provides app presets of 252, 365, 52, and 12. Custom app values must be finite and at least 1. This setting does not resample prices or change trade dates. Yahoo inputs use daily prices and 252 observations per year in the app.

Both engines need at least two training returns and two holdout returns. The CSV reader accepts at least five price rows. The chosen split must still meet both return counts. Results include a warning for fewer than 30 training returns. They also include a warning when the asset count equals or exceeds the training return count.

## Latest model holdings and profiles

The primary model uses historical arithmetic mean return. Latest holdings fit all supplied returns through the final price date. For weights `w`, the fitted annual mean is `mu @ w`, where `mu` is the annualized arithmetic mean of each asset over the complete fit period. This estimate is not compound growth (CAGR) or a forecast.

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

The app calls Extreme **Highest**. These labels are relative to the supplied assets and model. Highest does not add leverage or seek the largest possible variance. When the frontier has no volatility range, all three profiles use the same highest-return mix. Numerical ties can produce several equivalent allocations.

The three chronological, nonoverlapping windows remain diagnostic checks. Their observation counts are nearly equal. Each needs at least two returns, so profiles need at least six returns in total. Window dates identify the first and last return observations. Each annualized window mean is `mu_window @ w`; `worst_window_return` is the smallest of those three portfolio means. These diagnostics do not select primary weights. They are not confidence bounds or forecasts.

Reports warn when a window has fewer observations than the selected periods per year. They also identify profiles without distinct risk levels. Solver warnings remain visible, and tables report the actual estimates of the computed weights.

**Latest model holdings use an in-sample fit.** Their `as_of` date is the final supplied price date, not a guarantee of current market quotes. The classic frontier uses the same historical-mean objective on the initial training sample. Its holdout evaluates that initial allocation on later dates.

Historical profile backtests use the same historical-mean objective, but each fit uses only returns through the close before execution. Fixed rebalance retains the initial profile targets. Expanding and rolling methods rebuild their profiles at each scheduled fit. Latest full-history weights never enter those earlier trades. Three-window diagnostics also use only each fit's history.

The initial sample and rolling window each need at least six returns for the complete profile comparison. If either is too short, the app and CLI retain the original 12 combinations and explain why they omitted the profiles. A latest full-history fit can still be available when the earlier backtest fit is too short.

The Python API keeps `include_profiles=False` as its compatibility default. `run_backtests(..., include_profiles=True)` requests the additional profiles. Direct API calls reject shorter fits. The app and CLI use the fallback above.

### Optional weakest-window comparison

**Research → Efficient frontiers** retains a separate weakest-window comparison. The Python API selects it with `build_profiles(..., return_objective="worst_window")`. It does not change the primary model or its backtests.

This objective maximizes the lowest of the three portfolio window means. At each frontier target, it minimizes variance subject to `mu_window @ w >= target` in every window. Low uses the minimum-variance endpoint. Medium uses the midpoint of the endpoint return targets. Highest uses the maximum feasible weakest-window target, with minimum variance among tied solutions. Medium therefore uses a return midpoint in this optional model.

For this optional objective, the solver allows return-target slack of `1e-8 * S`. Here, `S` is the largest absolute annual asset mean across the three windows, or one if all means are zero. The final check rejects target shortfalls greater than `2e-8 * S`. This preserves the minimum-variance tie-break near numerical ties. A warning identifies cases where no feasible portfolio has positive estimated means in all three windows.

The optional chart is **Weakest-window frontier**. Its ETF markers use the same lowest-window-mean measure. The primary **Historical mean frontier** and its ETF markers use full-history arithmetic means. Neither chart guarantees future returns or minimum drawdown.

## Costs and constraints

All targets are long-only and fully invested after execution. An optional weight cap applies at each trade; price movements can take actual weights above it. The risk-free rate is a constant annual assumption for Sharpe and Sortino. It does not create an investable cash strategy.

Let `V` be wealth before a trade, `h` the current dollar holdings, `w` the target weights, and `c = cost_bps / 10000`. Post-trade wealth `V_after` solves:

```text
V_after + c × sum(abs(V_after × w - h)) = V
```

Fees charge both bought and sold notional. Entry from cash leaves `1 / (1 + c)` invested. A complete switch between two assets leaves `(1 - c) / (1 + c)` of pretrade wealth. There is no hidden division by two in turnover.

**Gross turnover** sums each trade's absolute notional divided by its own pretrade wealth. **Total fees** sum fee payments in units of initial capital. These denominators differ as wealth changes. Fees are also different from the full performance gap against a zero-fee counterfactual, which includes foregone compounding.

When no feasible maximum-Sharpe portfolio has positive estimated excess return, that strategy uses minimum-volatility weights for that execution. The warning and fallback count remain in the results; the interval is never discarded. Fixed rebalancing continues to restore its original target, including any original fallback.

## Passive benchmarks and evidence

SPY and QQQ comparisons use the same evaluation dates as each strategy. The backtest benchmarks stay in cash through the first evaluation interval, then buy at its close. The selected entry cost leaves `1 / (1 + cost_bps / 10000)` invested. The original holdout benchmarks enter at the final training close without costs. Neither benchmark path includes a final sale.

Benchmarks remain separate from the eligible assets and portfolio weight cap. The statistical comparison uses all strategies in the current run, even when chart selections display fewer curves. Latest model holdings remain an in-sample fit. [Benchmark methods and evidence](BENCHMARKS.md) explains the USD assumption, data coverage, HAC estimates, Holm correction, and limits on inference.

## Return and risk metrics

The original holdout and the backtests use the same metric definitions. Backtest returns include the selected fees. Let `p` denote observations per year, `r` the observed portfolio returns, and `rf` the annual risk-free rate.

| Metric | Definition |
| --- | --- |
| CAGR | Compound growth, annualized with `p` and the number of observed returns |
| Volatility | Sample standard deviation of `r`, multiplied by `sqrt(p)` |
| Sharpe | `(mean(r) * p - rf) / annual_volatility` |
| Sortino | `(mean(r) * p - rf) / (sqrt(mean(min(r - rf / p, 0)^2)) * sqrt(p))` |
| Calmar | `CAGR / abs(max_drawdown)` |

Sortino measures downside relative to the risk-free rate per observation. Its denominator includes all observations, with zero contributions from returns at or above that threshold. Calmar compares compound growth with the largest observed fall from a previous peak. A zero denominator makes these ratios undefined. The app shows no finite ratio in that case.

Annualized metrics use observation counts. They do not infer elapsed calendar years from dates. All input prices must use a common currency. The app does not perform currency conversion.

## Portfolio risk and diversification

Risk analysis describes the initial portfolio, with weights `w` and the annual covariance estimate `Sigma` from its training period. It uses the same covariance shrinkage as the optimizer.

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

The report highlights the highest net return, the smallest maximum drawdown, holding contributions and concentration. Its leading strategy is selected **after** observing the results. This is descriptive evidence, not a forecast or a recommended stock portfolio. Tiny solver residuals are excluded from displayed contributor/detractor headlines; full numeric holdings remain in the CSVs.

A fixed list chosen later in history creates universe-selection and survivorship bias even when every fit respects time order. Changing settings after seeing results also contaminates the evaluation period. The methods do not repair either problem. Different start dates, fee assumptions and market regimes can change the comparison.

The overview keeps the profile, maximum-Sharpe, and equal-weight comparisons in a fixed order. It does not use the test winner to select the next allocation. A sample-mean optimum can concentrate in past winners and perform poorly later. Estimation error is a known limitation of optimized portfolios; see the [DeMiguel, Garlappi, and Uppal research paper](https://warwick.ac.uk/fac/soc/wbs/subjects/finance/events/recentevents/pastevents/behaviouralfinance1day/uppal_paper_06.pdf). Equal weight is a comparison rule, not a claim of future outperformance.

## Reproduce and inspect

In the app, open **Test settings** in the market test to set trade intervals, costs, and rolling history. Select **Run the test again** after you change these assumptions. Use **Research → All tests** for every strategy, holding, and trade.

Select **Export** (or press E) to download the report and data. The bundle includes the market test on screen. A new portfolio calculation runs a new market test.

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

The selected risk level, allocation rule, and chart change the displayed results. They do not remove calculated strategies or assets from the exports.

Open `report.html` and use **Print / save PDF** to create a static report; the HTML charts remain interactive and work offline. Keep source data and generated reports local unless their redistribution is authorized.

Method references: [time-series cross-validation](https://otexts.com/fpp3/tscv.html), [self-financing portfolios and transaction costs](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf), and [portfolio simulation conventions](https://www.cvxportfolio.com/en/stable/simulator.html).
