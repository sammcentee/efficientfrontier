# Backtesting methods and accounting

The **Backtests & holdings** tab compares four methods with minimum-volatility, maximum-Sharpe and equal-weight targets. The CLI enables the same comparison with `--backtests`. All supplied assets remain eligible; there is no ticker-count or holdings-count cap.

| Method | Target weights | Estimation sample |
| --- | --- | --- |
| Buy and hold | Allocate once, then let weights drift | Initial training returns |
| Fixed rebalance | Restore the initial targets every N observations | Initial training returns |
| Expanding window | Estimate new targets every N observations | All returns available before the execution close |
| Rolling window | Estimate new targets every N observations | Latest L returns available before the execution close |

There are 12 method/portfolio combinations. Equal-weight targets do not depend on estimated returns, so the fixed, expanding and rolling equal-weight paths intentionally coincide. This is not three independent pieces of evidence. The original **holdout** remains available as a separate, cost-free calculation with its original entry timing.

## Timing and information

1. With `T` observed returns, the first `floor(train_fraction × T)` form the initial estimation period. The remaining dates form the common evaluation period.
2. Set initial wealth to one at the last initial-training close. Hold cash at zero interest until the next observed close. Deduct entry fees at that close. Establish positions there. The first evaluation interval therefore earns no asset return.
3. At each later execution close, value the old holdings with the latest observed price changes. Fit new targets with observations **only through the preceding close**. Deduct trading fees. Establish the new holdings. The new targets earn returns from the next observed interval.
4. Rebalance every `N` observed intervals. The default is 21. This is not an exact calendar-month schedule. Do not rebalance at the final close merely to report an unused target. The engine does not charge terminal liquidation fees.

The engine delays execution by one observed interval. A signal cannot receive a return that the fit already includes. It still assumes execution at the next supplied adjusted close, fractional holdings, and unlimited capacity. Spreads, slippage and commissions can be approximated by the fee setting; there is no volume-dependent market impact, tax or FX model. Input adjustments are accepted as supplied rather than independently reconstructing corporate actions.

`--rolling-window L` counts observed returns. The window must contain at least two returns and cannot exceed the initial training length. By default it equals that initial length. A shorter window changes both the initial rolling allocation and later allocations. A comparison with the expanding method therefore measures the full configuration and its different initial sample.

All schedules use the supplied price rows. With weekly data, `--rebalance-every 4` means four observed weekly intervals. With monthly data, it means four observed monthly intervals. Missing dates can make these intervals longer than the named calendar period.

The `--periods-per-year` option controls annualization. Its default is 252, and it accepts positive, finite numbers. App presets include 252, 365, 52, and 12. Custom app values must be finite and at least 1. This setting does not resample prices or change trade dates. Yahoo downloads remain daily.

Both engines need at least two training returns and two holdout returns. The CSV reader accepts at least five price rows. The chosen split must still meet both return counts. Results include a warning for fewer than 30 training returns. They also include a warning when the asset count equals or exceeds the training return count.

## Costs and constraints

All targets are long-only and fully invested after execution. An optional weight cap applies at each trade; price movements can take actual weights above it. The risk-free rate is a constant annual assumption for Sharpe and Sortino. It does not create an investable cash strategy.

Let `V` be wealth before a trade, `h` the current dollar holdings, `w` the target weights, and `c = cost_bps / 10000`. Post-trade wealth `V_after` solves:

```text
V_after + c × sum(abs(V_after × w - h)) = V
```

Fees charge both bought and sold notional. Entry from cash leaves `1 / (1 + c)` invested. A complete switch between two assets leaves `(1 - c) / (1 + c)` of pretrade wealth. There is no hidden division by two in turnover.

**Gross turnover** sums each trade's absolute notional divided by its own pretrade wealth. **Total fees** sum fee payments in units of initial capital. These denominators differ as wealth changes. Fees are also different from the full performance gap against a zero-fee counterfactual, which includes foregone compounding.

When no feasible maximum-Sharpe portfolio has positive estimated excess return, that strategy uses minimum-volatility weights for that execution. The warning and fallback count remain in the results; the interval is never discarded. Fixed rebalancing continues to restore its original target, including any original fallback.

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

## Reproduce and inspect

```bash
.venv/bin/python -m efficient_frontier --csv data/holdings.csv \
  --backtests --train-fraction .5 --rolling-window 252 \
  --rebalance-every 21 --cost-bps 10 --output results/holdings-study
```

The report includes `findings.md`, `backtest_metrics.csv`, `backtest_curve.csv` and one set of allocation, holdings and trade CSVs per combination under `backtests/`. `metadata.json` maps numbered file prefixes to strategy names. Trade logs record the exact estimation start/end dates as well as wealth, fees and turnover. All exports include the full input universe.

App selections change only the curves on screen. They do not remove strategies or assets from the exports.

Open `report.html` and use **Print / save PDF** to create a static report; the HTML charts remain interactive and work offline. Keep source data and generated reports local unless their redistribution is authorized.

Method references: [time-series cross-validation](https://otexts.com/fpp3/tscv.html), [self-financing portfolios and transaction costs](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf), and [portfolio simulation conventions](https://www.cvxportfolio.com/en/stable/simulator.html).
