# Backtesting methods and accounting

The **Backtests & holdings** tab compares four methods with minimum-volatility, maximum-Sharpe and equal-weight targets. The CLI enables the same comparison with `--backtests`. All supplied assets remain eligible; there is no ticker-count or holdings-count cap.

| Method | Target weights | Estimation sample |
| --- | --- | --- |
| Buy and hold | Allocate once, then let weights drift | Initial training returns |
| Fixed rebalance | Restore the initial targets every N sessions | Initial training returns |
| Expanding window | Estimate new targets every N sessions | All returns available before the execution day |
| Rolling window | Estimate new targets every N sessions | Latest L returns available before the execution day |

There are 12 method/portfolio combinations. Equal-weight targets do not depend on estimated returns, so the fixed, expanding and rolling equal-weight paths intentionally coincide. This is not three independent pieces of evidence. The original **holdout** remains available as a separate, cost-free calculation with its original entry timing.

## Timing and information

1. With `T` daily returns, the first `floor(train_fraction × T)` form the initial estimation period. The remaining dates are the common evaluation period.
2. Set initial wealth to one at the last initial-training close. Hold cash, earning zero interest, until the next observed close; deduct entry fees and establish positions there. The first evaluation session therefore earns no asset return.
3. At a later scheduled execution close, value the old holdings using that day's price changes. Fit new targets using observations **only through the preceding close**. Deduct trading fees and establish the new holdings. The new targets earn returns starting with the following session.
4. Rebalance every `N` observed sessions; the default is 21. This is not an exact calendar-month schedule. Do not rebalance at the final close merely to report an unused target. No terminal liquidation is charged.

This one-session delay prevents a signal from earning the return used to estimate it. It still assumes execution at the next supplied adjusted close, fractional holdings, and unlimited capacity. Spreads, slippage and commissions can be approximated by the fee setting; there is no volume-dependent market impact, tax or FX model. Input adjustments are accepted as supplied rather than independently reconstructing corporate actions.

`--rolling-window L` counts daily returns, not years. It must be between two and the initial training length. By default it equals that initial length. An explicit shorter window changes both the initial rolling allocation and subsequent allocations; comparing it with the expanding method tests the entire chosen configuration, not the window update rule in isolation.

## Costs and constraints

All targets are long-only and fully invested after execution. An optional weight cap applies at each trade; price movements can take actual weights above it. The risk-free rate is a constant annual assumption for estimation and realized Sharpe, not an investable cash strategy.

Let `V` be wealth before a trade, `h` the current dollar holdings, `w` the target weights, and `c = cost_bps / 10000`. Post-trade wealth `V_after` solves:

```text
V_after + c × sum(abs(V_after × w - h)) = V
```

Fees charge both bought and sold notional. Entry from cash leaves `1 / (1 + c)` invested. A complete switch between two assets leaves `(1 - c) / (1 + c)` of pretrade wealth. There is no hidden division by two in turnover.

**Gross turnover** sums each trade's absolute notional divided by its own pretrade wealth. **Total fees** sum fee payments in units of initial capital. These denominators differ as wealth changes. Fees are also different from the full performance gap against a zero-fee counterfactual, which includes foregone compounding.

When no feasible maximum-Sharpe portfolio has positive estimated excess return, that strategy uses minimum-volatility weights for that execution. The warning and fallback count remain in the results; the interval is never discarded. Fixed rebalancing continues to restore its original target, including any original fallback.

## Holdings and interpretation

- **P&L contribution:** each day's asset profit/loss added in initial-capital units. Sum of all asset contributions minus fees equals the strategy's net total return.
- **Average weight:** average exposure before each daily return, including the initial cash session and subsequent drift.
- **End weight:** final holding value divided by final portfolio wealth.
- **Selection frequency:** fraction of allocation events with target weight above `1e-6` (0.0001%). It is not the fraction of days held.
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

Open `report.html` and use **Print / save PDF** to create a static report; the HTML charts remain interactive and work offline. Keep source data and generated reports local unless their redistribution is authorized.

Method references: [time-series cross-validation](https://otexts.com/fpp3/tscv.html), [self-financing portfolios and transaction costs](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf), and [portfolio simulation conventions](https://www.cvxportfolio.com/en/stable/simulator.html).
