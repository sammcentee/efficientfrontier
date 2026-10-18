# Equal-volatility index research

This study asks whether a simple index rule beats both SPY and QQQ after trading costs at the same volatility. It does not change the app's portfolio model.

## Frozen rules

The protocol was saved before new strategy results. `index_research_protocol.json` records the candidate set, dates, costs, and comparison rules.

| Stock rule | Selection and target weights |
| --- | --- |
| Equal weight | Equal amounts across eligible stocks |
| Inverse volatility | All eligible stocks, weighted by inverse 252-return volatility, capped at 10% |
| Low-volatility 100 | The 100 stocks with lowest 252-return volatility; inverse-volatility weights, capped at 10% |
| Momentum 50 | The 50 highest returns from month-end m−12 through m−1; 2% per stock |
| Risk-adjusted momentum 50 | The same return divided by volatility over that window; 2% per stock |

Signals use the completed calendar month-end. Trades occur at the next trading close. Holdings drift between trades. A stock needs enough valid past prices and must pass the existing unchanged-price screen. These custom rules use adjusted total-return prices. They are adaptations, not replicas of published indexes.

The separate fund study uses SPMO, MTUM, SPLV, USMV, RSP, and QUAL. Each follows an externally maintained index. A seventh candidate holds QQQ only when its month-end adjusted level exceeds its ten-month moving average; otherwise it holds cash. The fund histories reflect the index rules in effect at each date, including rule changes. Fund results do not validate our custom stock rules.

## Two different risk comparisons

The executable rule estimates the desired portfolio's volatility from the prior 126 return observations. It scales risky weights to `min(1, 10% / estimated_volatility)` and holds the rest in cash. It trades at the next close after month-end. Every candidate and both benchmarks use the same rule. Realized volatility can differ from the target.

The exact-risk diagnostic scales each completed net daily return series by `q / realized_volatility`, where `q` is the lowest volatility among all compared paths. Every scale is at most one, and all normalized paths have the same observed volatility. This is a statistical normalization using future observations. It is not a tradable cash sleeve: the diagnostic does not simulate additional trades to maintain those exposures.

The primary selection metric is the smaller normalized annual arithmetic mean advantage over SPY and QQQ. Selection uses the executable risk-controlled paths in development only. CAGR and drawdown remain secondary measures. Raw unscaled paths are reported separately. A negative score means no candidate beats both references in development.

Cash earns zero in every path. The model charges 10 basis points on each amount bought or sold, including entry and risk adjustments. A fixed 25-basis-point stress checks trading-cost sensitivity. Fund expenses are already reflected in fund prices. Taxes, personal broker terms, and currency conversion are excluded from all comparisons.

## Data and evaluation limits

The custom stock study uses the existing 500-stock input from 18 October 2021 to 16 October 2026. It uses current constituents with complete histories. This introduces selection and survivor bias. The final period, 17 April 2025 onward, was already reviewed in other research. Treat this study as exploratory.

The fund study requests actual adjusted prices from 1 October 2015. It starts at the common inception date, then reserves at least one year for signal history. Development ends on 30 December 2022. Evaluation begins on 3 January 2023. We freeze the selected rule before inspecting its final performance. This is a chronological historical evaluation, not new prospective evidence. The set of surviving funds and known investment factors can also introduce research selection.

The local long stock corpus was rejected as a confirmation dataset. It uses raw share prices, separate dividends, split events, and corporate actions. It still has unresolved historical member quotes and terminal-value gaps. No missing security is silently converted into a successful surviving holding.

## Statistical uncertainty

Paired stationary bootstrap samples preserve shared dates across all candidates and benchmarks. The fixed settings are 10,000 repetitions, seed 20261018, and mean block lengths 10, 21, and 63 observations. The 21-day result is primary. Volatility scales are recomputed within each sample.

Simultaneous 97.5% lower bounds cover all candidates in each prespecified risk-controlled study. Bonferroni correction across the two study families targets an approximate 95% familywise confidence level, subject to the bootstrap assumptions. Raw paths and cost sensitivities remain descriptive. Passing a point estimate is weaker than both lower bounds exceeding zero. These bounds do not correct historical membership errors, unknown earlier searches, or all model assumptions. They do not assign a probability of future outperformance.

## Sources

- [Kenneth French: prior 2–12 month momentum](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html)
- [S&P momentum methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-momentum-indices.pdf)
- [S&P low-volatility methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-low-volatility-indices.pdf)
- [Faber: monthly trend rule](https://mebfaber.com/2017/12/13/episode-86-quantitative-approach-tactical-asset-allocation/)
- [Cederburg and coauthors: limits of volatility-managed portfolios](https://www.sciencedirect.com/science/article/pii/S0304405X2030132X)
- [Bailey and coauthors: backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
