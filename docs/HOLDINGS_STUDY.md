# Stock holdings: observed backtesting findings

Study run on 2026-10-03. These are descriptive comparisons, not stock recommendations or a prediction of future returns.

## Key observations

- The highest net return, **256.75%**, came from fixed maximum-Sharpe targets with heavy concentration: LLY's closing weight reached **63.48%**.
- Rolling maximum Sharpe returned **130.26%** with a **9.91%** maximum drawdown, versus **22.79%** for the fixed strategy. It also traded substantially more.
- In the later evaluation starting January 2025, rolling maximum Sharpe returned **28.07%**, only slightly above rebalanced equal weight's **27.39%**. The return advantage depended on the period.

The original universe was recorded after the beginning of the main evaluation. These findings include universe-selection hindsight; the later-period check and the limitations below are part of the result.

## Data and design

The study uses all 60 securities from `spy_holdings.ods`, with 1,697 complete adjusted daily observations from 2020-01-02 through 2026-10-02. No security was dropped and no missing price was filled. Yahoo/yfinance supplied adjusted closes in USD. The original `MMC` symbol was requested as `MRSH`, following the [issuer's January 2026 symbol change](https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html); the historical spreadsheet is unchanged. GOOG and GOOGL are two share classes of one issuer.

The original list was recorded in Git on **2024-10-27**. Applying it to earlier history introduces hindsight in the universe selection. Time-ordered estimation does not remove that problem. A second evaluation starting after that date is included below, but it is a sensitivity analysis chosen during this research, not an untouched prospective test.

Main configuration:

- Initial estimation: 848 daily returns, 2020-01-03 through 2023-05-16.
- Evaluation: 848 sessions, 2023-05-17 through 2026-10-02. The first session stays cash until entry at its close.
- Periodic trades: every 21 sessions, with information available at the preceding close. Entry plus 40 subsequent allocation events.
- Rolling estimation: latest 252 returns, including at initial entry. Expanding estimation retains the full prior sample.
- Long-only, fully invested targets, no additional position cap; 10% diagonal covariance shrinkage.
- Risk-free assumption: 2% annually for Sharpe. Cash during the initial delay earns zero interest.
- Fees: 10 basis points per unit bought or sold, including entry. No final liquidation, taxes, FX or volume-dependent impact.

The [methodology](BACKTESTING.md) defines execution, drift, fallback and attribution exactly. These results use the new delayed-execution engine, rather than the legacy cost-free holdout.

## Return, drawdown and trading

| Method / target | Net total return | CAGR | Annual volatility | Sharpe | Max drawdown | Gross turnover | Fees / initial capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy and hold / minimum volatility | 63.00% | 15.63% | 12.41% | 1.07 | -9.58% | 1.00× | 0.10% |
| Buy and hold / maximum Sharpe | 225.08% | 41.95% | 26.91% | 1.36 | -24.66% | 1.00× | 0.10% |
| Buy and hold / equal weight | 95.21% | 21.99% | 13.74% | 1.37 | -17.25% | 1.00× | 0.10% |
| Fixed rebalance / minimum volatility | 67.64% | 16.59% | 12.71% | 1.11 | -10.03% | 2.46× | 0.29% |
| Fixed rebalance / maximum Sharpe | 256.75% | 45.93% | 26.56% | 1.48 | -22.79% | 3.91× | 0.76% |
| Rebalanced equal weight¹ | 91.18% | 21.24% | 12.54% | 1.44 | -15.62% | 3.37× | 0.46% |
| Expanding window / minimum volatility | 62.75% | 15.57% | 11.60% | 1.13 | -9.69% | 3.13× | 0.38% |
| Expanding window / maximum Sharpe | 186.77% | 36.76% | 23.02% | 1.39 | -21.04% | 6.58× | 1.31% |
| Rolling window / minimum volatility | 66.25% | 16.31% | 10.40% | 1.31 | -10.61% | 13.30× | 1.64% |
| Rolling window / maximum Sharpe | 130.26% | 28.13% | 15.22% | 1.57 | -9.91% | 21.53× | 3.64% |

¹ Fixed, expanding and rolling equal-weight targets have identical paths, so they are combined in this table. The numeric exports retain all 12 combinations. There were no maximum-Sharpe fallbacks in this study.

Fixed maximum-Sharpe targets produced the highest net return, while rolling maximum Sharpe had the highest realized Sharpe. Those are different objectives: the rolling strategy's smaller drawdown and lower volatility came with substantially lower growth and more trading. The shorter rolling estimation sample also differs from the other methods; these numbers do not isolate a causal benefit of forgetting old data.

Minimum-volatility portfolios delivered 62.75–67.64% total returns, with maximum drawdowns of roughly 9.6–10.6%. They did not lead the return table, but demonstrate why selecting solely by the largest gain misses part of the research question.

## Which holdings mattered?

For the retrospectively highest-return combination, **fixed rebalance / maximum Sharpe**, only seven stocks had target weights above the reporting threshold. The original 60-stock universe did not imply a diversified allocation.

| Holding | Initial target | Average realized weight | Peak closing weight | P&L contribution / initial capital |
| --- | ---: | ---: | ---: | ---: |
| LLY | 55.80% | 55.68% | 63.48% | +123.99 percentage points |
| NVDA | 13.54% | 13.72% | 17.19% | +58.86 percentage points |
| TSLA | 18.96% | 18.87% | 28.18% | +46.13 percentage points |
| DE | 8.42% | 8.32% | 10.98% | +17.80 percentage points |
| ANET | 1.93% | 1.96% | 2.67% | +8.44 percentage points |
| XOM | 1.04% | 1.03% | 1.28% | +1.91 percentage points |
| HCA | 0.31% | 0.31% | 0.38% | +0.39 percentage points |

These are contributions to this portfolio, not the stocks' standalone percentage returns. Unrounded contributions across all holdings, less fees of 0.7564% of initial capital, reconcile to the portfolio's 256.7544% net return; displayed rounded rows can differ slightly. LLY accounted for almost half of gross portfolio profit. That concentration is a substantial part of the historical outcome and should accompany any headline return.

The rolling maximum-Sharpe portfolio had different leading contributors: JNJ +19.61 points, LLY +18.66, GE +17.19, NVDA +15.86 and T +11.64. Its allocations changed across 41 events. No individual holding's contribution establishes that it should be held now.

## Fee sensitivity

Same price input, settings and target schedules; fees vary on **both bought and sold notional**. Entries below are net total returns.

| Maximum-Sharpe method | 0 bps | 10 bps | 25 bps |
| --- | ---: | ---: | ---: |
| Buy and hold | 225.41% | 225.08% | 224.60% |
| Fixed rebalance | 258.15% | 256.75% | 254.67% |
| Expanding window | 188.67% | 186.77% | 183.96% |
| Rolling window | 135.28% | 130.26% | 122.94% |

The rolling strategy's 10-bps result was about **5.01 percentage points** below its zero-fee counterfactual, despite direct fee payments of 3.64% of initial capital. The extra difference is the return no longer earned on wealth consumed by earlier fees. All fee settings are illustrative, not broker-specific execution estimates.

## Later evaluation after the universe was recorded

Changing only the initial training fraction from 50% to 75% moves evaluation to **2025-01-27 through 2026-10-02**. Training-dependent initial targets therefore also change. The rolling window remains 252 returns and fees remain 10 bps.

| Maximum-Sharpe method | Net total return | Maximum drawdown | Realized Sharpe |
| --- | ---: | ---: | ---: |
| Buy and hold | 51.38% | -23.90% | 1.04 |
| Fixed rebalance | 60.42% | -24.08% | 1.18 |
| Expanding window | 34.43% | -21.18% | 0.85 |
| Rolling window | 28.07% | -10.25% | 0.97 |
| Rebalanced equal weight | 27.39% | -15.67% | 0.96 |

Rolling maximum Sharpe's advantage over rebalanced equal weight shrank from 39.09 percentage points in the main study to **0.68 points** here. Its smaller drawdown persisted, but the return advantage was sensitive to the evaluation interval. This later check reduces the earlier universe-timing concern; it does not eliminate strategy-selection bias or demonstrate future performance.

## Reproduce locally

Use an authorized CSV containing the same 60 adjusted price series and dates. The cached local study records the source, symbol mapping, fetch timestamps and input hashes in `results/holdings-study/input_provenance.json`. Provider revisions or another adjusted-price implementation can change results.

```bash
.venv/bin/python -m efficient_frontier --csv data/holdings.csv \
  --backtests --train-fraction .5 --rolling-window 252 \
  --rebalance-every 21 --cost-bps 10 --output results/holdings-study

# Repeat with --cost-bps 0 and --cost-bps 25 in different output folders.
# For the later period, change --train-fraction .5 to --train-fraction .75.
```

The local report contains interactive charts, Markdown findings and every holding/trade. **Print / save PDF** produces a static copy. Raw prices and generated market reports are ignored by Git; the software's license does not grant redistribution rights to the data. The figures above are derived research summaries, not redistributed price histories.
