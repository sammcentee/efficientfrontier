# Portfolio Lab user guide

Start with the [README quick start](../README.md#start-here-no-coding-needed). This guide covers setup details, input rules, and command-line reports. The [backtest guide](BACKTESTING.md) defines the model and accounting.

## Setup and troubleshooting

Use Python 3.14. The launchers create a project-local `.venv`. They install the runtime packages on first use and after `requirements.txt` changes. An incomplete package install is retried on the next launch.

On Windows, the current Python Install Manager can obtain Python 3.14 automatically. An older `py` launcher needs an installed 3.14 runtime. On macOS, install the 3.14 runtime before you open the `.command` launcher.

The launchers open a browser tab. The app binds to `127.0.0.1`, and Streamlit usage statistics are disabled. Keep the launcher window open while you use the app. Press **Ctrl+C** there to stop it. The server stays active after you close the browser tab.

| Problem | Action |
| --- | --- |
| Python was not found, or the launcher requests Python 3.14 | Install Python 3.14, then reopen the launcher. |
| Package download stopped | Check your internet connection and reopen the launcher. |
| The launcher window closes immediately | Extract the whole ZIP first. Keep its files together. |
| macOS says the `.command` file is not executable | Open Terminal. Type `bash `, drag `Start Portfolio Lab.command` into the window, then press Enter. |
| No browser tab opens | Visit `http://localhost:8501` or the Local URL from the launcher. |
| VS Code Remote/WSL cannot reach the app | Forward port 8501 from the Ports panel. |
| Port 8501 is in use | Stop the earlier launcher with Ctrl+C. Or pass `--server.port=8502` and use the new Local URL. |
| The launcher reports a different Python version in `.venv` | Remove only that project's `.venv` folder, then reopen the launcher. |

For a background or remote session, run:

```bash
bash run.sh --server.headless=true
```

To create an environment explicitly on Linux or WSL:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The synthetic demo and CSV analysis work offline after setup. Yahoo prices and the current Nasdaq list need internet access. Python 3.14 is the tested version. Native Windows startup has a smoke check. macOS startup still needs verification on a Mac.

## Settings and workflow

Select **Change**, edit the inputs, then select **Show portfolios**. This produces new portfolios and a new market test. **Change → Model assumptions** contains model controls. **Test settings** in the market test contains trade controls.

| Setting | Effect |
| --- | --- |
| Limit each holding / Largest holding allowed (%) | Caps target weights. A 25% cap needs at least four assets. Prices can move weights above the cap between trades. |
| Risk-free rate (%) | Sets the comparison rate for Sharpe and Sortino. It affects classic maximum-Sharpe weights and does not add an investable cash asset. |
| Prices for the first fit (%) | Controls the initial estimation sample. A larger share leaves fewer observations for later evaluation. It does not change the latest full-history fit. |
| Covariance shrinkage (%) | Reduces cross-asset covariances and preserves each asset's variance. |
| Trade interval | Sets the number of observed intervals between trades. With weekly data, four means four observed weeks. |
| Trading cost (basis points) | Charges each amount bought or sold. Ten basis points equal 0.10%. |
| Recent history for “Refit on recent prices” | Sets the number of past returns for rolling estimates. Zero uses the initial training length. |
| Price frequency | Sets the annualization factor for a CSV. It does not resample prices. |
| Price currency | Declares the common currency. The app does not check or convert currencies. SPY/QQQ comparisons need USD. |

Covariance describes how asset returns move together. At 0% shrinkage, the model uses sample covariance. At 100%, it sets cross-asset covariance to zero. The default 10% reduces these covariances by 10% and keeps each asset's variance. This manual assumption can reduce sensitivity to noisy estimates. It does not guarantee better results.

The adjustment is `(1 - shrinkage) × covariance + shrinkage × diag(covariance)`. The app does not estimate the shrinkage level automatically.

The app evaluates four allocation rules: buy and hold, fixed rebalance, expanding window, and rolling window. See [trade timing and costs](BACKTESTING.md#timing-and-information) for the exact definitions. The selected profile, rule, and chart only change the display. Exports retain all calculated strategies and the full statistical test family.

**Research → Risk breakdown** uses the initial model weights and the covariance from the training period. It describes the initial model, rather than later weight drift. Latest-profile estimates use all supplied history. These are different fits. See the [risk definitions](BACKTESTING.md#portfolio-risk-and-diversification).

## Classic frontier and risk choices

Use **Research → Efficient frontiers → Classic risk level** to select Low, Medium, or High. This choice updates the highlighted frontier point, its initial weights, and its later holdout result. It does not refit the model or change the overview's risk level.

| Classic level | Definition |
| --- | --- |
| Low | The frontier point with the least estimated volatility. |
| Medium | The sampled frontier point nearest halfway between Low and High volatility. |
| High | The frontier endpoint with the highest fitted arithmetic mean return. |

These levels compare the selected assets under the holding limit. They are not absolute risk ratings. A flat frontier gives the same mix for all three levels. High can put all its weight in one asset when there is no holding limit.

The frontier describes the first fit. At the same estimated volatility, a higher point has a higher fitted mean return. A Low point can have less return than an ETF with more risk. The chart uses annualized arithmetic means, not compounded growth. ETF points use the same fit dates, but the ETFs are separate references unless they are among the selected assets. See the [CVXPY portfolio model](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html) for the mean and variance definition.

The classic holdout tests the same initial weights on later prices, beside SPY and QQQ when available. It buys once, lets weights drift, and excludes trading costs. A mix above an ETF during fitting can lose to it in this later period. Return estimates can change, and compound growth differs from an arithmetic mean. The overview uses a separate weakest-window model, with the selected refit rule and fees. Its results do not test the classic frontier point.

## Price data

Upload a CSV with `Date` first and one asset per column. Use adjusted prices in a common currency. The file needs at least five complete price rows and one asset. Latest profiles need at least seven rows.

Both classic analysis and backtests need at least two returns in each of the training and evaluation periods. Five price rows meet this minimum only with a suitable split, such as 50/50. Profile backtests need six training returns for their three windows. The app warns about fewer than 30 training returns, or an asset count at least as large as the training return count. These warnings identify weak samples. A larger sample does not establish a reliable forecast.

The app rejects duplicate dates or columns, nonnumeric values, missing observations, and nonpositive prices. It sorts dates but does not remove assets or fill gaps in a CSV. It cannot confirm price adjustments, currency, or the interval between observations.

Set **Change → Model assumptions → Price frequency** to match the CSV:

| Interval | Observations per year |
| --- | ---: |
| Daily trading sessions | 252 |
| Daily calendar observations | 365 |
| Weekly | 52 |
| Monthly | 12 |
| Custom app setting | A finite number of at least 1 |

The CLI uses `--periods-per-year`, with a default of 252. It accepts any positive finite value. This factor changes annualization only. It does not convert daily prices to weekly or monthly prices.

Yahoo uses `auto_adjust=True` and the adjusted `Close` field. The end date is exclusive. The app caches price downloads for one hour. For **My tickers**, incomplete history stops the analysis. Change the symbols or dates and rerun. Nasdaq-100 input instead records exclusions under the [coverage rules](../README.md#the-full-nasdaq-100-universe).

Yahoo symbols retain exchange suffixes such as `VOD.L`, `IWDA.AS`, `BMW.DE`, and `7203.T`. Explicit aliases convert `BRK.B` to `BRK-B`, `BRK.A` to `BRK-A`, `BF.B` to `BF-B`, and `BF.A` to `BF-A`. Other periods remain unchanged. International inputs still need a common currency and compatible dates. Different market holidays can create gaps.

The original spreadsheet is a static list of 60 symbols. It is not a full S&P 500 universe or a historical membership record. The Yahoo preset uses Marsh's `MRSH` symbol in place of the spreadsheet's historical `MMC`. Availability and symbols can change.

Yahoo/yfinance is unofficial. The [yfinance project](https://github.com/ranaroussi/yfinance), [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), and [redistribution guidance](https://help.yahoo.com/kb/SLN2352.html) describe access and use limits. The project license does not grant permission to access or redistribute market data. Use synthetic prices for public examples.

## SPY and QQQ comparisons

The app runs the market test as part of **Show portfolios**. Its default display is **Refit on all past prices → Medium**. This display choice does not depend on observed returns.

Yahoo input obtains benchmark prices automatically. For a CSV, use SPY/QQQ columns already in the file, or a separate benchmark file with exactly the same dates. Downloads require an explicit choice. The Nasdaq loader still needs SPY/QQQ dates for its eligibility calendar when benchmark comparisons are disabled.

Synthetic demo results do not use actual market benchmarks. Comparisons need a USD declaration and complete benchmark prices for every asset date. See the [benchmark guide](BENCHMARKS.md) for sources, cost rules, exports, and statistical limits.

## Command-line options and exports

Run the CLI from the repository root. On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

```bash
# All available options
.venv/bin/python -m efficient_frontier --help

# Offline asset and SPY/QQQ files with identical dates
.venv/bin/python -m efficient_frontier --csv data/prices.csv \
  --benchmark-csv data/benchmarks.csv --currency USD --backtests

# Weekly prices: four observed weeks between trades, 52 returns per rolling fit
.venv/bin/python -m efficient_frontier --csv data/weekly-prices.csv \
  --periods-per-year 52 --backtests --rebalance-every 4 --rolling-window 52

# Explicit original spreadsheet universe
.venv/bin/python -m efficient_frontier --original-holdings --backtests

# Custom Yahoo symbols, with a 40% target-weight cap
.venv/bin/python -m efficient_frontier --tickers SPY,QQQ,IWM,EFA,TLT,GLD \
  --start 2020-01-01 --max-weight .40 --risk-free-rate .02 --output results/market
```

No source flag selects the offline demo. `--nasdaq100` selects current members under the same eligibility rules as the app. Yahoo inputs default to a start date of `2020-01-01`. Set `--start` and `--end` for a different period.

Yahoo inputs include SPY/QQQ comparisons by default. CSV analysis stays offline unless you request `--download-benchmarks`. Use `--no-benchmarks` to skip the comparison. `--currency` defaults to `USD`. This records a declaration without currency conversion.

The CLI and app use long-only, fully invested weights with no extra position cap by default. Pass `--max-weight .01` for a 1% starting-weight cap. This cap needs at least 100 eligible assets. The cap applies to targets. Prices can move actual weights above it between trades.

The default destination is `results/latest/`. Each run writes `report.html`, `report.zip`, `metadata.json`, and CSV inputs/results. Risk exports include `risk_summary.csv` and `risk_contributions.csv`. The [profile and backtest export list](BACKTESTING.md#reproduce-and-inspect) and [benchmark export list](BENCHMARKS.md#exports) explain the remaining files.

Generated files in the destination are replaced on rerun. Use a separate `--output` folder to preserve an experiment. A shorter rerun removes obsolete generated files and preserves unrelated files. The HTML includes Plotly JavaScript and works offline. Its **Print / save PDF** button uses the browser's print engine.

Keep inputs in ignored `data/` or `private/` folders. The default `results/` folder and root-level `prices.csv` are also ignored. A custom output location can need another ignore rule. Reports include all input prices. Review their contents and data rights before you share them.

CSV exports prefix formula-like asset labels with an apostrophe so spreadsheet software treats them as text. Ordinary symbols and numeric values stay unchanged. An escaped label retains that apostrophe if a program reloads the CSV.

## Classic model and original project

**Research** retains the classic mean-return frontier, minimum-volatility portfolio, maximum-Sharpe portfolio, and equal-weight reference. It estimates the model from the initial training returns. Its original holdout enters at the final training close without trade fees. Separate backtests delay execution and deduct selected costs, so their buy-and-hold result differs.

Maximum Sharpe needs a feasible positive estimated excess return. If none exists, the classic fit omits that portfolio and explains why. Backtests instead use a minimum-volatility fallback for that execution. Their warnings and fallback counts remain in the results.

Simple returns equal `price[t] / price[t-1] - 1`. The split uses `floor(training_fraction × number_of_returns)`. Annual arithmetic means and sample covariance use the chosen observations per year. Mean-return frontier estimates are not CAGR or forecasts. Singular covariance can produce several tied allocations. The [method guide](BACKTESTING.md) defines these estimates and their limits.

The original R script sampled 5–40 holdings from a fixed 60-ticker list. The Python optimizer uses the whole supplied universe, with no fixed input or selected-holdings cap. Forty frontier points means forty return targets, not forty assets. Large universes still face memory, solver, history, and provider limits. See [optimizer capacity](PERFORMANCE.md).

The historical R script replaced the daily mean with a terminal price-ratio expression, omitted the risk-free rate from Sharpe, and allocated ten million weight vectors. Its 12-worker cluster did not serve the serial loop. The Python app corrects those issues. The historical source and spreadsheet remain unchanged.

## Sector labels and exposure

Nasdaq coverage includes provider industry labels. The app calculates allocation and covariance risk. It does not infer ETF holdings or optimize sector percentages from these labels. A sector name does not establish suitability or diversification.

For sector comparisons, use one classification level and date. [GICS](https://www.msci.com/indexes/index-resources/gics) separates sectors, industry groups, industries, and sub-industries. Bonds and gold are separate asset classes. An ETF can contain several sectors and overlap with other holdings.

A broad benchmark can provide a market reference for sector weights. Its composition is not an optimal allocation for individual goals. The [SEC asset-allocation guide](https://www.investor.gov/introduction-investing/getting-started/asset-allocation) explains the role of horizon, risk tolerance, and diversification. Compare weights with shares of portfolio variance, drawdowns, and costs, rather than sector counts alone.
