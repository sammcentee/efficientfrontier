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

Choose the universe and **Annual volatility budget (%)** before **Build portfolio**. The default combines SPY stock holdings and Nasdaq-100 stocks. **Change → Model assumptions** contains the holding limit, development-period share, covariance settings, and trading costs.

| Setting | Effect |
| --- | --- |
| Annual volatility budget (%) | Sets a ceiling on fitted annual volatility. Future realized volatility and losses can exceed it. |
| Largest holding allowed (%) | Caps each stock target. A 10% limit needs at least ten eligible stocks. Price movements can take actual weights above the limit between trades. |
| Development history (%) | Reserves the first part of the history for initial fitting and blend selection. The remaining dates form the later test. |
| Covariance shrinkage (%) | Reduces cross-asset covariances and preserves each asset's variance. |
| Trade interval (CLI) | Counts observed price intervals between refits and trades. The default is 21 trading sessions. |
| Trading cost (basis points, CLI) | Charges each amount bought or sold. Ten basis points equal 0.10%. |
| Risk-free rate (%) | Sets the assumed comparison rate for Sharpe and Sortino. Cash in the ETF comparisons earns zero interest. |
| Price frequency | Sets annualization for a CSV. It does not resample prices. |
| Price currency | Declares the common currency. The app does not check or convert currencies. ETF comparisons require USD prices. |

The default stock holding limit is 10%. The six-asset synthetic demo explicitly uses 20%. Small custom universes can require a larger limit. Change it before the calculation; the app does not relax it automatically.

Covariance describes how asset returns move together. At 0% shrinkage, the model uses sample covariance. At 100%, it sets cross-asset covariance to zero. The adjustment is `(1 - shrinkage) × covariance + shrinkage × diag(covariance)`. The app does not estimate this setting automatically.

The stock rule uses historical arithmetic means. It fits a capped mean-return portfolio and a portfolio closest to equal weight under the same volatility budget. Earlier validation compares three fixed blends of those endpoints. It chooses the blend with the highest net validation CAGR, with ties favoring the diversified baseline. Each blend follows the same expanding-window refits and trade costs.

The selected blend stays fixed during the later test. Stock weights can change at scheduled refits, using only prices through the prior close. The app does not select a different blend after it sees the later result. The historical period can still contain research hindsight because developers or users already examined those prices.

The result status is a descriptive screen, not statistical proof of skill. Full support requires at least one year of return observations in both validation and the later test, both ETF references, and the stated comparison checks. Shorter studies remain available for research. A failed screen keeps the latest candidate collapsed with an explicit explanation.

**Latest candidate holdings** use all observed prices and the frozen blend. They can differ from the holdings used during the test. Holding contributions add up to total return before fees. Subtracting fees gives net total return.

**Annual mean** measures the arithmetic average of net test returns, scaled to a year. **CAGR** compounds the same returns. A gain of 50% followed by a loss of 50% has a zero arithmetic mean but loses 25% of the starting value. The fitted mean uses a different period and different weights.

At each fit, the long unchanged-price screen excludes stocks with at least 20 consecutive unchanged price intervals in the available past data. This can indicate stale or illiquid quotes; it does not prove false data. The export retains the original prices and dated eligibility decisions. If a held stock later triggers that screen, the study stops rather than assume a verified sale at the repeated quote.

## Classic frontier diagnostics

The app's risk choice is the continuous volatility budget. **Research** can explicitly run the legacy analysis from retained prices. It does not restore the old three-level risk selector or change the primary blend.

Legacy frontiers describe fitted arithmetic means and covariance. Their diagnostic targets and original holdouts are separate from the primary validation and later test. The legacy holdout buys the initial weights once and excludes trading costs. A strong fitted point can perform poorly later. See the [CVXPY portfolio model](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html) for the mean and variance definitions.

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

Yahoo uses `auto_adjust=True` and the adjusted `Close` field. The end date is exclusive. The app caches price downloads for one hour. For **My tickers**, incomplete history stops the analysis. Change the symbols or dates and rerun. The combined and Nasdaq-only inputs instead record exclusions under the [coverage rules](../README.md#stocks-from-spy-and-the-nasdaq-100).

Yahoo symbols retain exchange suffixes such as `VOD.L`, `IWDA.AS`, `BMW.DE`, and `7203.T`. Explicit aliases convert `BRK.B` to `BRK-B`, `BRK.A` to `BRK-A`, `BF.B` to `BF-B`, and `BF.A` to `BF-A`. Other periods remain unchanged. International inputs still need a common currency and compatible dates. Different market holidays can create gaps.

The original spreadsheet is a static list of 60 symbols. It is not a full S&P 500 universe or a historical membership record. The Yahoo preset uses Marsh's `MRSH` symbol in place of the spreadsheet's historical `MMC`. Availability and symbols can change.

Yahoo/yfinance is unofficial. The [yfinance project](https://github.com/ranaroussi/yfinance), [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), and [redistribution guidance](https://help.yahoo.com/kb/SLN2352.html) describe access and use limits. The project license does not grant permission to access or redistribute market data. Use synthetic prices for public examples.

## SPY and QQQ comparisons

The app runs validation and the later test as part of **Build portfolio**. The blend comes from earlier validation; the later test cannot choose it.

Yahoo input obtains benchmark prices automatically. For a CSV, use SPY/QQQ columns already in the file, or a separate benchmark file with exactly the same dates. Downloads require an explicit choice. Both stock-universe loaders still need SPY/QQQ dates for the eligibility calendar when benchmark comparisons are disabled.

Synthetic demo results do not use actual market benchmarks. Comparisons need a USD declaration and complete benchmark prices for every asset date. See the [benchmark guide](BENCHMARKS.md) for sources, cost rules, exports, and statistical limits.

## Command-line options and exports

Run the CLI from the repository root. On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

```bash
# Risk-budget study across current SPY and Nasdaq-100 stocks
.venv/bin/python -m efficient_frontier --sp500-nasdaq100 --target-volatility .20

# Offline six-asset synthetic example with an explicit feasible holding limit
.venv/bin/python -m efficient_frontier --target-volatility .20 --max-weight .20

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
