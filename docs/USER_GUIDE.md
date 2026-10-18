# Efficient Frontier Analysis for US Equities user guide

Start with the [README quick start](../README.md#setup). This guide covers setup details, input rules, and command-line reports. The [backtest guide](BACKTESTING.md) defines the model and accounting.

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

The **Efficient Frontier Analysis for US Equities** opening page introduces the model with typeset equations. Two compact number fields set **Annual volatility budget** and **Holding limit**. The **Settings** button opens the stock source, price history, and data options. Select **Demo** there for synthetic prices.

The default universe combines SPY stock holdings and Nasdaq-100 stocks. Select **Build portfolio** to download the stock prices and run the model. During calculation, the app shows progress and **Cancel**. **Settings** returns in the results navigation.

| Setting | Effect |
| --- | --- |
| Annual volatility budget | Limits fitted annual volatility. Future volatility and losses can exceed it. |
| Holding limit | Caps each stock target. A 10% limit needs at least ten eligible stocks. Prices can move actual weights above this limit. |
| Price frequency | Sets annualization for a CSV. It does not resample prices. |
| Price currency | Declares the common currency. The app does not check or convert currencies. ETF comparisons require USD prices. |

The initial volatility budget uses QQQ's adjusted daily returns over the latest trailing year:

```text
QQQ reference volatility = sqrt(252) × sample_standard_deviation(daily returns)
```

The app shows the reference's final price date. This one-year reference does not depend on the stock model's default five-year fit window. Opening the online app fetches the reference through a cache. It does not download the full stock universe until **Build portfolio**. A manual budget edit remains in place. This current reference sets a scenario budget; it does not reproduce a risk target known at the historical entry date.

If QQQ data is unavailable, the app labels a 20% manual fallback. That fallback is not a current QQQ estimate. The CLI continues to use the volatility budget supplied with `--target-volatility`.

The default stock holding limit is 10%. The six-asset synthetic demo explicitly uses 20%. Small custom universes can require a larger limit. Change it before the calculation. The app does not relax it automatically.

The model maximizes historical arithmetic mean return within the volatility budget and holding limit. Stock weights are long-only and sum to one.

## Latest fit and historical replay

Overview starts with the latest fitted holdings, then their mean–variance frontier, then an in-sample and out-of-sample comparison. Each holding shows its company name, weight, ticker, and a short business description when available. These descriptions come from current Yahoo Finance profiles; they are not historical records. CSV and demo studies stay offline and clearly label missing or simulated company information. The full weights remain available below the cards.

The comparison uses five walk-forward tests. The first fit uses the first 50% of returns. The remaining returns form five consecutive test windows of nearly equal length. Each window fits the model on all earlier stock prices only. It buys once at the next close and holds through that window, with no later trades. The in-sample result buys the same allocation and holds it over its own fit dates. Both sides use the same holding rule. Both are measured from the purchase close, so the one cash interval and the entry fee do not weigh more on the short test window.

The section shows the average annual mean on each side and one chart row for each window. A table gives the stock portfolio and equal-weight stocks on both sides of each window. An expander gives every path, with annual mean and volatility. SPY, QQQ, and equal-weight stocks use the same dates as the stock portfolio. Equal-weight stocks use no return estimates, so their change between the two sides is a rough market control. Their holdings differ from the stock portfolio, so the control is approximate.

In-sample results tend to look better. The optimizer favors stocks with high past means, within the volatility budget and holding limit, and part of each past mean is chance. The default volatility budget uses QQQ's latest year, which ends after the walk-forward fits. The section says so when that default applies. A window with an infeasible fit keeps its dates and the reason. Histories shorter than 29 returns cannot form five windows, and the section says so. The buy-and-hold growth chart, replay metrics, and exact return reconciliation follow.

The date summary above the chart separates the initial fit (first 70% of returns), replay (final 30%), and latest frontier (all returns). These are observation counts, not fixed calendar periods. Five years of daily prices gives roughly three and a half years for the initial fit and eighteen months for the replay. Annualized metrics describe units, not a one-year test. Each run shows its actual return dates.

The efficient frontier depends on the return estimates, covariance estimates, and allowed holdings. The optimizer finds the highest fitted mean within your limits. It does not maximize CAGR or the return-to-volatility ratio. It cannot ensure that the chosen stocks will have the highest later return.

| Number | What it measures |
| --- | --- |
| Fitted annual mean | Average return in the fitting sample, using that fit's target weights, before fees. |
| Fitted volatility | Sample volatility from the same fitting dates and target weights. The budget limits this estimate. |
| Fitted frontier | Feasible stock portfolios with the highest estimated return at each risk level. |
| Replay mean and volatility | Net results after one initial purchase, with no further trades and weights that drift with prices. |
| In-sample and out-of-sample means | Annual means of one walk-forward allocation while held, from the purchase close, on its fit dates and on its later test window. |

**The latest fit includes the replay dates. The replay did not use the latest holdings.** The comparison below the replay shows the two calculations side by side. **Latest fitted frontier · in-sample** uses one allocation chosen from the full history. **Historical replay · buy and hold** buys one allocation from the initial training sample and holds it through the later prices. Both panels report an annual arithmetic mean, but they use different dates, weights, and fee treatment.

SPY and QQQ use the same dates as the stocks within each panel. Their results also change between panels because the date ranges differ. They do not enter the stock allocations. **Data & method** shows the initial and latest fitted estimates. These are two separate allocations, not a sequence of replay trades.

Inline typeset equations show the optimization constraints, volatility estimate, buy-and-hold value path, and metric definitions. The fitted mean and volatility use the same fixed target weights for every observation in the estimation sample. They describe the starting allocation. A buy-and-hold path instead has weights that drift, even before differences in dates and fees.

If QQQ earns more with less volatility in the later test, it beat the portfolio rule on those dates. That is a real weakness in the test result. It does not show that the optimizer failed to maximize the earlier estimates. A mathematically optimal fit can still be a poor investment rule when those estimates do not persist.

## Fixed assumptions

The app fixes the other assumptions:

- Use the first 70% of return observations for the initial fit and the final 30% for the later test.
- Fit once on the initial training sample. Buy once, then hold without rebalancing.
- For the walk-forward tests, fit separately on the first 50%, 60%, 70%, 80%, and 90% of returns. Hold each allocation through the next tenth of the history only.
- Charge 10 basis points, or 0.10%, on the initial purchase. Do not charge a terminal sale.
- Use ordinary sample covariance, without shrinkage.

The initial fit uses prices through the close before execution. The first test interval stays in cash. The portfolio buys at that interval's close and earns returns from the next interval. It does not reset weights, add stocks, or sell stocks during the replay. The latest full-history fit remains separate.

The result status checks the model's later performance. A pass needs both ETF references, at least one year of later returns, and realized volatility within the budget. A pass also requires that no reference has higher net CAGR with no more volatility. These are descriptive checks, not statistical proof. The latest holdings remain visible regardless of the replay result.

**Annual mean** averages the net test returns and scales that average to a year. **CAGR** compounds the same returns. A 50% gain followed by a 50% loss has a zero arithmetic mean but loses 25% of the initial value. The fitted mean uses different dates and weights.

The initial and latest fits exclude stocks with at least 20 consecutive unchanged price intervals in their respective samples. This can indicate stale or illiquid quotes. It does not prove false data. The export retains the original prices and dated eligibility decisions. The later screen affects the separate latest allocation, not the stocks already held in the replay.

## Stock universe

The default combines State Street's current SPY stock holdings with Nasdaq's current Nasdaq-100 list. Duplicate symbols appear once. Separate share classes remain separate. SPY holdings are a proxy for S&P 500 membership, not a certified historical roster. Cash, derivatives, and rights in the holdings file do not enter the stock universe.

Both stock-universe loaders require complete positive adjusted prices on the SPY/QQQ market calendar. They record exclusions without filling gaps or shortening the history. **Data & method** and `universe_coverage.csv` show the source dates, counts, and reasons. An unavailable or incomplete constituent list stops the request.

Current membership and complete-history requirements favor surviving stocks. The results do not reconstruct past index membership. A shorter history can admit more stocks but also changes the estimation sample.

The app reserves SPY and QQQ for market comparisons. If either appears in a ticker list or CSV, the app separates it from stock allocations and reports that change. The separate general optimizer still accepts fund portfolios.

The optimizer considers every eligible stock and can assign zero weight. There is no fixed stock-count limit. Large inputs still face data, memory, and solver limits. See [optimizer capacity](PERFORMANCE.md).

## Price data

Upload a CSV with `Date` first and one asset per column. Use adjusted prices in a common currency. The file needs at least five complete price rows and one asset. The holding limit must be feasible for the eligible asset count.

The file layout is:

```csv
Date,STOCK_A,STOCK_B
2023-01-03,100.00,80.00
2023-01-04,101.00,79.50
2023-01-05,100.50,80.10
2023-01-06,102.00,80.40
2023-01-09,101.80,81.00
2023-01-10,102.10,80.70
```

The model needs at least two returns in both the initial training period and the later test. With the app's 70/30 split, supply at least six price rows. Five rows work only with a suitable CLI split, such as 50/50. Short histories and large asset counts make estimates less reliable. A larger sample does not establish a reliable forecast.

The app rejects duplicate dates or columns, nonnumeric values, missing observations, and nonpositive prices. It sorts dates without filling gaps. It cannot confirm price adjustments, currency, or the interval between observations.

Set **Settings → Data options → Price frequency** to match the CSV:

| Interval | Observations per year |
| --- | ---: |
| Daily trading sessions | 252 |
| Daily calendar observations | 365 |
| Weekly | 52 |
| Monthly | 12 |
| Custom app setting | A finite number of at least 1 |

The CLI uses `--periods-per-year`, with a default of 252. It accepts any positive finite value. This factor changes annualization only. It does not convert daily prices to weekly or monthly prices.

Yahoo uses `auto_adjust=True` and the adjusted `Close` field. The end date is exclusive. The app caches price downloads for one hour. For **My tickers**, incomplete history stops the analysis. Change the symbols or dates and rerun. The combined and Nasdaq-only inputs instead record exclusions under the [coverage rules](#stock-universe).

Yahoo symbols retain exchange suffixes such as `VOD.L`, `IWDA.AS`, `BMW.DE`, and `7203.T`. Explicit aliases convert `BRK.B` to `BRK-B`, `BRK.A` to `BRK-A`, `BF.B` to `BF-B`, and `BF.A` to `BF-A`. Other periods remain unchanged. International inputs still need a common currency and compatible dates. Different market holidays can create gaps.

The original spreadsheet is a static list of 60 symbols. It is not a full S&P 500 universe or a historical membership record. The Yahoo preset uses Marsh's `MRSH` symbol in place of the spreadsheet's historical `MMC`. Availability and symbols can change.

Yahoo/yfinance is unofficial. The [yfinance project](https://github.com/ranaroussi/yfinance), [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), and [redistribution guidance](https://help.yahoo.com/kb/SLN2352.html) describe access and use limits. The project license does not grant permission to access or redistribute market data. Use synthetic prices for public examples.

## SPY and QQQ comparisons

The app runs one later replay and five walk-forward tests as part of **Build portfolio**. This section describes the replay. It compares the model with equal-weight eligible stocks, SPY, and QQQ. All four paths buy once and hold. These three references have no volatility budget.

Equal-weight stocks give each initially eligible stock the same starting weight, then let those weights drift. SPY tracks the S&P 500, and QQQ tracks the Nasdaq-100. These references do not match the model's volatility. Compare each row's return and realized volatility together.

Yahoo input obtains benchmark prices automatically. For a CSV, use SPY/QQQ columns already in the file, or a separate benchmark file with exactly the same dates. Downloads require an explicit choice. Both stock-universe loaders still need SPY/QQQ dates for the eligibility calendar when benchmark comparisons are disabled.

Synthetic demo results do not use actual market benchmarks. Comparisons need a USD declaration and complete benchmark prices for every asset date. See the [benchmark guide](BENCHMARKS.md) for sources, cost rules, exports, and statistical limits.

## Command-line options and exports

Run the CLI from the repository root. On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

```bash
# The app's model across current SPY and Nasdaq-100 stocks
.venv/bin/python -m efficient_frontier --sp500-nasdaq100 --target-volatility .20

# Offline six-asset example with a feasible holding limit
.venv/bin/python -m efficient_frontier --target-volatility .20 --max-weight .20

# Offline stock and benchmark files with identical dates
.venv/bin/python -m efficient_frontier --csv data/prices.csv \
  --target-volatility .20 --benchmark-csv data/benchmarks.csv --currency USD

# All options, including separate legacy research tools
.venv/bin/python -m efficient_frontier --help
```

Use `--target-volatility` to run the app's stock model from the CLI. The CLI also accepts explicit test assumptions for reproduction. `--train-fraction .7` and `--cost-bps 10` match the app. The stock model always buys once and holds. The CLI rejects `--rebalance-every` with `--target-volatility`. That option belongs to the separate legacy `--backtests` tools.

`--shrinkage 0` matches the app's sample covariance. Price frequency uses `--periods-per-year`, which defaults to 252.

No source flag selects the offline demo. `--nasdaq100` selects current members under the same eligibility rules as the app. Yahoo inputs default to a start date of `2020-01-01`. Set `--start` and `--end` for a different period.

Yahoo inputs include SPY/QQQ comparisons by default. CSV analysis stays offline unless you request `--download-benchmarks`. Use `--no-benchmarks` to skip the comparison. `--currency` defaults to `USD`. This records a declaration without currency conversion.

The stock model uses long-only, fully invested weights and a 10% holding limit by default. Pass `--max-weight` to set a different limit. A 1% limit needs at least 100 eligible assets. The cap applies to the initial target weights. Prices can move actual weights above it during the holding period.

The default destination is `results/latest/`. Each run writes `report.html`, `report.zip`, `metadata.json`, and exact CSV inputs and results. The app export contains the later test, the walk-forward windows, latest weights, fitted estimates, and dated eligibility decisions. See the [export list](BACKTESTING.md#app-exports).

Generated files in the destination are replaced on rerun. Use a separate `--output` folder to preserve an experiment. A shorter rerun removes obsolete generated files and preserves unrelated files. The HTML includes Plotly JavaScript and works offline. Its **Print / save PDF** button uses the browser's print engine.

Keep inputs in ignored `data/` or `private/` folders. The default `results/` folder and root-level `prices.csv` are also ignored. A custom output location can need another ignore rule. Reports include all input prices. Review their contents and data rights before you share them.

CSV exports prefix formula-like asset labels with an apostrophe so spreadsheet software treats them as text. Ordinary symbols and numeric values stay unchanged. An escaped label retains that apostrophe if a program reloads the CSV.

## Legacy research tools

The CLI without `--target-volatility` retains the classic frontier tools. `--backtests` adds the earlier multi-method comparison. These tools remain separate from the app. They accept general asset portfolios, including funds. Their CLI covariance shrinkage defaults to 10%. The [legacy method reference](BACKTESTING.md#legacy-strategy-comparison) defines their objectives and exports.

```bash
# Legacy four-method comparison using supplied prices
.venv/bin/python -m efficient_frontier --csv data/prices.csv --backtests

# Legacy rolling-window study with weekly data
.venv/bin/python -m efficient_frontier --csv data/weekly-prices.csv \
  --periods-per-year 52 --backtests --rebalance-every 4 --rolling-window 52

# Legacy original spreadsheet universe
.venv/bin/python -m efficient_frontier --original-holdings --backtests
```

The classic analysis contains a mean-return frontier, minimum-volatility portfolio, maximum-Sharpe portfolio, and equal-weight reference. It estimates the model from the initial training returns. Its original holdout enters at the final training close without trade fees. Separate backtests delay execution and deduct selected costs, so their buy-and-hold result differs.

Maximum Sharpe needs a feasible positive estimated excess return. If none exists, the classic fit omits that portfolio and explains why. Backtests instead use a minimum-volatility fallback for that execution. Their warnings and fallback counts remain in the results.

Simple returns equal `price[t] / price[t-1] - 1`. The split uses `floor(training_fraction × number_of_returns)`. Annual arithmetic means and sample covariance use the chosen observations per year. Mean-return frontier estimates are not CAGR or forecasts. Singular covariance can produce several tied allocations. The [method guide](BACKTESTING.md) defines these estimates and their limits.

The original R script sampled 5–40 holdings from a fixed 60-ticker list. The Python optimizer uses the whole supplied universe, with no fixed input or selected-holdings cap. Forty frontier points means forty return targets, not forty assets. Large universes still face memory, solver, history, and provider limits. See [optimizer capacity](PERFORMANCE.md).

The historical R script replaced the daily mean with a terminal price-ratio expression, omitted the risk-free rate from Sharpe, and allocated ten million weight vectors. Its 12-worker cluster did not serve the serial loop. The current Python model uses arithmetic mean returns and an optimizer. It does not reproduce the R script's return calculation or random search. The historical source and spreadsheet remain unchanged.

## Historical membership

The separate historical-membership CLI and Python API accept dated member lists. The offline example contains member additions, removals, re-entry, and missing price history.

For real data, supply an adjusted-price CSV and the bundled Nasdaq-100 membership or your own membership CSV. This tool stays offline. The bundled community reconstruction covers **2010-01-01 through 2026-09-30**. Its provenance and known gaps appear in the report. It is not a certified complete index history.

You do not need a paid service. Supply suitable historical prices.

Historical prices use the same `Date,ASSET_A,ASSET_B,…` layout, but unavailable prices can be blank. Each identifier must refer to one security throughout its history. Use separate labels for reused tickers. The bundled data blocks known unsafe identities. It does not verify all other price histories. It does not automatically join old membership labels to today's Yahoo symbols.

A custom membership CSV has exactly two columns. Each row is the **full roster** effective on that date, not a list of changes:

```csv
date,tickers
2020-01-01,"ASSET_A,ASSET_B"
2020-06-01,"ASSET_B,ASSET_C"
```

Declare the final date covered by your source. The last roster remains effective through that date. Coverage must include the portfolio baseline date through the final valuation date. Training prices can predate membership coverage. Evaluation outside the declared coverage stops the test.

The study compares minimum volatility, maximum Sharpe, and equal weight. Each scheduled trade selects from membership effective on the trade date, using a full rolling window of returns through the prior close. New targets trade at the next close. The model assumes that effective membership is known before execution. It does not model announcement timestamps.

Removed members leave at the next scheduled rebalance, so the study is not an exact index tracker. A company can become eligible later when it has enough history. Missing prices for a held asset stop the test. The engine does not fill gaps or silently drop that holding. If no eligible portfolio meets the weight cap, it holds cash until the next allocation. Cash earns zero interest.

The historical report includes membership snapshots, sparse input prices, eligibility reasons at every trade, cash weights, allocations, trades, holdings, and fees. These records help inspect selection bias. They do not prove a complete survivorship-free result. This mode does not run the fixed-universe risk profiles or market benchmark comparisons.

```bash
# Fully offline example; rolling window defaults to 126 returns
.venv/bin/python -m efficient_frontier --historical-demo --output results/historical-demo

# Bundled historical membership with your supplied price history
.venv/bin/python -m efficient_frontier --historical-membership \
  --csv private/historical-prices.csv --output results/historical

# Your own full membership snapshots and explicit final covered date
.venv/bin/python -m efficient_frontier --membership-csv private/membership.csv \
  --membership-end 2024-12-31 --csv private/prices.csv \
  --rolling-window 126 --rebalance-every 21 --cost-bps 10 \
  --output results/custom-historical
```

Use enough initial training history for the selected rolling window. To reproduce the demo, use its `prices.csv`, `membership.csv`, coverage end from `metadata.json`, and the same settings. For the bundled source, keep `--historical-membership` when reproducing a run so its security-identity exclusions still apply.

## Sector labels and exposure

Nasdaq coverage includes provider industry labels. The app calculates allocation and covariance risk. It does not infer ETF holdings or optimize sector percentages from these labels. A sector name does not establish suitability or diversification.

For sector comparisons, use one classification level and date. [GICS](https://www.msci.com/indexes/index-resources/gics) separates sectors, industry groups, industries, and sub-industries. Bonds and gold are separate asset classes. An ETF can contain several sectors and overlap with other holdings.

A broad benchmark can provide a market reference for sector weights. Its composition is not an optimal allocation for individual goals. The [SEC asset-allocation guide](https://www.investor.gov/introduction-investing/getting-started/asset-allocation) explains the role of horizon, risk tolerance, and diversification. Compare weights with shares of portfolio variance, drawdowns, and costs, rather than sector counts alone.
