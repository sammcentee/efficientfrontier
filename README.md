![Portfolio Lab · Efficient Frontier](docs/images/header.svg)

[![CI](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml)

A local research app for portfolio allocations, risk, and historical tests. Start with the current Nasdaq-100, your own tickers, a price CSV, or the offline synthetic demo.

Compare Low, Medium, and Highest model portfolios. Inspect every weight, evaluate allocation rules against SPY and QQQ, and export an interactive report. This is a research tool, not an investment recommendation or a trading system.

## Start here: no coding needed

1. Install **Python 3.14** from the official [Windows](https://www.python.org/downloads/windows/) or [macOS](https://www.python.org/downloads/macos/) page. Linux needs Python 3.14 with `venv` support.
2. [Download the ZIP](https://github.com/sammcentee/efficientfrontier/archive/refs/heads/main.zip) and extract the whole folder.
3. Open the launcher for your computer:

| Computer | Launcher |
| --- | --- |
| Windows | Double-click `Start Portfolio Lab.bat`. |
| macOS | Double-click `Start Portfolio Lab.command`. |
| Linux or WSL | Run `bash run.sh` from the extracted folder. |

The first launch installs the packages and needs internet access. Later launches reuse the local `.venv`. A browser tab opens at **http://localhost:8501**. Keep the launcher window open. Press **Ctrl+C** there to stop the app.

For setup errors, ports, and remote sessions, see the [user guide](docs/USER_GUIDE.md#setup-and-troubleshooting). Python 3.14 is the tested version. Native Windows startup has a smoke check. macOS startup still needs a check on a Mac.

For a terminal setup:

```bash
git clone https://github.com/sammcentee/efficientfrontier.git
cd efficientfrontier
bash run.sh
```

## Your first study

1. Keep **Nasdaq-100** and **5 years**, or select **Change** for your own data.
2. Select **Show portfolios**. The app obtains prices, fits the portfolios, and runs the historical market test.
3. Choose **Low**, **Medium**, or **Highest**. Read the test comparison first, then inspect the latest proposed holdings.
4. Open **“Would this rule have beaten the market?”** Choose a **Rule**. **Test settings** changes the trade assumptions.
5. Open **Research** for frontiers, risk, data coverage, all tests, and the method.
6. Select **Export** for the report and data. The report includes the results on screen.

The app waits for **Show portfolios** before it requests data. Select **Try the offline demo** for a synthetic example without a data download. CSV analysis also works offline after setup.

Keyboard shortcuts: **Ctrl+Enter** on Windows/Linux or **Cmd+Enter** on macOS submits the setup. **L/M/H** selects a profile, **2** opens the market test, **4** opens Research, and **E** exports. See the [user guide](docs/USER_GUIDE.md) for settings and input rules.

![Portfolio Lab in an earlier layout](docs/images/portfolio-lab.png)

*Screenshot from an earlier app version. The current layout can differ.*

## What the profiles mean

The primary model uses historical arithmetic mean returns and covariance, with the chosen weight limit. Latest holdings fit all supplied history. The frontier minimizes estimated variance at each annualized arithmetic mean return target.

| Profile | Meaning within this model |
| --- | --- |
| Low | Minimum estimated volatility. |
| Medium | The sampled frontier point nearest halfway between Low and Highest volatility. |
| Highest | Highest fitted arithmetic mean return. |

These are relative model positions. Low can still contain risky assets, and profiles can coincide. Highest does not use leverage. CSV and JSON exports retain the name **Extreme** for Highest.

**Highest is not a recommendation.** Without a holding limit, maximizing the fitted mean can put the whole portfolio in one stock. Medium can also have much more risk than either ETF. The overview compares the later results of these rules with maximum Sharpe, equal weight, SPY, and QQQ. It shows arithmetic mean, compound growth, volatility, and drawdown on the same test dates. It does not automatically select the past winner.

Three historical windows show diagnostic mean returns. They do not select the primary weights. **Research** retains a separate weakest-window comparison. Its objective favors the lowest of the three window means.

**Latest holdings are an in-sample fit, not a forecast.** The fit date is the last supplied price date. Separate backtests use the same historical-mean objective with only data before each trade. They evaluate the allocation rule, rather than today's weights. Arithmetic mean return is not compound growth (CAGR). See [profile construction and evaluation](docs/BACKTESTING.md#latest-model-holdings-and-profiles).

## The full Nasdaq-100 universe

The default requests the [current official constituent list](https://api.nasdaq.com/api/quote/list-type/nasdaq100), with separate share classes. It admits only securities with complete positive adjusted prices on the fixed SPY/QQQ market dates for the selected history.

It records exclusions. It does not fill gaps or shorten the common history. Open **Research → Data** for counts and reasons. Reports retain `universe_coverage.csv` and the source date. An unavailable or incomplete constituent list stops the request.

Every eligible asset enters the optimization. Weights can be zero. There is no fixed ticker-count or holdings-count cap, though large inputs need more time and memory. The full weight table and exports retain every asset.

**This is historical analysis of current members.** It does not reconstruct past index membership. Current membership and complete-history eligibility introduce selection bias. A shorter history can admit more assets and also changes the estimation sample.

## Historical membership

Select **Explore historical membership** on the welcome screen, or **Historical membership** in the navigation after a study. Select **Run historical study** to try the offline example. It contains member additions, removals, re-entry, and missing price history.

For real data, upload an adjusted-price CSV and choose the bundled Nasdaq-100 membership or your own membership CSV. This mode stays offline. The bundled community reconstruction covers **2010-01-01 through 2026-09-30**. Its provenance and known gaps appear in the app and report. It is not a certified complete index history. No paid service is required; you must supply suitable historical prices.

Historical prices use the same `Date,ASSET_A,ASSET_B,…` layout, but unavailable prices can be blank. Each identifier must refer to one security throughout its history. Use separate labels for reused tickers. The bundled data blocks known unsafe identities; it does not verify all other price histories. It does not automatically join old membership labels to today's Yahoo symbols.

A custom membership CSV has exactly two columns. Each row is the **full roster** effective on that date, not a list of changes:

```csv
date,tickers
2020-01-01,"ASSET_A,ASSET_B"
2020-06-01,"ASSET_B,ASSET_C"
```

Declare the final date covered by your source. The last roster remains effective through that date. Coverage must include the portfolio baseline date through the final valuation date. Training prices can predate membership coverage; evaluation outside the declared coverage stops the test.

The study compares minimum volatility, maximum Sharpe, and equal weight. Each scheduled trade selects from membership effective on the trade date, using a full rolling window of returns through the prior close. New targets trade at the next close. The model assumes that effective membership is known before execution; it does not model announcement timestamps. Removed members leave at the next scheduled rebalance, so the study is not an exact index tracker. A company can become eligible later when it has enough history. Missing prices for a held asset stop the test; the engine does not fill gaps or silently drop that holding. If no eligible portfolio meets the weight cap, it holds cash until the next allocation. Cash earns zero interest.

The historical report includes membership snapshots, sparse input prices, eligibility reasons at every trade, cash weights, allocations, trades, holdings, and fees. These records help inspect selection bias; they do not prove a complete survivorship-free result. This mode does not run the fixed-universe risk profiles or market benchmark comparisons.

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

Use enough initial training history for the selected rolling window. The demo export can be reproduced with its `prices.csv`, `membership.csv`, coverage end from `metadata.json`, and matching settings. For the bundled source, keep `--historical-membership` when reproducing a run so its security-identity exclusions still apply.

## Your own prices

Select **Change → Upload CSV**. Put `Date` first, followed by one column per asset:

```csv
Date,ASSET_A,ASSET_B
2023-01-03,100.00,80.00
2023-01-04,101.00,79.50
2023-01-05,100.50,80.10
2023-01-06,102.00,80.40
2023-01-09,101.80,81.00
2023-01-10,102.10,80.70
2023-01-11,101.90,81.20
```

Use complete positive adjusted prices in one currency. The app does not convert currencies or fill missing prices. Set **Change → Model assumptions → Price frequency** to match the file. Annualization does not resample the data.

The classic analysis needs at least five price rows and a split with two returns in each period. Latest profiles need seven rows. See the [price-data rules](docs/USER_GUIDE.md#price-data) for short histories, exchange suffixes, and benchmarks.

## Reports without the browser

After the launcher creates `.venv`, run these commands from the repository root:

```bash
# Offline synthetic study with all four backtest rules
.venv/bin/python -m efficient_frontier --backtests

# Current Nasdaq-100 members with complete history from the chosen start
.venv/bin/python -m efficient_frontier --nasdaq100 --start 2021-01-01 --backtests

# Your price file, with no benchmark download
.venv/bin/python -m efficient_frontier --csv data/prices.csv --backtests
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. The default output is `results/latest/`. Open `report.html` for the offline report, or choose **Print / save PDF** in it. The ZIP includes the input prices and full results.

A rerun replaces generated files in that destination. Use a separate `--output` folder to preserve a study. Keep private inputs in ignored `data/` or `private/` folders. Review reports before you share them. See [CLI options and exports](docs/USER_GUIDE.md#command-line-options-and-exports).

## Historical findings and limits

The [recorded 60-stock study](docs/HOLDINGS_STUDY.md) evaluated 2023-05-17 through 2026-10-02, with 10-basis-point fees on bought and sold amounts. Fixed maximum Sharpe returned **256.75%** with heavy concentration. LLY's weight reached **63.48%**. Rolling maximum Sharpe had a smaller drawdown and more trades. Its advantage over equal weight fell to **0.68 percentage points** in the later-period check.

That study used a list recorded after the main evaluation began. It includes selection hindsight and does not establish future returns or statistically supported outperformance against SPY or QQQ. The linked study retains dates, fees, holdings, reproduction steps, and limitations.

Current benchmark comparisons use matched dates and entry rules. They account for all strategy/benchmark tests in the run. Neither a headline return nor a p-value gives the probability of future success. Results omit taxes, FX conversion, and real execution limits. See the [benchmark guide](docs/BENCHMARKS.md).

## Documentation and development

| Guide | Contents |
| --- | --- |
| [User guide](docs/USER_GUIDE.md) | Setup, settings, input rules, CLI options, and reports. |
| [Backtests and methods](docs/BACKTESTING.md) | Profile objectives, trade timing, fees, metrics, and risk. |
| [Market benchmarks](docs/BENCHMARKS.md) | SPY/QQQ data, matched comparisons, and statistical limits. |
| [Historical stock study](docs/HOLDINGS_STUDY.md) | Recorded findings and reproduction steps. |
| [Optimizer capacity](docs/PERFORMANCE.md) | Measured time and memory, with reproduction commands. |
| [Validation record](docs/VALIDATION.md) | Dated checks and their limits. |
| [Repository review](docs/REVIEW.md) | Findings, fixes, and remaining priorities from 9–10 October 2026. |
| [Contributing](CONTRIBUTING.md) | Development setup and review process. |

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

CI runs the automated checks and an offline CLI export. Provider responses are mocked in automated tests. Browser and live-data checks are separate and do not establish physical-phone support.

The app uses Python, CVXPY, and the compiled Rust Clarabel solver. The original `Efficient Frontier v1.12.R` and `spy_holdings.ods` remain historical files. R is not required.

## License and data access

Original code and documentation use the [MIT License](LICENSE). Third-party software and data retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md), the dated [public-source review](docs/PUBLIC_RELEASE_REVIEW.md), and [security reporting](SECURITY.md).

Yahoo access through yfinance is unofficial. A software license does not grant data-access or redistribution rights. Use the integration only where your access and intended use are authorized. Keep downloaded market prices and reports local unless redistribution is permitted.
