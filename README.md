![Portfolio Lab · Efficient Frontier](docs/images/header.svg)

[![CI](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml)

A local research app for portfolio allocations, risk, and historical tests. Start with stocks from SPY and the Nasdaq-100, your own tickers, a price CSV, or the offline synthetic demo.

Choose the stocks, history, volatility budget, and largest holding. One model finds the portfolio with the highest historical average return within those limits. One later test compares it with equal-weight stocks, SPY, and QQQ. Inspect every weight and export an offline report. This is a research tool.

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

1. Choose the stock universe and history. The default combines SPY's stock holdings with the current Nasdaq-100.
2. Set **Annual volatility budget (%)** before the calculation. The default is 20%, with a 10% limit per stock.
3. Select **Build portfolio**. The app fits the model on the first 70% of returns and tests it on the remaining 30%.
4. Read the later test and its comparison with equal-weight stocks, SPY, and QQQ. Inspect the latest holdings and trade record.
5. Open **Data & method** for data coverage and method details.
6. Select **Export** for the report and data. The report includes the results on screen.

The app waits for **Build portfolio** before it requests data. Select **Try the offline demo** for a synthetic example without a data download. CSV analysis also works offline after setup.

Keyboard shortcuts: **Ctrl+Enter** on Windows/Linux or **Cmd+Enter** on macOS submits the setup. **4** opens Data & method, and **E** exports. See the [user guide](docs/USER_GUIDE.md) for settings and input rules.

![Portfolio Lab in an earlier layout](docs/images/portfolio-lab.png)

*Screenshot from an earlier app version. The current layout can differ.*

## One model, one test

The model chooses stock weights with the highest historical arithmetic mean return, subject to two limits: fitted volatility and the largest holding. Weights are long-only and sum to 100%. An infeasible limit stops the calculation with an explanation.

The **Annual volatility budget** limits the model's estimated annual volatility. Future volatility can exceed it. It is not a maximum loss. The default budget is 20%, with at most 10% in each stock.

The app uses fixed test assumptions: 70% initial training history, a later 30% test, a refit every 21 observations, and 10-basis-point trading fees. Each refit uses all earlier returns. A 10% covariance shrinkage reduces estimated cross-asset covariances. Each trade uses prices through the preceding close.

The later test compares four paths: **Historical-mean portfolio**, **Equal-weight stocks**, **SPY**, and **QQQ**. Equal-weight stocks use the same trade schedule. The ETFs buy once and hold. The reference portfolios have no volatility budget.

The result checks whether the model stays within the budget in the later test. It also checks whether a reference has higher compound growth with no more volatility. Both ETFs and at least one year of later observations are required for a pass. These checks describe that historical period. They do not prove future outperformance.

The latest holdings use all supplied history and the same model rule. They can differ from the holdings in the test. See the [model and test details](docs/BACKTESTING.md#single-model-and-later-test).

## Stocks from SPY and the Nasdaq-100

The combined universe uses State Street's current SPY stock holdings and Nasdaq's current Nasdaq-100 list. It removes duplicate symbols while preserving separate share classes. SPY holdings are a practical proxy for the S&P 500, not a certified historical index roster. Cash, derivatives, and rights in the fund file do not become stock candidates.

The app records both sources, their dates, and exclusions. Prices must cover the same SPY/QQQ market dates. A long unchanged-price screen uses only the prices available at each fit. It flags possible stale or illiquid quotes without claiming that those quotes are false.

Current membership and complete-history requirements can favor survivors. The model uses only earlier prices at each historical fit. Prior review of the test period still creates research hindsight.

## The Nasdaq-only universe

The optional Nasdaq-100 source requests the [current official constituent list](https://api.nasdaq.com/api/quote/list-type/nasdaq100), with separate share classes. It admits only securities with complete positive adjusted prices on the fixed SPY/QQQ market dates for the selected history.

It records exclusions. It does not fill gaps or shorten the common history. Open **Data & method** for counts and reasons. Reports retain `universe_coverage.csv` and the source date. An unavailable or incomplete constituent list stops the request.

Every eligible asset enters the optimization. Weights can be zero. There is no fixed ticker-count or holdings-count cap, though large inputs need more time and memory. The full weight table and exports retain every asset.

**This is historical analysis of current members.** It does not reconstruct past index membership. Current membership and complete-history eligibility introduce selection bias. A shorter history can admit more assets and also changes the estimation sample.

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

Use complete positive adjusted prices in one currency. The app does not convert currencies or fill missing prices. Set **Change → Data options → Price frequency** to match the file. Annualization does not resample the data.

The model needs at least two returns in both the initial training period and the later test. A pass also requires at least one year of later test returns. See the [price-data rules](docs/USER_GUIDE.md#price-data) for short histories, exchange suffixes, and benchmarks.

## Reports without the browser

After the launcher creates `.venv`, run these commands from the repository root:

```bash
# Offline risk-budget study, with an explicit feasible limit for six demo assets
.venv/bin/python -m efficient_frontier --target-volatility .20 --max-weight .20

# Combined SPY and Nasdaq-100 stocks, 20% volatility budget and 10% per stock
.venv/bin/python -m efficient_frontier --sp500-nasdaq100 --target-volatility .20

# Your price file, with no benchmark download (at least ten eligible assets)
.venv/bin/python -m efficient_frontier --csv data/prices.csv --target-volatility .20
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`. The default output is `results/latest/`. Open `report.html` for the offline report, or choose **Print / save PDF** in it. The ZIP includes the input prices and full results.

A rerun replaces generated files in that destination. Use a separate `--output` folder to preserve a study. Keep private inputs in ignored `data/` or `private/` folders. Review reports before you share them. See [CLI options and exports](docs/USER_GUIDE.md#command-line-options-and-exports).

## Separate research tools

The command line and Python API retain the earlier frontier, multi-method backtests, and historical-membership study. They are separate tools. The app uses the single model above. See the [legacy tools](docs/USER_GUIDE.md#legacy-research-tools) and [historical-membership commands](docs/USER_GUIDE.md#historical-membership).

The [recorded 60-stock study](docs/HOLDINGS_STUDY.md) retains earlier findings, dates, fees, and reproduction steps. It used a stock list from after the test began. It does not establish future returns or outperformance against SPY or QQQ.

All studies omit taxes, currency conversion, and real execution limits. See the [benchmark guide](docs/BENCHMARKS.md) for data requirements and the separate legacy statistical tests.

## Documentation and development

| Guide | Contents |
| --- | --- |
| [User guide](docs/USER_GUIDE.md) | Setup, settings, input rules, CLI options, and reports. |
| [Model and backtests](docs/BACKTESTING.md) | The single model, later test, trade timing, fees, and separate legacy methods. |
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
