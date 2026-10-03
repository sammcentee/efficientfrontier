![Portfolio Lab · Efficient Frontier](docs/images/header.svg)

[![CI](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml)

A local research app built from the original `Efficient Frontier v1.12.R` project. Explore portfolio risk and return, inspect allocations, and compare their performance on a later period excluded from optimization.

**New to this? [Start here: no coding needed](#start-here-no-coding-needed).**

## What the stock study found

**Retrospective research:** the fixed list of **60 securities** was first recorded in **October 2024**. Applying it to earlier history introduces selection hindsight; these results are observations, not forecasts or stock recommendations.

The main test ran **17 May 2023–2 October 2026**, in USD, after a **0.10% fee on each amount bought or sold**, before tax and FX. Trades use information from the preceding close; no final liquidation is charged.

- **The highest return came with heavy concentration.** Estimate weights once, then restore them every 21 trading sessions (**fixed maximum Sharpe**): **+256.75%** total return. LLY started at **55.80%** of the portfolio and reached **63.48%** between trades.
- **The rolling method had a smaller fall, but more trading.** Recalculate weights every 21 sessions using the latest 252 daily returns (**rolling maximum Sharpe**): **+130.26%** return, with a **9.91%** worst peak-to-trough fall versus **22.79%** for fixed targets. Gross turnover was **21.53× versus 3.91×**.
- **The advantage changed with the period.** In the later test, **27 January 2025–2 October 2026**, rolling maximum Sharpe returned **28.07%**, versus **27.39%** for restoring equal allocations every 21 sessions—a gap of just **0.68 percentage points**. This was a sensitivity check chosen during research, not an untouched prospective test.

Maximum Sharpe targets estimated return above the risk-free rate per unit of volatility. These methods differed in their initial estimation windows as well as later updates; the comparison does not isolate the benefit of one setting. See the [comparison table](#stock-holdings-comparison) or [complete study](docs/HOLDINGS_STUDY.md) for holdings, costs and limitations.

The app runs in Python with the compiled Rust [Clarabel optimizer](https://clarabel.org/stable/python/getting_started_py/). It considers every supplied ticker, with no fixed ticker-count or holdings-count cap. The original R script and `spy_holdings.ods` are preserved as historical files; R is not needed to run the app.

**Pre-release:** original code and documentation are [MIT licensed](LICENSE). See the [release checklist](docs/RELEASING.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

The [public-source review](docs/PUBLIC_RELEASE_REVIEW.md) records the security, licensing and API checks. Report security issues through the process in [SECURITY.md](SECURITY.md).

![Portfolio Lab showing an efficient frontier for synthetic demonstration assets](docs/images/portfolio-lab.png)

## Start here: no coding needed

Portfolio Lab runs on your computer and opens in your usual web browser. You do not need Git, VS Code or R to use it.

1. **Install Python once.** On Windows, install the [Python Install Manager](https://www.python.org/downloads/windows/); the launcher can then download Python 3.14 for you. On macOS, install Python **3.14** from the [official macOS downloads](https://www.python.org/downloads/macos/). Linux users need Python 3.14 with `venv` support; see the terminal option below.
2. **[Download Portfolio Lab as a ZIP](https://github.com/sammcentee/efficientfrontier/archive/refs/heads/main.zip).** On Windows, right-click the downloaded ZIP and choose **Extract All**. On macOS, double-click the ZIP to unpack it. Open the extracted `efficientfrontier-main` folder before continuing.
3. **Start the app** using the launcher for your computer:

| Your computer | What to open |
| --- | --- |
| Windows | Double-click **`Start Portfolio Lab.bat`**. |
| macOS | Double-click **`Start Portfolio Lab.command`**. |
| Linux or WSL | Open a terminal in the extracted folder and run **`bash run.sh`**. |

The first launch needs an internet connection and may take several minutes to install the app's packages. Later launches reuse that setup; updated requirements trigger another package install. A browser tab opens automatically. If it does not, open **http://localhost:8501** or the **Local URL** shown in the launcher window.

Keep the launcher window open while using the app. To stop it, press **Ctrl+C** in that window. Closing the browser tab alone does not stop the app. Next time, open the same launcher again.

### Your first two minutes

1. Leave **Price data** on **Demo · synthetic**. The sample portfolio loads automatically; you do not need an account, an API key or a price file.
2. Hover over the **Efficient frontier** chart. Further right means more historical price variability; higher up means a higher return estimated from the training data. These estimates are not forecasts.
3. Look at the allocations and switch to **Original holdout** to see how the portfolios performed during the later period excluded from optimization.
4. Try one setting in the sidebar, then click **Build frontier**. When you are comfortable, switch **Price data** to **Upload CSV** or **Yahoo Finance**; the [price-data guide](#price-data) explains what to supply.

### Troubleshooting

| What happened | What to do |
| --- | --- |
| Python was not found, or the launcher asks for Python 3.14 | Install Python using the links above, then reopen the launcher. An older Windows Python launcher needs an installed 3.14 runtime; the current Python Install Manager can fetch it automatically. |
| Setup stopped while downloading packages | Check your internet connection and reopen the launcher. An incomplete package install is retried. |
| The app window disappeared immediately | Extract the entire ZIP first and keep its files together. Use the launcher from the extracted folder. |
| macOS reports that the `.command` file is not executable | Open Terminal, type `bash `, drag `Start Portfolio Lab.command` into the window, then press Enter. |
| No browser tab opened | Visit **http://localhost:8501**. For VS Code Remote/WSL, forward port **8501** from the **Ports** panel if needed. |
| Port 8501 is already in use | Close the earlier Portfolio Lab launcher with **Ctrl+C** and retry. From a terminal, you can instead pass `--server.port=8502` to the launcher and use the URL it prints. |

The synthetic demo and CSV analysis work offline after setup. Yahoo downloads need internet access. Python 3.14 is the tested version; native Windows startup has a smoke check, while macOS startup has not yet been tested on a Mac.

## Terminal setup on Linux or WSL

Install Git and Python 3.14, then clone the repository:

```bash
git clone https://github.com/sammcentee/efficientfrontier.git
cd efficientfrontier
./run.sh
```

Open **http://localhost:8501**. In VS Code Remote/WSL, forward port 8501 if the link does not open automatically. Stop the server with `Ctrl+C`.

The launcher creates a project-local `.venv`, installs dependencies on first use or when `requirements.txt` changes, and opens the browser. Pass `--server.headless=true` to leave browser opening to your remote setup. To install or update an existing environment explicitly:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The app binds to localhost. It needs no credentials or network connection for the synthetic demo or CSV analysis. Yahoo downloads require internet access.

## What you can do

- View the efficient frontier and allocations at individual target returns.
- Compare minimum volatility, maximum Sharpe, and equal-weight portfolios.
- Analyze one ticker or a larger custom universe; the original 60-ticker list is an optional preset.
- Optionally set a maximum starting position size, plus risk-free rate, training fraction, and covariance shrinkage. Position caps are off by default.
- Use Yahoo adjusted daily prices, upload a CSV, or explore a deterministic synthetic demo.
- Select the 60 tickers from the original spreadsheet as a Yahoo universe.
- Compare buy-and-hold performance on the chronological holdout: growth, volatility, Sharpe and drawdown.
- Compare buy-and-hold, periodic rebalancing, expanding-window and rolling-window backtests with delayed execution and configurable trading costs.
- Inspect each holding's contribution, allocation history, selection frequency and concentration after price drift.
- Download an offline interactive HTML report, Markdown findings and CSVs; use the report's **Print / save PDF** button for a static copy.

The demo is explicitly synthetic, with `DEMO_*` asset names. It is a software demonstration, not market history. Yahoo failures are reported rather than replaced with synthetic prices.

## Stock-holdings comparison

The [study above](#what-the-stock-study-found) used adjusted daily prices from **2 January 2020–2 October 2026**, with no additional position cap. The rolling method used 252 prior returns; the other initial estimates used 848. This table covers the main evaluation, **17 May 2023–2 October 2026**. Returns include the stated trading fees, before tax and FX.

| Maximum-Sharpe method | Net total return | CAGR | Maximum drawdown | Realized Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Buy and hold | 225.08% | 41.95% | -24.66% | 1.36 |
| Fixed rebalance | 256.75% | 45.93% | -22.79% | 1.48 |
| Expanding window | 186.77% | 36.76% | -21.04% | 1.39 |
| Rolling window, 252 returns | 130.26% | 28.13% | -9.91% | 1.57 |
| Equal weight, rebalanced baseline | 91.18% | 21.24% | -15.62% | 1.44 |

See [the complete stock study](docs/HOLDINGS_STUDY.md) for all strategies, fee sensitivity, the later-period check and reproduction instructions. [Backtesting methodology](docs/BACKTESTING.md) explains timing and accounting. Downloaded prices and generated market reports remain local and excluded from Git.

## Price data

Upload a CSV with `Date` first and one asset per column:

```csv
Date,ASSET_A,ASSET_B
2023-01-03,100.00,80.00
2023-01-04,101.00,79.50
```

The file needs at least 100 complete daily price observations and one or more assets. There is no hardcoded upper limit on columns. Supply adjusted prices in a common currency. Duplicate dates or columns, nonnumeric values, missing observations and nonpositive prices are rejected. Dates are sorted; assets are not silently removed and prices are not forward-filled. The app cannot infer adjustment status, currency, or whether an uploaded series is genuinely daily.

Yahoo uses `auto_adjust=True` and the adjusted `Close` field. The end date is exclusive. Downloads are cached for one hour in the app. If a requested ticker is unavailable or its history is incomplete, change the ticker list or requested dates and rerun. A narrow common trading calendar works best; cross-market holidays can cause gaps.

Downloaded market data is subject to the provider's terms. The [yfinance project](https://github.com/ranaroussi/yfinance) describes Yahoo's API as intended for personal use and links to the applicable data terms. A software license does not grant permission to redistribute downloaded prices or reports containing them. Use synthetic data for public examples.

The Yahoo integration is unofficial. [Yahoo's terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection without permission, and its [data redistribution guidance](https://help.yahoo.com/kb/SLN2352.html) restricts republication. Selecting Yahoo in this app does not grant that permission; use the integration only where your access and intended use are authorized. The offline demo and CSV input remain available without contacting Yahoo.

The original holdings file is a static list of 60 symbols, **not a full S&P 500 universe or a record of historical membership**. The Yahoo preset requests Marsh as `MRSH`, following its [January 2026 ticker change](https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html) from `MMC`; the historical spreadsheet stays unchanged. Other symbols and availability may change. Choosing today's survivors for a historical analysis introduces survivorship bias.

## Method

1. Calculate daily simple returns, `price[t] / price[t-1] - 1`.
2. Use the first `floor(training_fraction × number_of_returns)` observations for estimation. The rest are held aside.
3. Annualize arithmetic mean returns and sample covariance using 252 sessions per year.
4. Blend covariance toward its diagonal: `(1 - shrinkage) × covariance + shrinkage × diag(covariance)`. The default 10% is a user-controlled assumption, not an automatically fitted estimator.
5. Solve long-only, fully invested portfolios with nonnegative weights summing to one. Any asset can receive zero weight, so the optimizer can select subsets without sampling or enumerating combinations. A per-asset maximum is optional; the default is 100%, imposing no additional concentration restriction. The frontier minimizes variance at target returns on its efficient branch. CVXPY calls the compiled Clarabel solver.
6. Solve maximum Sharpe using a convex change of variables when a feasible portfolio has positive expected excess return. If none does, omit this portfolio and explain why. Near-zero risk yields an undefined Sharpe rather than an artificial infinity.
7. Allocate at the final training price and use each asset's adjusted-price growth throughout the holdout, without subsequent trading between assets. Distributions are reflected in the provider's price adjustments, rather than accumulated as separate cash. The first holdout return starts at the final training price. Equal weight uses the same timing and buy-and-hold convention.

Frontier returns are **historical arithmetic estimates**, not CAGR or forecasts. Holdout annualized growth compounds realized returns, using 252 observations per year. Holdout Sharpe subtracts the selected annual risk-free rate from annualized mean realized daily portfolio returns. Drawdown includes the initial capital, so a loss on the first holdout day counts.

The original holdout's weight cap applies when positions are established. Weights can drift above the cap during the holdout. Cash is not an investable asset; the risk-free input is used only for Sharpe. This original calculation excludes transaction costs, spreads, taxes, FX conversion and execution constraints. The separate backtest comparison deducts selected trading costs and delays execution by one session; its buy-and-hold result is therefore different. The asset universe is user-selected and fixed. Repeatedly selecting settings based on holdout performance contaminates that holdout. This is a research tool, not a trading system or an investment recommendation.

The **Backtests & holdings** tab is enabled by default and compares all four methods. You can disable it when exploring a large frontier alone. No full frontier is reconstructed at each refit; only a small set of portfolio targets is needed. Repeated fits still add computational work. See [the accounting and evaluation rules](docs/BACKTESTING.md).

With singular covariance, such as perfectly correlated assets and zero shrinkage, several allocations can tie for minimum variance at a target return. The curve may include equal-risk points with different returns; a unique allocation is not guaranteed.

## Ticker counts and performance

The historical R simulation sampled only 5–40 holdings from a fixed list of 60 tickers. The Python app accepts any nonempty supplied universe and optimizes weights across the entire universe. It imposes no minimum number of selected holdings, no maximum holdings count, and no fixed input-ticker cap. Forty frontier points means forty target-return levels, not forty assets. A single asset naturally produces one frontier point.

Position limits are optional. Enable **Limit weight per asset** in the app, or pass `--max-weight .01` for a 1% starting-weight cap. Caps below 5% are supported. A chosen cap must still allow the weights to sum to one; for example, a 1% cap needs at least 100 assets.

The computational work already runs in compiled numerical libraries and a Rust solver. The scaling improvements remove redundant bounds and replace a dense maximum-Sharpe constraint with a sparse equivalent; they preserve the same full covariance model. Rewriting the interface in C++ would not by itself change that model's computational cost.

Measured on this Linux/WSL machine, an uncapped 40-point frontier for 1,000 synthetic assets took **20.76 seconds** and **461 MiB peak process memory**. The previous Python implementation took 29.77 seconds and 601 MiB on identical inputs. See [the benchmark method and results](docs/PERFORMANCE.md) for timings at other sizes and commands to reproduce them.

There is no promise of unlimited hardware capacity: dense covariance storage grows quadratically with ticker count, and solving large dense systems becomes progressively more expensive. These timings exclude downloads, price-to-covariance estimation, UI rendering and report generation. Yahoo availability/rate limits and the browser's upload size limit are separate practical constraints. Large universes also need enough history for useful estimates; adding tickers or changing programming language does not guarantee better investment performance.

For large inputs, the correlation chart initially displays a selectable subset to keep the browser responsive. All supplied assets remain in the optimization, allocations and exports.

## Reports without the browser

```bash
# Offline synthetic example
.venv/bin/python -m efficient_frontier

# Your adjusted prices, without an additional position cap
.venv/bin/python -m efficient_frontier --csv prices.csv

# Four backtesting methods, trading every 21 sessions, with 10 bps costs
.venv/bin/python -m efficient_frontier --csv data/holdings.csv \
  --backtests --train-fraction .5 --rolling-window 252 \
  --rebalance-every 21 --cost-bps 10 --output results/holdings-study

# Yahoo example; these symbols are demonstration inputs
.venv/bin/python -m efficient_frontier \
  --tickers SPY,QQQ,IWM,EFA,TLT,GLD \
  --start 2020-01-01 --end 2026-10-03 \
  --max-weight .40 --risk-free-rate .02 --output results/market

# Original spreadsheet universe
.venv/bin/python -m efficient_frontier --original-holdings

# Optional 1% starting-weight limit, using a universe with at least 100 assets
.venv/bin/python -m efficient_frontier --csv data/prices.csv --max-weight .01
```

The default output is `results/latest/`. Each run writes `report.html`, `report.zip`, `metadata.json`, and the CSV inputs/results. Output files in that destination are replaced on rerun; use a different `--output` folder to preserve an experiment. Generated results and downloaded prices are excluded from Git. The HTML report includes Plotly JavaScript and works offline.

With `--backtests`, exports also contain `findings.md`, comparison metrics/curves, and each strategy's holdings, dated target allocations and trade ledger. Open the HTML report and choose **Print / save PDF** for a static copy. The PDF uses the browser's print engine, so no extra Python PDF package is required.

Keep local input files in `data/` or `private/`, which are ignored by Git. The default `results/` directory and a root-level `prices.csv` are also ignored; custom output locations may need an additional ignore rule. Reports include the full input prices, so review their contents before sharing.

CSV exports prefix formula-like asset labels with an apostrophe to keep them as text in spreadsheet software. Numeric values and ordinary ticker labels are unchanged; unusual escaped labels will include that apostrophe if reloaded programmatically.

## Verify

See the [validation record](docs/VALIDATION.md) for browser checks, automated coverage and remaining limitations.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests cover known optimization solutions, constraints, annualization, no lookahead, delayed execution, drift and rebalancing, self-financing trading fees, holdings attribution, drawdown, price validation, report export and app behavior. Network responses are mocked in unit tests.

[GitHub Actions](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml) runs the tests, dependency checks, and an offline demo on pushes to `main` and on pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Project layout

```text
app.py                       Interactive Streamlit app
efficient_frontier/core.py   Estimation, optimization and holdout evaluation
efficient_frontier/backtest.py  Delayed execution, refits, costs and attribution
efficient_frontier/backtest_report.py  Backtest findings and visualizations
efficient_frontier/data.py   Demo, CSV, Yahoo and original spreadsheet inputs
efficient_frontier/presentation.py  Charts and portable reports
efficient_frontier/__main__.py      Command-line runner
tests/                       Mathematical and integration checks
Efficient Frontier v1.12.R   Original script, unchanged
spy_holdings.ods             Original 60-ticker spreadsheet, unchanged
```

The original script overwrote the daily return mean with a terminal price-ratio expression, omitted the risk-free rate from Sharpe, and allocated ten million weight vectors. Its 12-worker cluster was unused by the serial simulation loop. The new implementation addresses those issues without altering the historical files.

Implementation references: [CVXPY quadratic programming](https://www.cvxpy.org/examples/basic/quadratic_program.html), [yfinance download arguments](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), and [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

## License

Original project code and documentation are available under the [MIT License](LICENSE), copyright © 2026 sammcentee. You may use, modify and distribute them, including commercially, provided copies or substantial portions retain the copyright and permission notices.

Third-party software and data are excluded from this license and retain their own terms and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MIT License does not grant rights to access or redistribute market data.
