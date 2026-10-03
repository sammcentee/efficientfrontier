# Efficient Frontier · Portfolio Lab

A local research app built from the original `Efficient Frontier v1.12.R` project. Explore portfolio risk and return, inspect allocations, and compare their performance on a later period excluded from optimization.

The original R script and `spy_holdings.ods` are preserved unchanged. The Python app replaces the ten-million-portfolio simulation with constrained convex optimization.

## Run on Linux

```bash
cd /root/projects/efficientfrontier
./run.sh
```

Open **http://localhost:8501**. In VS Code Remote/WSL, forward port 8501 if the link does not open automatically. Stop the server with `Ctrl+C`.

The launcher creates a project-local `.venv` and installs dependencies on first use. Python 3.14 is the tested runtime. To install or update an existing environment explicitly:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The app binds to localhost. It needs no credentials or network connection for the synthetic demo or CSV analysis. Yahoo downloads require internet access.

## What you can do

- View the efficient frontier and allocations at individual target returns.
- Compare minimum volatility, maximum Sharpe, and equal-weight portfolios.
- Set a maximum starting position size, risk-free rate, training fraction, and covariance shrinkage.
- Use Yahoo adjusted daily prices, upload a CSV, or explore a deterministic synthetic demo.
- Select the 60 tickers from the original spreadsheet as a Yahoo universe.
- Compare buy-and-hold performance on the chronological holdout: growth, volatility, Sharpe and drawdown.
- Download an offline interactive HTML report and CSVs containing prices, settings, weights, frontier points and holdout results.

The demo is explicitly synthetic, with `DEMO_*` asset names. It is a software demonstration, not market history. Yahoo failures are reported rather than replaced with synthetic prices.

## Price data

Upload a CSV with `Date` first and one asset per column:

```csv
Date,ASSET_A,ASSET_B
2023-01-03,100.00,80.00
2023-01-04,101.00,79.50
```

The file needs at least 100 complete daily price observations and two assets. Supply adjusted prices in a common currency. Duplicate dates or columns, nonnumeric values, missing observations and nonpositive prices are rejected. Dates are sorted; assets are not silently removed and prices are not forward-filled. The app cannot infer adjustment status, currency, or whether an uploaded series is genuinely daily.

Yahoo uses `auto_adjust=True` and the adjusted `Close` field. The end date is exclusive. Downloads are cached for one hour in the app. If a requested ticker is unavailable or its history is incomplete, change the ticker list or requested dates and rerun. A narrow common trading calendar works best; cross-market holidays can cause gaps.

The original holdings file is a static list of 60 symbols, **not a full S&P 500 universe or a record of historical membership**. Symbols and availability may have changed. Choosing today's survivors for a historical analysis introduces survivorship bias.

## Method

1. Calculate daily simple returns, `price[t] / price[t-1] - 1`.
2. Use the first `floor(training_fraction × number_of_returns)` observations for estimation. The rest are held aside.
3. Annualize arithmetic mean returns and sample covariance using 252 sessions per year.
4. Blend covariance toward its diagonal: `(1 - shrinkage) × covariance + shrinkage × diag(covariance)`. The default 10% is a user-controlled assumption, not an automatically fitted estimator.
5. Solve long-only, fully invested portfolios subject to `0 ≤ weight ≤ maximum_weight`. The frontier minimizes variance at target returns on its efficient branch. CVXPY uses the Clarabel solver.
6. Solve maximum Sharpe using a convex change of variables when a feasible portfolio has positive expected excess return. If none does, omit this portfolio and explain why. Near-zero risk yields an undefined Sharpe rather than an artificial infinity.
7. Allocate at the final training price and use each asset's adjusted-price growth throughout the holdout, without subsequent trading between assets. Distributions are reflected in the provider's price adjustments, rather than accumulated as separate cash. The first holdout return starts at the final training price. Equal weight uses the same timing and buy-and-hold convention.

Frontier returns are **historical arithmetic estimates**, not CAGR or forecasts. Holdout annualized growth compounds realized returns, using 252 observations per year. Holdout Sharpe subtracts the selected annual risk-free rate from annualized mean realized daily portfolio returns. Drawdown includes the initial capital, so a loss on the first holdout day counts.

The weight cap applies when positions are established. Weights can drift above the cap during the holdout. Cash is not an investable asset; the risk-free input is used only for Sharpe. Results exclude transaction costs, spreads, taxes, FX conversion and execution constraints. The asset universe is user-selected and fixed. Repeatedly selecting settings based on holdout performance contaminates that holdout. This is a research tool, not a trading system or an investment recommendation.

With singular covariance, such as perfectly correlated assets and zero shrinkage, several allocations can tie for minimum variance at a target return. The curve may include equal-risk points with different returns; a unique allocation is not guaranteed.

## Reports without the browser

```bash
# Offline synthetic example
.venv/bin/python -m efficient_frontier

# Your adjusted prices
.venv/bin/python -m efficient_frontier --csv prices.csv --max-weight .50

# Yahoo example; these symbols are demonstration inputs
.venv/bin/python -m efficient_frontier \
  --tickers SPY,QQQ,IWM,EFA,TLT,GLD \
  --start 2020-01-01 --end 2026-10-03 \
  --max-weight .40 --risk-free-rate .02 --output results/market

# Original spreadsheet universe
.venv/bin/python -m efficient_frontier --original-holdings --max-weight .10
```

The default output is `results/latest/`. Each run writes `report.html`, `report.zip`, `metadata.json`, and the CSV inputs/results. Output files in that destination are replaced on rerun; use a different `--output` folder to preserve an experiment. Generated results and downloaded prices are excluded from Git. The HTML report includes Plotly JavaScript and works offline.

## Verify

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests cover known two-asset optimization solutions, constraints, annualization, covariance shrinkage, no holdout lookahead, buy-and-hold accounting, drawdown, price validation, Yahoo response handling, report export and app behavior. Network responses are mocked in unit tests.

## Project layout

```text
app.py                       Interactive Streamlit app
efficient_frontier/core.py   Estimation, optimization and holdout evaluation
efficient_frontier/data.py   Demo, CSV, Yahoo and original spreadsheet inputs
efficient_frontier/presentation.py  Charts and portable reports
efficient_frontier/__main__.py      Command-line runner
tests/                       Mathematical and integration checks
Efficient Frontier v1.12.R   Original script, unchanged
spy_holdings.ods             Original 60-ticker spreadsheet, unchanged
```

The original script overwrote the daily return mean with a terminal price-ratio expression, omitted the risk-free rate from Sharpe, and allocated ten million weight vectors. Its 12-worker cluster was unused by the serial simulation loop. The new implementation addresses those issues without altering the historical files.

Implementation references: [CVXPY quadratic programming](https://www.cvxpy.org/examples/basic/quadratic_program.html), [yfinance download arguments](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), and [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
