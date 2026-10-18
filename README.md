# Efficient Frontier

A local Python app for stock portfolio optimization and historical tests. It estimates returns and covariance from supplied prices, then finds stock weights within a volatility budget and holding limit.

The default universe combines the stocks held by SPY with current Nasdaq-100 constituents. You can also use your own stock symbols, a price CSV, or synthetic demo data. SPY and QQQ provide comparisons and do not enter the stock allocation.

## Setup

Install **Python 3.14**, then [download and extract the repository](https://github.com/sammcentee/efficientfrontier/archive/refs/heads/main.zip).

| System | Start the app |
| --- | --- |
| Windows | Open `Start Portfolio Lab.bat`. |
| macOS | Open `Start Portfolio Lab.command`. |
| Linux or WSL | Run `bash run.sh` from the project folder. |

The first launch installs dependencies and needs internet access. The app opens at **http://localhost:8501**. Keep the launcher open and press **Ctrl+C** there to stop it. Demo and CSV studies work offline after setup.

For a Git checkout:

```bash
git clone https://github.com/sammcentee/efficientfrontier.git
cd efficientfrontier
bash run.sh
```

See the [user guide](docs/USER_GUIDE.md#setup-and-troubleshooting) for setup errors and platform limits.

## Run a study

1. Read the model equations on the **Efficient Frontier** opening page. Open **Settings** to choose stocks and history, or select the **Demo** stock source.
2. Set **Annual volatility budget (%)** and **Holding limit (%)** in the two number fields. The holding limit defaults to 10%.
3. Select **Build portfolio**.
4. Compare **Latest fit** with **Historical replay**, then inspect the frontier and holdings. Open **Data & method** for coverage and trades.
5. Select **Export** for the offline report and exact input/result CSVs.

The app starts the volatility budget at QQQ's latest one-year annualized volatility, with its price date shown. This reference is separate from the stock fit window. Your edits take precedence. If the reference is unavailable, the app labels its 20% manual fallback.

A 10% holding limit needs at least ten eligible stocks. The six-asset synthetic demo uses a 20% limit. An infeasible budget or holding limit stops the calculation.

## Model

The optimizer maximizes the estimated annual arithmetic mean return. Weights are long-only, sum to 100%, and obey the holding limit. The volatility budget limits the covariance-based estimate, not future volatility or losses. The objective is not CAGR or maximum Sharpe. The [R comparison](docs/R_METHOD.md) explains the original method, its return-formula error, and the corrected stock-only results.

The app fixes these assumptions:

| Assumption | Value |
| --- | --- |
| Initial training / later test | First 70% / final 30% of return observations |
| Holding rule | Fit once on the initial sample, buy once, then let weights drift |
| Execution | Close after the last price used in the fit |
| Trading fees | 0.10% of the initial purchase. No later trades or terminal sale |
| Covariance | Ordinary sample covariance, without shrinkage |

The first test interval stays in cash at zero interest. All replay paths then buy once and hold. The model omits taxes, currency conversion, and market impact. The [method reference](docs/BACKTESTING.md#portfolio-model) gives the equations and accounting.

## Read the results

**The latest fit includes the replay dates. The replay did not use the latest holdings.** The fit chooses one allocation using all supplied history. The replay buys one allocation from the initial training sample, then holds it through the later prices without rebalancing. The two panels use different dates, weights, and fee treatment.

| Result | Meaning |
| --- | --- |
| Fitted annual mean | Average return from a fit's history, using its target weights, before fees |
| Fitted volatility | Sample volatility from the fit's dates and target weights; the budget applies to this estimate |
| Fitted frontier | Best estimated stock returns across risk levels, with SPY/QQQ as separate reference points |
| Historical replay | Net performance of the initial allocation, bought once and held through later prices |
| Latest holdings | A new fit using all supplied history; these weights did not produce the earlier test |

Equal-weight stocks, SPY, and QQQ also buy once and hold. The equal-weight reference starts with equal weights but lets them drift. Within the replay, all comparisons use the same dates and fee rate, but their risks can differ. A benchmark with higher mean return and lower volatility beat the stock rule in that replay, even if the latest fitted portfolio sits above it on the fitted frontier chart.

The app displays typeset equations for the optimizer, volatility budget, buy-and-hold accounting, and performance metrics. The introduction appears beside the controls. Fitted moments describe the starting weights. They do not describe a historical buy-and-hold path.

The test status describes coverage, realized risk, and comparison results. It is not proof of future outperformance. Current member lists and complete-history requirements favor survivors. Prior review of the test dates also limits their value as new evidence.

## Command line

Run the app's model without a browser:

```bash
# Offline demo
.venv/bin/python -m efficient_frontier --target-volatility .20 --max-weight .20

# Supplied stock prices, with at least ten eligible stocks
.venv/bin/python -m efficient_frontier --csv data/prices.csv --target-volatility .20
```

On Windows, use `.venv\Scripts\python.exe`. Output goes to `results/latest/`. Use `--output` to preserve separate studies. Reports contain the input prices. Keep private data in ignored `data/` or `private/` folders.

## Reference

- [User guide](docs/USER_GUIDE.md): inputs, settings, reports, and separate legacy tools.
- [Model and accounting](docs/BACKTESTING.md): optimization, trade timing, fees, and metrics.
- [Benchmarks](docs/BENCHMARKS.md): SPY/QQQ comparisons and their limits.
- [Contributing](CONTRIBUTING.md): development setup and checks.
- [Validation record](docs/VALIDATION.md): dated checks and limitations.

The original `Efficient Frontier v1.12.R` and `spy_holdings.ods` remain historical files. R is not required. Earlier findings remain in the [stock study](docs/HOLDINGS_STUDY.md) and [research archive](research/README.md).

Code and documentation use the [MIT License](LICENSE). [Third-party terms](THIRD_PARTY_NOTICES.md) still apply. The software license does not grant rights to download or redistribute market data.
