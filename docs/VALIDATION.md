# Validation record

## Review and simplification: 10 October 2026

**485 tests passed** in a fresh local environment with Python 3.14.7 and the pinned dependencies. `pip check` passed.
The suite reports one existing pandas warning from the invalid-date rejection test.

Regression tests cover boolean and complex values in object columns, incomplete Nasdaq batches, approximate solver warnings, and risk-free-rate consistency.
App tests cover one market calculation per successful build and the original holdout fallback after a failed backtest.
Shared app/report tests check currency declarations, monthly observation labels, rolling return counts, and concentration text.

A fixed synthetic backtest used 64 assets and 360 price rows. All results for 24 strategies stayed exactly equal.
Solver calls fell from 91 to 78. This count does not establish a general runtime guarantee.

A mocked 100-stock request with four incomplete members fell from 105 requests to five. The same 96 assets remained eligible.
Whole-batch download failures still trigger individual retries. Every exclusion retains a reason.

The offline demo and a synthetic monthly CSV both completed CLI exports. The monthly fixture includes synthetic SPY/QQQ-labelled series.
Its 102 archive files matched the extracted files. The HTML rendered eight charts at 390 × 844 without external requests, uncaught JavaScript errors, or page overflow.
The report correctly states the currency declaration and the rolling count of return observations.

The background T3 preview completed the demo and the Highest keyboard shortcut before its desktop host disconnected.
Headless Chromium then completed monthly CSV input, changed test settings, the 102-file export, and Research navigation without uncaught app errors.
The app had no page overflow at 1440 × 900 or 390 × 844. The frequency-menu check waits for the drawer animation before interaction.
Native Windows, macOS, and physical-phone checks remain incomplete.
This review made no live Nasdaq or Yahoo data request. See the [review findings](REVIEW.md) for scope and remaining limits.

## Nasdaq-100 workspace: 4 October 2026

**391 tests passed**, including **28 Streamlit AppTest cases**. Dependency checks passed on Python 3.14 and Streamlit 1.65. The suite reports one existing pandas warning in the invalid-date rejection test.

The app now starts with the current Nasdaq-100 and five years of history. The main views are **Portfolio**, **Compare**, and **Research**. The app and exported reports use a light theme. The Deploy control is absent in the local app.

### Data and calculation checks

- Nasdaq returned 101 current securities, with a source date of 1 October 2026. Separate share classes remain separate assets.
- The default Yahoo request produced 1,255 market dates from 4 October 2021 through 2 October 2026. It included 92 securities and excluded nine. Every requested security appears in the coverage table with its status and reason.
- An independent audit used the browser's exported ZIP. All 103 files matched the extracted files. Coverage, price columns, dates, and latest weight totals reconciled.
- All 24 backtests reconciled holding profits less fees to net returns. The largest difference was `2e-15`.
- Both benchmark paths matched the saved prices, common dates, delayed entry, and 10 basis-point entry fee. All 48 comparisons, 96 statistical tests, and 144 evaluation-window rows remained in the export.

The default **Expanding window · Medium** test ran from 3 April 2025 through 2 October 2026. Its annualized growth was **9.36%**, versus **28.57% for SPY** and **41.07% for QQQ**. Both comparisons showed no clear statistical advantage after adjustment. This sample differs from the earlier six-asset checks below. It does not establish future performance.

### App checks

The interaction tests cover explicit calculation and export actions, pending inputs, applied settings, benchmark fees, and recovery after failures. They also cover repeated Nasdaq requests from the cache, replacement CSVs with the same filename, and removal of obsolete failure notes after a successful retry.

The background T3 browser completed the live Nasdaq request, risk selection, benchmark comparison, and report preparation. Navigation between views returned to the top. Repeated portfolio updates completed without the former progress-display cache error.

At 390 × 844, the navigation fit, metric cards stacked, and the comparison title remained inside its chart. The page had no horizontal overflow. Separate demo checks also covered 300-pixel width. The offline report rendered seven charts at 300 and 390 pixels without external resource requests. Its chart titles fit, and wide tables stayed in scroll areas.

A synthetic 100-asset timing check completed the main analysis in 0.17 seconds and all 24 backtests in 2.05 seconds. This measures local calculation only. It excludes downloads and does not guarantee performance on another computer.

These checks used Linux/WSL and a desktop browser. They do not establish physical-phone or native Windows/macOS support. Live data access was available during this run. Automated provider tests use mocked responses. Current membership and complete-history eligibility remain sources of historical selection bias. No new PDF was generated.

## Market benchmarks and historical evidence: 4 October 2026

**359 tests passed**, including **25 Streamlit AppTest cases**. Dependency checks passed on Python 3.14 and Streamlit 1.65 on Linux/WSL.

- SPY and QQQ comparisons cover both frontiers, the original holdout, all backtests, the app, CLI, and report exports.
- Data tests cover exact dates, separate downloads, weekly and monthly observations, malformed prices, and missing benchmark history. Comparisons do not fill or remove portfolio dates.
- App and CLI tests cover offline CSV inputs, explicit download choices, currency controls, and recovery from benchmark failures. Chart filters preserve all statistical tests and exports.
- Benchmark tests verify the original holdout with no fees and delayed backtest entry with the same fee rate as each portfolio.
- An independent dense-matrix calculation matched the HAC coefficients and standard errors within `2.78e-17`. A separate closed-testing calculation matched the Holm adjustment, including undefined tests.
- Tests cover short histories, constant benchmarks, insufficient residual variation, return scaling, and invalid equity paths. Undefined uncertainty remains undefined.
- The new app copy and benchmark guide passed the STE structural checks. Technical terms still require manual review.

The suite reports the existing pandas warning in the invalid-date rejection test. That test passes.

### Live data and exported results

Yahoo returned 1,697 adjusted-price rows from 2 January 2020 through 2 October 2026 for the six default example assets. The browser also completed a separate benchmark download for a four-asset portfolio without SPY or QQQ.

The default **Expanding window · Medium** comparison used a baseline of 20 September 2024 and entered at the next close. Results ended on 2 October 2026. The selected trading fee was 10 basis points per unit bought or sold.

| Evaluated path | Annualized growth | Medium portfolio difference |
| --- | ---: | ---: |
| Expanding window · Medium | 20.68% | — |
| S&P 500 proxy (SPY) | 17.32% | +3.36 percentage points |
| Nasdaq-100 proxy (QQQ) | 24.86% | −4.18 percentage points |

This fixed display choice beat SPY and trailed QQQ in this sample. Neither mean advantage nor alpha passed the adjusted tests against either benchmark. These results do not establish a persistent advantage or give a probability of future success.

Independent calculations checked the downloaded browser ZIP. Both benchmark curves matched the saved prices and entry fees within `2.22e-16`. All 48 strategy/benchmark pairs, 144 evaluation windows, relative wealth paths, and CAGR differences reconciled. The statistical family contained 96 tests and used 508 paired returns after the entry interval.

An independent share-and-trade replay also matched all 24 CLI portfolio curves within `1.94e-14`. Each recorded fit ended before execution. A repeat CLI run from the saved price files produced the same results without a data download.

### Browser checks for benchmark comparisons

The T3 Code collaborative browser showed benchmark markers and curves across the app. The Market comparison view showed the fixed default strategy, uncertainty details, and chronological evaluation windows.

At 390 × 844, the evidence cards stacked and the relative-performance title fit. The page had no horizontal overflow. The offline report rendered seven charts without external resource requests. Wide report tables remained inside scroll areas. The final app and report showed the revised statistical evidence labels.

These checks establish provider access only during this run. Automated Yahoo tests use mocked responses. The browser checks do not establish physical-phone or native Windows/macOS support. No new PDF was generated in this pass.

## Latest risk profiles and Yahoo default: 4 October 2026

**257 tests passed**, including **20 Streamlit AppTest cases**. Dependency checks passed on Python 3.14 and Streamlit 1.65 on Linux/WSL.

- Yahoo Finance is the initial app source. Tests confirm that downloads start only after **Build frontier**.
- Profile tests cover known solutions, weight limits, short histories, zero variance, data scaling, and nearly tied return estimates.
- An independent analytic reference matched 80 two-asset frontier cases across different caps and shrinkage levels. Maximum allocation error was below `3.2e-7`.
- A separate scale check confirmed the minimum-variance tie-break at the Extreme endpoint. Numerical tolerances are documented in [Backtesting](BACKTESTING.md#latest-model-holdings-and-profiles).
- A half-variance objective resolved a solver precision warning. Tests confirm that future approximate solver results produce explicit profile notes.
- Latest profiles use all supplied history. Changes to the training split do not change that fit. Backtest targets exclude future prices and execution-day returns.
- Profile backtests preserve the original 12 combinations and add 12 profile combinations. Costs and holding contributions reconcile.
- Reports contain profile weights, estimates, windows, and frontier data. Short-history reruns remove obsolete generated profile files and preserve unrelated files.

The suite still reports the pre-existing pandas warning in the invalid-date rejection test. That test passes.

### Browser and live-data checks

The T3 Code collaborative browser loaded live Yahoo prices for the six default example tickers. The request returned 1,697 adjusted-price rows from 2 January 2020 through 2 October 2026.

- The initial screen showed Yahoo Finance and waited for submission. Settings and backtest explanations were available before calculation.
- The Latest holdings view showed three dated profiles. The backtest charts initially showed expanding Low, Medium, and Extreme alongside fixed equal weight.
- The downloaded ZIP contained all 24 backtest holding files. Independent calculations confirmed profile weight totals, window boundaries, and annual arithmetic window means.
- At 390 × 844, the profile cards stacked vertically. Profile and backtest charts had no page-wide horizontal overflow.
- The offline report rendered six charts without external requests. Its chart titles fit the narrow viewport, and wide tables stayed in scroll areas.

This live request confirms provider access during this check only. Automated Yahoo tests still use mocked responses. These browser checks do not establish physical-phone or native Windows/macOS support. No new PDF was generated in this pass.

## General portfolio upgrade: 4 October 2026

**212 tests passed**. The suite contains **16 Streamlit AppTest cases**. Dependency checks passed. These checks used Python 3.14 and Streamlit 1.65 on Linux/WSL.

- Monthly, weekly, calendar-day, and custom annualization reach the estimates, backtests, metrics, interface, and reports. The selected factor does not resample prices.
- International Yahoo symbols retain exchange suffixes. Explicit share-class aliases still work. Provider responses remain mocked in automated tests.
- CSV inputs accept five complete price rows. Both engines still require two returns in each period. Direct API calls reject boolean and complex prices.
- Independent formulas verify Sortino, Calmar, concentration, and signed variance contributions. Tests cover undefined ratios and zero-risk portfolios.
- Default allocations, frontier values, equity curves, and previous metric columns match the original code within `1e-12` on the complete synthetic demo.
- Further comparisons confirmed unchanged backtest allocations, holdings, trades, and time boundaries. Fits exclude the return on the execution date.
- App tests verify applied settings, pending changes, risk selection, and chart filters. Filters preserve all strategy results and exports.
- CLI checks cover a monthly CSV report and an offline demo with all four backtest methods. Export checks cover the `Date` header and risk CSVs.

The suite reports one pre-existing pandas warning during the invalid-date rejection test. That test passes.

### Browser checks for this upgrade

These checks used the T3 Code collaborative browser:

- The app displayed the demo, portfolio risk view, holdout view, and selected backtest curves.
- A synthetic monthly CSV with 36 rows and two assets produced 24 training returns and 11 holdout returns.
- The downloaded monthly ZIP recorded 12 observations per year. Independent checks confirmed its CAGR, variance-share totals, and 12 strategy exports.
- At 390 × 844, the frontier, holdout, and backtest titles fit within their charts. The page had no horizontal overflow.
- The standalone report displayed all five charts, risk tables, and new ratios. It requested no external resources. Its narrow layout contained wide tables in scroll areas.
- The report print button called `window.print()`. This check did not produce a new PDF.

The preview host logged Electron startup errors when it opened the report tab. The charts still rendered. This was not a clean browser-console check.

These viewport checks used a desktop browser. They do not establish physical-phone support or native Windows/macOS coverage.

## Earlier baseline: 3 October 2026

The earlier checks used Python 3.14, Streamlit 1.65, and Chromium on Linux/WSL. The sections below record that baseline.

## Automated checks

At that time, **135 tests passed**, including **10 Streamlit AppTest cases**. Dependency checks also passed.

- Portfolio controls change the calculated weights, training split and selected frontier allocation. Single-asset and 64-asset inputs are covered.
- Backtest controls change trading intervals, estimation windows and fees; the holdings selector shows the corresponding strategy's records.
- Tests cover known optimization solutions, time-ordered estimation, delayed execution, drifting weights, self-financing fees and holdings attribution. Holding profits less fees reconcile to portfolio returns.
- Invalid settings and failed data requests remove previous results and downloads. CSV validation rejects malformed, incomplete or nonpositive data.
- CLI and report tests exercise export and reproduction. Rerunning without backtests now removes obsolete generated backtest files while preserving unrelated files.
- The original-holdings Yahoo preset requests Marsh's current `MRSH` symbol; the historical spreadsheet retains `MMC`.

Run these checks from the repository root:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

[CI](https://github.com/sammcentee/efficientfrontier/actions/workflows/ci.yml) repeats the automated suite and an offline command-line export on pushes to `main` and pull requests.

## Real-browser checks

These checks used Playwright against the running Streamlit app, in addition to AppTest:

- Opened the six-asset demo, checked chart labels and interactive hover, and selected a different strategy in **Backtests & holdings**.
- Uploaded an actual CSV with 500 synthetic observations and four assets. The app showed four assets, 349 training returns and 150 holdout returns.
- Changed the trading interval to 42 sessions and fees to 25 basis points. The downloaded ZIP recorded those settings and contained 12 strategy exports with reconciled holding contributions.
- Downloaded the prices CSV and verified it matched the upload. An invalid replacement CSV displayed an error and cleared the prior results and report download.
- Opened the downloaded HTML with the browser offline: all five charts rendered, frontier hover worked, and no external requests were made. The print button invoked printing; Chromium generated an 11-page PDF with searchable findings, chart labels and results.
- At 390 × 844 with touch emulation, opened and closed the sidebar, enabled the position-limit input and selected **Original holdout**. The page rendered without page-wide horizontal overflow. Long chart titles can clip at this width; this was not a full workflow on a physical phone.

No uncaught browser JavaScript errors occurred in the desktop app and offline-report flows. Browser checks are a local verification, not currently part of GitHub Actions.

## Checking the published findings

The [stock study](HOLDINGS_STUDY.md) figures were independently recomputed from the saved equity curves, trades and holdings for all 12 combinations. Returns, drawdowns, turnover, fees and contributions reconcile with the published summaries; recorded estimation dates precede their trades.

That verifies the calculations against those inputs. It does not remove the study's fixed-universe selection hindsight or establish future performance.

## Limits of this check

Yahoo responses are **mocked** in automated tests; these checks do not establish live Yahoo availability or permission to use its data. The current browser checks cover Chromium on Linux/WSL, not a full native Windows or macOS session. They also do not test every possible input, browser or portfolio size.
