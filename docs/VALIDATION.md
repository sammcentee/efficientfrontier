# Validation record

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
