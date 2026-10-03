# What has actually been tested

Checked on **3 October 2026 (UTC)** using Python 3.14, Streamlit 1.65 and Chromium on Linux/WSL. The README screenshot comes from the running app with synthetic data.

## Automated checks

**135 tests passed**, including **10 Streamlit AppTest cases**. Dependency checks also passed.

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
