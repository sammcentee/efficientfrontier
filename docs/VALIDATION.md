# Validation record

## Mathematical opening page and Nasdaq risk default: 18 October 2026

**876 tests passed** with Python 3.14 and the existing pandas invalid-date warning. The opening page displays three typeset model equations before any interaction. Settings, compact risk inputs, and the rainbow Build action replace the previous welcome controls.

The default budget uses QQQ adjusted daily returns over its latest complete trailing year. A live calculation returned **19.835042%** from 251 returns, ending 16 October 2026. Tests verify the sample standard deviation, annualization, date boundaries, incomplete-day exclusion, failed downloads, and preservation of saved or edited budgets. The interface and report retain two decimal places.

Headless Chromium verified the whole opening workflow at 1366 × 768 and 1440 × 900 without scrolling; Build ended at 700 pixels or less. At 390 pixels wide, the page stacks and scrolls without horizontal overflow. The native T3 browser sandbox was unavailable. The app server on port 8518 was restarted to load the changed modules and styles. Physical devices were not tested.

The live app check preserved an edited 21.25% budget through Settings, the demo calculation, all 14 result equations, and the exported report. Exported replay paths each retained one trade and zero rebalances. No browser errors occurred.

## Buy-and-hold replay and rendered mathematics: 18 October 2026

**859 tests passed** with Python 3.14 and one existing pandas invalid-date warning. After the browser check found joined LaTeX commands, the formatting correction passed all 40 app tests. All 14 displayed equations also passed the bundled KaTeX parser with strict errors enabled. Dependency and whitespace checks passed, and the stock-model CLI produced its offline report.

The stock model and equal-weight reference now make one initial purchase and hold. Their weights drift with prices. Every replay path records one trade, zero rebalances, one entry fee, and no terminal sale. The initial fit and latest full-history fit remain separate. The model revision hides old rebalanced results until the user rebuilds. The CLI rejects a rebalance interval for the stock model and retains it for legacy backtests.

An independent calculation checked 24 paths across fee rates of 0, 10, and 9,999 basis points and annualization factors of 12 and 252. Values matched the fixed-unit buy-and-hold formula within `4.44e-16` initial-capital units. Independently calculated arithmetic mean, sample volatility, CAGR, and drawdown matched within `1e-12`. Regression tests also cover initial target constraints, drift beyond the holding cap, prior-price information, benchmark parity, and exact export reproduction.

Headless Chromium verified both collapsed mathematical explanations, all 14 rendered equations, data navigation, and the exported report at 1440- and 390-pixel widths. No raw LaTeX commands, parser errors, clipped formulas, document overflow, or page errors remained. Exported demo paths each contained one trade. The offline report retained both charts. T3's native browser sandbox was unavailable, so checks used a separate background browser. The user's server remained running. The screenshot's market study was not rebuilt, and physical devices were not tested.

## Latest fit and historical replay distinction: 18 October 2026

**857 tests passed** with Python 3.14. The existing pandas invalid-date warning remains. The app and report now compare the latest full-history allocation with the replay of earlier allocations before showing the frontier. Labels state that the latest fit includes the replay dates. A replay callout identifies mean/volatility dominance independently of the existing CAGR-based performance checks.

Regression checks cover the comparison dates and metrics, dominance and return/risk tradeoffs, unchanged numeric exports, and passive app navigation. No optimization, allocation, or return-accounting code changed.

Headless Chromium checks covered the offline demo, exported report, and a synthetic CSV with a dominating QQQ reference at 1440- and 390-pixel widths. Both charts rendered, comparison columns stacked on mobile, and the verdict appeared above the replay table. There were no app exceptions, page errors, or document overflow. The T3 preview sandbox was unavailable; checks ran in a separate background browser. The user's running study was not restarted. The screenshot's market study was not rebuilt, and physical-device behavior was not tested.

## R comparison and full project review: 18 October 2026

**850 tests passed** with Python 3.14 and the installed pinned dependencies. The existing pandas warning from the invalid-date rejection test remains. Dependency and whitespace checks passed. Both the stock-model and legacy offline CLI reports completed.

The review covered the original R script, optimization, price and membership inputs, trade accounting, benchmarks, app state, CLI, reports, documentation, launchers, dependencies, and CI. The main model now uses sample covariance and separates SPY/QQQ reference columns from stock candidates. Its latest frontier and stock weights appear before the later test.

Confirmed defects corrected in this review:

- Equal-mean choices could select unnecessarily high risk. The stock model now minimizes variance when several portfolios attain the capped maximum mean, within its stated numerical tolerance.
- Singular covariance could give a dominated minimum-risk endpoint. Frontier and profile calculations now prefer higher return at the same minimum risk.
- Very small return or variance scales could cause solver failure or collapse frontier targets. Equivalent scaled constraints and removal of a redundant risk bound correct these cases.
- Saved-setting migration could replace a CSV/demo source with a live universe. It now preserves the source, files, and holding limit.
- Malformed membership CSV quoting could alter symbols silently. The parser now rejects malformed and incomplete rows. Future-dated Nasdaq source snapshots are also rejected.
- Report construction could mutate a source index name. Export cleanup could leave files from an earlier model. Both are corrected, along with benchmark risk labels and escaped chart text.

An independent reproduction used the original 60 stocks and 2020–2023 fit dates. Corrected arithmetic means still produce stock-only portfolios above SPY and QQQ at their respective sample volatilities. Four comparisons, including a 10% holding cap, pass feasibility and numerical global-bound checks. The app matches their annual means within 0.0001 percentage points. See the [R comparison](R_METHOD.md) for inputs, weights, the original formula error, and interpretation.

Independent analytic checks covered 42 two-stock frontier points at different holding caps and covariance scales. Maximum weight error was `1.54e-9`. Two further fixtures verify the equal-mean correction. An independent reconstruction checked 19 strategy paths for portfolio values, fees, prior-close signals, and benchmark parity. Changing future prices left earlier selections unchanged.

The original R file remains unchanged. The reproduction uses Python because `Rscript` was unavailable. Downloaded prices can differ from the 2024 vendor snapshot. These checks establish calculation agreement, not unseen index outperformance.

Headless Chromium checks covered the demo, data view, original 60-stock CSV, ETF exclusion, frontier references, and ZIP exports. Exported fit statistics reconciled with an independent calculation. Desktop and mobile widths of 1440 and 390 pixels had no horizontal overflow or JavaScript/app errors. The final chart-label and table-spacing corrections passed 55 focused app/report tests and a fresh-process browser check. The T3 preview sandbox was unavailable, so the browser ran separately in the background.

The Linux launcher passed a loopback health check, startup from a path with spaces, and clean interruption. Shell syntax and the three launcher regression tests passed. All 64 installed distributions returned no listed vulnerability or yanked-release warning from their release-specific PyPI records at review time. This was a feed check, not a complete dependency security audit. Native Windows/macOS execution, physical devices, a fresh installation, and a new Git-history secret scan were not performed.

## Fitted frontier and benchmark clarity: 18 October 2026

**741 tests passed**. Dependency and whitespace checks passed. The suite retains the existing pandas warning from the invalid-date rejection test.

The app and report now show the first and latest fitted estimates separately from the later historical test. Each fit has two risk measures: the adjusted covariance estimate that the budget limits, and ordinary historical volatility on the same dates and weights. `fit_estimates.csv` records all historical fits and the latest fit. These diagnostics use saved weights without another optimization. Tests verify covariance formulas, past-only samples, exact exports, and unchanged study state during navigation.

The saved 500-stock input reproduced the reported 23% budget and 30% holding-cap result exactly. The later test returned a 31.31% annual arithmetic mean with 24.20% realized volatility. QQQ returned 33.47% with 18.52% volatility. QQQ beat the rule on both measures during this test.

| Fit through | Fitted annual mean | Model volatility | Ordinary historical volatility |
| --- | ---: | ---: | ---: |
| 16 April 2025 | 55.1391% | 23.0000% | 23.5189% |
| 16 October 2026 | 54.8568% | 23.0000% | 23.7018% |

An independent calculation audited all 19 actual saved allocations: 18 historical trades and the latest weights. Every allocation met the holding cap. Model volatility exceeded the ceiling by at most `3.29e-12` from numerical error. A global objective bound placed every saved portfolio within **0.000129602 percentage points** of the maximum possible annual mean. This bound used a supporting hyperplane for the covariance quadratic form and an exact linear maximum over the capped simplex. It did not sample random portfolios. Independent SCS solves for the first and latest fit also agreed within 0.00000020 percentage points of annual mean.

This verifies optimization for the supplied estimates and constraints. It does not verify future returns. The later test uses changing earlier allocations. The latest portfolio did not generate that test path. The 10% covariance adjustment explains the model-versus-historical risk difference in the fitting table; it does not by itself explain the later realized risk.

Headless Chromium checks covered the updated app at 1440 and 390 pixels, plus the actual 23% / 30% report at 390 pixels. The pages had no horizontal overflow or JavaScript errors. Wide tables scroll within their containers. The offline report made no external requests. The T3 preview sandbox was unavailable, so a separate background browser performed these checks. The active app server stayed running. Native operating-system and physical-device behavior remain untested.

## Single model and later test: 18 October 2026

**734 tests passed** with Python 3.14 and the pinned dependencies. `pip check` and `git diff --check` passed.
The suite retains the existing pandas warning from the invalid-date rejection test.

The app now runs one historical-mean objective under a volatility ceiling and holding cap. One later test compares it with equal-weight stocks, SPY, and QQQ. Blend selection, a separate validation period, synthetic ETF/cash comparisons, and alternative app model menus are removed. Earlier CLI/API tools retain their own tests.

Tests cover an analytic constrained optimum, prior-close fits, future-price isolation, costs, stale quotes, infeasible limits, missing benchmarks, and exact exports. App checks cover fixed assumptions, retained inputs, explicit migration from old studies, and passive data views. Independent review reconstructed portfolio values, fees, and profit contributions.

A fresh combined-universe download supplied 500 complete stocks and both ETFs on 1,255 market dates, from 18 October 2021 through 16 October 2026. The source contained 518 unique stocks and recorded 18 incomplete-history exclusions. The model used a fixed 20% budget and 10% holding cap. All 18 historical allocations met both fitted limits within numerical tolerance. Independent calculations reconciled each path's profit contributions and fees, and the latest historical mean estimate.

The later test covered 17 April 2025 through 16 October 2026. The model's net CAGR was 31.25%, with 20.11% realized volatility. QQQ returned 37.36% with 18.52% volatility. The descriptive checks therefore failed. No parameter changed in response to the result.

The past-price screen excluded FER. Current membership, complete-history selection, and prior research on these dates limit interpretation.

The full-data CLI produced four comparison paths and one latest portfolio. The report contains one chart and no blend policy or validation exports. Reproduction tests compare exact numeric files.

Headless Chromium checks covered the synthetic app at 1440 pixels and 390 pixels, plus the full-data report at 390 pixels. Pages had no horizontal overflow or JavaScript errors. The report made no external requests. The T3 preview sandbox was unavailable, so these checks used a separate background browser. Native operating-system and physical-device behavior remain untested.

## Risk budget and combined stock universe: 16 October 2026

**726 tests passed** with Python 3.14 and the pinned dependencies. `pip check` and `git diff --check` passed.
The suite reports the existing pandas warning from the invalid-date rejection test.

The primary app replaces the three risk selectors with a volatility budget before the calculation. Tests cover explicit holding limits, infeasible settings, retained inputs, old-study migration, offline behavior, unsupported holdings, exports, and separate legacy diagnostics.
A model revision blocks stale results and exports until an explicit rebuild. A regression verifies that unchanged inputs still force a fresh calculation after a revision.
The selection tests verify constrained optima, feasible blends, dated price-quality decisions, costs, cash, both ETF references, the one-year evidence floor, and both realized-risk checks. They also verify raw ETF dominance rejection and valid return-versus-risk tradeoffs. Future prices cannot change earlier results or the frozen blend.

An independent share, cash, and fee reconstruction matched all 16 phase/strategy equity paths within `4.9e-15` initial-capital units on a saved 92-stock study. All 32 scheduled fits respected the position cap and fitted volatility ceiling within numerical tolerance.
This establishes accounting agreement, not market-data accuracy, statistical significance, or future outperformance.

The live source check combined 101 Nasdaq-100 securities and 503 SPY stocks into 518 unique stocks. Both provider dates and the independently published SPY holding count reconciled. The parser explicitly excluded cash and a contingent-value right. No downloaded source file entered the repository.
Five years of Yahoo prices supplied 500 complete stocks on 1,254 common market dates. A fixed 20% volatility budget, 10% holding limit, and 21-session schedule completed in about 61 seconds, excluding downloads. The long unchanged-price screen excluded FER from its fits.
One fit reported limited numerical accuracy. Its weight and fitted-risk feasibility checks passed, and the warning remains in the results.
The selected rule exceeded the realized-risk budget and correctly remained unsupported. QQQ also had higher net CAGR with lower realized volatility in the later test. No parameter changed in response to that outcome.

The risk-target CLI completed offline. The saved-price report archive retained 62 verified files, including both periods, all allocations, holdings, trades, source records, and policy decisions. Reproduction tests matched the numeric exports.
The background T3 browser displayed the new inputs and completed the synthetic app flow without an app exception or desktop page overflow.
Native viewport resizing disconnected the preview host. A 390-pixel iframe then verified the welcome controls and saved-price report layout, with wide result tables inside scroll areas. This is a desktop-browser layout check, not a physical-device test.

## Overview decisions and replay audit: 16 October 2026

**671 tests passed** with Python 3.14 and the pinned dependencies. `pip check` and `git diff --check` passed.
The suite reports the existing pandas warning from the invalid-date rejection test.

The overview and exported HTML now compare tested rules before proposed holdings. Tests check matched-date arithmetic means, CAGR, volatility, drawdown, explicit missing-profile results, and a fixed row order without automatic winner selection.
A regime-change fixture verifies that Medium and Highest can trail both ETFs despite higher fitted means. Display selections preserve all numeric exports, including every allocation, holding, and trade file.
Contribution tests verify visible gains and losses and reconciliation after fees. Unchanged-price diagnostics preserve the supplied prices and universe.

An independent share-and-fee replay of a saved 92-asset study matched both selected equity paths within `1.4e-15` initial-capital units.
Changing future prices left all earlier allocations unchanged. These checks establish calculation agreement, not future outperformance or freedom from current-membership selection bias.
Targeted price checks found no obvious split discontinuity explaining the selected losses. A separate long flat price history remains flagged for review; provider accuracy is not certified.

The open local study displayed the new comparison, selected Medium profile, replay contributions, and price warning without an app exception or keyboard-focus change.
The desktop page had no horizontal overflow at 1280 pixels. The saved-price HTML report kept its comparison table within a scroll area at 390 pixels.
The offline CLI completed all 24 strategies. Its 92 archive files matched the extracted outputs. Native operating-system and physical-device behavior remain outside these checks.

## Historical mean as the primary model: 15 October 2026

**651 tests passed** with Python 3.14 and the pinned dependencies. `pip check` and `git diff --check` passed.
The suite reports the existing pandas warning from the invalid-date rejection test.

The primary portfolios now use full-fit annual arithmetic mean returns and covariance. Their backtests use the same selection rule and 21-point grid, with only prior prices at each trade.
Tests deliberately separate a high-mean asset from an asset with stronger weak-window returns. They verify the selected weights, charts, ETF markers, CLI output, reports, and objective metadata.
The optional weakest-window comparison leaves the primary weights, backtests, and statistical evidence unchanged.

An independent reconstruction matched all 39 scheduled profile allocations in a synthetic fixture from each recorded fit interval.
A separate share-and-fee replay of a 92-asset saved-price study matched the highest-mean expanding-window equity path within `8.4e-10` initial-capital units.
These checks establish calculation agreement, not investment performance.

One valid rolling fit reached the solver's iteration limit. A regression now checks its weights against an independent constrained-variance solution.
An equivalent half-variance objective resolved that case without changes to the solver tolerances.

App tests cover one-time updates of open studies from retained prices, failure recovery, and preservation of inputs and view selections.
The open local study refreshed without a data download or a keyboard-focus change. The browser displayed the historical-mean overview without an app exception or page overflow at 1280 pixels.
The offline CLI completed all 24 strategies and the report archive. Fitted arithmetic means, realized CAGR, and mean-return uncertainty have separate labels.
Native operating-system and physical-device behavior remain outside these checks.

## Historical membership: 12 October 2026

**620 tests passed** with Python 3.14 and the pinned dependencies. `pip check` and `git diff --check` passed.
The suite reports the existing pandas warning from the invalid-date rejection test.

New tests cover effective-date entry, departure, reentry, short price histories, source coverage bounds, and known reused ticker identities.
Price or roster changes in the future cannot change earlier targets. A missing execution or held-asset price stops the study.
Cash transitions, fees, and holding contributions reconcile. A constant complete roster matches the existing rolling-window engine.
An independent share-accounting check also matched NAV through repeated cash exits and reentries.

CLI and Streamlit tests cover offline inputs, custom coverage dates, pending settings, error recovery, and preservation of the current study.
The synthetic CLI export completed. Reloaded inputs reproduced equity, cash, coverage, and metrics.
Report tests check HTML escaping, embedded chart resources, input snapshots, numeric exports, and source licenses.
The bundled history records 119 snapshots, known omissions, explicit coverage through 30 September 2026, and identity exclusions.

The background T3 preview ran the historical demo and displayed all three chart traces.
The historical page had no page overflow at 1440 × 900 and 390 × 844, and no Streamlit exception appeared.
The preview emitted Electron startup errors and disconnected during follow-up navigation, so the remaining navigation checks use AppTest.
No live historical market prices were downloaded. These checks do not establish data completeness, investment performance, or physical-device support.

## Classic frontier risk choices: 10 October 2026

**504 tests passed** with Python 3.14.7 and the pinned dependencies. `pip check` and `git diff --check` passed.
The suite reports the existing pandas warning from the invalid-date rejection test.

Tests verify actual Low, Medium, and High frontier weights, holding limits, and the nearest sampled midpoint of volatility.
The same initial weights produce each classic holdout path. A fixed regime-change example has higher fitted returns but lower later returns than both ETFs.
App tests verify that selecting a classic level changes its point, weights, and holdout results without solver calls or downloads.
The overview risk selection stays separate. ETF estimates and holdouts retain matched dates.

Flat-risk tests cover singular and zero covariance, equal and distinct means, and changes in covariance scale.
Explicit single-point analysis retains its original portfolios without false High or flat-risk labels.
Report tests verify the definitions, dates, and all classic weights and results in the existing CSV exports.
The synthetic monthly CLI fixture also completed its offline report and archive.

The background T3 browser checked all three classic choices at 1440 × 900 and 390 × 844.
The selected point and holdout curve agreed, with no page overflow or app exceptions.
No live Yahoo or Nasdaq request was made. These checks do not establish physical-phone or native Windows/macOS support.

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
