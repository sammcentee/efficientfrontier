# Repository review — 9–10 October 2026

This review starts from `claude/redesign` at `0a42423`, after the Nasdaq-100 work in PR #2 and the interface work in PR #3.
The default branch was `main` at `cdba36f`. These earlier changes remain separate draft pull requests.

## Assessment

The calculation engine has clear modules, chronological evaluation, explicit trade costs, and useful tests with known mathematical answers.
The latest interface already gives the user a focused overview. This review retains that layout.

The main weaknesses were unnecessary work, inconsistent input checks, and unverified claims in the labels.
The README also mixed setup, a user guide, and a long historical stock study. A new user needed too much context before the first launch.

## Changes from this review

| Finding | Change | Evidence |
| --- | --- | --- |
| Object columns silently converted boolean prices and discarded imaginary values. | Apply the same numeric checks to asset and benchmark prices. | Four regression cases failed before the fix. Numeric strings still work. |
| One incomplete Nasdaq member forced a retry for every member in its batch. | Validate each downloaded column independently. Retry only a failed batch. | A 100-stock case fell from 105 requests to five, with the same 96 eligible assets and four exclusions. |
| Backtests calculated a frontier endpoint and discarded it at every fit. | Request one minimum-volatility frontier point for backtest fits. Keep the normal chart frontier. | Solver calls fell from 91 to 78. All 24 strategy results stayed identical. |
| Successful app builds calculated market evidence twice. | Calculate it once. Retain the original holdout comparison if the backtest fails. | App tests check call counts and the failure path. |
| Approximate classic optimizer results had no project warning. | Report limited numerical accuracy once per optimization. | Tests cover minimum volatility, maximum Sharpe, and frontier solves. |
| Stand-alone benchmark comparisons could use a different risk-free rate from the asset analysis. | Retain the analysis rate and reject a mismatch. | A single-asset comparison confirms equal metrics under a matching nondefault rate. |
| Yahoo currency text claimed verified USD prices. Monthly CSV text described daily returns. A concentration note inferred profit dominance from capital weight. | State the currency assumption. Use observation counts. Describe allocation concentration directly. | App and report helpers share the corrected text. Regression cases cover the affected claims. |
| The README delayed setup and repeated reference material. | Put launch steps near the top. Keep short findings and link the detailed guides. | Local document links and anchors receive a separate check. |

The solver comparison used 64 synthetic assets, 360 price rows, and a fixed seed. It checks result equivalence, not investment performance.
Solver calls fell from 91 to 78, or 14.3%, for that case. Runtime savings depend on the data and machine.

## Limits and next priorities

- **Historical membership:** the Nasdaq study uses current members with complete history. It does not reconstruct past membership or include every delisted security.
- **Model risk:** three historical windows do not establish a future return floor. The risk labels compare portfolios inside this model.
- **Maintenance:** the app still needs substantial custom CSS and Streamlit state code. Keep future changes small and cover state transitions with tests.
- **External data:** provider access, quote currencies, and adjusted-price quality remain external assumptions. Offline CSV files give the most reproducible inputs.
- **Platforms:** browser checks cannot establish native Windows, macOS, or physical iPhone behavior.

See [validation records](VALIDATION.md) for the checks and their limits. The [stock study](HOLDINGS_STUDY.md) retains its original figures and caveats.
