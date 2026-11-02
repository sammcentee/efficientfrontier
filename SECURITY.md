# Security

This is a local portfolio research application under active development. Security fixes target the current `main` branch; there are no supported versioned releases yet.

## Report a vulnerability

Use GitHub's [private vulnerability reporting form](https://github.com/sammcentee/efficientfrontier/security/advisories/new) when available. Include the affected commit, reproduction steps using synthetic data, expected impact, and any proposed fix. Do not attach real portfolio data, credentials or account statements.

If the private form is unavailable, open an issue asking for a private reporting channel without disclosing exploit details or sensitive data.

## Local use and data handling

The desktop app opens no network port. Its window gets each page from the program through a custom protocol, and the program refuses posts from pages of other origins. Links to other sites open in the default browser, not in the app's window. The app collects no usage statistics. It keeps the study in memory until you close the window. It is not a public service: do not make a shared instance of it or upload other people's private data without access controls and a review of session/cache behavior.

The synthetic demo runs without a market-data network request. CSV analysis stays offline unless the user enables the benchmark download. A study from Yahoo Finance sends the requested symbols and dates directly to Yahoo. The user must have authorization for this access. The repository contains no API credentials or broker/trading integration.

Reports contain their input prices, allocations and settings. Keep private reports out of Git and review contents and provider terms before sharing. Formula-like text labels in CSV exports are escaped; numeric return and price values are preserved. HTML output escapes supplied metadata and table text.

See [the publication review](docs/PUBLIC_RELEASE_REVIEW.md) for the audit scope and limitations. Dependency and secret scans identify known patterns and advisories; they do not establish the absence of all vulnerabilities.
