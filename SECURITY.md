# Security

This is a local portfolio research application under active development. Security fixes target the current `main` branch; there are no supported versioned releases yet.

## Report a vulnerability

Use GitHub's [private vulnerability reporting form](https://github.com/sammcentee/efficientfrontier/security/advisories/new) when available. Include the affected commit, reproduction steps using synthetic data, expected impact, and any proposed fix. Do not attach real portfolio data, credentials or account statements.

If the private form is unavailable, open an issue asking for a private reporting channel without disclosing exploit details or sensitive data.

## Local use and data handling

The default server binds to `127.0.0.1`, with Streamlit usage statistics disabled. It is not configured as a public service: do not expose a shared instance or upload other people's private data without adding appropriate access controls and reviewing session/cache behavior.

The default synthetic demo and CSV analysis run without a market-data network request. Choosing Yahoo Finance sends the requested symbols and dates through yfinance to Yahoo and requires the user's authorized access. The repository contains no API credentials or broker/trading integration.

Reports contain their input prices, allocations and settings. Keep private reports out of Git and review contents and provider terms before sharing. Formula-like text labels in CSV exports are escaped; numeric return and price values are preserved. HTML output escapes supplied metadata and table text.

See [the publication review](docs/PUBLIC_RELEASE_REVIEW.md) for the audit scope and limitations. Dependency and secret scans identify known patterns and advisories; they do not establish the absence of all vulnerabilities.
