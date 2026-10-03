# Public-source review · 2026-10-03

Scope: publish the source of `sammcentee/efficientfrontier` on GitHub. This review does not approve a hosted application, commercial market-data service, or redistribution of generated market reports. The reviewed baseline was `1859595`, with four reachable commits; the publication-preparation changes were reviewed and scanned separately before changing visibility.

## Ownership and licensing

- The owner explicitly authorized public publication and confirmed that the original R script and `spy_holdings.ods` are theirs or permitted to publish.
- Project licensing was undecided at the original publication review. The owner subsequently selected the [MIT License](../LICENSE) for original project code and documentation. Third-party software and data retain their own terms; this choice does not expand data-access or redistribution rights.
- Installed metadata for the eight direct runtime/development dependencies agrees with [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Installed dependency code is not vendored in this repository.
- The preserved Plotly.js 4.1.1 license matches the versioned upstream license byte-for-byte. The complete Plotly notice is included in generated HTML and ZIP output. This does not constitute a complete license inventory of every component inside the generated upstream JavaScript bundle; generated bundles and reports are excluded from source publication.

## Secrets, private information and GitHub contents

- Gitleaks 8.30.1, with its downloaded release checksum verified, found no leaks in the four-commit baseline history. A separate scan covered all 41 unique historical blobs across 27 paths and eight decompressed spreadsheet members, with no findings. Publication changes were also scanned.
- The spreadsheet contains a header and 60 ticker symbols, with no prices, balances, account information, macros or external links. Its metadata contains software/version and editing timestamps, not personal contacts.
- The README image uses synthetic demo data and has no PNG metadata. Real downloaded prices, local reports, credentials, caches and environments are ignored and absent from the reviewed Git history.
- Ordinary Git author/committer names and email addresses remain in history and are visible with a public repository. No history rewrite was performed.
- At review time, GitHub had only the `main` branch, no tags, releases, release artifacts, issues, wiki, discussions or Pages deployment. Previous Actions runs used mocked data and the synthetic offline demo.
- CI uses SHA-pinned actions and read-only repository permissions. It does not run a privileged `pull_request_target` workflow.

## Dependencies and input/output handling

`pip-audit 2.10.1` queried the PyPI advisory service with `--strict` against `requirements-dev.txt`, which includes runtime requirements. It audited **61 resolved packages**, skipped none, and found **zero known vulnerabilities**. This was an isolated Python 3.14/Linux resolution; unpinned transitive dependencies may change, and package advisories do not cover unknown flaws or all native operating-system libraries. Tool/feed references: [pip-audit](https://pypi.org/project/pip-audit/), [PyPI JSON API](https://docs.pypi.org/api/json/), [PyPA advisory database](https://github.com/pypa/advisory-database).

The review reproduced spreadsheet-formula interpretation of uploaded asset names in CSV exports. The publication patch escapes formula-like row/column labels consistently in direct downloads and report ZIPs while preserving numeric values and ordinary symbols. Regression tests cover the behavior. HTML metadata, warnings, and table content are escaped.

## APIs and operating scope

Yahoo Finance through yfinance is the only market-data integration. It is selected explicitly; the default synthetic demo and CSV path do not contact Yahoo. The code contains no API key, broker connection, or order-execution capability.

The integration is unofficial. The [yfinance project](https://github.com/ranaroussi/yfinance) describes personal-use/research limits, but that description does not itself grant access rights. [Yahoo's terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection without permission, and [Yahoo's data guidance](https://help.yahoo.com/kb/SLN2352.html) restricts redistribution. Users must establish authorization for their access and intended use. No downloaded price dataset or market-data report is included in the public source.

The app binds to localhost, and Streamlit usage statistics are disabled. Source publication does not expose the running local application. Public hosting, authentication, shared caches and private uploads require a separate review.

## Review conclusion

No unresolved blocker was found for the authorized source publication within this scope. This is a record of checks and findings, not a guarantee that every vulnerability or legal issue has been excluded. Preserve the notices, keep real data out of public commits, and repeat relevant checks when dependencies, APIs or distributed artifacts change.
