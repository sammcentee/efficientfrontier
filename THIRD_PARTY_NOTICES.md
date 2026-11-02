# Third-party software and data

Original project code and documentation are licensed under the [MIT License](LICENSE). This file records third-party notices; third-party software and data retain their own terms and are not relicensed by the project's license.

## Rust dependencies

Cargo downloads these direct dependencies of the library, the CLI, and the desktop app from crates.io when it builds them. `Cargo.lock` pins every version. Except for the GLib backport below, their source code is not vendored in this repository. Licenses and notices remain available in each crate and its upstream project.

| Dependency | Upstream license | Project |
| --- | --- | --- |
| Clarabel | Apache-2.0 | [oxfordcontrol/Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) |
| nalgebra | Apache-2.0 | [dimforge/nalgebra](https://github.com/dimforge/nalgebra) |
| Tauri (`tauri`, `tauri-build`) | Apache-2.0 OR MIT | [tauri-apps/tauri](https://github.com/tauri-apps/tauri) |
| Tauri dialog and opener plugins | Apache-2.0 OR MIT | [tauri-apps/plugins-workspace](https://github.com/tauri-apps/plugins-workspace) |
| axum | MIT | [tokio-rs/axum](https://github.com/tokio-rs/axum) |
| Tokio | MIT | [tokio-rs/tokio](https://github.com/tokio-rs/tokio) |
| tower | MIT | [tower-rs/tower](https://github.com/tower-rs/tower) |
| ureq | MIT OR Apache-2.0 | [algesten/ureq](https://github.com/algesten/ureq) |
| chrono and chrono-tz | MIT OR Apache-2.0 | [chronotope/chrono](https://github.com/chronotope/chrono), [chronotope/chrono-tz](https://github.com/chronotope/chrono-tz) |
| clap | MIT OR Apache-2.0 | [clap-rs/clap](https://github.com/clap-rs/clap) |
| serde and serde_json | MIT OR Apache-2.0 | [serde-rs/serde](https://github.com/serde-rs/serde), [serde-rs/json](https://github.com/serde-rs/json) |
| indexmap | Apache-2.0 OR MIT | [indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) |
| zip | MIT | [zip-rs/zip2](https://github.com/zip-rs/zip2) |
| roxmltree | MIT OR Apache-2.0 | [RazrFalcon/roxmltree](https://github.com/RazrFalcon/roxmltree) |
| sha2 | MIT OR Apache-2.0 | [RustCrypto/hashes](https://github.com/RustCrypto/hashes) |
| libm | MIT | [rust-lang/compiler-builtins](https://github.com/rust-lang/compiler-builtins) |
| tempfile (tests only) | MIT OR Apache-2.0 | [Stebalien/tempfile](https://github.com/Stebalien/tempfile) |

This list covers direct dependencies, not the complete set of transitive crates. Run `cargo tree` for the full set. Preserve the crates' own notices when you redistribute a compiled binary.

### Vendored GLib security backport

The Linux desktop dependency graph uses the complete MIT-licensed `glib` 0.18.5 crate in [vendor/glib](vendor/glib). Its upstream [LICENSE](vendor/glib/LICENSE) and [COPYRIGHT](vendor/glib/COPYRIGHT) remain unchanged. The only source changes are the two-line fix from [gtk-rs/gtk-rs-core PR #1343](https://github.com/gtk-rs/gtk-rs-core/pull/1343/files) for [RUSTSEC-2024-0429](https://rustsec.org/advisories/RUSTSEC-2024-0429.html) / GHSA-wrw7-89jp-8q8g. The [backport note](vendor/glib/BACKPORT.md) records the archive checksum, patch, and regression command.

## System web views

The desktop app shows its pages in the web view of the operating system. On Windows, this is the Microsoft Edge WebView2 Runtime, a part of Windows 11 under Microsoft's terms. The `.exe` does not contain it. On Linux, the program uses the WebKitGTK and GTK libraries of the system and does not contain them.

## Plotly.js in the app and exported reports

The app and the standalone HTML reports use Plotly.js 4.1.1, so their charts work offline. The bundle in `assets/plotly/plotly.min.js` is the same file that Plotly Python 7.1.0 supplied to the earlier Python release. Its upstream MIT license is preserved verbatim in [assets/plotly/LICENSE.txt](assets/plotly/LICENSE.txt), sourced from the [versioned upstream license](https://github.com/plotly/plotly.js/blob/v4.1.1/LICENSE). The HTML and report ZIP also include this notice.

The JavaScript bundle also contains third-party components and retains their upstream license comments. The notice above covers Plotly itself; it is not a complete inventory of every bundled component. Before you redistribute an exported report, review the bundle's additional notices and the rights to its input data.

## KaTeX in the app

The app typesets its equations with KaTeX 0.19.1 from the npm package `katex`. The files in `assets/katex/` (the JavaScript, the style sheet, the auto-render extension and the WOFF2 fonts) are unchanged copies from that package. KaTeX is MIT licensed. Its license is preserved verbatim in [assets/katex/LICENSE.txt](assets/katex/LICENSE.txt). Exported reports do not include KaTeX.

## Bundled membership data

`assets/membership/` contains the pinned community reconstruction of Nasdaq-100 membership that the historical-membership study uses. Its license and provenance are in `assets/membership/LICENSE.n100tickers` and `assets/membership/provenance.json`. Reports of that study include the license.

## Market data and original holdings

Current stock-universe sources are the [Nasdaq-100 list](https://api.nasdaq.com/api/quote/list-type/nasdaq100) and [State Street's daily SPY holdings](https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx). SPY equity holdings serve as an S&P 500 proxy. They are not a certified historical index roster. The repository does not bundle downloaded source files. Reports retain source URLs, dates, and exclusions.

Downloaded prices are not included in the repository. The software's license does not grant redistribution rights to market data. An exported report includes its input prices; review the provider's terms before sharing it.

The Yahoo Finance integration is unofficial. The application calls the same public Yahoo endpoints that the [yfinance project](https://github.com/ranaroussi/yfinance) uses; yfinance describes that access as intended for personal use. [Yahoo's own terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) restrict automated collection without permission, and its [redistribution guidance](https://help.yahoo.com/kb/SLN2352.html) restricts republishing supplied data. This repository does not grant permission to access or redistribute Yahoo data. Publishing this client source does not establish permission for a hosted data service or commercial data use.

`spy_holdings.ods` is the original user-supplied list of 60 ticker symbols. On 2026-10-03 the repository owner confirmed that both the original R script and spreadsheet are theirs or permitted to publish. The spreadsheet contains no market prices or account information, and is not represented as an official or historical S&P 500 dataset. Screenshots in this repository use only the application's synthetic demonstration data.
