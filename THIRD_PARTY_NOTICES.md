# Third-party software and data

This file records dependency notices. It does not select or grant a license for Efficient Frontier's own code; project licensing remains undecided.

## Python dependencies

The application installs these direct dependencies using `requirements.txt`; their source code is not vendored in this repository. Licenses and bundled notices remain available in each installed distribution and its upstream project.

| Dependency | Upstream license | Project |
| --- | --- | --- |
| NumPy | BSD-3-Clause; distributions include additional bundled notices | [numpy/numpy](https://github.com/numpy/numpy) |
| pandas | BSD-3-Clause | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| CVXPY | Apache-2.0 | [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy) |
| Clarabel | Apache-2.0 | [oxfordcontrol/Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) |
| Plotly Python | MIT | [plotly/plotly.py](https://github.com/plotly/plotly.py) |
| Streamlit | Apache-2.0 | [streamlit/streamlit](https://github.com/streamlit/streamlit) |
| yfinance | Apache-2.0 | [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) |
| pytest (development only) | MIT | [pytest-dev/pytest](https://github.com/pytest-dev/pytest) |

This list covers direct dependencies, not a complete inventory of transitive or platform-specific bundled components. Preserve the distributions' own notices when redistributing an environment or installer.

## Plotly.js in exported reports

Standalone HTML reports embed Plotly.js so their charts work offline. Plotly Python 7.1.0 supplies Plotly.js 4.1.1. Its upstream MIT license is preserved verbatim in [plotly.js.LICENSE.txt](efficient_frontier/third_party/plotly.js.LICENSE.txt), sourced from the [versioned upstream license](https://github.com/plotly/plotly.js/blob/v4.1.1/LICENSE). The HTML and report ZIP also include this notice.

## Market data and original holdings

Downloaded prices are not included in the repository. The [yfinance project](https://github.com/ranaroussi/yfinance) directs users to Yahoo's terms for rights to downloaded data and describes the API as intended for personal use. The software's license does not grant redistribution rights to market data. An exported report includes its input prices; review the provider's terms before sharing it.

`spy_holdings.ods` is the original user-supplied list of 60 ticker symbols. Its external provenance and redistribution terms are undocumented. Confirm them before a public release; do not describe the file as an official or historical S&P 500 dataset. The README screenshot uses only the application's synthetic demonstration data.
