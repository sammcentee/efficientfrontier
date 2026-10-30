"""Synthetic prices of `tests/test_cli.py` for the Rust CLI tests.

The seeded NumPy draws of the Python fixtures cannot be repeated in Rust, so
this script writes the exact values as CSV. Run it with the Python release's
requirements from any folder:

    python tests/fixtures/cli/generate_cli.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# The `benchmark_inputs` fixture: ASSET_A and ASSET_B are the assets, SPY and QQQ the benchmarks.
rng = np.random.default_rng(72)
dates = pd.bdate_range("2022-01-03", periods=240, name="Date")
pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0004, .009, (240, 4)), axis=0)),
             index=dates, columns=["ASSET_A", "ASSET_B", "SPY", "QQQ"]).to_csv(HERE / "benchmark_inputs.csv")

# `test_monthly_csv_cli_uses_selected_annualization_and_exports_risk`.
rng = np.random.default_rng(9)
pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(.008, .03, (36, 2)), axis=0)),
    index=pd.date_range("2020-01-31", periods=36, freq="ME", name="Date"),
    columns=["FUND_A", "FUND_B"],
).to_csv(HERE / "monthly.csv")
