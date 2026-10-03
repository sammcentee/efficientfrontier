import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_cli_offline_export_and_csv_reproduction(tmp_path):
    output = tmp_path / "demo"
    run = subprocess.run([sys.executable, "-m", "efficient_frontier", "--output", str(output)],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "synthetic" in run.stdout
    assert (output / "report.html").stat().st_size > 1000
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["training_end"] < metadata["holdout_start"]
    reproduced = tmp_path / "reproduced"
    run = subprocess.run([sys.executable, "-m", "efficient_frontier", "--csv",
                          str(output / "prices.csv"), "--output", str(reproduced)],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    pd.testing.assert_frame_equal(pd.read_csv(output / "weights.csv"), pd.read_csv(reproduced / "weights.csv"),
                                  atol=1e-6, rtol=1e-6)
