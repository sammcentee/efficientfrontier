import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import numpy as np


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
    assert metadata["max_weight"] == 1.0
    assert metadata["latest_profiles"]["as_of"] == "2023-10-31"
    assert "Latest model holdings" in run.stdout
    latest_weights = pd.read_csv(output / "latest_profile_weights.csv", index_col=0)
    assert list(latest_weights.columns) == ["Low", "Medium", "Extreme"]
    np.testing.assert_allclose(latest_weights.sum(), 1)
    reproduced = tmp_path / "reproduced"
    run = subprocess.run([sys.executable, "-m", "efficient_frontier", "--csv",
                          str(output / "prices.csv"), "--output", str(reproduced)],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    pd.testing.assert_frame_equal(pd.read_csv(output / "weights.csv"), pd.read_csv(reproduced / "weights.csv"),
                                  atol=1e-6, rtol=1e-6)
    pd.testing.assert_frame_equal(pd.read_csv(output / "latest_profile_weights.csv"),
                                  pd.read_csv(reproduced / "latest_profile_weights.csv"), atol=1e-6, rtol=1e-6)


def test_explicit_empty_universe_does_not_fall_back_to_demo(tmp_path):
    output = tmp_path / "empty"
    run = subprocess.run([sys.executable, "-m", "efficient_frontier", "--tickers", "", "--output", str(output)],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 1
    assert "at least one" in run.stderr.lower()
    assert not output.exists()


def test_monthly_csv_cli_uses_selected_annualization_and_exports_risk(tmp_path):
    rng = np.random.default_rng(9)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(.008, .03, (36, 2)), axis=0)),
        index=pd.date_range("2020-01-31", periods=36, freq="ME", name="Date"),
        columns=["FUND_A", "FUND_B"],
    )
    source = tmp_path / "monthly.csv"
    prices.to_csv(source)
    output = tmp_path / "monthly"
    run = subprocess.run([
        sys.executable, "-m", "efficient_frontier", "--csv", str(source),
        "--periods-per-year", "12", "--backtests", "--rebalance-every", "3",
        "--rolling-window", "12", "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["periods_per_year"] == 12
    assert metadata["backtests"]["periods_per_year"] == 12
    assert metadata["backtests"]["rebalance_every"] == 3
    assert metadata["backtests"]["include_profiles"] is True
    windows = pd.read_csv(output / "latest_profile_windows.csv", index_col=0)
    assert windows.observations.sum() == 35
    latest = pd.read_csv(output / "latest_profile_summary.csv", index_col=0)
    window_returns = pd.read_csv(output / "latest_profile_window_returns.csv", index_col=0)
    np.testing.assert_allclose(latest.worst_window_return, window_returns.min())
    metrics = pd.read_csv(output / "holdout_metrics.csv", index_col=0)
    curve = pd.read_csv(output / "holdout_curve.csv", index_col=0)
    np.testing.assert_allclose(metrics.cagr, curve.iloc[-1].pow(12 / (len(curve) - 1)) - 1)
    assert {"sortino", "calmar"} <= set(metrics.columns)
    assert (output / "risk_summary.csv").is_file()
    contributions = pd.read_csv(output / "risk_contributions.csv", index_col=0)
    np.testing.assert_allclose(contributions.sum(), 1)
    assert "12 observations per year" in (output / "report.html").read_text()


def test_invalid_annualization_fails_before_export(tmp_path):
    output = tmp_path / "invalid"
    run = subprocess.run([
        sys.executable, "-m", "efficient_frontier", "--periods-per-year", "0",
        "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 1
    assert "periods_per_year" in run.stderr
    assert not output.exists()


def test_backtest_export_has_all_methods_trades_and_reconciled_holdings(tmp_path):
    output = tmp_path / "backtests"
    run = subprocess.run([sys.executable, "-m", "efficient_frontier", "--backtests",
                          "--rolling-window", "252", "--cost-bps", "10", "--output", str(output)],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["backtests"]["train_end"] < metadata["backtests"]["initial_execution"]
    assert len(metadata["backtest_files"]) == 24
    metrics = pd.read_csv(output / "backtest_metrics.csv", index_col=0)
    for prefix, strategy in metadata["backtest_files"].items():
        holdings = pd.read_csv(output / f"{prefix}_holdings.csv", index_col=0)
        trades = pd.read_csv(output / f"{prefix}_trades.csv", index_col=0)
        assert abs(holdings.pnl_contribution.sum() - trades.cost.sum() - metrics.loc[strategy, "total_return"]) < 1e-10
        assert all(pd.to_datetime(trades.train_end) < pd.to_datetime(trades.index))
    findings = (output / "findings.md").read_text()
    assert "synthetic" in findings and "Rolling window" in findings
    document = (output / "report.html").read_text()
    assert document.index("* plotly.js v") < document.index('<section id="backtesting">')
    assert document.count("* plotly.js v") == 1
    assert "Print / save PDF" in document


def test_reruns_remove_obsolete_backtest_outputs_but_preserve_unrelated_files(tmp_path):
    output = tmp_path / "reused"
    command = [sys.executable, "-m", "efficient_frontier", "--output", str(output)]
    run = subprocess.run([*command, "--backtests"], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    preserved = [output / "notes.txt", output / "backtests" / "notes.csv",
                 output / "backtests" / "13_custom.csv"]
    for path in preserved:
        path.write_text("Keep this user file.\n")
    obsolete = output / "backtests" / "25_holdings.csv"
    obsolete.write_text("old,output\n")

    run = subprocess.run([*command, "--backtests", "--cost-bps", "0"],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert not obsolete.exists()
    assert len(list((output / "backtests").glob("[0-9][0-9]_holdings.csv"))) == 24
    assert pd.read_csv(output / "backtest_metrics.csv").total_cost.eq(0).all()

    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    for name in ("findings.md", "backtest_metrics.csv", "backtest_curve.csv"):
        assert not (output / name).exists()
    assert set((output / "backtests").iterdir()) == set(preserved[1:])
    assert all(path.read_text() == "Keep this user file.\n" for path in preserved)
    assert "backtests" not in json.loads((output / "metadata.json").read_text())
    assert (output / "report.html").is_file()

    short_prices = pd.DataFrame({"AAA": [100, 102, 101, 103, 102], "BBB": [100, 101, 103, 102, 104]},
                                index=pd.bdate_range("2024-01-01", periods=5, name="Date"))
    source = tmp_path / "short.csv"
    short_prices.to_csv(source)
    run = subprocess.run([*command, "--csv", str(source), "--backtests"],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "at least seven prices" in run.stdout
    assert "at least six returns" in run.stdout
    assert not list(output.glob("latest_profile_*.csv"))
    assert len(pd.read_csv(output / "backtest_metrics.csv")) == 12
    assert not json.loads((output / "metadata.json").read_text())["backtests"]["include_profiles"]
