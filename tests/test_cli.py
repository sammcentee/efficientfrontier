import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark_inputs():
    rng = np.random.default_rng(72)
    dates = pd.bdate_range("2022-01-03", periods=240, name="Date")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0004, .009, (240, 4)), axis=0)),
                          index=dates, columns=["ASSET_A", "ASSET_B", "SPY", "QQQ"])
    return prices.iloc[:, :2], prices.loc[:, ["SPY", "QQQ"]]


@pytest.fixture
def nasdaq_universe(benchmark_inputs):
    from efficient_frontier.universe import NASDAQ100_URL, UNIVERSE_LIMITATION, UniversePrices, UniverseSnapshot

    prices, benchmarks = benchmark_inputs
    members = pd.DataFrame({"symbol": [*prices.columns, "SHORT"],
                            "name": ["Company A", "Company B", "Recent company"],
                            "industry": ["Technology", "Industrials", "Technology"]})
    snapshot = UniverseSnapshot(members, NASDAQ100_URL, "2026-10-02", "2026-10-04T12:00:00+00:00")
    coverage = members.assign(status=["included", "included", "excluded"],
                              reason=["", "", "Incomplete historical prices."],
                              observations=[len(prices), len(prices), 20],
                              first_date=[str(prices.index[0].date()), str(prices.index[0].date()), str(prices.index[-20].date())],
                              last_date=str(prices.index[-1].date()))
    metadata = {"universe": "Current Nasdaq-100 members", "universe_source_url": NASDAQ100_URL,
                "universe_source_date": snapshot.source_date, "universe_retrieved_at": snapshot.retrieved_at,
                "universe_requested": 3, "universe_included": 2, "universe_excluded": 1,
                "universe_limitation": UNIVERSE_LIMITATION, "universe_coverage": coverage.to_dict(orient="records")}
    return snapshot, UniversePrices(prices, coverage, benchmarks, metadata)


def forbid_download(*args, **kwargs):
    pytest.fail("This CLI invocation must not request network data.")


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


@pytest.mark.parametrize("reuse_columns", [False, True])
def test_yahoo_benchmarks_reuse_existing_columns_or_fetch_without_changing_universe(
    tmp_path, monkeypatch, capsys, benchmark_inputs, reuse_columns,
):
    from efficient_frontier import __main__ as cli

    assets, benchmarks = benchmark_inputs
    prices = pd.concat([assets, benchmarks], axis=1) if reuse_columns else assets
    portfolio_calls, benchmark_calls = [], []

    def download_portfolio(tickers, start, end):
        portfolio_calls.append(tickers)
        return prices

    def download_benchmarks(dates):
        benchmark_calls.append(dates)
        return benchmarks

    monkeypatch.setattr(cli, "download_prices", download_portfolio)
    monkeypatch.setattr(cli, "download_benchmarks", download_benchmarks)
    output = tmp_path / "yahoo"
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--tickers", ",".join(prices.columns),
                                      "--output", str(output)])
    assert cli.main() == 0
    assert portfolio_calls == [list(prices.columns)]
    assert len(benchmark_calls) == (0 if reuse_columns else 1)
    if benchmark_calls:
        pd.testing.assert_index_equal(benchmark_calls[0], prices.index)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["price_currency"] == "USD"
    assert metadata["benchmark_source"] == ("Asset columns: SPY, QQQ" if reuse_columns else "Yahoo Finance")
    assert "benchmarks_note" not in metadata
    weights = pd.read_csv(output / "weights.csv", index_col=0)
    assert list(weights.index) == list(prices.columns)
    benchmark_prices = pd.read_csv(output / "benchmark_prices.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(benchmark_prices, benchmarks, check_freq=False)
    evidence = pd.read_csv(output / "evidence_summary.csv", index_col=[0, 1])
    assert set(evidence.index.get_level_values(1)) == {"S&P 500 (SPY)", "Nasdaq-100 (QQQ)"}
    assert "Historical evidence for Minimum volatility" in capsys.readouterr().out


def test_csv_benchmark_source_stays_offline_until_download_is_explicit(
    tmp_path, monkeypatch, capsys, benchmark_inputs,
):
    from efficient_frontier import __main__ as cli

    assets, benchmarks = benchmark_inputs
    source, output = tmp_path / "assets.csv", tmp_path / "report"
    assets.to_csv(source)
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    command = ["efficient_frontier", "--csv", str(source), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", command)
    assert cli.main() == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert "CSV analysis stays offline" in metadata["benchmarks_note"]
    assert "--benchmark-csv" in capsys.readouterr().out
    assert not (output / "benchmark_prices.csv").exists()
    calls = []

    def download(dates):
        calls.append(dates)
        return benchmarks

    monkeypatch.setattr(cli, "download_benchmarks", download)
    monkeypatch.setattr(sys, "argv", [*command, "--download-benchmarks"])
    assert cli.main() == 0
    assert len(calls) == 1
    pd.testing.assert_index_equal(calls[0], assets.index)
    assert (output / "benchmark_prices.csv").is_file()
    assert json.loads((output / "metadata.json").read_text())["benchmark_source"] == "Yahoo Finance"


def test_benchmark_csv_reproduction_and_cleanup_preserve_unrelated_files(
    tmp_path, monkeypatch, capsys, benchmark_inputs,
):
    from efficient_frontier import __main__ as cli

    assets, benchmarks = benchmark_inputs
    # Explicit benchmark input takes priority over benchmark symbols in the asset file.
    prices = pd.concat([assets, benchmarks * 2], axis=1)
    source, benchmark_source = tmp_path / "assets.csv", tmp_path / "market.csv"
    prices.to_csv(source)
    benchmarks.to_csv(benchmark_source)
    output = tmp_path / "report"
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--csv", str(source),
                                      "--benchmark-csv", str(benchmark_source), "--backtests", "--output", str(output)])
    assert cli.main() == 0
    assert "Historical evidence for Expanding window · Medium" in capsys.readouterr().out
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["benchmark_source"] == "CSV: market.csv"
    actual_benchmarks = pd.read_csv(output / "benchmark_prices.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(actual_benchmarks, benchmarks, check_freq=False)
    original_weights = pd.read_csv(output / "weights.csv", index_col=0)
    original_metrics = pd.read_csv(output / "benchmark_holdout_metrics.csv", index_col=0)
    evidence = pd.read_csv(output / "evidence_summary.csv", index_col=[0, 1])
    assert len(evidence) == 48
    evaluation = pd.read_csv(output / "benchmark_backtest_curve.csv", index_col=0)
    assert evidence.observations.eq(len(evaluation) - 2).all()
    preserved = [output / "benchmark_notes.csv", output / "evidence_custom.csv"]
    for path in preserved:
        path.write_text("Keep this user file.\n")

    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--csv", str(output / "prices.csv"),
                                      "--benchmark-csv", str(output / "benchmark_prices.csv"), "--output", str(output)])
    assert cli.main() == 0
    pd.testing.assert_frame_equal(pd.read_csv(output / "weights.csv", index_col=0), original_weights,
                                  atol=1e-6, rtol=1e-6)
    pd.testing.assert_frame_equal(pd.read_csv(output / "benchmark_holdout_metrics.csv", index_col=0), original_metrics)
    assert not (output / "benchmark_backtest_metrics.csv").exists()
    assert not (output / "benchmark_backtest_curve.csv").exists()
    evidence = pd.read_csv(output / "evidence_summary.csv", index_col=[0, 1])
    evaluation = pd.read_csv(output / "benchmark_holdout_curve.csv", index_col=0)
    assert evidence.observations.eq(len(evaluation) - 1).all()

    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--csv", str(output / "prices.csv"),
                                      "--no-benchmarks", "--output", str(output)])
    assert cli.main() == 0
    generated = ["benchmark_prices.csv", "benchmark_holdout_metrics.csv", "benchmark_holdout_curve.csv",
                 "benchmark_training_estimates.csv", "benchmark_latest_estimates.csv",
                 "benchmark_backtest_metrics.csv", "benchmark_backtest_curve.csv",
                 "evidence_summary.csv", "evidence_windows.csv", "evidence_relative_curve.csv"]
    assert not any((output / name).exists() for name in generated)
    assert all(path.read_text() == "Keep this user file.\n" for path in preserved)
    assert (output / "report.html").is_file()


@pytest.mark.parametrize("benchmark_flag", ["--download-benchmarks", "--benchmark-csv"])
def test_non_usd_currency_skips_benchmark_io(tmp_path, monkeypatch, benchmark_inputs, benchmark_flag):
    from efficient_frontier import __main__ as cli

    assets, _ = benchmark_inputs
    source, output = tmp_path / "assets.csv", tmp_path / "report"
    assets.to_csv(source)
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    flags = [benchmark_flag] if benchmark_flag == "--download-benchmarks" else [benchmark_flag, str(tmp_path / "absent.csv")]
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--csv", str(source), "--currency", "eur",
                                      *flags, "--output", str(output)])
    assert cli.main() == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["price_currency"] == "EUR"
    assert "require USD" in metadata["benchmarks_note"]
    assert "no currency conversion" in metadata["benchmarks_note"]
    assert (output / "weights.csv").is_file()
    assert not (output / "benchmark_prices.csv").exists()


@pytest.mark.parametrize("failure", ["network", "missing_date"])
def test_benchmark_failures_preserve_portfolio_report(tmp_path, monkeypatch, benchmark_inputs, failure):
    from efficient_frontier import __main__ as cli

    assets, benchmarks = benchmark_inputs
    source, output = tmp_path / "assets.csv", tmp_path / "report"
    assets.to_csv(source)
    monkeypatch.setattr(cli, "download_prices", forbid_download)

    def failed_download(dates):
        raise ValueError("Benchmark service unavailable")

    monkeypatch.setattr(cli, "download_benchmarks", failed_download)
    if failure == "network":
        flags = ["--download-benchmarks"]
    else:
        benchmark_source = tmp_path / "missing.csv"
        benchmarks.drop(benchmarks.index[15]).to_csv(benchmark_source)
        flags = ["--benchmark-csv", str(benchmark_source)]
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--csv", str(source), *flags, "--output", str(output)])
    assert cli.main() == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert "unavailable" in metadata["benchmarks_note"]
    assert metadata["source"] == "CSV: assets.csv"
    assert not (output / "benchmark_prices.csv").exists()
    assert not (output / "evidence_summary.csv").exists()
    assert (output / "report.html").is_file()
    assert list(pd.read_csv(output / "weights.csv", index_col=0).index) == list(assets.columns)


def test_evidence_failure_preserves_valid_benchmark_paths(tmp_path, monkeypatch, benchmark_inputs):
    from efficient_frontier import __main__ as cli

    assets, benchmarks = benchmark_inputs
    source, output = tmp_path / "assets.csv", tmp_path / "report"
    pd.concat([assets, benchmarks], axis=1).to_csv(source)
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)

    def failed_evidence(*args, **kwargs):
        raise ValueError("Statistical comparison unavailable")

    monkeypatch.setattr(cli, "analyze_evidence", failed_evidence)
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--csv", str(source), "--output", str(output)])
    assert cli.main() == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert "benchmarks_note" not in metadata
    assert "Statistical comparison unavailable" in metadata["evidence_note"]
    assert (output / "benchmark_prices.csv").is_file()
    assert (output / "benchmark_holdout_curve.csv").is_file()
    assert not (output / "evidence_summary.csv").exists()


@pytest.mark.parametrize("benchmark_flag", ["--download-benchmarks", "--benchmark-csv"])
def test_demo_rejects_market_benchmarks_without_network(tmp_path, monkeypatch, capsys, benchmark_flag):
    from efficient_frontier import __main__ as cli

    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    flags = [benchmark_flag] if benchmark_flag == "--download-benchmarks" else [benchmark_flag, str(tmp_path / "absent.csv")]
    output = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", *flags, "--output", str(output)])
    assert cli.main() == 1
    assert "Synthetic demo data cannot use real market benchmarks" in capsys.readouterr().err
    assert not output.exists()


def test_nasdaq100_source_exports_coverage_and_reuses_downloaded_benchmarks(
    tmp_path, monkeypatch, capsys, nasdaq_universe,
):
    from efficient_frontier import __main__ as cli

    snapshot, universe = nasdaq_universe
    calls = []
    monkeypatch.setattr(cli, "fetch_nasdaq100", lambda: snapshot)

    def download(current_snapshot, start, end):
        calls.append((current_snapshot, start, end))
        return universe

    monkeypatch.setattr(cli, "download_universe_prices", download)
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    output = tmp_path / "nasdaq"
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--nasdaq100", "--start", "2022-01-01",
                                      "--end", "2023-01-01", "--output", str(output)])
    assert cli.main() == 0
    assert len(calls) == 1
    assert calls[0][0] is snapshot
    assert calls[0][1:] == ("2022-01-01", "2023-01-01")
    text = capsys.readouterr().out
    assert "Nasdaq-100 coverage: 2 of 3 securities included; 1 excluded." in text
    assert "not the members at each historical date" in text
    assert "2026-10-02" in text
    assert snapshot.source_url in text
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"] == "Nasdaq-100 · Yahoo Finance"
    assert metadata["universe_coverage"] == universe.coverage.to_dict(orient="records")
    assert metadata["universe_source_date"] == snapshot.source_date
    assert metadata["universe_retrieved_at"] == snapshot.retrieved_at
    assert metadata["universe_requested"] == 3
    assert metadata["universe_excluded"] == 1
    assert metadata["requested_start"] == "2022-01-01"
    assert metadata["requested_end_exclusive"] == "2023-01-01"
    assert metadata["benchmark_source"] == "Yahoo Finance · Nasdaq-100 download"
    assert list(pd.read_csv(output / "weights.csv", index_col=0).index) == list(universe.prices.columns)
    pd.testing.assert_frame_equal(pd.read_csv(output / "benchmark_prices.csv", index_col=0, parse_dates=True),
                                  universe.benchmarks, check_freq=False)
    coverage = pd.read_csv(output / "universe_coverage.csv")
    assert coverage.status.tolist() == ["included", "included", "excluded"]
    assert coverage.loc[coverage.symbol == "SHORT", "reason"].item() == "Incomplete historical prices."
    assert "Universe coverage" in (output / "report.html").read_text()

    # The default stays offline and removes only the obsolete generated coverage file.
    preserved = output / "universe_coverage_notes.csv"
    preserved.write_text("Keep this user file.\n")
    monkeypatch.setattr(cli, "fetch_nasdaq100", forbid_download)
    monkeypatch.setattr(cli, "download_universe_prices", forbid_download)
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--output", str(output)])
    assert cli.main() == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"] == "Demo · synthetic"
    assert "universe_coverage" not in metadata
    assert not (output / "universe_coverage.csv").exists()
    assert preserved.read_text() == "Keep this user file.\n"


@pytest.mark.parametrize("benchmark_mode", ["disabled", "other_currency", "csv", "download"])
def test_nasdaq100_source_preserves_explicit_benchmark_controls(
    tmp_path, monkeypatch, nasdaq_universe, benchmark_mode,
):
    from efficient_frontier import __main__ as cli

    snapshot, universe = nasdaq_universe
    monkeypatch.setattr(cli, "fetch_nasdaq100", lambda: snapshot)
    monkeypatch.setattr(cli, "download_universe_prices", lambda *args: universe)
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    if benchmark_mode == "disabled":
        flags = ["--no-benchmarks"]
    elif benchmark_mode == "other_currency":
        flags = ["--currency", "EUR"]
    elif benchmark_mode == "csv":
        path = tmp_path / "override.csv"
        (universe.benchmarks * 2).to_csv(path)
        flags = ["--benchmark-csv", str(path)]
    else:
        flags = ["--download-benchmarks"]
        calls = []

        def download(dates):
            calls.append(dates)
            return universe.benchmarks * 2

        monkeypatch.setattr(cli, "download_benchmarks", download)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--nasdaq100", *flags, "--output", str(output)])
    assert cli.main() == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["universe_included"] == 2
    assert (output / "universe_coverage.csv").is_file()
    if benchmark_mode in ("disabled", "other_currency"):
        assert "benchmarks_note" in metadata
        assert not (output / "benchmark_prices.csv").exists()
    else:
        assert metadata["benchmark_source"] == ("CSV: override.csv" if benchmark_mode == "csv" else "Yahoo Finance")
        pd.testing.assert_frame_equal(pd.read_csv(output / "benchmark_prices.csv", index_col=0, parse_dates=True),
                                      universe.benchmarks * 2, check_freq=False)
        if benchmark_mode == "download":
            assert len(calls) == 1
            pd.testing.assert_index_equal(calls[0], universe.prices.index)


@pytest.mark.parametrize("failure", ["snapshot", "coverage"])
def test_nasdaq100_failure_does_not_fall_back_to_demo(tmp_path, monkeypatch, capsys, nasdaq_universe, failure):
    from efficient_frontier import __main__ as cli
    from efficient_frontier.universe import UniversePrices

    snapshot, universe = nasdaq_universe
    monkeypatch.setattr(cli, "demo_prices", forbid_download)
    monkeypatch.setattr(cli, "download_prices", forbid_download)
    monkeypatch.setattr(cli, "download_benchmarks", forbid_download)
    if failure == "snapshot":
        def unavailable():
            raise ValueError("The current Nasdaq-100 list is unavailable or incomplete.")

        monkeypatch.setattr(cli, "fetch_nasdaq100", unavailable)
        monkeypatch.setattr(cli, "download_universe_prices", forbid_download)
        error = "current Nasdaq-100 list is unavailable"
    else:
        excluded = UniversePrices(
            universe.prices.iloc[:, :0], universe.coverage.assign(status="excluded", reason="No complete prices."),
            universe.benchmarks, {**universe.metadata, "universe_included": 0, "universe_excluded": 3},
        )
        monkeypatch.setattr(cli, "fetch_nasdaq100", lambda: snapshot)
        monkeypatch.setattr(cli, "download_universe_prices", lambda *args: excluded)
        error = "Only 0 of 3 Nasdaq-100 securities have complete prices. At least two are required. 3 securities were excluded."
    output = tmp_path / "failed"
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", "--nasdaq100", "--output", str(output)])
    assert cli.main() == 1
    assert error in capsys.readouterr().err
    assert not output.exists()
