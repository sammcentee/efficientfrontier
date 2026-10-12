import csv
from html.parser import HTMLParser
import io
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.historical_report import build_historical_report, historical_report_zip
from efficient_frontier.historical import historical_demo, load_historical_prices, run_historical_backtests
from efficient_frontier.membership import bundled_membership, load_membership_csv


@pytest.fixture
def historical_inputs():
    dates = pd.bdate_range("2024-01-02", periods=5)
    names = [f"Rolling window · {policy}" for policy in ("Minimum volatility", "Maximum Sharpe", "Equal weight")]
    prices = pd.DataFrame({"AAA": [100., 101., 103., 102., 104.],
                           "BBB": [np.nan, np.nan, 80., 81., 82.]}, index=dates)
    snapshots = pd.Series([("AAA",), ("AAA", "BBB")], index=dates[[0, 2]])
    membership = SimpleNamespace(
        snapshots=snapshots, coverage_start=dates[0], coverage_end=dates[-1], symbols=("AAA", "BBB"),
        metadata={"source": "Synthetic membership", "limitations": ["Fixture only"], "revision": "test-123"},
    )
    equity = pd.DataFrame({name: [1., .999, 1.01] for name in names}, index=dates[2:])
    metrics = pd.DataFrame({
        "total_return": [.01] * 3, "cagr": [.1] * 3, "volatility": [.12] * 3,
        "sharpe": [1.] * 3, "sortino": [1.2] * 3, "calmar": [np.nan] * 3,
        "max_drawdown": [-.001] * 3, "total_turnover": [1.] * 3, "total_cost": [.001] * 3,
        "rebalance_count": [1] * 3, "fallback_count": [0] * 3,
    }, index=names)
    holdings = pd.DataFrame({
        "average_weight": [.5, 0.], "end_weight": [1., 0.], "selection_frequency": [1., 0.],
        "max_target_weight": [1., 0.], "max_realized_weight": [1., 0.], "pnl_contribution": [.011, 0.],
    }, index=["AAA", "BBB"])
    allocations = pd.DataFrame({"AAA": [1.], "BBB": [0.]}, index=dates[[3]])
    trades = pd.DataFrame({"turnover": [1.], "cost": [.001], "cash_weight": [0.]}, index=dates[[3]])
    study = SimpleNamespace(
        equity=equity, metrics=metrics,
        cash_weights=pd.DataFrame({name: [1., 0., 0.] for name in names}, index=dates[2:]),
        coverage=pd.DataFrame({"date": dates[[3, 3]], "symbol": ["AAA", "BBB"],
                               "status": ["eligible", "excluded"], "reason": ["", "Insufficient prior history"]}),
        allocations={name: allocations.copy() for name in names},
        holdings={name: holdings.copy() for name in names}, trades={name: trades.copy() for name in names},
        warnings=["BBB is excluded from this fit."],
        settings={"rolling_window": 2, "rebalance_every": 21, "cost_bps": 10., "periods_per_year": 252},
    )
    return prices, membership, study


def test_bundle_preserves_sparse_inputs_membership_and_all_strategy_results(historical_inputs):
    prices, membership, study = historical_inputs
    files = build_historical_report(*historical_inputs)
    details = json.loads(files["metadata.json"])
    assert details["membership"]["revision"] == "test-123"
    assert details["membership"]["coverage_start"] == "2024-01-02"
    assert details["membership"]["coverage_end"] == "2024-01-08"
    assert details["settings"] == study.settings
    assert details["warnings"] == study.warnings
    recovered = pd.read_csv(io.BytesIO(files["prices.csv"]), index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(recovered, prices.rename_axis("Date"), check_freq=False)
    snapshots = pd.read_csv(io.BytesIO(files["membership.csv"]), index_col=0, parse_dates=True)
    assert list(snapshots.index) == list(membership.snapshots.index)
    assert list(snapshots.tickers) == ["AAA", "AAA,BBB"]
    for stem, strategy in details["backtest_files"].items():
        for kind in ("allocations", "holdings", "trades"):
            recovered = pd.read_csv(io.BytesIO(files[f"{stem}_{kind}.csv"]), index_col=0,
                                    parse_dates=kind != "holdings")
            expected = getattr(study, kind)[strategy].rename_axis("symbol" if kind == "holdings" else "date")
            pd.testing.assert_frame_equal(recovered, expected, check_freq=False)
    with zipfile.ZipFile(io.BytesIO(historical_report_zip(files))) as archive:
        assert set(archive.namelist()) == set(files)
        for path, content in files.items():
            assert archive.read(path) == content


def test_cash_and_exclusion_exports_keep_dates_and_values(historical_inputs):
    study = historical_inputs[2]
    files = build_historical_report(*historical_inputs)
    cash = pd.read_csv(io.BytesIO(files["cash_weights.csv"]), index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(cash, study.cash_weights.rename_axis("date"), check_freq=False)
    coverage = pd.read_csv(io.BytesIO(files["coverage.csv"]), parse_dates=["date"], keep_default_na=False)
    pd.testing.assert_frame_equal(coverage, study.coverage)
    metrics = pd.read_csv(io.BytesIO(files["metrics.csv"]), index_col=0)
    pd.testing.assert_frame_equal(metrics, study.metrics.rename_axis("strategy"))
    equity = pd.read_csv(io.BytesIO(files["equity.csv"]), index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(equity, study.equity.rename_axis("date"), check_freq=False)
    assert historical_inputs[0].index.name is None
    assert study.equity.index.name is None


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if (tag == "script" and name == "src") or (tag == "link" and name == "href"):
                self.external_resources.append(value)


def test_report_runs_offline_with_one_plotly_bundle_and_license(historical_inputs):
    files = build_historical_report(*historical_inputs)
    document = files["report.html"].decode()
    parser = ResourceParser()
    parser.feed(document)
    assert not parser.external_resources
    assert document.startswith("<!doctype html>")
    assert document.count("* plotly.js v") == 1
    assert document.count("Plotly.newPlot(") >= 2
    assert "Permission is hereby granted, free of charge" in document
    assert "Permission is hereby granted, free of charge" in files["THIRD_PARTY_NOTICES.txt"].decode()
    assert "Synthetic membership" in document and "Fixture only" in document
    assert "Portfolio baseline: 2024-01-04" in document
    assert "Evaluation returns: 2024-01-05 to 2024-01-08" in document
    assert "Removed companies remain held" in document
    assert "cash_weights.csv contains daily closing cash proportions" in document
    assert "Missing prices for a held asset or an execution stop the study" in document
    assert "complete historical reconstruction" in document
    assert "fixed input universe" not in document
    for name in historical_inputs[2].equity:
        assert name in document


def test_report_escapes_metadata_warnings_and_coverage(historical_inputs):
    _, membership, study = historical_inputs
    membership.metadata["source"] = '<script>alert("source")</script>'
    membership.metadata["<key>"] = ["<nested-value>"]
    study.settings["<setting>"] = "<value>"
    study.warnings = ['<img src=x onerror="warning()">']
    study.coverage.loc[1, "reason"] = '<script>alert("coverage")</script>'
    document = build_historical_report(*historical_inputs)["report.html"].decode()
    for unsafe in ('<script>alert("source")</script>', '<img src=x onerror="warning()">',
                   '<script>alert("coverage")</script>', "<setting>", "<nested-value>"):
        assert unsafe not in document
    assert '&lt;script&gt;alert(&quot;source&quot;)&lt;/script&gt;' in document
    assert "&lt;setting&gt;" in document.lower()
    assert "&lt;nested-value&gt;" in document


def test_csv_exports_escape_formula_text_without_changing_numeric_losses(historical_inputs):
    study = historical_inputs[2]
    study.coverage.loc[1, "reason"] = "=HYPERLINK(\"https://example.invalid\")"
    study.holdings[next(iter(study.holdings))].loc["AAA", "pnl_contribution"] = -.25
    files = build_historical_report(*historical_inputs)
    coverage = list(csv.DictReader(io.StringIO(files["coverage.csv"].decode())))
    assert coverage[1]["reason"].startswith("'=HYPERLINK")
    holdings = pd.read_csv(io.BytesIO(files["backtests/01_holdings.csv"]), index_col=0)
    assert holdings.loc["AAA", "pnl_contribution"] == -.25


def test_bundled_membership_export_carries_identity_rules_and_upstream_license(historical_inputs):
    membership = bundled_membership()
    files = build_historical_report(historical_inputs[0], membership, historical_inputs[2])
    metadata = json.loads(files["metadata.json"])
    assert metadata["membership"]["identity_exclusions"] == membership.metadata["identity_exclusions"]
    assert membership.metadata["source_commit"] == metadata["membership"]["source_commit"]
    license_text = files["LICENSE.n100tickers"].decode()
    assert "Permission is hereby granted, free of charge" in license_text
    assert license_text in files["THIRD_PARTY_NOTICES.txt"].decode()
    document = files["report.html"].decode()
    assert "keep --historical-membership with the same bundled data revision" in document
    assert "membership.csv alone does not preserve identity exclusions" in document
    assert "--membership-csv" in document and "does not restore the rules" in document


def test_demo_export_inputs_reproduce_equity_and_cash_paths():
    prices, membership = historical_demo()
    study = run_historical_backtests(prices, membership)
    files = build_historical_report(prices, membership, study)
    details = json.loads(files["metadata.json"])
    recovered_prices = load_historical_prices(io.BytesIO(files["prices.csv"]))
    recovered_membership = load_membership_csv(io.BytesIO(files["membership.csv"]),
                                               coverage_end=details["membership"]["coverage_end"])
    recovered = run_historical_backtests(recovered_prices, recovered_membership)
    pd.testing.assert_frame_equal(recovered.equity, study.equity, check_freq=False)
    pd.testing.assert_frame_equal(recovered.cash_weights, study.cash_weights, check_freq=False)
    for name in study.equity:
        total = study.holdings[name].pnl_contribution.sum() - study.metrics.loc[name, "total_cost"]
        assert total == pytest.approx(study.metrics.loc[name, "total_return"])
