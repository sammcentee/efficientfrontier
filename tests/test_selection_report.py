import copy
from dataclasses import dataclass
import html
from html.parser import HTMLParser
import io
import json
import re
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.backtest import BacktestStudy
from efficient_frontier.core import Portfolio
from efficient_frontier.selection_report import selection_report_html, selection_report_zip


@dataclass(frozen=True)
class PolicyFixture:
    blend: float
    validation_passed: bool
    reasons: tuple[str, ...]


@pytest.fixture
def selection_inputs():
    dates = pd.bdate_range("2024-01-02", periods=12)
    prices = pd.DataFrame({"AAA": np.linspace(100, 112, 12),
                           "BBB": np.linspace(80, 85, 12)}, index=dates)
    benchmarks = pd.DataFrame({"SPY": np.linspace(400, 430, 12),
                               "QQQ": np.linspace(350, 380, 12)}, index=dates)
    names = ["Selected stock rule", "Diversified baseline", "Blend 25%", "Blend 50%",
             "SPY", "QQQ", "SPY risk-matched", "QQQ risk-matched"]

    def phase(start, offset):
        phase_dates = dates[start:start + 4]
        metrics = pd.DataFrame({
            "annual_mean": [.031234567 + offset + i / 100 for i in range(len(names))],
            "cagr": [.021234567 + offset + i / 100 for i in range(len(names))],
            "volatility": [.12] * len(names), "max_drawdown": [-.025] * len(names),
            "total_return": [.02 + offset + i / 100 for i in range(len(names))],
            "total_cost": [.001234567] * len(names),
            "sharpe": [np.nan] * len(names), "fallback_count": [0] * len(names),
        }, index=names)
        equity = pd.DataFrame({name: [1., .999, .98, 1 + metrics.loc[name, "total_return"]]
                               for name in names}, index=phase_dates)
        allocations = pd.DataFrame({"AAA": [.6], "BBB": [.3], "CASH": [.1]}, index=phase_dates[[1]])
        holdings = pd.DataFrame({
            "average_weight": [.45, .25, .3], "end_weight": [.62, .28, .1],
            "selection_frequency": [1., 1., 1.], "max_target_weight": [.6, .3, .1],
            "max_realized_weight": [.62, .31, 1.], "pnl_contribution": [.025, -.004, 0.],
        }, index=["AAA", "BBB", "CASH"])
        trades = pd.DataFrame({"turnover": [.9], "cost": [.001234567], "cash_weight": [.1]},
                              index=phase_dates[[1]])
        return BacktestStudy(
            equity=equity, metrics=metrics, allocations={name: allocations.copy() for name in names},
            holdings={name: holdings.copy() for name in names}, trades={name: trades.copy() for name in names},
            warnings=[f"Phase starts at {phase_dates[0].date()}."],
            settings={"cost_bps": 10., "periods_per_year": 252, "target_volatility": .15},
        )

    study = SimpleNamespace(
        policy=PolicyFixture(.25, True, ("Validation passed its fixed screen.",)),
        validation=phase(3, 0.), test=phase(8, .01), selected=names[0],
        latest={
            names[0]: Portfolio(names[0], pd.Series({"AAA": .6, "BBB": .4}), .123456789, .14, .738976),
            names[1]: Portfolio(names[1], pd.Series({"AAA": .5, "BBB": .5}), .098765432, .12, .6563786),
        },
        supported=False, reasons=["The later test did not clear the descriptive screen."],
        settings={"target_volatility": .15, "max_weight": .6, "cost_bps": 10., "periods_per_year": 252,
                  "validation_start": "2024-01-05", "test_start": "2024-01-12", "rebalance_every": 21},
        warnings=["Current constituents can introduce survivorship bias."], benchmark_prices=benchmarks,
        eligibility=pd.DataFrame({"date": dates[[3, 8]], "ticker": ["AAA", "BBB"],
                                  "eligible": [True, False], "reason": ["", "Long unchanged-price run"]}),
    )
    metadata = {
        "source": "Synthetic adjusted prices", "universe": "Fixture stocks", "revision": "fixture-123",
        "universe_coverage": [{"ticker": "AAA", "status": "included", "reason": ""},
                              {"ticker": "OLD", "status": "excluded", "reason": "Missing earlier prices"}],
    }
    return study, prices, metadata


def test_zip_preserves_every_phase_and_strategy_numeric_result(selection_inputs):
    study, prices, metadata = selection_inputs
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs))) as archive:
        details = json.loads(archive.read("metadata.json"))
        assert len(details["result_files"]) == len(study.validation.equity.columns) + len(study.test.equity.columns)
        for phase_name in ("validation", "test"):
            phase = getattr(study, phase_name)
            for kind in ("metrics", "equity"):
                recovered = pd.read_csv(archive.open(f"{phase_name}/{kind}.csv"), index_col=0,
                                        parse_dates=kind == "equity")
                pd.testing.assert_frame_equal(recovered, getattr(phase, kind), check_freq=False, check_names=False)
            strategy_files = {stem: name for stem, name in details["result_files"].items()
                              if stem.startswith(f"{phase_name}/")}
            assert list(strategy_files.values()) == list(phase.equity.columns)
            for stem, name in strategy_files.items():
                assert re.fullmatch(rf"{phase_name}/\d+", stem)
                for kind in ("allocations", "holdings", "trades"):
                    recovered = pd.read_csv(archive.open(f"{stem}_{kind}.csv"), index_col=0,
                                            parse_dates=kind != "holdings")
                    pd.testing.assert_frame_equal(recovered, getattr(phase, kind)[name],
                                                  check_freq=False, check_names=False)
        for path, expected in (("prices.csv", prices), ("benchmark_prices.csv", study.benchmark_prices)):
            recovered = pd.read_csv(archive.open(path), index_col=0, parse_dates=True)
            pd.testing.assert_frame_equal(recovered, expected, check_freq=False, check_names=False)
        coverage = pd.read_csv(archive.open("universe_coverage.csv"), index_col=0, keep_default_na=False)
        pd.testing.assert_frame_equal(coverage, pd.DataFrame(metadata["universe_coverage"]))
        eligibility = pd.read_csv(archive.open("eligibility.csv"), index_col=0,
                                  parse_dates=["date"], keep_default_na=False)
        pd.testing.assert_frame_equal(eligibility, study.eligibility)


def test_zip_retains_policy_source_gate_reasons_and_exact_latest_estimates(selection_inputs):
    study, _, metadata = selection_inputs
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs))) as archive:
        details = json.loads(archive.read("metadata.json"))
        policy = json.loads(archive.read("policy.json"))
        assert policy == {"blend": .25, "validation_passed": True, "reasons": list(study.policy.reasons)}
        assert details["policy"] == policy
        assert details["settings"] == study.settings
        assert details["supported"] is False and details["reasons"] == study.reasons
        assert metadata["source"] in json.dumps(details)
        assert metadata["revision"] in json.dumps(details)
        assert study.warnings[0] in json.dumps(details)
        weights = pd.read_csv(archive.open("latest_weights.csv"), index_col=0)
        expected = pd.DataFrame({name: portfolio.weights for name, portfolio in study.latest.items()})
        pd.testing.assert_frame_equal(weights, expected, check_names=False)
        estimates = pd.read_csv(archive.open("latest_estimates.csv"), index_col=0)
        expected = pd.DataFrame({name: {key: getattr(portfolio, key) for key in
                                      ("expected_return", "volatility", "sharpe")}
                                 for name, portfolio in study.latest.items()}).T
        pd.testing.assert_frame_equal(estimates, expected, check_names=False)


def test_report_calls_preserve_all_input_objects(selection_inputs):
    study, prices, metadata = selection_inputs
    original = copy.deepcopy(selection_inputs)
    selection_report_html(*selection_inputs)
    selection_report_zip(*selection_inputs)
    saved_study, saved_prices, saved_metadata = original
    pd.testing.assert_frame_equal(prices, saved_prices)
    pd.testing.assert_frame_equal(study.benchmark_prices, saved_study.benchmark_prices)
    pd.testing.assert_frame_equal(study.eligibility, saved_study.eligibility)
    assert metadata == saved_metadata
    for key in ("policy", "selected", "supported", "reasons", "settings", "warnings"):
        assert getattr(study, key) == getattr(saved_study, key)
    for name, portfolio in study.latest.items():
        pd.testing.assert_series_equal(portfolio.weights, saved_study.latest[name].weights)
        for key in ("name", "expected_return", "volatility", "sharpe"):
            assert getattr(portfolio, key) == getattr(saved_study.latest[name], key)
    for phase_name in ("validation", "test"):
        phase, saved_phase = getattr(study, phase_name), getattr(saved_study, phase_name)
        for kind in ("equity", "metrics"):
            pd.testing.assert_frame_equal(getattr(phase, kind), getattr(saved_phase, kind))
        for kind in ("allocations", "holdings", "trades"):
            for name, frame in getattr(phase, kind).items():
                pd.testing.assert_frame_equal(frame, getattr(saved_phase, kind)[name])
        assert phase.settings == saved_phase.settings and phase.warnings == saved_phase.warnings


def test_result_tables_keep_backend_order_and_show_net_mean_separately_from_cagr(selection_inputs):
    study = selection_inputs[0]
    document = selection_report_html(*selection_inputs)
    for phase_name in ("validation", "test"):
        section = document.split(f'id="{phase_name}"', 1)[1].split("</section>", 1)[0]
        table = section[section.index("<table"):section.index("</table>")]
        positions = [table.index(f"<th>{html.escape(name)}</th>") for name in getattr(study, phase_name).equity]
        assert positions == sorted(positions)
        for key in ("annual_mean", "cagr", "volatility", "max_drawdown"):
            expected = getattr(study, phase_name).metrics.loc[study.selected, key]
            assert f"{expected:.2%}" in table
    assert "Later test (not used to choose the blend)" in document
    assert re.search(r"historical.{0,100}reviewed|reviewed.{0,100}historical", document, re.I | re.S)
    assert "descriptive screen" in document.lower()
    assert study.reasons[0] in document
    assert document.index('id="test"') < document.index('id="latest-selection"')


def test_unsupported_latest_holdings_are_inside_closed_details(selection_inputs):
    document = selection_report_html(*selection_inputs)
    section = document.split('id="latest-selection"', 1)[1].split("</section>", 1)[0]
    match = re.search(r"<details\b([^>]*)>(.*?)</details>", section, re.S)
    assert match is not None and not re.search(r"\bopen\b", match[1])
    assert "Experimental latest holdings" in match[2]
    assert "AAA" in match[2] and "BBB" in match[2]
    assert "60.00%" in match[2] and "40.00%" in match[2]


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if (tag == "script" and name == "src") or (tag == "link" and name == "href"):
                self.external_resources.append(value)


def test_report_is_offline_and_includes_plotly_license_once(selection_inputs):
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs))) as archive:
        document = archive.read("report.html").decode()
        parser = ResourceParser()
        parser.feed(document)
        assert not parser.external_resources
        assert document.lower().startswith("<!doctype html>")
        assert document.count("* plotly.js v") == 1
        assert document.count("Plotly.newPlot(") >= 2
        license_line = "Permission is hereby granted, free of charge"
        assert license_line in document
        assert license_line in archive.read("THIRD_PARTY_NOTICES.txt").decode()


def test_report_escapes_source_gate_reasons_warnings_and_asset_labels(selection_inputs):
    study, _, metadata = selection_inputs
    metadata["source"] = '<script>alert("source")</script>'
    study.reasons = ['<img src=x onerror="reason()">']
    study.warnings = ['<script>alert("warning")</script>']
    unsafe_asset = '<img src=x onerror="asset()">'
    for portfolio in study.latest.values():
        portfolio.weights.rename(index={"AAA": unsafe_asset}, inplace=True)
    document = selection_report_html(*selection_inputs)
    for unsafe in (metadata["source"], study.reasons[0], study.warnings[0], unsafe_asset):
        assert unsafe not in document
        assert html.escape(unsafe).split("&quot;")[0] in document


def test_coverage_csv_escapes_formulas_and_preserves_numeric_losses(selection_inputs):
    study, _, metadata = selection_inputs
    metadata["universe_coverage"][1]["reason"] = '=HYPERLINK("https://example.invalid")'
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs))) as archive:
        coverage = pd.read_csv(archive.open("universe_coverage.csv"), index_col=0, keep_default_na=False)
        assert coverage.iloc[1]["reason"].startswith("'=HYPERLINK")
        metrics = pd.read_csv(archive.open("test/metrics.csv"), index_col=0)
        assert metrics.loc[study.selected, "max_drawdown"] == -.025
        holdings = pd.read_csv(archive.open("test/01_holdings.csv"), index_col=0)
        assert holdings.loc["BBB", "pnl_contribution"] == -.004
