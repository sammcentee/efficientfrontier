import copy
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
from efficient_frontier.selection import BENCHMARK_NAMES, SELECTED, selection_fit_estimates, selection_frontier
from efficient_frontier.selection_report import (
    LATEST_LABEL, REPLAY_LABEL, selection_replay_findings_html, selection_replay_verdict_html,
    selection_benchmark_note, selection_chart, selection_fit_summary_html, selection_frontier_chart,
    selection_frontier_summary_html,
    selection_report_html, selection_report_zip,
)


@pytest.fixture
def selection_inputs():
    dates = pd.bdate_range("2024-01-02", periods=12)
    prices = pd.DataFrame({"AAA": np.linspace(100, 112, 12),
                           "BBB": np.linspace(80, 85, 12)}, index=dates)
    benchmarks = pd.DataFrame({"SPY": np.linspace(400, 430, 12),
                               "QQQ": np.linspace(350, 380, 12)}, index=dates)
    names = ["Historical-mean portfolio", "Equal-weight stocks", "SPY", "QQQ"]

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
        allocations = pd.DataFrame({"AAA": [.6], "BBB": [.4], "CASH": [0.]}, index=phase_dates[[1]])
        holdings = pd.DataFrame({
            "average_weight": [.45, .25, .3], "end_weight": [.62, .28, .1],
            "selection_frequency": [1., 1., 1.], "max_target_weight": [.6, .3, .1],
            "max_realized_weight": [.62, .31, 1.], "pnl_contribution": [.025, -.004, 0.],
        }, index=["AAA", "BBB", "CASH"])
        trades = pd.DataFrame({"train_end": [phase_dates[0]], "turnover": [1.],
                               "cost": [.001234567], "cash_weight": [0.]},
                              index=phase_dates[[1]])
        return BacktestStudy(
            equity=equity, metrics=metrics, allocations={name: allocations.copy() for name in names},
            holdings={name: holdings.copy() for name in names}, trades={name: trades.copy() for name in names},
            warnings=[f"Phase starts at {phase_dates[0].date()}."],
            settings={"cost_bps": 10., "periods_per_year": 252, "target_volatility": .15},
        )

    study = SimpleNamespace(
        test=phase(8, .01), selected=names[0],
        latest={
            names[0]: Portfolio(names[0], pd.Series({"AAA": .6, "BBB": .4}), .123456789, .14, .738976),
        },
        supported=False, reasons=["The later test did not clear the descriptive screen."],
        settings={"target_volatility": .15, "max_weight": .6, "cost_bps": 10., "periods_per_year": 252,
                  "model_revision": "single_mean_v1", "train_fraction": .7, "shrinkage": .1,
                  "latest_as_of": str(dates[-1].date()), "risk_free_rate": .02,
                  "initial_training_start": str(dates[1].date()), "initial_training_end": str(dates[8].date()),
                  "test_start": str(dates[9].date()), "holding_policy": "buy_and_hold"},
        warnings=["Current constituents can introduce survivorship bias."], benchmark_prices=benchmarks,
        eligibility=pd.DataFrame({"fit_date": [str(dates[-1].date())] * 2, "symbol": ["AAA", "BBB"],
                                  "eligible": [True, True], "reason": ["", ""]}),
    )
    metadata = {
        "source": "Synthetic adjusted prices", "universe": "Fixture stocks", "revision": "fixture-123",
        "universe_coverage": [{"ticker": "AAA", "status": "included", "reason": ""},
                              {"ticker": "OLD", "status": "excluded", "reason": "Missing earlier prices"}],
    }
    return study, prices, metadata


def test_chart_identifies_every_path_without_changing_returns(selection_inputs):
    study, _, _ = selection_inputs
    figure = selection_chart(study.test)
    assert [trace.name for trace in figure.data] == [
        REPLAY_LABEL if name == study.selected else name for name in study.test.equity
    ]
    assert all(trace.showlegend for trace in figure.data)
    assert figure.data[0].line.color != figure.data[1].line.color
    assert figure.data[1].line.dash == "dash"
    for name, trace in zip(study.test.equity.columns, figure.data):
        np.testing.assert_allclose(trace.y, study.test.equity[name] * 10_000)


def test_zip_preserves_every_test_strategy_and_numeric_result(selection_inputs):
    study, prices, metadata = selection_inputs
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs))) as archive:
        details = json.loads(archive.read("metadata.json"))
        assert len(details["result_files"]) == len(study.test.equity.columns)
        for kind in ("metrics", "equity"):
            recovered = pd.read_csv(archive.open(f"test/{kind}.csv"), index_col=0,
                                    parse_dates=kind == "equity")
            pd.testing.assert_frame_equal(recovered, getattr(study.test, kind), check_freq=False, check_names=False)
        assert list(details["result_files"].values()) == list(study.test.equity.columns)
        for stem, name in details["result_files"].items():
            assert re.fullmatch(r"test/\d+", stem)
            for kind in ("allocations", "holdings", "trades"):
                recovered = pd.read_csv(archive.open(f"{stem}_{kind}.csv"), index_col=0,
                                        parse_dates=[0, "train_end"] if kind == "trades" else kind != "holdings")
                pd.testing.assert_frame_equal(recovered, getattr(study.test, kind)[name],
                                              check_freq=False, check_names=False)
        for path, expected in (("prices.csv", prices), ("benchmark_prices.csv", study.benchmark_prices)):
            recovered = pd.read_csv(archive.open(path), index_col=0, parse_dates=True)
            pd.testing.assert_frame_equal(recovered, expected, check_freq=False, check_names=False)
        coverage = pd.read_csv(archive.open("universe_coverage.csv"), index_col=0, keep_default_na=False)
        pd.testing.assert_frame_equal(coverage, pd.DataFrame(metadata["universe_coverage"]))
        eligibility = pd.read_csv(archive.open("eligibility.csv"), index_col=0, keep_default_na=False)
        pd.testing.assert_frame_equal(eligibility, study.eligibility)
        fits = pd.read_csv(archive.open("fit_estimates.csv"), index_col=0, parse_dates=[0, "execution_date"])
        pd.testing.assert_frame_equal(fits, selection_fit_estimates(study, prices))


def test_comparison_precedes_frontier_and_keeps_latest_fit_separate_from_replay(selection_inputs):
    study, prices, _ = selection_inputs
    document = selection_report_html(*selection_inputs)
    section = document.split('id="fit-and-replay"', 1)[1].split("</section>", 1)[0]
    assert document.index('id="fit-and-replay"') < document.index('id="latest-frontier"') < document.index('id="test"')
    assert "<details" not in section
    assert "The latest fit includes the replay dates" in section
    assert "The replay did not use the latest holdings" in section
    assert "2024-01-03 to 2024-01-17" in section
    assert "2024-01-15 to 2024-01-17" in section
    assert "2024-01-12" not in section  # NAV baseline, not an evaluated return date.
    for value in (study.latest[study.selected].expected_return, study.latest[study.selected].volatility,
                  study.test.metrics.loc[study.selected, "annual_mean"], study.test.metrics.loc[study.selected, "volatility"]):
        assert f"{value:.2%}" in section
    assert "before fees" in section and "after fees" in section
    assert document.index('id="test"') < document.index('<details id="fit-estimates"')
    section = document.split('id="fit-estimates"', 1)[1].split("</details>", 1)[0]
    fits = selection_fit_estimates(study, prices)
    for date, fit in fits.iterrows():
        assert f"{fit.phase} · {date.date()}" in section
        for column in ("fitted_mean", "model_volatility", "historical_volatility"):
            assert f"{fit[column]:.2%}" in section
    assert "Fitted annual mean" in section and "Historical volatility" in section
    assert "The budget applies to model volatility" in section
    assert "latest weights did not produce the earlier test results" in section
    assert "not checks of optimizer accuracy" in document
    assert "Estimates for every saved fit" in document
    assert "2024-01-12" in section and "2024-01-17" in section


def test_benchmark_and_fit_explanations_use_the_actual_model_settings(selection_inputs):
    study, prices, _ = selection_inputs
    study.settings.update(shrinkage=.2)
    note = selection_benchmark_note(study)
    assert "at entry, then lets the weights drift" in note
    assert "All four paths buy once and hold" in note
    assert "SPY tracks the S&P 500" in note and "QQQ tracks the Nasdaq-100" in note
    assert "buy once and hold" in note
    assert "None is scaled to the model volatility budget" in note
    summary = selection_fit_summary_html(study, prices)
    assert "20% covariance adjustment" in summary
    assert "does not maximize the return-to-volatility ratio or CAGR" in summary


@pytest.mark.parametrize("mean,volatility,dominates", [
    (.3347, .1852, True),  # Screenshot: higher mean and lower risk.
    (.2827, .18, True),    # Equal mean, lower risk.
    (.30, .1922, True),    # Higher mean, equal risk.
    (.2827, .1922, False), # Identical outcome.
    (.26, .12, False),    # Lower risk with lower mean is a tradeoff.
    (.34, .22, False),    # Higher mean with higher risk is a tradeoff.
])
def test_replay_verdict_uses_mean_volatility_dominance_not_cagr(selection_inputs, mean, volatility, dominates):
    study = selection_inputs[0].test
    study.metrics = study.metrics.loc[["Historical-mean portfolio", "QQQ"]].copy()
    study.metrics.loc["Historical-mean portfolio", ["annual_mean", "volatility", "cagr"]] = [.2827, .1922, .3023]
    # Make CAGR point the opposite way to ensure it does not control this verdict.
    study.metrics.loc["QQQ", ["annual_mean", "volatility", "cagr"]] = [mean, volatility, .10 if dominates else .50]
    verdict = selection_replay_verdict_html(study)
    assert bool(verdict) == dominates
    if dominates:
        assert "QQQ dominated the stock rule" in verdict
        assert f"{mean:.2%} versus 28.27%" in verdict
        assert f"{volatility:.2%} versus 19.22%" in verdict


@pytest.mark.parametrize("total_return,volatility,return_finding,risk_finding", [
    (.6, .18, "20.00 percentage points higher", "2.00 percentage points lower"),
    (.2, .24, "20.00 percentage points lower", "4.00 percentage points higher"),
    (.4, .20, "the same", "the same"),
    (-.1, .20, "50.00 percentage points lower", "the same"),
    (.400001, .199999, "less than 0.01 percentage point higher", "less than 0.01 percentage point lower"),
])
def test_replay_findings_compare_total_return_and_risk_without_annualizing_again(
        selection_inputs, total_return, volatility, return_finding, risk_finding):
    study = selection_inputs[0].test
    study.metrics = study.metrics.rename(index=BENCHMARK_NAMES)
    study.metrics.loc[SELECTED, ["total_return", "volatility", "annual_mean", "cagr"]] = [total_return, volatility, -.9, -.8]
    study.metrics.loc[BENCHMARK_NAMES["SPY"], ["total_return", "volatility", "annual_mean", "cagr"]] = [.4, .2, .9, .8]
    before = study.metrics.copy(deep=True)
    findings = selection_replay_findings_html(study)
    spy = findings.split("Versus unscaled SPY", 1)[1].split("</p>", 1)[0]
    assert f"net total return was {return_finding}" in spy
    assert f"realized annual volatility {risk_finding}" in spy
    assert "in this replay" in spy
    assert "annual mean" not in findings and "CAGR" not in findings
    pd.testing.assert_frame_equal(study.metrics, before)


@pytest.mark.parametrize("available", [(), ("SPY",), ("QQQ",), ("SPY", "QQQ")])
def test_replay_findings_compare_each_available_etf_and_name_missing_comparisons(selection_inputs, available):
    study = selection_inputs[0].test
    study.metrics = study.metrics.rename(index=BENCHMARK_NAMES)
    study.metrics = study.metrics.drop(index=[name for symbol, name in BENCHMARK_NAMES.items() if symbol not in available])
    study.metrics.loc[SELECTED, ["total_return", "volatility"]] = [.5, .20]
    for symbol in available:
        study.metrics.loc[BENCHMARK_NAMES[symbol], ["total_return", "volatility"]] = [.4, .18] if symbol == "SPY" else [.7, .24]
    findings = selection_replay_findings_html(study)
    for symbol in BENCHMARK_NAMES:
        assert (f"Versus unscaled {symbol}" in findings) == (symbol in available)
    if "SPY" in available:
        assert "net total return was 10.00 percentage points higher" in findings
        assert "realized annual volatility 2.00 percentage points higher" in findings
    if "QQQ" in available:
        assert "net total return was 20.00 percentage points lower" in findings
        assert "realized annual volatility 4.00 percentage points lower" in findings
    missing = [symbol for symbol in BENCHMARK_NAMES if symbol not in available]
    assert ("unavailable" in findings) == bool(missing)
    if missing:
        assert " and ".join(missing) + " comparison" in findings
    assert "Equal-weight" not in findings


def test_replay_verdict_escapes_reference_name(selection_inputs):
    study = selection_inputs[0].test
    unsafe = '<img src=x onerror="reference()">'
    study.metrics = study.metrics.rename(index={"QQQ": unsafe})
    verdict = selection_replay_verdict_html(study)
    assert unsafe not in verdict and html.escape(unsafe) in verdict


def test_zip_retains_model_source_test_reasons_and_exact_latest_estimates(selection_inputs):
    study, _, metadata = selection_inputs
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs))) as archive:
        details = json.loads(archive.read("metadata.json"))
        assert "policy.json" not in archive.namelist()
        assert not any(name.startswith("validation/") for name in archive.namelist())
        assert "policy" not in details and "validation" not in details
        assert details["settings"] == study.settings
        assert details["settings"]["model_revision"] == "single_mean_v1"
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
    for key in ("selected", "supported", "reasons", "settings", "warnings"):
        assert getattr(study, key) == getattr(saved_study, key)
    for name, portfolio in study.latest.items():
        pd.testing.assert_series_equal(portfolio.weights, saved_study.latest[name].weights)
        for key in ("name", "expected_return", "volatility", "sharpe"):
            assert getattr(portfolio, key) == getattr(saved_study.latest[name], key)
    for kind in ("equity", "metrics"):
        pd.testing.assert_frame_equal(getattr(study.test, kind), getattr(saved_study.test, kind))
    for kind in ("allocations", "holdings", "trades"):
        for name, frame in getattr(study.test, kind).items():
            pd.testing.assert_frame_equal(frame, getattr(saved_study.test, kind)[name])
    assert study.test.settings == saved_study.test.settings and study.test.warnings == saved_study.test.warnings


def test_result_tables_keep_backend_order_and_show_net_mean_separately_from_cagr(selection_inputs):
    study = selection_inputs[0]
    document = selection_report_html(*selection_inputs)
    section = document.split('id="test"', 1)[1].split("</section>", 1)[0]
    table = section[section.index("<table"):section.index("</table>")]
    positions = [table.index(f"<th>{html.escape(REPLAY_LABEL if name == study.selected else name)}</th>")
                 for name in study.test.equity]
    assert positions == sorted(positions)
    for key in ("annual_mean", "cagr", "volatility", "max_drawdown"):
        expected = study.test.metrics.loc[study.selected, key]
        assert f"{expected:.2%}" in table
    assert "Historical replay · buy and hold" in document
    assert "no later rebalances" in document
    assert "refitted every" not in document
    assert re.search(r"historical.{0,100}reviewed|reviewed.{0,100}historical", document, re.I | re.S)
    assert "descriptive checks" in document.lower()
    assert study.reasons[0] in document
    assert document.index('id="latest-selection"') < document.index('id="test"')
    report_body = document.split("</head>", 1)[1].lower()
    for old_choice in ("blend", "risk-matched", 'id="validation"', "policy.json"):
        assert old_choice not in report_body


@pytest.mark.parametrize("supported", [False, True])
def test_latest_holdings_stay_visible_before_later_checks(selection_inputs, supported):
    selection_inputs[0].supported = supported
    document = selection_report_html(*selection_inputs)
    section = document.split('id="latest-selection"', 1)[1].split("</section>", 1)[0]
    assert "<details" not in section
    assert "AAA" in section and "BBB" in section
    assert "60.00%" in section and "40.00%" in section
    assert document.index('id="latest-frontier"') < document.index('id="latest-selection"') < document.index('id="selection-decision"')


def test_report_discloses_actual_test_assumptions(selection_inputs):
    study, _, _ = selection_inputs
    study.settings.update(train_fraction=.6, cost_bps=25., shrinkage=.2, periods_per_year=52.)
    document = selection_report_html(*selection_inputs)
    section = document.split('id="selection-assumptions"', 1)[1].split("</section>", 1)[0]
    for value in ("Initial training share: 60%", "Holding policy: buy once and hold",
                  "25 basis points (0.25%)", "Covariance shrinkage: 20%", "52 observations per year"):
        assert value in section


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
        assert document.count('class="plotly-graph-div"') == 2
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


def test_latest_frontier_chart_uses_fitted_means_and_matching_volatilities(selection_inputs):
    study, prices, _ = selection_inputs
    study.settings["shrinkage"] = 0.0
    data = selection_frontier(study, prices, frontier_points=5)
    frontier, _, comparisons, _ = data
    figure = selection_frontier_chart(data, study)
    np.testing.assert_allclose(figure.data[0].x, frontier.model_volatility)
    np.testing.assert_allclose(figure.data[0].y, frontier.fitted_mean)
    for trace, (name, row) in zip(figure.data[1:], comparisons.iterrows()):
        np.testing.assert_allclose(trace.x, [row.historical_volatility])
        np.testing.assert_allclose(trace.y, [row.fitted_mean])
        assert trace.marker.symbol == ("circle" if name == study.selected else "diamond")
        assert "Annual arithmetic mean" in trace.hovertemplate
    assert figure.layout.xaxis.title.text == "Annual volatility"
    assert figure.layout.yaxis.title.text == "Annual arithmetic mean"
    assert figure.layout.shapes[0].x0 == study.settings["target_volatility"]
    summary = selection_frontier_summary_html(study, prices, data)
    assert "same return dates" in summary and "ordinary sample covariance" in summary
    assert "before trading fees" in summary and "in-sample optimum" in summary
    assert LATEST_LABEL in summary
    assert "not portfolio candidates" in summary
    assert "None of the ETF positions is scaled" in summary
    assert "Model volatility" not in summary
    assert "2024-01-03 to 2024-01-17" in summary
    for row in comparisons.itertuples():
        assert f"{row.fitted_mean:.2%}" in summary and f"{row.historical_volatility:.2%}" in summary


def test_frontier_export_reuses_cached_points_and_preserves_every_weight(selection_inputs, monkeypatch):
    study, prices, _ = selection_inputs
    data = selection_frontier(study, prices, frontier_points=5)
    saved = copy.deepcopy(data)

    def unexpected_refit(*args, **kwargs):
        raise AssertionError("Cached report export must not solve the frontier again")

    monkeypatch.setattr("efficient_frontier.selection_report.selection_frontier", unexpected_refit)
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(*selection_inputs, frontier_data=data))) as archive:
        for filename, frame in zip(("latest/frontier.csv", "latest/frontier_weights.csv", "latest/fitted_comparison.csv"), data[:3]):
            recovered = pd.read_csv(archive.open(filename), index_col=0)
            pd.testing.assert_frame_equal(recovered, frame)
        document = archive.read("report.html").decode()
        assert document.index('id="latest-frontier"') < document.index('id="test"') < document.index('id="selection-decision"')
        assert "Stock model volatility uses 10% covariance shrinkage" in document
    for actual, expected in zip(data[:3], saved[:3]):
        pd.testing.assert_frame_equal(actual, expected)


def test_frontier_summary_escapes_reference_names_and_solver_warnings(selection_inputs):
    study, prices, _ = selection_inputs
    data = list(selection_frontier(study, prices, frontier_points=3))
    unsafe = '<img src=x onerror="reference()">'
    data[2] = data[2].rename(index={"S&P 500 (SPY)": unsafe})
    data[3] = ['<script>solver()</script>']
    document = selection_frontier_summary_html(study, prices, data)
    assert unsafe not in document and "&lt;img src=x" in document
    assert "<script>solver()</script>" not in document and "&lt;script&gt;solver()&lt;/script&gt;" in document
    assert selection_frontier_chart(data, study).data[2].name == html.escape(unsafe)
