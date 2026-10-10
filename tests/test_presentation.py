import csv
import html
import io
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.presentation import csv_text, frontier_chart, holdout_chart, latest_profile_chart, report_html, report_zip, weights_frame
from efficient_frontier.data import load_csv


@pytest.fixture
def report_analysis():
    dates = pd.bdate_range("2024-01-02", periods=6)
    returns = pd.DataFrame({"AAA": [0.01] * 6, "BBB": [0.005] * 6}, index=dates)
    portfolio = SimpleNamespace(weights=pd.Series({"AAA": 0.6, "BBB": 0.4}), expected_return=0.1, volatility=0.2, sharpe=0.3)
    return SimpleNamespace(
        portfolios={"Minimum volatility": portfolio},
        frontier=pd.DataFrame({"expected_return": [0.1, 0.15], "volatility": [0.2, 0.3], "sharpe": [0.3, 0.4]}),
        frontier_weights=pd.DataFrame({"AAA": [0.6, 0.7], "BBB": [0.4, 0.3]}),
        train_returns=returns.iloc[:4], test_returns=returns.iloc[4:],
        mean_returns=pd.Series({"AAA": 0.1, "BBB": 0.05}),
        covariance=pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["AAA", "BBB"], columns=["AAA", "BBB"]),
        periods_per_year=252,
        equity=pd.DataFrame({"Minimum volatility": [1.0, 1.008, 1.016064]}, index=dates[3:]),
        holdout_metrics=pd.DataFrame({"total_return": [0.016064], "cagr": [0.8], "volatility": [0.01], "sharpe": [0.3],
                                     "sortino": [1.25], "calmar": [2.5], "max_drawdown": [0.0]}, index=["Minimum volatility"]),
        warnings=['Review <script>alert("warning")</script>'],
    )


def test_report_escapes_source_parameters_and_warnings(report_analysis):
    document = report_html(report_analysis, {"source": 'Synthetic <script>alert("source")</script>', "<unsafe-key>": "<unsafe-value>"})
    assert "Synthetic demonstration data" in document
    assert '&lt;script&gt;alert(&quot;source&quot;)&lt;/script&gt;' in document
    assert '&lt;script&gt;alert(&quot;warning&quot;)&lt;/script&gt;' in document
    assert "&lt;unsafe-key&gt;" in document and "&lt;unsafe-value&gt;" in document
    assert 'alert("source")' not in document
    assert document.count("* plotly.js v") == 1
    assert "2024-01-02" in document and "2024-01-09" in document
    assert "There is no subsequent trading between assets" in document


def test_zip_contains_reproducible_results_and_assumptions(report_analysis):
    prices = pd.DataFrame({"AAA": [100.0, 101.0], "BBB": [100.0, 100.5]}, index=pd.bdate_range("2024-01-01", periods=2))
    archive = report_zip(report_analysis, prices, {"source": "Synthetic demo", "max_weight": 0.7})
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert set(bundle.namelist()) == {"report.html", "weights.csv", "frontier.csv", "frontier_weights.csv", "holdout_metrics.csv", "holdout_curve.csv", "prices.csv", "metadata.json", "THIRD_PARTY_NOTICES.txt", "risk_summary.csv", "risk_contributions.csv"}
        metadata = json.loads(bundle.read("metadata.json"))
        assert metadata["training_observations"] == 4
        assert metadata["holdout_observations"] == 2
        assert metadata["assets"] == ["AAA", "BBB"]
        assert metadata["annualization_days"] == 252
        assert metadata["periods_per_year"] == 252
        recovered = pd.read_csv(bundle.open("weights.csv"), index_col=0)
        pd.testing.assert_frame_equal(recovered, weights_frame(report_analysis))
        equity = pd.read_csv(bundle.open("holdout_curve.csv"), index_col=0)
        assert equity.iloc[0, 0] == 1.0
        assert bundle.read("report.html").startswith(b"<!doctype html>")


def test_report_data_table_shows_plain_labels_and_percents_and_metadata_stays_raw(report_analysis):
    prices = pd.DataFrame({"AAA": [100.0, 101.0], "BBB": [100.0, 100.5]}, index=pd.bdate_range("2024-01-01", periods=2))
    settings = {"source": "Synthetic demo", "compare_market": True, "train_fraction": 0.7, "risk_free_rate": 0.02,
                "max_weight": 1.0, "shrinkage": 0.1}
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, prices, settings))) as bundle:
        data = bundle.read("report.html").decode().split('<section id="data">')[1].split("</table>")[0]
        for row in ("<th>Market comparison</th><td>Yes</td>", "<th>Prices for the first fit</th><td>70%</td>",
                    "<th>Risk-free rate</th><td>2.00%</td>", "<th>Largest holding allowed</th><td>No limit</td>",
                    "<th>Covariance shrinkage</th><td>10%</td>", "<th>Observations per year</th><td>252</td>",
                    "<th>Assets</th><td>AAA, BBB</td>"):
            assert row in data
        assert "_" not in data and "Train fraction" not in data and "0.7<" not in data
        metadata = json.loads(bundle.read("metadata.json"))
        assert {key: metadata[key] for key in settings} == settings


def test_standalone_report_and_archive_include_complete_plotly_license(report_analysis):
    license_text = (Path(__file__).resolve().parents[1] / "efficient_frontier" / "third_party" / "plotly.js.LICENSE.txt").read_text(encoding="utf-8")
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED" in license_text
    document = report_html(report_analysis, {"source": "Synthetic demo"})
    assert f"<pre>{html.escape(license_text)}</pre>" in document
    assert 'the &quot;Software&quot;' in document
    archive = report_zip(report_analysis, pd.DataFrame(), {"source": "Synthetic demo"})
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert license_text in bundle.read("THIRD_PARTY_NOTICES.txt").decode("utf-8")
        assert html.escape(license_text) in bundle.read("report.html").decode("utf-8")


def test_charts_preserve_training_estimates_and_holdout_baseline(report_analysis):
    frontier = frontier_chart(report_analysis)
    np.testing.assert_array_equal(frontier.data[0].x, [0.2, 0.3])
    np.testing.assert_array_equal(frontier.data[0].y, [0.1, 0.15])
    assert len(frontier.data) == 2  # Maximum Sharpe can be absent.
    holdout = holdout_chart(report_analysis)
    np.testing.assert_allclose(holdout.data[0].y, [10_000, 10_080, 10_160.64])
    for chart in (frontier, holdout):
        assert len(chart.layout.title.text) <= 22
        assert chart.layout.title.font.size <= 20
        assert chart.layout.legend.maxheight <= chart.layout.margin.b


def test_csv_labels_cannot_be_spreadsheet_formulas_and_numbers_are_unchanged():
    labels = ['=HYPERLINK("https://example.invalid","example")', "\t+1+1", " -1+1", "\r@SUM(A1)", "BRK-B"]
    values = np.arange(25, dtype=float).reshape(5, 5) / 4 - 1
    frame = pd.DataFrame(values, index=pd.Index(labels, name="=index"), columns=labels)
    frame.columns.name = "@columns"
    original = frame.copy()
    rows = list(csv.reader(io.StringIO(csv_text(frame))))
    escaped = ["'" + label for label in labels[:4]] + [labels[4]]
    assert rows[0] == ["'=index", *escaped]
    assert [row[0] for row in rows[1:]] == escaped
    assert not any(label.lstrip().startswith(("=", "+", "-", "@"))
                   for label in rows[0] + [row[0] for row in rows[1:]])
    recovered = pd.read_csv(io.StringIO(csv_text(frame)), index_col=0)
    np.testing.assert_array_equal(recovered.to_numpy(), values)
    pd.testing.assert_frame_equal(frame, original)
    assert next(csv.reader(io.StringIO(csv_text(frame, index_label=" +index"))))[0] == "' +index"


def test_csv_preserves_normal_labels_dates_and_negative_numeric_values():
    frame = pd.DataFrame({"BRK-B": [100.0, 99.5], "return": [-0.01, 0.02]},
                         index=pd.bdate_range("2024-01-01", periods=2, name="Date"))
    recovered = pd.read_csv(io.StringIO(csv_text(frame)), index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(recovered, frame, check_freq=False)


def test_csv_escapes_formula_text_cells_without_changing_numeric_cells():
    frame = pd.DataFrame({"label": ["=1+1", "normal"], "return": [-.5, .1]})
    recovered = pd.read_csv(io.StringIO(csv_text(frame)), index_col=0)
    assert recovered.label.tolist() == ["'=1+1", "normal"]
    assert recovered["return"].tolist() == [-.5, .1]


def test_csv_escapes_each_multiindex_level_without_changing_values():
    frame = pd.DataFrame([[.25, -.5], [1.5, 2.0]],
                         index=pd.MultiIndex.from_tuples([("=strategy", "@benchmark"), ("normal", "safe")], names=["strategy", "benchmark"]),
                         columns=pd.MultiIndex.from_tuples([("+strategy", "-benchmark"), ("normal", "safe")], names=["strategy", "benchmark"]))
    original = frame.copy()
    recovered = pd.read_csv(io.StringIO(csv_text(frame)), header=[0, 1], index_col=[0, 1])
    assert recovered.index[0] == ("'=strategy", "'@benchmark")
    assert recovered.columns[0] == ("'+strategy", "'-benchmark")
    np.testing.assert_array_equal(recovered, original)
    pd.testing.assert_frame_equal(frame, original)


def test_report_zip_escapes_formula_asset_labels(report_analysis):
    label = "=1+1"
    report_analysis.portfolios["Minimum volatility"].weights.index = [label, "BBB"]
    report_analysis.frontier_weights.columns = [label, "BBB"]
    report_analysis.mean_returns.index = [label, "BBB"]
    report_analysis.covariance.index = report_analysis.covariance.columns = [label, "BBB"]
    prices = pd.DataFrame({label: [100.0, 101.0], "BBB": [100.0, 99.0]})
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, prices, {"source": "Synthetic demo"}))) as bundle:
        weights = list(csv.reader(io.StringIO(bundle.read("weights.csv").decode())))
        assert weights[1][0] == "'=1+1"
        contributions = list(csv.reader(io.StringIO(bundle.read("risk_contributions.csv").decode())))
        assert contributions[1][0] == "'=1+1"
        for name in ("prices.csv", "frontier_weights.csv"):
            header = next(csv.reader(io.StringIO(bundle.read(name).decode())))
            assert header[1:] == ["'=1+1", "BBB"]


def test_price_export_has_date_header_and_can_be_uploaded(report_analysis):
    prices = pd.DataFrame({"AAA": np.arange(6) + 100.0, "BBB": np.arange(6) + 200.0},
                          index=pd.bdate_range("2024-01-01", periods=6))
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, prices, {}))) as bundle:
        recovered = load_csv(io.BytesIO(bundle.read("prices.csv")))
    pd.testing.assert_frame_equal(recovered, prices.rename_axis("Date"), check_freq=False)
    assert prices.index.name is None


def test_universe_coverage_exports_all_members_and_escapes_provider_text(report_analysis):
    coverage = [
        {"symbol": "AAA", "company": "Example", "status": "included", "observations": 1260},
        {"symbol": "BBB", "company": '<script>alert("provider")</script>',
         "status": "excluded", "observations": 8, "reason": "=untrusted text"},
    ]
    metadata = {"universe_coverage": coverage, "universe_members": 2, "universe_included": 1}
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, pd.DataFrame(), metadata))) as bundle:
        frame = pd.read_csv(bundle.open("universe_coverage.csv"), index_col=0)
        assert frame.symbol.tolist() == ["AAA", "BBB"]
        assert frame.status.tolist() == ["included", "excluded"]
        assert frame.observations.tolist() == [1260, 8]
        assert frame.loc[1, "reason"] == "'=untrusted text"
        assert json.loads(bundle.read("metadata.json"))["universe_coverage"] == coverage
        document = bundle.read("report.html").decode()
        assert '<summary>Universe coverage</summary>' in document
        assert '&lt;script&gt;alert("provider")&lt;/script&gt;' in document
        assert '<script>alert("provider")</script>' not in document
        assert '<th>Universe coverage</th>' not in document


def test_report_uses_configured_annualization_and_preserves_risk_numbers(report_analysis):
    report_analysis.periods_per_year = 12.0
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, pd.DataFrame(), {"annualization_days": 252}))) as bundle:
        metadata = json.loads(bundle.read("metadata.json"))
        document = bundle.read("report.html").decode()
        assert metadata["periods_per_year"] == metadata["annualization_days"] == 12
        assert "12 observations per year" in document
        assert "252 trading days" not in document
        assert "<th>Sortino</th>" in document and "<td>1.250</td>" in document
        assert "<th>Calmar</th>" in document and "<td>2.500</td>" in document
        summary = pd.read_csv(bundle.open("risk_summary.csv"), index_col=0)
        assert summary.loc["Minimum volatility", "max_weight"] == pytest.approx(0.6)
        assert summary.loc["Minimum volatility", "effective_holdings"] == pytest.approx(1 / 0.52)
        assert summary.loc["Minimum volatility", "diversification_ratio"] == pytest.approx(0.24 / np.sqrt(0.0336))
        contributions = pd.read_csv(bundle.open("risk_contributions.csv"), index_col=0)
        np.testing.assert_allclose(contributions.iloc[:, 0], [0.5, 0.5])
        assert "Share of portfolio variance" in document
        assert "Negative shares" in document and "portfolio variance is zero" in document


def test_risk_tables_escape_labels_and_display_undefined_values(report_analysis):
    label = '<img src=x onerror="risk()">'
    report_analysis.portfolios["Minimum volatility"].weights.index = [label, "BBB"]
    report_analysis.mean_returns.index = [label, "BBB"]
    report_analysis.covariance.index = report_analysis.covariance.columns = [label, "BBB"]
    report_analysis.covariance.loc[:, :] = 0.0
    document = report_html(report_analysis, {})
    risk_section = document.split("<h2>Concentration and risk</h2>")[1].split("<h2>Realized holdout results</h2>")[0]
    assert label not in risk_section
    assert "&lt;img src=x onerror=" in risk_section
    assert risk_section.count("<td>—</td>") == 3
    assert "<td>60.00%</td>" in risk_section


@pytest.fixture
def latest_profiles():
    names = ["Low", "Medium", "Extreme"]
    summary = pd.DataFrame({
        "expected_return": [0.16, 0.14, 0.12], "volatility": [0.1, 0.15, 0.25], "sharpe": [1.4, 0.8, 0.4],
        "worst_window_return": [0.02, 0.04, 0.06], "max_weight": [0.8, 0.5, 0.8],
        "effective_holdings": [1 / 0.68, 2.0, 1 / 0.68],
    }, index=pd.Index(names, name="profile"))
    portfolios = {
        name: SimpleNamespace(weights=pd.Series({"AAA": weight, "BBB": 1 - weight}),
                              expected_return=summary.loc[name, "expected_return"],
                              volatility=summary.loc[name, "volatility"], sharpe=summary.loc[name, "sharpe"])
        for name, weight in zip(names, [0.8, 0.5, 0.2])
    }
    return SimpleNamespace(
        portfolios=portfolios, summary=summary,
        frontier=summary[["expected_return", "volatility", "sharpe", "worst_window_return"]].reset_index(drop=True),
        frontier_weights=pd.DataFrame({"AAA": [0.8, 0.5, 0.2], "BBB": [0.2, 0.5, 0.8]}),
        windows=pd.DataFrame({
            "start": ["2024-02-01", "2024-05-01", "2024-08-01"],
            "end": ["2024-04-01", "2024-07-01", "2024-10-01"], "observations": [3, 3, 3], "years": [0.25] * 3,
        }, index=pd.Index([1, 2, 3], name="window")),
        window_returns=pd.DataFrame({"Low": [0.2, 0.26, 0.02], "Medium": [0.2, 0.18, 0.04],
                                     "Extreme": [0.12, 0.18, 0.06]}, index=pd.Index([1, 2, 3], name="window")),
        as_of="2024-10-01", observations=9,
        settings={"periods_per_year": 12.0, "selection_method": "worst_window_return_frontier"}, warnings=[],
    )


def test_latest_profile_chart_uses_lowest_window_mean_not_full_history_mean(latest_profiles):
    chart = latest_profile_chart(latest_profiles)
    np.testing.assert_allclose(chart.data[0].x, [0.1, 0.15, 0.25])
    np.testing.assert_allclose(chart.data[0].y, [0.02, 0.04, 0.06])
    assert [trace.name for trace in chart.data[1:]] == ["Low", "Medium", "Highest"]
    for index, trace in enumerate(chart.data[1:]):
        assert trace.y[0] == pytest.approx([0.02, 0.04, 0.06][index])
    assert len({trace.marker.color for trace in chart.data[1:]}) == 3


def test_latest_profiles_are_prominent_separate_and_escape_report_content(report_analysis, latest_profiles):
    label = '<img src=x onerror="profile()">'
    latest_profiles.portfolios["Low"].weights.rename(index={"AAA": label}, inplace=True)
    latest_profiles.warnings = ['<script>alert("profile")</script>']
    document = report_html(report_analysis, {}, latest_profiles=latest_profiles)
    assert document.index('id="latest-profiles"') < document.index("<h2>Training estimates</h2>")
    profile_section = document.split('<section id="latest-profiles">')[1].split("</section>")[0]
    assert "Latest model holdings as of 1 Oct 2024" in profile_section
    assert "last price observation, not a live quote" in profile_section
    assert "not universal risk ratings" in profile_section
    assert "in-sample estimates, not forecasts or holdout results" in profile_section
    assert "12 observations per year" in profile_section
    assert "midpoint of this return range, not the midpoint of volatility" in profile_section
    assert label not in profile_section and "&lt;img src=x onerror=" in profile_section
    assert '&lt;script&gt;alert(&quot;profile&quot;)&lt;/script&gt;' in profile_section
    assert document.count("* plotly.js v") == 1
    assert "2024-02-01" in profile_section and "2024-10-01" in profile_section


def test_latest_profile_exports_preserve_values_and_fit_metadata(report_analysis, latest_profiles):
    archive = report_zip(report_analysis, pd.DataFrame(), {}, latest_profiles=latest_profiles)
    expected_frames = {
        "latest_profile_summary.csv": latest_profiles.summary,
        "latest_profile_weights.csv": weights_frame(latest_profiles),
        "latest_profile_windows.csv": latest_profiles.windows,
        "latest_profile_window_returns.csv": latest_profiles.window_returns,
        "latest_profile_frontier.csv": latest_profiles.frontier,
        "latest_profile_frontier_weights.csv": latest_profiles.frontier_weights,
    }
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name, frame in expected_frames.items():
            recovered = pd.read_csv(bundle.open(name), index_col=0)
            pd.testing.assert_frame_equal(recovered, frame)
        metadata = json.loads(bundle.read("metadata.json"))["latest_profiles"]
        assert metadata["as_of"] == "2024-10-01"
        assert metadata["observations"] == 9
        assert metadata["settings"] == latest_profiles.settings
        assert [window["window"] for window in metadata["windows"]] == [1, 2, 3]
        assert [window["observations"] for window in metadata["windows"]] == [3, 3, 3]
        assert "in-sample estimates" in metadata["interpretation"]


def test_latest_profile_weights_escape_formula_asset_names(report_analysis, latest_profiles):
    for portfolio in latest_profiles.portfolios.values():
        portfolio.weights.rename(index={"AAA": "=1+1"}, inplace=True)
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, pd.DataFrame(), {}, latest_profiles=latest_profiles))) as bundle:
        weights = pd.read_csv(bundle.open("latest_profile_weights.csv"), index_col=0)
        assert weights.index[0] == "'=1+1"
        np.testing.assert_allclose(weights.loc["'=1+1"], [0.8, 0.5, 0.2])


def test_report_summary_comes_first_and_names_the_selected_level(report_analysis, latest_profiles):
    document = report_html(report_analysis, {"source": "Synthetic demo", "universe": "Demo"}, latest_profiles=latest_profiles,
                           selected_profile="Extreme")
    assert document.index('id="summary"') < document.index('id="latest-profiles"')
    summary = document.split('<section id="summary">')[1].split("</section>")[0]
    assert 'The <span class="pl-nb">highest-risk</span> portfolio puts 80.0% in one asset.' in summary
    assert "Highest" in summary and "Extreme" not in summary
    assert "Market benchmarks" not in summary and "Backtesting findings" not in summary
    assert "Efficient Frontier · Research Report" not in document
    assert "<title>Portfolio Lab · Research report</title>" in document
    assert document.count("* plotly.js v") == 1


def test_report_view_is_recorded_only_when_requested_and_files_keep_extreme(report_analysis, latest_profiles):
    prices = pd.DataFrame({"AAA": [100.0, 101.0], "BBB": [100.0, 100.5]}, index=pd.bdate_range("2024-01-01", periods=2))
    plain = report_zip(report_analysis, prices, {}, latest_profiles=latest_profiles)
    chosen = report_zip(report_analysis, prices, {}, latest_profiles=latest_profiles, selected_profile="Extreme",
                        selected_method="Buy and hold")
    with zipfile.ZipFile(io.BytesIO(plain)) as bundle:
        assert "report_view" not in json.loads(bundle.read("metadata.json"))
    with zipfile.ZipFile(io.BytesIO(chosen)) as bundle:
        assert json.loads(bundle.read("metadata.json"))["report_view"] == {"risk_level": "Extreme", "rule": "Buy and hold"}
        assert list(pd.read_csv(bundle.open("latest_profile_weights.csv"), index_col=0).columns) == ["Low", "Medium", "Extreme"]
        assert "Sorted by Highest." in bundle.read("report.html").decode()
