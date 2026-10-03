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

from efficient_frontier.presentation import csv_text, frontier_chart, holdout_chart, report_html, report_zip, weights_frame


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
        equity=pd.DataFrame({"Minimum volatility": [1.0, 1.008, 1.016064]}, index=dates[3:]),
        holdout_metrics=pd.DataFrame({"total_return": [0.016064], "cagr": [0.8], "volatility": [0.01], "sharpe": [0.3], "max_drawdown": [0.0]}, index=["Minimum volatility"]),
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
        assert set(bundle.namelist()) == {"report.html", "weights.csv", "frontier.csv", "frontier_weights.csv", "holdout_metrics.csv", "holdout_curve.csv", "prices.csv", "metadata.json", "THIRD_PARTY_NOTICES.txt"}
        metadata = json.loads(bundle.read("metadata.json"))
        assert metadata["training_observations"] == 4
        assert metadata["holdout_observations"] == 2
        assert metadata["assets"] == ["AAA", "BBB"]
        assert metadata["annualization_days"] == 252
        recovered = pd.read_csv(bundle.open("weights.csv"), index_col=0)
        pd.testing.assert_frame_equal(recovered, weights_frame(report_analysis))
        equity = pd.read_csv(bundle.open("holdout_curve.csv"), index_col=0)
        assert equity.iloc[0, 0] == 1.0
        assert bundle.read("report.html").startswith(b"<!doctype html>")


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


def test_report_zip_escapes_formula_asset_labels(report_analysis):
    label = "=1+1"
    report_analysis.portfolios["Minimum volatility"].weights.index = [label, "BBB"]
    report_analysis.frontier_weights.columns = [label, "BBB"]
    prices = pd.DataFrame({label: [100.0, 101.0], "BBB": [100.0, 99.0]})
    with zipfile.ZipFile(io.BytesIO(report_zip(report_analysis, prices, {"source": "Synthetic demo"}))) as bundle:
        weights = list(csv.reader(io.StringIO(bundle.read("weights.csv").decode())))
        assert weights[1][0] == "'=1+1"
        for name in ("prices.csv", "frontier_weights.csv"):
            header = next(csv.reader(io.StringIO(bundle.read(name).decode())))
            assert header[1:] == ["'=1+1", "BBB"]
