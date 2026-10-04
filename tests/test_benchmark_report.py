import copy
import html
import io
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.backtest import run_backtests
from efficient_frontier.backtest_report import backtest_chart, backtest_html, findings_markdown
from efficient_frontier.benchmark_report import (
    benchmark_html, benchmark_markdown, default_evidence_strategy, evidence_chart, evidence_summary_frame,
)
from efficient_frontier.benchmarks import compare_benchmarks
from efficient_frontier.core import analyze
from efficient_frontier.evidence import analyze_evidence
from efficient_frontier.presentation import frontier_chart, holdout_chart, latest_profile_chart, report_html, report_zip
from efficient_frontier.profiles import build_profiles


@pytest.fixture(scope="module")
def comparison():
    rng = np.random.default_rng(21)
    dates = pd.bdate_range("2022-01-03", periods=241, name="Date")
    values = 100 * np.exp(np.cumsum(rng.normal(.0005, .012, (241, 5)), axis=0))
    prices = pd.DataFrame(values[:, :3], index=dates, columns=["A", "B", "C"])
    reference_prices = pd.DataFrame(values[:, 3:], index=dates, columns=["SPY", "QQQ"])
    analysis = analyze(prices, train_fraction=.6, frontier_points=5)
    profiles = build_profiles(prices, frontier_points=3)
    study = run_backtests(prices, train_fraction=.6, rebalance_every=40, rolling_window=80, include_profiles=True)
    benchmarks = compare_benchmarks(reference_prices, prices, analysis, study, profiles, risk_free_rate=.02)
    evidence = analyze_evidence(study.equity, benchmarks.backtest_equity)
    return SimpleNamespace(prices=prices, analysis=analysis, profiles=profiles, study=study, benchmarks=benchmarks, evidence=evidence)


def test_frontier_references_use_the_matching_estimation_objective(comparison):
    training = frontier_chart(comparison.analysis, comparison.benchmarks.training_estimates)
    latest = latest_profile_chart(comparison.profiles, comparison.benchmarks.latest_estimates)
    for position, name in enumerate(comparison.benchmarks.training_estimates.index):
        trace = training.data[-2 + position]
        assert trace.name == html.escape(name)
        assert trace.marker.symbol == "diamond"
        assert trace.y[0] == pytest.approx(comparison.benchmarks.training_estimates.loc[name, "expected_return"])
        assert latest.data[-2 + position].y[0] == pytest.approx(comparison.benchmarks.latest_estimates.loc[name, "worst_window_return"])


def test_equity_charts_keep_both_benchmarks_when_all_strategies_are_hidden(comparison):
    chart = backtest_chart(comparison.study, strategies=[], benchmark_equity=comparison.benchmarks.backtest_equity)
    assert len(chart.data) == 2
    np.testing.assert_allclose(chart.data[0].y, comparison.benchmarks.backtest_equity.iloc[:, 0] * 10_000)
    drawdown = backtest_chart(comparison.study, drawdown=True, strategies=[], benchmark_equity=comparison.benchmarks.backtest_equity)
    path = comparison.benchmarks.backtest_equity.iloc[:, 0]
    np.testing.assert_allclose(drawdown.data[0].y, path / path.cummax() - 1)
    holdout = holdout_chart(comparison.analysis, comparison.benchmarks.holdout_equity)
    np.testing.assert_allclose(holdout.data[-2].y, comparison.benchmarks.holdout_equity.iloc[:, 0] * 10_000)


def test_relative_wealth_chart_preserves_full_ratios_and_default_ignores_winner(comparison):
    evidence = copy.deepcopy(comparison.evidence)
    evidence.summary["total_return_difference"] = np.arange(len(evidence.summary)) * 100
    strategy = default_evidence_strategy(evidence)
    assert strategy == "Expanding window · Medium"
    chart = evidence_chart(evidence, strategy)
    for trace, benchmark in zip(chart.data, comparison.benchmarks.backtest_equity.columns):
        expected = comparison.study.equity[strategy] / comparison.benchmarks.backtest_equity[benchmark]
        expected /= expected.iloc[0]
        np.testing.assert_allclose(trace.y, expected)
        assert trace.y[0] == 1
    assert chart.layout.shapes[0].y0 == chart.layout.shapes[0].y1 == 1
    with pytest.raises(ValueError, match="Unknown evidence strategy"):
        evidence_chart(evidence, "Missing")
    empty = SimpleNamespace(summary=evidence.summary.iloc[:0])
    assert default_evidence_strategy(empty) is None


def test_evidence_format_keeps_ratios_pvalues_and_percentage_points_distinct(comparison):
    evidence = copy.deepcopy(comparison.evidence)
    evidence.summary["beta"] = 1.5
    evidence.summary["annual_advantage"] = .03
    evidence.summary["alpha"] = .02
    evidence.summary["advantage_pvalue_adjusted"] = .05
    evidence.summary["alpha_ci_low"] = np.nan
    display = evidence_summary_frame(evidence)
    assert display["Benchmark beta"].eq("1.500").all()
    assert display["Annual mean advantage"].eq("+3.00 pp").all()
    assert display["Annual alpha"].eq("2.00%").all()
    assert display["Mean Holm p-value"].eq("0.05").all()
    assert display["Alpha CI lower"].eq("—").all()
    assert evidence.summary.beta.eq(1.5).all()


def test_html_and_markdown_explain_matched_costs_and_statistical_limits(comparison):
    for document in (
        benchmark_html(comparison.benchmarks, comparison.evidence, backtest=True),
        benchmark_markdown(comparison.benchmarks, comparison.evidence, backtest=True),
        findings_markdown(comparison.study, "Fixture data", comparison.benchmarks, comparison.evidence),
    ):
        assert "95% HAC confidence intervals are pointwise" in document
        assert "Holm correction covers 96 tests" in document
        assert "at least 60 return observations" in document
        assert "same entry fee rate" in document
        assert "not measure the probability of future outperformance" in document
        assert "Positive alpha does not establish investment skill" in document
        assert "Inference omits the first evaluation session" in document
        assert "Expanding window · Medium" in document
        assert "2022-" in document
    assert "Both exclude fees" in benchmark_html(comparison.benchmarks)
    document = backtest_html(comparison.study, comparison.benchmarks, comparison.evidence)
    assert document.index("Market benchmarks") < document.index("Backtesting findings")
    assert "* plotly.js v" not in document


def test_benchmark_report_escapes_evidence_text(comparison):
    evidence = copy.deepcopy(comparison.evidence)
    evidence.warnings = ['<img src=x onerror="evidence()">']
    evidence.summary["status"] = "<script>status()</script>"
    document = benchmark_html(comparison.benchmarks, evidence, backtest=True)
    assert evidence.warnings[0] not in document
    assert "&lt;img src=x onerror=" in document
    assert "<script>status()</script>" not in document
    assert "&lt;script&gt;status()&lt;/script&gt;" in document


def test_report_exports_all_pairs_raw_prices_and_reproducible_curves(comparison):
    archive = report_zip(comparison.analysis, comparison.prices, {}, comparison.study,
                         comparison.profiles, comparison.benchmarks, comparison.evidence)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        expected = {
            "benchmark_prices.csv", "benchmark_holdout_metrics.csv", "benchmark_holdout_curve.csv",
            "benchmark_training_estimates.csv", "benchmark_latest_estimates.csv", "benchmark_backtest_metrics.csv",
            "benchmark_backtest_curve.csv", "evidence_summary.csv", "evidence_windows.csv", "evidence_relative_curve.csv",
        }
        assert expected <= set(bundle.namelist())
        prices = pd.read_csv(bundle.open("benchmark_prices.csv"), index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(prices, comparison.benchmarks.prices, check_freq=False)
        summary = pd.read_csv(bundle.open("evidence_summary.csv"), index_col=[0, 1], keep_default_na=False)
        np.testing.assert_allclose(summary.annual_advantage, comparison.evidence.summary.annual_advantage)
        pd.testing.assert_index_equal(summary.index, comparison.evidence.summary.index)
        relative = pd.read_csv(bundle.open("evidence_relative_curve.csv"), header=[0, 1], index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(relative, comparison.evidence.relative_equity, check_freq=False)
        windows = pd.read_csv(bundle.open("evidence_windows.csv"), index_col=0)
        pd.testing.assert_frame_equal(windows, comparison.evidence.windows)
        metadata = json.loads(bundle.read("metadata.json"))
        assert metadata["benchmarks"] == comparison.benchmarks.settings
        assert metadata["evidence"]["multiple_testing_tests"] == 96
        assert metadata["evidence"]["warnings"] == comparison.evidence.warnings
        document = bundle.read("report.html").decode()
        assert document.count("* plotly.js v") == 1
        assert "Evidence against benchmarks" in document


def test_omitted_market_comparison_is_explicit_without_fabricated_references(comparison):
    document = report_html(comparison.analysis, {
        "source": "Synthetic demo", "benchmarks_note": "No market comparison for synthetic prices.",
        "evidence_note": "No statistical inference without market benchmarks.",
    })
    assert "No market comparison for synthetic prices." in document
    assert "No statistical inference without market benchmarks." in document
    assert 'id="benchmarks"' not in document
