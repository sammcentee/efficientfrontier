from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.backtest_report import backtest_chart, backtest_html, findings, findings_markdown


@pytest.fixture
def study():
    names = [f"{method} · {policy}"
             for method in ("Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window")
             for policy in ("Minimum volatility", "Maximum Sharpe", "Equal weight")]
    metrics = pd.DataFrame({
        "total_return": np.arange(2, 14) / 100,
        "cagr": [0.15] * 12, "volatility": [0.1] * 12, "sharpe": [1.3] * 12,
        "sortino": [1.75] * 12, "calmar": [1.5] * 12,
        "max_drawdown": [-0.1] * 12, "total_turnover": [2.75] * 12,
        "total_cost": [0.02] * 12, "rebalance_count": [3] * 12, "fallback_count": [0] * 12,
    }, index=names)
    metrics.loc["Rolling window · Minimum volatility", "total_return"] = 0.25
    metrics.loc["Buy and hold · Equal weight", "max_drawdown"] = -0.02
    dates = pd.bdate_range("2024-01-02", periods=4)
    equity = pd.DataFrame({name: [1, 0.99, 0.98, 1 + metrics.loc[name, "total_return"]] for name in names}, index=dates)
    holdings = pd.DataFrame({
        "average_weight": [0.45, 0.25, 0.05], "end_weight": [0.65, 0.25, 0.1],
        "selection_frequency": [1.0, 1.0, 0.5], "max_target_weight": [0.7, 0.35, 0.15],
        "max_realized_weight": [0.72, 0.4, 0.2],
        "pnl_contribution": [0.24, -0.01, 0.04],
    }, index=["AAA", "BBB", "CCC"])
    return SimpleNamespace(
        equity=equity, metrics=metrics, holdings={name: holdings.copy() for name in names},
        allocations={}, trades={}, settings={"cost_bps": 10, "rebalance_every": 21, "rolling_window": 252},
        warnings=[],
    )


def test_findings_use_matching_equal_weight_and_signed_contributions(study):
    notes = findings(study)
    assert "Rolling window · Minimum volatility, +25.00%" in notes[0]
    assert "12.00 percentage points above Rolling window · Equal weight" in notes[0]
    assert "Buy and hold · Equal weight, -2.00%" in notes[1]
    assert "AAA, +24.00 percentage points" in notes[2]
    assert "BBB, -1.00 percentage points" in notes[3]
    assert "gross turnover 2.75×" in notes[4]
    assert "2.00% of initial capital (200.00 per 10,000" in notes[4]
    assert "AAA, 72.00% of portfolio value; its largest target was 70.00%" in notes[5]
    best = study.metrics.total_return.idxmax()
    assert study.holdings[best].pnl_contribution.sum() - study.metrics.loc[best, "total_cost"] == pytest.approx(0.25)


@pytest.mark.parametrize("contributions,absent,present", [
    ([-0.02, -0.01, -0.03], "No holding made a positive", "Largest holding detractor"),
    ([0.02, 0.01, 0.03], "No holding made a negative", "Largest positive holding contribution"),
    ([0.0, 0.0, 0.0], "No holding made a positive", "No holding made a negative"),
])
def test_findings_do_not_mislabel_positive_and_negative_assets(study, contributions, absent, present):
    best = study.metrics.total_return.idxmax()
    study.holdings[best]["pnl_contribution"] = contributions
    notes = "\n".join(findings(study))
    assert absent in notes and present in notes


def test_equal_weight_winner_and_fallbacks_are_explained(study):
    study.metrics.loc["Buy and hold · Equal weight", "total_return"] = 0.5
    study.metrics.loc["Rolling window · Maximum Sharpe", "fallback_count"] = 2
    notes = "\n".join(findings(study))
    assert "equal-weight comparison itself" in notes
    assert "fallback occurred 2 times" in notes


def test_solver_dust_is_not_a_named_detractor_or_a_displayed_holding(study):
    best = study.metrics.total_return.idxmax()
    study.holdings[best].loc["BBB", "pnl_contribution"] = -1e-9
    study.holdings[best].loc["BBB", "max_target_weight"] = 1e-8
    notes = "\n".join(findings(study))
    assert "No holding made a negative P&L contribution at the displayed precision" in notes
    assert "Largest holding detractor" not in notes
    assert "-0.00 percentage points" not in notes
    assert "| BBB |" not in findings_markdown(study, "Synthetic demo")
    assert "<th>BBB</th>" not in backtest_html(study)
    pd.testing.assert_series_equal(study.holdings[best].loc["BBB"], pd.Series({
        "average_weight": 0.25, "end_weight": 0.25, "selection_frequency": 1.0,
        "max_target_weight": 1e-8, "max_realized_weight": 0.4, "pnl_contribution": -1e-9,
    }, name="BBB"))  # Presentation does not remove assets from exported study data.


def test_charts_preserve_all_paths_and_initial_drawdown(study):
    comparison = backtest_chart(study)
    drawdown = backtest_chart(study, drawdown=True)
    assert len(comparison.data) == len(drawdown.data) == 12
    np.testing.assert_allclose(comparison.data[0].y, study.equity.iloc[:, 0] * 10_000)
    np.testing.assert_allclose(drawdown.data[0].y, [0.0, -0.01, -0.02, 0.0])
    assert comparison.data[0].line.dash == "solid"
    assert comparison.data[3].line.dash == "dash"
    assert comparison.data[6].line.dash == "dot"
    assert comparison.data[9].line.dash == "dashdot"
    assert len(comparison.layout.title.text) <= 22
    assert comparison.layout.legend.maxheight <= comparison.layout.margin.b


def test_chart_filter_preserves_full_study_and_requested_order(study):
    original = study.equity.copy()
    selected = [study.equity.columns[-1], study.equity.columns[0]]
    chart = backtest_chart(study, strategies=selected)
    assert [trace.name for trace in chart.data] == selected
    np.testing.assert_allclose(chart.data[0].y, original[selected[0]] * 10_000)
    drawdown = backtest_chart(study, drawdown=True, strategies=selected)
    np.testing.assert_allclose(drawdown.data[0].y, original[selected[0]] / original[selected[0]].cummax() - 1)
    assert len(backtest_chart(study, strategies=[]).data) == 0
    pd.testing.assert_frame_equal(study.equity, original)
    assert len(backtest_chart(study).data) == len(original.columns)


def test_chart_filter_rejects_unknown_strategy(study):
    with pytest.raises(ValueError, match="Unknown backtest strategies: Missing portfolio"):
        backtest_chart(study, strategies=["Missing portfolio"])


def test_html_escapes_asset_names_settings_and_warnings_without_bundle(study):
    best = study.metrics.total_return.idxmax()
    study.holdings[best].rename(index={"AAA": '<img src=x onerror="asset()">'}, inplace=True)
    study.settings["<setting>"] = '<script>alert("setting")</script>'
    study.warnings = ['<script>alert("warning")</script>']
    document = backtest_html(study)
    assert '<img src=x onerror="asset()">' not in document
    assert "&lt;img src=x onerror=" in document
    assert "&lt;setting&gt;" in document
    assert '&lt;script&gt;alert(&quot;setting&quot;)&lt;/script&gt;' in document
    assert '&lt;script&gt;alert(&quot;warning&quot;)&lt;/script&gt;' in document
    assert "* plotly.js v" not in document
    assert "<!doctype html>" not in document
    for strategy in study.metrics.index:
        assert strategy in document
    assert document.count("<td>3</td>") == 12  # Allocation event counts are not percentages.
    assert "Allocation events (incl. entry)" in document
    assert document.count("<td>1.75</td>") == 12
    assert document.count("<td>1.50</td>") == 12


@pytest.mark.parametrize("periods", [12.0, 52, 365])
def test_backtest_reports_use_configured_annualization(study, periods):
    study.settings["periods_per_year"] = periods
    for document in (backtest_html(study), findings_markdown(study, "Synthetic demo")):
        assert f"{periods:g} observations per year" in document
        assert "252 observations per year" not in document
        assert "Sortino" in document and "Calmar" in document


def test_markdown_has_all_strategies_dates_source_settings_and_limits(study):
    study.holdings[study.metrics.total_return.idxmax()].rename(index={"AAA": "A|<script>"}, inplace=True)
    document = findings_markdown(study, "Synthetic <demo>|source")
    assert "Synthetic &lt;demo&gt;\\|source" in document
    assert "A\\|&lt;script&gt;" in document
    assert "Portfolio baseline: 2024-01-02" in document
    assert "2024-01-03 to 2024-01-05 (3 observations)" in document
    assert "cost\\_bps: 10" in document
    assert "survivorship bias" in document
    assert "selected after evaluation" in document
    assert "not a forecast" in document
    assert "terminal liquidation is excluded" in document
    assert "equal-weight baselines coincide" in document
    assert "fee total excludes foregone compounding" in document
    for strategy in study.metrics.index:
        assert f"| {strategy} |" in document
