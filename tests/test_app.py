import base64
import html
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def offline_benchmark_downloads(monkeypatch):
    from efficient_frontier import data

    calls = []

    def unavailable(dates):
        calls.append(dates)
        raise ValueError("Benchmark downloads are unavailable in offline AppTests.")

    monkeypatch.setattr(data, "download_benchmarks", unavailable)
    return calls


@pytest.fixture
def market_prices():
    import streamlit as st

    st.cache_data.clear()
    rng = np.random.default_rng(902)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal([.0008, .0005, .0006], [.012, .009, .011], (360, 3)), axis=0)),
        index=pd.bdate_range("2021-01-04", periods=360, name="Date"),
        columns=["FUND_X", "SPY", "QQQ"],
    )


def chart_values(values):
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values)


def demo_app():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    return app.selectbox(key="source").select("Demo · synthetic").run()


def test_yahoo_is_default_and_only_submit_downloads_prices(monkeypatch):
    from efficient_frontier import data

    calls = []
    prices = data.demo_prices().iloc[:, :2].copy()
    prices.columns = ["DEFAULT-AAA", "DEFAULT-BBB"]

    def download(tickers, start, end):
        calls.append((tickers, start, end))
        return prices

    monkeypatch.setattr(data, "download_prices", download)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.selectbox(key="source").value == "Yahoo Finance"
    assert calls == []
    assert "result" not in app.session_state
    assert not app.metric
    assert not app.get("download_button")
    app.toggle(key="use_cap").set_value(True).run()
    assert calls == []
    app.number_input(key="cap").set_value(100.)
    app.text_area[0].set_value("DEFAULT-AAA, DEFAULT-BBB")
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(calls) == 1
    assert calls[0][0] == ["DEFAULT-AAA", "DEFAULT-BBB"]
    assert app.session_state.result[2]["source"] == "Yahoo Finance"
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)


def test_setting_and_backtest_explanations_are_available_before_first_run():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert "result" not in app.session_state
    labels = [item.label for item in app.expander]
    assert "Portfolio settings explained" in labels
    assert "Backtest methods explained" in labels
    explanation = "\n".join(item.value for item in app.markdown)
    for phrase in (
        "Covariance measures how asset returns move together",
        "reduces off-diagonal covariance toward zero",
        "keeps each asset's individual variance",
        "does not guarantee better results",
        "not automatic Ledoit-Wolf estimation",
        "does not add cash",
        "requires at least four assets",
        "later observations form the holdout, in date order",
        "Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window",
        "next observation's close", "One basis point is 0.01%",
        "does not change observation dates or trade dates",
    ):
        assert phrase in explanation


def test_demo_renders_and_invalid_cap_clears_old_result(offline_benchmark_downloads):
    app = demo_app()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "6"
    assert app.metric[3].value == "None"
    assert app.session_state.result[2]["max_weight"] == 1
    assert app.session_state.benchmarks is None
    assert app.session_state.evidence is None
    assert offline_benchmark_downloads == []
    assert any("synthetic" in item.value.lower() for item in app.info)
    app.toggle(key="use_cap").set_value(True).run()
    app.number_input(key="cap").set_value(5.0)
    app.button[0].click().run()
    assert not app.exception
    assert any("infeasible" in item.value.lower() for item in app.error)
    assert len(app.metric) == 0


def test_high_risk_free_rate_omits_sharpe_without_breaking_app():
    app = demo_app()
    app.number_input(key="risk_free").set_value(100.0)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any("Maximum Sharpe omitted" in item.value for item in app.warning)
    assert "Maximum Sharpe" not in app.session_state.result[0].portfolios


def test_switching_source_does_not_show_stale_demo_results():
    app = demo_app()
    app.selectbox(key="source").select("Upload CSV").run()
    assert not app.exception
    assert len(app.metric) == 0
    app.button[0].click().run()
    assert not app.exception
    assert any("Choose a CSV" in item.value for item in app.error)


def test_single_asset_renders_its_single_frontier_point(monkeypatch):
    from efficient_frontier import data
    prices = data.demo_prices().iloc[:, :1]
    monkeypatch.setattr(data, "demo_prices", lambda: prices)
    app = demo_app()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "1"
    assert len(app.session_state.result[0].frontier) == 1
    for portfolio in app.session_state.result[0].portfolios.values():
        np.testing.assert_allclose(portfolio.weights, [1.0])


def test_large_universe_chart_selection_keeps_all_assets_in_analysis(monkeypatch):
    from efficient_frontier import data
    rng = np.random.default_rng(44)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0003, .01, (150, 64)), axis=0)),
                          index=pd.bdate_range("2024-01-01", periods=150),
                          columns=[f"DEMO_{i}" for i in range(64)])
    monkeypatch.setattr(data, "demo_prices", lambda: prices)
    app = demo_app()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "64"
    assert len(app.multiselect(key="correlation_assets").value) == 20
    app.multiselect(key="correlation_assets").set_value([]).run()
    assert not app.exception
    assert len(app.session_state.result[0].portfolios["Minimum volatility"].weights) == 64


def test_backtest_controls_update_costs_and_can_disable_comparison():
    app = demo_app()
    assert not app.exception
    assert len(app.session_state.backtests.metrics) == 24
    assert app.session_state.backtests.settings["cost_bps"] == 10
    app.number_input(key="cost_bps").set_value(0.)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.backtests.metrics.total_cost.eq(0).all()
    app.checkbox(key="include_backtests").set_value(False)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.backtests is None


def test_portfolio_settings_apply_to_results_and_frontier_selection():
    app = demo_app()
    app.toggle(key="use_cap").set_value(True).run()
    app.number_input(key="cap").set_value(30.)
    app.number_input(key="risk_free").set_value(1.)
    app.slider[0].set_value(60)
    app.slider[1].set_value(35)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    analysis, prices, metadata = app.session_state.result
    assert {key: metadata[key] for key in ("max_weight", "risk_free_rate", "train_fraction", "shrinkage")} == {
        "max_weight": .3, "risk_free_rate": .01, "train_fraction": .6, "shrinkage": .35,
    }
    assert len(analysis.train_returns) == int((len(prices) - 1) * .6)
    assert len(analysis.test_returns) == len(prices) - 1 - len(analysis.train_returns)
    assert app.metric[3].value == "30%"
    for portfolio in analysis.portfolios.values():
        assert portfolio.weights.max() <= .3 + 1e-6
        np.testing.assert_allclose(portfolio.weights.sum(), 1.)
    assert analysis.frontier_weights.to_numpy().max() <= .3 + 1e-6

    app.select_slider[0].set_value(len(analysis.frontier) - 1).run()
    assert not app.exception
    displayed = next(item.value for item in app.dataframe if list(item.value.columns) == ["Weight"])
    expected = analysis.frontier_weights.iloc[-1].sort_values(ascending=False).rename("Weight").to_frame()
    pd.testing.assert_frame_equal(displayed, expected)


def test_backtest_schedule_and_holdings_selector_match_shown_results():
    app = demo_app()
    next(item for item in app.number_input if item.label == "Sessions between trades").set_value(42)
    next(item for item in app.number_input if item.label.startswith("Rolling estimation")).set_value(126)
    app.number_input(key="cost_bps").set_value(25.)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    study = app.session_state.backtests
    assert study.settings["rebalance_every"] == 42
    assert study.settings["rolling_window"] == 126
    assert study.settings["cost_bps"] == 25.
    selected = "Rolling window · Maximum Sharpe"
    app.selectbox(key="holding_strategy").select(selected).run()
    assert not app.exception
    displayed = next(item.value for item in app.dataframe if "pnl_contribution" in item.value.columns)
    pd.testing.assert_frame_equal(displayed, study.holdings[selected].sort_values("pnl_contribution", ascending=False))
    displayed_metrics = next(item.value for item in app.dataframe if "total_turnover" in item.value.columns)
    pd.testing.assert_frame_equal(displayed_metrics, study.metrics)
    allocations = study.allocations[selected]
    prices = app.session_state.result[1]
    np.testing.assert_array_equal(np.diff(prices.index.get_indexer(allocations.index)), 42)
    assert any(item.value.equals(allocations) for item in app.dataframe)
    for strategy, holdings in study.holdings.items():
        np.testing.assert_allclose(
            holdings.pnl_contribution.sum() - study.metrics.loc[strategy, "total_cost"],
            study.metrics.loc[strategy, "total_return"], atol=1e-12,
        )


def test_invalid_backtest_settings_clear_results_and_downloads():
    app = demo_app()
    assert len(app.get("download_button")) == 2
    next(item for item in app.number_input if item.label.startswith("Rolling estimation")).set_value(1)
    app.button[0].click().run()
    assert not app.exception
    assert any("rolling_window" in item.value for item in app.error)
    assert "result" not in app.session_state
    assert "backtests" not in app.session_state
    assert not app.metric
    assert not app.get("download_button")


def test_yahoo_inputs_reach_downloader_and_failure_clears_previous_results(monkeypatch):
    from datetime import date
    from efficient_frontier import data

    calls = []
    prices = data.demo_prices().iloc[:, :3].copy()
    prices.columns = ["AAPL", "MSFT", "BRK-B"]

    def download(tickers, start, end):
        calls.append((tickers, start, end))
        if tickers == ["MISSING-TEST"]:
            raise ValueError("Yahoo returned no prices for: MISSING-TEST.")
        return prices

    monkeypatch.setattr(data, "download_prices", download)
    app = demo_app()
    app.selectbox(key="source").select("Yahoo Finance").run()
    app.text_area[0].set_value("aapl, MSFT brk.b aapl")
    app.date_input[0].set_value(date(2020, 1, 1))
    app.date_input[1].set_value(date(2024, 1, 1))
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert calls == [(["AAPL", "MSFT", "BRK-B"], "2020-01-01", "2024-01-01")]
    analysis, actual_prices, metadata = app.session_state.result
    pd.testing.assert_frame_equal(actual_prices, prices)
    assert list(analysis.portfolios["Minimum volatility"].weights.index) == list(prices.columns)
    assert metadata["source"] == "Yahoo Finance"
    assert metadata["requested_end_exclusive"] == "2024-01-01"
    assert app.metric[0].value == "3"
    assert not any("DEMO DATA" in item.value for item in app.info)

    app.radio[0].set_value("Original 60 holdings").run()
    assert any("sidebar has changes" in item.value for item in app.info)
    assert any("Applied settings · Custom tickers" in item.value for item in app.caption)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(calls[-1][0]) == 60
    assert "MRSH" in calls[-1][0] and "MMC" not in calls[-1][0]
    app.radio[0].set_value("Custom tickers").run()
    app.text_area[0].set_value("MISSING-TEST")
    app.button[0].click().run()
    assert not app.exception
    assert any("MISSING-TEST" in item.value for item in app.error)
    assert "result" not in app.session_state
    assert "backtests" not in app.session_state
    assert not app.metric
    assert not app.get("download_button")


@pytest.mark.parametrize("preset,periods", [("Calendar days (365)", 365), ("Weekly (52)", 52), ("Monthly (12)", 12)])
def test_frequency_changes_annualization_without_resampling_prices(preset, periods):
    app = demo_app()
    original_prices = app.session_state.result[1].copy()
    app.selectbox(key="observation_frequency").select(preset).run()
    assert not app.exception
    assert app.session_state.result[2]["periods_per_year"] == 252
    assert any("sidebar has changes" in item.value for item in app.info)
    assert any("252 observations/year" in item.value for item in app.caption)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    analysis, prices, metadata = app.session_state.result
    assert analysis.periods_per_year == periods
    assert metadata["periods_per_year"] == periods
    assert app.session_state.backtests.settings["periods_per_year"] == periods
    pd.testing.assert_frame_equal(prices, original_prices)
    np.testing.assert_allclose(analysis.mean_returns, analysis.train_returns.mean() * periods)
    assert any(f"{periods} observations/year" in item.value for item in app.caption)
    assert any("does not resample prices or convert currency" in item.value for item in app.caption)
    assert not any("sidebar has changes" in item.value for item in app.info)


def test_custom_frequency_and_cap_changes_keep_applied_settings_clear():
    app = demo_app()
    app.selectbox(key="observation_frequency").select("Custom").run()
    app.number_input(key="periods_per_year").set_value(48.5).run()
    app.toggle(key="use_cap").set_value(True).run()
    assert any("sidebar has changes" in item.value for item in app.info)
    assert app.session_state.result[2]["max_weight"] == 1
    app.number_input(key="cap").set_value(30.)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    analysis, _, metadata = app.session_state.result
    assert analysis.periods_per_year == 48.5
    assert metadata["periods_per_year"] == 48.5
    assert metadata["use_cap"] is True
    assert metadata["max_weight"] == .3
    assert app.session_state.backtests.settings["periods_per_year"] == 48.5
    assert any("48.5 observations/year" in item.value and "Position limit 30.00%" in item.value
               for item in app.caption)
    assert not any("sidebar has changes" in item.value for item in app.info)


def test_risk_selector_displays_matching_weights_and_variance_contributions():
    from efficient_frontier.risk import risk_contributions, risk_summary

    app = demo_app()
    app.selectbox(key="risk_portfolio").select("Equal weight").run()
    assert not app.exception
    analysis = app.session_state.result[0]
    summary = next(item.value for item in app.dataframe if "effective_holdings" in item.value.columns)
    pd.testing.assert_frame_equal(summary, risk_summary(analysis))
    assert summary.loc["Equal weight", "effective_holdings"] == pytest.approx(len(analysis.mean_returns))
    displayed = next(item.value for item in app.dataframe if "Share of portfolio variance" in item.value.columns)
    expected = pd.DataFrame({"Allocation weight": analysis.portfolios["Equal weight"].weights,
                             "Share of portfolio variance": risk_contributions(analysis)["Equal weight"]})
    pd.testing.assert_frame_equal(displayed, expected.sort_values("Allocation weight", ascending=False))
    assert displayed["Share of portfolio variance"].sum() == pytest.approx(1.)
    holdout = next(item.value for item in app.dataframe if "Sortino ratio" in item.value.columns)
    np.testing.assert_allclose(holdout["Sortino ratio"], analysis.holdout_metrics.sortino, equal_nan=True)
    np.testing.assert_allclose(holdout["Calmar ratio"], analysis.holdout_metrics.calmar, equal_nan=True)


def test_backtest_chart_selection_preserves_all_results():
    app = demo_app()
    assert len(app.multiselect(key="chart_strategies").value) == 4
    assert {"Expanding window · Low", "Expanding window · Medium", "Expanding window · Extreme"} <= set(app.multiselect(key="chart_strategies").value)
    original_metrics = app.session_state.backtests.metrics.copy()
    selected = "Rolling window · Maximum Sharpe"
    app.multiselect(key="chart_strategies").set_value([selected]).run()
    assert not app.exception
    figures = [json.loads(item.proto.spec) for item in app.get("plotly_chart")]
    backtests = [figure for figure in figures if figure["layout"].get("title", {}).get("text", "").startswith("Backtest")]
    assert len(backtests) == 2
    assert all([trace["name"] for trace in figure["data"]] == [selected] for figure in backtests)
    displayed = next(item.value for item in app.dataframe if "total_turnover" in item.value.columns)
    pd.testing.assert_frame_equal(displayed, original_metrics)
    assert len(app.get("download_button")) == 2
    app.multiselect(key="chart_strategies").set_value([]).run()
    assert not app.exception
    assert any("Select at least one strategy" in item.value for item in app.info)
    pd.testing.assert_frame_equal(app.session_state.backtests.metrics, original_metrics)
    assert len(app.get("download_button")) == 2


def test_latest_holdings_use_all_history_and_do_not_depend_on_holdout_split():
    app = demo_app()
    assert not app.exception
    assert app.tabs[0].label == "Latest holdings"
    latest = app.session_state.latest_profiles
    prices = app.session_state.result[1]
    assert latest.as_of == str(prices.index[-1].date())
    assert latest.observations == len(prices) - 1
    assert list(latest.portfolios) == ["Low", "Medium", "Extreme"]
    weights = pd.DataFrame({name: p.weights for name, p in latest.portfolios.items()})
    assert any(item.value.equals(weights.rename_axis("ticker")) for item in app.dataframe)
    assert any("not out-of-sample results" in item.value for item in app.info)
    app.slider[0].set_value(50)
    app.button[0].click().run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, latest.frontier_weights)
    assert len(app.session_state.result[0].train_returns) < len(prices) * .6


def test_short_history_keeps_original_analysis_and_explains_missing_profiles(monkeypatch):
    from efficient_frontier import data

    prices = data.demo_prices().iloc[:5, :2]
    monkeypatch.setattr(data, "demo_prices", lambda: prices)
    app = demo_app()
    assert not app.exception
    assert not app.error
    assert app.session_state.latest_profiles is None
    assert len(app.session_state.backtests.metrics) == 12
    assert app.session_state.backtests.settings["include_profiles"] is False
    assert any("at least seven" in item.value for item in app.info)
    assert any("at least six" in message for message in app.session_state.backtests.warnings)


@pytest.mark.parametrize("reuse_columns", [False, True])
def test_yahoo_market_charts_and_evidence_keep_full_comparison_family(monkeypatch, market_prices, reuse_columns):
    from efficient_frontier import data

    prices = market_prices if reuse_columns else market_prices.set_axis(["FUND_X", "FUND_Y", "FUND_Z"], axis=1)
    benchmarks = market_prices.loc[:, ["SPY", "QQQ"]]
    calls = []
    monkeypatch.setattr(data, "download_prices", lambda *args: prices)

    def download(dates):
        calls.append(dates)
        return benchmarks

    monkeypatch.setattr(data, "download_benchmarks", download)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.text_area[0].set_value(",".join(prices.columns))
    app.number_input(key="cost_bps").set_value(35.)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(calls) == (0 if reuse_columns else 1)
    if calls:
        pd.testing.assert_index_equal(calls[0], prices.index)
    analysis, actual_prices, metadata = app.session_state.result
    comparison = app.session_state.benchmarks
    study = app.session_state.backtests
    evidence = app.session_state.evidence
    pd.testing.assert_frame_equal(actual_prices, prices)
    assert list(analysis.mean_returns.index) == list(prices.columns)
    pd.testing.assert_frame_equal(comparison.prices, benchmarks)
    pd.testing.assert_index_equal(comparison.holdout_equity.index, analysis.equity.index)
    pd.testing.assert_index_equal(comparison.backtest_equity.index, study.equity.index)
    assert comparison.settings["backtest_cost_bps"] == 35
    assert comparison.settings["holdout_cost_bps"] == 0
    np.testing.assert_allclose(comparison.backtest_equity.iloc[1], 1 / 1.0035)
    np.testing.assert_allclose(comparison.backtest_metrics.total_cost, 1 - 1 / 1.0035)
    assert metadata["benchmark_source"] == (
        "SPY and QQQ columns in the input prices" if reuse_columns else "Yahoo Finance · exact input dates"
    )
    assert evidence.settings["delayed_entry"] is True
    assert evidence.settings["multiple_testing_tests"] == 96
    assert len(evidence.summary) == 48
    assert app.selectbox(key="evidence_strategy").value == "Expanding window · Medium"
    assert study.metrics.total_return.idxmax() != "Expanding window · Medium"

    figures = {figure["layout"].get("title", {}).get("text", ""): figure
               for figure in (json.loads(item.proto.spec) for item in app.get("plotly_chart"))}
    benchmark_names = {"S&P 500 (SPY)", "Nasdaq-100 (QQQ)"}
    for title in ("Latest risk profiles", "Efficient frontier", "Holdout performance", "Backtest performance", "Backtest drawdowns"):
        names = [html.unescape(trace["name"]) for trace in figures[title]["data"]]
        assert sum(name in benchmark_names for name in names) == 2
    for title, estimates, objective in (
        ("Latest risk profiles", comparison.latest_estimates, "worst_window_return"),
        ("Efficient frontier", comparison.training_estimates, "expected_return"),
    ):
        for trace in figures[title]["data"]:
            name = html.unescape(trace["name"])
            if name in benchmark_names:
                np.testing.assert_allclose(chart_values(trace["x"]), [estimates.loc[name, "volatility"]])
                np.testing.assert_allclose(chart_values(trace["y"]), [estimates.loc[name, objective]])
    for title, equity in (("Holdout performance", comparison.holdout_equity),
                          ("Backtest performance", comparison.backtest_equity),
                          ("Backtest drawdowns", comparison.backtest_equity)):
        for trace in figures[title]["data"]:
            name = html.unescape(trace["name"])
            if name in benchmark_names:
                np.testing.assert_array_equal(pd.to_datetime(trace["x"]), equity.index)
                expected = equity[name] / equity[name].cummax() - 1 if title.endswith("drawdowns") else equity[name] * 10_000
                np.testing.assert_allclose(chart_values(trace["y"]), expected)

    original_evidence = evidence.summary.copy()
    app.multiselect(key="chart_strategies").set_value(["Rolling window · Low"]).run()
    app.selectbox(key="evidence_strategy").select("Buy and hold · Extreme").run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state.evidence.summary, original_evidence)
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96
    assert len(app.session_state.backtests.metrics) == 24
    for item in app.get("plotly_chart"):
        figure = json.loads(item.proto.spec)
        if figure["layout"].get("title", {}).get("text", "").startswith("Backtest"):
            assert {html.unescape(trace["name"]) for trace in figure["data"]} == benchmark_names | {"Rolling window · Low"}
    assert len(app.get("download_button")) == 2


def test_market_disable_currency_and_download_failure_remove_stale_comparisons(monkeypatch, market_prices):
    import streamlit as st
    from efficient_frontier import data

    prices = market_prices.set_axis(["FUND_X", "FUND_Y", "FUND_Z"], axis=1)
    calls, available = [], [True]
    monkeypatch.setattr(data, "download_prices", lambda *args: prices)

    def download(dates):
        calls.append(dates)
        if not available[0]:
            raise ValueError("Missing benchmark prices for one requested date.")
        return market_prices.loc[:, ["SPY", "QQQ"]]

    monkeypatch.setattr(data, "download_benchmarks", download)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.text_area[0].set_value(",".join(prices.columns))
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.benchmarks is not None
    assert app.session_state.evidence is not None
    assert len(calls) == 1
    app.checkbox(key="compare_market").set_value(False)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.benchmarks is None
    assert app.session_state.evidence is None
    assert "disabled" in app.session_state.result[2]["benchmarks_note"]
    assert len(calls) == 1
    assert len(app.get("download_button")) == 2

    app.checkbox(key="compare_market").set_value(True)
    app.button[0].click().run()
    assert app.session_state.benchmarks is not None
    app.selectbox(key="price_currency").select("Other currency")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.benchmarks is None
    assert app.session_state.evidence is None
    assert len(calls) == 1
    assert "require all input prices in USD" in app.session_state.result[2]["benchmarks_note"]
    assert len(app.get("download_button")) == 2

    app.selectbox(key="price_currency").select("USD")
    app.button[0].click().run()
    assert app.session_state.benchmarks is not None
    available[0] = False
    st.cache_data.clear()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(calls) == 2
    assert app.session_state.benchmarks is None
    assert app.session_state.evidence is None
    assert "Missing benchmark prices" in app.session_state.result[2]["benchmarks_note"]
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)
    assert len(app.get("download_button")) == 2
    names = [html.unescape(trace["name"]) for item in app.get("plotly_chart")
             for trace in json.loads(item.proto.spec)["data"] if "name" in trace]
    assert "S&P 500 (SPY)" not in names
    assert "Nasdaq-100 (QQQ)" not in names


def test_market_evidence_switches_to_cost_free_original_holdout(monkeypatch, market_prices):
    from efficient_frontier import data

    monkeypatch.setattr(data, "download_prices", lambda *args: market_prices)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.text_area[0].set_value(",".join(market_prices.columns))
    app.number_input(key="cost_bps").set_value(50.)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.evidence.settings["delayed_entry"] is True
    app.checkbox(key="include_backtests").set_value(False)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    analysis = app.session_state.result[0]
    comparison, evidence = app.session_state.benchmarks, app.session_state.evidence
    assert app.session_state.backtests is None
    assert comparison.backtest_equity is None
    assert comparison.backtest_metrics is None
    assert comparison.settings["holdout_cost_bps"] == 0
    assert evidence.settings["delayed_entry"] is False
    assert evidence.settings["inference_observations"] == len(analysis.test_returns)
    assert evidence.settings["multiple_testing_tests"] == len(analysis.portfolios) * 4
    assert app.selectbox(key="evidence_strategy").value == "Minimum volatility"
    expected = market_prices.loc[analysis.equity.index, ["SPY", "QQQ"]]
    expected = expected.div(expected.iloc[0]).rename(columns={"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)"})
    pd.testing.assert_frame_equal(comparison.holdout_equity, expected)
    assert any("no trading costs for either portfolio or benchmark" in item.value for item in app.caption)


def test_csv_benchmarks_use_uploaded_file_or_explicit_download_only(monkeypatch, market_prices):
    import streamlit as st
    from efficient_frontier import data

    prices = market_prices.set_axis(["FUND_X", "FUND_Y", "FUND_Z"], axis=1)
    benchmarks = market_prices.loc[:, ["SPY", "QQQ"]]
    assets_file = BytesIO(prices.to_csv().encode())
    assets_file.name = "portfolio.csv"
    market_file = BytesIO(benchmarks.to_csv().encode())
    market_file.name = "benchmarks.csv"
    uploads = {"Adjusted prices": assets_file, "Benchmark adjusted prices (optional)": None}
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploads[label])
    calls = []

    def download(dates):
        calls.append(dates)
        return benchmarks

    monkeypatch.setattr(data, "download_benchmarks", download)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.selectbox(key="source").select("Upload CSV").run()
    app.checkbox(key="include_backtests").set_value(False)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert calls == []
    assert app.session_state.benchmarks is None
    assert app.session_state.evidence is None
    assert "benchmark CSV" in app.session_state.result[2]["benchmarks_note"]

    uploads["Benchmark adjusted prices (optional)"] = market_file
    app.checkbox(key="download_benchmarks").set_value(True)
    app.button[0].click().run()
    assert not app.exception
    assert calls == []
    assert app.session_state.result[2]["benchmark_source"] == "CSV: benchmarks.csv"
    pd.testing.assert_frame_equal(app.session_state.benchmarks.prices, benchmarks, check_freq=False)
    assert list(app.session_state.result[0].mean_returns.index) == list(prices.columns)
    assert app.session_state.evidence is not None

    uploads["Benchmark adjusted prices (optional)"] = None
    app.checkbox(key="download_benchmarks").set_value(False)
    app.button[0].click().run()
    assert not app.exception
    assert calls == []
    assert app.session_state.benchmarks is None
    assert app.session_state.evidence is None
    app.checkbox(key="download_benchmarks").set_value(True)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(calls) == 1
    pd.testing.assert_index_equal(calls[0], prices.index)
    assert app.session_state.result[2]["benchmark_source"] == "Yahoo Finance · exact input dates"
    assert app.session_state.benchmarks is not None
    assert app.session_state.evidence is not None
    assert len(app.get("download_button")) == 2
