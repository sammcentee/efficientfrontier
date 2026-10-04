import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_demo_renders_and_invalid_cap_clears_old_result():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "6"
    assert app.metric[3].value == "None"
    assert app.session_state.result[2]["max_weight"] == 1
    assert any("synthetic" in item.value.lower() for item in app.info)
    app.toggle(key="use_cap").set_value(True).run()
    app.number_input(key="cap").set_value(5.0)
    app.button[0].click().run()
    assert not app.exception
    assert any("infeasible" in item.value.lower() for item in app.error)
    assert len(app.metric) == 0


def test_high_risk_free_rate_omits_sharpe_without_breaking_app():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.number_input(key="risk_free").set_value(100.0)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any("Maximum Sharpe omitted" in item.value for item in app.warning)
    assert "Maximum Sharpe" not in app.session_state.result[0].portfolios


def test_switching_source_does_not_show_stale_demo_results():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "64"
    assert len(app.multiselect(key="correlation_assets").value) == 20
    app.multiselect(key="correlation_assets").set_value([]).run()
    assert not app.exception
    assert len(app.session_state.result[0].portfolios["Minimum volatility"].weights) == 64


def test_backtest_controls_update_costs_and_can_disable_comparison():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert len(app.session_state.backtests.metrics) == 12
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
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

    app = AppTest.from_file(str(APP), default_timeout=20).run()
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
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert len(app.multiselect(key="chart_strategies").value) == 4
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
