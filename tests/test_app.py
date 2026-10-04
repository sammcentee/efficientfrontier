import base64
from datetime import date
import html
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def offline_app(monkeypatch):
    from efficient_frontier import data, universe

    st.cache_data.clear()
    demo = data.demo_prices().iloc[:240].copy()
    monkeypatch.setattr(data, "demo_prices", lambda: demo.copy())
    calls = []

    def blocked(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("An AppTest attempted an unmocked market download.")

    monkeypatch.setattr(data, "download_prices", blocked)
    monkeypatch.setattr(data, "download_benchmarks", blocked)
    monkeypatch.setattr(universe, "fetch_nasdaq100", blocked)
    monkeypatch.setattr(universe, "download_universe_prices", blocked)
    yield calls
    st.cache_data.clear()


@pytest.fixture
def market_prices():
    rng = np.random.default_rng(902)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal([.0008, .0005, .0006], [.012, .009, .011], (360, 3)), axis=0)),
        index=pd.bdate_range("2021-01-04", periods=360, name="Date"),
        columns=["FUND_X", "SPY", "QQQ"],
    )


@pytest.fixture
def nasdaq_data(monkeypatch, market_prices):
    from efficient_frontier import universe

    prices = market_prices.set_axis(["AAA", "BBB", "CCC"], axis=1)
    members = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC", "MISSING"],
                            "name": ["First company", "Second company", "Third company", "Missing company"],
                            "industry": ["Unknown"] * 4})
    coverage = members.assign(status=["included"] * 3 + ["excluded"], reason=[""] * 3 + ["Incomplete prices"])
    metadata = {
        "universe_requested": 4, "universe_included": 3, "universe_excluded": 1,
        "universe_coverage": coverage.to_dict(orient="records"), "universe_limitation": universe.UNIVERSE_LIMITATION,
        "universe_source_date": "2026-10-01", "universe_retrieved_at": "2026-10-04T09:00:00+00:00",
    }
    snapshot = universe.UniverseSnapshot(members, universe.NASDAQ100_URL, "2026-10-01", "2026-10-04T09:00:00+00:00")
    loaded = universe.UniversePrices(prices, coverage, market_prices[["SPY", "QQQ"]], metadata)
    calls = []

    def fetch():
        calls.append("members")
        return snapshot

    def download(actual_snapshot, start, end, *, progress=None):
        calls.append((start, end))
        pd.testing.assert_frame_equal(actual_snapshot.members, members)
        if progress is not None:
            progress(4, 4, "All fixture securities checked")
        return loaded

    monkeypatch.setattr(universe, "fetch_nasdaq100", fetch)
    monkeypatch.setattr(universe, "download_universe_prices", download)
    return SimpleNamespace(prices=prices, coverage=coverage, metadata=metadata, calls=calls)


def app_start():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def demo_app():
    app = app_start()
    app.selectbox(key="market").select("Demo").run()
    return app.button(key="build").click().run()


def market_app(monkeypatch, prices):
    from efficient_frontier import data

    monkeypatch.setattr(data, "download_prices", lambda *args: prices)
    app = app_start()
    app.selectbox(key="market").select("My tickers").run()
    app.text_area(key="ticker_text").set_value(",".join(prices.columns))
    return app.button(key="build").click().run()


def run_comparison(app):
    app.segmented_control(key="view").set_value("Compare").run()
    return app.button(key="run_comparison").click().run()


def research_view(app, topic):
    app.segmented_control(key="view").set_value("Research").run()
    return app.selectbox(key="research_topic").select(topic).run()


def figures(app):
    return [json.loads(item.proto.spec) for item in app.get("plotly_chart")]


def chart_values(values):
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values)


def test_nasdaq_is_default_and_no_network_or_work_starts_before_submit(nasdaq_data, offline_app):
    app = app_start()
    assert not app.exception
    assert app.selectbox(key="market").value == "Nasdaq-100"
    assert app.selectbox(key="market").options == ["Nasdaq-100", "My tickers", "Original 60 holdings", "Upload CSV", "Demo"]
    assert app.selectbox(key="history").value == "5 years"
    assert nasdaq_data.calls == offline_app == []
    assert "result" not in app.session_state
    assert not app.metric and not app.get("download_button")
    app.selectbox(key="history").select("10 years").run()
    assert nasdaq_data.calls == []
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert nasdaq_data.calls[0] == "members"
    assert len(nasdaq_data.calls) == 2
    pd.testing.assert_frame_equal(app.session_state.result[1], nasdaq_data.prices)
    assert app.session_state.result[2]["universe_requested"] == 4
    assert "backtests" not in app.session_state
    assert "report_bytes" not in app.session_state
    assert not app.get("download_button")


def test_nasdaq_coverage_and_company_names_preserve_every_checked_security(nasdaq_data):
    app = app_start().button(key="build").click().run()
    assert not app.exception
    holdings = next(item.value for item in app.dataframe if "Company" in item.value.columns)
    assert set(holdings.Symbol) == {"AAA", "BBB", "CCC"}
    assert set(holdings.Company) == {"First company", "Second company", "Third company"}
    assert holdings.Weight.sum() == pytest.approx(1)
    research_view(app, "Data & coverage")
    assert any(item.value.equals(nasdaq_data.coverage) for item in app.dataframe)
    assert any("not a historical reconstruction" in item.value for item in app.caption)
    assert app.session_state.result[2]["universe_coverage"][-1]["reason"] == "Incomplete prices"


def test_repeated_nasdaq_submit_replays_progress_cache_and_preserves_coverage(nasdaq_data):
    app = app_start().button(key="build").click().run()
    assert not app.exception and not app.error
    calls = nasdaq_data.calls.copy()
    assert len(calls) == 2
    weights = app.session_state.latest_profiles.frontier_weights.copy()

    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert nasdaq_data.calls == calls
    pd.testing.assert_frame_equal(app.session_state.result[1], nasdaq_data.prices)
    pd.testing.assert_frame_equal(app.session_state.coverage, nasdaq_data.coverage)
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, weights)
    assert app.session_state.result[2]["universe_coverage"] == nasdaq_data.metadata["universe_coverage"]
    assert not any("differs from this result" in item.value for item in app.info)


def test_demo_requires_submit_and_does_not_create_market_evidence(offline_app):
    app = app_start()
    app.selectbox(key="market").select("Demo").run()
    assert "result" not in app.session_state
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.segmented_control(key="view").value == "Portfolio"
    assert app.segmented_control(key="risk_profile").value == "Medium"
    assert app.segmented_control(key="risk_profile").options == ["Low", "Medium", "Highest"]
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert "backtests" not in app.session_state
    assert "report_bytes" not in app.session_state
    assert offline_app == []
    assert any("synthetic" in item.value for item in app.info)


def test_risk_choice_changes_display_without_refitting_or_dropping_holdings(monkeypatch):
    from efficient_frontier import profiles

    calls = []
    original = profiles.build_profiles

    def calculate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(profiles, "build_profiles", calculate)
    app = demo_app()
    latest = app.session_state.latest_profiles
    for profile in ("Low", "Extreme", "Medium"):
        app.segmented_control(key="risk_profile").set_value(profile).run()
        assert not app.exception
        displayed = next(item.value for item in app.dataframe if list(item.value.columns) == ["Symbol", "Weight"])
        expected = latest.portfolios[profile].weights.sort_values(ascending=False).rename("Weight").rename_axis("Symbol").reset_index()
        pd.testing.assert_frame_equal(displayed, expected)
        assert len(displayed) == len(app.session_state.result[1].columns)
    assert calls == [1]
    app.button(key="go_compare").click().run()
    assert app.segmented_control(key="view").value == "Compare"
    assert "backtests" not in app.session_state
    assert app.segmented_control(key="risk_profile").value == "Medium"


def test_settings_apply_on_submit_and_latest_fit_does_not_depend_on_split():
    app = demo_app()
    latest = app.session_state.latest_profiles
    prices = app.session_state.result[1].copy()
    app.slider(key="train_pct").set_value(50).run()
    assert app.session_state.result[2]["train_fraction"] == .7
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, latest.frontier_weights)
    assert app.session_state.latest_profiles.observations == len(prices) - 1
    assert app.session_state.latest_profiles.as_of == prices.index[-1].date().isoformat()
    assert len(app.session_state.result[0].train_returns) == int((len(prices) - 1) * .5)
    app.checkbox(key="use_cap").set_value(True).run()
    app.number_input(key="cap").set_value(30.)
    app.number_input(key="risk_free").set_value(1.)
    app.slider(key="shrink").set_value(35)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    analysis, _, metadata = app.session_state.result
    assert {key: metadata[key] for key in ("max_weight", "risk_free_rate", "shrinkage")} == {
        "max_weight": .3, "risk_free_rate": .01, "shrinkage": .35,
    }
    for portfolio in [*analysis.portfolios.values(), *app.session_state.latest_profiles.portfolios.values()]:
        assert portfolio.weights.max() <= .3 + 1e-6
        assert portfolio.weights.sum() == pytest.approx(1)


def test_invalid_new_fit_clears_previous_results_comparison_and_report():
    app = run_comparison(demo_app())
    app.button(key="prepare_report").click().run()
    assert "report_bytes" in app.session_state
    app.checkbox(key="use_cap").set_value(True).run()
    app.number_input(key="cap").set_value(5.)
    app.button(key="build").click().run()
    assert not app.exception
    assert any("infeasible" in item.value.lower() for item in app.error)
    for key in ("result", "latest_profiles", "backtests", "benchmarks", "evidence", "report_bytes"):
        assert key not in app.session_state
    assert not app.metric and not app.get("download_button")


def test_custom_tickers_dates_and_original_list_reach_only_explicit_downloads(monkeypatch, market_prices):
    from efficient_frontier import data

    calls = []

    def download(tickers, start, end):
        calls.append((tickers, start, end))
        if tickers == ["MISSING-TEST"]:
            raise ValueError("Yahoo returned no prices for: MISSING-TEST.")
        return market_prices

    monkeypatch.setattr(data, "download_prices", download)
    app = app_start()
    app.selectbox(key="market").select("My tickers").run()
    app.selectbox(key="history").select("Custom dates").run()
    app.text_area(key="ticker_text").set_value("aapl, MSFT brk.b aapl")
    app.date_input(key="start").set_value(date(2020, 1, 1))
    app.date_input(key="end").set_value(date(2024, 1, 1))
    assert calls == []
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == [(["AAPL", "MSFT", "BRK-B"], "2020-01-01", "2024-01-01")]
    app.selectbox(key="market").select("Original 60 holdings").run()
    assert len(calls) == 1
    assert app.session_state.result[2]["universe"] == "My tickers"
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="build").click().run()
    assert len(calls[-1][0]) == 60
    assert "MRSH" in calls[-1][0] and "MMC" not in calls[-1][0]
    app.selectbox(key="market").select("My tickers").run()
    app.text_area(key="ticker_text").set_value("MISSING-TEST")
    app.button(key="build").click().run()
    assert not app.exception
    assert any("MISSING-TEST" in item.value for item in app.error)
    assert "result" not in app.session_state


def test_ticker_draft_keeps_applied_result_and_downloads_only_after_submit(monkeypatch, market_prices):
    from efficient_frontier import data

    calls = []

    def download(tickers, start, end):
        calls.append(tickers)
        return market_prices.rename(columns={"FUND_X": tickers[0]})

    monkeypatch.setattr(data, "download_prices", download)
    app = app_start()
    app.selectbox(key="market").select("My tickers").run()
    app.text_area(key="ticker_text").set_value("FUND_X, SPY, QQQ")
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    original_inputs = app.session_state.applied_inputs.copy()
    original_weights = app.session_state.latest_profiles.frontier_weights.copy()
    assert not any("differs from this result" in item.value for item in app.info)

    app.text_area(key="ticker_text").set_value("FUND_NEW, SPY, QQQ").run()
    assert not app.exception
    assert calls == [["FUND_X", "SPY", "QQQ"]]
    assert any("differs from this result" in item.value for item in app.info)
    assert app.session_state.applied_inputs == original_inputs
    pd.testing.assert_frame_equal(app.session_state.result[1], market_prices)
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, original_weights)

    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == [["FUND_X", "SPY", "QQQ"], ["FUND_NEW", "SPY", "QQQ"]]
    assert list(app.session_state.result[1]) == ["FUND_NEW", "SPY", "QQQ"]
    assert app.session_state.applied_inputs["tickers"] == "FUND_NEW, SPY, QQQ"
    assert not any("differs from this result" in item.value for item in app.info)


def test_csv_content_and_benchmark_consent_edits_mark_pending_without_download(monkeypatch, market_prices, offline_app):
    prices = market_prices.set_axis(["FUND_X", "FUND_Y", "FUND_Z"], axis=1)
    benchmarks = market_prices[["SPY", "QQQ"]]
    assets_upload = BytesIO(prices.to_csv().encode())
    assets_upload.name = "portfolio.csv"
    benchmark_upload = BytesIO(benchmarks.to_csv().encode())
    benchmark_upload.name = "benchmarks.csv"
    originals = {"prices_upload": assets_upload, "benchmark_upload": benchmark_upload}
    uploads = originals.copy()
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploads[kwargs["key"]])
    app = app_start()
    app.selectbox(key="market").select("Upload CSV").run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    original_inputs = app.session_state.applied_inputs.copy()
    assert not any("differs from this result" in item.value for item in app.info)

    for key, frame in (("prices_upload", prices), ("benchmark_upload", benchmarks)):
        changed = frame.copy()
        changed.iloc[-1, 0] *= 1.05
        replacement = BytesIO(changed.to_csv().encode())
        replacement.name = originals[key].name
        uploads[key] = replacement
        app.run()
        assert not app.exception and not app.error
        assert any("differs from this result" in item.value for item in app.info)
        assert app.session_state.applied_inputs == original_inputs
        pd.testing.assert_frame_equal(app.session_state.result[1], prices, check_freq=False)
        pd.testing.assert_frame_equal(app.session_state.benchmarks.prices, benchmarks, check_freq=False)
        assert offline_app == []
        restored = BytesIO(originals[key].getvalue())
        restored.name = originals[key].name
        uploads[key] = restored
        app.run()
        assert not any("differs from this result" in item.value for item in app.info)

    uploads["benchmark_upload"] = None
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.benchmarks is None
    assert not any("differs from this result" in item.value for item in app.info)
    app.checkbox(key="download_benchmarks").set_value(True).run()
    assert not app.exception and not app.error
    assert any("differs from this result" in item.value for item in app.info)
    assert app.session_state.applied_inputs["download_benchmarks"] is False
    assert app.session_state.benchmarks is None
    pd.testing.assert_frame_equal(app.session_state.result[1], prices, check_freq=False)
    assert offline_app == []


def test_comparison_runs_only_on_request_and_display_changes_keep_cached_results(monkeypatch):
    from efficient_frontier import backtest

    calls = []
    original = backtest.run_backtests

    def calculate(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(backtest, "run_backtests", calculate)
    app = demo_app()
    app.segmented_control(key="view").set_value("Compare").run()
    assert calls == []
    app.number_input(key="rebalance_every").set_value(42)
    app.number_input(key="rolling_window").set_value(84)
    app.number_input(key="cost_bps").set_value(25.)
    app.button(key="run_comparison").click().run()
    assert not app.exception and not app.error
    study = app.session_state.backtests
    assert len(study.metrics) == 24
    assert study.settings["rebalance_every"] == 42
    assert study.settings["rolling_window"] == 84
    assert study.settings["cost_bps"] == 25
    assert len(calls) == 1
    app.selectbox(key="comparison_method").select("Rolling window").run()
    app.segmented_control(key="risk_profile").set_value("Extreme").run()
    app.segmented_control(key="comparison_chart").set_value("Drawdown").run()
    assert not app.exception
    assert len(calls) == 1
    assert [trace["name"] for trace in figures(app)[0]["data"]] == ["Rolling window · Highest"]
    pd.testing.assert_frame_equal(app.session_state.backtests.metrics, study.metrics)
    app.button(key="run_comparison").click().run()
    assert len(calls) == 1
    app.number_input(key="cost_bps").set_value(0.)
    app.button(key="run_comparison").click().run()
    assert len(calls) == 2
    assert app.session_state.backtests.metrics.total_cost.eq(0).all()


def test_failed_comparison_preserves_portfolios_and_clears_stale_derived_state(monkeypatch, market_prices):
    app = run_comparison(market_app(monkeypatch, market_prices))
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96
    app.button(key="prepare_report").click().run()
    original = app.session_state.latest_profiles.frontier_weights.copy()
    app.number_input(key="rolling_window").set_value(1)
    app.button(key="run_comparison").click().run()
    assert not app.exception
    assert any("rolling_window" in item.value for item in app.error)
    assert "backtests" not in app.session_state
    assert "report_bytes" not in app.session_state
    assert "evidence" not in app.session_state or app.session_state.evidence is None
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, original)
    assert "result" in app.session_state


def test_report_is_prepared_explicitly_and_new_comparison_invalidates_it(monkeypatch):
    from efficient_frontier import presentation

    calls = []
    original = presentation.report_zip

    def export(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(presentation, "report_zip", export)
    app = demo_app()
    assert calls == [] and not app.get("download_button")
    app.button(key="prepare_report").click().run()
    assert not app.exception and calls == [1]
    assert len(app.get("download_button")) == 1
    with zipfile.ZipFile(BytesIO(app.session_state.report_bytes)) as archive:
        assert "latest_profile_weights.csv" in archive.namelist()
        assert "backtest_metrics.csv" not in archive.namelist()
        exported = pd.read_csv(archive.open("latest_profile_weights.csv"), index_col=0)
        expected = pd.DataFrame({name: value.weights for name, value in app.session_state.latest_profiles.portfolios.items()}).rename_axis("ticker")
        pd.testing.assert_frame_equal(exported, expected)
    app.segmented_control(key="risk_profile").set_value("Low").run()
    app.button(key="prepare_report").click().run()
    assert calls == [1]
    run_comparison(app)
    assert "report_bytes" not in app.session_state
    assert not app.get("download_button")
    app.button(key="prepare_report").click().run()
    assert calls == [1, 1]
    with zipfile.ZipFile(BytesIO(app.session_state.report_bytes)) as archive:
        assert len(pd.read_csv(archive.open("backtest_metrics.csv"), index_col=0)) == 24


@pytest.mark.parametrize("frequency,periods", [("Daily calendar prices", 365), ("Weekly", 52), ("Monthly", 12), ("Custom", 48.5)])
def test_csv_frequency_scales_estimates_without_resampling(monkeypatch, market_prices, frequency, periods):
    upload = BytesIO(market_prices.to_csv().encode())
    upload.name = "portfolio.csv"
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: upload if kwargs["key"] == "prices_upload" else None)
    app = app_start()
    app.selectbox(key="market").select("Upload CSV").run()
    app.selectbox(key="frequency").select(frequency).run()
    if frequency == "Custom":
        app.number_input(key="periods_per_year").set_value(periods)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    analysis, prices, metadata = app.session_state.result
    assert analysis.periods_per_year == metadata["periods_per_year"] == periods
    pd.testing.assert_frame_equal(prices, market_prices, check_freq=False)
    np.testing.assert_allclose(analysis.mean_returns, analysis.train_returns.mean() * periods)
    assert app.session_state.latest_profiles.settings["periods_per_year"] == periods
    run_comparison(app)
    assert not app.exception and not app.error
    assert app.session_state.backtests.settings["periods_per_year"] == periods
    assert app.session_state.evidence.settings["periods_per_year"] == periods


def test_market_costs_charts_and_evidence_preserve_all_96_tests(monkeypatch, market_prices):
    app = market_app(monkeypatch, market_prices)
    assert not app.exception and not app.error
    assert app.session_state.evidence.settings["delayed_entry"] is False
    analysis = app.session_state.result[0]
    assert app.session_state.evidence.settings["multiple_testing_tests"] == len(analysis.portfolios) * 4
    research_view(app, "Efficient frontiers")
    comparison = app.session_state.benchmarks
    expected_points = {"Latest risk profiles": (comparison.latest_estimates, "worst_window_return"),
                       "Efficient frontier": (comparison.training_estimates, "expected_return")}
    names = {"S&P 500 (SPY)", "Nasdaq-100 (QQQ)"}
    for figure in figures(app):
        title = figure["layout"].get("title", {}).get("text")
        for trace in figure["data"]:
            name = html.unescape(trace.get("name", ""))
            if name in names and title in expected_points:
                estimates, objective = expected_points[title]
                np.testing.assert_allclose(chart_values(trace["y"]), [estimates.loc[name, objective]])
    app.segmented_control(key="view").set_value("Compare").run()
    app.number_input(key="cost_bps").set_value(35.)
    app.button(key="run_comparison").click().run()
    assert not app.exception and not app.error
    comparison, evidence, study = app.session_state.benchmarks, app.session_state.evidence, app.session_state.backtests
    assert len(study.metrics) == 24 and len(evidence.summary) == 48
    assert evidence.settings["multiple_testing_tests"] == 96
    assert evidence.settings["delayed_entry"] is True
    pd.testing.assert_index_equal(comparison.backtest_equity.index, study.equity.index)
    np.testing.assert_allclose(comparison.backtest_equity.iloc[1], 1 / 1.0035)
    np.testing.assert_allclose(comparison.backtest_metrics.total_cost, 1 - 1 / 1.0035)
    assert app.selectbox(key="comparison_method").value == "Expanding window"
    assert {html.unescape(trace["name"]) for trace in figures(app)[0]["data"]} == names | {"Expanding window · Medium"}
    original = evidence.summary.copy()
    app.selectbox(key="comparison_method").select("Buy and hold").run()
    app.segmented_control(key="risk_profile").set_value("Extreme").run()
    app.segmented_control(key="comparison_chart").set_value("Relative to markets").run()
    assert not app.exception
    assert len(figures(app)[0]["data"]) == 2
    pd.testing.assert_frame_equal(app.session_state.evidence.summary, original)
    research_view(app, "All backtests")
    app.multiselect(key="chart_strategies").set_value(["Rolling window · Low"]).run()
    assert {html.unescape(trace["name"]) for trace in figures(app)[0]["data"]} == names | {"Rolling window · Low"}
    app.multiselect(key="chart_strategies").set_value([]).run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state.backtests.metrics, study.metrics)
    pd.testing.assert_frame_equal(app.session_state.evidence.summary, original)
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96


def test_benchmark_disable_currency_and_download_failure_clear_old_evidence(monkeypatch, market_prices):
    from efficient_frontier import data

    prices = market_prices.set_axis(["FUND_X", "FUND_Y", "FUND_Z"], axis=1)
    calls, available = [], [True]

    def download(dates):
        calls.append(dates)
        if not available[0]:
            raise ValueError("Missing benchmark prices for one requested date.")
        return market_prices[["SPY", "QQQ"]]

    monkeypatch.setattr(data, "download_benchmarks", download)
    app = market_app(monkeypatch, prices)
    assert app.session_state.benchmarks is not None
    assert len(calls) == 1
    app.checkbox(key="compare_market").set_value(False)
    app.button(key="build").click().run()
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert len(calls) == 1
    app.checkbox(key="compare_market").set_value(True)
    app.selectbox(key="price_currency").select("Other currency")
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert "USD" in app.session_state.result[2]["benchmarks_note"]
    assert len(calls) == 1
    app.selectbox(key="price_currency").select("USD")
    app.button(key="build").click().run()
    assert app.session_state.benchmarks is not None
    available[0] = False
    st.cache_data.clear()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert "Missing benchmark prices" in app.session_state.result[2]["benchmarks_note"]
    assert len(calls) == 2
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)


def test_inference_failure_keeps_valid_benchmark_paths(monkeypatch, market_prices):
    from efficient_frontier import evidence

    def unavailable(*args, **kwargs):
        raise RuntimeError("Fixture inference unavailable.")

    monkeypatch.setattr(evidence, "analyze_evidence", unavailable)
    app = market_app(monkeypatch, market_prices)
    assert not app.exception and not app.error
    assert app.session_state.benchmarks is not None
    assert app.session_state.evidence is None
    assert "Fixture inference unavailable" in app.session_state.result[2]["evidence_note"]
    run_comparison(app)
    assert not app.exception and not app.error
    assert len(app.session_state.backtests.metrics) == 24
    assert app.session_state.benchmarks.backtest_equity is not None
    assert app.session_state.evidence is None


def test_benchmark_retry_removes_stale_failure_note_from_results_and_export(monkeypatch, market_prices):
    from efficient_frontier import benchmarks

    original = benchmarks.compare_benchmarks
    attempts = []

    def compare(benchmark_prices, prices, analysis, study=None, latest_profiles=None, **kwargs):
        if study is not None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("Fixture comparison failure.")
        return original(benchmark_prices, prices, analysis, study, latest_profiles, **kwargs)

    monkeypatch.setattr(benchmarks, "compare_benchmarks", compare)
    app = market_app(monkeypatch, market_prices)
    assert app.session_state.benchmarks is not None
    run_comparison(app)
    assert not app.exception and not app.error
    assert len(app.session_state.backtests.metrics) == 24
    assert "Fixture comparison failure" in app.session_state.result[2]["benchmarks_note"]
    assert "benchmarks" not in app.session_state or app.session_state.benchmarks is None
    assert "evidence" not in app.session_state or app.session_state.evidence is None

    app.button(key="run_comparison").click().run()
    assert not app.exception and not app.error
    assert attempts == [1, 1]
    assert app.session_state.benchmarks.backtest_equity is not None
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96
    assert "benchmarks_note" not in app.session_state.result[2]
    app.button(key="prepare_report").click().run()
    assert not app.exception
    with zipfile.ZipFile(BytesIO(app.session_state.report_bytes)) as archive:
        assert "benchmarks_note" not in json.loads(archive.read("metadata.json"))
        assert "Fixture comparison failure" not in archive.read("report.html").decode()


def test_csv_benchmark_upload_wins_and_download_requires_explicit_choice(monkeypatch, market_prices):
    from efficient_frontier import data

    prices = market_prices.set_axis(["FUND_X", "FUND_Y", "FUND_Z"], axis=1)
    upload = BytesIO(prices.to_csv().encode())
    upload.name = "portfolio.csv"
    benchmark_upload = BytesIO(market_prices[["SPY", "QQQ"]].to_csv().encode())
    benchmark_upload.name = "benchmarks.csv"
    uploads = {"prices_upload": upload, "benchmark_upload": None}
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploads[kwargs["key"]])
    calls = []

    def download(dates):
        calls.append(dates)
        return market_prices[["SPY", "QQQ"]]

    monkeypatch.setattr(data, "download_benchmarks", download)
    app = app_start()
    app.selectbox(key="market").select("Upload CSV").run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == [] and app.session_state.benchmarks is None
    uploads["benchmark_upload"] = benchmark_upload
    app.checkbox(key="download_benchmarks").set_value(True)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == []
    assert app.session_state.result[2]["benchmark_source"] == "CSV: benchmarks.csv"
    uploads["benchmark_upload"] = None
    app.checkbox(key="download_benchmarks").set_value(False)
    app.button(key="build").click().run()
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert calls == []
    app.checkbox(key="download_benchmarks").set_value(True)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert len(calls) == 1
    pd.testing.assert_index_equal(calls[0], prices.index)
    assert list(app.session_state.result[0].mean_returns.index) == list(prices.columns)


def test_research_risk_and_trade_tables_match_complete_results():
    from efficient_frontier.risk import risk_contributions

    app = run_comparison(demo_app())
    study = app.session_state.backtests
    research_view(app, "Risk breakdown")
    app.selectbox(key="risk_portfolio").select("Equal weight").run()
    result = app.session_state.result[0]
    displayed = next(item.value for item in app.dataframe if "Share of portfolio variance" in item.value.columns)
    expected = pd.DataFrame({"Allocation weight": result.portfolios["Equal weight"].weights,
                             "Share of portfolio variance": risk_contributions(result)["Equal weight"]}).sort_values("Allocation weight", ascending=False)
    pd.testing.assert_frame_equal(displayed, expected)
    research_view(app, "All backtests")
    selected = "Rolling window · Maximum Sharpe"
    app.selectbox(key="holding_strategy").select(selected).run()
    assert not app.exception
    assert any(item.value.equals(study.holdings[selected]) for item in app.dataframe)
    assert any(item.value.equals(study.allocations[selected]) for item in app.dataframe)
    assert any(item.value.equals(study.trades[selected]) for item in app.dataframe)
    assert any(item.value.equals(study.metrics) for item in app.dataframe)


def test_research_holdout_ratios_are_not_formatted_as_percentages():
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

    app = research_view(demo_app(), "Efficient frontiers")
    table = next(item for item in app.dataframe if "sortino" in item.value.columns)
    display = convert_arrow_bytes_to_pandas_df(table.proto.arrow_data.styler.display_values)
    for column in ("sharpe", "sortino", "calmar"):
        assert not display[column].str.contains("%", regex=False).any()
    assert display.total_return.str.contains("%", regex=False).all()


def test_large_universe_chart_limit_does_not_change_holdings(monkeypatch):
    from efficient_frontier import data

    rng = np.random.default_rng(44)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0003, .01, (150, 24)), axis=0)),
                          index=pd.bdate_range("2024-01-01", periods=150, name="Date"), columns=[f"DEMO_{i}" for i in range(24)])
    monkeypatch.setattr(data, "demo_prices", lambda: prices)
    app = demo_app()
    assert not app.exception and not app.error
    holdings = next(item.value for item in app.dataframe if "Weight" in item.value.columns)
    assert len(holdings) == 24
    research_view(app, "Data & coverage")
    assert len(app.multiselect(key="correlation_assets").value) == 15
    app.multiselect(key="correlation_assets").set_value([]).run()
    assert not app.exception
    assert len(app.session_state.result[0].mean_returns) == 24
    assert all(len(portfolio.weights) == 24 for portfolio in app.session_state.latest_profiles.portfolios.values())


def test_short_history_preserves_original_research_and_12_method_paths(monkeypatch):
    from efficient_frontier import data

    prices = data.demo_prices().iloc[:5, :2]
    monkeypatch.setattr(data, "demo_prices", lambda: prices)
    app = demo_app()
    assert not app.exception and not app.error
    assert app.session_state.latest_profiles is None
    assert any("at least seven" in item.value for item in app.info)
    run_comparison(app)
    assert not app.exception and not app.error
    assert len(app.session_state.backtests.metrics) == 12
    assert app.session_state.backtests.settings["include_profiles"] is False
    assert any("too short" in item.value for item in app.info)
    research_view(app, "Efficient frontiers")
    assert not app.exception
    assert len(figures(app)) == 2


def test_single_asset_and_missing_sharpe_do_not_break_profile_views(monkeypatch):
    from efficient_frontier import data

    prices = data.demo_prices().iloc[:, :1]
    monkeypatch.setattr(data, "demo_prices", lambda: prices)
    app = demo_app()
    app.number_input(key="risk_free").set_value(100.)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert "Maximum Sharpe" not in app.session_state.result[0].portfolios
    for portfolio in app.session_state.latest_profiles.portfolios.values():
        np.testing.assert_allclose(portfolio.weights, [1.])
    research_view(app, "Methodology")
    assert not app.exception
    assert any("Maximum Sharpe omitted" in item.value for item in app.markdown)


def test_missing_csv_clears_old_result_without_network(offline_app):
    app = demo_app()
    app.selectbox(key="market").select("Upload CSV").run()
    assert app.session_state.result[2]["universe"] == "Demo"
    app.button(key="build").click().run()
    assert not app.exception
    assert any("Choose an adjusted-price CSV" in item.value for item in app.error)
    assert "result" not in app.session_state
    assert offline_app == []
