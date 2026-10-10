import base64
from datetime import date
import html
from io import BytesIO
import json
from pathlib import Path
import re
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


def open_setup(app):
    return app.button(key="edit_setup").click().run()


def demo_app():
    return app_start().button(key="try_demo").click().run()


def market_app(monkeypatch, prices):
    from efficient_frontier import data

    monkeypatch.setattr(data, "download_prices", lambda *args: prices)
    app = open_setup(app_start())
    app.radio(key="market").set_value("My tickers").run()
    app.text_area(key="ticker_text").set_value(",".join(prices.columns))
    return app.button(key="build").click().run()


def run_comparison(app):
    """Run the market test again with the current Test settings."""
    return app.button(key="run_comparison").click().run()


def research_view(app, topic):
    app.segmented_control(key="view").set_value("Research").run()
    return app.segmented_control(key="research_topic").set_value(topic).run()


def figures(app):
    return [json.loads(item.proto.spec) for item in app.get("plotly_chart")]


def chart_values(values):
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values)


def html_text(app):
    """Every st.html fragment on the page, joined."""
    return "\n".join(item.proto.body for item in app.get("html"))


def test_nasdaq_is_default_and_no_network_or_work_starts_before_submit(nasdaq_data, offline_app):
    app = app_start()
    assert not app.exception
    assert app.button(key="build").label == "Show portfolios"
    assert nasdaq_data.calls == offline_app == []
    assert "result" not in app.session_state
    assert not app.metric and not app.get("download_button")
    open_setup(app)
    assert app.radio(key="market").value == "Nasdaq-100"
    assert app.radio(key="market").options == ["Nasdaq-100", "My tickers", "Original 60 holdings", "Upload CSV", "Demo"]
    assert app.segmented_control(key="history").value == "5 years"
    app.segmented_control(key="history").set_value("10 years").run()
    assert nasdaq_data.calls == []
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert nasdaq_data.calls[0] == "members"
    assert len(nasdaq_data.calls) == 2
    pd.testing.assert_frame_equal(app.session_state.result[1], nasdaq_data.prices)
    assert app.session_state.result[2]["universe_requested"] == 4
    assert "backtests" in app.session_state
    assert [button.key for button in app.get("download_button")] == ["export"]


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

    open_setup(app).button(key="build").click().run()
    assert not app.exception and not app.error
    assert nasdaq_data.calls == calls
    pd.testing.assert_frame_equal(app.session_state.result[1], nasdaq_data.prices)
    pd.testing.assert_frame_equal(app.session_state.coverage, nasdaq_data.coverage)
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, weights)
    assert app.session_state.result[2]["universe_coverage"] == nasdaq_data.metadata["universe_coverage"]
    assert not any("differs from this result" in item.value for item in app.info)


def test_demo_requires_submit_and_does_not_create_market_evidence(offline_app):
    app = open_setup(app_start())
    app.radio(key="market").set_value("Demo").run()
    assert "result" not in app.session_state
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.segmented_control(key="view").value == "Portfolio"
    assert app.segmented_control(key="risk_profile").value == "Medium"
    assert app.segmented_control(key="risk_profile").options == ["Low", "Medium", "Highest"]
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert "backtests" in app.session_state
    assert offline_app == []
    assert any("synthetic" in item.value for item in app.info)


def test_risk_choice_changes_display_without_refitting_or_dropping_holdings(monkeypatch):
    from efficient_frontier import profiles
    from efficient_frontier.story import LEVEL

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
        # The headline, stats and ledger show the selected level's own numbers.
        portfolio, page = latest.portfolios[profile], html_text(app)
        held = int((portfolio.weights >= .0005).sum())
        assert f'<span class="pl-num">{portfolio.volatility:.1%}</span><span class="pl-lbl">Volatility</span>' in page
        assert f'<span class="pl-num">{held}</span><span class="pl-lbl">Assets held</span>' in page
        assert f'<span class="pl-nb">{LEVEL[profile]}-risk</span>' in page
        selected_row = page.split('<tr class="on">')[1].split("</tr>")[0]
        assert f"{latest.summary.loc[profile, 'volatility']:.1%}" in selected_row and f"<td>{held}</td>" in selected_row
    assert calls == [1]
    app.button(key="key_market").click().run()
    assert app.segmented_control(key="view").value == "Portfolio"
    assert "backtests" in app.session_state
    assert app.segmented_control(key="risk_profile").value == "Medium"


def test_settings_apply_on_submit_and_latest_fit_does_not_depend_on_split():
    app = demo_app()
    latest = app.session_state.latest_profiles
    prices = app.session_state.result[1].copy()
    open_setup(app).slider(key="train_pct").set_value(50).run()
    assert app.session_state.result[2]["train_fraction"] == .7
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, latest.frontier_weights)
    assert app.session_state.latest_profiles.observations == len(prices) - 1
    assert app.session_state.latest_profiles.as_of == prices.index[-1].date().isoformat()
    assert len(app.session_state.result[0].train_returns) == int((len(prices) - 1) * .5)
    open_setup(app).checkbox(key="use_cap").set_value(True).run()
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
    open_setup(app).checkbox(key="use_cap").set_value(True).run()
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
    app = open_setup(app_start())
    app.radio(key="market").set_value("My tickers").run()
    app.segmented_control(key="history").set_value("Custom dates").run()
    app.text_area(key="ticker_text").set_value("aapl, MSFT brk.b aapl")
    app.date_input(key="start").set_value(date(2020, 1, 1))
    app.date_input(key="end").set_value(date(2024, 1, 1))
    assert calls == []
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == [(["AAPL", "MSFT", "BRK-B"], "2020-01-01", "2024-01-01")]
    open_setup(app).radio(key="market").set_value("Original 60 holdings").run()
    assert len(calls) == 1
    assert app.session_state.result[2]["universe"] == "My tickers"
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="build").click().run()
    assert len(calls[-1][0]) == 60
    assert "MRSH" in calls[-1][0] and "MMC" not in calls[-1][0]
    open_setup(app).radio(key="market").set_value("My tickers").run()
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
    app = open_setup(app_start())
    app.radio(key="market").set_value("My tickers").run()
    app.text_area(key="ticker_text").set_value("FUND_X, SPY, QQQ")
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    original_inputs = app.session_state.applied_inputs.copy()
    original_weights = app.session_state.latest_profiles.frontier_weights.copy()
    assert not any("differs from this result" in item.value for item in app.info)

    open_setup(app).text_area(key="ticker_text").set_value("FUND_NEW, SPY, QQQ").run()
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
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    original_inputs = app.session_state.applied_inputs.copy()
    assert not any("differs from this result" in item.value for item in app.info)

    open_setup(app)
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
    app.run()
    app.button(key="clear_benchmark_file").click().run()
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.benchmarks is None
    assert not any("differs from this result" in item.value for item in app.info)
    open_setup(app).checkbox(key="download_benchmarks").set_value(True).run()
    assert not app.exception and not app.error
    assert any("differs from this result" in item.value for item in app.info)
    assert app.session_state.applied_inputs["download_benchmarks"] is False
    assert app.session_state.benchmarks is None
    pd.testing.assert_frame_equal(app.session_state.result[1], prices, check_freq=False)
    assert offline_app == []


def test_market_test_runs_once_at_build_and_display_changes_keep_cached_results(monkeypatch):
    from efficient_frontier import backtest

    calls = []
    original = backtest.run_backtests

    def calculate(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(backtest, "run_backtests", calculate)
    app = demo_app()
    assert len(calls) == 1
    assert (calls[0]["rebalance_every"], calls[0]["rolling_window"], calls[0]["cost_bps"]) == (21, None, 10.)
    app.number_input(key="rebalance_every").set_value(42)
    app.number_input(key="rolling_window").set_value(84)
    app.number_input(key="cost_bps").set_value(25.)
    run_comparison(app)
    assert not app.exception and not app.error
    study = app.session_state.backtests
    assert len(study.metrics) == 24
    assert study.settings["rebalance_every"] == 42
    assert study.settings["rolling_window"] == 84
    assert study.settings["cost_bps"] == 25
    assert len(calls) == 2
    app.selectbox(key="comparison_method").select("Rolling window").run()
    app.segmented_control(key="risk_profile").set_value("Extreme").run()
    # The main chart and the result tile show the selected rule and level, not another strategy.
    selected = study.equity["Rolling window · Extreme"]
    np.testing.assert_allclose(chart_values(figures(app)[0]["data"][0]["y"]), selected * 10_000)
    assert f'<span class="pl-num">{selected.iloc[-1] * 10_000:,.0f}</span>' in html_text(app)
    app.segmented_control(key="comparison_chart").set_value("Drawdown").run()
    assert not app.exception
    assert len(calls) == 2
    assert [trace["name"] for trace in figures(app)[0]["data"]] == ["Your rule"]
    np.testing.assert_allclose(chart_values(figures(app)[0]["data"][0]["y"]), selected / selected.cummax() - 1)
    pd.testing.assert_frame_equal(app.session_state.backtests.metrics, study.metrics)
    run_comparison(app)
    assert len(calls) == 2
    app.number_input(key="cost_bps").set_value(0.)
    run_comparison(app)
    assert len(calls) == 3
    assert app.session_state.backtests.metrics.total_cost.eq(0).all()


@pytest.mark.parametrize("study_fails", [False, True])
def test_build_computes_market_comparison_and_evidence_once(monkeypatch, market_prices, study_fails):
    from efficient_frontier import backtest, benchmarks, evidence

    benchmark_calls, evidence_calls = [], []
    original_comparison, original_evidence = benchmarks.compare_benchmarks, evidence.analyze_evidence

    def compare(benchmark_prices, prices, analysis, study=None, latest_profiles=None, **kwargs):
        benchmark_calls.append(study is not None)
        return original_comparison(benchmark_prices, prices, analysis, study, latest_profiles, **kwargs)

    def estimate(*args, **kwargs):
        evidence_calls.append(kwargs.get("delayed_entry", False))
        return original_evidence(*args, **kwargs)

    def failed_study(*args, **kwargs):
        raise RuntimeError("Fixture market test failure.")

    monkeypatch.setattr(benchmarks, "compare_benchmarks", compare)
    monkeypatch.setattr(evidence, "analyze_evidence", estimate)
    if study_fails:
        monkeypatch.setattr(backtest, "run_backtests", failed_study)
    app = market_app(monkeypatch, market_prices)
    assert not app.exception
    assert benchmark_calls == evidence_calls == [not study_fails]
    assert app.session_state.benchmarks.holdout_equity is not None
    assert (app.session_state.benchmarks.backtest_equity is None) == study_fails
    assert app.session_state.evidence.settings["delayed_entry"] is not study_fails


def test_failed_comparison_preserves_portfolios_and_clears_stale_derived_state(monkeypatch, market_prices):
    app = run_comparison(market_app(monkeypatch, market_prices))
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96
    original = app.session_state.latest_profiles.frontier_weights.copy()
    app.number_input(key="rolling_window").set_value(1)
    app.button(key="run_comparison").click().run()
    assert not app.exception
    assert any("rolling_window" in item.value for item in app.error)
    assert "backtests" not in app.session_state
    assert "evidence" not in app.session_state or app.session_state.evidence is None
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, original)
    assert "result" in app.session_state


def test_export_is_built_on_click_from_current_results(monkeypatch):
    from streamlit.runtime.media_file_manager import MediaFileManager
    from efficient_frontier import presentation

    payloads, deferred = [], {}
    original, register = presentation.report_zip, MediaFileManager.add_deferred

    def export(*args, **kwargs):
        payloads.append(original(*args, **kwargs))
        return payloads[-1]

    def add_deferred(manager, data_callable, *args, **kwargs):
        file_id = register(manager, data_callable, *args, **kwargs)
        deferred[file_id] = data_callable
        return file_id

    monkeypatch.setattr(presentation, "report_zip", export)
    monkeypatch.setattr(MediaFileManager, "add_deferred", add_deferred)
    app = demo_app()
    assert not app.exception and payloads == []
    assert [button.key for button in app.get("download_button")] == ["export"]
    app.segmented_control(key="risk_profile").set_value("Low").run()
    assert payloads == []
    assert deferred[app.get("download_button")[0].proto.deferred_file_id]() is payloads[0]
    assert len(payloads) == 1
    with zipfile.ZipFile(BytesIO(payloads[0])) as archive:
        exported = pd.read_csv(archive.open("latest_profile_weights.csv"), index_col=0)
        expected = pd.DataFrame({name: value.weights for name, value in app.session_state.latest_profiles.portfolios.items()}).rename_axis("ticker")
        pd.testing.assert_frame_equal(exported, expected)
        assert len(pd.read_csv(archive.open("backtest_metrics.csv"), index_col=0)) == 24
        assert json.loads(archive.read("metadata.json"))["report_view"] == {"risk_level": "Low", "rule": "Expanding window"}


@pytest.mark.parametrize("frequency,periods", [("Daily calendar prices", 365), ("Weekly", 52), ("Monthly", 12), ("Custom", 48.5)])
def test_csv_frequency_scales_estimates_without_resampling(monkeypatch, market_prices, frequency, periods):
    upload = BytesIO(market_prices.to_csv().encode())
    upload.name = "portfolio.csv"
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: upload if kwargs["key"] == "prices_upload" else None)
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
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


def test_monthly_csv_labels_count_observations_and_do_not_claim_daily_returns(monkeypatch, market_prices):
    prices = market_prices.iloc[:180].set_axis(pd.date_range("2010-01-31", periods=180, freq="ME", name="Date"))
    upload = BytesIO(prices.to_csv().encode())
    upload.name = "monthly.csv"
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: upload if kwargs["key"] == "prices_upload" else None)
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
    app.selectbox(key="frequency").select("Monthly").run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.number_input(key="rebalance_every").label == "Trade every (price rows)"
    assert app.number_input(key="rolling_window").label == "Recent history for “Refit on recent prices” (return observations)"
    assert "The tests need at least 60 paired return observations." in html_text(app)
    assert not any("daily return pairs" in item.value for item in app.markdown)
    summary = next(item for item in app.dataframe if "Inference observations" in item.value.columns)
    assert json.loads(summary.proto.columns)["Inference observations"]["label"] == "Paired observations"
    windows = next(item for item in app.dataframe if "observations" in item.value.columns and "benchmark" in item.value.columns)
    assert json.loads(windows.proto.columns)["observations"]["label"] == "Observations"


def test_market_costs_charts_and_evidence_preserve_all_96_tests(monkeypatch, market_prices):
    app = market_app(monkeypatch, market_prices)
    assert not app.exception and not app.error
    assert app.session_state.evidence.settings["delayed_entry"] is True
    assert app.session_state.evidence.settings["multiple_testing_tests"] == len(app.session_state.backtests.metrics) * 4
    research_view(app, "Efficient frontiers")
    comparison = app.session_state.benchmarks
    expected_points = {"Risk frontier": (comparison.latest_estimates, "worst_window_return"),
                       "Classic frontier": (comparison.training_estimates, "expected_return")}
    names = {"S&P 500 (SPY)", "Nasdaq-100 (QQQ)"}
    found = set()
    for figure in figures(app):
        title = figure["layout"].get("title", {}).get("text")
        for trace in figure["data"]:
            name = html.unescape(trace.get("name", ""))
            if name in names and title in expected_points:
                estimates, objective = expected_points[title]
                np.testing.assert_allclose(chart_values(trace["y"]), [estimates.loc[name, objective]])
                found.add(title)
    assert found == set(expected_points)
    app.segmented_control(key="view").set_value("Portfolio").run()
    app.number_input(key="cost_bps").set_value(35.)
    run_comparison(app)
    assert not app.exception and not app.error
    comparison, evidence, study = app.session_state.benchmarks, app.session_state.evidence, app.session_state.backtests
    assert len(study.metrics) == 24 and len(evidence.summary) == 48
    assert evidence.settings["multiple_testing_tests"] == 96
    assert evidence.settings["delayed_entry"] is True
    pd.testing.assert_index_equal(comparison.backtest_equity.index, study.equity.index)
    np.testing.assert_allclose(comparison.backtest_equity.iloc[1], 1 / 1.0035)
    np.testing.assert_allclose(comparison.backtest_metrics.total_cost, 1 - 1 / 1.0035)
    assert app.selectbox(key="comparison_method").value == "Expanding window"
    assert {html.unescape(trace["name"]) for trace in figures(app)[0]["data"]} == names | {"Your rule"}
    original = evidence.summary.copy()
    app.selectbox(key="comparison_method").select("Buy and hold").run()
    app.segmented_control(key="risk_profile").set_value("Extreme").run()
    app.segmented_control(key="comparison_chart").set_value("Relative to markets").run()
    assert not app.exception
    assert len(figures(app)[0]["data"]) == 2
    pd.testing.assert_frame_equal(app.session_state.evidence.summary, original)
    research_view(app, "All backtests")
    app.multiselect(key="chart_strategies").set_value(["Rolling window · Low"]).run()
    # Research names each engine strategy with its rule label. The multiselect value stays the engine key.
    assert {html.unescape(trace["name"]) for trace in figures(app)[0]["data"]} == names | {"Refit on recent prices · Low"}
    app.multiselect(key="chart_strategies").set_value([]).run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state.backtests.metrics, study.metrics)
    pd.testing.assert_frame_equal(app.session_state.evidence.summary, original)
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96


def test_overview_answers_follow_the_selected_rule_and_level(monkeypatch, market_prices):
    """What it holds, the market answer, the 95% ranges and the stretch table show the selected rule and level only."""
    prices = market_prices.copy()
    # A late lead for FUND_X makes the answer differ between rules and levels.
    prices["FUND_X"] *= np.exp(.0005 * np.maximum(np.arange(len(prices)) - 252, 0))
    app = market_app(monkeypatch, prices)
    assert not app.exception and not app.error
    latest, evidence = app.session_state.latest_profiles, app.session_state.evidence

    def points(value):
        return "0.0" if round(value * 100, 1) == 0 else f"{value * 100:+.1f}".replace("-", "−")

    def numbers(selected):
        """The interval and stretch values that the evidence tiles must show for one rule and level."""
        ranges = [f"<b>{points(row.annual_advantage)}</b> points<small>range {points(row.advantage_ci_low)} to "
                  f"{points(row.advantage_ci_high)}</small>" for row in evidence.summary.xs(selected, level="strategy").itertuples()]
        stretches = [f"{row.relative_return:+.1%}".replace("-", "−") + f"<small>{'ahead' if row.relative_return > 0 else 'behind'}</small>"
                     for row in evidence.windows.loc[evidence.windows.strategy == selected].itertuples()]
        return ranges + stretches

    choices = [("Expanding window", "Medium"), ("Rolling window", "Extreme"), ("Rolling window", "Low")]
    expected = {f"{method} · {profile}": numbers(f"{method} · {profile}") for method, profile in choices}
    answers = {}
    for method, profile in choices:
        app.selectbox(key="comparison_method").select(method).run()
        app.segmented_control(key="risk_profile").set_value(profile).run()
        assert not app.exception
        page, selected = html_text(app), f"{method} · {profile}"
        weights = latest.portfolios[profile].weights
        held = weights[weights >= .0005].sort_values(ascending=False)
        rows = re.findall(r'<li class="pl-row"[^>]*><span class="pl-name"><b>([^<]+)</b>.*?<span class="pl-pct">([^<]+)</span>', page)
        assert rows == [(symbol, f"{weight:.1%}") for symbol, weight in held.items()]
        heading = page.split('id="pl-market"')[1].split("</h2>")[0].split('tabindex="-1">')[1]
        answers[selected] = html.unescape(re.sub("<[^>]+>", "", heading))
        assert all(value in page for value in expected[selected])
        assert not any(value in page for other, values in expected.items() if other != selected for value in values)
    assert answers == {"Expanding window · Medium": "Partly. It beat the S&P 500 but trailed the Nasdaq-100.",
                       "Rolling window · Extreme": "Yes. It beat both markets in this test.",
                       "Rolling window · Low": "Partly. It beat the S&P 500 but trailed the Nasdaq-100."}


def test_failed_market_test_in_a_build_keeps_the_holdout_market_results(monkeypatch, market_prices):
    app = market_app(monkeypatch, market_prices)
    app.number_input(key="rolling_window").set_value(len(app.session_state.result[0].train_returns) - 5)
    run_comparison(app)
    assert not app.exception and not app.error and app.session_state.study_error is None
    # A shorter first fit makes the stored rolling window too long, so the market test fails inside the build.
    open_setup(app).slider(key="train_pct").set_value(50).run()
    app.button(key="build").click().run()
    assert not app.exception
    assert app.session_state.result[2]["train_fraction"] == .5
    assert "rolling_window" in app.session_state.study_error
    assert any("rolling_window" in item.value for item in app.error)
    assert "backtests" not in app.session_state
    assert app.session_state.benchmarks.backtest_equity is None
    assert app.session_state.benchmarks.holdout_equity is not None
    assert app.session_state.evidence.settings["delayed_entry"] is False
    research_view(app, "Efficient frontiers")
    names = {html.unescape(trace.get("name", "")) for figure in figures(app) for trace in figure["data"]}
    assert {"S&P 500 (SPY)", "Nasdaq-100 (QQQ)"} <= names
    assert any("Diamonds are SPY and QQQ" in item.value for item in app.caption)
    app.segmented_control(key="view").set_value("Portfolio").run()
    app.number_input(key="rolling_window").set_value(0)
    run_comparison(app)
    assert not app.exception and not app.error
    assert app.session_state.study_error is None
    assert app.session_state.evidence.settings["delayed_entry"] is True


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
    open_setup(app).checkbox(key="compare_market").set_value(False)
    app.button(key="build").click().run()
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert len(calls) == 1
    open_setup(app).checkbox(key="compare_market").set_value(True)
    app.selectbox(key="price_currency").select("Other currency")
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert "USD" in app.session_state.result[2]["benchmarks_note"]
    assert len(calls) == 1
    open_setup(app).selectbox(key="price_currency").select("USD")
    app.button(key="build").click().run()
    assert app.session_state.benchmarks is not None
    available[0] = False
    st.cache_data.clear()
    open_setup(app).button(key="build").click().run()
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
    from efficient_frontier import benchmarks, presentation

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
    assert not app.exception and not app.error
    assert attempts == [1]
    assert len(app.session_state.backtests.metrics) == 24
    assert "Fixture comparison failure" in app.session_state.result[2]["benchmarks_note"]
    assert "benchmarks" not in app.session_state or app.session_state.benchmarks is None
    assert "evidence" not in app.session_state or app.session_state.evidence is None

    run_comparison(app)
    assert not app.exception and not app.error
    assert attempts == [1, 1]
    assert app.session_state.benchmarks.backtest_equity is not None
    assert app.session_state.evidence.settings["multiple_testing_tests"] == 96
    assert "benchmarks_note" not in app.session_state.result[2]
    state = app.session_state
    report = presentation.report_zip(*state.result, state.backtests, state.latest_profiles, state.benchmarks, state.evidence)
    with zipfile.ZipFile(BytesIO(report)) as archive:
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
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == [] and app.session_state.benchmarks is None
    uploads["benchmark_upload"] = benchmark_upload
    open_setup(app).checkbox(key="download_benchmarks").set_value(True)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert calls == []
    assert app.session_state.result[2]["benchmark_source"] == "CSV: benchmarks.csv"
    uploads["benchmark_upload"] = None
    open_setup(app).button(key="clear_benchmark_file").click().run()
    app.checkbox(key="download_benchmarks").set_value(False)
    app.button(key="build").click().run()
    assert app.session_state.benchmarks is None and app.session_state.evidence is None
    assert calls == []
    open_setup(app).checkbox(key="download_benchmarks").set_value(True)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert len(calls) == 1
    pd.testing.assert_index_equal(calls[0], prices.index)
    assert list(app.session_state.result[0].mean_returns.index) == list(prices.columns)


def test_research_risk_and_trade_tables_match_complete_results():
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
    from efficient_frontier.risk import risk_contributions
    from efficient_frontier.story import strategy_label

    def index_label(frame):
        return json.loads(frame.proto.columns).get("_index", {}).get("label")

    app = run_comparison(demo_app())
    study = app.session_state.backtests
    research_view(app, "Risk breakdown")
    app.selectbox(key="risk_portfolio").select("Equal weight").run()
    result = app.session_state.result[0]
    table = next(item for item in app.dataframe if "Share of portfolio variance" in item.value.columns)
    expected = pd.DataFrame({"Allocation weight": result.portfolios["Equal weight"].weights,
                             "Share of portfolio variance": risk_contributions(result)["Equal weight"]}).sort_values("Allocation weight", ascending=False)
    pd.testing.assert_frame_equal(table.value, expected)
    assert index_label(table) == "Symbol"
    research_view(app, "All backtests")
    selected = "Rolling window · Maximum Sharpe"
    app.selectbox(key="holding_strategy").select(selected).run()
    assert not app.exception
    tables = {name: next(item for item in app.dataframe if item.value.equals(frame)) for name, frame in (
        ("holdings", study.holdings[selected]), ("allocations", study.allocations[selected]), ("trades", study.trades[selected]),
        ("metrics", study.metrics.rename(index=strategy_label)))}
    assert "Holdings of Refit on recent prices · Maximum Sharpe" in html_text(app)
    # Plain headers, and trades in the units of the overview: 10,000 at the start and day-month-year dates.
    assert (index_label(tables["metrics"]), index_label(tables["holdings"])) == ("Strategy", "Symbol")
    first = study.trades[selected].iloc[0]
    shown = convert_arrow_bytes_to_pandas_df(tables["trades"].proto.arrow_data.styler.display_values).iloc[0]
    assert (shown.cost, shown.nav_before) == (f"{first.cost * 10_000:,.2f}", f"{first.nav_before * 10_000:,.0f}")
    assert re.fullmatch(r"\d{1,2} [A-Z][a-z]{2} \d{4}", shown.train_end)


def test_research_holdout_ratios_are_not_formatted_as_percentages():
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

    app = research_view(demo_app(), "Efficient frontiers")
    assert not any("Diamonds" in item.value or "SPY and QQQ markers" in item.value for item in app.caption)  # Demo has no markets.
    table = next(item for item in app.dataframe if "sortino" in item.value.columns)
    display = convert_arrow_bytes_to_pandas_df(table.proto.arrow_data.styler.display_values)
    for column in ("sharpe", "sortino", "calmar"):
        assert not display[column].str.contains("%", regex=False).any()
    assert display.total_return.str.contains("%", regex=False).all()


def test_classic_risk_choice_links_frontier_weights_and_holdout_without_refitting(monkeypatch, offline_app):
    from efficient_frontier import core, profiles

    app = research_view(demo_app(), "Efficient frontiers")
    analysis = app.session_state.result[0]
    assert app.segmented_control(key="classic_risk_profile").options == ["Low", "Medium", "High"]
    assert app.segmented_control(key="classic_risk_profile").value == "Medium"

    def no_solve(*args, **kwargs):
        raise AssertionError("Changing the classic risk level must not refit any model.")

    monkeypatch.setattr(core, "_solve", no_solve)
    monkeypatch.setattr(profiles, "_solve", no_solve)
    for level in ("Low", "High", "Medium"):
        app.segmented_control(key="classic_risk_profile").set_value(level).run()
        assert not app.exception and not app.error
        assert app.session_state.risk_profile == "Medium"
        charts = {figure["layout"]["title"]["text"]: figure for figure in figures(app)}
        points = {trace["name"]: trace for trace in charts["Classic frontier"]["data"]}
        assert points[level]["marker"]["size"] > points[next(name for name in ("Low", "Medium", "High") if name != level)]["marker"]["size"]
        np.testing.assert_allclose(chart_values(points[level]["x"]), [analysis.portfolios[level].volatility])
        np.testing.assert_allclose(chart_values(points[level]["y"]), [analysis.portfolios[level].expected_return])
        path = charts["Classic holdout"]["data"]
        assert [trace["name"] for trace in path] == [level]
        np.testing.assert_allclose(chart_values(path[0]["y"]), analysis.equity[level] * 10_000)
        weights = next(item.value for item in app.dataframe if list(item.value.columns) == ["Weight"])
        np.testing.assert_allclose(weights.Weight, analysis.portfolios[level].weights.loc[weights.index])
        assert len(weights) == len(analysis.mean_returns)
        metrics = next(item.value for item in app.dataframe if "sortino" in item.value.columns)
        pd.testing.assert_frame_equal(metrics, analysis.holdout_metrics.loc[[level]])
    assert offline_app == []
    captions = " ".join(item.value for item in app.caption)
    assert "not compounded growth or a forecast" in captions
    assert "same initial weights" in captions
    assert "weakest of three past stretches" in html_text(app)


def test_classic_higher_fit_return_can_lose_to_etfs_on_the_later_dates(monkeypatch):
    rng = np.random.default_rng(712)
    returns = rng.normal([.003, .0005, .001], [.015, .009, .012], (90, 3))
    returns[:63] -= returns[:63].mean(axis=0)
    returns[:63] += [.003, .0005, .001]
    returns[63:] -= returns[63:].mean(axis=0)
    returns[63:] += [-.006, .001, .002]
    prices = pd.DataFrame(np.vstack([np.ones(3), np.cumprod(1 + returns, axis=0)]) * 100,
                          index=pd.bdate_range("2024-01-01", periods=91, name="Date"), columns=["FUND_X", "SPY", "QQQ"])
    app = research_view(market_app(monkeypatch, prices), "Efficient frontiers")
    app.segmented_control(key="classic_risk_profile").set_value("High").run()
    assert not app.exception and not app.error
    analysis, benchmarks = app.session_state.result[0], app.session_state.benchmarks
    assert analysis.portfolios["High"].expected_return > benchmarks.training_estimates.expected_return.max()
    assert analysis.holdout_metrics.loc["High", "total_return"] < benchmarks.holdout_metrics.total_return.min()
    charts = {figure["layout"]["title"]["text"]: figure for figure in figures(app)}
    paths = {html.unescape(trace["name"]): trace for trace in charts["Classic holdout"]["data"]}
    assert set(paths) == {"High", "S&P 500 (SPY)", "Nasdaq-100 (QQQ)"}
    for name, equity in pd.concat([analysis.equity[["High"]], benchmarks.holdout_equity], axis=1).items():
        np.testing.assert_allclose(chart_values(paths[name]["y"]), equity * 10_000)
        assert pd.DatetimeIndex(paths[name]["x"]).equals(equity.index)
        assert chart_values(paths[name]["y"])[0] == 10_000
    captions = " ".join(item.value for item in app.caption)
    for date_value in (analysis.train_returns.index[0], analysis.train_returns.index[-1],
                       analysis.test_returns.index[0], analysis.test_returns.index[-1]):
        assert f"{date_value.day} {date_value:%b %Y}" in captions
    assert any("does not show which mix won on the later test dates" in item.value for item in app.markdown)


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
    heatmap = figures(app)[0]["layout"]
    assert heatmap["xaxis"]["dtick"] == heatmap["yaxis"]["dtick"] == 1  # Every row and column is labelled.
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
    app = open_setup(demo_app())
    app.number_input(key="risk_free").set_value(100.)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert "Maximum Sharpe" not in app.session_state.result[0].portfolios
    for portfolio in app.session_state.latest_profiles.portfolios.values():
        np.testing.assert_allclose(portfolio.weights, [1.])
    page = html_text(app)
    assert "Risk is relative to this one asset." in page and "compare mixes of this one asset only" in page
    assert "1 assets" not in page and not any("1 assets" in item.value for item in app.caption)
    research_view(app, "Methodology")
    assert not app.exception
    assert any("Maximum Sharpe omitted" in item.value for item in app.markdown)
    assert any("It does not measure every form of risk." in item.value for item in app.markdown)


def test_missing_csv_clears_old_result_without_network(offline_app):
    app = open_setup(demo_app())
    app.radio(key="market").set_value("Upload CSV").run()
    assert app.session_state.result[2]["universe"] == "Demo"
    app.button(key="build").click().run()
    assert not app.exception
    assert any("Choose an adjusted-price CSV" in item.value for item in app.error)
    assert "result" not in app.session_state
    assert offline_app == []


def test_welcome_has_no_work_and_one_click_demo(offline_app):
    app = app_start()
    assert not app.exception
    assert offline_app == [] and "result" not in app.session_state
    assert {"build", "try_demo", "edit_setup"} <= {button.key for button in app.button}
    app.button(key="try_demo").click().run()
    assert not app.exception and not app.error
    assert app.session_state.setup["market"] == "Demo"
    assert app.session_state.result[2]["universe"] == "Demo"
    assert len(app.session_state.backtests.metrics) == 24
    assert offline_app == []


def test_drawer_cancel_discards_draft_and_reopen_seeds_applied_values():
    app = open_setup(demo_app())
    app.slider(key="train_pct").set_value(50).run()
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="cancel_setup").click().run()
    assert not app.exception
    assert app.session_state.setup_open is False
    assert "train_pct" not in app.session_state
    assert app.session_state.setup["train_pct"] == 70
    assert app.session_state.result[2]["train_fraction"] == .7
    open_setup(app)
    assert app.slider(key="train_pct").value == 70
    assert not any("differs from this result" in item.value for item in app.info)
    assert not app.warning and not app.exception


def test_drawer_widgets_default_to_the_applied_setup_without_session_state_writes():
    app = open_setup(demo_app())
    app.number_input(key="risk_free").set_value(3.).run()
    app.slider(key="train_pct").set_value(60).run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert "risk_free" not in app.session_state and "train_pct" not in app.session_state
    # The drawer reads the applied setup as widget defaults. Nothing seeds widget keys, so a run that a second
    # click interrupts cannot leave the browser with bare widget defaults.
    open_setup(app)
    assert app.number_input(key="risk_free").value == 3. and app.slider(key="train_pct").value == 60
    assert app.radio(key="market").value == "Demo" and app.checkbox(key="compare_market").value is True
    assert not any("differs from this result" in item.value for item in app.info)
    assert not app.warning and not app.exception


def test_keys_set_risk_view_and_jump():
    app = demo_app()
    app.button(key="key_highest").click().run()
    assert app.segmented_control(key="risk_profile").value == "Extreme"
    app.button(key="key_research").click().run()
    assert app.segmented_control(key="view").value == "Research"
    assert app.session_state.risk_profile == "Extreme"
    app.button(key="key_low").click().run()
    assert app.session_state.risk_profile == "Low"
    app.button(key="key_evidence").click().run()
    assert app.segmented_control(key="view").value == "Portfolio"
    assert app.segmented_control(key="risk_profile").value == "Low"
    assert app.session_state.scroll_nonce == 2
    assert not app.exception


def test_test_settings_rerun_is_atomic_on_failure():
    app = demo_app()
    weights = app.session_state.latest_profiles.frontier_weights.copy()
    analysis = app.session_state.result[0]
    app.number_input(key="rolling_window").set_value(1)
    run_comparison(app)
    assert not app.exception
    assert any("rolling_window" in item.value for item in app.error)
    assert "rolling_window" in app.session_state.study_error
    assert "backtests" not in app.session_state
    assert app.session_state.result[0] is analysis
    pd.testing.assert_frame_equal(app.session_state.latest_profiles.frontier_weights, weights)
    app.number_input(key="rolling_window").set_value(0)
    run_comparison(app)
    assert not app.exception and not app.error
    assert app.session_state.study_error is None
    assert len(app.session_state.backtests.metrics) == 24


def test_failed_build_shows_error_state_and_change_reopens_the_applied_setup():
    app = open_setup(demo_app())
    app.checkbox(key="use_cap").set_value(True).run()
    app.number_input(key="cap").set_value(5.)
    app.button(key="build").click().run()
    assert not app.exception
    assert any("infeasible" in item.value.lower() for item in app.error)
    assert "These settings have no solution." in html_text(app)
    assert "A 5% limit needs at least 20 assets." in html_text(app)
    # A retry would fail the same way, so the primary action opens the drawer instead.
    assert "build" not in {button.key for button in app.button}
    assert app.button(key="fix_setup").label == "Change the study"
    assert "result" not in app.session_state
    app.button(key="fix_setup").click().run()
    assert app.session_state.setup_open is True
    assert app.checkbox(key="use_cap").value is True and app.number_input(key="cap").value == 5.
    app.number_input(key="cap").set_value(30.)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.result[2]["max_weight"] == .3


@pytest.mark.parametrize("failure,headline,action", [
    (ValueError("Yahoo returned no prices for: ZZZQQX. Check these symbols and their available history, or upload a CSV."),
     "Yahoo Finance had no prices for these symbols.", "fix_setup"),
    (ValueError("Yahoo price download failed. Check your connection and ticker symbols, retry later, or upload an adjusted-price CSV."),
     "We could not reach Yahoo Finance.", "build"),
    (OSError("Network is unreachable"), "We could not reach Yahoo Finance.", "build"),
])
def test_build_error_names_its_cause_and_offers_a_useful_action(monkeypatch, failure, headline, action):
    from efficient_frontier import data

    def download(*args):
        raise failure

    monkeypatch.setattr(data, "download_prices", download)
    app = open_setup(app_start())
    app.radio(key="market").set_value("My tickers").run()
    app.text_area(key="ticker_text").set_value("ZZZQQX")
    app.button(key="build").click().run()
    assert not app.exception
    assert any(str(failure) in item.value for item in app.error)
    assert f'<h1 class="pl-display">{headline}</h1>' in html_text(app)
    actions = {button.key for button in app.button}
    assert action in actions and ({"build", "fix_setup"} - {action}).isdisjoint(actions)
    assert app.button(key=action).label == {"build": "Try again", "fix_setup": "Change the study"}[action]
    app.button(key=action).click().run()
    if action == "fix_setup":
        assert app.radio(key="market").value == "My tickers" and app.text_area(key="ticker_text").value == "ZZZQQX"
    else:
        assert any(str(failure) in item.value for item in app.error)
