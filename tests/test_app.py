from datetime import date
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
    from efficient_frontier import companies, data, risk_reference, universe

    st.cache_data.clear()
    monkeypatch.setattr(risk_reference, "load_qqq_volatility_reference", lambda: {
        "volatility": .20, "first_return_date": "2025-10-17", "last_return_date": "2026-10-16",
        "observations": 251, "symbol": "QQQ", "source": "Yahoo Finance adjusted close", "periods_per_year": 252,
    })
    demo = data.demo_prices().iloc[:240].copy()
    monkeypatch.setattr(data, "demo_prices", lambda: demo.copy())
    calls = []

    def blocked(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("An AppTest attempted an unmocked market download.")

    def company_profile(symbol):
        calls.append(("company profile", symbol))
        return {"name": f"Company {symbol}", "description": f"Provides products and services for {symbol}.",
                "source_url": f"https://example.com/{symbol}"}

    monkeypatch.setattr(companies, "fetch_company_profile", company_profile)
    for module, names in ((data, ("download_prices", "download_benchmarks")),
                          (universe, ("fetch_nasdaq100", "fetch_sp500_nasdaq100", "download_universe_prices"))):
        for name in names:
            monkeypatch.setattr(module, name, blocked)
    yield calls
    st.cache_data.clear()


@pytest.fixture
def market_prices():
    rng = np.random.default_rng(902)
    returns = rng.normal(.0005, .008, (240, 12)) + rng.normal(0, .004, (240, 1))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)),
                        index=pd.bdate_range("2021-01-04", periods=240, name="Date"),
                        columns=[f"STOCK_{number:02d}" for number in range(12)])


@pytest.fixture
def benchmark_prices(market_prices):
    rng = np.random.default_rng(903)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal([.0004, .0006], [.009, .013],
                                                                         (len(market_prices), 2)), axis=0)),
                        index=market_prices.index, columns=["SPY", "QQQ"])


@pytest.fixture
def universe_data(monkeypatch, market_prices, benchmark_prices):
    from efficient_frontier import universe

    symbols = [*market_prices.columns, "MISSING"]
    members = pd.DataFrame({"symbol": symbols, "name": [f"Company {symbol}" for symbol in symbols],
                            "industry": ["Unknown"] * len(symbols)})
    coverage = members.assign(status=["included"] * 12 + ["excluded"], reason=[""] * 12 + ["Incomplete prices"])
    metadata = {"universe_requested": 13, "universe_included": 12, "universe_excluded": 1,
                "universe_coverage": coverage.to_dict(orient="records"),
                "universe_limitation": universe.UNIVERSE_LIMITATION,
                "universe_source_date": "2026-10-01", "universe_retrieved_at": "2026-10-16T09:00:00+00:00"}
    snapshot = universe.UniverseSnapshot(members, universe.NASDAQ100_URL, "2026-10-01", "2026-10-16T09:00:00+00:00")
    loaded = universe.UniversePrices(market_prices, coverage, benchmark_prices, metadata)
    calls = []

    def download(actual_snapshot, start, end, *, progress=None):
        calls.append((start, end))
        pd.testing.assert_frame_equal(actual_snapshot.members, members)
        if progress is not None:
            progress(13, 13, "All fixture securities checked")
        return loaded

    def combined():
        calls.append("combined members")
        return snapshot

    def nasdaq():
        calls.append("Nasdaq members")
        return snapshot

    monkeypatch.setattr(universe, "fetch_sp500_nasdaq100", combined)
    monkeypatch.setattr(universe, "fetch_nasdaq100", nasdaq)
    monkeypatch.setattr(universe, "download_universe_prices", download)
    return SimpleNamespace(prices=market_prices, coverage=coverage, metadata=metadata, calls=calls)


def app_start():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def open_setup(app):
    return app.button(key="edit_setup").click().run()


def demo_app():
    app = open_setup(app_start())
    app.radio(key="market").set_value("Demo").run()
    return app.button(key="build").click().run()


def html_text(app):
    return "\n".join(item.proto.body for item in app.get("html"))


def upload(frame, name):
    result = BytesIO(frame.to_csv().encode())
    result.name = name
    return result


def csv_app(monkeypatch, prices, benchmarks=None):
    uploads = {"prices_upload": upload(prices, "portfolio.csv"),
               "benchmark_upload": upload(benchmarks, "benchmarks.csv") if benchmarks is not None else None}
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploads[kwargs["key"]])
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
    return app.button(key="build").click().run(), uploads


def test_combined_default_and_risk_inputs_do_no_work_before_submit(monkeypatch, universe_data, offline_app):
    from efficient_frontier import selection

    def unexpected(*args, **kwargs):
        raise AssertionError("Opening the app or editing a draft must not fit a candidate.")

    monkeypatch.setattr(selection, "run_selection", unexpected)
    app = app_start()
    assert not app.exception
    assert app.button(key="build").label == "Build portfolio"
    assert app.button(key="edit_setup").label == "Settings"
    assert "try_demo" not in {item.key for item in app.button}
    assert len(app.get("latex")) == 3
    assert html_text(app).count("Efficient Frontier Analysis for US Equities") == 1
    assert "Constrained Mean–Variance Optimization" in html_text(app)
    assert 'class="pl-wordmark"' not in html_text(app)
    descriptions = "\n".join(item.value for item in app.markdown)
    for definition in ("sample mean returns", "annual covariance matrix", "initial weight vector",
                       "fraction in stock", "optimal weights", "vector of ones", "transpose",
                       "starting cash", "entry observation", "fee per unit bought", "sum across stocks"):
        assert definition in descriptions
    assert "Choose a risk budget" not in html_text(app)
    assert app.number_input(key="preview_target_vol").value == 20.
    assert app.number_input(key="preview_cap").value == 10.
    assert app.session_state.setup["market"] == "S&P 500 + Nasdaq-100"
    assert "result" not in app.session_state and "selection" not in app.session_state
    assert not app.get("download_button")
    app.number_input(key="preview_target_vol").set_value(18.).run()
    app.number_input(key="preview_cap").set_value(12.).run()
    open_setup(app)
    assert app.radio(key="market").value == "S&P 500 + Nasdaq-100"
    assert app.radio(key="market").options == ["S&P 500 + Nasdaq-100", "Nasdaq-100", "My tickers",
                                               "Original 60 holdings", "Upload CSV", "Demo"]
    assert app.number_input(key="target_vol").value == 18. and app.number_input(key="cap").value == 12.
    app.segmented_control(key="history").set_value("10 years").run()
    assert not app.exception and universe_data.calls == offline_app == []


def test_prebuild_target_and_cap_reach_the_real_selection(monkeypatch, universe_data):
    from efficient_frontier import selection

    original, calls = selection.run_selection, []

    def calculate(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "run_selection", calculate)
    app = app_start()
    app.number_input(key="preview_target_vol").set_value(18.).run()
    app.number_input(key="preview_cap").set_value(12.).run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert len(calls) == 1 and calls[0]["target_volatility"] == .18 and calls[0]["max_weight"] == .12
    assert universe_data.calls[0] == "combined members" and len(universe_data.calls) == 2
    assert app.session_state.result[0] is None
    assert app.session_state.result[2]["target_volatility"] == .18
    assert app.session_state.selection.settings["max_weight"] == .12
    assert "backtests" not in app.session_state and "latest_profiles" not in app.session_state
    pd.testing.assert_frame_equal(app.session_state.result[1], universe_data.prices)


def test_qqq_default_is_dated_and_does_not_overwrite_user_edits(monkeypatch, universe_data):
    from efficient_frontier import risk_reference

    calls = []

    def reference():
        calls.append(True)
        return {"volatility": .198350420824, "last_return_date": "2026-10-16"}

    monkeypatch.setattr(risk_reference, "load_qqq_volatility_reference", reference)
    app = app_start()
    assert not app.exception
    assert app.number_input(key="preview_target_vol").value == 19.84
    assert any("16 Oct 2026" in item.value and "QQQ 19.84%" in item.value for item in app.caption)
    app.number_input(key="preview_target_vol").set_value(17.25).run()
    open_setup(app)
    assert app.number_input(key="target_vol").value == 17.25
    app.button(key="cancel_setup").click().run()
    assert app.number_input(key="preview_target_vol").value == 17.25
    assert calls == [True]
    app.number_input(key="preview_target_vol").set_value(19.84).run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.selection.settings["target_volatility"] == pytest.approx(.1984)
    assert r"b=0.1984\;(19.84\%)" in "\n".join(item.proto.body for item in app.get("latex"))


def test_existing_manual_budget_equal_to_old_default_is_preserved(monkeypatch):
    from efficient_frontier import risk_reference

    saved = dict(app_start().session_state.setup)
    saved["market"] = "My tickers"
    monkeypatch.setattr(risk_reference, "load_qqq_volatility_reference", lambda: {
        "volatility": .198350420824, "last_return_date": "2026-10-16",
    })
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state.setup = saved
    app.session_state.setup_open = True
    app.run()
    assert not app.exception
    assert app.session_state.setup["target_vol"] == 20.
    assert app.number_input(key="target_vol").value == 20.


def test_unavailable_qqq_reference_keeps_manual_controls_and_math(monkeypatch):
    from efficient_frontier import risk_reference

    def unavailable():
        raise OSError("Network is unreachable")

    monkeypatch.setattr(risk_reference, "load_qqq_volatility_reference", unavailable)
    app = app_start()
    assert not app.exception
    assert len(app.get("latex")) == 3
    assert any("manual fallback, not a current Nasdaq-100 estimate" in item.value for item in app.caption)
    assert app.number_input(key="preview_target_vol").value == 20.
    app.number_input(key="preview_target_vol").set_value(23.).run()
    assert app.session_state.setup["target_vol"] == 23.


def test_demo_build_is_offline_with_an_explicit_feasible_cap(offline_app):
    app = demo_app()
    assert not app.exception and not app.error and offline_app == []
    study = app.session_state.selection
    assert app.session_state.setup["market"] == "Demo" and app.session_state.setup["cap"] == 20.
    assert study.settings["max_weight"] == .2
    assert study.settings["target_volatility"] == .2
    assert study.benchmark_prices is None and not study.supported
    assert any("synthetic" in item.value for item in app.info)
    assert study.latest[study.selected].weights.max() <= .2 + 1e-6
    assert study.latest[study.selected].weights.sum() == pytest.approx(1)
    assert study.settings["holding_policy"] == "buy_and_hold"
    assert "rebalance_every" not in study.settings
    for name in study.test.allocations:
        assert len(study.test.allocations[name]) == len(study.test.trades[name]) == 1
    initial = study.test.allocations[study.selected].iloc[0]
    final = study.test.holdings[study.selected]["end_weight"]
    assert not np.allclose(initial, final)
    holdings = "\n".join(item.proto.body for item in app.container(key="tile_holdings").get("html"))
    assert "Simulated " in holdings and "not an actual company" in holdings


def test_overview_shows_one_test_and_separate_latest_weights():
    app = demo_app()
    assert not app.exception
    page = html_text(app)
    nodes = list(app.main)
    positions = [next(i for i, node in enumerate(nodes) if node is app.container(key=key))
                 for key in ("tile_holdings", "tile_fitted_frontier", "tile_out_of_sample", "tile_final_test")]
    assert positions == sorted(positions)
    assert page.index("Latest fitted holdings") < page.index("Latest fitted frontier")
    assert page.index("Latest fitted frontier") < page.index("In sample versus out of sample")
    assert page.index("In sample versus out of sample") < page.index("Historical replay · buy and hold")
    assert page.index("Historical replay · buy and hold") < page.index("What drove the result?")
    assert page.index("What drove the result?") < page.index("Fit and replay use different holdings")
    assert "annual volatility budget.</h1>" not in page
    assert 'class="pl-context"' not in page
    assert "The latest fit includes the replay dates" in page
    assert "First historical fit" not in page
    assert "Historical volatility" in page and "ordinary sample covariance" in page
    assert len(app.get("plotly_chart")) == 3
    assert "validation" not in page.lower() and "blend" not in page.lower()
    holding_section = next(item for item in app.expander if item.label == "All weights, including negligible allocations")
    assert holding_section.proto.expanded is False
    assert not any('class="pl-holdings-grid"' in item.proto.body for panel in app.expander for item in panel.get("html"))
    assert any("These weights use all supplied prices" in item.value for item in app.caption)
    assert any("They form the selected point on the frontier below" in item.value for item in app.caption)
    assert any("the replay below uses a separate initial allocation" in item.value for item in app.markdown)
    assert app.segmented_control(key="view").options == ["Overview", "Data & method"]
    assert "risk_profile" not in {item.key for item in app.segmented_control}
    assert "comparison_method" not in {item.key for item in app.selectbox}
    assert not {"key_low", "key_medium", "key_highest", "open_historical", "run_legacy"} & {item.key for item in app.button}
    study = app.session_state.selection
    displayed = [item.value for item in app.dataframe if "annual_mean" in item.value.columns]
    assert len(displayed) == 1
    columns = ["annual_mean", "cagr", "volatility", "max_drawdown"]
    pd.testing.assert_frame_equal(displayed[0], study.test.metrics[columns].rename(
        index={study.selected: "Stock portfolio · buy and hold"}))
    assert list(study.latest) == [study.selected]


def test_mathematical_definitions_use_rendered_latex_and_applied_limits(universe_data):
    app = app_start()
    app.number_input(key="preview_target_vol").set_value(18.).run()
    app.number_input(key="preview_cap").set_value(12.).run()
    app.button(key="build").click().run()
    assert not app.exception
    assert "open_mathematics" not in {item.key for item in app.button}
    assert not any(item.label.startswith("Mathematics ·") for item in app.expander)
    assert "Mathematics ·" not in html_text(app)
    overview_equations = [item.proto.body for item in app.get("latex")]
    equations = "\n".join(overview_equations)
    # Adjacent raw strings must not turn spacing into unknown KaTeX commands.
    assert not re.search(r"\\(?:qquad|quad)[A-Za-z]", equations)
    assert r"\widehat\mu=A\bar r" in equations
    assert r"\widehat\Sigma=\frac{A}{m-1}" in equations
    assert r"b=0.18\;(18.00\%)" in equations
    assert r"h=0.12\;(12\%)" in equations
    assert r"V_e=\frac{V_{\mathrm{start}}}{1+c}" in equations
    assert r"c=0.001\;(0.10\%)" in equations
    assert r"V_t&=V_e\sum_i w_i g_{i,t}" in equations
    assert r"a_{i,t}=\frac{w_i g_{i,t}}{\sum_j w_j g_{j,t}}" in equations
    assert r"\frac{A}{N-1}" in equations
    assert r"\mathrm{CAGR}" in equations and r"\mathrm{MDD}" in equations
    assert r"\widehat\mu" not in html_text(app)
    text = "\n".join(item.value for item in app.markdown)
    assert "Neither limit triggers a trade" in text
    assert "no later refits, trades" in text
    assert "including the entry-fee interval" in text
    app.segmented_control(key="view").set_value("Data & method").run()
    assert not app.exception
    assert app.session_state.view == "Data & method"
    position = 0
    for equation in app.get("latex"):
        position = overview_equations.index(equation.proto.body, position) + 1
    assert not {"math_out_of_sample", "math_diagnosis"} & {item.key for item in app.container}
    assert not any(item.label.startswith("Mathematics ·") for item in app.expander)


def test_out_of_sample_section_precedes_replay_and_matches_walk_forward(universe_data):
    from efficient_frontier.out_of_sample import run_out_of_sample, portfolio_comparison
    from efficient_frontier.selection import BENCHMARK_NAMES

    app = app_start().button(key="build").click().run()
    assert not app.exception
    study = app.session_state.selection
    prices = app.session_state.result[1]
    expected = run_out_of_sample(study, prices)
    means = portfolio_comparison(expected)
    assert len(means) == len(expected.windows) == 5
    tile = app.container(key="tile_out_of_sample")
    nodes = list(app.main)
    replay_chart = app.container(key="tile_final_test").get("plotly_chart")[0]
    assert next(i for i, node in enumerate(nodes) if node is tile) < next(
        i for i, node in enumerate(nodes) if node is replay_chart)
    page = "\n".join(item.proto.body for item in tile.get("html"))
    assert "In sample versus out of sample" in page
    assert "each fit uses only earlier stock prices" in page
    assert f"Out of sample, the stock portfolio averaged {means.out_of_sample.mean():.1%} a year.".replace("-", "−") in page
    assert f"In sample, it averaged {means.in_sample.mean():.1%}.".replace("-", "−") in page
    assert f"{100 * means.change.mean():+.2f} pp".replace("-", "−") in page
    text = "\n".join(item.value for item in tile.markdown)
    lower = int((means.change < 0).sum())
    assert f"lower than the in-sample mean in {lower} of 5 windows" in text
    for name, label in ((BENCHMARK_NAMES["QQQ"], BENCHMARK_NAMES["QQQ"]), (BENCHMARK_NAMES["SPY"], BENCHMARK_NAMES["SPY"]),
                        ("Equal-weight stocks", "equal-weight stocks")):
        wins = int((means.out_of_sample > means[name]).sum())
        assert f"{label} in {wins} of 5" in text
    assert "average margin over them" in text
    assert list(tile.dataframe[0].value.columns[-2:]) == ["Equal weight in sample", "Equal weight out of sample"]
    table = tile.dataframe[0].value
    assert list(table.index) == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(table["In sample"], means.in_sample)
    np.testing.assert_allclose(table["Out of sample"], means.out_of_sample)
    for number, row in expected.windows.iterrows():
        assert table.loc[number, "Test dates"] == f"{row.test_start.date()} to {row.test_end.date()}"
    risk = portfolio_comparison(expected, "volatility")
    paths = next(panel for panel in tile.expander if panel.label == "Every path in each window").dataframe[0].value
    portfolio = paths.xs("Stock portfolio", level="Path")
    np.testing.assert_allclose(portfolio["In-sample volatility"], risk.in_sample)
    np.testing.assert_allclose(portfolio["Out-of-sample volatility"], risk.out_of_sample)
    np.testing.assert_allclose(paths.xs(BENCHMARK_NAMES["QQQ"], level="Path")["Out-of-sample mean"],
                               means[BENCHMARK_NAMES["QQQ"]])
    captions = "\n".join(item.value for item in tile.caption)
    assert "Window 3 uses the same fit as the replay" in captions
    assert "measured from the purchase close" in captions
    chart = tile.get("plotly_chart")[0]
    import base64

    def values(array):
        # Plotly serializes numeric arrays as base64 typed arrays.
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"]) if isinstance(array, dict) else array

    traces = {trace["name"]: trace for trace in json.loads(chart.proto.spec)["data"] if trace.get("name")}
    np.testing.assert_allclose(values(traces["Stock portfolio · in sample"]["x"]), means.in_sample)
    np.testing.assert_allclose(values(traces["Stock portfolio · out of sample"]["x"]), means.out_of_sample)
    equations = "\n".join(item.proto.body for item in tile.container(key="math_out_of_sample").get("latex"))
    assert r"\mu^{\rm in}_k=\frac{A}{s_k-1}\sum_{t=2}^{s_k}" in equations
    assert r"\mu^{\rm out}_k=\frac{A}{e_k-s_k-1}\sum_{t=s_k+2}^{e_k}" in equations
    assert rf"\approx {100 * means.change.mean():+.2f}\;\text{{pp}},\qquad |C|=5" in equations
    assert r"\sum_{k\in C}" in equations
    everything = html_text(app) + "\n".join(item.value for item in app.markdown)
    assert "This portfolio was dominated" not in everything and "No reference dominates" not in everything
    assert r"U_\lambda" not in "\n".join(item.proto.body for item in app.get("latex"))
    assert "math_proof" not in {item.key for item in app.container}


def test_short_history_explains_missing_walk_forward_and_export_says_so(monkeypatch, market_prices, benchmark_prices):
    from streamlit.runtime.media_file_manager import MediaFileManager

    deferred = {}
    register = MediaFileManager.add_deferred

    def add_deferred(manager, data_callable, *args, **kwargs):
        file_id = register(manager, data_callable, *args, **kwargs)
        deferred[file_id] = data_callable
        return file_id

    monkeypatch.setattr(MediaFileManager, "add_deferred", add_deferred)
    app, _ = csv_app(monkeypatch, market_prices.iloc[:20], benchmark_prices.iloc[:20])
    assert not app.exception and not app.error
    tile = app.container(key="tile_out_of_sample")
    assert any("at least 3 fit returns and 3 test returns" in item.value for item in tile.info)
    assert not tile.get("plotly_chart")
    assert app.container(key="tile_final_test").get("plotly_chart")
    exported = deferred[app.get("download_button")[0].proto.deferred_file_id]()
    with zipfile.ZipFile(BytesIO(exported)) as archive:
        assert not any(name.startswith("out_of_sample/") for name in archive.namelist())
        assert "at least 3 fit returns" in json.loads(archive.read("metadata.json"))["out_of_sample"]["unavailable"]


def test_short_daily_windows_are_described_in_less_than_a_month(monkeypatch, market_prices, benchmark_prices):
    app, _ = csv_app(monkeypatch, market_prices.iloc[:64], benchmark_prices.iloc[:64])
    assert not app.exception and not app.error
    captions = "\n".join(item.value for item in app.container(key="tile_out_of_sample").caption)
    assert "returns, less than one month" in captions and "0 months" not in captions


@pytest.mark.parametrize("failing", [{1}, {1, 2, 3, 4, 5}])
def test_infeasible_walk_forward_windows_are_named_without_hiding_the_rest(monkeypatch, universe_data, failing):
    from efficient_frontier import out_of_sample

    original, calls = out_of_sample._fit_targets, []

    def fit(*args, **kwargs):
        calls.append(1)
        if len(calls) in failing:
            raise ValueError("requested annual volatility 1.00% is below the feasible minimum 9.00%")
        return original(*args, **kwargs)

    monkeypatch.setattr(out_of_sample, "_fit_targets", fit)
    app = app_start().button(key="build").click().run()
    assert not app.exception
    tile = app.container(key="tile_out_of_sample")
    warnings = "\n".join(item.value for item in tile.warning)
    if len(failing) == 5:
        assert "No window had a feasible fit" in warnings
        assert not tile.get("plotly_chart") and not tile.dataframe
        assert sum("below the feasible minimum" in item.value for item in tile.caption) == 5
    else:
        assert "Window 1 has no result. requested annual volatility" in warnings
        assert list(tile.dataframe[0].value.index) == [2, 3, 4, 5]
        assert r"|C|=4" in "\n".join(item.proto.body for item in tile.get("latex"))
    assert app.container(key="tile_final_test").get("plotly_chart")


def test_qqq_default_budget_is_named_as_later_information(universe_data):
    app = app_start().button(key="build").click().run()
    assert not app.exception
    captions = "\n".join(item.value for item in app.container(key="math_out_of_sample").caption)
    assert ("The volatility budget equals QQQ's volatility over the year through 16 Oct 2026. "
            "That year ends after every walk-forward fit") in captions
    assert "overlaps" not in captions
    manual = app_start()
    manual.number_input(key="preview_target_vol").set_value(18.).run()
    manual.button(key="build").click().run()
    assert not manual.exception
    captions = "\n".join(item.value for item in manual.container(key="math_out_of_sample").caption)
    assert "QQQ's volatility" not in captions


def test_visible_replay_diagnosis_reconciles_the_saved_initial_allocation(monkeypatch, universe_data):
    from efficient_frontier import selection
    from efficient_frontier.selection_report import selection_replay_return_bridge

    app = app_start().button(key="build").click().run()
    assert not app.exception
    study = app.session_state.selection
    prices = app.session_state.result[1]
    bridge = selection_replay_return_bridge(study, prices)
    initial = study.test.allocations[study.selected].iloc[0].reindex(prices.columns)
    training = prices.loc[:study.settings["initial_training_end"]].pct_change(fill_method=None).iloc[1:]
    fitted_mean = float(training.mean() @ initial * study.settings["periods_per_year"])
    assert bridge.fitted_mean == pytest.approx(fitted_mean, abs=1e-12)

    replay = app.container(key="tile_final_test")
    diagnosis = replay.container(key="math_diagnosis")
    nodes = list(replay)
    diagnosis_position = next(index for index, node in enumerate(nodes) if node is diagnosis)
    for element in (replay.get("plotly_chart")[0], replay.dataframe[0]):
        assert next(index for index, node in enumerate(nodes) if node is element) < diagnosis_position
    assert not any("math_diagnosis" in {item.key for item in panel.container} for panel in app.expander)
    equations = [item.proto.body for item in diagnosis.get("latex")]
    rendered = "\n".join(equations)
    assert f"{fitted_mean:.2%}".replace("%", r"\%") in rendered
    assert f"{study.test.metrics.loc[study.selected, 'annual_mean']:.2%}".replace("%", r"\%") in rendered
    numeric_bridge = next(equation for equation in equations if r"\text{stock means}" in equation)
    for symbol, key in (("S", "mean_change"), ("D", "weight_drift"), ("E", "entry_delay"), ("F", "entry_fee")):
        match = re.search(rf"{symbol}&\\approx ([+-]?\d+\.\d+)(?:\\times10\^\{{(-?\d+)\}})?", numeric_bridge)
        assert match is not None
        displayed_points = float(match[1]) * 10 ** int(match[2] or 0)
        assert displayed_points == pytest.approx(100 * bridge[key], abs=.005)
    text = "\n".join(item.value for item in diagnosis.markdown)
    for name in study.test.metrics.index.drop(study.selected):
        assert name in text
    assert "accounting explanation" in text

    def unexpected(*args, **kwargs):
        raise AssertionError("Displaying saved replay mathematics must not rerun the stock model.")

    monkeypatch.setattr(selection, "run_selection", unexpected)
    # Alter the separate latest allocation: the initial replay proof must not change.
    study.latest[study.selected].weights.iloc[:] = 1 / len(prices.columns)
    study.latest[study.selected].expected_return += .15
    app.run()
    assert not app.exception and app.session_state.selection is study
    assert [item.proto.body for item in app.container(key="math_diagnosis").get("latex")] == equations


def test_combined_coverage_and_latest_holdings_preserve_every_input(universe_data, offline_app):
    app = app_start().button(key="build").click().run()
    assert not app.exception and not app.error
    study = app.session_state.selection
    holdings = next(item.value for item in app.dataframe if "Company" in item.value.columns)
    expected = study.latest[study.selected].weights.sort_values(ascending=False).rename("Weight").rename_axis("Symbol").reset_index()
    expected.insert(1, "Company", "Company " + expected.Symbol)
    pd.testing.assert_frame_equal(holdings, expected)
    assert set(holdings.Symbol) == set(universe_data.prices)
    weights = study.latest[study.selected].weights
    displayed = weights[weights > 1e-6].sort_values(ascending=False, kind="stable")
    cards_html = "\n".join(item.proto.body for item in app.container(key="tile_holdings").get("html"))
    cards = re.findall(r'<article class="pl-holding">(.*?)</article>', cards_html)
    assert len(cards) == len(displayed)
    for card, (symbol, weight) in zip(cards, displayed.items(), strict=True):
        assert f">Company {symbol}</a></h4>" in card
        assert f'<span class="pl-holding-symbol">{symbol}</span>' in card
        assert f'<strong class="pl-holding-weight">{weight:.2%}</strong>' in card
        assert f"<p>Provides products and services for {symbol}.</p>" in card
    initial = study.test.allocations[study.selected].iloc[0]
    replay_symbols = initial.index[initial > 1e-6]
    requested = {symbol for kind, symbol in offline_app if kind == "company profile"}
    assert requested == set(displayed.index) | set(replay_symbols)
    replay_holdings = next(item.value for item in app.dataframe if "What it does" in item.value.columns)
    expected_replay = study.test.holdings[study.selected].loc[replay_symbols].copy()
    expected_replay.insert(0, "Company", [f"Company {symbol}" for symbol in replay_symbols])
    expected_replay.insert(1, "What it does", [f"Provides products and services for {symbol}." for symbol in replay_symbols])
    pd.testing.assert_frame_equal(replay_holdings, expected_replay)
    pd.testing.assert_frame_equal(app.session_state.coverage, universe_data.coverage)
    assert app.session_state.result[2]["universe_coverage"][-1]["reason"] == "Incomplete prices"
    assert any("not a historical reconstruction" in item.value for item in app.caption)


@pytest.mark.parametrize("stalled", [False, True])
def test_unavailable_company_profiles_keep_results_visible_without_source_claims(monkeypatch, universe_data, stalled):
    import concurrent.futures
    from threading import Event
    from efficient_frontier import companies

    release, started = Event(), Event()
    pending = []
    original_wait = concurrent.futures.wait

    def unavailable(symbol):
        started.set()
        if stalled:
            release.wait()
        return {}

    def short_wait(futures, *, timeout):
        pending.extend(futures)
        return original_wait(futures, timeout=min(timeout, .05))

    monkeypatch.setattr(companies, "fetch_company_profile", unavailable)
    monkeypatch.setattr(concurrent.futures, "wait", short_wait)
    try:
        app = app_start().button(key="build").click().run(timeout=5)
        assert not app.exception and not app.error
        assert started.is_set()
        if stalled:
            assert not release.is_set() and any(future.running() for future in pending)
        study = app.session_state.selection
        holdings = app.container(key="tile_holdings")
        cards_html = "\n".join(item.proto.body for item in holdings.get("html"))
        for symbol in study.latest[study.selected].weights.loc[lambda weights: weights > 1e-6].index:
            assert f"<h4>Company {symbol}</h4>" in cards_html
        assert "Company description unavailable from the data provider." in cards_html
        assert "<a " not in cards_html
        captions = "\n".join(item.value for item in holdings.caption)
        assert "Yahoo Finance profiles" not in captions and "open its source" not in captions
        assert len(app.get("plotly_chart")) == 3
        assert app.container(key="tile_out_of_sample").container(key="math_out_of_sample")
    finally:
        release.set()
        original_wait([future for future in pending if not future.cancelled()], timeout=5)


def test_partial_market_prices_keep_available_comparison_and_name_missing_etf(monkeypatch, universe_data):
    from efficient_frontier import universe

    original = universe.download_universe_prices

    def partial(*args, **kwargs):
        loaded = original(*args, **kwargs)
        return universe.UniversePrices(loaded.prices, loaded.coverage, loaded.benchmarks[["SPY"]], loaded.metadata)

    monkeypatch.setattr(universe, "download_universe_prices", partial)
    app = app_start().button(key="build").click().run()
    assert not app.exception and not app.error
    study = app.session_state.selection
    assert list(study.benchmark_prices) == ["SPY"] and not study.supported
    assert list(study.test.equity) == ["Historical-mean portfolio", "Equal-weight stocks", "S&P 500 (SPY)"]
    assert any("unavailable for QQQ" in reason for reason in study.reasons)
    app.segmented_control(key="view").set_value("Data & method").run()
    assert "unavailable for QQQ" in html_text(app)
    pd.testing.assert_frame_equal(app.session_state.result[1], universe_data.prices)


def test_repeated_identical_build_uses_cached_inputs_and_selection(monkeypatch, universe_data):
    from efficient_frontier import selection

    original, fits = selection.run_selection, []
    original_frontier, frontiers = selection.selection_frontier, []

    def calculate(*args, **kwargs):
        fits.append(1)
        return original(*args, **kwargs)

    def frontier(*args, **kwargs):
        frontiers.append(1)
        return original_frontier(*args, **kwargs)

    monkeypatch.setattr(selection, "run_selection", calculate)
    monkeypatch.setattr(selection, "selection_frontier", frontier)
    app = app_start().button(key="build").click().run()
    assert not app.exception and len(fits) == len(frontiers) == 1
    equity = app.session_state.selection.test.equity.copy()
    open_setup(app).button(key="build").click().run()
    assert not app.exception and not app.error
    assert len(fits) == len(frontiers) == 1 and len(universe_data.calls) == 2
    app.segmented_control(key="view").set_value("Data & method").run()
    app.segmented_control(key="view").set_value("Portfolio").run()
    assert not app.exception and len(frontiers) == 1
    pd.testing.assert_frame_equal(app.session_state.selection.test.equity, equity)
    pd.testing.assert_frame_equal(app.session_state.coverage, universe_data.coverage)


def test_draft_cancel_and_reopen_preserve_applied_risk_and_prices():
    app = demo_app()
    original = app.session_state.selection
    prices = app.session_state.result[1].copy()
    open_setup(app).number_input(key="target_vol").set_value(25.).run()
    app.number_input(key="cap").set_value(30.).run()
    assert any("differs from this result" in item.value for item in app.info)
    assert app.session_state.selection is original
    assert app.session_state.result[2]["target_volatility"] == .2
    app.button(key="cancel_setup").click().run()
    assert not app.exception and not app.session_state.setup_open
    assert app.session_state.setup["target_vol"] == 20. and app.session_state.setup["cap"] == 20.
    open_setup(app)
    assert app.number_input(key="target_vol").value == 20. and app.number_input(key="cap").value == 20.
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)
    assert not any("differs from this result" in item.value for item in app.info)


def test_submit_applies_limits_and_fixed_assumptions_despite_old_custom_values():
    app = demo_app()
    app.session_state.setup = {**app.session_state.setup, "risk_free": 3., "train_pct": 60, "shrink": 35}
    for key, value in (("risk_free", 4.), ("train_pct", 80), ("shrink", 50)):
        app.session_state[key] = value
    open_setup(app)
    assert not {"risk_free", "train_pct", "shrink"} & {item.key for item in [*app.slider, *app.number_input]}
    app.number_input(key="target_vol").set_value(25.)
    app.number_input(key="cap").set_value(30.)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    metadata = app.session_state.result[2]
    assert {key: metadata[key] for key in ("target_volatility", "max_weight", "risk_free_rate", "train_fraction", "shrinkage")} == {
        "target_volatility": .25, "max_weight": .3, "risk_free_rate": .02, "train_fraction": .7, "shrinkage": 0.}
    assert app.session_state.selection.settings["holding_policy"] == "buy_and_hold"
    assert "rebalance_every" not in app.session_state.selection.settings
    assert app.session_state.selection.settings["cost_bps"] == 10.
    assert "target_vol" not in app.session_state and "cap" not in app.session_state
    open_setup(app)
    assert app.number_input(key="target_vol").value == 25. and app.number_input(key="cap").value == 30.
    assert {key: app.session_state.setup[key] for key in ("train_pct", "shrink", "risk_free")} == {
        "train_pct": 70, "shrink": 0, "risk_free": 2.}
    assert not any("differs from this result" in item.value for item in app.info)


def test_custom_tickers_dates_and_original_list_download_only_on_submit(monkeypatch, market_prices, benchmark_prices):
    from efficient_frontier import data

    calls = []

    def download(tickers, start, end):
        calls.append((tickers, start, end))
        return market_prices

    monkeypatch.setattr(data, "download_prices", download)
    monkeypatch.setattr(data, "download_benchmarks", lambda dates: benchmark_prices)
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
    study = app.session_state.selection
    open_setup(app).radio(key="market").set_value("Original 60 holdings").run()
    assert len(calls) == 1 and app.session_state.selection is study
    app.button(key="build").click().run()
    assert not app.exception and len(calls[-1][0]) == 60
    assert "MRSH" in calls[-1][0] and "MMC" not in calls[-1][0]


def test_csv_file_bytes_survive_reopen_and_cancel_then_explicit_removal(monkeypatch, market_prices, benchmark_prices, offline_app):
    app, uploads = csv_app(monkeypatch, market_prices, benchmark_prices)
    assert not app.exception and not app.error
    originals = {key: app.session_state.setup[key] for key in ("prices_file", "benchmark_file")}
    original = app.session_state.selection
    uploads.update(prices_upload=None, benchmark_upload=None)
    open_setup(app)
    assert any("Using portfolio.csv" in item.value for item in app.caption)
    assert any("Using benchmarks.csv" in item.value for item in app.caption)
    app.button(key="clear_prices_file").click().run()
    assert any("differs from this result" in item.value for item in app.info)
    app.button(key="cancel_setup").click().run()
    assert app.session_state.selection is original
    for key, value in originals.items():
        assert app.session_state.setup[key] == value
    open_setup(app).button(key="clear_benchmark_file").click().run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.setup["prices_file"] == originals["prices_file"]
    assert app.session_state.setup["benchmark_file"] is None
    assert app.session_state.selection.benchmark_prices is None
    assert offline_app == []


def test_same_filename_with_new_bytes_marks_pending_without_changing_inputs(monkeypatch, market_prices, benchmark_prices, offline_app):
    app, uploads = csv_app(monkeypatch, market_prices, benchmark_prices)
    original = app.session_state.selection
    applied = app.session_state.applied_inputs.copy()
    open_setup(app)
    changed = market_prices.copy()
    changed.iloc[-1, 0] *= 1.05
    uploads["prices_upload"] = upload(changed, "portfolio.csv")
    app.run()
    assert not app.exception and any("differs from this result" in item.value for item in app.info)
    assert app.session_state.selection is original and app.session_state.applied_inputs == applied
    pd.testing.assert_frame_equal(app.session_state.result[1], market_prices, check_freq=False)
    assert offline_app == []
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.session_state.result[1], changed, check_freq=False)
    assert app.session_state.applied_inputs["prices"] != applied["prices"]


def test_csv_download_requires_explicit_submit_and_uploaded_benchmarks_win(monkeypatch, market_prices, benchmark_prices):
    from efficient_frontier import data

    calls = []

    def download(dates):
        calls.append(dates.copy())
        return benchmark_prices

    monkeypatch.setattr(data, "download_benchmarks", download)
    app, uploads = csv_app(monkeypatch, market_prices)
    assert not app.exception and calls == [] and app.session_state.selection.benchmark_prices is None
    open_setup(app).checkbox(key="download_benchmarks").set_value(True).run()
    assert calls == [] and app.session_state.applied_inputs["download_benchmarks"] is False
    app.button(key="build").click().run()
    assert not app.exception and not app.error and len(calls) == 1
    pd.testing.assert_frame_equal(app.session_state.selection.benchmark_prices, benchmark_prices, check_freq=False)
    uploads["benchmark_upload"] = upload(benchmark_prices * 2, "benchmarks.csv")
    open_setup(app).button(key="build").click().run()
    assert not app.exception and len(calls) == 1
    assert app.session_state.result[2]["benchmark_source"] == "CSV: benchmarks.csv"
    pd.testing.assert_frame_equal(app.session_state.selection.benchmark_prices, benchmark_prices * 2, check_freq=False)


@pytest.mark.parametrize("symbols", [["SPY"], ["QQQ"], ["SPY", "QQQ"]])
def test_csv_comparison_columns_never_become_stock_holdings(monkeypatch, market_prices, benchmark_prices, offline_app, symbols):
    combined = pd.concat([market_prices, benchmark_prices[symbols]], axis=1)
    app, _ = csv_app(monkeypatch, combined)
    assert not app.exception and not app.error
    study = app.session_state.selection
    pd.testing.assert_frame_equal(app.session_state.result[1], market_prices, check_freq=False)
    pd.testing.assert_frame_equal(study.benchmark_prices, benchmark_prices[symbols], check_freq=False)
    assert set(study.latest[study.selected].weights.index) == set(market_prices)
    assert app.session_state.result[2]["excluded_candidate_symbols"] == symbols
    assert any("excluded from candidate holdings" in item.value for item in app.info)
    open_setup(app).checkbox(key="download_benchmarks").set_value(True).run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error and offline_app == []
    assert app.session_state.result[2]["benchmark_source"] == "Input columns: " + ", ".join(symbols)


@pytest.mark.parametrize("comparison", ["disabled", "other_currency", "override"])
def test_csv_comparison_controls_do_not_put_etfs_back_in_candidate_universe(
    monkeypatch, market_prices, benchmark_prices, offline_app, comparison,
):
    app, uploads = csv_app(monkeypatch, pd.concat([market_prices, benchmark_prices], axis=1))
    open_setup(app)
    if comparison == "disabled":
        app.checkbox(key="compare_market").set_value(False).run()
    elif comparison == "other_currency":
        app.selectbox(key="price_currency").select("Other currency").run()
    else:
        uploads["benchmark_upload"] = upload(benchmark_prices[["QQQ"]] * 2, "override.csv")
        app.run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error and offline_app == []
    pd.testing.assert_frame_equal(app.session_state.result[1], market_prices, check_freq=False)
    if comparison == "override":
        pd.testing.assert_frame_equal(app.session_state.selection.benchmark_prices, benchmark_prices[["QQQ"]] * 2, check_freq=False)
        assert app.session_state.result[2]["benchmark_source"] == "CSV: override.csv"
    else:
        assert app.session_state.selection.benchmark_prices is None


def test_csv_with_only_comparison_etfs_fails_actionably(monkeypatch, benchmark_prices, offline_app):
    app, _ = csv_app(monkeypatch, benchmark_prices)
    assert not app.exception and "selection" not in app.session_state
    assert any("stock" in item.value.lower() and "SPY" in item.value for item in app.error)
    assert app.button(key="fix_setup").label == "Change the study"
    assert offline_app == []


def test_single_stock_after_comparison_split_requires_explicit_full_holding_cap(monkeypatch, market_prices, benchmark_prices):
    stock = market_prices.iloc[:, :1]
    app, _ = csv_app(monkeypatch, pd.concat([stock, benchmark_prices[["SPY"]]], axis=1))
    assert not app.exception and app.error
    open_setup(app).number_input(key="cap").set_value(100.)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    weights = app.session_state.selection.latest[app.session_state.selection.selected].weights
    assert list(weights.index) == list(stock.columns)
    assert weights.iloc[0] == pytest.approx(1.)


@pytest.mark.parametrize("change", ["disable", "currency", "missing prices"])
def test_unavailable_benchmarks_clear_previous_comparison_without_changing_assets(monkeypatch, market_prices, benchmark_prices, change):
    app, uploads = csv_app(monkeypatch, market_prices, benchmark_prices)
    assert not app.exception and app.session_state.selection.benchmark_prices is not None
    open_setup(app)
    if change == "disable":
        app.checkbox(key="compare_market").set_value(False)
    elif change == "currency":
        app.selectbox(key="price_currency").select("Other currency")
    else:
        uploads["benchmark_upload"] = upload(benchmark_prices.iloc[1:], "incomplete.csv")
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    assert app.session_state.selection.benchmark_prices is None and not app.session_state.selection.supported
    assert app.session_state.benchmark_prices is None
    assert app.session_state.result[2]["benchmarks_note"]
    pd.testing.assert_frame_equal(app.session_state.result[1], market_prices, check_freq=False)


@pytest.mark.parametrize("train_fraction", [.70, .71])
def test_short_uploaded_history_shows_actual_fit_replay_and_frontier_windows(monkeypatch, market_prices, train_fraction):
    from efficient_frontier import selection

    prices = market_prices.iloc[:48].copy()
    source = upload(prices, "short-history.csv")
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: source if kwargs["key"] == "prices_upload" else None)
    original = selection.run_selection

    def fit(*args, **kwargs):
        # Exercise saved study settings even when they differ from the UI default.
        return original(*args, **{**kwargs, "train_fraction": train_fraction})

    monkeypatch.setattr(selection, "run_selection", fit)
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
    app.selectbox(key="frequency").select("Custom").run()
    app.number_input(key="periods_per_year").set_value(48.5)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    settings = app.session_state.selection.settings
    assert settings["periods_per_year"] == 48.5
    assert settings["train_fraction"] == train_fraction
    split = int(np.floor((len(prices) - 1) * train_fraction))

    def displayed_day(stamp):
        return f"{stamp.day} {stamp:%b %Y}"

    first_return = displayed_day(prices.index[1])
    train_end = displayed_day(prices.index[split])
    replay_start = displayed_day(prices.index[split + 1])
    last_return = displayed_day(prices.index[-1])
    replay = app.container(key="tile_final_test")
    summary = replay.container(key="replay_dates")
    windows = re.findall(r"<article>(.*?)</article>", "\n".join(item.proto.body for item in summary.get("html")))
    assert len(windows) == 3
    assert f"Initial fit · first {train_fraction:.0%}" in windows[0]
    assert f"{first_return} to {train_end}" in windows[0]
    assert f"{split} returns choose the replay holdings" in windows[0]
    assert f"Replay · final {1 - train_fraction:.0%}" in windows[1]
    assert f"{replay_start} to {last_return}" in windows[1]
    assert f"{len(prices) - 1 - split} later returns form the replay chart and table" in windows[1]
    assert "Latest frontier · full history" in windows[2]
    assert f"{first_return} to {last_return}" in windows[2]
    assert "including the replay dates" in windows[2]
    assert displayed_day(prices.index[0]) not in "\n".join(windows)
    nodes = list(replay)
    assert next(i for i, node in enumerate(nodes) if node is summary) < next(
        i for i, node in enumerate(nodes) if node is replay.get("plotly_chart")[0])
    captions = "\n".join(item.value for item in replay.caption)
    assert "split uses observations, not a fixed one-year test" in captions
    assert "Annualized return and volatility describe units, not the chart’s duration" in captions

    from efficient_frontier.out_of_sample import window_bounds

    # The walk-forward windows use their own fixed split, independent of the replay's train_fraction.
    table = app.container(key="tile_out_of_sample").dataframe[0].value
    bounds = window_bounds(len(prices) - 1)
    assert len(table) == len(bounds)
    for (fit_end, test_end), (_, row) in zip(bounds, table.iterrows()):
        assert row["Test dates"] == f"{prices.index[fit_end + 1].date()} to {prices.index[test_end].date()}"
    captions = "\n".join(item.value for item in app.container(key="tile_out_of_sample").caption)
    assert f"Each fit starts on {first_return} and ends at the close before its test dates" in captions
    assert table.iloc[-1]["Test dates"].endswith(str(prices.index[-1].date()))
    frontier = app.container(key="tile_fitted_frontier")
    description = next(item for item in frontier.markdown if item.value.startswith("**Full-history fit"))
    assert f"**Full-history fit · {first_return} to {last_return}.**" in description.value
    assert "includes the replay dates" in description.value
    nodes = list(frontier)
    assert next(i for i, node in enumerate(nodes) if node is description) < next(
        i for i, node in enumerate(nodes) if node is frontier.get("plotly_chart")[0])


@pytest.mark.parametrize("frequency,periods", [("Monthly", 12), ("Custom", 48.5)])
def test_csv_frequency_reaches_later_test_without_resampling(monkeypatch, market_prices, frequency, periods):
    prices = market_prices.set_axis(pd.date_range("2000-01-31", periods=len(market_prices), freq="ME", name="Date"))
    source = upload(prices, "monthly.csv")
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: source if kwargs["key"] == "prices_upload" else None)
    app = open_setup(app_start())
    app.radio(key="market").set_value("Upload CSV").run()
    app.selectbox(key="frequency").select(frequency).run()
    if frequency == "Custom":
        app.number_input(key="periods_per_year").set_value(periods)
    app.button(key="build").click().run()
    assert not app.exception and not app.error
    study = app.session_state.selection
    assert app.session_state.result[2]["periods_per_year"] == periods
    assert study.settings["periods_per_year"] == periods
    assert study.test.settings["periods_per_year"] == periods
    for key in ("math_fit", "math_replay"):
        equations = "\n".join(item.proto.body for item in app.container(key=key).get("latex"))
        assert f"A={periods:g}" in equations
        assert "A=252" not in equations
    pd.testing.assert_frame_equal(app.session_state.result[1], prices, check_freq=False)


def test_export_is_deferred_and_captures_completed_selection_not_draft(monkeypatch):
    from streamlit.runtime.media_file_manager import MediaFileManager
    from efficient_frontier import selection_report

    payloads, deferred = [], {}
    original, register = selection_report.selection_report_zip, MediaFileManager.add_deferred

    def export(*args, **kwargs):
        payloads.append(original(*args, **kwargs))
        return payloads[-1]

    def add_deferred(manager, data_callable, *args, **kwargs):
        file_id = register(manager, data_callable, *args, **kwargs)
        deferred[file_id] = data_callable
        return file_id

    monkeypatch.setattr(selection_report, "selection_report_zip", export)
    monkeypatch.setattr(MediaFileManager, "add_deferred", add_deferred)
    app = demo_app()
    assert not app.exception and payloads == []
    study = app.session_state.selection
    open_setup(app).number_input(key="target_vol").set_value(25.).run()
    assert payloads == []
    exported = deferred[app.get("download_button")[0].proto.deferred_file_id]()
    assert exported is payloads[0] and len(payloads) == 1
    with zipfile.ZipFile(BytesIO(exported)) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["settings"]["target_volatility"] == .2
        assert metadata["supported"] == study.supported and metadata["selected"] == study.selected
        assert not any(name.startswith("validation/") for name in archive.namelist())
        assert "policy.json" not in archive.namelist()
        assert {"latest/frontier.csv", "latest/frontier_weights.csv", "latest/fitted_comparison.csv",
                "out_of_sample/windows.csv", "out_of_sample/metrics.csv", "out_of_sample/weights.csv"} <= set(archive.namelist())
        frame = pd.read_csv(archive.open("test/equity.csv"), index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(frame, study.test.equity, check_freq=False, check_names=False)
        weights = pd.read_csv(archive.open("latest_weights.csv"), index_col=0)
        expected = pd.DataFrame({name: portfolio.weights for name, portfolio in study.latest.items()})
        pd.testing.assert_frame_equal(weights, expected, check_names=False)


def test_infeasible_rebuild_clears_stale_results_and_preserves_failed_settings():
    app = open_setup(demo_app())
    app.number_input(key="cap").set_value(5.)
    app.button(key="build").click().run()
    assert not app.exception and app.error
    assert "These settings have no solution." in html_text(app)
    assert "A 5% limit needs at least 20 assets." in html_text(app)
    for key in ("result", "selection", "latest_profiles", "backtests", "benchmarks", "evidence", "benchmark_prices"):
        assert key not in app.session_state
    assert not app.get("download_button")
    app.button(key="fix_setup").click().run()
    assert app.number_input(key="cap").value == 5. and app.number_input(key="target_vol").value == 20.
    app.number_input(key="cap").set_value(20.)
    app.button(key="build").click().run()
    assert not app.exception and not app.error and "selection" in app.session_state


def test_infeasible_volatility_target_does_not_relax_the_limits():
    app = app_start()
    app.number_input(key="preview_target_vol").set_value(1.).run()
    open_setup(app).radio(key="market").set_value("Demo").run()
    app.button(key="build").click().run()
    assert not app.exception and app.error
    assert "These settings have no solution." in html_text(app)
    assert "No constraint is relaxed automatically." in html_text(app)
    assert app.session_state.setup["target_vol"] == 1. and app.session_state.setup["cap"] == 20.
    assert "selection" not in app.session_state and not app.get("download_button")
    app.button(key="fix_setup").click().run()
    assert app.number_input(key="target_vol").value == 1. and app.number_input(key="cap").value == 20.


def test_missing_csv_clears_stale_result_without_network(offline_app):
    app = open_setup(demo_app())
    app.radio(key="market").set_value("Upload CSV").run()
    app.button(key="build").click().run()
    assert not app.exception and any("Choose an adjusted-price CSV" in item.value for item in app.error)
    assert "result" not in app.session_state and "selection" not in app.session_state
    assert not app.get("download_button") and offline_app == []


@pytest.mark.parametrize("failure,headline,action", [
    (ValueError("Yahoo returned no prices for: ZZZQQX. Check these symbols and their available history, or upload a CSV."),
     "Yahoo Finance had no prices for these symbols.", "fix_setup"),
    (OSError("Network is unreachable"), "We could not reach Yahoo Finance.", "build"),
])
def test_download_error_names_its_cause_and_keeps_a_useful_action(monkeypatch, failure, headline, action):
    from efficient_frontier import data

    def download(*args):
        raise failure

    monkeypatch.setattr(data, "download_prices", download)
    app = open_setup(app_start())
    app.radio(key="market").set_value("My tickers").run()
    app.text_area(key="ticker_text").set_value("ZZZQQX")
    app.button(key="build").click().run()
    assert not app.exception and any(str(failure) in item.value for item in app.error)
    assert headline in html_text(app) and app.button(key=action)


def test_old_study_waits_for_intentional_migration_and_retains_input_prices(monkeypatch, offline_app):
    from efficient_frontier import selection

    app = demo_app()
    prices = app.session_state.result[1].copy()
    del app.session_state["selection"]
    app.session_state.setup.pop("target_vol")

    def unexpected(*args, **kwargs):
        raise AssertionError("A saved study must not migrate or download automatically.")

    monkeypatch.setattr(selection, "run_selection", unexpected)
    app.run()
    assert not app.exception and "selection" not in app.session_state
    assert any("previous selection model" in item.value for item in app.info)
    assert not app.get("download_button")
    assert not app.get("plotly_chart") and not app.get("latex")
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)
    app.button(key="migrate_selection").click().run()
    assert not app.exception and app.session_state.setup_open
    assert app.radio(key="market").value == "Demo"
    assert app.number_input(key="target_vol").value == 20. and app.number_input(key="cap").value == 20.
    assert "selection" not in app.session_state and offline_app == []


def test_old_csv_setup_keeps_source_files_and_explicit_uncapped_setting(monkeypatch, market_prices, offline_app):
    app, _ = csv_app(monkeypatch, market_prices)
    stored_file = app.session_state.setup["prices_file"]
    prices = app.session_state.result[1].copy()
    app.session_state.setup.pop("target_vol")
    app.session_state.setup["use_cap"] = False
    del app.session_state["selection"]
    app.run()
    app.button(key="migrate_selection").click().run()
    assert not app.exception
    assert app.radio(key="market").value == "Upload CSV"
    assert app.number_input(key="cap").value == 100.
    assert app.session_state.setup["prices_file"] == stored_file
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)
    assert offline_app == []


def test_changed_model_blocks_old_results_and_rebuilds_identical_inputs(monkeypatch, offline_app):
    from efficient_frontier import selection

    calls = []
    original = selection.run_selection

    def select(*args, **kwargs):
        calls.append("selection")
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "run_selection", select)
    app = demo_app()
    prices = app.session_state.result[1].copy()
    setup = dict(app.session_state.setup)
    assert calls == ["selection"] and app.get("download_button")
    monkeypatch.setattr(selection, "MODEL_REVISION", "test_new_revision")
    app.run()
    assert not app.exception and calls == ["selection"]
    assert any("previous selection model" in item.value for item in app.info)
    assert not app.get("download_button")
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)
    assert app.session_state.setup == setup
    app.button(key="migrate_selection").click().run()
    app.button(key="build").click().run()
    assert not app.exception and not app.error and calls == ["selection", "selection"]
    assert app.session_state.result[2]["selection_model_revision"] == "test_new_revision"
    assert app.session_state.selection.settings["model_revision"] == "test_new_revision"
    assert app.get("download_button") and offline_app == []
    pd.testing.assert_frame_equal(app.session_state.result[1], prices)


def test_data_method_is_passive_and_shows_only_the_completed_model(monkeypatch):
    from efficient_frontier import backtest, core, profiles, selection

    calls = []
    original = selection.run_selection

    def select(*args, **kwargs):
        calls.append("selection")
        return original(*args, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError("The simplified app must not run legacy models.")

    monkeypatch.setattr(selection, "run_selection", select)
    for module, name in ((core, "analyze"), (profiles, "build_profiles"), (backtest, "run_backtests")):
        monkeypatch.setattr(module, name, unexpected)
    app = demo_app()
    assert not app.exception and calls == ["selection"]
    study = app.session_state.selection
    app.segmented_control(key="view").set_value("Data & method").run()
    assert not app.exception
    assert "One stock model" in html_text(app)
    assert not {"run_legacy", "open_historical"} & {item.key for item in app.button}
    assert "research_topic" not in {item.key for item in app.segmented_control}
    assert "compare_worst_window" not in {item.key for item in app.checkbox}
    tables = [item.value for item in app.dataframe]
    for expected in (app.session_state.result[1], study.eligibility, study.test.allocations[study.selected],
                     study.test.holdings[study.selected], study.test.trades[study.selected],
                     selection.selection_fit_estimates(study, app.session_state.result[1])):
        assert any(table.equals(expected) for table in tables)
    captions = "\n".join(item.value for item in app.caption)
    assert "0.10%" in captions and "2% risk-free rate" in captions
    assert "buy once and hold" in captions
    assert "not checks of optimizer accuracy" in html_text(app)
    assert app.session_state.result[0] is None and app.session_state.selection is study
    nonce = app.session_state.get("scroll_nonce", 0)
    app.run()
    assert app.session_state.view == "Data & method" and app.session_state.get("scroll_nonce", 0) == nonce
    app.segmented_control(key="view").set_value("Portfolio").run()
    assert not app.exception and app.session_state.selection is study and calls == ["selection"]
    assert all(key not in app.session_state for key in ("backtests", "latest_profiles", "benchmarks", "evidence"))


@pytest.mark.parametrize("old_view", ["Research", "Historical membership"])
def test_removed_saved_view_returns_to_portfolio_without_rebuild(old_view, monkeypatch):
    from efficient_frontier import selection

    app = demo_app()
    study = app.session_state.selection
    app.session_state.view = old_view

    def unexpected(*args, **kwargs):
        raise AssertionError("A view migration must not rebuild the model.")

    monkeypatch.setattr(selection, "run_selection", unexpected)
    app.run()
    assert not app.exception and app.session_state.view == "Portfolio"
    assert app.session_state.selection is study


def test_navigation_shortcuts_preserve_selection_without_old_risk_shortcuts():
    app = demo_app()
    selection = app.session_state.selection
    app.button(key="key_data_method").click().run()
    assert app.session_state.view == "Data & method"
    app.button(key="key_market").click().run()
    assert app.session_state.view == "Portfolio" and app.session_state.scroll_nonce == 2
    app.button(key="key_holdings").click().run()
    assert app.session_state.selection is selection and app.session_state.scroll_nonce == 3
    assert not {"key_low", "key_medium", "key_highest"} & {item.key for item in app.button}
    assert not app.exception
