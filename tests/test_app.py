from pathlib import Path

import numpy as np
import pandas as pd
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
    assert len(app.multiselect[0].value) == 20
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert len(app.session_state.result[0].portfolios["Minimum volatility"].weights) == 64
