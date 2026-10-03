from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_demo_renders_and_invalid_cap_clears_old_result():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "6"
    assert any("synthetic" in item.value.lower() for item in app.info)
    app.slider[0].set_value(5)
    app.button[0].click().run()
    assert not app.exception
    assert any("infeasible" in item.value.lower() for item in app.error)
    assert len(app.metric) == 0


def test_high_risk_free_rate_omits_sharpe_without_breaking_app():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.number_input[0].set_value(100.0)
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
