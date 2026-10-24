from io import BytesIO
import json
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from efficient_frontier import __main__ as cli
from efficient_frontier.historical import historical_demo


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from efficient_frontier import data, universe

    def forbidden(*args, **kwargs):
        pytest.fail("Historical interfaces must not request network data.")

    for module, names in ((cli, ("download_prices", "download_benchmarks", "fetch_nasdaq100", "download_universe_prices")),
                          (data, ("download_prices", "download_benchmarks")),
                          (universe, ("fetch_nasdaq100", "download_universe_prices"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def invoke(monkeypatch, output, *args):
    monkeypatch.setattr(sys, "argv", ["efficient_frontier", *map(str, args), "--output", str(output)])
    return cli.main()


def test_demo_cli_is_offline_and_exported_inputs_reproduce_results(tmp_path, monkeypatch):
    output = tmp_path / "demo"
    assert invoke(monkeypatch, output, "--historical-demo", "--cost-bps", "7", "--currency", "EUR") == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["settings"]["rolling_window"] == 126
    assert metadata["settings"]["price_currency"] == "EUR"
    assert len(metadata["backtest_files"]) == 3
    prices = pd.read_csv(output / "prices.csv")
    assert prices.isna().any().any()
    assert (pd.read_csv(output / "coverage.csv").status == "excluded").any()
    assert pd.read_csv(output / "cash_weights.csv", index_col=0).iloc[0].eq(1).all()
    with ZipFile(output / "report.zip") as archive:
        assert archive.read("report.html") == (output / "report.html").read_bytes()
        assert set(metadata["backtest_files"]) == {"backtests/01", "backtests/02", "backtests/03"}
    replay = tmp_path / "replay"
    assert invoke(monkeypatch, replay, "--csv", output / "prices.csv", "--membership-csv", output / "membership.csv",
                  "--membership-end", metadata["membership"]["coverage_end"], "--cost-bps", "7", "--currency", "EUR") == 0
    for name in ("equity.csv", "metrics.csv", "coverage.csv", "cash_weights.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(output / name), pd.read_csv(replay / name))
    metrics = pd.read_csv(output / "metrics.csv", index_col=0)
    for stem, strategy in metadata["backtest_files"].items():
        holdings = pd.read_csv(output / f"{stem}_holdings.csv", index_col=0)
        trades = pd.read_csv(output / f"{stem}_trades.csv", index_col=0)
        assert np.isclose(holdings.pnl_contribution.sum() - trades.cost.sum(), metrics.loc[strategy, "total_return"])
        assert (pd.to_datetime(trades.train_end) < pd.to_datetime(trades.index)).all()


@pytest.mark.parametrize("args,message", [
    (["--historical-membership"], "requires --csv"),
    (["--historical-demo", "--csv", "prices.csv"], "omit --csv"),
    (["--historical-demo", "--tickers", "AAPL,MSFT"], "automatic price downloads"),
    (["--historical-demo", "--nasdaq100"], "automatic price downloads"),
    (["--historical-demo", "--download-benchmarks"], "benchmark comparisons"),
    (["--historical-demo", "--benchmark-csv", "prices.csv"], "benchmark comparisons"),
    (["--membership-csv", "members.csv", "--csv", "prices.csv"], "requires --membership-end"),
    (["--membership-end", "2020-01-01"], "requires --membership-csv"),
    (["--historical-membership", "--csv", "prices.csv", "--membership-end", "2020-01-01"], "only to --membership-csv"),
    (["--historical-demo", "--currency", " "], "currency"),
])
def test_conflicting_inputs_fail_before_output_or_network(tmp_path, monkeypatch, capsys, args, message):
    output = tmp_path / "invalid"
    assert invoke(monkeypatch, output, *args) == 1
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_historical_and_fixed_exports_remove_only_stale_generated_files(tmp_path, monkeypatch):
    output = tmp_path / "reused"
    assert invoke(monkeypatch, output, "--backtests") == 0
    keep = output / "backtests" / "notes.csv"
    keep.write_text("User notes\n")
    assert invoke(monkeypatch, output, "--historical-demo") == 0
    assert not (output / "weights.csv").exists()
    assert not (output / "backtest_metrics.csv").exists()
    assert len(list((output / "backtests").glob("[0-9][0-9]_holdings.csv"))) == 3
    assert (output / "membership.csv").exists()
    assert invoke(monkeypatch, output) == 0
    assert not (output / "membership.csv").exists()
    assert not (output / "cash_weights.csv").exists()
    assert not (output / "coverage.csv").exists()
    assert (output / "weights.csv").exists()
    assert keep.read_text() == "User notes\n"


def test_bundled_history_records_missing_prices_and_preserves_provenance(tmp_path, monkeypatch):
    prices, _ = historical_demo()
    prices = prices.iloc[:, :2].set_axis(["AAPL", "MSFT"], axis=1)
    source = tmp_path / "verified.csv"
    prices.to_csv(source)
    output = tmp_path / "bundled"
    assert invoke(monkeypatch, output, "--historical-membership", "--csv", source) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["membership"]["source_commit"]
    assert metadata["membership"]["identity_exclusions"]
    coverage = pd.read_csv(output / "coverage.csv")
    assert coverage.loc[coverage.status == "eligible", "symbol"].isin(["AAPL", "MSFT"]).all()
    assert coverage.loc[coverage.status == "excluded", "reason"].str.contains("No price column").any()
    assert (output / "LICENSE.n100tickers").is_file()
    assert invoke(monkeypatch, output, "--historical-demo") == 0
    assert not (output / "LICENSE.n100tickers").exists()


def start():
    return AppTest.from_string(
        "from efficient_frontier.historical_app import render_historical\nrender_historical()",
        default_timeout=30,
    ).run()


def test_historical_component_waits_for_submit_and_keeps_applied_settings():
    app = start()
    assert not app.exception and "historical_result" not in app.session_state
    assert not app.get("download_button")
    app.button(key="historical_run").click().run()
    assert not app.exception and not app.error
    result = app.session_state.historical_result
    assert len(result[3].equity.columns) == 3
    assert any("simulated data" in item.value for item in app.info)
    app.number_input(key="historical_fees").set_value(0.).run()
    assert app.session_state.historical_result[3].settings["cost_bps"] == 10
    assert any("last completed settings" in item.value for item in app.info)
    app.button(key="historical_run").click().run()
    assert app.session_state.historical_result[3].metrics.total_cost.eq(0).all()
    assert not app.exception


def test_missing_historical_input_is_clear():
    app = start()
    assert not app.exception
    app.radio(key="historical_source").set_value("Nasdaq-100 membership + price CSV").run()
    app.button(key="historical_run").click().run()
    assert not app.exception
    assert any("Choose an adjusted-price CSV" in item.value for item in app.error)
    assert "historical_result" not in app.session_state
    assert not app.get("download_button")


def test_custom_upload_uses_declared_coverage_and_keeps_sparse_input(monkeypatch):
    prices, membership = historical_demo()
    price_upload = BytesIO(prices.to_csv().encode())
    price_upload.name = "sparse.csv"
    members = pd.DataFrame({"date": membership.snapshots.index.strftime("%Y-%m-%d"),
                            "tickers": [",".join(roster) for roster in membership.snapshots]})
    members_upload = BytesIO(members.to_csv(index=False).encode())
    members_upload.name = "members.csv"
    uploads = {"historical_prices_file": price_upload, "historical_members_file": members_upload}
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploads[kwargs["key"]])
    app = start()
    app.radio(key="historical_source").set_value("Custom membership + price CSV").run()
    app.date_input(key="historical_coverage_end").set_value(membership.coverage_end.date()).run()
    app.button(key="historical_run").click().run()
    assert not app.exception and not app.error
    saved = app.session_state.historical_result
    pd.testing.assert_frame_equal(saved[1], prices, check_freq=False)
    assert saved[2].coverage_end == membership.coverage_end
    assert saved[3].settings["source"] == "CSV: sparse.csv"
    app.date_input(key="historical_coverage_end").set_value(prices.index[340].date()).run()
    app.button(key="historical_run").click().run()
    assert not app.exception
    assert any("outside coverage" in item.value for item in app.error)
    assert app.session_state.historical_result[2].coverage_end == membership.coverage_end
