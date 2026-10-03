from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.data import (
    demo_prices, download_prices, load_csv, original_tickers, parse_tickers, validate_prices,
)


@pytest.fixture
def prices():
    return demo_prices().iloc[:120, :2].rename(columns={"DEMO_EQUITY": "AAA", "DEMO_TECH": "BBB"})


def test_demo_is_complete_deterministic_and_clearly_synthetic():
    data = demo_prices()
    pd.testing.assert_frame_equal(data, demo_prices())
    assert data.shape == (1000, 6)
    assert all(name.startswith("DEMO_") for name in data.columns)
    pd.testing.assert_frame_equal(validate_prices(data), data, check_freq=False)


def test_validation_sorts_and_normalizes_daily_dates_without_changing_values(prices):
    unsorted = prices.iloc[::-1].copy()
    unsorted.index = unsorted.index + pd.Timedelta(hours=16)
    pd.testing.assert_frame_equal(validate_prices(unsorted), prices, check_freq=False)


@pytest.mark.parametrize("bad_value, message", [
    (None, "Missing prices"), (np.nan, "Missing prices"),
    (np.inf, "finite"), (-np.inf, "finite"), (0, "positive"), (-1, "positive"),
])
def test_validation_rejects_incomplete_or_invalid_prices(prices, bad_value, message):
    prices.iloc[8, 1] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_prices(prices)


def test_validation_rejects_text(prices):
    prices = prices.astype(object)
    prices.iloc[3, 0] = "bad price"
    with pytest.raises(ValueError, match="numeric"):
        validate_prices(prices)


def test_validation_rejects_duplicate_days_after_normalization(prices):
    duplicate = prices.iloc[[0]].copy()
    duplicate.index += pd.Timedelta(hours=8)
    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_prices(pd.concat([prices, duplicate]))


@pytest.mark.parametrize("kind, message", [
    ("short", "100"), ("no_assets", "at least one asset"),
    ("duplicate_asset", "unique"), ("numeric_index", "dates"),
    ("missing_date", "valid date"), ("invalid_date", "valid date"),
])
def test_validation_rejects_invalid_shape_and_dates(prices, kind, message):
    if kind == "short":
        prices = prices.iloc[:99]
    elif kind == "no_assets":
        prices = prices.iloc[:, :0]
    elif kind == "duplicate_asset":
        prices.columns = ["AAA", "AAA"]
    elif kind == "numeric_index":
        prices = prices.reset_index(drop=True)
    else:
        dates = list(prices.index.strftime("%Y-%m-%d"))
        dates[0] = None if kind == "missing_date" else "invalid"
        prices.index = dates
    with pytest.raises(ValueError, match=message):
        validate_prices(prices)


@pytest.mark.parametrize("source_type", ["string", "bytes", "path"])
def test_csv_accepts_filelike_objects_and_paths(prices, source_type, tmp_path):
    content = prices.to_csv().replace("Date,", "date,", 1)
    if source_type == "string":
        source = StringIO(content)
    elif source_type == "bytes":
        source = BytesIO(content.encode())
    else:
        source = tmp_path / "prices.csv"
        source.write_text(content)
    pd.testing.assert_frame_equal(load_csv(source), prices, check_freq=False)


@pytest.mark.parametrize("asset_count", [1, 128])
def test_csv_preserves_any_nonempty_asset_universe(prices, asset_count):
    rng = np.random.default_rng(12)
    table = pd.DataFrame(
        rng.lognormal(mean=4, sigma=0.1, size=(len(prices), asset_count)),
        index=prices.index,
        columns=[f"ASSET_{number:03d}" for number in reversed(range(asset_count))],
    )
    pd.testing.assert_frame_equal(load_csv(StringIO(table.to_csv())), table, check_freq=False)


@pytest.mark.parametrize("header, message", [
    ("Date,AAA,AAA", "duplicate headers"),
    ("Date,AAA,aaa", "duplicate headers"),
    ("Date,AAA,", "nonempty"),
    ("Time,AAA,BBB", "first CSV column"),
])
def test_csv_rejects_invalid_headers(prices, header, message):
    content = header + "\n" + prices.to_csv().split("\n", 1)[1]
    with pytest.raises(ValueError, match=message):
        load_csv(StringIO(content))


def test_csv_rejects_missing_values_and_duplicate_dates(prices):
    sparse = prices.copy()
    sparse.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="Missing prices"):
        load_csv(StringIO(sparse.to_csv()))
    with pytest.raises(ValueError, match="Duplicate dates"):
        load_csv(StringIO(pd.concat([prices, prices.iloc[[0]]]).to_csv()))


def test_empty_csv_is_actionable():
    with pytest.raises(ValueError, match="nonempty"):
        load_csv(StringIO(""))


@pytest.mark.parametrize("row", ["2020-01-01,100", "2020-01-01,100,200,300"])
def test_csv_rejects_ragged_rows(row):
    with pytest.raises(ValueError, match="same number of columns"):
        load_csv(StringIO("Date,AAA,BBB\n" + row))


def test_ticker_normalization():
    assert parse_tickers(" aapl, msft\nBRK.B\tAAPL,brk-b  ") == ["AAPL", "MSFT", "BRK-B"]


def test_ticker_normalization_preserves_a_large_universe_in_input_order():
    symbols = [f"TICKER{number:03d}" for number in reversed(range(128))]
    assert parse_tickers(", ".join(symbols).lower()) == symbols


@pytest.mark.parametrize("symbols", [["AAA"], ["AAA", "BBB"]])
@pytest.mark.parametrize("ticker_level_first", [False, True])
def test_download_selects_adjusted_close_in_requested_order(monkeypatch, prices, ticker_level_first, symbols):
    expected = prices.loc[:, symbols]
    close = expected.iloc[:, ::-1]
    raw = pd.concat({"Open": close * 2, "Close": close}, axis=1)
    if ticker_level_first:
        raw = raw.swaplevel(axis=1)
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = download_prices(symbols, "2020-01-01", "2021-01-01")
    pd.testing.assert_frame_equal(result, expected, check_freq=False)
    assert calls[0][0][0] == symbols
    assert calls[0][1]["auto_adjust"] is True
    assert calls[0][1]["actions"] is False
    assert calls[0][1]["multi_level_index"] is True
    assert calls[0][1]["keepna"] is True


@pytest.mark.parametrize("symbols", [[], ["", " "]])
def test_download_rejects_an_empty_universe_before_fetching(monkeypatch, symbols):
    def download(*args, **kwargs):
        pytest.fail("An empty universe must be rejected before downloading prices.")

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    with pytest.raises(ValueError, match="at least one ticker"):
        download_prices(symbols, "2020-01-01", "2021-01-01")


@pytest.mark.parametrize("failure, message", [
    ("empty", "no prices"), ("absent", "BBB"), ("all_missing", "BBB"),
    ("partial", "Missing prices"), ("no_close", "adjusted close"),
    ("flat", "unexpected"), ("network", "connection"),
])
def test_download_failures_never_fall_back_or_drop_assets(monkeypatch, prices, failure, message):
    raw = pd.concat({"Close": prices}, axis=1)
    if failure == "empty":
        raw = pd.DataFrame()
    elif failure == "absent":
        raw = raw.drop(columns=[("Close", "BBB")])
    elif failure == "all_missing":
        raw[("Close", "BBB")] = np.nan
    elif failure == "partial":
        raw.iloc[5, 1] = np.nan
    elif failure == "no_close":
        raw = pd.concat({"Open": prices}, axis=1)
    elif failure == "flat":
        raw = prices

    def download(*args, **kwargs):
        if failure == "network":
            raise RuntimeError("Offline")
        return raw

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    with pytest.raises(ValueError, match=message):
        download_prices(["AAA", "BBB"], "2020-01-01", "2021-01-01")


def test_original_tickers_reads_bundled_ods():
    tickers = original_tickers()
    assert len(tickers) == len(set(tickers)) == 60
    assert tickers[:3] == ["AAPL", "NVDA", "AMZN"]
    assert all(ticker.isalpha() and ticker.isupper() for ticker in tickers)


def test_original_tickers_missing_file_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="could not be read"):
        original_tickers(tmp_path / "absent.ods")
