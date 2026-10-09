from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.data import (
    BENCHMARK_SYMBOLS, demo_prices, download_benchmarks, download_prices, load_csv,
    original_tickers, parse_tickers, validate_benchmark_prices, validate_prices,
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


@pytest.mark.parametrize("bad_value", [True, np.bool_(True), 100 + 1j, np.complex128(100 + 1j)])
def test_validation_rejects_boolean_or_complex_values_in_object_columns(prices, bad_value):
    prices = prices.astype(object)
    prices.iloc[3, 0] = bad_value
    with pytest.raises(ValueError, match="numeric"):
        validate_prices(prices)


def test_validation_rejects_duplicate_days_after_normalization(prices):
    duplicate = prices.iloc[[0]].copy()
    duplicate.index += pd.Timedelta(hours=8)
    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_prices(pd.concat([prices, duplicate]))


@pytest.mark.parametrize("kind, message", [
    ("short", "5"), ("no_assets", "at least one asset"),
    ("duplicate_asset", "unique"), ("numeric_index", "dates"),
    ("missing_date", "valid date"), ("invalid_date", "valid date"),
])
def test_validation_rejects_invalid_shape_and_dates(prices, kind, message):
    if kind == "short":
        prices = prices.iloc[:4]
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


@pytest.mark.parametrize("frequency", ["B", "W-FRI", "MS"])
def test_csv_accepts_five_daily_weekly_or_monthly_observations(prices, frequency):
    table = prices.iloc[:5].copy()
    table.index = pd.date_range("2020-01-01", periods=5, freq=frequency, name="Date")
    pd.testing.assert_frame_equal(load_csv(StringIO(table.to_csv())), table, check_freq=False)


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


def test_ticker_normalization_preserves_exchange_suffixes_and_explicit_share_classes():
    assert parse_tickers(
        "vod.l, BMW.DE iwda.as\n7203.T brk.b BF.B brk.a bf.a VOD.L brk-b bf-b"
    ) == ["VOD.L", "BMW.DE", "IWDA.AS", "7203.T", "BRK-B", "BF-B", "BRK-A", "BF-A"]


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


@pytest.mark.parametrize("ticker_level_first", [False, True])
def test_download_preserves_exchange_suffixes_and_normalizes_share_classes(
    monkeypatch, prices, ticker_level_first,
):
    symbols = ["VOD.L", "BMW.DE", "IWDA.AS", "7203.T", "BRK-B", "BF-B"]
    expected = pd.concat([prices.iloc[:, 0]] * len(symbols), axis=1)
    expected.columns = symbols
    close = expected.rename(columns={"BRK-B": "BRK.B", "BF-B": "BF.B"}).iloc[:, ::-1]
    close.columns = close.columns.str.lower()
    raw = pd.concat({"Close": close}, axis=1)
    if ticker_level_first:
        raw = raw.swaplevel(axis=1)

    def download(requested, **kwargs):
        assert requested == symbols
        return raw

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = download_prices(
        ["vod.l", "bmw.de", "iwda.as", "7203.t", "brk.b", "bf.b", "VOD.L"],
        "2020-01-01", "2021-01-01",
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_download_rejects_duplicate_columns_after_share_class_normalization(monkeypatch, prices):
    close = prices.copy()
    close.columns = ["BRK.B", "BRK-B"]
    raw = pd.concat({"Close": close}, axis=1)
    monkeypatch.setattr("efficient_frontier.data.yf.download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="duplicate asset columns"):
        download_prices(["BRK.B"], "2020-01-01", "2021-01-01")


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


def test_original_preset_updates_renamed_symbol_without_changing_historical_list():
    historical = original_tickers()
    current = original_tickers(current_symbols=True)
    assert "MMC" in historical and "MRSH" not in historical
    assert "MRSH" in current and "MMC" not in current
    assert len(current) == len(historical) == 60
    assert current == ["MRSH" if ticker == "MMC" else ticker for ticker in historical]


@pytest.fixture
def benchmark_prices(prices):
    return prices.rename(columns={"AAA": "SPY", "BBB": "QQQ"})


def test_benchmark_validation_preserves_expected_dates_and_canonical_column_order(benchmark_prices):
    expected_dates = benchmark_prices.index.copy()
    table = benchmark_prices.iloc[::-1, ::-1].rename(columns={"SPY": " spy ", "QQQ": "qqq"})
    table.index += pd.Timedelta(hours=16)
    result = validate_benchmark_prices(table, expected_dates)
    assert BENCHMARK_SYMBOLS == ("SPY", "QQQ")
    pd.testing.assert_frame_equal(result, benchmark_prices, check_freq=False)
    pd.testing.assert_index_equal(expected_dates, benchmark_prices.index)
    assert list(table.columns) == ["qqq", " spy "]


@pytest.mark.parametrize("columns", [["SPY", "OTHER"], ["SPY", "spy"], ["QQQ", "QQQ"]])
def test_benchmark_validation_requires_both_unique_benchmark_symbols(benchmark_prices, columns):
    table = benchmark_prices.copy()
    table.columns = columns
    with pytest.raises(ValueError, match="unique|SPY and QQQ"):
        validate_benchmark_prices(table, benchmark_prices.index)


def test_benchmark_validation_rejects_extra_columns(benchmark_prices):
    table = benchmark_prices.assign(OTHER=100)
    with pytest.raises(ValueError, match="SPY and QQQ"):
        validate_benchmark_prices(table, benchmark_prices.index)


@pytest.mark.parametrize("change", ["missing", "extra", "different"])
def test_benchmark_validation_requires_the_exact_expected_dates(benchmark_prices, change):
    expected_dates = benchmark_prices.index[:10]
    table = benchmark_prices.iloc[:10].copy()
    if change == "missing":
        table = table.iloc[1:]
    elif change == "extra":
        table = benchmark_prices.iloc[:11]
    else:
        table.index = table.index[:-1].append(pd.DatetimeIndex([benchmark_prices.index[10]]))
    with pytest.raises(ValueError, match="exactly match"):
        validate_benchmark_prices(table, expected_dates)


@pytest.mark.parametrize("value, message", [
    (np.nan, "Missing prices"), (np.inf, "finite"), (0, "positive"), (-1, "positive"),
])
def test_benchmark_validation_rejects_invalid_prices(benchmark_prices, value, message):
    table = benchmark_prices.copy()
    table.iloc[3, 0] = value
    with pytest.raises(ValueError, match=message):
        validate_benchmark_prices(table, benchmark_prices.index)


@pytest.mark.parametrize("change, message", [
    ("duplicate", "Duplicate dates"), ("missing", "valid date"),
    ("boolean", "numeric"), ("complex", "numeric"),
    ("object_boolean", "numeric"), ("object_complex", "numeric"),
])
def test_benchmark_validation_rejects_invalid_dates_and_numeric_types(benchmark_prices, change, message):
    table = benchmark_prices.copy()
    if change == "duplicate":
        extra = table.iloc[:1].copy()
        extra.index += pd.Timedelta(hours=8)
        table = pd.concat([table, extra])
    elif change == "missing":
        table.index = pd.DatetimeIndex([pd.NaT, *table.index[1:]])
    elif change == "boolean":
        table = table > 0
    elif change == "complex":
        table = table.astype(complex) + 1j
    else:
        table = table.astype(object)
        table.iloc[0, 0] = True if change == "object_boolean" else 100 + 1j
    with pytest.raises(ValueError, match=message):
        validate_benchmark_prices(table, benchmark_prices.index)


@pytest.mark.parametrize("change", ["short", "reversed", "duplicate", "intraday", "timezone", "missing"])
def test_benchmark_expected_dates_are_never_changed_or_fetched_when_invalid(
    monkeypatch, benchmark_prices, change,
):
    expected = benchmark_prices.index
    if change == "short":
        expected = expected[:4]
    elif change == "reversed":
        expected = expected[::-1]
    elif change == "duplicate":
        expected = expected.append(expected[:1])
    elif change == "intraday":
        expected = expected + pd.Timedelta(hours=16)
    elif change == "timezone":
        expected = expected.tz_localize("UTC")
    else:
        expected = pd.DatetimeIndex([pd.NaT, *expected[1:]])

    def download(*args, **kwargs):
        pytest.fail("Invalid expected dates must fail before a network request.")

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    original = expected.copy()
    for call in (lambda: validate_benchmark_prices(benchmark_prices, expected),
                 lambda: download_benchmarks(expected)):
        with pytest.raises(ValueError):
            call()
    pd.testing.assert_index_equal(expected, original)


@pytest.mark.parametrize("stride", [1, 5, 21])
def test_benchmark_download_selects_every_requested_date_without_filling(
    monkeypatch, benchmark_prices, stride,
):
    expected_dates = benchmark_prices.index[::stride][:5]
    table = benchmark_prices.copy()
    # A price outside the requested observations must not affect the comparison.
    excluded = table.index.difference(expected_dates)
    table.loc[excluded[0], "SPY"] = np.nan
    raw = pd.concat({"Close": table.iloc[:, ::-1]}, axis=1)
    calls = []

    def download(symbols, **kwargs):
        calls.append((symbols, kwargs))
        return raw

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = download_benchmarks(expected_dates)
    pd.testing.assert_frame_equal(result, benchmark_prices.loc[expected_dates], check_freq=False)
    assert calls[0][0] == ["SPY", "QQQ"]
    assert calls[0][1]["start"] == expected_dates[0].date().isoformat()
    assert calls[0][1]["end"] == (expected_dates[-1] + pd.Timedelta(days=1)).date().isoformat()
    assert calls[0][1]["interval"] == "1d"


@pytest.mark.parametrize("failure, message", [
    ("date", "missing.*dates"), ("price", "Missing prices"), ("symbol", "QQQ"),
])
def test_benchmark_download_never_drops_an_expected_date_or_symbol(
    monkeypatch, benchmark_prices, failure, message,
):
    expected_dates = benchmark_prices.index[::5][:5]
    table = benchmark_prices.copy()
    if failure == "date":
        table = table.drop(index=expected_dates[2])
    elif failure == "price":
        table.loc[expected_dates[2], "QQQ"] = np.nan
    else:
        table = table.drop(columns="QQQ")
    raw = pd.concat({"Close": table}, axis=1)
    monkeypatch.setattr("efficient_frontier.data.yf.download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match=message):
        download_benchmarks(expected_dates)
