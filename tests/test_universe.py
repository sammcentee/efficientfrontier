import copy
from io import BytesIO, StringIO
import json
from urllib.error import URLError
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from efficient_frontier import universe
from efficient_frontier.universe import (
    UniverseSnapshot, download_universe_prices, fetch_nasdaq100, fetch_sp500_nasdaq100, fetch_spy_holdings,
)


@pytest.fixture
def payload():
    symbols = ["GOOG", "GOOGL", "BRK.B"] + [f"STOCK{i}" for i in range(98)]
    return {
        "status": {"rCode": 200},
        "data": {
            "date": "Oct 1, 2026", "totalrecords": len(symbols),
            "data": {"rows": [{"symbol": s, "companyName": f"{s} Company", "sector": ""} for s in symbols]},
        },
    }


def test_fetch_uses_official_complete_list_and_preserves_share_classes(monkeypatch, payload):
    calls = []

    def fetch(request, timeout):
        calls.append((request.full_url, timeout))
        return StringIO(json.dumps(payload))

    monkeypatch.setattr(universe, "urlopen", fetch)
    result = fetch_nasdaq100()
    assert calls == [(universe.NASDAQ100_URL, 20)]
    assert result.members["symbol"].tolist()[:3] == ["GOOG", "GOOGL", "BRK-B"]
    assert len(result.members) == 101
    assert result.source_date == "2026-10-01"
    assert pd.Timestamp(result.retrieved_at).tz is not None
    assert set(result.members["industry"]) == {"Unknown"}


@pytest.mark.parametrize("failure", [
    "truncated", "too_short", "status", "duplicate", "alias_duplicate", "missing_name",
    "invalid_symbol", "missing_date", "null_data", "wrong_rows", "missing_total",
])
def test_fetch_rejects_incomplete_or_invalid_membership(monkeypatch, payload, failure):
    malformed = copy.deepcopy(payload)
    data = malformed["data"]
    rows = data["data"]["rows"]
    if failure == "truncated":
        rows.pop()
    elif failure == "too_short":
        rows[:] = rows[:5]
        data["totalrecords"] = 5
    elif failure == "status":
        malformed["status"]["rCode"] = 500
    elif failure == "duplicate":
        rows[1]["symbol"] = rows[0]["symbol"]
    elif failure == "alias_duplicate":
        rows[1]["symbol"] = "BRK-B"
    elif failure == "missing_name":
        rows[1]["companyName"] = " "
    elif failure == "invalid_symbol":
        rows[1]["symbol"] = "BAD SYMBOL"
    elif failure == "missing_date":
        data["date"] = None
    elif failure == "null_data":
        malformed["data"] = None
    elif failure == "wrong_rows":
        data["data"]["rows"] = {}
    else:
        del data["totalrecords"]
    monkeypatch.setattr(universe, "urlopen", lambda *args, **kwargs: StringIO(json.dumps(malformed)))
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        fetch_nasdaq100()


@pytest.mark.parametrize("failure", [URLError("offline"), OSError("timeout"), ValueError("bad JSON")])
def test_fetch_failure_has_no_silent_fallback(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(universe, "urlopen", fail)
    with pytest.raises(ValueError, match="choose your own stocks"):
        fetch_nasdaq100()


def snapshot(symbols):
    return UniverseSnapshot(
        pd.DataFrame({"symbol": symbols, "name": symbols, "industry": "Unknown"}),
        universe.NASDAQ100_URL, "2026-10-01", "2026-10-04T12:00:00+00:00",
    )


def price_table(symbols, dates=None):
    if dates is None:
        dates = pd.bdate_range("2020-01-01", periods=10, name="Date")
    return pd.DataFrame({s: np.arange(len(dates), dtype=float) + 100 + i for i, s in enumerate(symbols)}, index=dates)


def test_incomplete_members_do_not_redownload_complete_batch_peers(monkeypatch):
    symbols = [f"STOCK{i}" for i in range(100)]
    excluded = [symbols[i] for i in (0, 25, 50, 75)]
    calls = []

    def download(requested, **kwargs):
        calls.append(requested)
        prices = price_table(requested)
        if excluded[0] in requested:
            prices.loc[prices.index[:3], excluded[0]] = np.nan
        if excluded[1] in requested:
            prices.loc[prices.index[4], excluded[1]] = np.nan
        if excluded[2] in requested:
            prices = prices.drop(columns=excluded[2])
        if excluded[3] in requested:
            prices.loc[prices.index[4], excluded[3]] = 0
        return pd.concat({"Close": prices}, axis=1)

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = download_universe_prices(snapshot(symbols), "2020-01-01", "2021-01-01")
    assert calls == [["SPY", "QQQ"], *[symbols[i:i + 25] for i in range(0, 100, 25)]]
    assert result.prices.columns.tolist() == [symbol for symbol in symbols if symbol not in excluded]
    pd.testing.assert_index_equal(result.prices.index, result.benchmarks.index)
    coverage = result.coverage.set_index("symbol")
    assert coverage.loc[excluded, "status"].eq("excluded").all()
    assert coverage.loc[excluded[0], "reason"] == "Price dates differ from market dates: 3 missing, 0 extra."
    assert coverage.loc[excluded[1], "reason"] == "Price dates differ from market dates: 1 missing, 0 extra."
    assert "no prices" in coverage.loc[excluded[2], "reason"]
    assert "positive" in coverage.loc[excluded[3], "reason"]
    assert coverage.loc[excluded[0], "observations"] == 7
    assert coverage.loc[excluded[0], "first_date"] == "2020-01-06"
    assert result.metadata["universe_included"] == 96
    assert result.metadata["universe_excluded"] == 4


def test_all_missing_member_is_excluded_without_an_individual_request(monkeypatch):
    calls = []

    def download(requested, **kwargs):
        calls.append(requested)
        prices = price_table(requested)
        if "MISSING" in requested:
            prices["MISSING"] = np.nan
        return pd.concat({"Close": prices}, axis=1)

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = download_universe_prices(snapshot(["COMPLETE", "MISSING"]), "2020-01-01", "2021-01-01")
    assert calls == [["SPY", "QQQ"], ["COMPLETE", "MISSING"]]
    assert result.prices.columns.tolist() == ["COMPLETE"]
    coverage = result.coverage.set_index("symbol")
    assert coverage.loc["MISSING", "status"] == "excluded"
    assert "no prices" in coverage.loc["MISSING", "reason"]


@pytest.mark.parametrize("failure", ["network", "format"])
def test_whole_batch_failure_retries_individual_requests(monkeypatch, failure):
    symbols = ["AAA", "BBB", "CCC"]
    calls = []

    def download(requested, **kwargs):
        calls.append(requested)
        prices = price_table(requested)
        if requested == symbols:
            if failure == "network":
                raise RuntimeError("Batch request failed.")
            return prices
        return pd.concat({"Close": prices}, axis=1)

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = download_universe_prices(snapshot(symbols), "2020-01-01", "2021-01-01")
    assert calls == [["SPY", "QQQ"], symbols, *[[symbol] for symbol in symbols]]
    assert result.prices.columns.tolist() == symbols
    assert result.coverage["status"].eq("included").all()


def test_download_batches_all_securities_in_order_without_ranking(monkeypatch):
    symbols = [f"STOCK{i}" for i in range(101)]
    calls, updates = [], []

    def download(symbols, start, end, **kwargs):
        calls.append(symbols)
        return price_table(symbols)

    monkeypatch.setattr(universe, "download_prices", download)
    monkeypatch.setattr(universe, "_download_adjusted_close", download)
    result = download_universe_prices(snapshot(symbols), "2020-01-01", "2021-01-01", progress=lambda *x: updates.append(x))
    assert calls[0] == ["SPY", "QQQ"]
    assert [len(batch) for batch in calls[1:]] == [25, 25, 25, 25, 1]
    assert result.prices.columns.tolist() == symbols
    assert result.coverage["symbol"].tolist() == symbols
    assert set(result.coverage["status"]) == {"included"}
    assert result.benchmarks.columns.tolist() == ["SPY", "QQQ"]
    assert result.prices.index.equals(result.benchmarks.index)
    assert updates[0][0:2] == (0, 101)
    assert updates[-1][0:2] == (101, 101)
    assert result.metadata["universe_requested"] == result.metadata["universe_included"] == 101
    assert result.metadata["universe_excluded"] == 0
    assert "survivors" in result.metadata["universe_limitation"]


def test_failed_batch_retries_each_member_and_keeps_coverage(monkeypatch):
    symbols = ["COMPLETE", "SHORT", "GAP", "UNAVAILABLE", "EXTRA"]
    calls = []

    def download(symbols, start, end, **kwargs):
        calls.append(symbols)
        if symbols == ["SPY", "QQQ"]:
            return price_table(symbols)
        if len(symbols) > 1:
            raise ValueError("Incomplete batch.")
        symbol = symbols[0]
        if symbol == "UNAVAILABLE":
            raise ValueError("Yahoo returned no prices for UNAVAILABLE.")
        prices = price_table(symbols)
        if symbol == "SHORT":
            prices = prices.iloc[3:]
        elif symbol == "GAP":
            prices = prices.drop(prices.index[4])
        elif symbol == "EXTRA":
            prices.loc[pd.Timestamp("2020-01-18")] = 100.0
        return prices

    monkeypatch.setattr(universe, "download_prices", download)
    monkeypatch.setattr(universe, "_download_adjusted_close", download)
    result = download_universe_prices(snapshot(symbols), "2020-01-01", "2021-01-01")
    coverage = result.coverage.set_index("symbol")
    assert calls[2:] == [[s] for s in symbols]
    assert result.prices.columns.tolist() == ["COMPLETE"]
    assert len(result.prices) == 10
    assert coverage.loc["SHORT", "reason"] == "Price dates differ from market dates: 3 missing, 0 extra."
    assert coverage.loc["GAP", "reason"] == "Price dates differ from market dates: 1 missing, 0 extra."
    assert coverage.loc["EXTRA", "reason"] == "Price dates differ from market dates: 0 missing, 1 extra."
    assert "no prices" in coverage.loc["UNAVAILABLE", "reason"]
    assert coverage.loc["SHORT", "observations"] == 7
    assert result.metadata["universe_included"] == 1
    assert result.metadata["universe_excluded"] == 4
    assert len(result.metadata["universe_coverage"]) == 5
    assert result.metadata["universe_coverage"][3]["observations"] is None
    json.dumps(result.metadata, allow_nan=False)


def test_unavailable_universe_preserves_exclusion_details(monkeypatch):
    def download(symbols, start, end, **kwargs):
        if symbols == ["SPY", "QQQ"]:
            return price_table(symbols)
        raise ValueError("Yahoo returned no prices.")

    monkeypatch.setattr(universe, "download_prices", download)
    monkeypatch.setattr(universe, "_download_adjusted_close", download)
    result = download_universe_prices(snapshot(["AAA", "BBB"]), "2020-01-01", "2021-01-01")
    assert result.prices.shape == (10, 0)
    assert result.coverage["status"].tolist() == ["excluded", "excluded"]
    assert result.metadata["universe_included"] == 0


def test_benchmark_failure_stops_before_stock_requests(monkeypatch):
    calls = []

    def fail(symbols, start, end):
        calls.append(symbols)
        raise ValueError("Yahoo unavailable.")

    monkeypatch.setattr(universe, "download_prices", fail)
    with pytest.raises(ValueError, match="Yahoo unavailable"):
        download_universe_prices(snapshot(["AAA", "BBB"]), "2020-01-01", "2021-01-01")
    assert calls == [["SPY", "QQQ"]]


@pytest.mark.parametrize("symbols", [[], ["AAA", "AAA"]])
def test_download_requires_nonempty_unique_members(symbols):
    with pytest.raises(ValueError, match="unique security symbols"):
        download_universe_prices(snapshot(symbols), "2020-01-01", "2021-01-01")


def spy_rows(stock_count=500):
    symbols = ["GOOG", "GOOGL", "BRK.B", "BF.B"] + [f"STOCK{i}" for i in range(stock_count - 4)]
    rows = [
        ["Fund Name:", "State Street® SPDR® S&P 500® ETF Trust"],
        ["Ticker Symbol:", "SPY"], ["Holdings:", "As of 01-Oct-2026"], [],
        ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"],
    ]
    rows += [[f"{symbol} Company", symbol, f"{i:09}", f"{i:07}", 99.5 / stock_count,
              "-", 1000, "USD"] for i, symbol in enumerate(symbols)]
    rows += [["US DOLLAR", "-", "999USDZ92", "-", .5, "-", 5000, "USD"],
             ["TPG INC", "2602335D", "436CVR021", "-", .000003, "-", 500, "USD"],
             [], ["Disclosure text, not a holding."]]
    return rows


def spy_workbook(rows, *, shared=False):
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = ElementTree.Element("worksheet", xmlns=namespace)
    data = ElementTree.SubElement(sheet, "sheetData")
    strings = ElementTree.Element("sst", xmlns=namespace)
    for i, values in enumerate(rows, 1):
        row = ElementTree.SubElement(data, "row", r=str(i))
        for column, value in zip("ABCDEFGH", values):
            cell = ElementTree.SubElement(row, "c", r=f"{column}{i}")
            if isinstance(value, (float, int)):
                ElementTree.SubElement(cell, "v").text = str(value)
            elif shared:
                cell.set("t", "s")
                ElementTree.SubElement(cell, "v").text = str(len(strings))
                ElementTree.SubElement(ElementTree.SubElement(strings, "si"), "t").text = value
            else:
                cell.set("t", "inlineStr")
                ElementTree.SubElement(ElementTree.SubElement(cell, "is"), "t").text = value
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ElementTree.tostring(sheet))
        if shared:
            archive.writestr("xl/sharedStrings.xml", ElementTree.tostring(strings))
    return output.getvalue()


def spy_page(count=501, date="Oct 1 2026"):
    return (f"<section><h2>Fund Characteristics <span>as of {date}</span></h2>"
            f"<table><tr><th>Number of Holdings</th><td>{count}</td></tr></table></section>"
            "<section><h2>Index Characteristics</h2><th>Number of Holdings</th><td>999</td></section>").encode()


def mock_spy(monkeypatch, rows, *, count=501, date="Oct 1 2026", shared=False):
    calls = []

    def fetch(request, timeout):
        calls.append((request.full_url, timeout))
        if request.full_url == universe.SPY_HOLDINGS_URL:
            return BytesIO(spy_workbook(rows, shared=shared))
        assert request.full_url == universe.SPY_PRODUCT_URL
        return BytesIO(spy_page(count, date))

    monkeypatch.setattr(universe, "urlopen", fetch)
    return calls


@pytest.mark.parametrize("shared", [False, True])
def test_spy_uses_primary_dated_full_holdings_and_preserves_share_classes(monkeypatch, shared):
    calls = mock_spy(monkeypatch, spy_rows(), shared=shared)
    result = fetch_spy_holdings()
    assert calls == [(universe.SPY_HOLDINGS_URL, 20), (universe.SPY_PRODUCT_URL, 20)]
    assert len(result.members) == 500
    assert result.members.symbol.tolist()[:4] == ["GOOG", "GOOGL", "BRK-B", "BF-B"]
    assert set(result.members.industry) == {"Unknown"}
    assert result.source_date == "2026-10-01"
    assert "proxy" in result.name and "not a certified" in result.limitation
    source = result.source_records[0]
    assert source["reported_fund_holdings"] == 501
    assert source["stock_count"] == 500
    assert [row["reason"] for row in source["excluded_holdings"]] == ["Cash holding", "Non-stock right or warrant"]
    json.dumps(source, allow_nan=False)


@pytest.mark.parametrize("name,symbol", [("SPDR S&P 500 ETF TRUST", "SPY"),
                                        ("S&P 500 FUTURE", "ESZ6"), ("Treasury MONEY MARKET", "CASHXX")])
def test_spy_excludes_funds_and_derivatives_from_stock_candidates(monkeypatch, name, symbol):
    rows = spy_rows()
    rows.insert(-2, [name, symbol, "123456789", "1234567", .001, "-", 1000, "USD"])
    mock_spy(monkeypatch, rows, count=502)
    result = fetch_spy_holdings()
    assert len(result.members) == 500
    assert symbol not in set(result.members.symbol)
    assert result.source_records[0]["excluded_holdings"][-1]["reason"] == "Fund or derivative holding"


@pytest.mark.parametrize("failure", [
    "fund_ticker", "fund_name", "missing_date", "future_date", "header", "currency", "symbol",
    "identifier", "sedol", "weight_nan", "weight_zero", "shares_nan", "duplicate", "alias_duplicate",
    "truncated", "count", "count_date",
])
def test_spy_rejects_wrong_fund_invalid_rows_and_incomplete_snapshots(monkeypatch, failure):
    rows = spy_rows(stock_count=501)
    count, date = 502, "Oct 1 2026"
    if failure == "fund_ticker":
        rows[1][1] = "QQQ"
    elif failure == "fund_name":
        rows[0][1] = "Some other fund"
    elif failure == "missing_date":
        rows[2][1] = ""
    elif failure == "future_date":
        rows[2][1] = "As of 01-Oct-2080"
    elif failure == "header":
        rows[4][0] = "Unexpected header"
    elif failure == "currency":
        rows[5][7] = "EUR"
    elif failure == "symbol":
        rows[5][1] = "INVALID SYMBOL"
    elif failure == "identifier":
        rows[5][2] = ""
    elif failure == "sedol":
        rows[5][3] = "-"
    elif failure == "weight_nan":
        rows[5][4] = float("nan")
    elif failure == "weight_zero":
        rows[5][4] = 0
    elif failure == "shares_nan":
        rows[5][6] = float("nan")
    elif failure == "duplicate":
        rows[6][1] = rows[5][1]
    elif failure == "alias_duplicate":
        rows[5][1] = "BRK-B"
    elif failure == "truncated":
        rows.pop(10)  # Still 500 stocks: the separate published count detects this truncation.
    elif failure == "count":
        count += 1
    elif failure == "count_date":
        date = "Sep 30 2026"
    mock_spy(monkeypatch, rows, count=count, date=date)
    with pytest.raises(ValueError, match="SPY stock holdings are unavailable or incomplete"):
        fetch_spy_holdings()


@pytest.mark.parametrize("response", [b"not a workbook", b"<html>Access denied</html>"])
def test_spy_rejects_bad_download_without_fallback(monkeypatch, response):
    monkeypatch.setattr(universe, "urlopen", lambda *args, **kwargs: BytesIO(response))
    with pytest.raises(ValueError, match="choose Nasdaq-100 or your own stocks"):
        fetch_spy_holdings()


def test_union_combines_sources_without_dropping_share_classes_or_hiding_dates(monkeypatch):
    mock_spy(monkeypatch, spy_rows())
    nasdaq = snapshot(["GOOG", "GOOGL", "BRK.B", "TECHONLY"])
    nasdaq = UniverseSnapshot(nasdaq.members, nasdaq.source_url, "2026-10-02", nasdaq.retrieved_at,
                              source_records=({"name": "Nasdaq", "url": nasdaq.source_url, "date": "2026-10-02"},))
    monkeypatch.setattr(universe, "fetch_nasdaq100", lambda: nasdaq)
    result = fetch_sp500_nasdaq100()
    assert len(result.members) == 501
    assert result.members.symbol.is_unique
    assert result.members.symbol.tolist()[:5] == ["GOOG", "GOOGL", "BRK-B", "TECHONLY", "BF-B"]
    assert result.source_date == "2026-10-01"
    assert [source["date"] for source in result.source_records] == ["2026-10-02", "2026-10-01"]
    assert "S&P 500" in result.name and "Nasdaq-100" in result.name


@pytest.mark.parametrize("failed_source", ["fetch_nasdaq100", "fetch_spy_holdings"])
def test_union_requires_both_sources_without_partial_fallback(monkeypatch, failed_source):
    monkeypatch.setattr(universe, "fetch_nasdaq100", lambda: snapshot(["AAA"]))
    monkeypatch.setattr(universe, "fetch_spy_holdings", lambda: snapshot(["BBB"]))

    def fail():
        raise ValueError("Source unavailable.")

    monkeypatch.setattr(universe, failed_source, fail)
    with pytest.raises(ValueError, match="Source unavailable"):
        fetch_sp500_nasdaq100()


def test_downloader_preserves_snapshot_specific_provenance_and_limitation(monkeypatch):
    original = snapshot(["AAA", "BBB"])
    custom = UniverseSnapshot(original.members, "https://example.com/holdings", "2026-10-01", original.retrieved_at,
                              name="Combined stocks", limitation="Current holdings proxy",
                              source_records=({"url": "https://example.com/holdings", "date": "2026-10-01"},))
    monkeypatch.setattr(universe, "download_prices", lambda symbols, *args: price_table(symbols))
    monkeypatch.setattr(universe, "_download_adjusted_close", lambda symbols, *args, **kwargs: price_table(symbols))
    result = download_universe_prices(custom, "2020-01-01", "2021-01-01")
    assert result.metadata["universe"] == custom.name
    assert result.metadata["universe_sources"] == list(custom.source_records)
    assert result.metadata["universe_limitation"] == custom.limitation
    assert result.prices.index.equals(result.benchmarks.index)
