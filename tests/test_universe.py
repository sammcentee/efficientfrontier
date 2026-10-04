import copy
from io import StringIO
import json
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from efficient_frontier import universe
from efficient_frontier.universe import UniverseSnapshot, download_universe_prices, fetch_nasdaq100


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


def test_download_batches_all_securities_in_order_without_ranking(monkeypatch):
    symbols = [f"STOCK{i}" for i in range(101)]
    calls, updates = [], []

    def download(symbols, start, end):
        calls.append(symbols)
        return price_table(symbols)

    monkeypatch.setattr(universe, "download_prices", download)
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

    def download(symbols, start, end):
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
    def download(symbols, start, end):
        if symbols == ["SPY", "QQQ"]:
            return price_table(symbols)
        raise ValueError("Yahoo returned no prices.")

    monkeypatch.setattr(universe, "download_prices", download)
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
