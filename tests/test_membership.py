from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from efficient_frontier.membership import MembershipHistory, bundled_membership, load_membership_csv


CSV = 'date,tickers\n2020-01-01,"B,A"\n2020-01-06,"A,C"\n2020-01-10,"B,C"\n'


def test_effective_dates_are_inclusive_and_rosters_replace_each_other():
    history = load_membership_csv(StringIO(CSV), coverage_end="2020-01-12")
    assert history.at("2020-01-01") == ("B", "A")
    assert history.at("2020-01-05") == ("B", "A")
    assert history.at("2020-01-06") == ("A", "C")
    assert history.at("2020-01-10") == ("B", "C")
    assert history.at(pd.Timestamp("2020-01-12")) == ("B", "C")
    assert history.symbols == ("B", "A", "C")


@pytest.mark.parametrize("date", ["2019-12-31", "2020-01-13"])
def test_lookup_does_not_extend_coverage(date):
    history = load_membership_csv(StringIO(CSV), "2020-01-12")
    with pytest.raises(ValueError, match="outside coverage"):
        history.at(date)


@pytest.mark.parametrize("date", [
    "2020-01-01T12:00:00", "01/02/2020", "2020-02-30", "NaT", None, 20200101,
    pd.NaT, pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01 12:00"),
])
def test_lookup_requires_unambiguous_calendar_dates(date):
    history = load_membership_csv(StringIO(CSV), "2020-01-12")
    with pytest.raises(ValueError, match="date"):
        history.at(date)


@pytest.mark.parametrize("csv", [
    "", "date,tickers\n", "date,symbol\n2020-01-01,A\n", "date,tickers,extra\n2020-01-01,A,x\n",
    'date,tickers\n2020-01-01,"A,A"\n', 'date,tickers\n2020-01-01,"A, A"\n',
    'date,tickers\n2020-01-01,"A,"\n', "date,tickers\n2020-01-01,\n",
    "date,tickers\n2020-01-01,A\n2020-01-01,B\n",
    "date,tickers\n2020-01-02,A\n2020-01-01,B\n",
    "date,tickers\n2020-01-01,BAD SYMBOL\n", "date,tickers\n2020-01-01,=A\n",
    "date,tickers\n2020-02-30,A\n", "date,tickers\n01/02/2020,A\n",
    "date,tickers\ngarbage,2020-01-01,A\n",
])
def test_csv_rejects_invalid_or_ambiguous_snapshots(csv):
    with pytest.raises(ValueError):
        load_membership_csv(StringIO(csv), "2020-01-31")


@pytest.mark.parametrize("coverage_end", ["2019-12-31", "2020-01-09", "2020-02-30", None, ""])
def test_coverage_end_is_required_and_cannot_precede_last_snapshot(coverage_end):
    with pytest.raises(ValueError):
        load_membership_csv(StringIO(CSV), coverage_end)


def test_custom_keys_are_preserved_without_yahoo_aliases(tmp_path):
    path = tmp_path / "members.csv"
    path.write_text('date,tickers\n2020-01-01,"BRK.B,BRK-B,NA,OLD_DELL,security:123"\n')
    history = load_membership_csv(path, "2020-01-02", source="Verified security IDs")
    assert history.at("2020-01-01") == ("BRK.B", "BRK-B", "NA", "OLD_DELL", "security:123")
    assert history.identity_exclusions("1990-01-01", through="2030-01-01") == {}
    assert history.source == "Verified security IDs"
    assert history.metadata["coverage_end"] == "2020-01-02"
    assert history.metadata["symbol_count"] == 5
    assert "not independently verified" in history.limitations[0]
    json.dumps(history.metadata)


def test_upload_bytes_and_text_have_the_same_snapshots():
    from_bytes = load_membership_csv(BytesIO(CSV.encode()), "2020-01-12")
    from_text = load_membership_csv(StringIO(CSV), "2020-01-12")
    pd.testing.assert_series_equal(from_bytes.snapshots, from_text.snapshots)


def test_constructor_validates_direct_input_and_copies_rosters():
    roster = ["A", "B"]
    snapshots = pd.Series([roster], index=pd.to_datetime(["2020-01-01"]))
    history = MembershipHistory(snapshots, "2020-01-02")
    roster.append("C")
    assert history.at("2020-01-01") == ("A", "B")
    with pytest.raises(ValueError, match="nonempty roster"):
        MembershipHistory(pd.Series(["A,B"], index=snapshots.index), "2020-01-02")
    with pytest.raises(ValueError, match="at least one"):
        MembershipHistory(pd.Series(dtype=object), "2020-01-02")
    with pytest.raises(ValueError, match="source"):
        MembershipHistory(snapshots, "2020-01-02", source="")


def test_identity_exclusions_use_inclusive_query_and_exclusive_rule_end():
    history = MembershipHistory(
        pd.Series([("A", "B")], index=pd.to_datetime(["2020-01-01"])), "2020-01-12",
        _provenance={"identity_exclusions": [
            {"symbol": "A", "start": None, "end": "2019-01-01", "reason": "Reused ticker"},
            {"symbol": "B", "start": "2019-06-01", "end": None, "reason": "Unresolved identity"},
        ]},
    )
    assert history.identity_exclusions("2018-01-01") == {"A": "Reused ticker"}
    assert history.identity_exclusions("2019-01-01") == {}
    assert history.identity_exclusions("2019-05-31", through="2019-06-01") == {"B": "Unresolved identity"}
    assert set(history.identity_exclusions("2018-01-01", through="2020-01-01")) == {"A", "B"}
    with pytest.raises(ValueError, match="on or after"):
        history.identity_exclusions("2020-01-02", through="2020-01-01")


def test_bundle_is_pinned_and_keeps_historical_labels():
    history = bundled_membership()
    assert history.coverage_start == pd.Timestamp("2010-01-01")
    assert history.coverage_end == pd.Timestamp("2026-09-30")
    assert len(history.snapshots) == 119
    assert len(history.symbols) == 255
    assert "PCLN" in history.at("2010-01-01")
    assert "BKNG" not in history.at("2010-01-01")
    assert "GOOGL" not in history.at("2014-04-02")
    assert "GOOGL" in history.at("2014-04-03")
    assert "GOOG" in history.identity_exclusions("2014-04-02")
    assert "GOOG" not in history.identity_exclusions("2014-04-03")
    assert "GOOG" in history.identity_exclusions("2014-04-02", through="2014-04-03")
    assert "CMCSK" in " ".join(history.limitations)
    assert "LVNTA" in " ".join(history.limitations)
    assert "LILA" in " ".join(history.limitations)
    directory = Path(__file__).resolve().parents[1] / "efficient_frontier" / "membership_data"
    assert hashlib.sha256((directory / "nasdaq100_membership_snapshots.csv").read_bytes()).hexdigest() == history.metadata["snapshot_sha256"]
    assert "Copyright (c) 2021 - 2025 Jeff McCarrell" in (directory / "LICENSE.n100tickers").read_text()
    assert history.metadata["source_commit"] == "cf7c89419941316ff8523726ccba136125f2c11c"


@pytest.mark.parametrize("symbol", ["DELL", "ALTR", "LIFE", "SNDK", "FOX", "FOXA", "GOLD", "NWSA", "BBBY", "LMCA", "LMCK"])
def test_bundle_blocks_known_reused_tickers_in_old_price_history(symbol):
    assert symbol in bundled_membership().identity_exclusions("2010-01-01", through="2016-01-01")


@pytest.mark.parametrize(("symbol", "start"), [
    ("GOOG", "2014-04-03"), ("SNDK", "2025-02-24"), ("DELL", "2018-12-28"),
    ("FOX", "2019-03-19"), ("FOXA", "2019-03-19"), ("NWSA", "2013-07-01"),
])
def test_current_identity_needs_price_history_from_the_safe_start(symbol, start):
    history = bundled_membership()
    previous = pd.Timestamp(start) - pd.Timedelta(days=1)
    assert symbol not in history.identity_exclusions(start, through="2026-09-30")
    assert symbol in history.identity_exclusions(previous, through="2026-09-30")
