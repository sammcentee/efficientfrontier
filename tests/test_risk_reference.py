from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.risk_reference import load_qqq_volatility_reference


def install_clock(monkeypatch, hour=17, day=16):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, day, hour, tzinfo=ZoneInfo("America/New_York")).astimezone(tz)

    monkeypatch.setattr("efficient_frontier.risk_reference.datetime", Clock)


@pytest.fixture
def qqq_prices():
    dates = pd.bdate_range("2025-09-25", "2026-10-19", name="Date")
    returns = .0007 + np.sin(np.arange(len(dates) - 1)) * .013
    return pd.DataFrame({"QQQ": 100 * np.r_[1, np.cumprod(1 + returns)]}, index=dates)


def test_reference_uses_only_qqq_adjusted_closes_and_trailing_year_sample_risk(monkeypatch, qqq_prices):
    install_clock(monkeypatch)
    calls = []

    def download(symbols, **kwargs):
        calls.append((symbols, kwargs))
        # Return extra dates too, to test the helper's own window selection.
        return pd.concat({"Close": qqq_prices, "Open": qqq_prices * 1.1}, axis=1)

    monkeypatch.setattr("efficient_frontier.data.yf.download", download)
    result = load_qqq_volatility_reference()
    daily = qqq_prices.QQQ.pct_change(fill_method=None)
    expected = daily.loc[(daily.index > "2025-10-16") & (daily.index <= "2026-10-16")]
    assert result["volatility"] == pytest.approx(np.std(expected, ddof=1) * np.sqrt(252), abs=1e-14)
    assert result["first_return_date"] == "2025-10-17"
    assert result["last_return_date"] == "2026-10-16"
    assert result["observations"] == len(expected)
    assert result["periods_per_year"] == 252
    assert result["symbol"] == "QQQ"
    assert result["source"] == "Yahoo Finance adjusted close"
    assert len(calls) == 1 and calls[0][0] == ["QQQ"]
    assert calls[0][1]["end"] == "2026-10-17"
    assert calls[0][1]["auto_adjust"] is True
    assert calls[0][1]["interval"] == "1d"


@pytest.mark.parametrize("hour, day, last_date, end_date", [
    (15, 16, "2026-10-15", "2026-10-16"),
    (16, 16, "2026-10-16", "2026-10-17"),
    (12, 18, "2026-10-16", "2026-10-18"),
])
def test_incomplete_today_is_excluded_and_weekends_use_last_observed_close(
    monkeypatch, qqq_prices, hour, day, last_date, end_date,
):
    install_clock(monkeypatch, hour, day)
    calls = []

    def download(symbols, start, end):
        calls.append((symbols, start, end))
        return qqq_prices

    monkeypatch.setattr("efficient_frontier.risk_reference.download_prices", download)
    result = load_qqq_volatility_reference()
    assert result["last_return_date"] == last_date
    assert calls[0][2] == pd.Timestamp(end_date)
    boundary = pd.Timestamp(last_date) - pd.DateOffset(years=1)
    daily = qqq_prices.QQQ.pct_change(fill_method=None)
    expected = daily.loc[(daily.index > boundary) & (daily.index <= last_date)]
    assert result["volatility"] == pytest.approx(expected.std(ddof=1) * np.sqrt(252))


def test_explicit_past_date_is_inclusive_even_before_todays_close(monkeypatch, qqq_prices):
    install_clock(monkeypatch, hour=10)
    monkeypatch.setattr("efficient_frontier.risk_reference.download_prices", lambda *args: qqq_prices)
    result = load_qqq_volatility_reference(date(2026, 10, 15))
    assert result["last_return_date"] == "2026-10-15"


@pytest.mark.parametrize("problem, message", [
    ("short", "full trailing year"),
    ("sparse", "200 finite daily returns"),
    ("missing", "200 finite daily returns"),
    ("infinite", "200 finite daily returns"),
    ("flat", "greater than zero"),
    ("stale", "no recent completed close"),
    ("empty", "no recent completed close"),
])
def test_invalid_reference_never_substitutes_an_old_or_invented_value(monkeypatch, qqq_prices, problem, message):
    install_clock(monkeypatch)
    prices = qqq_prices.copy()
    if problem == "short":
        prices = prices.iloc[-50:]
    elif problem == "sparse":
        prices = prices.iloc[::3]
    elif problem == "missing":
        prices.iloc[-3, 0] = np.nan
    elif problem == "infinite":
        prices.iloc[-3, 0] = np.inf
    elif problem == "flat":
        prices[:] = 100
    elif problem == "stale":
        prices = prices.loc[:"2026-10-01"]
    elif problem == "empty":
        prices = prices.iloc[:0]
    monkeypatch.setattr("efficient_frontier.risk_reference.download_prices", lambda *args: prices)
    with pytest.raises(ValueError, match=message):
        load_qqq_volatility_reference()


def test_download_failure_propagates_without_a_fallback(monkeypatch):
    install_clock(monkeypatch)

    def fail(*args):
        raise ValueError("Yahoo price download failed.")

    monkeypatch.setattr("efficient_frontier.risk_reference.download_prices", fail)
    with pytest.raises(ValueError, match="Yahoo price download failed"):
        load_qqq_volatility_reference()


def test_future_date_fails_before_a_download(monkeypatch):
    install_clock(monkeypatch)

    def fail(*args):
        pytest.fail("A future date must not reach Yahoo.")

    monkeypatch.setattr("efficient_frontier.risk_reference.download_prices", fail)
    with pytest.raises(ValueError, match="today or an earlier date"):
        load_qqq_volatility_reference(date(2026, 10, 17))
