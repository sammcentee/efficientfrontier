"""A current Nasdaq-100 risk reference from QQQ adjusted daily closes."""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import download_prices


def load_qqq_volatility_reference(as_of: date | None = None) -> dict:
    """Measure the last calendar year's daily simple-return volatility.

    ``as_of`` is an inclusive date. Today's bar is excluded before 16:00 New
    York time, including early-close days. Yahoo's download end is exclusive.
    The observed last close anchors the year. No missing prices are filled.
    """
    now = datetime.now(ZoneInfo("America/New_York"))
    requested = pd.Timestamp(as_of or now.date()).normalize()
    today = pd.Timestamp(now.date())
    if pd.isna(requested) or requested > today:
        raise ValueError("The QQQ reference date must be today or an earlier date.")
    end = requested + pd.Timedelta(days=1)
    if requested == today and now.time() < time(16):
        end = requested
    # Extra closes cover weekends at both boundaries and the first return's base.
    start = end - pd.DateOffset(years=1) - pd.Timedelta(days=14)
    prices = download_prices(["QQQ"], start, end)
    prices = prices.loc[(prices.index >= start) & (prices.index < end), "QQQ"]
    if prices.empty or prices.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError("QQQ has no recent completed close. Retry the current volatility reference later.")
    last_close = prices.index[-1]
    boundary = last_close - pd.DateOffset(years=1)
    if not (prices.index <= boundary).any():
        raise ValueError("QQQ prices do not cover the full trailing year and its starting close.")
    returns = prices.pct_change(fill_method=None).loc[lambda values: values.index > boundary]
    if len(returns) < 200 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("The QQQ trailing-year reference needs at least 200 finite daily returns.")
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    if not np.isfinite(volatility) or volatility <= 1e-12:
        raise ValueError("The QQQ trailing-year volatility must be finite and greater than zero.")
    return {
        "symbol": "QQQ", "source": "Yahoo Finance adjusted close",
        "volatility": volatility, "periods_per_year": 252,
        "first_return_date": str(returns.index[0].date()),
        "last_return_date": str(returns.index[-1].date()),
        "observations": len(returns),
    }
