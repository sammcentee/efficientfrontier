"""Current Nasdaq-100 members and explicit price coverage for each security."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from .data import BENCHMARK_SYMBOLS, _download_adjusted_close, download_prices, parse_tickers, validate_prices


NASDAQ100_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
UNIVERSE_LIMITATION = (
    "This study uses current Nasdaq-100 members, not the members at each historical date. "
    "It excludes securities without complete prices for the selected period. "
    "Both choices can favor survivors and distort historical results. "
    "This is not a historical reconstruction of the Nasdaq-100."
)


@dataclass(frozen=True)
class UniverseSnapshot:
    members: pd.DataFrame
    source_url: str
    source_date: str
    retrieved_at: str


@dataclass(frozen=True)
class UniversePrices:
    prices: pd.DataFrame
    coverage: pd.DataFrame
    benchmarks: pd.DataFrame
    metadata: dict


def fetch_nasdaq100() -> UniverseSnapshot:
    """Read the complete current list from Nasdaq. Keep each share class."""
    request = Request(NASDAQ100_URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        data = payload["data"]
        rows = data["data"]["rows"]
        if payload["status"]["rCode"] != 200 or not isinstance(rows, list):
            raise ValueError("Invalid response status.")
        if len(rows) < 100 or len(rows) != int(data["totalrecords"]):
            raise ValueError("Incomplete constituent list.")
        members = []
        for row in rows:
            symbol, name = row["symbol"], row["companyName"]
            if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]*", symbol):
                raise ValueError("Invalid security symbol.")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Missing company name.")
            members.append({
                "symbol": parse_tickers(symbol)[0], "name": name.strip(),
                "industry": str(row.get("sector") or "Unknown").strip() or "Unknown",
            })
        members = pd.DataFrame(members)
        if members["symbol"].duplicated().any():
            raise ValueError("Duplicate security symbols.")
        if not isinstance(data["date"], str):
            raise ValueError("Missing provider date.")
        source_date = pd.Timestamp(data["date"])
        if pd.isna(source_date):
            raise ValueError("Missing provider date.")
    except (URLError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError(
            "The current Nasdaq-100 list is unavailable or incomplete. "
            "Retry later, choose your own stocks, or upload a price CSV."
        ) from exc
    return UniverseSnapshot(
        members=members, source_url=NASDAQ100_URL, source_date=source_date.date().isoformat(),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


def download_universe_prices(snapshot: UniverseSnapshot, start, end, *, progress=None) -> UniversePrices:
    """Fetch all members in batches. Report every exclusion without shorter dates.

    SPY and QQQ provide the common daily calendar. No missing value receives a fill.
    The optional callback receives (completed, total, message).
    """
    symbols = snapshot.members["symbol"].tolist()
    total = len(symbols)
    if not total or len(set(symbols)) != total:
        raise ValueError("The universe must contain unique security symbols.")
    if progress is not None:
        progress(0, total, "Read market dates from SPY and QQQ.")
    benchmarks = download_prices(list(BENCHMARK_SYMBOLS), start, end)
    expected_dates = benchmarks.index
    accepted, coverage = {}, []

    def record(symbol, prices=None, error=None):
        member = snapshot.members.loc[snapshot.members["symbol"] == symbol].iloc[0]
        reason = str(error) if error is not None else ""
        observations, first_date, last_date = None, None, None
        if prices is not None:
            # Inspect each member's actual dates, then require the full market calendar below.
            prices = prices.dropna(how="all")
            observations = len(prices)
            try:
                prices = validate_prices(prices)
            except ValueError as exc:
                reason = f"Yahoo returned no prices for: {symbol}." if prices.empty else str(exc)
                prices = None
        if prices is not None:
            first_date, last_date = prices.index[0].date().isoformat(), prices.index[-1].date().isoformat()
            if not prices.index.equals(expected_dates):
                missing = len(expected_dates.difference(prices.index))
                extra = len(prices.index.difference(expected_dates))
                reason = f"Price dates differ from market dates: {missing} missing, {extra} extra."
            else:
                accepted[symbol] = prices[symbol]
        coverage.append({
            "symbol": symbol, "name": member["name"], "industry": member["industry"],
            "status": "excluded" if reason else "included", "reason": reason,
            "observations": observations, "first_date": first_date, "last_date": last_date,
        })
        if progress is not None:
            progress(len(coverage), total, f"Checked {len(coverage)} of {total} securities.")

    for offset in range(0, total, 25):
        batch = symbols[offset:offset + 25]
        if progress is not None:
            progress(offset, total, f"Download securities {offset + 1} to {offset + len(batch)} of {total}.")
        try:
            prices = _download_adjusted_close(batch, start, end, allow_incomplete=True)
        except ValueError:
            # Retry whole-batch download failures separately from member coverage.
            for symbol in batch:
                try:
                    individual = _download_adjusted_close([symbol], start, end, allow_incomplete=True)
                except ValueError as exc:
                    record(symbol, error=exc)
                else:
                    record(symbol, prices=individual)
        else:
            for symbol in batch:
                record(symbol, prices=prices.loc[:, [symbol]])

    prices = pd.DataFrame(accepted, index=expected_dates)
    metadata = {
        "universe": "Current Nasdaq-100 members",
        "universe_source_url": snapshot.source_url,
        "universe_source_date": snapshot.source_date,
        "universe_retrieved_at": snapshot.retrieved_at,
        "universe_requested": total,
        "universe_included": len(accepted),
        "universe_excluded": total - len(accepted),
        "universe_calendar": "Yahoo adjusted SPY and QQQ dates",
        "universe_start": expected_dates[0].date().isoformat(),
        "universe_end": expected_dates[-1].date().isoformat(),
        "universe_eligibility": "Complete positive adjusted prices on every market date in the selected period.",
        "universe_limitation": UNIVERSE_LIMITATION,
        "universe_coverage": coverage,
    }
    return UniversePrices(prices, pd.DataFrame(coverage), benchmarks, metadata)
