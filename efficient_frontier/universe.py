"""Current stock universes and explicit price coverage for each security."""

from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from io import BytesIO
import json
import math
import re
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pandas as pd

from .data import BENCHMARK_SYMBOLS, _download_adjusted_close, download_prices, parse_tickers, validate_prices


NASDAQ100_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
SPY_HOLDINGS_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
SPY_PRODUCT_URL = "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy"
UNIVERSE_LIMITATION = (
    "This study uses current Nasdaq-100 members, not the members at each historical date. "
    "It excludes securities without complete prices for the selected period. "
    "Both choices can favor survivors and distort historical results. "
    "This is not a historical reconstruction of the Nasdaq-100."
)
SPY_LIMITATION = (
    "The S&P 500 stock list uses current SPY equity holdings published by State Street as a proxy, "
    "not a certified index constituent list. Cash, funds, derivatives and non-stock rights are excluded. "
    "This study uses today's stock universe and requires complete prices for the selected period. "
    "Both choices can favor survivors and distort historical results. "
    "This is not a historical reconstruction of either index."
)


@dataclass(frozen=True)
class UniverseSnapshot:
    members: pd.DataFrame
    source_url: str
    source_date: str
    retrieved_at: str
    name: str = "Current Nasdaq-100 members"
    limitation: str = UNIVERSE_LIMITATION
    source_records: tuple[dict, ...] = ()


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
        if pd.isna(source_date) or source_date.date() > datetime.now(timezone.utc).date():
            raise ValueError("Missing or future provider date.")
    except (URLError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise ValueError(
            "The current Nasdaq-100 list is unavailable or incomplete. "
            "Retry later, choose your own stocks, or upload a price CSV."
        ) from exc
    retrieved_at = datetime.now(timezone.utc).isoformat()
    source_date = source_date.date().isoformat()
    return UniverseSnapshot(
        members=members, source_url=NASDAQ100_URL, source_date=source_date,
        retrieved_at=retrieved_at,
        source_records=({"name": "Nasdaq-100 constituents", "url": NASDAQ100_URL,
                         "date": source_date, "retrieved_at": retrieved_at, "stock_count": len(members)},),
    )


def _spy_worksheet(content: bytes) -> list[dict]:
    """Read the provider's single-sheet XLSX without an Excel dependency."""
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(BytesIO(content)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(item.itertext()) for item in
                       ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(f"{namespace}sheetData/{namespace}row"):
        cells = {}
        for cell in row:
            reference = cell.get("r", "")
            if not re.fullmatch(r"[A-Z]+[1-9][0-9]*", reference):
                raise ValueError("Invalid worksheet cell reference.")
            value = cell.find(f"{namespace}v")
            text = value.text if value is not None else ""
            if cell.get("t") == "s":
                index = int(text)
                if index < 0:
                    raise ValueError("Invalid shared string reference.")
                text = strings[index]
            elif cell.get("t") == "inlineStr":
                text = "".join(cell.itertext())
            cells[re.sub(r"\d", "", reference)] = (text or "").strip()
        rows.append(cells)
    return rows


def _spy_reported_count(document: str) -> tuple[str, int]:
    """Read the dated fund count, distinct from the index count on the same page."""
    for section in re.findall(r"<section\b[^>]*>(.*?)</section>", document, flags=re.S):
        text = " ".join(unescape(re.sub(r"<[^>]+>", " ", section)).split())
        match = re.search(
            r"Fund Characteristics as of (\w+ \d{1,2} \d{4}).*?Number of Holdings (\d+)", text,
        )
        if match:
            return pd.Timestamp(match[1]).date().isoformat(), int(match[2])
    raise ValueError("Missing dated SPY fund holding count.")


def fetch_spy_holdings() -> UniverseSnapshot:
    """Read SPY stocks as an S&P 500 proxy, with an independent fund-count check."""
    try:
        with urlopen(Request(SPY_HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
            rows = _spy_worksheet(response.read())
        metadata = {row.get("A"): row.get("B", "") for row in rows[:3]}
        if metadata.get("Ticker Symbol:") != "SPY" or not re.search(
            r"S&P 500.*ETF Trust", metadata.get("Fund Name:", ""),
        ):
            raise ValueError("The holdings file does not identify the SPY fund.")
        dated = re.fullmatch(r"As of (.+)", metadata.get("Holdings:", ""))
        source_date = pd.Timestamp(dated[1]) if dated else pd.NaT
        if pd.isna(source_date) or source_date.date() > datetime.now(timezone.utc).date():
            raise ValueError("Missing or future holdings date.")
        source_date = source_date.date().isoformat()
        header = ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"]
        header_rows = [i for i, row in enumerate(rows) if [row.get(c) for c in "ABCDEFGH"] == header]
        if len(header_rows) != 1:
            raise ValueError("Missing or repeated holdings header.")
        members, excluded, weights, cash_count = [], [], [], 0
        for row in rows[header_rows[0] + 1:]:
            if not any(row.get(column) for column in "BCDEFGH"):
                continue  # The provider places disclosure text below the holdings table.
            name, symbol, identifier = row.get("A", ""), row.get("B", ""), row.get("C", "")
            weight, shares = float(row.get("E", "")), float(row.get("G", ""))
            if not name or not math.isfinite(weight) or not math.isfinite(shares) or row.get("H") != "USD":
                raise ValueError("Invalid holding identity, value or currency.")
            weights.append(weight)
            reason = ""
            if name.upper() in {"US DOLLAR", "CASH", "CASH USD"}:
                reason, cash_count = "Cash holding", cash_count + 1
            elif "CVR" in identifier or re.search(r"\b(CVR|CONTINGENT VALUE|WARRANT|RIGHTS?)\b", name, re.I):
                reason = "Non-stock right or warrant"
            elif symbol in BENCHMARK_SYMBOLS or re.search(r"\b(ETF|ETN|FUTURES?|SWAP|OPTIONS?|MONEY MARKET)\b", name, re.I):
                reason = "Fund or derivative holding"
            if reason:
                excluded.append({"symbol": symbol, "name": name, "identifier": identifier, "reason": reason})
                continue
            if (not re.fullmatch(r"[A-Z][A-Z0-9.-]*", symbol)
                    or not re.fullmatch(r"[A-Z0-9]{9}", identifier)
                    or not re.fullmatch(r"[A-Z0-9]{6}\d", row.get("D", ""))
                    or weight <= 0 or shares <= 0):
                raise ValueError("Unrecognized stock holding.")
            sector = row.get("F", "")
            members.append({"symbol": parse_tickers(symbol)[0], "name": name,
                            "industry": sector if sector and sector != "-" else "Unknown"})
        members = pd.DataFrame(members)
        if not 500 <= len(members) <= 525 or members["symbol"].duplicated().any():
            raise ValueError("Incomplete or duplicate stock holdings.")
        if not 95 <= sum(weights) <= 105:
            raise ValueError("Incomplete holdings weights.")
        with urlopen(Request(SPY_PRODUCT_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
            count_date, reported_count = _spy_reported_count(response.read().decode("utf-8"))
        if count_date != source_date or reported_count != len(weights) - cash_count:
            raise ValueError("The holdings file does not match the dated published fund count.")
    except (URLError, OSError, ValueError, TypeError, KeyError, IndexError, BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError(
            "The current SPY stock holdings are unavailable or incomplete. "
            "Retry later, choose Nasdaq-100 or your own stocks, or upload a price CSV."
        ) from exc
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return UniverseSnapshot(
        members, SPY_HOLDINGS_URL, source_date, retrieved_at,
        name="Current S&P 500 stocks (SPY holdings proxy)", limitation=SPY_LIMITATION,
        source_records=({"name": "State Street SPY holdings (S&P 500 proxy)", "url": SPY_HOLDINGS_URL,
                         "date": source_date, "retrieved_at": retrieved_at, "stock_count": len(members),
                         "reported_fund_holdings": reported_count, "count_source_url": SPY_PRODUCT_URL,
                         "excluded_holdings": excluded},),
    )


def fetch_sp500_nasdaq100() -> UniverseSnapshot:
    """Combine both complete sources; retain share classes and deduplicate overlap."""
    nasdaq, spy = fetch_nasdaq100(), fetch_spy_holdings()
    members = pd.concat([nasdaq.members, spy.members], ignore_index=True)
    members["symbol"] = members["symbol"].map(lambda symbol: parse_tickers(symbol)[0])
    members = members.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    return UniverseSnapshot(
        members, SPY_HOLDINGS_URL, min(nasdaq.source_date, spy.source_date),
        datetime.now(timezone.utc).isoformat(),
        name="Current S&P 500 (SPY holdings proxy) + Nasdaq-100 stocks", limitation=SPY_LIMITATION,
        source_records=nasdaq.source_records + spy.source_records,
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
        "universe": snapshot.name,
        "universe_source_url": snapshot.source_url,
        "universe_source_date": snapshot.source_date,
        "universe_retrieved_at": snapshot.retrieved_at,
        "universe_sources": list(snapshot.source_records),
        "universe_requested": total,
        "universe_included": len(accepted),
        "universe_excluded": total - len(accepted),
        "universe_calendar": "Yahoo adjusted SPY and QQQ dates",
        "universe_start": expected_dates[0].date().isoformat(),
        "universe_end": expected_dates[-1].date().isoformat(),
        "universe_eligibility": "Complete positive adjusted prices on every market date in the selected period.",
        "universe_limitation": snapshot.limitation,
        "universe_coverage": coverage,
    }
    return UniversePrices(prices, pd.DataFrame(coverage), benchmarks, metadata)
