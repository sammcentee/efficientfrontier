"""Explicit, complete price observations for portfolio research."""

import csv
from io import StringIO
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import yfinance as yf


def _normalize_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    aliases = {"BRK.A": "BRK-A", "BRK.B": "BRK-B", "BF.A": "BF-A", "BF.B": "BF-B"}
    return aliases.get(symbol, symbol)


def parse_tickers(text: str) -> list[str]:
    """Normalize Yahoo symbols and remove duplicates. Preserve exchange suffixes and input order."""
    return list(dict.fromkeys(
        _normalize_ticker(token)
        for token in re.split(r"[,\s]+", text.strip()) if token
    ))


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return prices in date order. Never fill gaps or discard an asset or row."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a table with dates and asset columns.")
    result = prices.copy()
    result.columns = [str(column).strip() for column in result.columns]
    if len(result.columns) == 0:
        raise ValueError("Provide at least one asset.")
    if any(not column for column in result.columns) or result.columns.has_duplicates:
        raise ValueError("Asset column names must be nonempty and unique.")
    if pd.api.types.is_numeric_dtype(result.index.dtype):
        raise ValueError("The price index must contain dates, not row numbers.")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(result.index, errors="raise"))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Every row must have a valid date.") from exc
    if dates.hasnans:
        raise ValueError("Every row must have a valid date; blank dates are not allowed.")
    if dates.has_duplicates:
        raise ValueError("Duplicate dates are not allowed; provide one price per asset per day.")
    result.index = dates.rename("Date")
    if len(result) < 5:
        raise ValueError("Provide at least 5 price observations for every asset.")
    try:
        if any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
               for dtype in result.dtypes):
            raise ValueError("Boolean or complex prices are invalid.")
        result = result.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("All prices must be numeric values.") from exc
    if result.isna().any().any():
        missing = ", ".join(result.columns[result.isna().any()])
        raise ValueError(
            f"Missing prices for: {missing}. Choose a common date range with complete "
            "history or correct the CSV; missing prices are never filled."
        )
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError("All prices must be finite and strictly positive.")
    return result.sort_index()


def load_csv(source) -> pd.DataFrame:
    """Load daily, weekly, or monthly prices from a CSV with Date as its first column."""
    try:
        content = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        content = content.lstrip("\ufeff")
        reader = csv.reader(StringIO(content), strict=True)
        headers = next(reader)
        headers = [header.strip() for header in headers]
        rows = list(reader)
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError("Upload a nonempty UTF-8 CSV with Date as the first column.") from exc
    if not headers or headers[0].casefold() != "date":
        raise ValueError("The first CSV column must be named Date.")
    if any(not header for header in headers) or len(set(h.casefold() for h in headers)) != len(headers):
        raise ValueError("CSV column headers must be nonempty and unique; duplicate headers are invalid.")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Every CSV row must have the same number of columns as its header.")
    table = pd.DataFrame(rows, columns=headers)
    return validate_prices(table.set_index(headers[0]))


def download_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """Fetch Yahoo adjusted close prices; an unavailable asset is an error."""
    symbols = parse_tickers(" ".join(tickers))
    if not symbols:
        raise ValueError("Enter at least one ticker.")
    try:
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(start_date) or pd.isna(end_date) or start_date >= end_date:
            raise ValueError("Invalid date range.")
    except (ValueError, TypeError) as exc:
        raise ValueError("Choose valid start and end dates, with the start before the end.") from exc
    try:
        raw = yf.download(
            symbols, start=start_date.date().isoformat(), end=end_date.date().isoformat(),
            auto_adjust=True, actions=False, progress=False, threads=False,
            multi_level_index=True, keepna=True, interval="1d",
        )
    except Exception as exc:
        raise ValueError(
            "Yahoo price download failed. Check your connection and ticker symbols, "
            "retry later, or upload an adjusted-price CSV."
        ) from exc
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Yahoo returned no prices. Check the tickers and date range, or upload a CSV.")
    if not isinstance(raw.columns, pd.MultiIndex) or raw.columns.nlevels != 2:
        raise ValueError("Yahoo returned an unexpected price table. Retry or upload an adjusted-price CSV.")
    price_levels = [level for level in range(2) if "Close" in raw.columns.get_level_values(level)]
    if not price_levels:
        raise ValueError("Yahoo returned no adjusted close prices. Retry or upload an adjusted-price CSV.")
    prices = raw.xs("Close", axis=1, level=price_levels[0]).copy()
    prices.columns = [_normalize_ticker(str(symbol)) for symbol in prices.columns]
    if prices.columns.has_duplicates:
        raise ValueError("Yahoo returned duplicate asset columns. Retry or upload an adjusted-price CSV.")
    missing = [symbol for symbol in symbols if symbol not in prices.columns or prices[symbol].isna().all()]
    if missing:
        raise ValueError(
            f"Yahoo returned no prices for: {', '.join(missing)}. Check these symbols "
            "and their available history, or upload a CSV."
        )
    return validate_prices(prices.loc[:, symbols])


def demo_prices() -> pd.DataFrame:
    """Generate reproducible synthetic prices, with no external data or fallback."""
    rng = np.random.default_rng(212)
    dates = pd.bdate_range("2020-01-01", periods=1000, name="Date")
    names = ["DEMO_EQUITY", "DEMO_TECH", "DEMO_BONDS", "DEMO_GOLD", "DEMO_PROPERTY", "DEMO_GLOBAL"]
    market = rng.normal(0, 1, (len(dates), 1))
    noise = rng.normal(0, 1, (len(dates), len(names)))
    exposure = np.array([0.65, 0.7, -0.1, 0.05, 0.5, 0.6])
    volatility = np.array([0.17, 0.25, 0.045, 0.14, 0.19, 0.16])
    drift = np.array([0.085, 0.115, 0.025, 0.045, 0.065, 0.075])
    shocks = market * exposure + noise * np.sqrt(1 - exposure ** 2)
    log_returns = (drift - volatility ** 2 / 2) / 252 + shocks * volatility / np.sqrt(252)
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=names)


def original_tickers(path=None, *, current_symbols=False) -> list[str]:
    """Read the historical list, optionally using known current Yahoo symbols."""
    source = Path(path) if path is not None else Path(__file__).resolve().parent.parent / "spy_holdings.ods"
    table_ns = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    try:
        with ZipFile(source) as archive:
            root = ElementTree.fromstring(archive.read("content.xml"))
        cells = []
        for row in root.iter(f"{table_ns}table-row"):
            cell = row.find(f"{table_ns}table-cell")
            if cell is not None:
                value = "".join(cell.itertext()).strip()
                if value:
                    cells.append(value)
    except (OSError, BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError("The original spy_holdings.ods ticker list could not be read.") from exc
    if not cells or cells[0] != "Tickers" or not all(re.fullmatch(r"[A-Z]+", ticker) for ticker in cells[1:]):
        raise ValueError("The original holdings file must have a Tickers column containing stock symbols.")
    tickers = list(dict.fromkeys(cells[1:]))
    if not tickers:
        raise ValueError("The original holdings file contains no tickers.")
    if current_symbols:
        # Marsh changed its NYSE symbol on 2026-01-14; the ODS stays historical.
        # Source: https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html
        tickers = ["MRSH" if ticker == "MMC" else ticker for ticker in tickers]
    return tickers
