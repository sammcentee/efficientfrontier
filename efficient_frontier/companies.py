"""Current company identities for display, separate from portfolio price data."""

from html import unescape
import re
from urllib.parse import quote

from yfinance.data import YfData

from .data import parse_tickers


def _text(value) -> str:
    return " ".join(unescape(value).split()) if isinstance(value, str) else ""


def _business_sentence(summary: str) -> str:
    """Keep the provider's first sentence without cutting company abbreviations."""
    for boundary in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", summary):
        sentence = summary[:boundary.start()]
        if re.search(r"\b(?:Inc|Corp|Ltd|Co|S\.A|N\.V|U\.S|U\.K)\.$", sentence):
            continue
        return sentence
    return summary


def fetch_company_profile(symbol: str) -> dict[str, str]:
    """Return current Yahoo name, one business sentence and source URL, or {}.

    Missing fields stay empty. A failed profile never changes the portfolio.
    Cache this optional metadata at the UI boundary. The pinned yfinance
    transport handles Yahoo cookies and permits an explicit request timeout;
    Ticker.get_info does not expose one and fetches unrelated financial fields.
    """
    symbols = parse_tickers(symbol)
    if len(symbols) != 1:
        return {}
    ticker = symbols[0]
    escaped = quote(ticker, safe="")
    try:
        response = YfData().get_raw_json(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{escaped}",
            params={"modules": "price,assetProfile", "formatted": "false", "symbol": ticker},
            timeout=8,
        )
        result = response["quoteSummary"]["result"][0]
        price = result.get("price") or {}
        profile = result.get("assetProfile") or {}
        name = _text(price.get("longName")) or _text(price.get("shortName"))
        description = _business_sentence(_text(profile.get("longBusinessSummary")))
    except Exception:
        return {}
    if not name and not description:
        return {}
    return {
        "name": name,
        "description": description,
        "source_url": f"https://finance.yahoo.com/quote/{escaped}/profile/",
    }
