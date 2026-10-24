import pytest

from efficient_frontier.companies import fetch_company_profile


def mock_profile(monkeypatch, *, price=None, profile=None, response=None):
    calls = []

    def fetch(self, url, **kwargs):
        calls.append((url, kwargs))
        return response if response is not None else {
            "quoteSummary": {"result": [{"price": price, "assetProfile": profile}]},
        }

    monkeypatch.setattr("efficient_frontier.companies.YfData.get_raw_json", fetch)
    return calls


def test_profile_uses_bounded_narrow_lookup_and_preserves_share_class(monkeypatch):
    calls = mock_profile(
        monkeypatch,
        price={"longName": "Berkshire Hathaway Inc.", "shortName": "Berkshire"},
        profile={"longBusinessSummary": (
            "Berkshire Hathaway Inc. operates insurance and other businesses worldwide. "
            "It has several subsidiaries."
        )},
    )
    assert fetch_company_profile("brk.b") == {
        "name": "Berkshire Hathaway Inc.",
        "description": "Berkshire Hathaway Inc. operates insurance and other businesses worldwide.",
        "source_url": "https://finance.yahoo.com/quote/BRK-B/profile/",
    }
    assert calls == [(
        "https://query2.finance.yahoo.com/v10/finance/quoteSummary/BRK-B",
        {"params": {"modules": "price,assetProfile", "formatted": "false", "symbol": "BRK-B"},
         "timeout": 8},
    )]


@pytest.mark.parametrize("summary, expected", [
    ("Acme Corp. Designs products worldwide. It also sells services.",
     "Acme Corp. Designs products worldwide."),
    ("Acme operates in the U.S. and Canada. Its products include tools.",
     "Acme operates in the U.S. and Canada."),
    ("Acme makes tools! It sells them worldwide.", "Acme makes tools!"),
    ("Acme makes tools", "Acme makes tools"),
    ("  Acme makes\n tools &amp; parts.   It sells them worldwide. ",
     "Acme makes tools & parts."),
])
def test_description_keeps_one_clean_business_sentence(monkeypatch, summary, expected):
    mock_profile(monkeypatch, profile={"longBusinessSummary": summary})
    assert fetch_company_profile("ACME")["description"] == expected


def test_missing_description_is_not_invented_from_sector(monkeypatch):
    mock_profile(monkeypatch, price={"shortName": "Acme"}, profile={"sector": "Technology"})
    assert fetch_company_profile("ACME") == {
        "name": "Acme", "description": "",
        "source_url": "https://finance.yahoo.com/quote/ACME/profile/",
    }


@pytest.mark.parametrize("response", [
    {}, {"quoteSummary": {"result": None}}, {"quoteSummary": {"result": []}},
    {"quoteSummary": {"result": [{}]}},
    {"quoteSummary": {"result": [{"price": {"longName": 3}, "assetProfile": []}]}},
])
def test_missing_or_invalid_profile_is_unavailable(monkeypatch, response):
    mock_profile(monkeypatch, response=response)
    assert fetch_company_profile("ACME") == {}


def test_network_failure_is_optional_metadata_only(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("Provider did not respond")

    monkeypatch.setattr("efficient_frontier.companies.YfData.get_raw_json", fail)
    assert fetch_company_profile("ACME") == {}


@pytest.mark.parametrize("symbol", ["", "MSFT NVDA"])
def test_missing_or_multiple_symbols_do_not_fetch(monkeypatch, symbol):
    calls = mock_profile(monkeypatch)
    assert fetch_company_profile(symbol) == {}
    assert calls == []
