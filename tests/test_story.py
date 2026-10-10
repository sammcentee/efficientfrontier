from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from efficient_frontier import story
from efficient_frontier.story import (
    clarity_text, evidence_answer, evidence_lead, headline_html, headline_text, interval_html, list_html, market_answer,
    market_lead, short_company_name, stretch_table_html,
)

SPY, QQQ = "S&P 500 (SPY)", "Nasdaq-100 (QQQ)"
SELECTED = "Expanding window · Medium"


def weights(values):
    return pd.Series(values, index=[f"A{i:02d}" for i in range(len(values))])


@pytest.mark.parametrize("values,profile,universe,expected", [
    ([.264, .162, .159, .074, .068, .067, .060, .055, .045, .024, .022, .0001], "Medium", "Nasdaq-100",
     "A medium-risk portfolio of 11 Nasdaq-100 stocks."),
    ([.946, .054], "Extreme", "Nasdaq-100", "The highest-risk portfolio puts 94.6% in one stock."),
    ([.5, .5], "Extreme", "Original 60 holdings", "The highest-risk portfolio puts 50.0% in one stock."),
    ([.4, .3, .2, .1], "Low", "Original 60 holdings", "A low-risk portfolio of 4 stocks from your original 60."),
    ([.4, .3, .2, .1, .0004], "Low", "My tickers", "A low-risk portfolio of 4 of your tickers."),
    ([.3, .3, .2, .2], "Medium", "Upload CSV", "A medium-risk portfolio of 4 assets from your file."),
    ([.3, .3, .2, .2], "Extreme", "Demo", "A highest-risk portfolio of 4 simulated assets."),
    ([1.0], "Low", "Demo", "The low-risk portfolio puts 100.0% in one asset."),
])
def test_headline_names_the_level_count_and_concentration(values, profile, universe, expected):
    assert headline_text(weights(values), profile, universe) == expected


def test_headline_html_keeps_hyphenated_phrases_together():
    html = headline_html(weights([.6, .4]), "Medium", "Nasdaq-100")
    assert '<span class="pl-nb">medium-risk</span>' in html
    html = headline_html(weights([.3, .3, .2, .2]), "Low", "Nasdaq-100")
    assert html.endswith('<span class="pl-nb">Nasdaq-100 stocks</span>.')


@pytest.mark.parametrize("raw,short", [
    ("Vertex Pharmaceuticals Incorporated Common Stock", "Vertex Pharmaceuticals"),
    ("Alphabet Inc. Class A Common Stock", "Alphabet"),
    ("Alphabet Inc. Class C Capital Stock", "Alphabet"),
    ("Seagate Technology Holdings PLC Ordinary Shares (Ireland)", "Seagate Technology Holdings"),
    ("NVIDIA Corporation Common Stock", "NVIDIA"),
    ("O'Reilly Automotive, Inc. Common Stock", "O'Reilly Automotive"),
    ("ASML Holding N.V. New York Registry Shares", "ASML Holding"),
    ("AstraZeneca PLC American Depositary Shares", "AstraZeneca"),
    ("Ferrovial SE Ordinary Shares", "Ferrovial"),
    ("Cisco Systems, Inc. Common Stock (DE)", "Cisco Systems"),
    ("Copart, Inc. (DE) Common Stock", "Copart"),
    ("Strategy Inc Common Stock Class A", "Strategy"),
    ("Shopify Inc. Class A Subordinate Voting Shares", "Shopify"),
    ("Warner Bros. Discovery, Inc. Series A Common Stock", "Warner Bros. Discovery"),
    ("Apple", "Apple"),
    ("Company", "Company"),
])
def test_short_company_name_drops_security_type_and_legal_suffixes(raw, short):
    assert short_company_name(raw) == short


def test_short_company_name_for_every_nasdaq_100_member():
    """Raw names from the Nasdaq.com list of 1 Oct 2026. The short names in the fixture were typed by hand."""
    members = pd.read_csv(Path(__file__).parent / "fixtures" / "nasdaq100_names.csv")
    assert len(members) == 101
    shown = members["name"].map(short_company_name)
    assert shown[shown != members["short_name"]].to_dict() == {}


def market_study(cagr):
    return SimpleNamespace(metrics=pd.DataFrame({"cagr": [cagr], "total_return": [cagr * 1.5]}, index=[SELECTED]))


def markets(spy, qqq):
    return SimpleNamespace(backtest_equity=pd.DataFrame(), backtest_metrics=pd.DataFrame(
        {"cagr": [spy, qqq], "total_return": [spy * 1.5, qqq * 1.5]}, index=pd.Index([SPY, QQQ], name="benchmark")))


@pytest.mark.parametrize("rule,spy,qqq,expected", [
    (.094, .286, .411, "No. It trailed both markets."),
    (.30, .20, .25, "Yes. It beat both markets in this test."),
    (.25, .20, .30, "Partly. It beat the S&P 500 but trailed the Nasdaq-100."),
    (.25, .30, .20, "Partly. It beat the Nasdaq-100 but trailed the S&P 500."),
    (.2004, .2, .3, "Partly. It matched the S&P 500 but trailed the Nasdaq-100."),
])
def test_market_answer_compares_growth_at_a_tenth_of_a_point(rule, spy, qqq, expected):
    assert market_answer(market_study(rule), SELECTED, markets(spy, qqq)) == expected


def test_market_answer_without_markets_states_growth():
    assert market_answer(market_study(.072), SELECTED, None) == "It grew 7.2% a year."
    no_test = SimpleNamespace(backtest_equity=None, backtest_metrics=None)
    assert market_answer(market_study(-.031), SELECTED, no_test) == "It grew −3.1% a year."


def test_market_lead_describes_the_rule_and_its_costs():
    settings = {"test_start": "2025-04-03", "test_end": "2026-10-02", "rebalance_every": 21, "cost_bps": 10.0,
                "rolling_window": 252, "periods_per_year": 252}
    study = SimpleNamespace(settings=settings)
    assert market_lead(study, SELECTED) == (
        "We replayed the medium-risk rule over the last 18 months. About once a month it refitted, using only prices "
        "that it could have seen at the time. It paid 0.10% on every trade.")
    assert "Every 42 trading days it refitted on the most recent 252 trading days" in market_lead(
        SimpleNamespace(settings={**settings, "rebalance_every": 42}), "Rolling window · Extreme")
    assert "minimum-volatility" in market_lead(study, "Buy and hold · Minimum volatility")
    assert "Every 21 price rows" in market_lead(SimpleNamespace(settings={**settings, "periods_per_year": 12}), SELECTED)


def evidence(statuses, gaps=(-.19, -.32)):
    index = pd.MultiIndex.from_tuples([(SELECTED, SPY), (SELECTED, QQQ)], names=["strategy", "benchmark"])
    summary = pd.DataFrame({"status": statuses, "cagr_difference": gaps}, index=index)
    return SimpleNamespace(summary=summary, settings={"multiple_testing_tests": 96})


@pytest.mark.parametrize("statuses,answer", [
    (["No clear statistical advantage"] * 2, "Not clear, either way."),
    (["Historical mean underperformance"] * 2, "Clearly behind both markets, in this sample."),
    (["Historical mean advantage", "Historical mean advantage and alpha"], "Clearly ahead of both markets, in this sample."),
    (["Historical alpha"] * 2, "Some evidence of alpha. No clear lead in average return."),
    (["Insufficient history"] * 2, "Too little history to judge."),
    (["Uncertainty unavailable"] * 2, "The uncertainty could not be estimated."),
    (["No clear statistical advantage", "Historical mean underperformance"],
     "Not clear against the S&P 500. Clearly behind the Nasdaq-100."),
    (["Historical mean advantage", "Historical alpha"],
     "Clearly ahead of the S&P 500, in this sample. Some evidence of alpha against the Nasdaq-100."),
])
def test_evidence_answer_for_each_status_and_mixed_results(statuses, answer):
    assert evidence_answer(evidence(statuses), SELECTED) == answer


def test_evidence_lead_and_clarity_follow_the_statuses():
    unclear = evidence(["No clear statistical advantage"] * 2)
    assert evidence_lead(unclear, SELECTED, 18) == (
        "The rule trailed, but 18 months is a short test. After we allow for chance, for returns that cluster in time, "
        "and for all 96 tests in this study, neither gap is statistically clear.")
    assert evidence_lead(unclear, SELECTED, 40).startswith("The rule trailed. After we allow")
    assert evidence_lead(evidence(["No clear statistical advantage"] * 2, (.1, -.1)), SELECTED, 18).startswith(
        "The rule led one market and trailed the other")
    mixed = evidence(["No clear statistical advantage", "Historical mean underperformance"])
    assert evidence_lead(mixed, SELECTED, 18) == "After we allow for chance and for all 96 tests, the two markets give different results."
    assert clarity_text(unclear, SELECTED) == "Neither gap is statistically clear. See How sure can we be."
    assert clarity_text(mixed, SELECTED) == "Only the gap to the Nasdaq-100 is statistically clear in this sample. See How sure can we be."
    assert clarity_text(evidence(["Historical mean underperformance"] * 2), SELECTED).startswith("Both gaps are statistically clear")
    assert clarity_text(evidence(["Insufficient history"] * 2), SELECTED) == "The statistics could not be estimated for this test."
    assert clarity_text(None, SELECTED) == "The statistics could not be estimated for this test."


def interval_evidence(names):
    index = pd.MultiIndex.from_tuples([(SELECTED, name) for name in names], names=["strategy", "benchmark"])
    return SimpleNamespace(summary=pd.DataFrame({
        "annual_advantage": [-.166, -.269], "advantage_ci_low": [-.358, -.546], "advantage_ci_high": [.025, .007]}, index=index))


def test_interval_html_escapes_labels_and_places_zero_on_a_padded_axis():
    html = interval_html(interval_evidence(['<img src=x onerror="a()"> (X)', QQQ]), SELECTED)
    assert "<img" not in html and "vs &lt;img src=x onerror=&quot;a()&quot;&gt;" in html
    # Range -54.6 to +2.5, padded 10%, on 20-point steps: -80 to +20, so zero sits at 80%.
    assert 'class="f-zero" style="left:80.00%"' in html
    assert '<span class="l" style="width:80.00%">← Rule behind</span><span class="r">Rule ahead →</span>' in html
    assert "<b>−16.6</b> points<small>range −35.8 to +2.5</small>" in html
    assert [tick in html for tick in ("−80<", "−60<", "−40<", "−20<", ">0<", "+20<")] == [True] * 6


def test_stretch_table_marks_ahead_and_behind_and_escapes_markets():
    rows = []
    for market, changes in ((SPY, (-.111, .054, -.163)), ("<b>Index</b> (X)", (-.172, .072, -.23))):
        for number, (start, end, change) in enumerate(zip(("2025-04-04", "2025-10-03", "2026-04-06"),
                                                          ("2025-10-02", "2026-04-02", "2026-10-02"), changes), start=1):
            rows.append({"strategy": SELECTED, "benchmark": market, "window": number, "start": start, "end": end,
                         "relative_return": change})
    html = stretch_table_html(SimpleNamespace(windows=pd.DataFrame(rows)), SELECTED)
    assert "<th>Apr 2025 to Oct 2025</th><th>Oct 2025 to Apr 2026</th><th>Apr 2026 to Oct 2026</th>" in html
    assert '<i class="w-dot on" aria-hidden="true"></i>+5.4%<small>ahead</small>' in html
    assert '<i class="w-dot" aria-hidden="true"></i>−11.1%<small>behind</small>' in html
    assert "<td>S&amp;P 500</td>" in html and "&lt;b&gt;Index&lt;/b&gt;" in html and "<b>Index" not in html


def test_list_shows_ten_rows_and_one_rest_row_beyond_twelve_holdings():
    top = [.099, .08, .07, .065, .06, .058, .055, .052, .05, .05]
    values = top + [.04] + [.321 / 21] * 21 + [.00001]
    html = list_html(weights(values), {"A00": "Ferrovial <SE>"})
    assert html.count('class="pl-row"') == 10 and html.count('class="pl-row is-rest"') == 1
    assert "<b>22 more holdings</b><small>The largest of these is 4.0%.</small>" in html
    assert '<span class="pl-pct">36.1%</span></li></ul>' in html
    assert "<b>Ferrovial &lt;SE&gt;</b><small>A00</small>" in html
    assert '<i style="width:100.0%"></i>' in html
    small = list_html(weights([.5, .3, .2, .00001]))
    assert small.count('class="pl-row"') == 3 and "more holdings" not in small and "<b>A00</b></span>" in small


def test_one_asset_reads_in_the_singular():
    assert story.these_assets(92, "Nasdaq-100") == "these 92 stocks"
    assert story.these_assets(1, "Demo") == "this one asset"
    items = story.fine_print_items({"universe": "Demo", "source": "Demo · synthetic"}, pd.DataFrame(columns=["A"]))
    assert "compare mixes of this one asset only" in items[1]


def test_result_tile_without_markets_does_not_repeat_the_answer_or_claim_a_failed_test():
    settings = {"test_start": "2025-04-03", "test_end": "2026-10-02", "initial_execution": "2025-04-03"}
    study = SimpleNamespace(settings=settings, equity=pd.DataFrame({SELECTED: [1.0, 1.073]}, index=pd.to_datetime(["2025-04-02", "2026-10-02"])),
                            metrics=pd.DataFrame({"cagr": [.061], "total_return": [.073]}, index=[SELECTED]))
    html = story.result_tile_html(study, SELECTED, None, None, {"universe": "Demo"})
    assert '<p class="pl-verdict">In total, it gained 7.3%.</p>' in html
    assert "a year," not in html and "could not be estimated" not in html and "pl-clarity" not in html


def test_story_module_has_no_streamlit_dependency():
    assert "streamlit" not in Path(story.__file__).read_text(encoding="utf-8")
