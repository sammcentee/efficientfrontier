import html
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from efficient_frontier import story
from efficient_frontier.metrics import performance_metrics
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
    assert market_answer(market_study(-.031), SELECTED, no_test) == "It lost 3.1% a year."


def test_market_lead_describes_the_rule_and_its_costs():
    settings = {"test_start": "2025-04-03", "test_end": "2026-10-02", "rebalance_every": 21, "cost_bps": 10.0,
                "rolling_window": 252, "periods_per_year": 252}
    study = SimpleNamespace(settings=settings)
    lead = market_lead(study, SELECTED)
    assert "We replayed the medium-risk rule over the last 18 months." in lead
    assert "About once a month it refitted using all earlier returns." in lead
    assert "annual historical arithmetic mean returns and volatility" in lead
    assert "returns through the prior close; targets took effect at the following close" in lead
    assert "first interval stayed in cash" in lead
    assert "0.10% on each amount bought or sold" in lead
    assert "later realised compound growth (CAGR), not the fitted mean return" in lead
    assert "Every 42 trading days it refitted on the most recent 252 return observations" in market_lead(
        SimpleNamespace(settings={**settings, "rebalance_every": 42}), "Rolling window · Extreme")
    assert "minimum-volatility" in market_lead(study, "Buy and hold · Minimum volatility")
    assert "Every 21 price rows" in market_lead(SimpleNamespace(settings={**settings, "periods_per_year": 12}), SELECTED)


def evidence(statuses, gaps=(-.19, -.32)):
    index = pd.MultiIndex.from_tuples([(SELECTED, SPY), (SELECTED, QQQ)], names=["strategy", "benchmark"])
    summary = pd.DataFrame({"status": statuses, "cagr_difference": gaps}, index=index)
    return SimpleNamespace(summary=summary, settings={"multiple_testing_tests": 96})


@pytest.mark.parametrize("statuses,answer", [
    (["No clear statistical advantage"] * 2, "No statistically clear difference in mean return."),
    (["Historical mean underperformance"] * 2, "Lower mean returns than both markets, in this sample."),
    (["Historical mean advantage", "Historical mean advantage and alpha"], "Higher mean returns than both markets, in this sample."),
    (["Historical alpha"] * 2, "Some evidence of alpha. No clear lead in average return."),
    (["Insufficient history"] * 2, "Too little history to judge."),
    (["Uncertainty unavailable"] * 2, "The uncertainty could not be estimated."),
    (["No clear statistical advantage", "Historical mean underperformance"],
     "No clear mean-return difference against the S&P 500. Lower mean returns than the Nasdaq-100, in this sample."),
    (["Historical mean advantage", "Historical alpha"],
     "Higher mean returns than the S&P 500, in this sample. Some evidence of alpha against the Nasdaq-100."),
])
def test_evidence_answer_for_each_status_and_mixed_results(statuses, answer):
    assert evidence_answer(evidence(statuses), SELECTED) == answer


def test_evidence_lead_and_clarity_follow_the_statuses():
    unclear = evidence(["No clear statistical advantage"] * 2)
    lead = evidence_lead(unclear, SELECTED, 18)
    assert "trailed both markets on realised compound growth (CAGR)" in lead
    assert "18 months is a short test" in lead
    assert "annual arithmetic mean return gaps, not CAGR gaps" in lead
    assert "Neither annual mean return gap is statistically clear" in lead
    assert "all 96 tests" in lead
    assert "does not establish equal performance" in lead
    assert "short test" not in evidence_lead(unclear, SELECTED, 40)
    assert evidence_lead(evidence(["No clear statistical advantage"] * 2, (.1, -.1)), SELECTED, 18).startswith(
        "The rule led the S&P 500 and trailed the Nasdaq-100")
    mixed = evidence(["No clear statistical advantage", "Historical mean underperformance"])
    assert "statistical evidence differs by market after adjustment for all 96 tests" in evidence_lead(mixed, SELECTED, 18)
    assert clarity_text(unclear, SELECTED).startswith("Neither annual mean return gap is statistically clear.")
    assert "Only the annual mean return gap to the Nasdaq-100 is statistically clear" in clarity_text(mixed, SELECTED)
    assert clarity_text(evidence(["Historical mean underperformance"] * 2), SELECTED).startswith("Both annual mean return gaps are statistically clear")
    assert "does not test the CAGR gaps" in clarity_text(mixed, SELECTED)
    assert clarity_text(evidence(["Insufficient history"] * 2), SELECTED) == "The statistics could not be estimated for this test."
    assert clarity_text(None, SELECTED) == "The statistics could not be estimated for this test."


@pytest.mark.parametrize("status,gaps,growth,mean", [
    ("Historical mean underperformance", (.12, .03), "led both markets", "statistically negative"),
    ("Historical mean advantage", (-.12, -.03), "trailed both markets", "statistically positive"),
    ("No clear statistical advantage", (.12, .03), "led both markets", "Neither annual mean return gap is statistically clear"),
    ("No clear statistical advantage", (0., 0.), "matched both markets", "Neither annual mean return gap is statistically clear"),
])
def test_cagr_outcome_remains_separate_when_mean_return_evidence_differs(status, gaps, growth, mean):
    comparison = evidence([status] * 2, gaps)
    lead = evidence_lead(comparison, SELECTED, 40)
    assert f"{growth} on realised compound growth (CAGR) in this sample" in lead
    assert mean in lead
    assert "annual arithmetic mean return gaps, not CAGR gaps" in lead
    assert "cannot beat" not in lead


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


@pytest.mark.parametrize("expected_return,shown", [(.234, "23.4%"), (-.234, "−23.4%")])
def test_overview_uses_full_history_mean_even_when_worst_window_has_opposite_sign(expected_return, shown):
    portfolios = {
        name: SimpleNamespace(weights=pd.Series({"AAA": .6, "BBB": .4}), volatility=.15)
        for name in ("Low", "Medium", "Extreme")
    }
    latest = SimpleNamespace(
        portfolios=portfolios,
        summary=pd.DataFrame({"expected_return": [expected_return] * 3, "worst_window_return": [-expected_return] * 3,
                              "volatility": [.15] * 3, "effective_holdings": [1 / (.6 ** 2 + .4 ** 2)] * 3},
                             index=list(portfolios)),
        windows=pd.DataFrame({"years": [1., 1., 1.]}),
    )
    stats = story.stats_html(latest, "Medium", "Nasdaq-100")
    why = " ".join(story.why_text(latest, "Medium", "Nasdaq-100"))
    ledger = story.levels_ledger_html(latest, "Medium")
    for text in (stats, why, ledger):
        assert shown in text
        assert "worst stretch" not in text and "Weakest past stretch" not in text
    assert "Fitted annual mean" in stats and "Full-history estimate before fees" in stats
    assert "arithmetic mean" in why and "arithmetic mean" in ledger
    assert "all 3 years" in why
    assert "same prices that selected today's holdings" in why
    assert "not compound growth (CAGR) or a forecast" in why
    assert "measures realised growth on later prices" in why
    assert "midpoint between Low and Highest volatility" in why
    assert "Highest maximises the fitted annual mean return" in why
    assert "Historical mean*" in ledger and "not later realised compound growth (CAGR)" in ledger
    latest.summary["worst_window_return"] = .876
    assert story.stats_html(latest, "Medium", "Nasdaq-100") == stats
    assert " ".join(story.why_text(latest, "Medium", "Nasdaq-100")) == why
    assert story.levels_ledger_html(latest, "Medium") == ledger


def test_method_defines_mean_frontier_and_separates_fit_growth_and_uncertainty():
    method = story.METHOD_MD
    assert "annual arithmetic mean returns and covariance" in method
    assert "nearest to the midpoint" in method and "Low and Highest volatility" in method
    assert "maximises the fitted annual arithmetic mean return" in method
    assert "Each fit selects the same mean-based risk levels from earlier returns" in method
    assert "previous close" in method and "next close" in method
    assert "annual arithmetic mean return gaps, not CAGR gaps" in method
    assert "unclear mean gap does not establish equal performance or rule out future outperformance" in method
    assert "worst stretch" not in method
    assert story.setting_text("return_objective", "historical_mean") == "Annual historical arithmetic mean"


def test_one_asset_reads_in_the_singular():
    assert story.these_assets(92, "Nasdaq-100") == "these 92 stocks"
    assert story.these_assets(1, "Demo") == "this one asset"
    items = story.fine_print_items({"universe": "Demo", "source": "Demo · synthetic"}, pd.DataFrame(columns=["A"]))
    assert "compare mixes of this one asset only" in items[1]


@pytest.mark.parametrize("source", ["Yahoo Finance", "Upload CSV", "CSV: monthly-prices.csv", "Nasdaq-100 · Yahoo Finance"])
def test_currency_note_describes_the_declaration_without_claiming_verification(source):
    prices = pd.DataFrame(columns=["VOD.L", "AAPL"])
    items = story.fine_print_items({"source": source, "universe": "My tickers", "price_currency": "USD"}, prices)
    assert items[-1] == "<b>Currency.</b> You declared USD prices. The app does not check or convert currencies."
    items = story.fine_print_items({"source": source, "universe": "My tickers", "price_currency": "Other currency"}, prices)
    assert "You declared a common price currency." in items[-1]
    assert "Market comparisons are off" in items[-1]


@pytest.mark.parametrize("source", ["Demo · synthetic", "Demo"])
def test_currency_note_identifies_actual_demo_prices(source):
    items = story.fine_print_items({"source": source, "universe": "Demo", "price_currency": "USD"}, pd.DataFrame(columns=["A"]))
    assert items[-1] == "<b>Currency.</b> The demo prices are simulated. No currency conversion occurs."


def test_concentration_callout_describes_allocation_without_claiming_risk_or_profit_shares():
    html = story.callout_html(pd.Series({"BONDS": .5, "STOCKS": .5}), "Upload CSV")
    assert "At least half of it is BONDS." in html
    assert "This is a large allocation to one asset." in html
    assert "gains and losses" not in html
    assert story.callout_html(pd.Series({"BONDS": .4, "STOCKS": .3, "CASH": .3}), "Upload CSV") == ""


def test_result_tile_without_markets_does_not_repeat_the_answer_or_claim_a_failed_test():
    settings = {"test_start": "2025-04-03", "test_end": "2026-10-02", "initial_execution": "2025-04-03"}
    study = SimpleNamespace(settings=settings, equity=pd.DataFrame({SELECTED: [1.0, 1.073]}, index=pd.to_datetime(["2025-04-02", "2026-10-02"])),
                            metrics=pd.DataFrame({"cagr": [.061], "total_return": [.073]}, index=[SELECTED]))
    html = story.result_tile_html(study, SELECTED, None, None, {"universe": "Demo"})
    assert '<p class="pl-verdict">In total, it gained 7.3%.</p>' in html
    assert "a year," not in html and "could not be estimated" not in html and "pl-clarity" not in html


@pytest.fixture
def decision_study():
    dates = pd.bdate_range("2024-01-02", periods=4)
    paths = {
        "Equal weight": [1., .99, 1.04, 1.08],
        "Extreme": [1., .99, 1.3, 1.05],
        "Medium": [1., .99, .85, .93],
        "Low": [1., .995, 1.01, 1.03],
        "Maximum Sharpe": [1., .99, 1.08, 1.06],
    }
    equity = pd.DataFrame({f"Expanding window · {name}": path for name, path in paths.items()}, index=dates)
    metrics = pd.DataFrame({name: performance_metrics(path, 0., 12) for name, path in equity.items()}).T
    metrics["total_cost"] = .01
    study = SimpleNamespace(
        equity=equity, metrics=metrics,
        settings={"periods_per_year": 12, "test_start": str(dates[1].date()), "test_end": str(dates[-1].date())},
        holdings={SELECTED: pd.DataFrame({"pnl_contribution": [.09, -.14, -.03, .02, .015, -.01, -.005, 0.]},
                                         index=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "UNHELD"])},
    )
    market_equity = pd.DataFrame({QQQ: [1., .99, 1.04, 1.12], SPY: [1., .99, 1.05, 1.10]}, index=dates)
    market_metrics = pd.DataFrame({name: performance_metrics(path, 0., 12) for name, path in market_equity.items()}).T
    benchmarks = SimpleNamespace(backtest_equity=market_equity, backtest_metrics=market_metrics)
    return study, benchmarks


def table_rows(document):
    return [[html.unescape(re.sub(r"<[^>]+>", "", cell)) for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row)]
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", document)]


def test_decision_panel_keeps_fixed_order_and_matches_net_paths_and_metrics(decision_study):
    study, benchmarks = decision_study
    document = story.decision_panel_html(study, SELECTED, benchmarks)
    rows = table_rows(document)
    assert rows[0] == ["Rule / ETF", "Test mean / year", "Growth / year (CAGR)", "Volatility / year", "Maximum drawdown"]
    assert [row[0] for row in rows[1:]] == ["Low", "Medium · Selected", "Highest", "Maximum Sharpe", "Equal weight", SPY, QQQ]
    assert document.count('class="on"') == 1
    for cells, name in zip(rows[1:6], [f"Expanding window · {name}" for name in
                                     ("Low", "Medium", "Extreme", "Maximum Sharpe", "Equal weight")]):
        path = study.equity[name]
        expected_mean = sum(path.iloc[i] / path.iloc[i - 1] - 1 for i in range(1, len(path))) * 12 / 3
        expected = [expected_mean, *(study.metrics.loc[name, key] for key in ("cagr", "volatility", "max_drawdown"))]
        assert cells[1:] == [f"{value:.1%}".replace("-", "−") for value in expected]
    assert rows[-2][1:] == [f"{value:.1%}".replace("-", "−") for value in
                           [(.99 - 1 + 1.05 / .99 - 1 + 1.10 / 1.05 - 1) * 4,
                            *(benchmarks.backtest_metrics.loc[SPY, key] for key in ("cagr", "volatility", "max_drawdown"))]]
    assert "Test returns: 3 Jan 2024 to 5 Jan 2024" in document
    assert "Baseline: 2 Jan 2024" in document and "(Expanding window)" in document
    assert "including the entry interval and fees" in document
    assert "For compound growth, this test gives no reason to prefer this rule to either ETF" in document
    assert "Medium is not the market average and can be riskier than both ETFs" in document
    assert "Latest proposed weights fit all selected prices" in document
    assert "replay uses earlier fits" in document
    # Reversing the numerical ranking must not change row order or the chosen profile.
    study.metrics["cagr"] = [-value for value in study.metrics.cagr]
    changed = table_rows(story.decision_panel_html(study, SELECTED, benchmarks))
    assert [row[0] for row in changed] == [row[0] for row in rows]


@pytest.mark.parametrize("policy,purpose", [
    ("Low", "Low targets the least fitted volatility"),
    ("Extreme", "Without a holding cap, it can allocate to one asset"),
])
def test_decision_panel_explains_selected_rule_without_recommending_it(decision_study, policy, purpose):
    study, benchmarks = decision_study
    document = story.decision_panel_html(study, f"Expanding window · {policy}", benchmarks)
    assert purpose in document
    assert "These are model rules, not recommendations" in document
    assert f"{story.display_label(policy)} · Selected" in document


def test_decision_panel_qualifies_cagr_verdict_when_net_arithmetic_mean_beats_etfs(decision_study):
    study, benchmarks = decision_study
    study.equity[SELECTED] = [1., .99, .6, 1.09]
    metrics = performance_metrics(study.equity[SELECTED], 0., 12)
    study.metrics.loc[SELECTED, list(metrics)] = list(metrics.values())
    rows = table_rows(story.decision_panel_html(study, SELECTED, benchmarks))
    assert float(rows[2][1].rstrip("%")) > max(float(row[1].rstrip("%")) for row in rows[-2:])
    document = story.decision_panel_html(study, SELECTED, benchmarks)
    assert "No. It trailed both markets" in document
    assert "For compound growth, this test gives no reason" in document
    assert "no return-based reason" not in document


def test_decision_panel_does_not_replace_a_missing_profile(decision_study):
    study, benchmarks = decision_study
    study.equity = study.equity.rename(columns={SELECTED: "Expanding window · Minimum volatility"})
    study.metrics = study.metrics.rename(index={SELECTED: "Expanding window · Minimum volatility"})
    document = story.decision_panel_html(study, SELECTED, benchmarks)
    assert "Medium was not backtested. Its result is unavailable" in document
    assert "no other rule replaces the selection" in document
    assert "Minimum volatility" not in document
    assert 'class="on"' not in document
    assert "No. It trailed" not in document


def test_decision_panel_handles_absent_or_unmatched_etfs_without_claiming_a_market_result(decision_study):
    study, benchmarks = decision_study
    for missing in (None, SimpleNamespace(backtest_equity=None, backtest_metrics=None)):
        document = story.decision_panel_html(study, SELECTED, missing)
        assert "Matching ETF results are unavailable" in document
        assert "It lost" in document and "a year" in document
        assert "trailed both markets" not in document
        assert len(table_rows(document)) == 6
    benchmarks.backtest_equity.index = benchmarks.backtest_equity.index + pd.Timedelta(days=1)
    assert "Matching ETF results are unavailable" in story.decision_panel_html(study, SELECTED, benchmarks)


def test_decision_panel_marks_partial_benchmark_coverage(decision_study):
    study, benchmarks = decision_study
    benchmarks.backtest_equity = benchmarks.backtest_equity.drop(columns=QQQ)
    benchmarks.backtest_metrics = benchmarks.backtest_metrics.drop(index=QQQ)
    document = story.decision_panel_html(study, SELECTED, benchmarks)
    assert "It trailed the S&amp;P 500 in this test" in document
    assert "One or both ETF comparisons are unavailable" in document
    assert "either ETF" not in document


def test_result_drivers_show_largest_gains_and_losses_and_reconcile_fees(decision_study):
    study, _ = decision_study
    document = story.result_drivers_html(study, SELECTED)
    rows = table_rows(document)
    assert rows[1:] == [["AAA", "+900.00"], ["DDD", "+200.00"], ["EEE", "+150.00"],
                       ["BBB", "−1,400.00"], ["CCC", "−300.00"], ["FFF", "−100.00"],
                       ["Other holdings", "−50.00"], ["Trading fees", "−100.00"], ["Net change", "−700.00"]]
    amounts = [float(row[1].replace(",", "").replace("−", "-")) for row in rows[1:]]
    assert sum(amounts[:-1]) == pytest.approx(amounts[-1])
    assert amounts[-1] == pytest.approx((study.equity[SELECTED].iloc[-1] - 1) * 10_000)
    assert "UNHELD" not in document
    assert "historical replay, not the latest proposed weights" in document
    assert "All holding contributions less trading fees equal the net change" in document


def test_result_drivers_do_not_hide_losses_behind_larger_gains_or_show_rounding_residuals(decision_study):
    study, _ = decision_study
    study.holdings[SELECTED] = pd.DataFrame({"pnl_contribution": [.3, .2, .1, -.03, -.02, -.01, 1e-14, -1e-14]},
                                          index=["GAIN1", "GAIN2", "GAIN3", "LOSS1", "LOSS2", "LOSS3", "TINY_GAIN", "TINY_LOSS"])
    study.metrics.loc[SELECTED, "total_return"] = .53
    document = story.result_drivers_html(study, SELECTED)
    assert [row[0] for row in table_rows(document)[1:]] == ["GAIN1", "GAIN2", "GAIN3", "LOSS1", "LOSS2", "LOSS3", "Trading fees", "Net change"]
    assert "TINY_" not in document
    assert "three largest gains and three largest losses" in document


def test_result_drivers_escape_names_and_show_explicit_missing_or_zero_results(decision_study):
    study, _ = decision_study
    label = '<img src=x onerror="bad()">'
    study.holdings[SELECTED] = study.holdings[SELECTED].rename(index={"BBB": label})
    document = story.result_drivers_html(study, SELECTED)
    assert label not in document and html.escape(label) in document
    assert "are unavailable" in story.result_drivers_html(study, "Expanding window · Extreme")
    study.holdings[SELECTED]["pnl_contribution"] = 0.
    study.metrics.loc[SELECTED, "total_return"] = -.01
    document = story.result_drivers_html(study, SELECTED)
    assert "No holding contributed a gain or loss" in document
    assert table_rows(document)[1:] == [["Trading fees", "−100.00"], ["Net change", "−100.00"]]


def test_result_tile_uses_loss_words_and_leaves_uncertainty_for_the_evidence_section(decision_study):
    study, benchmarks = decision_study
    benchmarks.backtest_metrics.loc[SPY, "cagr"] = -.02
    document = story.result_tile_html(study, SELECTED, benchmarks, evidence(["Historical mean underperformance"] * 2),
                                      {"universe": "Demo"})
    assert "It lost" in document and "S&amp;P 500 lost 2.0% a year" in document
    assert "grew −" not in document
    assert "pl-clarity" not in document and "statistically" not in document
    assert "How sure can we be" not in document


def test_combined_universe_context_keeps_history_and_coverage_counts():
    universe = "S&P 500 + Nasdaq-100"
    prices = pd.DataFrame(100., index=pd.bdate_range("2021-10-18", "2026-10-15"), columns=["AAA", "BBB", "CCC"])
    metadata = {"universe": universe, "universe_included": 3, "universe_requested": 5,
                "requested_start": "2021-10-16", "requested_end_exclusive": "2026-10-16"}
    result = story.context_result_html(metadata, prices)
    assert "S&amp;P 500 + Nasdaq-100" in result and "3 of 5 stocks" in result
    assert "5 years" in result and "15 Oct 2026" in result
    assert "5 years" in story.context_setup_html({"market": universe, "history": "5 years"})
    assert story.these_assets(3, universe) == "these 3 stocks"


def test_story_module_has_no_streamlit_dependency():
    assert "streamlit" not in Path(story.__file__).read_text(encoding="utf-8")
