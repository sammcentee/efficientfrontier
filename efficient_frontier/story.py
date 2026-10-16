"""Plain-language answers, HTML fragments and story charts shared by the app and the report.

Every function is pure: it reads engine results and returns text, HTML or a Plotly figure.
HTML fragments use the ``.pl-*`` classes in ``style.STORY_CSS`` and escape every data string.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .backtest_report import backtest_chart
from .benchmark_report import evidence_chart
from .style import ACCENT, CANVAS, INK_2, INK_3, display_label, style_chart


LEVEL_LABELS = {"Low": "Low", "Medium": "Medium", "Extreme": "Highest"}
VIEW_LABELS = {"Portfolio": "Overview", "Research": "Research"}
RULE_LABELS = {"Expanding window": "Refit on all past prices", "Rolling window": "Refit on recent prices",
               "Fixed rebalance": "Keep the first mix", "Buy and hold": "Buy once, then hold"}
CHART_LABELS = {"Growth": "Growth", "Drawdown": "Falls from peak", "Relative to markets": "Versus markets"}
TOPIC_LABELS = {"Efficient frontiers": "Frontier", "Risk breakdown": "Risk", "Data & coverage": "Data",
                "All backtests": "All tests", "Methodology": "Method"}
SETTING_LABELS = {
    "source": "Source", "universe": "Stocks", "price_currency": "Price currency", "compare_market": "Market comparison",
    "train_fraction": "Prices for the first fit", "risk_free_rate": "Risk-free rate", "max_weight": "Largest holding allowed",
    "shrinkage": "Covariance shrinkage", "periods_per_year": "Observations per year", "currency_assumption": "Currency assumption",
    "requested_start": "From", "requested_end_exclusive": "Until (not included)", "universe_source_url": "Index list source",
    "universe_source_date": "Index list date", "universe_retrieved_at": "Index list retrieved", "universe_requested": "Securities checked",
    "universe_included": "Securities used", "universe_excluded": "Securities left out", "universe_limitation": "Survivorship note",
    "benchmark_source": "Market price source", "benchmarks_note": "Market note", "evidence_note": "Evidence note",
    "input_file": "Price file", "universe_calendar": "Market calendar", "universe_start": "Index prices from",
    "universe_end": "Index prices to", "universe_eligibility": "Eligibility", "training_start": "First fit from",
    "training_end": "First fit to", "training_observations": "Returns in the first fit", "holdout_start": "Holdout from",
    "holdout_end": "Holdout to", "holdout_observations": "Returns in the holdout", "assets": "Assets",
    "holdout_strategy": "Holdout rule", "distributions": "Distributions", "estimation": "Estimates", "exclusions": "Not included",
    "rebalance_every": "Trade every (price rows)", "rolling_window": "Recent history for refits (return observations)",
    "cost_bps": "Trading cost", "split": "Last price row of the first fit", "include_profiles": "Risk levels tested",
    "trading_days_per_year": "Trading days per year", "cash_interest_rate": "Interest on cash", "train_start": "First fit from",
    "train_end": "First fit to", "test_start": "Test from", "test_end": "Test to", "initial_execution": "First trade",
    "execution": "Trade timing", "cost_convention": "Cost rule",
    "return_objective": "Return objective", "selection_method": "Portfolio selection",
}
STAT_LABELS = {
    "CAGR difference": "Growth gap", "Annual mean advantage": "Average gap", "Mean CI lower": "95% range, low",
    "Mean CI upper": "95% range, high", "Mean p-value": "p (one test)", "Mean Holm p-value": "Adjusted p (Holm)",
    "Benchmark beta": "Beta", "Annual alpha": "Alpha", "Windows ahead": "Stretches ahead", "Inference observations": "Paired observations",
    "Historical evidence": "Result",
}
WORDS = {
    "95% range": "A 95% confidence interval for the annual arithmetic mean return gap, not the compound growth (CAGR) gap. "
                 "A range that crosses zero includes no difference. It allows for returns that cluster in time (Newey–West).",
    "Average gap": "The average difference per return observation between your rule and the market, scaled to a year. "
                   "It differs from the growth gap, which compounds the whole path.",
    "Points": "Percentage points. If your rule grew 9.4% and the market grew 28.6%, the gap is 19.2 points, not 19.2%.",
}
# Format with k = the number of tests, for example GLOSSARY_MD.format(k=96). k = "the" reads "all the tests".
GLOSSARY_MD = (
    "### Words used here\n"
    f"**95% range.** {WORDS['95% range']}\n\n"
    f"**Average gap.** {WORDS['Average gap']}\n\n"
    f"**Points.** {WORDS['Points']}\n\n"
    "**Adjusted p.** How surprising the gap would be if there were no real difference, after we allow for all {k} tests "
    "in this report (Holm). It is not the chance that the result was luck.\n"
)
METHOD_MD = """
### What the model does
It chooses weights for all eligible assets. Weights are zero or more and add up to 100%.
Today's holdings use all of the selected history to estimate annual arithmetic mean returns and covariance.
The frontier finds the least volatile mix for each target mean return. These fitted means are not compound growth (CAGR) or forecasts.

### What the risk levels mean
**Low** is the mix with the least modelled volatility. **Medium** is the sampled frontier point nearest to the midpoint
between Low and Highest volatility. **Highest** maximises the fitted annual arithmetic mean return, with the least variance among equal solutions.
The levels are relative to the selected assets. They are not absolute limits or guarantees. Exports call Highest "Extreme".
Volatility measures how much past returns moved. It does not measure every form of risk.

### Covariance shrinkage
Covariance describes how returns move together. The estimates can be noisy.
At 0%, the model uses the sample covariance. At 100%, it ignores correlations and keeps each asset's own variance.
The default 10% reduces the cross-asset terms by 10%. This is a manual setting, not Ledoit–Wolf.

### The market test
**Buy once, then hold** (Buy and hold) invests once. **Keep the first mix** (Fixed rebalance) trades back to the first targets.
**Refit on all past prices** (Expanding window) refits with all earlier returns. **Refit on recent prices** (Rolling window) uses a fixed recent window.
Each fit selects the same mean-based risk levels from earlier returns. Each trade uses information to the previous close
and executes at the next close. The first interval stays in cash. The later test reports realised compound growth (CAGR).
Fees apply to each amount bought or sold. There is no final sale. Taxes and currency conversion are not included.

### Reading the evidence
SPY and QQQ are ETF stand-ins, not raw index series. They use the same dates and fee rate as the rule.
The realised CAGR comparison shows which rule grew more in this sample. The confidence ranges and adjusted tests concern
annual arithmetic mean return gaps, not CAGR gaps. An unclear mean gap does not establish equal performance or rule out future outperformance.
The frontier does not promise to beat either market.
The Holm adjustment covers every test in this report. It cannot correct earlier searches that you did before.
Today's members, the full-history requirement and repeated changes after review can all distort a backtest.
Save the rules and the report. Then judge them on later data that you have not seen.
"""
SKELETON_HTML = ('<div class="pl-skel-rows" aria-hidden="true">'
                 + '<div class="pl-skel-row"><span class="pl-skel"></span><span class="pl-skel"></span><span class="pl-skel"></span></div>' * 4
                 + '</div>')

LEVEL = {"Low": "low", "Medium": "medium", "Extreme": "highest"}
NOUN = {"Nasdaq-100": ("Nasdaq-100 stocks", "stock"), "Original 60 holdings": ("stocks from your original 60", "stock"),
        "My tickers": ("of your tickers", "stock"), "Upload CSV": ("assets from your file", "asset"),
        "Demo": ("simulated assets", "asset")}
PLACE = {"Nasdaq-100": "the Nasdaq-100", "My tickers": "your tickers", "Original 60 holdings": "your original 60 holdings",
         "Upload CSV": "your CSV file", "Demo": "the demo data"}
YAHOO = ("Nasdaq-100", "My tickers", "Original 60 holdings")
HELD = 0.0005
INFO_SVG = ('<svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">'
            '<circle cx="9" cy="9" r="7.25"/><path d="M9 8v5" stroke-linecap="round"/>'
            '<circle cx="9" cy="5.5" r=".9" fill="currentColor" stroke="none"/></svg>')
# One trailing comma, bracket note such as "(DE)", security type, share class or legal form. Names drop them one at a time.
_NAME_SUFFIX = re.compile(r"(\s*,|\s*\([^)]*\)|\s+(Common Stock|Common Shares|Capital Stock|Ordinary Shares|American Depositary Shares"
                          r"|New York Registry Shares|Subordinate Voting Shares|Class [A-Z]|Series [A-Z]|Inc\.?|Incorporated"
                          r"|Corporation|Corp\.?|Company|plc|PLC|N\.V\.|Ltd\.?|Limited|S\.A\.|AG|SE))\s*$")


# ----------------------------------------------------------------------------- formatting helpers

def _minus(text: str) -> str:
    return text.replace("-", "−")


def _pct(value: float, digits: int = 1, plus: bool = False) -> str:
    """Format a fraction as a percent with a true minus sign."""
    return _minus(f"{value:+.{digits}%}" if plus else f"{value:.{digits}%}")


def _growth(value: float) -> str:
    return f"{'lost' if value < 0 else 'grew'} {_pct(abs(value))}"


def _date(value) -> str:
    """Show a date as "2 Oct 2026" (no %-d, which fails on Windows)."""
    day = pd.Timestamp(value)
    return f"{day.day} {day:%b %Y}"


def _e(value) -> str:
    return html.escape(str(value))


def _market(name) -> str:
    """Return "S&P 500" for "S&P 500 (SPY)"."""
    return re.sub(r"\s*\([^)]*\)$", "", str(name))


def _noun(universe: str) -> str:
    return NOUN.get(universe, ("assets", "asset"))[1]


def these_assets(count: int, universe: str) -> str:
    """Return "these 92 stocks", or "this one asset" when there is only one."""
    return f"this one {_noun(universe)}" if count == 1 else f"these {count} {_noun(universe)}s"


def _nb(text: str) -> str:
    """Escape text and keep each hyphenated word on one line."""
    return re.sub(r"\S+-\S+", lambda match: f'<span class="pl-nb">{match.group(0)}</span>', html.escape(text))


def _held(weights: pd.Series) -> pd.Series:
    return weights[weights >= HELD].sort_values(ascending=False, kind="stable")


def _matched(benchmarks: Any):
    return benchmarks if benchmarks is not None and getattr(benchmarks, "backtest_equity", None) is not None else None


def short_company_name(name) -> str:
    """Drop the security type, share class, bracket notes and legal suffixes for display. Exported files keep the raw names."""
    text = str(name).strip()
    while True:
        shorter = _NAME_SUFFIX.sub("", text)
        if shorter == text or not shorter.strip():
            return text
        text = shorter


def strategy_label(name) -> str:
    """Show an engine strategy such as "Expanding window · Extreme" as "Refit on all past prices · Highest"."""
    method, separator, policy = str(name).partition(" · ")
    if not separator:
        return display_label(name)
    return f"{RULE_LABELS.get(method, method)} · {LEVEL_LABELS.get(policy, policy)}"


def setting_label(key) -> str:
    return SETTING_LABELS.get(key, str(key).replace("_", " ").capitalize())


def setting_text(key, value) -> str:
    """A study setting as display text. Fractions show as percents. CSV and JSON files keep the raw values."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key in ("train_fraction", "shrinkage"):
        return f"{value * 100:g}%"
    if key in ("risk_free_rate", "cash_interest_rate"):
        return f"{value:.2%}"
    if key == "max_weight":
        return "No limit" if value >= 1 else f"{value * 100:g}%"
    if key == "cost_bps":
        return f"{value:g} basis points ({value / 100:.2f}%)"
    if key in ("periods_per_year", "trading_days_per_year"):
        return f"{value:g}"
    if key == "return_objective":
        return {"historical_mean": "Annual historical arithmetic mean",
                "worst_window": "Lowest annual mean across three historical windows"}.get(value, str(value))
    if key == "selection_method" and value == "historical_mean_frontier":
        return "Historical mean and variance frontier"
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return str(value)


def company_names(coverage) -> dict:
    """Map each symbol to its short company name. Coverage is a frame or a list of records."""
    if coverage is None:
        return {}
    frame = pd.DataFrame(coverage)
    if frame.empty or not {"symbol", "name"} <= set(frame.columns):
        return {}
    return {row.symbol: short_company_name(row.name) for row in frame[["symbol", "name"]].itertuples(index=False)}


def months_tested(study: Any) -> int:
    """Length of the held-back test period in months."""
    start, end = pd.Timestamp(study.settings["test_start"]), pd.Timestamp(study.settings["test_end"])
    return round((end - start).days / 30.44)


# ----------------------------------------------------------------------------- headlines and context

def headline_text(weights: pd.Series, profile: str, universe: str) -> str:
    """Answer "What would I hold?" in one sentence."""
    held = _held(weights)
    if held.max() >= 0.5:
        return f"The {LEVEL[profile]}-risk portfolio puts {held.max():.1%} in one {_noun(universe)}."
    return f"A {LEVEL[profile]}-risk portfolio of {len(held)} {NOUN.get(universe, ('assets', 'asset'))[0]}."


def headline_html(weights: pd.Series, profile: str, universe: str) -> str:
    """The headline as escaped HTML. "Nasdaq-100 stocks" and hyphenated words do not break."""
    text = headline_text(weights, profile, universe)
    return '<span class="pl-nb">Nasdaq-100 stocks</span>'.join(_nb(part) for part in text.split("Nasdaq-100 stocks"))


def welcome_headline_html(market: str) -> str:
    return _nb(f"Find three portfolios in {PLACE.get(market, 'your data')}.")


def work_headline_html(market: str, rerun: bool = False) -> str:
    return _nb("Updating your portfolios." if rerun else f"Finding portfolios in {PLACE.get(market, 'your data')}.")


def _history(metadata: dict) -> str | None:
    """Return "5 years" or "10 years" when the Yahoo request used a preset, else None."""
    start, end = metadata.get("requested_start"), metadata.get("requested_end_exclusive")
    if not start or not end:
        return None
    for years in (5, 10):
        if pd.Timestamp(end) - pd.DateOffset(years=years) == pd.Timestamp(start):
            return f"{years} years"
    return None


def _context(parts: list[str], extra: str = "") -> str:
    return '<p class="pl-context">' + '<span class="pl-sep">·</span>'.join(parts) + extra + "</p>"


def context_result_html(metadata: dict, prices: pd.DataFrame) -> str:
    """The context line for a result, for example "Nasdaq-100 · 92 of 101 stocks · 5 years to 2 Oct 2026"."""
    universe = str(metadata.get("universe", ""))
    first, last = _date(prices.index[0]), _date(prices.index[-1])
    history = _history(metadata) if universe in YAHOO else None
    period = (f'<span>{history}<span class="hide-m"> to {last}</span></span>' if history
              else f"<span>{first} to {last}</span>")
    if universe == "Nasdaq-100":
        count = f'{metadata.get("universe_included", len(prices.columns))} of {metadata.get("universe_requested", len(prices.columns))} stocks'
        parts = ["<span>Nasdaq-100</span>", f"<span>{_e(count)}</span>", period]
    else:
        label = f'CSV: {metadata["input_file"]}' if universe == "Upload CSV" and metadata.get("input_file") else universe
        count = f"{len(prices.columns)} {_noun(universe)}{'' if len(prices.columns) == 1 else 's'}"
        parts = [f"<span>{_e(label)}</span>", f"<span>{count}</span>", period]
    synthetic = any(word in str(metadata.get("source", "")).lower() for word in ("demo", "synthetic"))
    return _context(parts, ' <span class="pl-badge">Simulated</span>' if synthetic else "")


def context_setup_html(setup: dict, badge: str | None = None) -> str:
    """The context line before a result, from the applied setup, for example "Nasdaq-100 · 5 years"."""
    market = setup.get("market", "")
    if market == "Upload CSV":
        label = f"CSV: {setup['prices_file'][0]}" if setup.get("prices_file") else "Upload CSV"
    else:
        label = market
    parts = [f"<span>{_e(label)}</span>"]
    if market in YAHOO:
        history = setup.get("history", "5 years")
        if history == "Custom dates":
            history = f"{_date(setup['start'])} to {_date(pd.Timestamp(setup['end']) - pd.Timedelta(days=1))}"
        parts.append(f"<span>{_e(history)}</span>")
    return _context(parts, f' <span class="pl-badge">{_e(badge)}</span>' if badge else "")


# ----------------------------------------------------------------------------- holdings

def stats_html(latest: Any, profile: str, universe: str, names: dict | None = None) -> str:
    """Fitted mean return, volatility and largest holding, divided by hairlines."""
    weights = latest.portfolios[profile].weights
    largest = weights.idxmax()
    stats = (
        (_pct(latest.summary.loc[profile, "expected_return"]), "Fitted annual mean", "Full-history estimate before fees"),
        (_pct(latest.portfolios[profile].volatility), "Volatility", "Yearly, from past prices"),
        (_pct(weights.max()), "Largest holding", (names or {}).get(largest, largest)),
    )
    return '<div class="pl-stats">' + "".join(
        f'<div class="pl-stat"><span class="pl-num">{number}</span><span class="pl-lbl">{label}</span><small>{_e(sub)}</small></div>'
        for number, label, sub in stats) + "</div>"


def list_html(weights: pd.Series, names: dict | None = None) -> str:
    """The proposed holdings and one bar row per position. More than 12 rows end in one rest row."""
    held = _held(weights)
    top = held.iloc[0]
    shown = held if len(held) <= 12 else held.iloc[:10]
    rows = []
    for index, (symbol, weight) in enumerate(shown.items()):
        name = (names or {}).get(symbol)
        label = f"<b>{_e(name)}</b><small>{_e(symbol)}</small>" if name else f"<b>{_e(symbol)}</b>"
        rows.append(f'<li class="pl-row" style="--i:{index}"><span class="pl-name">{label}</span>'
                    f'<span class="pl-bar"><i style="width:{weight / top * 100:.1f}%"></i></span>'
                    f'<span class="pl-pct">{_pct(weight)}</span></li>')
    if len(held) > 12:
        rest = 1 - shown.sum()
        rows.append(f'<li class="pl-row is-rest" style="--i:10"><span class="pl-name"><b>{len(held) - 10} more holdings</b>'
                    f'<small>The largest of these is {_pct(held.iloc[10])}.</small></span>'
                    f'<span class="pl-bar"><i style="width:{min(rest / top, 1) * 100:.1f}%"></i></span>'
                    f'<span class="pl-pct">{_pct(rest)}</span></li>')
    return ('<div class="pl-head"><h3 class="pl-title">Proposed holdings</h3><small>Share of the portfolio</small></div>'
            f'<ul class="pl-list">{"".join(rows)}</ul>')


def callout_html(weights: pd.Series, universe: str, names: dict | None = None) -> str:
    """A concentration note when one holding is 50% or more. Empty otherwise."""
    largest = weights.idxmax()
    if weights.max() < 0.5:
        return ""
    name = (names or {}).get(largest, largest)
    title = f"Almost all of it is {name}." if weights.max() >= 0.8 else f"At least half of it is {name}."
    kind = "company" if _noun(universe) == "stock" else "asset"
    return (f'<div class="pl-callout"><span class="pl-glyph" aria-hidden="true">!</span><p><b>{_e(title)}</b>'
            f"This is a large allocation to one {kind}. To spread the risk, select Change, "
            "then Model assumptions, and set a holding limit.</p></div>")


def why_text(latest: Any, profile: str, universe: str) -> list[str]:
    """The three paragraphs of "Why these stocks?"."""
    noun = f"{_noun(universe)}s"
    years = round(float(latest.windows["years"].sum()), 1)
    count = len(latest.portfolios[profile].weights)
    mixes = f"The model tries weighted mixes of all {count} {noun}." if count > 1 else f"The model has one {_noun(universe)} to hold."
    return [
        f"{mixes} It uses all {years:g} years to estimate annual arithmetic mean returns and volatility. "
        "The frontier finds the least volatile mix for each target mean return.",
        f"This mix has a fitted annual arithmetic mean of {_pct(latest.summary.loc[profile, 'expected_return'])}. "
        "It uses the same prices that selected today's holdings. It is not compound growth (CAGR) or a forecast. "
        "The next section fits the rule using earlier prices and measures realised growth on later prices.",
        "Low has the least fitted volatility. Medium is the sampled frontier point nearest to the midpoint between Low and Highest volatility. "
        f"Highest maximises the fitted annual mean return. These levels are relative to the selected {noun}; "
        "Highest uses no borrowing and does not maximise every measure of risk.",
    ]


def levels_ledger_html(latest: Any, selected: str, names: dict | None = None, universe: str = "Nasdaq-100") -> str:
    """The "How the three levels compare" table with its footnote."""
    ramp = {"Low": "var(--pl-risk-low)", "Medium": "var(--pl-risk-med)", "Extreme": "var(--pl-risk-high)"}
    count_label = "Stocks held" if _noun(universe) == "stock" else "Assets held"
    rows = []
    for profile in latest.summary.index:
        weights = latest.portfolios[profile].weights
        largest = weights.idxmax()
        css = ' class="on"' if profile == selected else ""
        rows.append(
            f"<tr{css}>"
            f'<td><i class="pl-ldot" style="background:{ramp.get(profile, "var(--pl-rest)")}"></i>{_e(display_label(profile))}</td>'
            f'<td>{_pct(latest.summary.loc[profile, "volatility"])}</td>'
            f'<td class="hide-m">{_e((names or {}).get(largest, largest))} {_pct(weights.max())}</td>'
            f"<td>{len(_held(weights))}</td><td>{_pct(latest.summary.loc[profile, 'expected_return'])}</td></tr>")
    return ('<table class="pl-ledger"><thead><tr><th>Level</th><th>Volatility</th><th class="hide-m">Largest holding</th>'
            f'<th>{count_label}</th><th>Historical mean*</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="pl-small">*Annual arithmetic mean fitted to all selected prices. This is not later realised '
            "compound growth (CAGR) or a forecast.</p>")


# ----------------------------------------------------------------------------- market test

def market_answer(study: Any, selected: str, benchmarks: Any) -> str:
    """Answer "Would this rule have beaten the market?" from CAGR, compared at 0.1 percentage point."""
    cagr = study.metrics.loc[selected, "cagr"]
    matched = _matched(benchmarks)
    if matched is None:
        return f"It {_growth(cagr)} a year."
    outcomes = {}
    for name, row in matched.backtest_metrics.iterrows():
        mine, theirs = round(cagr * 1000), round(row["cagr"] * 1000)
        outcomes[_market(name)] = "beat" if mine > theirs else "trailed" if mine < theirs else "matched"
    if len(outcomes) == 1:
        name, outcome = next(iter(outcomes.items()))
        return f"It {outcome} the {name} in this test."
    if len(set(outcomes.values())) == 1:
        outcome = next(iter(outcomes.values()))
        return {"trailed": "No. It trailed both markets.", "beat": "Yes. It beat both markets in this test.",
                "matched": "It matched both markets in this test."}[outcome]
    rank = {"beat": 0, "matched": 1, "trailed": 2}
    (better, first), (worse, second) = sorted(((outcome, name) for name, outcome in outcomes.items()), key=lambda item: rank[item[0]])
    return f"Partly. It {better} the {first} but {worse} the {second}."


def decision_panel_html(study: Any, selected: str, benchmarks: Any) -> str:
    """Compare the requested rule with fixed alternatives, without choosing a winner."""
    method, _, policy = str(selected).partition(" · ")
    available = selected in study.equity and selected in study.metrics.index
    matched = _matched(benchmarks)
    if matched is not None and not matched.backtest_equity.index.equals(study.equity.index):
        matched = None
    methods = ("Low", "Medium", "Extreme", "Maximum Sharpe", "Equal weight")
    rows = [(f"{method} · {name}", display_label(name), study.equity, study.metrics) for name in methods
            if f"{method} · {name}" in study.equity and f"{method} · {name}" in study.metrics.index]
    etf_count = 0
    if matched is not None:
        for ticker in ("SPY", "QQQ"):
            name = next((name for name in matched.backtest_metrics.index
                         if str(name) == ticker or str(name).endswith(f"({ticker})")), None)
            if name is not None and name in matched.backtest_equity:
                rows.append((name, str(name), matched.backtest_equity, matched.backtest_metrics))
                etf_count += 1
    table_rows = []
    for name, label, equity, metrics in rows:
        row = metrics.loc[name]
        mean = equity[name].pct_change(fill_method=None).iloc[1:].mean() * study.settings["periods_per_year"]
        css = ' class="on"' if name == selected else ""
        label = _e(label) + (" · Selected" if name == selected else "")
        values = (mean, row["cagr"], row["volatility"], row["max_drawdown"])
        table_rows.append(f"<tr{css}><th scope=\"row\">{label}</th>"
                          + "".join(f"<td>{_pct(value)}</td>" for value in values) + "</tr>")
    if available:
        verdict = market_answer(study, selected, matched)
        implication = ""
        if matched is not None and etf_count == len(matched.backtest_metrics) == 2 and all(
            round(study.metrics.loc[selected, "cagr"] * 1000) < round(value * 1000)
            for value in matched.backtest_metrics["cagr"]
        ):
            implication = " For compound growth, this test gives no reason to prefer this rule to either ETF."
    else:
        verdict = f"{strategy_label(selected)} was not backtested. Its result is unavailable."
        implication = " The table shows the available comparisons; no other rule replaces the selection."
    purpose = {
        "Low": "Low targets the least fitted volatility.",
        "Medium": "Medium selects the sampled frontier point nearest to the midpoint of Low and Highest fitted volatility. "
                  "Medium is not the market average and can be riskier than both ETFs.",
        "Extreme": "Highest maximises the fitted historical arithmetic mean return. Without a holding cap, it can allocate to one asset.",
        "Maximum Sharpe": "Maximum Sharpe targets fitted excess return per unit of volatility.",
        "Equal weight": "Equal weight assigns the same target weight to each asset.",
    }.get(policy, "")
    dates = study.equity.index
    missing = ("<p>Matching ETF results are unavailable for these test dates.</p>" if matched is None else
               "<p>One or both ETF comparisons are unavailable for these test dates.</p>" if etf_count < 2 else "")
    return (
        '<h3 class="pl-title">Compare rules and ETFs</h3>'
        f'<p class="pl-sub">Selected: {_e(strategy_label(selected))}. Test returns: {_date(dates[1])} to {_date(dates[-1])}. '
        f'Baseline: {_date(dates[0])}. Rule: {_e(RULE_LABELS.get(method, method))} ({_e(method)}).</p>'
        f'<p class="pl-verdict">{_e(verdict)}{_e(implication)}</p>{missing}'
        '<div class="scroll" style="overflow-x:auto"><table class="pl-ledger"><thead><tr><th>Rule / ETF</th>'
        '<th>Test mean / year</th><th>Growth / year (CAGR)</th><th>Volatility / year</th><th>Maximum drawdown</th>'
        f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>'
        '<p class="pl-small">Test mean is the arithmetic mean of net portfolio returns, including the entry interval and fees, '
        'scaled to a year. CAGR compounds the same net path. Rows follow a fixed order, not a performance ranking.</p>'
        f'<p>{_e(purpose)} These are model rules, not recommendations. Risk differences are shown in the table.</p>'
        '<p>Latest proposed weights fit all selected prices. This replay uses earlier fits, so its holdings and weights '
        'can differ from the latest proposal.</p>'
    )


def result_drivers_html(study: Any, selected: str) -> str:
    """Show gains and losses from the replay and reconcile all holdings less fees."""
    heading = '<h3 class="pl-title">What drove the result?</h3>'
    if selected not in study.holdings or selected not in study.metrics.index:
        return heading + f"<p>Holding contributions for {_e(strategy_label(selected))} are unavailable.</p>"
    contributions = study.holdings[selected]["pnl_contribution"]
    visible = contributions[(contributions * 10_000).round(2) != 0]
    gains = visible[visible > 0].sort_values(ascending=False, kind="stable").head(3)
    losses = visible[visible < 0].sort_values(kind="stable").head(3)
    shown = [*gains.index, *losses.index]
    amounts = [(str(name), float(contributions.loc[name])) for name in shown]
    other = float(contributions.drop(shown).sum())
    if round(other * 10_000, 2):
        amounts.append(("Other holdings", other))
    amounts += [("Trading fees", -float(study.metrics.loc[selected, "total_cost"])),
                ("Net change", float(study.metrics.loc[selected, "total_return"]))]
    rows = "".join(
        f'<tr><th scope="row">{_e(name)}</th><td>{_minus(f"{value * 10_000:+,.2f}") if value else "0.00"}</td></tr>'
        for name, value in amounts
    )
    note = ("Up to three largest gains and three largest losses are shown, with gains first. "
            "Remaining contributions are combined. Amounts that round to zero are not listed separately."
            if len(visible) else "No holding contributed a gain or loss in this test." if not contributions.any() else
            "No holding contribution is visible at this precision.")
    return (
        heading + f'<p class="pl-sub">{_e(strategy_label(selected))}. {note}</p>'
        '<table class="pl-ledger"><thead><tr><th>Holding / cost</th><th>Gain or loss per initial 10,000</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p class="pl-small">These contributions come from holdings in the historical replay, not the latest proposed weights. '
        'All holding contributions less trading fees equal the net change. Amounts use initial capital as the denominator.</p>'
    )


def market_lead(study: Any, selected: str) -> str:
    """The lead under the market answer: what was replayed, how often it traded and what it paid."""
    method, _, policy = str(selected).partition(" · ")
    level = LEVEL.get(policy, policy.lower().replace(" ", "-"))
    settings = study.settings
    every, cost = int(settings["rebalance_every"]), settings["cost_bps"] / 100
    unit = "trading days" if settings.get("periods_per_year", 252) == 252 else "price rows"
    rule = {
        "Expanding window": (("About once a month" if every == 21 and unit == "trading days" else f"Every {every} {unit}")
                             + " it refitted using all earlier returns."),
        "Rolling window": f"Every {every} {unit} it refitted on the most recent {settings['rolling_window']} return observations.",
        "Fixed rebalance": f"It kept the mix from the first fit. Every {every} {unit} it traded back to that mix.",
        "Buy and hold": "It bought the mix from the first fit once and then held it.",
    }.get(method, "")
    selection = (" Each fit selected the risk level from a frontier of annual historical arithmetic mean returns and volatility."
                 if policy in LEVEL else "")
    return (f"We replayed the {level}-risk rule over the last {months_tested(study)} months. {rule}{selection} "
            "Fits used returns through the prior close; targets took effect at the following close. "
            f"The first interval stayed in cash. It paid {cost:.2f}% on each amount bought or sold. "
            "The result is later realised compound growth (CAGR), not the fitted mean return.")


def result_tile_html(study: Any, selected: str, benchmarks: Any, evidence: Any, metadata: dict) -> str:
    """Kicker, big numbers, realised result and the must-read note for the result tile."""
    matched = _matched(benchmarks)
    equity, metrics = study.equity, study.metrics.loc[selected]
    columns = [("rule", "Your rule", equity[selected].iloc[-1], metrics["total_return"], metrics["cagr"])]
    if matched is not None:
        for name, row in matched.backtest_metrics.iterrows():
            key = "spy" if "SPY" in str(name) else "qqq"
            columns.append((key, _market(name), matched.backtest_equity[name].iloc[-1], row["total_return"], row["cagr"]))
    numbers = "".join(
        f'<div class="pl-bignum"><span class="pl-key"><i class="{key}"></i>{_e(label)}</span>'
        f'<span class="pl-num">{value * 10_000:,.0f}</span><small><span>{_pct(total, plus=True)}</span>'
        f'<span class="pl-sep"> · </span><span>{_pct(cagr)} a year</span></small></div>'
        for key, label, value, total, cagr in columns)
    if matched is not None:
        markets = " and the ".join(f"{_e(label)} {_growth(cagr)} a year" for _, label, _, _, cagr in columns[1:])
        verdict = f"It {_growth(metrics['cagr'])} a year. <span>The {markets}.</span>"
    else:
        # The section answer already gives the yearly growth. Without markets there is no gap to judge.
        total = metrics["total_return"]
        verdict = f"In total, it {'gained' if total >= 0 else 'lost'} {_pct(abs(total))}."
    months = months_tested(study)
    nasdaq = metadata.get("universe") == "Nasdaq-100"
    if nasdaq:
        bold = "A test on today’s winners." if months >= 36 else "A short test on today’s winners."
        why = ("The stock list is today’s Nasdaq-100, so companies that left the index are missing. "
               "That tends to flatter past results.")
    else:
        bold = "A fixed list." if months >= 36 else "A short test on a fixed list."
        why = "The asset list is fixed for the whole period. That can favour survivors."
    held_back = (f"Only the last {months} months ({_date(study.settings['test_start'])} to {_date(study.settings['test_end'])}) "
                 "were held back for this test.")
    return (f'<span class="pl-kicker">10,000 on {_date(equity.index[0])} became, by {_date(equity.index[-1])}</span>'
            f'<div class="pl-bignums">{numbers}</div><p class="pl-verdict">{verdict}</p>'
            f'<p class="pl-mustread">{INFO_SVG}<span><b>{bold}</b> {held_back} {why}</span></p>')


def chart_footnote(study: Any, benchmarks: Any) -> str:
    """The cash-first and entry-cost note under the market chart."""
    start, entry = _date(study.equity.index[0]), _date(study.settings["initial_execution"])
    if _matched(benchmarks) is None:
        return f"It starts with 10,000 in cash on {start} and invests at the close on {entry}. Taxes are not included."
    return (f"All three start with 10,000 in cash on {start} and invest at the close on {entry}. The two markets pay the same "
            f"{study.settings['cost_bps'] / 100:.2f}% entry cost. Taxes are not included.")


def section_header_html(anchor: str, eyebrow: str, answer: str, lead: str = "") -> str:
    """A section that opens with its question as the eyebrow and its answer as the headline."""
    lead_html = f'<p class="pl-lead">{_e(lead)}</p>' if lead else ""
    return (f'<section class="pl-section" id="{_e(anchor)}"><p class="pl-eyebrow">{_e(eyebrow)}</p>'
            f'<h2 class="pl-answer" tabindex="-1">{_nb(answer)}</h2>{lead_html}</section>')


# ----------------------------------------------------------------------------- evidence

_CATEGORY = {"No clear statistical advantage": "unclear", "Historical mean underperformance": "behind",
             "Historical mean advantage": "ahead", "Historical mean advantage and alpha": "ahead",
             "Historical alpha": "alpha", "Insufficient history": "insufficient", "Uncertainty unavailable": "unavailable"}


def _statuses(evidence: Any, selected: str) -> list[tuple[str, str, float]]:
    rows = evidence.summary.xs(selected, level="strategy")
    return [(_market(name), _CATEGORY.get(str(row["status"]), "unavailable"), row["cagr_difference"]) for name, row in rows.iterrows()]


def evidence_answer(evidence: Any, selected: str) -> str:
    """Answer "How sure can we be?"."""
    if evidence is None:
        return "The uncertainty could not be estimated."
    statuses = _statuses(evidence, selected)
    categories = {category for _, category, _ in statuses}
    if len(categories) == 1:
        return {"unclear": "No statistically clear difference in mean return.", "behind": "Lower mean returns than both markets, in this sample.",
                "ahead": "Higher mean returns than both markets, in this sample.",
                "alpha": "Some evidence of alpha. No clear lead in average return.",
                "insufficient": "Too little history to judge.",
                "unavailable": "The uncertainty could not be estimated."}[categories.pop()]
    short = {"unclear": "No clear mean-return difference against the {m}.", "behind": "Lower mean returns than the {m}, in this sample.",
             "ahead": "Higher mean returns than the {m}, in this sample.", "alpha": "Some evidence of alpha against the {m}.",
             "insufficient": "Too little history against the {m}.", "unavailable": "No estimate against the {m}."}
    return " ".join(short[category].format(m=name) for name, category, _ in statuses)


def evidence_lead(evidence: Any, selected: str, months: int, metadata: dict | None = None) -> str:
    """Separate realised compound growth from uncertainty in arithmetic mean returns."""
    if evidence is None:
        return (metadata or {}).get("evidence_note", "")
    statuses = _statuses(evidence, selected)
    tests = evidence.settings.get("multiple_testing_tests", 0)
    directions = ["led" if gap > 0 else "trailed" if gap < 0 else "matched" for _, _, gap in statuses]
    comparison = (f"{directions[0]} both markets" if len(set(directions)) == 1 else
                  " and ".join(f"{direction} the {name}" for direction, (name, _, _) in zip(directions, statuses)))
    growth = f"The rule {comparison} on realised compound growth (CAGR) in this sample."
    if months < 36:
        growth += f" {months} months is a short test."
    distinction = "The confidence ranges and adjusted tests concern annual arithmetic mean return gaps, not CAGR gaps."
    categories = {category for _, category, _ in statuses}
    if len(categories) > 1:
        inference = f"The statistical evidence differs by market after adjustment for all {tests} tests."
    else:
        inference = {
            "unclear": f"Neither annual mean return gap is statistically clear after allowing for returns that cluster in time "
                       f"and all {tests} tests. This does not establish equal performance.",
            "behind": f"The annual mean return gaps are statistically negative after adjustment for all {tests} tests. "
                      "This describes the sample, not future performance.",
            "ahead": f"The annual mean return gaps are statistically positive after adjustment for all {tests} tests. "
                     "This describes the sample, not future performance.",
            "alpha": "Alpha adjusts for exposure to one market. It does not prove skill.",
            "insufficient": f"The tests need at least {evidence.settings.get('inference_min_observations', 60)} paired return observations.",
            "unavailable": (metadata or {}).get("evidence_note", "The mean-return uncertainty could not be estimated."),
        }[categories.pop()]
    return f"{growth} {distinction} {inference}"


def clarity_text(evidence: Any, selected: str) -> str:
    """One line on statistical clarity for the result tile."""
    unknown = "The statistics could not be estimated for this test."
    if evidence is None:
        return unknown
    statuses = _statuses(evidence, selected)
    if any(category in ("insufficient", "unavailable") for _, category, _ in statuses):
        return unknown
    clear = [name for name, category, _ in statuses if category in ("ahead", "behind")]
    if not clear:
        return "Neither annual mean return gap is statistically clear. This does not test the CAGR gaps. See How sure can we be."
    if len(clear) == len(statuses):
        return "Both annual mean return gaps are statistically clear in this sample. This does not test the CAGR gaps. See How sure can we be."
    return (f"Only the annual mean return gap to the {clear[0]} is statistically clear in this sample. "
            "This does not test the CAGR gaps. See How sure can we be.")


def _signed(value: float) -> str:
    """One decimal with a sign and a true minus. Zero has no sign."""
    return "0.0" if round(value, 1) == 0 else _minus(f"{value:+.1f}")


def _tick(value: float) -> str:
    value = round(value, 6)
    return "0" if value == 0 else _minus(f"{value:+g}")


def _interval_axis(values: list[float]) -> tuple[float, float, float]:
    """Padded axis limits on whole steps, with 4 to 6 ticks."""
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    span = (high - low) or 1.0
    low, high = low - span * 0.1, high + span * 0.1
    preferred = (5, 10, 20, 25, 50)
    others = sorted({m * 10.0 ** k for k in range(-2, 5) for m in (1, 2, 2.5, 5)} - set(preferred))
    best = None
    for step in (*preferred, *others):
        first, last = math.floor(low / step) * step, math.ceil(high / step) * step
        count = round((last - first) / step) + 1
        if 4 <= count <= 6:
            return first, last, step
        if best is None or abs(count - 5) < abs(best[3] - 5):
            best = (first, last, step, count)
    return best[:3]


def interval_html(evidence: Any, strategy: str) -> str:
    """Average yearly gap with its 95% range against each market, as an HTML forest plot."""
    rows = evidence.summary.xs(strategy, level="strategy")
    data = [(_market(name), row["annual_advantage"] * 100, row["advantage_ci_low"] * 100, row["advantage_ci_high"] * 100)
            for name, row in rows.iterrows()]
    values = [value for _, *numbers in data for value in numbers if math.isfinite(value)] or [0.0]
    low, high, step = _interval_axis(values)

    def position(value):
        return (value - low) / (high - low) * 100

    zero = position(0)
    body = []
    for name, estimate, ci_low, ci_high in data:
        has_range = math.isfinite(ci_low) and math.isfinite(ci_high)
        span = (f'<i class="f-ci" style="left:{position(ci_low):.2f}%;width:{position(ci_high) - position(ci_low):.2f}%"></i>'
                if has_range else "")
        dot = f'<i class="f-dot" style="left:{position(estimate):.2f}%"></i>' if math.isfinite(estimate) else ""
        detail = f"range {_signed(ci_low)} to {_signed(ci_high)}" if has_range else "no range for this test"
        body.append(f'<div class="f-row"><div class="f-label">vs {_e(name)}</div>'
                    f'<div class="f-value"><b>{_signed(estimate) if math.isfinite(estimate) else "—"}</b> points<small>{detail}</small></div>'
                    f'<div class="f-plot"><i class="f-zero" style="left:{zero:.2f}%"></i>{span}{dot}</div></div>')
    count = round((high - low) / step) + 1
    ticks = "".join(f'<span style="left:{position(low + i * step):.2f}%">{_tick(low + i * step)}</span>' for i in range(count))
    return ('<div class="pl-forest"><div class="f-sides"><div></div><div class="mid">'
            f'<span class="l" style="width:{zero:.2f}%">← Rule behind</span>'
            '<span class="r">Rule ahead →</span></div></div>'
            + "".join(body) + f'<div class="f-axis"><div></div><div class="ticks">{ticks}</div></div></div>')


def _windows(evidence: Any, strategy: str) -> pd.DataFrame:
    return evidence.windows.loc[evidence.windows["strategy"] == strategy]


def stretch_summary_html(evidence: Any, strategy: str) -> str:
    """The large "1 of 3" summary for the stretch tile."""
    rows = evidence.summary.xs(strategy, level="strategy")
    total = evidence.settings.get("window_count", 3)
    counts = [int(value) for value in rows["windows_ahead"]]
    if len(set(counts)) == 1:
        number, label = f"{counts[0]} of {total}", "stretches ahead of each market"
    else:
        number = " · ".join(f"{count} of {total}" for count in counts)
        label = "stretches ahead of " + " · ".join(f"the {_market(name)}" for name in rows.index)
    return f'<div class="pl-stat"><span class="pl-num">{number}</span><span class="pl-lbl">{_e(label)}</span></div>'


def stretch_table_html(evidence: Any, strategy: str) -> str:
    """Change against each market in each of the three test stretches."""
    windows = _windows(evidence, strategy)
    markets = list(dict.fromkeys(windows["benchmark"]))
    first = windows.loc[windows["benchmark"] == markets[0]]
    head = "".join(f"<th>{pd.Timestamp(row.start):%b %Y} to {pd.Timestamp(row.end):%b %Y}</th>" for row in first.itertuples())
    body = []
    for market in markets:
        cells = []
        for row in windows.loc[windows["benchmark"] == market].itertuples():
            ahead = row.relative_return > 0
            cells.append(f'<td><i class="w-dot{" on" if ahead else ""}" aria-hidden="true"></i>{_pct(row.relative_return, plus=True)}'
                         f'<small>{"ahead" if ahead else "behind"}</small></td>')
        body.append(f"<tr><td>{_e(_market(market))}</td>{''.join(cells)}</tr>")
    return f'<table class="pl-wtable"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# ----------------------------------------------------------------------------- fine print and work card

def fine_print_items(metadata: dict, prices: pd.DataFrame, coverage_place: str = "Research › Data") -> list[str]:
    """The four caveats as HTML paragraphs' content. ``coverage_place`` says where the coverage table is."""
    universe = metadata.get("universe", "")
    count = len(prices.columns)
    items = [
        "<b>Research, not advice.</b> Every number comes from past prices. None of it is a forecast.",
        f"<b>Relative risk.</b> Low, Medium and Highest compare mixes of {these_assets(count, universe)} only. "
        "They are not absolute risk ratings.",
    ]
    if universe == "Nasdaq-100":
        items.append(f"<b>Survivors only.</b> The list is today’s Nasdaq-100. {_e(metadata.get('universe_excluded', 0))} of "
                     f"{_e(metadata.get('universe_requested', count))} stocks were left out for missing prices. "
                     f"See {_e(coverage_place)}.")
    else:
        items.append("<b>Fixed list.</b> The study keeps the same assets for the whole period. That can favour survivors.")
    source = str(metadata.get("source", ""))
    if metadata.get("price_currency", "USD") != "USD":
        currency = ("You declared a common price currency. Market comparisons are off, because they need USD prices. "
                    "No currency conversion occurs.")
    elif "synthetic" in source.lower() or source == "Demo":
        currency = "The demo prices are simulated. No currency conversion occurs."
    else:
        currency = "You declared USD prices. The app does not check or convert currencies."
    items.append(f"<b>Currency.</b> {currency}")
    return items


def fine_print_html(metadata: dict, prices: pd.DataFrame) -> str:
    return ('<section class="pl-fine"><h3>The fine print, in plain words</h3>'
            + "".join(f"<p>{item}</p>" for item in fine_print_items(metadata, prices)) + "</section>")


def steps_html(market: str, step: int, *, compare: bool = True, detail: str | None = None) -> str:
    """The work timeline. Steps before ``step`` are done, ``step`` is running, later steps wait. Use 5 for all done."""
    second = {"Upload CSV": "Read your price file", "Demo": "Make the simulated prices"}.get(
        market, "Download daily prices from Yahoo Finance")
    fourth = ("Replay the test period against the S&P 500 and the Nasdaq-100" if compare and market != "Demo"
              else "Replay the test period")
    labels = {1: "Read today’s Nasdaq-100 list", 2: second, 3: "Find the three portfolios", 4: fourth}
    if market != "Nasdaq-100":
        del labels[1]
    items = []
    for number, label in labels.items():
        state = "done" if number < step else "now" if number == step else "next"
        sub = f"<small>{_e(detail)}</small>" if detail and number == 2 and state == "now" else ""
        items.append(f'<li class="{state}"><i aria-hidden="true">{"✓" if state == "done" else ""}</i>'
                     f"<span>{_e(label)}{sub}</span></li>")
    return f'<ol class="pl-steps">{"".join(items)}</ol>'


# ----------------------------------------------------------------------------- story chart

def add_test_windows(figure: go.Figure, windows: pd.DataFrame) -> go.Figure:
    """Shade test stretches 1 and 3 and label all three. Bands go after the base chart's own shapes."""
    for number, row in enumerate(windows.itertuples(), start=1):
        if number in (1, 3):
            figure.add_vrect(x0=row.start, x1=row.end, fillcolor=CANVAS, opacity=1, layer="below", line_width=0)
        figure.add_annotation(text=f"Stretch {number}", xref="x", x=row.start, xanchor="left", xshift=4, yref="paper", y=1,
                              yanchor="top", font={"size": 11, "color": INK_3}, showarrow=False)
    return figure


def add_end_labels(figure: go.Figure, fmt: str) -> go.Figure:
    """Write each line's last value at the right edge, pushed apart so the labels do not overlap."""
    lines = [trace for trace in figure.data if trace.mode == "lines" and trace.y is not None and len(trace.y)]
    if not lines:
        return figure
    values = pd.concat([pd.Series(trace.y, dtype=float) for trace in lines]).dropna()
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * 0.06 or abs(high) * 0.06 or 1.0
    low, high = low - pad, high + pad
    figure.update_yaxes(range=[low, high])
    gap = (high - low) * 0.07
    labels = sorted(((float(trace.y[-1]), trace) for trace in lines), key=lambda item: item[0])
    previous = None
    for value, trace in labels:
        position = value if previous is None else max(value, previous + gap)
        previous = position
        figure.add_annotation(text=f'<span style="color:{trace.line.color}">●</span> {fmt.format(value)}', xref="paper", x=1,
                              xanchor="left", xshift=6, yref="y", y=position, font={"size": 12, "color": INK_2}, showarrow=False)
    return figure


def story_chart(study: Any, selected: str, mode: str, benchmarks: Any = None, evidence: Any = None) -> go.Figure:
    """The overview market chart: growth, falls from peak or the ratio to each market, with test stretches."""
    matched = _matched(benchmarks)
    relative = mode == "Relative to markets" and matched is not None and evidence is not None
    drawdown = mode == "Drawdown"
    if relative:
        figure = evidence_chart(evidence, selected)
        figure.update_traces(hovertemplate="%{fullData.name}: %{y:.2f}×<extra></extra>")
    else:
        figure = backtest_chart(study, drawdown=drawdown, strategies=[selected],
                                benchmark_equity=matched.backtest_equity if matched is not None else None)
        figure.data[0].name = "Your rule"
        figure.data[0].line = {"color": ACCENT, "width": 2.5}
        if drawdown:
            figure.data[0].update(fill="tozeroy", fillcolor="rgba(0,113,227,.10)")
        for trace in figure.data[1:]:
            trace.line.width = 1.75
        figure.update_traces(hovertemplate="%{fullData.name}: " + ("%{y:.1%}" if drawdown else "%{y:,.0f}") + "<extra></extra>")
    style_chart(figure, None, height=320, hovermode="x unified", story=True)
    figure.update_xaxes(tickformat="%b %Y", hoverformat="%-d %b %Y", nticks=6)
    figure.update_yaxes(tickformat=".2f" if relative else ".0%" if drawdown else "~s", ticksuffix="×" if relative else "")
    if evidence is not None and selected in set(evidence.windows["strategy"]):
        windows = _windows(evidence, selected)
        add_test_windows(figure, windows.loc[windows["benchmark"] == windows["benchmark"].iloc[0]])
        if drawdown:
            # Room above the 0% line for the stretch labels. It is less than one tick step, so no tick shows above 0%.
            low = float(pd.concat([pd.Series(trace.y, dtype=float) for trace in figure.data]).min())
            if low < 0:
                figure.update_yaxes(range=[low * 1.06, -low * 0.12])
    if not drawdown:
        add_end_labels(figure, "{:.2f}×" if relative else "{:,.0f}")
    return figure

