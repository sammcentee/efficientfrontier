"""Run with ./run.sh from the repository root."""

from datetime import date
from hashlib import sha256
import html
from io import BytesIO
import math
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from efficient_frontier import presentation, story
from efficient_frontier.backtest import run_backtests
from efficient_frontier.backtest_report import HOLDINGS, METRICS, backtest_chart
from efficient_frontier.benchmark_report import evidence_summary_frame
from efficient_frontier.benchmarks import compare_benchmarks
from efficient_frontier.core import analyze
from efficient_frontier.data import BENCHMARK_SYMBOLS, demo_prices, download_benchmarks, download_prices, load_csv, original_tickers, parse_tickers, validate_benchmark_prices
from efficient_frontier.evidence import analyze_evidence
from efficient_frontier.presentation import csv_text, frontier_chart, holdout_chart, latest_profile_chart, weights_frame
from efficient_frontier.profiles import build_profiles
from efficient_frontier.risk import risk_contributions, risk_summary
from efficient_frontier.style import ACCENT, APP_CSS, GRID, PLOTLY_CONFIG, style_chart
from efficient_frontier.universe import download_universe_prices, fetch_nasdaq100


st.set_page_config(page_title="Portfolio Lab", page_icon=":material/donut_large:", layout="wide", initial_sidebar_state="collapsed")
st.html(f"<style>{APP_CSS}</style>")

MARKETS = ["Nasdaq-100", "My tickers", "Original 60 holdings", "Upload CSV", "Demo"]
MARKET_CAPTIONS = ["Today's 101 members. Prices from Yahoo Finance.", "Type Yahoo Finance symbols.",
                   "Your original fixed list. Marsh uses its current MRSH symbol, formerly MMC.",
                   "Your own adjusted prices. Stays offline.", "Six simulated assets. Works offline."]
FREQUENCIES = {"Daily market prices": 252, "Daily calendar prices": 365, "Weekly": 52, "Monthly": 12, "Custom": None}
METHODS = ["Expanding window", "Rolling window", "Fixed rebalance", "Buy and hold"]
TOPICS = ["Efficient frontiers", "Risk breakdown", "Data & coverage", "All backtests", "Methodology"]
DEFAULT_TEST = {"rebalance_every": 21, "rolling_window": 0, "cost_bps": 10.0}
DEFAULT_SETUP = {
    "market": "Nasdaq-100", "history": "5 years", "ticker_text": "AAPL, MSFT, AMZN, GOOGL, NVDA, META",
    "start": date(2020, 1, 1), "end": date.today(), "use_cap": False, "cap": 40.0, "risk_free": 2.0,
    "train_pct": 70, "shrink": 10, "frequency": "Daily market prices", "periods_per_year": 252.0,
    "price_currency": "USD", "compare_market": True, "download_benchmarks": False,
    "prices_file": None, "benchmark_file": None,
}
FILE_KEYS = ("prices_file", "benchmark_file")
DRAWER_KEYS = [key for key in DEFAULT_SETUP if key not in FILE_KEYS]
SETTING_KEYS = ("train_fraction", "risk_free_rate", "max_weight", "shrinkage", "periods_per_year")
CHART_TITLES = {"Growth": "Growth of 10,000", "Drawdown": "Falls from peak", "Relative to markets": "Versus the markets"}
COVERAGE_COLUMNS = {"industry": None, "symbol": "Symbol", "name": "Company", "status": "Status", "reason": "Reason",
                    "observations": st.column_config.NumberColumn("Price rows"), "first_date": "First date", "last_date": "Last date"}
DATE_COLUMN = st.column_config.DateColumn("Date", format="D MMM YYYY")
PREVIEW_HTML = ('<ol class="pl-preview"><li><b>What it holds.</b><span>The stocks and weights at each risk level.</span></li>'
                '<li><b>Would it have beaten the market?</b><span>A replay of the last part of the history against the '
                'S&amp;P 500 and the Nasdaq-100.</span></li><li><b>How sure can we be?</b><span>The statistics, in plain words. '
                'The detail is one click away.</span></li></ol>')
KEY_LEGEND = ('<span class="pl-keys">Keys: <kbd>L</kbd> <kbd>M</kbd> <kbd>H</kbd> risk · <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> '
              '<kbd>4</kbd> sections · <kbd>S</kbd> setup · <kbd>E</kbd> export</span>')
SCROLL_JS = """<script>/* __NONCE__ */
(() => {
  const main = document.querySelector('[data-testid="stMain"]');
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const go = () => {
    if ("__TARGET__" === "top") {
      if (document.activeElement?.matches?.(".pl-answer")) document.activeElement.blur();
      main?.scrollTo({top: 0, behavior: "instant"}); return;
    }
    const el = document.getElementById("__TARGET__");
    if (!el) return;
    el.scrollIntoView({behavior: reduce ? "instant" : "smooth", block: "start"});
    el.querySelector("h2")?.focus({preventScroll: true});
  };
  requestAnimationFrame(() => setTimeout(go, 60));
})();
</script>"""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, start, end):
    return download_prices(tickers, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_members():
    return fetch_nasdaq100()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe(start, end):
    progress_bar = st.progress(0, text="Read the current Nasdaq-100 list")
    return download_universe_prices(
        fetch_members(), start, end,
        progress=lambda done, total, message: progress_bar.progress(done / max(total, 1), text=message),
    )


@st.cache_data(ttl=3600, show_spinner=False,
               hash_funcs={pd.DatetimeIndex: lambda dates: (tuple(dates.astype(str)), dates.name)})
def fetch_benchmarks(dates):
    return download_benchmarks(dates)


@st.cache_data(show_spinner=False, max_entries=2)
def calculate(prices, settings):
    return analyze(prices, **settings)


@st.cache_data(show_spinner=False, max_entries=2)
def calculate_latest(prices, settings):
    return build_profiles(prices, **{key: value for key, value in settings.items() if key != "train_fraction"})


@st.cache_data(show_spinner=False, max_entries=2)
def compare_backtests(prices, settings, options):
    return run_backtests(prices, **settings, **options)


@st.cache_data(show_spinner=False, max_entries=2)
def calculate_market(benchmark_prices, prices, result, latest, settings, study=None):
    benchmarks = compare_benchmarks(benchmark_prices, prices, result, study, latest,
                                    risk_free_rate=settings["risk_free_rate"])
    try:
        evidence = analyze_evidence(
            study.equity if study is not None else result.equity,
            benchmarks.backtest_equity if study is not None else benchmarks.holdout_equity,
            risk_free_rate=settings["risk_free_rate"], periods_per_year=settings["periods_per_year"],
            delayed_entry=study is not None,
        )
    except (ValueError, RuntimeError) as exc:
        return benchmarks, None, f"Historical uncertainty estimates are unavailable: {exc}"
    return benchmarks, evidence, ""


# ----------------------------------------------------------------------------- state and callbacks

def init_state():
    defaults = {"setup": dict(DEFAULT_SETUP), "setup_open": False, "view": "Portfolio", "risk_profile": "Medium",
                "comparison_method": "Expanding window", "comparison_chart": "Growth", "research_topic": "Efficient frontiers"}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_state(name, value):
    st.session_state[name] = value


def jump(view, anchor):
    st.session_state.view = view
    st.session_state.scroll_to = anchor
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1


def start_build():
    st.session_state.pending_build = True
    st.session_state.build_error = None
    st.session_state.scroll_to = "top"


def try_demo():
    st.session_state.setup = {**st.session_state.setup, "market": "Demo"}
    start_build()


def cancel_work():
    st.session_state.pop("pending_build", None)
    st.session_state.pop("pending_study", None)


def open_setup():
    """Open the drawer. Its widgets take their defaults from the applied setup, so an interrupted run cannot blank them."""
    state = st.session_state
    if state.get("pending_build") or state.get("pending_study") or state.get("setup_open"):
        return
    state.draft_clear = set()
    state.setup_open = True
    state.setup_opened_at = time.monotonic()


def close_setup():
    st.session_state.setup_open = False


def dismiss_setup():
    """Esc or a click outside the drawer. The second click of a double click on Change lands outside the new drawer,
    so the drawer ignores a dismiss in its first 0.5 s (the Windows double-click time). The next run draws it again."""
    if time.monotonic() - st.session_state.get("setup_opened_at", 0.0) >= 0.5:
        close_setup()


def remove_file(name):
    st.session_state.draft_clear = st.session_state.get("draft_clear", set()) | {name}


def submit_test_settings():
    # A new key mounts a closed popover. The browser cannot reopen it with its stale open state.
    st.session_state.test_nonce = st.session_state.get("test_nonce", 0) + 1
    st.session_state.pending_study = True


def read_draft(uploads):
    """A new setup from the drawer widgets. Files are (name, bytes): a new upload, else the stored file unless removed."""
    state = st.session_state
    draft = {key: state.get(key, state.setup[key]) for key in DRAWER_KEYS}
    for name in FILE_KEYS:
        upload = uploads.get(name)
        draft[name] = ((upload.name, upload.getvalue()) if upload is not None
                       else None if name in state.get("draft_clear", set()) else state.setup[name])
    return draft


def resolve(setup):
    """The source, dates, settings and input choices that a setup requests."""
    market = setup["market"]
    source = "Demo · synthetic" if market == "Demo" else "Upload CSV" if market == "Upload CSV" else "Yahoo Finance"
    csv = source == "Upload CSV"
    start, end = date(2020, 1, 1), date.today()
    if source == "Yahoo Finance":
        if setup["history"] == "Custom dates":
            start, end = setup["start"], setup["end"]
        else:
            start = (pd.Timestamp(end) - pd.DateOffset(years=5 if setup["history"] == "5 years" else 10)).date()
    periods_per_year = 252.0
    if csv:
        periods_per_year = FREQUENCIES[setup["frequency"]]
        if periods_per_year is None:
            periods_per_year = setup["periods_per_year"]
    settings = dict(train_fraction=setup["train_pct"] / 100, risk_free_rate=setup["risk_free"] / 100,
                    max_weight=setup["cap"] / 100 if setup["use_cap"] else 1.0, shrinkage=setup["shrink"] / 100,
                    periods_per_year=periods_per_year)
    ticker_text = setup["ticker_text"] if market == "My tickers" else ""
    files = {name: setup[name] if csv else None for name in FILE_KEYS}
    download_for_csv = bool(setup["download_benchmarks"]) if csv else False
    input_choices = {
        "tickers": ticker_text.strip(), "download_benchmarks": download_for_csv,
        **{name: (upload[0], sha256(upload[1]).hexdigest()) if upload is not None else None
           for name, upload in (("prices", files["prices_file"]), ("benchmarks", files["benchmark_file"]))},
    }
    return {"market": market, "source": source, "start": start, "end": end, "settings": settings,
            "price_currency": setup["price_currency"], "compare_market": setup["compare_market"],
            "download_for_csv": download_for_csv, "ticker_text": ticker_text, **files, "input_choices": input_choices}


def differs(plan):
    """True when a resolved setup would not reproduce the applied result."""
    metadata = st.session_state.result[2]
    return (metadata["universe"] != plan["market"] or st.session_state.get("applied_inputs") != plan["input_choices"]
            or any(metadata[key] != value for key, value in plan["settings"].items())
            or metadata["price_currency"] != plan["price_currency"] or metadata["compare_market"] != plan["compare_market"]
            or plan["source"] == "Yahoo Finance" and (metadata.get("requested_start") != plan["start"].isoformat()
                                                      or metadata.get("requested_end_exclusive") != plan["end"].isoformat()))


def apply_notes(metadata, notes):
    """Set each note with text and remove each note set to None."""
    for key, value in notes.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value


def make_report_callable():
    """A zero-argument export over the objects on screen now. It reads no session state when it runs."""
    result, prices, metadata = st.session_state.result
    objects = (result, prices, metadata, st.session_state.get("backtests"), st.session_state.get("latest_profiles"),
               st.session_state.get("benchmarks"), st.session_state.get("evidence"))
    profile, method = st.session_state.risk_profile, st.session_state.comparison_method
    return lambda: presentation.report_zip(*objects, selected_profile=profile, selected_method=method)


def settings_frame(metadata):
    """Study settings with plain labels. The metadata does not change."""
    return pd.DataFrame([{"Setting": story.setting_label(key), "Value": story.setting_text(key, value)}
                         for key, value in metadata.items() if key != "universe_coverage"])


def day(value):
    stamp = pd.Timestamp(value)
    return f"{stamp.day} {stamp:%b %Y}"


# ----------------------------------------------------------------------------- slow work

def run_study(prices, result, latest, metadata, settings, options, benchmark_prices):
    """Run the market test and its market step. Writes nothing. Returns (study, benchmarks, evidence, notes, error)."""
    try:
        profile_backtests = min(len(result.train_returns), options["rolling_window"] or len(result.train_returns)) >= 6
        study = compare_backtests(prices, settings, dict(**{**options, "rolling_window": options["rolling_window"] or None},
                                                         include_profiles=profile_backtests))
        if not profile_backtests:
            study.warnings.append("Risk-profile tests need six returns in each fit window. The original targets remain available.")
    except (ValueError, RuntimeError, OSError) as exc:
        return None, None, None, {}, str(exc)
    benchmarks = evidence = None
    notes = {"evidence_note": None}
    if benchmark_prices is not None:
        try:
            benchmarks, evidence, note = calculate_market(benchmark_prices, prices, result, latest, settings, study)
            notes["benchmarks_note"] = None
            if note:
                notes["evidence_note"] = note
        except (ValueError, RuntimeError, OSError) as exc:
            notes["benchmarks_note"] = f"Market comparison unavailable: {exc}"
    return study, benchmarks, evidence, notes, None


def error_kind(exc):
    """Name the failure for the error state. A retry helps only "yahoo" (network) and "other"."""
    message = str(exc)
    if isinstance(exc, OSError) or message.startswith("Yahoo price download failed"):
        return "yahoo"
    if message.startswith("Yahoo returned no prices"):
        return "no_prices"
    if "position cap is infeasible" in message:
        return "infeasible"
    if message == "Choose an adjusted-price CSV first.":
        return "no_file"
    return "other"


def build(plan, mark, progress_slot):
    """Fetch prices, fit the portfolios and run the default market test. Writes the session state once, at the end."""
    market, source, settings = plan["market"], plan["source"], plan["settings"]
    start, end = plan["start"], plan["end"]
    loaded = None
    try:
        metadata = {"source": source, "universe": market, "price_currency": plan["price_currency"],
                    "compare_market": plan["compare_market"], **settings}
        metadata["currency_assumption"] = "The user declares a common price currency. No verification or currency conversion occurs."
        universe_benchmarks = None
        if source == "Yahoo Finance":
            metadata.update(requested_start=start.isoformat(), requested_end_exclusive=end.isoformat())
            if market == "Nasdaq-100":
                mark(2)
                with progress_slot:
                    loaded = fetch_universe(start.isoformat(), end.isoformat())
                prices, universe_benchmarks = loaded.prices, loaded.benchmarks
                metadata.update(loaded.metadata)
                metadata["universe"] = market
                if len(prices.columns) < 2:
                    raise ValueError("Fewer than two Nasdaq-100 securities have complete prices. Review coverage below or choose a shorter history.")
            else:
                tickers = original_tickers(current_symbols=True) if market == "Original 60 holdings" else parse_tickers(plan["ticker_text"])
                prices = fetch_prices(tickers, start.isoformat(), end.isoformat())
        elif market == "Demo":
            prices = demo_prices()
        else:
            if plan["prices_file"] is None:
                raise ValueError("Choose an adjusted-price CSV first.")
            prices = load_csv(BytesIO(plan["prices_file"][1]))
            metadata["input_file"] = plan["prices_file"][0]
        mark(3)
        result = calculate(prices, settings)
        latest = calculate_latest(prices, settings) if len(prices) >= 7 else None
        mark(4)
        benchmark_prices = benchmarks = evidence = None
        if market == "Demo":
            metadata["benchmarks_note"] = "Synthetic prices have no real-market benchmark comparison."
        elif not plan["compare_market"]:
            metadata["benchmarks_note"] = "Market comparisons are off. To turn them on, select Change, then Model assumptions."
        elif plan["price_currency"] != "USD":
            metadata["benchmarks_note"] = "Market comparisons require USD prices. No currency conversion occurs."
        else:
            try:
                if plan["benchmark_file"] is not None:
                    benchmark_prices = validate_benchmark_prices(load_csv(BytesIO(plan["benchmark_file"][1])), prices.index)
                    metadata["benchmark_source"] = "CSV: " + plan["benchmark_file"][0]
                elif plan["download_for_csv"]:
                    benchmark_prices = fetch_benchmarks(prices.index)
                    metadata["benchmark_source"] = "Yahoo Finance"
                elif universe_benchmarks is not None:
                    benchmark_prices = universe_benchmarks
                    metadata["benchmark_source"] = "Yahoo Finance"
                elif set(BENCHMARK_SYMBOLS).issubset(prices.columns):
                    benchmark_prices = prices.loc[:, list(BENCHMARK_SYMBOLS)]
                    metadata["benchmark_source"] = "SPY and QQQ input columns"
                elif source == "Yahoo Finance":
                    benchmark_prices = fetch_benchmarks(prices.index)
                    metadata["benchmark_source"] = "Yahoo Finance"
                else:
                    metadata["benchmarks_note"] = ("Add a market prices file, include SPY and QQQ columns, or turn on the Yahoo download. "
                                                   "Select Change, then Model assumptions.")
            except (ValueError, RuntimeError, OSError) as exc:
                metadata["benchmarks_note"] = f"Market comparison unavailable: {exc}"
                benchmark_prices = None
        options = st.session_state.get("backtest_options", DEFAULT_TEST)
        study, study_benchmarks, study_evidence, notes, study_error = run_study(
            prices, result, latest, metadata, settings, options, benchmark_prices)
    except (ValueError, RuntimeError, OSError) as exc:
        for key in ("result", "latest_profiles", "backtests", "benchmarks", "evidence", "benchmark_prices", "coverage"):
            st.session_state.pop(key, None)
        if loaded is not None:
            st.session_state.coverage = loaded.coverage
        st.session_state.build_error = str(exc)
        st.session_state.build_error_kind = error_kind(exc)
        st.session_state.pop("pending_build", None)
        st.rerun()
    if study is not None:
        apply_notes(metadata, notes)
        benchmarks, evidence = study_benchmarks, study_evidence
    elif benchmark_prices is not None:
        try:
            benchmarks, evidence, note = calculate_market(benchmark_prices, prices, result, latest, settings)
            if note:
                metadata["evidence_note"] = note
        except (ValueError, RuntimeError, OSError) as exc:
            metadata["benchmarks_note"] = f"Market comparison unavailable: {exc}"
            benchmark_prices = None
    st.session_state.update(
        result=(result, prices, metadata), applied_inputs=plan["input_choices"], latest_profiles=latest,
        benchmark_prices=benchmark_prices, coverage=loaded.coverage if loaded is not None else None,
        benchmarks=benchmarks, evidence=evidence, backtest_options=options, study_error=study_error, build_error=None,
        scroll_to="top")
    if study is None:
        st.session_state.pop("backtests", None)
    else:
        st.session_state.backtests = study
    for key in ("pending_build", "holding_strategy", "chart_strategies", "risk_portfolio", "correlation_assets"):
        st.session_state.pop(key, None)
    st.rerun()


# ----------------------------------------------------------------------------- shared pieces

def scroll(target):
    st.html(SCROLL_JS.replace("__TARGET__", target).replace("__NONCE__", str(st.session_state.get("scroll_nonce", 0))),
            unsafe_allow_javascript=True)


def context_line(text, align="center"):
    with st.container(horizontal=True, horizontal_alignment=align, vertical_alignment="center", gap="small"):
        st.html(text, width="content")
        st.button("Change", key="edit_setup", type="tertiary", icon=":material/tune:", shortcut="S", on_click=open_setup,
                  disabled=busy)


def title_html(title, sub=""):
    return f'<h3 class="pl-title">{html.escape(title)}</h3>' + (f'<p class="pl-sub">{html.escape(sub)}</p>' if sub else "")


def stat_strip(stats):
    return '<div class="pl-stats">' + "".join(
        f'<div class="pl-stat"><span class="pl-num">{html.escape(str(number))}</span><span class="pl-lbl">{label}</span>'
        f'<small>{html.escape(sub)}</small></div>' for number, label, sub in stats) + "</div>"


def ratio(value):
    """Two decimals for a ratio, without a "-0.00"."""
    return f"{0.0 if abs(value) < .005 else value:.2f}"


def research_chart(figure):
    """Draw a Research chart. The legend can grow on a phone, so it does not scroll or hide entries."""
    st.plotly_chart(figure.update_layout(legend_maxheight=0.5), width="stretch", theme=None, config=PLOTLY_CONFIG)


def try_buttons(fix=False):
    """The error actions. When a retry would fail the same way, the primary action opens the drawer instead."""
    with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="small"):
        if not st.session_state.setup_open:
            if fix:
                st.button("Change the study", key="fix_setup", type="primary", icon=":material/tune:", on_click=open_setup)
            else:
                st.button("Try again", key="build", type="primary", shortcut="Mod+Enter", on_click=start_build)
        if st.session_state.setup["market"] != "Demo":
            st.button("Try the offline demo", key="try_demo", type="tertiary", icon=":material/science:", on_click=try_demo)


# ----------------------------------------------------------------------------- drawer

def file_slot(label, key, name, clear_key, caption=None, help=None):
    upload = st.file_uploader(label, type=["csv"], key=key, help=help)
    if caption:
        st.caption(caption)
    stored = st.session_state.setup[name]
    if upload is None and stored is not None and name not in st.session_state.get("draft_clear", set()):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.caption(f"Using {stored[0]}. Upload a file to replace it.")
            st.button("Remove", key=clear_key, type="tertiary", on_click=remove_file, args=(name,))
    return upload


@st.dialog("Change the study", width="small", position="right", on_dismiss=dismiss_setup)
def setup_sheet():
    state = st.session_state
    setup = state.setup
    st.html('<p class="pl-sub">Choose the stocks and the history. Nothing changes until you select Show portfolios.</p>')
    uploads = {}
    market = st.radio("Stocks", MARKETS, index=MARKETS.index(setup["market"]), key="market", captions=MARKET_CAPTIONS)
    if market == "My tickers":
        st.text_area("Stock symbols", setup["ticker_text"], key="ticker_text",
                     help="Separate Yahoo Finance symbols with commas. Exchange suffixes such as .L work.")
    elif market == "Upload CSV":
        uploads["prices_file"] = file_slot(
            "Adjusted prices (.csv)", "prices_upload", "prices_file", "clear_prices_file",
            caption="Use Date as the first column, then one asset per column. Prices must be complete and use a common currency.")
    if market in ("Nasdaq-100", "My tickers", "Original 60 holdings"):
        history = st.segmented_control("History", ["5 years", "10 years", "Custom dates"], default=setup["history"], key="history",
                                       required=True, format_func=lambda value: "Custom" if value == "Custom dates" else value)
        if history == "Custom dates":
            left, right = st.columns(2)
            left.date_input("From", setup["start"], key="start")
            right.date_input("Until", setup["end"], key="end", help="The last day is not included.")
        else:
            end = date.today()
            start = (pd.Timestamp(end) - pd.DateOffset(years=5 if history == "5 years" else 10)).date()
            st.caption(f"{day(start)} to {day(end - pd.Timedelta(days=1))}")
    with st.expander("Model assumptions  :gray[holding limit, risk-free rate, test split, shrinkage, currency]"):
        st.caption("The defaults work without changes. These settings change the assumptions, not your risk choice.")
        use_cap = st.checkbox("Limit each holding", setup["use_cap"], key="use_cap")
        st.number_input("Largest holding allowed (%)", 0.0, 100.0, setup["cap"], step=1.0, key="cap", disabled=not use_cap,
                        help="A 25% limit needs at least four assets. The limit applies when weights are set. Prices can move weights later.")
        st.number_input("Risk-free rate (%)", -10.0, 100.0, setup["risk_free"], step=.25, key="risk_free",
                        help="A yearly rate for Sharpe and Sortino. It affects the classic maximum-Sharpe mix. It does not add cash.")
        st.slider("Prices for the first fit (%)", 50, 90, setup["train_pct"], step=5, key="train_pct",
                  help="The earliest prices fit the starting model. The rest test it. This does not change today's holdings.")
        st.slider("Covariance shrinkage (%)", 0, 100, setup["shrink"], step=5, key="shrink",
                  help="Pulls stock-to-stock relationships toward zero to reduce noise. 0% keeps them as measured. 100% ignores them.")
        if market == "Upload CSV":
            frequency = st.selectbox("Price frequency", list(FREQUENCIES), index=list(FREQUENCIES).index(setup["frequency"]), key="frequency")
            if FREQUENCIES[frequency] is None:
                st.number_input("Observations per year", min_value=1.0, value=setup["periods_per_year"], key="periods_per_year")
            st.caption("Frequency scales yearly figures. It does not resample prices.")
        currencies = ["USD", "Other currency"]
        st.selectbox("Price currency", currencies, index=currencies.index(setup["price_currency"]), key="price_currency",
                     help="Declare the currency of every price. The app does not check or convert currencies.")
        st.caption("SPY and QQQ comparisons need USD prices. No currency conversion occurs.")
        st.checkbox("Compare with the S&P 500 and the Nasdaq-100", setup["compare_market"], key="compare_market", disabled=market == "Demo")
        if market == "Upload CSV":
            uploads["benchmark_file"] = file_slot("Market prices (optional)", "benchmark_upload", "benchmark_file",
                                                  "clear_benchmark_file", help="Use Date,SPY,QQQ on exactly the same dates as your asset file.")
            st.checkbox("Download SPY and QQQ from Yahoo Finance for this file", setup["download_benchmarks"], key="download_benchmarks")
            st.caption("CSV analysis stays offline unless you select this download. SPY and QQQ columns in your asset file also work.")
    draft = read_draft(uploads)
    if "result" in state and differs(resolve(draft)):
        st.info("Your setup differs from this result. Select Show portfolios to apply it.")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel", key="cancel_setup"):
            close_setup()
            st.rerun()
        if st.button("Show portfolios", key="build", type="primary", shortcut="Mod+Enter", on_click=set_state, args=("view", "Portfolio")):
            state.setup = draft
            state.setup_open = False
            start_build()
            st.rerun()


# ----------------------------------------------------------------------------- welcome, work and error

def hero_work():
    setup = st.session_state.setup
    plan = resolve(setup)
    market = plan["market"]
    compare = plan["compare_market"] and plan["price_currency"] == "USD"
    with st.container(key="hero", horizontal_alignment="center"):
        context_line(story.context_setup_html(setup, badge="First run · about 25 s" if market == "Nasdaq-100" and not ready else None))
        st.html(f'<h1 class="pl-display pl-working">{story.work_headline_html(market, rerun=ready and not differs(plan))}</h1>')
        steps = st.empty()
        steps.html(story.steps_html(market, 1 if market == "Nasdaq-100" else 2, compare=compare))
        progress_slot = st.container(key="work_progress")
        st.html('<p class="pl-note">Prices stay in memory for one hour. A second run with the same dates is almost instant.</p>')
        st.button("Cancel", key="cancel_build", type="tertiary", on_click=cancel_work)
    with st.container(border=True, key="tile_skeleton"):
        st.html(story.SKELETON_HTML)
    st.session_state.pop("scroll_to", None)
    scroll("top")
    build(plan, lambda number: steps.html(story.steps_html(market, number, compare=compare)), progress_slot)


def welcome_or_error():
    setup = st.session_state.setup
    market = setup["market"]
    error = st.session_state.get("build_error")
    with st.container(key="hero", horizontal_alignment="center"):
        context_line(story.context_setup_html(setup))
        if error is None:
            st.html(f'<h1 class="pl-display">{story.welcome_headline_html(market)}</h1>'
                    '<p class="pl-lead pl-center">Low, medium and highest relative risk. Built from past prices, '
                    "then tested on prices that the model did not see.</p>")
            if not st.session_state.setup_open:
                st.button("Show portfolios", key="build", type="primary", shortcut="Mod+Enter", on_click=start_build)
            note = ("The first Nasdaq-100 run downloads about 100 price series. It takes about 25 seconds. "
                    "Nothing downloads until you select Show portfolios." if market == "Nasdaq-100"
                    else "Prices download from Yahoo Finance when you select Show portfolios."
                    if market in ("My tickers", "Original 60 holdings") else "This runs offline.")
            st.html(f'<p class="pl-note">{note}</p>')
            if market != "Demo":
                st.button("Try the offline demo", key="try_demo", type="tertiary", icon=":material/science:", on_click=try_demo)
            st.html(PREVIEW_HTML + '<p class="pl-foot pl-center" style="margin-top:32px">Historical research, not a forecast or financial advice.</p>')
            return
        kind = st.session_state.get("build_error_kind", "other")
        headline = {"yahoo": "We could not reach Yahoo Finance.", "no_prices": "Yahoo Finance had no prices for these symbols.",
                    "infeasible": "These settings have no solution.", "no_file": "The study needs a price file."}.get(
            kind, "The study could not run.")
        st.html(f'<h1 class="pl-display">{html.escape(headline)}</h1>')
        st.error(error)
        if kind == "infeasible":
            cap = setup["cap"]
            need = (f"A {cap:g}% limit needs at least {math.ceil(100 / cap - 1e-9)} "
                    f"{'assets' if market in ('Demo', 'Upload CSV') else 'stocks'}. " if cap > 0 else "")
            st.html(f'<p class="pl-lead pl-center">{need}Raise or remove the holding limit in Model assumptions.</p>')
        try_buttons(fix=kind in ("no_prices", "infeasible", "no_file"))
    coverage = st.session_state.get("coverage")
    if coverage is not None:
        with st.container(border=True, key="tile_missing"):
            st.html(title_html("Which stocks were missing"))
            st.dataframe(coverage, hide_index=True, width="stretch", column_config=COVERAGE_COLUMNS)


# ----------------------------------------------------------------------------- overview

def overview():
    result, prices, metadata = st.session_state.result
    latest = st.session_state.get("latest_profiles")
    coverage = st.session_state.get("coverage")
    profile = st.session_state.risk_profile
    universe = metadata["universe"]
    noun = "asset" if universe in ("Demo", "Upload CSV") else "stock"
    names = story.company_names(coverage) if universe == "Nasdaq-100" else {}

    with st.container(key="hero", horizontal_alignment="center"):
        context_line(story.context_result_html(metadata, prices))
        if "synthetic" in metadata["source"]:
            st.info("Demo · synthetic prices. These are not market results.")
        if latest is None:
            st.html('<h1 class="pl-display">There is not enough history to set risk levels.</h1>')
            st.info("Latest risk profiles need at least seven complete price rows. Research still shows the classic model.")
        else:
            st.html(f'<h1 class="pl-display">{story.headline_html(latest.portfolios[profile].weights, profile, universe)}</h1>')
            st.segmented_control("Risk level", ["Low", "Medium", "Extreme"], key="risk_profile", required=True,
                                 format_func=story.LEVEL_LABELS.get, label_visibility="collapsed", width="stretch",
                                 persist_state="session", disabled=busy)
            st.html(f'<p class="pl-note">Risk is relative to {story.these_assets(len(prices.columns), universe)}. '
                    "Built from past prices. Not a forecast.</p>")

    if latest is not None:
        holdings_tiles(latest, prices, profile, universe, noun, names, coverage)
    market_section(result, prices, metadata, latest, profile)
    st.html(story.fine_print_html(metadata, prices))
    with st.container(horizontal_alignment="center"):
        st.button("Open Research", key="open_research", icon=":material/chevron_right:", icon_position="right",
                  on_click=jump, args=("Research", "top"), disabled=busy)
    source = {"Yahoo Finance": "Yahoo Finance", "Upload CSV": f"CSV file {metadata.get('input_file', '')}".strip()}.get(
        metadata["source"], "simulated")
    index_list = (f" · Index list: Nasdaq.com, {day(metadata['universe_source_date'])}"
                  if universe == "Nasdaq-100" and metadata.get("universe_source_date") else "")
    st.html(f'<div class="pl-foot"><span>Portfolio Lab · Prices: {html.escape(source)}{index_list}</span>{KEY_LEGEND}</div>')


def holdings_tiles(latest, prices, profile, universe, noun, names, coverage):
    weights = latest.portfolios[profile].weights
    allocations = weights.sort_values(ascending=False)
    with st.container(border=True, key="tile_holdings"):
        callout = story.callout_html(weights, universe, names)
        if callout:
            st.html(callout)
        st.html(story.stats_html(latest, profile, universe, names) + story.list_html(weights, names))
        count = len(allocations)
        with st.expander(f"Every {noun} and its exact weight  :gray[{count} row{'' if count == 1 else 's'}]"):
            holdings = allocations.rename("Weight").rename_axis("Symbol").reset_index()
            if coverage is not None:
                holdings.insert(1, "Company", holdings["Symbol"].map(names))
            # Weight before Company, so the weight stays in view on a phone. The frame keeps its columns.
            order = ["Symbol", "Weight", "Company"] if coverage is not None else None
            st.dataframe(holdings, hide_index=True, width="stretch", column_order=order, column_config={
                "Weight": st.column_config.ProgressColumn("Weight", format="percent", min_value=0.0,
                                                          max_value=float(allocations.max()), color=ACCENT),
                "Company": st.column_config.TextColumn("Company")})
            small = int((allocations < .0005).sum())
            listed = f"All {count} {noun}s appear here, sorted by weight." if count > 1 else f"The one {noun} appears here."
            st.caption(f"{listed} {small} {'has' if small == 1 else 'have'} a weight under 0.05%. "
                       "Each portfolio adds up to 100%. Weights under 0.005% show as 0%.")
        with st.expander(f"Why these {noun}s?  :gray[how the model chose them]"):
            for paragraph in story.why_text(latest, profile, universe):
                st.write(paragraph)
            percent = {name: st.column_config.NumberColumn(story.LEVEL_LABELS.get(name, name), format="percent")
                       for name in latest.window_returns.columns}
            st.dataframe(latest.window_returns, width="stretch", column_config={"_index": "Stretch", **percent})
            st.dataframe(latest.windows, hide_index=True, width="stretch", column_config={
                "start": "From", "end": "To", "observations": "Price rows",
                "years": st.column_config.NumberColumn("Years", format="%.1f")})
    st.caption(f"Weights fitted on all prices from {day(prices.index[0])} to {day(prices.index[-1])}. "
               "Very small weights can show as 0.0%.")
    for note in latest.warnings:
        st.caption(note)
    with st.container(border=True, key="tile_levels"):
        st.html('<div class="pl-head"><h3 class="pl-title">How the three levels compare</h3>'
                '<span class="pl-keys"><kbd>L</kbd> <kbd>M</kbd> <kbd>H</kbd></span></div>'
                + story.levels_ledger_html(latest, profile, names, universe))


def test_controls():
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.selectbox("Rule", METHODS, key="comparison_method", format_func=story.RULE_LABELS.get, label_visibility="collapsed",
                     width=320, persist_state="session")
        with st.popover("Test settings", key=f"test_settings_{st.session_state.get('test_nonce', 0)}", on_change="rerun",
                        type="tertiary", icon=":material/tune:"):
            with st.form("comparison_settings", border=False):
                options = st.session_state.get("backtest_options", DEFAULT_TEST)
                st.number_input("Trade every (price rows)", 1, value=options["rebalance_every"], key="rebalance_every",
                                help="21 trading days is about one month. For a CSV file, this counts price rows.")
                st.number_input("Trading cost (basis points)", 0.0, 9999.0, options["cost_bps"], key="cost_bps",
                                help="10 basis points = 0.10% of each amount bought or sold. Entry counts. There is no final sale.")
                st.number_input("Recent history for “Refit on recent prices” (return observations)", 0, value=options["rolling_window"],
                                key="rolling_window", help="0 uses the same length as the first fit.")
                st.caption("All four rules use the same dates and fees. Each trade uses information to the previous close "
                           "and trades at the next close.")
                st.form_submit_button("Run the test again", key="run_comparison", type="primary", on_click=submit_test_settings)


def rerun_study(result, prices, metadata, latest):
    """Run the market test again with the Test settings. Writes the session state once, at the end."""
    options = {key: st.session_state.get(key, DEFAULT_TEST[key]) for key in DEFAULT_TEST}
    settings = {key: metadata[key] for key in SETTING_KEYS}
    study, benchmarks, evidence, notes, error = run_study(prices, result, latest, metadata, settings, options,
                                                          st.session_state.get("benchmark_prices"))
    if error is None:
        st.session_state.update(backtests=study, benchmarks=benchmarks, evidence=evidence, backtest_options=options, study_error=None)
        apply_notes(metadata, notes)
    else:
        for key in ("backtests", "benchmarks", "evidence"):
            st.session_state.pop(key, None)
        metadata.pop("evidence_note", None)
        st.session_state.study_error = error
    for key in ("pending_study", "holding_strategy", "chart_strategies"):
        st.session_state.pop(key, None)
    st.rerun()


def market_section(result, prices, metadata, latest, profile):
    study = st.session_state.get("backtests")
    benchmarks = st.session_state.get("benchmarks")
    evidence = st.session_state.get("evidence")
    matched = benchmarks if benchmarks is not None and benchmarks.backtest_equity is not None else None
    eyebrow = "Would this rule have beaten the market?" if matched is not None or study is None else "How did this rule do in the test period?"
    method = st.session_state.comparison_method
    selected = f"{method} · {profile}"
    fallback = study is not None and selected not in study.equity
    if fallback:
        selected = f"{method} · Minimum volatility"

    if st.session_state.get("pending_study"):
        answer = story.market_answer(study, selected, benchmarks) if study is not None else "The market test did not run."
        st.html(story.section_header_html("pl-market", eyebrow, answer))
        options = {key: st.session_state.get(key, DEFAULT_TEST[key]) for key in DEFAULT_TEST}
        unit = "trading days" if metadata["periods_per_year"] == 252 else "price rows"
        with st.container(key="study_work"):
            st.html('<p class="pl-lead">Running the market test again.</p>'
                    f'<p class="pl-sub">{html.escape(story.RULE_LABELS[method])} · every {options["rebalance_every"]} {unit} · '
                    f'{options["cost_bps"] / 100:.2f}% per trade</p>')
            st.button("Cancel", key="cancel_study", type="tertiary", on_click=cancel_work)
        rerun_study(result, prices, metadata, latest)

    if st.session_state.get("study_error"):
        st.html(story.section_header_html("pl-market", eyebrow, "The market test did not run.",
                                          "The settings below did not work for this data. Change them and run the test again."))
        test_controls()
        st.error(st.session_state.study_error)
        return
    if study is None:
        st.html(story.section_header_html("pl-market", eyebrow, "The market test did not run."))
        test_controls()
        st.info("Select Test settings, then Run the test again.")
        return

    st.html(story.section_header_html("pl-market", eyebrow, story.market_answer(study, selected, benchmarks),
                                      story.market_lead(study, selected)))
    test_controls()
    if fallback:
        st.info("This sample is too short for the risk profiles. The comparison shows minimum volatility.")
    with st.container(border=True, key="tile_result"):
        st.html(story.result_tile_html(study, selected, benchmarks, evidence, metadata))
    if matched is None:
        st.info(metadata.get("benchmarks_note", "Market benchmarks are unavailable."))

    mode = st.session_state.comparison_chart
    relative = matched is not None and evidence is not None
    windows = evidence is not None and selected in set(evidence.windows["strategy"])
    with st.container(border=True, key="tile_chart"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.html(f'<h3 class="pl-title">{CHART_TITLES[mode if relative or mode != "Relative to markets" else "Growth"]}</h3>',
                    width="stretch")
            st.segmented_control("Chart", ["Growth", "Drawdown", "Relative to markets"], key="comparison_chart", required=True,
                                 format_func=story.CHART_LABELS.get, label_visibility="collapsed", persist_state="session")
        if mode == "Relative to markets" and not relative:
            st.caption("Versus markets needs market prices. The chart shows growth.")
        st.plotly_chart(story.story_chart(study, selected, mode, benchmarks, evidence), width="stretch", theme=None, config=PLOTLY_CONFIG)
        st.caption(story.chart_footnote(study, benchmarks))
        if windows:
            st.caption("Shaded bands are the three test stretches in How sure can we be.")
        for note in study.warnings:
            st.caption(note)
        with st.expander("Returns, falls and trading costs  :gray[full table]"):
            comparison = study.metrics.loc[[selected]].rename(index={selected: "Your rule"})
            if matched is not None:
                comparison = pd.concat([comparison, matched.backtest_metrics])
            formats = {name: "{:.0f}" if name.endswith("count") else ratio if name in ("sharpe", "sortino", "calmar")
                       else "{:.2f}" if name == "total_turnover" else "{:.2%}" for name in comparison}
            st.dataframe(comparison.style.format(formats, na_rep="—"), column_config={name: label for name, (label, _) in METRICS.items()}, width="stretch")
    evidence_section(study, matched, evidence, selected, metadata)


def evidence_section(study, matched, evidence, selected, metadata):
    eyebrow = "How sure can we be?"
    if matched is None:
        st.html(story.section_header_html("pl-evidence", eyebrow, "There is no market to compare with.",
                                          "The statistics compare the rule with the S&P 500 and the Nasdaq-100. This study has no market prices."))
        return
    if evidence is None:
        st.html(story.section_header_html("pl-evidence", eyebrow, "The uncertainty could not be estimated."))
        if metadata.get("evidence_note"):
            st.info(metadata["evidence_note"])
        return
    months = story.months_tested(study)
    st.html(story.section_header_html("pl-evidence", eyebrow, story.evidence_answer(evidence, selected),
                                      story.evidence_lead(evidence, selected, months, metadata)))
    left, right = st.columns([7, 5], gap="small")
    with left.container(border=True, key="tile_gap"):
        st.html(title_html("Average yearly gap, with 95% range", "Your rule minus each market, in percentage points. "
                           "This is an arithmetic average, so it differs from the growth gap above.")
                + story.interval_html(evidence, selected))
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key="words"):
            st.html('<span class="pl-words">Words used here</span>', width="content")
            for label, key in (("95% range", "word_range"), ("Average gap", "word_gap"), ("Points", "word_points")):
                with st.popover(label, key=key, type="tertiary", icon=":material/help:"):
                    st.markdown(story.WORDS[label])
    with right.container(border=True, key="tile_stretch"):
        st.html(title_html("Stretch by stretch") + story.stretch_summary_html(evidence, selected)
                + '<p class="pl-sub">The test period, split into three equal stretches. These differ from the three stretches '
                "that fitted today's holdings.</p>" + story.stretch_table_html(evidence, selected))
    with st.container(border=True, key="tile_stats"):
        with st.expander("Show the statistics  :gray[alpha, beta, p-values, method notes]"):
            st.dataframe(evidence_summary_frame(evidence).xs(selected, level="strategy"), column_config={"_index": "Market", **story.STAT_LABELS},
                         width="stretch")
            windows = evidence.windows.loc[evidence.windows.strategy == selected].drop(columns="strategy")
            st.dataframe(windows.style.format({"relative_return": "{:+.2%}", "annual_advantage": lambda value: f"{value * 100:+.2f} pp"}),
                         hide_index=True, width="stretch", column_config={
                             "benchmark": "Market", "window": "Stretch", "start": "From", "end": "To", "observations": "Observations",
                             "relative_return": "Change vs market", "annual_advantage": "Average gap"})
            settings = evidence.settings
            strategies = evidence.summary.index.get_level_values("strategy").nunique()
            markets = evidence.summary.index.get_level_values(1).nunique()
            st.markdown(f"Ranges use Newey–West standard errors ({settings.get('hac_lags', 0)} lags) on "
                        f"{settings.get('inference_observations', 0)} return pairs. The Holm adjustment covers all "
                        f"{settings.get('multiple_testing_tests', 0)} tests: {strategies} rules and levels, {markets} markets and 2 measures. "
                        "A p-value is not the chance that the rule will win. Alpha adjusts for exposure to one market. "
                        "It does not prove skill. The average gap is an arithmetic difference. Growth compounds the whole path.")
            if evidence.warnings:
                st.markdown("\n".join(f"- {note}" for note in evidence.warnings))
    st.html('<div class="pl-settle"><h3 class="pl-title">What would settle it?</h3><p>More time, or prices that nobody has '
            "looked at yet. Export the report now. Then test the same rule on the next year of prices.</p></div>")
    st.caption(f"These tests describe the last {months} months. They cannot tell you how the rule will do next.")


# ----------------------------------------------------------------------------- research

def research():
    result, prices, metadata = st.session_state.result
    latest = st.session_state.get("latest_profiles")
    study = st.session_state.get("backtests")
    benchmarks = st.session_state.get("benchmarks")
    evidence = st.session_state.get("evidence")
    coverage = st.session_state.get("coverage")
    profile = st.session_state.risk_profile
    method = st.session_state.comparison_method
    context_line(story.context_result_html(metadata, prices), align="left")
    st.html('<h1 class="pl-page">Research</h1><p class="pl-lead">Everything behind the overview, for checks and for reuse.</p>')
    topic = st.segmented_control("Topic", TOPICS, key="research_topic", required=True, format_func=story.TOPIC_LABELS.get,
                                 label_visibility="collapsed", persist_state="session")
    if topic == "Efficient frontiers":
        if latest is not None:
            with st.container(border=True, key="tile_r_frontier"):
                estimates = benchmarks.latest_estimates if benchmarks is not None else None
                research_chart(latest_profile_chart(latest, estimates, profile))
                st.caption("Full-history fit. The vertical axis is the return in the weakest of three past stretches. "
                           "It is in-sample, not a forecast." + (" Diamonds are SPY and QQQ on the same dates." if estimates is not None else ""))
                st.caption(f"Highlighted: {story.LEVEL_LABELS[profile]}. Change the level on the overview, or press L, M or H.")
                with st.expander("All three portfolios' weights"):
                    st.dataframe(weights_frame(latest).style.format("{:.2%}"), column_config={"_index": "Symbol", "Extreme": "Highest"},
                                 width="stretch")
        with st.container(border=True, key="tile_r_classic"):
            st.html(title_html("The classic model, for reference", "The app's first model. It fits average returns on the first "
                               f"{metadata['train_fraction'] * 100:g}% of prices, to {day(result.train_returns.index[-1])}. "
                               "Then it holds each mix unchanged on the rest. It stays here for comparison."))
            with st.expander("Classic frontier"):
                estimates = benchmarks.training_estimates if benchmarks is not None else None
                research_chart(frontier_chart(result, estimates))
                st.dataframe(weights_frame(result).style.format("{:.2%}"), column_config={"_index": "Symbol"}, width="stretch")
                st.caption("This model uses only the first part of the history." + (
                    " The SPY and QQQ markers use the same dates. They do not follow your holding limit." if estimates is not None else ""))
            with st.expander("Classic holdout, no trading costs"):
                research_chart(holdout_chart(result, benchmarks.holdout_equity if benchmarks is not None else None))
                metrics = result.holdout_metrics if benchmarks is None else pd.concat([result.holdout_metrics, benchmarks.holdout_metrics])
                st.dataframe(metrics.style.format({name: ratio if name in ("sharpe", "sortino", "calmar") else "{:.2%}" for name in metrics}, na_rep="—"),
                             column_config={"_index": "Portfolio", **{name: label for name, (label, _) in METRICS.items()}}, width="stretch")
                st.caption("Each mix is bought once at the split close and then left to drift. This check has no fees. "
                           "Its entry date differs from the market test.")
    elif topic == "Risk breakdown":
        with st.container(border=True, key="tile_r_risk"):
            with st.container(key="r_lead"):
                st.caption("Where the risk comes from. These checks use the classic model's mixes and its training-period covariance. "
                           "The risk level does not apply here.")
            st.dataframe(risk_summary(result).style.format({"max_weight": "{:.2%}", "effective_holdings": "{:.2f}", "diversification_ratio": "{:.2f}"}, na_rep="—"),
                         column_config={"_index": "Portfolio", "max_weight": "Largest weight", "effective_holdings": "Effective holdings",
                                        "diversification_ratio": "Diversification ratio"}, width="stretch")
            risk_portfolio = st.selectbox("Portfolio", list(result.portfolios), key="risk_portfolio")
            frame = pd.DataFrame({"Allocation weight": result.portfolios[risk_portfolio].weights,
                                  "Share of portfolio variance": risk_contributions(result)[risk_portfolio]}).sort_values("Allocation weight", ascending=False)
            figure = px.bar(frame.head(20).rename_axis("Asset").reset_index(), y="Asset", x=list(frame.columns), orientation="h", barmode="group",
                            color_discrete_sequence=[ACCENT, "#8e8e93"])
            style_chart(figure, "Allocation and risk", height=max(380, 110 + 30 * min(20, len(frame))))
            figure.update_traces(marker_line_width=0)
            figure.update_xaxes(title=None, tickformat=".0%", showgrid=True, gridcolor=GRID)
            figure.update_yaxes(title=None, autorange="reversed", dtick=1, showgrid=False)
            figure.update_layout(legend_title_text="")
            st.plotly_chart(figure, width="stretch", theme=None, config=PLOTLY_CONFIG)
            st.dataframe(frame.style.format("{:.2%}", na_rep="—"), column_config={"_index": "Symbol"}, width="stretch")
            st.caption("The chart shows up to 20 assets. The table has every asset. Variance shares can be negative. "
                       "When portfolio variance is zero, the shares are undefined.")
    elif topic == "Data & coverage":
        with st.container(border=True, key="tile_r_data"):
            st.html(title_html("The data behind this result"))
            with st.container(key="mustread"):
                if coverage is not None:
                    st.caption(metadata.get("universe_limitation", "Current Nasdaq-100 membership and complete-history requirements can favor surviving stocks. This is not historical index membership."))
                else:
                    st.caption("Fixed list. The study keeps the same assets for the whole period. That can favour survivors.")
            if coverage is not None:
                listed = f"List of {day(metadata['universe_source_date'])}" if metadata.get("universe_source_date") else "Today's list"
                st.html(stat_strip(((metadata.get("universe_requested", len(coverage)), "Checked", listed),
                                    (metadata.get("universe_included", len(prices.columns)), "Used", "Complete prices"),
                                    (metadata.get("universe_excluded", 0), "Left out", "Missing or late prices"))))
                st.dataframe(coverage, hide_index=True, width="stretch", column_config=COVERAGE_COLUMNS)
        with st.container(border=True, key="tile_r_settings"):
            st.html(title_html("Study settings"))
            st.dataframe(settings_frame(metadata), hide_index=True, width="stretch")
            with st.expander("Prices"):
                st.dataframe(prices, width="stretch", height=300, column_config={"_index": DATE_COLUMN})
                st.download_button("Download prices.csv", csv_text(prices, index_label="Date"), "prices.csv", "text/csv")
            with st.expander("Correlations"):
                chart_assets = st.multiselect("Assets in this chart", list(prices), default=list(prices)[:15], key="correlation_assets")
                if chart_assets:
                    figure = px.imshow(result.train_returns[chart_assets].corr(), zmin=-1, zmax=1, aspect="auto",
                                       color_continuous_scale=[[0, "#D0453E"], [.5, "#F2F2F4"], [1, "#0071E3"]])
                    style_chart(figure, "Correlations", height=480)
                    figure.update_xaxes(dtick=1)
                    figure.update_yaxes(dtick=1)
                    figure.update_coloraxes(colorbar={"thickness": 10, "len": .8, "tickfont": {"size": 11}})
                    st.plotly_chart(figure, width="stretch", theme=None, config=PLOTLY_CONFIG)
                st.caption("Training-period correlations. This selection does not change the portfolio universe.")
    elif topic == "All backtests":
        if study is None:
            st.info("The market test has not run. Select Test settings in the overview to run it.")
            st.button("Go to the market test", key="go_market", on_click=jump, args=("Portfolio", "pl-market"))
            return
        with st.container(key="r_lead"):
            st.caption(f"Every rule and level that the market test ran: {len(study.equity.columns)} strategies over the same "
                       f"{story.months_tested(study)} months. Use it to check that the overview is not a lucky pick.")
        if "holding_strategy" not in st.session_state:
            preferred = f"{method} · {profile}"
            st.session_state.holding_strategy = preferred if preferred in study.equity else study.equity.columns[0]
        with st.container(border=True, key="tile_r_tests"):
            strategies = st.multiselect("Strategies in this chart", list(study.equity), format_func=story.strategy_label, key="chart_strategies",
                                        default=[name for name in (f"{method} · {level}" for level in ("Low", "Medium", "Extreme")) if name in study.equity])
            holding = st.selectbox("Inspect one strategy", list(study.equity), key="holding_strategy", format_func=story.strategy_label)
            if strategies:
                st.plotly_chart(backtest_chart(study, strategies=strategies, benchmark_equity=benchmarks.backtest_equity if benchmarks is not None else None,
                                               focus=holding, label=story.strategy_label), width="stretch", theme=None, config=PLOTLY_CONFIG)
            st.caption("Blue: the strategy in Inspect one strategy. Grey: the other strategies in this chart. "
                       "Chart selections do not change any result or test.")
            formats = {name: "{:.0f}" if name.endswith("count") else ratio if name in ("sharpe", "sortino", "calmar")
                       else "{:.2f}×" if name == "total_turnover" else "{:.2%}" for name in study.metrics}
            st.dataframe(study.metrics.rename(index=story.strategy_label).style.format(formats, na_rep="—"),
                         column_config={"_index": "Strategy", **{name: label for name, (label, _) in METRICS.items()}}, width="stretch")
        with st.container(border=True, key="tile_r_holdings"):
            st.html(title_html(f"Holdings of {story.strategy_label(holding)}"))
            st.dataframe(study.holdings[holding].style.format("{:.2%}"),
                         column_config={"_index": "Symbol", **{name: label for name, (label, _) in HOLDINGS.items()}}, width="stretch")
            st.caption("Profit and loss contributions use initial capital. Their sum, less fees, equals the strategy's net total return.")
            with st.expander("Target weights and trades"):
                st.dataframe(study.allocations[holding].style.format("{:.2%}"), column_config={"_index": DATE_COLUMN}, width="stretch")
                st.dataframe(study.trades[holding].style.format({
                    "turnover": "{:.2%}", "cost": lambda value: f"{value * 10_000:,.2f}", "nav_before": lambda value: f"{value * 10_000:,.0f}",
                    "nav_after": lambda value: f"{value * 10_000:,.0f}", "train_start": day, "train_end": day}), width="stretch", column_config={
                    "_index": DATE_COLUMN, "turnover": "Turnover", "cost": "Fee", "nav_before": "Value before", "nav_after": "Value after",
                    "train_start": "Fit from", "train_end": "Fit to"})
                st.caption("Values and fees are per 10,000 at the start.")
            if evidence is not None:
                with st.expander("Every statistical comparison"):
                    comparisons = evidence_summary_frame(evidence).rename(index=story.strategy_label, level="strategy").reset_index()
                    st.dataframe(comparisons, hide_index=True, width="stretch",
                                 column_config={"strategy": "Strategy", "benchmark": "Market", **story.STAT_LABELS})
    else:
        st.markdown(story.METHOD_MD)
        for note in result.warnings:
            st.write(note)
        st.markdown(story.GLOSSARY_MD.format(k=evidence.settings["multiple_testing_tests"] if evidence is not None else "the"))


# ----------------------------------------------------------------------------- page

init_state()
ready = "result" in st.session_state
busy = bool(st.session_state.get("pending_build") or st.session_state.get("pending_study"))

with st.container(key="localnav", horizontal=True, vertical_alignment="center", gap="medium"):
    st.html('<span class="pl-wordmark">Portfolio Lab</span>', width="stretch")
    if ready:
        st.segmented_control("View", ["Portfolio", "Research"], key="view", required=True, format_func=story.VIEW_LABELS.get,
                             label_visibility="collapsed", disabled=busy)
        st.download_button("Export", data=make_report_callable(), file_name="portfolio-lab.zip", mime="application/zip",
                           key="export", type="primary", icon=":material/ios_share:", on_click="ignore", shortcut="E",
                           disabled=busy, help="One zip file. The report opens offline and prints to PDF.")

KEYS = (("key_holdings", "Holdings", "1", jump, ("Portfolio", "top")),
        ("key_market", "Market test", "2", jump, ("Portfolio", "pl-market")),
        ("key_evidence", "Evidence", "3", jump, ("Portfolio", "pl-evidence")),
        ("key_research", "Research", "4", jump, ("Research", "top")),
        ("key_low", "Low", "L", set_state, ("risk_profile", "Low")),
        ("key_medium", "Medium", "M", set_state, ("risk_profile", "Medium")),
        ("key_highest", "Highest", "H", set_state, ("risk_profile", "Extreme")))
if ready:
    with st.container(key="keys"):
        for key, label, shortcut, callback, args in KEYS:
            st.button(label, key=key, shortcut=shortcut, on_click=callback, args=args, disabled=busy)

if st.session_state.setup_open:
    setup_sheet()

if st.session_state.get("pending_build"):
    hero_work()
elif not ready:
    welcome_or_error()
elif st.session_state.view == "Portfolio":
    overview()
else:
    research()

target = st.session_state.pop("scroll_to", None)
if target is None and st.session_state.get("previous_view") != st.session_state.get("view"):
    target = "top"
if target:
    scroll(target)
st.session_state.previous_view = st.session_state.get("view")
