"""Run with ./run.sh from the repository root."""

from datetime import date
from hashlib import sha256
import html
from io import BytesIO
import math
import time

import pandas as pd
import streamlit as st

from efficient_frontier import story
from efficient_frontier.backtest_report import HOLDINGS
from efficient_frontier.data import demo_prices, download_benchmarks, download_prices, load_csv, original_tickers, parse_tickers, stale_price_notes
from efficient_frontier.presentation import csv_text
from efficient_frontier.risk_reference import load_qqq_volatility_reference
from efficient_frontier.style import APP_CSS, PLOTLY_CONFIG
from efficient_frontier.universe import download_universe_prices, fetch_nasdaq100, fetch_sp500_nasdaq100
from efficient_frontier.selection import MODEL_REVISION as SELECTION_MODEL_REVISION, run_selection, selection_fit_estimates, selection_frontier, split_stock_prices, validate_selection_benchmarks
from efficient_frontier.selection_report import selection_report_zip, selection_summary_html, selection_metrics_frame, selection_chart, SELECTION_METRIC_LABELS, selection_benchmark_note, FIT_METRIC_LABELS, selection_frontier_chart, selection_frontier_summary_html, selection_comparison_html, selection_replay_verdict_html, REPLAY_LABEL


st.set_page_config(page_title="Efficient Frontier Generator: Constrained Mean–Variance Optimization", page_icon=":material/donut_large:", layout="wide", initial_sidebar_state="collapsed")
st.html(f"<style>{APP_CSS}</style>")

MARKETS = ["S&P 500 + Nasdaq-100", "Nasdaq-100", "My tickers", "Original 60 holdings", "Upload CSV", "Demo"]
MARKET_CAPTIONS = ["Today’s combined members, without duplicate symbols. Prices from Yahoo Finance.", "Today's 101 members. Prices from Yahoo Finance.", "Type Yahoo Finance symbols.",
                   "Your original fixed list. Marsh uses its current MRSH symbol, formerly MMC.",
                   "Your own adjusted prices. Stays offline.", "Six simulated assets. Works offline."]
FREQUENCIES = {"Daily market prices": 252, "Daily calendar prices": 365, "Weekly": 52, "Monthly": 12, "Custom": None}
DEFAULT_TEST = {"cost_bps": 10.0}
FIXED_ASSUMPTIONS = {"train_pct": 70, "shrink": 0, "risk_free": 2.0}
DEFAULT_SETUP = {
    "market": "S&P 500 + Nasdaq-100", "history": "5 years", "ticker_text": "AAPL, MSFT, AMZN, GOOGL, NVDA, META, JPM, XOM, JNJ, PG, KO, UNH",
    "start": date(2020, 1, 1), "end": date.today(), "use_cap": True, "cap": 10.0, "target_vol": 20.0, "risk_free": 2.0,
    "train_pct": 70, "shrink": 0, "frequency": "Daily market prices", "periods_per_year": 252.0,
    "price_currency": "USD", "compare_market": True, "download_benchmarks": False,
    "prices_file": None, "benchmark_file": None,
}
FILE_KEYS = ("prices_file", "benchmark_file")
DRAWER_KEYS = [key for key in DEFAULT_SETUP if key not in FILE_KEYS]
COVERAGE_COLUMNS = {"industry": None, "symbol": "Symbol", "name": "Company", "status": "Status", "reason": "Reason",
                    "observations": st.column_config.NumberColumn("Price rows"), "first_date": "First date", "last_date": "Last date"}
DATE_COLUMN = st.column_config.DateColumn("Date", format="D MMM YYYY")
PREVIEW_HTML = ('<ol class="pl-preview"><li><b>Set two limits.</b><span>Set annual volatility and the initial holding cap.</span></li>'
                '<li><b>Buy once and hold.</b><span>Fit on the first 70% of returns; hold through the final 30%.</span></li>'
                '<li><b>Inspect the latest weights.</b><span>Fit all history separately. These weights do not produce the replay.</span></li></ol>')
KEY_LEGEND = '<span class="pl-keys">Keys: <kbd>1</kbd> overview · <kbd>2</kbd> test · <kbd>4</kbd> data &amp; method · <kbd>S</kbd> setup · <kbd>E</kbd> export</span>'
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
def fetch_members(market):
    return fetch_sp500_nasdaq100() if market == "S&P 500 + Nasdaq-100" else fetch_nasdaq100()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe(market, start, end):
    progress_bar = st.progress(0, text=f"Read the current {market} list")
    return download_universe_prices(
        fetch_members(market), start, end,
        progress=lambda done, total, message: progress_bar.progress(done / max(total, 1), text=message),
    )


@st.cache_data(ttl=3600, show_spinner=False,
               hash_funcs={pd.DatetimeIndex: lambda dates: (tuple(dates.astype(str)), dates.name)})
def fetch_benchmarks(dates):
    return download_benchmarks(dates)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_default_risk(reference_day):
    return load_qqq_volatility_reference()


@st.cache_data(show_spinner=False, max_entries=2)
def calculate_selection(prices, benchmark_prices, settings, target_volatility, model_revision):
    """Include the selection model revision in the cached calculation identity."""
    return run_selection(prices, benchmark_prices=benchmark_prices, target_volatility=target_volatility,
                         **settings, cost_bps=DEFAULT_TEST["cost_bps"])


@st.cache_data(show_spinner=False, max_entries=2)
def calculate_frontier(study, prices, model_revision):
    """Retain the latest fitted frontier across view changes and exports."""
    return selection_frontier(study, prices)


# ----------------------------------------------------------------------------- state and callbacks

def init_state():
    new_setup = "setup" not in st.session_state
    defaults = {"setup": dict(DEFAULT_SETUP), "setup_open": False, "view": "Portfolio"}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.view not in ("Portfolio", "Data & method"):
        st.session_state.view = "Portfolio"
    if "target_vol" not in st.session_state.setup:
        # Add the new risk input without changing the saved source, files or holding limit.
        setup = st.session_state.setup
        st.session_state.setup = {**DEFAULT_SETUP, **setup, "target_vol": 20.0, "use_cap": True,
                                  "cap": setup.get("cap", 10.0) if setup.get("use_cap", True) else 100.0}
    if "risk_reference_checked" not in st.session_state:
        st.session_state.risk_reference_checked = True
        setup = st.session_state.setup
        if "result" not in st.session_state and setup["market"] not in ("Demo", "Upload CSV"):
            try:
                reference = fetch_default_risk(date.today())
                st.session_state.qqq_reference = reference
                if new_setup:
                    st.session_state.setup = {**setup, "target_vol": round(reference["volatility"] * 100, 2)}
            except (OSError, ValueError):
                st.session_state.risk_reference_unavailable = True


def set_state(name, value):
    st.session_state[name] = value


def jump(view, anchor):
    st.session_state.view = view
    st.session_state.scroll_to = anchor
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1


def start_build():
    st.session_state.setup = {**st.session_state.setup, **FIXED_ASSUMPTIONS}
    st.session_state.pending_build = True
    st.session_state.build_error = None
    st.session_state.scroll_to = "top"


def change_market():
    if st.session_state.market == "Demo":
        st.session_state.cap = 20.0


def cancel_work():
    st.session_state.pop("pending_build", None)


def open_setup():
    """Open the drawer. Its widgets take their defaults from the applied setup, so an interrupted run cannot blank them."""
    state = st.session_state
    if state.get("pending_build") or state.get("setup_open"):
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


def read_draft(uploads):
    """A new setup from the drawer widgets. Files are (name, bytes): a new upload, else the stored file unless removed."""
    state = st.session_state
    draft = {key: state.get(key, state.setup[key]) for key in DRAWER_KEYS}
    for name in FILE_KEYS:
        upload = uploads.get(name)
        draft[name] = ((upload.name, upload.getvalue()) if upload is not None
                       else None if name in state.get("draft_clear", set()) else state.setup[name])
    draft.update(FIXED_ASSUMPTIONS)
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
    settings = dict(train_fraction=FIXED_ASSUMPTIONS["train_pct"] / 100, risk_free_rate=FIXED_ASSUMPTIONS["risk_free"] / 100,
                    max_weight=setup["cap"] / 100, shrinkage=FIXED_ASSUMPTIONS["shrink"] / 100,
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
            "target_volatility": setup["target_vol"] / 100,
            "download_for_csv": download_for_csv, "ticker_text": ticker_text, **files, "input_choices": input_choices}


def differs(plan):
    """True when a resolved setup would not reproduce the applied result."""
    metadata = st.session_state.result[2]
    return (metadata["universe"] != plan["market"] or st.session_state.get("applied_inputs") != plan["input_choices"]
            or any(metadata.get(key) != value for key, value in plan["settings"].items())
            or metadata.get("target_volatility") != plan["target_volatility"]
            or metadata["price_currency"] != plan["price_currency"] or metadata["compare_market"] != plan["compare_market"]
            or plan["source"] == "Yahoo Finance" and (metadata.get("requested_start") != plan["start"].isoformat()
                                                      or metadata.get("requested_end_exclusive") != plan["end"].isoformat()))


def make_report_callable():
    """Capture the completed selection and its inputs without recalculating on export."""
    _, prices, metadata = st.session_state.result
    selection = st.session_state.selection
    return lambda: selection_report_zip(selection, prices, metadata,
                                        frontier_data=calculate_frontier(selection, prices, SELECTION_MODEL_REVISION))


def settings_frame(metadata):
    """Study settings with plain labels. The metadata does not change."""
    return pd.DataFrame([{"Setting": "History for the initial fit" if key == "train_fraction" else story.setting_label(key),
                          "Value": f"{value:.2%}" if key == "target_volatility" else story.setting_text(key, value)}
                         for key, value in metadata.items() if key not in ("universe_coverage", "profile_model_revision", "selection_model_revision")])


def day(value):
    stamp = pd.Timestamp(value)
    return f"{stamp.day} {stamp:%b %Y}"


# ----------------------------------------------------------------------------- slow work

def error_kind(exc):
    """Name the failure for the error state. A retry helps only "yahoo" (network) and "other"."""
    message = str(exc)
    if isinstance(exc, OSError) or message.startswith("Yahoo price download failed"):
        return "yahoo"
    if message.startswith("Yahoo returned no prices"):
        return "no_prices"
    if any(phrase in message.lower() for phrase in ("infeasible", "risk budget", "below the feasible minimum")):
        return "infeasible"
    if message == "Choose an adjusted-price CSV first.":
        return "no_file"
    if message.startswith("Provide stock price columns"):
        return "no_stocks"
    return "other"


def build(plan, mark, progress_slot):
    """Load inputs and build one selection study. Replace the applied result only after success."""
    market, source, settings = plan["market"], plan["source"], plan["settings"]
    start, end = plan["start"], plan["end"]
    loaded = None
    try:
        metadata = {"source": source, "universe": market, "price_currency": plan["price_currency"],
                    "compare_market": plan["compare_market"], "target_volatility": plan["target_volatility"], **settings}
        metadata["currency_assumption"] = "The user declares a common price currency. No verification or currency conversion occurs."
        universe_benchmarks = None
        if source == "Yahoo Finance":
            metadata.update(requested_start=start.isoformat(), requested_end_exclusive=end.isoformat())
            if market in ("S&P 500 + Nasdaq-100", "Nasdaq-100"):
                mark(2)
                with progress_slot:
                    loaded = fetch_universe(market, start.isoformat(), end.isoformat())
                prices, universe_benchmarks = loaded.prices, loaded.benchmarks
                metadata.update(loaded.metadata)
                metadata["universe"] = market
                if len(prices.columns) < 2:
                    raise ValueError("Fewer than two selected securities have complete prices. Review coverage below or choose a shorter history.")
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
        mark(4)
        prices, embedded_benchmarks = split_stock_prices(prices)
        if embedded_benchmarks is not None:
            metadata["excluded_candidate_symbols"] = list(embedded_benchmarks.columns)
            metadata["candidate_note"] = "Comparison ETFs excluded from candidate holdings: " + ", ".join(embedded_benchmarks.columns) + "."
        benchmark_prices = None
        if market == "Demo":
            metadata["benchmarks_note"] = "Synthetic prices have no real-market benchmark comparison."
        elif not plan["compare_market"]:
            metadata["benchmarks_note"] = "Market comparisons are off. To turn them on, select Change, then Data options."
        elif plan["price_currency"] != "USD":
            metadata["benchmarks_note"] = "Market comparisons require USD prices. No currency conversion occurs."
        else:
            try:
                if plan["benchmark_file"] is not None:
                    benchmark_prices = load_csv(BytesIO(plan["benchmark_file"][1]))
                    metadata["benchmark_source"] = "CSV: " + plan["benchmark_file"][0]
                elif embedded_benchmarks is not None:
                    benchmark_prices = embedded_benchmarks
                    metadata["benchmark_source"] = "Input columns: " + ", ".join(embedded_benchmarks.columns)
                elif plan["download_for_csv"]:
                    benchmark_prices = fetch_benchmarks(prices.index)
                    metadata["benchmark_source"] = "Yahoo Finance"
                elif universe_benchmarks is not None:
                    benchmark_prices = universe_benchmarks
                    metadata["benchmark_source"] = "Yahoo Finance"
                elif source == "Yahoo Finance":
                    benchmark_prices = fetch_benchmarks(prices.index)
                    metadata["benchmark_source"] = "Yahoo Finance"
                else:
                    metadata["benchmarks_note"] = ("Add a market prices file, include SPY and QQQ columns, or turn on the Yahoo download. "
                                                   "Select Change, then Data options.")
                if benchmark_prices is not None:
                    benchmark_prices = validate_selection_benchmarks(benchmark_prices, prices.index)
            except (ValueError, RuntimeError, OSError) as exc:
                metadata["benchmarks_note"] = f"Market comparison unavailable: {exc}"
                benchmark_prices = None
        selection = calculate_selection(prices, benchmark_prices, settings, plan["target_volatility"], SELECTION_MODEL_REVISION)
    except (ValueError, RuntimeError, OSError) as exc:
        for key in ("result", "selection", "latest_profiles", "backtests", "benchmarks", "evidence", "benchmark_prices", "coverage"):
            st.session_state.pop(key, None)
        if loaded is not None:
            st.session_state.coverage = loaded.coverage
        st.session_state.build_error = str(exc)
        st.session_state.build_error_kind = error_kind(exc)
        st.session_state.pop("pending_build", None)
        st.rerun()
    metadata["selection_model_revision"] = SELECTION_MODEL_REVISION
    st.session_state.update(result=(None, prices, metadata), selection=selection,
                            applied_inputs=plan["input_choices"], benchmark_prices=benchmark_prices,
                            coverage=loaded.coverage if loaded is not None else None,
                            build_error=None, scroll_to="top")
    for key in ("pending_build", "pending_study", "latest_profiles", "backtests", "benchmarks", "evidence",
                "profile_refresh_error", "holding_strategy", "chart_strategies", "risk_portfolio", "correlation_assets",
                "preview_target_vol", "preview_cap"):
        st.session_state.pop(key, None)
    st.rerun()


# ----------------------------------------------------------------------------- shared pieces

def scroll(target):
    st.html(SCROLL_JS.replace("__TARGET__", target).replace("__NONCE__", str(st.session_state.get("scroll_nonce", 0))),
            unsafe_allow_javascript=True)


def context_line(text, align="center"):
    with st.container(horizontal=True, horizontal_alignment=align, vertical_alignment="center", gap="small"):
        st.html(text, width="content")
        st.button("Settings", key="edit_setup", type="tertiary", icon=":material/tune:", shortcut="S", on_click=open_setup,
                  disabled=busy)


def title_html(title, sub=""):
    return f'<h3 class="pl-title">{html.escape(title)}</h3>' + (f'<p class="pl-sub">{html.escape(sub)}</p>' if sub else "")


def stat_strip(stats):
    return '<div class="pl-stats">' + "".join(
        f'<div class="pl-stat"><span class="pl-num">{html.escape(str(number))}</span><span class="pl-lbl">{label}</span>'
        f'<small>{html.escape(sub)}</small></div>' for number, label, sub in stats) + "</div>"


def try_buttons(fix=False):
    """The error actions. When a retry would fail the same way, the primary action opens the drawer instead."""
    with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center", gap="small"):
        if not st.session_state.setup_open:
            if fix:
                st.button("Change the study", key="fix_setup", type="primary", icon=":material/tune:", on_click=open_setup)
            else:
                st.button("Try again", key="build", type="primary", shortcut="Mod+Enter", on_click=start_build)


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
    st.html('<p class="pl-sub">Choose the stocks and the history. Nothing changes until you select Build portfolio.</p>')
    uploads = {}
    market = st.radio("Stocks", MARKETS, index=MARKETS.index(setup["market"]), key="market", captions=MARKET_CAPTIONS, on_change=change_market)
    if market == "My tickers":
        st.text_area("Stock symbols", setup["ticker_text"], key="ticker_text",
                     help="Separate Yahoo Finance symbols with commas. Exchange suffixes such as .L work. SPY and QQQ serve only as comparisons.")
    elif market == "Upload CSV":
        uploads["prices_file"] = file_slot(
            "Adjusted prices (.csv)", "prices_upload", "prices_file", "clear_prices_file",
            caption="Use Date as the first column, then one stock per column. Prices must be complete and use a common currency. "
                    "SPY and QQQ columns serve only as comparisons and cannot become holdings.")
    if market in ("S&P 500 + Nasdaq-100", "Nasdaq-100", "My tickers", "Original 60 holdings"):
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
    st.number_input("Annual volatility budget", 1.0, 100.0, setup["target_vol"], step=0.1, format="%.2f", key="target_vol",
              help="A fitted annual volatility ceiling. Realized volatility can exceed it. No borrowing or automatic cash allocation.")
    risk_reference()
    st.number_input("Holding limit", 1.0, 100.0, setup["cap"], step=1.0, format="%g", key="cap",
                    help="A 10% limit needs at least ten assets. If the risk budget cannot be met, the build stops.")
    if market == "Demo":
        st.caption("The demo has six simulated assets. Its default holding limit is 20%, shown above; you can change it. At least 17% is needed.")
    st.caption("One fixed method: fit on the first 70% of return observations, then buy once and hold through the remaining 30%. "
               "The entry fee is 0.10% of the amount bought. There is no later rebalancing.")
    with st.expander("Data options  :gray[frequency, currency, market comparisons]"):
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
                                                  "clear_benchmark_file", help="Use Date with SPY, QQQ, or both on exactly the same dates as your stock file.")
            st.checkbox("Download SPY and QQQ from Yahoo Finance for this file", setup["download_benchmarks"], key="download_benchmarks")
            st.caption("CSV analysis stays offline unless you select this download. Local comparison columns take priority over a download.")
    draft = read_draft(uploads)
    if "result" in state and differs(resolve(draft)):
        st.info("Your setup differs from this result. Select Build portfolio to apply it.")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("Cancel", key="cancel_setup"):
            close_setup()
            st.rerun()
        if st.button("Build portfolio", key="build", type="primary", shortcut="Mod+Enter", on_click=set_state, args=("view", "Portfolio")):
            state.setup = draft
            state.pop("preview_target_vol", None)
            state.pop("preview_cap", None)
            state.setup_open = False
            start_build()
            st.rerun()


# ----------------------------------------------------------------------------- welcome, work and error

def hero_work():
    setup = st.session_state.setup
    plan = resolve(setup)
    with st.container(key="hero", horizontal_alignment="center"):
        context_line(story.context_setup_html(setup))
        st.html('<h1 class="pl-display pl-working">Build and check one candidate.</h1>')
        steps = st.empty()
        labels = {2: "Load the selected prices", 3: "Prepare the risk budget", 4: "Fit one stock model, then replay buy and hold"}
        steps.info("Read the selected universe")
        progress_slot = st.container(key="work_progress")
        st.button("Cancel", key="cancel_build", type="tertiary", on_click=cancel_work)
    build(plan, lambda number: steps.info(labels[number]), progress_slot)


def welcome_mathematics(setup):
    periods = resolve(setup)["settings"]["periods_per_year"]
    with st.container(key="landing_math"):
        with st.container(key="landing_estimates"):
            st.latex(r"\widehat\mu=A\bar r,\qquad "
                     r"\widehat\Sigma=A\operatorname{Cov}_{\mathrm{sample}}(r)")
            st.markdown(
                r"$\boldsymbol{r}:$ vector of stock returns; $\boldsymbol{\bar r}:$ sample mean returns; "
                rf"$\boldsymbol{{A={periods:g}}}:$ observations per year. "
                r"$\boldsymbol{\widehat\mu}:$ fitted annual returns; $\boldsymbol{\widehat\Sigma}:$ annual covariance matrix; "
                r"$\boldsymbol{\operatorname{Cov}_{\mathrm{sample}}}:$ sample covariance."
            )
        with st.container(key="landing_optimization"):
            st.latex(r"\begin{aligned}"
                     r"w^*&=\underset{w}{\operatorname{arg\,max}}\;\widehat\mu^{\mathsf T}w\\[4pt]"
                     r"\text{subject to}\quad&\mathbf1^{\mathsf T}w=1,\quad 0\le w_i\le h,\\"
                     r"&\sqrt{w^{\mathsf T}\widehat\Sigma w}\le b"
                     r"\end{aligned}")
            st.markdown(
                r"$\boldsymbol{w}:$ initial weight vector; $\boldsymbol{w_i}:$ fraction in stock $\boldsymbol{i}$; $\boldsymbol{w^*}:$ optimal weights. "
                r"$\boldsymbol{b}:$ fitted annual volatility budget; $\boldsymbol{h}:$ initial holding limit; "
                r"$\boldsymbol{\mathbf1}:$ vector of ones; $\boldsymbol{\mathsf T}:$ transpose; "
                r"$\boldsymbol{\operatorname{arg\,max}}:$ weights that maximize fitted return."
            )
        with st.container(key="landing_growth"):
            st.latex(r"V_t=\frac{V_{\mathrm{start}}}{1+c}"
                     r"\sum_i w_i\frac{p_{i,t}}{p_{i,e}}\qquad(t\ge e)")
            st.markdown(
                r"$\boldsymbol{V_t}:$ value at observation $\boldsymbol{t}$; $\boldsymbol{V_{\mathrm{start}}}:$ starting cash; "
                r"$\boldsymbol{p_{i,t}}:$ adjusted price of stock $\boldsymbol{i}$ at $\boldsymbol{t}$; $\boldsymbol{e}:$ entry observation. "
                r"$\boldsymbol{c=0.10\%}:$ fee per unit bought; $\boldsymbol{\sum_i}:$ sum across stocks."
            )


def welcome_or_error():
    setup = st.session_state.setup
    market = setup["market"]
    error = st.session_state.get("build_error")
    if error is None:
        with st.container(key="landing"):
            st.html('<h1 class="pl-display">Efficient Frontier Generator '
                    '<span class="pl-method-title">Constrained Mean–Variance Optimization</span></h1>')
            mathematics, controls = st.columns([2.2, 1], gap="large")
            with mathematics:
                welcome_mathematics(setup)
            with controls:
                with st.container(key="landing_controls"):
                    if not st.session_state.setup_open:
                        target = st.number_input("Annual volatility budget", 1.0, 100.0, setup["target_vol"],
                                                 step=0.1, format="%.2f", key="preview_target_vol",
                                                 help="b: the maximum fitted annual standard deviation at the initial allocation.")
                        reference = st.session_state.get("qqq_reference")
                        if reference:
                            st.caption(f"Nasdaq-100 proxy · QQQ {reference['volatility']:.2%}\n\n"
                                       f"One year through {day(reference['last_return_date'])}. "
                                       "Daily sample volatility × √252. This is a separate window from the portfolio fit.")
                        elif st.session_state.get("risk_reference_unavailable"):
                            st.caption("QQQ volatility is unavailable. The 20% starting value is a manual fallback, not a current Nasdaq-100 estimate.")
                        cap = st.number_input("Holding limit", 1.0, 100.0, setup["cap"], step=1.0,
                                              format="%g", key="preview_cap",
                                              help="h: the maximum initial weight in one stock. A 10% cap requires at least ten stocks.")
                        st.session_state.setup = {**setup, "target_vol": target, "cap": cap, "use_cap": True}
            st.html(PREVIEW_HTML)
            if not st.session_state.setup_open:
                with st.container(key="landing_actions", horizontal=True, horizontal_alignment="center",
                                  vertical_alignment="center", gap="medium"):
                    st.button("Build portfolio", key="build", type="primary", shortcut="Mod+Enter", on_click=start_build)
                    st.button("Settings", key="edit_setup", type="tertiary", icon=":material/tune:", shortcut="S", on_click=open_setup)
        return
    with st.container(key="hero", horizontal_alignment="center"):
        context_line(story.context_setup_html(setup))
        kind = st.session_state.get("build_error_kind", "other")
        headline = {"yahoo": "We could not reach Yahoo Finance.", "no_prices": "Yahoo Finance had no prices for these symbols.",
                    "infeasible": "These settings have no solution.", "no_file": "The study needs a price file.",
                    "no_stocks": "The portfolio needs stock prices."}.get(
            kind, "The study could not run.")
        st.html(f'<h1 class="pl-display">{html.escape(headline)}</h1>')
        st.error(error)
        if kind == "infeasible":
            cap = setup["cap"]
            need = (f"A {cap:g}% limit needs at least {math.ceil(100 / cap - 1e-9)} "
                    f"{'assets' if market in ('Demo', 'Upload CSV') else 'stocks'}. " if cap > 0 else "")
            st.html(f'<p class="pl-lead pl-center">{need}Review the holding limit and annual volatility budget in Change the study. No constraint is relaxed automatically.</p>')
        try_buttons(fix=kind in ("no_prices", "infeasible", "no_file", "no_stocks"))
    coverage = st.session_state.get("coverage")
    if coverage is not None:
        with st.container(border=True, key="tile_missing"):
            st.html(title_html("Which stocks were missing"))
            st.dataframe(coverage, hide_index=True, width="stretch", column_config=COVERAGE_COLUMNS)


# ----------------------------------------------------------------------------- overview

def risk_reference():
    benchmark_prices = st.session_state.get("benchmark_prices")
    if benchmark_prices is None:
        return
    periods = st.session_state.result[2]["periods_per_year"]
    volatility = benchmark_prices.pct_change(fill_method=None).dropna().std() * math.sqrt(periods)
    st.caption("Reference from the saved input history: " + ", ".join(f"{name} {value:.1%}" for name, value in volatility.items())
               + " annual historical volatility from ordinary sample covariance. "
               "These are observations over the full input history, not the replay or future limits.")


def selection_fit_math(study):
    settings = study.settings
    with st.container(key="math_fit"):
        st.html(title_html("Mathematics · portfolio selection"))
        st.markdown("**Estimate returns and covariance**")
        st.write("For each stock, use simple returns between consecutive adjusted prices. "
                 "The fit contains m return observations. A is the number of observations per year.")
        st.latex(r"r_{i,s}=\frac{p_{i,s}}{p_{i,s-1}}-1,\qquad "
                 r"\bar r=\frac{1}{m}\sum_{s=1}^{m}r_s")
        st.latex(r"\widehat\mu=A\bar r,\qquad "
                 rf"A={settings['periods_per_year']:g}")
        st.latex(r"\widehat\Sigma=\frac{A}{m-1}\sum_{s=1}^{m}"
                 r"(r_s-\bar r)(r_s-\bar r)^{\mathsf T}")
        st.caption("μ̂ contains annual arithmetic mean returns. Σ̂ is the annual sample covariance matrix. "
                   "This app uses no covariance shrinkage.")
        st.markdown("**Choose the highest fitted mean within your limits**")
        st.latex(r"\begin{aligned}"
                 r"w^*&=\underset{w}{\operatorname{arg\,max}}\;\widehat\mu^{\mathsf T}w\\[6pt]"
                 r"\text{subject to}\quad&\mathbf 1^{\mathsf T}w=1,\\"
                 r"&0\le w_i\le h,\\"
                 r"&\sqrt{w^{\mathsf T}\widehat\Sigma w}\le b"
                 r"\end{aligned}")
        st.latex(rf"b={settings['target_volatility']:g}\;"
                 rf"({settings['target_volatility']:.2%}),\qquad "
                 rf"h={settings['max_weight']:g}\;({settings['max_weight']:.0%})".replace("%", r"\%"))
        st.write("The weights w sum to one, with no short sales. h caps each initial holding. "
                 "b caps estimated annual volatility when the allocation is chosen. "
                 "After purchase, weights drift and realized volatility can exceed b. "
                 "Neither limit triggers a trade.")
        st.markdown("**An estimate and a replay answer different questions**")
        st.write(f"The replay chooses its initial weights from the first {settings['train_fraction']:.0%} of return observations. "
                 "The latest fit uses all supplied history, including the replay dates, and describes a separate starting allocation.")
        st.latex(r"\widehat\mu_p=\widehat\mu^{\mathsf T}w,\qquad "
                 r"\widehat\sigma_p=\sqrt{w^{\mathsf T}\widehat\Sigma w}")
        st.caption("These fitted moments apply one fixed weight vector to each historical return observation, before fees. "
                   "They are not the result of buying once and letting weights drift through that history. "
                   "The optimizer maximizes fitted arithmetic mean, not CAGR or the Sharpe ratio.")


def selection_replay_math(study):
    settings = study.settings
    fee = settings["cost_bps"] / 10_000
    with st.container(key="math_replay"):
        st.html(title_html("Mathematics · buy-and-hold returns"))
        st.markdown("**One purchase, then no rebalancing**")
        st.write(f"Fit once through {day(settings['initial_training_end'])}. "
                 f"Buy at the next close, {day(settings['test_start'])}. "
                 "The portfolio stays in cash until that purchase. "
                 "There are no later refits, trades, volatility adjustments, or final liquidation.")
        st.write("Let e be the entry close, wᵢ the initial stock weights, and pᵢ,ₜ the adjusted prices. "
                 "Vstart is the starting cash and c is the fee per unit invested.")
        st.latex(r"V_e=\frac{V_{\mathrm{start}}}{1+c},\qquad "
                 rf"c={fee:g}\;({fee:.2%})".replace("%", r"\%"))
        st.caption("The entry fee is c × Vₑ. It comes out of starting cash before the holdings begin to earn returns.")
        st.latex(r"\begin{aligned}"
                 r"g_{i,t}&=\frac{p_{i,t}}{p_{i,e}}\\[8pt]"
                 r"V_t&=V_e\sum_i w_i g_{i,t}\quad(t\ge e)"
                 r"\end{aligned}")
        st.latex(r"a_{i,t}=\frac{w_i g_{i,t}}{\sum_j w_j g_{j,t}}")
        st.write("gᵢ,ₜ is each stock’s growth factor since entry. aᵢ,ₜ is its actual portfolio weight. "
                 "Faster-growing holdings gain weight without a new trade. "
                 "The initial holding limit can therefore be exceeded later.")
        st.caption("The calculation follows the supplied adjusted-price convention. "
                   "For Yahoo prices, this includes split and dividend adjustments. "
                   "It does not separately model cash dividends or literal share counts across corporate actions.")
        st.markdown("**Measure the completed net-return path**")
        st.write("The table uses N observed returns, including the entry-fee interval. "
                 "The chart starts at Vstart. The entry interval has the fee but no stock return.")
        st.latex(r"R_t=\frac{V_t}{V_{t-1}}-1,\qquad "
                 r"\bar R=\frac{1}{N}\sum_{t=1}^{N}R_t")
        st.latex(r"\mu_{\mathrm{realized}}=A\bar R,\qquad "
                 rf"A={settings['periods_per_year']:g}")
        st.latex(r"\sigma_{\mathrm{realized}}="
                 r"\sqrt{\frac{A}{N-1}\sum_{t=1}^{N}(R_t-\bar R)^2}")
        st.latex(r"\mathrm{CAGR}=\left(\frac{V_N}{V_{\mathrm{start}}}\right)^{A/N}-1")
        st.latex(r"\mathrm{MDD}=\min_{0\le t\le N}"
                 r"\left(\frac{V_t}{\max_{0\le s\le t}V_s}-1\right)")
        st.caption("The annual mean scales the average one-period return. CAGR compounds the observed path. "
                   "Maximum drawdown is the worst fall from a previous peak, including starting cash. "
                   "The budget constrains fitted volatility, not this realized sample standard deviation.")


def selection_period(study, title, benchmark_note):
    st.html(title_html(title))
    st.write("Choose the stock allocation once using earlier prices, buy once, then let the holdings drift. "
             "This path does not use the latest fitted weights.")
    verdict = selection_replay_verdict_html(study)
    if verdict:
        st.html(verdict)
    st.caption(f"Return dates: {day(study.equity.index[1])} to {day(study.equity.index[-1])}. "
               f"Starting capital at the {day(study.equity.index[0])} close. "
               "All rows use the same dates and fee rate. Each row measures its own net return path. "
               "Annual mean scales the average return to one year. CAGR measures compound growth.")
    columns = ["annual_mean", "cagr", "volatility", "max_drawdown"]
    metrics = selection_metrics_frame(study)[columns].rename(index={"Historical-mean portfolio": REPLAY_LABEL})
    st.dataframe(metrics.style.format("{:.2%}", na_rep="—"), width="stretch",
                 column_config={"_index": "Rule or baseline", **SELECTION_METRIC_LABELS})
    st.plotly_chart(selection_chart(study), width="stretch", theme=None, config=PLOTLY_CONFIG)
    st.caption(benchmark_note)


def overview():
    _, prices, metadata = st.session_state.result
    selection = st.session_state.selection
    with st.container(key="hero", horizontal_alignment="center"):
        context_line(story.context_result_html(metadata, prices))
        st.html(f'<h1 class="pl-display">{metadata["target_volatility"]:.2%} annual volatility budget.</h1>')
        st.html(stat_strip(((f"{metadata['target_volatility']:.2%}", "Annual volatility budget", "Fitted ceiling"),
                            (f"{metadata['max_weight']:.0%}", "Holding limit", "Maximum target weight"))))
        if "synthetic" in metadata["source"]:
            st.info("Demo · synthetic prices. These are not market results. The six-asset demo uses a 20% holding limit by default.")
        if metadata.get("candidate_note"):
            st.info(metadata["candidate_note"])
    with st.container(border=True, key="tile_fit_replay"):
        st.html(selection_comparison_html(selection, prices))
    with st.container(border=True, key="tile_fitted_frontier"):
        frontier_data = calculate_frontier(selection, prices, SELECTION_MODEL_REVISION)
        st.html(selection_frontier_summary_html(selection, prices, frontier_data))
        st.plotly_chart(selection_frontier_chart(frontier_data, selection), width="stretch", theme=None, config=PLOTLY_CONFIG)
        selection_fit_math(selection)
    names = story.company_names(st.session_state.get("coverage"))
    portfolio = selection.latest[selection.selected]
    with st.expander(f"Latest fitted holdings · fitted through {day(prices.index[-1])}", expanded=True):
        st.caption("These weights use all supplied prices, including the replay dates. "
                   "The historical replay bought a separate allocation once, before the later returns.")
        if not selection.supported:
            st.warning("The buy-and-hold replay did not meet the performance checks. It used a separate initial allocation.")
        frame = portfolio.weights.sort_values(ascending=False).rename("Weight").rename_axis("Symbol").reset_index()
        if names:
            frame.insert(1, "Company", frame.Symbol.map(names))
        st.dataframe(frame, hide_index=True, width="stretch", column_config={"Weight": st.column_config.NumberColumn("Weight", format="percent")})
        for note in stale_price_notes(prices):
            st.warning(note)
    st.html('<div id="pl-market"></div>')
    with st.container(border=True, key="tile_final_test"):
        selection_period(selection.test, "Historical replay · buy and hold", selection_benchmark_note(selection))
        selection_replay_math(selection)
        st.html(story.result_drivers_html(selection.test, selection.selected))
        with st.expander("What drove the stock rule’s test result?"):
            st.dataframe(selection.test.holdings[selection.selected].style.format("{:.2%}"), width="stretch",
                         column_config={"_index": "Symbol", **{name: label for name, (label, _) in HOLDINGS.items()}})
    if selection.warnings:
        st.warning(selection.warnings[0])
        with st.expander("All data and model notes"):
            for note in selection.warnings:
                st.write(note)
            st.dataframe(selection.eligibility, hide_index=True, width="stretch")
    if metadata.get("universe_limitation"):
        st.caption(metadata["universe_limitation"])
    st.button("Open Data & method", key="open_data_method", on_click=jump, args=("Data & method", "top"))
    st.html(KEY_LEGEND)


# ----------------------------------------------------------------------------- data and method

def data_method():
    _, prices, metadata = st.session_state.result
    selection = st.session_state.selection
    study = selection.test
    coverage = st.session_state.get("coverage")
    context_line(story.context_result_html(metadata, prices), align="left")
    st.html('<h1 class="pl-page">Data &amp; method</h1>')
    with st.container(border=True, key="tile_method"):
        st.html(title_html("One stock model"))
        st.markdown(
            "1. **Estimate returns and risk.** Use average past returns and the relationships between stocks. "
            "Use sample covariance, as in the original R model.\n"
            "2. **Choose stock weights.** Maximize estimated average return within your volatility budget and holding limit. "
            "Keep the portfolio fully invested, with no borrowing or short sales. "
            "This does not maximize compound growth or the return-to-volatility ratio.\n"
            "3. **Buy once and hold.** Use the first 70% of return observations for one initial fit. "
            "Buy once at the next close, then let those holdings drift through the remaining 30%. "
            "There are no later refits or rebalances.\n"
            "4. **Fit the latest weights.** Apply the same model to all supplied history. "
            "These weights describe a separate starting allocation. They do not produce the replay."
        )
        st.caption("The initial target uses prices through the prior close and trades at the next close. "
                   "The first test interval stays in cash with zero interest. "
                   f"The model charges {selection.settings['cost_bps'] / 10_000:.2%} on the amount bought at entry. "
                   "There are no later trades or fees, and no final sale.")
        st.caption(selection_benchmark_note(selection))
        st.caption("The fixed 2% risk-free rate affects Sharpe and Sortino metrics in the export only. "
                   "It does not change stock weights or pay interest on cash. "
                   "Taxes, currency conversion, liquidity, and future membership changes are excluded.")
    selection_fit_math(selection)
    selection_replay_math(selection)
    with st.expander("Historical performance checks"):
        st.html(selection_summary_html(selection, metadata))
    with st.expander("Initial and latest fitted estimates"):
        st.caption("Each row uses only prices available at its fit date and the saved target weights. "
                   "Both volatility estimates use ordinary sample covariance because this model has no covariance shrinkage. "
                   "Neither is the later realized volatility. The same table is exported as fit_estimates.csv.")
        st.dataframe(selection_fit_estimates(selection, prices).style.format(
            {key: "{:.2%}" for key in (*FIT_METRIC_LABELS, "max_weight")}), width="stretch",
            column_config={"_index": st.column_config.DateColumn("Fitted through", format="D MMM YYYY"),
                           "execution_date": st.column_config.DateColumn("Traded on", format="D MMM YYYY"),
                           "phase": "Fit", "observations": "Return observations", "max_weight": "Largest holding",
                           **FIT_METRIC_LABELS})
    with st.container(border=True, key="tile_data"):
        st.html(title_html("The data behind this result"))
        st.caption(metadata.get("universe_limitation") or
                   "Fixed list. The study keeps the same input assets for the whole period. This can favor surviving stocks.")
        if coverage is not None:
            listed = f"List of {day(metadata['universe_source_date'])}" if metadata.get("universe_source_date") else "Today's list"
            st.html(stat_strip(((metadata.get("universe_requested", len(coverage)), "Checked", listed),
                                (metadata.get("universe_included", len(prices.columns)), "Used", "Complete prices"),
                                (metadata.get("universe_excluded", 0), "Left out", "Missing or late prices"))))
            st.dataframe(coverage, hide_index=True, width="stretch", column_config=COVERAGE_COLUMNS)
        with st.expander("Price eligibility at each fit"):
            st.caption("The model excludes stocks with at least 20 consecutive unchanged price intervals in past data. "
                       "Repeated prices do not prove an invalid quote. Original price columns remain in the export.")
            st.dataframe(selection.eligibility, hide_index=True, width="stretch")
        with st.expander("Input prices"):
            st.dataframe(prices, width="stretch", height=300, column_config={"_index": DATE_COLUMN})
            st.download_button("Download prices.csv", csv_text(prices, index_label="Date"), "prices.csv", "text/csv")
        with st.expander("Source and settings"):
            st.dataframe(settings_frame({**metadata, **DEFAULT_TEST}), hide_index=True, width="stretch")
    with st.expander("Exact replay holdings, initial weights, and entry trade"):
        st.caption("These tables describe the stock portfolio in the buy-and-hold replay. "
                   "The export also contains every comparison path and trade.")
        st.dataframe(study.holdings[selection.selected].style.format("{:.2%}"), width="stretch",
                     column_config={"_index": "Symbol", **{name: label for name, (label, _) in HOLDINGS.items()}})
        st.dataframe(study.allocations[selection.selected].style.format("{:.2%}"),
                     column_config={"_index": DATE_COLUMN}, width="stretch")
        st.dataframe(study.trades[selection.selected].style.format({
            "turnover": "{:.2%}", "cost": lambda value: f"{value * 10_000:,.2f}",
            "nav_before": lambda value: f"{value * 10_000:,.0f}", "nav_after": lambda value: f"{value * 10_000:,.0f}",
            "train_start": day, "train_end": day}), width="stretch", column_config={
                "_index": DATE_COLUMN, "turnover": "Turnover", "cost": "Fee", "nav_before": "Value before",
                "nav_after": "Value after", "train_start": "Fit from", "train_end": "Fit to"})
        st.caption("Values and fees are per 10,000 at the start. Holding contributions less fees equal the net total return.")
    st.html(KEY_LEGEND)


# ----------------------------------------------------------------------------- page

init_state()
ready = "result" in st.session_state
current_selection = (ready and "selection" in st.session_state
                     and st.session_state.result[2].get("selection_model_revision") == SELECTION_MODEL_REVISION)
busy = bool(st.session_state.get("pending_build"))

if ready:
    with st.container(key="localnav", horizontal=True, vertical_alignment="center", gap="medium"):
        st.html('<span class="pl-wordmark">Efficient Frontier Generator</span>', width="stretch")
        st.segmented_control("View", ["Portfolio", "Data & method"], key="view", required=True,
                             format_func=lambda name: story.VIEW_LABELS.get(name, name),
                             label_visibility="collapsed", disabled=busy)
        if current_selection:
            st.download_button("Export", data=make_report_callable(), file_name="portfolio-lab.zip", mime="application/zip",
                               key="export", type="primary", icon=":material/ios_share:", on_click="ignore", shortcut="E",
                               disabled=busy, help="One zip file. The report opens offline and prints to PDF.")

KEYS = (("key_holdings", "Overview", "1", jump, ("Portfolio", "top")),
        ("key_market", "Later test", "2", jump, ("Portfolio", "pl-market")),
        ("key_data_method", "Data & method", "4", jump, ("Data & method", "top")))
if ready:
    with st.container(key="keys"):
        for key, label, shortcut, callback, args in KEYS:
            st.button(label, key=key, shortcut=shortcut, on_click=callback, args=args, disabled=busy)

if st.session_state.setup_open:
    setup_sheet()

if st.session_state.get("pending_build"):
    hero_work()
elif ready and not current_selection:
    st.info("This saved study uses a previous selection model. Your input prices are retained. "
            "Rebuild for buy and hold: one initial purchase, no later rebalancing. The previous results are hidden.")
    st.button("Set risk budget and rebuild", key="migrate_selection", on_click=open_setup)
elif not ready:
    welcome_or_error()
elif st.session_state.view == "Portfolio":
    overview()
else:
    data_method()

target = st.session_state.pop("scroll_to", None)
if target is None and st.session_state.get("previous_view") != st.session_state.get("view"):
    target = "top"
if target:
    scroll(target)
st.session_state.previous_view = st.session_state.get("view")
