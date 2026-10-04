"""Run with ./run.sh from the repository root."""

from datetime import date
from hashlib import sha256
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from efficient_frontier.backtest import run_backtests
from efficient_frontier.backtest_report import HOLDINGS, METRICS, backtest_chart
from efficient_frontier.benchmark_report import evidence_chart, evidence_summary_frame
from efficient_frontier.benchmarks import compare_benchmarks
from efficient_frontier.core import analyze
from efficient_frontier.data import BENCHMARK_SYMBOLS, demo_prices, download_benchmarks, download_prices, load_csv, original_tickers, parse_tickers, validate_benchmark_prices
from efficient_frontier.evidence import analyze_evidence
from efficient_frontier.presentation import csv_text, frontier_chart, holdout_chart, latest_profile_chart, report_zip, weights_frame
from efficient_frontier.profiles import build_profiles
from efficient_frontier.risk import risk_contributions, risk_summary
from efficient_frontier.style import APP_CSS, COLORS, style_chart
from efficient_frontier.universe import download_universe_prices, fetch_nasdaq100


st.set_page_config(page_title="Portfolio Lab", page_icon="◒", layout="wide")
st.set_option("client.toolbarMode", "viewer")
st.markdown(APP_CSS, unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False, max_entries=2)
def export_report(result, prices, metadata, study, latest, benchmarks, evidence):
    return report_zip(result, prices, metadata, study, latest, benchmarks, evidence)


def show_compare():
    st.session_state.view = "Compare"


st.title("Portfolio Lab")
st.caption("Choose your risk. Understand what you hold.")
ready = "result" in st.session_state
with st.expander("Market & settings", expanded=not ready):
    market = st.selectbox("Investment universe", ["Nasdaq-100", "My tickers", "Original 60 holdings", "Upload CSV", "Demo"], key="market")
    source = "Demo · synthetic" if market == "Demo" else "Upload CSV" if market == "Upload CSV" else "Yahoo Finance"
    if market == "Nasdaq-100":
        st.caption("Start with the full current Nasdaq-100 list, including separate share classes. Yahoo Finance supplies the prices.")
    history = st.selectbox("History", ["5 years", "10 years", "Custom dates"], key="history") if source == "Yahoo Finance" else None
    with st.container():
        ticker_text = ""
        uploaded = benchmark_upload = None
        if market == "My tickers":
            ticker_text = st.text_area("Stock symbols", "AAPL, MSFT, AMZN, GOOGL, NVDA, META", key="ticker_text",
                                       help="Separate Yahoo Finance symbols with commas. International exchange suffixes are supported.")
        elif market == "Original 60 holdings":
            st.caption("Your original static stock list. Marsh uses its current MRSH symbol, formerly MMC.")
        elif market == "Upload CSV":
            uploaded = st.file_uploader("Adjusted prices", type=["csv"], key="prices_upload")
            st.caption("Use Date as the first column, then one asset per column. Prices must be complete and use a common currency.")
        elif market == "Demo":
            st.caption("Six simulated assets. Use this offline example to try the controls.")
        start, end = date(2020, 1, 1), date.today()
        if source == "Yahoo Finance":
            if history == "Custom dates":
                left, right = st.columns(2)
                start = left.date_input("From", start, key="start")
                end = right.date_input("Until (exclusive)", end, key="end")
            else:
                start = (pd.Timestamp(end) - pd.DateOffset(years=5 if history == "5 years" else 10)).date()
                st.caption(f"{start:%d %b %Y} to {end:%d %b %Y} · end date excluded")
        with st.expander("Fine-tune the model"):
            st.caption("The defaults work without changes. These controls change the assumptions, not your risk choice.")
            left, right = st.columns(2)
            use_cap = left.checkbox("Limit each holding", value=False, key="use_cap")
            cap = right.number_input("Maximum holding (%)", 0.0, 100.0, 40.0, 1.0, key="cap", disabled=not use_cap,
                                     help="A 25% cap needs at least four assets. The limit applies when weights are set. Prices can change weights later.")
            risk_free = left.number_input("Risk-free rate (%)", -10.0, 100.0, 2.0, .25, key="risk_free",
                                          help="Annual comparison rate for Sharpe and Sortino. It affects maximum-Sharpe weights and does not add cash.")
            train_pct = right.slider("Initial training data (%)", 50, 90, 70, 5, key="train_pct",
                                     help="The earliest prices fit the initial model. Later prices evaluate it. This split does not change the latest holdings.")
            shrink = st.slider("Covariance shrinkage (%)", 0, 100, 10, 5, key="shrink",
                               help="Reduce reliance on noisy relationships between stocks. 0% keeps sample correlations. 100% ignores correlations and keeps each stock's variance.")
            st.caption("Covariance measures how returns move together. Shrinkage reduces these estimated relationships while it keeps each stock's variance. It can reduce sensitivity to noise.")
            periods_per_year = 252.0
            if source == "Upload CSV":
                frequencies = {"Daily market prices": 252, "Daily calendar prices": 365, "Weekly": 52, "Monthly": 12, "Custom": None}
                frequency = st.selectbox("Price frequency", list(frequencies), key="frequency")
                periods_per_year = frequencies[frequency]
                if periods_per_year is None:
                    periods_per_year = st.number_input("Observations per year", min_value=1.0, value=252.0, key="periods_per_year")
                st.caption("Frequency scales annual figures. It does not resample prices.")
            price_currency = st.selectbox("Price currency", ["USD", "Other currency"], key="price_currency",
                                          help="Declare the currency of every input price. The app does not verify currencies or perform conversions.")
            compare_market = st.checkbox("Include S&P 500 and Nasdaq-100 benchmarks", True, key="compare_market", disabled=market == "Demo")
            download_for_csv = False
            if source == "Upload CSV":
                benchmark_upload = st.file_uploader("Benchmark prices (optional)", type=["csv"], key="benchmark_upload",
                                                    help="Use Date,SPY,QQQ on exactly the same dates as your asset file.")
                download_for_csv = st.checkbox("Download Yahoo benchmarks for this CSV", False, key="download_benchmarks")
                st.caption("CSV analysis stays offline unless you select this download. Both SPY and QQQ columns in your asset file also work.")
            st.caption("SPY and QQQ comparisons require USD prices. No currency conversion occurs.")
        submitted = st.button("Update portfolios" if ready else "Find portfolios", type="primary", width="stretch", key="build")

settings = dict(train_fraction=train_pct / 100, risk_free_rate=risk_free / 100,
                max_weight=cap / 100 if use_cap else 1.0, shrinkage=shrink / 100,
                periods_per_year=periods_per_year)
input_choices = {
    "tickers": ticker_text.strip(), "download_benchmarks": download_for_csv,
    **{name: (upload.name, sha256(upload.getvalue()).hexdigest()) if upload is not None else None
       for name, upload in (("prices", uploaded), ("benchmarks", benchmark_upload))},
}
if submitted:
    for key in ("result", "latest_profiles", "backtests", "benchmarks", "evidence", "benchmark_prices", "report_bytes", "coverage"):
        st.session_state.pop(key, None)
    try:
        with st.status("Prepare your portfolios", expanded=True) as progress_status:
            metadata = {"source": source, "universe": market, "price_currency": price_currency,
                        "compare_market": compare_market, **settings}
            metadata["currency_assumption"] = "The user declares a common price currency. No verification or currency conversion occurs."
            universe_benchmarks = None
            if source == "Yahoo Finance":
                metadata.update(requested_start=start.isoformat(), requested_end_exclusive=end.isoformat())
                if market == "Nasdaq-100":
                    loaded = fetch_universe(start.isoformat(), end.isoformat())
                    st.session_state.coverage = loaded.coverage
                    prices, universe_benchmarks = loaded.prices, loaded.benchmarks
                    metadata.update(loaded.metadata)
                    metadata["universe"] = market
                    if len(prices.columns) < 2:
                        raise ValueError("Fewer than two Nasdaq-100 securities have complete prices. Review coverage below or choose a shorter history.")
                else:
                    tickers = original_tickers(current_symbols=True) if market == "Original 60 holdings" else parse_tickers(ticker_text)
                    prices = fetch_prices(tickers, start.isoformat(), end.isoformat())
            elif market == "Demo":
                prices = demo_prices()
            else:
                if uploaded is None:
                    raise ValueError("Choose an adjusted-price CSV first.")
                prices = load_csv(BytesIO(uploaded.getvalue()))
                metadata["input_file"] = uploaded.name
            st.write(f"Find portfolio combinations across {len(prices.columns)} assets")
            result = calculate(prices, settings)
            latest = calculate_latest(prices, settings) if len(prices) >= 7 else None
            benchmark_prices = benchmarks = evidence = None
            if market == "Demo":
                metadata["benchmarks_note"] = "Synthetic prices have no real-market benchmark comparison."
            elif not compare_market:
                metadata["benchmarks_note"] = "Market comparisons are off in Market & settings."
            elif price_currency != "USD":
                metadata["benchmarks_note"] = "Market comparisons require USD prices. No currency conversion occurs."
            else:
                try:
                    if benchmark_upload is not None:
                        benchmark_prices = validate_benchmark_prices(load_csv(BytesIO(benchmark_upload.getvalue())), prices.index)
                        metadata["benchmark_source"] = "CSV: " + benchmark_upload.name
                    elif download_for_csv:
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
                        metadata["benchmarks_note"] = "Add a benchmark CSV, include SPY and QQQ columns, or enable a Yahoo benchmark download in Market & settings."
                    if benchmark_prices is not None:
                        benchmarks, evidence, note = calculate_market(benchmark_prices, prices, result, latest, settings)
                        if note:
                            metadata["evidence_note"] = note
                except (ValueError, RuntimeError, OSError) as exc:
                    metadata["benchmarks_note"] = f"Market comparison unavailable: {exc}"
                    benchmark_prices = None
            st.session_state.result = (result, prices, metadata)
            st.session_state.applied_inputs = input_choices
            st.session_state.latest_profiles = latest
            st.session_state.benchmark_prices = benchmark_prices
            st.session_state.benchmarks = benchmarks
            st.session_state.evidence = evidence
            st.session_state.view = "Portfolio"
            st.session_state.scroll_to_start = True
            progress_status.update(label="Your portfolios are ready", state="complete", expanded=False)
        st.rerun()
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))

if "result" not in st.session_state:
    if st.session_state.get("coverage") is not None:
        st.dataframe(st.session_state.coverage, hide_index=True, width="stretch")
    else:
        st.subheader("Start with a market. Find your balance.")
        st.write("Explore the Nasdaq-100, choose a relative risk level, and see the resulting stock weights.")
        for column, title, body in zip(st.columns(3), ("1 · Choose", "2 · Understand", "3 · Compare"), (
            "Start with the full index or bring your own prices.",
            "See which stocks the model holds and why the risk levels differ.",
            "Test the allocation rules against the S&P 500 and Nasdaq-100.",
        )):
            with column:
                st.markdown(f"**{title}**")
                st.caption(body)
    st.caption("Historical research. Model estimates do not predict future returns.")
    st.stop()

result, prices, metadata = st.session_state.result
latest = st.session_state.get("latest_profiles")
study = st.session_state.get("backtests")
benchmarks = st.session_state.get("benchmarks")
evidence = st.session_state.get("evidence")
applied_settings = {key: metadata[key] for key in settings}
if (metadata["universe"] != market or st.session_state.get("applied_inputs") != input_choices
        or any(metadata[key] != value for key, value in settings.items())
        or metadata["price_currency"] != price_currency or metadata["compare_market"] != compare_market
        or source == "Yahoo Finance" and (metadata.get("requested_start") != start.isoformat()
                                          or metadata.get("requested_end_exclusive") != end.isoformat())):
    st.info("Your setup differs from this result. Select Update portfolios to apply it.")
st.caption(f"{metadata['universe']} · {len(prices.columns)} assets analyzed · {prices.index[0]:%d %b %Y}–{prices.index[-1]:%d %b %Y}")
coverage = st.session_state.get("coverage")
if coverage is not None:
    excluded = int((coverage["status"] == "excluded").sum())
    st.caption(f"{len(coverage)} current index securities checked · {excluded} excluded for incomplete or unavailable prices. See Research → Data & coverage.")
if "synthetic" in metadata["source"]:
    st.info("Demo · synthetic prices. These are not market results.")

st.session_state.setdefault("view", "Portfolio")
view = st.segmented_control("View", ["Portfolio", "Compare", "Research"], required=True,
                            key="view", label_visibility="collapsed", width="stretch")
profile = st.segmented_control("Risk level", ["Low", "Medium", "Extreme"], default="Medium", required=True,
                               format_func=lambda value: "Highest" if value == "Extreme" else value,
                               key="risk_profile", help="Relative positions on this market's frontier. Low is minimum modeled risk. Highest is the maximum weakest-window return target.")

if view == "Portfolio":
    if latest is None:
        st.info("Latest risk profiles need at least seven complete price rows. Research still contains the original frontier and holdout.")
    else:
        selected = latest.portfolios[profile]
        label = "Highest" if profile == "Extreme" else profile
        st.subheader(f"Your {label.lower()} risk portfolio")
        descriptions = {"Low": "The smallest estimated volatility for this market and these settings.",
                        "Medium": "The middle return target between the low and highest risk profiles.",
                        "Extreme": "The highest achievable weakest-window return target, with the smallest variance among tied solutions."}
        st.caption(descriptions[profile])
        columns = st.columns(3)
        columns[0].metric("Estimated annual risk", f"{selected.volatility:.1%}", help="Annualized volatility from the full input history. It does not measure every form of risk.")
        columns[1].metric("Weakest historical window", f"{latest.summary.loc[profile, 'worst_window_return']:.1%}", help="The lowest annual arithmetic mean across three historical windows. It is an estimate, not a guaranteed return.")
        columns[2].metric("Largest holding", f"{selected.weights.max():.1%}", help="Enable a holding limit in Market & settings to reduce target concentration.")
        allocations = selected.weights.sort_values(ascending=False)
        chart_weights = allocations[allocations >= .0005].head(10).copy()
        remainder = float(allocations.sum() - chart_weights.sum())
        if remainder >= .0005:
            chart_weights.loc["Other holdings"] = remainder
        chart_weights = chart_weights.sort_values()
        figure = px.bar(x=chart_weights.values, y=chart_weights.index, orientation="h", text=chart_weights.values,
                        color_discrete_sequence=[COLORS[profile]])
        figure.update_traces(texttemplate="%{x:.1%}", textposition="auto", hovertemplate="%{y}<br>Allocation: %{x:.2%}<extra></extra>")
        style_chart(figure, "Where the portfolio invests", height=max(330, 110 + 29 * len(chart_weights)))
        figure.update_xaxes(title=None, tickformat=".0%")
        figure.update_yaxes(title=None)
        figure.update_layout(margin=dict(l=15, r=25, t=50, b=35), showlegend=False)
        st.plotly_chart(figure, width="stretch", theme=None)
        st.caption("The chart shows the largest allocations. Every holding below includes small positions and the complete portfolio.")
        with st.expander("Every holding"):
            holdings = allocations.rename("Weight").rename_axis("Symbol").reset_index()
            if coverage is not None:
                holdings = holdings.merge(coverage[["symbol", "name"]], left_on="Symbol", right_on="symbol", how="left").drop(columns="symbol").rename(columns={"name": "Company"})
                holdings = holdings[["Symbol", "Company", "Weight"]]
            st.dataframe(holdings.style.format({"Weight": "{:.2%}"}), hide_index=True, width="stretch")
            st.caption("Every input asset appears here. Each portfolio sums to 100%. Tiny weights can round to zero.")
        st.caption(f"Fit through {latest.as_of}. These weights use the full selected history. Use Compare to evaluate allocation rules on later historical prices.")
        st.button("Compare with the market", on_click=show_compare, type="primary", key="go_compare")
        with st.expander("Why these holdings?"):
            st.write("The model compares weighted combinations of all eligible stocks. It divides history into three consecutive windows and favors a stronger weakest-window mean.")
            st.write("Each risk level selects a different target on that frontier. The optimizer finds the least volatile allocation at the target.")
            st.write("Low is relative to the selected stocks. Highest uses no leverage and does not maximize every possible measure of risk.")
            st.dataframe(latest.window_returns.style.format("{:.2%}"), width="stretch")
            st.dataframe(latest.windows, hide_index=True, width="stretch")
        if latest.warnings:
            with st.expander("Model notes"):
                for note in latest.warnings:
                    st.write(note)

elif view == "Compare":
    st.subheader("How did the rules perform?")
    st.caption("Rebuild portfolios from prices available before each trade. Compare the result with passive SPY and QQQ holdings.")
    options = st.session_state.get("backtest_options", {"rebalance_every": 21, "rolling_window": 0, "cost_bps": 10.0})
    with st.expander("Comparison settings", expanded=study is None):
        with st.form("comparison_settings"):
            left, right = st.columns(2)
            rebalance_every = left.number_input("Observations between trades", 1, value=options["rebalance_every"], key="rebalance_every")
            cost_bps = right.number_input("Trading cost (basis points)", 0.0, 9999.0, options["cost_bps"], key="cost_bps",
                                          help="10 basis points means 0.1% of each amount bought or sold. Entry is included. No final sale is charged.")
            rolling_window = st.number_input("Rolling history (0 = initial training length)", 0, value=options["rolling_window"], key="rolling_window")
            st.caption("All four methods share dates and fee rates. A trade uses the prior close's information and executes at the next close.")
            run_comparison = st.form_submit_button("Run comparison" if study is None else "Update comparison", type="primary", key="run_comparison")
    if run_comparison:
        for key in ("backtests", "benchmarks", "evidence", "report_bytes"):
            st.session_state.pop(key, None)
        study = benchmarks = evidence = None
        metadata.pop("evidence_note", None)
        try:
            with st.spinner("Test the portfolio rules across the selected history…"):
                profile_backtests = min(len(result.train_returns), rolling_window or len(result.train_returns)) >= 6
                options = dict(rebalance_every=rebalance_every, rolling_window=rolling_window, cost_bps=cost_bps)
                study = compare_backtests(prices, applied_settings, dict(**{**options, "rolling_window": rolling_window or None}, include_profiles=profile_backtests))
                if not profile_backtests:
                    study.warnings.append("Risk-profile tests need six returns in each fit window. The original targets remain available.")
                st.session_state.backtest_options = options
                st.session_state.backtests = study
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
        if study is not None and st.session_state.get("benchmark_prices") is not None:
            try:
                benchmarks, evidence, note = calculate_market(st.session_state.benchmark_prices, prices, result, latest, applied_settings, study)
                st.session_state.benchmarks, st.session_state.evidence = benchmarks, evidence
                metadata.pop("benchmarks_note", None)
                if note:
                    metadata["evidence_note"] = note
            except (ValueError, RuntimeError, OSError) as exc:
                metadata["benchmarks_note"] = f"Market comparison unavailable: {exc}"
        if study is not None:
            st.rerun()
    if study is None:
        st.write("Run the comparison to see growth, losses, and evidence against the benchmarks. Your selected risk level stays the same.")
    else:
        method = st.selectbox("Allocation rule", ["Expanding window", "Rolling window", "Fixed rebalance", "Buy and hold"], key="comparison_method")
        selected = f"{method} · {profile}"
        if selected not in study.equity:
            selected = f"{method} · Minimum volatility"
            st.info("This sample is too short for the risk profiles. The comparison shows minimum volatility.")
        selected_label = selected.replace(" · Extreme", " · Highest")
        st.caption(f"{selected_label} · {study.settings['test_start']}–{study.settings['test_end']} · after {study.settings['cost_bps']:g} basis-point trading fees")
        matched_benchmarks = benchmarks if benchmarks is not None and benchmarks.backtest_equity is not None else None
        columns = st.columns(3 if matched_benchmarks is not None else 1)
        columns[0].metric("Portfolio annual growth", f"{study.metrics.loc[selected, 'cagr']:.1%}")
        if matched_benchmarks is not None:
            for column, (name, row) in zip(columns[1:], matched_benchmarks.backtest_metrics.iterrows()):
                column.metric(name, f"{row['cagr']:.1%}")
        else:
            st.info(metadata.get("benchmarks_note", "Market benchmarks are unavailable."))
        chart_mode = st.segmented_control("Chart", ["Growth", "Drawdown", "Relative to markets"], default="Growth", required=True, key="comparison_chart")
        benchmark_equity = matched_benchmarks.backtest_equity if matched_benchmarks is not None else None
        if chart_mode == "Relative to markets" and matched_benchmarks is not None and evidence is not None:
            figure = evidence_chart(evidence, selected)
        else:
            if chart_mode == "Relative to markets":
                st.caption("Relative performance needs market benchmarks. Growth is shown below.")
            figure = backtest_chart(study, strategies=[selected], drawdown=chart_mode == "Drawdown", benchmark_equity=benchmark_equity)
            figure.update_traces(name=selected_label, selector={"name": selected})
        st.plotly_chart(figure, width="stretch", theme=None)
        st.caption("All paths start at the same close, stay in cash for one interval, then enter. Both benchmarks pay the same entry fee rate.")
        if matched_benchmarks is not None and evidence is not None:
            st.subheader("Is there evidence of an advantage?")
            rows = evidence.summary.xs(selected, level="strategy")
            for column, (name, row) in zip(st.columns(2), rows.iterrows()):
                with column:
                    st.metric(f"Annual growth vs {name}", f"{row['cagr_difference'] * 100:+.2f} pp")
                    st.caption(str(row["status"]))
            st.caption("A historical lead does not establish future outperformance. The tests allow for return dependence and multiple comparisons.")
            with st.expander("Consistency and uncertainty"):
                windows = evidence.windows.loc[evidence.windows.strategy == selected].drop(columns="strategy")
                st.dataframe(windows.style.format({"relative_return": "{:+.2%}", "annual_advantage": lambda value: f"{value * 100:+.2f} pp"}), hide_index=True, width="stretch")
                st.caption("Three consecutive evaluation windows. They differ from the windows used to fit today's holdings.")
                st.dataframe(evidence_summary_frame(evidence).xs(selected, level="strategy"), width="stretch")
                st.write("Mean advantage is an arithmetic return difference. Annual growth compounds the full path. Alpha adjusts for exposure to one benchmark and does not establish skill.")
                st.write("Individual 95% intervals use Newey–West errors. Holm adjustment covers every strategy, both benchmarks, and both tested measures.")
                st.write("A p-value is not the probability of luck or future success. Short samples retain descriptive results without inference.")
                for note in evidence.warnings:
                    st.caption(note)
        elif metadata.get("evidence_note"):
            st.info(metadata["evidence_note"])
        with st.expander("Returns, losses, and trading costs"):
            comparison = study.metrics.loc[[selected]]
            if matched_benchmarks is not None:
                comparison = pd.concat([comparison, matched_benchmarks.backtest_metrics])
            formats = {name: "{:.0f}" if name.endswith("count") else "{:.2f}" if name in ("sharpe", "sortino", "calmar", "total_turnover") else "{:.2%}" for name in comparison}
            st.dataframe(comparison.style.format(formats, na_rep="—"), column_config={name: label for name, (label, _) in METRICS.items()}, width="stretch")
        if study.warnings:
            with st.expander("Comparison notes"):
                for note in study.warnings:
                    st.write(note)

else:
    topic = st.selectbox("Research view", ["Efficient frontiers", "Risk breakdown", "Data & coverage", "All backtests", "Methodology"], key="research_topic")
    if topic == "Efficient frontiers":
        if latest is not None:
            st.plotly_chart(latest_profile_chart(latest, benchmarks.latest_estimates if benchmarks is not None else None), width="stretch", theme=None)
            st.caption("Full-history fit. The vertical axis is the lowest annual arithmetic mean across three windows. It is not a forecast.")
            with st.expander("All risk-profile allocations"):
                st.dataframe(weights_frame(latest).style.format("{:.2%}"), width="stretch")
        with st.expander("Original mean-return frontier"):
            st.plotly_chart(frontier_chart(result, benchmarks.training_estimates if benchmarks is not None else None), width="stretch", theme=None)
            st.dataframe(weights_frame(result).style.format("{:.2%}"), width="stretch")
            st.caption("This model uses the initial training period. Passive benchmark markers use the same dates and do not follow your holding limit.")
        with st.expander("Original holdout · no trading fees"):
            st.plotly_chart(holdout_chart(result, benchmarks.holdout_equity if benchmarks is not None else None), width="stretch", theme=None)
            metrics = result.holdout_metrics if benchmarks is None else pd.concat([result.holdout_metrics, benchmarks.holdout_metrics])
            st.dataframe(metrics.style.format({name: "{:.2f}" if name in ("sharpe", "sortino", "calmar") else "{:.2%}" for name in metrics}, na_rep="—"), width="stretch")
            st.caption("Allocate once at the split close, then let weights drift. This legacy check excludes fees and uses a different entry date from Compare.")
    elif topic == "Risk breakdown":
        st.caption("These diagnostics use the original training-period portfolios and covariance.")
        st.dataframe(risk_summary(result).style.format({"max_weight": "{:.2%}", "effective_holdings": "{:.2f}", "diversification_ratio": "{:.2f}"}, na_rep="—"), width="stretch")
        risk_portfolio = st.selectbox("Portfolio", list(result.portfolios), key="risk_portfolio")
        frame = pd.DataFrame({"Allocation weight": result.portfolios[risk_portfolio].weights,
                              "Share of portfolio variance": risk_contributions(result)[risk_portfolio]}).sort_values("Allocation weight", ascending=False)
        figure = px.bar(frame.head(20).rename_axis("Asset").reset_index(), y="Asset", x=list(frame.columns), orientation="h", barmode="group", color_discrete_sequence=["#175bc0", "#a9b4c4"])
        style_chart(figure, "Allocation and risk", height=max(380, 110 + 30 * min(20, len(frame))))
        figure.update_xaxes(title=None, tickformat=".0%")
        figure.update_yaxes(title=None, autorange="reversed")
        st.plotly_chart(figure, width="stretch", theme=None)
        st.dataframe(frame.style.format("{:.2%}", na_rep="—"), width="stretch")
        st.caption("The chart shows up to 20 assets. The table includes every asset. Variance shares can be negative. Zero portfolio variance gives undefined shares.")
    elif topic == "Data & coverage":
        st.subheader("The data behind this result")
        if coverage is not None:
            st.dataframe(coverage, hide_index=True, width="stretch")
            st.caption(metadata.get("universe_limitation", "Current Nasdaq-100 membership and complete-history requirements can favor surviving stocks. This is not historical index membership."))
        st.dataframe(pd.DataFrame([{"Setting": key, "Value": str(value)} for key, value in metadata.items() if key != "universe_coverage"]), hide_index=True, width="stretch")
        with st.expander("Prices"):
            st.dataframe(prices, width="stretch", height=300)
            st.download_button("Download prices", csv_text(prices, index_label="Date"), "prices.csv", "text/csv")
        with st.expander("Correlations"):
            chart_assets = st.multiselect("Assets in this chart", list(prices), default=list(prices)[:15], key="correlation_assets")
            if chart_assets:
                figure = px.imshow(result.train_returns[chart_assets].corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
                style_chart(figure, "Training correlations")
                st.plotly_chart(figure, width="stretch", theme=None)
            st.caption("This chart selection does not change the portfolio universe.")
    elif topic == "All backtests":
        if study is None:
            st.info("Run the comparison first to inspect every strategy and its trades.")
            st.button("Go to Compare", on_click=show_compare)
        else:
            strategies = st.multiselect("Strategies in this chart", list(study.equity), default=[name for name in study.equity if name.startswith("Expanding window") and name.split(" · ")[-1] in ("Low", "Medium", "Extreme")], key="chart_strategies")
            if strategies:
                st.plotly_chart(backtest_chart(study, strategies=strategies, benchmark_equity=benchmarks.backtest_equity if benchmarks is not None else None), width="stretch", theme=None)
            st.caption("Chart selections preserve every result and the full statistical test family.")
            st.dataframe(study.metrics, width="stretch")
            selected = st.selectbox("Inspect holdings and trades", list(study.equity), key="holding_strategy")
            st.dataframe(study.holdings[selected].style.format("{:.2%}"), column_config={name: label for name, (label, _) in HOLDINGS.items()}, width="stretch")
            st.caption("Holding profit/loss contributions use initial capital. Their sum less fees equals the strategy's net total return.")
            with st.expander("Target weights and trades"):
                st.dataframe(study.allocations[selected].style.format("{:.2%}"), width="stretch")
                st.dataframe(study.trades[selected], width="stretch")
            if evidence is not None:
                with st.expander("Every statistical comparison"):
                    st.dataframe(evidence_summary_frame(evidence), width="stretch")
    else:
        st.markdown("""
### What the model does
The model chooses weights across all eligible assets. Weights are nonnegative and sum to 100%.
The latest holdings use all selected history. Three chronological windows supply annual arithmetic return estimates.
The model favors a stronger weakest window and minimizes volatility at each return target.

### What the risk choices mean
**Low** selects minimum modeled volatility. **Medium** selects the middle weakest-window return target.
**Highest** selects the maximum target, with minimum variance among tied solutions. Exports call this profile **Extreme**.
The labels are relative to the selected market. They are not absolute risk limits or guarantees.

### Covariance shrinkage
Covariance describes how stock returns move together. These estimates can be noisy.
At 0%, the model uses sample covariance. At 100%, it ignores correlations and keeps individual stock variances.
The default 10% reduces cross-stock covariance by 10%. This manual setting is not automatic Ledoit–Wolf estimation.

### Historical comparison
**Buy and hold** allocates once. **Fixed rebalance** restores the initial targets.
**Expanding window** refits from all prior returns. **Rolling window** uses a fixed recent history.
Each trade uses prior-close information and executes at the next close. The first interval stays in cash.
Fees apply to each amount bought or sold. There is no final sale. Taxes and currency conversion are excluded.

### How to read the evidence
SPY and QQQ are ETF proxies, not raw index series. Their paths share the portfolio's dates and fee rates.
The efficient frontier does not promise to beat either benchmark. Confidence intervals describe historical uncertainty.
Holm adjustment covers all tests in this report, but it cannot correct unknown earlier strategy searches.
Current constituents, complete-history requirements, and repeated changes after review can distort a backtest.
Save the rules and report, then evaluate later data that you have not inspected.
""")
        for note in result.warnings:
            st.write(note)

st.divider()
with st.popover("Export your research"):
    st.caption("Save an offline report, exact inputs, all allocations, and any completed comparisons.")
    if st.button("Prepare report", key="prepare_report"):
        with st.spinner("Prepare your report…"):
            st.session_state.report_bytes = export_report(result, prices, metadata, study, latest, benchmarks, evidence)
    if "report_bytes" in st.session_state:
        st.download_button("Download report and data", st.session_state.report_bytes, "portfolio-lab.zip", "application/zip", type="primary")
st.caption("Historical research · Relative risk levels · Future performance remains unknown")
scroll_to_start = st.session_state.pop("scroll_to_start", False)
if st.session_state.get("previous_view") != view or scroll_to_start:
    st.html("""<script>
document.querySelector('[data-testid="stMain"]')?.scrollTo({top: 0, behavior: 'instant'});
</script>""", unsafe_allow_javascript=True)
st.session_state.previous_view = view
