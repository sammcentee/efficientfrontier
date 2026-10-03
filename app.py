"""Run with ./run.sh from the repository root."""

from datetime import date
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from efficient_frontier.core import analyze
from efficient_frontier.data import demo_prices, download_prices, load_csv, original_tickers, parse_tickers
from efficient_frontier.presentation import frontier_chart, holdout_chart, report_zip, weights_frame


st.set_page_config(page_title="Efficient Frontier · Portfolio Lab", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1440px; padding-top: 2.7rem;}
h1 {letter-spacing: -0.045em; font-weight: 700 !important;}
[data-testid="stMetric"] {background: #131e30; border: 1px solid #24334a;
  border-radius: 12px; padding: 16px 20px;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
[data-testid="stSidebar"] {border-right: 1px solid #24334a;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, start, end):
    return download_prices(tickers, start, end)


@st.cache_data(show_spinner=False)
def calculate(prices, settings):
    return analyze(prices, **settings)


@st.cache_data(show_spinner=False)
def export_report(analysis, prices, metadata):
    return report_zip(analysis, prices, metadata)


with st.sidebar:
    st.markdown("### ◈ Portfolio Lab")
    st.caption("Your efficient frontier, rebuilt.")
    source = st.selectbox("Price data", ["Demo · synthetic", "Yahoo Finance", "Upload CSV"], key="source")
    universe = "Custom tickers"
    if source == "Yahoo Finance":
        universe = st.radio("Universe", ["Custom tickers", "Original 60 holdings"])
    with st.form("analysis_settings"):
        if source == "Yahoo Finance":
            if universe == "Original 60 holdings":
                tickers = original_tickers()
                st.caption(f"{len(tickers)} tickers from your original spreadsheet. This is a static list, not historical index membership.")
            else:
                ticker_text = st.text_area("Tickers", "SPY, QQQ, IWM, EFA, TLT, GLD", height=90)
            start = st.date_input("Start date", date(2020, 1, 1))
            end = st.date_input("End date (exclusive)", date.today())
        elif source == "Upload CSV":
            uploaded = st.file_uploader("Adjusted daily prices", type=["csv"])
            st.caption("Date in the first column; one asset per remaining column. Use a common currency and complete daily observations.")
        else:
            st.caption("Six simulated assets. Repeatable data for exploring the app; no market performance claims.")
        st.markdown("**Portfolio constraints**")
        cap = st.slider("Maximum weight per asset", 5, 100, 40, step=5, format="%d%%")
        risk_free = st.number_input("Annual risk-free rate (%)", min_value=-10.0, max_value=100.0, value=2.0, step=0.25)
        train_pct = st.slider("Data used for training", 50, 90, 70, step=5, format="%d%%")
        shrink = st.slider("Covariance shrinkage", 0, 100, 10, step=5, format="%d%%", help="Blend the training covariance toward its diagonal. 0% uses the sample covariance; 100% removes estimated correlations.")
        submitted = st.form_submit_button("Build frontier", type="primary", width="stretch")
    st.caption("Long-only · Fully invested · No leverage")
    st.divider()
    st.caption("Historical research. Estimated returns are not forecasts. Costs, taxes and currency conversion are excluded.")

st.caption("PORTFOLIO RESEARCH / EFFICIENT FRONTIER")
st.title("Find the balance.")
st.markdown("Explore the trade-off between risk and return, then test your allocations on the data held aside.")

settings = dict(train_fraction=train_pct / 100, risk_free_rate=risk_free / 100,
                max_weight=cap / 100, shrinkage=shrink / 100)
if submitted or ("result" not in st.session_state and source == "Demo · synthetic"):
    st.session_state.pop("result", None)
    try:
        with st.spinner("Preparing prices and solving the frontier…"):
            if source == "Demo · synthetic":
                prices = demo_prices()
            elif source == "Yahoo Finance":
                if universe == "Custom tickers":
                    tickers = parse_tickers(ticker_text)
                prices = fetch_prices(tickers, start.isoformat(), end.isoformat())
            else:
                if uploaded is None:
                    raise ValueError("Choose a CSV of adjusted daily prices before building the frontier.")
                prices = load_csv(BytesIO(uploaded.getvalue()))
            result = calculate(prices, settings)
            metadata = {"source": source, **settings}
            if source == "Yahoo Finance":
                metadata.update(requested_start=start.isoformat(), requested_end_exclusive=end.isoformat(), universe=universe)
            st.session_state.result = (result, prices, metadata)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

if "result" not in st.session_state or st.session_state.result[2]["source"] != source:
    st.info("Choose your data and settings, then select Build frontier.")
    st.stop()

result, prices, metadata = st.session_state.result
if "synthetic" in metadata["source"]:
    st.info("DEMO DATA · These prices are synthetic. Use Yahoo Finance or upload your own adjusted prices to research real assets.")
else:
    st.caption(f"{metadata['source']} · {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y} · adjusted daily prices")
for warning in result.warnings:
    st.warning(warning)

cols = st.columns(4)
cols[0].metric("Assets", str(len(prices.columns)))
cols[1].metric("Training sessions", f"{len(result.train_returns):,}")
cols[2].metric("Holdout sessions", f"{len(result.test_returns):,}")
cols[3].metric("Maximum starting weight", f"{metadata['max_weight']:.0%}")
st.caption(f"Train: {result.train_returns.index[0]:%d %b %Y} – {result.train_returns.index[-1]:%d %b %Y}  ·  Holdout: {result.test_returns.index[0]:%d %b %Y} – {result.test_returns.index[-1]:%d %b %Y}")

frontier_tab, holdout_tab, data_tab = st.tabs(["Efficient frontier", "Holdout performance", "Data & methodology"])
with frontier_tab:
    st.plotly_chart(frontier_chart(result), width="stretch", theme=None)
    st.caption("The curve shows minimum estimated volatility at each target return, under your weight limit. All estimates use the training period only.")
    comparison = pd.DataFrame({name: {"Estimated annual return": p.expected_return,
                                    "Annual volatility": p.volatility, "Sharpe ratio": p.sharpe}
                               for name, p in result.portfolios.items()}).T
    st.dataframe(comparison.style.format({"Estimated annual return": "{:.2%}", "Annual volatility": "{:.2%}", "Sharpe ratio": "{:.2f}"}), width="stretch")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Portfolio allocations")
        st.dataframe(weights_frame(result).style.format("{:.2%}"), width="stretch")
    with right:
        st.subheader("Explore the curve")
        point = st.select_slider("Target-return point", options=list(range(len(result.frontier))),
                                 format_func=lambda i: f"{result.frontier.iloc[i]['expected_return']:.2%} estimated annual return")
        frontier_weights = result.frontier_weights.iloc[point].sort_values(ascending=False)
        st.dataframe(frontier_weights.rename("Weight").to_frame().style.format("{:.2%}"), width="stretch")

with holdout_tab:
    st.subheader("What happened after training?")
    st.markdown("Allocate once at the end of training, then hold through the later period without trading between assets. Adjusted-price growth reflects the data provider's treatment of distributions; weights drift. The equal-weight portfolio follows the same rule.")
    st.plotly_chart(holdout_chart(result), width="stretch", theme=None)
    metrics = result.holdout_metrics.rename(columns={"total_return": "Total return", "cagr": "Annualized growth", "volatility": "Annual volatility", "sharpe": "Sharpe ratio", "max_drawdown": "Max drawdown"})
    st.dataframe(metrics.style.format({col: "{:.2f}" if col == "Sharpe ratio" else "{:.2%}" for col in metrics.columns}), width="stretch")
    st.caption("The starting weight cap applies at allocation; later weights can exceed it as prices move. Annualized growth compounds realized returns, while frontier returns are arithmetic estimates. Repeatedly tuning settings against this holdout makes it less independent.")

with data_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Training correlations")
        fig = px.imshow(result.train_returns.corr(), zmin=-1, zmax=1,
                        color_continuous_scale="Tealrose", aspect="auto")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", height=430, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        st.subheader("How it works")
        st.markdown(f"""
- Daily simple returns from adjusted closing prices; no forward-filling or silent asset removal.
- The earliest **{metadata['train_fraction']:.0%}** of return observations estimates the portfolio. The remaining observations are held aside.
- Annual return = mean daily return × 252. Annual covariance = daily covariance × 252.
- Covariance blends **{metadata['shrinkage']:.0%}** toward its diagonal to temper estimated correlations.
- Convex optimization finds minimum-volatility portfolios and, when positive excess return is feasible, maximum Sharpe.
- Sharpe uses the **{metadata['risk_free_rate']:.2%}** annual risk-free assumption; cash is not an investable asset in this model.
- All weights are nonnegative, sum to 100%, and obey the starting position limit.
""")
        st.caption("Use daily observations in a common currency. The app cannot verify whether a CSV is adjusted, daily, or in a common currency. A fixed ticker list can introduce survivorship bias. Historical averages are sensitive to the chosen period.")
    st.subheader("Price observations")
    st.dataframe(prices, width="stretch", height=260)
    st.download_button("Download prices CSV", prices.to_csv(index_label="Date"), "prices.csv", "text/csv")

st.divider()
st.download_button("Download research report + CSVs", export_report(result, prices, metadata),
                   "efficient-frontier-report.zip", "application/zip", type="primary")
st.caption("Includes a standalone interactive HTML report, exact input prices, allocations, frontier points, holdout results and settings.")
