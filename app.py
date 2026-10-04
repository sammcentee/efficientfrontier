"""Run with ./run.sh from the repository root."""

from datetime import date
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from efficient_frontier.core import analyze
from efficient_frontier.backtest import run_backtests
from efficient_frontier.backtest_report import HOLDINGS, METRICS, backtest_chart, findings
from efficient_frontier.data import demo_prices, download_prices, load_csv, original_tickers, parse_tickers
from efficient_frontier.presentation import csv_text, frontier_chart, holdout_chart, latest_profile_chart, report_zip, weights_frame
from efficient_frontier.profiles import build_profiles
from efficient_frontier.risk import risk_contributions, risk_summary


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


@st.cache_data(show_spinner=False, max_entries=2)
def calculate(prices, settings):
    return analyze(prices, **settings)


@st.cache_data(show_spinner=False, max_entries=2)
def compare_backtests(prices, settings, backtest_settings):
    return run_backtests(prices, **settings, **backtest_settings)


@st.cache_data(show_spinner=False, max_entries=2)
def calculate_latest(prices, settings):
    return build_profiles(prices, **{key: value for key, value in settings.items() if key != "train_fraction"})


@st.cache_data(show_spinner=False, max_entries=2)
def export_report(analysis, prices, metadata, study, latest_profiles):
    return report_zip(analysis, prices, metadata, study, latest_profiles)


with st.sidebar:
    st.markdown("### ◈ Portfolio Lab")
    st.caption("Your efficient frontier, rebuilt.")
    source = st.selectbox("Price data", ["Yahoo Finance", "Demo · synthetic", "Upload CSV"], key="source")
    universe = "Custom tickers"
    if source == "Yahoo Finance":
        universe = st.radio("Universe", ["Custom tickers", "Original 60 holdings"])
    use_cap = st.toggle("Limit weight per asset", value=False, key="use_cap",
                        help="Limit concentration at each allocation. A fully invested portfolio needs enough assets to satisfy the cap. Weights can drift above it between trades.")
    with st.expander("Observation frequency"):
        frequencies = {"Trading days (252)": 252, "Calendar days (365)": 365,
                       "Weekly (52)": 52, "Monthly (12)": 12, "Custom": None}
        frequency = st.selectbox("Annualization basis", list(frequencies), key="observation_frequency")
        periods_per_year = frequencies[frequency]
        if periods_per_year is None:
            periods_per_year = st.number_input("Observations per year", min_value=1.0, value=252.0,
                                               step=1.0, key="periods_per_year")
        st.caption("This assumption applies to returns between supplied rows. It does not resample prices or convert currency.")
        if source != "Upload CSV":
            st.caption("Demo and Yahoo prices use daily observations. Weekly and monthly settings require data at those intervals.")
    with st.form("analysis_settings"):
        if source == "Yahoo Finance":
            if universe == "Original 60 holdings":
                tickers = original_tickers(current_symbols=True)
                st.caption(f"{len(tickers)} tickers from your original spreadsheet. This is a static list, not historical index membership.")
                st.caption("Marsh's renamed ticker is requested as MRSH (formerly MMC).")
            else:
                ticker_text = st.text_area("Tickers", "SPY, QQQ, IWM, EFA, TLT, GLD", height=90)
            start = st.date_input("Start date", date(2020, 1, 1))
            end = st.date_input("End date (exclusive)", date.today())
        elif source == "Upload CSV":
            uploaded = st.file_uploader("Adjusted prices", type=["csv"])
            st.caption("Use dates in the first column and one asset per subsequent column. Supply complete observations at consistent intervals in a common currency.")
        else:
            st.caption("Six simulated assets. Repeatable data for exploring the app; no market performance claims.")
        with st.expander("Portfolio settings", expanded=True):
            cap = st.number_input("Maximum weight per asset (%)", min_value=0.0, max_value=100.0,
                                  value=40.0, step=1.0, format="%.4f", key="cap",
                                  help="For example, a 25% cap needs at least four assets. The cap applies to target weights, not later weights after prices change.") if use_cap else 100.0
            risk_free = st.number_input("Annual risk-free rate (%)", min_value=-10.0, max_value=100.0, value=2.0, step=0.25, key="risk_free",
                                        help="Annual comparison rate for Sharpe and Sortino ratios. It also affects Maximum Sharpe allocations. It does not add cash to the portfolio.")
            train_pct = st.slider("Data used for training", 50, 90, 70, step=5, format="%d%%",
                                  help="The earliest observations estimate the initial portfolio. Later observations evaluate it. More training data leaves less data for evaluation.")
            shrink = st.slider("Covariance shrinkage", 0, 100, 10, step=5, format="%d%%",
                               help="Reduce reliance on noisy historical co-movement. 0% uses sample correlations. 100% ignores correlations but keeps individual asset variances. This is a manual setting, not automatic Ledoit-Wolf estimation.")
            st.caption("Open Portfolio settings explained on the main page for examples and trade-offs.")
        with st.expander("Backtest comparison"):
            include_backtests = st.checkbox("Compare four backtesting methods", value=True, key="include_backtests")
            rebalance_every = st.number_input("Sessions between trades", min_value=1, value=21, step=1,
                                             help="Count supplied observations between trades. This schedule applies to fixed, expanding, and rolling methods. Buy and hold allocates only once.")
            rolling_window = st.number_input("Rolling estimation sessions (0 = initial training length)", min_value=0, value=0, step=21,
                                            help="The rolling method estimates each new target from this many recent returns. A shorter window adapts faster but uses less evidence.")
            cost_bps = st.number_input("Trading cost (basis points per bought or sold unit)", min_value=0., max_value=9999., value=10., step=1., key="cost_bps",
                                       help="One basis point is 0.01%. A cost of 10 means 0.1% of each amount bought or sold. Both sides of a rebalance incur costs.")
            st.caption("Each session is one supplied observation. The schedule does not count calendar months.")
            st.caption("Targets use the previous close's data and trade at the next close. Repeated fits take longer for large universes.")
        submitted = st.form_submit_button("Build frontier", type="primary", width="stretch")
    st.caption("No ticker-count limit · Long-only · Fully invested")
    st.divider()
    st.caption("Historical research. Estimated returns are not forecasts. Backtests include the selected trading costs; taxes and currency conversion are excluded.")

st.caption("PORTFOLIO RESEARCH / EFFICIENT FRONTIER")
st.title("Find the balance.")
st.markdown("Find model holdings for low, medium, or extreme relative risk. Favor returns that hold up across several historical windows.")

with st.expander("Portfolio settings explained"):
    st.markdown("""
**Weight limit.** A cap limits each asset's target weight and controls concentration.
A 25% cap requires at least four assets for a fully invested portfolio.
Price changes can move actual weights above the cap between trades.

**Risk-free rate.** This annual comparison rate affects Sharpe and Sortino ratios and the Maximum Sharpe allocation.
It does not add cash or an interest-bearing asset to the portfolio.

**Training split.** The earliest return observations estimate the initial portfolio.
The later observations form the holdout, in date order.
More training data can support estimates, but leaves fewer observations for evaluation.
Repeated choices based on holdout results reduce its independence.
The separate Latest holdings view uses all supplied observations through the last price date.
The training split does not change that latest fit.

**Covariance shrinkage.** Covariance measures how asset returns move together.
Estimates from historical data can be noisy, especially with few observations or many assets.
The slider reduces off-diagonal covariance toward zero while it keeps each asset's individual variance.

At **0%**, the model uses sample correlations.
At **100%**, the model ignores correlations between assets but keeps individual variances.
Intermediate values reduce reliance on these estimated relationships.
Shrinkage can reduce dependence on noisy correlations, but it does not guarantee better results.
This is a manual choice, not automatic Ledoit-Wolf estimation.
""")

with st.expander("Backtest methods explained"):
    st.markdown("""
Each method evaluates allocations on the same later observations.
When history permits, each method includes Low, Medium, and Extreme profiles plus the three original portfolio targets.
Each profile fit divides its available history into three windows and favors the weakest window's estimated return.

| Method | What happens |
| --- | --- |
| Buy and hold | Allocate once, then let weights change with prices. |
| Fixed rebalance | Restore the original target weights at each scheduled trade. The estimates do not change. |
| Expanding window | Estimate new target weights from all prior observations at each scheduled trade. The sample grows over time. |
| Rolling window | Estimate new target weights from a fixed number of recent returns. Older observations leave the sample. |

**Trade timing.** Each target uses data through a previous close.
The trade executes at the next observation's close.
The first evaluation interval stays in cash without interest.
New weights do not earn the return from the interval before execution.

**Schedule and annualization.** A session is one supplied observation, not one calendar day or month.
The trade interval counts these observations.
The annualization assumption scales return and risk measures. It does not change observation dates or trade dates.

**Costs.** One basis point is 0.01%.
A setting of 10 basis points charges 0.1% of each amount bought or sold.
For example, a sale of 1,000 units plus a purchase of 1,000 units costs 2 units.
The comparison includes entry costs and excludes a final liquidation.
The model excludes taxes and currency conversion.

The separate Original holdout allocates at the split without costs.
Its entry timing differs from the four-method comparison.
These historical comparisons do not predict which method will perform best in the future.
""")

settings = dict(train_fraction=train_pct / 100, risk_free_rate=risk_free / 100,
                max_weight=cap / 100, shrinkage=shrink / 100, periods_per_year=periods_per_year)
if submitted or ("result" not in st.session_state and source == "Demo · synthetic"):
    st.session_state.pop("result", None)
    st.session_state.pop("backtests", None)
    st.session_state.pop("latest_profiles", None)
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
                    raise ValueError("Choose a CSV of adjusted prices before building the frontier.")
                prices = load_csv(BytesIO(uploaded.getvalue()))
            result = calculate(prices, settings)
            latest_profiles = calculate_latest(prices, settings) if len(prices) >= 7 else None
            profile_backtests = min(len(result.train_returns), rolling_window or len(result.train_returns)) >= 6
            study = compare_backtests(prices, settings, dict(rebalance_every=rebalance_every,
                                      rolling_window=rolling_window or None, cost_bps=cost_bps,
                                      include_profiles=profile_backtests)) if include_backtests else None
            if study is not None and not profile_backtests:
                study.warnings.append("Risk-profile backtests need at least six returns in both the initial and rolling fit windows. Only the original targets are shown.")
            metadata = {"source": source, "use_cap": use_cap, **settings}
            if source == "Yahoo Finance":
                metadata.update(requested_start=start.isoformat(), requested_end_exclusive=end.isoformat(), universe=universe)
            st.session_state.result = (result, prices, metadata)
            st.session_state.backtests = study
            st.session_state.latest_profiles = latest_profiles
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

if "result" not in st.session_state or st.session_state.result[2]["source"] != source:
    st.info("Choose your data and settings, then select Build frontier.")
    st.stop()

result, prices, metadata = st.session_state.result
study = st.session_state.get("backtests")
latest_profiles = st.session_state.get("latest_profiles")
if "synthetic" in metadata["source"]:
    st.info("DEMO DATA · These prices are synthetic. Use Yahoo Finance or upload your own adjusted prices to research real assets.")
else:
    st.caption(f"{metadata['source']} · {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y} · adjusted prices")
if (use_cap != metadata["use_cap"] or periods_per_year != metadata["periods_per_year"]
        or (source == "Yahoo Finance" and universe != metadata["universe"])):
    st.info("The sidebar has changes. Select Build frontier to apply them. Results below use the last completed run.")
applied_cap = "No position limit" if metadata["max_weight"] == 1 else f"Position limit {metadata['max_weight']:.2%}"
st.caption(f"Applied settings · {metadata.get('universe', metadata['source'])} · "
           f"{metadata['periods_per_year']:g} observations/year · Training {metadata['train_fraction']:.0%} · "
           f"Risk-free rate {metadata['risk_free_rate']:.2%} · Shrinkage {metadata['shrinkage']:.0%} · {applied_cap}")
if study is not None:
    st.caption(f"Applied backtest settings · Trade every {study.settings['rebalance_every']} observations · "
               f"Rolling window {study.settings['rolling_window']} returns · Costs {study.settings['cost_bps']:g} basis points")
for warning in result.warnings:
    st.warning(warning)

cols = st.columns(4)
cols[0].metric("Assets", str(len(prices.columns)))
cols[1].metric("Training observations", f"{len(result.train_returns):,}")
cols[2].metric("Holdout observations", f"{len(result.test_returns):,}")
cols[3].metric("Position weight limit", "None" if metadata["max_weight"] == 1 else f"{metadata['max_weight'] * 100:g}%")
st.caption(f"Train: {result.train_returns.index[0]:%d %b %Y} – {result.train_returns.index[-1]:%d %b %Y}  ·  Holdout: {result.test_returns.index[0]:%d %b %Y} – {result.test_returns.index[-1]:%d %b %Y}")

latest_tab, frontier_tab, risk_tab, backtest_tab, holdout_tab, data_tab = st.tabs(["Latest holdings", "Efficient frontier", "Portfolio risk", "Backtests & holdings", "Original holdout", "Data & methodology"])
with latest_tab:
    st.subheader("Latest model holdings")
    if latest_profiles is None:
        st.info("Supply at least seven price observations for three historical windows with two returns each.")
    else:
        st.caption(f"As of {latest_profiles.as_of} · All {latest_profiles.observations:,} supplied returns · Three historical windows")
        st.write("These are model-optimal allocations for the selected assets, constraints, and worst-window return objective. Risk labels are relative to this frontier.")
        descriptions = {
            "Low": "Minimum estimated volatility on this frontier.",
            "Medium": "The middle worst-window return target on this frontier.",
            "Extreme": "The highest worst-window return target and highest modeled risk on this frontier.",
        }
        for column, (name, portfolio) in zip(st.columns(3), latest_profiles.portfolios.items()):
            with column:
                st.markdown(f"### {name}")
                st.caption(descriptions[name])
                st.metric("Estimated annual volatility", f"{portfolio.volatility:.2%}")
                st.metric("Weakest window estimate", f"{latest_profiles.summary.loc[name, 'worst_window_return']:.2%}")
                leaders = portfolio.weights.sort_values(ascending=False).head(5)
                st.dataframe(leaders.rename("Largest allocations").to_frame().style.format("{:.2%}"), width="stretch")
        st.caption("The cards show up to five assets. The complete allocations below include every asset and sum to 100% per profile.")
        st.dataframe(weights_frame(latest_profiles).style.format("{:.2%}"), width="stretch")
        st.plotly_chart(latest_profile_chart(latest_profiles), width="stretch", theme=None)
        with st.expander("How the consistency preference works", expanded=True):
            st.write("The model splits the supplied returns into three consecutive, non-overlapping windows. It calculates each allocation's annual arithmetic return estimate in each window.")
            st.write("The frontier minimizes volatility at targets for the weakest window estimate. This favors a stronger weakest period over a high average from one strong period.")
            st.write("This objective does not minimize drawdown or require positive returns in every year. Extreme does not add leverage or deliberately maximize variance.")
            st.dataframe(latest_profiles.windows, width="stretch")
            st.dataframe(latest_profiles.window_returns.style.format("{:.2%}"), width="stretch")
        st.info("These latest weights use all supplied history, including the original holdout. The window estimates describe that fit. They are not out-of-sample results.")
        for warning in latest_profiles.warnings:
            st.warning(warning)

with frontier_tab:
    st.plotly_chart(frontier_chart(result), width="stretch", theme=None)
    st.caption("The curve shows minimum estimated volatility at each target return, using every supplied ticker. A position limit applies only if enabled. All estimates use the training period only.")
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
        if len(result.frontier) == 1:
            point = 0
            st.caption("These inputs have a single frontier point.")
        else:
            point = st.select_slider("Target-return point", options=list(range(len(result.frontier))),
                                     format_func=lambda i: f"{result.frontier.iloc[i]['expected_return']:.2%} estimated annual return")
        frontier_weights = result.frontier_weights.iloc[point].sort_values(ascending=False)
        st.dataframe(frontier_weights.rename("Weight").to_frame().style.format("{:.2%}"), width="stretch")

with risk_tab:
    st.subheader("Allocation and portfolio risk")
    diagnostics = risk_summary(result)
    st.dataframe(diagnostics.style.format({"max_weight": "{:.2%}", "effective_holdings": "{:.2f}",
                                          "diversification_ratio": "{:.2f}"}, na_rep="—"),
                 column_config={"max_weight": "Largest weight", "effective_holdings": "Effective holdings",
                                "diversification_ratio": "Diversification ratio"}, width="stretch")
    st.caption("Effective holdings equals 1 / sum of squared weights. Equal weights give the asset count. Greater concentration reduces this value.")
    st.caption("The diversification ratio compares weighted asset volatility with portfolio volatility. Both use the training covariance.")
    risk_portfolio = st.selectbox("Inspect portfolio risk for", list(result.portfolios), key="risk_portfolio")
    risk_weights = result.portfolios[risk_portfolio].weights
    contribution = risk_contributions(result)[risk_portfolio]
    risk_frame = pd.DataFrame({"Allocation weight": risk_weights, "Share of portfolio variance": contribution})
    risk_frame = risk_frame.sort_values("Allocation weight", ascending=False)
    chart_rows = risk_frame.head(20).rename_axis("Asset").reset_index()
    fig = px.bar(chart_rows, y="Asset", x=["Allocation weight", "Share of portfolio variance"],
                 orientation="h", barmode="group", color_discrete_sequence=["#40d4be", "#ffcb77"])
    fig.update_layout(template="plotly_dark", paper_bgcolor="#111c2e", plot_bgcolor="#111c2e",
                      height=max(380, 100 + 30 * len(chart_rows)),
                      margin=dict(l=10, r=20, t=20, b=100), legend=dict(title=None, orientation="h", y=-0.2))
    fig.update_xaxes(title="Weight / share of variance", tickformat=".0%")
    fig.update_yaxes(title=None, autorange="reversed")
    st.plotly_chart(fig, width="stretch", theme=None)
    if len(risk_frame) > 20:
        st.caption("The chart shows the 20 largest allocations. The table includes every asset.")
    st.dataframe(risk_frame.style.format("{:.2%}", na_rep="—"), width="stretch")
    st.caption("Risk shares use the training covariance and can be negative when assets offset risk. Zero portfolio variance gives undefined risk shares.")

with holdout_tab:
    st.subheader("What happened after training?")
    st.markdown("Allocate once at the end of training, then hold through the later period without trading between assets. Adjusted-price growth reflects the data provider's treatment of distributions; weights drift. The equal-weight portfolio follows the same rule.")
    st.plotly_chart(holdout_chart(result), width="stretch", theme=None)
    metrics = result.holdout_metrics.rename(columns={"total_return": "Total return", "cagr": "Annualized growth", "volatility": "Annual volatility", "sharpe": "Sharpe ratio", "sortino": "Sortino ratio", "calmar": "Calmar ratio", "max_drawdown": "Max drawdown"})
    st.dataframe(metrics.style.format({col: "{:.2f}" if col.endswith("ratio") else "{:.2%}" for col in metrics.columns}, na_rep="—"), width="stretch")
    st.caption("Sortino compares excess returns with downside deviation. Calmar divides annualized growth by maximum drawdown. A zero denominator gives an undefined ratio.")
    st.caption("The starting weight cap applies at allocation; later weights can exceed it as prices move. Annualized growth compounds realized returns, while frontier returns are arithmetic estimates. Repeatedly tuning settings against this holdout makes it less independent.")

with backtest_tab:
    if study is None:
        st.info("Enable Compare four backtesting methods and build the frontier to include this study.")
    else:
        st.subheader("Backtesting methods and holding contributions")
        st.caption("All methods share evaluation dates and costs. Buy and hold lets weights drift; fixed rebalancing restores the initial targets; expanding and rolling windows estimate new targets. Trades execute one session after the last estimation close. These are retrospective comparisons, not forecasts.")
        for item in findings(study):
            st.write(item)
        preferred = ("Expanding window · Low", "Expanding window · Medium", "Expanding window · Extreme",
                     "Fixed rebalance · Equal weight") if study.settings.get("include_profiles") else (
                         "Fixed rebalance · Maximum Sharpe", "Expanding window · Maximum Sharpe",
                         "Rolling window · Maximum Sharpe", "Fixed rebalance · Equal weight")
        default_strategies = [name for name in study.equity.columns if name in preferred]
        if study.settings.get("include_profiles"):
            st.caption("Low, Medium, and Extreme here use only information available before each trade. These backtests do not apply the latest weights to past dates.")
        chart_strategies = st.multiselect("Strategies shown in charts", list(study.equity.columns),
                                          default=default_strategies, key="chart_strategies")
        st.caption("This selection controls both charts. Tables and downloads include every strategy.")
        if chart_strategies:
            st.plotly_chart(backtest_chart(study, strategies=chart_strategies), width="stretch", theme=None)
            st.plotly_chart(backtest_chart(study, drawdown=True, strategies=chart_strategies), width="stretch", theme=None)
        else:
            st.info("Select at least one strategy to show its performance and drawdown.")
        formats = {column: ("{:.0f}" if column.endswith("count") else "{:.2f}" if column in ("sharpe", "sortino", "calmar", "total_turnover") else "{:.2%}") for column in study.metrics.columns}
        st.dataframe(study.metrics.style.format(formats, na_rep="—"),
                     column_config={column: label for column, (label, _) in METRICS.items()}, width="stretch")
        selected = st.selectbox("Inspect holdings for", list(study.equity.columns), key="holding_strategy")
        holdings = study.holdings[selected].sort_values("pnl_contribution", ascending=False)
        st.dataframe(holdings.style.format("{:.2%}"),
                     column_config={column: label for column, (label, _) in HOLDINGS.items()}, width="stretch")
        st.caption("Contributions are profit/loss as a fraction of initial capital, before separately charged fees. Their sum minus fees equals the strategy's net total return. Selection frequency counts target allocations; average weight includes drift and the initial cash session.")
        st.subheader("Target allocations through time")
        st.dataframe(study.allocations[selected].style.format("{:.2%}"), width="stretch")
        if study.warnings:
            with st.expander("Backtest notes"):
                for warning in study.warnings:
                    st.write(warning)

with data_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Training correlations")
        chart_assets = list(prices.columns)
        if len(chart_assets) > 30:
            chart_assets = st.multiselect("Assets shown in the correlation chart", chart_assets,
                                         default=chart_assets[:20], key="correlation_assets")
            st.caption("This selection controls the chart. All supplied assets remain in the portfolio analysis.")
        if chart_assets:
            fig = px.imshow(result.train_returns[chart_assets].corr(), zmin=-1, zmax=1,
                            color_continuous_scale="Tealrose", aspect="auto")
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", height=430, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        st.subheader("How it works")
        st.markdown(f"""
- Simple returns between supplied price observations. No forward-fill or silent asset removal.
- The original mean-return model uses the earliest **{metadata['train_fraction']:.0%}** of return observations. The remaining observations form its holdout.
- Latest holdings use all supplied history and three windows. Their fit includes the original holdout. Each backtest fit uses only prior data.
- Annual return = mean return × **{metadata['periods_per_year']:g}**. Annual covariance = covariance per observation × **{metadata['periods_per_year']:g}**.
- Covariance blends **{metadata['shrinkage']:.0%}** toward its diagonal to temper estimated correlations.
- Convex optimization finds minimum-volatility portfolios and, when positive excess return is feasible, maximum Sharpe.
- Sharpe uses the **{metadata['risk_free_rate']:.2%}** annual risk-free assumption; cash is not an investable asset in this model.
- All weights are nonnegative and sum to 100%. A starting position limit is optional; there is no minimum or maximum number of holdings.
""")
        st.caption("Use observations at consistent intervals in a common currency. The annualization assumption does not resample prices or convert currency. The app cannot verify CSV frequency, adjustments, or currency. A fixed ticker list can introduce survivorship bias.")
    st.subheader("Price observations")
    st.dataframe(prices, width="stretch", height=260)
    st.download_button("Download prices CSV", csv_text(prices, index_label="Date"), "prices.csv", "text/csv")

st.divider()
st.download_button("Download research report + CSVs", export_report(result, prices, metadata, study, latest_profiles),
                   "efficient-frontier-report.zip", "application/zip", type="primary")
st.caption("Includes an offline HTML report with Print / save PDF, exact inputs and results. Enabled backtests add findings.md, performance curves, holding contributions, target allocations and trading costs.")
