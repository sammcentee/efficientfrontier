"""Offline interface for portfolios with dated membership and sparse prices."""

from datetime import date
from hashlib import sha256
from io import BytesIO

import streamlit as st

from .backtest_report import HOLDINGS, METRICS, backtest_chart
from .historical import historical_demo, load_historical_prices, run_historical_backtests
from .historical_report import build_historical_report, historical_report_zip
from .membership import bundled_membership, load_membership_csv
from .style import PLOTLY_CONFIG


def _file(upload):
    return None if upload is None else (upload.name, upload.getvalue())


def _signature(source, prices_file, members_file, coverage_end, settings, currency):
    files = tuple(None if item is None else (item[0], sha256(item[1]).hexdigest())
                  for item in (prices_file, members_file))
    return source, files, coverage_end, tuple(settings.items()), currency


def render_historical():
    st.title("Historical membership")
    st.write("Test portfolios against the members recorded at each date. A company can enter only while it is a member, "
             "with enough prior price history. The three portfolios use the same rolling window and trade dates.")
    source = st.radio("Historical data", ["Offline demo", "Nasdaq-100 membership + price CSV", "Custom membership + price CSV"],
                      key="historical_source")
    prices_file = members_file = None
    coverage_end = None
    if source == "Offline demo":
        st.info("Synthetic prices and membership. This example includes additions, removals, re-entry, and missing history. "
                "It does not measure market performance.")
    else:
        st.caption("Price CSV: Date,ASSET_A,ASSET_B,… Use adjusted prices in one currency. Leave unavailable prices blank. "
                   "Use stable security identities: a reused ticker needs a separate label in both files. "
                   "The test stops if a held asset has no valuation price.")
        prices_file = _file(st.file_uploader("Adjusted-price CSV", type=["csv"], key="historical_prices_file"))
        if source == "Custom membership + price CSV":
            st.caption('Membership CSV: date,tickers. Each row is a full roster, for example 2020-01-01,"ASSET_A,ASSET_B". '
                       "A roster takes effect on that date and replaces the prior roster.")
            members_file = _file(st.file_uploader("Membership CSV", type=["csv"], key="historical_members_file"))
            coverage_end = st.date_input("Membership covered through", value=date.today(), key="historical_coverage_end")
            st.caption("Declare the final date checked by your source. The last change date alone does not establish coverage.")
        else:
            membership = bundled_membership()
            st.caption(f"Bundled source: {membership.source}. Coverage: {membership.coverage_start.date()} to {membership.coverage_end.date()}.")
            with st.expander("Membership source and limits"):
                for note in membership.metadata.get("limitations", []):
                    st.write(note)
                st.json(membership.metadata)
    with st.expander("Test settings"):
        left, right = st.columns(2)
        with left:
            train = st.slider("Initial training share (%)", 10, 90, 70, key="historical_train")
            window = st.number_input("Rolling window (returns)", min_value=2, value=126, step=1, key="historical_window")
            interval = st.number_input("Rebalance every (observations)", min_value=1, value=21, step=1, key="historical_interval")
            fees = st.number_input("Fee on buys and sells (basis points)", min_value=0., value=10., step=1., key="historical_fees")
        with right:
            cap = st.number_input("Maximum target weight (%)", min_value=.01, max_value=100., value=100., key="historical_cap")
            risk_free = st.number_input("Annual risk-free rate (%)", value=2., key="historical_risk_free")
            shrinkage = st.slider("Covariance shrinkage (%)", 0, 100, 10, key="historical_shrinkage")
            periods = st.number_input("Observations per year", min_value=1., value=252., key="historical_periods")
        currency = st.text_input("Common price currency", value="USD", key="historical_currency").strip().upper()
        st.caption("Targets use prices through the prior close and trade at the next close. Removed members leave at the next "
                   "scheduled trade. Cash earns no interest. If too few eligible securities meet the cap, the portfolio holds cash.")
    settings = dict(train_fraction=train / 100, rolling_window=int(window), rebalance_every=int(interval),
                    cost_bps=fees, max_weight=cap / 100, risk_free_rate=risk_free / 100,
                    shrinkage=shrinkage / 100, periods_per_year=periods)
    signature = _signature(source, prices_file, members_file, coverage_end, settings, currency)
    if st.button("Run historical study", key="historical_run", type="primary"):
        try:
            if not currency:
                raise ValueError("Declare the common price currency, such as USD or EUR.")
            if source == "Offline demo":
                prices, membership = historical_demo()
            else:
                if prices_file is None:
                    raise ValueError("Choose an adjusted-price CSV first.")
                prices = load_historical_prices(BytesIO(prices_file[1]))
                if source == "Custom membership + price CSV":
                    if members_file is None:
                        raise ValueError("Choose a membership CSV first.")
                    membership = load_membership_csv(BytesIO(members_file[1]), coverage_end=coverage_end,
                                                     source="CSV: " + members_file[0])
                else:
                    membership = bundled_membership()
            with st.spinner("Check membership, fit targets, and replay trades"):
                study = run_historical_backtests(prices, membership, **settings)
                study.settings.update(source="Historical membership demo · synthetic" if source == "Offline demo" else "CSV: " + prices_file[0],
                                      price_currency=currency,
                                      currency_assumption="All prices must share the declared currency. No currency conversion occurs.")
                files = build_historical_report(prices, membership, study)
                bundle = historical_report_zip(files)
            st.session_state.historical_result = (signature, prices, membership, study, bundle)
            st.session_state.pop("historical_error", None)
        except (ValueError, RuntimeError, OSError) as exc:
            st.session_state.historical_error = str(exc)
    if st.session_state.get("historical_error"):
        st.error(st.session_state.historical_error)
    saved = st.session_state.get("historical_result")
    if saved is None:
        return
    saved_signature, prices, membership, study, bundle = saved
    if saved_signature != signature:
        st.info("The results below use the last completed settings. Select Run historical study to apply your changes.")
    st.subheader("Historical test results")
    st.caption(f"{study.settings['source']} · {prices.index[0].date()} to {prices.index[-1].date()} · "
               f"Declared currency: {study.settings['price_currency']}. No currency conversion.")
    if "synthetic" in study.settings["source"]:
        st.info("These results use simulated data, not market prices.")
    st.download_button("Export historical report", bundle, file_name="historical-membership.zip", mime="application/zip",
                       key="historical_export", on_click="ignore")
    st.plotly_chart(backtest_chart(study), width="stretch", theme=None, config=PLOTLY_CONFIG)
    metric_formats = {name: "{:.0f}" if name.endswith("count") else "{:.2f}"
                      if name in ("sharpe", "sortino", "calmar") else "{:.2f}×"
                      if name == "total_turnover" else "{:.2%}" for name in study.metrics}
    st.dataframe(study.metrics.style.format(metric_formats, na_rep="—"), width="stretch",
                 column_config={"_index": "Strategy", **{name: label for name, (label, _) in METRICS.items()}})
    st.caption("Minimum volatility, maximum Sharpe, and equal weight use only the eligible members at each trade. "
               "These results do not establish an advantage over the Nasdaq-100 index.")
    for note in study.warnings:
        st.warning(note)
    with st.expander("Membership and price coverage", expanded=True):
        st.caption("Every membership and price decision at each scheduled trade is retained. An excluded company still appears in this record.")
        st.dataframe(study.coverage, hide_index=True, width="stretch")
        st.json(membership.metadata)
    strategy = st.selectbox("Inspect historical strategy", list(study.equity), key="historical_strategy")
    with st.expander("Holdings, target weights, and trades"):
        st.dataframe(study.holdings[strategy].style.format("{:.2%}"), width="stretch",
                     column_config={"_index": "Symbol", **{name: label for name, (label, _) in HOLDINGS.items()}})
        st.dataframe(study.allocations[strategy].style.format("{:.2%}"), width="stretch")
        st.dataframe(study.trades[strategy].style.format({
            "turnover": "{:.2%}", "cost": "{:.4%}", "nav_before": "{:.4f}", "nav_after": "{:.4f}",
            "cash_weight": "{:.2%}"}), width="stretch")
        st.caption("Portfolio values start at 1. Fees are shown as a share of initial capital.")
    with st.expander("Cash through time"):
        st.dataframe(study.cash_weights.style.format("{:.2%}"), width="stretch")
    st.caption("Membership sources can contain errors and omissions. Adjusted prices can also contain hindsight from later corporate actions. "
               "This mode records changing membership; it does not certify a complete survivorship-free data set. Taxes, FX, and liquidity limits are omitted.")
