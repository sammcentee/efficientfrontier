//! The settings drawer, "Change the study" (`setup_sheet`).
//!
//! The server renders every control with the applied setup. The browser shows
//! the controls of the selected market (`data-when-*`), disables the hidden ones
//! so that they keep their stored values, and asks the server whether the draft
//! differs from the applied result.

use efficient_frontier::frame::{Date, std};
use efficient_frontier::pyfmt::{self, Json};

use super::format::day;
use super::html::{caption, esc, icon};
use super::widgets::{FileSlot, NumberInput, checkbox, date_input, radio_list, segmented, select};
use crate::build::Applied;
use crate::session::Session;
use crate::setup::{Currency, Frequency, History, Market, differs, resolve, years_before};

/// A block that the browser shows only for the listed values of a control.
fn when(control: &str, values: &[&str], current: &str, body: &str) -> String {
    let hidden = if values.contains(&current) { "" } else { " hidden" };
    format!(r#"<div data-when-{control}="{}"{hidden}>{body}</div>"#, esc(&values.join("|")))
}

/// `risk_reference()`: the comparison ETFs' volatility over the applied input history.
fn risk_reference(applied: Option<&Applied>) -> String {
    let Some((applied, benchmarks)) =
        applied.and_then(|applied| applied.benchmark_prices.as_ref().map(|b| (applied, b)))
    else {
        return String::new();
    };
    let periods = applied.metadata.get("periods_per_year").and_then(Json::as_f64).unwrap_or(252.0);
    let returns = benchmarks.pct_change();
    let parts: Vec<String> = benchmarks
        .columns
        .iter()
        .enumerate()
        .map(|(j, name)| format!("{name} {}", pyfmt::format(std(&returns.col(j), 1) * periods.sqrt(), ".1%")))
        .collect();
    caption(&format!(
        "Reference from the saved input history: {} annual historical volatility from ordinary sample covariance. \
         These are observations over the full input history, not the replay or future limits.",
        parts.join(", ")
    ))
}

/// The drawer for a session, as an HTML fragment.
pub fn drawer(session: &Session, today: Date) -> String {
    let setup = &session.setup;
    let market = setup.market.label();
    let markets: Vec<(&str, &str)> = Market::ALL.iter().map(|m| (m.label(), m.caption())).collect();
    let yahoo: Vec<&str> = Market::ALL.iter().filter(|m| m.uses_yahoo()).map(|m| m.label()).collect();
    let tickers = format!(
        r#"<div class="pl-field" id="field_ticker_text">{}<textarea id="ticker_text" name="ticker_text" rows="3" spellcheck="false">{}</textarea></div>"#,
        super::widgets::label(
            "ticker_text",
            "Stock symbols",
            Some(
                "Separate Yahoo Finance symbols with commas. Exchange suffixes such as .L work. SPY and QQQ serve only as comparisons."
            )
        ),
        esc(&setup.ticker_text)
    );
    let prices_file = FileSlot {
        id: "prices_upload",
        label: "Adjusted prices (.csv)",
        caption: Some(
            "Use Date as the first column, then one stock per column. Prices must be complete and use a common currency. \
             SPY and QQQ columns serve only as comparisons and cannot become holdings.",
        ),
        help: None,
        stored: setup.prices_file.as_ref().map(|file| file.name.as_str()),
        clear: "clear_prices_file",
    };
    let histories: Vec<(&str, &str)> = History::ALL.iter().map(|h| (h.label(), h.short_label())).collect();
    let preset = |years: u32, history: History| {
        let end = today;
        let last = end.pred_opt().unwrap_or(end);
        when(
            "history",
            &[history.label()],
            setup.history.label(),
            &caption(&format!("{} to {}", day(years_before(end, years)), day(last))),
        )
    };
    let history = format!(
        "{}{}{}{}",
        segmented("history", "History", &histories, setup.history.label()),
        when(
            "history",
            &[History::Custom.label()],
            setup.history.label(),
            &format!(
                r#"<div class="pl-columns">{}{}</div>"#,
                date_input("start", "From", setup.start, None),
                date_input("end", "Until", setup.end, Some("The last day is not included."))
            )
        ),
        preset(5, History::FiveYears),
        preset(10, History::TenYears)
    );
    let target = NumberInput {
        id: "target_vol",
        name: "target_vol",
        label: "Annual volatility budget",
        value: setup.target_vol,
        format: ".2f",
        step: 0.1,
        min: Some(1.0),
        max: Some(100.0),
        percent: true,
        help: Some(
            "A fitted annual volatility ceiling. Realized volatility can exceed it. No borrowing or automatic cash allocation.",
        ),
    };
    let cap = NumberInput {
        id: "cap",
        name: "cap",
        label: "Holding limit",
        value: setup.cap,
        format: "g",
        step: 1.0,
        help: Some("A 10% limit needs at least ten assets. If the risk budget cannot be met, the build stops."),
        ..target
    };
    let frequencies: Vec<&str> = Frequency::ALL.iter().map(|f| f.label()).collect();
    let periods = NumberInput {
        id: "periods_per_year",
        name: "periods_per_year",
        label: "Observations per year",
        value: setup.periods_per_year,
        format: ".2f",
        step: 0.01,
        min: Some(1.0),
        max: None,
        percent: false,
        help: None,
    };
    let csv_frequency = format!(
        "{}{}{}",
        select("frequency", "Price frequency", &frequencies, setup.frequency.label(), None),
        when("frequency", &[Frequency::Custom.label()], setup.frequency.label(), &periods.html()),
        caption("Frequency scales yearly figures. It does not resample prices.")
    );
    let currencies: Vec<&str> = Currency::ALL.iter().map(|c| c.label()).collect();
    let benchmark_file = FileSlot {
        id: "benchmark_upload",
        label: "Market prices (optional)",
        caption: None,
        help: Some("Use Date with SPY, QQQ, or both on exactly the same dates as your stock file."),
        stored: setup.benchmark_file.as_ref().map(|file| file.name.as_str()),
        clear: "clear_benchmark_file",
    };
    let csv_benchmarks = format!(
        "{}{}{}",
        benchmark_file.html(),
        checkbox(
            "download_benchmarks",
            "Download SPY and QQQ from Yahoo Finance for this file",
            setup.download_benchmarks,
            false
        ),
        caption(
            "CSV analysis stays offline unless you select this download. Local comparison columns take priority over a download."
        )
    );
    let csv = [Market::UploadCsv.label()];
    let data_options = format!(
        r#"<details class="pl-expander" id="data_options"><summary><span class="pl-expander-label">Data options <span class="pl-gray">frequency, currency, market comparisons</span></span><span class="pl-expander-icon" aria-hidden="true">{chevron}</span></summary><div class="pl-expander-body">{frequency}{currency}{currency_note}{compare}{benchmarks}</div></details>"#,
        chevron = icon::CHEVRON,
        frequency = when("market", &csv, market, &csv_frequency),
        currency = select(
            "price_currency",
            "Price currency",
            &currencies,
            setup.price_currency.label(),
            Some("Declare the currency of every price. The app does not check or convert currencies.")
        ),
        currency_note = caption("SPY and QQQ comparisons need USD prices. No currency conversion occurs."),
        compare = checkbox(
            "compare_market",
            "Compare with the S&P 500 and the Nasdaq-100",
            setup.compare_market,
            setup.market == Market::Demo
        ),
        benchmarks = when("market", &csv, market, &csv_benchmarks),
    );
    let applied = session.applied.as_deref();
    let pending = applied.is_some_and(|applied| differs(&resolve(setup, today), &applied.metadata, &applied.inputs));
    let differs_note = format!(
        r#"<div class="pl-alert pl-alert-info" role="status" id="pl-differs"{}><p>Your setup differs from this result. Select Build portfolio to apply it.</p></div>"#,
        if pending { "" } else { " hidden" }
    );
    format!(
        r#"<div class="pl-drawer-backdrop" data-dismiss></div><aside class="pl-drawer" role="dialog" aria-modal="true" aria-labelledby="pl-drawer-title" tabindex="-1"><header class="pl-drawer-head"><h2 id="pl-drawer-title">Change the study</h2><button type="button" class="pl-icon-btn" data-close aria-label="Close">{close}</button></header><form class="pl-drawer-body" id="setup_form" method="post" action="/build" enctype="multipart/form-data" data-draft="/api/draft"><input type="hidden" name="form" value="drawer"><input type="hidden" name="clear_prices_file" value=""><input type="hidden" name="clear_benchmark_file" value=""><p class="pl-sub">Choose the stocks and the history. Nothing changes until you select Build portfolio.</p>{markets}{tickers}{prices_file}{history}{target}{reference}{cap}{demo}{method}{data_options}{differs_note}<div class="pl-drawer-actions"><button type="button" class="pl-btn pl-btn-secondary" id="cancel_setup" data-close>Cancel</button><button type="submit" class="pl-btn pl-btn-build" id="drawer_build" data-build><span>Build portfolio</span><kbd data-mod-key>Ctrl + Enter</kbd></button></div></form></aside>"#,
        close = icon::CLOSE,
        markets = radio_list("market", "Stocks", &markets, market),
        tickers = when("market", &[Market::MyTickers.label()], market, &tickers),
        prices_file = when("market", &csv, market, &prices_file.html()),
        history = when("market", &yahoo, market, &history),
        target = target.html(),
        reference = risk_reference(applied),
        cap = cap.html(),
        demo = when(
            "market",
            &[Market::Demo.label()],
            market,
            &caption(
                "The demo has six simulated assets. Its default holding limit is 20%, shown above; you can change it. At least 17% is needed."
            )
        ),
        method = caption(
            "One fixed method: fit on the first 70% of return observations, then buy once and hold through the remaining 30%. \
             The entry fee is 0.10% of the amount bought. There is no later rebalancing."
        ),
    )
}
