//! The model mathematics (`selection_fit_math`, `selection_replay_math`) and the
//! Data & method page (`data_method`, `settings_frame`).

use efficient_frontier::frame::{Cell, Table};
use efficient_frontier::pyfmt::{self, Dict, Json};
use efficient_frontier::selection::{SelectionStudy, selection_fit_estimates};
use efficient_frontier::selection_report::{FIT_METRIC_LABELS, selection_benchmark_note, selection_summary_html};
use efficient_frontier::story;

use super::format::{day, day_text};
use super::holding_label;
use super::html::{KEY_LEGEND, caption, esc, expander, guarded, latex, md, plain_caption, stat_strip, title_html};
use super::landing::coverage_table;
use super::table::{cell, default_format, frame, same_label, table};
use crate::build::Applied;
use crate::guard::guard;
use crate::session::Session;
use crate::setup::COST_BPS;

/// A study setting as display text: `f"{value:g}"`.
fn g(settings: &Dict, key: &str) -> String {
    pyfmt::g(settings.f64(key))
}

/// `selection_fit_math(study)`: the estimates, the optimization and fitted moments.
pub fn fit_math(study: &SelectionStudy) -> String {
    let settings = &study.settings;
    let target = settings.f64("target_volatility");
    let cap = settings.f64("max_weight");
    let limits = format!(
        r"b={}\;({}),\qquad h={}\;({})",
        pyfmt::g(target),
        pyfmt::format(target, ".2%"),
        pyfmt::g(cap),
        pyfmt::format(cap, ".0%")
    )
    .replace('%', r"\%");
    let parts = [
        md("**Estimate returns and covariance**"),
        md("For each stock, use simple returns between consecutive adjusted prices. \
            The fit contains m return observations. A is the number of observations per year."),
        latex(concat!(r"r_{i,s}=\frac{p_{i,s}}{p_{i,s-1}}-1,\qquad ", r"\bar r=\frac{1}{m}\sum_{s=1}^{m}r_s")),
        latex(&format!(r"\widehat\mu=A\bar r,\qquad A={}", g(settings, "periods_per_year"))),
        latex(concat!(r"\widehat\Sigma=\frac{A}{m-1}\sum_{s=1}^{m}", r"(r_s-\bar r)(r_s-\bar r)^{\mathsf T}")),
        caption(
            "μ̂ contains annual arithmetic mean returns. Σ̂ is the annual sample covariance matrix. \
             This app uses no covariance shrinkage.",
        ),
        md("**Choose the highest fitted mean within your limits**"),
        latex(concat!(
            r"\begin{aligned}",
            r"w^*&=\underset{w}{\operatorname{arg\,max}}\;\widehat\mu^{\mathsf T}w\\[6pt]",
            r"\text{subject to}\quad&\mathbf 1^{\mathsf T}w=1,\\",
            r"&0\le w_i\le h,\\",
            r"&\sqrt{w^{\mathsf T}\widehat\Sigma w}\le b",
            r"\end{aligned}"
        )),
        latex(&limits),
        md("The weights w sum to one, with no short sales. h caps each initial holding. \
            b caps estimated annual volatility when the allocation is chosen. \
            After purchase, weights drift and realized volatility can exceed b. \
            Neither limit triggers a trade."),
        md("**An estimate and a replay answer different questions**"),
        md(&format!(
            "The replay chooses its initial weights from the first {} of return observations. \
             The latest fit uses all supplied history, including the replay dates, and describes a separate starting allocation.",
            pyfmt::format(settings.f64("train_fraction"), ".0%")
        )),
        latex(concat!(
            r"\widehat\mu_p=\widehat\mu^{\mathsf T}w,\qquad ",
            r"\widehat\sigma_p=\sqrt{w^{\mathsf T}\widehat\Sigma w}"
        )),
        caption(
            "These fitted moments apply one fixed weight vector to each historical return observation, before fees. \
             They are not the result of buying once and letting weights drift through that history. \
             The optimizer maximizes fitted arithmetic mean, not CAGR or the Sharpe ratio.",
        ),
    ];
    format!(r#"<div class="pl-math" id="math_fit">{}</div>"#, parts.concat())
}

/// `selection_replay_math(study)`: one purchase, weight drift and the realized measures.
pub fn replay_math(study: &SelectionStudy) -> String {
    let settings = &study.settings;
    let fee = settings.f64("cost_bps") / 10_000.0;
    let periods = g(settings, "periods_per_year");
    let parts = [
        md("**One purchase, then no rebalancing**"),
        md(&format!(
            "Fit once through {}. Buy at the next close, {}. The portfolio stays in cash until that purchase. \
             There are no later refits, trades, volatility adjustments, or final liquidation.",
            day_text(settings.str("initial_training_end")),
            day_text(settings.str("test_start"))
        )),
        md("Let e be the entry close, wᵢ the initial stock weights, and pᵢ,ₜ the adjusted prices. \
            Vstart is the starting cash and c is the fee per unit invested."),
        latex(
            &format!(
                r"V_e=\frac{{V_{{\mathrm{{start}}}}}}{{1+c}},\qquad c={}\;({})",
                pyfmt::g(fee),
                pyfmt::format(fee, ".2%")
            )
            .replace('%', r"\%"),
        ),
        caption("The entry fee is c × Vₑ. It comes out of starting cash before the holdings begin to earn returns."),
        latex(concat!(
            r"\begin{aligned}",
            r"g_{i,t}&=\frac{p_{i,t}}{p_{i,e}}\\[8pt]",
            r"V_t&=V_e\sum_i w_i g_{i,t}\quad(t\ge e)",
            r"\end{aligned}"
        )),
        latex(r"a_{i,t}=\frac{w_i g_{i,t}}{\sum_j w_j g_{j,t}}"),
        md("gᵢ,ₜ is each stock’s growth factor since entry. aᵢ,ₜ is its actual portfolio weight. \
            Faster-growing holdings gain weight without a new trade. \
            The initial holding limit can therefore be exceeded later."),
        caption(
            "The calculation follows the supplied adjusted-price convention. \
             For Yahoo prices, this includes split and dividend adjustments. \
             It does not separately model cash dividends or literal share counts across corporate actions.",
        ),
        md("**Measure the completed net-return path**"),
        md("The table uses N observed returns, including the entry-fee interval. \
            The chart starts at Vstart. The entry interval has the fee but no stock return."),
        latex(concat!(r"R_t=\frac{V_t}{V_{t-1}}-1,\qquad ", r"\bar R=\frac{1}{N}\sum_{t=1}^{N}R_t")),
        latex(&format!(r"\mu_{{\mathrm{{realized}}}}=A\bar R,\qquad A={periods}")),
        latex(concat!(r"\sigma_{\mathrm{realized}}=", r"\sqrt{\frac{A}{N-1}\sum_{t=1}^{N}(R_t-\bar R)^2}")),
        latex(r"\mathrm{CAGR}=\left(\frac{V_N}{V_{\mathrm{start}}}\right)^{A/N}-1"),
        latex(concat!(r"\mathrm{MDD}=\min_{0\le t\le N}", r"\left(\frac{V_t}{\max_{0\le s\le t}V_s}-1\right)")),
        caption(
            "The annual mean scales the average one-period return. CAGR compounds the observed path. \
             Maximum drawdown is the worst fall from a previous peak, including starting cash. \
             The budget constrains fitted volatility, not this realized sample standard deviation.",
        ),
    ];
    format!(r#"<div class="pl-math" id="math_replay">{}</div>"#, parts.concat())
}

/// `settings_frame(metadata)`: study settings with plain labels.
fn settings_table(metadata: &Dict) -> String {
    let mut all = metadata.clone();
    all.insert("cost_bps", COST_BPS);
    let rows: Vec<(String, String)> = all
        .iter()
        .filter(|(key, _)| !matches!(*key, "universe_coverage" | "profile_model_revision" | "selection_model_revision"))
        .map(|(key, value)| {
            let setting = if key == "train_fraction" {
                "History for the initial fit".to_string()
            } else {
                story::setting_label(key)
            };
            let text = if key == "target_volatility" {
                pyfmt::format(value.as_f64().unwrap_or(f64::NAN), ".2%")
            } else {
                story::setting_text(key, value)
            };
            (setting, text)
        })
        .collect();
    let mut grid = super::table::Grid::new(vec!["Setting".into(), "Value".into()], 0);
    for (setting, text) in rows {
        grid.push(vec![esc(&setting), esc(&text)]);
    }
    grid.html()
}

/// Every value of a weights table as `{:.2%}`.
fn percent_cells(_: &str, value: &Cell) -> String {
    match value {
        Cell::Float(number) => pyfmt::format(*number, ".2%"),
        other => cell(other),
    }
}

/// The replay holdings of the stock portfolio, with the labels of `HOLDINGS`.
pub fn holdings_table(holdings: &Table) -> String {
    table(holdings, Some(&["Symbol"]), &holding_label, &percent_cells).html()
}

/// The entry trade: per-10,000 values and fit dates as days.
fn trades_table(trades: &Table) -> String {
    let label = |name: &str| {
        Some(
            match name {
                "turnover" => "Turnover",
                "cost" => "Fee",
                "nav_before" => "Value before",
                "nav_after" => "Value after",
                "train_start" => "Fit from",
                "train_end" => "Fit to",
                other => other,
            }
            .to_string(),
        )
    };
    let format = |name: &str, value: &Cell| match (name, value) {
        ("_index", Cell::Date(date)) => day(*date),
        ("turnover", Cell::Float(v)) => pyfmt::format(*v, ".2%"),
        ("cost", Cell::Float(v)) => pyfmt::format(v * 10_000.0, ",.2f"),
        ("nav_before" | "nav_after", Cell::Float(v)) => pyfmt::format(v * 10_000.0, ",.0f"),
        ("train_start" | "train_end", Cell::Str(text)) => esc(&day_text(text)),
        (_, other) => cell(other),
    };
    table(trades, Some(&["Date"]), &label, &format).html()
}

/// `selection_fit_estimates`: each saved fit on its own dates.
fn estimates_table(estimates: &Table) -> String {
    let label = |name: &str| {
        let label = match name {
            "phase" => "Fit",
            "execution_date" => "Traded on",
            "observations" => "Return observations",
            "max_weight" => "Largest holding",
            other => FIT_METRIC_LABELS.iter().find(|(key, _)| *key == other).map_or(other, |(_, label)| label),
        };
        Some(label.to_string())
    };
    let format = |name: &str, value: &Cell| match value {
        Cell::Date(date) => day(*date),
        Cell::Float(v) if name == "max_weight" || FIT_METRIC_LABELS.iter().any(|(key, _)| *key == name) => {
            pyfmt::format(*v, ".2%")
        }
        other => cell(other),
    };
    table(estimates, Some(&["Fitted through"]), &label, &format).html()
}

/// The number of a coverage statistic: the metadata value, else a count.
fn count(metadata: &Dict, key: &str, fallback: usize) -> String {
    metadata.get(key).map_or_else(|| fallback.to_string(), Json::py_str)
}

/// `data_method()`: the method, the mathematics and every input table.
pub fn data_method(session: &Session, applied: &Applied) -> String {
    let study = &applied.selection;
    let prices = &applied.prices;
    let metadata = &applied.metadata;
    let selected = &study.selected;
    let test = &study.test;
    let benchmark_note = guarded(|| caption(&selection_benchmark_note(study)));
    let mut html = String::from(r#"<h1 class="pl-page">Data &amp; method</h1>"#);
    html.push_str(&format!(
        r#"<section class="pl-tile" id="tile_method">{}{}{}{}{}</section>"#,
        title_html("One stock model", ""),
        md("1. **Estimate returns and risk.** Use average past returns and the relationships between stocks. \
            Use sample covariance, as in the original R model.\n\
            2. **Choose stock weights.** Maximize estimated average return within your volatility budget and holding limit. \
            Keep the portfolio fully invested, with no borrowing or short sales. \
            This does not maximize compound growth or the return-to-volatility ratio.\n\
            3. **Buy once and hold.** Use the first 70% of return observations for one initial fit. \
            Buy once at the next close, then let those holdings drift through the remaining 30%. \
            The replay has no later refits or rebalances.\n\
            4. **Test out of sample.** Fit separately on the first 50%, 60%, 70%, 80%, and 90% of returns. \
            Hold each allocation through the next tenth of the history, which its fit did not use. \
            Compare each result with the same allocation on its own fit dates.\n\
            5. **Fit the latest weights.** Apply the same model to all supplied history. \
            These weights describe a separate starting allocation. They do not produce the replay."),
        caption(&format!(
            "The initial target uses prices through the prior close and trades at the next close. \
             The first test interval stays in cash with zero interest. \
             The model charges {} on the amount bought at entry. \
             There are no later trades or fees, and no final sale.",
            pyfmt::format(study.settings.f64("cost_bps") / 10_000.0, ".2%")
        )),
        benchmark_note,
        caption(
            "The fixed 2% risk-free rate affects Sharpe and Sortino metrics in the export only. \
             It does not change stock weights or pay interest on cash. \
             Taxes, currency conversion, liquidity, and future membership changes are excluded."
        ),
    ));
    html.push_str(&fit_math(study));
    html.push_str(&replay_math(study));
    html.push_str(&expander("Historical performance checks", &guarded(|| selection_summary_html(study, metadata))));
    let estimates = match guard(|| selection_fit_estimates(study, prices)) {
        Ok(estimates) => estimates_table(&estimates),
        Err(message) => super::html::failed(&message),
    };
    html.push_str(&expander(
        "Initial and latest fitted estimates",
        &(caption(
            "Each row uses only prices available at its fit date and the saved target weights. \
             Both volatility estimates use ordinary sample covariance because this model has no covariance shrinkage. \
             Neither is the later realized volatility. The same table is exported as fit_estimates.csv.",
        ) + &estimates),
    ));
    let mut data = title_html("The data behind this result", "");
    data.push_str(&match metadata.get("universe_limitation").and_then(Json::as_str).filter(|text| !text.is_empty()) {
        Some(text) => plain_caption(text),
        None => caption(
            "Fixed list. The study keeps the same input assets for the whole period. This can favor surviving stocks.",
        ),
    });
    if let Some(coverage) = &session.coverage {
        let listed = match metadata.get("universe_source_date").and_then(Json::as_str).filter(|text| !text.is_empty()) {
            Some(date) => format!("List of {}", day_text(date)),
            None => "Today's list".to_string(),
        };
        data.push_str(&stat_strip(&[
            (count(metadata, "universe_requested", coverage.nrows()), "Checked", &listed),
            (count(metadata, "universe_included", prices.ncols()), "Used", "Complete prices"),
            (count(metadata, "universe_excluded", 0), "Left out", "Missing or late prices"),
        ]));
        data.push_str(&coverage_table(coverage));
    }
    data.push_str(&expander(
        "Price eligibility at each fit",
        &(caption(
            "The model excludes stocks with at least 20 consecutive unchanged price intervals in past data. \
             Repeated prices do not prove an invalid quote. Original price columns remain in the export.",
        ) + &table(&study.eligibility, None, &same_label, &default_format).html()),
    ));
    data.push_str(&format!(
        r#"<details class="pl-expander" data-src="/api/prices?result={id}"><summary><span class="pl-expander-label">Input prices</span><span class="pl-expander-icon" aria-hidden="true">{chevron}</span></summary><div class="pl-expander-body"><div class="pl-lazy" data-lazy><p class="pl-caption">Loading the input prices…</p></div><a class="pl-btn pl-btn-secondary" href="/download/prices.csv?result={id}" download="prices.csv" id="download_prices">Download prices.csv</a></div></details>"#,
        id = applied.id,
        chevron = super::html::icon::CHEVRON
    ));
    data.push_str(&expander("Source and settings", &guarded(|| settings_table(metadata))));
    html.push_str(&format!(r#"<section class="pl-tile" id="tile_data">{data}</section>"#));
    let replay = [
        caption(
            "These tables describe the stock portfolio in the buy-and-hold replay. \
             The export also contains every comparison path and trade.",
        ),
        holdings_table(&test.holdings[selected]),
        frame(&test.allocations[selected], "Date", &|v| pyfmt::format(v, ".2%")).html(),
        trades_table(&test.trades[selected]),
        caption(
            "Values and fees are per 10,000 at the start. Holding contributions less fees equal the net total return.",
        ),
    ]
    .concat();
    html.push_str(&expander("Exact replay holdings, initial weights, and entry trade", &replay));
    html.push_str(KEY_LEGEND);
    html
}
