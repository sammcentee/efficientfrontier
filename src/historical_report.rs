//! Portable inputs, results and offline charts for historical membership studies.
//!
//! Port of `efficient_frontier/historical_report.py` in the Python release (commit 6c6df6e).
//!
//! The report charts use the fixed element ids `historical-growth` and
//! `historical-drawdown` instead of random ones.

use indexmap::IndexMap;

use crate::backtest_report::{ChartOptions, HOLDINGS, METRICS, backtest_chart, formatted};
use crate::error::Result;
use crate::frame::{Cell, Frame, Table, iso};
use crate::historical::HistoricalStudy;
use crate::html_table::{HtmlOptions, data_table, to_html};
use crate::membership::{MembershipHistory, N100TICKERS_LICENSE};
use crate::plot::{PLOTLY_JS, PLOTLY_JS_LICENSE};
use crate::pyfmt::{self, Dict, Json, escape};
use crate::selection_report::write_zip;
use crate::story::capitalize;
use crate::style::{plotly_config, report_css};

pub const ASSUMPTIONS: [&str; 7] = [
    "The three portfolios refit at each scheduled trade. Each fit uses the configured rolling \
     window of returns through the prior close. New targets execute at the following close and \
     earn no execution-day return. The first evaluation interval stays in cash.",
    "Each trade uses the membership effective on its execution date. Membership publication \
     timestamps are not verified. Removed companies remain held until \
     the next scheduled rebalance. An eligible asset needs the full training window of prices. \
     The coverage file records exclusions at each trade; the study does not require every company \
     to have prices for the full study period.",
    "Missing prices for a held asset or an execution stop the study. Prices are never filled. \
     Cash earns zero interest. If the eligible assets cannot form a portfolio within the holding \
     limit, the portfolio stays in cash for that allocation.",
    "Trading costs apply to purchases and sales, including entry. No final liquidation is charged. \
     Holding P&L contributions less fees reconcile to net total return. Gross turnover counts both \
     purchases and sales divided by portfolio value before each trade. Taxes, currency conversion \
     and liquidity constraints are excluded. Adjusted prices must use one currency basis.",
    "Historical membership reduces one source of survivorship bias. It does not guarantee complete \
     history: community records can omit changes, and price providers can omit delisted companies \
     or reuse symbols. Read the source metadata and coverage exclusions before interpreting results.",
    "To rerun a study with bundled history, keep --historical-membership with the same bundled data \
     revision, exported prices and settings. membership.csv alone does not preserve identity \
     exclusions. metadata.json records those rules, but loading the snapshots with --membership-csv \
     treats the symbols as user-supplied stable identifiers and does not restore the rules.",
    "These are retrospective comparisons, not forecasts. Choosing a strategy after seeing its \
     results can overfit the evaluated period.",
];

fn metadata_table(values: &Dict) -> String {
    let rows: String = values
        .iter()
        .map(|(key, value)| {
            let text = match value {
                Json::Dict(_) | Json::List(_) => value.dumps(None),
                other => other.py_str(),
            };
            format!("<tr><th>{}</th><td>{}</td></tr>", escape(&capitalize(&key.replace('_', " "))), escape(&text))
        })
        .collect();
    format!("<div class=\"scroll\"><table class=\"metadata\">{rows}</table></div>")
}

/// `frame.set_index(column)` for a table with a default index.
fn indexed_by(frame: &Table, column: &str) -> Table {
    let mut table = Table::new(Some(column), frame.column(column).to_vec());
    for (name, cells) in frame.columns.iter().zip(&frame.data) {
        if name != column {
            table.push(name, cells.clone());
        }
    }
    table
}

/// The closing cash weight of each strategy on the last date, as percent text.
fn closing_cash(cash_weights: &Frame) -> Table {
    let last = cash_weights.nrows().saturating_sub(1);
    let values = (0..cash_weights.ncols())
        .map(|j| Cell::Str(format!("{}%", pyfmt::format(cash_weights.get(last, j) * 100.0, ".2f"))))
        .collect();
    Table::labelled(None, &cash_weights.columns).with("Closing cash weight", values)
}

/// Return the report, input copies and all numeric results as portable files, in export order.
pub fn build_historical_report(
    prices: &Frame,
    membership: &MembershipHistory,
    historical: &HistoricalStudy,
) -> IndexMap<String, Vec<u8>> {
    let study = &historical.study;
    let snapshots = membership.snapshots();
    let membership_frame = Table::new(Some("date"), snapshots.iter().map(|(date, _)| Cell::Date(*date)).collect())
        .with("tickers", snapshots.iter().map(|(_, symbols)| Cell::Str(symbols.join(","))).collect());
    let strategy_files: Vec<(String, String)> = study
        .equity
        .columns
        .iter()
        .enumerate()
        .map(|(i, name)| (format!("backtests/{:02}", i + 1), name.clone()))
        .collect();
    let mut source = membership.metadata();
    source.insert("coverage_start", iso(membership.coverage_start()));
    source.insert("coverage_end", iso(membership.coverage_end()));
    let details = crate::dict! {
        "study" => "Historical membership rolling-window backtest",
        "membership" => source.clone(),
        "price_symbols" => prices.columns.clone(),
        "price_start" => iso(prices.index[0]),
        "price_end" => iso(prices.index[prices.nrows() - 1]),
        "settings" => study.settings.clone(),
        "warnings" => study.warnings.clone(),
        "backtest_files" => Dict(strategy_files.iter().map(|(stem, name)| (stem.clone(), Json::from(name.as_str()))).collect()),
        "assumptions" => ASSUMPTIONS.to_vec(),
    };
    let mut csv: Vec<(String, String)> = vec![
        ("prices.csv".into(), prices.to_csv("Date")),
        ("membership.csv".into(), membership_frame.to_csv(None)),
        ("equity.csv".into(), study.equity.to_csv("date")),
        ("metrics.csv".into(), study.metrics.to_csv(Some("strategy"))),
        ("cash_weights.csv".into(), historical.cash_weights.to_csv("date")),
        ("coverage.csv".into(), indexed_by(&historical.coverage, "date").to_csv(None)),
    ];
    let mut holdings_sections = String::new();
    for (stem, name) in &strategy_files {
        csv.push((format!("{stem}_allocations.csv"), study.allocations[name].to_csv("date")));
        csv.push((format!("{stem}_holdings.csv"), study.holdings[name].to_csv(Some("symbol"))));
        csv.push((format!("{stem}_trades.csv"), study.trades[name].to_csv(Some("date"))));
        holdings_sections.push_str(&format!(
            "<details><summary>{}</summary><div class=\"scroll\">{}</div></details>",
            escape(name),
            data_table(&formatted(&study.holdings[name], &HOLDINGS))
        ));
    }
    let mut files: IndexMap<String, Vec<u8>> = csv.into_iter().map(|(name, text)| (name, text.into_bytes())).collect();
    files.insert("metadata.json".into(), Json::Dict(details).dumps(Some(2)).into_bytes());
    let mut notices = format!("Plotly.js (embedded in report.html)\n\n{PLOTLY_JS_LICENSE}");
    if source.get("license").is_some_and(|license| license.py_str().contains("LICENSE.n100tickers")) {
        files.insert("LICENSE.n100tickers".into(), N100TICKERS_LICENSE.as_bytes().to_vec());
        notices.push_str(&format!("\n\nn100tickers membership data\n\n{N100TICKERS_LICENSE}"));
    }
    files.insert("THIRD_PARTY_NOTICES.txt".into(), notices.clone().into_bytes());
    let config = plotly_config();
    let charts: String = [(false, "historical-growth"), (true, "historical-drawdown")]
        .iter()
        .map(|(drawdown, id)| {
            let chart = backtest_chart(study, &ChartOptions { drawdown: *drawdown, ..Default::default() })
                .expect("every study path is known");
            format!("<div class=\"chart\">{}</div>", chart.to_html(id, &config))
        })
        .collect();
    let notes: String = study.warnings.iter().map(|note| format!("<li>{}</li>", escape(note))).collect();
    let assumptions: String = ASSUMPTIONS.iter().map(|note| format!("<p>{}</p>", escape(note))).collect();
    let dates = &study.equity.index;
    let periods = study.settings.get("periods_per_year").and_then(Json::as_f64).unwrap_or(252.0);
    let notes_aside =
        if notes.is_empty() { String::new() } else { format!("<aside><h2>Study notes</h2><ul>{notes}</ul></aside>") };
    let document = format!(
        "<!doctype html>\n\
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
<title>Portfolio Lab · Historical membership</title>\n\
<style>{css}</style><script>{PLOTLY_JS}</script></head><body><main>\n\
<p class=\"eyebrow\">Historical research</p><h1>Historical membership</h1>\n\
<p>Three rolling-window portfolios use dated membership and the price history available before each trade.</p>\n\
<p>Portfolio baseline: {baseline}. Evaluation returns: {first} to {last}.\n\
Charts start with 10,000 units and include modeled trading costs.</p>\n\
<section><h2>Membership source and coverage</h2>{source_table}\n\
<p>Read the source limits alongside the results. Dated membership does not establish a complete historical reconstruction.</p></section>\n\
<section><h2>Portfolio results</h2>{charts}<div class=\"scroll\">{metrics}</div>\n\
<h3>Cash at the final close</h3><div class=\"scroll\">{cash}</div>\n\
<p>cash_weights.csv contains daily closing cash proportions, including the initial cash balance.</p></section>\n\
<section><h2>Holdings</h2><p>Asset P&amp;L contributions use initial capital as the denominator.\n\
Fees are accounted for separately. Allocations, trades and all holdings are included in the CSV files.</p>\n\
{holdings_sections}</section>\n\
<section><h2>Settings</h2>{settings_table}<h2>Assumptions</h2>{assumptions}\n\
<p>CAGR and volatility use {periods} observations per year.\n\
An em dash marks an undefined ratio.</p>\n\
<details><summary>Eligibility at each trade</summary><div class=\"scroll\">{coverage}</div></details></section>\n\
{notes_aside}\n\
<footer>This report works offline. metadata.json maps numbered result files to portfolio names.\n\
<details><summary>Third-party notices</summary><pre>{notices}</pre></details></footer>\n\
</main></body></html>",
        css = report_css(),
        baseline = iso(dates[0]),
        first = iso(dates[1]),
        last = iso(dates[dates.len() - 1]),
        source_table = metadata_table(&source),
        metrics = data_table(&formatted(&study.metrics, &METRICS)),
        cash = data_table(&closing_cash(&historical.cash_weights)),
        settings_table = metadata_table(&study.settings),
        periods = pyfmt::g(periods),
        coverage = to_html(&historical.coverage, &HtmlOptions::REPORT_NO_INDEX),
        notices = escape(&notices),
    );
    files.insert("report.html".into(), document.into_bytes());
    files
}

/// Package files returned by [`build_historical_report`] without changing them.
pub fn historical_report_zip(files: &IndexMap<String, Vec<u8>>) -> Result<Vec<u8>> {
    let entries: Vec<(String, &[u8])> =
        files.iter().map(|(name, content)| (name.clone(), content.as_slice())).collect();
    write_zip(&entries)
}
