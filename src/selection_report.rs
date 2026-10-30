//! Offline evidence and reproducible exports for the single portfolio model.
//!
//! Port of `efficient_frontier/selection_report.py`. Owner: reports agent.
//! The web app calls the public functions below; keep these signatures.
//!
//! Charts in the exported report use fixed element ids (`chart-latest-frontier`,
//! `chart-out-of-sample` and `chart-replay`) instead of random ones, so the same
//! inputs give the same report.

use std::io::Write as _;

use serde_json::{Value, json};

use crate::backtest::BacktestStudy;
use crate::backtest_report::{ChartOptions, backtest_chart};
use crate::benchmark_report::{np_mean, values_json};
use crate::data::stale_price_notes;
use crate::error::{Error, Result};
use crate::frame::{Cell, Date, Frame, Series, Table, iso};
use crate::html_table::{HtmlOptions, data_table, to_html};
use crate::out_of_sample::{OutOfSampleStudy, portfolio_comparison, run_default_out_of_sample, sample_values};
use crate::plot::{Figure, PLOTLY_JS, PLOTLY_JS_LICENSE, merge};
use crate::pyfmt::{self, Dict, Json, escape};
use crate::selection::{
    BASELINE, SELECTED, SelectionFrontier, SelectionStudy, selection_fit_estimates, selection_frontier,
};
use crate::story::truthy;
use crate::style::{ACCENT, ChartStyle, GRID, INK_3, REST, benchmark_color, plotly_config, report_css, style_chart};

pub const SELECTION_METRIC_LABELS: [(&str, &str); 6] = [
    ("annual_mean", "Net arithmetic mean / year"),
    ("cagr", "Net compound growth / year (CAGR)"),
    ("volatility", "Realized volatility / year"),
    ("max_drawdown", "Maximum drawdown"),
    ("total_return", "Net total return"),
    ("total_cost", "Fees / initial capital"),
];
pub const FIT_METRIC_LABELS: [(&str, &str); 3] = [
    ("fitted_mean", "Fitted annual mean"),
    ("model_volatility", "Model volatility"),
    ("historical_volatility", "Historical volatility"),
];
pub const LATEST_LABEL: &str = "Latest stock allocation";
pub const REPLAY_LABEL: &str = "Stock portfolio · buy and hold";
pub const HISTORY_NOTE: &str = "The replay buys the initial allocation once and holds it. Historical data reviewed earlier can introduce research hindsight. \
A chronological split alone is not fresh prospective evidence. These descriptive checks do not prove future outperformance.";
pub const OUT_OF_SAMPLE_TITLE: &str = "In sample versus out of sample";
/// Marker symbols of the reference paths in the walk-forward chart.
pub const OUT_OF_SAMPLE_SYMBOLS: [(&str, &str); 3] =
    [("Nasdaq-100 (QQQ)", "diamond"), ("S&P 500 (SPY)", "triangle-up"), (BASELINE, "square")];
/// Display names of the engine paths in the walk-forward tables.
pub const PATH_LABELS: [(&str, &str); 2] = [(SELECTED, "Stock portfolio"), (BASELINE, "Equal-weight stocks")];

const SPY: &str = "S&P 500 (SPY)";
const QQQ: &str = "Nasdaq-100 (QQQ)";

/// `selection_replay_return_bridge`: the initial fitted mean reconciled to the replay mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReturnBridge {
    pub fitted_mean: f64,
    pub realized_mean: f64,
    pub mean_change: f64,
    pub weight_drift: f64,
    pub entry_delay: f64,
    pub entry_fee: f64,
}

/// `out_of_sample_findings`: averages over completed walk-forward windows.
#[derive(Clone, Debug, PartialEq)]
pub struct OutOfSampleFindings {
    pub completed: usize,
    pub windows: usize,
    pub in_sample: f64,
    pub out_of_sample: f64,
    pub change: f64,
    pub lower: usize,
    pub margin_in_sample: f64,
    pub margin_out_of_sample: f64,
    /// Reference path name and the count of windows in which the portfolio beat it out of sample.
    pub higher_than: Vec<(String, usize)>,
    pub test_returns: i64,
}

/// Keep the engine's fixed strategy order and exact realized metrics.
pub fn selection_metrics_frame(study: &BacktestStudy) -> Table {
    study.metrics.select(&SELECTION_METRIC_LABELS.map(|(column, _)| column))
}

/// Identify every comparison path in the app and exported report.
pub fn selection_chart(study: &BacktestStudy) -> Figure {
    let options = ChartOptions { focus: Some(SELECTED), ..Default::default() };
    let mut figure = backtest_chart(study, &options).expect("every study path is known");
    for (name, trace) in study.equity.columns.iter().zip(figure.data.iter_mut()) {
        let color = match name.as_str() {
            SELECTED => ACCENT,
            BASELINE => INK_3,
            other => benchmark_color(other, INK_3),
        };
        let dash = match name.as_str() {
            BASELINE => "dash",
            SPY => "dot",
            _ => "solid",
        };
        let label = if name == SELECTED { Value::from(REPLAY_LABEL) } else { trace["name"].clone() };
        merge(trace, &json!({"name": label, "showlegend": true, "line": {"color": color, "dash": dash}}));
    }
    figure
}

/// `fig.update_xaxes(title=text, ...)`: plotly.py replaces the whole axis title with `{"text": text}`.
fn axis_title(figure: &mut Figure, axis: &str, text: &str) {
    figure.layout[axis]["title"] = json!({"text": text});
}

/// Plot the latest stock frontier and passive ETF references on their fit dates.
pub fn selection_frontier_chart(frontier: &SelectionFrontier, study: &SelectionStudy) -> Figure {
    let hover = "Annual volatility: %{x:.2%}<br>Annual arithmetic mean: %{y:.2%}<extra>%{fullData.name}</extra>";
    let shrinkage = study.settings.f64("shrinkage") != 0.0;
    let points = &frontier.frontier;
    let mut figure = Figure::with_trace(json!({
        "type": "scatter", "x": values_json(&points.f64s("model_volatility")), "y": values_json(&points.f64s("fitted_mean")),
        "mode": "lines", "name": "Stock efficient frontier", "line": {"color": INK_3, "width": 2.5}, "hovertemplate": hover,
    }));
    let comparisons = &frontier.comparisons;
    let volatility = comparisons.f64s("model_volatility");
    let mean = comparisons.f64s("fitted_mean");
    for (i, name) in comparisons.index_labels().iter().enumerate() {
        let selected = name == SELECTED;
        let label = if selected { LATEST_LABEL } else { name.as_str() };
        let risk_label = if selected && shrinkage { "Model volatility" } else { "Annual volatility" };
        figure.add_trace(json!({
            "type": "scatter", "x": values_json(&[volatility[i]]), "y": values_json(&[mean[i]]), "mode": "markers",
            "name": escape(label),
            "marker": {
                "symbol": if selected { "circle" } else { "diamond" }, "size": if selected { 15 } else { 11 },
                "color": if selected { ACCENT } else { benchmark_color(name, INK_3) },
            },
            "hovertemplate": format!("{risk_label}: %{{x:.2%}}<br>Annual arithmetic mean: %{{y:.2%}}<extra>%{{fullData.name}}</extra>"),
        }));
    }
    if shrinkage {
        figure.data[0]["hovertemplate"] = Value::from(hover.replace("Annual volatility", "Model volatility"));
    }
    figure.add_vline(
        json!(study.settings.f64("target_volatility")),
        json!({"dash": "dot", "color": INK_3}),
        Some(("Volatility budget", "top left")),
    );
    style_chart(&mut figure, ChartStyle { title: Some("Latest fitted frontier · in-sample"), ..Default::default() });
    figure.update_layout(json!({"legend": {"maxheight": 120}}));
    axis_title(&mut figure, "xaxis", "Annual volatility");
    figure.update_xaxes(json!({"tickformat": ".0%", "rangemode": "tozero"}));
    axis_title(&mut figure, "yaxis", "Annual arithmetic mean");
    figure.update_yaxes(json!({"tickformat": ".0%"}));
    figure
}

/// The number in the row labelled `row` and the named column of a strategy table.
fn metric(metrics: &Table, row: &str, column: &str) -> f64 {
    metrics.f64_at(row, column)
}

/// Contrast the latest allocation with the initial buy-and-hold allocation.
pub fn selection_comparison_html(study: &SelectionStudy, prices: &Frame) -> String {
    let latest = &study.latest[&study.selected];
    let replay = &study.test;
    let dates = &replay.equity.index;
    format!(
        "<h2>Fit and replay use different holdings</h2>\
         <p><strong>The latest fit includes the replay dates. \
         The replay did not use the latest holdings.</strong></p>\
         <div class=\"pl-fit-replay\"><article>\
         <h3>Latest fit · in-sample</h3>\
         <p class=\"pl-fit-dates\">{} to {}</p>\
         <dl><dt>Annual arithmetic mean · before fees</dt><dd>{}</dd>\
         <dt>Fitted model volatility</dt><dd>{}</dd></dl>\
         <p>One allocation chosen using the full history. These fixed-weight return estimates describe its starting risk and return, not a buy-and-hold path.</p>\
         </article><article><h3>Historical replay · buy and hold</h3>\
         <p class=\"pl-fit-dates\">{} to {}</p>\
         <dl><dt>Annual arithmetic mean · after fees</dt><dd>{}</dd>\
         <dt>Realized volatility</dt><dd>{}</dd></dl>\
         <p>One initial allocation chosen before the replay. Buy once, pay the entry fee, then hold. \
         Weights drift with prices; there are no later rebalances.</p></article></div>\
         <p>Different dates, allocations, and fee treatment. A high fitted mean does not establish that the rule beats the benchmarks.</p>",
        iso(prices.index[1]),
        iso(prices.index[prices.nrows() - 1]),
        pyfmt::pct(latest.expected_return, 2),
        pyfmt::pct(latest.volatility, 2),
        iso(dates[1]),
        iso(dates[dates.len() - 1]),
        pyfmt::pct(metric(&replay.metrics, &study.selected, "annual_mean"), 2),
        pyfmt::pct(metric(&replay.metrics, &study.selected, "volatility"), 2),
    )
}

/// Compare net total returns and observed risk with each available ETF.
pub fn selection_replay_findings_html(study: &BacktestStudy) -> String {
    let difference = |value: f64, reference: f64| {
        let points = 100.0 * (value - reference);
        if points == 0.0 {
            return "the same".to_string();
        }
        let amount = pyfmt::format(points.abs(), ".2f");
        let size = if amount == "0.00" {
            "less than 0.01 percentage point".to_string()
        } else {
            format!("{amount} percentage points")
        };
        format!("{size} {}", if points > 0.0 { "higher" } else { "lower" })
    };
    let metrics = &study.metrics;
    let mut rows = String::new();
    let mut missing = Vec::new();
    for (symbol, name) in crate::selection::BENCHMARK_NAMES {
        if metrics.row_of(name).is_none() {
            missing.push(symbol);
            continue;
        }
        rows.push_str(&format!(
            "<p><strong>Versus unscaled {symbol} in this replay:</strong> net total return was {}, \
             with realized annual volatility {}.</p>",
            difference(metric(metrics, SELECTED, "total_return"), metric(metrics, name, "total_return")),
            difference(metric(metrics, SELECTED, "volatility"), metric(metrics, name, "volatility")),
        ));
    }
    if !missing.is_empty() {
        rows.push_str(&format!(
            "<p>{} comparison {} unavailable for this replay.</p>",
            missing.join(" and "),
            if missing.len() == 1 { "is" } else { "data are" }
        ));
    }
    format!("<div class=\"pl-replay-findings\">{rows}</div>")
}

/// Classify every observed mean/volatility pair against the stock portfolio.
///
/// Retain original metric values and row order. Dominance requires one strict
/// improvement and no deterioration. The portfolio's own row is equal.
/// Index `strategy`; columns `annual_mean`, `volatility`, `classification`.
pub fn selection_replay_dominance(study: &BacktestStudy) -> Table {
    let mut comparisons = study.metrics.select(&["annual_mean", "volatility"]);
    let (selected_mean, selected_volatility) =
        (metric(&study.metrics, SELECTED, "annual_mean"), metric(&study.metrics, SELECTED, "volatility"));
    let means = comparisons.f64s("annual_mean");
    let volatilities = comparisons.f64s("volatility");
    let classes = means
        .iter()
        .zip(&volatilities)
        .map(|(mean, volatility)| {
            let equal = *mean == selected_mean && *volatility == selected_volatility;
            let class = if equal {
                "equal"
            } else if *mean >= selected_mean && *volatility <= selected_volatility {
                "reference_dominates"
            } else if *mean <= selected_mean && *volatility >= selected_volatility {
                "portfolio_dominates"
            } else {
                "tradeoff"
            };
            Cell::from(class)
        })
        .collect();
    comparisons.push("classification", classes);
    comparisons
}

/// The rows of `prices` on the given dates, in that order.
fn rows_on(prices: &Frame, dates: &[Date]) -> Frame {
    let rows: Vec<Vec<f64>> = dates
        .iter()
        .map(|date| prices.row(prices.index.binary_search(date).expect("replay dates are price dates")))
        .collect();
    Frame::from_rows(dates.to_vec(), prices.columns.clone(), &rows)
}

/// Column means, as `DataFrame.mean()`.
fn column_means(frame: &Frame) -> Vec<f64> {
    (0..frame.ncols()).map(|j| np_mean(&frame.col(j))).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Reconcile initial fitted mean to replay mean in annual arithmetic units.
///
/// Separate changed stock means, weight drift, the cash entry interval and
/// its fee. Use saved initial weights, without refitting or using latest weights.
pub fn selection_replay_return_bridge(study: &SelectionStudy, prices: &Frame) -> ReturnBridge {
    let allocation = &study.test.allocations[&study.selected];
    let weights = Series::new(allocation.columns.clone(), allocation.row(0)).reindex(&prices.columns, 0.0).values;
    let training_end = Date::parse_from_str(study.settings.str("initial_training_end"), "%Y-%m-%d")
        .expect("the training end is an ISO date");
    let training = column_means(&prices.rows_through(training_end).pct_change());
    let replay_prices = rows_on(prices, &study.test.equity.index);
    let returns = replay_prices.pct_change();
    let periods = study.settings.f64("periods_per_year");
    let scale = periods / returns.nrows() as f64;
    let purchase = replay_prices.row(1);
    let positions: Vec<Vec<f64>> = (1..replay_prices.nrows())
        .map(|i| replay_prices.row(i).iter().zip(&purchase).zip(&weights).map(|((p, b), w)| p / b * w).collect())
        .collect();
    let mut drift_rows = Vec::new();
    for k in 1..positions.len() {
        let total: f64 = positions[k - 1].iter().sum();
        let row = returns.row(k);
        let terms: Vec<f64> =
            positions[k - 1].iter().zip(&weights).zip(&row).map(|((p, w), r)| (p / total - w) * r).collect();
        drift_rows.push(terms.iter().filter(|v| !v.is_nan()).sum::<f64>());
    }
    let drift: f64 = drift_rows.iter().sum();
    let fee = study.settings.f64("cost_bps") / 10_000.0;
    let replay_means = column_means(&returns);
    let change: Vec<f64> = replay_means.iter().zip(&training).map(|(r, t)| r - t).collect();
    ReturnBridge {
        fitted_mean: dot(&training, &weights) * periods,
        realized_mean: metric(&study.test.metrics, &study.selected, "annual_mean"),
        mean_change: dot(&change, &weights) * periods,
        weight_drift: drift * scale,
        entry_delay: -dot(&returns.row(0), &weights) * scale,
        entry_fee: -fee / (1.0 + fee) * scale,
    }
}

/// State observed mean/volatility dominance without changing the CAGR checks.
/// Empty text when there is no verdict to show.
pub fn selection_replay_verdict_html(study: &BacktestStudy) -> String {
    let metrics = &study.metrics;
    let (selected_mean, selected_volatility) =
        (metric(metrics, SELECTED, "annual_mean"), metric(metrics, SELECTED, "volatility"));
    let names = metrics.index_labels();
    let means = metrics.f64s("annual_mean");
    let volatilities = metrics.f64s("volatility");
    let dominates: Vec<usize> = (0..names.len())
        .filter(|i| names[*i] != SELECTED)
        .filter(|i| {
            means[*i] >= selected_mean
                && volatilities[*i] <= selected_volatility
                && (means[*i] > selected_mean || volatilities[*i] < selected_volatility)
        })
        .collect();
    let Some(best) = dominates.iter().copied().fold(None, |best: Option<usize>, i| match best {
        Some(b) if means[b] >= means[i] => Some(b),
        _ => Some(i),
    }) else {
        return String::new();
    };
    format!(
        "<p class=\"pl-replay-verdict\"><strong>{} dominated the stock rule on mean and volatility in this replay.</strong><br>\
         Annual arithmetic mean: {} versus {}. Realized volatility: {} versus {}.</p>",
        escape(&names[best]),
        pyfmt::pct(means[best], 2),
        pyfmt::pct(selected_mean, 2),
        pyfmt::pct(volatilities[best], 2),
        pyfmt::pct(selected_volatility, 2),
    )
}

/// Every value as a percent with two decimals, or an em dash when missing, in a report table.
fn percent_table(frame: &Table) -> String {
    let mut display = frame.clone();
    for column in &mut display.data {
        for cell in column.iter_mut() {
            let text =
                if cell.is_null() { "—".to_string() } else { pyfmt::pct(cell.as_f64().unwrap_or(f64::NAN), 2) };
            *cell = Cell::Str(text);
        }
    }
    data_table(&display)
}

/// Explain the frontier and show references on the same dates and return measure.
pub fn selection_frontier_summary_html(study: &SelectionStudy, prices: &Frame, frontier: &SelectionFrontier) -> String {
    let shrinkage = study.settings.f64("shrinkage");
    let columns: &[&str] = if shrinkage != 0.0 {
        &["fitted_mean", "model_volatility", "historical_volatility"]
    } else {
        &["fitted_mean", "historical_volatility"]
    };
    let mut display = frontier.comparisons.select(columns);
    display.rename_columns(&FIT_METRIC_LABELS);
    display.rename_index(&[(study.selected.as_str(), LATEST_LABEL)]);
    let risk_note = if shrinkage != 0.0 {
        format!(
            "Stock model volatility uses {} covariance shrinkage; ETF volatility does not. \
             Historical volatility uses ordinary sample covariance for every row. \
             The frontier and budget use stock model volatility, so their risk estimates differ from the ETF markers.",
            pyfmt::pct(shrinkage, 0)
        )
    } else {
        "Every row uses ordinary sample covariance, the same return dates, and the same annualization. \
         The frontier and volatility budget use this same risk measure."
            .to_string()
    };
    let notes: String = frontier.warnings.iter().map(|warning| format!("<li>{}</li>", escape(warning))).collect();
    let mut html = format!(
        "<h2>Latest fitted frontier · in-sample</h2>\
         <p>Return dates: {} to {}. \
         The same dates choose and evaluate these weights, before trading fees. \
         This is an in-sample optimum, not a historical trading result.</p>\
         <p>The line shows the highest fitted mean available at each volatility among the eligible stocks, \
         with no short selling and at most {} in one stock. \
         The selected point maximizes that mean within the volatility budget. \
         These limits apply to the starting allocation; buy-and-hold weights and realized risk can later exceed them.</p>\
         <p>{risk_note}</p><div class=\"scroll\">{}</div>\
         <p>SPY and QQQ use the same fit dates. They are not portfolio candidates and do not follow the stock holding limit. \
         None of the ETF positions is scaled to match the selected volatility.</p>",
        iso(prices.index[1]),
        iso(prices.index[prices.nrows() - 1]),
        pyfmt::pct(study.settings.f64("max_weight"), 0),
        percent_table(&display),
    );
    if !notes.is_empty() {
        html.push_str(&format!("<ul>{notes}</ul>"));
    }
    html
}

/// The reasons in order, without repeats (`list(dict.fromkeys(items))`).
fn unique(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for item in items {
        if !seen.contains(&item) {
            seen.push(item);
        }
    }
    seen
}

/// Describe the one model and its later historical check.
pub fn selection_summary_html(study: &SelectionStudy, metadata: &Dict) -> String {
    let heading =
        if study.supported { "Replay: performance checks passed" } else { "Replay: performance checks not met" };
    let notes: String =
        unique(study.reasons.iter().cloned()).iter().map(|reason| format!("<li>{}</li>", escape(reason))).collect();
    let mut html = format!(
        "<h2>{heading}</h2><p>Annual volatility budget: {}. \
         Largest holding: {}. \
         The model maximizes the fitted historical mean within the volatility budget and holding limit.</p>\
         <p>The budget limits estimated volatility. Future volatility and losses can exceed it. \
         The later test checks the same rule against equal-weight stocks, SPY, and QQQ. \
         These are performance checks, not checks of optimizer accuracy.</p>\
         <p>A pass requires both ETFs and at least one year of test observations. \
         Realized test volatility must stay within the budget. No reference can have higher CAGR with equal or lower volatility.</p>",
        pyfmt::pct(study.settings.f64("target_volatility"), 2),
        pyfmt::pct(study.settings.f64("max_weight"), 1),
    );
    if !notes.is_empty() {
        html.push_str(&format!("<ul>{notes}</ul>"));
    }
    if let Some(note) = metadata.get("benchmarks_note").filter(|note| truthy(Some(note))) {
        html.push_str(&format!("<p>{}</p>", escape(&note.py_str())));
    }
    html.push_str(&format!("<p class=\"source\">{HISTORY_NOTE}</p>"));
    html
}

/// `f"{value:.{digits}%}"` with a true minus sign.
pub fn percent(value: f64, digits: usize) -> String {
    crate::pyfmt::format(value, &format!(".{digits}%")).replace('-', "−")
}

/// A difference of two rates in percentage points, with a true minus sign.
pub fn points(value: f64) -> String {
    format!("{} pp", crate::pyfmt::format(100.0 * value, "+.2f")).replace('-', "−")
}

fn date_range(start: Date, end: Date) -> String {
    format!("{} to {}", iso(start), iso(end))
}

/// The row of each completed window in `result.windows`, in the order of `windows`.
fn window_rows(result: &OutOfSampleStudy, windows: &[Cell]) -> Vec<usize> {
    windows
        .iter()
        .map(|window| result.windows.index[0].iter().position(|cell| cell == window).expect("a completed window"))
        .collect()
}

fn window_date(result: &OutOfSampleStudy, column: &str, row: usize) -> Date {
    result.windows.column(column)[row].as_date().expect("window dates")
}

fn window_labels(result: &OutOfSampleStudy, windows: &[Cell]) -> Vec<String> {
    window_rows(result, windows)
        .into_iter()
        .zip(windows)
        .map(|(row, number)| {
            let (start, end) = (window_date(result, "test_start", row), window_date(result, "test_end", row));
            format!(
                "{} · {} ’{}–{} ’{}",
                number.py_str(),
                start.format("%b"),
                start.format("%y"),
                end.format("%b"),
                end.format("%y")
            )
        })
        .collect()
}

fn floats(values: &[f64]) -> Vec<Cell> {
    values.iter().map(|value| Cell::Float(*value)).collect()
}

/// One row per completed window: the stock portfolio on both sides, then equal-weight stocks on both sides.
pub fn out_of_sample_frame(result: &OutOfSampleStudy) -> Table {
    let means = portfolio_comparison(result, "annual_mean");
    let columns =
        ["Test dates", "In sample", "Out of sample", "Change", "Equal weight in sample", "Equal weight out of sample"];
    if means.nrows() == 0 {
        let mut table = Table::new(Some("Window"), Vec::new());
        for column in columns {
            table.push(column, Vec::new());
        }
        return table;
    }
    let windows = means.index[0].clone();
    let inside = sample_values(result, "in_sample", "annual_mean");
    let dates = window_rows(result, &windows)
        .into_iter()
        .map(|row| Cell::Str(date_range(window_date(result, "test_start", row), window_date(result, "test_end", row))))
        .collect();
    Table::new(Some("Window"), windows)
        .with("Test dates", dates)
        .with("In sample", means.column("in_sample").to_vec())
        .with("Out of sample", means.column("out_of_sample").to_vec())
        .with("Change", means.column("change").to_vec())
        .with("Equal weight in sample", inside.column(BASELINE).to_vec())
        .with("Equal weight out of sample", means.column(BASELINE).to_vec())
}

/// Every path in every completed window: annual mean and volatility on both sides.
pub fn out_of_sample_paths_frame(result: &OutOfSampleStudy) -> Table {
    let windows = result.completed_windows();
    let paths = result.strategies();
    let (mut numbers, mut labels) = (Vec::new(), Vec::new());
    let mut rows: Vec<[f64; 5]> = Vec::new();
    for window in &windows {
        for path in &paths {
            let value = |sample: &str, measure: &str| result.value(*window, sample, path, measure);
            let (inside, outside) = (value("in_sample", "annual_mean"), value("out_of_sample", "annual_mean"));
            numbers.push(Cell::Int(*window));
            labels.push(Cell::from(
                PATH_LABELS.iter().find(|(name, _)| name == path).map_or(path.as_str(), |(_, label)| label),
            ));
            rows.push([
                inside,
                outside,
                outside - inside,
                value("in_sample", "volatility"),
                value("out_of_sample", "volatility"),
            ]);
        }
    }
    let mut table = Table::multi(vec![Some("Window".into()), Some("Path".into())], vec![numbers, labels]);
    let names = ["In-sample mean", "Out-of-sample mean", "Change", "In-sample volatility", "Out-of-sample volatility"];
    for (k, name) in names.iter().enumerate() {
        table.push(*name, rows.iter().map(|row| Cell::Float(row[k])).collect());
    }
    table
}

/// The display format of one column of the two frames above: `points` for
/// "Change", none for "Test dates", and `percent` (2 digits) otherwise.
pub fn out_of_sample_format(column: &str) -> Option<fn(f64) -> String> {
    match column {
        "Test dates" => None,
        "Change" => Some(points),
        _ => Some(|value| percent(value, 2)),
    }
}

/// Average the completed windows and count each comparison. `None` when no window completed.
pub fn out_of_sample_findings(result: &OutOfSampleStudy) -> Option<OutOfSampleFindings> {
    let means = portfolio_comparison(result, "annual_mean");
    if means.is_empty() {
        return None;
    }
    let inside = means.f64s("in_sample");
    let outside = means.f64s("out_of_sample");
    let change = means.f64s("change");
    let equal_in = sample_values(result, "in_sample", "annual_mean").f64s(BASELINE);
    let equal_out = means.f64s(BASELINE);
    let difference = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<f64>>();
    let references =
        means.columns.iter().filter(|name| !["in_sample", "out_of_sample", "change"].contains(&name.as_str()));
    let higher_than = references
        .map(|name| {
            let reference = means.f64s(name);
            (name.clone(), outside.iter().zip(&reference).filter(|(o, r)| o > r).count())
        })
        .collect();
    let rows = window_rows(result, &means.index[0]);
    let test_returns: Vec<f64> =
        rows.iter().map(|row| result.windows.column("test_returns")[*row].as_f64().unwrap_or(f64::NAN)).collect();
    Some(OutOfSampleFindings {
        completed: means.nrows(),
        windows: result.windows.nrows(),
        in_sample: np_mean(&inside),
        out_of_sample: np_mean(&outside),
        change: np_mean(&change),
        lower: change.iter().filter(|value| **value < 0.0).count(),
        margin_in_sample: np_mean(&difference(&inside, &equal_in)),
        margin_out_of_sample: np_mean(&difference(&outside, &equal_out)),
        higher_than,
        test_returns: np_mean(&test_returns).round_ties_even() as i64,
    })
}

fn join(parts: &[String]) -> String {
    match parts.len() {
        0..=2 => parts.join(" and "),
        n => format!("{}, and {}", parts[..n - 1].join(", "), parts[n - 1]),
    }
}

/// The headline and the explanation sentences.
pub fn out_of_sample_text(findings: &OutOfSampleFindings) -> (String, Vec<String>) {
    let count = findings.completed;
    let headline = format!(
        "Out of sample, the stock portfolio averaged {} a year. In sample, it averaged {}.",
        percent(findings.out_of_sample, 1),
        percent(findings.in_sample, 1)
    );
    let mut lines = vec![
        format!(
            "The out-of-sample mean was lower than the in-sample mean in {} of {count} window{}.",
            findings.lower,
            if count != 1 { "s" } else { "" }
        ),
        format!(
            "Equal-weight stocks use no return estimates. The stock portfolio’s average margin over them was \
             {} a year in sample and {} out of sample.",
            points(findings.margin_in_sample),
            points(findings.margin_out_of_sample)
        ),
    ];
    let parts: Vec<String> = findings
        .higher_than
        .iter()
        .map(|(name, wins)| {
            format!("{} in {wins} of {count}", if name == BASELINE { "equal-weight stocks" } else { name })
        })
        .collect();
    lines.push(format!("Out of sample, the stock portfolio had a higher annual mean than {}.", join(&parts)));
    (headline, lines)
}

/// Plot each window's in-sample and out-of-sample annual mean on one axis, with references out of sample.
pub fn out_of_sample_chart(result: &OutOfSampleStudy) -> Figure {
    let means = portfolio_comparison(result, "annual_mean");
    let windows = if means.nrows() == 0 { Vec::new() } else { means.index[0].clone() };
    let labels = window_labels(result, &windows);
    let rows = window_rows(result, &windows);
    let dates = |start: &str, end: &str| -> Vec<String> {
        rows.iter().map(|row| date_range(window_date(result, start, *row), window_date(result, end, *row))).collect()
    };
    let (fit_dates, test_dates) = (dates("fit_start", "fit_end"), dates("test_start", "test_end"));
    let hover = |kind: &str| {
        format!(
            "Window %{{customdata[0]}} · {kind} %{{customdata[1]}}<br>%{{fullData.name}}: %{{x:.2%}}<extra></extra>"
        )
    };
    let custom = |dates: &[String]| -> Value {
        Value::Array(windows.iter().zip(dates).map(|(window, text)| json!([window.as_int(), text])).collect())
    };
    let column = |name: &str| if means.nrows() == 0 { Vec::new() } else { means.f64s(name) };
    let (inside, outside) = (column("in_sample"), column("out_of_sample"));
    let mut connectors_x = Vec::new();
    let mut connectors_y = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        connectors_x.extend([json!(inside[i]), json!(outside[i]), Value::Null]);
        connectors_y.extend([json!(label), json!(label), Value::Null]);
    }
    let mut figure = Figure::with_trace(json!({
        "type": "scatter", "x": connectors_x, "y": connectors_y, "mode": "lines", "line": {"color": REST, "width": 2},
        "hoverinfo": "skip", "showlegend": false,
    }));
    for name in [BASELINE, SPY, QQQ] {
        if means.position(name).is_none() {
            continue;
        }
        let symbol =
            OUT_OF_SAMPLE_SYMBOLS.iter().find(|(path, _)| *path == name).map_or("circle", |(_, symbol)| symbol);
        let color = if name == BASELINE { INK_3 } else { benchmark_color(name, INK_3) };
        figure.add_trace(json!({
            "type": "scatter", "x": values_json(&means.f64s(name)), "y": labels, "mode": "markers",
            "name": escape(&format!("{name} · out of sample")),
            "marker": {"symbol": symbol, "size": 10, "color": color, "line": {"width": 1.5, "color": "#ffffff"}},
            "customdata": custom(&test_dates), "hovertemplate": hover("test dates"),
        }));
    }
    figure.add_trace(json!({
        "type": "scatter", "x": values_json(&inside), "y": labels, "mode": "markers", "name": "Stock portfolio · in sample",
        "marker": {"symbol": "circle-open", "size": 14, "color": ACCENT, "line": {"width": 2.5}},
        "customdata": custom(&fit_dates), "hovertemplate": hover("fit dates"),
    }));
    figure.add_trace(json!({
        "type": "scatter", "x": values_json(&outside), "y": labels, "mode": "markers",
        "name": "Stock portfolio · out of sample",
        "marker": {"symbol": "circle", "size": 14, "color": ACCENT, "line": {"width": 2, "color": "#ffffff"}},
        "customdata": custom(&test_dates), "hovertemplate": hover("test dates"),
    }));
    figure.add_vline(json!(0), json!({"color": INK_3, "width": 1}), None);
    let count = labels.len() as u32;
    style_chart(&mut figure, ChartStyle { height: 190 + 58 * count, ..Default::default() });
    figure.update_layout(json!({
        "legend": {"maxheight": 120, "y": -90.0 / (58.0 * count as f64 + 86.0), "yanchor": "top"}, "margin": {"b": 130},
    }));
    axis_title(&mut figure, "xaxis", "Annual mean while held");
    figure.update_xaxes(json!({"tickformat": ".0%", "showgrid": true, "gridcolor": GRID}));
    axis_title(&mut figure, "yaxis", "Test window");
    figure.update_yaxes(json!({"type": "category", "autorange": "reversed", "showgrid": false}));
    figure
}

/// Format every column except "Test dates" for display.
fn formatted(frame: &Table) -> Table {
    let mut display = frame.clone();
    for (name, column) in display.columns.iter().zip(display.data.iter_mut()) {
        if let Some(format) = out_of_sample_format(name) {
            for cell in column.iter_mut() {
                *cell = Cell::Str(format(cell.as_f64().unwrap_or(f64::NAN)));
            }
        }
    }
    display
}

/// Report section with the walk-forward findings, the window tables, and the chart.
///
/// `result` is the walk-forward result, or the reason that it is unavailable.
pub fn out_of_sample_html(result: std::result::Result<&OutOfSampleStudy, &str>) -> String {
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            return format!(
                "<section id=\"out-of-sample\"><h2>{OUT_OF_SAMPLE_TITLE}</h2><p>{}</p></section>",
                escape(error)
            );
        }
    };
    let findings = out_of_sample_findings(result);
    let statuses = result.windows.column("status");
    let mut notes: String = result.windows.index[0]
        .iter()
        .zip(statuses)
        .filter(|(_, status)| status.as_str() != Some(""))
        .map(|(number, status)| {
            format!("<li>Window {} has no result. {}</li>", number.py_str(), escape(&status.py_str()))
        })
        .collect();
    notes.extend(result.notes.iter().map(|note| format!("<li>{}</li>", escape(note))));
    let method = format!(
        "<p>Each of {} windows fits the model on earlier stock prices only. \
         The volatility budget and holding limit are the same fixed inputs for every window. It buys once at the next close \
         and holds through the window without trades. The in-sample result buys the same weights at the start of the fit dates \
         and holds them to the end of the fit. Both sides are measured from the purchase close, so the cash interval and the \
         entry fee do not enter these figures. \
         The test windows do not overlap and cover the final {} of returns.</p>",
        result.windows.nrows(),
        pyfmt::pct(1.0 - result.settings.f64("first_fit_fraction"), 0)
    );
    let Some(findings) = findings else {
        return format!(
            "<section id=\"out-of-sample\"><h2>{OUT_OF_SAMPLE_TITLE}</h2>{method}\
             <p>No window had a feasible fit.</p><ul>{notes}</ul></section>"
        );
    };
    let (headline, lines) = out_of_sample_text(&findings);
    let chart = out_of_sample_chart(result).to_html("chart-out-of-sample", &plotly_config());
    let main = data_table(&formatted(&out_of_sample_frame(result)));
    let paths = data_table(&formatted(&out_of_sample_paths_frame(result)));
    let mut html = format!(
        "<section id=\"out-of-sample\"><h2>{OUT_OF_SAMPLE_TITLE}</h2><p><strong>{}</strong></p>",
        escape(&headline)
    );
    for line in &lines {
        html.push_str(&format!("<p>{}</p>", escape(line)));
    }
    html.push_str(&method);
    html.push_str(&format!(
        "<div class=\"chart\">{chart}</div><h3>Annual mean while held</h3><div class=\"scroll\">{main}</div>\
         <h3>Every path in each window</h3><div class=\"scroll\">{paths}</div>"
    ));
    if !notes.is_empty() {
        html.push_str(&format!("<ul>{notes}</ul>"));
    }
    html.push_str(&format!(
        "<p>Each fit starts on {} and ends at the close before its test dates. \
         Each window is short, so its annual mean is noisy. These are historical checks on dates you can already know, \
         not forecasts.</p></section>",
        iso(window_date(result, "fit_start", 0))
    ));
    html
}

/// Separate fitted estimates from the later results of earlier weights.
pub fn selection_fit_summary_html(study: &SelectionStudy, prices: &Frame) -> String {
    let estimates = selection_fit_estimates(study, prices);
    let fits = estimates.take_rows(&[0, estimates.nrows() - 1]);
    let shrinkage = study.settings.f64("shrinkage");
    let columns: Vec<&str> = if shrinkage != 0.0 {
        FIT_METRIC_LABELS.iter().map(|(column, _)| *column).collect()
    } else {
        vec!["fitted_mean", "historical_volatility"]
    };
    let mut display = fits.select(&columns);
    display.rename_columns(&FIT_METRIC_LABELS);
    display.index_names = vec![None];
    display.index = vec![
        (0..fits.nrows())
            .map(|i| {
                let date = fits.index[0][i].as_date().map_or_else(String::new, iso);
                Cell::Str(format!("{} · {date}", fits.column("phase")[i].py_str()))
            })
            .collect(),
    ];
    let risk_note = if shrinkage != 0.0 {
        format!(
            "Model volatility uses a {} covariance adjustment toward individual stock variances. \
             Historical volatility uses the ordinary sample covariance with the same dates and weights. \
             The budget applies to model volatility.",
            pyfmt::pct(shrinkage, 0)
        )
    } else {
        "Historical volatility uses ordinary sample covariance, with no covariance adjustment. \
         The volatility budget applies to this same measure."
            .to_string()
    };
    format!(
        "<h2>Estimates at each fit</h2>\
         <p>Each fit uses returns from {} through its date below. \
         These estimates describe the data used for each fit. They are not forecasts.</p>\
         <div class=\"scroll\">{}</div>\
         <p>The model maximizes the fitted arithmetic mean within the model volatility budget and holding limit. \
         It does not maximize the return-to-volatility ratio or CAGR.</p>\
         <p>{risk_note}</p>\
         <p>The initial fit supplies the only purchase in the historical replay. The holdings then drift with prices. \
         The latest weights did not produce the earlier test results.</p>",
        iso(prices.index[1]),
        percent_table(&display)
    )
}

/// Define comparison holdings and their different risk levels.
pub fn selection_benchmark_note(study: &SelectionStudy) -> String {
    let _ = study;
    "Equal-weight stocks divides money equally across eligible stocks at entry, then lets the weights drift. \
     SPY tracks the S&P 500. QQQ tracks the Nasdaq-100. All four paths buy once and hold, \
     with the same entry delay and fee rate. \
     Each reference has its own volatility. None is scaled to the model volatility budget."
        .to_string()
}

/// The latest weights (index `symbol`) and estimates (index `strategy`) of each latest portfolio.
fn latest_frames(study: &SelectionStudy) -> (Table, Table) {
    let mut symbols: Vec<String> = Vec::new();
    for portfolio in study.latest.values() {
        for label in &portfolio.weights.labels {
            if !symbols.contains(label) {
                symbols.push(label.clone());
            }
        }
    }
    let mut weights = Table::labelled(Some("symbol"), &symbols);
    for (name, portfolio) in &study.latest {
        weights.push(name, symbols.iter().map(|symbol| Cell::from(portfolio.weights.get(symbol))).collect());
    }
    let names: Vec<&String> = study.latest.keys().collect();
    let field =
        |value: fn(&crate::core::Portfolio) -> f64| floats(&study.latest.values().map(value).collect::<Vec<_>>());
    let estimates = Table::labelled(Some("strategy"), &names)
        .with("expected_return", field(|p| p.expected_return))
        .with("volatility", field(|p| p.volatility))
        .with("sharpe", field(|p| p.sharpe));
    (weights, estimates)
}

/// `pd.DataFrame(records)` for a list of JSON objects: one column per key in order
/// of first appearance. Numbers with missing values become floats with NaN, as in pandas.
pub fn records_table(records: &[Json]) -> Table {
    let mut columns: Vec<&str> = Vec::new();
    for record in records {
        for key in record.as_dict().map(|fields| fields.keys().collect::<Vec<_>>()).unwrap_or_default() {
            if !columns.contains(&key) {
                columns.push(key);
            }
        }
    }
    let mut table = Table::range(records.len());
    for column in columns {
        let values: Vec<Option<&Json>> =
            records.iter().map(|record| record.as_dict().and_then(|fields| fields.get(column))).collect();
        let missing = |value: &Option<&Json>| value.is_none_or(|value| value.is_null());
        let numeric = values.iter().all(|value| missing(value) || matches!(value, Some(Json::Int(_) | Json::Float(_))));
        let floats = numeric
            && values.iter().any(|value| !missing(value))
            && values.iter().any(|value| missing(value) || matches!(value, Some(Json::Float(_))));
        let cells = values
            .iter()
            .map(|value| match value {
                None | Some(Json::Null) if floats => Cell::Float(f64::NAN),
                None | Some(Json::Null) => Cell::Null,
                Some(Json::Int(number)) if floats => Cell::Float(*number as f64),
                Some(Json::Int(number)) => Cell::Int(*number),
                Some(Json::Float(number)) => Cell::Float(*number),
                Some(Json::Bool(flag)) => Cell::Bool(*flag),
                Some(Json::Str(text)) => Cell::Str(text.clone()),
                Some(other) => Cell::Str(other.py_repr()),
            })
            .collect();
        table.push(column, cells);
    }
    table
}

fn coverage_table(metadata: &Dict) -> Option<Table> {
    let records = metadata.get("universe_coverage").filter(|value| truthy(Some(value)))?;
    Some(records_table(records.as_list().unwrap_or_default()))
}

fn details(study: &SelectionStudy, prices: &Frame, metadata: &Dict) -> Dict {
    let result_files: Dict = Dict(
        study
            .test
            .equity
            .columns
            .iter()
            .enumerate()
            .map(|(i, name)| (format!("test/{:02}", i + 1), Json::from(name.as_str())))
            .collect(),
    );
    let warnings = unique(study.warnings.iter().chain(&study.test.warnings).cloned().chain(stale_price_notes(prices)));
    let mut details = metadata.clone();
    details.insert("study", "Single historical-mean portfolio");
    details.insert("settings", study.settings.clone());
    details.insert("selected", study.selected.as_str());
    details.insert("supported", study.supported);
    details.insert("reasons", study.reasons.clone());
    details.insert("warnings", warnings);
    details.insert("interpretation", HISTORY_NOTE);
    details.insert("price_start", iso(prices.index[0]));
    details.insert("price_end", iso(prices.index[prices.nrows() - 1]));
    details.insert("price_symbols", prices.columns.clone());
    details.insert("test", study.test.settings.clone());
    details.insert("result_files", result_files);
    details
}

/// The supplied walk-forward result, or a new one. A too-short history gives its reason.
fn walk_forward(
    study: &SelectionStudy,
    prices: &Frame,
    supplied: Option<&OutOfSampleStudy>,
) -> Result<std::result::Result<OutOfSampleStudy, String>> {
    if let Some(result) = supplied {
        return Ok(Ok(result.clone()));
    }
    match run_default_out_of_sample(study, prices) {
        Ok(result) => Ok(Ok(result)),
        Err(Error::Value(reason)) => Ok(Err(reason)),
        Err(other) => Err(other),
    }
}

fn frontier_or_new(
    study: &SelectionStudy,
    prices: &Frame,
    frontier: Option<&SelectionFrontier>,
) -> Result<SelectionFrontier> {
    match frontier {
        Some(frontier) => Ok(frontier.clone()),
        None => selection_frontier(study, prices, 30),
    }
}

/// Show the latest fitted stock weights, the walk-forward tests, and one later replay.
///
/// `frontier` and `out_of_sample` reuse earlier results; `None` calculates them.
pub fn selection_report_html(
    study: &SelectionStudy,
    prices: &Frame,
    metadata: &Dict,
    frontier: Option<&SelectionFrontier>,
    out_of_sample: Option<&OutOfSampleStudy>,
) -> Result<String> {
    let frontier = frontier_or_new(study, prices, frontier)?;
    let out_of_sample = walk_forward(study, prices, out_of_sample)?;
    Ok(report_html(study, prices, metadata, &frontier, &out_of_sample))
}

fn report_html(
    study: &SelectionStudy,
    prices: &Frame,
    metadata: &Dict,
    frontier: &SelectionFrontier,
    out_of_sample: &std::result::Result<OutOfSampleStudy, String>,
) -> String {
    let details = details(study, prices, metadata);
    let config = plotly_config();
    let out_of_sample_files = match out_of_sample {
        Err(_) => "There are no out_of_sample files because the history is too short for the walk-forward windows.\n",
        Ok(result) if result.metrics.nrows() == 0 => {
            "out_of_sample/windows.csv records each walk-forward window and the reason it has no result.\n"
        }
        Ok(_) => {
            "out_of_sample/windows.csv, metrics.csv, weights.csv, and eligibility.csv record each \
             walk-forward window, its results, its allocation, and its eligibility decisions.\n"
        }
    };
    let frontier_chart = selection_frontier_chart(frontier, study).to_html("chart-latest-frontier", &config);
    let result = &study.test;
    let mut metrics = selection_metrics_frame(result);
    metrics.rename_columns(&SELECTION_METRIC_LABELS);
    metrics.rename_index(&[(study.selected.as_str(), REPLAY_LABEL)]);
    let chart = selection_chart(result).to_html("chart-replay", &config);
    let dates = &result.equity.index;
    let test_section = format!(
        "<section id=\"test\"><h2>Historical replay · buy and hold</h2>\
         <p>The rule chooses one initial allocation using earlier prices, then holds it through the replay. \
         This path does not use the latest fitted weights.</p>{}\
         <p>Baseline: {}. Evaluation returns: {} to {}.</p>\
         <p>All rows use the same replay dates and fee rate.</p><p>{}</p>\
         <p>Arithmetic means average the net period returns, including entry fees. CAGR compounds the same returns. \
         The table keeps a fixed order and does not choose a model from the results.</p>\
         <div class=\"scroll\">{}</div><div class=\"chart\">{chart}</div></section>",
        selection_replay_verdict_html(result),
        iso(dates[0]),
        iso(dates[1]),
        iso(dates[dates.len() - 1]),
        selection_benchmark_note(study),
        percent_table(&metrics),
    );
    let (weights, estimates) = latest_frames(study);
    let order = Series::new(weights.index_labels(), weights.f64s(&study.selected)).descending_order();
    let mut estimates_shown = estimates.select(&["expected_return", "volatility"]);
    estimates_shown.rename_columns(&[
        ("expected_return", "Fitted arithmetic mean / year"),
        ("volatility", "Fitted volatility / year"),
    ]);
    let latest = format!(
        "<p>These weights fit all supplied history through the final price date. The replay bought a separate initial allocation once. \
         Latest fitted means are annual arithmetic estimates, not realized CAGR or forecasts.</p>\
         <p>All stocks are shown, largest weight first. Displayed weights are rounded; \
         latest_weights.csv preserves every weight at full precision.</p>\
         <div class=\"scroll\">{}</div><div class=\"scroll\">{}</div>",
        percent_table(&weights.take_rows(&order)),
        percent_table(&estimates_shown)
    );
    let mut fit_history = selection_fit_estimates(study, prices);
    for column in ["fitted_mean", "model_volatility", "historical_volatility", "max_weight"] {
        let values = fit_history.f64s(column).iter().map(|value| Cell::Str(pyfmt::pct(*value, 2))).collect();
        fit_history.push(column, values);
    }
    let execution = fit_history
        .column("execution_date")
        .iter()
        .map(|cell| Cell::Str(cell.as_date().map_or_else(|| "—".to_string(), iso)))
        .collect();
    fit_history.push("execution_date", execution);
    fit_history.index = vec![fit_history.index[0].iter().map(|cell| Cell::Str(cell.py_str())).collect()];
    fit_history.index_names = vec![Some("Fit through".into())];
    fit_history.rename_columns(&[
        ("fitted_mean", "Fitted annual mean"),
        ("model_volatility", "Model volatility"),
        ("historical_volatility", "Historical volatility"),
        ("phase", "Fit"),
        ("execution_date", "Execution date"),
        ("observations", "Return observations"),
        ("max_weight", "Largest holding"),
    ]);
    let warnings = details.get("warnings").and_then(Json::as_list).unwrap_or_default();
    let notes: String = warnings.iter().map(|note| format!("<li>{}</li>", escape(&note.py_str()))).collect();
    let source_rows: String = metadata
        .iter()
        .filter(|(key, _)| *key != "universe_coverage")
        .map(|(key, value)| {
            let text = match value {
                Json::List(_) | Json::Dict(_) => value.dumps(None),
                other => other.py_str(),
            };
            format!("<tr><th>{}</th><td>{}</td></tr>", escape(&key.replace('_', " ")), escape(&text))
        })
        .collect();
    let coverage = coverage_table(metadata).map_or_else(String::new, |table| {
        format!(
            "<details><summary>Universe coverage</summary><div class=\"scroll\">{}</div></details>",
            to_html(&table, &HtmlOptions::REPORT_NO_INDEX)
        )
    });
    let settings = &study.settings;
    let out_of_sample_section = out_of_sample_html(out_of_sample.as_ref().map_err(String::as_str));
    let notes_aside = if notes.is_empty() {
        String::new()
    } else {
        format!("<aside><h3>Data and model notes</h3><ul>{notes}</ul></aside>")
    };
    format!(
        "<!doctype html>\n\
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
<title>Portfolio Lab · Historical-mean portfolio</title><style>{css}</style><script>{PLOTLY_JS}</script></head>\n\
<body><main><p class=\"eyebrow\">Historical research</p><h1>One model: latest fit and historical replay</h1>\n\
<button onclick=\"window.print()\">Print / save PDF</button>\n\
<section id=\"fit-and-replay\">{comparison}</section>\n\
<section id=\"latest-frontier\">{frontier_summary}\n\
<div class=\"chart\">{frontier_chart}</div></section>\n\
<section id=\"latest-selection\"><h2>Latest fitted stock weights · {last_date}</h2>{latest}</section>\n\
{out_of_sample_section}\n\
{test_section}\n\
<section id=\"selection-assumptions\"><h2>Method and limits</h2>\n\
<details id=\"selection-decision\"><summary>Historical performance checks</summary>{summary}</details>\n\
<details id=\"fit-estimates\"><summary>First and latest fitted estimates</summary>{fit_summary}</details>\n\
<p>The model maximizes historical arithmetic mean return under a fitted volatility budget and a per-stock holding limit.\n\
Stock weights are long-only and sum to one at entry. There are no later rebalances.</p>\n\
<p>Initial training share: {train_fraction}.\n\
Holding policy: buy once and hold.\n\
Fee rate: {cost_bps} basis points ({cost_rate}) per amount bought or sold.\n\
Covariance shrinkage: {shrinkage}.\n\
Annualization: {periods} observations per year.</p>\n\
<p>The initial allocation uses prices through the close before entry. The first evaluation interval stays in cash.\n\
The entry fee applies to the amount invested. There are no later trades or terminal sale. Cash earns zero interest.\n\
Equal-weight stocks, SPY, and QQQ also buy once and hold. Position weights drift after entry.\n\
Taxes, currency conversion, liquidity, and future membership changes are not modeled.</p>\n\
<p>Past-only eligibility checks can exclude long unchanged-price runs at a fit. eligibility.csv records each decision for the initial and latest fits.\n\
The original input prices and excluded asset columns remain in the export.</p>\n\
<details><summary>Estimates for every saved fit</summary><div class=\"scroll\">\n\
{fit_history}</div></details>\n\
<p>Current constituent lists and complete-history requirements introduce selection and survivorship bias.\n\
The later test is a descriptive historical check, not proof of future outperformance.</p>\n\
{notes_aside}</section>\n\
<section id=\"selection-sources\"><h2>Source and reproduction</h2><div class=\"scroll\"><table class=\"metadata\">{source_rows}</table></div>{coverage}\n\
<p>metadata.json records the model revision, settings, input dates, source details, test reasons, walk-forward settings and notes, and every numbered result file.\n\
fit_estimates.csv records the mean and both volatility measures for every saved fit.\n\
{out_of_sample_files}latest/frontier.csv and latest/frontier_weights.csv preserve each plotted stock mix.\n\
latest/fitted_comparison.csv records the same-date portfolio and ETF estimates.\n\
The test exports include every path, allocation, trade, and holding.</p></section>\n\
<footer>This report works offline.<details><summary>Third-party notice: Plotly.js (MIT)</summary>\n\
<pre>{license}</pre></details></footer></main></body></html>",
        css = report_css(),
        comparison = selection_comparison_html(study, prices),
        frontier_summary = selection_frontier_summary_html(study, prices, frontier),
        last_date = iso(prices.index[prices.nrows() - 1]),
        summary = selection_summary_html(study, metadata),
        fit_summary = selection_fit_summary_html(study, prices),
        train_fraction = pyfmt::pct(settings.f64("train_fraction"), 0),
        cost_bps = pyfmt::g(settings.f64("cost_bps")),
        cost_rate = pyfmt::pct(settings.f64("cost_bps") / 10_000.0, 2),
        shrinkage = pyfmt::pct(settings.f64("shrinkage"), 0),
        periods = pyfmt::g(settings.f64("periods_per_year")),
        fit_history = data_table(&fit_history),
        license = escape(PLOTLY_JS_LICENSE),
    )
}

/// Export exact inputs, model settings, every walk-forward window, and every later-test result.
/// `frontier` and `out_of_sample` reuse earlier results; `None` calculates them.
pub fn selection_report_zip(
    study: &SelectionStudy,
    prices: &Frame,
    metadata: &Dict,
    frontier: Option<&SelectionFrontier>,
    out_of_sample: Option<&OutOfSampleStudy>,
) -> Result<Vec<u8>> {
    let mut details = details(study, prices, metadata);
    let frontier = frontier_or_new(study, prices, frontier)?;
    let out_of_sample = walk_forward(study, prices, out_of_sample)?;
    details.insert("frontier_warnings", frontier.warnings.clone());
    let (weights, estimates) = latest_frames(study);
    let mut files: Vec<(String, String)> = vec![
        ("prices.csv".into(), prices.to_csv("Date")),
        ("latest_weights.csv".into(), weights.to_csv(None)),
        ("latest_estimates.csv".into(), estimates.to_csv(None)),
        ("fit_estimates.csv".into(), selection_fit_estimates(study, prices).to_csv(None)),
        ("latest/frontier.csv".into(), frontier.frontier.to_csv(None)),
        ("latest/frontier_weights.csv".into(), frontier.weights.to_csv(None)),
        ("latest/fitted_comparison.csv".into(), frontier.comparisons.to_csv(None)),
    ];
    if let Some(benchmarks) = &study.benchmark_prices {
        files.push(("benchmark_prices.csv".into(), benchmarks.to_csv("Date")));
    }
    files.push(("eligibility.csv".into(), study.eligibility.to_csv(None)));
    if let Some(coverage) = coverage_table(metadata) {
        files.push(("universe_coverage.csv".into(), coverage.to_csv(None)));
    }
    match &out_of_sample {
        Ok(result) => {
            let mut settings = result.settings.clone();
            settings.insert("notes", result.notes.clone());
            details.insert("out_of_sample", settings);
            files.push(("out_of_sample/windows.csv".into(), result.windows.to_csv(None)));
            if result.metrics.nrows() > 0 {
                files.push(("out_of_sample/metrics.csv".into(), result.metrics.to_csv(None)));
                files.push(("out_of_sample/weights.csv".into(), result.weights.to_csv(None)));
                files.push(("out_of_sample/eligibility.csv".into(), result.eligibility.to_csv(None)));
            }
        }
        Err(reason) => details.insert("out_of_sample", crate::dict! { "unavailable" => reason.as_str() }),
    }
    let test = &study.test;
    files.push(("test/metrics.csv".into(), test.metrics.to_csv(Some("strategy"))));
    files.push(("test/equity.csv".into(), test.equity.to_csv("date")));
    for (number, name) in test.equity.columns.iter().enumerate() {
        let stem = format!("test/{:02}", number + 1);
        files.push((format!("{stem}_allocations.csv"), test.allocations[name].to_csv("date")));
        files.push((format!("{stem}_holdings.csv"), test.holdings[name].to_csv(Some("symbol"))));
        files.push((format!("{stem}_trades.csv"), test.trades[name].to_csv(Some("date"))));
    }
    let report = report_html(study, prices, metadata, &frontier, &out_of_sample);
    let mut entries = vec![
        ("report.html".to_string(), report),
        ("metadata.json".to_string(), Json::Dict(details).dumps(Some(2))),
        ("THIRD_PARTY_NOTICES.txt".to_string(), PLOTLY_JS_LICENSE.to_string()),
    ];
    entries.extend(files);
    write_zip(&entries)
}

/// A deflated ZIP archive of the given files, in the given order.
pub fn write_zip<T: AsRef<[u8]>>(entries: &[(String, T)]) -> Result<Vec<u8>> {
    let failure = |error: zip::result::ZipError| Error::Io(format!("Could not write the report archive: {error}"));
    let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let mut archive = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, text) in entries {
        archive.start_file(name.as_str(), options).map_err(failure)?;
        archive.write_all(text.as_ref())?;
    }
    Ok(archive.finish().map_err(failure)?.into_inner())
}
