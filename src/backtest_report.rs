//! Descriptive findings and portable views of a backtesting study.
//!
//! Port of `efficient_frontier/backtest_report.py`. Owner: reports agent.

use std::cmp::Ordering;

use serde_json::{Value, json};

use crate::backtest::BacktestStudy;
use crate::benchmark_report::{
    add_benchmark_paths, benchmark_html, benchmark_markdown, dates_json, drawdown, np_sum, values_json,
};
use crate::benchmarks::BenchmarkComparison;
use crate::error::Result;
use crate::evidence::EvidenceAnalysis;
use crate::frame::{Cell, Frame, Table, iso};
use crate::html_table::data_table;
use crate::plot::Figure;
use crate::pyfmt::{self, Json, escape};
use crate::story::{setting_label, setting_text, truthy};
use crate::style::{ACCENT, ChartStyle, GRID, INK_2, REST, color, display_label, plotly_config, style_chart};
use crate::value_error;

/// How a value shows in report tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueFormat {
    /// `12.34%`.
    Percent,
    /// `1.23`.
    Number,
    /// `2.75×`.
    Multiple,
    /// A whole number.
    Count,
}

/// Strategy metrics in table order: column, label and format.
pub const METRICS: [(&str, &str, ValueFormat); 11] = [
    ("total_return", "Net total return", ValueFormat::Percent),
    ("cagr", "CAGR", ValueFormat::Percent),
    ("volatility", "Annual volatility", ValueFormat::Percent),
    ("sharpe", "Sharpe", ValueFormat::Number),
    ("sortino", "Sortino", ValueFormat::Number),
    ("calmar", "Calmar", ValueFormat::Number),
    ("max_drawdown", "Maximum drawdown", ValueFormat::Percent),
    ("total_turnover", "Gross turnover", ValueFormat::Multiple),
    ("total_cost", "Fees / initial capital", ValueFormat::Percent),
    ("rebalance_count", "Allocation events (incl. entry)", ValueFormat::Count),
    ("fallback_count", "Sharpe fallbacks", ValueFormat::Count),
];
/// Holding measures in table order: column, label and format.
pub const HOLDINGS: [(&str, &str, ValueFormat); 6] = [
    ("pnl_contribution", "P&L / initial capital", ValueFormat::Percent),
    ("average_weight", "Average weight", ValueFormat::Percent),
    ("end_weight", "End weight", ValueFormat::Percent),
    ("selection_frequency", "Selection frequency", ValueFormat::Percent),
    ("max_target_weight", "Maximum target weight", ValueFormat::Percent),
    ("max_realized_weight", "Maximum realized weight", ValueFormat::Percent),
];
pub const BEST_IN_HINDSIGHT: &str =
    "Best in hindsight. These results were picked after the test. They are not a recommendation.";
pub const ASSUMPTIONS: [&str; 6] = [
    "Buy and hold allocates once. Fixed rebalance restores the original fitted targets at \
     each scheduled trade. Expanding window refits using all prior observations; rolling \
     window refits using only the most recent configured number of returns. Rebalance \
     intervals count observed sessions, not calendar months. Fixed-rebalance, expanding-window \
     and rolling-window equal-weight baselines coincide because they use the same targets and schedule.",
    "All methods use the same evaluation dates and fixed input universe. \
     Targets use information through the prior close and execute at the following close. \
     The first evaluation session is held in cash earning zero interest. Weights drift \
     between trades; no execution-day return is earned using the new target.",
    "Trading fees are deducted from portfolio wealth. Gross turnover counts both purchases \
     and sales, each divided by portfolio value before that trade and summed across events. \
     Entry is included and terminal liquidation is excluded. Fees are shown as a share of \
     initial capital. Asset P&L contributions, less \
     fees, reconcile to the net total return. The fee total excludes foregone compounding, so it \
     does not measure the full performance difference from a zero-fee run. Taxes, FX conversion and liquidity constraints \
     are not modeled.",
    "Average weight uses exposure at the start of each evaluation session, including the \
     initial cash session. Selection frequency is the share of allocation events with an \
     asset target above 0.0001%; it is not the fraction of observations held.",
    "Prices must be adjusted and share a common currency basis. \
     A fixed universe can introduce survivorship bias.",
    "These findings describe the evaluated sample. The highest-return strategy is selected \
     after evaluation; it is not a forecast or a recommendation to hold its assets. Repeated \
     strategy selection using these results can overfit the evaluation period.",
];

fn assumptions(study: &BacktestStudy) -> Vec<String> {
    let settings = &study.settings;
    let periods = settings.get("periods_per_year").and_then(Json::as_f64).unwrap_or(252.0);
    let mut notes: Vec<String> = ASSUMPTIONS.iter().map(|note| note.to_string()).collect();
    if truthy(settings.get("include_profiles")) {
        let objective = if settings.get("profile_return_objective").and_then(Json::as_str) == Some("worst_window") {
            "Low, Medium, and Highest use the weakest-window objective, with a return target in each of three historical windows. "
        } else {
            "Low, Medium, and Highest use the same historical arithmetic mean return objective as the primary latest holdings. \
             Low selects minimum fitted volatility, Medium selects the sampled point nearest the midpoint of endpoint volatility, \
             and Highest selects maximum fitted arithmetic mean return. Three-window estimates are diagnostics, not optimization targets. "
        };
        notes.push(format!(
            "{objective}These are relative risk levels within each fit, not universal risk ratings. \
             Profile backtests fit only the history available before each allocation. \
             The latest model holdings use all supplied history and remain separate from these backtests and the original holdout."
        ));
    }
    notes.push(format!(
        "CAGR and volatility use {} observations per year. \
         Sortino measures excess return relative to downside deviation. \
         Calmar divides CAGR by the absolute maximum drawdown. \
         An em dash marks a ratio with a zero denominator.",
        pyfmt::g(periods)
    ));
    notes
}

/// One value in a report table. Missing values show as an em dash.
pub fn format_value(cell: &Cell, kind: ValueFormat) -> String {
    if cell.is_null() {
        return "—".into();
    }
    let value = cell.as_f64().unwrap_or(f64::NAN);
    match kind {
        ValueFormat::Percent => pyfmt::pct(value, 2),
        ValueFormat::Count => match cell {
            Cell::Int(count) => count.to_string(),
            _ => (value.trunc() as i64).to_string(),
        },
        ValueFormat::Number => pyfmt::format(value, ".2f"),
        ValueFormat::Multiple => format!("{}×", pyfmt::format(value, ".2f")),
    }
}

/// The given columns as display text under their labels, with the same index.
pub fn formatted(frame: &Table, columns: &[(&str, &str, ValueFormat)]) -> Table {
    let mut display = Table { columns_name: None, columns: Vec::new(), data: Vec::new(), ..frame.clone() };
    for (column, label, kind) in columns {
        display.push(*label, frame.column(column).iter().map(|cell| Cell::Str(format_value(cell, *kind))).collect());
    }
    display
}

/// `frame.rename(index=display_label)` on the first index level.
fn display_index(mut frame: Table) -> Table {
    if let Some(level) = frame.index.first_mut() {
        for cell in level {
            *cell = Cell::Str(display_label(&cell.py_str()));
        }
    }
    frame
}

/// Positions in descending order of `key`, stable, with NaN last, as `sort_values(ascending=False)`.
fn descending(positions: &[usize], key: impl Fn(usize) -> f64) -> Vec<usize> {
    let mut order = positions.to_vec();
    order.sort_by(|a, b| match (key(*a).is_nan(), key(*b).is_nan()) {
        (false, false) => key(*b).partial_cmp(&key(*a)).unwrap_or(Ordering::Equal),
        (nan_a, nan_b) => nan_a.cmp(&nan_b),
    });
    order
}

/// Positions in ascending order of `key`, stable, with NaN last, as `sort_values()`.
fn ascending(positions: &[usize], key: impl Fn(usize) -> f64) -> Vec<usize> {
    let mut order = positions.to_vec();
    order.sort_by(|a, b| match (key(*a).is_nan(), key(*b).is_nan()) {
        (false, false) => key(*a).partial_cmp(&key(*b)).unwrap_or(Ordering::Equal),
        (nan_a, nan_b) => nan_a.cmp(&nan_b),
    });
    order
}

/// The first position of the largest value, skipping NaN, as `Series.idxmax()`.
fn idxmax(values: &[f64], positions: impl Iterator<Item = usize>) -> Option<usize> {
    positions.filter(|i| !values[*i].is_nan()).fold(None, |best, i| match best {
        Some(b) if values[b] >= values[i] => Some(b),
        _ => Some(i),
    })
}

/// The first position of the smallest value, skipping NaN, as `Series.idxmin()`.
fn idxmin(values: &[f64], positions: impl Iterator<Item = usize>) -> Option<usize> {
    positions.filter(|i| !values[*i].is_nan()).fold(None, |best, i| match best {
        Some(b) if values[b] <= values[i] => Some(b),
        _ => Some(i),
    })
}

/// Up to ten holdings targeted above 0.0001%, by largest absolute contribution.
fn top_holdings(study: &BacktestStudy, strategy: &str) -> Table {
    let holdings = &study.holdings[strategy];
    let targets = holdings.f64s("max_target_weight");
    let contributions = holdings.f64s("pnl_contribution");
    let kept: Vec<usize> = (0..holdings.nrows()).filter(|i| targets[*i] > 1e-6).collect();
    let mut order = descending(&kept, |i| contributions[i].abs());
    order.truncate(10);
    holdings.take_rows(&order)
}

/// The strategy with the highest net total return.
fn best_strategy(study: &BacktestStudy) -> String {
    let total = study.metrics.f64s("total_return");
    let row = idxmax(&total, 0..total.len()).expect("a backtest study has at least one strategy");
    study.metrics.index_labels()[row].clone()
}

/// Describe measured outcomes without treating selected winners as forecasts.
pub fn findings(study: &BacktestStudy) -> Vec<String> {
    let metrics = &study.metrics;
    let names = metrics.index_labels();
    let total = metrics.f64s("total_return");
    let best_row = idxmax(&total, 0..total.len()).expect("a backtest study has at least one strategy");
    let best = &names[best_row];
    let value = |column: &str, row: usize| metrics.column(column)[row].as_f64().unwrap_or(f64::NAN);
    let method = best.split(" · ").next().unwrap_or(best);
    let benchmark = format!("{method} · Equal weight");
    let mut outcome = format!("Highest net return in this sample: {best}, {}.", pyfmt::format(total[best_row], "+.2%"));
    if benchmark == *best {
        outcome.push_str(" The equal-weight comparison itself had the highest net return.");
    } else if let Some(row) = metrics.row_of(&benchmark) {
        let difference = (total[best_row] - total[row]) * 100.0;
        let direction = if difference >= 0.0 { "above" } else { "below" };
        outcome.push_str(&format!(
            " This was {} percentage points {direction} {benchmark}.",
            pyfmt::format(difference.abs(), ".2f")
        ));
    }
    let drawdowns = metrics.f64s("max_drawdown");
    let safest = idxmax(&drawdowns, 0..drawdowns.len()).expect("a drawdown");
    let mut result = vec![
        outcome,
        format!(
            "Smallest maximum drawdown in this sample: {}, {} from a previous portfolio high.",
            names[safest],
            pyfmt::pct(drawdowns[safest], 2)
        ),
    ];
    let holdings = &study.holdings[best.as_str()];
    let assets = holdings.index_labels();
    let contributions = holdings.f64s("pnl_contribution");
    // Two decimal places in percentage points cannot meaningfully show smaller values.
    let positions = 0..contributions.len();
    match idxmax(&contributions, positions.clone().filter(|i| contributions[*i] >= 5e-5)) {
        Some(asset) => result.push(format!(
            "Largest positive holding contribution for {best}: {}, {} percentage points of initial capital before fees.",
            assets[asset],
            pyfmt::format(contributions[asset] * 100.0, "+.2f")
        )),
        None => result.push(format!(
            "No holding made a positive P&L contribution at the displayed precision for {best} in this sample."
        )),
    }
    match idxmin(&contributions, positions.filter(|i| contributions[*i] <= -5e-5)) {
        Some(asset) => result.push(format!(
            "Largest holding detractor for {best}: {}, {} percentage points of initial capital before fees.",
            assets[asset],
            pyfmt::format(contributions[asset] * 100.0, "+.2f")
        )),
        None => result.push(format!(
            "No holding made a negative P&L contribution at the displayed precision for {best} in this sample."
        )),
    }
    let cost = value("total_cost", best_row);
    result.push(format!(
        "Trading for {best}: gross turnover {}× (summed traded-notional / pretrade-value ratios); \
         modeled fees consumed {} of initial capital ({} per 10,000 initial units).",
        pyfmt::format(value("total_turnover", best_row), ".2f"),
        pyfmt::pct(cost, 2),
        pyfmt::format(cost * 10_000.0, ".2f")
    ));
    let realized = holdings.f64s("max_realized_weight");
    let targets = holdings.f64s("max_target_weight");
    if let Some(largest) = idxmax(&realized, 0..realized.len()) {
        result.push(format!(
            "Largest observed closing position for {best}: {}, {} of portfolio value; \
             its largest target was {}. Price changes can move realized weights above target limits.",
            assets[largest],
            pyfmt::pct(realized[largest], 2),
            pyfmt::pct(targets[largest], 2)
        ));
    }
    let fallbacks: Vec<f64> =
        metrics.f64s("fallback_count").iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
    let fallbacks = np_sum(&fallbacks) as i64;
    if fallbacks != 0 {
        result.push(format!(
            "Maximum-Sharpe fallback occurred {fallbacks} times across the strategy paths; \
             inspect the warnings and trade logs before interpreting that comparison."
        ));
    }
    result
}

/// Options of [`backtest_chart`], with the Python defaults.
#[derive(Clone, Copy, Debug)]
pub struct ChartOptions<'a> {
    /// Plot the fall from the running peak instead of the value of 10,000.
    pub drawdown: bool,
    /// The strategies to plot, in this order. `None` plots every path.
    pub strategies: Option<&'a [&'a str]>,
    pub benchmark_equity: Option<&'a Frame>,
    /// The highlighted strategy. Every other strategy is a thin grey line without a legend entry.
    pub focus: Option<&'a str>,
    /// Names each strategy trace.
    pub label: fn(&str) -> String,
}

impl Default for ChartOptions<'_> {
    fn default() -> Self {
        Self { drawdown: false, strategies: None, benchmark_equity: None, focus: None, label: display_label }
    }
}

/// Plot selected paths without changes to the complete study data.
///
/// With a focus strategy, that path is blue and every other strategy is a thin
/// grey line without a legend entry. An unknown strategy name is an error.
pub fn backtest_chart(study: &BacktestStudy, options: &ChartOptions) -> Result<Figure> {
    let equity = &study.equity;
    let names: Vec<&str> = match options.strategies {
        Some(names) => names.to_vec(),
        None => equity.columns.iter().map(String::as_str).collect(),
    };
    let unknown: Vec<&str> = names.iter().copied().filter(|name| equity.position(name).is_none()).collect();
    if !unknown.is_empty() {
        return Err(value_error!("Unknown backtest strategies: {}", unknown.join(", ")));
    }
    let mut figure = Figure::new();
    let measure = if options.drawdown { "Drawdown: %{y:.2%}" } else { "Net portfolio value: %{y:,.2f}" };
    for name in names {
        let policy = name.split_once(" · ").map_or("", |(_, policy)| policy);
        let path = equity.column(name).expect("a known strategy");
        let values: Vec<f64> =
            if options.drawdown { drawdown(&path) } else { path.iter().map(|value| value * 10_000.0).collect() };
        let (line, legend) = match options.focus {
            None => (json!({"color": color(policy, REST), "width": 2}), true),
            Some(focus) if focus == name => (json!({"color": ACCENT, "width": 2.5}), true),
            Some(_) => (json!({"color": REST, "width": 1.25}), false),
        };
        figure.add_trace(json!({
            "type": "scatter", "x": dates_json(&equity.index), "y": values_json(&values),
            "name": escape(&(options.label)(name)), "mode": "lines", "line": line, "showlegend": legend,
            "hovertemplate": format!("%{{x|%Y-%m-%d}}<br>{measure}<extra>%{{fullData.name}}</extra>"),
        }));
    }
    add_benchmark_paths(&mut figure, options.benchmark_equity, options.drawdown);
    let (title, height) = if options.drawdown { ("Falls from peak", 360) } else { ("Growth of 10,000", 420) };
    style_chart(&mut figure, ChartStyle { title: Some(title), height, hovermode: "x unified", story: false });
    figure.update_yaxes(json!({"tickformat": if options.drawdown { ".0%" } else { "~s" }}));
    Ok(figure)
}

/// Use the requested strategy, else the fixed default that does not depend on results.
fn focus_strategy(study: &BacktestStudy, focus: Option<&str>) -> String {
    let columns = &study.equity.columns;
    if let Some(focus) = focus.filter(|focus| columns.iter().any(|name| name == focus)) {
        return focus.to_string();
    }
    let defaults = ["Expanding window · Medium", "Minimum volatility", "Expanding window · Minimum volatility"];
    defaults
        .iter()
        .find(|name| columns.iter().any(|column| column == *name))
        .map_or_else(|| columns[0].clone(), |name| name.to_string())
}

/// Escape text for a Markdown table or list: HTML entities, backslashes, line breaks and Markdown symbols.
pub fn markdown(value: &str) -> String {
    let text = escape(value).replace('\\', "\\\\").replace(['\n', '\r'], " ");
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        if "|`*_[]".contains(character) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn markdown_table(frame: &Table) -> String {
    let first =
        if frame.index_names.first().and_then(Option::as_deref) == Some("Asset") { "Asset" } else { "Strategy" };
    let header: Vec<String> = std::iter::once(first.to_string()).chain(frame.columns.iter().cloned()).collect();
    let row =
        |cells: Vec<String>| format!("| {} |", cells.iter().map(|cell| markdown(cell)).collect::<Vec<_>>().join(" | "));
    let mut lines = vec![row(header.clone()), format!("| {} |", vec!["---"; header.len()].join(" | "))];
    let labels = frame.index_labels();
    for (i, label) in labels.iter().enumerate() {
        lines.push(row(std::iter::once(label.clone())
            .chain(frame.data.iter().map(|column| column[i].py_str()))
            .collect()));
    }
    lines.join("\n")
}

/// Create a standalone findings report without a Markdown-table dependency.
pub fn findings_markdown(
    study: &BacktestStudy,
    source: &str,
    benchmarks: Option<&BenchmarkComparison>,
    evidence: Option<&EvidenceAnalysis>,
) -> String {
    let best = best_strategy(study);
    let mut holdings = formatted(&top_holdings(study, &best), &HOLDINGS);
    holdings.index_names = vec![Some("Asset".into())];
    let dates = &study.equity.index;
    let settings: Vec<String> = study
        .settings
        .iter()
        .map(|(key, value)| format!("- {}: {}", markdown(key), markdown(&value.py_str())))
        .collect();
    let notes: Vec<String> = study.warnings.iter().map(|note| format!("- {}", markdown(note))).collect();
    let comparison = benchmarks
        .map_or_else(String::new, |benchmarks| format!("{}\n", benchmark_markdown(benchmarks, evidence, true)));
    let bullets: Vec<String> =
        findings(study).iter().map(|note| format!("- {}", markdown(&display_label(note)))).collect();
    let mut document = format!(
        "# Backtesting findings\n\nSource: {}\n\n\
         Portfolio baseline: {}. Evaluation returns: {} to {} ({} observations).\n\n",
        markdown(source),
        iso(dates[0]),
        iso(dates[1]),
        iso(dates[dates.len() - 1]),
        dates.len() - 1
    );
    document.push_str(&comparison);
    document.push_str(&format!(
        "## Observed findings\n\n{BEST_IN_HINDSIGHT}\n\n{}\n\n\
         These are retrospective comparisons; the selected strategy is not a forecast.\n\n\
         ## All method and portfolio combinations\n\n{}\n\n\
         ## Holdings for {}\n\n\
         Up to 10 largest absolute P&L contributions are shown, restricted to assets targeted above 0.0001%. \
         Contributions use initial capital \
         as the denominator, with trading fees accounted for separately. The CSV export contains all holdings.\n\n\
         {}\n\n## Settings\n\n{}\n\n## Assumptions and interpretation\n\n{}\n",
        bullets.join("\n"),
        markdown_table(&display_index(formatted(&study.metrics, &METRICS))),
        markdown(&display_label(&best)),
        markdown_table(&holdings),
        settings.join("\n"),
        assumptions(study).join("\n\n")
    ));
    if !notes.is_empty() {
        document.push_str(&format!("\n## Study notes\n\n{}\n", notes.join("\n")));
    }
    document
}

/// The holding contributions as horizontal bars, smallest first.
fn holdings_chart(holdings: &Table) -> Figure {
    let contributions = holdings.f64s("pnl_contribution");
    let order = ascending(&(0..holdings.nrows()).collect::<Vec<_>>(), |i| contributions[i]);
    let labels = holdings.index_labels();
    let values: Vec<f64> = order.iter().map(|i| contributions[*i]).collect();
    let names: Vec<String> = order.iter().map(|i| escape(&labels[*i])).collect();
    let colors: Vec<&str> = values.iter().map(|value| if *value >= 0.0 { ACCENT } else { "#86868b" }).collect();
    let mut bars = Figure::with_trace(json!({
        "type": "bar", "x": values_json(&values), "y": names, "orientation": "h",
        "marker": {"color": colors, "line": {"width": 0}}, "text": values_json(&values), "texttemplate": "%{x:+.1%}",
        "textposition": "outside", "textfont": {"size": 12, "color": INK_2}, "cliponaxis": false,
        "hovertemplate": "%{y}<br>P&L / initial capital: %{x:.2%}<extra></extra>",
    }));
    style_chart(&mut bars, ChartStyle { title: Some("Holding contributions"), height: 430, ..Default::default() });
    bars.update_layout(json!({"showlegend": false, "margin": {"l": 8, "r": 56, "t": 44, "b": 24}}));
    bars.update_xaxes(json!({"tickformat": ".1%", "showgrid": true, "gridcolor": GRID}));
    bars.update_yaxes(json!({"showgrid": false, "dtick": 1}));
    bars
}

/// Return escaped report sections; the parent document supplies plotly.js.
///
/// Chart ids start with `backtest-`, so the document must not hold two backtest sections.
pub fn backtest_html(
    study: &BacktestStudy,
    benchmarks: Option<&BenchmarkComparison>,
    evidence: Option<&EvidenceAnalysis>,
    focus: Option<&str>,
) -> String {
    let best = best_strategy(study);
    let focus = focus_strategy(study, focus);
    let holdings = top_holdings(study, &best);
    let mut holding_table = formatted(&holdings, &HOLDINGS);
    holding_table.index_names = vec![Some("Asset".into())];
    let config: Value = plotly_config();
    let bullets: String =
        findings(study).iter().map(|note| format!("<li>{}</li>", escape(&display_label(note)))).collect();
    let warnings: String = study.warnings.iter().map(|note| format!("<li>{}</li>", escape(note))).collect();
    let assumptions: String = assumptions(study).iter().map(|note| format!("<p>{}</p>", escape(note))).collect();
    let settings: String = study
        .settings
        .iter()
        .map(|(key, value)| {
            format!("<tr><th>{}</th><td>{}</td></tr>", escape(&setting_label(key)), escape(&setting_text(key, value)))
        })
        .collect();
    let benchmark_equity = benchmarks.and_then(|benchmarks| benchmarks.backtest_equity.as_ref());
    let comparison = benchmarks.map_or_else(String::new, |benchmarks| benchmark_html(benchmarks, evidence, true));
    let chart = |drawdown: bool| {
        let options = ChartOptions { drawdown, benchmark_equity, focus: Some(focus.as_str()), ..Default::default() };
        backtest_chart(study, &options).expect("every study strategy is known")
    };
    let mut document = format!(
        "<section id=\"backtesting\">{comparison}<h2>Backtesting findings</h2>\
         <p>{BEST_IN_HINDSIGHT}</p><ul>{bullets}</ul><p class=\"muted\">These are retrospective comparisons; \
         the selected strategy is not a forecast.</p>\
         <p class=\"muted\">Blue: {}. Grey: the other strategies.</p>\
         <div class=\"chart\">{}</div><div class=\"chart\">{}</div>\
         <h2>All method and portfolio combinations</h2><div class=\"scroll\">{}</div>\
         <h2>Holdings for {}</h2><p>Up to 10 largest absolute P&amp;L contributions \
         are shown, restricted to assets targeted above 0.0001%. \
         Contributions use initial capital as the denominator; fees are accounted for separately. \
         The CSV export contains all holdings.</p>\
         <div class=\"chart\">{}</div><div class=\"scroll\">{}</div>\
         <h2>Backtest settings</h2><div class=\"scroll\"><table class=\"metadata\">{settings}</table></div>\
         <h2>Backtest assumptions</h2>{assumptions}",
        escape(&display_label(&focus)),
        chart(false).to_html("backtest-growth", &config),
        chart(true).to_html("backtest-drawdown", &config),
        data_table(&display_index(formatted(&study.metrics, &METRICS))),
        escape(&display_label(&best)),
        holdings_chart(&holdings).to_html("backtest-holdings", &config),
        data_table(&holding_table),
    );
    if !warnings.is_empty() {
        document.push_str(&format!("<aside><h2>Backtest notes</h2><ul>{warnings}</ul></aside>"));
    }
    document.push_str("</section>");
    document
}
