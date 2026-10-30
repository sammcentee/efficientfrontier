//! Benchmark comparisons and uncertainty in historical portfolio results.
//!
//! Port of `efficient_frontier/benchmark_report.py`. Owner: reports agent.
//!
//! It also holds the pandas numeric helpers that the report modules share:
//! [`np_sum`], [`np_mean`], [`drawdown`] and the chart value helpers.

use serde_json::{Value, json};

use crate::benchmarks::BenchmarkComparison;
use crate::error::Result;
use crate::evidence::EvidenceAnalysis;
use crate::frame::{Cell, Date, Frame, Table, iso};
use crate::html_table::{HtmlOptions, data_table, to_html};
use crate::plot::Figure;
use crate::pyfmt::{self, Json, escape};
use crate::style::{ChartStyle, INK_2, INK_3, INK_4, benchmark_color, display_label, plotly_config, style_chart};
use crate::value_error;

/// Column labels of the evidence summary.
pub const EVIDENCE_LABELS: [(&str, &str); 20] = [
    ("total_return_difference", "Total return difference"),
    ("cagr_difference", "CAGR difference"),
    ("annual_advantage", "Annual mean advantage"),
    ("advantage_ci_low", "Mean CI lower"),
    ("advantage_ci_high", "Mean CI upper"),
    ("advantage_pvalue", "Mean p-value"),
    ("advantage_pvalue_adjusted", "Mean Holm p-value"),
    ("beta", "Benchmark beta"),
    ("alpha", "Annual alpha"),
    ("alpha_ci_low", "Alpha CI lower"),
    ("alpha_ci_high", "Alpha CI upper"),
    ("alpha_pvalue", "Alpha p-value"),
    ("alpha_pvalue_adjusted", "Alpha Holm p-value"),
    ("information_ratio", "Information ratio"),
    ("tracking_error", "Annual tracking error"),
    ("windows_ahead", "Windows ahead"),
    ("observations", "Inference observations"),
    ("status", "Historical evidence"),
    ("advantage_note", "Mean inference note"),
    ("alpha_note", "Alpha inference note"),
];

/// The default color of a reference without its own color.
const OTHER_BENCHMARK: &str = "#737b85";

/// Dates as `YYYY-MM-DD` text for chart axes.
pub fn dates_json(dates: &[Date]) -> Value {
    Value::Array(dates.iter().map(|date| Value::String(iso(*date))).collect())
}

/// Numbers for chart traces. NaN becomes `null`, as plotly.js expects.
pub fn values_json(values: &[f64]) -> Value {
    Value::Array(values.iter().map(|value| json!(value)).collect())
}

/// `numpy.sum` of a float array: pairwise summation in blocks of eight, the
/// order that pandas `Series.sum()` and `Series.mean()` use. The order matters
/// when a sum is rounded for display.
pub fn np_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |total, value| total + value);
    }
    if n <= 128 {
        let mut partial = [0.0; 8];
        partial.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for (j, total) in partial.iter_mut().enumerate() {
                *total += values[i + j];
            }
            i += 8;
        }
        let mut total = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for value in &values[i..] {
            total += value;
        }
        return total;
    }
    let half = n / 2;
    let half = half - half % 8;
    np_sum(&values[..half]) + np_sum(&values[half..])
}

/// pandas `Series.mean()`: NaN values are left out. NaN when no value remains.
pub fn np_mean(values: &[f64]) -> f64 {
    let count = values.iter().filter(|value| !value.is_nan()).count();
    if count == 0 {
        return f64::NAN;
    }
    let filled: Vec<f64> = values.iter().map(|value| if value.is_nan() { 0.0 } else { *value }).collect();
    np_sum(&filled) / count as f64
}

/// `path / path.cummax() - 1`: the fall from the running peak. NaN values stay NaN.
pub fn drawdown(path: &[f64]) -> Vec<f64> {
    let mut peak = f64::NAN;
    path.iter()
        .map(|value| {
            if value.is_nan() {
                return f64::NAN;
            }
            peak = if peak.is_nan() { *value } else { peak.max(*value) };
            value / peak - 1.0
        })
        .collect()
}

/// Add one line per benchmark: the value of 10,000, or the fall from the running peak.
pub fn add_benchmark_paths(figure: &mut Figure, equity: Option<&Frame>, drawdown_view: bool) {
    let Some(equity) = equity else { return };
    for (j, name) in equity.columns.iter().enumerate() {
        let path = equity.col(j);
        let values: Vec<f64> =
            if drawdown_view { drawdown(&path) } else { path.iter().map(|value| value * 10_000.0).collect() };
        let measure = if drawdown_view { "Drawdown: %{y:.2%}" } else { "Benchmark value: %{y:,.2f}" };
        figure.add_trace(json!({
            "type": "scatter", "x": dates_json(&equity.index), "y": values_json(&values), "name": escape(name),
            "mode": "lines", "line": {"color": benchmark_color(name, OTHER_BENCHMARK), "width": 1.75},
            "hovertemplate": format!("%{{x|%Y-%m-%d}}<br>{measure}<extra>%{{fullData.name}}</extra>"),
        }));
    }
}

/// Return the ETF symbol from a label such as "S&P 500 (SPY)".
pub fn ticker(name: &str) -> String {
    match name.rfind('(') {
        Some(start) if name.ends_with(')') => name[start + 1..name.len() - 1].to_string(),
        _ => name.to_string(),
    }
}

/// Add one diamond per benchmark at its historical volatility and return estimate.
///
/// `estimates` has one row per benchmark name, a `volatility` column and the
/// `objective` column. `return_label` replaces the default hover label of the return.
pub fn add_benchmark_estimates(
    figure: &mut Figure,
    estimates: Option<&Table>,
    objective: &str,
    return_label: Option<&str>,
) {
    let Some(estimates) = estimates else { return };
    let label = return_label.unwrap_or(if objective == "worst_window_return" {
        "Lowest annual window mean"
    } else {
        "Historical arithmetic mean, yearly"
    });
    let volatility = estimates.f64s("volatility");
    let returns = estimates.f64s(objective);
    for (i, name) in estimates.index_labels().iter().enumerate() {
        figure.add_trace(json!({
            "type": "scatter", "x": values_json(&[volatility[i]]), "y": values_json(&[returns[i]]),
            "name": escape(name), "mode": "markers+text",
            "marker": {"symbol": "diamond", "size": 11, "color": benchmark_color(name, OTHER_BENCHMARK)},
            "text": [escape(&ticker(name))], "textposition": "bottom center",
            "textfont": {"size": 12, "color": INK_2},
            "hovertemplate": format!("Historical volatility, yearly: %{{x:.2%}}<br>{label}: %{{y:.2%}}<extra>%{{fullData.name}}</extra>"),
        }));
    }
}

/// Choose a fixed default without reference to realized performance.
pub fn default_evidence_strategy(evidence: &EvidenceAnalysis) -> Option<String> {
    let mut names: Vec<String> = Vec::new();
    for cell in &evidence.summary.index[0] {
        let name = cell.py_str();
        if !names.contains(&name) {
            names.push(name);
        }
    }
    ["Expanding window · Medium", "Minimum volatility", "Expanding window · Minimum volatility"]
        .iter()
        .find(|name| names.iter().any(|known| known == *name))
        .map(|name| name.to_string())
        .or_else(|| names.first().cloned())
}

/// Plot portfolio wealth divided by benchmark wealth, with a baseline of one.
pub fn evidence_chart(evidence: &EvidenceAnalysis, strategy: &str) -> Result<Figure> {
    let paths = evidence
        .relative_equity
        .strategy(strategy)
        .ok_or_else(|| value_error!("Unknown evidence strategy: {strategy}"))?;
    let mut figure = Figure::new();
    for (j, benchmark) in paths.columns.iter().enumerate() {
        figure.add_trace(json!({
            "type": "scatter", "x": dates_json(&paths.index), "y": values_json(&paths.col(j)), "name": escape(benchmark),
            "mode": "lines", "line": {"color": benchmark_color(benchmark, OTHER_BENCHMARK), "width": 2},
            "hovertemplate": "%{x|%Y-%m-%d}<br>Relative wealth: %{y:.3f}×<extra>%{fullData.name}</extra>",
        }));
    }
    figure.add_hline(json!(1), json!({"color": INK_4, "width": 1}), None);
    figure.add_annotation(json!({
        "font": {"size": 11, "color": INK_3}, "showarrow": false, "text": "Level with the market",
        "x": 1, "xanchor": "right", "xref": "x domain", "y": 1, "yanchor": "bottom", "yref": "y",
    }));
    style_chart(
        &mut figure,
        ChartStyle { title: Some("Versus the markets"), height: 380, hovermode: "x unified", story: false },
    );
    figure.update_yaxes(json!({"tickformat": ".2f", "ticksuffix": "×"}));
    Ok(figure)
}

/// One evidence value as display text. Missing values show as an em dash.
fn evidence_value(evidence: &EvidenceAnalysis, column: &str, cell: &Cell) -> String {
    if cell.is_null() {
        return "—".into();
    }
    if matches!(column, "status" | "advantage_note" | "alpha_note") {
        return cell.py_str();
    }
    let value = cell.as_f64().unwrap_or(f64::NAN);
    if column.contains("pvalue") {
        return pyfmt::format(value, ".4g");
    }
    match column {
        "windows_ahead" => {
            let count = evidence.settings.get("window_count").map_or_else(|| "3".to_string(), Json::py_str);
            format!("{}/{count}", value.trunc() as i64)
        }
        "observations" => (value.trunc() as i64).to_string(),
        "beta" | "information_ratio" => pyfmt::format(value, ".3f"),
        "total_return_difference"
        | "cagr_difference"
        | "annual_advantage"
        | "advantage_ci_low"
        | "advantage_ci_high" => {
            format!("{} pp", pyfmt::format(value * 100.0, "+.2f"))
        }
        _ => pyfmt::pct(value, 2),
    }
}

/// Format estimates and p-values without changes to the numeric evidence.
pub fn evidence_summary_frame(evidence: &EvidenceAnalysis) -> Table {
    let mut frame = evidence.summary.clone();
    for (column, cells) in frame.columns.iter().zip(frame.data.iter_mut()) {
        for cell in cells.iter_mut() {
            *cell = Cell::Str(evidence_value(evidence, column, cell));
        }
    }
    frame.rename_columns(&EVIDENCE_LABELS);
    frame
}

fn evidence_notes(evidence: &EvidenceAnalysis) -> Vec<String> {
    let settings = &evidence.settings;
    let text = |key: &str, default: &str| settings.get(key).map_or_else(|| default.to_string(), Json::py_str);
    let confidence = settings.get("confidence_level").and_then(Json::as_f64).unwrap_or(0.95);
    let mut notes = vec![
        "Relative wealth divides portfolio value by benchmark value after both start at one. A value above one means the portfolio is ahead.".to_string(),
        "Annual mean advantage is the arithmetic mean return difference times observations per year. It is not the CAGR difference.".to_string(),
        format!(
            "Inference requires at least {} return observations. \
             The {} HAC confidence intervals are pointwise, not simultaneous bands.",
            text("inference_min_observations", "60"),
            pyfmt::pct(confidence, 0)
        ),
        format!(
            "HAC uses {} lags for serial correlation and unequal return variance. \
             Holm correction covers {} tests across all strategy/benchmark pairs and both mean-advantage and alpha hypotheses.",
            text("hac_lags", "0"),
            text("multiple_testing_tests", "0")
        ),
        format!(
            "Inference uses {} paired returns from {} to {}.",
            text("inference_observations", "0"),
            text("inference_start", "None"),
            text("inference_end", "None")
        ),
        "The p-values do not measure the probability of future outperformance. These estimates describe the observed sample and use large-sample approximations.".to_string(),
        "Alpha uses a separate excess-return regression for each benchmark. Beta measures only exposure to that benchmark. Positive alpha does not establish investment skill.".to_string(),
        format!(
            "The {} chronological windows describe consistency within the evaluated sample. \
             They are not independent replications of the strategy.",
            text("window_count", "3")
        ),
    ];
    if crate::story::truthy(settings.get("delayed_entry")) {
        notes.push("Inference omits the first evaluation session and its entry fee. Total returns, relative wealth, and window results retain entry costs.".to_string());
    }
    notes
}

/// Format the window results and give the columns their report labels.
fn window_frame(windows: &Table) -> Table {
    let mut display = windows.clone();
    for (column, cells) in display.columns.iter().zip(display.data.iter_mut()) {
        let format: fn(f64) -> String = match column.as_str() {
            "relative_return" => |value| pyfmt::pct(value, 2),
            "annual_advantage" => |value| format!("{} pp", pyfmt::format(value * 100.0, "+.2f")),
            _ => continue,
        };
        for cell in cells.iter_mut() {
            *cell = Cell::Str(if cell.is_null() { "—".into() } else { format(cell.as_f64().unwrap_or(f64::NAN)) });
        }
    }
    display.rename_columns(&[
        ("strategy", "Strategy"),
        ("benchmark", "Benchmark"),
        ("window", "Window"),
        ("start", "Start"),
        ("end", "End"),
        ("observations", "Observations"),
        ("relative_return", "Relative wealth change"),
        ("annual_advantage", "Annual mean advantage"),
    ]);
    display
}

/// The rows of the window table whose strategy matches, with the strategy column kept or dropped.
fn windows_of(evidence: &EvidenceAnalysis, strategy: &str) -> Table {
    let strategies = evidence.windows.column("strategy");
    let rows: Vec<usize> = (0..strategies.len()).filter(|i| strategies[*i].as_str() == Some(strategy)).collect();
    evidence.windows.take_rows(&rows)
}

/// The window table with display labels for every strategy.
fn labelled_windows(evidence: &EvidenceAnalysis) -> Table {
    let mut windows = evidence.windows.clone();
    let strategies = windows.column("strategy").iter().map(|cell| Cell::Str(display_label(&cell.py_str()))).collect();
    windows.push("strategy", strategies);
    windows
}

fn drop_column(table: &Table, column: &str) -> Table {
    let keep: Vec<&str> = table.columns.iter().map(String::as_str).filter(|name| *name != column).collect();
    table.select(&keep)
}

fn benchmark_metrics(frame: &Table) -> Table {
    let labels = [
        ("total_return", "Total return"),
        ("cagr", "CAGR"),
        ("volatility", "Annual volatility"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("calmar", "Calmar"),
        ("max_drawdown", "Maximum drawdown"),
        ("total_cost", "Fees / initial capital"),
    ];
    let present: Vec<&str> =
        labels.iter().map(|(column, _)| *column).filter(|column| frame.position(column).is_some()).collect();
    let mut result = frame.select(&present);
    for (column, cells) in result.columns.iter().zip(result.data.iter_mut()) {
        let ratio = matches!(column.as_str(), "sharpe" | "sortino" | "calmar");
        for cell in cells.iter_mut() {
            let value = cell.as_f64().unwrap_or(f64::NAN);
            *cell = Cell::Str(if cell.is_null() {
                "—".into()
            } else if ratio {
                pyfmt::format(value, ".3f")
            } else {
                pyfmt::pct(value, 2)
            });
        }
    }
    result.rename_columns(&labels);
    result
}

fn comparison_parts(benchmarks: &BenchmarkComparison, backtest: bool) -> (Table, Vec<String>) {
    let (equity, metrics) = if backtest {
        (
            benchmarks.backtest_equity.as_ref().expect("backtest benchmark paths"),
            benchmarks.backtest_metrics.as_ref().expect("backtest benchmark metrics"),
        )
    } else {
        (&benchmarks.holdout_equity, &benchmarks.holdout_metrics)
    };
    let notes = vec![
        format!(
            "Evaluation baseline: {}. Last observation: {}. Benchmark paths use the same dates as the portfolio paths.",
            iso(equity.index[0]),
            iso(equity.index[equity.nrows() - 1])
        ),
        "SPY and QQQ are USD ETF proxies for the S&P 500 and Nasdaq-100. They are not index series.".to_string(),
        "Benchmark markers use the same dates and return objective as each frontier chart. The independent ETF references do not follow portfolio weight limits.".to_string(),
        "Stock frontier volatility uses the configured model covariance. ETF marker volatility uses the ordinary sample variance. \
         With covariance shrinkage, these are different risk estimates. ETF positions are not scaled to match portfolio volatility."
            .to_string(),
        if backtest {
            "Backtest benchmarks stay in cash for the first interval, pay the same entry fee rate as the portfolios, then hold. No terminal sale occurs."
        } else {
            "The original holdout and its benchmarks allocate at the initial close. Both exclude fees."
        }
        .to_string(),
    ];
    (benchmark_metrics(metrics), notes)
}

/// `frame.rename(index=display_label, level="strategy")` on the first index level.
fn display_strategies(mut frame: Table) -> Table {
    if let Some(level) = frame.index.first_mut() {
        for cell in level {
            *cell = Cell::Str(display_label(&cell.py_str()));
        }
    }
    frame
}

/// `summary.xs(strategy, level="strategy")`: the rows of one strategy, indexed by benchmark.
fn summary_of(summary: &Table, strategy: &str) -> Table {
    let rows: Vec<usize> = (0..summary.nrows()).filter(|i| summary.index[0][*i].as_str() == Some(strategy)).collect();
    let mut table = summary.take_rows(&rows);
    table.index.remove(0);
    table.index_names.remove(0);
    table
}

/// Return report sections without another plotly.js bundle.
///
/// The evidence chart uses the element id `evidence-chart`.
pub fn benchmark_html(benchmarks: &BenchmarkComparison, evidence: Option<&EvidenceAnalysis>, backtest: bool) -> String {
    let (metrics, notes) = comparison_parts(benchmarks, backtest);
    let mut document = String::from("<section id=\"benchmarks\"><h2>Market benchmarks</h2>");
    for note in &notes {
        document.push_str(&format!("<p>{}</p>", escape(note)));
    }
    document.push_str(&format!("<div class=\"scroll\">{}</div>", data_table(&metrics)));
    if let Some(evidence) = evidence {
        let strategy = default_evidence_strategy(evidence);
        let summary = evidence_summary_frame(evidence);
        document.push_str("<h2>Evidence against benchmarks</h2>");
        for note in evidence_notes(evidence) {
            document.push_str(&format!("<p class=\"muted\">{}</p>", escape(&note)));
        }
        if let Some(strategy) = &strategy {
            document.push_str(&format!(
                "<h3>{}</h3><p>This default does not depend on realized returns.</p>",
                escape(&display_label(strategy))
            ));
            document.push_str(&format!("<div class=\"scroll\">{}</div>", data_table(&summary_of(&summary, strategy))));
            let chart = evidence_chart(evidence, strategy).expect("the default strategy has relative paths");
            document
                .push_str(&format!("<div class=\"chart\">{}</div>", chart.to_html("evidence-chart", &plotly_config())));
            let windows = window_frame(&drop_column(&windows_of(evidence, strategy), "strategy"));
            document.push_str("<h3>Results in each window</h3>");
            document
                .push_str(&format!("<div class=\"scroll\">{}</div>", to_html(&windows, &HtmlOptions::REPORT_NO_INDEX)));
        }
        document.push_str("<details><summary>All strategy comparisons</summary>");
        document
            .push_str(&format!("<div class=\"scroll\">{}</div></details>", data_table(&display_strategies(summary))));
        let windows = window_frame(&labelled_windows(evidence));
        document.push_str("<details><summary>Historical windows for every comparison</summary>");
        document.push_str(&format!(
            "<div class=\"scroll\">{}</div></details>",
            to_html(&windows, &HtmlOptions::REPORT_NO_INDEX)
        ));
        if !evidence.warnings.is_empty() {
            let items: String = evidence.warnings.iter().map(|note| format!("<li>{}</li>", escape(note))).collect();
            document.push_str(&format!("<aside><h3>Evidence notes</h3><ul>{items}</ul></aside>"));
        }
    }
    document.push_str("</section>");
    document
}

/// `frame.reset_index()` as Markdown: index levels first, then every column.
fn markdown_table(frame: &Table) -> String {
    let index_names: Vec<String> = frame
        .index_names
        .iter()
        .enumerate()
        .map(|(level, name)| match name {
            Some(name) => name.clone(),
            None if frame.index.len() == 1 => "index".into(),
            None => format!("level_{level}"),
        })
        .collect();
    let header: Vec<String> = index_names.into_iter().chain(frame.columns.iter().cloned()).collect();
    let row = |cells: Vec<String>| {
        format!(
            "| {} |",
            cells.iter().map(|cell| crate::backtest_report::markdown(cell)).collect::<Vec<_>>().join(" | ")
        )
    };
    let mut lines = vec![row(header.clone()), format!("| {} |", vec!["---"; header.len()].join(" | "))];
    for i in 0..frame.nrows() {
        let cells =
            frame.index.iter().map(|level| level[i].py_str()).chain(frame.data.iter().map(|column| column[i].py_str()));
        lines.push(row(cells.collect()));
    }
    lines.join("\n")
}

/// The benchmark section and the evidence as Markdown.
pub fn benchmark_markdown(
    benchmarks: &BenchmarkComparison,
    evidence: Option<&EvidenceAnalysis>,
    backtest: bool,
) -> String {
    use crate::backtest_report::markdown;
    let (metrics, notes) = comparison_parts(benchmarks, backtest);
    let mut document = format!(
        "## Market benchmarks\n\n{}\n\n{}\n",
        notes.iter().map(|note| markdown(note)).collect::<Vec<_>>().join("\n\n"),
        markdown_table(&metrics)
    );
    if let Some(evidence) = evidence {
        document.push_str(&format!(
            "\n## Evidence against benchmarks\n\n{}\n\n",
            evidence_notes(evidence).iter().map(|note| markdown(note)).collect::<Vec<_>>().join("\n\n")
        ));
        if let Some(strategy) = default_evidence_strategy(evidence) {
            document.push_str(&format!(
                "Default comparison: {}. This choice does not depend on realized returns.\n\n",
                markdown(&display_label(&strategy))
            ));
        }
        document.push_str(&format!("{}\n", markdown_table(&display_strategies(evidence_summary_frame(evidence)))));
        let windows = window_frame(&labelled_windows(evidence));
        let indexed = set_index(&windows, &["Strategy", "Benchmark", "Window"]);
        document.push_str(&format!("\n### Historical windows\n\n{}\n", markdown_table(&indexed)));
        if !evidence.warnings.is_empty() {
            let notes: Vec<String> = evidence.warnings.iter().map(|note| format!("- {}", markdown(note))).collect();
            document.push_str(&format!("\n{}\n", notes.join("\n")));
        }
    }
    document
}

/// `frame.set_index(columns)`.
fn set_index(frame: &Table, columns: &[&str]) -> Table {
    let mut table = Table::multi(
        columns.iter().map(|name| Some(name.to_string())).collect(),
        columns.iter().map(|name| frame.column(name).to_vec()).collect(),
    );
    for (name, cells) in frame.columns.iter().zip(&frame.data) {
        if !columns.contains(&name.as_str()) {
            table.push(name, cells.clone());
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::cells;

    #[test]
    fn benchmark_paths_show_value_or_drawdown() {
        let day = |d| chrono::NaiveDate::from_ymd_opt(2024, 1, d).unwrap();
        let equity = Frame::from_columns(
            vec![day(2), day(3), day(4)],
            vec![("S&P 500 (SPY)".into(), vec![1.0, 1.1, 0.99]), ("<X>".into(), vec![1.0, 0.9, 1.0])],
        );
        let mut figure = Figure::new();
        add_benchmark_paths(&mut figure, Some(&equity), false);
        add_benchmark_paths(&mut figure, Some(&equity), true);
        add_benchmark_paths(&mut figure, None, true);
        assert_eq!(figure.data.len(), 4);
        assert_eq!(figure.data[0]["x"], json!(["2024-01-02", "2024-01-03", "2024-01-04"]));
        assert_eq!(figure.data[0]["y"], json!([10000.0, 11000.0, 9900.0]));
        assert_eq!(figure.data[0]["line"]["color"], "#8e8e93");
        assert_eq!(figure.data[1]["name"], "&lt;X&gt;");
        assert_eq!(figure.data[1]["line"]["color"], "#737b85");
        let fall = figure.data[2]["y"].as_array().unwrap();
        assert_eq!(fall[0], 0.0);
        assert!((fall[2].as_f64().unwrap() + 0.1).abs() < 1e-12);
        assert!(figure.data[3]["hovertemplate"].as_str().unwrap().contains("Drawdown: %{y:.2%}"));
    }

    #[test]
    fn estimates_mark_each_etf_with_its_ticker() {
        let estimates = Table::labelled(Some("benchmark"), &["S&P 500 (SPY)", "Nasdaq-100 (QQQ)"])
            .with("volatility", cells(&[0.15, 0.2]))
            .with("expected_return", cells(&[0.1, 0.12]));
        let mut figure = Figure::new();
        add_benchmark_estimates(&mut figure, Some(&estimates), "expected_return", Some("Fit arithmetic mean, yearly"));
        assert_eq!(figure.data[1]["text"], json!(["QQQ"]));
        assert_eq!(figure.data[1]["marker"]["color"], "#545458");
        assert!(figure.data[0]["hovertemplate"].as_str().unwrap().contains("Fit arithmetic mean, yearly: %{y:.2%}"));
        assert_eq!(ticker("S&P 500 (SPY)"), "SPY");
        assert_eq!(ticker("SPY"), "SPY");
    }
}
