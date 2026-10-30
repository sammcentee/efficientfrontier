//! Benchmark comparisons and uncertainty in historical portfolio results.
//!
//! Port of `efficient_frontier/benchmark_report.py`. Owner: reports agent.
//!
//! This wave ports the chart helpers that the backtest and story components use.
//! Wave 2 ports `default_evidence_strategy`, `evidence_chart`,
//! `evidence_summary_frame`, `benchmark_html`, `benchmark_markdown` and their
//! private helpers, which need the benchmark comparison and evidence types.

use serde_json::{Value, json};

use crate::frame::{Date, Frame, Table, iso};
use crate::plot::Figure;
use crate::pyfmt::escape;
use crate::style::{INK_2, benchmark_color};

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
