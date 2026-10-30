//! Offline evidence and reproducible exports for the single portfolio model.
//!
//! Port of `efficient_frontier/selection_report.py`. Owner: reports agent.
//! The web app calls the public functions below; keep these signatures.

use crate::backtest::BacktestStudy;
use crate::error::Result;
use crate::frame::{Frame, Table};
use crate::out_of_sample::OutOfSampleStudy;
use crate::plot::Figure;
use crate::pyfmt::Dict;
use crate::selection::{SelectionFrontier, SelectionStudy};

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
    let _ = study;
    todo!("selection_report")
}

/// Identify every comparison path in the app and exported report.
pub fn selection_chart(study: &BacktestStudy) -> Figure {
    let _ = study;
    todo!("selection_report")
}

/// Plot the latest stock frontier and passive ETF references on their fit dates.
pub fn selection_frontier_chart(frontier: &SelectionFrontier, study: &SelectionStudy) -> Figure {
    let _ = (frontier, study);
    todo!("selection_report")
}

pub fn selection_comparison_html(study: &SelectionStudy, prices: &Frame) -> String {
    let _ = (study, prices);
    todo!("selection_report")
}

pub fn selection_replay_findings_html(study: &BacktestStudy) -> String {
    let _ = study;
    todo!("selection_report")
}

/// Classify every observed mean/volatility pair against the stock portfolio.
/// Index `strategy`; columns `annual_mean`, `volatility`, `classification`.
pub fn selection_replay_dominance(study: &BacktestStudy) -> Table {
    let _ = study;
    todo!("selection_report")
}

/// Reconcile initial fitted mean to replay mean in annual arithmetic units.
pub fn selection_replay_return_bridge(study: &SelectionStudy, prices: &Frame) -> ReturnBridge {
    let _ = (study, prices);
    todo!("selection_report")
}

/// Empty text when there is no verdict to show.
pub fn selection_replay_verdict_html(study: &BacktestStudy) -> String {
    let _ = study;
    todo!("selection_report")
}

pub fn selection_frontier_summary_html(study: &SelectionStudy, prices: &Frame, frontier: &SelectionFrontier) -> String {
    let _ = (study, prices, frontier);
    todo!("selection_report")
}

pub fn selection_summary_html(study: &SelectionStudy, metadata: &Dict) -> String {
    let _ = (study, metadata);
    todo!("selection_report")
}

/// `f"{value:.{digits}%}"` with a true minus sign.
pub fn percent(value: f64, digits: usize) -> String {
    crate::pyfmt::format(value, &format!(".{digits}%")).replace('-', "−")
}

/// A difference of two rates in percentage points, with a true minus sign.
pub fn points(value: f64) -> String {
    format!("{} pp", crate::pyfmt::format(100.0 * value, "+.2f")).replace('-', "−")
}

/// One row per window for the app table.
pub fn out_of_sample_frame(result: &OutOfSampleStudy) -> Table {
    let _ = result;
    todo!("selection_report")
}

/// Every path in each window, with annual mean and volatility.
pub fn out_of_sample_paths_frame(result: &OutOfSampleStudy) -> Table {
    let _ = result;
    todo!("selection_report")
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
    let _ = result;
    todo!("selection_report")
}

/// The headline and the explanation sentences.
pub fn out_of_sample_text(findings: &OutOfSampleFindings) -> (String, Vec<String>) {
    let _ = findings;
    todo!("selection_report")
}

pub fn out_of_sample_chart(result: &OutOfSampleStudy) -> Figure {
    let _ = result;
    todo!("selection_report")
}

pub fn selection_benchmark_note(study: &SelectionStudy) -> String {
    let _ = study;
    todo!("selection_report")
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
    let _ = (study, prices, metadata, frontier, out_of_sample);
    todo!("selection_report")
}
