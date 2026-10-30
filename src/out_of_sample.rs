//! Walk-forward tests of the historical-mean stock model.
//!
//! Each window fits the model on earlier returns only. It buys that allocation
//! once and holds it through the next window, which the fit did not see. The same
//! allocation is also bought and held over its own fit dates. That is the
//! in-sample result. Both sides use the same holding rule.
//!
//! Both sides are measured from the purchase close. The first interval of each
//! path is cash and pays the one entry fee. A short test window would give that
//! one interval far more weight than a long fit window, so it is left out of every
//! walk-forward measure.

use crate::backtest::BacktestStudy;
use crate::dict;
use crate::error::Result;
use crate::frame::{Cell, Frame, Table, mean};
use crate::metrics::{path_returns, performance_metrics};
use crate::pyfmt::Dict;
use crate::selection::{SELECTED, SelectionStudy, concat_rows, fit_targets, simulate};
use crate::value_error;

pub const FIRST_FIT_FRACTION: f64 = 0.5;
pub const WINDOW_COUNT: usize = 5;
pub const MINIMUM_RETURNS: usize = 3;
pub const METRICS: [&str; 5] = ["annual_mean", "volatility", "cagr", "total_return", "max_drawdown"];

#[derive(Clone, Debug, PartialEq)]
pub struct OutOfSampleStudy {
    /// Index `window`; columns `fit_start`, `fit_end`, `test_start`, `test_end`,
    /// `fit_returns`, `test_returns`, `status`. An empty status means a feasible fit.
    pub windows: Table,
    /// Index levels `window`, `sample`, `strategy`; columns [`METRICS`].
    pub metrics: Table,
    /// Index `window`; one column per price column.
    pub weights: Table,
    /// The eligibility decisions of each feasible fit, with a `window` column.
    pub eligibility: Table,
    pub notes: Vec<String>,
    pub settings: Dict,
}

/// Return `(last fit return, last test return)` for each consecutive window.
///
/// The float arithmetic matches the replay split, so the 70% window uses the
/// same fit as the replay.
pub fn window_bounds(count: usize, windows: usize, first_fit: f64) -> Result<Vec<(usize, usize)>> {
    let mut edges: Vec<usize> = (0..windows)
        .map(|k| (count as f64 * (first_fit + (1.0 - first_fit) * k as f64 / windows as f64)).floor() as usize)
        .collect();
    edges.push(count);
    let bounds: Vec<(usize, usize)> = edges.windows(2).map(|pair| (pair[0], pair[1])).collect();
    if edges[0] < MINIMUM_RETURNS || bounds.iter().any(|(split, end)| end < split || end - split < MINIMUM_RETURNS) {
        return Err(value_error!(
            "Out-of-sample tests need at least {MINIMUM_RETURNS} fit returns and {MINIMUM_RETURNS} test returns \
             in each of {windows} windows. This history has {count} returns."
        ));
    }
    Ok(bounds)
}

/// Measure each path from its purchase close, after the cash interval and entry fee.
fn held_metrics(path: &BacktestStudy, settings: &Dict) -> Vec<(String, [f64; 5])> {
    let risk_free_rate = settings.f64("risk_free_rate");
    let periods = settings.f64("periods_per_year");
    path.equity
        .columns
        .iter()
        .enumerate()
        .map(|(j, strategy)| {
            let held = &path.equity.col(j)[1..];
            let metrics = performance_metrics(held, risk_free_rate, periods);
            let annual_mean = mean(&path_returns(held)) * periods;
            (
                strategy.clone(),
                [annual_mean, metrics.volatility, metrics.cagr, metrics.total_return, metrics.max_drawdown],
            )
        })
        .collect()
}

/// Fit on returns before each window, then hold through that window only.
///
/// Window k fits on returns 1..s_k and keeps interval s_k + 1 in cash. It buys
/// at close s_k + 1 and holds to close e_k. The in-sample side keeps interval 1
/// in cash, buys the same weights at close 1, and holds to close s_k. The
/// windows do not overlap and cover the returns after the first fit. A window
/// whose fit is infeasible keeps its dates and reason.
pub fn run_out_of_sample(
    study: &SelectionStudy,
    prices: &Frame,
    windows: usize,
    first_fit: f64,
) -> Result<OutOfSampleStudy> {
    let settings = &study.settings;
    let benchmarks = study.benchmark_prices.as_ref();
    let simulation = dict! {
        "cost_bps" => settings.get("cost_bps").cloned().unwrap(),
        "risk_free_rate" => settings.get("risk_free_rate").cloned().unwrap(),
        "periods_per_year" => settings.get("periods_per_year").cloned().unwrap(),
    };
    let dates = &prices.index;
    let mut window_rows: Vec<(usize, usize, usize, String)> = Vec::new();
    let mut metric_index: Vec<(usize, &'static str, String)> = Vec::new();
    let mut metric_values: Vec<[f64; 5]> = Vec::new();
    let mut weight_rows: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut eligibility: Vec<Table> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for (position, (split, end)) in window_bounds(prices.nrows() - 1, windows, first_fit)?.into_iter().enumerate() {
        let number = position + 1;
        let fit_prices = prices.rows(0, split + 1);
        let fit = match fit_targets(
            &fit_prices,
            settings.f64("target_volatility"),
            settings.f64("max_weight"),
            settings.f64("shrinkage"),
            settings.f64("periods_per_year"),
        ) {
            Ok(fit) => fit,
            Err(error) => {
                window_rows.push((number, split, end, error.message().to_string()));
                continue;
            }
        };
        window_rows.push((number, split, end, String::new()));
        weight_rows.push((number, fit.selected.values.clone()));
        let mut coverage = fit.coverage.clone();
        coverage.push("window", vec![Cell::Int(number as i64); coverage.nrows()]);
        eligibility.push(coverage);
        notes.extend(fit.warnings.iter().map(|message| format!("Window {number}: {message}")));
        let inside_benchmarks = benchmarks.map(|frame| frame.rows(0, split + 1));
        let outside_benchmarks = benchmarks.map(|frame| frame.rows(0, end + 1));
        let paths = [
            ("in_sample", simulate(&fit_prices, inside_benchmarks.as_ref(), 0, split, &fit, &simulation)),
            (
                "out_of_sample",
                simulate(&prices.rows(0, end + 1), outside_benchmarks.as_ref(), split, end, &fit, &simulation),
            ),
        ];
        for (sample, path) in &paths {
            for (strategy, values) in held_metrics(path, &simulation) {
                metric_index.push((number, sample, strategy));
                metric_values.push(values);
            }
        }
    }
    let window_table = Table::new(Some("window"), window_rows.iter().map(|row| Cell::Int(row.0 as i64)).collect())
        .with("fit_start", vec![Cell::Date(dates[1]); window_rows.len()])
        .with("fit_end", window_rows.iter().map(|row| Cell::Date(dates[row.1])).collect())
        .with("test_start", window_rows.iter().map(|row| Cell::Date(dates[row.1 + 1])).collect())
        .with("test_end", window_rows.iter().map(|row| Cell::Date(dates[row.2])).collect())
        .with("fit_returns", window_rows.iter().map(|row| Cell::Int(row.1 as i64)).collect())
        .with("test_returns", window_rows.iter().map(|row| Cell::Int((row.2 - row.1) as i64)).collect())
        .with("status", window_rows.iter().map(|row| Cell::Str(row.3.clone())).collect());
    let mut metrics = Table::multi(
        vec![Some("window".into()), Some("sample".into()), Some("strategy".into())],
        vec![
            metric_index.iter().map(|row| Cell::Int(row.0 as i64)).collect(),
            metric_index.iter().map(|row| Cell::from(row.1)).collect(),
            metric_index.iter().map(|row| Cell::Str(row.2.clone())).collect(),
        ],
    );
    for (k, name) in METRICS.iter().enumerate() {
        metrics.push(*name, metric_values.iter().map(|values| Cell::Float(values[k])).collect());
    }
    let mut weights = Table::new(
        if weight_rows.is_empty() { None } else { Some("window") },
        weight_rows.iter().map(|row| Cell::Int(row.0 as i64)).collect(),
    );
    for (j, column) in prices.columns.iter().enumerate() {
        weights.push(column, weight_rows.iter().map(|row| Cell::Float(row.1[j])).collect());
    }
    let eligibility =
        if eligibility.is_empty() { Table::range(0) } else { concat_rows(&eligibility.iter().collect::<Vec<_>>()) };
    let mut unique_notes: Vec<String> = Vec::new();
    for note in notes {
        if !unique_notes.contains(&note) {
            unique_notes.push(note);
        }
    }
    let mut study_settings = dict! {
        "windows" => windows,
        "first_fit_fraction" => first_fit,
        "holding_policy" => "buy_and_hold",
        "measured_from" => "purchase close; the cash interval and entry fee are excluded",
    };
    study_settings.update(&simulation);
    Ok(OutOfSampleStudy {
        windows: window_table,
        metrics,
        weights,
        eligibility,
        notes: unique_notes,
        settings: study_settings,
    })
}

/// [`run_out_of_sample`] with the default five windows after a 50% first fit.
pub fn run_default_out_of_sample(study: &SelectionStudy, prices: &Frame) -> Result<OutOfSampleStudy> {
    run_out_of_sample(study, prices, WINDOW_COUNT, FIRST_FIT_FRACTION)
}

impl OutOfSampleStudy {
    /// The strategies in engine path order.
    pub fn strategies(&self) -> Vec<String> {
        let mut order: Vec<String> = Vec::new();
        if let Some(level) = self.metrics.index.get(2) {
            for cell in level {
                let name = cell.py_str();
                if !order.contains(&name) {
                    order.push(name);
                }
            }
        }
        order
    }

    /// The completed window numbers, in increasing order.
    pub fn completed_windows(&self) -> Vec<i64> {
        let mut windows: Vec<i64> = Vec::new();
        for cell in &self.metrics.index[0] {
            let number = cell.as_int().unwrap();
            if !windows.contains(&number) {
                windows.push(number);
            }
        }
        windows.sort_unstable();
        windows
    }

    /// One measure for one window, side, and path.
    pub fn value(&self, window: i64, sample: &str, strategy: &str, measure: &str) -> f64 {
        let values = self.metrics.column(measure);
        (0..self.metrics.nrows())
            .find(|i| {
                self.metrics.index[0][*i].as_int() == Some(window)
                    && self.metrics.index[1][*i].as_str() == Some(sample)
                    && self.metrics.index[2][*i].as_str() == Some(strategy)
            })
            .map_or(f64::NAN, |i| values[i].as_f64().unwrap_or(f64::NAN))
    }
}

/// One row per completed window and one column per path, for one side of the split, in engine path order.
pub fn sample_values(result: &OutOfSampleStudy, sample: &str, measure: &str) -> Table {
    let windows = result.completed_windows();
    let mut table = Table::new(Some("window"), windows.iter().map(|w| Cell::Int(*w)).collect());
    for strategy in result.strategies() {
        table.push(
            &strategy,
            windows.iter().map(|w| Cell::Float(result.value(*w, sample, &strategy, measure))).collect(),
        );
    }
    table
}

/// One row per completed window: the stock portfolio in and out of sample, then each reference out of sample.
pub fn portfolio_comparison(result: &OutOfSampleStudy, measure: &str) -> Table {
    if result.metrics.nrows() == 0 {
        return Table::range(0)
            .with("in_sample", Vec::new())
            .with("out_of_sample", Vec::new())
            .with("change", Vec::new());
    }
    let inside = sample_values(result, "in_sample", measure);
    let outside = sample_values(result, "out_of_sample", measure);
    let in_values = inside.f64s(SELECTED);
    let out_values = outside.f64s(SELECTED);
    let mut frame = Table { index_names: inside.index_names.clone(), index: inside.index.clone(), ..Table::range(0) };
    frame.push("in_sample", in_values.iter().map(|v| Cell::Float(*v)).collect());
    frame.push("out_of_sample", out_values.iter().map(|v| Cell::Float(*v)).collect());
    frame.push("change", out_values.iter().zip(&in_values).map(|(o, i)| Cell::Float(o - i)).collect());
    for strategy in &outside.columns {
        if strategy != SELECTED {
            frame.push(strategy, outside.column(strategy).to_vec());
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_consecutive_and_cover_the_final_half() {
        for count in [29, 30, 100, 251, 1259] {
            let bounds = window_bounds(count, WINDOW_COUNT, FIRST_FIT_FRACTION).unwrap();
            assert_eq!(bounds.len(), 5);
            assert_eq!(bounds[0].0, count / 2);
            assert_eq!(bounds[4].1, count);
            for pair in bounds.windows(2) {
                assert_eq!(pair[0].1, pair[1].0);
            }
            // The 70% window uses the replay's split.
            assert_eq!(bounds[2].0, (count as f64 * 0.7).floor() as usize);
        }
    }

    #[test]
    fn too_short_history_raises_an_explicit_reason() {
        let error = window_bounds(28, WINDOW_COUNT, FIRST_FIT_FRACTION).unwrap_err();
        assert_eq!(
            error.message(),
            "Out-of-sample tests need at least 3 fit returns and 3 test returns in each of 5 windows. This history has 28 returns."
        );
    }
}
