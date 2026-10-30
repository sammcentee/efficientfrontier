//! Paired historical comparisons with serial-dependence and multiple-test controls.

use std::collections::HashSet;

use nalgebra::{DMatrix, DVector};

use crate::dict;
use crate::error::Result;
use crate::frame::{Cell, Date, Frame, Table, csv_field, float_text, iso, mean, std, text_label};
use crate::metrics::validate_periods_per_year;
use crate::profiles::array_split;
use crate::pyfmt::Dict;
use crate::value_error;

/// The two-sided 95% normal quantile.
const NORMAL_QUANTILE: f64 = 1.959963984540054;
/// The fewest paired observations that support inference.
pub const INFERENCE_MIN_OBSERVATIONS: usize = 60;

/// Options for [`analyze_evidence`], with the Python defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvidenceOptions {
    pub risk_free_rate: f64,
    pub periods_per_year: f64,
    /// Exclude the first cash and entry interval from the statistical estimates.
    pub delayed_entry: bool,
}

impl Default for EvidenceOptions {
    fn default() -> Self {
        Self { risk_free_rate: 0.02, periods_per_year: 252.0, delayed_entry: true }
    }
}

/// Strategy wealth divided by benchmark wealth, both normalized to one at the first date.
///
/// Python stores these paths as one DataFrame with two column levels named
/// `strategy` and `benchmark`. Here each column has a `(strategy, benchmark)`
/// key. The order is strategy-major: every benchmark for the first strategy,
/// then every benchmark for the next strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct RelativeEquity {
    /// The equity dates.
    pub index: Vec<Date>,
    /// The `(strategy, benchmark)` key of each column.
    pub columns: Vec<(String, String)>,
    /// Shape `(index.len(), columns.len())`.
    pub values: DMatrix<f64>,
}

impl RelativeEquity {
    /// The strategies in column order, without repeats
    /// (`columns.get_level_values("strategy").unique()`).
    pub fn strategies(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for (strategy, _) in &self.columns {
            if !names.contains(strategy) {
                names.push(strategy.clone());
            }
        }
        names
    }

    /// The paths of one strategy with one column per benchmark
    /// (`xs(strategy, axis=1, level="strategy")`). `None` for an unknown strategy.
    pub fn strategy(&self, strategy: &str) -> Option<Frame> {
        let positions: Vec<usize> = (0..self.columns.len()).filter(|j| self.columns[*j].0 == strategy).collect();
        if positions.is_empty() {
            return None;
        }
        let columns =
            positions.iter().map(|j| (self.columns[*j].1.clone(), self.values.column(*j).iter().copied().collect()));
        Some(Frame::from_columns(self.index.clone(), columns.collect()))
    }

    /// Python `csv_text(relative_equity)`: one header row for each column level,
    /// then a row with the index name when the dates have one (the price loaders
    /// name them `Date`), then one row per date.
    pub fn to_csv(&self, index_name: Option<&str>) -> String {
        let mut text = String::new();
        let mut write_row = |fields: Vec<String>| {
            text.push_str(&fields.iter().map(|field| csv_field(field)).collect::<Vec<_>>().join(","));
            text.push_str("\r\n");
        };
        for (level, name) in ["strategy", "benchmark"].into_iter().enumerate() {
            let mut fields = vec![text_label(name)];
            fields.extend(self.columns.iter().map(|key| text_label(if level == 0 { &key.0 } else { &key.1 })));
            write_row(fields);
        }
        if let Some(name) = index_name {
            let mut fields = vec![text_label(name)];
            fields.extend(self.columns.iter().map(|_| String::new()));
            write_row(fields);
        }
        for (i, date) in self.index.iter().enumerate() {
            let mut fields = vec![iso(*date)];
            fields.extend(self.values.row(i).iter().map(|value| float_text(*value)));
            write_row(fields);
        }
        text
    }
}

/// Paired comparisons of every strategy with every benchmark.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceAnalysis {
    /// Index levels `strategy` and `benchmark`. Columns `total_return_difference`,
    /// `cagr_difference`, `annual_advantage`, `beta`, `alpha`, `tracking_error`,
    /// `information_ratio`, `observations` (integer), `advantage_note`, `alpha_note`
    /// (text, empty when inference is available), `advantage_ci_low`, `advantage_ci_high`,
    /// `advantage_pvalue`, `alpha_ci_low`, `alpha_ci_high`, `alpha_pvalue`,
    /// `windows_ahead` (integer), `advantage_pvalue_adjusted`, `alpha_pvalue_adjusted`,
    /// and `status` (text).
    pub summary: Table,
    /// Default index, one row per strategy, benchmark and window. Columns `strategy`,
    /// `benchmark`, `window` (integer from one), `start`, `end` (ISO dates of the first
    /// and last return), `observations` (integer), `relative_return`, `annual_advantage`.
    pub windows: Table,
    pub relative_equity: RelativeEquity,
    pub settings: Dict,
    pub warnings: Vec<String>,
}

fn validate_equity(equity: &Frame) -> Result<()> {
    let unique: HashSet<&String> = equity.columns.iter().collect();
    if equity.is_empty() || unique.len() != equity.ncols() {
        return Err(value_error!("Equity must be a nonempty DataFrame with unique column names."));
    }
    if equity.index.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(value_error!("Equity dates must be unique, increasing, and nonmissing."));
    }
    if equity.nrows() < 2 {
        return Err(value_error!("Equity requires an initial value and at least one later value."));
    }
    if equity.values.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(value_error!("Equity values must be finite and strictly positive."));
    }
    Ok(())
}

fn largest_magnitude(values: &[f64]) -> f64 {
    values.iter().fold(f64::NEG_INFINITY, |largest, value| largest.max(value.abs()))
}

/// `numpy.linalg.lstsq(design, values, rcond=None)[0]`, through the singular value decomposition.
fn least_squares(design: &DMatrix<f64>, values: &DVector<f64>) -> DVector<f64> {
    let svd = design.clone().svd(true, true);
    let largest = svd.singular_values.iter().fold(0.0_f64, |high, value| high.max(*value));
    let cutoff = f64::EPSILON * design.nrows().max(design.ncols()) as f64 * largest;
    svd.solve(values, cutoff).expect("both sets of singular vectors are computed")
}

/// Fit an intercept, with an optional benchmark slope, and Newey-West errors.
///
/// Returns the coefficients, their standard errors, and a note that explains
/// undefined errors.
fn fit_hac(values: &[f64], benchmark: Option<&[f64]>, lags: usize) -> (Vec<f64>, Vec<f64>, &'static str) {
    let count = values.len();
    let terms = if benchmark.is_some() { 2 } else { 1 };
    let undefined = vec![f64::NAN; terms];
    if count == 0 {
        return (undefined.clone(), undefined, "No observations remain after the entry interval.");
    }
    let mut design = DMatrix::from_element(count, terms, 1.0);
    let mut transform = DMatrix::identity(terms, terms);
    if let Some(benchmark) = benchmark {
        let center = mean(benchmark);
        let scale = std(benchmark, 0);
        if scale <= 32.0 * f64::EPSILON * largest_magnitude(benchmark).max(1.0) {
            return (undefined.clone(), undefined, "Benchmark variation is too small to identify alpha and beta.");
        }
        for (i, value) in benchmark.iter().enumerate() {
            design[(i, 1)] = (value - center) / scale;
        }
        transform = DMatrix::from_row_slice(2, 2, &[1.0, -center / scale, 0.0, 1.0 / scale]);
    }
    let values = DVector::from_column_slice(values);
    let fitted = least_squares(&design, &values);
    let coefficients: Vec<f64> = (&transform * &fitted).iter().copied().collect();
    if count < INFERENCE_MIN_OBSERVATIONS {
        return (coefficients, undefined, "At least 60 paired observations are required for inference.");
    }
    let residuals = &values - &design * &fitted;
    if largest_magnitude(residuals.as_slice()) <= 32.0 * f64::EPSILON * largest_magnitude(values.as_slice()).max(1.0) {
        return (coefficients, undefined, "Residual variation is too small for uncertainty estimates.");
    }
    let mut scores = design.clone();
    for (mut row, residual) in scores.row_iter_mut().zip(residuals.iter()) {
        row *= *residual;
    }
    let mut meat = scores.transpose() * &scores;
    for lag in 1..=lags {
        let cross = scores.rows(lag, count - lag).transpose() * scores.rows(0, count - lag);
        meat += (&cross + cross.transpose()) * (1.0 - lag as f64 / (lags + 1) as f64);
    }
    let inverse = (design.transpose() * &design).try_inverse().expect("the design columns are independent");
    let covariance =
        &transform * &inverse * &meat * &inverse * transform.transpose() * count as f64 / (count - terms) as f64;
    let errors: Vec<f64> = covariance
        .diagonal()
        .iter()
        .map(|variance| variance.max(0.0).sqrt())
        .map(|error| if error <= 0.0 { f64::NAN } else { error })
        .collect();
    let note = if errors[0].is_finite() { "" } else { "The uncertainty estimate is undefined." };
    (coefficients, errors, note)
}

/// Adjust one fixed family. Undefined tests retain their place in the family.
fn holm_adjust(pvalues: &[f64]) -> Vec<f64> {
    let values: Vec<f64> = pvalues.iter().map(|value| if value.is_finite() { *value } else { 1.0 }).collect();
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut adjusted = vec![f64::NAN; values.len()];
    let mut running = f64::NEG_INFINITY;
    for (rank, position) in order.into_iter().enumerate() {
        running = running.max(values[position] * (values.len() - rank) as f64);
        if pvalues[position].is_finite() {
            adjusted[position] = running.min(1.0);
        }
    }
    adjusted
}

/// A 95% confidence interval and a two-sided p-value, in annual units.
#[derive(Clone, Copy)]
struct Test {
    ci_low: f64,
    ci_high: f64,
    pvalue: f64,
}

impl Test {
    /// Normal inference for an estimate per period and its standard error.
    fn new(estimate: f64, error: f64, periods_per_year: f64) -> Self {
        Test {
            ci_low: (estimate - NORMAL_QUANTILE * error) * periods_per_year,
            ci_high: (estimate + NORMAL_QUANTILE * error) * periods_per_year,
            pvalue: if error.is_finite() { libm::erfc((estimate / error).abs() / 2f64.sqrt()) } else { f64::NAN },
        }
    }
}

/// The estimates of one strategy and benchmark pair.
struct Row {
    strategy: String,
    benchmark: String,
    total_return_difference: f64,
    cagr_difference: f64,
    annual_advantage: f64,
    beta: f64,
    alpha: f64,
    tracking_error: f64,
    information_ratio: f64,
    advantage_note: &'static str,
    alpha_note: &'static str,
    advantage_test: Test,
    alpha_test: Test,
    windows_ahead: usize,
}

/// One chronological window of one pair.
struct WindowRow {
    strategy: String,
    benchmark: String,
    window: usize,
    start: Date,
    end: Date,
    observations: usize,
    relative_return: f64,
    annual_advantage: f64,
}

/// Compare net equity paths on identical dates without a future forecast.
///
/// Full-path results and nonoverlap windows retain entry costs. Statistical
/// estimates exclude the first cash and entry interval when delayed_entry is
/// true. HAC uses Bartlett weights and a finite-sample covariance correction.
/// Holm adjustment covers all strategy-benchmark pairs and both return tests.
/// Individual confidence intervals do not include a multiple-test adjustment.
pub fn analyze_evidence(
    strategy_equity: &Frame,
    benchmark_equity: &Frame,
    options: &EvidenceOptions,
) -> Result<EvidenceAnalysis> {
    validate_equity(strategy_equity)?;
    validate_equity(benchmark_equity)?;
    if strategy_equity.index != benchmark_equity.index {
        return Err(value_error!("Strategy and benchmark equity must have exactly the same dates."));
    }
    let periods_per_year = validate_periods_per_year(options.periods_per_year)?;
    let EvidenceOptions { risk_free_rate, delayed_entry, .. } = *options;
    if !risk_free_rate.is_finite() {
        return Err(value_error!("risk_free_rate must be a finite number."));
    }
    let strategy_returns = strategy_equity.pct_change();
    let benchmark_returns = benchmark_equity.pct_change();
    if strategy_returns.values.iter().chain(benchmark_returns.values.iter()).any(|value| !value.is_finite()) {
        return Err(value_error!("Equity paths must produce finite returns."));
    }
    let offset = usize::from(delayed_entry);
    let full_count = strategy_returns.nrows();
    let inference_dates = &strategy_returns.index[offset.min(full_count)..];
    let count = inference_dates.len();
    let bandwidth = (4.0 * (count as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
    let lags = count.saturating_sub(1).min(bandwidth);
    let chunks = array_split(full_count, full_count.min(3));
    let normalized = |equity: &Frame, j: usize| -> Vec<f64> {
        let column = equity.col(j);
        column.iter().map(|value| value / column[0]).collect()
    };
    let per_period_rate = risk_free_rate / periods_per_year;
    let mut rows: Vec<Row> = Vec::new();
    let mut window_rows: Vec<WindowRow> = Vec::new();
    let mut relative_columns: Vec<((String, String), Vec<f64>)> = Vec::new();
    for (s, strategy) in strategy_equity.columns.iter().enumerate() {
        for (b, benchmark) in benchmark_equity.columns.iter().enumerate() {
            let full_strategy = strategy_returns.col(s);
            let full_benchmark = benchmark_returns.col(b);
            let active: Vec<f64> = (offset..full_count).map(|i| full_strategy[i] - full_benchmark[i]).collect();
            let (advantage, advantage_error, advantage_note) = fit_hac(&active, None, lags);
            let excess =
                |returns: &[f64]| returns[offset..].iter().map(|value| value - per_period_rate).collect::<Vec<_>>();
            let (coefficients, alpha_error, alpha_note) =
                fit_hac(&excess(&full_strategy), Some(&excess(&full_benchmark)), lags);
            let strategy_path = normalized(strategy_equity, s);
            let benchmark_path = normalized(benchmark_equity, b);
            let relative: Vec<f64> = strategy_path.iter().zip(&benchmark_path).map(|(a, z)| a / z).collect();
            let strategy_growth = *strategy_path.last().unwrap();
            let benchmark_growth = *benchmark_path.last().unwrap();
            let mut tracking_error = if count > 1 { std(&active, 1) * periods_per_year.sqrt() } else { f64::NAN };
            let spread = active.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - active.iter().copied().fold(f64::INFINITY, f64::min);
            if count > 0 && spread <= 32.0 * f64::EPSILON * largest_magnitude(&active).max(1.0) {
                tracking_error = 0.0;
            }
            let growth = |value: f64| (value.ln() * periods_per_year / full_count as f64).exp_m1();
            let annual_advantage = advantage[0] * periods_per_year;
            let mut ahead = 0;
            for (number, positions) in chunks.iter().enumerate() {
                let (first, last) = (positions.start, positions.end - 1);
                let relative_return = relative[last + 1] / relative[first] - 1.0;
                ahead += usize::from(relative_return > 0.0);
                let differences: Vec<f64> = positions.clone().map(|i| full_strategy[i] - full_benchmark[i]).collect();
                window_rows.push(WindowRow {
                    strategy: strategy.clone(),
                    benchmark: benchmark.clone(),
                    window: number + 1,
                    start: strategy_returns.index[first],
                    end: strategy_returns.index[last],
                    observations: positions.len(),
                    relative_return,
                    annual_advantage: mean(&differences) * periods_per_year,
                });
            }
            rows.push(Row {
                strategy: strategy.clone(),
                benchmark: benchmark.clone(),
                total_return_difference: strategy_growth - benchmark_growth,
                cagr_difference: growth(strategy_growth) - growth(benchmark_growth),
                annual_advantage,
                beta: coefficients[1],
                alpha: coefficients[0] * periods_per_year,
                tracking_error,
                information_ratio: if tracking_error > 0.0 { annual_advantage / tracking_error } else { f64::NAN },
                advantage_note,
                alpha_note,
                advantage_test: Test::new(advantage[0], advantage_error[0], periods_per_year),
                alpha_test: Test::new(coefficients[0], alpha_error[0], periods_per_year),
                windows_ahead: ahead,
            });
            relative_columns.push(((strategy.clone(), benchmark.clone()), relative));
        }
    }

    let pvalues: Vec<f64> = rows.iter().flat_map(|row| [row.advantage_test.pvalue, row.alpha_test.pvalue]).collect();
    let adjusted = holm_adjust(&pvalues);
    let statuses: Vec<&str> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let advantage_passes = adjusted[2 * i] <= 0.05;
            let alpha_passes = adjusted[2 * i + 1] <= 0.05 && row.alpha > 0.0;
            if count < INFERENCE_MIN_OBSERVATIONS {
                "Insufficient history"
            } else if !row.advantage_test.pvalue.is_finite() && !row.alpha_test.pvalue.is_finite() {
                "Uncertainty unavailable"
            } else if advantage_passes && row.annual_advantage > 0.0 {
                if alpha_passes { "Historical mean advantage and alpha" } else { "Historical mean advantage" }
            } else if advantage_passes && row.annual_advantage < 0.0 {
                "Historical mean underperformance"
            } else if alpha_passes {
                "Historical alpha"
            } else {
                "No clear statistical advantage"
            }
        })
        .collect();
    let floats = |value: fn(&Row) -> f64| rows.iter().map(|row| Cell::Float(value(row))).collect::<Vec<_>>();
    let texts = |value: fn(&Row) -> &str| rows.iter().map(|row| Cell::from(value(row))).collect::<Vec<_>>();
    let summary = Table::multi(
        vec![Some("strategy".to_string()), Some("benchmark".to_string())],
        vec![texts(|row| &row.strategy), texts(|row| &row.benchmark)],
    )
    .with("total_return_difference", floats(|row| row.total_return_difference))
    .with("cagr_difference", floats(|row| row.cagr_difference))
    .with("annual_advantage", floats(|row| row.annual_advantage))
    .with("beta", floats(|row| row.beta))
    .with("alpha", floats(|row| row.alpha))
    .with("tracking_error", floats(|row| row.tracking_error))
    .with("information_ratio", floats(|row| row.information_ratio))
    .with("observations", vec![Cell::from(count); rows.len()])
    .with("advantage_note", texts(|row| row.advantage_note))
    .with("alpha_note", texts(|row| row.alpha_note))
    .with("advantage_ci_low", floats(|row| row.advantage_test.ci_low))
    .with("advantage_ci_high", floats(|row| row.advantage_test.ci_high))
    .with("advantage_pvalue", floats(|row| row.advantage_test.pvalue))
    .with("alpha_ci_low", floats(|row| row.alpha_test.ci_low))
    .with("alpha_ci_high", floats(|row| row.alpha_test.ci_high))
    .with("alpha_pvalue", floats(|row| row.alpha_test.pvalue))
    .with("windows_ahead", rows.iter().map(|row| Cell::from(row.windows_ahead)).collect())
    .with("advantage_pvalue_adjusted", adjusted.iter().step_by(2).map(|value| Cell::Float(*value)).collect())
    .with("alpha_pvalue_adjusted", adjusted.iter().skip(1).step_by(2).map(|value| Cell::Float(*value)).collect())
    .with("status", statuses.into_iter().map(Cell::from).collect());

    let windows = Table::range(window_rows.len())
        .with("strategy", window_rows.iter().map(|row| Cell::from(row.strategy.as_str())).collect())
        .with("benchmark", window_rows.iter().map(|row| Cell::from(row.benchmark.as_str())).collect())
        .with("window", window_rows.iter().map(|row| Cell::from(row.window)).collect())
        .with("start", window_rows.iter().map(|row| Cell::from(iso(row.start))).collect())
        .with("end", window_rows.iter().map(|row| Cell::from(iso(row.end))).collect())
        .with("observations", window_rows.iter().map(|row| Cell::from(row.observations)).collect())
        .with("relative_return", window_rows.iter().map(|row| Cell::Float(row.relative_return)).collect())
        .with("annual_advantage", window_rows.iter().map(|row| Cell::Float(row.annual_advantage)).collect());
    let mut values = DMatrix::zeros(strategy_equity.nrows(), relative_columns.len());
    for (j, (_, path)) in relative_columns.iter().enumerate() {
        values.column_mut(j).copy_from_slice(path);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(value_error!("Equity paths must produce finite relative values."));
    }
    let relative_equity = RelativeEquity {
        index: strategy_equity.index.clone(),
        columns: relative_columns.into_iter().map(|(key, _)| key).collect(),
        values,
    };
    let mut warnings: Vec<String> = [
        "These tests describe historical paths. They do not estimate the probability of future outperformance.",
        "Holm adjustment covers this report only. It does not account for prior strategy searches or repeated changes after review.",
        "HAC inference assumes consistently spaced observations and stable enough return relationships. Structural changes can invalidate the estimates.",
    ]
    .map(String::from)
    .to_vec();
    if delayed_entry {
        warnings.push(
            "Statistical estimates exclude the first cash and entry interval. Full-path results and windows retain entry costs."
                .to_string(),
        );
    }
    if count < INFERENCE_MIN_OBSERVATIONS {
        warnings.push(
            "Fewer than 60 paired observations remain. The report shows descriptive results without statistical inference."
                .to_string(),
        );
    }
    if chunks.iter().any(|positions| (positions.len() as f64) < periods_per_year) {
        warnings.push(
            "At least one comparison window has less than one year of observations. Window results can be unstable."
                .to_string(),
        );
    }
    let settings = dict! {
        "periods_per_year" => periods_per_year,
        "risk_free_rate" => risk_free_rate,
        "delayed_entry" => delayed_entry,
        "inference_min_observations" => INFERENCE_MIN_OBSERVATIONS,
        "inference_observations" => count,
        "evaluation_observations" => full_count,
        "hac_lags" => lags,
        "confidence_level" => 0.95,
        "multiple_testing_method" => "Holm",
        "multiple_testing_tests" => rows.len() * 2,
        "pvalue_sides" => "two-sided",
        "inference_start" => inference_dates.first().map(|date| iso(*date)),
        "inference_end" => inference_dates.last().map(|date| iso(*date)),
        "window_count" => chunks.len(),
    };
    Ok(EvidenceAnalysis { summary, windows, relative_equity, settings, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a.is_nan() && e.is_nan()) || (a - e).abs() <= 1e-12, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn holm_uses_the_full_family_and_retains_undefined_tests() {
        assert_all_close(&holm_adjust(&[0.01, 0.04, 0.03, f64::NAN]), &[0.04, 0.09, 0.09, f64::NAN]);
        assert_all_close(&holm_adjust(&[0.02, 0.02, 0.8]), &[0.06, 0.06, 0.8]);
    }

    #[test]
    fn least_squares_recovers_an_exact_line() {
        let design = DMatrix::from_row_slice(3, 2, &[1.0, -1.0, 1.0, 0.0, 1.0, 1.0]);
        let fitted = least_squares(&design, &DVector::from_column_slice(&[1.0, 3.0, 5.0]));
        assert!((fitted[0] - 3.0).abs() < 1e-12 && (fitted[1] - 2.0).abs() < 1e-12, "{fitted}");
    }

    #[test]
    fn hac_notes_explain_undefined_estimates() {
        assert_eq!(fit_hac(&[], None, 0).2, "No observations remain after the entry interval.");
        assert_eq!(
            fit_hac(&[0.1, 0.2], Some(&[0.3, 0.3]), 0).2,
            "Benchmark variation is too small to identify alpha and beta."
        );
        let (coefficients, errors, note) = fit_hac(&[0.1, 0.3], None, 0);
        assert!((coefficients[0] - 0.2).abs() < 1e-15 && errors[0].is_nan());
        assert_eq!(note, "At least 60 paired observations are required for inference.");
        assert_eq!(fit_hac(&[0.5; 60], None, 2).2, "Residual variation is too small for uncertainty estimates.");
    }

    #[test]
    fn relative_equity_writes_two_header_levels() {
        let day = |d| Date::from_ymd_opt(2020, 1, d).unwrap();
        let equity = RelativeEquity {
            index: vec![day(1), day(2)],
            columns: vec![("=A".into(), "S&P, 500".into()), ("=A".into(), "QQQ".into())],
            values: DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.5, f64::NAN]),
        };
        assert_eq!(
            equity.to_csv(Some("Date")),
            "strategy,'=A,'=A\r\nbenchmark,\"S&P, 500\",QQQ\r\nDate,,\r\n2020-01-01,1.0,1.0\r\n2020-01-02,1.5,\r\n"
        );
        assert_eq!(equity.to_csv(None).lines().count(), 4);
        assert_eq!(equity.strategies(), vec!["=A".to_string()]);
        assert_eq!(equity.strategy("=A").unwrap().columns, vec!["S&P, 500".to_string(), "QQQ".to_string()]);
        assert!(equity.strategy("B").is_none());
    }
}
