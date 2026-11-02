//! Relative risk profiles from historical means, with optional window research.

use std::ops::Range;
use std::str::FromStr;

use indexmap::IndexMap;
use nalgebra::{Cholesky, DVector};

use crate::core::{
    Frontier, Optimization, Portfolio, TRADING_DAYS, dot, estimates, linspace, optimize, quadratic_form, solve_checked,
    validate_price_frame,
};
use crate::dict;
use crate::error::{Error, Result};
use crate::frame::{Cell, Frame, Series, Table, iso};
use crate::metrics::validate_periods_per_year;
use crate::pyfmt::Dict;
use crate::solver::{Program, Row, Status};
use crate::{runtime_error, value_error};

/// The profile names, from the lowest to the highest estimated risk.
pub const PROFILE_NAMES: [&str; 3] = ["Low", "Medium", "Extreme"];
/// The number of chronological windows.
pub const WINDOW_COUNT: usize = 3;

const INACCURATE_WARNING: &str = "The profile solver reported limited numerical accuracy. \
    Weight and return checks passed, but the optimum can be approximate.";
/// The allowed shortfall of a worst-window target, in solver units.
const RETURN_TOLERANCE: f64 = 1e-8;

/// The estimate that orders the profile frontier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReturnObjective {
    /// Minimize variance at each full-sample arithmetic mean target (`historical_mean`).
    #[default]
    HistoricalMean,
    /// Maximize the weakest of the three window means (`worst_window`).
    WorstWindow,
}

impl ReturnObjective {
    /// The Python name of the objective.
    pub fn as_str(self) -> &'static str {
        match self {
            ReturnObjective::HistoricalMean => "historical_mean",
            ReturnObjective::WorstWindow => "worst_window",
        }
    }

    /// The `selection_method` setting of the objective.
    pub fn selection_method(self) -> &'static str {
        match self {
            ReturnObjective::HistoricalMean => "historical_mean_frontier",
            ReturnObjective::WorstWindow => "worst_window_return_frontier",
        }
    }
}

impl FromStr for ReturnObjective {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        match text {
            "historical_mean" => Ok(ReturnObjective::HistoricalMean),
            "worst_window" => Ok(ReturnObjective::WorstWindow),
            _ => Err(value_error!("return_objective must be historical_mean or worst_window.")),
        }
    }
}

/// Options for [`build_profiles`], with the Python defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileOptions {
    pub risk_free_rate: f64,
    pub max_weight: f64,
    pub shrinkage: f64,
    pub periods_per_year: f64,
    pub frontier_points: usize,
    pub return_objective: ReturnObjective,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.02,
            max_weight: 1.0,
            shrinkage: 0.1,
            periods_per_year: TRADING_DAYS,
            frontier_points: 21,
            return_objective: ReturnObjective::HistoricalMean,
        }
    }
}

/// Low, Medium, and Extreme profiles with their frontier and window diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileAnalysis {
    /// `Low`, `Medium`, and `Extreme`, in that order. Each name is also the portfolio name.
    pub portfolios: IndexMap<String, Portfolio>,
    /// Default index, one row per frontier point in order of increasing target return.
    /// Columns `expected_return`, `volatility`, `sharpe`, `worst_window_return`.
    pub frontier: Table,
    /// Default index, one row per frontier point; one column per asset.
    pub frontier_weights: Table,
    /// Index `window` (1, 2, 3). Columns `start` and `end` (ISO dates of the first
    /// and last return), `observations` (integer) and `years` (observations per year).
    pub windows: Table,
    /// Index `window`; one column per profile with the annual window mean of its allocation.
    pub window_returns: Table,
    /// Index `profile` (the three names). Columns `expected_return`, `volatility`,
    /// `sharpe`, `worst_window_return`, `max_weight`, `effective_holdings`.
    pub summary: Table,
    /// The last price date, `YYYY-MM-DD`.
    pub as_of: String,
    /// The number of returns.
    pub observations: usize,
    pub settings: Dict,
    pub warnings: Vec<String>,
}

/// `numpy.array_split(numpy.arange(count), sections)` as ranges of positions.
pub(crate) fn array_split(count: usize, sections: usize) -> Vec<Range<usize>> {
    let (each, extras) = (count / sections, count % sections);
    let mut start = 0;
    (0..sections)
        .map(|section| {
            let size = each + usize::from(section < extras);
            start += size;
            start - size..start
        })
        .collect()
}

/// The first position of the smallest value, as `numpy.argmin`.
fn argmin(values: &[f64]) -> usize {
    (0..values.len()).fold(0, |best, i| if values[i] < values[best] { i } else { best })
}

/// The smallest window mean of an allocation.
fn worst_window(window_means: &[Vec<f64>], weights: &[f64]) -> f64 {
    window_means.iter().map(|means| dot(means, weights)).fold(f64::INFINITY, f64::min)
}

/// Coefficients over the weights, which start at variable `offset`.
fn weight_row(values: &[f64], offset: usize) -> Row {
    values.iter().enumerate().map(|(j, value)| (j + offset, *value)).collect()
}

/// Long-only, capped and fully invested weights. CVXPY puts a score variable
/// before the weights, so `offset` is the position of the first weight.
fn budget(n: usize, offset: usize, max_weight: f64) -> Program {
    let mut program = Program::new(n + offset).equal(weight_row(&vec![1.0; n], offset), 1.0);
    for j in offset..offset + n {
        program = program.at_least(vec![(j, 1.0)], 0.0);
    }
    for j in offset..offset + n {
        program = program.at_most(vec![(j, 1.0)], max_weight);
    }
    program
}

/// Maximize a score (variable 0) that no window mean exceeds.
fn score_rows(mut program: Program, solver_means: &[Vec<f64>]) -> Program {
    let n = program.variables() - 1;
    for means in solver_means {
        let mut row = vec![(0, -1.0)];
        row.extend(weight_row(means, 1));
        program = program.at_least(row, 0.0);
    }
    let mut objective = vec![0.0; n + 1];
    objective[0] = 1.0;
    program.maximize(&objective)
}

/// Add the accuracy warning once when the solver reports limited accuracy.
fn note_accuracy(status: Status, warnings: &mut Vec<String>) {
    if status == Status::OptimalInaccurate && !warnings.iter().any(|warning| warning == INACCURATE_WARNING) {
        warnings.push(INACCURATE_WARNING.to_string());
    }
}

/// Use all supplied returns to select Low, Medium, and Extreme profiles.
///
/// By default, minimize variance at each full-sample arithmetic mean target.
/// Low has minimum volatility. Medium is the sampled point nearest halfway
/// between the endpoint volatilities. Extreme has maximum feasible mean return.
/// At tied minimum risk, prefer the highest objective return.
/// Three chronological windows supply diagnostics for these same allocations.
///
/// The optional worst_window objective maximizes the weakest window return.
/// Its Medium uses the midpoint return target, with minimum variance as a
/// tie-break. Targets allow 1e-8 times the largest absolute window estimate.
/// All profile labels express relative risk, not a return guarantee.
pub fn build_profiles(prices: &Frame, options: &ProfileOptions) -> Result<ProfileAnalysis> {
    validate_price_frame(prices)?;
    let periods_per_year = validate_periods_per_year(options.periods_per_year)?;
    let ProfileOptions { risk_free_rate, max_weight, shrinkage, frontier_points, return_objective, .. } = *options;
    if !risk_free_rate.is_finite() {
        return Err(value_error!("risk_free_rate must be finite."));
    }
    if !max_weight.is_finite() || !(max_weight > 0.0 && max_weight <= 1.0) {
        return Err(value_error!("max_weight must be greater than zero and at most one."));
    }
    if prices.ncols() as f64 * max_weight < 1.0 - 1e-12 {
        return Err(value_error!("The position cap is infeasible: asset count * max_weight < 1."));
    }
    if !shrinkage.is_finite() || !(0.0..=1.0).contains(&shrinkage) {
        return Err(value_error!("shrinkage must be between zero and one."));
    }
    if frontier_points < 3 {
        return Err(value_error!("frontier_points must be an integer of at least three."));
    }
    if return_objective == ReturnObjective::WorstWindow && frontier_points % 2 != 1 {
        return Err(value_error!("worst_window frontier_points must be an odd integer of at least three."));
    }
    let returns = prices.pct_change();
    if returns.nrows() < 6 {
        return Err(value_error!("Profiles require at least six returns, with two returns per historical window."));
    }
    let n = prices.ncols();
    let labels = prices.columns.clone();
    let chunks = array_split(returns.nrows(), WINDOW_COUNT);
    let window_means: Vec<Vec<f64>> = chunks
        .iter()
        .map(|chunk| returns.rows(chunk.start, chunk.end).means().iter().map(|mean| mean * periods_per_year).collect())
        .collect();
    let (mean_series, labelled_covariance, _) = estimates(&returns, periods_per_year, shrinkage);
    let mean_returns = mean_series.values.clone();
    let covariance = labelled_covariance.values.clone();
    if window_means.iter().flatten().any(|value| !value.is_finite())
        || covariance.iter().any(|value| !value.is_finite())
    {
        return Err(value_error!("Return estimates and covariance must be finite."));
    }

    let covariance_scale = covariance.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let solver_covariance = if covariance_scale != 0.0 { &covariance / covariance_scale } else { covariance.clone() };
    let largest_mean = window_means.iter().flatten().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let return_scale = if largest_mean != 0.0 { largest_mean } else { 1.0 };
    let solver_means: Vec<Vec<f64>> =
        window_means.iter().map(|means| means.iter().map(|mean| mean / return_scale).collect()).collect();
    let mut warnings: Vec<String> = Vec::new();

    let solve = |program: &Program, warnings: &mut Vec<String>| -> Result<Vec<f64>> {
        let solution = solve_checked(program)?;
        note_accuracy(solution.status, warnings);
        Ok(solution.x)
    };
    let portfolio = |name: &str, values: &[f64]| -> Result<Portfolio> {
        let sum: f64 = values.iter().sum();
        let lowest = values.iter().copied().fold(f64::INFINITY, f64::min);
        let highest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if values.iter().any(|value| !value.is_finite())
            || (sum - 1.0).abs() > 1e-6
            || lowest < -1e-6
            || highest > max_weight + 1e-6
        {
            return Err(runtime_error!("Solver returned infeasible weights for {name}."));
        }
        let clipped: Vec<f64> = values.iter().map(|value| value.clamp(0.0, max_weight)).collect();
        let total: f64 = clipped.iter().sum();
        let allocation: Vec<f64> = clipped.iter().map(|value| value / total).collect();
        let expected = dot(&mean_returns, &allocation);
        let volatility = quadratic_form(&allocation, &covariance).max(0.0).sqrt();
        let sharpe = if volatility > 0.0 { (expected - risk_free_rate) / volatility } else { f64::NAN };
        Ok(Portfolio {
            name: name.to_string(),
            weights: Series::new(labels.clone(), allocation),
            expected_return: expected,
            volatility,
            sharpe,
        })
    };

    let (frontier, middle) = match return_objective {
        ReturnObjective::HistoricalMean => {
            let Optimization { frontier, warnings: messages, .. } =
                optimize(&mean_series, &labelled_covariance, risk_free_rate, max_weight, frontier_points)?;
            warnings = messages;
            let last = frontier.len() - 1;
            let midpoint_volatility = (frontier.volatility[0] + frontier.volatility[last]) / 2.0;
            let distances: Vec<f64> =
                frontier.volatility.iter().map(|volatility| (volatility - midpoint_volatility).abs()).collect();
            if frontier.expected_return[last] <= 0.0 {
                warnings.push("No feasible portfolio has a positive estimated historical mean return.".to_string());
            }
            (frontier, argmin(&distances))
        }
        ReturnObjective::WorstWindow => {
            // The standard half-variance scale leaves the frontier unchanged.
            let mut minimum = portfolio(
                "Low",
                &solve(&budget(n, 0, max_weight).quadratic(solver_covariance.clone()), &mut warnings)?,
            )?;
            let singular_covariance = match Cholesky::new(solver_covariance.clone()) {
                Some(factor) => factor.l().diagonal().iter().fold(f64::INFINITY, |low, value| low.min(*value)) <= 1e-7,
                None => true,
            };
            if singular_covariance {
                let target = &solver_covariance * DVector::from_column_slice(&minimum.weights.values);
                let mut program = budget(n, 1, max_weight);
                for (i, value) in target.iter().enumerate() {
                    program = program.equal(weight_row(solver_covariance.row(i).transpose().as_slice(), 1), *value);
                }
                minimum = portfolio("Low", &solve(&score_rows(program, &solver_means), &mut warnings)?[1..])?;
            }
            let low_score = worst_window(&solver_means, &minimum.weights.values);
            let endpoint = portfolio(
                "Extreme",
                &solve(&score_rows(budget(n, 1, max_weight), &solver_means), &mut warnings)?[1..],
            )?;
            let high_score = low_score.max(worst_window(&solver_means, &endpoint.weights.values));
            let mut points = vec![minimum.clone()];
            if high_score - low_score <= RETURN_TOLERANCE {
                points = vec![minimum; frontier_points];
            } else {
                for required in linspace(low_score, high_score, frontier_points).into_iter().skip(1) {
                    // Price ratios can break an exact return tie at floating-point precision.
                    let mut program = budget(n, 0, max_weight).quadratic(solver_covariance.clone());
                    for means in &solver_means {
                        program = program.at_least(weight_row(means, 0), required - RETURN_TOLERANCE);
                    }
                    let point = portfolio("Frontier", &solve(&program, &mut warnings)?)?;
                    if worst_window(&solver_means, &point.weights.values) < required - 2.0 * RETURN_TOLERANCE {
                        return Err(runtime_error!("Profile solution missed its required worst-window return."));
                    }
                    points.push(point);
                }
            }
            if high_score <= 0.0 {
                warnings.push(
                    "No feasible portfolio has a positive estimated return in all three historical windows."
                        .to_string(),
                );
            }
            let frontier = Frontier {
                labels: labels.clone(),
                expected_return: points.iter().map(|point| point.expected_return).collect(),
                volatility: points.iter().map(|point| point.volatility).collect(),
                sharpe: points.iter().map(|point| point.sharpe).collect(),
                weights: points.iter().map(|point| point.weights.values.clone()).collect(),
            };
            (frontier, frontier_points / 2)
        }
    };

    let floats = |values: &[f64]| values.iter().map(|value| Cell::Float(*value)).collect::<Vec<_>>();
    let worst_window_return: Vec<f64> =
        frontier.weights.iter().map(|weights| worst_window(&window_means, weights)).collect();
    let last = frontier.len() - 1;
    let selected = [0, middle, last];
    let mut portfolios = IndexMap::new();
    for (name, position) in PROFILE_NAMES.iter().zip(selected) {
        portfolios.insert(
            name.to_string(),
            Portfolio {
                name: name.to_string(),
                weights: Series::new(labels.clone(), frontier.weights[position].clone()),
                expected_return: frontier.expected_return[position],
                volatility: frontier.volatility[position],
                sharpe: frontier.sharpe[position],
            },
        );
    }
    let mut frontier_table = frontier.points_table();
    frontier_table.push("worst_window_return", floats(&worst_window_return));
    let mut summary = frontier_table.take_rows(&selected);
    summary.index_names = vec![Some("profile".to_string())];
    summary.index = vec![PROFILE_NAMES.iter().map(|name| Cell::from(*name)).collect()];
    summary.push("max_weight", portfolios.values().map(|point| Cell::Float(point.weights.max())).collect());
    summary.push(
        "effective_holdings",
        portfolios.values().map(|point| Cell::Float(1.0 / dot(&point.weights.values, &point.weights.values))).collect(),
    );
    let window_index: Vec<Cell> = (1..=chunks.len()).map(Cell::from).collect();
    let years: Vec<f64> = chunks.iter().map(|chunk| chunk.len() as f64 / periods_per_year).collect();
    let windows = Table::new(Some("window"), window_index.clone())
        .with("start", chunks.iter().map(|chunk| Cell::from(iso(returns.index[chunk.start]))).collect())
        .with("end", chunks.iter().map(|chunk| Cell::from(iso(returns.index[chunk.end - 1]))).collect())
        .with("observations", chunks.iter().map(|chunk| Cell::from(chunk.len())).collect())
        .with("years", floats(&years));
    let mut window_returns = Table::new(Some("window"), window_index);
    for (name, point) in &portfolios {
        let values: Vec<f64> = window_means.iter().map(|means| dot(means, &point.weights.values)).collect();
        window_returns.push(name, floats(&values));
    }
    if years.iter().any(|value| *value < 1.0) {
        warnings.push(
            "At least one historical window has less than one year of observations. Estimates can be unstable."
                .to_string(),
        );
    }
    if frontier.len() == 1 || frontier.volatility[last] - frontier.volatility[0] <= 1e-8 * covariance_scale.sqrt() {
        warnings.push(
            "The profiles have the same or nearly the same estimated risk. The data provide no distinct risk levels."
                .to_string(),
        );
    }
    Ok(ProfileAnalysis {
        portfolios,
        frontier_weights: frontier.weights_table(),
        frontier: frontier_table,
        windows,
        window_returns,
        summary,
        as_of: iso(*prices.index.last().expect("validated prices have dates")),
        observations: returns.nrows(),
        settings: dict! {
            "periods_per_year" => periods_per_year,
            "risk_free_rate" => risk_free_rate,
            "max_weight" => max_weight,
            "shrinkage" => shrinkage,
            "window_count" => WINDOW_COUNT,
            "return_objective" => return_objective.as_str(),
            "selection_method" => return_objective.selection_method(),
        },
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_split_matches_numpy() {
        assert_eq!(array_split(19, 3), vec![0..7, 7..13, 13..19]);
        assert_eq!(array_split(6, 3), vec![0..2, 2..4, 4..6]);
        assert_eq!(array_split(2, 2), vec![0..1, 1..2]);
        assert_eq!(array_split(1, 1), vec![0..1]);
    }

    #[test]
    fn objective_names_round_trip_and_reject_unknown_text() {
        for objective in [ReturnObjective::HistoricalMean, ReturnObjective::WorstWindow] {
            assert_eq!(objective.as_str().parse::<ReturnObjective>().unwrap(), objective);
        }
        assert_eq!(
            "unknown".parse::<ReturnObjective>().unwrap_err().message(),
            "return_objective must be historical_mean or worst_window."
        );
        assert_eq!(ReturnObjective::WorstWindow.selection_method(), "worst_window_return_frontier");
    }

    #[test]
    fn approximate_solver_status_adds_one_visible_warning() {
        // Port of test_approximate_solver_status_is_visible_in_the_result: every
        // worst-window solve passes its status here.
        let mut warnings = Vec::new();
        note_accuracy(Status::Optimal, &mut warnings);
        assert!(warnings.is_empty());
        note_accuracy(Status::OptimalInaccurate, &mut warnings);
        note_accuracy(Status::OptimalInaccurate, &mut warnings);
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("limited numerical accuracy") && warnings[0].contains("optimum can be approximate")
        );
    }

    #[test]
    fn argmin_returns_the_first_minimum() {
        assert_eq!(argmin(&[3.0, 1.0, 1.0, 2.0]), 1);
        assert_eq!(argmin(&[0.5]), 0);
    }

    #[test]
    fn budget_matches_the_cvxpy_layout_with_a_leading_score() {
        let program = score_rows(budget(2, 1, 0.9), &[vec![0.1, 0.2], vec![0.3, 0.1]]);
        assert_eq!(program.variables(), 3);
        let solution = program.solve().unwrap();
        // Maximize min(0.1 a + 0.2 b, 0.3 a + 0.1 b) with a + b = 1: a = 1/3.
        assert!((solution.x[1] - 1.0 / 3.0).abs() < 1e-7, "{:?}", solution.x);
        assert!((solution.x[0] - (0.1 / 3.0 + 0.4 / 3.0)).abs() < 1e-7);
    }
}
