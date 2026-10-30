//! Long-only mean-variance optimization and a chronological holdout experiment.

use std::collections::HashSet;

use indexmap::IndexMap;
use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};

use crate::error::Result;
use crate::frame::{Cell, Frame, LabeledMatrix, Series, Table};
use crate::metrics::{Performance, performance_metrics, validate_periods_per_year};
use crate::solver::{Program, Row, Solution, Status};
use crate::{runtime_error, value_error};

pub const TRADING_DAYS: f64 = 252.0;
pub const CLASSIC_PROFILE_NAMES: [&str; 3] = ["Low", "Medium", "High"];

const INACCURATE_WARNING: &str = "The portfolio solver reported limited numerical accuracy. \
    Weight and return checks passed, but the optimum can be approximate.";

#[derive(Clone, Debug, PartialEq)]
pub struct Portfolio {
    pub name: String,
    pub weights: Series,
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe: f64,
}

/// Frontier points in order of increasing required return, with one weight row per point.
#[derive(Clone, Debug, PartialEq)]
pub struct Frontier {
    pub labels: Vec<String>,
    pub expected_return: Vec<f64>,
    pub volatility: Vec<f64>,
    pub sharpe: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
}

impl Frontier {
    pub fn len(&self) -> usize {
        self.expected_return.len()
    }

    pub fn is_empty(&self) -> bool {
        self.expected_return.is_empty()
    }

    /// The `frontier` DataFrame: a default index and three estimate columns.
    pub fn points_table(&self) -> Table {
        Table::range(self.len())
            .with("expected_return", self.expected_return.iter().map(|v| Cell::Float(*v)).collect())
            .with("volatility", self.volatility.iter().map(|v| Cell::Float(*v)).collect())
            .with("sharpe", self.sharpe.iter().map(|v| Cell::Float(*v)).collect())
    }

    /// The `frontier_weights` DataFrame: a default index and one column per asset.
    pub fn weights_table(&self) -> Table {
        let mut table = Table::range(self.len());
        for (j, label) in self.labels.iter().enumerate() {
            table.push(label, self.weights.iter().map(|row| Cell::Float(row[j])).collect());
        }
        table
    }

    fn keep_last(&mut self) {
        let last = self.len() - 1;
        self.expected_return = vec![self.expected_return[last]];
        self.volatility = vec![self.volatility[last]];
        self.sharpe = vec![self.sharpe[last]];
        self.weights = vec![self.weights[last].clone()];
    }
}

/// The result of [`optimize`].
#[derive(Clone, Debug, PartialEq)]
pub struct Optimization {
    pub portfolios: IndexMap<String, Portfolio>,
    pub frontier: Frontier,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub portfolios: IndexMap<String, Portfolio>,
    pub frontier: Frontier,
    pub train_returns: Frame,
    pub test_returns: Frame,
    pub equity: Frame,
    pub holdout_metrics: IndexMap<String, Performance>,
    pub mean_returns: Series,
    pub covariance: LabeledMatrix,
    pub warnings: Vec<String>,
    pub periods_per_year: f64,
    pub risk_free_rate: f64,
}

/// Check that prices are a nonempty, positive frame with unique labels and increasing dates.
pub fn validate_price_frame(prices: &Frame) -> Result<()> {
    let unique: HashSet<&String> = prices.columns.iter().collect();
    if prices.is_empty() || unique.len() != prices.ncols() {
        return Err(value_error!("prices must be a nonempty DataFrame with unique asset names."));
    }
    if prices.index.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(value_error!("Prices must have unique, increasing, nonmissing dates."));
    }
    if prices.values.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(value_error!("Prices must be finite and strictly positive; resolve missing data first."));
    }
    Ok(())
}

/// CVXPY's checked solve: anything but an optimal status is an error.
pub(crate) fn solve_checked(program: &Program) -> Result<Solution> {
    let solution =
        program.solve().map_err(|error| runtime_error!("Portfolio optimization failed: {}", error.message()))?;
    if !solution.status.is_optimal() {
        return Err(runtime_error!("Portfolio optimization failed: {}", solution.status.name()));
    }
    Ok(solution)
}

/// `numpy.linspace(start, stop, count)`.
pub(crate) fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count).map(|i| i as f64 * step + start).collect();
    values[count - 1] = stop;
    values
}

/// Dot product of two slices.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `w' S w`.
pub(crate) fn quadratic_form(weights: &[f64], matrix: &DMatrix<f64>) -> f64 {
    let w = DVector::from_column_slice(weights);
    (w.transpose() * matrix * &w)[(0, 0)]
}

/// The full-investment row `sum(w) == 1` over the first `n` variables.
pub(crate) fn ones_row(n: usize) -> Row {
    (0..n).map(|j| (j, 1.0)).collect()
}

/// `(variable, coefficient)` pairs for a dense coefficient vector.
pub(crate) fn dense_row(values: &[f64]) -> Row {
    values.iter().copied().enumerate().collect()
}

/// A factor `F` with `F'F = matrix` for a positive semidefinite matrix.
///
/// Cholesky gives the factor for a positive definite matrix. Otherwise, as in
/// CVXPY's `decomp_quad`, the factor keeps eigenvalues above `1e6 * eps` of the
/// largest absolute eigenvalue.
pub(crate) fn psd_factor(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    if let Some(cholesky) = Cholesky::new(matrix.clone()) {
        return cholesky.l().transpose();
    }
    let eigen = SymmetricEigen::new(matrix.clone());
    let scale = eigen.eigenvalues.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let cutoff = 1e6 * f64::EPSILON;
    let kept: Vec<usize> =
        (0..eigen.eigenvalues.len()).filter(|i| scale > 0.0 && eigen.eigenvalues[*i] / scale > cutoff).collect();
    let mut factor = DMatrix::zeros(kept.len().max(1), matrix.ncols());
    for (row, i) in kept.iter().enumerate() {
        let root = eigen.eigenvalues[*i].sqrt();
        for j in 0..matrix.ncols() {
            factor[(row, j)] = root * eigen.eigenvectors[(j, *i)];
        }
    }
    factor
}

/// Optimize annual arithmetic returns and annual covariance, in decimal units.
///
/// Sharpe uses the supplied annual risk-free rate. Its convex formulation requires
/// a feasible positive excess return; otherwise that portfolio is omitted.
/// Minimum-risk ties prefer higher expected return.
/// A single frontier point retains only minimum volatility, without extra solves.
pub fn optimize(
    mean_returns: &Series,
    covariance: &LabeledMatrix,
    risk_free_rate: f64,
    max_weight: f64,
    frontier_points: usize,
) -> Result<Optimization> {
    if mean_returns.is_empty() {
        return Err(value_error!("mean_returns must be a nonempty pandas Series."));
    }
    let labels = mean_returns.labels.clone();
    let label_set: HashSet<&String> = labels.iter().collect();
    if label_set.len() != labels.len() {
        return Err(value_error!("Asset names must be unique."));
    }
    let rows: HashSet<&String> = covariance.index.iter().collect();
    let columns: HashSet<&String> = covariance.columns.iter().collect();
    if rows.len() != covariance.index.len()
        || columns.len() != covariance.columns.len()
        || rows != label_set
        || columns != label_set
    {
        return Err(value_error!("Covariance rows and columns must match the asset names."));
    }
    let n = labels.len();
    let mu = mean_returns.values.clone();
    let row_positions: Vec<usize> =
        labels.iter().map(|label| covariance.index.iter().position(|name| name == label).unwrap()).collect();
    let column_positions: Vec<usize> =
        labels.iter().map(|label| covariance.columns.iter().position(|name| name == label).unwrap()).collect();
    let mut sigma = DMatrix::from_fn(n, n, |i, j| covariance.values[(row_positions[i], column_positions[j])]);
    if mu.iter().any(|value| !value.is_finite()) || sigma.iter().any(|value| !value.is_finite()) {
        return Err(value_error!("Expected returns and covariance must be finite."));
    }
    if !risk_free_rate.is_finite() {
        return Err(value_error!("risk_free_rate must be finite."));
    }
    if !max_weight.is_finite() || !(max_weight > 0.0 && max_weight <= 1.0) {
        return Err(value_error!("max_weight must be greater than zero and at most one."));
    }
    if n as f64 * max_weight < 1.0 - 1e-12 {
        return Err(value_error!("The position cap is infeasible: asset count * max_weight < 1."));
    }
    if frontier_points < 1 {
        return Err(value_error!("frontier_points must be a positive integer."));
    }
    let symmetric =
        (0..n).all(|i| (0..n).all(|j| (sigma[(i, j)] - sigma[(j, i)]).abs() <= 1e-12 + 1e-10 * sigma[(j, i)].abs()));
    if !symmetric {
        return Err(value_error!("Covariance must be symmetric."));
    }
    sigma = (&sigma + sigma.transpose()) / 2.0;
    let covariance_scale = sigma.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    // Positive rescaling preserves the optimum and meaningful solver tolerances.
    let solver_covariance = if covariance_scale != 0.0 { &sigma / covariance_scale } else { sigma.clone() };
    let singular_covariance = match Cholesky::new(solver_covariance.clone()) {
        // Roundoff can let a singular matrix pass Cholesky.
        Some(factor) => factor.l().diagonal().iter().fold(f64::INFINITY, |low, value| low.min(*value)) <= 1e-7,
        None => {
            // Cholesky verifies the usual positive-definite case quickly. Singular
            // covariance is also valid, allowing tiny negative roundoff.
            let lowest =
                SymmetricEigen::new(sigma.clone()).eigenvalues.iter().fold(f64::INFINITY, |low, value| low.min(*value));
            if lowest < -1e-10 * covariance_scale {
                return Err(value_error!("Covariance must be positive semidefinite."));
            }
            true
        }
    };
    // A common return offset and positive rescaling do not change the frontier.
    let highest_mu = mu.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lowest_mu = mu.iter().copied().fold(f64::INFINITY, f64::min);
    let return_range = highest_mu - lowest_mu;
    let average_mu = mu.iter().sum::<f64>() / n as f64;
    let solver_returns: Vec<f64> = if return_range != 0.0 {
        mu.iter().map(|value| (value - average_mu) / return_range).collect()
    } else {
        vec![0.0; n]
    };

    let mut warnings: Vec<String> = Vec::new();
    let solve = |program: &Program, warnings: &mut Vec<String>| -> Result<Vec<f64>> {
        let solution = solve_checked(program)?;
        if solution.status == Status::OptimalInaccurate && !warnings.iter().any(|w| w == INACCURATE_WARNING) {
            warnings.push(INACCURATE_WARNING.to_string());
        }
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
        let weights: Vec<f64> = clipped.iter().map(|value| value / total).collect();
        let expected = dot(&mu, &weights);
        let volatility = quadratic_form(&weights, &sigma).max(0.0).sqrt();
        let sharpe = if volatility > 1e-12 { (expected - risk_free_rate) / volatility } else { f64::NAN };
        Ok(Portfolio {
            name: name.to_string(),
            weights: Series::new(labels.clone(), weights),
            expected_return: expected,
            volatility,
            sharpe,
        })
    };

    // Half-variance has the same minimizer and improves solver scaling near
    // frontier corners where most nonnegativity constraints become active.
    let base = || {
        let mut program = Program::new(n).quadratic(solver_covariance.clone()).equal(ones_row(n), 1.0);
        for j in 0..n {
            program = program.at_least(vec![(j, 1.0)], 0.0);
        }
        if max_weight < 1.0 {
            for j in 0..n {
                program = program.at_most(vec![(j, 1.0)], max_weight);
            }
        }
        program
    };
    let mut minimum = portfolio("Minimum volatility", &solve(&base(), &mut warnings)?)?;
    if singular_covariance && return_range != 0.0 {
        // Equal-variance minimizers differ only in the covariance null space.
        // Maximize return there so the low-risk endpoint is not dominated.
        let mut program = base().quadratic(DMatrix::zeros(n, n)).maximize(&solver_returns);
        let target = &solver_covariance * DVector::from_column_slice(&minimum.weights.values);
        for i in 0..n {
            program = program.equal(dense_row(solver_covariance.row(i).transpose().as_slice()), target[i]);
        }
        minimum = portfolio("Minimum volatility", &solve(&program, &mut warnings)?)?;
    }
    let mut portfolios = IndexMap::new();
    portfolios.insert(minimum.name.clone(), minimum.clone());

    // The maximum feasible return is a linear program with a greedy exact solution.
    let mut highest_weights = vec![0.0; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| (-mu[*a]).partial_cmp(&(-mu[*b])).unwrap());
    let mut remaining = 1.0;
    for index in order {
        highest_weights[index] = max_weight.min(remaining);
        remaining -= highest_weights[index];
        if remaining <= 1e-12 {
            break;
        }
    }
    let excess: Vec<f64> = mu.iter().map(|value| value - risk_free_rate).collect();
    let excess_scale = excess.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if dot(&excess, &highest_weights) > 1e-10 * excess_scale {
        // Substitute y = w / ((mu-rf)'w); minimizing variance gives maximum Sharpe.
        let scaled_excess: Vec<f64> = excess.iter().map(|value| value / excess_scale).collect();
        let variables = if max_weight < 1.0 { n + 1 } else { n };
        let mut p = DMatrix::zeros(variables, variables);
        p.view_mut((0, 0), (n, n)).copy_from(&(&solver_covariance * 2.0));
        let mut program = Program::new(variables).quadratic(p);
        for j in 0..n {
            program = program.at_least(vec![(j, 1.0)], 0.0);
        }
        program = program.equal(dense_row(&scaled_excess), 1.0);
        if max_weight < 1.0 {
            // A shared scalar keeps the cap constraints sparse as assets grow.
            let mut normalizer: Row = vec![(n, 1.0)];
            normalizer.extend((0..n).map(|j| (j, -1.0)));
            program = program.equal(normalizer, 0.0);
            for j in 0..n {
                program = program.at_most(vec![(j, 1.0), (n, -max_weight)], 0.0);
            }
        }
        let raw = solve(&program, &mut warnings)?;
        let total: f64 = raw[..n].iter().sum();
        let normalized: Vec<f64> = raw[..n].iter().map(|value| value / total).collect();
        let maximum = portfolio("Maximum Sharpe", &normalized)?;
        portfolios.insert(maximum.name.clone(), maximum);
    } else {
        warnings.push(
            "Maximum Sharpe omitted: no feasible portfolio has positive expected excess return \
             above the risk-free rate."
                .to_string(),
        );
    }
    let equal = portfolio("Equal weight", &vec![1.0 / n as f64; n])?;
    portfolios.insert(equal.name.clone(), equal);
    if portfolios.values().any(|p| !p.sharpe.is_finite()) {
        warnings.push("Sharpe is undefined for portfolios with zero estimated volatility.".to_string());
    }

    let mut frontier_portfolios = vec![minimum.clone()];
    let low_score = dot(&solver_returns, &minimum.weights.values);
    let high_score = dot(&solver_returns, &highest_weights);
    if frontier_points > 1 && high_score > low_score + 1e-10 {
        for required in linspace(low_score, high_score, frontier_points).into_iter().skip(1) {
            let program = base().at_least(dense_row(&solver_returns), required);
            let point = portfolio("Frontier", &solve(&program, &mut warnings)?)?;
            if dot(&solver_returns, &point.weights.values) < required - 1e-6 {
                return Err(runtime_error!("Frontier solution missed its required return."));
            }
            frontier_portfolios.push(point);
        }
    }
    let mut frontier = Frontier {
        labels: labels.clone(),
        expected_return: frontier_portfolios.iter().map(|p| p.expected_return).collect(),
        volatility: frontier_portfolios.iter().map(|p| p.volatility).collect(),
        sharpe: frontier_portfolios.iter().map(|p| p.sharpe).collect(),
        weights: frontier_portfolios.iter().map(|p| p.weights.values.clone()).collect(),
    };
    if frontier.len() > 1 {
        let spread = frontier.volatility.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - frontier.volatility.iter().copied().fold(f64::INFINITY, f64::min);
        if spread <= 1e-8 * covariance_scale.sqrt() {
            // Equal-risk points with a lower return are dominated by the endpoint.
            frontier.keep_last();
        }
    }
    Ok(Optimization { portfolios, frontier, warnings })
}

/// Options for [`analyze`], with the Python defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyzeOptions {
    pub train_fraction: f64,
    pub risk_free_rate: f64,
    pub max_weight: f64,
    pub shrinkage: f64,
    pub frontier_points: usize,
    pub periods_per_year: f64,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            train_fraction: 0.7,
            risk_free_rate: 0.02,
            max_weight: 1.0,
            shrinkage: 0.1,
            frontier_points: 40,
            periods_per_year: TRADING_DAYS,
        }
    }
}

/// Annual mean returns and shrunk annual covariance of a return sample.
pub(crate) fn estimates(
    returns: &Frame,
    periods_per_year: f64,
    shrinkage: f64,
) -> (Series, LabeledMatrix, DMatrix<f64>) {
    let mean = returns.means() * periods_per_year;
    let sample = returns.covariance() * periods_per_year;
    let covariance = shrink(&sample, shrinkage);
    (
        Series::new(returns.columns.clone(), mean.iter().copied().collect()),
        LabeledMatrix::square(returns.columns.clone(), covariance),
        sample,
    )
}

/// `(1 - shrinkage) * sample + shrinkage * diag(diag(sample))`.
pub(crate) fn shrink(sample: &DMatrix<f64>, shrinkage: f64) -> DMatrix<f64> {
    let diagonal = DMatrix::from_diagonal(&sample.diagonal());
    sample * (1.0 - shrinkage) + diagonal * shrinkage
}

/// Fit earlier observations, then hold fixed adjusted-price exposures.
///
/// Means and sample covariance use `periods_per_year` for annualization.
/// Covariance shrinks toward its diagonal. Holdout Sharpe and Sortino use annual
/// arithmetic excess returns. Calmar uses CAGR. No fees, tax or FX.
/// A multipoint frontier sample also supplies Low, Medium, and High risk levels.
pub fn analyze(prices: &Frame, options: &AnalyzeOptions) -> Result<Analysis> {
    validate_price_frame(prices)?;
    let periods_per_year = validate_periods_per_year(options.periods_per_year)?;
    let AnalyzeOptions { train_fraction, risk_free_rate, max_weight, shrinkage, frontier_points, .. } = *options;
    if !train_fraction.is_finite() || !(train_fraction > 0.0 && train_fraction < 1.0) {
        return Err(value_error!("train_fraction must be between zero and one."));
    }
    if !shrinkage.is_finite() || !(0.0..=1.0).contains(&shrinkage) {
        return Err(value_error!("shrinkage must be between zero and one."));
    }
    let returns = prices.pct_change();
    let split = (returns.nrows() as f64 * train_fraction).floor() as usize;
    if split < 2 || returns.nrows() - split < 2 {
        return Err(value_error!("At least two training returns and two holdout returns are required."));
    }
    let train_returns = returns.rows(0, split);
    let test_returns = returns.rows(split, returns.nrows());
    let (mean_returns, covariance, _) = estimates(&train_returns, periods_per_year, shrinkage);
    let Optimization { mut portfolios, frontier, mut warnings } =
        optimize(&mean_returns, &covariance, risk_free_rate, max_weight, frontier_points)?;
    if frontier_points > 1 {
        let last = frontier.len() - 1;
        let midpoint_volatility = (frontier.volatility[0] + frontier.volatility[last]) / 2.0;
        let mut middle = 0;
        for (i, volatility) in frontier.volatility.iter().enumerate() {
            if (volatility - midpoint_volatility).abs() < (frontier.volatility[middle] - midpoint_volatility).abs() {
                middle = i;
            }
        }
        for (name, position) in CLASSIC_PROFILE_NAMES.iter().zip([0, middle, last]) {
            portfolios.insert(
                name.to_string(),
                Portfolio {
                    name: name.to_string(),
                    weights: Series::new(frontier.labels.clone(), frontier.weights[position].clone()),
                    expected_return: frontier.expected_return[position],
                    volatility: frontier.volatility[position],
                    sharpe: frontier.sharpe[position],
                },
            );
        }
        if frontier.len() == 1 {
            warnings.push(
                "The classic frontier provides no distinct risk levels. Low, Medium, and High use the same allocation."
                    .to_string(),
            );
        }
    }
    if split < 30 {
        warnings.push(format!(
            "Only {split} training returns are available (fewer than 30). Return and risk estimates can be unstable."
        ));
    }
    if prices.ncols() >= split {
        warnings.push(format!(
            "The training sample has {split} returns for {} assets. \
             Sample covariance is singular before shrinkage. Use more history or fewer assets.",
            prices.ncols()
        ));
    }

    // Hold each asset's adjusted-price exposure from the final training close.
    let held = prices.rows(split, prices.nrows());
    let base: Vec<f64> = prices.row(split);
    let mut columns = Vec::new();
    let mut holdout_metrics = IndexMap::new();
    for (name, portfolio) in &portfolios {
        let path: Vec<f64> = (0..held.nrows())
            .map(|i| (0..held.ncols()).map(|j| held.get(i, j) / base[j] * portfolio.weights.values[j]).sum())
            .collect();
        holdout_metrics.insert(name.clone(), performance_metrics(&path, risk_free_rate, periods_per_year));
        columns.push((name.clone(), path));
    }
    let equity = Frame::from_columns(held.index.clone(), columns);
    Ok(Analysis {
        portfolios,
        frontier,
        train_returns,
        test_returns,
        equity,
        holdout_metrics,
        mean_returns,
        covariance,
        warnings,
        periods_per_year,
        risk_free_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn analytic_minimum_variance_and_maximum_sharpe() {
        let names = labels(&["A", "B"]);
        let mean = Series::new(names.clone(), vec![0.10, 0.20]);
        let covariance = LabeledMatrix::square(names, DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.16]));
        let result = optimize(&mean, &covariance, 0.0, 1.0, 5).unwrap();
        let minimum = &result.portfolios["Minimum volatility"];
        assert!((minimum.weights.values[0] - 0.8).abs() < 1e-6);
        // Tangency with zero correlation: w ∝ Σ⁻¹μ = (2.5, 1.25).
        let sharpe = &result.portfolios["Maximum Sharpe"];
        assert!((sharpe.weights.values[0] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(result.frontier.len(), 5);
        assert!((result.frontier.expected_return[4] - 0.20).abs() < 1e-6);
    }

    #[test]
    fn covariance_alignment_uses_labels() {
        let mean = Series::new(labels(&["A", "B"]), vec![0.10, 0.20]);
        let covariance =
            LabeledMatrix::square(labels(&["B", "A"]), DMatrix::from_row_slice(2, 2, &[0.16, 0.0, 0.0, 0.04]));
        let result = optimize(&mean, &covariance, 0.0, 1.0, 1).unwrap();
        assert!((result.portfolios["Minimum volatility"].weights.values[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn invalid_inputs_are_rejected_with_python_messages() {
        let mean = Series::new(labels(&["A", "B"]), vec![0.1, 0.2]);
        let indefinite =
            LabeledMatrix::square(labels(&["A", "B"]), DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]));
        assert_eq!(
            optimize(&mean, &indefinite, 0.0, 1.0, 1).unwrap_err().message(),
            "Covariance must be positive semidefinite."
        );
        let identity = LabeledMatrix::square(labels(&["A", "B"]), DMatrix::identity(2, 2));
        assert_eq!(
            optimize(&mean, &identity, 0.0, 0.4, 1).unwrap_err().message(),
            "The position cap is infeasible: asset count * max_weight < 1."
        );
        let asymmetric =
            LabeledMatrix::square(labels(&["A", "B"]), DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]));
        assert_eq!(optimize(&mean, &asymmetric, 0.0, 1.0, 1).unwrap_err().message(), "Covariance must be symmetric.");
    }

    #[test]
    fn no_positive_excess_return_omits_maximum_sharpe() {
        let mean = Series::new(labels(&["A", "B"]), vec![0.01, 0.015]);
        let covariance = LabeledMatrix::square(labels(&["A", "B"]), DMatrix::identity(2, 2) * 0.04);
        let result = optimize(&mean, &covariance, 0.05, 1.0, 1).unwrap();
        assert!(!result.portfolios.contains_key("Maximum Sharpe"));
        assert!(result.warnings[0].starts_with("Maximum Sharpe omitted"));
    }

    #[test]
    fn analysis_holds_training_weights_through_the_holdout() {
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let index: Vec<_> = (0..12).map(|i| start + chrono::Days::new(i)).collect();
        let a: Vec<f64> = (0..12).map(|i| 100.0 * (1.0 + 0.01 * (i % 3) as f64 + 0.002 * i as f64)).collect();
        let b: Vec<f64> = (0..12).map(|i| 50.0 * (1.0 + 0.02 * ((i + 1) % 4) as f64)).collect();
        let prices = Frame::from_columns(index, vec![("A".into(), a), ("B".into(), b)]);
        let analysis = analyze(&prices, &AnalyzeOptions { frontier_points: 5, ..Default::default() }).unwrap();
        assert_eq!(analysis.train_returns.nrows(), 7);
        assert_eq!(analysis.equity.nrows(), 12 - 7);
        for (j, name) in analysis.equity.columns.iter().enumerate() {
            assert_eq!(analysis.equity.get(0, j), 1.0, "{name}");
        }
        assert!(analysis.warnings.iter().any(|w| w.starts_with("Only 7 training returns")));
        assert_eq!(analysis.portfolios.keys().last().unwrap(), "High");
    }
}
