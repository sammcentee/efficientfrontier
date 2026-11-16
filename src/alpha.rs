//! Out-of-sample alpha of the stock portfolio, and the noise in fitted mean returns.
//!
//! Alpha is the intercept of an ordinary least squares regression of the
//! portfolio's daily excess returns on a reference's daily excess returns:
//! `r_P - r_f = alpha + beta (r_B - r_f) + e`. The walk-forward windows supply the
//! returns, so no stock return in the regression was known to the fit that chose
//! the weights. Fixed inputs, such as a volatility budget from a later year, are
//! outside this guarantee.

use crate::frame::{Cell, Frame};
use crate::out_of_sample::OutOfSampleStudy;
use crate::selection::SELECTED;

/// The regression of the stock portfolio's excess returns on one reference's.
#[derive(Clone, Debug, PartialEq)]
pub struct Alpha {
    pub reference: String,
    /// The intercept per year: periods per year times the daily intercept.
    pub annual: f64,
    /// The ordinary least squares standard error of `annual`.
    pub standard_error: f64,
    pub beta: f64,
    pub r_squared: f64,
    pub observations: usize,
}

impl Alpha {
    pub fn t_statistic(&self) -> f64 {
        self.annual / self.standard_error
    }
}

/// Regress `portfolio` on `reference`, both as simple returns per period.
///
/// `None` with fewer than three returns, or when the reference does not vary.
pub fn alpha(
    reference: &str,
    portfolio: &[f64],
    benchmark: &[f64],
    risk_free_rate: f64,
    periods: f64,
) -> Option<Alpha> {
    let n = portfolio.len();
    if n < 3 || benchmark.len() != n {
        return None;
    }
    let rate = risk_free_rate / periods;
    let y: Vec<f64> = portfolio.iter().map(|r| r - rate).collect();
    let x: Vec<f64> = benchmark.iter().map(|r| r - rate).collect();
    let count = n as f64;
    let (x_mean, y_mean) = (x.iter().sum::<f64>() / count, y.iter().sum::<f64>() / count);
    let sxx: f64 = x.iter().map(|v| (v - x_mean).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - x_mean) * (b - y_mean)).sum();
    let syy: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let beta = sxy / sxx;
    let intercept = y_mean - beta * x_mean;
    let residual: f64 = x.iter().zip(&y).map(|(a, b)| (b - intercept - beta * a).powi(2)).sum();
    let variance = residual / (count - 2.0);
    let standard_error = (variance * (1.0 / count + x_mean * x_mean / sxx)).sqrt();
    Some(Alpha {
        reference: reference.to_string(),
        annual: intercept * periods,
        standard_error: standard_error * periods,
        beta,
        r_squared: if syy > 0.0 { 1.0 - residual / syy } else { f64::NAN },
        observations: n,
    })
}

/// The alpha of the stock portfolio against each other path of the walk-forward tests.
pub fn out_of_sample_alpha(result: &OutOfSampleStudy) -> Vec<Alpha> {
    let returns = &result.held_returns;
    let Some(portfolio) = returns.column(SELECTED) else { return Vec::new() };
    let rate = result.settings.f64("risk_free_rate");
    let periods = result.settings.f64("periods_per_year");
    returns
        .columns
        .iter()
        .enumerate()
        .filter(|(_, name)| *name != SELECTED)
        .filter_map(|(j, name)| alpha(name, &portfolio, &returns.col(j), rate, periods))
        .collect()
}

/// How much of the spread in fitted annual means across stocks exceeds sampling noise.
#[derive(Clone, Debug, PartialEq)]
pub struct MeanNoise {
    pub stocks: usize,
    pub years: f64,
    /// The sample standard deviation of the fitted annual means across stocks.
    pub spread: f64,
    /// The root mean square standard error of each fitted mean, relative to the
    /// average stock on the same dates.
    pub noise: f64,
}

impl MeanNoise {
    /// The share of the variance of fitted means that exceeds sampling noise, at least zero.
    ///
    /// The estimate treats each mean as constant over the whole history, so a mean
    /// that changes over time also counts as signal. It is an upper limit on
    /// lasting differences between stocks.
    pub fn signal_share(&self) -> f64 {
        if self.spread > 0.0 { (1.0 - (self.noise / self.spread).powi(2)).max(0.0) } else { 0.0 }
    }
}

/// Measure [`MeanNoise`] for the stocks in `prices`.
///
/// Each stock's return is compared with the average stock return on the same
/// date. That removes the market move, which every fitted mean shares, so the
/// standard errors describe the differences between stocks that the optimizer
/// ranks. The comparison also removes 1/n of each stock's own error, so the
/// noise variance is scaled back by n/(n - 1). `None` with fewer than two
/// stocks or three returns.
pub fn mean_noise(prices: &Frame, periods: f64) -> Option<MeanNoise> {
    let returns = prices.pct_change();
    let (m, n) = (returns.nrows(), returns.ncols());
    if n < 2 || m < 3 {
        return None;
    }
    let average: Vec<f64> = (0..m).map(|i| returns.row(i).iter().sum::<f64>() / n as f64).collect();
    let sample_deviation = |values: &[f64]| {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0)).sqrt()
    };
    let mut means = Vec::with_capacity(n);
    let mut errors = 0.0;
    for j in 0..n {
        let column = returns.col(j);
        means.push(periods * column.iter().sum::<f64>() / m as f64);
        let relative: Vec<f64> = column.iter().zip(&average).map(|(r, a)| r - a).collect();
        errors += (periods * sample_deviation(&relative) / (m as f64).sqrt()).powi(2);
    }
    Some(MeanNoise {
        stocks: n,
        years: m as f64 / periods,
        spread: sample_deviation(&means),
        noise: (errors / (n as f64 - 1.0)).sqrt(),
    })
}

/// Ranks from 1, with the average rank for ties.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        for position in &order[start..end] {
            ranks[*position] = (start + end + 1) as f64 / 2.0;
        }
        start = end;
    }
    ranks
}

/// The Spearman rank correlation of two equal-length samples. `None` without variation.
pub fn rank_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    let (x, y) = (ranks(a), ranks(b));
    let n = x.len() as f64;
    let (x_mean, y_mean) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(&y).map(|(p, q)| (p - x_mean) * (q - y_mean)).sum();
    let sxx: f64 = x.iter().map(|p| (p - x_mean).powi(2)).sum();
    let syy: f64 = y.iter().map(|q| (q - y_mean).powi(2)).sum();
    (sxx > 0.0 && syy > 0.0).then(|| sxy / (sxx * syy).sqrt())
}

/// For each completed walk-forward window, the rank correlation across the
/// window's eligible stocks between their mean return on the fit dates and
/// their mean return on the test dates after the purchase close.
///
/// A value near zero means that the ranking that the optimizer used told
/// little about the later ranking.
pub fn rank_persistence(prices: &Frame, result: &OutOfSampleStudy) -> Vec<f64> {
    let returns = prices.pct_change();
    let windows = &result.windows;
    let eligibility = &result.eligibility;
    if eligibility.position("window").is_none() {
        return Vec::new();
    }
    let mut values = Vec::new();
    for row in 0..windows.nrows() {
        if !windows.column("status")[row].py_str().is_empty() {
            continue;
        }
        let number = &windows.index[0][row];
        let split = windows.column("fit_returns")[row].as_int().unwrap_or(0) as usize;
        let end = split + windows.column("test_returns")[row].as_int().unwrap_or(0) as usize;
        let symbols: Vec<String> = (0..eligibility.nrows())
            .filter(|i| eligibility.column("window")[*i] == *number)
            .filter(|i| eligibility.column("eligible")[*i] == Cell::Bool(true))
            .map(|i| eligibility.column("symbol")[i].py_str())
            .collect();
        let Some(chosen) = returns.select(&symbols) else { continue };
        if split == 0 || end <= split + 1 || end > chosen.nrows() {
            continue;
        }
        let mean = |j: usize, start: usize, stop: usize| {
            chosen.col(j)[start..stop].iter().sum::<f64>() / (stop - start) as f64
        };
        // Row s holds return s + 1: the fit uses returns 1..s_k, the test s_k + 2..e_k.
        let fitted: Vec<f64> = (0..chosen.ncols()).map(|j| mean(j, 0, split)).collect();
        let later: Vec<f64> = (0..chosen.ncols()).map(|j| mean(j, split + 1, end)).collect();
        if let Some(value) = rank_correlation(&fitted, &later) {
            values.push(value);
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Date;

    // numpy and scipy values for the small examples below.
    const SPREAD: f64 = 1.0710552910457956;
    const NOISE: f64 = 2.2422625429743914;
    const SPEARMAN: f64 = 0.36842105263157904;

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9 * expected.abs().max(1.0), "{actual} != {expected}");
    }

    #[test]
    fn alpha_matches_ordinary_least_squares() {
        let benchmark = [0.010, -0.020, 0.005, 0.015, -0.010, 0.002];
        let portfolio = [0.012, -0.018, 0.009, 0.020, -0.011, 0.001];
        let result = alpha("SPY", &portfolio, &benchmark, 0.02, 252.0).unwrap();
        // numpy.linalg.lstsq on the excess returns, with the usual OLS standard error.
        close(result.beta, 1.0929687499999996);
        close(result.annual, 0.45604999999999996);
        close(result.standard_error, 0.24919374239735634);
        close(result.r_squared, 0.9775056936231421);
        assert_eq!(result.observations, 6);
        close(result.t_statistic(), 0.45604999999999996 / 0.24919374239735634);
    }

    #[test]
    fn alpha_needs_three_returns_and_a_varying_reference() {
        assert!(alpha("SPY", &[0.01, 0.02], &[0.0, 0.01], 0.0, 252.0).is_none());
        assert!(alpha("SPY", &[0.01, 0.02, 0.03], &[0.01, 0.01, 0.01], 0.0, 252.0).is_none());
    }

    #[test]
    fn a_portfolio_that_copies_its_reference_has_no_alpha() {
        let benchmark = [0.010, -0.020, 0.005, 0.015, -0.010];
        let result = alpha("SPY", &benchmark, &benchmark, 0.02, 252.0).unwrap();
        close(result.beta, 1.0);
        assert!(result.annual.abs() < 1e-12 && result.standard_error.abs() < 1e-12);
        close(result.r_squared, 1.0);
    }

    fn frame(columns: Vec<(&str, Vec<f64>)>) -> Frame {
        let rows = columns[0].1.len();
        let start = Date::from_ymd_opt(2020, 1, 1).unwrap();
        let index = (0..rows).map(|i| start + chrono::Days::new(i as u64)).collect();
        Frame::from_columns(index, columns.into_iter().map(|(name, values)| (name.to_string(), values)).collect())
    }

    #[test]
    fn mean_noise_compares_the_spread_of_means_with_their_standard_errors() {
        let prices = frame(vec![
            ("A", vec![100.0, 101.0, 100.5, 102.0, 103.0]),
            ("B", vec![50.0, 49.0, 49.5, 50.5, 50.0]),
            ("C", vec![20.0, 20.4, 20.2, 20.1, 20.6]),
        ]);
        let noise = mean_noise(&prices, 252.0).unwrap();
        assert_eq!(noise.stocks, 3);
        close(noise.years, 4.0 / 252.0);
        close(noise.spread, SPREAD);
        close(noise.noise, NOISE);
        assert_eq!(noise.signal_share(), 0.0);
    }

    #[test]
    fn rank_correlation_matches_spearman_with_ties() {
        close(rank_correlation(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]).unwrap(), 1.0);
        close(rank_correlation(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]).unwrap(), -1.0);
        // scipy.stats.spearmanr([1, 2, 2, 5, 3], [2, 1, 4, 3, 3]).
        close(rank_correlation(&[1.0, 2.0, 2.0, 5.0, 3.0], &[2.0, 1.0, 4.0, 3.0, 3.0]).unwrap(), SPEARMAN);
        assert!(rank_correlation(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn a_large_true_difference_dominates_the_noise() {
        let steady =
            |rate: f64| (0..60).map(|i| 100.0 * (1.0 + rate).powi(i) * (1.0 + 0.001 * (i % 2) as f64)).collect();
        let prices = frame(vec![("A", steady(0.004)), ("B", steady(0.0)), ("C", steady(-0.004))]);
        let noise = mean_noise(&prices, 252.0).unwrap();
        assert!(noise.signal_share() > 0.99, "{noise:?}");
    }
}
