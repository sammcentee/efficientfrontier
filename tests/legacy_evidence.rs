//! Port of `tests/test_evidence.py`. The Holm adjustment test is a unit test in `src/evidence.rs`.

mod legacy_common;

use efficient_frontier::evidence::{EvidenceAnalysis, EvidenceOptions, analyze_evidence};
use efficient_frontier::frame::{Cell, Frame, Table, iso, mean, std};
use legacy_common::{approx, assert_allclose, business_days, date, numbers, test_data};
use nalgebra::{DMatrix, DVector};

/// `equity(returns, name)`: one path from 1.0 on business days from 2020-01-01.
fn equity(returns: &[f64], name: &str) -> Frame {
    let mut growth = 1.0;
    let mut path = vec![1.0];
    for value in returns {
        growth *= 1.0 + value;
        path.push(growth);
    }
    Frame::from_columns(business_days(date(2020, 1, 1), returns.len() + 1), vec![(name.to_string(), path)])
}

/// `sample(count)`: NumPy draws of `default_rng(100)` with serially correlated residuals.
fn sample(count: usize) -> (Frame, Frame) {
    let data = test_data(&format!("evidence_sample_{count}"));
    (equity(&numbers(&data["strategy"]), "Strategy"), equity(&numbers(&data["benchmark"]), "SPY"))
}

/// The columns of several frames with the same dates.
fn combine(frames: &[&Frame]) -> Frame {
    let columns = frames
        .iter()
        .flat_map(|frame| frame.columns.iter().enumerate().map(|(j, name)| (name.clone(), frame.col(j))))
        .collect();
    Frame::from_columns(frames[0].index.clone(), columns)
}

fn rename(frame: &Frame, names: &[&str]) -> Frame {
    let mut renamed = frame.clone();
    renamed.columns = names.iter().map(|name| name.to_string()).collect();
    renamed
}

fn reversed_columns(frame: &Frame) -> Frame {
    frame.select_positions(&(0..frame.ncols()).rev().collect::<Vec<_>>())
}

fn evidence(strategy: &Frame, benchmark: &Frame, options: EvidenceOptions) -> EvidenceAnalysis {
    analyze_evidence(strategy, benchmark, &options).unwrap()
}

fn immediate() -> EvidenceOptions {
    EvidenceOptions { delayed_entry: false, ..Default::default() }
}

fn value(table: &Table, column: &str, row: usize) -> f64 {
    table.column(column)[row].as_f64().unwrap_or(f64::NAN)
}

fn text(table: &Table, column: &str, row: usize) -> String {
    table.column(column)[row].py_str()
}

#[test]
fn hac_matches_independent_dense_kernel_reference() {
    let (strategy, benchmark) = sample(180);
    let result = evidence(&strategy, &benchmark, immediate());
    let y = strategy.pct_change().col(0);
    let x = benchmark.pct_change().col(0);
    let count = y.len();
    let n = count as f64;
    let lags = (4.0 * (n / 100.0).powf(2.0 / 9.0)).floor() as usize;
    let kernel =
        DMatrix::from_fn(count, count, |i, j| (1.0 - (i as f64 - j as f64).abs() / (lags + 1) as f64).max(0.0));
    let active: Vec<f64> = y.iter().zip(&x).map(|(a, b)| a - b).collect();
    let center = mean(&active);
    let demeaned = DVector::from_iterator(count, active.iter().map(|value| value - center));
    let advantage_error = ((demeaned.transpose() * &kernel * &demeaned)[(0, 0)] / (n * n) * n / (n - 1.0)).sqrt();
    let rate = 0.02 / 252.0;
    let design = DMatrix::from_fn(count, 2, |i, j| if j == 0 { 1.0 } else { x[i] - rate });
    let target = DVector::from_iterator(count, y.iter().map(|value| value - rate));
    let inverse = (design.transpose() * &design).try_inverse().unwrap();
    let coefficients = &inverse * design.transpose() * &target;
    let residual = &target - &design * &coefficients;
    let weighted = (&residual * residual.transpose()).component_mul(&kernel);
    let covariance = &inverse * design.transpose() * weighted * &design * &inverse * (n / (n - 2.0));
    let alpha_error = covariance[(0, 0)].sqrt();
    let summary = &result.summary;
    approx(value(summary, "annual_advantage", 0), center * 252.0);
    approx(value(summary, "beta", 0), coefficients[1]);
    approx(value(summary, "alpha", 0), coefficients[0] * 252.0);
    approx(value(summary, "advantage_ci_low", 0), (center - 1.959963984540054 * advantage_error) * 252.0);
    approx(value(summary, "alpha_ci_high", 0), (coefficients[0] + 1.959963984540054 * alpha_error) * 252.0);
    approx(value(summary, "advantage_pvalue", 0), libm::erfc((center / advantage_error).abs() / 2f64.sqrt()));
    approx(value(summary, "alpha_pvalue", 0), libm::erfc((coefficients[0] / alpha_error).abs() / 2f64.sqrt()));
    assert_eq!(result.settings.int("hac_lags"), lags as i64);
    let deviation = std(&active, 1);
    approx(value(summary, "tracking_error", 0), deviation * 252f64.sqrt());
    approx(value(summary, "information_ratio", 0), center / deviation * 252f64.sqrt());
    assert!(advantage_error > deviation / n.sqrt());
}

#[test]
fn all_strategies_and_benchmarks_share_one_fixed_test_family() {
    let (strategy, benchmark) = sample(180);
    let strategy = combine(&[&strategy, &rename(&benchmark, &["Clone"])]);
    let benchmark = combine(&[&benchmark, &rename(&benchmark, &["QQQ"])]);
    let result = evidence(&strategy, &benchmark, immediate());
    assert_eq!(result.settings.int("multiple_testing_tests"), 8);
    let rows = result.summary.nrows();
    let original: Vec<f64> = (0..rows)
        .flat_map(|i| [value(&result.summary, "advantage_pvalue", i), value(&result.summary, "alpha_pvalue", i)])
        .collect();
    let pvalues: Vec<f64> = original.iter().map(|p| if p.is_finite() { *p } else { 1.0 }).collect();
    let mut order: Vec<usize> = (0..pvalues.len()).collect();
    order.sort_by(|a, b| pvalues[*a].partial_cmp(&pvalues[*b]).unwrap());
    let mut expected = vec![f64::NAN; pvalues.len()];
    let mut maximum: f64 = 0.0;
    for (rank, position) in order.into_iter().enumerate() {
        maximum = maximum.max((pvalues[position] * (pvalues.len() - rank) as f64).min(1.0));
        if original[position].is_finite() {
            expected[position] = maximum;
        }
    }
    let actual: Vec<f64> = (0..rows)
        .flat_map(|i| {
            [value(&result.summary, "advantage_pvalue_adjusted", i), value(&result.summary, "alpha_pvalue_adjusted", i)]
        })
        .collect();
    assert_allclose(&actual, &expected, 0.0, 1e-7);
    let reordered = evidence(&reversed_columns(&strategy), &reversed_columns(&benchmark), immediate());
    let sorted = |table: &Table| -> Table {
        let mut positions: Vec<usize> = (0..table.nrows()).collect();
        positions.sort_by_key(|i| (table.index[0][*i].py_str(), table.index[1][*i].py_str()));
        table.take_rows(&positions)
    };
    let (a, b) = (sorted(&result.summary), sorted(&reordered.summary));
    assert_eq!(a.index, b.index);
    assert_eq!(a.columns, b.columns);
    for (column, (x, y)) in a.columns.iter().zip(a.data.iter().zip(&b.data)) {
        for (left, right) in x.iter().zip(y) {
            match (left, right) {
                (Cell::Float(l), Cell::Float(r)) => assert_allclose(&[*l], &[*r], 0.0, 1e-5),
                _ => assert_eq!(left, right, "{column}"),
            }
        }
    }
}

#[test]
fn entry_exclusion_changes_inference_but_keeps_costs_in_full_path_results() {
    let (strategy, benchmark) = sample(180);
    let mut strategy_returns = vec![-0.2];
    strategy_returns.extend(strategy.pct_change().col(0));
    let mut benchmark_returns = vec![-0.01];
    benchmark_returns.extend(benchmark.pct_change().col(0));
    let strategy = equity(&strategy_returns, "Strategy");
    let benchmark = equity(&benchmark_returns, "SPY");
    let delayed = evidence(&strategy, &benchmark, EvidenceOptions::default());
    let at_once = evidence(&strategy, &benchmark, immediate());
    assert_eq!(value(&delayed.summary, "observations", 0), 180.0);
    assert_eq!(value(&at_once.summary, "observations", 0), 181.0);
    let differences: Vec<f64> = strategy_returns[1..].iter().zip(&benchmark_returns[1..]).map(|(a, b)| a - b).collect();
    approx(value(&delayed.summary, "annual_advantage", 0), mean(&differences) * 252.0);
    assert!(value(&delayed.summary, "annual_advantage", 0) > value(&at_once.summary, "annual_advantage", 0));
    let growth = |returns: &[f64]| returns.iter().map(|value| 1.0 + value).product::<f64>();
    approx(
        value(&delayed.summary, "total_return_difference", 0),
        growth(&strategy_returns) - growth(&benchmark_returns),
    );
    assert_eq!(value(&delayed.summary, "cagr_difference", 0), value(&at_once.summary, "cagr_difference", 0));
    assert_eq!(delayed.windows, at_once.windows);
    assert_eq!(delayed.relative_equity, at_once.relative_equity);
    assert_eq!(delayed.settings.str("inference_start"), iso(strategy.index[2]));
}

#[test]
fn window_dates_cover_all_returns_once_and_relative_returns_compound() {
    let (strategy, benchmark) = sample(19);
    let result = evidence(&strategy, &benchmark, EvidenceOptions { periods_per_year: 12.0, ..immediate() });
    assert_eq!(result.windows.f64s("observations"), [7.0, 6.0, 6.0]);
    let (s, b) = (strategy.pct_change(), benchmark.pct_change());
    let (sr, br) = (s.col(0), b.col(0));
    for (row, (start, end)) in [(0, 7), (7, 13), (13, 19)].into_iter().enumerate() {
        let growth = |returns: &[f64]| returns[start..end].iter().map(|value| 1.0 + value).product::<f64>();
        approx(value(&result.windows, "relative_return", row), growth(&sr) / growth(&br) - 1.0);
        let differences: Vec<f64> = (start..end).map(|i| sr[i] - br[i]).collect();
        approx(value(&result.windows, "annual_advantage", row), mean(&differences) * 12.0);
        assert_eq!(text(&result.windows, "start", row), iso(s.index[start]));
        assert_eq!(text(&result.windows, "end", row), iso(s.index[end - 1]));
    }
    let ahead = result.windows.f64s("relative_return").iter().filter(|value| **value > 0.0).count();
    assert_eq!(value(&result.summary, "windows_ahead", 0), ahead as f64);
    let compounded: f64 = result.windows.f64s("relative_return").iter().map(|value| 1.0 + value).product();
    let relative = &result.relative_equity.values;
    approx(compounded, relative[(relative.nrows() - 1, 0)]);
}

#[test]
fn initial_capital_and_labels_are_preserved() {
    let (strategy, benchmark) = sample(180);
    let reference = evidence(&strategy, &benchmark, EvidenceOptions::default());
    let mut rich_strategy = strategy.clone();
    rich_strategy.values *= 100.0;
    let mut rich_benchmark = benchmark.clone();
    rich_benchmark.values *= 500.0;
    let scaled = evidence(&rich_strategy, &rich_benchmark, EvidenceOptions::default());
    assert_eq!(reference.summary.columns, scaled.summary.columns);
    for column in &reference.summary.columns {
        match &reference.summary.column(column)[0] {
            Cell::Float(_) => {
                assert_allclose(&scaled.summary.f64s(column), &reference.summary.f64s(column), 1e-10, 1e-10)
            }
            _ => assert_eq!(scaled.summary.column(column), reference.summary.column(column), "{column}"),
        }
    }
    assert_allclose(
        scaled.relative_equity.values.as_slice(),
        reference.relative_equity.values.as_slice(),
        1e-12,
        1e-12,
    );
    assert_eq!(scaled.relative_equity.values[(0, 0)], 1.0);
    assert_eq!(scaled.summary.index_names, [Some("strategy".to_string()), Some("benchmark".to_string())]);
    assert_eq!(scaled.relative_equity.columns, [("Strategy".to_string(), "SPY".to_string())]);
}

#[test]
fn inference_requires_sixty_paired_observations() {
    for count in [59, 60] {
        let (strategy, benchmark) = sample(count);
        let result = evidence(&strategy, &benchmark, immediate());
        let summary = &result.summary;
        assert_eq!(value(summary, "observations", 0), count as f64);
        assert!(value(summary, "beta", 0).is_finite());
        assert!(value(summary, "alpha", 0).is_finite());
        assert_eq!(value(summary, "advantage_pvalue", 0).is_finite(), count >= 60);
        assert_eq!(value(summary, "alpha_pvalue", 0).is_finite(), count >= 60);
        if count < 60 {
            assert_eq!(text(summary, "status", 0), "Insufficient history");
            assert!(text(summary, "advantage_note", 0).contains("60 paired observations"));
        }
    }
}

#[test]
fn one_return_keeps_descriptive_results_with_no_post_entry_inference() {
    let result = evidence(&equity(&[0.01], "Strategy"), &equity(&[0.02], "SPY"), EvidenceOptions::default());
    let summary = &result.summary;
    assert_eq!(value(summary, "observations", 0), 0.0);
    assert!((value(summary, "total_return_difference", 0) + 0.01).abs() <= 1e-12);
    assert!(value(summary, "annual_advantage", 0).is_nan());
    assert!(value(summary, "alpha_pvalue", 0).is_nan());
    assert_eq!(result.windows.nrows(), 1);
    assert!(result.settings.get("inference_start").unwrap().is_null());
}

#[test]
fn identical_paths_do_not_create_certainty_from_zero_residuals() {
    let (_, benchmark) = sample(180);
    let result = evidence(&rename(&benchmark, &["Strategy"]), &benchmark, EvidenceOptions::default());
    let summary = &result.summary;
    assert_eq!(value(summary, "total_return_difference", 0), 0.0);
    approx(value(summary, "beta", 0), 1.0);
    assert!(value(summary, "alpha", 0).abs() <= 1e-12);
    assert_eq!(value(summary, "tracking_error", 0), 0.0);
    assert!(value(summary, "information_ratio", 0).is_nan());
    assert!(value(summary, "advantage_pvalue", 0).is_nan());
    assert!(value(summary, "alpha_pvalue", 0).is_nan());
    assert_eq!(text(summary, "status", 0), "Uncertainty unavailable");
    assert_allclose(
        result.relative_equity.values.as_slice(),
        &vec![1.0; result.relative_equity.values.len()],
        0.0,
        1e-7,
    );
}

#[test]
fn constant_benchmark_does_not_create_identifiable_alpha() {
    let (strategy, _) = sample(180);
    let result = evidence(&strategy, &equity(&[0.0; 180], "Cash"), immediate());
    let summary = &result.summary;
    assert!(value(summary, "beta", 0).is_nan());
    assert!(value(summary, "alpha", 0).is_nan());
    assert!(value(summary, "alpha_pvalue", 0).is_nan());
    assert!(value(summary, "advantage_pvalue", 0).is_finite());
    assert!(text(summary, "alpha_note", 0).contains("Benchmark variation"));
    assert_eq!(result.settings.int("multiple_testing_tests"), 2);
}

#[test]
fn strong_historical_signal_gets_a_directional_status() {
    let data = test_data("evidence_signal");
    let (benchmark, noise) = (numbers(&data["benchmark"]), numbers(&data["noise"]));
    for (advantage, status) in
        [(0.002, "Historical mean advantage and alpha"), (-0.002, "Historical mean underperformance")]
    {
        let strategy: Vec<f64> = benchmark.iter().zip(&noise).map(|(b, e)| b + advantage + e).collect();
        let result = evidence(&equity(&strategy, "Strategy"), &equity(&benchmark, "SPY"), immediate());
        assert_eq!(text(&result.summary, "status", 0), status);
        assert!(value(&result.summary, "advantage_pvalue_adjusted", 0) < 0.05);
        assert!(result.warnings.iter().any(|warning| warning.contains("future outperformance")));
    }
}

#[test]
fn higher_market_exposure_is_not_reported_as_significant_alpha() {
    let benchmark = numbers(&test_data("evidence_exposure"));
    let strategy: Vec<f64> = benchmark.iter().map(|value| 2.0 * value).collect();
    let result = evidence(
        &equity(&strategy, "Strategy"),
        &equity(&benchmark, "SPY"),
        EvidenceOptions { risk_free_rate: 0.0, ..immediate() },
    );
    let summary = &result.summary;
    approx(value(summary, "beta", 0), 2.0);
    assert!(value(summary, "alpha", 0).abs() <= 1e-12);
    assert!(value(summary, "alpha_pvalue", 0).is_nan());
    assert_eq!(text(summary, "status", 0), "Historical mean advantage");
}

#[test]
fn annualization_scales_estimates_and_intervals_but_not_inference() {
    let (strategy, benchmark) = sample(180);
    let monthly = evidence(
        &strategy,
        &benchmark,
        EvidenceOptions { periods_per_year: 12.0, risk_free_rate: 0.0, ..Default::default() },
    );
    let daily = evidence(
        &strategy,
        &benchmark,
        EvidenceOptions { periods_per_year: 252.0, risk_free_rate: 0.0, ..Default::default() },
    );
    let scaled =
        |column: &str, factor: f64| monthly.summary.f64s(column).iter().map(|value| value * factor).collect::<Vec<_>>();
    for column in ["annual_advantage", "alpha", "advantage_ci_low", "alpha_ci_high"] {
        assert_allclose(&daily.summary.f64s(column), &scaled(column, 21.0), 0.0, 1e-7);
    }
    assert_allclose(&daily.summary.f64s("tracking_error"), &scaled("tracking_error", 21f64.sqrt()), 0.0, 1e-7);
    assert_allclose(&daily.summary.f64s("information_ratio"), &scaled("information_ratio", 21f64.sqrt()), 0.0, 1e-7);
    for column in ["beta", "advantage_pvalue_adjusted", "alpha_pvalue_adjusted"] {
        assert_allclose(&daily.summary.f64s(column), &monthly.summary.f64s(column), 0.0, 1e-7);
    }
}

#[test]
fn invalid_equity_values_are_rejected() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let (mut strategy, benchmark) = sample(180);
        strategy.values[(3, 0)] = bad;
        let error = analyze_evidence(&strategy, &benchmark, &EvidenceOptions::default()).unwrap_err();
        assert!(error.message().contains("finite and strictly positive"), "{error}");
    }
}

#[test]
fn mismatched_dates_are_rejected_without_silent_alignment() {
    let (strategy, benchmark) = sample(180);
    let error =
        analyze_evidence(&strategy.rows(1, strategy.nrows()), &benchmark, &EvidenceOptions::default()).unwrap_err();
    assert!(error.message().contains("exactly the same dates"), "{error}");
}

#[test]
fn unsorted_dates_are_rejected() {
    let (strategy, benchmark) = sample(180);
    let reverse = |frame: &Frame| {
        let order: Vec<usize> = (0..frame.nrows()).rev().collect();
        Frame::from_rows(
            order.iter().map(|i| frame.index[*i]).collect(),
            frame.columns.clone(),
            &order.iter().map(|i| frame.row(*i)).collect::<Vec<_>>(),
        )
    };
    let error = analyze_evidence(&reverse(&strategy), &reverse(&benchmark), &EvidenceOptions::default()).unwrap_err();
    assert!(error.message().contains("increasing"), "{error}");
}

#[test]
fn duplicate_names_are_rejected() {
    let (strategy, benchmark) = sample(180);
    let error =
        analyze_evidence(&combine(&[&strategy, &strategy]), &benchmark, &EvidenceOptions::default()).unwrap_err();
    assert!(error.message().contains("unique column names"), "{error}");
}

#[test]
fn invalid_settings_are_rejected() {
    let (strategy, benchmark) = sample(180);
    for (options, message) in [
        (EvidenceOptions { periods_per_year: 0.0, ..Default::default() }, "periods_per_year"),
        (EvidenceOptions { risk_free_rate: f64::INFINITY, ..Default::default() }, "risk_free_rate"),
    ] {
        let error = analyze_evidence(&strategy, &benchmark, &options).unwrap_err();
        assert!(error.is_value() && error.message().contains(message), "{error}");
    }
}
