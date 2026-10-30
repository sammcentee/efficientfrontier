//! Port of `tests/test_profiles.py`.

mod legacy_common;

use efficient_frontier::core::{AnalyzeOptions, analyze};
use efficient_frontier::frame::{Frame, Table};
use efficient_frontier::profiles::{ProfileAnalysis, ProfileOptions, ReturnObjective, build_profiles};
use legacy_common::{
    approx, assert_allclose, date, month_ends, monthly_prices, regime_prices, regime_returns, repeat, tile,
};

const OBJECTIVES: [ReturnObjective; 2] = [ReturnObjective::HistoricalMean, ReturnObjective::WorstWindow];

fn profiles(prices: &Frame, options: ProfileOptions) -> ProfileAnalysis {
    build_profiles(prices, &options).unwrap()
}

fn options() -> ProfileOptions {
    ProfileOptions::default()
}

fn worst_window() -> ProfileOptions {
    ProfileOptions { return_objective: ReturnObjective::WorstWindow, ..options() }
}

fn weights(result: &ProfileAnalysis, name: &str) -> Vec<f64> {
    result.portfolios[name].weights.values.clone()
}

/// The rows of a weight table.
fn rows(table: &Table) -> Vec<Vec<f64>> {
    (0..table.nrows()).map(|i| table.data.iter().map(|column| column[i].as_f64().unwrap()).collect()).collect()
}

fn summary(result: &ProfileAnalysis, profile: &str, column: &str) -> f64 {
    result.summary.f64_at(profile, column)
}

/// Sample variance of each regime asset: `prices.pct_change().var()`.
fn regime_variances() -> [f64; 2] {
    let returns = regime_returns();
    let variance = |values: &[f64]| efficient_frontier::frame::var(values, 1);
    [variance(&returns[0]), variance(&returns[1])]
}

fn increasing(values: &[f64], slack: f64) -> bool {
    values.windows(2).all(|pair| pair[1] - pair[0] >= -slack)
}

#[test]
fn analytic_profiles_prefer_consistent_windows_over_the_full_sample_winner() {
    let result =
        profiles(&regime_prices(), ProfileOptions { shrinkage: 1.0, periods_per_year: 12.0, ..worst_window() });
    let [steady, transient] = regime_variances();
    let low = [transient / (steady + transient), steady / (steady + transient)];
    let extreme = [1.0, 0.0];
    assert_allclose(&weights(&result, "Low"), &low, 2e-6, 0.0);
    assert_allclose(&weights(&result, "Extreme"), &extreme, 2e-6, 0.0);
    assert_allclose(&weights(&result, "Medium"), &[(low[0] + 1.0) / 2.0, low[1] / 2.0], 2e-6, 0.0);
    assert!((summary(&result, "Extreme", "worst_window_return") - 0.12).abs() <= 1e-8);
    assert!(summary(&result, "Extreme", "expected_return") < summary(&result, "Low", "expected_return"));
    assert!(increasing(&result.frontier.f64s("volatility"), 1e-9));
    let worst = result.frontier.f64s("worst_window_return");
    assert!(increasing(&worst, 1e-9));
    let linear: Vec<f64> = (0..21).map(|i| worst[0] + (worst[20] - worst[0]) * i as f64 / 20.0).collect();
    assert_allclose(&worst, &linear, 2e-7, 0.0);
    assert_eq!(result.portfolios.keys().collect::<Vec<_>>(), ["Low", "Medium", "Extreme"]);
    assert_eq!(result.summary.index_labels(), ["Low", "Medium", "Extreme"]);
    assert_eq!(result.settings.str("return_objective"), "worst_window");
    assert_eq!(result.settings.str("selection_method"), "worst_window_return_frontier");
}

#[test]
fn default_profiles_follow_full_sample_means_and_midpoint_volatility() {
    let result = profiles(&regime_prices(), ProfileOptions { shrinkage: 1.0, periods_per_year: 12.0, ..options() });
    let [steady, transient] = regime_variances();
    let low = [transient / (steady + transient), steady / (steady + transient)];
    let extreme = [0.0, 1.0];
    let expected_weights: Vec<Vec<f64>> = (0..21)
        .map(|i| {
            let t = i as f64 / 20.0;
            vec![low[0] + t * (extreme[0] - low[0]), low[1] + t * (extreme[1] - low[1])]
        })
        .collect();
    let expected_volatility: Vec<f64> = expected_weights
        .iter()
        .map(|w| (w[0] * w[0] * steady * 12.0 + w[1] * w[1] * transient * 12.0).sqrt())
        .collect();
    let midpoint = (expected_volatility[0] + expected_volatility[20]) / 2.0;
    let distances: Vec<f64> = expected_volatility.iter().map(|v| (v - midpoint).abs()).collect();
    let middle = (0..21).fold(0, |best, i| if distances[i] < distances[best] { i } else { best });
    for (actual, expected) in rows(&result.frontier_weights).iter().zip(&expected_weights) {
        assert_allclose(actual, expected, 2e-6, 0.0);
    }
    let expected_returns: Vec<f64> = expected_weights.iter().map(|w| w[0] * 0.12 + w[1] * 0.2).collect();
    assert_allclose(&result.frontier.f64s("expected_return"), &expected_returns, 1e-8, 0.0);
    assert_allclose(&result.frontier.f64s("volatility"), &expected_volatility, 1e-7, 0.0);
    assert_allclose(&weights(&result, "Medium"), &expected_weights[middle], 2e-6, 0.0);
    assert_ne!(middle, 10, "halfway in volatility differs from halfway in return");
    assert!((summary(&result, "Extreme", "expected_return") - 0.2).abs() <= 1e-8);
    assert!((summary(&result, "Extreme", "worst_window_return") + 0.24).abs() <= 1e-8);
    assert_eq!(result.settings.str("return_objective"), "historical_mean");
    assert_eq!(result.settings.str("selection_method"), "historical_mean_frontier");
    assert!(!result.warnings.iter().any(|message| message.contains("positive estimated return in all three")));
}

#[test]
fn mean_profiles_agree_with_classic_frontier_on_the_same_estimation_data() {
    let prices = regime_prices();
    // Six unchanged future prices after the profile history.
    let last = prices.nrows() - 1;
    let mut index = prices.index.clone();
    index.extend(month_ends(date(2021, 8, 31), 6));
    let mut values: Vec<Vec<f64>> = (0..prices.nrows()).map(|i| prices.row(i)).collect();
    values.extend(std::iter::repeat_n(prices.row(last), 6));
    let extended = Frame::from_rows(index, prices.columns.clone(), &values);
    for frontier_points in [3, 21, 40] {
        let settings = ProfileOptions {
            shrinkage: 0.4,
            max_weight: 0.9,
            risk_free_rate: 0.03,
            periods_per_year: 12.0,
            frontier_points,
            ..options()
        };
        let result = profiles(&prices, settings);
        let classic = analyze(
            &extended,
            &AnalyzeOptions {
                train_fraction: 0.75,
                risk_free_rate: 0.03,
                max_weight: 0.9,
                shrinkage: 0.4,
                frontier_points,
                periods_per_year: 12.0,
            },
        )
        .unwrap();
        assert_eq!(result.frontier_weights, classic.frontier.weights_table(), "{frontier_points} points");
        assert_eq!(
            result.frontier.select(&["expected_return", "volatility", "sharpe"]),
            classic.frontier.points_table()
        );
        for (name, classic_name) in [("Low", "Low"), ("Medium", "Medium"), ("Extreme", "High")] {
            assert_eq!(weights(&result, name), classic.portfolios[classic_name].weights.values);
        }
    }
}

#[test]
fn position_cap_applies_to_every_frontier_point() {
    for objective in OBJECTIVES {
        let result = profiles(
            &regime_prices(),
            ProfileOptions {
                max_weight: 0.9,
                shrinkage: 1.0,
                periods_per_year: 12.0,
                return_objective: objective,
                ..options()
            },
        );
        let points = rows(&result.frontier_weights);
        for row in &points {
            assert!((row.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
            assert!(row.iter().all(|w| *w >= 0.0 && *w <= 0.900001), "{row:?}");
        }
        let expected = if objective == ReturnObjective::HistoricalMean { [0.1, 0.9] } else { [0.9, 0.1] };
        assert_allclose(&weights(&result, "Extreme"), &expected, 2e-6, 0.0);
        let expected_return = expected[0] * 0.12 + expected[1] * 0.2;
        assert!((summary(&result, "Extreme", "expected_return") - expected_return).abs() <= 1e-8);
        if objective == ReturnObjective::HistoricalMean {
            assert!(increasing(&result.frontier.f64s("expected_return"), 1e-9));
        }
        assert!(increasing(&result.frontier.f64s("volatility"), 1e-9));
    }
}

#[test]
fn binding_cap_gives_identical_profiles_with_a_warning() {
    for objective in OBJECTIVES {
        let result = profiles(
            &regime_prices(),
            ProfileOptions { max_weight: 0.5, frontier_points: 3, return_objective: objective, ..options() },
        );
        for row in rows(&result.frontier_weights) {
            assert_allclose(&row, &[0.5, 0.5], 1e-9, 0.0);
        }
        assert!(result.warnings.iter().any(|warning| warning.contains("no distinct risk levels")));
    }
}

#[test]
fn windows_cover_every_return_once_and_include_the_latest_date() {
    let base = regime_prices();
    let last = base.row(base.nrows() - 1);
    let mut index = base.index.clone();
    index.push(date(2021, 8, 31));
    let mut values: Vec<Vec<f64>> = (0..base.nrows()).map(|i| base.row(i)).collect();
    values.push(vec![last[0] * 1.02, last[1] * 0.96]);
    let prices = Frame::from_rows(index, base.columns.clone(), &values);
    let returns = prices.pct_change();
    for objective in OBJECTIVES {
        let result = profiles(
            &prices,
            ProfileOptions { periods_per_year: 12.0, frontier_points: 3, return_objective: objective, ..options() },
        );
        assert_eq!(result.windows.f64s("observations"), [7.0, 6.0, 6.0]);
        assert_eq!(result.observations, 19);
        assert_eq!(result.as_of, "2021-08-31");
        for (number, (start, end)) in [(0, 7), (7, 13), (13, 19)].into_iter().enumerate() {
            let chunk = returns.rows(start, end);
            assert_eq!(result.windows.column("start")[number].py_str(), efficient_frontier::frame::iso(chunk.index[0]));
            assert_eq!(
                result.windows.column("end")[number].py_str(),
                efficient_frontier::frame::iso(*chunk.index.last().unwrap())
            );
            approx(result.windows.f64s("years")[number], (end - start) as f64 / 12.0);
            for (name, portfolio) in &result.portfolios {
                let means = chunk.means();
                let expected: f64 = means.iter().zip(&portfolio.weights.values).map(|(m, w)| m * w).sum::<f64>() * 12.0;
                approx(result.window_returns.f64s(name)[number], expected);
            }
        }
        let lowest: Vec<f64> = result
            .portfolios
            .keys()
            .map(|name| result.window_returns.f64s(name).into_iter().fold(f64::INFINITY, f64::min))
            .collect();
        assert_allclose(&lowest, &result.summary.f64s("worst_window_return"), 0.0, 1e-7);
    }
}

#[test]
fn price_units_and_asset_order_do_not_change_profiles() {
    let prices = regime_prices();
    let scaled_values: Vec<Vec<f64>> = (0..prices.nrows())
        .map(|i| {
            let row = prices.row(i);
            vec![row[1] * 1000.0, row[0] * 0.01]
        })
        .collect();
    let rescaled = Frame::from_rows(prices.index.clone(), vec!["Transient".into(), "Steady".into()], &scaled_values);
    for objective in OBJECTIVES {
        let settings = ProfileOptions { frontier_points: 3, return_objective: objective, ..options() };
        let reference = profiles(&prices, settings);
        let reordered = profiles(&rescaled, settings);
        assert_eq!(reordered.frontier_weights.columns, ["Transient", "Steady"]);
        for name in reference.portfolios.keys() {
            let aligned = reordered.portfolios[name].weights.reindex(&prices.columns, f64::NAN);
            assert_allclose(&weights(&reference, name), &aligned.values, 2e-6, 0.0);
        }
        assert_eq!(reference.summary.columns, reordered.summary.columns);
        assert_eq!(reference.summary.index, reordered.summary.index);
        for column in &reference.summary.columns {
            assert_allclose(&reordered.summary.f64s(column), &reference.summary.f64s(column), 1e-6, 1e-6);
        }
    }
}

#[test]
fn six_returns_and_one_asset_are_valid() {
    let prices = monthly_prices(&[vec![0.01, -0.02, 0.01, 0.02, 0.03, -0.01]], &["Only"]);
    for objective in OBJECTIVES {
        let result =
            profiles(&prices, ProfileOptions { periods_per_year: 12.0, return_objective: objective, ..options() });
        for row in rows(&result.frontier_weights) {
            assert_allclose(&row, &[1.0], 0.0, 1e-7);
        }
        assert_eq!(result.frontier.nrows(), if objective == ReturnObjective::HistoricalMean { 1 } else { 21 });
        assert_eq!(result.windows.f64s("observations"), [2.0, 2.0, 2.0]);
        assert!(result.summary.f64s("effective_holdings").iter().all(|value| *value == 1.0));
        assert!(result.warnings.iter().any(|warning| warning.contains("no distinct risk levels")));
        assert!(result.warnings.iter().any(|warning| warning.contains("less than one year")));
    }
}

#[test]
fn zero_covariance_preserves_undefined_sharpe_and_flat_profiles() {
    let prices = monthly_prices(&[vec![0.0; 9], vec![0.0; 9]], &["A", "B"]);
    for objective in OBJECTIVES {
        let result = profiles(&prices, ProfileOptions { frontier_points: 3, return_objective: objective, ..options() });
        assert!(result.frontier.f64s("volatility").iter().all(|value| *value == 0.0));
        assert!(result.frontier.f64s("worst_window_return").iter().all(|value| *value == 0.0));
        assert!(result.frontier.f64s("sharpe").iter().all(|value| value.is_nan()));
        let points = rows(&result.frontier_weights);
        for row in &points {
            assert_allclose(row, &points[0], 0.0, 1e-7);
        }
        assert!(result.warnings.iter().any(|warning| warning.contains("no distinct risk levels")));
        assert!(result.warnings.iter().any(|warning| warning.contains("No feasible portfolio")));
    }
}

#[test]
fn non_daily_annualization_scales_risk_and_return_without_changing_weights() {
    let prices = regime_prices();
    let monthly = profiles(&prices, ProfileOptions { periods_per_year: 12.0, frontier_points: 3, ..options() });
    let daily = profiles(&prices, ProfileOptions { periods_per_year: 252.0, frontier_points: 3, ..options() });
    for (a, b) in rows(&daily.frontier_weights).iter().zip(rows(&monthly.frontier_weights)) {
        assert_allclose(a, &b, 1e-6, 0.0);
    }
    let scaled = |table: &Table, column: &str, factor: f64| -> Vec<f64> {
        table.f64s(column).iter().map(|value| value * factor).collect()
    };
    assert_allclose(
        &daily.summary.f64s("expected_return"),
        &scaled(&monthly.summary, "expected_return", 21.0),
        0.0,
        1e-5,
    );
    assert_allclose(
        &daily.summary.f64s("worst_window_return"),
        &scaled(&monthly.summary, "worst_window_return", 21.0),
        0.0,
        1e-5,
    );
    assert_allclose(
        &daily.summary.f64s("volatility"),
        &scaled(&monthly.summary, "volatility", 21f64.sqrt()),
        0.0,
        1e-5,
    );
    assert_eq!(monthly.settings.f64("periods_per_year"), 12.0);
    assert_eq!(monthly.settings.int("window_count"), 3);
    assert_eq!(monthly.settings.str("selection_method"), "historical_mean_frontier");
}

#[test]
fn three_and_twenty_one_points_select_the_same_worst_window_profiles() {
    let prices = regime_prices();
    let short = profiles(&prices, ProfileOptions { frontier_points: 3, ..worst_window() });
    let full = profiles(&prices, ProfileOptions { frontier_points: 21, ..worst_window() });
    for name in short.portfolios.keys() {
        assert_allclose(&weights(&short, name), &weights(&full, name), 2e-6, 0.0);
    }
}

#[test]
fn extreme_profile_minimizes_variance_when_the_best_return_has_multiple_solutions() {
    for return_scale in [1.0, 1e-6] {
        let first: Vec<f64> =
            tile(&[-0.005, 0.005, -0.005, 0.005], 3).iter().map(|v| (0.01 + v) * return_scale).collect();
        let second: Vec<f64> =
            tile(&[-0.015, -0.015, 0.015, 0.015], 3).iter().map(|v| (0.01 + v) * return_scale).collect();
        let third: Vec<f64> = repeat(&[-0.02, 0.02, 0.02], 4).iter().map(|v| v * return_scale).collect();
        let prices = monthly_prices(&[first, second, third], &["A", "B", "C"]);
        let result = profiles(
            &prices,
            ProfileOptions { shrinkage: 1.0, periods_per_year: 12.0, frontier_points: 3, ..worst_window() },
        );
        assert_allclose(&weights(&result, "Extreme"), &[0.9, 0.1, 0.0], 3e-6, 0.0);
        let worst = summary(&result, "Extreme", "worst_window_return");
        assert!((worst - 0.12 * return_scale).abs() <= 1e-6 * 0.12 * return_scale, "{worst}");
    }
}

#[test]
fn tiny_returns_preserve_the_worst_window_frontier_weights() {
    let prices = regime_prices();
    let returns = regime_returns();
    let tiny: Vec<Vec<f64>> = returns.iter().map(|column| column.iter().map(|value| value * 1e-6).collect()).collect();
    let reference = profiles(&prices, ProfileOptions { frontier_points: 3, ..worst_window() });
    let result = profiles(
        &monthly_prices(&tiny, &["Steady", "Transient"]),
        ProfileOptions { frontier_points: 3, ..worst_window() },
    );
    for (a, b) in rows(&result.frontier_weights).iter().zip(rows(&reference.frontier_weights)) {
        assert_allclose(a, &b, 3e-6, 0.0);
    }
    let expected: Vec<f64> = reference.summary.f64s("volatility").iter().map(|value| value * 1e-6).collect();
    assert_allclose(&result.summary.f64s("volatility"), &expected, 0.0, 1e-5);
}

#[test]
fn nonpositive_worst_window_returns_remain_feasible_with_a_warning() {
    let falling: Vec<Vec<f64>> =
        regime_returns().iter().map(|column| column.iter().map(|value| value - 0.1).collect()).collect();
    let result = profiles(
        &monthly_prices(&falling, &["Steady", "Transient"]),
        ProfileOptions { periods_per_year: 12.0, frontier_points: 3, ..worst_window() },
    );
    assert!(result.summary.f64s("worst_window_return").iter().all(|value| *value < 0.0));
    assert_allclose(&weights(&result, "Extreme"), &[1.0, 0.0], 2e-6, 0.0);
    assert!(result.warnings.iter().any(|warning| warning.contains("No feasible portfolio")));
}

#[test]
fn invalid_settings_fail_with_a_clear_error() {
    let cases: [(ProfileOptions, &str); 9] = [
        (ProfileOptions { max_weight: 0.0, ..options() }, "max_weight"),
        (ProfileOptions { max_weight: 1.1, ..options() }, "max_weight"),
        (ProfileOptions { max_weight: 0.4, ..options() }, "infeasible"),
        (ProfileOptions { shrinkage: -0.1, ..options() }, "shrinkage"),
        (ProfileOptions { shrinkage: f64::NAN, ..options() }, "shrinkage"),
        (ProfileOptions { risk_free_rate: f64::INFINITY, ..options() }, "risk_free_rate"),
        (ProfileOptions { periods_per_year: 0.0, ..options() }, "periods_per_year"),
        (ProfileOptions { frontier_points: 2, ..options() }, "integer"),
        (ProfileOptions { frontier_points: 4, ..worst_window() }, "odd integer"),
    ];
    for (settings, message) in cases {
        let error = build_profiles(&regime_prices(), &settings).unwrap_err();
        assert!(error.is_value() && error.message().contains(message), "{error} should mention {message}");
    }
    // Python rejects other objective names; Rust parses them into the enum first.
    let error = "unknown".parse::<ReturnObjective>().unwrap_err();
    assert!(error.is_value() && error.message().contains("return_objective"));
}

#[test]
fn profiles_reject_less_than_six_returns() {
    let error = build_profiles(&regime_prices().rows(0, 6), &options()).unwrap_err();
    assert!(error.message().contains("at least six returns"), "{error}");
}

#[test]
fn profiles_reject_missing_prices_without_filling_them() {
    let mut prices = regime_prices();
    prices.values[(3, 1)] = f64::NAN;
    let error = build_profiles(&prices, &options()).unwrap_err();
    assert!(error.message().contains("resolve missing data"), "{error}");
}

#[test]
fn profiles_reject_dates_out_of_order() {
    let prices = regime_prices();
    let order: Vec<usize> = (0..prices.nrows()).rev().collect();
    let reversed = Frame::from_rows(
        order.iter().map(|i| prices.index[*i]).collect(),
        prices.columns.clone(),
        &order.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
    );
    let error = build_profiles(&reversed, &options()).unwrap_err();
    assert!(error.message().contains("increasing"), "{error}");
}

#[test]
fn worst_window_low_uses_best_return_among_identical_risk_allocations() {
    for return_scale in [1.0, 1e-6] {
        let noise = tile(&[-0.01, 0.01, -0.01, 0.01], 6);
        let independent = tile(&[-0.015, -0.015, 0.015, 0.015], 6);
        let columns = vec![
            noise.iter().map(|v| (0.001 + v) * return_scale).collect(),
            noise.iter().map(|v| (0.002 + v) * return_scale).collect(),
            independent.iter().map(|v| (0.003 + v) * return_scale).collect(),
        ];
        let result = profiles(
            &monthly_prices(&columns, &["A", "B", "C"]),
            ProfileOptions { shrinkage: 0.0, periods_per_year: 12.0, frontier_points: 5, ..worst_window() },
        );
        // A and B have the same covariance. B has a higher mean in all three windows.
        assert_allclose(&weights(&result, "Low"), &[0.0, 9.0 / 13.0, 4.0 / 13.0], 3e-7, 0.0);
        assert!(result.frontier_weights.f64s("A").iter().all(|w| w.abs() <= 2e-6));
        assert!(result.frontier.f64s("worst_window_return").windows(2).all(|pair| pair[1] > pair[0]));
        assert!(result.frontier.f64s("volatility").windows(2).all(|pair| pair[1] > pair[0]));
    }
}
