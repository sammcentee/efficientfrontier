//! Port of `tests/test_benchmarks.py`.

mod legacy_common;

use chrono::{Days, Weekday};
use efficient_frontier::backtest::{BacktestOptions, run_backtests};
use efficient_frontier::benchmarks::compare_benchmarks;
use efficient_frontier::core::{Analysis, AnalyzeOptions, analyze};
use efficient_frontier::frame::{Date, Frame, Table, mean, std};
use efficient_frontier::profiles::{ProfileOptions, build_profiles};
use legacy_common::{approx, assert_allclose, calendar_days, column, date, month_starts, weeks};

const NAMES: [&str; 2] = ["S&P 500 (SPY)", "Nasdaq-100 (QQQ)"];

/// Asset and benchmark prices on eleven dates.
fn histories(dates: Vec<Date>) -> (Frame, Frame) {
    let assets = Frame::from_columns(
        dates.clone(),
        vec![
            ("A".into(), vec![100.0, 102.0, 99.0, 104.0, 101.0, 103.0, 105.0, 107.0, 104.0, 109.0, 111.0]),
            ("B".into(), vec![100.0, 101.0, 100.0, 102.0, 103.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0]),
        ],
    );
    let benchmarks = Frame::from_columns(
        dates,
        vec![
            ("SPY".into(), vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 100.0, 200.0, 220.0, 198.0, 210.0]),
            ("QQQ".into(), vec![100.0, 103.0, 102.0, 104.0, 100.0, 107.0, 100.0, 80.0, 88.0, 92.0, 84.0]),
        ],
    );
    (assets, benchmarks)
}

fn monthly() -> (Frame, Frame) {
    histories(month_starts(date(2020, 1, 1), 11))
}

fn analysis(assets: &Frame, options: AnalyzeOptions) -> Analysis {
    analyze(assets, &options).unwrap()
}

fn settings(train_fraction: f64, risk_free_rate: f64, periods_per_year: f64) -> (AnalyzeOptions, BacktestOptions) {
    (
        AnalyzeOptions { train_fraction, risk_free_rate, periods_per_year, ..Default::default() },
        BacktestOptions { train_fraction, risk_free_rate, periods_per_year, ..Default::default() },
    )
}

/// The numeric cells of one labelled table row.
fn table_row(table: &Table, label: &str) -> Vec<f64> {
    let i = table.row_of(label).unwrap_or_else(|| panic!("row {label:?} is missing"));
    table.data.iter().map(|values| values[i].as_f64().unwrap_or(f64::NAN)).collect()
}

#[test]
fn holdout_and_backtests_use_their_own_entry_timing_and_costs() {
    let (assets, prices) = monthly();
    let (analyze_options, backtest_options) = settings(0.6, 0.03, 12.0);
    let analysis = analysis(&assets, analyze_options);
    let study = run_backtests(&assets, &BacktestOptions { cost_bps: 100.0, ..backtest_options }).unwrap();
    let result = compare_benchmarks(&prices, &assets, &analysis, Some(&study), None, 0.03).unwrap();
    let after_entry = 1.0 / 1.01;
    assert_allclose(&column(&result.holdout_equity, NAMES[0]), &[1.0, 2.0, 2.2, 1.98, 2.1], 0.0, 1e-7);
    assert_allclose(&column(&result.holdout_equity, NAMES[1]), &[1.0, 0.8, 0.88, 0.92, 0.84], 0.0, 1e-7);
    let backtest = result.backtest_equity.as_ref().unwrap();
    let path = |values: [f64; 4]| {
        [1.0, values[0] * after_entry, values[1] * after_entry, values[2] * after_entry, values[3] * after_entry]
    };
    assert_allclose(&column(backtest, NAMES[0]), &path([1.0, 1.1, 0.99, 1.05]), 0.0, 1e-7);
    assert_allclose(&column(backtest, NAMES[1]), &path([1.0, 1.1, 1.15, 1.05]), 0.0, 1e-7);
    let metrics = result.backtest_metrics.as_ref().unwrap();
    for name in NAMES {
        approx(metrics.f64_at(name, "total_turnover"), after_entry);
        approx(metrics.f64_at(name, "total_cost"), 0.01 / 1.01);
        approx(metrics.f64_at(name, "total_return"), 1.05 / 1.01 - 1.0);
        assert_eq!(metrics.cell(name, "rebalance_count").as_int(), Some(1));
        assert_eq!(metrics.cell(name, "fallback_count").as_int(), Some(0));
    }
    assert_eq!(result.holdout_equity.index, analysis.equity.index);
    assert_eq!(backtest.index, study.equity.index);
    assert_eq!(result.settings.f64("holdout_cost_bps"), 0.0);
    assert_eq!(result.settings.f64("backtest_cost_bps"), 100.0);
}

#[test]
fn benchmarks_match_independent_single_asset_engine_results() {
    for (dates, periods) in [
        (month_starts(date(2020, 1, 1), 11), 12.0),
        (weeks(date(2020, 1, 1), 11, Weekday::Fri), 52.0),
        (calendar_days(date(2020, 1, 1), 11), 365.0),
    ] {
        let (assets, prices) = histories(dates);
        let (analyze_options, backtest_options) = settings(0.6, 0.07, periods);
        let backtest_options = BacktestOptions { cost_bps: 40.0, ..backtest_options };
        let analysis = analysis(&assets, analyze_options);
        let study = run_backtests(&assets, &backtest_options).unwrap();
        let result = compare_benchmarks(&prices, &assets, &analysis, Some(&study), None, 0.07).unwrap();
        let backtest = result.backtest_equity.as_ref().unwrap();
        for (symbol, name) in ["SPY", "QQQ"].into_iter().zip(NAMES) {
            let single = prices.select(&[symbol]).unwrap();
            let single_analysis = analyze(&single, &analyze_options).unwrap();
            let single_study = run_backtests(&single, &backtest_options).unwrap();
            assert_allclose(
                &column(&result.holdout_equity, name),
                &column(&single_analysis.equity, "Equal weight"),
                0.0,
                1e-7,
            );
            assert_allclose(
                &column(backtest, name),
                &column(&single_study.equity, "Buy and hold · Equal weight"),
                0.0,
                1e-7,
            );
            let expected: Vec<f64> =
                single_analysis.holdout_metrics["Equal weight"].fields().iter().map(|(_, value)| *value).collect();
            assert_allclose(&table_row(&result.holdout_metrics, name), &expected, 0.0, 1e-7);
            assert_allclose(
                &table_row(result.backtest_metrics.as_ref().unwrap(), name),
                &table_row(&single_study.metrics, "Buy and hold · Equal weight"),
                0.0,
                1e-7,
            );
        }
        assert_eq!(result.settings.f64("periods_per_year"), periods);
    }
}

#[test]
fn estimates_use_the_same_training_dates_and_latest_windows() {
    let (assets, prices) = monthly();
    let analysis = analysis(
        &assets,
        AnalyzeOptions {
            train_fraction: 0.6,
            periods_per_year: 12.0,
            risk_free_rate: 0.08,
            shrinkage: 1.0,
            ..Default::default()
        },
    );
    let latest =
        build_profiles(&assets, &ProfileOptions { periods_per_year: 12.0, risk_free_rate: 0.08, ..Default::default() })
            .unwrap();
    let result = compare_benchmarks(&prices, &assets, &analysis, None, Some(&latest), 0.08).unwrap();
    for (j, name) in NAMES.iter().enumerate() {
        let values = prices.col(j);
        let returns: Vec<f64> = values.windows(2).map(|pair| pair[1] / pair[0] - 1.0).collect();
        let train = &returns[..6];
        let (expected, volatility) = (mean(train) * 12.0, std(train, 1) * 12f64.sqrt());
        approx(result.training_estimates.f64_at(name, "expected_return"), expected);
        approx(result.training_estimates.f64_at(name, "volatility"), volatility);
        approx(result.training_estimates.f64_at(name, "sharpe"), (expected - 0.08) / volatility);
        approx(result.latest_estimates.f64_at(name, "expected_return"), mean(&returns) * 12.0);
        approx(result.latest_estimates.f64_at(name, "volatility"), std(&returns, 1) * 12f64.sqrt());
        // numpy.array_split of ten returns into three windows: 4, 3 and 3.
        let worst = [&returns[..4], &returns[4..7], &returns[7..]]
            .iter()
            .map(|chunk| mean(chunk) * 12.0)
            .fold(f64::INFINITY, f64::min);
        approx(result.latest_estimates.f64_at(name, "worst_window_return"), worst);
    }
    assert!(result.backtest_equity.is_none());
    assert!(result.backtest_metrics.is_none());
}

#[test]
fn missing_optional_results_do_not_create_backtests_or_window_estimates() {
    let (assets, prices) = monthly();
    let analysis = analysis(&assets, AnalyzeOptions { train_fraction: 0.6, ..Default::default() });
    let result = compare_benchmarks(&prices, &assets, &analysis, None, None, 0.02).unwrap();
    assert!(result.backtest_equity.is_none());
    assert!(result.backtest_metrics.is_none());
    assert!(result.latest_estimates.f64s("worst_window_return").iter().all(|value| value.is_nan()));
}

#[test]
fn benchmarks_reject_missing_or_extra_full_input_dates() {
    let (assets, prices) = monthly();
    let analysis = analysis(&assets, AnalyzeOptions { train_fraction: 0.6, ..Default::default() });
    let rows: Vec<usize> = (0..prices.nrows()).filter(|i| *i != 2).collect();
    let dropped = Frame::from_rows(
        rows.iter().map(|i| prices.index[*i]).collect(),
        prices.columns.clone(),
        &rows.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
    );
    let last = prices.nrows() - 1;
    let mut index = prices.index.clone();
    index.push(prices.index[last] + Days::new(1));
    let mut values: Vec<Vec<f64>> = (0..prices.nrows()).map(|i| prices.row(i)).collect();
    values.push(prices.row(last));
    let extra = Frame::from_rows(index, prices.columns.clone(), &values);
    for changed in [dropped, extra] {
        let error = compare_benchmarks(&changed, &assets, &analysis, None, None, 0.02).unwrap_err();
        assert!(error.is_value() && error.message().to_lowercase().contains("date"), "{error}");
    }
}

#[test]
fn benchmarks_do_not_change_or_expand_the_original_results() {
    // Rust passes shared references, so the inputs cannot change; this checks the results keep their shape.
    let (assets, prices) = monthly();
    let analysis = analysis(
        &assets,
        AnalyzeOptions { train_fraction: 0.6, periods_per_year: 12.0, max_weight: 0.6, ..Default::default() },
    );
    let study = run_backtests(
        &assets,
        &BacktestOptions {
            train_fraction: 0.6,
            periods_per_year: 12.0,
            max_weight: 0.6,
            include_profiles: true,
            ..Default::default()
        },
    )
    .unwrap();
    let latest =
        build_profiles(&assets, &ProfileOptions { periods_per_year: 12.0, max_weight: 0.6, ..Default::default() })
            .unwrap();
    let original = (format!("{analysis:?}"), study.clone(), latest.clone());
    let result = compare_benchmarks(&prices, &assets, &analysis, Some(&study), Some(&latest), 0.02).unwrap();
    assert_eq!(format!("{analysis:?}"), original.0);
    assert_eq!(study, original.1);
    assert_eq!(latest, original.2);
    assert_eq!(analysis.mean_returns.labels, ["A", "B"]);
    assert_eq!(study.equity.ncols(), 24);
    assert_eq!(result.prices.columns, ["SPY", "QQQ"]);
}

#[test]
fn constant_benchmark_prices_have_undefined_training_and_holdout_sharpe() {
    let (assets, mut prices) = monthly();
    prices.values.fill(100.0);
    let analysis =
        analysis(&assets, AnalyzeOptions { train_fraction: 0.6, periods_per_year: 12.0, ..Default::default() });
    let result = compare_benchmarks(&prices, &assets, &analysis, None, None, 0.02).unwrap();
    assert!(result.training_estimates.f64s("sharpe").iter().all(|value| value.is_nan()));
    assert!(result.latest_estimates.f64s("sharpe").iter().all(|value| value.is_nan()));
    assert!(result.holdout_metrics.f64s("sharpe").iter().all(|value| value.is_nan()));
}

#[test]
fn benchmarks_reject_nonfinite_risk_free_rates() {
    let (assets, prices) = monthly();
    let analysis = analysis(&assets, AnalyzeOptions::default());
    for risk_free_rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = compare_benchmarks(&prices, &assets, &analysis, None, None, risk_free_rate).unwrap_err();
        assert!(error.message().contains("risk_free_rate"), "{error}");
    }
}

#[test]
fn benchmark_comparison_rejects_a_study_with_different_evaluation_dates() {
    let (assets, prices) = monthly();
    let analysis = analysis(&assets, AnalyzeOptions { train_fraction: 0.6, ..Default::default() });
    let study = run_backtests(&assets, &BacktestOptions { train_fraction: 0.5, ..Default::default() }).unwrap();
    let error = compare_benchmarks(&prices, &assets, &analysis, Some(&study), None, 0.02).unwrap_err();
    assert!(error.message().contains("same evaluation dates"), "{error}");
}

#[test]
fn benchmark_comparison_rejects_inconsistent_annualization() {
    let (assets, prices) = monthly();
    let analysis =
        analysis(&assets, AnalyzeOptions { train_fraction: 0.6, periods_per_year: 12.0, ..Default::default() });
    let study =
        run_backtests(&assets, &BacktestOptions { train_fraction: 0.6, periods_per_year: 252.0, ..Default::default() })
            .unwrap();
    let error = compare_benchmarks(&prices, &assets, &analysis, Some(&study), None, 0.02).unwrap_err();
    assert!(error.message().contains("periods_per_year"), "{error}");
}

#[test]
fn standalone_benchmarks_reject_a_rate_that_differs_from_analysis() {
    let (assets, prices) = monthly();
    let analysis = analysis(&assets, AnalyzeOptions { risk_free_rate: 0.07, ..Default::default() });
    assert_eq!(analysis.risk_free_rate, 0.07);
    let error = compare_benchmarks(&prices, &assets, &analysis, None, None, 0.02).unwrap_err();
    assert_eq!(error.message(), "Analysis and benchmarks must use the same risk_free_rate.");
}

#[test]
fn standalone_benchmarks_match_portfolio_metrics_with_a_nondefault_rate() {
    let (_, prices) = monthly();
    let mut assets = prices.select(&["SPY"]).unwrap();
    assets.columns = vec!["A".into()];
    let analysis =
        analysis(&assets, AnalyzeOptions { risk_free_rate: 0.07, periods_per_year: 12.0, ..Default::default() });
    let result = compare_benchmarks(&prices, &assets, &analysis, None, None, 0.07).unwrap();
    let expected: Vec<f64> =
        analysis.holdout_metrics["Equal weight"].fields().iter().map(|(_, value)| *value).collect();
    assert_allclose(&table_row(&result.holdout_metrics, NAMES[0]), &expected, 0.0, 1e-7);
}

#[test]
fn benchmark_comparison_rejects_inconsistent_risk_free_rates() {
    let (assets, prices) = monthly();
    let analysis =
        analysis(&assets, AnalyzeOptions { train_fraction: 0.6, risk_free_rate: 0.02, ..Default::default() });
    let study =
        run_backtests(&assets, &BacktestOptions { train_fraction: 0.6, risk_free_rate: 0.03, ..Default::default() })
            .unwrap();
    let error = compare_benchmarks(&prices, &assets, &analysis, Some(&study), None, 0.02).unwrap_err();
    assert!(error.message().contains("risk_free_rate"), "{error}");
    let latest = build_profiles(&assets, &ProfileOptions { risk_free_rate: 0.03, ..Default::default() }).unwrap();
    let error = compare_benchmarks(&prices, &assets, &analysis, None, Some(&latest), 0.02).unwrap_err();
    assert!(error.message().contains("risk_free_rate"), "{error}");
}
