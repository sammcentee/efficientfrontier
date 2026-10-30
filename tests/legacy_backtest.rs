//! Port of `tests/test_backtest.py`.
//!
//! Python replaces the optimizer and the profile builder with recording stubs
//! in three tests. Rust has no such seam, so those ports check the same rules
//! through the dates of each trade's fit window and through the real results.

mod legacy_common;

use chrono::Weekday;
use efficient_frontier::backtest::{BacktestOptions, BacktestStudy, METHOD_NAMES, run_backtests};
use efficient_frontier::frame::{Date, Frame, Table};
use legacy_common::{
    approx, assert_allclose, business_days, column, date, month_starts, numbers, parse_date, prices_from_columns,
    test_data, tile, weeks,
};

fn history() -> Frame {
    Frame::from_columns(
        business_days(date(2020, 1, 1), 8),
        vec![
            ("A".into(), vec![100.0, 103.0, 101.0, 100.0, 100.0, 200.0, 100.0, 100.0]),
            ("B".into(), vec![100.0, 101.0, 104.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        ],
    )
}

fn half() -> BacktestOptions {
    BacktestOptions { train_fraction: 0.5, ..Default::default() }
}

fn daily() -> BacktestOptions {
    BacktestOptions { rebalance_every: 1, ..half() }
}

fn run(prices: &Frame, options: BacktestOptions) -> BacktestStudy {
    run_backtests(prices, &options).unwrap()
}

fn metric(study: &BacktestStudy, key: &str, name: &str) -> f64 {
    study.metrics.f64_at(key, name)
}

fn holding(study: &BacktestStudy, key: &str, name: &str) -> Vec<f64> {
    study.holdings[key].f64s(name)
}

fn trade_dates(table: &Table) -> Vec<Date> {
    table.index[0].iter().map(|cell| cell.as_date().unwrap()).collect()
}

fn text_dates(table: &Table, name: &str) -> Vec<Date> {
    table.column(name).iter().map(|cell| parse_date(cell.as_str().unwrap())).collect()
}

/// Rows of a frame up to and including `last`, as `loc[:last]`.
fn through(frame: &Frame, last: Date) -> Frame {
    frame.rows_through(last)
}

/// Multiply the prices of column `j` from row `start` on.
fn scaled(prices: &Frame, start: usize, j: usize, factor: f64) -> Frame {
    let mut changed = prices.clone();
    for i in start..changed.nrows() {
        changed.values[(i, j)] *= factor;
    }
    changed
}

#[test]
fn buy_and_hold_drift_and_daily_rebalance_have_analytic_paths() {
    let study = run(&history(), BacktestOptions { cost_bps: 0.0, ..daily() });
    let buy = "Buy and hold · Equal weight";
    let fixed = "Fixed rebalance · Equal weight";
    assert_allclose(&column(&study.equity, buy), &[1.0, 1.0, 1.5, 1.0, 1.0], 0.0, 1e-7);
    assert_allclose(&column(&study.equity, fixed), &[1.0, 1.0, 1.5, 1.125, 1.125], 0.0, 1e-7);
    assert_allclose(&holding(&study, buy, "average_weight"), &[5.0 / 12.0, 1.0 / 3.0], 0.0, 1e-7);
    assert_allclose(&holding(&study, buy, "end_weight"), &[0.5, 0.5], 0.0, 1e-7);
    assert_allclose(&holding(&study, fixed, "pnl_contribution"), &[0.125, 0.0], 1e-15, 1e-7);
    assert_eq!(metric(&study, buy, "rebalance_count"), 1.0);
    assert_eq!(metric(&study, fixed, "rebalance_count"), 3.0);
    approx(metric(&study, fixed, "total_turnover"), 1.0 + 1.0 / 3.0 + 1.0 / 3.0);
    assert_eq!(*study.allocations[fixed].index.last().unwrap(), history().index[6]);
    assert_eq!(study.equity.ncols(), 12);
}

#[test]
fn entry_cost_and_rebalances_conserve_wealth_and_contributions() {
    let study = run(&history(), BacktestOptions { cost_bps: 100.0, ..daily() });
    assert!(study.equity.row(0).iter().all(|value| *value == 1.0));
    assert_allclose(&study.equity.row(1), &[1.0 / 1.01; 12], 0.0, 1e-7);
    for (key, trades) in &study.trades {
        let (before, after, cost) = (trades.f64s("nav_before"), trades.f64s("nav_after"), trades.f64s("cost"));
        let turnover = trades.f64s("turnover");
        let difference: Vec<f64> = before.iter().zip(&after).map(|(b, a)| b - a).collect();
        assert_allclose(&difference, &cost, 1e-14, 1e-7);
        let charged: Vec<f64> = turnover.iter().zip(&before).map(|(t, b)| 0.01 * t * b).collect();
        assert_allclose(&cost, &charged, 1e-14, 1e-7);
        approx(metric(&study, key, "total_cost"), cost.iter().sum());
        let reconciled = holding(&study, key, "pnl_contribution").iter().sum::<f64>() - cost.iter().sum::<f64>();
        assert!((metric(&study, key, "total_return") - reconciled).abs() <= 1e-13, "{key}");
        assert!(metric(&study, key, "max_drawdown") <= -(1.0 - 1.0 / 1.01), "{key}");
    }
}

#[test]
fn execution_day_jump_is_not_earned_before_initial_entry() {
    let mut prices = history();
    prices.values[(4, 0)] = 200.0;
    let study = run(&prices, BacktestOptions { rebalance_every: 2, cost_bps: 0.0, ..half() });
    assert_allclose(&column(&study.equity, "Buy and hold · Equal weight"), &[1.0, 1.0, 1.0, 0.75, 0.75], 0.0, 1e-7);
    assert_eq!(study.allocations["Fixed rebalance · Equal weight"].index, [prices.index[4], prices.index[6]]);
}

#[test]
fn realized_concentration_includes_drift_before_rebalancing() {
    let study = run(&history(), BacktestOptions { max_weight: 0.5, cost_bps: 0.0, ..daily() });
    let buy = "Buy and hold · Equal weight";
    let fixed = "Fixed rebalance · Equal weight";
    assert_allclose(&holding(&study, buy, "max_target_weight"), &[0.5, 0.5], 0.0, 1e-7);
    assert_allclose(&holding(&study, buy, "max_realized_weight"), &[2.0 / 3.0, 0.5], 0.0, 1e-7);
    assert_allclose(&holding(&study, fixed, "max_realized_weight"), &[2.0 / 3.0, 2.0 / 3.0], 0.0, 1e-7);
    assert_eq!(study.settings.f64("trading_days_per_year"), 252.0);
    assert_eq!(study.settings.f64("cash_interest_rate"), 0.0);
}

#[test]
fn full_switch_charges_both_sell_and_buy_legs() {
    // Python forces the optimizer to switch from A to B. The switch trade sells
    // every A dollar and buys B with the rest.
    let nav = 1.37;
    let after = efficient_frontier::backtest::after_cost(nav, &[nav, 0.0], &[0.0, 1.0], 0.01);
    approx(after, nav * 0.99 / 1.01);
    approx(nav - after, 0.01 * (nav + after));
}

#[test]
fn future_changes_do_not_affect_prior_equity_or_targets() {
    for periods_per_year in [12.0, 252.0, 365.0] {
        let options = BacktestOptions { periods_per_year, ..daily() };
        let prices = history();
        let first = run(&prices, options);
        let second = run(&scaled(&prices, 5, 0, 1.4), options);
        assert_eq!(first.equity.rows(0, 2), second.equity.rows(0, 2));
        for key in first.allocations.keys() {
            // Even the close-5 trade cannot observe the changed close-5 price.
            assert_eq!(
                through(&first.allocations[key], prices.index[5]),
                through(&second.allocations[key], prices.index[5]),
                "{key}"
            );
        }
    }
}

#[test]
fn training_windows_end_before_execution_and_rolling_is_bounded() {
    for periods_per_year in [12.0, 52.0, 252.0] {
        let prices = history();
        let study = run(&prices, BacktestOptions { rolling_window: Some(2), periods_per_year, ..daily() });
        let dates = &prices.index;
        // The fit windows of Python's recording stub, as `(start, end)` in returns:
        // the initial fit, the expanding windows and the bounded rolling windows.
        for (key, trades) in &study.trades {
            let windows: Vec<(usize, usize)> = match key.split(" · ").next().unwrap() {
                "Buy and hold" => vec![(0, 3)],
                "Fixed rebalance" => vec![(0, 3); 3],
                "Expanding window" => vec![(0, 3), (0, 4), (0, 5)],
                _ => vec![(1, 3), (2, 4), (3, 5)],
            };
            let starts: Vec<Date> = windows.iter().map(|(start, _)| dates[start + 1]).collect();
            let ends: Vec<Date> = windows.iter().map(|(_, end)| dates[*end]).collect();
            assert_eq!(text_dates(trades, "train_start"), starts, "{key}");
            assert_eq!(text_dates(trades, "train_end"), ends, "{key}");
            assert!(text_dates(trades, "train_end").iter().zip(trade_dates(trades)).all(|(end, trade)| *end < trade));
        }
        assert_eq!(study.settings.int("rolling_window"), 2);
        assert_eq!(study.settings.int("split"), 3);
    }
}

#[test]
fn equal_weight_refits_match_fixed_rebalancing() {
    let study = run(&history(), daily());
    let fixed = "Fixed rebalance · Equal weight";
    for method in ["Expanding window", "Rolling window"] {
        let key = format!("{method} · Equal weight");
        assert_allclose(&column(&study.equity, &key), &column(&study.equity, fixed), 0.0, 1e-7);
        assert_eq!(study.allocations[&key], study.allocations[fixed]);
        assert_eq!(study.holdings[&key], study.holdings[fixed]);
    }
}

#[test]
fn single_asset_only_entry_cost_and_correct_exposure() {
    let prices = history().select(&["A"]).unwrap();
    let study = run(&prices, BacktestOptions { cost_bps: 25.0, ..daily() });
    let expected = [1.0, 1.0 / 1.0025, 2.0 / 1.0025, 1.0 / 1.0025, 1.0 / 1.0025];
    for key in &study.equity.columns {
        assert_allclose(&column(&study.equity, key), &expected, 0.0, 1e-7);
        approx(holding(&study, key, "average_weight")[0], 0.75);
        approx(holding(&study, key, "end_weight")[0], 1.0);
        approx(holding(&study, key, "selection_frequency")[0], 1.0);
        approx(metric(&study, key, "total_turnover"), 1.0 / 1.0025);
    }
}

#[test]
fn nonpositive_excess_falls_back_without_dropping_curves() {
    let study = run(&history(), BacktestOptions { risk_free_rate: 1000.0, ..daily() });
    for method in METHOD_NAMES {
        let (minimum, maximum) = (format!("{method} · Minimum volatility"), format!("{method} · Maximum Sharpe"));
        assert_allclose(&column(&study.equity, &minimum), &column(&study.equity, &maximum), 0.0, 1e-7);
        assert_eq!(metric(&study, &maximum, "fallback_count"), metric(&study, &maximum, "rebalance_count"));
        assert_eq!(metric(&study, &minimum, "fallback_count"), 0.0);
    }
    assert!(
        study.warnings.iter().any(|w| w.contains("2020-01-07 · Buy and hold: Maximum Sharpe uses Minimum volatility"))
    );
    assert!(!study.warnings.iter().any(|warning| warning.contains("omitted")));
}

#[test]
fn invalid_settings() {
    let cases: [(BacktestOptions, &str); 13] = [
        (BacktestOptions { train_fraction: 1.0, ..half() }, "train_fraction"),
        (BacktestOptions { train_fraction: 0.01, ..half() }, "two training"),
        (BacktestOptions { risk_free_rate: f64::NAN, ..half() }, "risk_free_rate"),
        (BacktestOptions { shrinkage: -0.1, ..half() }, "shrinkage"),
        (BacktestOptions { max_weight: 0.4, ..half() }, "infeasible"),
        (BacktestOptions { max_weight: 1.1, ..half() }, "max_weight"),
        (BacktestOptions { rebalance_every: 0, ..half() }, "rebalance_every"),
        (BacktestOptions { rolling_window: Some(1), ..half() }, "rolling_window"),
        (BacktestOptions { rolling_window: Some(4), ..half() }, "rolling_window"),
        (BacktestOptions { cost_bps: -1.0, ..half() }, "cost_bps"),
        (BacktestOptions { cost_bps: f64::INFINITY, ..half() }, "cost_bps"),
        (BacktestOptions { cost_bps: 10_000.0, ..half() }, "cost_bps"),
        (BacktestOptions { periods_per_year: f64::NAN, ..half() }, "periods_per_year"),
    ];
    for (options, message) in cases {
        let error = run_backtests(&history(), &options).unwrap_err();
        assert!(error.is_value() && error.message().contains(message), "{error} should mention {message}");
    }
}

#[test]
fn invalid_prices() {
    let prices = history();
    let mut duplicate = prices.clone();
    duplicate.columns = vec!["A".into(), "A".into()];
    let order: Vec<usize> = (0..prices.nrows()).rev().collect();
    let reversed = Frame::from_rows(
        order.iter().map(|i| prices.index[*i]).collect(),
        prices.columns.clone(),
        &order.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
    );
    let mut missing = prices.clone();
    missing.values.column_mut(0).fill(f64::NAN);
    let mut zero = prices.clone();
    zero.values.column_mut(0).fill(0.0);
    let cases = [
        (duplicate, "unique asset"),
        (reversed, "increasing"),
        (missing, "finite and strictly positive"),
        (zero, "finite and strictly positive"),
        (prices.rows(0, 0), "nonempty"),
    ];
    for (frame, message) in cases {
        let error = run_backtests(&frame, &BacktestOptions::default()).unwrap_err();
        assert!(error.message().contains(message), "{error} should mention {message}");
    }
}

#[test]
fn periodic_backtest_metrics_use_selected_annualization() {
    let values = vec![100.0, 103.0, 101.0, 100.0, 100.0, 90.0, 108.0, 108.0];
    for (dates, periods_per_year) in
        [(month_starts(date(2020, 1, 1), 8), 12.0), (weeks(date(2020, 1, 1), 8, Weekday::Sun), 52.0)]
    {
        let prices = Frame::from_columns(dates, vec![("A".into(), values.clone())]);
        let study = run(&prices, BacktestOptions { risk_free_rate: 0.12, cost_bps: 0.0, periods_per_year, ..half() });
        let annual_excess = 0.025 * periods_per_year - 0.12;
        let target = 0.12 / periods_per_year;
        let downside = (((0.1 + target).powi(2) + 2.0 * target.powi(2)) * periods_per_year / 4.0).sqrt();
        for key in &study.equity.columns {
            approx(metric(&study, key, "total_return"), 0.08);
            approx(metric(&study, key, "cagr"), 1.08f64.powf(periods_per_year / 4.0) - 1.0);
            approx(metric(&study, key, "volatility"), (0.0475 / 3.0 * periods_per_year).sqrt());
            approx(metric(&study, key, "sortino"), annual_excess / downside);
            approx(metric(&study, key, "calmar"), metric(&study, key, "cagr") / 0.1);
        }
        assert_eq!(study.settings.f64("periods_per_year"), periods_per_year);
        assert_eq!(study.settings.f64("trading_days_per_year"), periods_per_year);
    }
}

#[test]
fn backtest_default_periods_match_explicit_trading_year() {
    let implicit = run(&history(), half());
    let explicit = run(&history(), BacktestOptions { periods_per_year: 252.0, ..half() });
    assert_eq!(implicit.equity, explicit.equity);
    assert_eq!(format!("{:?}", implicit.metrics), format!("{:?}", explicit.metrics));
    assert_eq!(implicit.settings, explicit.settings);
}

#[test]
fn backtest_rejects_invalid_annualization() {
    for periods_per_year in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = run_backtests(&history(), &BacktestOptions { periods_per_year, ..Default::default() }).unwrap_err();
        assert!(error.message().contains("periods_per_year"), "{error}");
    }
}

#[test]
fn backtest_ratios_are_undefined_for_flat_zero_cost_equity() {
    let mut prices = history();
    for i in 4..prices.nrows() {
        prices.values.row_mut(i).fill(100.0);
    }
    let study = run(&prices, BacktestOptions { risk_free_rate: 0.0, cost_bps: 0.0, periods_per_year: 12.0, ..half() });
    for name in ["sharpe", "sortino", "calmar"] {
        assert!(study.metrics.f64s(name).iter().all(|value| value.is_nan()), "{name}");
    }
}

/// `profile_history()`: twenty monthly NumPy returns from `default_rng(304)`.
fn profile_history() -> Frame {
    let returns: Vec<Vec<f64>> =
        test_data("backtest_profile_returns").as_array().unwrap().iter().map(numbers).collect();
    prices_from_columns(&returns, &["A", "B"], month_starts(date(2020, 1, 1), 21))
}

fn profile_settings() -> BacktestOptions {
    BacktestOptions {
        train_fraction: 0.6,
        rolling_window: Some(8),
        rebalance_every: 2,
        risk_free_rate: 0.02,
        periods_per_year: 12.0,
        cost_bps: 25.0,
        ..Default::default()
    }
}

#[test]
fn optional_profiles_preserve_original_strategies_and_reconcile_costs() {
    let prices = profile_history();
    let original = run(&prices, profile_settings());
    let extended = run(&prices, BacktestOptions { include_profiles: true, ..profile_settings() });
    assert_eq!(original.equity.ncols(), 12);
    assert_eq!(extended.equity.ncols(), 24);
    assert!(!original.settings.bool("include_profiles"));
    assert!(extended.settings.bool("include_profiles"));
    assert!(original.settings.get("profile_return_objective").unwrap().is_null());
    assert_eq!(extended.settings.str("profile_return_objective"), "historical_mean");
    assert_eq!(extended.equity.select(&original.equity.columns).unwrap(), original.equity);
    let originals: Vec<usize> =
        original.metrics.index_labels().iter().map(|key| extended.metrics.row_of(key).unwrap()).collect();
    assert_eq!(format!("{:?}", extended.metrics.take_rows(&originals)), format!("{:?}", original.metrics));
    for method in METHOD_NAMES {
        let low = &extended.allocations[&format!("{method} · Low")];
        let minimum = &extended.allocations[&format!("{method} · Minimum volatility")];
        assert_allclose(low.values.as_slice(), minimum.values.as_slice(), 2e-5, 1e-7);
        for profile in ["Low", "Medium", "Extreme"] {
            let key = format!("{method} · {profile}");
            let trades = &extended.trades[&key];
            let charged: Vec<f64> =
                trades.f64s("turnover").iter().zip(trades.f64s("nav_before")).map(|(t, b)| 0.0025 * t * b).collect();
            assert_allclose(&trades.f64s("cost"), &charged, 1e-13, 1e-7);
            let reconciled = holding(&extended, &key, "pnl_contribution").iter().sum::<f64>()
                - trades.f64s("cost").iter().sum::<f64>();
            assert!((metric(&extended, &key, "total_return") - reconciled).abs() <= 1e-13, "{key}");
            assert_eq!(metric(&extended, &key, "fallback_count"), 0.0);
        }
    }
}

#[test]
fn profile_refits_never_observe_execution_day_or_future_prices() {
    let options = BacktestOptions { include_profiles: true, cost_bps: 10.0, ..profile_settings() };
    let prices = profile_history();
    let first = run(&prices, options);
    let second = run(&scaled(&prices, 15, 0, 1.4), options);
    assert_eq!(through(&first.equity, prices.index[14]), through(&second.equity, prices.index[14]));
    for method in METHOD_NAMES {
        for profile in ["Low", "Medium", "Extreme"] {
            let key = format!("{method} · {profile}");
            assert_eq!(
                through(&first.allocations[&key], prices.index[15]),
                through(&second.allocations[&key], prices.index[15]),
                "{key}"
            );
        }
    }
}

#[test]
fn profile_backtests_use_the_prior_mean_frontier_and_same_medium_grid() {
    let steady: Vec<f64> = tile(&[-0.015, 0.015], 9).iter().map(|v| 0.01 + v).collect();
    let transient: Vec<f64> = legacy_common::regime_returns()[1].clone();
    let mut columns = vec![steady.clone(), transient.clone()];
    columns[0].extend([0.1; 6]);
    columns[1].extend([-0.1; 6]);
    let prices = prices_from_columns(&columns, &["Steady", "Transient"], month_starts(date(2020, 1, 1), 25));
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    assert!(mean(&columns[0]) > mean(&columns[1]), "the future reverses the mean ranking");
    let study = run(
        &prices,
        BacktestOptions {
            train_fraction: 0.75,
            shrinkage: 1.0,
            periods_per_year: 12.0,
            include_profiles: true,
            rebalance_every: 2,
            ..Default::default()
        },
    );
    let variance = [efficient_frontier::frame::var(&steady, 1), efficient_frontier::frame::var(&transient, 1)];
    let low = [variance[1] / (variance[0] + variance[1]), variance[0] / (variance[0] + variance[1])];
    let frontier: Vec<[f64; 2]> =
        (0..21).map(|i| i as f64 / 20.0).map(|t| [low[0] - t * low[0], low[1] + t * (1.0 - low[1])]).collect();
    let volatility: Vec<f64> =
        frontier.iter().map(|w| (w[0] * w[0] * variance[0] * 12.0 + w[1] * w[1] * variance[1] * 12.0).sqrt()).collect();
    let midpoint = (volatility[0] + volatility[20]) / 2.0;
    let middle =
        (0..21).fold(
            0,
            |best, i| {
                if (volatility[i] - midpoint).abs() < (volatility[best] - midpoint).abs() { i } else { best }
            },
        );
    for method in METHOD_NAMES {
        assert_allclose(&study.allocations[&format!("{method} · Extreme")].row(0), &[0.0, 1.0], 2e-6, 0.0);
        assert_allclose(&study.allocations[&format!("{method} · Medium")].row(0), &frontier[middle], 2e-6, 0.0);
    }
}

#[test]
fn profile_fit_slices_match_expanding_and_rolling_return_windows() {
    let prices = profile_history();
    let study = run(
        &prices,
        BacktestOptions {
            train_fraction: 0.6,
            rolling_window: Some(8),
            rebalance_every: 2,
            risk_free_rate: 0.03,
            periods_per_year: 12.0,
            max_weight: 0.7,
            shrinkage: 0.2,
            include_profiles: true,
            ..Default::default()
        },
    );
    let dates = &prices.index;
    // Python records these profile fits: prices.iloc[start:end + 1] for each `(start, end)`.
    let expanding = [(0, 12), (0, 14), (0, 16), (0, 18)];
    let rolling = [(4, 12), (6, 14), (8, 16), (10, 18)];
    for (method, windows) in [("Expanding window", expanding), ("Rolling window", rolling)] {
        for profile in ["Low", "Medium", "Extreme"] {
            let trades = &study.trades[&format!("{method} · {profile}")];
            let starts: Vec<Date> = windows.iter().map(|(start, _)| dates[start + 1]).collect();
            let ends: Vec<Date> = windows.iter().map(|(_, end)| dates[*end]).collect();
            assert_eq!(text_dates(trades, "train_start"), starts);
            assert_eq!(text_dates(trades, "train_end"), ends);
        }
    }
    // Every fit has short windows; the profile message appears once.
    assert_eq!(study.warnings.iter().filter(|message| message.contains("less than one year")).count(), 1);
    for trades in study.trades.values() {
        assert!(text_dates(trades, "train_end").iter().zip(trade_dates(trades)).all(|(end, trade)| *end < trade));
    }
}

#[test]
fn profiles_require_six_returns_in_each_fit_window() {
    for options in [
        BacktestOptions { train_fraction: 0.25, include_profiles: true, ..Default::default() },
        BacktestOptions { train_fraction: 0.6, rolling_window: Some(5), include_profiles: true, ..Default::default() },
    ] {
        let error = run_backtests(&profile_history(), &options).unwrap_err();
        assert!(error.message().contains("at least six training returns"), "{error}");
    }
}

#[test]
fn profiles_accept_six_initial_training_returns() {
    let study = run(
        &profile_history().rows(0, 11),
        BacktestOptions {
            train_fraction: 0.6,
            rolling_window: Some(6),
            periods_per_year: 12.0,
            include_profiles: true,
            ..Default::default()
        },
    );
    assert_eq!(study.settings.int("split"), 6);
    assert_eq!(study.equity.ncols(), 24);
}
