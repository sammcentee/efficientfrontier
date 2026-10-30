//! Rolling tests with dated membership: the behavior of the Python test suite.
//!
//! The ports of `tests/test_historical.py` use its seeded prices, exported by
//! `tests/fixtures/golden/generate_historical.py`. The last two tests check
//! the library part of `tests/test_historical_interfaces.py`: the CLI and
//! report exports themselves belong to those modules.

mod common;

use common::golden_dir;
use efficient_frontier::frame::{Cell, Date, Frame, Table, iso};
use efficient_frontier::historical::{
    HistoricalOptions, HistoricalStudy, historical_demo, load_historical_prices, load_historical_prices_path,
    run_historical_backtests,
};
use efficient_frontier::membership::{MembershipHistory, bundled_membership, load_membership_csv};
use efficient_frontier::pyfmt::Json;
use efficient_frontier::{Result, dict};

const EQUAL: &str = "Rolling window · Equal weight";
const DEFAULT_CHANGES: [(usize, &str); 3] = [(0, "A,B"), (26, "A,C,MISSING"), (32, "A,B,C")];

/// The seeded 45-day, three-asset prices of the Python tests.
fn prices() -> Frame {
    load_historical_prices_path(&golden_dir().join("historical_test_prices.csv")).unwrap()
}

/// Membership with full rosters that take effect on the given price rows.
fn history(prices: &Frame, changes: &[(usize, &str)]) -> MembershipHistory {
    let mut text = String::from("date,tickers\n");
    for (row, symbols) in changes {
        text.push_str(&format!("{},\"{symbols}\"\n", iso(prices.index[*row])));
    }
    load_membership_csv(text.as_bytes(), iso(prices.index[prices.nrows() - 1]).as_str(), None).unwrap()
}

fn options() -> HistoricalOptions {
    HistoricalOptions {
        train_fraction: 0.5,
        rolling_window: Some(5),
        rebalance_every: 3,
        cost_bps: 10.0,
        ..HistoricalOptions::default()
    }
}

fn run(prices: &Frame, membership: &MembershipHistory, options: HistoricalOptions) -> Result<HistoricalStudy> {
    run_historical_backtests(prices, membership, &options)
}

fn row_of(index: &[Date], date: Date) -> usize {
    index.iter().position(|value| *value == date).unwrap_or_else(|| panic!("{date} is missing"))
}

/// The target weight of `symbol` in `strategy` at the trade on `date`.
fn weight(study: &HistoricalStudy, strategy: &str, date: Date, symbol: &str) -> f64 {
    let allocation = &study.study.allocations[strategy];
    allocation.get(row_of(&allocation.index, date), allocation.position(symbol).unwrap())
}

/// A number from the trade on `date`.
fn trade(study: &HistoricalStudy, strategy: &str, date: Date, column: &str) -> f64 {
    let trades = &study.study.trades[strategy];
    let row = trades.index[0].iter().position(|cell| cell.as_date() == Some(date)).unwrap();
    trades.f64s(column)[row]
}

/// The coverage decisions of one symbol, as `(status, reason)` pairs.
fn decisions(study: &HistoricalStudy, symbol: &str) -> Vec<(String, String)> {
    let coverage = &study.coverage;
    (0..coverage.nrows())
        .filter(|i| coverage.column("symbol")[*i].as_str() == Some(symbol))
        .map(|i| (coverage.column("status")[i].py_str(), coverage.column("reason")[i].py_str()))
        .collect()
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-6 * expected.abs().max(1e-12)
}

#[test]
fn addition_departure_and_reentry_apply_at_trade_date() {
    let prices = prices();
    let dates = &prices.index;
    let result = run(&prices, &history(&prices, &DEFAULT_CHANGES), options()).unwrap();
    assert_eq!(weight(&result, EQUAL, dates[23], "C"), 0.0);
    assert_eq!(weight(&result, EQUAL, dates[26], "C"), 0.5);
    assert_eq!(weight(&result, EQUAL, dates[26], "B"), 0.0);
    assert!(close(weight(&result, EQUAL, dates[32], "B"), 1.0 / 3.0));
    assert!(result.study.allocations[EQUAL].column("MISSING").unwrap().iter().all(|value| *value == 0.0));
    let missing = decisions(&result, "MISSING");
    assert!(!missing.is_empty());
    assert!(missing.iter().all(|(status, reason)| status == "excluded" && reason.contains("No price column")));
    assert!(result.study.warnings.iter().any(|note| note.contains("survivorship bias")));
}

#[test]
fn effective_change_between_trades_sells_at_next_scheduled_trade() {
    let prices = prices();
    let dates = &prices.index;
    let membership = history(&prices, &[(0, "A,B"), (24, "A,C")]);
    let result = run(&prices, &membership, HistoricalOptions { cost_bps: 0.0, ..options() }).unwrap();
    assert_eq!(weight(&result, EQUAL, dates[23], "B"), 0.5);
    assert_eq!(weight(&result, EQUAL, dates[26], "B"), 0.0);
    let expected: f64 = (0..2).map(|j| 0.5 * prices.get(25, j) / prices.get(23, j)).sum();
    let equity = &result.study.equity;
    assert!(close(equity.get(row_of(&equity.index, dates[25]), equity.position(EQUAL).unwrap()), expected));
}

#[test]
fn short_history_becomes_eligible_using_only_prior_window() {
    let mut prices = prices();
    for i in 0..24 {
        prices.values[(i, 2)] = f64::NAN;
    }
    let dates = prices.index.clone();
    let result = run(&prices, &history(&prices, &DEFAULT_CHANGES), options()).unwrap();
    assert_eq!(weight(&result, EQUAL, dates[26], "C"), 0.0);
    assert_eq!(weight(&result, EQUAL, dates[29], "C"), 0.0);
    assert!(close(weight(&result, EQUAL, dates[32], "C"), 1.0 / 3.0));
    assert!(decisions(&result, "C")[0].1.contains("complete prior prices"));
    assert!(result.study.equity.values.iter().all(|value| value.is_finite()));
}

#[test]
fn future_prices_and_future_rosters_do_not_change_prior_targets() {
    let prices = prices();
    let dates = &prices.index;
    let original = run(&prices, &history(&prices, &DEFAULT_CHANGES), options()).unwrap();
    let mut altered = prices.clone();
    for i in 33..altered.nrows() {
        for (j, factor) in [3.0, 0.2, 7.0].into_iter().enumerate() {
            altered.values[(i, j)] *= factor;
        }
    }
    let future_roster = history(&prices, &[(0, "A,B"), (26, "A,C,MISSING"), (32, "A,B,C"), (33, "A,C")]);
    let changed = run(&altered, &future_roster, options()).unwrap();
    for (name, allocation) in &original.study.allocations {
        let other = &changed.study.allocations[name];
        let through = allocation.rows_through(dates[32]);
        assert_eq!(through.index, other.rows_through(dates[32]).index);
        for (a, b) in through.values.iter().zip(other.rows_through(dates[32]).values.iter()) {
            assert!((a - b).abs() <= 1e-10 + 1e-10 * b.abs(), "{name}: {a} vs {b}");
        }
    }
    assert_eq!(original.study.equity.rows_through(dates[32]), changed.study.equity.rows_through(dates[32]));
}

#[test]
fn execution_day_prices_cannot_change_that_days_targets_or_earn_first_return() {
    let prices = prices();
    let membership = history(&prices, &[(0, "A,B,C")]);
    let original = run(&prices, &membership, options()).unwrap();
    let mut altered = prices.clone();
    for (j, factor) in [10.0, 0.5, 2.0].into_iter().enumerate() {
        altered.values[(23, j)] *= factor;
    }
    let changed = run(&altered, &membership, options()).unwrap();
    for (name, allocation) in &original.study.allocations {
        assert_eq!(allocation.row(0), changed.study.allocations[name].row(0), "{name}");
        let equity = &changed.study.equity;
        assert!(close(equity.get(1, equity.position(name).unwrap()), 1.0 / 1.001), "{name}");
        let trades = &changed.study.trades[name];
        for (date, train_end) in trades.index[0].iter().zip(trades.column("train_end")) {
            assert!(train_end.py_str() < iso(date.as_date().unwrap()), "{name}");
        }
    }
}

#[test]
fn future_price_gap_cannot_silently_exclude_a_survivor() {
    let mut prices = prices();
    prices.values[(40, 2)] = f64::NAN;
    let error = run(&prices, &history(&prices, &DEFAULT_CHANGES), options()).unwrap_err();
    assert!(error.message().starts_with("Missing price for held asset") && error.message().contains(": C."), "{error}");
}

#[test]
fn missing_execution_price_fails_instead_of_excluding_selected_new_member() {
    let mut prices = prices();
    prices.values[(26, 2)] = f64::NAN;
    let error = run(&prices, &history(&prices, &DEFAULT_CHANGES), options()).unwrap_err();
    assert!(error.message().starts_with("Missing execution price") && error.message().contains(": C."), "{error}");
}

#[test]
fn missing_departed_price_is_not_a_free_liquidation() {
    let mut prices = prices();
    let membership = history(&prices, &[(0, "A,B"), (24, "A,C")]);
    prices.values[(24, 1)] = f64::NAN;
    let error = run(&prices, &membership, options()).unwrap_err();
    assert!(error.message().starts_with("Missing price for held asset") && error.message().contains(": B."), "{error}");
}

#[test]
fn cash_exit_reentry_fees_and_contributions_reconcile() {
    let prices = prices();
    let dates = &prices.index;
    let membership = history(&prices, &[(0, "A,B"), (26, "A"), (32, "A,B")]);
    let result =
        run(&prices, &membership, HistoricalOptions { max_weight: 0.5, cost_bps: 100.0, ..options() }).unwrap();
    let study = &result.study;
    let cash = &result.cash_weights;
    for name in &study.equity.columns {
        assert!(close(
            trade(&result, name, dates[26], "nav_after"),
            trade(&result, name, dates[26], "nav_before") * 0.99
        ));
        assert!(close(trade(&result, name, dates[26], "turnover"), 1.0));
        assert!(close(
            trade(&result, name, dates[32], "nav_after"),
            trade(&result, name, dates[32], "nav_before") / 1.01
        ));
        let j = cash.position(name).unwrap();
        for i in row_of(&cash.index, dates[26])..=row_of(&cash.index, dates[31]) {
            assert_eq!(cash.get(i, j), 1.0, "{name}");
        }
        let allocation = &study.allocations[name];
        assert_eq!(allocation.row(row_of(&allocation.index, dates[26])).iter().sum::<f64>(), 0.0);
        assert_eq!(cash.get(row_of(&cash.index, dates[32]), j), 0.0);
        let holdings = &study.holdings[name];
        let costs: f64 = study.trades[name].f64s("cost").iter().sum();
        let reconciled = holdings.f64s("pnl_contribution").iter().sum::<f64>() - costs;
        assert!((reconciled - study.metrics.f64_at(name, "total_return")).abs() <= 1e-12, "{name}");
        let end = holdings.f64s("end_weight").iter().sum::<f64>() + cash.get(cash.nrows() - 1, j);
        assert!(close(end, 1.0), "{name}");
    }
    assert!(study.warnings.iter().any(|message| message.contains("cash")));
}

#[test]
fn empty_price_coverage_remains_cash_with_complete_audit() {
    let prices = prices();
    let result = run(&prices, &history(&prices, &[(0, "UNPRICED")]), options()).unwrap();
    assert!(result.study.equity.values.iter().all(|value| *value == 1.0));
    assert!(result.cash_weights.values.iter().all(|value| *value == 1.0));
    assert!(result.coverage.column("status").iter().all(|status| status.as_str() == Some("excluded")));
    assert!(result.study.metrics.f64s("total_cost").iter().all(|cost| *cost == 0.0));
    for weights in result.study.allocations.values() {
        assert!(weights.values.iter().all(|value| *value == 0.0));
    }
}

#[test]
fn one_eligible_member_is_valid_when_cap_is_one() {
    let prices = prices();
    let result = run(&prices, &history(&prices, &[(0, "A")]), options()).unwrap();
    for weights in result.study.allocations.values() {
        assert!(weights.column("A").unwrap().iter().all(|value| (value - 1.0).abs() < 1e-7));
    }
}

/// Python counts the calls of a patched `identity_exclusions`. Here exclusion
/// rules with exclusive ends at the window edges check the same interval.
#[test]
fn identity_hazard_in_training_range_blocks_otherwise_current_safe_symbol() {
    let prices = prices();
    let dates = &prices.index;
    let reason = "Ticker identifies another security before this date.";
    let hazard = |start: Option<Date>, end: Option<Date>| {
        let rule = dict! {"symbol" => "C", "start" => start.map(iso), "end" => end.map(iso), "reason" => reason};
        MembershipHistory::new(
            vec![(dates[0], vec!["A".to_string(), "B".to_string(), "C".to_string()])],
            dates[44],
            "Patched identities",
            vec![],
            dict! {"identity_exclusions" => vec![Json::Dict(rule)]},
        )
        .unwrap()
    };
    // Python: excluded when the window starts on or before row 21.
    let result = run(&prices, &hazard(None, Some(dates[22])), options()).unwrap();
    assert_eq!(weight(&result, EQUAL, dates[23], "C"), 0.0);
    assert!(close(weight(&result, EQUAL, dates[29], "C"), 1.0 / 3.0));
    assert!(decisions(&result, "C")[0].1.contains("another security"));
    // The first trade, on row 23, checks rows 17 through 23.
    for (rule, first) in
        [((None, Some(dates[18])), true), ((Some(dates[23]), None), true), ((None, Some(dates[17])), false)]
    {
        let result = run(&prices, &hazard(rule.0, rule.1), options()).unwrap();
        assert_eq!(decisions(&result, "C")[0].0 == "excluded", first, "{rule:?}");
    }
}

#[test]
fn membership_coverage_cannot_be_extended_by_last_snapshot() {
    let prices = prices();
    let membership = load_membership_csv(b"date,tickers\n2021-01-04,\"A,B\"\n", "2021-02-15", None).unwrap();
    let error = run(&prices, &membership, options()).unwrap_err();
    assert!(error.message().contains("coverage"), "{error}");
}

/// Python also rejects booleans and non-integer counts, which the option types exclude.
#[test]
fn invalid_settings_fail() {
    let prices = prices();
    let membership = history(&prices, &DEFAULT_CHANGES);
    let base = options();
    let cases = [
        ("train_fraction", HistoricalOptions { train_fraction: 1.0, ..base }),
        ("rolling_window", HistoricalOptions { rolling_window: Some(1), ..base }),
        ("rolling_window", HistoricalOptions { rolling_window: Some(23), ..base }),
        ("rebalance_every", HistoricalOptions { rebalance_every: 0, ..base }),
        ("max_weight", HistoricalOptions { max_weight: 0.0, ..base }),
        ("max_weight", HistoricalOptions { max_weight: f64::INFINITY, ..base }),
        ("shrinkage", HistoricalOptions { shrinkage: -1.0, ..base }),
        ("cost_bps", HistoricalOptions { cost_bps: 10_000.0, ..base }),
        ("risk_free_rate", HistoricalOptions { risk_free_rate: f64::NAN, ..base }),
        ("periods_per_year", HistoricalOptions { periods_per_year: 0.0, ..base }),
    ];
    for (key, options) in cases {
        let error = run(&prices, &membership, options).unwrap_err();
        assert!(error.is_value() && error.message().contains(key), "{key}: {error}");
    }
}

#[test]
fn sparse_csv_roundtrip_preserves_missing_rows() {
    let mut prices = prices();
    for i in 0..10 {
        prices.values[(i, 2)] = f64::NAN;
    }
    let result = load_historical_prices(prices.to_csv("Date").as_bytes()).unwrap();
    assert_eq!((&result.index, &result.columns), (&prices.index, &prices.columns));
    for (a, b) in result.values.iter().zip(prices.values.iter()) {
        assert!(a == b || (a.is_nan() && b.is_nan()));
    }
}

fn reconciles(study: &HistoricalStudy) {
    for name in &study.study.equity.columns {
        let contributions: f64 = study.study.holdings[name].f64s("pnl_contribution").iter().sum();
        let metrics = &study.study.metrics;
        let reconciled = contributions - metrics.f64_at(name, "total_cost");
        assert!((reconciled - metrics.f64_at(name, "total_return")).abs() <= 1e-12, "{name}");
    }
}

#[test]
fn demo_exercises_eligibility_changes_and_reconciles() {
    let (prices, membership) = historical_demo();
    let result = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    let weights = &result.study.allocations[EQUAL];
    for symbol in ["NEWCO", "BETA", "GAMMA"] {
        let column = weights.column(symbol).unwrap();
        assert!(column.contains(&0.0) && column.iter().any(|value| *value > 0.0), "{symbol}");
    }
    reconciles(&result);
}

/// The CLI writes prices.csv and membership.csv; reloading them reproduces the study.
#[test]
fn demo_exported_inputs_reproduce_results() {
    let (prices, membership) = historical_demo();
    let options = HistoricalOptions { cost_bps: 7.0, ..HistoricalOptions::default() };
    let study = run_historical_backtests(&prices, &membership, &options).unwrap();
    assert_eq!(study.study.settings.int("rolling_window"), 126);
    assert_eq!(study.study.allocations.len(), 3);
    assert!(prices.values.iter().any(|value| value.is_nan()));
    assert!(study.coverage.column("status").iter().any(|status| status.as_str() == Some("excluded")));
    assert!(study.cash_weights.row(0).iter().all(|value| *value == 1.0));
    reconciles(&study);
    for (name, trades) in &study.study.trades {
        for (date, train_end) in trades.index[0].iter().zip(trades.column("train_end")) {
            assert!(train_end.py_str() < iso(date.as_date().unwrap()), "{name}");
        }
    }
    let snapshots = membership.snapshots();
    let exported = Table::new(Some("date"), snapshots.iter().map(|(date, _)| Cell::Date(*date)).collect())
        .with("tickers", snapshots.iter().map(|(_, roster)| Cell::Str(roster.join(","))).collect())
        .to_csv(None);
    let reloaded =
        load_membership_csv(exported.as_bytes(), iso(membership.coverage_end()).as_str(), Some("CSV: membership.csv"))
            .unwrap();
    let replay_prices = load_historical_prices(prices.to_csv("Date").as_bytes()).unwrap();
    let replay = run_historical_backtests(&replay_prices, &reloaded, &options).unwrap();
    assert_eq!(study.study.equity.to_csv("Date"), replay.study.equity.to_csv("Date"));
    assert_eq!(study.study.metrics.to_csv(None), replay.study.metrics.to_csv(None));
    assert_eq!(study.coverage.to_csv(None), replay.coverage.to_csv(None));
    assert_eq!(study.cash_weights.to_csv("date"), replay.cash_weights.to_csv("date"));
}

#[test]
fn bundled_history_records_missing_prices_and_preserves_provenance() {
    let (prices, _) = historical_demo();
    let mut prices = prices.select_positions(&[0, 1]);
    prices.columns = vec!["AAPL".to_string(), "MSFT".to_string()];
    let study = run_historical_backtests(&prices, &bundled_membership(), &HistoricalOptions::default()).unwrap();
    let membership = study.study.settings.get("membership").and_then(Json::as_dict).unwrap();
    assert!(!membership.str("source_commit").is_empty());
    assert!(membership.get("identity_exclusions").and_then(Json::as_list).is_some_and(|rules| !rules.is_empty()));
    let coverage = &study.coverage;
    let statuses = coverage.column("status");
    let symbols = coverage.column("symbol");
    for i in (0..coverage.nrows()).filter(|i| statuses[*i].as_str() == Some("eligible")) {
        assert!(["AAPL", "MSFT"].contains(&symbols[i].as_str().unwrap()));
    }
    assert!(coverage.column("reason").iter().any(|reason| reason.py_str().contains("No price column")));
}
