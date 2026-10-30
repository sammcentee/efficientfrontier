//! Historical membership studies, membership data and input parsing match the Python release.
//!
//! Golden files come from `tests/fixtures/golden/generate_historical.py`. Both
//! versions solve the same convex programs with Clarabel, so weights agree to
//! solver tolerance, and paths and metrics follow from the same accounting.

mod common;

use common::{assert_close, dict_json, fixture, frame_json, golden_dir, table_json};
use efficient_frontier::frame::{Frame, iso};
use efficient_frontier::historical::{
    HistoricalOptions, HistoricalStudy, historical_demo, load_historical_prices, load_historical_prices_path,
    run_historical_backtests,
};
use efficient_frontier::membership::{MembershipHistory, bundled_membership, load_membership_csv};
use serde_json::{Value, json};

/// Weights, paths and metrics agree far inside solver tolerance. The largest
/// observed difference is about 2e-10, in the bundled membership study.
const TOLERANCE: f64 = 1e-8;

fn options(given: &Value) -> HistoricalOptions {
    let default = HistoricalOptions::default();
    let number = |key: &str, fallback: f64| match &given[key] {
        Value::Null => fallback,
        Value::String(text) if text == "inf" => f64::INFINITY,
        value => value.as_f64().unwrap(),
    };
    let nan =
        |key: &str, fallback: f64| if given.get(key) == Some(&Value::Null) { f64::NAN } else { number(key, fallback) };
    let count = |key: &str, fallback: usize| given.get(key).and_then(Value::as_u64).map_or(fallback, |v| v as usize);
    HistoricalOptions {
        train_fraction: number("train_fraction", default.train_fraction),
        rolling_window: match given.get("rolling_window") {
            None => default.rolling_window,
            Some(Value::Null) => None,
            Some(value) => Some(value.as_u64().unwrap() as usize),
        },
        rebalance_every: count("rebalance_every", default.rebalance_every),
        risk_free_rate: nan("risk_free_rate", default.risk_free_rate),
        max_weight: number("max_weight", default.max_weight),
        shrinkage: number("shrinkage", default.shrinkage),
        cost_bps: number("cost_bps", default.cost_bps),
        periods_per_year: number("periods_per_year", default.periods_per_year),
    }
}

/// Frames in golden files keep pandas index names; date frames here have none.
fn strip_names(value: &Value) -> Value {
    let mut value = value.clone();
    value.as_object_mut().unwrap().remove("index_names");
    value
}

fn sparse_prices(name: &str) -> Frame {
    load_historical_prices_path(&golden_dir().join(name)).expect("golden prices")
}

/// Compare every result of a study and return the largest numeric difference.
fn compare_study(name: &str, study: &HistoricalStudy, expected: &Value) -> f64 {
    let result = &study.study;
    let mut largest: f64 = 0.0;
    largest = largest.max(assert_close(&dict_json(&result.settings), &expected["settings"], 0.0, "settings"));
    assert_eq!(json!(result.settings.keys().collect::<Vec<_>>()), expected["settings_keys"], "{name}: settings order");
    let membership = result.settings.get("membership").and_then(|value| value.as_dict()).unwrap();
    assert_eq!(json!(membership.keys().collect::<Vec<_>>()), expected["membership_keys"], "{name}: membership order");
    assert_eq!(json!(result.warnings), expected["warnings"], "{name}: warnings");
    largest = largest.max(assert_close(&table_json(&study.coverage), &expected["coverage"], 0.0, "coverage"));
    largest =
        largest.max(assert_close(&frame_json(&result.equity), &strip_names(&expected["equity"]), TOLERANCE, "equity"));
    largest = largest.max(assert_close(
        &frame_json(&study.cash_weights),
        &strip_names(&expected["cash_weights"]),
        TOLERANCE,
        "cash weights",
    ));
    largest = largest.max(assert_close(&table_json(&result.metrics), &expected["metrics"], TOLERANCE, "metrics"));
    assert_eq!(result.allocations.len(), 3, "{name}: strategies");
    for (strategy, allocation) in &result.allocations {
        largest = largest.max(assert_close(
            &frame_json(allocation),
            &strip_names(&expected["allocations"][strategy]),
            TOLERANCE,
            &format!("allocations {strategy}"),
        ));
        largest = largest.max(assert_close(
            &table_json(&result.holdings[strategy]),
            &expected["holdings"][strategy],
            TOLERANCE,
            &format!("holdings {strategy}"),
        ));
        largest = largest.max(assert_close(
            &table_json(&result.trades[strategy]),
            &expected["trades"][strategy],
            TOLERANCE,
            &format!("trades {strategy}"),
        ));
    }
    largest
}

fn check_study(name: &str) {
    let golden = fixture(&format!("{name}.json"));
    let spec = &golden["membership"];
    let (prices, membership) = match spec["kind"].as_str().unwrap() {
        "demo" => historical_demo(),
        "bundled" => (sparse_prices(&format!("{name}_prices.csv")), bundled_membership()),
        "csv" => (
            sparse_prices(&format!("{name}_prices.csv")),
            load_membership_csv(
                spec["csv"].as_str().unwrap().as_bytes(),
                spec["coverage_end"].as_str().unwrap(),
                spec["source"].as_str(),
            )
            .unwrap(),
        ),
        other => panic!("unknown membership kind {other}"),
    };
    let study = run_historical_backtests(&prices, &membership, &options(&golden["options"]))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let largest = compare_study(name, &study, &golden["study"]);
    eprintln!("{name}: largest difference {largest:e}");
}

#[test]
fn demo_study_matches_python() {
    check_study("historical_demo");
}

#[test]
fn bundled_membership_study_matches_python() {
    check_study("historical_bundled");
}

#[test]
fn custom_weekly_study_with_cash_and_fallbacks_matches_python() {
    check_study("historical_custom");
}

fn test_prices() -> Frame {
    sparse_prices("historical_test_prices.csv")
}

/// Membership CSV text for `(row, tickers)` changes on the price dates.
fn roster(prices: &Frame, changes: &Value) -> String {
    let mut text = String::from("date,tickers\n");
    for change in changes.as_array().unwrap() {
        let row = change[0].as_u64().unwrap() as usize;
        text.push_str(&format!("{},\"{}\"\n", iso(prices.index[row]), change[1].as_str().unwrap()));
    }
    text
}

#[test]
fn fixed_roster_matches_the_python_rolling_engine() {
    let golden = fixture("historical_fixed.json");
    let prices = test_prices();
    let text = roster(&prices, &golden["changes"]);
    let membership = load_membership_csv(text.as_bytes(), iso(prices.index[44]).as_str(), None).unwrap();
    let study = run_historical_backtests(&prices, &membership, &options(&golden["options"])).unwrap();
    let rolling = &golden["rolling"];
    let mut largest: f64 = 0.0;
    largest = largest.max(assert_close(
        &frame_json(&study.study.equity),
        &strip_names(&rolling["equity"]),
        TOLERANCE,
        "equity",
    ));
    largest = largest.max(assert_close(
        &strip_names(&table_json(&study.study.metrics)),
        &strip_names(&rolling["metrics"]),
        TOLERANCE,
        "metrics",
    ));
    for (strategy, allocation) in &study.study.allocations {
        largest = largest.max(assert_close(
            &frame_json(allocation),
            &strip_names(&rolling["allocations"][strategy]),
            TOLERANCE,
            &format!("allocations {strategy}"),
        ));
        largest = largest.max(assert_close(
            &strip_names(&table_json(&study.study.holdings[strategy])),
            &strip_names(&rolling["holdings"][strategy]),
            TOLERANCE,
            &format!("holdings {strategy}"),
        ));
    }
    eprintln!("historical_fixed: largest difference {largest:e}");
}

#[test]
fn error_messages_match_python() {
    let golden = fixture("historical_errors.json");
    let base = test_prices();
    for case in golden["cases"].as_array().unwrap() {
        let mut prices = base.clone();
        for cell in case["nan"].as_array().unwrap() {
            let (row, column) = (cell[0].as_u64().unwrap() as usize, cell[1].as_u64().unwrap() as usize);
            prices.values[(row, column)] = f64::NAN;
        }
        let text = roster(&prices, &case["changes"]);
        let membership = load_membership_csv(text.as_bytes(), case["coverage_end"].as_str().unwrap(), None).unwrap();
        let error = run_historical_backtests(&prices, &membership, &options(&case["options"])).unwrap_err();
        assert!(error.is_value(), "{case}");
        assert_eq!(error.message(), case["error"].as_str().unwrap(), "{case}");
    }
}

fn snapshots_json(history: &MembershipHistory) -> Value {
    json!(history.snapshots().iter().map(|(date, roster)| json!([iso(*date), roster])).collect::<Vec<_>>())
}

#[test]
fn membership_csv_parsing_matches_pandas() {
    let golden = fixture("historical_parsing.json");
    for case in golden["membership"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        match load_membership_csv(text.as_bytes(), "2020-01-31", None) {
            Ok(history) => assert_eq!(snapshots_json(&history), case["snapshots"], "{text:?}"),
            Err(error) => assert_eq!(json!(error.message()), case["error"], "{text:?}"),
        }
    }
    let text = "date,tickers\n2020-01-01,\"B,A\"\n2020-01-06,\"A,C\"\n2020-01-10,\"B,C\"\n";
    let history = load_membership_csv(text.as_bytes(), "2020-01-12", None).unwrap();
    for case in golden["lookups"].as_array().unwrap() {
        let date = case["date"].as_str().unwrap();
        match history.at(date) {
            Ok(roster) => assert_eq!(json!(roster), case["roster"], "{date:?}"),
            Err(error) => assert_eq!(json!(error.message()), case["error"], "{date:?}"),
        }
    }
    let text = "date,tickers\n2020-01-01,\"B,A\"\n2020-01-10,\"B,C\"\n";
    for case in golden["coverage_ends"].as_array().unwrap() {
        let coverage_end = case["coverage_end"].as_str().unwrap();
        match load_membership_csv(text.as_bytes(), coverage_end, None) {
            Ok(history) => {
                assert_close(&dict_json(&history.metadata()), &case["metadata"], 0.0, coverage_end);
            }
            Err(error) => assert_eq!(json!(error.message()), case["error"], "{coverage_end:?}"),
        }
    }
}

#[test]
fn price_csv_parsing_matches_pandas() {
    let golden = fixture("historical_parsing.json");
    for case in golden["prices"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        match load_historical_prices(text.as_bytes()) {
            Ok(prices) => {
                assert!(case.get("prices").is_some(), "{text:?} loads in Rust but fails in Python: {}", case["error"]);
                assert_close(&frame_json(&prices), &strip_names(&case["prices"]), 0.0, &format!("{text:?}"));
            }
            Err(error) => assert_eq!(json!(error.message()), case["error"], "{text:?}"),
        }
    }
}

#[test]
fn bundled_membership_matches_python() {
    let golden = fixture("historical_membership.json");
    let history = bundled_membership();
    assert_eq!(json!(iso(history.coverage_start())), golden["coverage_start"]);
    assert_eq!(json!(iso(history.coverage_end())), golden["coverage_end"]);
    assert_eq!(json!(history.source()), golden["source"]);
    assert_eq!(json!(history.limitations()), golden["limitations"]);
    assert_eq!(json!(history.symbols()), golden["symbols"]);
    assert_eq!(snapshots_json(&history), golden["snapshots"]);
    let metadata = history.metadata();
    assert_close(&dict_json(&metadata), &golden["metadata"], 0.0, "metadata");
    assert_eq!(json!(metadata.keys().collect::<Vec<_>>()), golden["metadata_keys"]);
    for query in golden["exclusions"].as_array().unwrap() {
        let start = query["start"].as_str().unwrap();
        let exclusions = history.identity_exclusions(start, query["through"].as_str()).unwrap();
        let pairs: Vec<Value> = exclusions.iter().map(|(symbol, reason)| json!([symbol, reason])).collect();
        assert_eq!(json!(pairs), query["result"], "{query}");
    }
}
