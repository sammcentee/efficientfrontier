//! The stock model and the classic analysis match the Python release.
//!
//! Golden files come from `tests/fixtures/golden/generate_selection.py`. Both
//! versions solve the same convex programs with Clarabel, so weights agree to
//! solver tolerance, and paths and metrics follow from the same accounting.

mod common;

use common::{assert_close, dict_json, fixture, frame_json, prices, table_json};
use efficient_frontier::core::{AnalyzeOptions, analyze};
use efficient_frontier::out_of_sample::run_default_out_of_sample;
use efficient_frontier::selection::{SelectionOptions, run_selection, selection_fit_estimates, selection_frontier};
use serde_json::{Value, json};

/// Weights and estimates agree to solver tolerance.
const TOLERANCE: f64 = 2e-6;

fn options(golden: &Value) -> SelectionOptions {
    let given = &golden["options"];
    let default = SelectionOptions::default();
    let number = |key: &str, fallback: f64| given.get(key).and_then(Value::as_f64).unwrap_or(fallback);
    SelectionOptions {
        target_volatility: number("target_volatility", default.target_volatility),
        max_weight: number("max_weight", default.max_weight),
        train_fraction: number("train_fraction", default.train_fraction),
        cost_bps: number("cost_bps", default.cost_bps),
        shrinkage: number("shrinkage", default.shrinkage),
        periods_per_year: number("periods_per_year", default.periods_per_year),
        risk_free_rate: number("risk_free_rate", default.risk_free_rate),
    }
}

fn check_selection(name: &str) {
    let golden = fixture(&format!("{name}.json"));
    let prices = prices(&format!("{name}_prices.csv"));
    let benchmarks = golden["benchmarks"].as_bool().unwrap().then(|| common::prices(&format!("{name}_benchmarks.csv")));
    let result = run_selection(&prices, benchmarks.as_ref(), &options(&golden));
    if let Some(error) = golden.get("error") {
        assert_eq!(result.unwrap_err().message(), error.as_str().unwrap(), "{name}");
        return;
    }
    let study = result.unwrap_or_else(|error| panic!("{name}: {error}"));
    let expected = &golden["selection"];
    let mut largest: f64 = 0.0;
    largest = largest.max(assert_close(&dict_json(&study.settings), &expected["settings"], 0.0, "settings"));
    largest =
        largest.max(assert_close(&dict_json(&study.test.settings), &expected["test_settings"], 0.0, "test settings"));
    assert_eq!(json!(study.supported), expected["supported"], "{name}: supported");
    assert_eq!(json!(study.reasons), expected["reasons"], "{name}: reasons");
    assert_eq!(json!(study.warnings), expected["warnings"], "{name}: warnings");
    assert_eq!(json!(study.test.warnings), expected["test_warnings"], "{name}: test warnings");
    largest = largest.max(assert_close(
        &frame_json(&study.test.equity),
        &strip_names(&expected["equity"]),
        TOLERANCE,
        "equity",
    ));
    largest = largest.max(assert_close(&table_json(&study.test.metrics), &expected["metrics"], 1e-5, "metrics"));
    for (strategy, allocation) in &study.test.allocations {
        let golden_allocation = strip_names(&expected["allocations"][strategy]);
        largest = largest.max(assert_close(
            &frame_json(allocation),
            &golden_allocation,
            TOLERANCE,
            &format!("allocations {strategy}"),
        ));
        largest = largest.max(assert_close(
            &table_json(&study.test.holdings[strategy]),
            &expected["holdings"][strategy],
            TOLERANCE,
            &format!("holdings {strategy}"),
        ));
        largest = largest.max(assert_close(
            &table_json(&study.test.trades[strategy]),
            &expected["trades"][strategy],
            TOLERANCE,
            &format!("trades {strategy}"),
        ));
    }
    let weights = &study.latest[&study.selected];
    let latest: serde_json::Map<String, Value> =
        weights.weights.iter().map(|(label, value)| (label.to_string(), json!(value))).collect();
    largest =
        largest.max(assert_close(&Value::Object(latest), &expected["latest_weights"], TOLERANCE, "latest weights"));
    largest = largest.max(assert_close(
        &json!(weights.expected_return),
        &expected["latest_expected"],
        TOLERANCE,
        "latest expected",
    ));
    largest = largest.max(assert_close(
        &json!(weights.volatility),
        &expected["latest_volatility"],
        TOLERANCE,
        "latest volatility",
    ));
    largest = largest.max(assert_close(&table_json(&study.eligibility), &expected["eligibility"], 0.0, "eligibility"));
    largest = largest.max(assert_close(
        &table_json(&selection_fit_estimates(&study, &prices)),
        &golden["fit_estimates"],
        TOLERANCE,
        "fit estimates",
    ));

    let frontier = selection_frontier(&study, &prices, 30).unwrap();
    largest =
        largest.max(assert_close(&table_json(&frontier.frontier), &golden["frontier"]["frontier"], 1e-5, "frontier"));
    largest = largest.max(assert_close(
        &table_json(&frontier.weights),
        &golden["frontier"]["weights"],
        1e-4,
        "frontier weights",
    ));
    largest = largest.max(assert_close(
        &table_json(&frontier.comparisons),
        &golden["frontier"]["comparisons"],
        TOLERANCE,
        "comparisons",
    ));
    assert_eq!(json!(frontier.warnings), golden["frontier"]["warnings"], "{name}: frontier warnings");

    let expected = &golden["out_of_sample"];
    match run_default_out_of_sample(&study, &prices) {
        Err(error) => assert_eq!(error.message(), expected["error"].as_str().unwrap()),
        Ok(result) => {
            largest = largest.max(assert_close(&table_json(&result.windows), &expected["windows"], 0.0, "windows"));
            largest = largest.max(assert_close(
                &table_json(&result.metrics),
                &expected["metrics"],
                1e-5,
                "walk-forward metrics",
            ));
            // A nearly flat optimum can move a weight by a few 1e-6 while the
            // objective agrees to 1e-10 (selection_market, window 5).
            largest = largest.max(assert_close(
                &table_json(&result.weights),
                &expected["weights"],
                1e-5,
                "walk-forward weights",
            ));
            largest = largest.max(assert_close(
                &table_json(&result.eligibility),
                &expected["eligibility"],
                0.0,
                "walk-forward eligibility",
            ));
            assert_eq!(json!(result.notes), expected["notes"], "{name}: notes");
            largest = largest.max(assert_close(
                &dict_json(&result.settings),
                &expected["settings"],
                0.0,
                "walk-forward settings",
            ));
        }
    }
    eprintln!("{name}: largest difference {largest:e}");
}

/// Frames in golden files keep pandas index names; date frames here have none.
fn strip_names(value: &Value) -> Value {
    let mut value = value.clone();
    value.as_object_mut().unwrap().remove("index_names");
    value
}

#[test]
fn demo_selection_matches_python() {
    check_selection("selection_demo");
}

#[test]
fn market_selection_with_benchmarks_matches_python() {
    check_selection("selection_market");
}

#[test]
fn stale_price_screen_shrinkage_and_costs_match_python() {
    check_selection("selection_stale");
}

#[test]
fn weekly_prices_and_another_split_match_python() {
    check_selection("selection_weekly");
}

#[test]
fn infeasible_budget_message_matches_python() {
    check_selection("selection_infeasible");
}

fn check_analysis(name: &str) {
    let golden = fixture(&format!("{name}.json"));
    let prices = prices(&format!("{name}_prices.csv"));
    let given = &golden["options"];
    let default = AnalyzeOptions::default();
    let options = AnalyzeOptions {
        max_weight: given.get("max_weight").and_then(Value::as_f64).unwrap_or(default.max_weight),
        shrinkage: given.get("shrinkage").and_then(Value::as_f64).unwrap_or(default.shrinkage),
        frontier_points: given
            .get("frontier_points")
            .and_then(Value::as_u64)
            .map_or(default.frontier_points, |v| v as usize),
        ..default
    };
    let analysis = analyze(&prices, &options).unwrap();
    let mut largest: f64 = 0.0;
    let portfolios: serde_json::Map<String, Value> = analysis
        .portfolios
        .iter()
        .map(|(key, p)| {
            let sharpe = if p.sharpe.is_nan() { Value::Null } else { json!(p.sharpe) };
            (key.clone(), json!({"weights": p.weights.values, "expected_return": p.expected_return, "volatility": p.volatility, "sharpe": sharpe}))
        })
        .collect();
    largest = largest.max(assert_close(&Value::Object(portfolios), &golden["portfolios"], 1e-5, "portfolios"));
    largest = largest.max(assert_close(
        &table_json(&analysis.frontier.points_table()),
        &golden["frontier"],
        1e-5,
        "frontier",
    ));
    largest = largest.max(assert_close(
        &table_json(&analysis.frontier.weights_table()),
        &golden["frontier_weights"],
        1e-4,
        "frontier weights",
    ));
    let metrics: Vec<Value> = analysis
        .holdout_metrics
        .values()
        .map(|m| {
            json!(m.fields().iter().map(|(_, v)| if v.is_nan() { Value::Null } else { json!(v) }).collect::<Vec<_>>())
        })
        .collect();
    largest = largest.max(assert_close(&json!(metrics), &golden["holdout_metrics"]["data"], 1e-5, "holdout metrics"));
    assert_eq!(json!(analysis.holdout_metrics.keys().collect::<Vec<_>>()), golden["holdout_metrics"]["index"]);
    largest = largest.max(assert_close(&frame_json(&analysis.equity), &strip_names(&golden["equity"]), 1e-5, "equity"));
    assert_eq!(json!(analysis.warnings), golden["warnings"], "{name}: warnings");
    eprintln!("{name}: largest difference {largest:e}");
}

#[test]
fn demo_classic_analysis_matches_python() {
    check_analysis("analyze_demo");
}

#[test]
fn capped_classic_analysis_matches_python() {
    check_analysis("analyze_capped");
}

#[test]
fn singular_classic_analysis_matches_python() {
    check_analysis("analyze_singular");
}
