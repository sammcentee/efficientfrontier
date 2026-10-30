//! The legacy research tools match the Python release.
//!
//! Golden files come from `tests/fixtures/golden/generate_legacy.py`. Both
//! versions solve the same convex programs with Clarabel, so weights agree to
//! solver tolerance, and paths and metrics follow from the same accounting.
//! Each check prints its largest numeric difference (`--nocapture` shows them).

mod common;

use common::{assert_close, dict_json, fixture, frame_json, table_json};
use efficient_frontier::backtest::{BacktestOptions, run_backtests};
use efficient_frontier::benchmarks::compare_benchmarks;
use efficient_frontier::core::{AnalyzeOptions, Portfolio, analyze};
use efficient_frontier::data::demo_prices;
use efficient_frontier::evidence::{EvidenceOptions, analyze_evidence};
use efficient_frontier::frame::{Date, Frame, iso};
use efficient_frontier::profiles::{ProfileOptions, build_profiles};
use indexmap::IndexMap;
use serde_json::{Value, json};

/// Weights and estimates agree to solver tolerance.
const TOLERANCE: f64 = 2e-6;

/// The largest difference of each named comparison in one case.
#[derive(Default)]
struct Differences(Vec<(String, f64)>);

impl Differences {
    fn check(&mut self, label: &str, actual: &Value, expected: &Value, tolerance: f64) {
        let largest = assert_close(actual, expected, tolerance, label);
        self.0.push((label.to_string(), largest));
    }

    fn report(&self, case: &str) {
        let (label, largest) = self
            .0
            .iter()
            .fold(("none", 0.0), |best, (label, value)| if *value > best.1 { (label.as_str(), *value) } else { best });
        eprintln!("{case}: largest difference {largest:e} ({label})");
    }
}

fn load_prices(case: &Value) -> Frame {
    let name = case["prices"].as_str().unwrap();
    let mut frame = if name == "demo" { demo_prices() } else { common::prices(&format!("{name}.csv")) };
    if let Some(rows) = case.get("rows").and_then(Value::as_u64) {
        frame = frame.rows(0, rows as usize);
    }
    if let Some(columns) = case.get("columns").and_then(Value::as_u64) {
        frame = frame.select_positions(&(0..columns as usize).collect::<Vec<_>>());
    }
    frame
}

fn number(options: &Value, key: &str, fallback: f64) -> f64 {
    options.get(key).and_then(Value::as_f64).unwrap_or(fallback)
}

fn profile_options(options: &Value) -> ProfileOptions {
    let default = ProfileOptions::default();
    ProfileOptions {
        risk_free_rate: number(options, "risk_free_rate", default.risk_free_rate),
        max_weight: number(options, "max_weight", default.max_weight),
        shrinkage: number(options, "shrinkage", default.shrinkage),
        periods_per_year: number(options, "periods_per_year", default.periods_per_year),
        frontier_points: options
            .get("frontier_points")
            .and_then(Value::as_u64)
            .map_or(default.frontier_points, |v| v as usize),
        return_objective: options
            .get("return_objective")
            .and_then(Value::as_str)
            .map_or(default.return_objective, |text| text.parse().unwrap()),
    }
}

fn backtest_options(options: &Value) -> BacktestOptions {
    let default = BacktestOptions::default();
    BacktestOptions {
        train_fraction: number(options, "train_fraction", default.train_fraction),
        risk_free_rate: number(options, "risk_free_rate", default.risk_free_rate),
        max_weight: number(options, "max_weight", default.max_weight),
        shrinkage: number(options, "shrinkage", default.shrinkage),
        rebalance_every: options
            .get("rebalance_every")
            .and_then(Value::as_u64)
            .map_or(default.rebalance_every, |v| v as usize),
        rolling_window: options.get("rolling_window").and_then(Value::as_u64).map(|v| v as usize),
        cost_bps: number(options, "cost_bps", default.cost_bps),
        periods_per_year: number(options, "periods_per_year", default.periods_per_year),
        include_profiles: options.get("include_profiles").and_then(Value::as_bool).unwrap_or(false),
    }
}

/// Date frames in golden files keep pandas index names; Rust frames have none.
fn strip_names(value: &Value) -> Value {
    let mut value = value.clone();
    value.as_object_mut().unwrap().remove("index_names");
    value
}

/// A date frame from its golden split form.
fn frame_from_json(value: &Value) -> Frame {
    let index: Vec<Date> =
        value["index"].as_array().unwrap().iter().map(|day| day.as_str().unwrap().parse().unwrap()).collect();
    let columns: Vec<String> =
        value["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let rows: Vec<Vec<f64>> = value["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
        .collect();
    Frame::from_rows(index, columns, &rows)
}

fn portfolios_json(portfolios: &IndexMap<String, Portfolio>) -> Value {
    let map: serde_json::Map<String, Value> = portfolios
        .iter()
        .map(|(name, p)| {
            let sharpe = if p.sharpe.is_nan() { Value::Null } else { json!(p.sharpe) };
            (
                name.clone(),
                json!({"weights": p.weights.values, "labels": p.weights.labels, "expected_return": p.expected_return,
                       "volatility": p.volatility, "sharpe": sharpe}),
            )
        })
        .collect();
    Value::Object(map)
}

fn case_name(kind: &str, number: usize, case: &Value) -> String {
    format!("{kind} {number} ({} {})", case["prices"].as_str().unwrap_or("market"), case["options"])
}

#[test]
fn profiles_match_python() {
    let golden = fixture("legacy_profiles.json");
    for (number, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let name = case_name("profiles", number, case);
        let result = build_profiles(&load_prices(case), &profile_options(&case["options"]))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let expected = &case["result"];
        let mut differences = Differences::default();
        differences.check("portfolios", &portfolios_json(&result.portfolios), &expected["portfolios"], TOLERANCE);
        differences.check("frontier", &table_json(&result.frontier), &expected["frontier"], TOLERANCE);
        differences.check(
            "frontier weights",
            &table_json(&result.frontier_weights),
            &expected["frontier_weights"],
            TOLERANCE,
        );
        differences.check("windows", &table_json(&result.windows), &expected["windows"], 0.0);
        differences.check(
            "window returns",
            &table_json(&result.window_returns),
            &expected["window_returns"],
            TOLERANCE,
        );
        differences.check("summary", &table_json(&result.summary), &expected["summary"], TOLERANCE);
        differences.check("settings", &dict_json(&result.settings), &expected["settings"], 0.0);
        assert_eq!(result.as_of, expected["as_of"].as_str().unwrap(), "{name}: as_of");
        assert_eq!(json!(result.observations), expected["observations"], "{name}: observations");
        assert_eq!(json!(result.warnings), expected["warnings"], "{name}: warnings");
        differences.report(&name);
    }
}

#[test]
fn backtests_match_python() {
    let golden = fixture("legacy_backtests.json");
    for (number, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let name = case_name("backtests", number, case);
        let study = run_backtests(&load_prices(case), &backtest_options(&case["options"]))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let expected = &case["result"];
        let mut differences = Differences::default();
        differences.check("settings", &dict_json(&study.settings), &expected["settings"], 0.0);
        assert_eq!(json!(study.warnings), expected["warnings"], "{name}: warnings");
        differences.check("equity", &frame_json(&study.equity), &strip_names(&expected["equity"]), TOLERANCE);
        differences.check("metrics", &table_json(&study.metrics), &expected["metrics"], 1e-5);
        let keys: Vec<&String> = study.allocations.keys().collect();
        let golden_keys: Vec<&String> = expected["allocations"].as_object().unwrap().keys().collect();
        assert_eq!(keys.len(), golden_keys.len(), "{name}: path count");
        for key in keys {
            let label = |part: &str| format!("{part} {key}");
            differences.check(
                &label("allocations"),
                &frame_json(&study.allocations[key]),
                &strip_names(&expected["allocations"][key]),
                TOLERANCE,
            );
            differences.check(
                &label("holdings"),
                &table_json(&study.holdings[key]),
                &expected["holdings"][key],
                TOLERANCE,
            );
            differences.check(&label("trades"), &table_json(&study.trades[key]), &expected["trades"][key], TOLERANCE);
        }
        differences.report(&name);
    }
}

#[test]
fn benchmarks_match_python() {
    let golden = fixture("legacy_benchmarks.json");
    let assets = common::prices("legacy_market_prices.csv");
    let benchmark_prices = common::prices("legacy_market_benchmark_prices.csv");
    let settings = &golden["settings"];
    let risk_free_rate = settings["risk_free_rate"].as_f64().unwrap();
    let analysis = analyze(
        &assets,
        &AnalyzeOptions {
            train_fraction: settings["train_fraction"].as_f64().unwrap(),
            risk_free_rate,
            max_weight: settings["max_weight"].as_f64().unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let mut study_options = backtest_options(&golden["backtest_options"]);
    study_options.train_fraction = settings["train_fraction"].as_f64().unwrap();
    study_options.risk_free_rate = risk_free_rate;
    study_options.max_weight = settings["max_weight"].as_f64().unwrap();
    let study = run_backtests(&assets, &study_options).unwrap();
    let latest = build_profiles(&assets, &profile_options(&golden["profile_options"])).unwrap();
    for (label, study, latest) in [("full", Some(&study), Some(&latest)), ("plain", None, None)] {
        let result = compare_benchmarks(&benchmark_prices, &assets, &analysis, study, latest, risk_free_rate).unwrap();
        let expected = &golden[label];
        let mut differences = Differences::default();
        differences.check("prices", &frame_json(&result.prices), &strip_names(&expected["prices"]), 0.0);
        differences.check(
            "holdout equity",
            &frame_json(&result.holdout_equity),
            &strip_names(&expected["holdout_equity"]),
            1e-12,
        );
        differences.check("holdout metrics", &table_json(&result.holdout_metrics), &expected["holdout_metrics"], 1e-10);
        match &result.backtest_equity {
            Some(equity) => {
                differences.check(
                    "backtest equity",
                    &frame_json(equity),
                    &strip_names(&expected["backtest_equity"]),
                    1e-12,
                );
                differences.check(
                    "backtest metrics",
                    &table_json(result.backtest_metrics.as_ref().unwrap()),
                    &expected["backtest_metrics"],
                    1e-10,
                );
            }
            None => {
                assert!(expected["backtest_equity"].is_null() && expected["backtest_metrics"].is_null());
                assert!(result.backtest_metrics.is_none());
            }
        }
        differences.check(
            "training estimates",
            &table_json(&result.training_estimates),
            &expected["training_estimates"],
            1e-12,
        );
        differences.check(
            "latest estimates",
            &table_json(&result.latest_estimates),
            &expected["latest_estimates"],
            1e-12,
        );
        differences.check("settings", &dict_json(&result.settings), &expected["settings"], 0.0);
        differences.report(&format!("benchmarks {label}"));
    }
}

#[test]
fn evidence_matches_python() {
    let golden = fixture("legacy_evidence.json");
    for (number, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let options = &case["options"];
        let name = format!("evidence {number} ({options})");
        let result = analyze_evidence(
            &frame_from_json(&case["strategy_equity"]),
            &frame_from_json(&case["benchmark_equity"]),
            &EvidenceOptions {
                risk_free_rate: options["risk_free_rate"].as_f64().unwrap(),
                periods_per_year: options["periods_per_year"].as_f64().unwrap(),
                delayed_entry: options["delayed_entry"].as_bool().unwrap(),
            },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut differences = Differences::default();
        differences.check("summary", &table_json(&result.summary), &case["summary"], 1e-9);
        differences.check("windows", &table_json(&result.windows), &case["windows"], 1e-12);
        let relative = &result.relative_equity;
        let actual = json!({
            "index": relative.index.iter().map(|date| iso(*date)).collect::<Vec<_>>(),
            "columns": relative.columns.iter().map(|(s, b)| vec![s.clone(), b.clone()]).collect::<Vec<_>>(),
            "data": (0..relative.index.len()).map(|i| relative.values.row(i).iter().copied().collect::<Vec<_>>()).collect::<Vec<_>>(),
        });
        differences.check("relative equity", &actual, &case["relative_equity"], 1e-12);
        differences.check("settings", &dict_json(&result.settings), &case["settings"], 0.0);
        assert_eq!(json!(result.warnings), case["warnings"], "{name}: warnings");
        differences.report(&name);
    }
}
