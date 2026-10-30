//! Behavior of the selection report, ported from `tests/test_selection_report.py`
//! and the report tests of `tests/test_out_of_sample.py`.

mod common;

use common::table_json;

use std::io::Read as _;

use chrono::NaiveDate;
use efficient_frontier::backtest::BacktestStudy;
use efficient_frontier::core::Portfolio;
use efficient_frontier::dict;
use efficient_frontier::frame::{Cell, Date, Frame, Series, Table, cells};
use efficient_frontier::out_of_sample::{
    OutOfSampleStudy, portfolio_comparison, run_default_out_of_sample, sample_values,
};
use efficient_frontier::pyfmt::{Dict, Json, escape, pct};
use efficient_frontier::selection::{
    BASELINE, SELECTED, SelectionOptions, SelectionStudy, run_selection, selection_fit_estimates, selection_frontier,
};
use efficient_frontier::selection_report::*;
use indexmap::IndexMap;

const NAMES: [&str; 4] = ["Historical-mean portfolio", "Equal-weight stocks", "SPY", "QQQ"];
const SPY: &str = "S&P 500 (SPY)";
const QQQ: &str = "Nasdaq-100 (QQQ)";

fn day(text: &str) -> Date {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
}

/// `pd.bdate_range(start, periods=count)`.
fn business_days(start: &str, count: usize) -> Vec<Date> {
    let mut dates = Vec::new();
    let mut date = day(start);
    while dates.len() < count {
        if chrono::Datelike::weekday(&date).number_from_monday() <= 5 {
            dates.push(date);
        }
        date = date.succ_opt().unwrap();
    }
    dates
}

/// `numpy.linspace(start, stop, count)`.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let step = (stop - start) / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count).map(|i| i as f64 * step + start).collect();
    values[count - 1] = stop;
    values
}

fn phase(dates: &[Date], start: usize, offset: f64) -> BacktestStudy {
    let phase_dates = dates[start..start + 4].to_vec();
    let series = |base: f64| (0..4).map(|i| base + offset + i as f64 / 100.0).collect::<Vec<f64>>();
    let total = series(0.02);
    let metrics = Table::labelled(None, &NAMES)
        .with("annual_mean", cells(&series(0.031234567)))
        .with("cagr", cells(&series(0.021234567)))
        .with("volatility", cells(&[0.12; 4]))
        .with("max_drawdown", cells(&[-0.025; 4]))
        .with("total_return", cells(&total))
        .with("total_cost", cells(&[0.001234567; 4]))
        .with("sharpe", cells(&[f64::NAN; 4]))
        .with("fallback_count", cells(&[0_i64; 4]));
    let equity = Frame::from_columns(
        phase_dates.clone(),
        NAMES.iter().zip(&total).map(|(name, total)| (name.to_string(), vec![1.0, 0.999, 0.98, 1.0 + total])).collect(),
    );
    let allocation =
        Frame::from_rows(vec![phase_dates[1]], vec!["AAA".into(), "BBB".into(), "CASH".into()], &[vec![0.6, 0.4, 0.0]]);
    let holdings = Table::labelled(None, &["AAA", "BBB", "CASH"])
        .with("average_weight", cells(&[0.45, 0.25, 0.3]))
        .with("end_weight", cells(&[0.62, 0.28, 0.1]))
        .with("selection_frequency", cells(&[1.0, 1.0, 1.0]))
        .with("max_target_weight", cells(&[0.6, 0.3, 0.1]))
        .with("max_realized_weight", cells(&[0.62, 0.31, 1.0]))
        .with("pnl_contribution", cells(&[0.025, -0.004, 0.0]));
    let trades = Table::new(None, vec![Cell::Date(phase_dates[1])])
        .with("train_end", vec![Cell::Date(phase_dates[0])])
        .with("turnover", cells(&[1.0]))
        .with("cost", cells(&[0.001234567]))
        .with("cash_weight", cells(&[0.0]));
    let each =
        |value: &dyn Fn() -> Table| NAMES.iter().map(|name| (name.to_string(), value())).collect::<IndexMap<_, _>>();
    BacktestStudy {
        equity,
        metrics,
        allocations: NAMES.iter().map(|name| (name.to_string(), allocation.clone())).collect(),
        holdings: each(&|| holdings.clone()),
        trades: each(&|| trades.clone()),
        warnings: vec![format!("Phase starts at {}.", phase_dates[0])],
        settings: dict! {"cost_bps" => 10.0, "periods_per_year" => 252, "target_volatility" => 0.15},
    }
}

/// The `selection_inputs` fixture: a hand-made study, its prices and metadata.
fn inputs() -> (SelectionStudy, Frame, Dict) {
    let dates = business_days("2024-01-02", 12);
    let prices = Frame::from_columns(
        dates.clone(),
        vec![("AAA".into(), linspace(100.0, 112.0, 12)), ("BBB".into(), linspace(80.0, 85.0, 12))],
    );
    let benchmarks = Frame::from_columns(
        dates.clone(),
        vec![("SPY".into(), linspace(400.0, 430.0, 12)), ("QQQ".into(), linspace(350.0, 380.0, 12))],
    );
    let latest = Portfolio {
        name: SELECTED.into(),
        weights: Series::new(vec!["AAA".into(), "BBB".into()], vec![0.6, 0.4]),
        expected_return: 0.123456789,
        volatility: 0.14,
        sharpe: 0.738976,
    };
    let iso = |i: usize| dates[i].to_string();
    let study = SelectionStudy {
        test: phase(&dates, 8, 0.01),
        latest: [(SELECTED.to_string(), latest)].into_iter().collect(),
        selected: SELECTED.into(),
        supported: false,
        reasons: vec!["The later test did not clear the descriptive screen.".into()],
        settings: dict! {
            "target_volatility" => 0.15, "max_weight" => 0.6, "cost_bps" => 10.0, "periods_per_year" => 252,
            "model_revision" => "single_mean_v1", "train_fraction" => 0.7, "shrinkage" => 0.1,
            "latest_as_of" => iso(11), "risk_free_rate" => 0.02, "initial_training_start" => iso(1),
            "initial_training_end" => iso(8), "test_start" => iso(9), "holding_policy" => "buy_and_hold",
        },
        warnings: vec!["Current constituents can introduce survivorship bias.".into()],
        benchmark_prices: Some(benchmarks),
        eligibility: Table::range(2)
            .with("fit_date", cells(&[iso(11), iso(11)]))
            .with("symbol", cells(&["AAA", "BBB"]))
            .with("eligible", cells(&[true, true]))
            .with("reason", cells(&["", ""])),
    };
    let metadata = dict! {
        "source" => "Synthetic adjusted prices", "universe" => "Fixture stocks", "revision" => "fixture-123",
        "universe_coverage" => Json::List(vec![
            Json::Dict(dict! {"ticker" => "AAA", "status" => "included", "reason" => ""}),
            Json::Dict(dict! {"ticker" => "OLD", "status" => "excluded", "reason" => "Missing earlier prices"}),
        ]),
    };
    (study, prices, metadata)
}

fn report(study: &SelectionStudy, prices: &Frame, metadata: &Dict) -> String {
    selection_report_html(study, prices, metadata, None, None).unwrap()
}

/// The text files of a ZIP export, in member order.
fn unzip(bytes: &[u8]) -> IndexMap<String, String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let mut files = IndexMap::new();
    for i in 0..archive.len() {
        let mut member = archive.by_index(i).unwrap();
        let mut text = String::new();
        member.read_to_string(&mut text).unwrap();
        files.insert(member.name().unwrap().to_string(), text);
    }
    files
}

fn export(study: &SelectionStudy, prices: &Frame, metadata: &Dict) -> IndexMap<String, String> {
    unzip(&selection_report_zip(study, prices, metadata, None, None).unwrap())
}

/// The text between `start` and the next `end` after it.
fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    let rest = text.split_once(start).unwrap_or_else(|| panic!("{start} is missing")).1;
    rest.split_once(end).map_or(rest, |(inside, _)| inside)
}

fn set_metric(study: &mut BacktestStudy, row: &str, values: &[(&str, f64)]) {
    let i = study.metrics.row_of(row).unwrap();
    for (column, value) in values {
        let j = study.metrics.position(column).unwrap();
        study.metrics.data[j][i] = Cell::Float(*value);
    }
}

fn keep_rows(study: &mut BacktestStudy, rows: &[&str]) {
    let positions: Vec<usize> = rows.iter().map(|row| study.metrics.row_of(row).unwrap()).collect();
    study.metrics = study.metrics.take_rows(&positions);
}

#[test]
fn chart_identifies_every_path_without_changing_returns() {
    let (study, _, _) = inputs();
    let figure = selection_chart(&study.test);
    let names: Vec<&str> = figure.data.iter().map(|trace| trace["name"].as_str().unwrap()).collect();
    assert_eq!(names, [REPLAY_LABEL, "Equal-weight stocks", "SPY", "QQQ"]);
    assert!(figure.data.iter().all(|trace| trace["showlegend"] == true));
    assert_ne!(figure.data[0]["line"]["color"], figure.data[1]["line"]["color"]);
    assert_eq!(figure.data[1]["line"]["dash"], "dash");
    for (j, trace) in figure.data.iter().enumerate() {
        let expected: Vec<f64> = study.test.equity.col(j).iter().map(|value| value * 10_000.0).collect();
        let actual: Vec<f64> = trace["y"].as_array().unwrap().iter().map(|value| value.as_f64().unwrap()).collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn zip_preserves_every_test_strategy_and_numeric_result() {
    let (study, prices, metadata) = inputs();
    let files = export(&study, &prices, &metadata);
    let details: serde_json::Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    let result_files = details["result_files"].as_object().unwrap();
    assert_eq!(result_files.len(), study.test.equity.ncols());
    let names: Vec<&str> = result_files.values().map(|value| value.as_str().unwrap()).collect();
    assert_eq!(names, NAMES);
    assert_eq!(files["test/metrics.csv"], study.test.metrics.to_csv(Some("strategy")));
    assert_eq!(files["test/equity.csv"], study.test.equity.to_csv("date"));
    for (number, name) in NAMES.iter().enumerate() {
        let stem = format!("test/{:02}", number + 1);
        assert!(result_files.contains_key(&stem));
        assert_eq!(files[&format!("{stem}_allocations.csv")], study.test.allocations[*name].to_csv("date"));
        assert_eq!(files[&format!("{stem}_holdings.csv")], study.test.holdings[*name].to_csv(Some("symbol")));
        assert_eq!(files[&format!("{stem}_trades.csv")], study.test.trades[*name].to_csv(Some("date")));
    }
    assert_eq!(files["prices.csv"], prices.to_csv("Date"));
    assert_eq!(files["benchmark_prices.csv"], study.benchmark_prices.as_ref().unwrap().to_csv("Date"));
    assert_eq!(
        files["universe_coverage.csv"],
        ",ticker,status,reason\r\n0,AAA,included,\r\n1,OLD,excluded,Missing earlier prices\r\n"
    );
    assert_eq!(files["eligibility.csv"], study.eligibility.to_csv(None));
    assert_eq!(files["fit_estimates.csv"], selection_fit_estimates(&study, &prices).to_csv(None));
}

#[test]
fn comparison_precedes_frontier_and_keeps_latest_fit_separate_from_replay() {
    let (study, prices, metadata) = inputs();
    let document = report(&study, &prices, &metadata);
    let section = between(&document, "id=\"fit-and-replay\"", "</section>");
    assert!(document.find("id=\"fit-and-replay\"") < document.find("id=\"latest-frontier\""));
    assert!(document.find("id=\"latest-frontier\"") < document.find("id=\"test\""));
    assert!(!section.contains("<details"));
    assert!(section.contains("The latest fit includes the replay dates"));
    assert!(section.contains("The replay did not use the latest holdings"));
    assert!(section.contains("2024-01-03 to 2024-01-17") && section.contains("2024-01-15 to 2024-01-17"));
    assert!(!section.contains("2024-01-12"), "the NAV baseline is not an evaluated return date");
    let latest = &study.latest[SELECTED];
    for value in [
        latest.expected_return,
        latest.volatility,
        study.test.metrics.f64_at(SELECTED, "annual_mean"),
        study.test.metrics.f64_at(SELECTED, "volatility"),
    ] {
        assert!(section.contains(&pct(value, 2)));
    }
    assert!(section.contains("before fees") && section.contains("after fees"));
    assert!(document.find("id=\"test\"") < document.find("<details id=\"fit-estimates\""));
    let section = between(&document, "id=\"fit-estimates\"", "</details>");
    let fits = selection_fit_estimates(&study, &prices);
    for i in 0..fits.nrows() {
        let label = format!("{} · {}", fits.column("phase")[i].py_str(), fits.index[0][i].py_str());
        assert!(section.contains(&label), "{label}");
        for column in ["fitted_mean", "model_volatility", "historical_volatility"] {
            assert!(section.contains(&pct(fits.f64s(column)[i], 2)));
        }
    }
    assert!(section.contains("Fitted annual mean") && section.contains("Historical volatility"));
    assert!(section.contains("The budget applies to model volatility"));
    assert!(section.contains("latest weights did not produce the earlier test results"));
    assert!(
        document.contains("not checks of optimizer accuracy") && document.contains("Estimates for every saved fit")
    );
    assert!(section.contains("2024-01-12") && section.contains("2024-01-17"));
}

#[test]
fn benchmark_and_fit_explanations_use_the_actual_model_settings() {
    let (mut study, prices, _) = inputs();
    study.settings.insert("shrinkage", 0.2);
    let note = selection_benchmark_note(&study);
    for phrase in [
        "at entry, then lets the weights drift",
        "All four paths buy once and hold",
        "SPY tracks the S&P 500",
        "QQQ tracks the Nasdaq-100",
        "None is scaled to the model volatility budget",
    ] {
        assert!(note.contains(phrase), "{phrase}");
    }
    let summary = selection_fit_summary_html(&study, &prices);
    assert!(summary.contains("20% covariance adjustment"));
    assert!(summary.contains("does not maximize the return-to-volatility ratio or CAGR"));
}

#[test]
fn replay_verdict_uses_mean_volatility_dominance_not_cagr() {
    let cases = [
        (0.3347, 0.1852, true),
        (0.2827, 0.18, true),
        (0.30, 0.1922, true),
        (0.2827, 0.1922, false),
        (0.26, 0.12, false),
        (0.34, 0.22, false),
    ];
    for (mean, volatility, dominates) in cases {
        let mut study = inputs().0.test;
        keep_rows(&mut study, &[SELECTED, "QQQ"]);
        set_metric(&mut study, SELECTED, &[("annual_mean", 0.2827), ("volatility", 0.1922), ("cagr", 0.3023)]);
        // CAGR points the opposite way, so it cannot control the verdict.
        let cagr = if dominates { 0.10 } else { 0.50 };
        set_metric(&mut study, "QQQ", &[("annual_mean", mean), ("volatility", volatility), ("cagr", cagr)]);
        let verdict = selection_replay_verdict_html(&study);
        assert_eq!(!verdict.is_empty(), dominates, "{mean} {volatility}");
        if dominates {
            assert!(verdict.contains("QQQ dominated the stock rule"));
            assert!(verdict.contains(&format!("{} versus 28.27%", pct(mean, 2))));
            assert!(verdict.contains(&format!("{} versus 19.22%", pct(volatility, 2))));
        }
    }
}

#[test]
fn replay_dominance_uses_unrounded_mean_and_volatility() {
    let up = |value: f64| f64::from_bits(value.to_bits() + 1);
    let down = |value: f64| f64::from_bits(value.to_bits() - 1);
    let cases = [
        (0.30, 0.18, "reference_dominates"),
        (0.30, 0.20, "reference_dominates"),
        (0.28, 0.18, "reference_dominates"),
        (0.26, 0.22, "portfolio_dominates"),
        (0.26, 0.20, "portfolio_dominates"),
        (0.28, 0.22, "portfolio_dominates"),
        (0.28, 0.20, "equal"),
        (0.30, 0.22, "tradeoff"),
        (0.26, 0.18, "tradeoff"),
        (up(0.28), 0.20, "reference_dominates"),
        (0.28, down(0.20), "reference_dominates"),
        (down(0.28), 0.20, "portfolio_dominates"),
        (up(0.28), up(0.20), "tradeoff"),
    ];
    for (mean, volatility, classification) in cases {
        let mut study = inputs().0.test;
        keep_rows(&mut study, &[SELECTED, "QQQ"]);
        let values = [("annual_mean", 0.28), ("volatility", 0.20), ("cagr", -0.50), ("total_return", -0.40)];
        set_metric(&mut study, SELECTED, &values);
        let values = [("annual_mean", mean), ("volatility", volatility), ("cagr", 0.60), ("total_return", 0.70)];
        set_metric(&mut study, "QQQ", &values);
        study.settings.insert("periods_per_year", 52);
        let before = study.metrics.clone();
        let comparisons = selection_replay_dominance(&study);
        assert_eq!(comparisons.columns, ["annual_mean", "volatility", "classification"]);
        assert_eq!(comparisons.f64s("annual_mean"), before.f64s("annual_mean"));
        assert_eq!(comparisons.f64s("volatility"), before.f64s("volatility"));
        assert_eq!(comparisons.cell(SELECTED, "classification"), &Cell::from("equal"));
        assert_eq!(comparisons.cell("QQQ", "classification"), &Cell::from(classification), "{mean} {volatility}");
        assert_eq!(table_json(&study.metrics), table_json(&before));
    }
}

#[test]
fn replay_dominance_keeps_each_available_baseline_without_inventing_missing_etfs() {
    let mut study = inputs().0.test;
    keep_rows(&mut study, &["Equal-weight stocks", SELECTED, "SPY"]);
    set_metric(&mut study, SELECTED, &[("annual_mean", 0.28), ("volatility", 0.20)]);
    set_metric(&mut study, "Equal-weight stocks", &[("annual_mean", 0.26), ("volatility", 0.22)]);
    set_metric(&mut study, "SPY", &[("annual_mean", 0.30), ("volatility", 0.18)]);
    let comparisons = selection_replay_dominance(&study);
    assert_eq!(comparisons.index, study.metrics.index);
    assert_eq!(comparisons.column("classification"), cells(&["portfolio_dominates", "equal", "reference_dominates"]));
}

/// Prices from returns, with business-day dates from 2020-01-01.
fn prices_from_returns(columns: &[Vec<f64>], names: &[&str]) -> Frame {
    let count = columns[0].len();
    let mut values: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, returns) in names.iter().zip(columns) {
        let mut path = vec![100.0];
        let mut growth = 1.0;
        for value in returns {
            growth *= 1.0 + value;
            path.push(100.0 * growth);
        }
        values.push((name.to_string(), path));
    }
    Frame::from_columns(business_days("2020-01-01", count + 1), values)
}

#[test]
fn return_bridge_reconciles_real_buy_and_hold_with_fees_and_drift() {
    let noise: Vec<f64> = (0..60).map(|i| if i % 2 == 0 { -0.001 } else { 0.001 }).collect();
    let a: Vec<f64> = (0..60).map(|i| if i < 42 { 0.012 } else { -0.02 } + noise[i]).collect();
    let b: Vec<f64> = (0..60).map(|i| if i < 42 { 0.004 } else { 0.016 } - noise[i]).collect();
    let c: Vec<f64> = (0..60).map(|i| 0.001 + noise[i] / 4.0).collect();
    let prices = prices_from_returns(&[a, b, c], &["A", "B", "C"]);
    for cost_bps in [0.0, 10.0, 500.0] {
        for periods in [12.0, 252.0] {
            let options = SelectionOptions {
                target_volatility: 0.8,
                max_weight: 0.6,
                cost_bps,
                periods_per_year: periods,
                ..Default::default()
            };
            let mut study = run_selection(&prices, None, &options).unwrap();
            let original = (prices.clone(), study.test.metrics.clone());
            let bridge = selection_replay_return_bridge(&study, &prices);
            let parts = bridge.mean_change + bridge.weight_drift + bridge.entry_delay + bridge.entry_fee;
            assert!((bridge.fitted_mean + parts - bridge.realized_mean).abs() < 1e-10);
            let values = study.test.equity.column(SELECTED).unwrap();
            let n = (values.len() - 1) as f64;
            let returns: Vec<f64> = values.windows(2).map(|pair| pair[1] / pair[0] - 1.0).collect();
            let realized = returns.iter().sum::<f64>() / n * periods;
            assert!((bridge.realized_mean - realized).abs() < 1e-12);
            assert!((bridge.entry_fee - (values[1] / values[0] - 1.0) * periods / n).abs() < 1e-12);
            // Independently value fixed units, rather than reconstructing daily weights.
            let initial = study.test.allocations[SELECTED].row(0)[..3].to_vec();
            let first = study.test.equity.index[0];
            let start = prices.index.iter().position(|date| *date == first).unwrap();
            let replay: Vec<Vec<f64>> = (start..prices.nrows()).map(|i| prices.row(i)).collect();
            let constant: Vec<f64> = replay
                .windows(2)
                .map(|pair| (0..3).map(|j| (pair[1][j] / pair[0][j] - 1.0) * initial[j]).sum::<f64>())
                .collect();
            assert!((bridge.entry_delay + constant[0] * periods / n).abs() < 1e-12);
            let units: Vec<f64> = (0..3).map(|j| initial[j] / replay[1][j]).collect();
            let held: Vec<f64> = replay[1..].iter().map(|row| (0..3).map(|j| units[j] * row[j]).sum::<f64>()).collect();
            let drift: f64 = held
                .windows(2)
                .zip(&constant[1..])
                .map(|(pair, constant)| pair[1] / pair[0] - 1.0 - constant)
                .sum::<f64>();
            assert!(bridge.weight_drift.abs() > 1e-5);
            assert!((bridge.weight_drift - drift * periods / n).abs() < 1e-12);
            assert!(bridge.mean_change < 0.0);
            let fitted = selection_fit_estimates(&study, &prices).f64s("fitted_mean")[0];
            assert!((bridge.fitted_mean - fitted).abs() < 1e-12);
            assert_eq!((prices.clone(), study.test.metrics.clone()), original);
            // A separate all-history allocation must not affect the initial replay bridge.
            let weights = &mut study.latest.get_mut(SELECTED).unwrap().weights;
            weights.values = vec![1.0 / 3.0; 3];
            assert_eq!(selection_replay_return_bridge(&study, &prices), bridge);
        }
    }
}

fn renamed_benchmarks(mut study: BacktestStudy) -> BacktestStudy {
    study.metrics.rename_index(&[("SPY", SPY), ("QQQ", QQQ)]);
    study
}

#[test]
fn replay_findings_compare_total_return_and_risk_without_annualizing_again() {
    let cases = [
        (0.6, 0.18, "20.00 percentage points higher", "2.00 percentage points lower"),
        (0.2, 0.24, "20.00 percentage points lower", "4.00 percentage points higher"),
        (0.4, 0.20, "the same", "the same"),
        (-0.1, 0.20, "50.00 percentage points lower", "the same"),
        (0.400001, 0.199999, "less than 0.01 percentage point higher", "less than 0.01 percentage point lower"),
    ];
    for (total_return, volatility, return_finding, risk_finding) in cases {
        let mut study = renamed_benchmarks(inputs().0.test);
        let values =
            [("total_return", total_return), ("volatility", volatility), ("annual_mean", -0.9), ("cagr", -0.8)];
        set_metric(&mut study, SELECTED, &values);
        set_metric(&mut study, SPY, &[("total_return", 0.4), ("volatility", 0.2), ("annual_mean", 0.9), ("cagr", 0.8)]);
        let before = study.metrics.clone();
        let findings = selection_replay_findings_html(&study);
        let spy = between(&findings, "Versus unscaled SPY", "</p>");
        assert!(spy.contains(&format!("net total return was {return_finding}")), "{spy}");
        assert!(spy.contains(&format!("realized annual volatility {risk_finding}")), "{spy}");
        assert!(spy.contains("in this replay"));
        assert!(!findings.contains("annual mean") && !findings.contains("CAGR"));
        assert_eq!(table_json(&study.metrics), table_json(&before));
    }
}

#[test]
fn replay_findings_compare_each_available_etf_and_name_missing_comparisons() {
    for available in [&[][..], &["SPY"][..], &["QQQ"][..], &["SPY", "QQQ"][..]] {
        let mut study = renamed_benchmarks(inputs().0.test);
        let mut rows = vec![SELECTED, "Equal-weight stocks"];
        rows.extend(available.iter().map(|symbol| if *symbol == "SPY" { SPY } else { QQQ }));
        keep_rows(&mut study, &rows);
        set_metric(&mut study, SELECTED, &[("total_return", 0.5), ("volatility", 0.20)]);
        for symbol in available {
            let (name, values) = if *symbol == "SPY" { (SPY, [0.4, 0.18]) } else { (QQQ, [0.7, 0.24]) };
            set_metric(&mut study, name, &[("total_return", values[0]), ("volatility", values[1])]);
        }
        let findings = selection_replay_findings_html(&study);
        for symbol in ["SPY", "QQQ"] {
            assert_eq!(findings.contains(&format!("Versus unscaled {symbol}")), available.contains(&symbol));
        }
        if available.contains(&"SPY") {
            assert!(findings.contains("net total return was 10.00 percentage points higher"));
            assert!(findings.contains("realized annual volatility 2.00 percentage points higher"));
        }
        if available.contains(&"QQQ") {
            assert!(findings.contains("net total return was 20.00 percentage points lower"));
            assert!(findings.contains("realized annual volatility 4.00 percentage points lower"));
        }
        let missing: Vec<&str> = ["SPY", "QQQ"].into_iter().filter(|symbol| !available.contains(symbol)).collect();
        assert_eq!(findings.contains("unavailable"), !missing.is_empty());
        if !missing.is_empty() {
            assert!(findings.contains(&format!("{} comparison", missing.join(" and "))));
        }
        assert!(!findings.contains("Equal-weight"));
    }
}

#[test]
fn replay_verdict_escapes_reference_name() {
    let mut study = inputs().0.test;
    let unsafe_name = "<img src=x onerror=\"reference()\">";
    study.metrics.rename_index(&[("QQQ", unsafe_name)]);
    let verdict = selection_replay_verdict_html(&study);
    assert!(!verdict.contains(unsafe_name) && verdict.contains(&escape(unsafe_name)));
}

#[test]
fn zip_retains_model_source_test_reasons_and_exact_latest_estimates() {
    let (study, prices, metadata) = inputs();
    let files = export(&study, &prices, &metadata);
    assert!(!files.contains_key("policy.json") && !files.keys().any(|name| name.starts_with("validation/")));
    let details: serde_json::Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    assert!(details.get("policy").is_none() && details.get("validation").is_none());
    assert_eq!(details["settings"], common::dict_json(&study.settings));
    assert_eq!(details["settings"]["model_revision"], "single_mean_v1");
    assert_eq!(details["supported"], false);
    assert_eq!(details["reasons"], serde_json::json!(study.reasons));
    let text = &files["metadata.json"];
    for value in ["Synthetic adjusted prices", "fixture-123", study.warnings[0].as_str()] {
        assert!(text.contains(value));
    }
    assert_eq!(files["latest_weights.csv"], format!("symbol,{SELECTED}\r\nAAA,0.6\r\nBBB,0.4\r\n"));
    assert_eq!(
        files["latest_estimates.csv"],
        format!("strategy,expected_return,volatility,sharpe\r\n{SELECTED},0.123456789,0.14,0.738976\r\n")
    );
}

#[test]
fn report_calls_preserve_all_input_objects() {
    let (study, prices, metadata) = inputs();
    let saved = (study.clone(), prices.clone(), metadata.clone());
    report(&study, &prices, &metadata);
    export(&study, &prices, &metadata);
    assert_eq!(saved.1, prices);
    assert_eq!(saved.2, metadata);
    assert_eq!(saved.0.test.equity, study.test.equity);
    assert_eq!(table_json(&saved.0.test.metrics), table_json(&study.test.metrics));
    assert_eq!(saved.0.test.allocations, study.test.allocations);
    assert_eq!(saved.0.test.holdings, study.test.holdings);
    assert_eq!(saved.0.test.trades, study.test.trades);
    assert_eq!(saved.0.latest, study.latest);
    assert_eq!((saved.0.settings, saved.0.reasons, saved.0.warnings), (study.settings, study.reasons, study.warnings));
}

#[test]
fn result_tables_keep_backend_order_and_show_net_mean_separately_from_cagr() {
    let (study, prices, metadata) = inputs();
    let document = report(&study, &prices, &metadata);
    let section = between(&document, "id=\"test\"", "</section>");
    let table = between(section, "<table", "</table>");
    let positions: Vec<usize> = NAMES
        .iter()
        .map(|name| {
            table.find(&format!("<th>{}</th>", escape(if *name == SELECTED { REPLAY_LABEL } else { name }))).unwrap()
        })
        .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    for key in ["annual_mean", "cagr", "volatility", "max_drawdown"] {
        assert!(table.contains(&pct(study.test.metrics.f64_at(SELECTED, key), 2)));
    }
    assert!(document.contains("Historical replay · buy and hold") && document.contains("no later rebalances"));
    assert!(!document.contains("refitted every"));
    assert!(document.contains("Historical data reviewed earlier"));
    assert!(document.to_lowercase().contains("descriptive checks"));
    assert!(document.contains(&study.reasons[0]));
    assert!(document.find("id=\"latest-selection\"") < document.find("id=\"test\""));
    let body = document.split_once("</head>").unwrap().1.to_lowercase();
    for old_choice in ["blend", "risk-matched", "id=\"validation\"", "policy.json"] {
        assert!(!body.contains(old_choice));
    }
}

#[test]
fn latest_holdings_stay_visible_before_later_checks() {
    for supported in [false, true] {
        let (mut study, prices, metadata) = inputs();
        study.supported = supported;
        let document = report(&study, &prices, &metadata);
        let section = between(&document, "id=\"latest-selection\"", "</section>");
        assert!(!section.contains("<details"));
        assert!(section.contains("AAA") && section.contains("BBB"));
        assert!(section.contains("60.00%") && section.contains("40.00%"));
        assert!(document.find("id=\"latest-frontier\"") < document.find("id=\"latest-selection\""));
        assert!(document.find("id=\"latest-selection\"") < document.find("id=\"selection-decision\""));
    }
}

#[test]
fn report_discloses_actual_test_assumptions() {
    let (mut study, prices, metadata) = inputs();
    study
        .settings
        .update(&dict! {"train_fraction" => 0.6, "cost_bps" => 25.0, "shrinkage" => 0.2, "periods_per_year" => 52.0});
    let document = report(&study, &prices, &metadata);
    let section = between(&document, "id=\"selection-assumptions\"", "</section>");
    for value in [
        "Initial training share: 60%",
        "Holding policy: buy once and hold",
        "25 basis points (0.25%)",
        "Covariance shrinkage: 20%",
        "52 observations per year",
    ] {
        assert!(section.contains(value), "{value}");
    }
}

#[test]
fn report_is_offline_and_includes_plotly_license_once() {
    let (study, prices, metadata) = inputs();
    let files = export(&study, &prices, &metadata);
    let document = &files["report.html"];
    assert!(!document.contains("<script src") && !document.contains("<link"));
    assert!(document.to_lowercase().starts_with("<!doctype html>"));
    assert_eq!(document.matches("* plotly.js v").count(), 1);
    assert_eq!(document.matches("class=\"plotly-graph-div\"").count(), 2);
    let license_line = "Permission is hereby granted, free of charge";
    assert!(document.contains(license_line) && files["THIRD_PARTY_NOTICES.txt"].contains(license_line));
}

#[test]
fn report_escapes_source_gate_reasons_warnings_and_asset_labels() {
    let (mut study, prices, mut metadata) = inputs();
    metadata.insert("source", "<script>alert(\"source\")</script>");
    study.reasons = vec!["<img src=x onerror=\"reason()\">".into()];
    study.warnings = vec!["<script>alert(\"warning\")</script>".into()];
    let unsafe_asset = "<img src=x onerror=\"asset()\">";
    study.latest.get_mut(SELECTED).unwrap().weights.labels[0] = unsafe_asset.into();
    let document = report(&study, &prices, &metadata);
    for unsafe_text in [metadata.str("source"), study.reasons[0].as_str(), study.warnings[0].as_str(), unsafe_asset] {
        assert!(!document.contains(unsafe_text));
        assert!(document.contains(escape(unsafe_text).split("&quot;").next().unwrap()));
    }
}

#[test]
fn coverage_csv_escapes_formulas_and_preserves_numeric_losses() {
    let (study, prices, mut metadata) = inputs();
    let mut records = metadata.get("universe_coverage").unwrap().as_list().unwrap().to_vec();
    if let Json::Dict(record) = &mut records[1] {
        record.insert("reason", "=HYPERLINK(\"https://example.invalid\")");
    }
    metadata.insert("universe_coverage", Json::List(records));
    let files = export(&study, &prices, &metadata);
    assert!(files["universe_coverage.csv"].contains("\"'=HYPERLINK(\"\"https://example.invalid\"\")\""));
    assert!(files["test/metrics.csv"].contains(",-0.025,"));
    assert!(files["test/01_holdings.csv"].contains("BBB,0.25,0.28,1.0,0.3,0.31,-0.004"));
}

#[test]
fn latest_frontier_chart_uses_fitted_means_and_matching_volatilities() {
    let (mut study, prices, _) = inputs();
    study.settings.insert("shrinkage", 0.0);
    let frontier = selection_frontier(&study, &prices, 5).unwrap();
    let figure = selection_frontier_chart(&frontier, &study);
    let numbers = |value: &serde_json::Value| -> Vec<f64> {
        value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    };
    assert_eq!(numbers(&figure.data[0]["x"]), frontier.frontier.f64s("model_volatility"));
    assert_eq!(numbers(&figure.data[0]["y"]), frontier.frontier.f64s("fitted_mean"));
    let comparisons = &frontier.comparisons;
    for (i, (trace, name)) in figure.data[1..].iter().zip(comparisons.index_labels()).enumerate() {
        assert_eq!(numbers(&trace["x"]), [comparisons.f64s("historical_volatility")[i]]);
        assert_eq!(numbers(&trace["y"]), [comparisons.f64s("fitted_mean")[i]]);
        assert_eq!(trace["marker"]["symbol"], if name == SELECTED { "circle" } else { "diamond" });
        assert!(trace["hovertemplate"].as_str().unwrap().contains("Annual arithmetic mean"));
    }
    let layout = figure.full_layout();
    assert_eq!(layout["xaxis"]["title"]["text"], "Annual volatility");
    assert_eq!(layout["yaxis"]["title"]["text"], "Annual arithmetic mean");
    assert_eq!(layout["shapes"][0]["x0"], 0.15);
    let summary = selection_frontier_summary_html(&study, &prices, &frontier);
    for phrase in [
        "same return dates",
        "ordinary sample covariance",
        "before trading fees",
        "in-sample optimum",
        LATEST_LABEL,
        "not portfolio candidates",
        "None of the ETF positions is scaled",
        "2024-01-03 to 2024-01-17",
    ] {
        assert!(summary.contains(phrase), "{phrase}");
    }
    assert!(!summary.contains("Model volatility"));
    for i in 0..comparisons.nrows() {
        assert!(summary.contains(&pct(comparisons.f64s("fitted_mean")[i], 2)));
        assert!(summary.contains(&pct(comparisons.f64s("historical_volatility")[i], 2)));
    }
}

#[test]
fn frontier_export_reuses_supplied_points_and_preserves_every_weight() {
    let (study, prices, metadata) = inputs();
    let mut frontier = selection_frontier(&study, &prices, 5).unwrap();
    // A changed copy shows that the export writes the supplied points instead of solving again.
    frontier.frontier.data[0][0] = Cell::Float(0.987654321);
    let saved = frontier.clone();
    let files = unzip(&selection_report_zip(&study, &prices, &metadata, Some(&frontier), None).unwrap());
    assert_eq!(files["latest/frontier.csv"], frontier.frontier.to_csv(None));
    assert!(files["latest/frontier.csv"].contains("0.987654321"));
    assert_eq!(files["latest/frontier_weights.csv"], frontier.weights.to_csv(None));
    assert_eq!(files["latest/fitted_comparison.csv"], frontier.comparisons.to_csv(None));
    let document = &files["report.html"];
    assert!(document.find("id=\"latest-frontier\"") < document.find("id=\"test\""));
    assert!(document.find("id=\"test\"") < document.find("id=\"selection-decision\""));
    assert!(document.contains("Stock model volatility uses 10% covariance shrinkage"));
    assert_eq!(frontier, saved);
}

#[test]
fn frontier_summary_escapes_reference_names_and_solver_warnings() {
    let (study, prices, _) = inputs();
    let mut frontier = selection_frontier(&study, &prices, 3).unwrap();
    let unsafe_name = "<img src=x onerror=\"reference()\">";
    frontier.comparisons.rename_index(&[(SPY, unsafe_name)]);
    frontier.warnings = vec!["<script>solver()</script>".into()];
    let document = selection_frontier_summary_html(&study, &prices, &frontier);
    assert!(!document.contains(unsafe_name) && document.contains("&lt;img src=x"));
    assert!(!document.contains("<script>solver()</script>"));
    assert!(document.contains("&lt;script&gt;solver()&lt;/script&gt;"));
    assert_eq!(selection_frontier_chart(&frontier, &study).data[2]["name"], escape(unsafe_name));
}

// ----------------------------------------------------------------------------- walk-forward report pieces

/// The `market` fixture of `tests/test_out_of_sample.py`, saved by the report generator.
fn walk_market() -> (Frame, Frame) {
    (common::prices("reports_walk_prices.csv"), common::prices("reports_walk_benchmarks.csv"))
}

fn walk_study(prices: &Frame, benchmarks: &Frame) -> SelectionStudy {
    let options = SelectionOptions { target_volatility: 0.2, max_weight: 0.15, cost_bps: 25.0, ..Default::default() };
    run_selection(prices, Some(benchmarks), &options).unwrap()
}

/// Mark window `number` as failed, as the release does when its fit raises.
fn fail_window(result: &mut OutOfSampleStudy, number: i64, reason: &str) {
    let row = result.windows.index[0].iter().position(|cell| *cell == Cell::Int(number)).unwrap();
    let status = result.windows.position("status").unwrap();
    result.windows.data[status][row] = Cell::from(reason);
    let keep = |table: &Table, cells: &[Cell]| -> Table {
        table.take_rows(&(0..table.nrows()).filter(|i| cells[*i] != Cell::Int(number)).collect::<Vec<_>>())
    };
    result.metrics = keep(&result.metrics, &result.metrics.index[0].clone());
    result.weights = keep(&result.weights, &result.weights.index[0].clone());
    if result.eligibility.position("window").is_some() {
        let mut eligibility = keep(&result.eligibility, result.eligibility.column("window"));
        eligibility.index = vec![(0..eligibility.nrows() as i64).map(Cell::Int).collect()];
        result.eligibility = eligibility;
    }
    result.notes.retain(|note| !note.starts_with(&format!("Window {number}: ")));
}

#[test]
fn comparison_frame_and_findings_use_completed_windows() {
    let (prices, benchmarks) = walk_market();
    let study = walk_study(&prices, &benchmarks);
    let result = run_default_out_of_sample(&study, &prices).unwrap();
    let means = portfolio_comparison(&result, "annual_mean");
    assert_eq!(means.columns[..3], ["in_sample", "out_of_sample", "change"]);
    let findings = out_of_sample_findings(&result).unwrap();
    assert_eq!(findings.completed, 5);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (inside, outside) = (means.f64s("in_sample"), means.f64s("out_of_sample"));
    assert!((findings.in_sample - mean(&inside)).abs() < 1e-12);
    assert!((findings.out_of_sample - mean(&outside)).abs() < 1e-12);
    assert_eq!(findings.lower, means.f64s("change").iter().filter(|value| **value < 0.0).count());
    let equal_in = sample_values(&result, "in_sample", "annual_mean").f64s(BASELINE);
    let margin: Vec<f64> = inside.iter().zip(&equal_in).map(|(a, b)| a - b).collect();
    assert!((findings.margin_in_sample - mean(&margin)).abs() < 1e-12);
    let names: Vec<&str> = findings.higher_than.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, [BASELINE, SPY, QQQ]);
    let (headline, lines) = out_of_sample_text(&findings);
    assert!(headline.contains(&percent(mean(&outside), 1)) && headline.contains(&percent(mean(&inside), 1)));
    assert!(lines[0].contains(&format!("in {} of 5 windows", findings.lower)));
    assert!(lines[1].contains("average margin over them"));
    assert!(lines[2].contains(", and Nasdaq-100 (QQQ) in"));
    let frame = out_of_sample_frame(&result);
    assert_eq!(frame.index[0], cells(&[1_i64, 2, 3, 4, 5]));
    assert_eq!(
        frame.columns,
        ["Test dates", "In sample", "Out of sample", "Change", "Equal weight in sample", "Equal weight out of sample"]
    );
    assert_eq!(frame.f64s("Equal weight in sample"), equal_in);
    let paths = out_of_sample_paths_frame(&result);
    assert_eq!(paths.nrows(), 5 * 4);
    assert_eq!(paths.index[1][..4], cells(&["Stock portfolio", "Equal-weight stocks", SPY, QQQ]));
    let stock: Vec<f64> = (0..paths.nrows())
        .filter(|i| paths.index[1][*i] == Cell::from("Stock portfolio"))
        .map(|i| paths.f64s("Change")[i])
        .collect();
    assert_eq!(stock, means.f64s("change"));
}

#[test]
fn chart_hover_names_the_dates_each_marker_measures() {
    let (prices, benchmarks) = walk_market();
    let result = run_default_out_of_sample(&walk_study(&prices, &benchmarks), &prices).unwrap();
    let figure = out_of_sample_chart(&result);
    let trace = |name: &str| figure.data.iter().find(|trace| trace["name"] == name).unwrap().clone();
    let (inside, outside) = (trace("Stock portfolio · in sample"), trace("Stock portfolio · out of sample"));
    assert!(inside["hovertemplate"].as_str().unwrap().contains("fit dates"));
    assert!(outside["hovertemplate"].as_str().unwrap().contains("test dates"));
    let date = |column: &str| result.windows.column(column)[0].py_str();
    assert_eq!(inside["customdata"][0][1], format!("{} to {}", date("fit_start"), date("fit_end")));
    assert_eq!(outside["customdata"][0][1], format!("{} to {}", date("test_start"), date("test_end")));
}

#[test]
fn reference_sentence_lists_one_or_two_paths_without_a_serial_comma() {
    let cases: [(Vec<(&str, usize)>, &str); 2] = [
        (vec![(BASELINE, 2)], "than equal-weight stocks in 2 of 5."),
        (vec![(BASELINE, 2), (QQQ, 1)], "than equal-weight stocks in 2 of 5 and Nasdaq-100 (QQQ) in 1 of 5."),
    ];
    for (references, expected) in cases {
        let findings = OutOfSampleFindings {
            completed: 5,
            windows: 5,
            in_sample: 0.1,
            out_of_sample: -0.02,
            change: -0.12,
            lower: 4,
            margin_in_sample: 0.03,
            margin_out_of_sample: -0.01,
            higher_than: references.iter().map(|(name, wins)| (name.to_string(), *wins)).collect(),
            test_returns: 125,
        };
        let (headline, lines) = out_of_sample_text(&findings);
        assert!(headline.contains("−2.0%"));
        assert!(lines.last().unwrap().ends_with(expected));
        assert!(lines[1].contains("+3.00 pp a year in sample and −1.00 pp out of sample"));
    }
}

#[test]
fn infeasible_window_keeps_its_dates_and_reason() {
    let (prices, benchmarks) = walk_market();
    let mut result = run_default_out_of_sample(&walk_study(&prices, &benchmarks), &prices).unwrap();
    fail_window(&mut result, 1, "requested annual volatility 1.00% is below the feasible minimum");
    assert_eq!(portfolio_comparison(&result, "annual_mean").index[0], cells(&[2_i64, 3, 4, 5]));
    assert_eq!(out_of_sample_findings(&result).unwrap().completed, 4);
    assert!(out_of_sample_html(Ok(&result)).contains("Window 1 has no result. requested annual volatility"));
}

#[test]
fn all_windows_infeasible_report_no_result_and_export_only_the_windows() {
    let (prices, benchmarks) = walk_market();
    let study = walk_study(&prices, &benchmarks);
    let mut result = run_default_out_of_sample(&study, &prices).unwrap();
    for number in 1..=5 {
        fail_window(&mut result, number, "The 15.0% position cap is infeasible");
    }
    assert!(out_of_sample_findings(&result).is_none());
    assert!(portfolio_comparison(&result, "annual_mean").nrows() == 0);
    assert!(out_of_sample_html(Ok(&result)).contains("No window had a feasible fit"));
    let metadata = dict! {"source" => "Synthetic"};
    let files = unzip(&selection_report_zip(&study, &prices, &metadata, None, Some(&result)).unwrap());
    let names: Vec<&String> = files.keys().filter(|name| name.starts_with("out_of_sample/")).collect();
    assert_eq!(names, ["out_of_sample/windows.csv"]);
    assert_eq!(files["out_of_sample/windows.csv"].matches("position cap is infeasible").count(), 5);
    assert!(files["report.html"].contains("the reason it has no result"));
}

#[test]
fn walk_forward_fit_notes_are_kept_in_the_report() {
    let (mut prices, benchmarks) = walk_market();
    let study = walk_study(&prices, &benchmarks);
    let first = prices.get(0, 0);
    for i in 0..40 {
        prices.values[(i, 0)] = first;
    }
    let result = run_default_out_of_sample(&study, &prices).unwrap();
    assert!(!result.notes.is_empty() && result.notes.iter().all(|note| note.starts_with("Window ")));
    assert!(out_of_sample_html(Ok(&result)).contains("unchanged-price screen excluded S00"));
}

#[test]
fn missing_benchmarks_keep_stock_and_equal_weight_results() {
    let (prices, _) = walk_market();
    let options = SelectionOptions { target_volatility: 0.2, max_weight: 0.15, ..Default::default() };
    let plain = run_selection(&prices, None, &options).unwrap();
    let findings = out_of_sample_findings(&run_default_out_of_sample(&plain, &prices).unwrap()).unwrap();
    assert_eq!(findings.higher_than.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(), [BASELINE]);
}

#[test]
fn report_and_zip_include_walk_forward_results() {
    let (prices, benchmarks) = walk_market();
    let study = walk_study(&prices, &benchmarks);
    let metadata = dict! {"source" => "Synthetic adjusted prices", "universe" => "Fixture stocks"};
    let result = run_default_out_of_sample(&study, &prices).unwrap();
    let files = unzip(&selection_report_zip(&study, &prices, &metadata, None, None).unwrap());
    for name in ["windows", "metrics", "weights", "eligibility"] {
        assert!(files.contains_key(&format!("out_of_sample/{name}.csv")));
    }
    assert_eq!(files["out_of_sample/metrics.csv"], result.metrics.to_csv(None));
    let details: serde_json::Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    assert_eq!(details["out_of_sample"]["windows"], 5);
    assert_eq!(details["out_of_sample"]["notes"], serde_json::json!(result.notes));
    let document = &files["report.html"];
    let section = between(document, "id=\"out-of-sample\"", "</section>");
    let (headline, _) = out_of_sample_text(&out_of_sample_findings(&result).unwrap());
    assert!(section.contains(&escape(&headline)));
    assert!(document.find("id=\"out-of-sample\"") < document.find("id=\"test\""));
}

#[test]
fn report_explains_a_history_too_short_for_walk_forward() {
    let (prices, benchmarks) = walk_market();
    let (short, short_benchmarks) = (prices.rows(0, 16), benchmarks.rows(0, 16));
    let options = SelectionOptions { target_volatility: 0.5, max_weight: 0.15, ..Default::default() };
    let study = run_selection(&short, Some(&short_benchmarks), &options).unwrap();
    let document = report(&study, &short, &dict! {"source" => "Synthetic", "universe" => "Fixture"});
    assert!(document.contains("at least 3 fit returns and 3 test returns"));
    assert!(document.contains("There are no out_of_sample files"));
    let files = export(&study, &short, &dict! {"source" => "Synthetic"});
    assert!(!files.keys().any(|name| name.starts_with("out_of_sample/")));
    let details: serde_json::Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    assert!(details["out_of_sample"].get("unavailable").is_some());
}
