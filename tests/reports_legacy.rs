//! The legacy, benchmark and historical reports, ported from `tests/test_backtest_report.py`,
//! `tests/test_benchmark_report.py`, `tests/test_presentation.py` and `tests/test_historical_report.py`.

mod common;

use std::io::Read as _;

use chrono::NaiveDate;
use efficient_frontier::backtest::{BacktestOptions, BacktestStudy, run_backtests};
use efficient_frontier::backtest_report::{ChartOptions, backtest_chart, backtest_html, findings, findings_markdown};
use efficient_frontier::benchmark_report::{
    benchmark_html, benchmark_markdown, default_evidence_strategy, evidence_chart, evidence_summary_frame,
};
use efficient_frontier::benchmarks::{BenchmarkComparison, compare_benchmarks};
use efficient_frontier::core::{Analysis, AnalyzeOptions, Frontier, Portfolio, analyze};
use efficient_frontier::dict;
use efficient_frontier::evidence::{EvidenceAnalysis, EvidenceOptions, analyze_evidence};
use efficient_frontier::frame::{Cell, Date, Frame, LabeledMatrix, Series, Table, cells};
use efficient_frontier::historical::{
    HistoricalOptions, historical_demo, load_historical_prices, run_historical_backtests,
};
use efficient_frontier::historical_report::build_historical_report;
use efficient_frontier::membership::{bundled_membership, load_membership_csv};
use efficient_frontier::metrics::Performance;
use efficient_frontier::presentation::{self, ReportParts, report_html, report_zip};
use efficient_frontier::profiles::{ProfileAnalysis, ProfileOptions, ReturnObjective, build_profiles};
use efficient_frontier::pyfmt::{Dict, Json};
use efficient_frontier::story;
use indexmap::IndexMap;
use nalgebra::DMatrix;
use serde_json::Value;

fn day(text: &str) -> Date {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
}

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

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
}

// ----------------------------------------------------------------------------- backtest report

/// The `study` fixture of `tests/test_backtest_report.py`.
fn study() -> BacktestStudy {
    let names: Vec<String> = ["Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window"]
        .iter()
        .flat_map(|method| {
            ["Minimum volatility", "Maximum Sharpe", "Equal weight"].map(|policy| format!("{method} · {policy}"))
        })
        .collect();
    let mut total: Vec<f64> = (2..14).map(|i| i as f64 / 100.0).collect();
    total[9] = 0.25;
    let mut drawdown = vec![-0.1; 12];
    drawdown[2] = -0.02;
    let metrics = Table::labelled(None, &names)
        .with("total_return", cells(&total))
        .with("cagr", cells(&[0.15; 12]))
        .with("volatility", cells(&[0.1; 12]))
        .with("sharpe", cells(&[1.3; 12]))
        .with("sortino", cells(&[1.75; 12]))
        .with("calmar", cells(&[1.5; 12]))
        .with("max_drawdown", cells(&drawdown))
        .with("total_turnover", cells(&[2.75; 12]))
        .with("total_cost", cells(&[0.02; 12]))
        .with("rebalance_count", cells(&[3_i64; 12]))
        .with("fallback_count", cells(&[0_i64; 12]));
    let dates = vec![day("2024-01-02"), day("2024-01-03"), day("2024-01-04"), day("2024-01-05")];
    let equity = Frame::from_columns(
        dates,
        names.iter().zip(&total).map(|(name, total)| (name.clone(), vec![1.0, 0.99, 0.98, 1.0 + total])).collect(),
    );
    let holdings = Table::labelled(None, &["AAA", "BBB", "CCC"])
        .with("average_weight", cells(&[0.45, 0.25, 0.05]))
        .with("end_weight", cells(&[0.65, 0.25, 0.1]))
        .with("selection_frequency", cells(&[1.0, 1.0, 0.5]))
        .with("max_target_weight", cells(&[0.7, 0.35, 0.15]))
        .with("max_realized_weight", cells(&[0.72, 0.4, 0.2]))
        .with("pnl_contribution", cells(&[0.24, -0.01, 0.04]));
    BacktestStudy {
        equity,
        metrics,
        allocations: IndexMap::new(),
        holdings: names.iter().map(|name| (name.clone(), holdings.clone())).collect(),
        trades: IndexMap::new(),
        warnings: Vec::new(),
        settings: dict! {"cost_bps" => 10, "rebalance_every" => 21, "rolling_window" => 252},
    }
}

const BEST: &str = "Rolling window · Minimum volatility";

fn set_cell(table: &mut Table, row: &str, column: &str, value: Cell) {
    let (i, j) = (table.row_of(row).unwrap(), table.position(column).unwrap());
    table.data[j][i] = value;
}

#[test]
fn findings_use_matching_equal_weight_and_signed_contributions() {
    let study = study();
    let notes = findings(&study);
    assert!(notes[0].contains("Rolling window · Minimum volatility, +25.00%"));
    assert!(notes[0].contains("12.00 percentage points above Rolling window · Equal weight"));
    assert!(notes[1].contains("Buy and hold · Equal weight, -2.00%"));
    assert!(notes[2].contains("AAA, +24.00 percentage points"));
    assert!(notes[3].contains("BBB, -1.00 percentage points"));
    assert!(notes[4].contains("gross turnover 2.75×"));
    assert!(notes[4].contains("2.00% of initial capital (200.00 per 10,000"));
    assert!(notes[5].contains("AAA, 72.00% of portfolio value; its largest target was 70.00%"));
}

#[test]
fn findings_do_not_mislabel_positive_and_negative_assets() {
    let cases = [
        ([-0.02, -0.01, -0.03], "No holding made a positive", "Largest holding detractor"),
        ([0.02, 0.01, 0.03], "No holding made a negative", "Largest positive holding contribution"),
        ([0.0, 0.0, 0.0], "No holding made a positive", "No holding made a negative"),
    ];
    for (contributions, absent, present) in cases {
        let mut study = study();
        study.holdings.get_mut(BEST).unwrap().push("pnl_contribution", cells(&contributions));
        let notes = findings(&study).join("\n");
        assert!(notes.contains(absent) && notes.contains(present));
    }
}

#[test]
fn equal_weight_winner_and_fallbacks_are_explained() {
    let mut study = study();
    set_cell(&mut study.metrics, "Buy and hold · Equal weight", "total_return", Cell::Float(0.5));
    set_cell(&mut study.metrics, "Rolling window · Maximum Sharpe", "fallback_count", Cell::Int(2));
    let notes = findings(&study).join("\n");
    assert!(notes.contains("equal-weight comparison itself") && notes.contains("fallback occurred 2 times"));
}

#[test]
fn solver_dust_is_not_a_named_detractor_or_a_displayed_holding() {
    let mut study = study();
    let holdings = study.holdings.get_mut(BEST).unwrap();
    set_cell(holdings, "BBB", "pnl_contribution", Cell::Float(-1e-9));
    set_cell(holdings, "BBB", "max_target_weight", Cell::Float(1e-8));
    let saved = holdings.clone();
    let notes = findings(&study).join("\n");
    assert!(notes.contains("No holding made a negative P&L contribution at the displayed precision"));
    assert!(!notes.contains("Largest holding detractor") && !notes.contains("-0.00 percentage points"));
    assert!(!findings_markdown(&study, "Synthetic demo", None, None).contains("| BBB |"));
    assert!(!backtest_html(&study, None, None, None).contains("<th>BBB</th>"));
    assert_eq!(study.holdings[BEST], saved, "presentation does not remove assets from exported study data");
}

#[test]
fn charts_preserve_all_paths_and_initial_drawdown() {
    let study = study();
    let comparison = backtest_chart(&study, &ChartOptions::default()).unwrap();
    let drawdown = backtest_chart(&study, &ChartOptions { drawdown: true, ..Default::default() }).unwrap();
    assert_eq!((comparison.data.len(), drawdown.data.len()), (12, 12));
    let expected: Vec<f64> = study.equity.col(0).iter().map(|value| value * 10_000.0).collect();
    assert_eq!(numbers(&comparison.data[0]["y"]), expected);
    let fall = numbers(&drawdown.data[0]["y"]);
    for (actual, expected) in fall.iter().zip([0.0, -0.01, -0.02, 0.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(comparison.data.iter().all(|trace| trace["line"].get("dash").is_none()));
    let layout = comparison.full_layout();
    assert!(layout["title"]["text"].as_str().unwrap().chars().count() <= 22);
    assert!(layout["legend"]["maxheight"].as_f64().unwrap() <= layout["margin"]["b"].as_f64().unwrap());
}

#[test]
fn chart_filter_preserves_full_study_and_requested_order() {
    let study = study();
    let selected = [study.equity.columns[11].as_str(), study.equity.columns[0].as_str()];
    let chart = backtest_chart(&study, &ChartOptions { strategies: Some(&selected), ..Default::default() }).unwrap();
    let names: Vec<&str> = chart.data.iter().map(|trace| trace["name"].as_str().unwrap()).collect();
    assert_eq!(names, selected);
    let empty: [&str; 0] = [];
    assert!(
        backtest_chart(&study, &ChartOptions { strategies: Some(&empty), ..Default::default() })
            .unwrap()
            .data
            .is_empty()
    );
    let error =
        backtest_chart(&study, &ChartOptions { strategies: Some(&["Missing portfolio"]), ..Default::default() })
            .unwrap_err();
    assert_eq!(error.message(), "Unknown backtest strategies: Missing portfolio");
}

#[test]
fn chart_styles_new_profiles_and_keeps_all_existing_paths() {
    let mut study = study();
    let first = study.equity.col(0);
    for method in ["Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window"] {
        for profile in ["Low", "Medium", "Extreme"] {
            let mut columns: Vec<(String, Vec<f64>)> =
                study.equity.columns.iter().enumerate().map(|(j, name)| (name.clone(), study.equity.col(j))).collect();
            columns.push((format!("{method} · {profile}"), first.clone()));
            study.equity = Frame::from_columns(study.equity.index.clone(), columns);
        }
    }
    let chart = backtest_chart(&study, &ChartOptions::default()).unwrap();
    assert_eq!(chart.data.len(), 24);
    let colors: std::collections::HashSet<&str> =
        chart.data[12..15].iter().map(|t| t["line"]["color"].as_str().unwrap()).collect();
    assert_eq!(colors.len(), 3);
}

#[test]
fn focus_mode_highlights_one_strategy_and_keeps_every_path() {
    let study = study();
    let focus = "Expanding window · Equal weight";
    let chart = backtest_chart(&study, &ChartOptions { focus: Some(focus), ..Default::default() }).unwrap();
    assert_eq!(chart.data.len(), 12);
    for trace in &chart.data {
        if trace["name"] == focus {
            assert!(
                trace["line"]["color"] == "#0071e3" && trace["line"]["width"] == 2.5 && trace["showlegend"] == true
            );
        } else {
            assert!(trace["line"]["color"] == "#c7c7cc" && trace["showlegend"] == false);
        }
    }
}

#[test]
fn profile_backtest_reports_explain_separate_past_only_fits() {
    let mut study = study();
    study.settings.update(&dict! {"include_profiles" => true, "profile_return_objective" => "historical_mean"});
    for document in [backtest_html(&study, None, None, None), findings_markdown(&study, "Synthetic demo", None, None)] {
        for phrase in [
            "relative risk levels within each fit",
            "only the history available before each allocation",
            "latest model holdings use all supplied history",
            "separate from these backtests and the original holdout",
            "same historical arithmetic mean return objective as the primary latest holdings",
            "Three-window estimates are diagnostics, not optimization targets",
        ] {
            assert!(document.contains(phrase), "{phrase}");
        }
        assert!(!document.contains("return target in each of three"));
    }
    study.settings.insert("profile_return_objective", "worst_window");
    for document in [backtest_html(&study, None, None, None), findings_markdown(&study, "Synthetic demo", None, None)] {
        assert!(document.contains("use the weakest-window objective"));
        assert!(document.contains("return target in each of three historical windows"));
        assert!(!document.contains("diagnostics, not optimization targets"));
    }
}

#[test]
fn html_escapes_asset_names_settings_and_warnings_without_bundle() {
    let mut study = study();
    study.holdings.get_mut(BEST).unwrap().rename_index(&[("AAA", "<img src=x onerror=\"asset()\">")]);
    study.settings.insert("<setting>", "<script>alert(\"setting\")</script>");
    study.warnings = vec!["<script>alert(\"warning\")</script>".into()];
    let document = backtest_html(&study, None, None, None);
    assert!(!document.contains("<img src=x onerror=\"asset()\">") && document.contains("&lt;img src=x onerror="));
    assert!(document.contains("&lt;setting&gt;"));
    assert!(document.contains("&lt;script&gt;alert(&quot;setting&quot;)&lt;/script&gt;"));
    assert!(document.contains("&lt;script&gt;alert(&quot;warning&quot;)&lt;/script&gt;"));
    assert!(!document.contains("* plotly.js v") && !document.contains("<!doctype html>"));
    for strategy in study.metrics.index_labels() {
        assert!(document.contains(&strategy));
    }
    assert_eq!(document.matches("<td>3</td>").count(), 12, "allocation event counts are not percentages");
    assert!(document.contains("Allocation events (incl. entry)"));
    assert_eq!(document.matches("<td>1.75</td>").count(), 12);
    assert_eq!(document.matches("<td>1.50</td>").count(), 12);
}

#[test]
fn backtest_settings_table_shows_plain_labels_and_percents() {
    let mut study = study();
    study.settings.update(&dict! {
        "train_fraction" => 0.7, "risk_free_rate" => 0.02, "include_profiles" => true, "test_start" => "2024-01-03",
    });
    let document = backtest_html(&study, None, None, None);
    let settings = document.split("<h2>Backtest settings</h2>").nth(1).unwrap().split("</table>").next().unwrap();
    for row in [
        "<th>Trading cost</th><td>10 basis points (0.10%)</td>",
        "<th>Trade every (price rows)</th><td>21</td>",
        "<th>Recent history for refits (return observations)</th><td>252</td>",
        "<th>Prices for the first fit</th><td>70%</td>",
        "<th>Risk-free rate</th><td>2.00%</td>",
        "<th>Risk levels tested</th><td>Yes</td>",
        "<th>Test from</th><td>2024-01-03</td>",
    ] {
        assert!(settings.contains(row), "{row}");
    }
    assert!(!settings.contains('_'));
}

#[test]
fn backtest_reports_use_configured_annualization() {
    for (periods, text) in [(Json::Float(12.0), "12"), (Json::Int(52), "52"), (Json::Int(365), "365")] {
        let mut study = study();
        study.settings.insert("periods_per_year", periods);
        for document in
            [backtest_html(&study, None, None, None), findings_markdown(&study, "Synthetic demo", None, None)]
        {
            assert!(document.contains(&format!("{text} observations per year")));
            assert!(!document.contains("252 observations per year"));
            assert!(document.contains("Sortino") && document.contains("Calmar"));
        }
    }
}

#[test]
fn markdown_has_all_strategies_dates_source_settings_and_limits() {
    let mut study = study();
    study.holdings.get_mut(BEST).unwrap().rename_index(&[("AAA", "A|<script>")]);
    let document = findings_markdown(&study, "Synthetic <demo>|source", None, None);
    for phrase in [
        "Synthetic &lt;demo&gt;\\|source",
        "A\\|&lt;script&gt;",
        "Portfolio baseline: 2024-01-02",
        "2024-01-03 to 2024-01-05 (3 observations)",
        "cost\\_bps: 10",
        "survivorship bias",
        "selected after evaluation",
        "not a forecast",
        "terminal liquidation is excluded",
        "equal-weight baselines coincide",
        "fee total excludes foregone compounding",
    ] {
        assert!(document.contains(phrase), "{phrase}");
    }
    for strategy in study.metrics.index_labels() {
        assert!(document.contains(&format!("| {strategy} |")));
    }
}

// ----------------------------------------------------------------------------- benchmark report

struct Comparison {
    prices: Frame,
    analysis: Analysis,
    profiles: ProfileAnalysis,
    study: BacktestStudy,
    benchmarks: BenchmarkComparison,
    evidence: EvidenceAnalysis,
}

/// The `comparison` fixture, on the legacy market prices instead of a seeded random draw.
fn comparison() -> Comparison {
    let prices = common::prices("legacy_market_prices.csv");
    let references = common::prices("legacy_market_benchmark_prices.csv");
    let analysis =
        analyze(&prices, &AnalyzeOptions { train_fraction: 0.6, frontier_points: 5, ..Default::default() }).unwrap();
    let profiles = build_profiles(&prices, &ProfileOptions { frontier_points: 3, ..Default::default() }).unwrap();
    let options = BacktestOptions {
        train_fraction: 0.6,
        rebalance_every: 40,
        rolling_window: Some(80),
        include_profiles: true,
        ..Default::default()
    };
    let study = run_backtests(&prices, &options).unwrap();
    let benchmarks = compare_benchmarks(&references, &prices, &analysis, Some(&study), Some(&profiles), 0.02).unwrap();
    let evidence =
        analyze_evidence(&study.equity, benchmarks.backtest_equity.as_ref().unwrap(), &EvidenceOptions::default())
            .unwrap();
    Comparison { prices, analysis, profiles, study, benchmarks, evidence }
}

#[test]
fn frontier_references_use_the_matching_estimation_objective() {
    let case = comparison();
    let training = presentation::frontier_chart(&case.analysis, Some(&case.benchmarks.training_estimates), None);
    let latest = presentation::latest_profile_chart(&case.profiles, Some(&case.benchmarks.latest_estimates), None);
    for (position, name) in case.benchmarks.training_estimates.index_labels().iter().enumerate() {
        let trace = &training.data[training.data.len() - 2 + position];
        let other = &latest.data[latest.data.len() - 2 + position];
        assert_eq!(trace["name"], efficient_frontier::pyfmt::escape(name));
        assert_eq!(trace["marker"]["symbol"], "diamond");
        assert_eq!(numbers(&trace["y"])[0], case.benchmarks.training_estimates.f64_at(name, "expected_return"));
        assert_eq!(numbers(&other["y"])[0], case.benchmarks.latest_estimates.f64_at(name, "expected_return"));
        assert!(other["hovertemplate"].as_str().unwrap().contains("Full-history arithmetic mean"));
        assert!(trace["hovertemplate"].as_str().unwrap().contains("Historical volatility, yearly"));
    }
    for chart in [&training, &latest] {
        let count = chart.data.len() - 2;
        assert!(
            chart.data[..count]
                .iter()
                .all(|t| t["hovertemplate"].as_str().unwrap().contains("Model volatility, yearly"))
        );
    }
}

#[test]
fn optional_weakest_window_frontier_keeps_matched_benchmark_estimates() {
    let case = comparison();
    let options =
        ProfileOptions { frontier_points: 3, return_objective: ReturnObjective::WorstWindow, ..Default::default() };
    let profiles = build_profiles(&case.prices, &options).unwrap();
    let chart = presentation::latest_profile_chart(&profiles, Some(&case.benchmarks.latest_estimates), None);
    assert_eq!(chart.full_layout()["title"]["text"], "Weakest-window frontier");
    let estimates = &case.benchmarks.latest_estimates;
    for (trace, name) in chart.data[chart.data.len() - 2..].iter().zip(estimates.index_labels()) {
        assert_eq!(numbers(&trace["y"])[0], estimates.f64_at(&name, "worst_window_return"));
        assert!(trace["hovertemplate"].as_str().unwrap().contains("Lowest annual window mean"));
    }
}

#[test]
fn equity_charts_keep_both_benchmarks_when_all_strategies_are_hidden() {
    let case = comparison();
    let equity = case.benchmarks.backtest_equity.as_ref().unwrap();
    let none: [&str; 0] = [];
    let options = ChartOptions { strategies: Some(&none), benchmark_equity: Some(equity), ..Default::default() };
    let chart = backtest_chart(&case.study, &options).unwrap();
    assert_eq!(chart.data.len(), 2);
    assert_eq!(numbers(&chart.data[0]["y"]), equity.col(0).iter().map(|v| v * 10_000.0).collect::<Vec<_>>());
    let holdout = presentation::holdout_chart(&case.analysis, Some(&case.benchmarks.holdout_equity), None).unwrap();
    let expected: Vec<f64> = case.benchmarks.holdout_equity.col(0).iter().map(|v| v * 10_000.0).collect();
    assert_eq!(numbers(&holdout.data[holdout.data.len() - 2]["y"]), expected);
}

#[test]
fn relative_wealth_chart_preserves_full_ratios_and_default_ignores_winner() {
    let case = comparison();
    let mut evidence = case.evidence.clone();
    let count = evidence.summary.nrows();
    evidence.summary.push("total_return_difference", (0..count).map(|i| Cell::Float(i as f64 * 100.0)).collect());
    let strategy = default_evidence_strategy(&evidence).unwrap();
    assert_eq!(strategy, "Expanding window · Medium");
    let chart = evidence_chart(&evidence, &strategy).unwrap();
    let equity = case.benchmarks.backtest_equity.as_ref().unwrap();
    let path = case.study.equity.column(&strategy).unwrap();
    for (j, trace) in chart.data.iter().enumerate() {
        let ratio: Vec<f64> = path.iter().zip(equity.col(j)).map(|(a, b)| a / b).collect();
        let expected: Vec<f64> = ratio.iter().map(|value| value / ratio[0]).collect();
        let actual = numbers(&trace["y"]);
        assert!(actual.iter().zip(&expected).all(|(a, e)| (a - e).abs() < 1e-12));
        assert_eq!(actual[0], 1.0);
    }
    let layout = chart.full_layout();
    assert_eq!((layout["shapes"][0]["y0"].as_f64(), layout["shapes"][0]["y1"].as_f64()), (Some(1.0), Some(1.0)));
    assert_eq!(evidence_chart(&evidence, "Missing").unwrap_err().message(), "Unknown evidence strategy: Missing");
    evidence.summary = evidence.summary.take_rows(&[]);
    assert_eq!(default_evidence_strategy(&evidence), None);
}

#[test]
fn evidence_format_keeps_ratios_pvalues_and_percentage_points_distinct() {
    let case = comparison();
    let mut evidence = case.evidence.clone();
    let count = evidence.summary.nrows();
    for (column, value) in [
        ("beta", 1.5),
        ("annual_advantage", 0.03),
        ("alpha", 0.02),
        ("advantage_pvalue_adjusted", 0.05),
        ("alpha_ci_low", f64::NAN),
    ] {
        evidence.summary.push(column, vec![Cell::Float(value); count]);
    }
    let display = evidence_summary_frame(&evidence);
    for (column, text) in [
        ("Benchmark beta", "1.500"),
        ("Annual mean advantage", "+3.00 pp"),
        ("Annual alpha", "2.00%"),
        ("Mean Holm p-value", "0.05"),
        ("Alpha CI lower", "—"),
    ] {
        assert!(display.column(column).iter().all(|cell| cell.as_str() == Some(text)), "{column}");
    }
    assert!(evidence.summary.f64s("beta").iter().all(|value| *value == 1.5));
}

#[test]
fn html_and_markdown_explain_matched_costs_and_statistical_limits() {
    let case = comparison();
    let (benchmarks, evidence) = (&case.benchmarks, Some(&case.evidence));
    for document in [
        benchmark_html(benchmarks, evidence, true),
        benchmark_markdown(benchmarks, evidence, true),
        findings_markdown(&case.study, "Fixture data", Some(benchmarks), evidence),
    ] {
        for phrase in [
            "95% HAC confidence intervals are pointwise",
            "Holm correction covers 96 tests",
            "at least 60 return observations",
            "same entry fee rate",
            "With covariance shrinkage, these are different risk estimates",
            "not scaled to match portfolio volatility",
            "not measure the probability of future outperformance",
            "Positive alpha does not establish investment skill",
            "Inference omits the first evaluation session",
            "Expanding window · Medium",
            "2019-",
        ] {
            assert!(document.contains(phrase), "{phrase}");
        }
    }
    assert!(benchmark_html(benchmarks, None, false).contains("Both exclude fees"));
    let document = backtest_html(&case.study, Some(benchmarks), evidence, None);
    assert!(document.find("Market benchmarks") < document.find("Backtesting findings"));
    assert!(!document.contains("* plotly.js v"));
}

#[test]
fn benchmark_report_escapes_evidence_text() {
    let case = comparison();
    let mut evidence = case.evidence.clone();
    evidence.warnings = vec!["<img src=x onerror=\"evidence()\">".into()];
    let count = evidence.summary.nrows();
    evidence.summary.push("status", vec![Cell::from("<script>status()</script>"); count]);
    let document = benchmark_html(&case.benchmarks, Some(&evidence), true);
    assert!(!document.contains(&evidence.warnings[0]) && document.contains("&lt;img src=x onerror="));
    assert!(
        !document.contains("<script>status()</script>") && document.contains("&lt;script&gt;status()&lt;/script&gt;")
    );
}

fn parts(case: &Comparison) -> ReportParts<'_> {
    ReportParts {
        study: Some(&case.study),
        latest_profiles: Some(&case.profiles),
        benchmarks: Some(&case.benchmarks),
        evidence: Some(&case.evidence),
        ..Default::default()
    }
}

#[test]
fn report_exports_all_pairs_raw_prices_and_reproducible_curves() {
    let case = comparison();
    let files = unzip(&report_zip(&case.analysis, &case.prices, &Dict::new(), &parts(&case)).unwrap());
    for name in [
        "benchmark_prices.csv",
        "benchmark_holdout_metrics.csv",
        "benchmark_holdout_curve.csv",
        "benchmark_training_estimates.csv",
        "benchmark_latest_estimates.csv",
        "benchmark_backtest_metrics.csv",
        "benchmark_backtest_curve.csv",
        "evidence_summary.csv",
        "evidence_windows.csv",
        "evidence_relative_curve.csv",
    ] {
        assert!(files.contains_key(name), "{name}");
    }
    assert_eq!(files["benchmark_prices.csv"], case.benchmarks.prices.to_csv("Date"));
    assert_eq!(files["evidence_relative_curve.csv"], case.evidence.relative_equity.to_csv(Some("Date")));
    let metadata: Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    assert_eq!(metadata["backtest_files"].as_object().unwrap().len(), 24);
    assert_eq!(metadata["backtests"]["profile_return_objective"], "historical_mean");
    assert_eq!(metadata["latest_profiles"]["settings"]["return_objective"], "historical_mean");
    assert_eq!(metadata["evidence"]["multiple_testing_tests"], 96);
    assert_eq!(files["latest_profile_summary.csv"], case.profiles.summary.to_csv(None));
    let document = &files["report.html"];
    assert_eq!(document.matches("* plotly.js v").count(), 1);
    assert!(document.contains("Evidence against benchmarks"));
}

#[test]
fn report_shares_live_decision_context_before_proposed_holdings() {
    let case = comparison();
    for method in ["Buy and hold", "Expanding window"] {
        let selected = format!("{method} · Extreme");
        let panel = story::decision_panel_html(&case.study, &selected, Some(&case.benchmarks));
        let drivers = story::result_drivers_html(&case.study, &selected);
        let view = ReportParts { selected_profile: Some("Extreme"), selected_method: Some(method), ..parts(&case) };
        let document = report_html(&case.analysis, &Dict::new(), &view, None).unwrap();
        let summary = document.split("<section id=\"summary\">").nth(1).unwrap().split("</section>").next().unwrap();
        assert!(summary.contains(&panel) && summary.contains(&drivers));
        assert!(summary.find(&panel) < summary.find("Latest proposed holdings"));
        assert!(summary.contains(
            "These weights use all selected prices. The test replays earlier fits of this rule, with different holdings."
        ));
    }
}

#[test]
fn decision_context_selection_preserves_every_numeric_export() {
    let case = comparison();
    let archives: Vec<IndexMap<String, String>> = [("Low", "Buy and hold"), ("Extreme", "Expanding window")]
        .iter()
        .map(|(profile, method)| {
            let view = ReportParts { selected_profile: Some(profile), selected_method: Some(method), ..parts(&case) };
            unzip(&report_zip(&case.analysis, &case.prices, &Dict::new(), &view).unwrap())
        })
        .collect();
    let csv: Vec<&String> = archives[0].keys().filter(|name| name.ends_with(".csv")).collect();
    assert_eq!(csv.iter().filter(|name| name.starts_with("backtests/")).count(), 24 * 3);
    for name in csv {
        assert_eq!(archives[0][name], archives[1][name], "{name}");
    }
    assert_ne!(archives[0]["report.html"], archives[1]["report.html"]);
}

// ----------------------------------------------------------------------------- classic report

/// The `report_analysis` fixture of `tests/test_presentation.py`.
fn report_analysis() -> Analysis {
    let dates: Vec<Date> =
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"].map(day).to_vec();
    let returns = |range: std::ops::Range<usize>| {
        Frame::from_columns(
            dates[range.clone()].to_vec(),
            vec![("AAA".into(), vec![0.01; range.len()]), ("BBB".into(), vec![0.005; range.len()])],
        )
    };
    let portfolio = Portfolio {
        name: "Minimum volatility".into(),
        weights: Series::new(vec!["AAA".into(), "BBB".into()], vec![0.6, 0.4]),
        expected_return: 0.1,
        volatility: 0.2,
        sharpe: 0.3,
    };
    let metrics = Performance {
        total_return: 0.016064,
        cagr: 0.8,
        volatility: 0.01,
        sharpe: 0.3,
        max_drawdown: 0.0,
        sortino: 1.25,
        calmar: 2.5,
    };
    Analysis {
        portfolios: [("Minimum volatility".to_string(), portfolio)].into_iter().collect(),
        frontier: Frontier {
            labels: vec!["AAA".into(), "BBB".into()],
            expected_return: vec![0.1, 0.15],
            volatility: vec![0.2, 0.3],
            sharpe: vec![0.3, 0.4],
            weights: vec![vec![0.6, 0.4], vec![0.7, 0.3]],
        },
        train_returns: returns(0..4),
        test_returns: returns(4..6),
        equity: Frame::from_columns(
            dates[3..].to_vec(),
            vec![("Minimum volatility".into(), vec![1.0, 1.008, 1.016064])],
        ),
        holdout_metrics: [("Minimum volatility".to_string(), metrics)].into_iter().collect(),
        mean_returns: Series::new(vec!["AAA".into(), "BBB".into()], vec![0.1, 0.05]),
        covariance: LabeledMatrix::square(
            vec!["AAA".into(), "BBB".into()],
            DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]),
        ),
        warnings: vec!["Review <script>alert(\"warning\")</script>".into()],
        periods_per_year: 252.0,
        risk_free_rate: 0.02,
    }
}

fn small_prices() -> Frame {
    Frame::from_columns(
        vec![day("2024-01-01"), day("2024-01-02")],
        vec![("AAA".into(), vec![100.0, 101.0]), ("BBB".into(), vec![100.0, 100.5])],
    )
}

#[test]
fn report_escapes_source_parameters_and_warnings() {
    let metadata =
        dict! {"source" => "Synthetic <script>alert(\"source\")</script>", "<unsafe-key>" => "<unsafe-value>"};
    let document = report_html(&report_analysis(), &metadata, &ReportParts::default(), None).unwrap();
    assert!(document.contains("Synthetic demonstration data"));
    assert!(document.contains("&lt;script&gt;alert(&quot;source&quot;)&lt;/script&gt;"));
    assert!(document.contains("&lt;script&gt;alert(&quot;warning&quot;)&lt;/script&gt;"));
    assert!(document.contains("&lt;unsafe-key&gt;") && document.contains("&lt;unsafe-value&gt;"));
    assert!(!document.contains("alert(\"source\")"));
    assert_eq!(document.matches("* plotly.js v").count(), 1);
    assert!(document.contains("2024-01-02") && document.contains("2024-01-09"));
    assert!(document.contains("There is no subsequent trading between assets"));
}

#[test]
fn zip_contains_reproducible_results_and_assumptions() {
    let analysis = report_analysis();
    let metadata = dict! {"source" => "Synthetic demo", "max_weight" => 0.7};
    let files = unzip(&report_zip(&analysis, &small_prices(), &metadata, &ReportParts::default()).unwrap());
    let mut names: Vec<&str> = files.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "THIRD_PARTY_NOTICES.txt",
            "frontier.csv",
            "frontier_weights.csv",
            "holdout_curve.csv",
            "holdout_metrics.csv",
            "metadata.json",
            "prices.csv",
            "report.html",
            "risk_contributions.csv",
            "risk_summary.csv",
            "weights.csv",
        ]
    );
    let metadata: Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    assert_eq!(metadata["training_observations"], 4);
    assert_eq!(metadata["holdout_observations"], 2);
    assert_eq!(metadata["assets"], serde_json::json!(["AAA", "BBB"]));
    assert_eq!(metadata["annualization_days"], 252.0);
    assert_eq!(files["weights.csv"], "ticker,Minimum volatility\r\nAAA,0.6\r\nBBB,0.4\r\n");
    assert!(files["holdout_curve.csv"].starts_with("Date,Minimum volatility\r\n2024-01-05,1.0\r\n"));
    assert!(files["prices.csv"].starts_with("Date,AAA,BBB\r\n"));
    assert!(files["report.html"].starts_with("<!doctype html>"));
    let license = files["THIRD_PARTY_NOTICES.txt"].as_str();
    assert!(
        license.starts_with("Plotly.js (embedded in report.html)\n\n")
            && license.contains("Permission is hereby granted")
    );
}

#[test]
fn report_zip_escapes_formula_labels_and_exports_coverage() {
    let label = "=1+1";
    let mut analysis = report_analysis();
    analysis.portfolios.get_mut("Minimum volatility").unwrap().weights.labels[0] = label.into();
    analysis.frontier.labels[0] = label.into();
    analysis.mean_returns.labels[0] = label.into();
    analysis.covariance.index[0] = label.into();
    analysis.covariance.columns[0] = label.into();
    let prices = Frame::from_columns(
        vec![day("2024-01-01"), day("2024-01-02")],
        vec![(label.into(), vec![100.0, 101.0]), ("BBB".into(), vec![100.0, 99.0])],
    );
    let coverage = Json::List(vec![Json::Dict(
        dict! {"symbol" => "AAA", "name" => "@Formula Inc.", "observations" => Json::Null},
    )]);
    let metadata = dict! {"source" => "Synthetic demo", "universe_coverage" => coverage};
    let files = unzip(&report_zip(&analysis, &prices, &metadata, &ReportParts::default()).unwrap());
    assert!(files["weights.csv"].lines().nth(1).unwrap().starts_with("'=1+1,"));
    assert!(files["risk_contributions.csv"].lines().nth(1).unwrap().starts_with("'=1+1,"));
    for name in ["prices.csv", "frontier_weights.csv"] {
        let header: Vec<&str> = files[name].lines().next().unwrap().split(',').collect();
        assert_eq!(header[1..], ["'=1+1", "BBB"], "{name}");
    }
    assert_eq!(files["universe_coverage.csv"], ",symbol,name,observations\r\n0,AAA,'@Formula Inc.,\r\n");
    assert!(files["report.html"].contains("<summary>Universe coverage</summary>"));
}

#[test]
fn report_view_is_recorded_only_when_requested_and_latest_profiles_are_separate() {
    let prices = efficient_frontier::data::demo_prices();
    let analysis = analyze(&prices, &AnalyzeOptions::default()).unwrap();
    let profiles = build_profiles(&prices, &ProfileOptions::default()).unwrap();
    let base = ReportParts { latest_profiles: Some(&profiles), ..Default::default() };
    let plain = unzip(&report_zip(&analysis, &prices, &Dict::new(), &base).unwrap());
    let metadata: Value = serde_json::from_str(&plain["metadata.json"]).unwrap();
    assert!(metadata.get("report_view").is_none());
    assert!(plain["latest_profile_summary.csv"].contains("Extreme"));
    let view = ReportParts { selected_profile: Some("Extreme"), ..base };
    let chosen = unzip(&report_zip(&analysis, &prices, &Dict::new(), &view).unwrap());
    let metadata: Value = serde_json::from_str(&chosen["metadata.json"]).unwrap();
    assert_eq!(metadata["report_view"], serde_json::json!({"risk_level": "Extreme", "rule": "Expanding window"}));
    let document = &chosen["report.html"];
    assert!(document.find("<section id=\"summary\">") < document.find("<section id=\"latest-profiles\">"));
    assert!(document.contains("Sorted by Highest."));
    assert!(document.contains("<span class=\"pl-nb\">highest-risk</span> portfolio"));
}

// ----------------------------------------------------------------------------- historical report

#[test]
fn historical_report_runs_offline_with_one_plotly_bundle_and_license() {
    let (prices, membership) = historical_demo();
    let study = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    let files = build_historical_report(&prices, &membership, &study);
    let document = String::from_utf8(files["report.html"].clone()).unwrap();
    assert!(
        document.starts_with("<!doctype html>") && !document.contains("<script src") && !document.contains("<link")
    );
    assert_eq!(document.matches("* plotly.js v").count(), 1);
    assert!(document.matches("Plotly.newPlot(").count() >= 2);
    let notices = String::from_utf8(files["THIRD_PARTY_NOTICES.txt"].clone()).unwrap();
    assert!(
        document.contains("Permission is hereby granted, free of charge")
            && notices.contains("Permission is hereby granted")
    );
    for phrase in [
        "Removed companies remain held",
        "cash_weights.csv contains daily closing cash proportions",
        "Missing prices for a held asset or an execution stop the study",
        "complete historical reconstruction",
    ] {
        assert!(document.contains(phrase), "{phrase}");
    }
    assert!(!document.contains("fixed input universe"));
    for name in &study.study.equity.columns {
        assert!(document.contains(name.as_str()));
    }
    assert!(!files.contains_key("LICENSE.n100tickers"));
}

#[test]
fn historical_report_escapes_metadata_warnings_and_coverage() {
    let (prices, membership) = historical_demo();
    let mut study = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    study.study.settings.insert("<setting>", "<value>");
    study.study.warnings = vec!["<img src=x onerror=\"warning()\">".into()];
    let reason = study.coverage.position("reason").unwrap();
    study.coverage.data[reason][1] = Cell::from("<script>alert(\"coverage\")</script>");
    let document =
        String::from_utf8(build_historical_report(&prices, &membership, &study)["report.html"].clone()).unwrap();
    for unsafe_text in ["<img src=x onerror=\"warning()\">", "<script>alert(\"coverage\")</script>", "<setting>"] {
        assert!(!document.contains(unsafe_text), "{unsafe_text}");
    }
    assert!(document.to_lowercase().contains("&lt;setting&gt;"));
}

#[test]
fn historical_csv_exports_escape_formula_text_without_changing_numeric_losses() {
    let (prices, membership) = historical_demo();
    let mut study = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    let reason = study.coverage.position("reason").unwrap();
    study.coverage.data[reason][1] = Cell::from("=HYPERLINK(\"https://example.invalid\")");
    let files = build_historical_report(&prices, &membership, &study);
    let coverage = String::from_utf8(files["coverage.csv"].clone()).unwrap();
    assert!(coverage.lines().nth(2).unwrap().contains(",\"'=HYPERLINK(\"\"https://example.invalid\"\")\""));
}

#[test]
fn bundled_membership_export_carries_identity_rules_and_upstream_license() {
    let (prices, membership) = historical_demo();
    let study = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    let bundled = bundled_membership();
    let files = build_historical_report(&prices, &bundled, &study);
    let metadata: Value = serde_json::from_slice(&files["metadata.json"]).unwrap();
    let source = bundled.metadata();
    assert_eq!(metadata["membership"]["source_commit"], common::json_value(source.get("source_commit").unwrap()));
    assert_eq!(
        metadata["membership"]["identity_exclusions"],
        common::json_value(source.get("identity_exclusions").unwrap())
    );
    let license = String::from_utf8(files["LICENSE.n100tickers"].clone()).unwrap();
    assert!(license.contains("Permission is hereby granted, free of charge"));
    assert!(String::from_utf8(files["THIRD_PARTY_NOTICES.txt"].clone()).unwrap().contains(&license));
    let document = String::from_utf8(files["report.html"].clone()).unwrap();
    for phrase in [
        "keep --historical-membership with the same bundled data revision",
        "membership.csv alone does not preserve identity exclusions",
        "--membership-csv",
        "does not restore the rules",
    ] {
        assert!(document.contains(phrase), "{phrase}");
    }
}

#[test]
fn demo_export_inputs_reproduce_equity_and_cash_paths() {
    let (prices, membership) = historical_demo();
    let study = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    let files = build_historical_report(&prices, &membership, &study);
    let details: Value = serde_json::from_slice(&files["metadata.json"]).unwrap();
    let recovered_prices = load_historical_prices(&files["prices.csv"]).unwrap();
    let coverage_end = details["membership"]["coverage_end"].as_str().unwrap();
    let recovered_membership = load_membership_csv(&files["membership.csv"], day(coverage_end), None).unwrap();
    let recovered =
        run_historical_backtests(&recovered_prices, &recovered_membership, &HistoricalOptions::default()).unwrap();
    assert_eq!(recovered.study.equity, study.study.equity);
    assert_eq!(recovered.cash_weights, study.cash_weights);
    for name in &study.study.equity.columns {
        let total: f64 = study.study.holdings[name].f64s("pnl_contribution").iter().sum::<f64>()
            - study.study.metrics.f64_at(name, "total_cost");
        assert!((total - study.study.metrics.f64_at(name, "total_return")).abs() < 1e-12);
    }
}

#[test]
fn report_uses_configured_annualization_and_preserves_risk_numbers() {
    let mut analysis = report_analysis();
    analysis.periods_per_year = 12.0;
    let empty = Frame::from_columns(Vec::new(), Vec::new());
    let files =
        unzip(&report_zip(&analysis, &empty, &dict! {"annualization_days" => 252}, &ReportParts::default()).unwrap());
    let metadata: Value = serde_json::from_str(&files["metadata.json"]).unwrap();
    assert_eq!(
        (metadata["periods_per_year"].as_f64(), metadata["annualization_days"].as_f64()),
        (Some(12.0), Some(12.0))
    );
    let document = &files["report.html"];
    assert!(document.contains("12 observations per year") && !document.contains("252 trading days"));
    assert!(document.contains("<th>Sortino</th>") && document.contains("<td>1.250</td>"));
    assert!(document.contains("<th>Calmar</th>") && document.contains("<td>2.500</td>"));
    let summary = efficient_frontier::risk::risk_summary(&analysis);
    assert!((summary.f64_at("Minimum volatility", "max_weight") - 0.6).abs() < 1e-12);
    assert!((summary.f64_at("Minimum volatility", "effective_holdings") - 1.0 / 0.52).abs() < 1e-12);
    assert!((summary.f64_at("Minimum volatility", "diversification_ratio") - 0.24 / 0.0336_f64.sqrt()).abs() < 1e-12);
    assert_eq!(files["risk_summary.csv"], summary.to_csv(None));
    let contributions = efficient_frontier::risk::risk_contributions(&analysis);
    assert!(contributions.f64s("Minimum volatility").iter().all(|share| (share - 0.5).abs() < 1e-12));
    assert!(document.contains("Share of portfolio variance"));
    assert!(document.contains("Negative shares") && document.contains("portfolio variance is zero"));
}

#[test]
fn risk_tables_escape_labels_and_display_undefined_values() {
    let label = "<img src=x onerror=\"risk()\">";
    let mut analysis = report_analysis();
    analysis.portfolios.get_mut("Minimum volatility").unwrap().weights.labels[0] = label.into();
    analysis.mean_returns.labels[0] = label.into();
    analysis.covariance = LabeledMatrix::square(vec![label.into(), "BBB".into()], DMatrix::zeros(2, 2));
    let document = report_html(&analysis, &Dict::new(), &ReportParts::default(), None).unwrap();
    let risk = document.split("<h2>Concentration and risk</h2>").nth(1).unwrap();
    let risk = risk.split("<h2>Realized holdout results</h2>").next().unwrap();
    assert!(!risk.contains(label) && risk.contains("&lt;img src=x onerror="));
    assert_eq!(risk.matches("<td>—</td>").count(), 3);
    assert!(risk.contains("<td>60.00%</td>"));
}

#[test]
fn optional_weakest_window_report_keeps_its_objective() {
    let prices = efficient_frontier::data::demo_prices();
    let options = ProfileOptions { return_objective: ReturnObjective::WorstWindow, ..Default::default() };
    let profiles = build_profiles(&prices, &options).unwrap();
    let parts = ReportParts { latest_profiles: Some(&profiles), ..Default::default() };
    let document = report_html(&report_analysis(), &Dict::new(), &parts, None).unwrap();
    let section =
        document.split("<section id=\"latest-profiles\">").nth(1).unwrap().split("</section>").next().unwrap();
    assert!(section.contains("define the weakest-window fit"));
    assert!(section.contains("midpoint of this return range, not the midpoint of volatility"));
    assert!(section.contains("Weakest-window frontier"));
    assert!(!section.contains("diagnostics, not optimization targets"));
}

#[test]
fn report_shows_escaped_stale_price_notes_beside_proposed_holdings() {
    let label = "<script>alert(\"stale\")</script>";
    let dates: Vec<Date> = (0..45)
        .map(|i| day("2024-01-01") + chrono::Days::new(i))
        .filter(|date| chrono::Datelike::weekday(date).number_from_monday() <= 5)
        .take(31)
        .collect();
    let rising: Vec<f64> = (0..31).map(|i| 100.0 + 10.0 * i as f64 / 30.0).collect();
    let prices = Frame::from_columns(dates, vec![(label.into(), vec![100.0; 31]), ("BBB".into(), rising)]);
    let saved = prices.clone();
    let notes = efficient_frontier::data::stale_price_notes(&prices);
    assert!(!notes.is_empty());
    let demo = efficient_frontier::data::demo_prices();
    let profiles = build_profiles(&demo, &ProfileOptions::default()).unwrap();
    let parts = ReportParts { latest_profiles: Some(&profiles), ..Default::default() };
    let document = report_html(&report_analysis(), &Dict::new(), &parts, Some(&prices)).unwrap();
    let summary = document.split("<section id=\"summary\">").nth(1).unwrap().split("</section>").next().unwrap();
    assert!(summary.find("Latest proposed holdings") < summary.find("Price checks"));
    for note in &notes {
        assert!(summary.contains(&efficient_frontier::pyfmt::escape(note)));
    }
    assert!(!summary.contains(label));
    assert_eq!(prices, saved);
}

#[test]
fn report_discloses_an_unavailable_profile_before_fallback_details() {
    let case = comparison();
    let options =
        BacktestOptions { train_fraction: 0.6, rebalance_every: 40, rolling_window: Some(80), ..Default::default() };
    let study = run_backtests(&case.prices, &options).unwrap();
    assert_eq!(study.equity.ncols(), 12);
    let parts = ReportParts {
        study: Some(&study),
        latest_profiles: Some(&case.profiles),
        benchmarks: Some(&case.benchmarks),
        selected_profile: Some("Extreme"),
        selected_method: Some("Expanding window"),
        ..Default::default()
    };
    let document = report_html(&case.analysis, &Dict::new(), &parts, None).unwrap();
    let summary = document.split("<section id=\"summary\">").nth(1).unwrap().split("</section>").next().unwrap();
    assert!(summary.contains(&story::decision_panel_html(
        &study,
        "Expanding window · Extreme",
        Some(&case.benchmarks)
    )));
    assert!(summary.contains("The requested Highest profile was not included in this backtest."));
    assert!(summary.contains("The detailed result below shows Expanding window · Minimum volatility."));
}

#[test]
fn omitted_market_comparison_is_explicit_without_fabricated_references() {
    let case = comparison();
    let metadata = dict! {
        "source" => "Synthetic demo", "benchmarks_note" => "No market comparison for synthetic prices.",
        "evidence_note" => "No statistical inference without market benchmarks.",
    };
    let document = report_html(&case.analysis, &metadata, &ReportParts::default(), None).unwrap();
    assert!(document.contains("No market comparison for synthetic prices."));
    assert!(document.contains("No statistical inference without market benchmarks."));
    assert!(!document.contains("id=\"benchmarks\""));
}
