//! The selection report, its ZIP export and the app-facing report pieces match the Python release.
//!
//! Golden files come from `tests/fixtures/golden/generate_reports.py`. Each
//! `reports_<case>.zip` holds the Python ZIP export (with plotly.js replaced by
//! a marker), the app-facing HTML snippets and tables, and chart JSON.
//!
//! Text compares exactly. Chart blocks in HTML compare separately: the Python
//! blocks use random element ids and base64 arrays, so both sides replace each
//! block with a marker, and the chart JSON compares with a numeric tolerance.
//! Numbers in CSV and JSON files compare with a tolerance where the model
//! results differ at solver tolerance; see [`csv_tolerance`].

mod common;

use std::collections::BTreeMap;
use std::io::Read as _;

use common::{assert_close, golden_dir, table_json};
use efficient_frontier::backtest::{BacktestOptions, BacktestStudy, run_backtests};
use efficient_frontier::backtest_report::{ChartOptions, backtest_chart, backtest_html, findings_markdown};
use efficient_frontier::benchmark_report::{benchmark_html, benchmark_markdown, evidence_chart};
use efficient_frontier::benchmarks::{BenchmarkComparison, compare_benchmarks};
use efficient_frontier::core::{Analysis, AnalyzeOptions, analyze};
use efficient_frontier::data::demo_prices;
use efficient_frontier::data::read_csv_rows;
use efficient_frontier::dict;
use efficient_frontier::evidence::{EvidenceAnalysis, EvidenceOptions, analyze_evidence};
use efficient_frontier::frame::{Cell, Frame, Table};
use efficient_frontier::historical::{HistoricalOptions, historical_demo, run_historical_backtests};
use efficient_frontier::historical_report::{build_historical_report, historical_report_zip};
use efficient_frontier::out_of_sample::{OutOfSampleStudy, run_default_out_of_sample};
use efficient_frontier::plot::{Figure, PLOTLY_JS};
use efficient_frontier::presentation::{self, ReportParts};
use efficient_frontier::profiles::{ProfileAnalysis, ProfileOptions, build_profiles};
use efficient_frontier::pyfmt::{Dict, Json};
use efficient_frontier::selection::{
    SelectionFrontier, SelectionOptions, SelectionStudy, run_selection, selection_frontier,
};
use efficient_frontier::selection_report as report;
use efficient_frontier::story;
use indexmap::IndexMap;
use serde_json::{Value, json};

/// The inputs of one golden case, as in `CASES` of the generator.
struct Inputs {
    prices: &'static str,
    benchmarks: Option<&'static str>,
    rows: Option<usize>,
    options: SelectionOptions,
    metadata: Dict,
    first_window_fails: Option<&'static str>,
}

fn coverage() -> Json {
    let record = |symbol: &str, name: &str, industry: &str, status: &str, reason: &str, observed: bool| {
        Json::Dict(dict! {
            "symbol" => symbol, "name" => name, "industry" => industry, "status" => status, "reason" => reason,
            "observations" => if observed { Json::Int(701) } else { Json::Null },
            "first_date" => if observed { Json::from("2019-01-01") } else { Json::Null },
            "last_date" => if observed { Json::from("2021-09-08") } else { Json::Null },
        })
    };
    Json::List(vec![
        record("S00", "Alpha Inc. Common Stock", "Technology", "included", "", true),
        record("S01", "Beta & Sons, Inc. Class A", "Retail", "included", "", true),
        record("OLD", "=Old Holdings PLC", "Utilities <old>", "excluded", "Yahoo returned no prices for: OLD.", false),
    ])
}

fn inputs(case: &str) -> Inputs {
    let options = |target_volatility: f64, max_weight: f64| SelectionOptions {
        target_volatility,
        max_weight,
        ..SelectionOptions::default()
    };
    match case {
        "demo" => Inputs {
            prices: "selection_demo_prices.csv",
            benchmarks: None,
            rows: None,
            options: options(0.2, 0.2),
            metadata: dict! {
                "source" => "Demo · synthetic", "universe" => "Demo", "price_currency" => "USD", "compare_market" => false,
                "benchmarks_note" => "Synthetic prices have no real-market benchmark comparison.",
                "selection_model_revision" => "stock_mean_buy_hold_v3",
            },
            first_window_fails: None,
        },
        "market" => Inputs {
            prices: "selection_market_prices.csv",
            benchmarks: Some("selection_market_benchmarks.csv"),
            rows: None,
            options: options(0.18, 0.1),
            metadata: dict! {
                "source" => "Yahoo Finance", "universe" => "Nasdaq-100", "price_currency" => "USD", "compare_market" => true,
                "requested_start" => "2019-01-01", "requested_end_exclusive" => "2021-09-09",
                "universe_sources" => vec!["https://example.invalid/nasdaq-100", "<b>backup list</b>"],
                "universe_requested" => 42, "universe_included" => 40, "universe_excluded" => 2,
                "universe_limitation" => "Current members only; former members are missing.",
                "benchmark_source" => "Yahoo Finance",
                "notes" => dict! {"weights" => vec![Json::Int(1), Json::Float(2.5), Json::Null, Json::Bool(true)], "ok" => "é"},
                "universe_coverage" => coverage(), "selection_model_revision" => "stock_mean_buy_hold_v3",
            },
            first_window_fails: None,
        },
        "stale" => Inputs {
            prices: "selection_stale_prices.csv",
            benchmarks: Some("selection_stale_benchmarks.csv"),
            rows: None,
            options: SelectionOptions { shrinkage: 0.2, cost_bps: 25.0, ..options(0.25, 0.15) },
            metadata: dict! {
                "source" => "Upload CSV", "universe" => "Upload CSV", "input_file" => "prices <stale>.csv",
                "price_currency" => "USD", "benchmark_source" => "Input columns: SPY, QQQ",
                "selection_model_revision" => "stock_mean_buy_hold_v3",
            },
            first_window_fails: None,
        },
        "walk" => Inputs {
            prices: "reports_walk_prices.csv",
            benchmarks: Some("reports_walk_benchmarks.csv"),
            rows: None,
            options: SelectionOptions { cost_bps: 25.0, ..options(0.2, 0.15) },
            metadata: dict! {"source" => "Synthetic adjusted prices", "universe" => "Fixture stocks"},
            first_window_fails: Some("requested annual volatility 1.00% is below the feasible minimum"),
        },
        "short" => Inputs {
            prices: "reports_walk_prices.csv",
            benchmarks: Some("reports_walk_benchmarks.csv"),
            rows: Some(16),
            options: options(0.5, 0.15),
            metadata: dict! {"source" => "Synthetic", "universe" => "Fixture"},
            first_window_fails: None,
        },
        other => panic!("unknown case {other}"),
    }
}

/// The Python run with a failed first fit: window 1 keeps its dates and reason, and nothing else.
fn fail_first_window(result: &mut OutOfSampleStudy, reason: &str) {
    let status = result.windows.position("status").unwrap();
    result.windows.data[status][0] = Cell::from(reason);
    let keep = |table: &Table, cells: &[Cell]| -> Table {
        let rows: Vec<usize> = (0..table.nrows()).filter(|i| cells[*i] != Cell::Int(1)).collect();
        table.take_rows(&rows)
    };
    result.metrics = keep(&result.metrics, &result.metrics.index[0].clone());
    result.weights = keep(&result.weights, &result.weights.index[0].clone());
    let mut eligibility = keep(&result.eligibility, result.eligibility.column("window"));
    eligibility.index = vec![(0..eligibility.nrows() as i64).map(Cell::Int).collect()];
    result.eligibility = eligibility;
    result.notes.retain(|note| !note.starts_with("Window 1: "));
}

struct Case {
    study: SelectionStudy,
    prices: Frame,
    metadata: Dict,
    frontier: SelectionFrontier,
    walk: Result<OutOfSampleStudy, String>,
}

fn run_case(name: &str) -> Case {
    let inputs = inputs(name);
    let mut prices = common::prices(inputs.prices);
    let mut benchmarks = inputs.benchmarks.map(common::prices);
    if let Some(rows) = inputs.rows {
        prices = prices.rows(0, rows);
        benchmarks = benchmarks.map(|frame| frame.rows(0, rows));
    }
    let study = run_selection(&prices, benchmarks.as_ref(), &inputs.options).unwrap();
    let frontier = selection_frontier(&study, &prices, 30).unwrap();
    let walk = match run_default_out_of_sample(&study, &prices) {
        Ok(mut result) => {
            if let Some(reason) = inputs.first_window_fails {
                fail_first_window(&mut result, reason);
            }
            Ok(result)
        }
        Err(error) => Err(error.message().to_string()),
    };
    Case { study, prices, metadata: inputs.metadata, frontier, walk }
}

/// The text files of one golden archive.
fn golden(name: &str) -> BTreeMap<String, String> {
    let path = golden_dir().join(format!("reports_{name}.zip"));
    let file = std::fs::File::open(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut archive = zip::ZipArchive::new(file).unwrap();
    let mut files = BTreeMap::new();
    for i in 0..archive.len() {
        let mut member = archive.by_index(i).unwrap();
        let mut text = String::new();
        member.read_to_string(&mut text).unwrap();
        files.insert(member.name().unwrap().to_string(), text);
    }
    files
}

/// Replace each chart block after `<div class="chart">` with `[chart]`, and return the blocks.
fn charts(html: &str) -> (String, Vec<String>) {
    let marker = "<div class=\"chart\">";
    let mut text = String::new();
    let mut blocks = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find(marker) {
        let body = start + marker.len();
        text.push_str(&rest[..body]);
        let script = rest[body..].find("</script>").expect("a chart script") + body;
        let end = rest[script..].find("</div>").expect("a chart end") + script + "</div>".len();
        blocks.push(rest[body..end].to_string());
        text.push_str("[chart]");
        rest = &rest[end..];
    }
    text.push_str(rest);
    (text, blocks)
}

/// Order rows of equal displayed weights by symbol in the latest weights table.
///
/// Weights at the holding cap agree to about 1e-11 in both versions, so their
/// largest-first order is solver noise.
fn tied_weights_by_symbol(html: &str) -> String {
    let Some(section) = html.find("<section id=\"latest-selection\">") else { return html.to_string() };
    let start = html[section..].find("<tbody>").unwrap() + section + "<tbody>".len();
    let end = html[start..].find("</tbody>").unwrap() + start;
    let rows: Vec<&str> = html[start..end].split_inclusive("</tr>").collect();
    let (body, tail) = rows.split_at(rows.len() - 1);
    let cell = |row: &str, tag: &str| {
        let open = format!("<{tag}>");
        let from = row.find(&open).map_or(0, |i| i + open.len());
        row[from..].split('<').next().unwrap_or("").to_string()
    };
    let mut ordered: Vec<&str> = Vec::new();
    for group in body.chunk_by(|a, b| cell(a, "td") == cell(b, "td")) {
        let mut group = group.to_vec();
        group.sort_by_key(|row| cell(row, "th"));
        ordered.extend(group);
    }
    format!("{}{}{}{}", &html[..start], ordered.concat(), tail.concat(), &html[end..])
}

/// The figure data and layout in a chart block written by [`Figure::to_html`].
fn block_figure(block: &str) -> Value {
    let mut rest = &block[block.find("Plotly.newPlot(").unwrap() + "Plotly.newPlot(".len()..];
    let mut values = Vec::new();
    for _ in 0..3 {
        rest = rest.trim_start().trim_start_matches(',').trim_start();
        let mut stream = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        values.push(stream.next().unwrap().unwrap());
        rest = &rest[stream.byte_offset()..];
    }
    json!({"data": values[1], "layout": values[2]})
}

/// Collects every difference, so one run reports them all.
#[derive(Default)]
struct Report {
    problems: Vec<String>,
    /// Displayed numbers one unit apart in their last digit, at a rounding boundary.
    rounding: Vec<String>,
    largest: f64,
}

/// Text split into numbers (digits with `,` and `.` inside) and other runs.
fn tokens(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = 0;
    let mut numeric = None;
    let characters: Vec<(usize, char)> = text.char_indices().collect();
    for (k, (i, c)) in characters.iter().enumerate() {
        let next_digit = characters.get(k + 1).is_some_and(|(_, next)| next.is_ascii_digit());
        let digit = c.is_ascii_digit() || (numeric == Some(true) && (*c == ',' || *c == '.') && next_digit);
        if numeric.is_some_and(|kind| kind != digit) {
            tokens.push(&text[start..*i]);
            start = *i;
        }
        numeric = Some(digit);
    }
    if start < text.len() {
        tokens.push(&text[start..]);
    }
    tokens
}

/// Two displayed numbers with the same decimals, one unit apart in the last digit.
fn one_unit_apart(actual: &str, expected: &str) -> bool {
    let decimals = |text: &str| text.split_once('.').map_or(0, |(_, fraction)| fraction.len());
    let number = |text: &str| text.replace(',', "").parse::<f64>().ok();
    match (number(actual), number(expected)) {
        (Some(a), Some(e)) if decimals(actual) == decimals(expected) => {
            ((a - e).abs() * 10f64.powi(decimals(actual) as i32) - 1.0).abs() < 1e-6
        }
        _ => false,
    }
}

impl Report {
    /// Exact text, except displayed numbers one unit apart in their last digit.
    /// The model results agree to solver tolerance, so a value next to a rounding
    /// boundary can round the other way; [`Report::finish`] lists each case.
    fn text(&mut self, what: &str, actual: &str, expected: &str) {
        if actual == expected {
            return;
        }
        let (a, e) = (tokens(actual), tokens(expected));
        if a.len() == e.len() {
            let mut close = Vec::new();
            let all = a.iter().zip(&e).all(|(x, y)| {
                x == y || {
                    let near = one_unit_apart(x, y);
                    if near {
                        close.push(format!("{what}: {x} versus Python {y}"));
                    }
                    near
                }
            });
            if all {
                self.rounding.extend(close);
                return;
            }
        }
        let position =
            actual.bytes().zip(expected.bytes()).position(|(a, e)| a != e).unwrap_or(actual.len().min(expected.len()));
        let window = |text: &str| {
            let mut start = position.saturating_sub(120).min(text.len());
            while !text.is_char_boundary(start) {
                start -= 1;
            }
            let mut end = (position + 160).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            text[start..end].to_string()
        };
        self.problems.push(format!(
            "{what}: first difference at byte {position}\n  actual:   {:?}\n  expected: {:?}",
            window(actual),
            window(expected)
        ));
    }

    fn values(&mut self, what: &str, actual: &Value, expected: &Value, tolerance: f64) {
        let result = std::panic::catch_unwind(|| assert_close(actual, expected, tolerance, what));
        match result {
            Ok(difference) => self.largest = self.largest.max(difference),
            Err(error) => {
                let message = error
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| error.downcast_ref::<&str>().map(|text| text.to_string()))
                    .unwrap_or_default();
                self.problems.push(message);
            }
        }
    }

    /// HTML or text with chart blocks compared separately.
    fn html(&mut self, what: &str, actual: &str, expected: &str) -> Vec<String> {
        let (actual_text, actual_blocks) = charts(actual);
        let (expected_text, expected_blocks) = charts(expected);
        self.text(what, &actual_text, &expected_text);
        if actual_blocks.len() != expected_blocks.len() {
            self.problems.push(format!("{what}: {} charts, expected {}", actual_blocks.len(), expected_blocks.len()));
        }
        actual_blocks
    }

    fn finish(self, name: &str) {
        eprintln!("{name}: largest numeric difference {:e}", self.largest);
        for note in &self.rounding {
            eprintln!("{name}: rounding boundary: {note}");
        }
        assert!(
            self.problems.is_empty(),
            "{name}: {} differences\n{}",
            self.problems.len(),
            self.problems.join("\n\n")
        );
    }
}

/// Numbers in CSV files: absolute tolerance by file, as in `tests/golden_selection.rs`.
///
/// Weights come from Clarabel solves in both versions and agree to solver
/// tolerance; frontier weights along a flat frontier move most. Prices,
/// windows, eligibility and coverage are inputs or counts and compare exactly.
fn csv_tolerance(name: &str) -> f64 {
    match name {
        "prices.csv"
        | "benchmark_prices.csv"
        | "eligibility.csv"
        | "universe_coverage.csv"
        | "out_of_sample/windows.csv"
        | "out_of_sample/eligibility.csv" => 0.0,
        "latest/frontier_weights.csv" => 1e-4,
        "latest/frontier.csv" | "out_of_sample/weights.csv" | "out_of_sample/metrics.csv" | "test/metrics.csv" => 1e-5,
        _ => 2e-6,
    }
}

fn compare_csv(report: &mut Report, name: &str, actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let tolerance = csv_tolerance(name);
    let parse = |text: &str| read_csv_rows(text).unwrap_or_else(|error| panic!("{name}: {error}"));
    let (actual_rows, expected_rows) = (parse(actual), parse(expected));
    if actual_rows.len() != expected_rows.len() {
        report.problems.push(format!("{name}: {} rows, expected {}", actual_rows.len(), expected_rows.len()));
        return;
    }
    for (i, (a, e)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        if a.len() != e.len() {
            report.problems.push(format!("{name} row {i}: {a:?} vs expected {e:?}"));
            continue;
        }
        for (x, y) in a.iter().zip(e) {
            if x == y {
                continue;
            }
            match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(x), Ok(y)) if (x - y).abs() <= tolerance => report.largest = report.largest.max((x - y).abs()),
                _ => report.problems.push(format!("{name} row {i}: {x:?} vs expected {y:?}")),
            }
        }
    }
    if !actual.ends_with("\r\n") || actual.contains("\r\n\r\n") {
        report.problems.push(format!("{name}: line endings differ"));
    }
}

/// `json.dumps(..., indent=2)` text: the same lines in the same order; numbers within `tolerance`.
fn compare_json_text(report: &mut Report, name: &str, actual: &str, expected: &str, tolerance: f64) {
    let (a, e): (Vec<&str>, Vec<&str>) = (actual.lines().collect(), expected.lines().collect());
    if a.len() != e.len() {
        report.text(name, actual, expected);
        return;
    }
    for (x, y) in a.iter().zip(&e) {
        if x == y {
            continue;
        }
        let number = |line: &str| -> Option<(String, f64)> {
            let (head, tail) = line.rsplit_once(": ").unwrap_or(("", line.trim()));
            let value = tail.trim().trim_end_matches(',');
            value.parse::<f64>().ok().map(|number| (head.to_string(), number))
        };
        match (number(x), number(y)) {
            (Some((hx, vx)), Some((hy, vy))) if hx == hy && (vx - vy).abs() <= tolerance * vy.abs().max(1.0) => {
                report.largest = report.largest.max((vx - vy).abs());
            }
            _ => report.problems.push(format!("{name}: {x:?} vs expected {y:?}")),
        }
    }
}

/// The figure JSON without the template, which plotly.py and this port write differently.
fn figure_json(figure: &Figure) -> Value {
    let mut value = figure.to_json();
    value["layout"].as_object_mut().unwrap().remove("template");
    value
}

fn findings_json(findings: &report::OutOfSampleFindings) -> Value {
    let higher: serde_json::Map<String, Value> =
        findings.higher_than.iter().map(|(name, count)| (name.clone(), json!(count))).collect();
    json!({
        "completed": findings.completed, "windows": findings.windows, "in_sample": findings.in_sample,
        "out_of_sample": findings.out_of_sample, "change": findings.change, "lower": findings.lower,
        "margin_in_sample": findings.margin_in_sample, "margin_out_of_sample": findings.margin_out_of_sample,
        "higher_than": higher, "test_returns": findings.test_returns,
    })
}

/// Compare every member of an exported archive with the golden archive. Returns the report's chart blocks.
fn compare_archive(
    result: &mut Report,
    bytes: &[u8],
    golden: &BTreeMap<String, String>,
    json_tolerance: f64,
) -> Vec<String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<String> =
        (0..archive.len()).map(|i| archive.by_index(i).unwrap().name().unwrap().to_string()).collect();
    let expected_names: Vec<String> = serde_json::from_str(&golden["names.json"]).unwrap();
    result.values("archive names", &json!(names), &json!(expected_names), 0.0);
    let mut blocks = Vec::new();
    for member in &names {
        let mut file = archive.by_name(member).unwrap();
        assert_eq!(file.compression(), zip::CompressionMethod::Deflated, "{member}");
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        let Some(expected) = golden.get(&format!("archive/{member}")) else { continue };
        if member == "report.html" {
            assert_eq!(text.matches(PLOTLY_JS).count(), 1, "plotly.js appears once");
            let text = tied_weights_by_symbol(&text.replace(PLOTLY_JS, "[plotly.js]"));
            blocks = result.html("report.html", &text, &tied_weights_by_symbol(expected));
            let mut ids: Vec<&str> =
                text.match_indices("class=\"plotly-graph-div\"").map(|(i, _)| &text[..i]).collect();
            ids.iter_mut().for_each(|prefix| *prefix = prefix.rsplit("id=\"").next().unwrap_or(""));
            let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
            assert_eq!(unique.len(), ids.len(), "chart ids are unique");
        } else if member.ends_with(".json") {
            compare_json_text(result, member, &text, expected, json_tolerance);
        } else if member.ends_with(".csv") {
            compare_csv(result, member, &text, expected);
        } else {
            result.html(member, &text, expected);
        }
    }
    blocks
}

fn check(name: &str) {
    let case = run_case(name);
    let golden = golden(name);
    let mut result = Report::default();
    let (study, prices) = (&case.study, &case.prices);

    // The ZIP export: member order, every CSV, metadata.json, the notice and the report.
    let walk = case.walk.as_ref().ok();
    let bytes = report::selection_report_zip(study, prices, &case.metadata, Some(&case.frontier), walk).unwrap();
    let blocks = compare_archive(&mut result, &bytes, &golden, 1e-6);
    let figures = [report::selection_frontier_chart(&case.frontier, study), report::selection_chart(&study.test)];
    // Each embedded chart is the figure of the public chart function.
    let embedded: Vec<Value> = blocks.iter().map(|block| block_figure(block)).collect();
    result.values("report frontier chart", &embedded[0], &figures[0].to_json(), 0.0);
    result.values("report replay chart", &embedded[embedded.len() - 1], &figures[1].to_json(), 0.0);

    // App-facing snippets and tables.
    let snippets: Value = serde_json::from_str(&golden["snippets.json"]).unwrap();
    let snippet = |key: &str| snippets[key].as_str().unwrap_or_else(|| panic!("snippet {key}")).to_string();
    let texts = [
        ("selection_comparison_html", report::selection_comparison_html(study, prices)),
        ("selection_replay_findings_html", report::selection_replay_findings_html(&study.test)),
        ("selection_replay_verdict_html", report::selection_replay_verdict_html(&study.test)),
        ("selection_frontier_summary_html", report::selection_frontier_summary_html(study, prices, &case.frontier)),
        ("selection_summary_html", report::selection_summary_html(study, &case.metadata)),
        ("selection_benchmark_note", report::selection_benchmark_note(study)),
        ("selection_fit_summary_html", report::selection_fit_summary_html(study, prices)),
        ("out_of_sample_html", report::out_of_sample_html(case.walk.as_ref().map_err(String::as_str))),
        ("result_drivers_html", story::result_drivers_html(&study.test, &study.selected)),
        ("context_result_html", story::context_result_html(&case.metadata, prices)),
        ("fine_print_html", story::fine_print_html(&case.metadata, prices)),
        ("backtest_html", backtest_html(&study.test, None, None, None)),
        ("findings_markdown", findings_markdown(&study.test, "Golden <source>", None, None)),
    ];
    for (key, actual) in &texts {
        result.html(key, actual, &snippet(key));
    }
    let settings: Vec<[String; 2]> = study
        .settings
        .iter()
        .map(|(key, value)| [story::setting_label(key), story::setting_text(key, value)])
        .collect();
    result.values("settings", &json!(settings), &snippets["settings"], 0.0);
    result.values(
        "selection_metrics_frame",
        &table_json(&report::selection_metrics_frame(&study.test)),
        &snippets["selection_metrics_frame"],
        1e-5,
    );
    result.values(
        "selection_replay_dominance",
        &table_json(&report::selection_replay_dominance(&study.test)),
        &snippets["selection_replay_dominance"],
        1e-5,
    );
    let bridge = report::selection_replay_return_bridge(study, prices);
    let bridge = json!({
        "fitted_mean": bridge.fitted_mean, "realized_mean": bridge.realized_mean, "mean_change": bridge.mean_change,
        "weight_drift": bridge.weight_drift, "entry_delay": bridge.entry_delay, "entry_fee": bridge.entry_fee,
    });
    result.values("selection_replay_return_bridge", &bridge, &snippets["selection_replay_return_bridge"], 1e-5);
    if let Ok(walk) = &case.walk
        && let Some(findings) = report::out_of_sample_findings(walk)
    {
        result.values("out_of_sample_findings", &findings_json(&findings), &snippets["out_of_sample_findings"], 1e-5);
        let (headline, lines) = report::out_of_sample_text(&findings);
        result.values("out_of_sample_text", &json!([headline, lines]), &snippets["out_of_sample_text"], 0.0);
        result.values(
            "out_of_sample_frame",
            &table_json(&report::out_of_sample_frame(walk)),
            &snippets["out_of_sample_frame"],
            1e-5,
        );
        result.values(
            "out_of_sample_paths_frame",
            &table_json(&report::out_of_sample_paths_frame(walk)),
            &snippets["out_of_sample_paths_frame"],
            1e-5,
        );
    }

    // Chart JSON. Frontier points move at solver tolerance.
    let figures: Value = serde_json::from_str(&golden["charts.json"]).unwrap();
    let drawdown = backtest_chart(&study.test, &ChartOptions { drawdown: true, ..Default::default() }).unwrap();
    let mut actual = vec![
        ("selection_chart", report::selection_chart(&study.test), 2e-6 * 10_000.0),
        ("selection_frontier_chart", report::selection_frontier_chart(&case.frontier, study), 1e-5),
        ("backtest_chart_drawdown", drawdown, 2e-6),
    ];
    if let Ok(walk) = &case.walk
        && report::out_of_sample_findings(walk).is_some()
    {
        actual.push(("out_of_sample_chart", report::out_of_sample_chart(walk), 1e-5));
    }
    assert_eq!(actual.len(), figures.as_object().unwrap().len(), "{name}: chart count");
    for (key, figure, tolerance) in &actual {
        result.values(key, &figure_json(figure), &figures[*key], *tolerance);
    }
    result.finish(name);
}

#[test]
fn demo_report_matches_python() {
    check("demo");
}

#[test]
fn market_report_with_benchmarks_and_coverage_matches_python() {
    check("market");
}

#[test]
fn stale_prices_shrinkage_and_costs_report_matches_python() {
    check("stale");
}

#[test]
fn walk_forward_report_with_a_failed_window_matches_python() {
    check("walk");
}

#[test]
fn too_short_walk_forward_history_report_matches_python() {
    check("short");
}

// ----------------------------------------------------------------------------- legacy and historical reports

struct Legacy {
    prices: Frame,
    analysis: Analysis,
    study: Option<BacktestStudy>,
    latest: Option<ProfileAnalysis>,
    benchmarks: Option<BenchmarkComparison>,
    evidence: Option<EvidenceAnalysis>,
    metadata: Dict,
}

/// The inputs of `legacy_full`, `legacy_plain` and `legacy_demo` in the generator.
fn legacy(name: &str) -> Legacy {
    let settings = AnalyzeOptions { train_fraction: 0.6, risk_free_rate: 0.03, max_weight: 0.4, ..Default::default() };
    match name {
        "full" | "plain" => {
            let market = common::prices("legacy_market_prices.csv");
            let market_benchmarks = common::prices("legacy_market_benchmark_prices.csv");
            let analysis = analyze(&market, &settings).unwrap();
            let full = name == "full";
            let study = full.then(|| {
                let options = BacktestOptions {
                    train_fraction: 0.6,
                    risk_free_rate: 0.03,
                    max_weight: 0.4,
                    rebalance_every: 20,
                    cost_bps: 15.0,
                    include_profiles: true,
                    ..Default::default()
                };
                run_backtests(&market, &options).unwrap()
            });
            let latest = full.then(|| {
                build_profiles(&market, &ProfileOptions { risk_free_rate: 0.03, max_weight: 0.4, ..Default::default() })
                    .unwrap()
            });
            let benchmarks =
                compare_benchmarks(&market_benchmarks, &market, &analysis, study.as_ref(), latest.as_ref(), 0.03)
                    .unwrap();
            let (equity, reference) = match &study {
                Some(study) => (&study.equity, benchmarks.backtest_equity.as_ref().unwrap()),
                None => (&analysis.equity, &benchmarks.holdout_equity),
            };
            let options = EvidenceOptions { risk_free_rate: 0.03, periods_per_year: 252.0, delayed_entry: full };
            let evidence = analyze_evidence(equity, reference, &options).unwrap();
            let metadata = if full {
                dict! {
                    "source" => "CSV: legacy_market_prices.csv", "price_currency" => "USD",
                    "currency_assumption" => "The user declares all asset prices in USD. No currency conversion occurs.",
                    "train_fraction" => 0.6, "risk_free_rate" => 0.03, "max_weight" => 0.4,
                    "benchmark_source" => "CSV: legacy <benchmarks>.csv", "notes" => vec![Json::from("é"), Json::Int(1)],
                }
            } else {
                dict! {
                    "source" => "CSV: legacy_market_prices.csv", "price_currency" => "USD",
                    "train_fraction" => 0.6, "risk_free_rate" => 0.03, "max_weight" => 0.4,
                    "benchmark_source" => "CSV: legacy_market_benchmark_prices.csv", "universe_coverage" => coverage(),
                }
            };
            Legacy {
                prices: market,
                analysis,
                study,
                latest,
                benchmarks: Some(benchmarks),
                evidence: Some(evidence),
                metadata,
            }
        }
        "demo" => {
            let prices = demo_prices();
            let analysis = analyze(&prices, &AnalyzeOptions::default()).unwrap();
            let latest = build_profiles(&prices, &ProfileOptions::default()).unwrap();
            let metadata = dict! {
                "source" => "Demo · synthetic", "universe" => "Demo", "price_currency" => "USD",
                "benchmarks_note" => "Synthetic demo data has no real market benchmark comparison.",
            };
            Legacy { prices, analysis, study: None, latest: Some(latest), benchmarks: None, evidence: None, metadata }
        }
        other => panic!("unknown legacy case {other}"),
    }
}

fn check_legacy(name: &str, view: (Option<&str>, Option<&str>)) {
    let case = legacy(name);
    let golden = golden(&format!("legacy_{name}"));
    let mut result = Report::default();
    let parts = ReportParts {
        study: case.study.as_ref(),
        latest_profiles: case.latest.as_ref(),
        benchmarks: case.benchmarks.as_ref(),
        evidence: case.evidence.as_ref(),
        selected_profile: view.0,
        selected_method: view.1,
    };
    let bytes = presentation::report_zip(&case.analysis, &case.prices, &case.metadata, &parts).unwrap();
    compare_archive(&mut result, &bytes, &golden, 1e-5);

    let snippets: Value = serde_json::from_str(&golden["snippets.json"]).unwrap();
    let mut texts: Vec<(&str, String)> = Vec::new();
    let mut charts: Vec<(String, Figure, f64)> = Vec::new();
    let benchmarks = case.benchmarks.as_ref();
    charts.push((
        "frontier_chart".into(),
        presentation::frontier_chart(&case.analysis, benchmarks.map(|b| &b.training_estimates), Some("Medium")),
        1e-5,
    ));
    charts.push((
        "holdout_chart".into(),
        presentation::holdout_chart(&case.analysis, benchmarks.map(|b| &b.holdout_equity), None).unwrap(),
        1e-5 * 10_000.0,
    ));
    if let Some(latest) = &case.latest {
        charts.push((
            "latest_profile_chart".into(),
            presentation::latest_profile_chart(latest, benchmarks.map(|b| &b.latest_estimates), Some("Extreme")),
            1e-5,
        ));
        let why = story::why_text(latest, "Medium", "Nasdaq-100");
        result.values("why_text", &json!(why), &snippets["why_text"], 0.0);
        let names: IndexMap<String, String> = [("A00".to_string(), "<Name>".to_string())].into_iter().collect();
        texts.push(("stats_html", story::stats_html(latest, "Low", "Upload CSV", Some(&names))));
        texts.push(("levels_ledger_html", story::levels_ledger_html(latest, "Extreme", None, "Demo")));
    }
    if let Some(benchmarks) = benchmarks {
        let backtest = case.study.is_some();
        texts.push(("benchmark_html", benchmark_html(benchmarks, case.evidence.as_ref(), backtest)));
        texts.push(("benchmark_markdown", benchmark_markdown(benchmarks, case.evidence.as_ref(), backtest)));
    }
    if let Some(study) = &case.study {
        let selected = "Expanding window · Medium";
        let evidence = case.evidence.as_ref();
        for mode in ["Growth", "Drawdown", "Relative to markets"] {
            let figure = story::story_chart(study, selected, mode, benchmarks, evidence).unwrap();
            charts.push((format!("story_chart {mode}"), figure, 1e-5 * 10_000.0));
        }
        charts.push(("evidence_chart".into(), evidence_chart(evidence.unwrap(), selected).unwrap(), 1e-5));
        let metadata = dict! {"universe" => "Nasdaq-100"};
        let note = dict! {"evidence_note" => "x"};
        texts.extend([
            ("decision_panel_html", story::decision_panel_html(study, "Rolling window · Low", benchmarks)),
            ("market_answer", story::market_answer(study, selected, benchmarks)),
            ("market_lead", story::market_lead(study, selected)),
            ("result_tile_html", story::result_tile_html(study, selected, benchmarks, evidence, &metadata)),
            ("chart_footnote", story::chart_footnote(study, benchmarks)),
            ("evidence_answer", story::evidence_answer(evidence, selected)),
            ("evidence_lead", story::evidence_lead(evidence, selected, story::months_tested(study), Some(&note))),
            ("clarity_text", story::clarity_text(evidence, selected)),
            ("interval_html", story::interval_html(evidence.unwrap(), selected)),
            ("stretch_summary_html", story::stretch_summary_html(evidence.unwrap(), selected)),
            ("stretch_table_html", story::stretch_table_html(evidence.unwrap(), selected)),
            ("backtest_html", backtest_html(study, benchmarks, evidence, Some("Rolling window · Extreme"))),
            ("findings_markdown", findings_markdown(study, "Golden", benchmarks, evidence)),
        ]);
    }
    for (key, actual) in &texts {
        let expected = snippets[*key].as_str().unwrap_or_else(|| panic!("snippet {key}"));
        result.html(key, actual, expected);
    }
    let figures: Value = serde_json::from_str(&golden["charts.json"]).unwrap();
    assert_eq!(charts.len(), figures.as_object().unwrap().len(), "{name}: chart count");
    for (key, figure, tolerance) in &charts {
        result.values(key, &figure_json(figure), &figures[key.as_str()], *tolerance);
    }
    result.finish(&format!("legacy {name}"));
}

#[test]
fn legacy_report_with_backtests_benchmarks_and_evidence_matches_python() {
    check_legacy("full", (Some("Extreme"), Some("Rolling window")));
}

#[test]
fn legacy_holdout_report_with_benchmarks_and_evidence_matches_python() {
    check_legacy("plain", (None, None));
}

#[test]
fn legacy_demo_report_with_profiles_matches_python() {
    check_legacy("demo", (None, None));
}

#[test]
fn historical_demo_report_matches_python() {
    let (prices, membership) = historical_demo();
    let study = run_historical_backtests(&prices, &membership, &HistoricalOptions::default()).unwrap();
    let files = build_historical_report(&prices, &membership, &study);
    let bytes = historical_report_zip(&files).unwrap();
    let golden = golden("historical_demo");
    let mut result = Report::default();
    compare_archive(&mut result, &bytes, &golden, 1e-8);
    result.finish("historical demo");
}
