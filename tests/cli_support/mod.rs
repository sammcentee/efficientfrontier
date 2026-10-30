//! Shared fakes, fixtures and file checks for the ports of `tests/test_cli.py`.
//!
//! [`Fake`] replaces the monkeypatched functions of the Python tests. A source
//! without an answer forbids its call, as `forbid_download` does.

#![allow(dead_code)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use efficient_frontier::cli::{Args, Sources, run};
use efficient_frontier::error::Result;
use efficient_frontier::evidence::{self, EvidenceAnalysis, EvidenceOptions};
use efficient_frontier::frame::{Cell, Date, Frame, Table, cells, iso};
use efficient_frontier::pyfmt::{Dict, Json};
use efficient_frontier::universe::{NASDAQ100_URL, UNIVERSE_LIMITATION, UniversePrices};
use efficient_frontier::{data, dict, value_error};
use serde_json::Value;

pub const FORBIDDEN: &str = "This CLI invocation must not request network data.";

/// Command-line arguments from text and paths: `argv!["--csv", source, "--output", output]`.
#[macro_export]
macro_rules! argv {
    ($($item:expr),* $(,)?) => {
        Vec::from([$(std::ffi::OsString::from(AsRef::<std::ffi::OsStr>::as_ref(&$item))),*])
    };
}

/// Offline replacements for the CLI's downloads. Every call is recorded.
#[derive(Default)]
pub struct Fake {
    /// The answer of `download_prices`. `None` forbids the call.
    pub prices: Option<Frame>,
    /// The answer of `download_benchmarks`. `None` forbids the call.
    pub benchmarks: Option<Result<Frame>>,
    /// The universe that `universe` serves (`true` for S&P 500 + Nasdaq-100) and its answer.
    /// `None` forbids both universes; the other universe is always forbidden.
    pub universe: Option<(bool, Result<UniversePrices>)>,
    /// Forbid `demo_prices`, as the Python tests that monkeypatch it.
    pub forbid_demo: bool,
    /// Make `analyze_evidence` fail with this message.
    pub evidence_error: Option<String>,
    pub price_calls: Vec<Vec<String>>,
    pub benchmark_calls: Vec<Vec<Date>>,
    pub universe_calls: Vec<(bool, Date, Date)>,
}

impl Sources for Fake {
    fn universe(&mut self, sp500_nasdaq100: bool, start: Date, end: Date) -> Result<UniversePrices> {
        self.universe_calls.push((sp500_nasdaq100, start, end));
        match &self.universe {
            Some((served, answer)) if *served == sp500_nasdaq100 => answer.clone(),
            _ => panic!("{FORBIDDEN}"),
        }
    }

    fn download_prices(&mut self, tickers: &[String], _: Date, _: Date) -> Result<Frame> {
        self.price_calls.push(tickers.to_vec());
        Ok(self.prices.clone().expect(FORBIDDEN))
    }

    fn download_benchmarks(&mut self, dates: &[Date]) -> Result<Frame> {
        self.benchmark_calls.push(dates.to_vec());
        self.benchmarks.clone().expect(FORBIDDEN)
    }

    fn original_tickers(&mut self) -> Result<Vec<String>> {
        panic!("{FORBIDDEN}")
    }

    fn demo_prices(&mut self) -> Frame {
        assert!(!self.forbid_demo, "{FORBIDDEN}");
        data::demo_prices()
    }

    fn analyze_evidence(
        &mut self,
        strategy_equity: &Frame,
        benchmark_equity: &Frame,
        options: &EvidenceOptions,
    ) -> Result<EvidenceAnalysis> {
        match &self.evidence_error {
            Some(message) => Err(value_error!("{message}")),
            None => evidence::analyze_evidence(strategy_equity, benchmark_equity, options),
        }
    }
}

/// The exit code and console text of one run.
pub struct Run {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// `cli.main()` with `sys.argv` set to the arguments: run in this process with `sources`.
pub fn run_cli(arguments: &[OsString], sources: &mut Fake) -> Run {
    let mut command = vec![OsString::from("efficient-frontier")];
    command.extend_from_slice(arguments);
    let args = Args::try_parse_from(command).expect("valid arguments");
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let code = run(args, sources, &mut stdout, &mut stderr);
    Run { code, stdout: String::from_utf8(stdout).unwrap(), stderr: String::from_utf8(stderr).unwrap() }
}

/// `subprocess.run([sys.executable, "-m", "efficient_frontier", ...])`: run the real program.
pub fn run_bin(arguments: &[OsString]) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_efficient-frontier")).args(arguments).output().unwrap();
    Run {
        code: output.status.code().expect("an exit code"),
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

/// Run an exported reproduce command offline: in this process with every download
/// forbidden, then with the real program in place of its first word.
pub fn reproduce(command: &str) {
    let words = shlex_split(command);
    assert_eq!(words[0], "efficient-frontier", "{command}");
    let arguments: Vec<OsString> = words[1..].iter().map(OsString::from).collect();
    let run = run_cli(&arguments, &mut Fake { forbid_demo: true, ..Fake::default() });
    assert_eq!(run.code, 0, "{}", run.stderr);
    let run = run_bin(&arguments);
    assert_eq!(run.code, 0, "{}", run.stderr);
}

/// Python's `shlex.split` for the output of `shlex.join`: plain words, single
/// quotes, and a single quote inside double quotes.
pub fn shlex_split(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word: Option<String> = None;
    let mut chars = command.chars();
    while let Some(c) = chars.next() {
        match c {
            ' ' => words.extend(word.take()),
            '\'' | '"' => word.get_or_insert_with(String::new).extend(chars.by_ref().take_while(|next| *next != c)),
            c => word.get_or_insert_with(String::new).push(c),
        }
    }
    words.extend(word);
    words
}

pub fn day(text: &str) -> Date {
    Date::parse_from_str(text, "%Y-%m-%d").unwrap()
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli").join(name)
}

/// The `benchmark_inputs` fixture: assets `ASSET_A` and `ASSET_B`, and benchmarks `SPY` and `QQQ`.
pub fn benchmark_inputs() -> (Frame, Frame) {
    let prices = data::load_csv_path(&fixture("benchmark_inputs.csv")).unwrap();
    (prices.select(&["ASSET_A", "ASSET_B"]).unwrap(), prices.select(&["SPY", "QQQ"]).unwrap())
}

/// The `nasdaq_universe` fixture: two complete members, one excluded member and the benchmarks.
pub fn nasdaq_universe() -> UniversePrices {
    let (prices, benchmarks) = benchmark_inputs();
    let dates = &prices.index;
    let count = dates.len() as i64;
    let (first, last) = (iso(dates[0]), iso(dates[dates.len() - 1]));
    let coverage = Table::range(3)
        .with("symbol", cells(&["ASSET_A", "ASSET_B", "SHORT"]))
        .with("name", cells(&["Company A", "Company B", "Recent company"]))
        .with("industry", cells(&["Technology", "Industrials", "Technology"]))
        .with("status", cells(&["included", "included", "excluded"]))
        .with("reason", cells(&["", "", "Incomplete historical prices."]))
        .with("observations", cells(&[count, count, 20]))
        .with("first_date", cells(&[first.clone(), first, iso(dates[dates.len() - 20])]))
        .with("last_date", cells(&[last.clone(), last.clone(), last]));
    let metadata = dict! {
        "universe" => "Current Nasdaq-100 members",
        "universe_source_url" => NASDAQ100_URL,
        "universe_source_date" => "2026-10-02",
        "universe_retrieved_at" => "2026-10-04T12:00:00+00:00",
        "universe_requested" => 3,
        "universe_included" => 2,
        "universe_excluded" => 1,
        "universe_limitation" => UNIVERSE_LIMITATION,
        "universe_coverage" => records(&coverage),
    };
    UniversePrices { prices, coverage, benchmarks, metadata }
}

/// `DataFrame.to_dict(orient="records")` of a table with text and integer columns.
pub fn records(table: &Table) -> Json {
    let row = |i: usize| {
        let mut record = Dict::new();
        for (name, column) in table.columns.iter().zip(&table.data) {
            let value = match &column[i] {
                Cell::Int(value) => Json::Int(*value),
                Cell::Str(value) => Json::Str(value.clone()),
                other => panic!("unexpected coverage value {other:?}"),
            };
            record.insert(name.clone(), value);
        }
        Json::Dict(record)
    };
    Json::List((0..table.nrows()).map(row).collect())
}

/// A value of the CLI's JSON type as `serde_json` reads it from a file.
pub fn json_value(value: &Json) -> Value {
    serde_json::from_str(&value.dumps(None)).unwrap()
}

/// `frame.to_csv(path)`.
pub fn write_csv(path: &Path, frame: &Frame) {
    std::fs::write(path, frame.to_csv("Date")).unwrap();
}

/// `frame * factor`.
pub fn scaled(frame: &Frame, factor: f64) -> Frame {
    Frame::new(frame.index.clone(), frame.columns.clone(), &frame.values * factor)
}

/// `pandas.concat([left, right], axis=1)` for frames with the same dates.
pub fn concat(left: &Frame, right: &Frame) -> Frame {
    assert_eq!(left.index, right.index);
    let columns = left.columns.iter().chain(&right.columns).cloned();
    let values = (0..left.ncols()).map(|j| left.col(j)).chain((0..right.ncols()).map(|j| right.col(j)));
    Frame::from_columns(left.index.clone(), columns.zip(values).collect())
}

/// `frame.drop(frame.index[row])`.
pub fn drop_row(frame: &Frame, row: usize) -> Frame {
    let kept: Vec<usize> = (0..frame.nrows()).filter(|i| *i != row).collect();
    let rows: Vec<Vec<f64>> = kept.iter().map(|i| frame.row(*i)).collect();
    Frame::from_rows(kept.iter().map(|i| frame.index[*i]).collect(), frame.columns.clone(), &rows)
}

/// `json.loads((output / "metadata.json").read_text())`.
pub fn metadata(output: &Path) -> Value {
    serde_json::from_str(&read(&output.join("metadata.json"))).unwrap()
}

pub fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// A CSV file as text: the header row and the data rows.
pub struct Csv {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Csv {
    pub fn read(path: &Path) -> Csv {
        let mut rows = data::read_csv_rows(&read(path)).unwrap();
        let header = rows.remove(0);
        Csv { header, rows }
    }

    /// The column names after the first (index) column.
    pub fn columns(&self) -> &[String] {
        &self.header[1..]
    }

    /// The first column, read as the index.
    pub fn index(&self) -> Vec<&str> {
        self.rows.iter().map(|row| row[0].as_str()).collect()
    }

    pub fn column(&self, name: &str) -> Vec<&str> {
        let j = self.header.iter().position(|header| header == name).unwrap_or_else(|| panic!("no column {name:?}"));
        self.rows.iter().map(|row| row[j].as_str()).collect()
    }

    /// A column as numbers. Empty cells are NaN.
    pub fn floats(&self, name: &str) -> Vec<f64> {
        self.column(name).iter().map(|text| number(text)).collect()
    }

    /// The number in the row whose index is `row`.
    pub fn float(&self, row: &str, column: &str) -> f64 {
        let i = self.index().iter().position(|label| *label == row).unwrap_or_else(|| panic!("no row {row:?}"));
        self.floats(column)[i]
    }
}

fn number(text: &str) -> f64 {
    if text.is_empty() { f64::NAN } else { text.parse().unwrap_or_else(|_| panic!("not a number: {text:?}")) }
}

/// `numpy.isclose`: NaN only equals NaN.
pub fn close(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= atol + rtol * expected.abs()
}

/// `pandas.testing.assert_frame_equal(pd.read_csv(actual), pd.read_csv(expected), atol=atol, rtol=rtol)`:
/// the same header and text, and numbers within the tolerance.
pub fn assert_csv_close(actual: &Path, expected: &Path, atol: f64, rtol: f64) {
    let (a, e) = (Csv::read(actual), Csv::read(expected));
    assert_eq!(a.header, e.header, "{}", actual.display());
    assert_eq!(a.rows.len(), e.rows.len(), "{}", actual.display());
    for (i, (row_a, row_e)) in a.rows.iter().zip(&e.rows).enumerate() {
        for (j, (x, y)) in row_a.iter().zip(row_e).enumerate() {
            match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(x), Ok(y)) => {
                    assert!(close(x, y, atol, rtol), "{} row {i} column {j}: {x} vs {y}", actual.display())
                }
                _ => assert_eq!(x, y, "{} row {i} column {j}", actual.display()),
            }
        }
    }
}

/// `pandas.testing.assert_frame_equal(pd.read_csv(path, index_col=0, parse_dates=True), expected,
/// check_freq=False)`, with the default tolerance of pandas.
pub fn assert_csv_frame(path: &Path, expected: &Frame) {
    let csv = Csv::read(path);
    assert_eq!(csv.header[0], "Date", "{}", path.display());
    assert_eq!(csv.columns(), expected.columns, "{}", path.display());
    let dates: Vec<String> = expected.index.iter().map(|date| iso(*date)).collect();
    assert_eq!(csv.index(), dates, "{}", path.display());
    for (j, name) in expected.columns.iter().enumerate() {
        for (i, (actual, wanted)) in csv.floats(name).into_iter().zip(expected.col(j)).enumerate() {
            assert!(close(actual, wanted, 1e-8, 1e-5), "{} {name} row {i}: {actual} vs {wanted}", path.display());
        }
    }
}
