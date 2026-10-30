//! Reproducible command-line analysis and report export.
//!
//! Port of `efficient_frontier/__main__.py`. The flags, defaults, checks,
//! messages and exported files are those of the Python release. Network
//! access goes through [`Sources`], so tests can replace every download.

use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use clap::{ArgGroup, Parser};

use crate::backtest;
use crate::benchmarks::{self, BenchmarkComparison};
use crate::core::{self, Analysis};
use crate::error::Result;
use crate::evidence;
use crate::frame::{Cell, Date, Frame, Table};
use crate::profiles;
use crate::pyfmt::{self, Dict, Json};
use crate::selection::{self, SelectionOptions};
use crate::selection_report::{selection_metrics_frame, selection_report_zip};
use crate::value_error;
use crate::{benchmark_report, data, historical, historical_report, membership, presentation, universe};

/// The command-line options. Help texts follow the Python release.
#[derive(Clone, Debug, Parser)]
#[command(
    name = "efficient-frontier",
    version,
    about = "Build portfolio frontiers, market comparisons, and historical evidence.",
    group(ArgGroup::new("source").args(["csv", "tickers", "original_holdings", "nasdaq100", "sp500_nasdaq100"])),
    group(ArgGroup::new("membership").args(["historical_membership", "membership_csv", "historical_demo"])),
    group(ArgGroup::new("benchmark").args(["benchmark_csv", "download_benchmarks", "no_benchmarks"]))
)]
pub struct Args {
    /// Adjusted prices: Date,ASSET1,ASSET2,…
    #[arg(long)]
    pub csv: Option<PathBuf>,
    /// Comma-separated Yahoo Finance symbols; no ticker-count cap
    #[arg(long)]
    pub tickers: Option<String>,
    /// Use the original 60 tickers from spy_holdings.ods
    #[arg(long)]
    pub original_holdings: bool,
    /// Fetch current Nasdaq-100 members and retain securities with complete prices for the selected period
    #[arg(long)]
    pub nasdaq100: bool,
    /// With --target-volatility, use the deduplicated S&P 500 (SPY holdings proxy) and Nasdaq-100 stock universe
    #[arg(long = "sp500-nasdaq100")]
    pub sp500_nasdaq100: bool,
    /// Use bundled historical Nasdaq-100 membership with supplied --csv prices; stays offline
    #[arg(long)]
    pub historical_membership: bool,
    /// Complete dated membership snapshots; requires --csv prices and --membership-end
    #[arg(long)]
    pub membership_csv: Option<PathBuf>,
    /// Offline synthetic study with changing membership and sparse price history
    #[arg(long)]
    pub historical_demo: bool,
    /// Final covered date for a custom membership CSV, YYYY-MM-DD
    #[arg(long)]
    pub membership_end: Option<String>,
    /// Offline USD benchmark prices: Date,SPY,QQQ on exactly the asset dates; --target-volatility also accepts one ETF
    #[arg(long)]
    pub benchmark_csv: Option<PathBuf>,
    /// Download SPY and QQQ from Yahoo, including for CSV portfolios; unavailable for synthetic demo data
    #[arg(long)]
    pub download_benchmarks: bool,
    /// Skip market benchmark comparisons
    #[arg(long)]
    pub no_benchmarks: bool,
    /// Declare the common asset-price currency; default USD. Benchmarks require USD. No currency conversion.
    #[arg(long, default_value = "USD")]
    pub currency: String,
    #[arg(long, default_value = "2020-01-01")]
    pub start: String,
    /// Yahoo end date, exclusive; defaults to today
    #[arg(long)]
    pub end: Option<String>,
    #[arg(long, default_value_t = 0.7)]
    pub train_fraction: f64,
    /// Annual decimal, e.g. .02
    #[arg(long, default_value_t = 0.02)]
    pub risk_free_rate: f64,
    /// Per-asset cap as a decimal; default .10 with --target-volatility, otherwise 1 (no additional cap)
    #[arg(long)]
    pub max_weight: Option<f64>,
    /// Run single-portfolio selection with an annual volatility target as a decimal, e.g. .20 for 20 percent; omit a source flag to use the offline demo
    #[arg(long)]
    pub target_volatility: Option<f64>,
    /// Covariance shrinkage toward zero correlations: 0 uses sample covariance; 1 keeps only individual variances. Default 0 with --target-volatility, otherwise .1
    #[arg(long)]
    pub shrinkage: Option<f64>,
    /// Observations per year: 252 trading days, 365 calendar days, 52 weeks, or 12 months; no resampling
    #[arg(long)]
    pub periods_per_year: Option<f64>,
    /// Compare buy-and-hold, fixed rebalancing, expanding and rolling windows
    #[arg(long)]
    pub backtests: bool,
    /// Legacy backtest trading interval in observations; default 21. Not used by the buy-and-hold stock model
    #[arg(long)]
    pub rebalance_every: Option<i64>,
    /// Rolling estimation returns; default is 126 for historical membership, otherwise the initial training length
    #[arg(long)]
    pub rolling_window: Option<i64>,
    /// Backtest fee per unit of bought or sold notional, in basis points
    #[arg(long, default_value_t = 10.0)]
    pub cost_bps: f64,
    #[arg(long, default_value = "results/latest")]
    pub output: PathBuf,
}

/// The shared model settings of every mode, after the defaults of `main()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub train_fraction: f64,
    pub risk_free_rate: f64,
    pub max_weight: f64,
    pub shrinkage: f64,
    pub periods_per_year: f64,
}

impl Settings {
    fn entries(&self) -> [(&'static str, f64); 5] {
        [
            ("train_fraction", self.train_fraction),
            ("risk_free_rate", self.risk_free_rate),
            ("max_weight", self.max_weight),
            ("shrinkage", self.shrinkage),
            ("periods_per_year", self.periods_per_year),
        ]
    }
}

/// Every network download of the CLI. Tests replace these with offline fakes.
pub trait Sources {
    /// `fetch_sp500_nasdaq100()` or `fetch_nasdaq100()`, then `download_universe_prices`.
    fn universe(&mut self, sp500_nasdaq100: bool, start: Date, end: Date) -> Result<universe::UniversePrices>;
    fn download_prices(&mut self, tickers: &[String], start: Date, end: Date) -> Result<Frame>;
    fn download_benchmarks(&mut self, dates: &[Date]) -> Result<Frame>;
    fn original_tickers(&mut self) -> Result<Vec<String>>;
    fn demo_prices(&mut self) -> Frame;
}

/// The real downloads.
pub struct LiveSources;

impl Sources for LiveSources {
    fn universe(&mut self, sp500_nasdaq100: bool, start: Date, end: Date) -> Result<universe::UniversePrices> {
        let snapshot = if sp500_nasdaq100 { universe::fetch_sp500_nasdaq100()? } else { universe::fetch_nasdaq100()? };
        universe::download_universe_prices(&snapshot, start, end, None)
    }

    fn download_prices(&mut self, tickers: &[String], start: Date, end: Date) -> Result<Frame> {
        data::download_prices(tickers, start, end)
    }

    fn download_benchmarks(&mut self, dates: &[Date]) -> Result<Frame> {
        data::download_benchmarks(dates)
    }

    fn original_tickers(&mut self) -> Result<Vec<String>> {
        data::original_tickers(None, true)
    }

    fn demo_prices(&mut self) -> Frame {
        data::demo_prices()
    }
}

/// Files that earlier runs of any mode can leave in the output folder.
const GENERATED_FILES: [&str; 49] = [
    "weights.csv",
    "frontier.csv",
    "frontier_weights.csv",
    "holdout_metrics.csv",
    "holdout_curve.csv",
    "risk_summary.csv",
    "risk_contributions.csv",
    "membership.csv",
    "equity.csv",
    "metrics.csv",
    "cash_weights.csv",
    "coverage.csv",
    "LICENSE.n100tickers",
    "findings.md",
    "backtest_metrics.csv",
    "backtest_curve.csv",
    "latest_profile_summary.csv",
    "latest_profile_weights.csv",
    "latest_profile_windows.csv",
    "latest_profile_window_returns.csv",
    "latest_profile_frontier.csv",
    "latest_profile_frontier_weights.csv",
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
    "universe_coverage.csv",
    "policy.json",
    "latest_weights.csv",
    "latest_estimates.csv",
    "fit_estimates.csv",
    "eligibility.csv",
    "latest/frontier.csv",
    "latest/frontier_weights.csv",
    "latest/fitted_comparison.csv",
    "out_of_sample/windows.csv",
    "out_of_sample/metrics.csv",
    "out_of_sample/weights.csv",
    "out_of_sample/eligibility.csv",
    "validation/metrics.csv",
    "validation/equity.csv",
    "test/metrics.csv",
    "test/equity.csv",
];

/// `NN_kind.csv` for a two-digit number and a known kind.
fn numbered_export(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() > 3
        && bytes[0].is_ascii_digit()
        && bytes[1].is_ascii_digit()
        && bytes[2] == b'_'
        && ["holdings.csv", "allocations.csv", "trades.csv"].contains(&&name[3..])
}

/// Replace generated results, including files from the other study modes.
///
/// Write `report.zip`, remove known generated files that the new bundle does
/// not contain, and extract the bundle. Unrelated files stay.
pub fn write_report_bundle(bundle: &[u8], output: &Path) -> Result<()> {
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("report.zip"), bundle)?;
    let mut archive = zip::ZipArchive::new(Cursor::new(bundle))
        .map_err(|error| value_error!("The report archive is invalid: {error}"))?;
    let current: Vec<String> = archive.file_names().filter_map(|name| name.ok().map(|name| name.to_string())).collect();
    let mut previous: Vec<PathBuf> = GENERATED_FILES.iter().map(|name| output.join(name)).collect();
    for folder in ["backtests", "validation", "test"] {
        if let Ok(entries) = std::fs::read_dir(output.join(folder)) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if numbered_export(&name) {
                    previous.push(entry.path());
                }
            }
        }
    }
    for path in previous {
        let relative = path.strip_prefix(output).unwrap_or(&path).to_string_lossy().replace('\\', "/");
        if path.is_file() && !current.contains(&relative) {
            std::fs::remove_file(&path)?;
        }
    }
    for index in 0..archive.len() {
        let mut file =
            archive.by_index(index).map_err(|error| value_error!("The report archive is invalid: {error}"))?;
        let Some(relative) = file.enclosed_name() else { continue };
        let target = output.join(relative);
        if file.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        std::fs::write(&target, content)?;
    }
    Ok(())
}

/// POSIX shell quoting, as Python's `shlex.join`.
pub fn shlex_join(parts: &[String]) -> String {
    parts
        .iter()
        .map(|part| {
            let safe = !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c));
            if safe { part.clone() } else { format!("'{}'", part.replace('\'', "'\"'\"'")) }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Text for `print(frame.round(decimals).to_string())`, laid out as pandas does.
///
/// Numeric headers and values get a leading sign space, float columns share the
/// fewest decimals that keep every rounded value, columns are right-aligned
/// and joined by one space, and every line is padded to the full width.
pub fn console_table(table: &Table, decimals: usize) -> String {
    let rows = table.nrows();
    let named_index = table.index_names.iter().any(Option::is_some);
    let mut index: Vec<String> = Vec::new();
    if named_index {
        index.push(table.index_names.iter().map(|name| name.clone().unwrap_or_default()).collect::<Vec<_>>().join(" "));
    }
    index.extend((0..rows).map(|i| table.index.iter().map(|level| level[i].py_str()).collect::<Vec<_>>().join(" ")));
    let index_width = index.iter().map(|text| text.chars().count()).max().unwrap_or(0);
    let mut index_column = vec![String::new()];
    index_column.extend(index.iter().map(|text| pad_right(text, index_width)));
    let mut columns = vec![index_column];
    for (name, cells) in table.columns.iter().zip(&table.data) {
        let numeric =
            cells.iter().all(|cell| matches!(cell, Cell::Float(_) | Cell::Int(_) | Cell::Bool(_) | Cell::Null));
        let header = if numeric { format!(" {name}") } else { name.clone() };
        let values = if cells.iter().all(|cell| matches!(cell, Cell::Float(_) | Cell::Null)) {
            float_texts(cells, decimals)
        } else {
            cells
                .iter()
                .map(|cell| match cell {
                    Cell::Int(value) if *value >= 0 => format!(" {value}"),
                    Cell::Int(value) => value.to_string(),
                    other => format!(" {}", other.py_str()),
                })
                .collect()
        };
        let width = values.iter().chain([&header]).map(|text| text.chars().count()).max().unwrap_or(0);
        let mut column = vec![pad_left(&header, width)];
        if named_index {
            column.push(String::new());
        }
        column.extend(values.iter().map(|text| pad_left(text, width)));
        columns.push(column);
    }
    // pandas `adjoin(1, ...)`: every column but the last is left-justified to its width plus one space.
    let widths: Vec<usize> = columns
        .iter()
        .enumerate()
        .map(|(k, column)| {
            column.iter().map(|text| text.chars().count()).max().unwrap_or(0) + usize::from(k + 1 < columns.len())
        })
        .collect();
    (0..columns[0].len())
        .map(|line| {
            columns.iter().zip(&widths).map(|(column, width)| pad_right(&column[line], *width)).collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// pandas float column text after `round(decimals)`: six decimals, then trailing
/// zeros trimmed together, keeping one decimal; a leading space for a nonnegative sign.
fn float_texts(cells: &[Cell], decimals: usize) -> Vec<String> {
    let scale = 10f64.powi(decimals as i32);
    let mut texts: Vec<Option<String>> = cells
        .iter()
        .map(|cell| {
            let value = cell.as_f64().unwrap_or(f64::NAN);
            if value.is_nan() {
                return None;
            }
            let rounded = (value * scale).round_ties_even() / scale;
            let text = format!("{rounded:.6}");
            Some(if text.starts_with('-') { text } else { format!(" {text}") })
        })
        .collect();
    while texts.iter().flatten().count() > 0 && texts.iter().flatten().all(|text| text.ends_with('0')) {
        for text in texts.iter_mut().flatten() {
            text.pop();
        }
    }
    texts
        .into_iter()
        .map(|text| match text {
            Some(text) if text.ends_with('.') => format!("{text}0"),
            Some(text) => text,
            None => "NaN".to_string(),
        })
        .collect()
}

fn pad_left(text: &str, width: usize) -> String {
    format!("{}{text}", " ".repeat(width.saturating_sub(text.chars().count())))
}

fn pad_right(text: &str, width: usize) -> String {
    format!("{text}{}", " ".repeat(width.saturating_sub(text.chars().count())))
}

/// A Yahoo date argument, parsed as pandas reads a timestamp.
fn parse_date(text: &str) -> Result<Date> {
    data::parse_price_dates(&[text.to_string()])
        .ok()
        .and_then(|dates| dates.first().copied())
        .ok_or_else(|| value_error!("Choose valid start and end dates, with the start before the end."))
}

fn today() -> String {
    chrono::Local::now().date_naive().format("%Y-%m-%d").to_string()
}

/// Selection mode: one historical-mean stock model with a buy-and-hold replay.
fn selection_cli(args: &Args, settings: &Settings, sources: &mut dyn Sources, out: &mut dyn Write) -> Result<i32> {
    if args.rolling_window.is_some() {
        return Err(value_error!("--target-volatility buys once and holds; omit --rolling-window."));
    }
    if args.rebalance_every.is_some() {
        return Err(value_error!(
            "--target-volatility buys once and holds; omit --rebalance-every. That option applies to legacy backtests only."
        ));
    }
    let target_volatility = args.target_volatility.expect("selection mode has a target");
    let synthetic = args.csv.is_none()
        && args.tickers.is_none()
        && !args.original_holdings
        && !args.nasdaq100
        && !args.sp500_nasdaq100;
    if synthetic && (args.benchmark_csv.is_some() || args.download_benchmarks) {
        return Err(value_error!(
            "Synthetic demo data cannot use real market benchmarks. Select --csv or a market source first."
        ));
    }
    let end_text = args.end.clone().unwrap_or_else(today);
    let mut metadata = Dict::new();
    let (prices, source, universe_benchmarks) = if args.sp500_nasdaq100 || args.nasdaq100 {
        let loaded = sources.universe(args.sp500_nasdaq100, parse_date(&args.start)?, parse_date(&end_text)?)?;
        metadata.update(&loaded.metadata);
        let source =
            if args.sp500_nasdaq100 { "S&P 500 + Nasdaq-100 · Yahoo Finance" } else { "Nasdaq-100 · Yahoo Finance" };
        (loaded.prices, source.to_string(), Some((loaded.benchmarks, loaded.coverage)))
    } else if let Some(path) = &args.csv {
        (data::load_csv_path(path)?, format!("CSV: {}", file_name(path)), None)
    } else if args.tickers.is_some() || args.original_holdings {
        let symbols = if args.original_holdings {
            sources.original_tickers()?
        } else {
            data::parse_tickers(args.tickers.as_deref().unwrap_or_default())
        };
        let start = parse_date(&args.start)?;
        let end = parse_date(&end_text)?;
        (sources.download_prices(&symbols, start, end)?, "Yahoo Finance".to_string(), None)
    } else {
        (sources.demo_prices(), "Demo · synthetic".to_string(), None)
    };
    let (prices, embedded) = selection::split_stock_prices(&prices)?;
    let currency = args.currency.trim().to_uppercase();
    metadata.insert("source", source.as_str());
    metadata.insert("price_currency", currency.as_str());
    metadata.insert(
        "currency_assumption",
        format!("All prices are declared in {currency}. No currency conversion occurs."),
    );
    if let Some(embedded) = &embedded {
        metadata.insert("excluded_candidate_symbols", Json::from(embedded.columns.clone()));
        metadata.insert(
            "candidate_note",
            format!("Comparison ETFs excluded from candidate holdings: {}.", embedded.columns.join(", ")),
        );
    }
    if let Some((_, coverage)) = &universe_benchmarks {
        metadata.insert("universe_coverage", records(coverage));
    }
    if universe_benchmarks.is_some() || source == "Yahoo Finance" {
        metadata.insert("requested_start", args.start.as_str());
        metadata.insert("requested_end_exclusive", end_text.as_str());
    }
    let mut benchmark_prices = None;
    if args.no_benchmarks {
        metadata.insert(
            "benchmarks_note",
            "Market benchmarks are disabled by --no-benchmarks; ETF advantage cannot be established.",
        );
    } else if synthetic {
        metadata.insert(
            "benchmarks_note",
            "Synthetic demo data has no real market comparison; ETF advantage cannot be established.",
        );
    } else if currency != "USD" {
        metadata.insert(
            "benchmarks_note",
            format!(
                "Market comparisons require USD prices. Declared currency: {currency}. No currency conversion occurs."
            ),
        );
    } else {
        let attempt = (|| -> Result<Option<Frame>> {
            let chosen = if let Some(path) = &args.benchmark_csv {
                let frame = data::load_csv_path(path)?;
                metadata.insert("benchmark_source", format!("CSV: {}", file_name(path)));
                Some(frame)
            } else if let Some(embedded) = &embedded {
                metadata.insert("benchmark_source", format!("Input columns: {}", embedded.columns.join(", ")));
                Some(embedded.clone())
            } else if args.download_benchmarks {
                let frame = sources.download_benchmarks(&prices.index)?;
                metadata.insert("benchmark_source", "Yahoo Finance");
                Some(frame)
            } else if let Some((benchmarks, _)) = &universe_benchmarks {
                metadata.insert("benchmark_source", "Yahoo Finance · universe download");
                Some(benchmarks.clone())
            } else if source == "Yahoo Finance" {
                let frame = sources.download_benchmarks(&prices.index)?;
                metadata.insert("benchmark_source", "Yahoo Finance");
                Some(frame)
            } else {
                metadata.insert(
                    "benchmarks_note",
                    "CSV analysis stays offline. Supply --benchmark-csv to establish ETF comparisons.",
                );
                None
            };
            chosen.map(|frame| selection::validate_selection_benchmarks(&frame, &prices.index)).transpose()
        })();
        match attempt {
            Ok(frame) => benchmark_prices = frame,
            Err(error) => {
                metadata.insert("benchmarks_note", format!("Market benchmarks are unavailable: {error}"));
            }
        }
    }
    let study = selection::run_selection(
        &prices,
        benchmark_prices.as_ref(),
        &SelectionOptions {
            target_volatility,
            max_weight: settings.max_weight,
            train_fraction: settings.train_fraction,
            cost_bps: args.cost_bps,
            shrinkage: settings.shrinkage,
            periods_per_year: settings.periods_per_year,
            risk_free_rate: settings.risk_free_rate,
        },
    )?;
    let export_directory = absolute(&args.output);
    let mut reproduce = vec![
        "efficient-frontier".to_string(),
        "--csv".into(),
        export_directory.join("prices.csv").to_string_lossy().to_string(),
        "--target-volatility".into(),
        pyfmt::repr(target_volatility),
    ];
    for (key, value) in settings.entries() {
        reproduce.push(format!("--{}", key.replace('_', "-")));
        // argparse keeps the integer default 252 when the flag is absent.
        let default_periods = key == "periods_per_year" && args.periods_per_year.is_none();
        reproduce.push(if default_periods { "252".to_string() } else { pyfmt::repr(value) });
    }
    reproduce.extend(["--cost-bps".into(), pyfmt::repr(args.cost_bps), "--currency".into(), currency.clone()]);
    if benchmark_prices.is_some() {
        reproduce.extend([
            "--benchmark-csv".into(),
            export_directory.join("benchmark_prices.csv").to_string_lossy().to_string(),
        ]);
    } else {
        reproduce.push("--no-benchmarks".into());
    }
    reproduce.extend(["--output".into(), export_directory.join("reproduced").to_string_lossy().to_string()]);
    metadata.insert("reproduce_command", shlex_join(&reproduce));
    metadata.insert(
        "reproduce_note",
        "Run with an efficient-frontier build of the same version, for example cargo run --release --bin efficient-frontier -- \
         followed by these arguments in the project checkout. Update file paths if the export folder moves.",
    );
    write_report_bundle(&selection_report_zip(&study, &prices, &metadata, None, None)?, &args.output)?;
    writeln!(out, "Source: {source}")?;
    if let Some(Json::Str(note)) = metadata.get("candidate_note") {
        writeln!(out, "Note: {note}")?;
    }
    writeln!(
        out,
        "Annual volatility target: {}; holding cap: {}",
        pyfmt::format(target_volatility, ".1%"),
        pyfmt::format(settings.max_weight, ".1%")
    )?;
    writeln!(out, "One rule: maximize historical mean return within the volatility budget and holding limit.")?;
    writeln!(out, "{}", if study.supported { "Historical checks passed." } else { "Historical checks did not pass." })?;
    let mut printed: Vec<&String> = Vec::new();
    for note in study.reasons.iter().chain(&study.warnings) {
        if !printed.contains(&note) {
            writeln!(out, "Note: {note}")?;
            printed.push(note);
        }
    }
    if let Some(Json::Str(note)) = metadata.get("benchmarks_note") {
        writeln!(out, "Note: {note}")?;
    }
    writeln!(out, "\nHistorical replay (buy once and hold):")?;
    writeln!(out, "{}", console_table(&selection_metrics_frame(&study.test), 4))?;
    writeln!(
        out,
        "Earlier research on these dates can introduce hindsight. A historical test alone is not unseen prospective evidence."
    )?;
    writeln!(out, "Report: {}", absolute(&args.output.join("report.html")).display())?;
    Ok(0)
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(|| path.to_string_lossy().to_string(), |name| name.to_string_lossy().to_string())
}

/// An absolute path without resolving symbolic links of missing folders.
fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// `DataFrame.to_dict(orient="records")` for metadata.
fn records(table: &Table) -> Json {
    Json::List(
        (0..table.nrows())
            .map(|i| {
                let mut row = Dict::new();
                for (name, column) in table.columns.iter().zip(&table.data) {
                    let value = match &column[i] {
                        Cell::Null => Json::Null,
                        Cell::Bool(value) => Json::Bool(*value),
                        Cell::Int(value) => Json::Int(*value),
                        Cell::Float(value) => Json::Float(*value),
                        Cell::Str(value) => Json::Str(value.clone()),
                        Cell::Date(value) => Json::Str(crate::frame::iso(*value)),
                    };
                    row.insert(name.clone(), value);
                }
                Json::Dict(row)
            })
            .collect(),
    )
}

/// A nonnegative count from the command line. Negative values become zero, which
/// the model rejects with the same message as Python does for any invalid count.
fn count(value: i64) -> usize {
    usize::try_from(value).unwrap_or(0)
}

/// Historical-membership mode: rolling-window portfolios with dated rosters.
fn historical_cli(args: &Args, settings: &Settings, out: &mut dyn Write) -> Result<i32> {
    if args.tickers.is_some() || args.original_holdings || args.nasdaq100 {
        return Err(value_error!(
            "Historical membership uses supplied --csv prices or --historical-demo; automatic price downloads are unavailable."
        ));
    }
    if args.benchmark_csv.is_some() || args.download_benchmarks {
        return Err(value_error!("Historical membership does not support market benchmark comparisons."));
    }
    let (prices, membership, source) = if args.historical_demo {
        if args.csv.is_some() || args.membership_end.is_some() {
            return Err(value_error!(
                "--historical-demo supplies both prices and membership; omit --csv and --membership-end."
            ));
        }
        let (prices, membership) = historical::historical_demo();
        (prices, membership, "Historical membership demo · synthetic".to_string())
    } else {
        let Some(csv) = &args.csv else {
            return Err(value_error!(
                "Historical membership requires --csv with adjusted prices for stable security identities."
            ));
        };
        let membership = if let Some(path) = &args.membership_csv {
            let Some(end) = &args.membership_end else {
                return Err(value_error!(
                    "--membership-csv requires --membership-end to declare the final covered date."
                ));
            };
            membership::load_membership_csv_path(path, end.as_str(), Some(&format!("CSV: {}", file_name(path))))?
        } else {
            if args.membership_end.is_some() {
                return Err(value_error!("--membership-end applies only to --membership-csv."));
            }
            membership::bundled_membership()
        };
        (historical::load_historical_prices_path(csv)?, membership, format!("CSV: {}", file_name(csv)))
    };
    let mut result = historical::run_historical_backtests(
        &prices,
        &membership,
        &historical::HistoricalOptions {
            train_fraction: settings.train_fraction,
            rolling_window: Some(args.rolling_window.map_or(126, count)),
            rebalance_every: count(args.rebalance_every.unwrap_or(21)),
            risk_free_rate: settings.risk_free_rate,
            max_weight: settings.max_weight,
            shrinkage: settings.shrinkage,
            cost_bps: args.cost_bps,
            periods_per_year: settings.periods_per_year,
        },
    )?;
    let currency = args.currency.trim().to_uppercase();
    result.study.settings.update(&crate::dict! {
        "source" => source.as_str(),
        "price_currency" => currency.as_str(),
        "currency_assumption" => "All prices must share the declared currency. No currency conversion occurs.",
    });
    let files = historical_report::build_historical_report(&prices, &membership, &result);
    write_report_bundle(&historical_report::historical_report_zip(&files)?, &args.output)?;
    writeln!(out, "Source: {source}")?;
    writeln!(out, "Declared price currency: {currency} (no currency conversion)")?;
    writeln!(
        out,
        "Membership coverage: {} to {}",
        crate::frame::iso(membership.coverage_start()),
        crate::frame::iso(membership.coverage_end())
    )?;
    for warning in &result.study.warnings {
        writeln!(out, "Note: {warning}")?;
    }
    writeln!(out, "\nHistorical membership · rolling-window portfolios after trading costs:")?;
    writeln!(out, "{}", console_table(&result.study.metrics, 4))?;
    writeln!(out, "Eligibility decisions: {}", absolute(&args.output.join("coverage.csv")).display())?;
    writeln!(out, "Report: {}", absolute(&args.output.join("report.html")).display())?;
    Ok(0)
}

/// Holdout metrics as the Python DataFrame: index `portfolio`, one column per measure.
fn holdout_table(analysis: &Analysis) -> Table {
    let names: Vec<&String> = analysis.holdout_metrics.keys().collect();
    let mut table = Table::labelled(Some("portfolio"), &names);
    for (k, field) in crate::metrics::PERFORMANCE_FIELDS.iter().enumerate() {
        table.push(*field, analysis.holdout_metrics.values().map(|m| Cell::Float(m.fields()[k].1)).collect());
    }
    table
}

/// The legacy classic analysis, backtests, benchmarks and evidence.
fn legacy_cli(args: &Args, settings: &Settings, sources: &mut dyn Sources, out: &mut dyn Write) -> Result<i32> {
    let currency = args.currency.trim().to_uppercase();
    let synthetic = args.csv.is_none() && args.tickers.is_none() && !args.original_holdings && !args.nasdaq100;
    if synthetic && (args.benchmark_csv.is_some() || args.download_benchmarks) {
        return Err(value_error!(
            "Synthetic demo data cannot use real market benchmarks. Select --csv or --tickers first."
        ));
    }
    let end_text = args.end.clone().unwrap_or_else(today);
    let mut universe = None;
    let (prices, source) = if args.nasdaq100 {
        let loaded = sources.universe(false, parse_date(&args.start)?, parse_date(&end_text)?)?;
        if loaded.prices.ncols() < 2 {
            let requested = loaded.coverage.nrows();
            let included = loaded.prices.ncols();
            return Err(value_error!(
                "Only {included} of {requested} Nasdaq-100 securities have complete prices. \
                 At least two are required. {} securities were excluded.",
                requested - included
            ));
        }
        let prices = loaded.prices.clone();
        universe = Some(loaded);
        (prices, "Nasdaq-100 · Yahoo Finance".to_string())
    } else if let Some(path) = &args.csv {
        (data::load_csv_path(path)?, format!("CSV: {}", file_name(path)))
    } else if args.tickers.is_some() || args.original_holdings {
        let tickers = if args.original_holdings {
            sources.original_tickers()?
        } else {
            data::parse_tickers(args.tickers.as_deref().unwrap_or_default())
        };
        let (start, end) = (parse_date(&args.start)?, parse_date(&end_text)?);
        (sources.download_prices(&tickers, start, end)?, "Yahoo Finance".to_string())
    } else {
        (sources.demo_prices(), "Demo · synthetic".to_string())
    };
    let analysis = core::analyze(
        &prices,
        &core::AnalyzeOptions {
            train_fraction: settings.train_fraction,
            risk_free_rate: settings.risk_free_rate,
            max_weight: settings.max_weight,
            shrinkage: settings.shrinkage,
            periods_per_year: settings.periods_per_year,
            ..Default::default()
        },
    )?;
    let latest_profiles = if prices.nrows() >= 7 {
        Some(profiles::build_profiles(
            &prices,
            &profiles::ProfileOptions {
                risk_free_rate: settings.risk_free_rate,
                max_weight: settings.max_weight,
                shrinkage: settings.shrinkage,
                periods_per_year: settings.periods_per_year,
                return_objective: profiles::ReturnObjective::HistoricalMean,
                ..Default::default()
            },
        )?)
    } else {
        None
    };
    let train = analysis.train_returns.nrows() as i64;
    let profile_backtests = train.min(args.rolling_window.unwrap_or(train)) >= 6;
    let mut study = if args.backtests {
        Some(backtest::run_backtests(
            &prices,
            &backtest::BacktestOptions {
                train_fraction: settings.train_fraction,
                risk_free_rate: settings.risk_free_rate,
                max_weight: settings.max_weight,
                shrinkage: settings.shrinkage,
                rebalance_every: count(args.rebalance_every.unwrap_or(21)),
                rolling_window: args.rolling_window.map(count),
                cost_bps: args.cost_bps,
                periods_per_year: settings.periods_per_year,
                include_profiles: profile_backtests,
            },
        )?)
    } else {
        None
    };
    if let Some(study) = study.as_mut().filter(|_| !profile_backtests) {
        study.warnings.push(
            "Risk-profile backtests need at least six returns in both the initial and rolling fit windows. \
             Only the original targets are shown."
                .to_string(),
        );
    }
    let mut metadata = universe.as_ref().map_or_else(Dict::new, |loaded| loaded.metadata.clone());
    metadata.insert("source", source.as_str());
    metadata.insert("price_currency", currency.as_str());
    metadata.insert(
        "currency_assumption",
        format!(
            "The user declares all asset prices in {currency}. No currency conversion occurs. \
             USD benchmark ETFs require the same currency basis."
        ),
    );
    for (key, value) in settings.entries() {
        metadata.insert(key, value);
    }
    if let Some(loaded) = &universe {
        metadata.insert("universe_coverage", records(&loaded.coverage));
    }
    if latest_profiles.is_none() {
        metadata.insert(
            "latest_profiles_note",
            "Latest profiles need at least seven prices for three windows with two returns each.",
        );
    }
    if source == "Yahoo Finance" || args.nasdaq100 {
        metadata.insert("requested_start", args.start.as_str());
        metadata.insert("requested_end_exclusive", end_text.as_str());
    }
    let mut benchmarks = None;
    let mut evidence = None;
    if args.no_benchmarks {
        metadata.insert("benchmarks_note", "Market benchmarks are disabled by --no-benchmarks.");
    } else if synthetic {
        metadata.insert("benchmarks_note", "Synthetic demo data has no real market benchmark comparison.");
    } else if currency != "USD" {
        metadata.insert(
            "benchmarks_note",
            format!(
                "Market benchmarks require USD prices. The declared currency is {currency}, and no currency conversion occurs."
            ),
        );
    } else {
        let attempt = (|| -> Result<Option<BenchmarkComparison>> {
            let chosen = if let Some(path) = &args.benchmark_csv {
                metadata.insert("benchmark_source", format!("CSV: {}", file_name(path)));
                Some(data::load_csv_path(path)?)
            } else if args.download_benchmarks {
                metadata.insert("benchmark_source", "Yahoo Finance");
                Some(sources.download_benchmarks(&prices.index)?)
            } else if let Some(loaded) = &universe {
                metadata.insert("benchmark_source", "Yahoo Finance · Nasdaq-100 download");
                Some(loaded.benchmarks.clone())
            } else if let Some(columns) = prices.select(&data::BENCHMARK_SYMBOLS) {
                metadata.insert("benchmark_source", format!("Asset columns: {}", data::BENCHMARK_SYMBOLS.join(", ")));
                Some(columns)
            } else if source == "Yahoo Finance" {
                metadata.insert("benchmark_source", "Yahoo Finance");
                Some(sources.download_benchmarks(&prices.index)?)
            } else {
                metadata.insert(
                    "benchmarks_note",
                    "CSV analysis stays offline. Supply --benchmark-csv or --download-benchmarks, or include SPY and QQQ price columns.",
                );
                None
            };
            let Some(frame) = chosen else { return Ok(None) };
            let frame = data::validate_benchmark_prices(&frame, &prices.index)?;
            Ok(Some(benchmarks::compare_benchmarks(
                &frame,
                &prices,
                &analysis,
                study.as_ref(),
                latest_profiles.as_ref(),
                args.risk_free_rate,
            )?))
        })();
        match attempt {
            Ok(comparison) => benchmarks = comparison,
            Err(error) => {
                metadata.insert("benchmarks_note", format!("Market benchmarks are unavailable: {error}"));
            }
        }
    }
    if let Some(comparison) = &benchmarks {
        let (strategy_equity, benchmark_equity) = match (&study, &comparison.backtest_equity) {
            (Some(study), Some(backtest)) => (&study.equity, backtest),
            _ => (&analysis.equity, &comparison.holdout_equity),
        };
        let options = evidence::EvidenceOptions {
            risk_free_rate: args.risk_free_rate,
            periods_per_year: settings.periods_per_year,
            delayed_entry: study.is_some(),
        };
        match evidence::analyze_evidence(strategy_equity, benchmark_equity, &options) {
            Ok(result) => evidence = Some(result),
            Err(error) => {
                metadata.insert("evidence_note", format!("Historical evidence is unavailable: {error}"));
            }
        }
    }
    let parts = presentation::ReportParts {
        study: study.as_ref(),
        latest_profiles: latest_profiles.as_ref(),
        benchmarks: benchmarks.as_ref(),
        evidence: evidence.as_ref(),
        ..Default::default()
    };
    let bundle = presentation::report_zip(&analysis, &prices, &metadata, &parts)?;
    write_report_bundle(&bundle, &args.output)?;

    writeln!(out, "Source: {source}")?;
    writeln!(out, "Declared price currency: {currency} (no currency conversion)")?;
    if let Some(loaded) = &universe {
        let (included, requested) = (prices.ncols(), loaded.coverage.nrows());
        let text = |key: &str| metadata.get(key).map(Json::py_str).unwrap_or_default();
        writeln!(
            out,
            "Nasdaq-100 coverage: {included} of {requested} securities included; {} excluded.",
            requested - included
        )?;
        writeln!(out, "Membership source date: {}", text("universe_source_date"))?;
        writeln!(out, "Membership source: {}", text("universe_source_url"))?;
        writeln!(out, "Note: {}", text("universe_limitation"))?;
        writeln!(out, "Coverage: {}", absolute(&args.output.join("universe_coverage.csv")).display())?;
    }
    if let Some(profiles) = &latest_profiles {
        writeln!(
            out,
            "\nLatest model holdings as of {} (historical mean frontier; full-history fit, not an out-of-sample result):",
            profiles.as_of
        )?;
        writeln!(out, "{}", console_table(&profiles.summary, 4))?;
        for (name, portfolio) in &profiles.portfolios {
            let leaders: Vec<String> = portfolio
                .weights
                .descending_order()
                .into_iter()
                .take(5)
                .map(|i| {
                    format!("{} {}", portfolio.weights.labels[i], pyfmt::format(portfolio.weights.values[i], ".2%"))
                })
                .collect();
            writeln!(out, "{name} largest allocations: {}", leaders.join(", "))?;
        }
        for warning in &profiles.warnings {
            writeln!(out, "Note: {warning}")?;
        }
    } else if let Some(note) = metadata.get("latest_profiles_note") {
        writeln!(out, "Note: {}", note.py_str())?;
    }
    for warning in &analysis.warnings {
        writeln!(out, "Note: {warning}")?;
    }
    writeln!(out, "\nOriginal chronological holdout:")?;
    writeln!(out, "{}", console_table(&holdout_table(&analysis), 4))?;
    if let Some(study) = &study {
        for warning in &study.warnings {
            writeln!(out, "Note: {warning}")?;
        }
        writeln!(out, "\nBacktest comparison (after the selected trading costs):")?;
        writeln!(out, "{}", console_table(&study.metrics, 4))?;
        writeln!(out, "Findings: {}", absolute(&args.output.join("findings.md")).display())?;
    }
    if let Some(comparison) = &benchmarks {
        let source = metadata.get("benchmark_source").map(Json::py_str).unwrap_or_default();
        writeln!(out, "\nMarket benchmarks ({source}; matched dates and entry rules):")?;
        let table = match (&study, &comparison.backtest_metrics) {
            (Some(_), Some(metrics)) => metrics,
            _ => &comparison.holdout_metrics,
        };
        writeln!(out, "{}", console_table(&table.select(&["total_return", "cagr", "volatility", "max_drawdown"]), 4))?;
    }
    if let Some(result) = &evidence {
        if let Some(selected) = benchmark_report::default_evidence_strategy(result) {
            writeln!(out, "\nHistorical evidence for {selected} (fixed display choice, not the best result):")?;
            let rows: Vec<usize> = (0..result.summary.nrows())
                .filter(|i| result.summary.index[0][*i].as_str() == Some(&selected))
                .collect();
            let mut table = result.summary.take_rows(&rows);
            table.index.remove(0);
            table.index_names.remove(0);
            let columns = [
                "annual_advantage",
                "advantage_ci_low",
                "advantage_ci_high",
                "alpha",
                "alpha_pvalue_adjusted",
                "status",
            ];
            writeln!(out, "{}", console_table(&table.select(&columns), 4))?;
        }
        for warning in &result.warnings {
            writeln!(out, "Note: {warning}")?;
        }
    }
    for note in ["benchmarks_note", "evidence_note"] {
        if let Some(value) = metadata.get(note) {
            writeln!(out, "Note: {}", value.py_str())?;
        }
    }
    writeln!(out, "Report: {}", absolute(&args.output.join("report.html")).display())?;
    Ok(0)
}

/// Run the CLI with parsed arguments. Returns the process exit code.
pub fn run(mut args: Args, sources: &mut dyn Sources, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    if args.target_volatility.is_none() && args.rebalance_every.is_none() {
        args.rebalance_every = Some(21);
    }
    let selection_mode = args.target_volatility.is_some();
    let settings = Settings {
        train_fraction: args.train_fraction,
        risk_free_rate: args.risk_free_rate,
        max_weight: args.max_weight.unwrap_or(if selection_mode { 0.1 } else { 1.0 }),
        shrinkage: args.shrinkage.unwrap_or(if selection_mode { 0.0 } else { 0.1 }),
        periods_per_year: args.periods_per_year.unwrap_or(252.0),
    };
    let result = (|| -> Result<i32> {
        let currency = args.currency.trim().to_uppercase();
        if currency.is_empty() {
            return Err(value_error!("currency must identify the common price currency, such as USD or EUR."));
        }
        if args.historical_demo || args.historical_membership || args.membership_csv.is_some() {
            if selection_mode || args.sp500_nasdaq100 {
                return Err(value_error!(
                    "Historical membership is a separate study; omit --target-volatility and --sp500-nasdaq100."
                ));
            }
            return historical_cli(&args, &settings, out);
        }
        if args.membership_end.is_some() {
            return Err(value_error!("--membership-end requires --membership-csv."));
        }
        if selection_mode {
            return selection_cli(&args, &settings, sources, out);
        }
        if args.sp500_nasdaq100 {
            return Err(value_error!(
                "--sp500-nasdaq100 requires --target-volatility, for example .20 for 20% annual volatility."
            ));
        }
        legacy_cli(&args, &settings, sources, out)
    })();
    match result {
        Ok(code) => code,
        Err(error) => {
            let _ = writeln!(err, "Analysis failed: {error}");
            1
        }
    }
}

/// The program entry point: parse arguments and run with real downloads.
pub fn main() -> i32 {
    let args = Args::parse();
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    run(args, &mut LiveSources, &mut stdout.lock(), &mut stderr.lock())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle(names: &[&str]) -> Vec<u8> {
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut archive = zip::ZipWriter::new(&mut buffer);
            for name in names {
                archive.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
                archive.write_all(b"generated\n").unwrap();
            }
            archive.finish().unwrap();
        }
        buffer.into_inner()
    }

    #[test]
    fn reruns_clean_only_known_exports() {
        let folder = tempfile::tempdir().unwrap();
        let generated = [
            "policy.json",
            "latest_weights.csv",
            "latest_estimates.csv",
            "fit_estimates.csv",
            "eligibility.csv",
            "latest/frontier.csv",
            "latest/frontier_weights.csv",
            "latest/fitted_comparison.csv",
            "validation/metrics.csv",
            "validation/equity.csv",
            "validation/01_allocations.csv",
            "test/metrics.csv",
            "test/equity.csv",
            "test/01_holdings.csv",
            "test/01_trades.csv",
            "out_of_sample/windows.csv",
            "out_of_sample/metrics.csv",
            "out_of_sample/weights.csv",
            "out_of_sample/eligibility.csv",
        ];
        let mut first = vec!["report.html"];
        first.extend(generated);
        write_report_bundle(&bundle(&first), folder.path()).unwrap();
        let notes = [
            folder.path().join("notes.txt"),
            folder.path().join("validation/notes.csv"),
            folder.path().join("test/01_custom.csv"),
        ];
        for path in &notes {
            std::fs::write(path, "Keep this user file.\n").unwrap();
        }
        write_report_bundle(&bundle(&["report.html", "weights.csv"]), folder.path()).unwrap();
        for name in generated {
            assert!(!folder.path().join(name).exists(), "{name}");
        }
        for path in &notes {
            assert_eq!(std::fs::read_to_string(path).unwrap(), "Keep this user file.\n");
        }
        assert!(folder.path().join("report.zip").is_file());
    }

    #[test]
    fn shell_quoting_matches_shlex() {
        let parts = ["efficient-frontier", "--csv", "/tmp/my prices/prices.csv", "it's", ""].map(String::from);
        assert_eq!(shlex_join(&parts), "efficient-frontier --csv '/tmp/my prices/prices.csv' 'it'\"'\"'s' ''");
    }

    #[test]
    fn console_tables_match_pandas_layout() {
        let table =
            Table::labelled(Some("strategy"), &["Historical-mean portfolio", "Equal-weight stocks", "S&P 500 (SPY)"])
                .with("total_return", vec![Cell::Float(0.123456), Cell::Float(-0.5), Cell::Null])
                .with("cagr", vec![Cell::Float(1.0), Cell::Float(2.0), Cell::Float(3.0)])
                .with("rebalance_count", vec![Cell::Int(1), Cell::Int(2), Cell::Int(3)])
                .with("name", vec![Cell::from("a"), Cell::from("bb"), Cell::from("ccc")]);
        // The exact output of pandas 3.0.6 for the same frame.
        let expected = "                           total_return  cagr  rebalance_count name\n\
                        strategy                                                           \n\
                        Historical-mean portfolio        0.1235   1.0                1    a\n\
                        Equal-weight stocks             -0.5000   2.0                2   bb\n\
                        S&P 500 (SPY)                       NaN   3.0                3  ccc";
        assert_eq!(console_table(&table, 4), expected);
    }
}
