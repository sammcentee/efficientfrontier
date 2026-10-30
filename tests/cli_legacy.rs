//! The classic analysis of the CLI with market benchmarks, evidence and the Nasdaq-100 source.
//!
//! Ports of the in-process tests of `tests/test_cli.py`. A [`Fake`] replaces
//! the monkeypatched downloads, so every run stays offline.

mod cli_support;

use std::collections::BTreeSet;
use std::path::Path;

use cli_support::*;
use efficient_frontier::frame::{Frame, cells};
use efficient_frontier::universe::{NASDAQ100_URL, UniversePrices};
use efficient_frontier::value_error;

fn ok(run: &Run) {
    assert_eq!(run.code, 0, "{}", run.stderr);
}

fn note(metadata: &serde_json::Value, key: &str) -> String {
    metadata[key].as_str().unwrap_or_else(|| panic!("no {key} in {metadata}")).to_string()
}

fn keep(path: &Path) {
    std::fs::write(path, "Keep this user file.\n").unwrap();
}

fn kept(path: &Path) -> bool {
    read(path) == "Keep this user file.\n"
}

fn yahoo_benchmarks_reuse_existing_columns_or_fetch_without_changing_universe(reuse_columns: bool) {
    let (assets, benchmarks) = benchmark_inputs();
    let prices = if reuse_columns { concat(&assets, &benchmarks) } else { assets };
    let mut fake = Fake { prices: Some(prices.clone()), benchmarks: Some(Ok(benchmarks.clone())), ..Fake::default() };
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("yahoo");
    let run = run_cli(&argv!["--tickers", prices.columns.join(","), "--output", output], &mut fake);
    ok(&run);
    assert_eq!(fake.price_calls, std::slice::from_ref(&prices.columns));
    assert_eq!(fake.benchmark_calls.len(), if reuse_columns { 0 } else { 1 });
    if let Some(dates) = fake.benchmark_calls.first() {
        assert_eq!(dates, &prices.index);
    }
    let metadata = metadata(&output);
    assert_eq!(metadata["price_currency"], "USD");
    assert_eq!(metadata["benchmark_source"], if reuse_columns { "Asset columns: SPY, QQQ" } else { "Yahoo Finance" });
    assert!(metadata.get("benchmarks_note").is_none());
    assert_eq!(Csv::read(&output.join("weights.csv")).index(), prices.columns);
    assert_csv_frame(&output.join("benchmark_prices.csv"), &benchmarks);
    let evidence = Csv::read(&output.join("evidence_summary.csv"));
    let compared: BTreeSet<&str> = evidence.rows.iter().map(|row| row[1].as_str()).collect();
    assert_eq!(compared, BTreeSet::from(["S&P 500 (SPY)", "Nasdaq-100 (QQQ)"]));
    assert!(run.stdout.contains("Historical evidence for Minimum volatility"));
}

#[test]
fn yahoo_benchmarks_fetch_without_changing_universe() {
    yahoo_benchmarks_reuse_existing_columns_or_fetch_without_changing_universe(false);
}

#[test]
fn yahoo_benchmarks_reuse_existing_columns() {
    yahoo_benchmarks_reuse_existing_columns_or_fetch_without_changing_universe(true);
}

#[test]
fn csv_benchmark_source_stays_offline_until_download_is_explicit() {
    let (assets, benchmarks) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("assets.csv"), folder.path().join("report"));
    write_csv(&source, &assets);
    let command = argv!["--csv", source, "--output", output];
    let run = run_cli(&command, &mut Fake::default());
    ok(&run);
    assert!(note(&metadata(&output), "benchmarks_note").contains("CSV analysis stays offline"));
    assert!(run.stdout.contains("--benchmark-csv"));
    assert!(!output.join("benchmark_prices.csv").exists());

    let mut fake = Fake { benchmarks: Some(Ok(benchmarks)), ..Fake::default() };
    ok(&run_cli(&[command, argv!["--download-benchmarks"]].concat(), &mut fake));
    assert_eq!(fake.benchmark_calls, [assets.index]);
    assert!(output.join("benchmark_prices.csv").is_file());
    assert_eq!(metadata(&output)["benchmark_source"], "Yahoo Finance");
}

#[test]
fn benchmark_csv_reproduction_and_cleanup_preserve_unrelated_files() {
    let (assets, benchmarks) = benchmark_inputs();
    // Explicit benchmark input takes priority over benchmark symbols in the asset file.
    let prices = concat(&assets, &scaled(&benchmarks, 2.0));
    let folder = tempfile::tempdir().unwrap();
    let (source, benchmark_source) = (folder.path().join("assets.csv"), folder.path().join("market.csv"));
    write_csv(&source, &prices);
    write_csv(&benchmark_source, &benchmarks);
    let output = folder.path().join("report");
    let run = run_cli(
        &argv!["--csv", source, "--benchmark-csv", benchmark_source, "--backtests", "--output", output],
        &mut Fake::default(),
    );
    ok(&run);
    assert!(run.stdout.contains("Historical evidence for Expanding window · Medium"));
    assert_eq!(metadata(&output)["benchmark_source"], "CSV: market.csv");
    assert_csv_frame(&output.join("benchmark_prices.csv"), &benchmarks);
    let (original_weights, original_metrics) = (folder.path().join("weights.csv"), folder.path().join("metrics.csv"));
    std::fs::copy(output.join("weights.csv"), &original_weights).unwrap();
    std::fs::copy(output.join("benchmark_holdout_metrics.csv"), &original_metrics).unwrap();
    let evidence = Csv::read(&output.join("evidence_summary.csv"));
    assert_eq!(evidence.rows.len(), 48);
    let evaluation = Csv::read(&output.join("benchmark_backtest_curve.csv"));
    let observations = (evaluation.rows.len() - 2) as f64;
    assert!(evidence.floats("observations").iter().all(|count| *count == observations));
    let preserved = [output.join("benchmark_notes.csv"), output.join("evidence_custom.csv")];
    preserved.iter().for_each(|path| keep(path));

    let (prices_csv, benchmarks_csv) = (output.join("prices.csv"), output.join("benchmark_prices.csv"));
    ok(&run_cli(
        &argv!["--csv", prices_csv, "--benchmark-csv", benchmarks_csv, "--output", output],
        &mut Fake::default(),
    ));
    assert_csv_close(&output.join("weights.csv"), &original_weights, 1e-6, 1e-6);
    assert_csv_close(&output.join("benchmark_holdout_metrics.csv"), &original_metrics, 1e-8, 1e-5);
    assert!(!output.join("benchmark_backtest_metrics.csv").exists());
    assert!(!output.join("benchmark_backtest_curve.csv").exists());
    let evidence = Csv::read(&output.join("evidence_summary.csv"));
    let evaluation = Csv::read(&output.join("benchmark_holdout_curve.csv"));
    let observations = (evaluation.rows.len() - 1) as f64;
    assert!(evidence.floats("observations").iter().all(|count| *count == observations));

    ok(&run_cli(&argv!["--csv", prices_csv, "--no-benchmarks", "--output", output], &mut Fake::default()));
    let generated = [
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
    ];
    assert!(generated.iter().all(|name| !output.join(name).exists()));
    assert!(preserved.iter().all(|path| kept(path)));
    assert!(output.join("report.html").is_file());
}

/// `--download-benchmarks` alone, or `--benchmark-csv` with a file that does not exist.
fn benchmark_flags(flag: &str, folder: &Path) -> Vec<std::ffi::OsString> {
    if flag == "--download-benchmarks" { argv![flag] } else { argv![flag, folder.join("absent.csv")] }
}

fn non_usd_currency_skips_benchmark_io(benchmark_flag: &str) {
    let (assets, _) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("assets.csv"), folder.path().join("report"));
    write_csv(&source, &assets);
    let flags = benchmark_flags(benchmark_flag, folder.path());
    let arguments = [argv!["--csv", source, "--currency", "eur"], flags, argv!["--output", output]].concat();
    ok(&run_cli(&arguments, &mut Fake::default()));
    let metadata = metadata(&output);
    assert_eq!(metadata["price_currency"], "EUR");
    let note = note(&metadata, "benchmarks_note");
    assert!(note.contains("require USD") && note.contains("no currency conversion"), "{note}");
    assert!(output.join("weights.csv").is_file());
    assert!(!output.join("benchmark_prices.csv").exists());
}

#[test]
fn non_usd_currency_skips_benchmark_download() {
    non_usd_currency_skips_benchmark_io("--download-benchmarks");
}

#[test]
fn non_usd_currency_skips_benchmark_csv() {
    non_usd_currency_skips_benchmark_io("--benchmark-csv");
}

fn benchmark_failures_preserve_portfolio_report(failure: &str) {
    let (assets, benchmarks) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("assets.csv"), folder.path().join("report"));
    write_csv(&source, &assets);
    let mut fake = Fake { benchmarks: Some(Err(value_error!("Benchmark service unavailable"))), ..Fake::default() };
    let flags = if failure == "network" {
        argv!["--download-benchmarks"]
    } else {
        let benchmark_source = folder.path().join("missing.csv");
        write_csv(&benchmark_source, &drop_row(&benchmarks, 15));
        argv!["--benchmark-csv", benchmark_source]
    };
    ok(&run_cli(&[argv!["--csv", source], flags, argv!["--output", output]].concat(), &mut fake));
    let metadata = metadata(&output);
    assert!(note(&metadata, "benchmarks_note").contains("unavailable"));
    assert_eq!(metadata["source"], "CSV: assets.csv");
    assert!(!output.join("benchmark_prices.csv").exists());
    assert!(!output.join("evidence_summary.csv").exists());
    assert!(output.join("report.html").is_file());
    assert_eq!(Csv::read(&output.join("weights.csv")).index(), assets.columns);
}

#[test]
fn benchmark_network_failure_preserves_portfolio_report() {
    benchmark_failures_preserve_portfolio_report("network");
}

#[test]
fn benchmark_missing_date_preserves_portfolio_report() {
    benchmark_failures_preserve_portfolio_report("missing_date");
}

#[test]
fn evidence_failure_preserves_valid_benchmark_paths() {
    let (assets, benchmarks) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("assets.csv"), folder.path().join("report"));
    write_csv(&source, &concat(&assets, &benchmarks));
    let mut fake = Fake { evidence_error: Some("Statistical comparison unavailable".into()), ..Fake::default() };
    ok(&run_cli(&argv!["--csv", source, "--output", output], &mut fake));
    let metadata = metadata(&output);
    assert!(metadata.get("benchmarks_note").is_none());
    assert!(note(&metadata, "evidence_note").contains("Statistical comparison unavailable"));
    assert!(output.join("benchmark_prices.csv").is_file());
    assert!(output.join("benchmark_holdout_curve.csv").is_file());
    assert!(!output.join("evidence_summary.csv").exists());
}

fn demo_rejects_market_benchmarks_without_network(benchmark_flag: &str) {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("demo");
    let arguments = [benchmark_flags(benchmark_flag, folder.path()), argv!["--output", output]].concat();
    let run = run_cli(&arguments, &mut Fake::default());
    assert_eq!(run.code, 1);
    assert!(run.stderr.contains("Synthetic demo data cannot use real market benchmarks"), "{}", run.stderr);
    assert!(!output.exists());
}

#[test]
fn demo_rejects_benchmark_download_without_network() {
    demo_rejects_market_benchmarks_without_network("--download-benchmarks");
}

#[test]
fn demo_rejects_benchmark_csv_without_network() {
    demo_rejects_market_benchmarks_without_network("--benchmark-csv");
}

#[test]
fn nasdaq100_source_exports_coverage_and_reuses_downloaded_benchmarks() {
    let universe = nasdaq_universe();
    let mut fake = Fake { universe: Some((false, Ok(universe.clone()))), ..Fake::default() };
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("nasdaq");
    let arguments = argv!["--nasdaq100", "--start", "2022-01-01", "--end", "2023-01-01", "--output", output];
    let run = run_cli(&arguments, &mut fake);
    ok(&run);
    assert_eq!(fake.universe_calls, [(false, day("2022-01-01"), day("2023-01-01"))]);
    let text = &run.stdout;
    assert!(text.contains("Nasdaq-100 coverage: 2 of 3 securities included; 1 excluded."));
    assert!(text.contains("not the members at each historical date"));
    assert!(text.contains("2026-10-02"));
    assert!(text.contains(NASDAQ100_URL));
    let metadata = metadata(&output);
    assert_eq!(metadata["source"], "Nasdaq-100 · Yahoo Finance");
    assert_eq!(metadata["universe_coverage"], json_value(&records(&universe.coverage)));
    assert_eq!(metadata["universe_source_date"], "2026-10-02");
    assert_eq!(metadata["universe_retrieved_at"], "2026-10-04T12:00:00+00:00");
    assert_eq!(metadata["universe_requested"], 3);
    assert_eq!(metadata["universe_excluded"], 1);
    assert_eq!(metadata["requested_start"], "2022-01-01");
    assert_eq!(metadata["requested_end_exclusive"], "2023-01-01");
    assert_eq!(metadata["benchmark_source"], "Yahoo Finance · Nasdaq-100 download");
    assert_eq!(Csv::read(&output.join("weights.csv")).index(), universe.prices.columns);
    assert_csv_frame(&output.join("benchmark_prices.csv"), &universe.benchmarks);
    let coverage = Csv::read(&output.join("universe_coverage.csv"));
    assert_eq!(coverage.column("status"), ["included", "included", "excluded"]);
    let short = coverage.column("symbol").iter().position(|symbol| *symbol == "SHORT").unwrap();
    assert_eq!(coverage.column("reason")[short], "Incomplete historical prices.");
    assert!(read(&output.join("report.html")).contains("Universe coverage"));

    // The default stays offline and removes only the obsolete generated coverage file.
    let preserved = output.join("universe_coverage_notes.csv");
    keep(&preserved);
    ok(&run_cli(&argv!["--output", output], &mut Fake::default()));
    let metadata = cli_support::metadata(&output);
    assert_eq!(metadata["source"], "Demo · synthetic");
    assert!(metadata.get("universe_coverage").is_none());
    assert!(!output.join("universe_coverage.csv").exists());
    assert!(kept(&preserved));
}

fn nasdaq100_source_preserves_explicit_benchmark_controls(benchmark_mode: &str) {
    let universe = nasdaq_universe();
    let mut fake = Fake { universe: Some((false, Ok(universe.clone()))), ..Fake::default() };
    let folder = tempfile::tempdir().unwrap();
    let doubled = scaled(&universe.benchmarks, 2.0);
    let flags = match benchmark_mode {
        "disabled" => argv!["--no-benchmarks"],
        "other_currency" => argv!["--currency", "EUR"],
        "csv" => {
            let path = folder.path().join("override.csv");
            write_csv(&path, &doubled);
            argv!["--benchmark-csv", path]
        }
        _ => {
            fake.benchmarks = Some(Ok(doubled.clone()));
            argv!["--download-benchmarks"]
        }
    };
    let output = folder.path().join("report");
    ok(&run_cli(&[argv!["--nasdaq100"], flags, argv!["--output", output]].concat(), &mut fake));
    let metadata = metadata(&output);
    assert_eq!(metadata["universe_included"], 2);
    assert!(output.join("universe_coverage.csv").is_file());
    if matches!(benchmark_mode, "disabled" | "other_currency") {
        assert!(metadata.get("benchmarks_note").is_some());
        assert!(!output.join("benchmark_prices.csv").exists());
    } else {
        let expected = if benchmark_mode == "csv" { "CSV: override.csv" } else { "Yahoo Finance" };
        assert_eq!(metadata["benchmark_source"], expected);
        assert_csv_frame(&output.join("benchmark_prices.csv"), &doubled);
        if benchmark_mode == "download" {
            assert_eq!(fake.benchmark_calls, std::slice::from_ref(&universe.prices.index));
        }
    }
}

#[test]
fn nasdaq100_source_preserves_disabled_benchmarks() {
    nasdaq100_source_preserves_explicit_benchmark_controls("disabled");
}

#[test]
fn nasdaq100_source_preserves_other_currency() {
    nasdaq100_source_preserves_explicit_benchmark_controls("other_currency");
}

#[test]
fn nasdaq100_source_preserves_benchmark_csv() {
    nasdaq100_source_preserves_explicit_benchmark_controls("csv");
}

#[test]
fn nasdaq100_source_preserves_benchmark_download() {
    nasdaq100_source_preserves_explicit_benchmark_controls("download");
}

fn nasdaq100_failure_does_not_fall_back_to_demo(failure: &str) {
    let universe = nasdaq_universe();
    let (answer, error) = if failure == "snapshot" {
        let unavailable = value_error!("The current Nasdaq-100 list is unavailable or incomplete.");
        (Err(unavailable), "current Nasdaq-100 list is unavailable")
    } else {
        let mut metadata = universe.metadata.clone();
        metadata.insert("universe_included", 0);
        metadata.insert("universe_excluded", 3);
        let excluded = UniversePrices {
            prices: Frame::from_columns(universe.prices.index.clone(), Vec::new()),
            coverage: universe
                .coverage
                .clone()
                .with("status", cells(&["excluded"; 3]))
                .with("reason", cells(&["No complete prices."; 3])),
            benchmarks: universe.benchmarks.clone(),
            metadata,
        };
        let message = "Only 0 of 3 Nasdaq-100 securities have complete prices. At least two are required. \
                       3 securities were excluded.";
        (Ok(excluded), message)
    };
    let mut fake = Fake { universe: Some((false, answer)), forbid_demo: true, ..Fake::default() };
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("failed");
    let run = run_cli(&argv!["--nasdaq100", "--output", output], &mut fake);
    assert_eq!(run.code, 1);
    assert!(run.stderr.contains(error), "{}", run.stderr);
    assert!(!output.exists());
}

#[test]
fn nasdaq100_snapshot_failure_does_not_fall_back_to_demo() {
    nasdaq100_failure_does_not_fall_back_to_demo("snapshot");
}

#[test]
fn nasdaq100_coverage_failure_does_not_fall_back_to_demo() {
    nasdaq100_failure_does_not_fall_back_to_demo("coverage");
}
