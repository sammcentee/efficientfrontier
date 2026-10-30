//! The risk-target selection of the CLI (`--target-volatility`) and the cleanup of earlier exports.
//!
//! Ports of the in-process tests of `tests/test_cli.py`. A [`Fake`] replaces
//! the monkeypatched downloads, so every run stays offline.

mod cli_support;

use std::collections::BTreeSet;
use std::io::{Cursor, Write};

use cli_support::*;
use efficient_frontier::cli::write_report_bundle;
use efficient_frontier::universe::UniversePrices;

fn ok(run: &Run) {
    assert_eq!(run.code, 0, "{}", run.stderr);
}

fn texts(value: &serde_json::Value) -> Vec<&str> {
    value.as_array().unwrap().iter().map(|item| item.as_str().unwrap()).collect()
}

fn risk_target_cli_exports_complete_offline_inputs_and_reproduces(shrinkage: Option<&str>) {
    let (prices, benchmarks) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, market) = (folder.path().join("stocks.csv"), folder.path().join("markets.csv"));
    write_csv(&source, &prices);
    write_csv(&market, &benchmarks);
    let output = folder.path().join("risk target");
    let shrinkage_flags = shrinkage.map_or_else(Vec::new, |value| argv!["--shrinkage", value]);
    let arguments = [
        argv!["--csv", source, "--benchmark-csv", market, "--target-volatility", ".20", "--max-weight", ".5"],
        shrinkage_flags,
        argv!["--output", output],
    ]
    .concat();
    ok(&run_cli(&arguments, &mut Fake::default()));
    let metadata = metadata(&output);
    let settings = &metadata["settings"];
    assert_eq!(settings["target_volatility"], 0.2);
    assert_eq!(settings["max_weight"], 0.5);
    assert_eq!(settings["shrinkage"], shrinkage.map_or(0.0, |value| value.parse().unwrap()));
    assert_eq!(settings["holding_policy"], "buy_and_hold");
    assert!(settings.get("rebalance_every").is_none());
    let command = metadata["reproduce_command"].as_str().unwrap();
    assert!(!command.contains("--rebalance-every"));
    assert_eq!(metadata["selected"], "Historical-mean portfolio");
    assert!(!output.join("latest_profile_weights.csv").exists());
    assert_csv_frame(&output.join("prices.csv"), &prices);
    assert_csv_frame(&output.join("benchmark_prices.csv"), &benchmarks);
    assert!(output.join("eligibility.csv").is_file());
    assert!(!output.join("policy.json").exists());
    assert!(!output.join("validation").exists());
    let result_files = metadata["result_files"].as_object().unwrap();
    let phases: BTreeSet<&str> = result_files.keys().map(|key| key.split('/').next().unwrap()).collect();
    assert_eq!(phases, BTreeSet::from(["test"]));
    assert_eq!(result_files.len(), 4);
    let metrics = Csv::read(&output.join("test/metrics.csv"));
    assert!(metrics.floats("trade_count").iter().all(|count| *count == 1.0));
    assert!(metrics.floats("rebalance_count").iter().all(|count| *count == 0.0));
    for stem in result_files.keys() {
        assert_eq!(Csv::read(&output.join(format!("{stem}_trades.csv"))).rows.len(), 1, "{stem}");
    }
    reproduce(command);
    for name in ["latest_weights.csv", "test/equity.csv", "test/metrics.csv"] {
        assert_csv_close(&output.join("reproduced").join(name), &output.join(name), 1e-8, 1e-8);
    }
}

#[test]
fn risk_target_cli_exports_complete_offline_inputs_and_reproduces_with_default_shrinkage() {
    risk_target_cli_exports_complete_offline_inputs_and_reproduces(None);
}

#[test]
fn risk_target_cli_exports_complete_offline_inputs_and_reproduces_with_shrinkage() {
    risk_target_cli_exports_complete_offline_inputs_and_reproduces(Some("0.25"));
}

#[test]
fn risk_target_combined_source_is_explicit_and_reuses_matching_etf_prices() {
    let universe = nasdaq_universe();
    let mut fake = Fake { universe: Some((true, Ok(universe.clone()))), ..Fake::default() };
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("combined");
    let arguments = argv![
        "--sp500-nasdaq100",
        "--target-volatility",
        ".20",
        "--max-weight",
        ".5",
        "--start",
        "2022-01-01",
        "--end",
        "2023-01-01",
        "--output",
        output
    ];
    ok(&run_cli(&arguments, &mut fake));
    assert_eq!(fake.universe_calls, [(true, day("2022-01-01"), day("2023-01-01"))]);
    let metadata = metadata(&output);
    assert_eq!(metadata["source"], "S&P 500 + Nasdaq-100 · Yahoo Finance");
    assert_eq!(metadata["universe_coverage"], json_value(&records(&universe.coverage)));
    assert_eq!(metadata["benchmark_source"], "Yahoo Finance · universe download");
    assert_csv_frame(&output.join("benchmark_prices.csv"), &universe.benchmarks);
}

#[test]
fn risk_target_csv_without_markets_is_explicitly_unsupported() {
    let (prices, _) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("stocks.csv"), folder.path().join("uncompared"));
    write_csv(&source, &prices);
    let arguments = argv!["--csv", source, "--target-volatility", ".20", "--max-weight", ".5", "--output", output];
    let run = run_cli(&arguments, &mut Fake::default());
    ok(&run);
    let metadata = metadata(&output);
    assert_eq!(metadata["supported"], false);
    assert!(metadata["benchmarks_note"].as_str().unwrap().contains("stays offline"));
    assert!(texts(&metadata["reasons"]).iter().any(|reason| {
        let reason = reason.to_lowercase();
        reason.contains("benchmark") && reason.contains("unavailable")
    }));
    assert!(!output.join("benchmark_prices.csv").exists());
    assert!(run.stdout.contains("Historical checks did not pass"));
}

fn risk_target_separates_embedded_etfs_offline_and_reproduces(symbols: &[&str]) {
    let (prices, benchmarks) = benchmark_inputs();
    let embedded = benchmarks.select(symbols).unwrap();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("combined.csv"), folder.path().join("study"));
    write_csv(&source, &concat(&prices, &embedded));
    let arguments = argv![
        "--csv",
        source,
        "--target-volatility",
        ".2",
        "--max-weight",
        ".5",
        "--download-benchmarks",
        "--output",
        output
    ];
    let run = run_cli(&arguments, &mut Fake::default());
    ok(&run);
    let metadata = metadata(&output);
    assert_eq!(texts(&metadata["excluded_candidate_symbols"]), symbols);
    assert!(run.stdout.contains("excluded from candidate holdings"));
    assert_csv_frame(&output.join("prices.csv"), &prices);
    assert_csv_frame(&output.join("benchmark_prices.csv"), &embedded);
    let holdings = Csv::read(&output.join("latest_weights.csv"));
    assert!(holdings.index().iter().all(|symbol| !symbols.contains(symbol)));
    reproduce(metadata["reproduce_command"].as_str().unwrap());
    let equity = "test/equity.csv";
    assert_csv_close(&output.join("reproduced").join(equity), &output.join(equity), 1e-8, 1e-8);
}

#[test]
fn risk_target_separates_embedded_spy_offline_and_reproduces() {
    risk_target_separates_embedded_etfs_offline_and_reproduces(&["SPY"]);
}

#[test]
fn risk_target_separates_embedded_qqq_offline_and_reproduces() {
    risk_target_separates_embedded_etfs_offline_and_reproduces(&["QQQ"]);
}

#[test]
fn risk_target_separates_embedded_spy_and_qqq_offline_and_reproduces() {
    risk_target_separates_embedded_etfs_offline_and_reproduces(&["SPY", "QQQ"]);
}

fn risk_target_excludes_etfs_even_when_comparisons_are_off(flags: &[&str]) {
    let (prices, benchmarks) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("combined.csv"), folder.path().join("study"));
    write_csv(&source, &concat(&prices, &benchmarks));
    let flags: Vec<std::ffi::OsString> = flags.iter().map(Into::into).collect();
    let arguments =
        [argv!["--csv", source, "--target-volatility", ".2", "--max-weight", ".5"], flags, argv!["--output", output]]
            .concat();
    ok(&run_cli(&arguments, &mut Fake::default()));
    assert_csv_frame(&output.join("prices.csv"), &prices);
    assert!(!output.join("benchmark_prices.csv").exists());
}

#[test]
fn risk_target_excludes_etfs_when_benchmarks_are_disabled() {
    risk_target_excludes_etfs_even_when_comparisons_are_off(&["--no-benchmarks"]);
}

#[test]
fn risk_target_excludes_etfs_for_another_currency() {
    risk_target_excludes_etfs_even_when_comparisons_are_off(&["--currency", "EUR"]);
}

#[test]
fn risk_target_preserves_partial_universe_benchmark() {
    let loaded = nasdaq_universe();
    let partial = UniversePrices { benchmarks: loaded.benchmarks.select(&["QQQ"]).unwrap(), ..loaded };
    let mut fake = Fake { universe: Some((true, Ok(partial.clone()))), ..Fake::default() };
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("partial");
    let arguments = argv!["--sp500-nasdaq100", "--target-volatility", ".2", "--max-weight", ".5", "--output", output];
    ok(&run_cli(&arguments, &mut fake));
    let metadata = metadata(&output);
    assert_eq!(metadata["supported"], false);
    assert!(texts(&metadata["reasons"]).iter().any(|reason| reason.contains("SPY") && reason.contains("unavailable")));
    assert_csv_frame(&output.join("benchmark_prices.csv"), &partial.benchmarks);
}

#[test]
fn risk_target_rejects_only_etf_inputs_before_export() {
    let (_, benchmarks) = benchmark_inputs();
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("funds.csv"), folder.path().join("invalid"));
    write_csv(&source, &benchmarks);
    let run = run_cli(&argv!["--csv", source, "--target-volatility", ".2", "--output", output], &mut Fake::default());
    assert_eq!(run.code, 1);
    assert!(!output.exists());
    assert!(run.stderr.to_lowercase().contains("stock") && run.stderr.contains("SPY"), "{}", run.stderr);
}

#[test]
fn risk_target_default_cap_is_not_silently_relaxed_for_demo() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("infeasible");
    let run = run_cli(&argv!["--target-volatility", ".20", "--output", output], &mut Fake::default());
    assert_eq!(run.code, 1);
    assert!(run.stderr.contains("10.0%"), "{}", run.stderr);
    assert!(!output.exists());
}

fn risk_target_rejects_incompatible_modes_before_download(flags: &[&str], message: &str) {
    let folder = tempfile::tempdir().unwrap();
    let flags: Vec<std::ffi::OsString> = flags.iter().map(Into::into).collect();
    let arguments = [flags, argv!["--output", folder.path().join("invalid")]].concat();
    let run = run_cli(&arguments, &mut Fake { forbid_demo: true, ..Fake::default() });
    assert_eq!(run.code, 1);
    assert!(run.stderr.contains(message), "{}", run.stderr);
}

#[test]
fn risk_target_rejects_historical_demo_before_download() {
    let flags = ["--target-volatility", ".2", "--historical-demo"];
    risk_target_rejects_incompatible_modes_before_download(&flags, "Historical membership is a separate study");
}

#[test]
fn combined_universe_requires_a_risk_target_before_download() {
    risk_target_rejects_incompatible_modes_before_download(&["--sp500-nasdaq100"], "requires --target-volatility");
}

#[test]
fn risk_target_rejects_rolling_window_before_download() {
    let flags = ["--target-volatility", ".2", "--rolling-window", "20"];
    risk_target_rejects_incompatible_modes_before_download(&flags, "omit --rolling-window");
}

#[test]
fn risk_target_rejects_rebalance_interval_before_download() {
    let flags = ["--target-volatility", ".2", "--rebalance-every", "21"];
    risk_target_rejects_incompatible_modes_before_download(&flags, "omit --rebalance-every");
}

/// A report bundle whose files all contain `generated`.
fn bundle(names: &[&str]) -> Vec<u8> {
    let mut buffer = Cursor::new(Vec::new());
    let mut archive = zip::ZipWriter::new(&mut buffer);
    for name in names {
        archive.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
        archive.write_all(b"generated\n").unwrap();
    }
    archive.finish().unwrap();
    buffer.into_inner()
}

#[test]
fn reruns_clean_only_known_selection_exports() {
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
    write_report_bundle(&bundle(&[&["report.html"][..], &generated].concat()), folder.path()).unwrap();
    let notes = [
        folder.path().join("notes.txt"),
        folder.path().join("validation/notes.csv"),
        folder.path().join("test/01_custom.csv"),
    ];
    for path in &notes {
        std::fs::write(path, "Keep this user file.\n").unwrap();
    }
    write_report_bundle(&bundle(&["report.html", "weights.csv"]), folder.path()).unwrap();
    assert!(generated.iter().all(|name| !folder.path().join(name).exists()));
    assert!(notes.iter().all(|path| read(path) == "Keep this user file.\n"));
}
