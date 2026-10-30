//! The `efficient-frontier` program in a separate process, offline.
//!
//! Ports of the subprocess tests of `tests/test_cli.py`, and checks of the
//! argument parser: the help text and the exclusive flag groups.

mod cli_support;

use std::collections::BTreeSet;

use chrono::Months;
use clap::Parser;
use cli_support::*;
use efficient_frontier::cli::Args;
use efficient_frontier::frame::Frame;

fn ok(run: &Run) {
    assert_eq!(run.code, 0, "{}", run.stderr);
}

fn sum(values: &[f64]) -> f64 {
    values.iter().filter(|value| !value.is_nan()).sum()
}

#[test]
fn offline_export_and_csv_reproduction() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("demo");
    let run = run_bin(&argv!["--output", output]);
    ok(&run);
    assert!(run.stdout.contains("synthetic"));
    assert!(std::fs::metadata(output.join("report.html")).unwrap().len() > 1000);
    let metadata = metadata(&output);
    assert!(metadata["training_end"].as_str().unwrap() < metadata["holdout_start"].as_str().unwrap());
    assert_eq!(metadata["max_weight"], 1.0);
    assert_eq!(metadata["shrinkage"], 0.1);
    assert_eq!(metadata["latest_profiles"]["as_of"], "2023-10-31");
    assert_eq!(metadata["latest_profiles"]["settings"]["return_objective"], "historical_mean");
    assert!(run.stdout.contains("Latest model holdings"));
    let latest = Csv::read(&output.join("latest_profile_weights.csv"));
    assert_eq!(latest.columns(), ["Low", "Medium", "Extreme"]);
    for name in latest.columns() {
        assert!(close(sum(&latest.floats(name)), 1.0, 0.0, 1e-7), "{name}");
    }
    let reproduced = folder.path().join("reproduced");
    ok(&run_bin(&argv!["--csv", output.join("prices.csv"), "--output", reproduced]));
    for name in ["weights.csv", "latest_profile_weights.csv"] {
        assert_csv_close(&reproduced.join(name), &output.join(name), 1e-6, 1e-6);
    }
}

#[test]
fn explicit_empty_universe_does_not_fall_back_to_demo() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("empty");
    let run = run_bin(&argv!["--tickers", "", "--output", output]);
    assert_eq!(run.code, 1);
    assert!(run.stderr.to_lowercase().contains("at least one"), "{}", run.stderr);
    assert!(!output.exists());
}

#[test]
fn monthly_csv_cli_uses_selected_annualization_and_exports_risk() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("monthly");
    ok(&run_bin(&argv![
        "--csv",
        fixture("monthly.csv"),
        "--periods-per-year",
        "12",
        "--backtests",
        "--rebalance-every",
        "3",
        "--rolling-window",
        "12",
        "--output",
        output,
    ]));
    let metadata = metadata(&output);
    assert_eq!(metadata["periods_per_year"], 12.0);
    assert_eq!(metadata["backtests"]["periods_per_year"], 12.0);
    assert_eq!(metadata["backtests"]["rebalance_every"], 3);
    assert_eq!(metadata["backtests"]["include_profiles"], true);
    assert_eq!(metadata["backtests"]["profile_return_objective"], "historical_mean");
    let windows = Csv::read(&output.join("latest_profile_windows.csv"));
    assert_eq!(sum(&windows.floats("observations")), 35.0);
    let latest = Csv::read(&output.join("latest_profile_summary.csv"));
    let window_returns = Csv::read(&output.join("latest_profile_window_returns.csv"));
    assert_eq!(window_returns.columns(), latest.index());
    for profile in latest.index() {
        let worst = window_returns.floats(profile).into_iter().fold(f64::INFINITY, f64::min);
        assert!(close(latest.float(profile, "worst_window_return"), worst, 0.0, 1e-7), "{profile}");
    }
    let metrics = Csv::read(&output.join("holdout_metrics.csv"));
    let curve = Csv::read(&output.join("holdout_curve.csv"));
    let intervals = (curve.rows.len() - 1) as f64;
    for portfolio in metrics.index() {
        let growth = *curve.floats(portfolio).last().unwrap();
        let cagr = growth.powf(12.0 / intervals) - 1.0;
        assert!(close(metrics.float(portfolio, "cagr"), cagr, 0.0, 1e-7), "{portfolio}");
    }
    assert!(metrics.columns().contains(&"sortino".to_string()) && metrics.columns().contains(&"calmar".to_string()));
    assert!(output.join("risk_summary.csv").is_file());
    let contributions = Csv::read(&output.join("risk_contributions.csv"));
    for name in contributions.columns() {
        assert!(close(sum(&contributions.floats(name)), 1.0, 0.0, 1e-7), "{name}");
    }
    assert!(read(&output.join("report.html")).contains("12 observations per year"));
}

/// `100 * numpy.vstack([numpy.ones(2), numpy.cumprod(1 + returns, axis=0)])` on month ends from January 2020.
fn monthly_prices(returns: [Vec<f64>; 2], names: [&str; 2]) -> Frame {
    let first = day("2020-02-01");
    let dates = (0..=returns[0].len() as u32).map(|i| first + Months::new(i) - chrono::Days::new(1)).collect();
    let columns = returns
        .iter()
        .zip(names)
        .map(|(column, name)| {
            let mut growth = 1.0;
            let mut path = vec![100.0];
            for value in column {
                growth *= 1.0 + value;
                path.push(100.0 * growth);
            }
            (name.to_string(), path)
        })
        .collect();
    Frame::from_columns(dates, columns)
}

#[test]
fn cli_primary_profile_maximizes_mean_despite_a_negative_window() {
    let steady: Vec<f64> = (0..36).map(|i| 0.01 + [-0.005, 0.005][i % 2]).collect();
    let higher: Vec<f64> = (0..36).map(|i| [0.04, -0.02, 0.06][i / 12] + [-0.001, 0.001][i % 2]).collect();
    let prices = monthly_prices([steady, higher], ["Steady", "HigherMean"]);
    let folder = tempfile::tempdir().unwrap();
    let (source, output) = (folder.path().join("regimes.csv"), folder.path().join("mean"));
    write_csv(&source, &prices);
    let run = run_bin(&argv![
        "--csv",
        source,
        "--periods-per-year",
        "12",
        "--backtests",
        "--rebalance-every",
        "3",
        "--output",
        output
    ]);
    ok(&run);
    assert!(run.stdout.contains("historical mean frontier"));
    let weights = Csv::read(&output.join("latest_profile_weights.csv"));
    let summary = Csv::read(&output.join("latest_profile_summary.csv"));
    assert!(weights.float("HigherMean", "Extreme") > 1.0 - 1e-6);
    assert!((summary.float("Extreme", "expected_return") - 0.32).abs() <= 1e-7);
    assert!((summary.float("Extreme", "worst_window_return") + 0.24).abs() <= 1e-7);
    // `prices.pct_change(fill_method=None).iloc[1:].mean() * 12`, by asset.
    let returns = prices.pct_change();
    let mean: Vec<f64> = (0..returns.ncols()).map(|j| sum(&returns.col(j)) / returns.nrows() as f64 * 12.0).collect();
    for profile in summary.index() {
        let expected: f64 =
            prices.columns.iter().zip(&mean).map(|(asset, mean)| mean * weights.float(asset, profile)).sum();
        assert!(close(summary.float(profile, "expected_return"), expected, 1e-8, 1e-7), "{profile}");
    }
    let metadata = metadata(&output);
    assert_eq!(metadata["latest_profiles"]["settings"]["selection_method"], "historical_mean_frontier");
    assert_eq!(metadata["backtests"]["profile_return_objective"], "historical_mean");
}

#[test]
fn invalid_annualization_fails_before_export() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("invalid");
    let run = run_bin(&argv!["--periods-per-year", "0", "--output", output]);
    assert_eq!(run.code, 1);
    assert!(run.stderr.contains("periods_per_year"), "{}", run.stderr);
    assert!(!output.exists());
}

#[test]
fn backtest_export_has_all_methods_trades_and_reconciled_holdings() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("backtests");
    ok(&run_bin(&argv!["--backtests", "--rolling-window", "252", "--cost-bps", "10", "--output", output]));
    let metadata = metadata(&output);
    let backtests = &metadata["backtests"];
    assert!(backtests["train_end"].as_str().unwrap() < backtests["initial_execution"].as_str().unwrap());
    let files = metadata["backtest_files"].as_object().unwrap();
    assert_eq!(files.len(), 24);
    let metrics = Csv::read(&output.join("backtest_metrics.csv"));
    for (prefix, strategy) in files {
        let holdings = Csv::read(&output.join(format!("{prefix}_holdings.csv")));
        let trades = Csv::read(&output.join(format!("{prefix}_trades.csv")));
        let total = metrics.float(strategy.as_str().unwrap(), "total_return");
        let difference = sum(&holdings.floats("pnl_contribution")) - sum(&trades.floats("cost")) - total;
        assert!(difference.abs() < 1e-10, "{prefix}: {difference}");
        for (train_end, date) in trades.column("train_end").into_iter().zip(trades.index()) {
            assert!(day(&train_end[..10]) < day(&date[..10]), "{prefix}: {train_end} {date}");
        }
    }
    let findings = read(&output.join("findings.md"));
    assert!(findings.contains("synthetic") && findings.contains("Rolling window"));
    let document = read(&output.join("report.html"));
    assert!(document.find("* plotly.js v").unwrap() < document.find("<section id=\"backtesting\">").unwrap());
    assert_eq!(document.matches("* plotly.js v").count(), 1);
    assert!(document.contains("Print / save PDF"));
}

/// The numbered holdings exports in `folder`.
fn holdings_files(folder: &std::path::Path) -> usize {
    std::fs::read_dir(folder)
        .unwrap()
        .filter_map(|entry| entry.unwrap().file_name().into_string().ok())
        .filter(|name| {
            let bytes = name.as_bytes();
            name.len() == 15 && bytes[..2].iter().all(u8::is_ascii_digit) && name.ends_with("_holdings.csv")
        })
        .count()
}

#[test]
fn reruns_remove_obsolete_backtest_outputs_but_preserve_unrelated_files() {
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("reused");
    let command = argv!["--output", output];
    ok(&run_bin(&[command.clone(), argv!["--backtests"]].concat()));
    let backtests = output.join("backtests");
    let preserved = [output.join("notes.txt"), backtests.join("notes.csv"), backtests.join("13_custom.csv")];
    for path in &preserved {
        std::fs::write(path, "Keep this user file.\n").unwrap();
    }
    let obsolete = backtests.join("25_holdings.csv");
    std::fs::write(&obsolete, "old,output\n").unwrap();

    ok(&run_bin(&[command.clone(), argv!["--backtests", "--cost-bps", "0"]].concat()));
    assert!(!obsolete.exists());
    assert_eq!(holdings_files(&backtests), 24);
    assert!(Csv::read(&output.join("backtest_metrics.csv")).floats("total_cost").iter().all(|cost| *cost == 0.0));

    ok(&run_bin(&command));
    for name in ["findings.md", "backtest_metrics.csv", "backtest_curve.csv"] {
        assert!(!output.join(name).exists(), "{name}");
    }
    let remaining: BTreeSet<_> = std::fs::read_dir(&backtests).unwrap().map(|entry| entry.unwrap().path()).collect();
    assert_eq!(remaining, BTreeSet::from([preserved[1].clone(), preserved[2].clone()]));
    for path in &preserved {
        assert_eq!(read(path), "Keep this user file.\n");
    }
    assert!(metadata(&output).get("backtests").is_none());
    assert!(output.join("report.html").is_file());

    let dates = vec![day("2024-01-01"), day("2024-01-02"), day("2024-01-03"), day("2024-01-04"), day("2024-01-05")];
    let short_prices = Frame::from_columns(
        dates,
        vec![
            ("AAA".into(), vec![100.0, 102.0, 101.0, 103.0, 102.0]),
            ("BBB".into(), vec![100.0, 101.0, 103.0, 102.0, 104.0]),
        ],
    );
    let source = folder.path().join("short.csv");
    write_csv(&source, &short_prices);
    let run = run_bin(&[command, argv!["--csv", source, "--backtests"]].concat());
    ok(&run);
    assert!(run.stdout.contains("at least seven prices"));
    assert!(run.stdout.contains("at least six returns"));
    let profiles = std::fs::read_dir(&output)
        .unwrap()
        .filter(|entry| entry.as_ref().unwrap().file_name().to_string_lossy().starts_with("latest_profile_"))
        .count();
    assert_eq!(profiles, 0);
    assert_eq!(Csv::read(&output.join("backtest_metrics.csv")).rows.len(), 12);
    assert_eq!(metadata(&output)["backtests"]["include_profiles"], false);
}

/// Every option of the Python release's `argparse` parser.
const FLAGS: [&str; 26] = [
    "--csv",
    "--tickers",
    "--original-holdings",
    "--nasdaq100",
    "--sp500-nasdaq100",
    "--historical-membership",
    "--membership-csv",
    "--historical-demo",
    "--membership-end",
    "--benchmark-csv",
    "--download-benchmarks",
    "--no-benchmarks",
    "--currency",
    "--start",
    "--end",
    "--train-fraction",
    "--risk-free-rate",
    "--max-weight",
    "--target-volatility",
    "--shrinkage",
    "--periods-per-year",
    "--backtests",
    "--rebalance-every",
    "--rolling-window",
    "--cost-bps",
    "--output",
];

#[test]
fn help_lists_every_flag() {
    let run = run_bin(&argv!["--help"]);
    ok(&run);
    // Each option starts its own line; the description follows on the next lines.
    let listed: BTreeSet<&str> = run
        .stdout
        .lines()
        .filter(|line| line.trim_start().starts_with('-'))
        .flat_map(|line| line.split_whitespace().take(2))
        .map(|word| word.trim_end_matches(','))
        .filter(|word| word.starts_with("--"))
        .collect();
    for flag in FLAGS.into_iter().chain(["--help"]) {
        assert!(listed.contains(flag), "{flag} is missing from the help:\n{}", run.stdout);
    }
}

#[test]
fn numeric_options_take_negative_values_as_argparse_does() {
    let decimals = [
        "--train-fraction",
        "--risk-free-rate",
        "--max-weight",
        "--target-volatility",
        "--shrinkage",
        "--periods-per-year",
        "--cost-bps",
    ];
    for flag in decimals {
        for value in ["-0.01", "-.01", "-1e-3"] {
            let parsed = Args::try_parse_from(["efficient-frontier", flag, value]);
            assert!(parsed.is_ok(), "{flag} {value}: {:?}", parsed.err());
        }
    }
    for flag in ["--rebalance-every", "--rolling-window"] {
        assert!(Args::try_parse_from(["efficient-frontier", flag, "-3"]).is_ok(), "{flag}");
    }
    assert_eq!(Args::try_parse_from(["efficient-frontier", "--risk-free-rate", "-.01"]).unwrap().risk_free_rate, -0.01);
    // A flag in place of the value is still a usage error.
    let error = Args::try_parse_from(["efficient-frontier", "--risk-free-rate", "--backtests"]).unwrap_err();
    assert_eq!(error.exit_code(), 2);
    // The Python release runs this command and exports the negative rate.
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("negative");
    ok(&run_bin(&argv!["--risk-free-rate", "-0.01", "--output", output]));
    assert_eq!(metadata(&output)["risk_free_rate"], -0.01);
}

/// The three mutually exclusive groups of the Python parser.
const GROUPS: [&[&[&str]]; 3] = [
    &[
        &["--csv", "prices.csv"],
        &["--tickers", "AAA"],
        &["--original-holdings"],
        &["--nasdaq100"],
        &["--sp500-nasdaq100"],
    ],
    &[&["--historical-membership"], &["--membership-csv", "members.csv"], &["--historical-demo"]],
    &[&["--benchmark-csv", "markets.csv"], &["--download-benchmarks"], &["--no-benchmarks"]],
];

#[test]
fn conflicting_flags_in_one_exclusive_group_exit_with_code_2() {
    for group in GROUPS {
        for (i, first) in group.iter().enumerate() {
            for second in &group[i + 1..] {
                let arguments = [&["efficient-frontier"][..], *first, *second].concat();
                let error = Args::try_parse_from(&arguments).unwrap_err();
                assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict, "{arguments:?}");
                assert_eq!(error.exit_code(), 2, "{arguments:?}");
            }
        }
    }
    let folder = tempfile::tempdir().unwrap();
    let output = folder.path().join("conflict");
    let run = run_bin(&argv!["--tickers", "AAA", "--nasdaq100", "--output", output]);
    assert_eq!(run.code, 2);
    assert!(run.stdout.is_empty());
    assert!(run.stderr.contains("--nasdaq100") && run.stderr.contains("--tickers"), "{}", run.stderr);
    assert!(!output.exists());
}
