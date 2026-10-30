//! Live smoke tests against Yahoo Finance, Nasdaq and State Street.
//!
//! The tests need the network, so they are ignored by default. Run them one
//! at a time, as the providers expect gentle use:
//! `cargo test --test network_live -- --ignored --nocapture --test-threads=1`.
//! Each test prints what it received, for a comparison with the Python release.

use chrono::NaiveDate;
use efficient_frontier::companies::fetch_company_profile;
use efficient_frontier::data::{download_adjusted_close, download_benchmarks, download_prices};
use efficient_frontier::frame::Date;
use efficient_frontier::risk_reference::load_qqq_volatility_reference;
use efficient_frontier::universe::{fetch_nasdaq100, fetch_spy_holdings};

fn day(text: &str) -> Date {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
}

fn symbols(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| name.to_string()).collect()
}

#[test]
#[ignore = "needs the network"]
fn qqq_and_spy_daily_prices() {
    let prices = download_prices(&symbols(&["QQQ", "SPY"]), day("2026-10-01"), day("2026-10-15")).unwrap();
    print!("{}", prices.to_csv("Date"));
    assert_eq!(prices.columns, ["QQQ", "SPY"]);
    assert_eq!(prices.index.first(), Some(&day("2026-10-01")));
    assert!(prices.index.iter().all(|date| *date < day("2026-10-15")));
}

#[test]
#[ignore = "needs the network"]
fn a_bad_symbol_is_reported_and_never_dropped() {
    let request = symbols(&["SPY", "NOSUCHSYMBOLZZ"]);
    let error = download_prices(&request, day("2026-10-01"), day("2026-10-15")).unwrap_err();
    println!("{error}");
    assert_eq!(
        error.message(),
        "Yahoo returned no prices for: NOSUCHSYMBOLZZ. Check these symbols and their available history, or upload a CSV."
    );
    let incomplete = download_adjusted_close(&request, day("2026-10-01"), day("2026-10-15"), true).unwrap();
    print!("{}", incomplete.to_csv("Date"));
    assert!(incomplete.col(1).iter().all(|value| value.is_nan()));
}

#[test]
#[ignore = "needs the network"]
fn other_exchanges_use_their_own_calendar_dates() {
    let request = symbols(&["VOD.L", "7203.T"]);
    let prices = download_adjusted_close(&request, day("2026-10-01"), day("2026-10-15"), true).unwrap();
    print!("{}", prices.to_csv("Date"));
    assert_eq!(prices.columns, request);
    assert!(prices.index.iter().all(|date| *date >= day("2026-10-01") && *date < day("2026-10-15")));
}

#[test]
#[ignore = "needs the network"]
fn benchmarks_on_exact_dates() {
    let dates: Vec<Date> = ["2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22", "2026-09-29"].map(day).to_vec();
    let prices = download_benchmarks(&dates).unwrap();
    print!("{}", prices.to_csv("Date"));
    assert_eq!(prices.index, dates);
    assert_eq!(prices.columns, ["SPY", "QQQ"]);
}

#[test]
#[ignore = "needs the network"]
fn current_nasdaq100_members() {
    let snapshot = fetch_nasdaq100().unwrap();
    println!(
        "{} members, source date {}, retrieved {}",
        snapshot.members.len(),
        snapshot.source_date,
        snapshot.retrieved_at
    );
    for member in &snapshot.members {
        println!("{},{},{}", member.symbol, member.name, member.industry);
    }
    assert!(snapshot.members.len() >= 100);
}

#[test]
#[ignore = "needs the network"]
fn current_spy_holdings() {
    let snapshot = fetch_spy_holdings().unwrap();
    let record = &snapshot.source_records[0];
    println!(
        "{} stocks, source date {}, reported fund holdings {:?}, excluded {:?}",
        snapshot.members.len(),
        snapshot.source_date,
        record.get("reported_fund_holdings"),
        record.get("excluded_holdings").map(|excluded| excluded.py_repr())
    );
    for member in &snapshot.members {
        println!("{},{},{}", member.symbol, member.name, member.industry);
    }
    assert!((500..=525).contains(&snapshot.members.len()));
}

#[test]
#[ignore = "needs the network"]
fn one_company_profile() {
    let profile = fetch_company_profile("MSFT").unwrap();
    println!("{profile:#?}");
    assert!(!profile.name.is_empty());
    assert_eq!(profile.source_url, "https://finance.yahoo.com/quote/MSFT/profile/");
}

#[test]
#[ignore = "needs the network"]
fn current_qqq_reference() {
    let reference = load_qqq_volatility_reference(None).unwrap();
    println!("{reference:#?}");
    assert!(reference.observations >= 200);
    assert!(reference.volatility > 0.0);
}
