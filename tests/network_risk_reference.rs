//! QQQ's trailing one-year volatility as the starting risk budget.
//!
//! Port of `tests/test_risk_reference.py`. The clock and `download_prices`
//! are parameters of `load_qqq_volatility_reference_with`, in place of the
//! monkeypatched `datetime` and download.

mod network_support;

use std::collections::HashMap;

use chrono::{DateTime, Days, Months, TimeZone, Utc};
use chrono_tz::America::New_York;
use efficient_frontier::error::Result;
use efficient_frontier::frame::{Date, Frame, std};
use efficient_frontier::risk_reference::load_qqq_volatility_reference_with;
use efficient_frontier::value_error;
use network_support::*;

/// New York wall-clock time on a day of October 2026.
fn clock(hour: u32, day: u32) -> DateTime<Utc> {
    New_York.with_ymd_and_hms(2026, 10, day, hour, 0, 0).unwrap().with_timezone(&Utc)
}

/// Business days from 2025-09-25 to 2026-10-19 with smooth synthetic returns.
fn qqq_prices() -> Frame {
    let mut dates = business_days("2025-09-25", 400);
    dates.retain(|date| *date <= day("2026-10-19"));
    let mut prices = vec![100.0];
    for i in 0..dates.len() - 1 {
        prices.push(prices[i] * (1.0 + 0.0007 + (i as f64).sin() * 0.013));
    }
    Frame::from_columns(dates, vec![("QQQ".into(), prices)])
}

/// The annualized sample volatility of the returns after `boundary`, through `last`.
fn expected_volatility(prices: &Frame, boundary: Date, last: Date) -> (f64, usize) {
    let returns = prices.pct_change();
    let values: Vec<f64> = (0..returns.nrows())
        .filter(|i| returns.index[*i] > boundary && returns.index[*i] <= last)
        .map(|i| returns.get(i, 0))
        .collect();
    (std(&values, 1) * 252_f64.sqrt(), values.len())
}

#[test]
fn reference_uses_only_qqq_adjusted_closes_and_trailing_year_sample_risk() {
    let prices = qqq_prices();
    let mut calls = Vec::new();
    // Return extra dates too, to test the helper's own window selection.
    let download = |symbols: &[String], start: Date, end: Date| -> Result<Frame> {
        calls.push((symbols.to_vec(), start, end));
        Ok(prices.clone())
    };
    let result = load_qqq_volatility_reference_with(clock(17, 16), None, download).unwrap();
    let (volatility, observations) = expected_volatility(&prices, day("2025-10-16"), day("2026-10-16"));
    assert!((result.volatility - volatility).abs() < 1e-14);
    assert_eq!(result.first_return_date, day("2025-10-17"));
    assert_eq!(result.last_return_date, day("2026-10-16"));
    assert_eq!(result.observations, observations);
    assert_eq!(result.periods_per_year, 252);
    assert_eq!(result.symbol, "QQQ");
    assert_eq!(result.source, "Yahoo Finance adjusted close");
    assert_eq!(calls, [(vec!["QQQ".to_string()], day("2025-10-03"), day("2026-10-17"))]);
}

#[test]
fn incomplete_today_is_excluded_and_weekends_use_last_observed_close() {
    for (hour, day_of_month, last_date, end_date) in [
        (15, 16, "2026-10-15", "2026-10-16"),
        (16, 16, "2026-10-16", "2026-10-17"),
        (12, 18, "2026-10-16", "2026-10-18"),
    ] {
        let prices = qqq_prices();
        let mut ends = Vec::new();
        let download = |_: &[String], _: Date, end: Date| -> Result<Frame> {
            ends.push(end);
            Ok(prices.clone())
        };
        let result = load_qqq_volatility_reference_with(clock(hour, day_of_month), None, download).unwrap();
        assert_eq!(result.last_return_date, day(last_date));
        assert_eq!(ends, [day(end_date)]);
        let boundary = day(last_date).checked_sub_months(Months::new(12)).unwrap();
        let (volatility, _) = expected_volatility(&prices, boundary, day(last_date));
        assert!((result.volatility - volatility).abs() <= 1e-12 * volatility);
    }
}

#[test]
fn explicit_past_date_is_inclusive_even_before_todays_close() {
    let prices = qqq_prices();
    let result =
        load_qqq_volatility_reference_with(clock(10, 16), Some(day("2026-10-15")), |_, _, _| Ok(prices.clone()))
            .unwrap();
    assert_eq!(result.last_return_date, day("2026-10-15"));
}

#[test]
fn invalid_reference_never_substitutes_an_old_or_invented_value() {
    let cases = [
        ("short", "QQQ prices do not cover the full trailing year and its starting close."),
        ("sparse", "The QQQ trailing-year reference needs at least 200 finite daily returns."),
        ("missing", "The QQQ trailing-year reference needs at least 200 finite daily returns."),
        ("infinite", "The QQQ trailing-year reference needs at least 200 finite daily returns."),
        ("flat", "The QQQ trailing-year volatility must be finite and greater than zero."),
        ("stale", "QQQ has no recent completed close. Retry the current volatility reference later."),
        ("empty", "QQQ has no recent completed close. Retry the current volatility reference later."),
    ];
    for (problem, message) in cases {
        let mut prices = qqq_prices();
        let count = prices.nrows();
        match problem {
            "short" => prices = prices.rows(count - 50, count),
            "sparse" => {
                let kept: Vec<usize> = (0..count).step_by(3).collect();
                prices = Frame::from_rows(
                    kept.iter().map(|i| prices.index[*i]).collect(),
                    prices.columns.clone(),
                    &kept.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
                );
            }
            "missing" => prices.values[(count - 3, 0)] = f64::NAN,
            "infinite" => prices.values[(count - 3, 0)] = f64::INFINITY,
            "flat" => prices.values.fill(100.0),
            "stale" => prices = prices.rows_through(day("2026-10-01")),
            _ => prices = prices.rows(0, 0),
        }
        let error = load_qqq_volatility_reference_with(clock(17, 16), None, |_, _, _| Ok(prices)).unwrap_err();
        assert!(error.is_value());
        assert_eq!(error.message(), message, "{problem}");
    }
}

#[test]
fn download_failure_propagates_without_a_fallback() {
    let error = load_qqq_volatility_reference_with(clock(17, 16), None, |_, _, _| {
        Err(value_error!("Yahoo price download failed."))
    })
    .unwrap_err();
    assert_eq!(error.message(), "Yahoo price download failed.");
}

#[test]
fn future_date_fails_before_a_download() {
    let error = load_qqq_volatility_reference_with(clock(17, 16), Some(day("2026-10-17")), |_, _, _| {
        panic!("A future date must not reach Yahoo.")
    })
    .unwrap_err();
    assert_eq!(error.message(), "The QQQ reference date must be today or an earlier date.");
}

#[test]
fn the_new_york_date_follows_daylight_saving_time() {
    // 04:30 UTC on 2 November 2026 is 23:30 on 1 November in New York, after the clocks went back.
    let now = Utc.with_ymd_and_hms(2026, 11, 2, 4, 30, 0).unwrap();
    let error = load_qqq_volatility_reference_with(now, Some(day("2026-11-02")), |_, _, _| {
        panic!("A future date must not reach Yahoo.")
    })
    .unwrap_err();
    assert_eq!(error.message(), "The QQQ reference date must be today or an earlier date.");
    let mut ends = Vec::new();
    let _ = load_qqq_volatility_reference_with(now, None, |_, _, end| {
        ends.push(end);
        Err(value_error!("offline"))
    });
    assert_eq!(ends, [day("2026-11-02")]);
}

#[test]
fn the_reference_reads_yahoo_adjusted_closes() {
    // End to end through the fake network: one QQQ chart request, adjusted closes, New York dates.
    let prices = qqq_prices();
    let (result, calls) =
        offline(yahoo(HashMap::from([("QQQ".to_string(), Chart::from_column(&prices, "QQQ"))])), || {
            load_qqq_volatility_reference_with(clock(17, 16), None, efficient_frontier::data::download_prices)
        });
    let result = result.unwrap();
    let (volatility, observations) = expected_volatility(&prices, day("2025-10-16"), day("2026-10-16"));
    assert!((result.volatility - volatility).abs() < 1e-14);
    assert_eq!(result.observations, observations);
    assert_eq!(chart_symbols(&calls), ["QQQ"]);
    assert_eq!(calls[0].param("period2"), Some(midnight(New_York, "2026-10-17").to_string().as_str()));
    let start = day("2026-10-17").checked_sub_months(Months::new(12)).unwrap() - Days::new(14);
    assert_eq!(calls[0].param("period1"), Some(midnight(New_York, &start.to_string()).to_string().as_str()));
}
