//! Shared constructions for the ports of the legacy research tests.
//!
//! The helpers rebuild the pandas date ranges and price paths of the Python
//! tests, so the same numbers reach the same functions.

#![allow(dead_code)]

use chrono::{Datelike, Days, Months, Weekday};
use efficient_frontier::frame::{Date, Frame};
use serde_json::Value;

pub fn date(year: i32, month: u32, day: u32) -> Date {
    Date::from_ymd_opt(year, month, day).unwrap()
}

pub fn parse_date(text: &str) -> Date {
    Date::parse_from_str(text, "%Y-%m-%d").unwrap()
}

/// `pandas.bdate_range(start, periods=count)`: weekdays from `start`.
pub fn business_days(start: Date, count: usize) -> Vec<Date> {
    let mut dates = Vec::with_capacity(count);
    let mut day = start;
    while dates.len() < count {
        if day.weekday().number_from_monday() <= 5 {
            dates.push(day);
        }
        day = day + Days::new(1);
    }
    dates
}

/// `pandas.date_range(start, periods=count, freq="D")`.
pub fn calendar_days(start: Date, count: usize) -> Vec<Date> {
    (0..count as u64).map(|i| start + Days::new(i)).collect()
}

/// `pandas.date_range(start, periods=count, freq="W-<weekday>")`: the first such weekday on or after `start`.
pub fn weeks(start: Date, count: usize, weekday: Weekday) -> Vec<Date> {
    let mut first = start;
    while first.weekday() != weekday {
        first = first + Days::new(1);
    }
    (0..count as u64).map(|i| first + Days::new(7 * i)).collect()
}

/// `pandas.date_range(start, periods=count, freq="MS")` for a month start.
pub fn month_starts(start: Date, count: usize) -> Vec<Date> {
    (0..count as u32).map(|i| start + Months::new(i)).collect()
}

/// `pandas.date_range(start, periods=count, freq="ME")` for a month end.
pub fn month_ends(start: Date, count: usize) -> Vec<Date> {
    let first = start.with_day(1).unwrap();
    (0..count as u32).map(|i| first + Months::new(i + 1) - Days::new(1)).collect()
}

/// `numpy.tile(pattern, repeats)`.
pub fn tile(pattern: &[f64], repeats: usize) -> Vec<f64> {
    (0..pattern.len() * repeats).map(|i| pattern[i % pattern.len()]).collect()
}

/// `numpy.repeat(values, repeats)`.
pub fn repeat(values: &[f64], repeats: usize) -> Vec<f64> {
    values.iter().flat_map(|value| std::iter::repeat_n(*value, repeats)).collect()
}

/// `100 * numpy.vstack([ones, numpy.cumprod(1 + returns, axis=0)])` with the given dates.
///
/// `columns[j]` holds the returns of asset `j`.
pub fn prices_from_columns(columns: &[Vec<f64>], names: &[&str], index: Vec<Date>) -> Frame {
    let rows = columns[0].len() + 1;
    assert_eq!(index.len(), rows, "one date per price");
    let values = columns
        .iter()
        .zip(names)
        .map(|(returns, name)| {
            let mut growth = 1.0;
            let mut path = vec![100.0];
            for value in returns {
                growth *= 1.0 + value;
                path.push(100.0 * growth);
            }
            (name.to_string(), path)
        })
        .collect();
    Frame::from_columns(index, values)
}

/// `prices_from_returns` of `tests/test_profiles.py`: month ends from 2020-01-31.
pub fn monthly_prices(columns: &[Vec<f64>], names: &[&str]) -> Frame {
    prices_from_columns(columns, names, month_ends(date(2020, 1, 31), columns[0].len() + 1))
}

/// The two-regime history of `tests/test_profiles.py`.
pub fn regime_returns() -> Vec<Vec<f64>> {
    let steady = tile(&[-0.015, 0.015], 9).into_iter().map(|value| 0.01 + value).collect();
    let transient =
        repeat(&[0.03, -0.02, 0.04], 6).into_iter().zip(tile(&[-0.001, 0.001], 9)).map(|(a, b)| a + b).collect();
    vec![steady, transient]
}

pub fn regime_prices() -> Frame {
    monthly_prices(&regime_returns(), &["Steady", "Transient"])
}

/// Numbers from the Python test data in `tests/fixtures/golden/legacy_test_data.json`.
pub fn test_data(key: &str) -> Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden/legacy_test_data.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let data: Value = serde_json::from_str(&text).unwrap();
    data[key].clone()
}

pub fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|item| item.as_f64().unwrap()).collect()
}

/// `numpy.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)`, with equal NaNs.
pub fn assert_allclose(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len(), "lengths differ: {actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        let close = (a.is_nan() && e.is_nan()) || (a - e).abs() <= atol + rtol * e.abs();
        assert!(close, "{actual:?} vs {expected:?} (atol {atol:e}, rtol {rtol:e})");
    }
}

/// `pytest.approx(expected, rel=1e-6, abs=1e-12)`.
pub fn approx(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= (1e-6 * expected.abs()).max(1e-12), "{actual} vs approximately {expected}");
}

/// A column of a frame by name.
pub fn column(frame: &Frame, name: &str) -> Vec<f64> {
    frame.column(name).unwrap_or_else(|| panic!("column {name:?} is missing"))
}
