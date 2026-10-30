//! A current Nasdaq-100 risk reference from QQQ adjusted daily closes.
//!
//! Port of `efficient_frontier/risk_reference.py` in the Python release
//! (commit 6c6df6e). The app uses QQQ's trailing one-year volatility as its
//! starting volatility budget.

use chrono::{DateTime, Days, Months, NaiveTime, Utc};
use chrono_tz::America::New_York;

use crate::data::download_prices;
use crate::error::Result;
use crate::frame::{Date, Frame, std};
use crate::value_error;

/// QQQ's trailing one-year volatility and the returns that measure it.
#[derive(Clone, Debug, PartialEq)]
pub struct VolatilityReference {
    /// `QQQ`.
    pub symbol: &'static str,
    /// `Yahoo Finance adjusted close`.
    pub source: &'static str,
    /// The annualized sample standard deviation of daily simple returns.
    pub volatility: f64,
    /// 252.
    pub periods_per_year: u32,
    pub first_return_date: Date,
    pub last_return_date: Date,
    pub observations: usize,
}

/// Measure the last calendar year's daily simple-return volatility.
///
/// `as_of` is an inclusive date. Today's bar is excluded before 16:00 New
/// York time, including early-close days. Yahoo's download end is exclusive.
/// The observed last close anchors the year. No missing prices are filled.
pub fn load_qqq_volatility_reference(as_of: Option<Date>) -> Result<VolatilityReference> {
    load_qqq_volatility_reference_with(Utc::now(), as_of, download_prices)
}

/// [`load_qqq_volatility_reference`] at the instant `now`, with `download` in
/// place of [`download_prices`]. `download` receives the symbols and the
/// inclusive start and exclusive end dates.
pub fn load_qqq_volatility_reference_with(
    now: DateTime<Utc>,
    as_of: Option<Date>,
    download: impl FnOnce(&[String], Date, Date) -> Result<Frame>,
) -> Result<VolatilityReference> {
    let now = now.with_timezone(&New_York);
    let today = now.date_naive();
    let requested = as_of.unwrap_or(today);
    if requested > today {
        return Err(value_error!("The QQQ reference date must be today or an earlier date."));
    }
    let close = NaiveTime::from_hms_opt(16, 0, 0).expect("a valid time");
    let end = if requested == today && now.time() < close { requested } else { requested + Days::new(1) };
    // Extra closes cover weekends at both boundaries and the first return's base.
    let start = year_before(end) - Days::new(14);
    let prices = download(&["QQQ".to_string()], start, end)?;
    let column = prices.column("QQQ").expect("the download returns the requested symbol");
    let (dates, closes): (Vec<Date>, Vec<f64>) =
        prices.index.iter().copied().zip(column).filter(|(date, _)| *date >= start && *date < end).unzip();
    let Some(&last_close) = dates.last().filter(|last| **last >= end - Days::new(7)) else {
        return Err(value_error!("QQQ has no recent completed close. Retry the current volatility reference later."));
    };
    let boundary = year_before(last_close);
    if !dates.iter().any(|date| *date <= boundary) {
        return Err(value_error!("QQQ prices do not cover the full trailing year and its starting close."));
    }
    let returns: Vec<(Date, f64)> = (1..dates.len())
        .filter(|i| dates[*i] > boundary)
        .map(|i| (dates[i], closes[i] / closes[i - 1] - 1.0))
        .collect();
    if returns.len() < 200 || !returns.iter().all(|(_, value)| value.is_finite()) {
        return Err(value_error!("The QQQ trailing-year reference needs at least 200 finite daily returns."));
    }
    let values: Vec<f64> = returns.iter().map(|(_, value)| *value).collect();
    let volatility = std(&values, 1) * 252_f64.sqrt();
    if !volatility.is_finite() || volatility <= 1e-12 {
        return Err(value_error!("The QQQ trailing-year volatility must be finite and greater than zero."));
    }
    Ok(VolatilityReference {
        symbol: "QQQ",
        source: "Yahoo Finance adjusted close",
        volatility,
        periods_per_year: 252,
        first_return_date: returns[0].0,
        last_return_date: returns[returns.len() - 1].0,
        observations: returns.len(),
    })
}

/// `date - pandas.DateOffset(years=1)`: February 29 becomes February 28.
fn year_before(date: Date) -> Date {
    date.checked_sub_months(Months::new(12)).expect("dates stay in range")
}
