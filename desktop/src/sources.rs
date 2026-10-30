//! The only module that calls the network data functions.
//!
//! Every call to the library's universe, company-profile and QQQ-reference
//! modules is in this file. The rest of the app sees the [`Sources`] trait, so
//! tests supply their own data instead of the network, as the Python tests
//! patch these functions.

use efficient_frontier::error::{Error, Result};
use efficient_frontier::frame::{Date, Frame, Table};
use efficient_frontier::pyfmt::Dict;
use efficient_frontier::{companies, data, risk_reference, universe};
use std::sync::Arc;

use crate::cache::{Cache, ONE_HOUR};
use crate::setup::Market;

/// The progress callback of a universe download: `(completed, total, message)`.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, &str);

/// True once the build that asked for the data is cancelled.
pub type Cancelled<'a> = &'a dyn Fn() -> bool;

/// The message of a download that a cancelled build stopped.
pub const CANCELLED: &str = "The build was cancelled.";

/// The error of a download that a cancelled build stopped. It is a runtime
/// error, so the universe download stops instead of retrying the batch symbol
/// by symbol, as it does after a value error.
pub fn cancelled_error() -> Error {
    Error::Runtime(CANCELLED.into())
}

/// A current company name, one business sentence and the profile URL.
/// Every field is empty when the provider has no profile (Python's `{}`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompanyProfile {
    pub name: String,
    pub description: String,
    pub source_url: String,
}

/// QQQ's trailing one-year volatility, the default volatility budget.
#[derive(Clone, Debug, PartialEq)]
pub struct QqqReference {
    /// Annual volatility as a fraction.
    pub volatility: f64,
    pub last_return_date: Date,
}

/// The prices of one current universe and the coverage of every member (`universe.UniversePrices`).
#[derive(Clone, Debug)]
pub struct UniversePrices {
    /// Complete prices of the included members.
    pub prices: Frame,
    /// One row per member, with columns `symbol`, `name`, `industry`, `status`,
    /// `reason`, `observations`, `first_date` and `last_date`.
    pub coverage: Table,
    /// SPY and QQQ on the same dates, when available.
    pub benchmarks: Option<Frame>,
    /// The `universe_*` metadata of the study, including `universe_coverage`.
    pub metadata: Dict,
}

/// Market data for the app. [`Live`] downloads it; tests supply fixtures.
pub trait Sources: Send + Sync {
    /// The current members of a universe market and their complete prices (`fetch_universe`).
    /// The download stops before its next batch once `cancelled` returns true.
    fn universe_prices(
        &self,
        market: Market,
        start: Date,
        end: Date,
        progress: Progress,
        cancelled: Cancelled,
    ) -> Result<UniversePrices>;
    /// Yahoo adjusted prices; `end` is exclusive (`data.download_prices`).
    fn prices(&self, tickers: &[String], start: Date, end: Date) -> Result<Frame>;
    /// SPY and QQQ on exactly these dates (`data.download_benchmarks`).
    fn benchmarks(&self, dates: &[Date]) -> Result<Frame>;
    /// The original 60 holdings with current Yahoo symbols (`data.original_tickers(current_symbols=True)`).
    fn original_tickers(&self) -> Result<Vec<String>>;
    /// A current company profile; empty when unavailable (`companies.fetch_company_profile`).
    fn company_profile(&self, symbol: &str) -> CompanyProfile;
    /// QQQ's trailing-year volatility (`risk_reference.load_qqq_volatility_reference`).
    fn qqq_reference(&self) -> Result<QqqReference>;
}

/// The network sources. The member lists are cached for one hour, as `fetch_members` in `app.py`.
pub struct Live {
    members: Cache<Market, Arc<UniverseSnapshot>>,
}

impl Default for Live {
    fn default() -> Self {
        Self { members: Cache::new(Some(ONE_HOUR), None) }
    }
}

impl Sources for Live {
    fn universe_prices(
        &self,
        market: Market,
        start: Date,
        end: Date,
        progress: Progress,
        cancelled: Cancelled,
    ) -> Result<UniversePrices> {
        let snapshot = self.members.get_or_try(market, || fetch_members(market).map(Arc::new))?;
        if cancelled() {
            return Err(cancelled_error());
        }
        download_universe_prices(&snapshot, start, end, progress, cancelled)
    }

    fn prices(&self, tickers: &[String], start: Date, end: Date) -> Result<Frame> {
        data::download_prices(tickers, start, end)
    }

    fn benchmarks(&self, dates: &[Date]) -> Result<Frame> {
        data::download_benchmarks(dates)
    }

    fn original_tickers(&self) -> Result<Vec<String>> {
        data::original_tickers(None, true)
    }

    fn company_profile(&self, symbol: &str) -> CompanyProfile {
        fetch_company_profile(symbol)
    }

    fn qqq_reference(&self) -> Result<QqqReference> {
        load_qqq_volatility_reference()
    }
}

// ----------------------------------------------------------------------------- network functions
// The library's `universe`, `companies` and `risk_reference` modules, in the
// shapes that the app uses.

pub use efficient_frontier::universe::UniverseSnapshot;

/// The current members: `universe::fetch_sp500_nasdaq100()` for [`Market::Combined`],
/// else `universe::fetch_nasdaq100()`.
pub fn fetch_members(market: Market) -> Result<UniverseSnapshot> {
    match market {
        Market::Combined => universe::fetch_sp500_nasdaq100(),
        _ => universe::fetch_nasdaq100(),
    }
}

/// A batch download that fails, before it reaches the network, once the build is cancelled.
pub fn cancellable<'a>(
    cancelled: Cancelled<'a>,
    mut download: impl FnMut(&[String], Date, Date, bool) -> Result<Frame> + 'a,
) -> impl FnMut(&[String], Date, Date, bool) -> Result<Frame> + 'a {
    move |symbols, start, end, allow_incomplete| {
        if cancelled() {
            return Err(cancelled_error());
        }
        download(symbols, start, end, allow_incomplete)
    }
}

/// `universe::download_universe_prices(snapshot, start, end, progress)`, as [`UniversePrices`].
/// `progress` receives `(completed, total, message)`. A cancelled build stops the
/// download at its next batch of symbols.
pub fn download_universe_prices(
    snapshot: &UniverseSnapshot,
    start: Date,
    end: Date,
    progress: Progress,
    cancelled: Cancelled,
) -> Result<UniversePrices> {
    let mut download = cancellable(cancelled, data::download_adjusted_close);
    let loaded = universe::download_universe_prices_with(snapshot, start, end, Some(progress), &mut download)?;
    Ok(UniversePrices {
        prices: loaded.prices,
        coverage: loaded.coverage,
        benchmarks: Some(loaded.benchmarks),
        metadata: loaded.metadata,
    })
}

/// `companies::fetch_company_profile(symbol)`: empty fields when unavailable.
pub fn fetch_company_profile(symbol: &str) -> CompanyProfile {
    companies::fetch_company_profile(symbol)
        .map(|company| CompanyProfile {
            name: company.name,
            description: company.description,
            source_url: company.source_url,
        })
        .unwrap_or_default()
}

/// `risk_reference::load_qqq_volatility_reference()`: QQQ's volatility and its last return date.
pub fn load_qqq_volatility_reference() -> Result<QqqReference> {
    let reference = risk_reference::load_qqq_volatility_reference(None)?;
    Ok(QqqReference { volatility: reference.volatility, last_return_date: reference.last_return_date })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use efficient_frontier::universe::{Member, UniverseSnapshot};

    use super::*;

    /// Complete prices on ten business days for the requested symbols.
    fn prices(symbols: &[String]) -> Frame {
        let start = Date::from_ymd_opt(2024, 1, 1).unwrap();
        let dates: Vec<Date> = (0..10).map(|day| start + chrono::Days::new(day)).collect();
        let columns =
            symbols.iter().map(|symbol| (symbol.clone(), (0..10).map(|i| 100.0 + i as f64).collect())).collect();
        Frame::from_columns(dates, columns)
    }

    #[test]
    fn a_cancelled_universe_download_stops_at_the_next_batch() {
        let members: Vec<Member> = (0..60)
            .map(|i| Member { symbol: format!("S{i:02}"), name: format!("Stock {i}"), industry: "Unknown".into() })
            .collect();
        let snapshot = UniverseSnapshot::new(members, "https://example.com", "2026-10-01", "2026-10-30T09:00:00+00:00");
        let start = Date::from_ymd_opt(2024, 1, 1).unwrap();
        let end = Date::from_ymd_opt(2024, 2, 1).unwrap();
        let calls = Cell::new(0);
        // The benchmark download and the first batch run; then the user cancels.
        let cancelled = || calls.get() >= 2;
        let mut download = cancellable(&cancelled, |symbols: &[String], _, _, _| {
            calls.set(calls.get() + 1);
            Ok(prices(symbols))
        });
        let error = universe::download_universe_prices_with(&snapshot, start, end, None, &mut download).unwrap_err();
        assert_eq!(error.message(), CANCELLED);
        assert!(error.is_runtime());
        assert_eq!(calls.get(), 2, "batches 2 and 3 of 25 symbols were never requested");
        let never = || false;
        let mut download = cancellable(&never, |symbols: &[String], _, _, _| Ok(prices(symbols)));
        let loaded = universe::download_universe_prices_with(&snapshot, start, end, None, &mut download).unwrap();
        assert_eq!(loaded.prices.ncols(), 60);
    }
}
