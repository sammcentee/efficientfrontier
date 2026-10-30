//! The only module that calls the network data functions.
//!
//! Every call to the library's universe, company-profile and QQQ-reference
//! modules is in this file. The rest of the app sees the [`Sources`] trait, so
//! tests supply their own data instead of the network, as the Python tests
//! patch these functions.

use efficient_frontier::error::Result;
use efficient_frontier::frame::{Date, Frame, Table};
use efficient_frontier::pyfmt::Dict;
use efficient_frontier::{companies, data, risk_reference, universe};
use std::sync::Arc;

use crate::cache::{Cache, ONE_HOUR};
use crate::setup::Market;

/// The progress callback of a universe download: `(completed, total, message)`.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, &str);

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
    fn universe_prices(&self, market: Market, start: Date, end: Date, progress: Progress) -> Result<UniversePrices>;
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
    fn universe_prices(&self, market: Market, start: Date, end: Date, progress: Progress) -> Result<UniversePrices> {
        let snapshot = self.members.get_or_try(market, || fetch_members(market).map(Arc::new))?;
        download_universe_prices(&snapshot, start, end, progress)
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

/// `universe::download_universe_prices(snapshot, start, end, progress)`, as [`UniversePrices`].
/// `progress` receives `(completed, total, message)`.
pub fn download_universe_prices(
    snapshot: &UniverseSnapshot,
    start: Date,
    end: Date,
    progress: Progress,
) -> Result<UniversePrices> {
    let loaded = universe::download_universe_prices(snapshot, start, end, Some(progress))?;
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
