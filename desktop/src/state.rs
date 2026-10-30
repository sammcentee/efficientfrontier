//! The shared server state: sessions, caches, data sources and settings.

use std::sync::Arc;
use std::time::Duration;

use efficient_frontier::frame::Date;
use efficient_frontier::pyfmt;

use crate::cache::Caches;
use crate::guard::guard;
use crate::session::{Session, Sessions};
use crate::setup::{Market, Setup};
use crate::sources::{Live, QqqReference, Sources};

/// How long the overview waits for company profiles (`wait(pending, timeout=10)`).
pub const PROFILE_TIMEOUT: Duration = Duration::from_secs(10);

pub struct AppState {
    pub sessions: Sessions,
    pub caches: Caches,
    pub sources: Arc<dyn Sources>,
    /// The longest wait for company profiles.
    pub profile_timeout: Duration,
    /// Today's local date. Tests can fix it.
    pub clock: fn() -> Date,
}

impl AppState {
    pub fn new(sources: Arc<dyn Sources>) -> Self {
        Self {
            sessions: Sessions::default(),
            caches: Caches::default(),
            sources,
            profile_timeout: PROFILE_TIMEOUT,
            clock: local_today,
        }
    }

    /// The state of the real app, with network sources.
    pub fn live() -> Self {
        Self::new(Arc::new(Live::default()))
    }

    pub fn today(&self) -> Date {
        (self.clock)()
    }

    /// A new browser session with the defaults of `init_state`.
    ///
    /// Before the first result of a market that downloads prices, the volatility
    /// budget starts at QQQ's trailing-year volatility, rounded to two decimals.
    /// When the reference is unavailable, the 20% default stays and the landing
    /// page says that it is a manual fallback.
    pub fn new_session(&self) -> Session {
        let today = self.today();
        let mut session = Session::new(Setup::new(today));
        if session.applied.is_none() && !matches!(session.setup.market, Market::Demo | Market::UploadCsv) {
            match self.qqq_reference(today) {
                Ok(reference) => {
                    session.setup.target_vol = round2(reference.volatility * 100.0);
                    session.qqq_reference = Some(reference);
                }
                Err(_) => session.risk_reference_unavailable = true,
            }
        }
        session
    }

    /// `fetch_default_risk(date.today())`, cached for one hour per day.
    fn qqq_reference(&self, today: Date) -> Result<QqqReference, String> {
        guard(|| self.caches.qqq_reference.get_or_try(today, || self.sources.qqq_reference()))?
            .map_err(|error| error.message().to_string())
    }
}

/// Today's date on this computer, as `date.today()`.
pub fn local_today() -> Date {
    chrono::Local::now().date_naive()
}

/// Python's `round(value, 2)`: correctly rounded, with ties to even.
pub fn round2(value: f64) -> f64 {
    pyfmt::format(value, ".2f").parse().expect("a formatted number")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_python() {
        assert_eq!(round2(0.198350420824 * 100.0), 19.84);
        assert_eq!(round2(20.0), 20.0);
        assert_eq!(round2(0.125), 0.12);
        assert_eq!(round2(2.675), 2.67);
    }
}
