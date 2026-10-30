//! In-memory caches with the behavior of `st.cache_data`.
//!
//! Downloads, company profiles and the QQQ reference expire one hour after they
//! were stored. The selection, frontier and walk-forward results keep the two
//! most recently used studies. Failures are not cached, because Streamlit does
//! not cache exceptions. The caches are shared by every browser session.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use chrono::Datelike;
use efficient_frontier::frame::{Date, Frame};
use efficient_frontier::out_of_sample::OutOfSampleStudy;
use efficient_frontier::selection::{SelectionFrontier, SelectionStudy};
use indexmap::IndexMap;
use sha2::{Digest, Sha256};

use crate::session::lock;
use crate::setup::{Market, ModelSettings};
use crate::sources::{CompanyProfile, QqqReference, UniversePrices};

pub const ONE_HOUR: Duration = Duration::from_secs(3600);

/// A SHA-256 digest that identifies cached inputs.
pub type Key = [u8; 32];

/// Values by key, with an optional lifetime and an optional entry limit.
///
/// A full cache drops its least recently used entry. Lookups scan a short list,
/// which is faster than hashing for the few entries that one local user creates.
pub struct Cache<K, V> {
    ttl: Option<Duration>,
    capacity: Option<usize>,
    entries: Mutex<VecDeque<(K, V, Instant)>>,
    computed: AtomicUsize,
}

impl<K: PartialEq, V: Clone> Cache<K, V> {
    pub fn new(ttl: Option<Duration>, capacity: Option<usize>) -> Self {
        Self { ttl, capacity, entries: Mutex::new(VecDeque::new()), computed: AtomicUsize::new(0) }
    }

    /// The stored value, which becomes the most recently used entry.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut entries = lock(&self.entries);
        if let Some(ttl) = self.ttl {
            entries.retain(|(_, _, stored)| stored.elapsed() < ttl);
        }
        let position = entries.iter().position(|(stored, _, _)| stored == key)?;
        let entry = entries.remove(position)?;
        let value = entry.1.clone();
        entries.push_back(entry);
        Some(value)
    }

    pub fn insert(&self, key: K, value: V) {
        let mut entries = lock(&self.entries);
        entries.retain(|(stored, _, _)| stored != &key);
        entries.push_back((key, value, Instant::now()));
        if let Some(capacity) = self.capacity {
            while entries.len() > capacity {
                entries.pop_front();
            }
        }
    }

    /// The stored value, or the computed one. An error is returned and not stored.
    /// The lock is not held while `compute` runs.
    pub fn get_or_try<E>(&self, key: K, compute: impl FnOnce() -> Result<V, E>) -> Result<V, E> {
        if let Some(value) = self.get(&key) {
            return Ok(value);
        }
        let value = compute()?;
        self.computed.fetch_add(1, Ordering::Relaxed);
        self.insert(key, value.clone());
        Ok(value)
    }

    /// [`Cache::get_or_try`] for a computation that cannot fail.
    pub fn get_or_insert_with(&self, key: K, compute: impl FnOnce() -> V) -> V {
        match self.get_or_try(key, || Ok::<V, std::convert::Infallible>(compute())) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }

    /// How many values this cache has computed: the misses that succeeded.
    #[cfg(test)]
    pub fn computed(&self) -> usize {
        self.computed.load(Ordering::Relaxed)
    }
}

/// Every cache of the app, as the `@st.cache_data` functions of `app.py`.
pub struct Caches {
    /// `fetch_universe(market, start, end)`.
    pub universe: Cache<(Market, Date, Date), Arc<UniversePrices>>,
    /// `fetch_prices(tickers, start, end)`.
    pub prices: Cache<(Vec<String>, Date, Date), Arc<Frame>>,
    /// `fetch_benchmarks(dates)`, by a digest of the dates.
    pub benchmarks: Cache<Key, Arc<Frame>>,
    /// `fetch_default_risk(reference_day)`.
    pub qqq_reference: Cache<Date, QqqReference>,
    /// `fetch_company_profiles(symbols)`, at most 32 symbol lists.
    pub profiles: Cache<Vec<String>, Arc<IndexMap<String, CompanyProfile>>>,
    /// `calculate_selection`, by [`selection_key`].
    pub selection: Cache<Key, Arc<SelectionStudy>>,
    /// `calculate_frontier`, by the selection key of its study.
    pub frontier: Cache<Key, Arc<SelectionFrontier>>,
    /// `calculate_out_of_sample`, by the selection key of its study.
    pub out_of_sample: Cache<Key, Arc<OutOfSampleStudy>>,
}

impl Default for Caches {
    fn default() -> Self {
        Self {
            universe: Cache::new(Some(ONE_HOUR), None),
            prices: Cache::new(Some(ONE_HOUR), None),
            benchmarks: Cache::new(Some(ONE_HOUR), None),
            qqq_reference: Cache::new(Some(ONE_HOUR), None),
            profiles: Cache::new(Some(ONE_HOUR), Some(32)),
            selection: Cache::new(None, Some(2)),
            frontier: Cache::new(None, Some(2)),
            out_of_sample: Cache::new(None, Some(2)),
        }
    }
}

fn hash_text(hasher: &mut Sha256, text: &str) {
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

fn hash_dates(hasher: &mut Sha256, dates: &[Date]) {
    hasher.update((dates.len() as u64).to_le_bytes());
    for date in dates {
        hasher.update(date.num_days_from_ce().to_le_bytes());
    }
}

/// Feed a frame's dates, labels and exact values to a hasher.
fn hash_frame(hasher: &mut Sha256, frame: &Frame) {
    hash_dates(hasher, &frame.index);
    hasher.update((frame.ncols() as u64).to_le_bytes());
    for column in &frame.columns {
        hash_text(hasher, column);
    }
    let bytes: Vec<u8> = frame.values.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect();
    hasher.update(&bytes);
}

/// The identity of a list of dates.
pub fn dates_key(dates: &[Date]) -> Key {
    let mut hasher = Sha256::new();
    hash_dates(&mut hasher, dates);
    hasher.finalize().into()
}

/// The identity of one selection study: its prices, comparison prices, settings,
/// volatility budget and model revision.
pub fn selection_key(
    prices: &Frame,
    benchmarks: Option<&Frame>,
    settings: &ModelSettings,
    target_volatility: f64,
    revision: &str,
) -> Key {
    let mut hasher = Sha256::new();
    hash_frame(&mut hasher, prices);
    match benchmarks {
        Some(frame) => {
            hasher.update([1]);
            hash_frame(&mut hasher, frame);
        }
        None => hasher.update([0]),
    }
    for (name, value) in settings.entries() {
        hash_text(&mut hasher, name);
        hasher.update(value.as_f64().unwrap_or(f64::NAN).to_bits().to_le_bytes());
    }
    hasher.update(target_volatility.to_bits().to_le_bytes());
    hash_text(&mut hasher, revision);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::{Setup, resolve};

    fn frame(values: &[f64]) -> Frame {
        let start = Date::from_ymd_opt(2021, 1, 4).unwrap();
        let index = (0..values.len() as u64).map(|day| start + chrono::Days::new(day)).collect();
        Frame::from_columns(index, vec![("A".into(), values.to_vec())])
    }

    #[test]
    fn entries_expire_and_the_least_recently_used_entry_leaves_first() {
        let cache: Cache<&str, i32> = Cache::new(None, Some(2));
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.get(&"a"), Some(1));
        cache.insert("c", 3);
        assert_eq!((cache.get(&"a"), cache.get(&"b"), cache.get(&"c")), (Some(1), None, Some(3)));
        let expired: Cache<&str, i32> = Cache::new(Some(Duration::ZERO), None);
        expired.insert("a", 1);
        assert_eq!(expired.get(&"a"), None);
    }

    #[test]
    fn failures_are_not_cached() {
        let cache: Cache<u8, i32> = Cache::new(Some(ONE_HOUR), None);
        assert_eq!(cache.get_or_try(1, || Err::<i32, &str>("offline")), Err("offline"));
        assert_eq!(cache.get_or_try(1, || Ok::<i32, &str>(7)), Ok(7));
        assert_eq!(cache.get_or_try(1, || Ok::<i32, &str>(8)), Ok(7));
        assert_eq!(cache.computed(), 1);
    }

    #[test]
    fn selection_keys_follow_every_input() {
        let today = Date::from_ymd_opt(2026, 10, 30).unwrap();
        let settings = resolve(&Setup::new(today), today).settings;
        let prices = frame(&[1.0, 2.0, 3.0]);
        let key = selection_key(&prices, None, &settings, 0.2, "v3");
        assert_eq!(key, selection_key(&prices.clone(), None, &settings, 0.2, "v3"));
        assert_ne!(key, selection_key(&frame(&[1.0, 2.0, 3.5]), None, &settings, 0.2, "v3"));
        assert_ne!(key, selection_key(&prices, Some(&prices), &settings, 0.2, "v3"));
        assert_ne!(key, selection_key(&prices, None, &settings, 0.21, "v3"));
        assert_ne!(key, selection_key(&prices, None, &ModelSettings { max_weight: 0.2, ..settings }, 0.2, "v3"));
        assert_ne!(key, selection_key(&prices, None, &settings, 0.2, "v4"));
        assert_ne!(dates_key(&prices.index), dates_key(&prices.index[1..]));
    }
}
