//! Browser sessions: a cookie identifies the server-side state of one browser.
//!
//! This replaces `st.session_state`. The state lives in memory. Every tab of one
//! browser shares the cookie, so it shares the study, and a reload keeps it.
//! The desktop keeps its study for the lifetime of the window, including idle
//! time. A store with explicit limits can expire idle sessions or remove the
//! least recently used session when it is full.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::http::header::COOKIE;
use axum::http::{HeaderMap, HeaderValue};
use efficient_frontier::frame::Table;
use efficient_frontier::story::VIEW_LABELS;
use sha2::{Digest, Sha256};

use crate::build::{Applied, BuildError, Job};
use crate::setup::{Setup, hex};
use crate::sources::QqqReference;

/// The name of the session cookie.
pub const COOKIE_NAME: &str = "pl_session";
/// The most sessions that the server keeps.
pub const MAX_SESSIONS: usize = 32;
/// The former desktop idle limit, also used by tests of explicit expiry.
#[cfg(test)]
pub const IDLE_LIMIT: Duration = Duration::from_secs(24 * 60 * 60);

/// The two result pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Portfolio,
    DataMethod,
}

impl View {
    pub const ALL: [View; 2] = [View::Portfolio, View::DataMethod];

    /// The stored value and form value.
    pub fn key(self) -> &'static str {
        match self {
            View::Portfolio => "Portfolio",
            View::DataMethod => "Data & method",
        }
    }

    /// The segment text: `story.VIEW_LABELS.get(name, name)`.
    pub fn label(self) -> &'static str {
        VIEW_LABELS.iter().find(|(key, _)| *key == self.key()).map_or(self.key(), |(_, label)| label)
    }

    pub fn parse(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|view| view.key() == key)
    }
}

/// The state of one browser, as the keys of `st.session_state` in `app.py`.
#[derive(Clone, Debug)]
pub struct Session {
    pub setup: Setup,
    pub view: View,
    /// `result` and `selection`: the last successful study.
    pub applied: Option<Arc<Applied>>,
    /// `coverage`: the member coverage of the last universe download, also after a failure.
    pub coverage: Option<Arc<Table>>,
    /// `build_error` and `build_error_kind`.
    pub build_error: Option<BuildError>,
    /// `pending_build`: the running build, if any.
    pub job: Option<Arc<Job>>,
    pub qqq_reference: Option<QqqReference>,
    pub risk_reference_unavailable: bool,
}

impl Session {
    pub fn new(setup: Setup) -> Self {
        Self {
            setup,
            view: View::Portfolio,
            applied: None,
            coverage: None,
            build_error: None,
            job: None,
            qqq_reference: None,
            risk_reference_unavailable: false,
        }
    }
}

pub type SessionRef = Arc<Mutex<Session>>;

struct Entry {
    session: SessionRef,
    last_used: Instant,
}

/// Every session by cookie value, with a size limit and an optional idle limit.
pub struct Sessions {
    map: Mutex<HashMap<String, Entry>>,
    capacity: usize,
    idle_limit: Option<Duration>,
    #[cfg(test)]
    clock: Mutex<Option<Instant>>,
}

impl Default for Sessions {
    fn default() -> Self {
        Self::new(MAX_SESSIONS, None)
    }
}

impl Sessions {
    pub fn new(capacity: usize, idle_limit: Option<Duration>) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            capacity: capacity.max(1),
            idle_limit,
            #[cfg(test)]
            clock: Mutex::new(None),
        }
    }

    /// The session of a cookie, which counts as a use of it.
    pub fn get(&self, id: &str) -> Option<SessionRef> {
        let (found, retired) = {
            let mut map = lock(&self.map);
            let retired = self.remove_idle(&mut map);
            let found = map.get_mut(id).map(|entry| {
                entry.last_used = self.now();
                entry.session.clone()
            });
            (found, retired)
        };
        retire(retired);
        found
    }

    /// Store a new session under a new random identifier. When the store is
    /// full, the least recently used session makes room.
    pub fn insert(&self, session: Session) -> (String, SessionRef) {
        let id = new_id();
        let handle = Arc::new(Mutex::new(session));
        let retired = {
            let mut map = lock(&self.map);
            let mut retired = self.remove_idle(&mut map);
            while map.len() >= self.capacity {
                let oldest = map.iter().min_by_key(|(_, entry)| entry.last_used).map(|(key, _)| key.clone());
                match oldest.and_then(|key| map.remove(&key)) {
                    Some(entry) => retired.push(entry.session),
                    None => break,
                }
            }
            map.insert(id.clone(), Entry { session: handle.clone(), last_used: self.now() });
            retired
        };
        retire(retired);
        (id, handle)
    }

    /// How many sessions the store keeps.
    #[cfg(test)]
    pub fn count(&self) -> usize {
        lock(&self.map).len()
    }

    #[cfg(test)]
    pub fn advance_clock(&self, elapsed: Duration) {
        let mut clock = lock(&self.clock);
        *clock = Some(clock.unwrap_or_else(Instant::now) + elapsed);
    }

    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = *lock(&self.clock) {
            return now;
        }
        Instant::now()
    }

    fn remove_idle(&self, map: &mut HashMap<String, Entry>) -> Vec<SessionRef> {
        let Some(idle_limit) = self.idle_limit else { return Vec::new() };
        let now = self.now();
        let idle: Vec<String> = map
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.last_used) >= idle_limit)
            .map(|(key, _)| key.clone())
            .collect();
        idle.iter().filter_map(|key| map.remove(key)).map(|entry| entry.session).collect()
    }
}

/// Cancel the builds of removed sessions; their memory goes with the last reference.
fn retire(sessions: Vec<SessionRef>) {
    for session in sessions {
        if let Some(job) = lock(&session).job.take() {
            job.cancel();
        }
    }
}

/// Lock a mutex. A panic in another holder does not make the state unusable:
/// every update below is a whole-field assignment, so the data stays consistent.
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// 128 unpredictable bits as hexadecimal text.
///
/// `RandomState` carries random SipHash keys from the operating system, so its
/// hashes of a counter are unpredictable without an extra dependency.
fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let random = RandomState::new();
    let mut digest = Sha256::new();
    for salt in 0..4_u64 {
        let mut hasher = random.build_hasher();
        hasher.write_u64(salt);
        digest.update(hasher.finish().to_le_bytes());
    }
    digest.update(COUNTER.fetch_add(1, Ordering::Relaxed).to_le_bytes());
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    digest.update(now.as_nanos().to_le_bytes());
    hex(&digest.finalize()[..16])
}

/// The session identifier in a request's cookies.
pub fn cookie_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all(COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .find(|(name, _)| *name == COOKIE_NAME)
        .map(|(_, value)| value.to_string())
}

/// The `Set-Cookie` value for a new session. `SameSite=Lax` keeps the cookie
/// off requests that other sites make in the background, but keeps it on a
/// link from another site, so open tabs do not lose their study.
pub fn set_cookie(id: &str) -> HeaderValue {
    HeaderValue::from_str(&format!("{COOKIE_NAME}={id}; Path=/; HttpOnly; SameSite=Lax"))
        .expect("hexadecimal cookie value")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_unique_hexadecimal_tokens() {
        let first = new_id();
        let second = new_id();
        assert_eq!(first.len(), 32);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    #[test]
    fn the_cookie_is_found_among_others() {
        let mut headers = HeaderMap::new();
        headers.insert(COOKIE, HeaderValue::from_static("theme=dark; pl_session=abc123; other=1"));
        assert_eq!(cookie_id(&headers).as_deref(), Some("abc123"));
        assert_eq!(cookie_id(&HeaderMap::new()), None);
        let value = set_cookie("abc123");
        assert_eq!(value.to_str().unwrap(), "pl_session=abc123; Path=/; HttpOnly; SameSite=Lax");
    }

    fn session() -> Session {
        Session::new(Setup::new(efficient_frontier::frame::Date::from_ymd_opt(2026, 10, 30).unwrap()))
    }

    #[test]
    fn a_full_store_removes_the_least_recently_used_session_and_cancels_its_build() {
        let pause = || std::thread::sleep(Duration::from_millis(5));
        let sessions = Sessions::new(2, Some(IDLE_LIMIT));
        let (first, first_handle) = sessions.insert(session());
        let job = crate::build::Job::new();
        lock(&first_handle).job = Some(job.clone());
        pause();
        let (second, _) = sessions.insert(session());
        pause();
        assert!(sessions.get(&first).is_some(), "a use makes the first session the most recent");
        pause();
        let (third, _) = sessions.insert(session());
        assert_eq!(sessions.count(), 2);
        assert!(sessions.get(&second).is_none(), "the least recently used session left");
        assert!(!job.is_cancelled());
        pause();
        assert!(sessions.get(&third).is_some());
        pause();
        sessions.insert(session());
        assert!(sessions.get(&first).is_none());
        assert!(job.is_cancelled(), "a removed session's build stops");
    }

    #[test]
    fn idle_sessions_are_removed() {
        let sessions = Sessions::new(MAX_SESSIONS, Some(IDLE_LIMIT));
        let (id, _) = sessions.insert(session());
        assert!(sessions.get(&id).is_some());
        sessions.advance_clock(IDLE_LIMIT);
        assert!(sessions.get(&id).is_none());
        assert_eq!(sessions.count(), 0);
    }

    #[test]
    fn default_desktop_session_survives_long_idle() {
        let sessions = Sessions::default();
        let (id, handle) = sessions.insert(session());
        let job = crate::build::Job::new();
        lock(&handle).job = Some(job.clone());
        sessions.advance_clock(IDLE_LIMIT + Duration::from_secs(1));
        assert!(Arc::ptr_eq(&sessions.get(&id).expect("the desktop session survives"), &handle));
        assert!(!job.is_cancelled(), "idle time does not cancel the desktop's build");
        assert_eq!(sessions.count(), 1);
    }

    #[test]
    fn sessions_are_stored_by_identifier() {
        let sessions = Sessions::default();
        let today = efficient_frontier::frame::Date::from_ymd_opt(2026, 10, 30).unwrap();
        let (id, handle) = sessions.insert(Session::new(Setup::new(today)));
        assert!(Arc::ptr_eq(&sessions.get(&id).unwrap(), &handle));
        assert!(sessions.get("unknown").is_none());
        assert_eq!(View::Portfolio.label(), "Overview");
        assert_eq!(View::DataMethod.label(), "Data & method");
        assert_eq!(View::parse("Data & method"), Some(View::DataMethod));
    }
}
