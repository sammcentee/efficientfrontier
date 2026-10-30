//! Browser sessions: a cookie identifies the server-side state of one browser.
//!
//! This replaces `st.session_state`. The state lives in memory for the life of
//! the process. Every tab of one browser shares the cookie, so it shares the
//! study, and a reload keeps it.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

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

/// Every session by cookie value.
#[derive(Default)]
pub struct Sessions {
    map: Mutex<HashMap<String, SessionRef>>,
}

impl Sessions {
    pub fn get(&self, id: &str) -> Option<SessionRef> {
        lock(&self.map).get(id).cloned()
    }

    /// Store a new session under a new random identifier.
    pub fn insert(&self, session: Session) -> (String, SessionRef) {
        let id = new_id();
        let handle = Arc::new(Mutex::new(session));
        lock(&self.map).insert(id.clone(), handle.clone());
        (id, handle)
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

/// The `Set-Cookie` value for a new session. `SameSite=Strict` keeps other
/// sites from posting to this server with the session.
pub fn set_cookie(id: &str) -> HeaderValue {
    HeaderValue::from_str(&format!("{COOKIE_NAME}={id}; Path=/; HttpOnly; SameSite=Strict"))
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
        assert!(value.to_str().unwrap().starts_with("pl_session=abc123; Path=/; HttpOnly; SameSite=Strict"));
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
