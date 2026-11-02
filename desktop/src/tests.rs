//! Router tests: the intent of `tests/test_app.py`, driven in-process.
//!
//! [`Fake`] replaces the network, as the Python tests patch the download
//! functions. A test client keeps the session cookie and follows builds through
//! `/api/job`. The last tests send the window's requests through the bridge.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{HeaderMap, Method, Request, StatusCode, header};
use efficient_frontier::error::{Error, Result};
use efficient_frontier::frame::{Cell, Date, Frame, Table, iso};
use efficient_frontier::pyfmt::{Dict, Json};
use efficient_frontier::selection::SELECTED;
use efficient_frontier::{dict, io_error, value_error};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

use crate::bridge::{Bridge, Forwarded, webview_request};
use crate::build::Applied;
use crate::render::overview::budget_note;
use crate::routes::OLDER_STUDY;
use crate::session::{IDLE_LIMIT, Session, Sessions, lock};
use crate::setup::{Market, hex};
use crate::sources::{Cancelled, CompanyProfile, Progress, QqqReference, Sources, UniversePrices, cancelled_error};
use crate::state::AppState;

// Run with --release to check the optimizer-sensitive RUSTSEC-2024-0429 regression.
#[cfg(target_os = "linux")]
mod glib_variant_str_iter {
    use glib::variant::ToVariant;

    const VALUES: [&str; 6] = ["", "alpha", "Éire", "日本語", "omega", "end"];

    #[test]
    fn forward() {
        let variant = VALUES.to_variant();
        assert_eq!(variant.array_iter_str().unwrap().collect::<Vec<_>>(), VALUES);

        let mut iter = variant.array_iter_str().unwrap();
        assert_eq!(iter.nth(2), Some("Éire"));
        assert_eq!(iter.next(), Some("日本語"));
        assert_eq!(iter.last(), Some("end"));
    }

    #[test]
    fn backward() {
        let variant = VALUES.to_variant();
        assert_eq!(
            variant.array_iter_str().unwrap().rev().collect::<Vec<_>>(),
            VALUES.into_iter().rev().collect::<Vec<_>>()
        );

        let mut iter = variant.array_iter_str().unwrap();
        assert_eq!(iter.nth_back(2), Some("日本語"));
        assert_eq!(iter.next_back(), Some("Éire"));
        assert_eq!(iter.last(), Some("alpha"));
    }

    #[test]
    fn mixed_and_empty() {
        let variant = VALUES.to_variant();
        let mut iter = variant.array_iter_str().unwrap();
        assert_eq!(iter.len(), 6);
        assert_eq!(iter.next(), Some(""));
        assert_eq!(iter.next_back(), Some("end"));
        assert_eq!(iter.nth(1), Some("Éire"));
        assert_eq!(iter.nth_back(1), Some("日本語"));
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.last(), None);

        let empty = Vec::<&str>::new().to_variant();
        let mut iter = empty.array_iter_str().unwrap();
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}

// ----------------------------------------------------------------------------- fixtures

fn day(text: &str) -> Date {
    Date::parse_from_str(text, "%Y-%m-%d").unwrap()
}

fn today() -> Date {
    day("2026-10-30")
}

/// A small deterministic generator of normal draws (xorshift and Box-Muller).
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn business_days(start: Date, count: usize) -> Vec<Date> {
    let mut dates = Vec::with_capacity(count);
    let mut date = start;
    while dates.len() < count {
        if chrono::Datelike::weekday(&date).number_from_monday() <= 5 {
            dates.push(date);
        }
        date = date.succ_opt().unwrap();
    }
    dates
}

/// Twelve stocks with a common factor on 240 business days, as the `market_prices` fixture.
fn market_prices() -> Frame {
    let dates = business_days(day("2021-01-04"), 240);
    let mut rng = Rng(902);
    let mut levels = [100.0_f64; 12];
    let mut rows = Vec::new();
    for _ in &dates {
        let common = 0.004 * rng.normal();
        for level in &mut levels {
            *level *= (0.0005 + 0.008 * rng.normal() + common).exp();
        }
        rows.push(levels.to_vec());
    }
    let columns = (0..12).map(|number| format!("STOCK_{number:02}")).collect();
    Frame::from_rows(dates, columns, &rows)
}

/// SPY and QQQ on the same dates, as the `benchmark_prices` fixture.
fn benchmark_prices(dates: &[Date]) -> Frame {
    let mut rng = Rng(903);
    let (mut spy, mut qqq) = (100.0_f64, 100.0_f64);
    let rows: Vec<Vec<f64>> = dates
        .iter()
        .map(|_| {
            spy *= (0.0004 + 0.009 * rng.normal()).exp();
            qqq *= (0.0006 + 0.013 * rng.normal()).exp();
            vec![spy, qqq]
        })
        .collect();
    Frame::from_rows(dates.to_vec(), vec!["SPY".into(), "QQQ".into()], &rows)
}

/// Two frames side by side on the same dates.
fn join(left: &Frame, right: &Frame) -> Frame {
    let mut columns: Vec<(String, Vec<f64>)> =
        left.columns.iter().enumerate().map(|(j, name)| (name.clone(), left.col(j))).collect();
    columns.extend(right.columns.iter().enumerate().map(|(j, name)| (name.clone(), right.col(j))));
    Frame::from_columns(left.index.clone(), columns)
}

fn first_rows(frame: &Frame, rows: usize) -> Frame {
    frame.rows(0, rows)
}

fn csv(frame: &Frame) -> Vec<u8> {
    frame.to_csv("Date").into_bytes()
}

const UNIVERSE_LIMITATION: &str = "This study uses current Nasdaq-100 members, not the members at each historical date. \
It excludes securities without complete prices for the selected period. Both choices can favor survivors and distort historical results. \
This is not a historical reconstruction of the Nasdaq-100.";

/// The `universe_data` fixture: twelve included members and one excluded.
fn universe_fixture() -> UniversePrices {
    let prices = market_prices();
    let mut symbols = prices.columns.clone();
    symbols.push("MISSING".into());
    let excluded = |symbol: &String| symbol == "MISSING";
    let coverage = Table::range(symbols.len())
        .with("symbol", symbols.iter().map(|s| Cell::from(s.as_str())).collect())
        .with("name", symbols.iter().map(|s| Cell::from(format!("Company {s}"))).collect())
        .with("industry", vec![Cell::from("Unknown"); symbols.len()])
        .with("status", symbols.iter().map(|s| Cell::from(if excluded(s) { "excluded" } else { "included" })).collect())
        .with(
            "reason",
            symbols.iter().map(|s| Cell::from(if excluded(s) { "Incomplete prices" } else { "" })).collect(),
        );
    let records: Vec<Json> = symbols
        .iter()
        .map(|s| {
            Json::Dict(dict! {
                "symbol" => s.as_str(), "name" => format!("Company {s}"), "industry" => "Unknown",
                "status" => if excluded(s) { "excluded" } else { "included" },
                "reason" => if excluded(s) { "Incomplete prices" } else { "" },
            })
        })
        .collect();
    let metadata = dict! {
        "universe_requested" => 13, "universe_included" => 12, "universe_excluded" => 1,
        "universe_coverage" => Json::List(records), "universe_limitation" => UNIVERSE_LIMITATION,
        "universe_source_date" => "2026-10-01", "universe_retrieved_at" => "2026-10-16T09:00:00+00:00",
    };
    let benchmarks = benchmark_prices(&prices.index);
    UniversePrices { prices, coverage, benchmarks: Some(benchmarks), metadata }
}

// ----------------------------------------------------------------------------- fake sources

/// A latch that holds a fake download until the test releases it.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn wait(&self) {
        let mut open = lock(&self.open);
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
    }

    fn release(&self) {
        *lock(&self.open) = true;
        self.changed.notify_all();
    }
}

#[derive(Clone)]
enum Profiles {
    Named,
    Empty,
    Stalled(Arc<Gate>),
}

/// The test data sources. An unset source fails as an unmocked download.
struct Fake {
    calls: Mutex<Vec<String>>,
    qqq: Result<QqqReference>,
    qqq_panics: bool,
    universe: Option<UniversePrices>,
    universe_gate: Option<Arc<Gate>>,
    /// Download the universe in four batches; after batch 1, wait for this gate.
    batch_gate: Option<Arc<Gate>>,
    universe_panics: bool,
    /// Universe downloads running now, and the most at once.
    running: AtomicUsize,
    most_running: AtomicUsize,
    prices: Option<Result<Frame>>,
    benchmarks: Option<Frame>,
    profiles: Profiles,
}

impl Default for Fake {
    fn default() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            qqq: Ok(QqqReference { volatility: 0.20, last_return_date: day("2026-10-16") }),
            qqq_panics: false,
            universe: None,
            universe_gate: None,
            batch_gate: None,
            universe_panics: false,
            running: AtomicUsize::new(0),
            most_running: AtomicUsize::new(0),
            prices: None,
            benchmarks: None,
            profiles: Profiles::Named,
        }
    }
}

impl Fake {
    fn universe() -> Self {
        Self { universe: Some(universe_fixture()), ..Self::default() }
    }

    fn log(&self, call: String) {
        lock(&self.calls).push(call);
    }

    fn calls(&self) -> Vec<String> {
        lock(&self.calls).clone()
    }

    /// Every call except the QQQ reference.
    fn downloads(&self) -> Vec<String> {
        self.calls().into_iter().filter(|call| call != "qqq" && !call.starts_with("profile ")).collect()
    }

    fn profile_requests(&self) -> Vec<String> {
        self.calls().into_iter().filter_map(|call| call.strip_prefix("profile ").map(str::to_string)).collect()
    }
}

fn blocked() -> Error {
    Error::Runtime("An app test attempted an unmocked market download.".into())
}

impl Fake {
    /// The universe download, batch by batch when `batch_gate` is set. A cancelled
    /// build stops before its next batch, as the live download does.
    fn download_universe(&self, progress: Progress, cancelled: Cancelled) -> Result<UniversePrices> {
        if let Some(gate) = &self.universe_gate {
            progress(3, 13, "Checked 3 of 13 securities.");
            gate.wait();
        }
        if let Some(gate) = &self.batch_gate {
            for batch in 1..=4 {
                if cancelled() {
                    self.log(format!("stopped before batch {batch}"));
                    return Err(cancelled_error());
                }
                self.log(format!("batch {batch}"));
                progress(25 * batch, 100, &format!("Checked {} of 100 securities.", 25 * batch));
                if batch == 1 {
                    gate.wait();
                }
            }
        }
        if self.universe_panics {
            todo!("simulated universe download failure")
        }
        let loaded = self.universe.clone().ok_or_else(blocked)?;
        progress(13, 13, "All fixture securities checked");
        Ok(loaded)
    }
}

impl Sources for Fake {
    fn universe_prices(
        &self,
        market: Market,
        start: Date,
        end: Date,
        progress: Progress,
        cancelled: Cancelled,
    ) -> Result<UniversePrices> {
        self.log(format!("universe {} {} {}", market.label(), iso(start), iso(end)));
        let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_running.fetch_max(running, Ordering::SeqCst);
        let result = self.download_universe(progress, cancelled);
        self.running.fetch_sub(1, Ordering::SeqCst);
        result
    }

    fn prices(&self, tickers: &[String], start: Date, end: Date) -> Result<Frame> {
        self.log(format!("prices {} {} {}", tickers.join(","), iso(start), iso(end)));
        self.prices.clone().unwrap_or_else(|| Err(blocked()))
    }

    fn benchmarks(&self, dates: &[Date]) -> Result<Frame> {
        self.log(format!("benchmarks {}", dates.len()));
        self.benchmarks.clone().ok_or_else(blocked)
    }

    fn original_tickers(&self) -> Result<Vec<String>> {
        self.log("original tickers".into());
        efficient_frontier::data::original_tickers(None, true)
    }

    fn company_profile(&self, symbol: &str) -> CompanyProfile {
        self.log(format!("profile {symbol}"));
        match &self.profiles {
            Profiles::Named => CompanyProfile {
                name: format!("Company {symbol}"),
                description: format!("Provides products and services for {symbol}."),
                source_url: format!("https://example.com/{symbol}"),
            },
            Profiles::Empty => CompanyProfile::default(),
            Profiles::Stalled(gate) => {
                gate.wait();
                CompanyProfile::default()
            }
        }
    }

    fn qqq_reference(&self) -> Result<QqqReference> {
        self.log("qqq".into());
        if self.qqq_panics {
            todo!("simulated QQQ reference failure")
        }
        self.qqq.clone()
    }
}

// ----------------------------------------------------------------------------- test client

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: String,
}

/// One browser: the app's router and a cookie jar of one session.
struct TestApp {
    router: Router,
    state: Arc<AppState>,
    fake: Arc<Fake>,
    cookie: Option<String>,
}

fn encode(text: &str) -> String {
    text.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (byte as char).to_string(),
            b' ' => "+".into(),
            other => format!("%{other:02X}"),
        })
        .collect()
}

const BOUNDARY: &str = "plTestBoundary7MA4YWxkTrZu0gW";

/// A file part: field name, file name and bytes.
type File<'a> = (&'a str, &'a str, Vec<u8>);

fn multipart(fields: &[(&str, &str)], files: &[File]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend(
            format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").bytes(),
        );
    }
    for (name, file_name, bytes) in files {
        body.extend(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{file_name}\"\r\nContent-Type: text/csv\r\n\r\n"
            )
            .bytes(),
        );
        body.extend(bytes);
        body.extend(b"\r\n");
    }
    body.extend(format!("--{BOUNDARY}--\r\n").bytes());
    body
}

impl TestApp {
    fn new(fake: Fake) -> Self {
        Self::with(fake, |_| {})
    }

    fn with(fake: Fake, configure: impl FnOnce(&mut AppState)) -> Self {
        let fake = Arc::new(fake);
        let mut state = AppState::new(fake.clone());
        state.clock = today;
        configure(&mut state);
        let state = Arc::new(state);
        Self { router: crate::routes::router(state.clone()), state, fake, cookie: None }
    }

    async fn send(&mut self, method: Method, path: &str, content_type: Option<String>, body: Vec<u8>) -> Reply {
        self.send_with(method, path, &[], content_type, body).await
    }

    /// A request with extra headers, such as the browser's `Origin` and `Sec-Fetch-*`.
    async fn send_with(
        &mut self,
        method: Method,
        path: &str,
        headers: &[(&str, &str)],
        content_type: Option<String>,
        body: Vec<u8>,
    ) -> Reply {
        let mut request = Request::builder().method(method).uri(path);
        if let Some(cookie) = &self.cookie {
            request = request.header(header::COOKIE, cookie);
        }
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        if let Some(content_type) = content_type {
            request = request.header(header::CONTENT_TYPE, content_type);
        }
        let response = self.router.clone().oneshot(request.body(Body::from(body)).unwrap()).await.unwrap();
        if let Some(value) = response.headers().get(header::SET_COOKIE) {
            self.cookie = value.to_str().unwrap().split(';').next().map(str::to_string);
        }
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        Reply { status, headers, body: String::from_utf8_lossy(&bytes).into_owned() }
    }

    async fn get(&mut self, path: &str) -> Reply {
        self.send(Method::GET, path, None, Vec::new()).await
    }

    /// The current page, which must render.
    async fn page(&mut self) -> String {
        let reply = self.get("/").await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        reply.body
    }

    async fn drawer(&mut self) -> String {
        let reply = self.get("/drawer").await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        reply.body
    }

    async fn form(&mut self, path: &str, fields: &[(&str, &str)]) -> Reply {
        let body: Vec<String> =
            fields.iter().map(|(name, value)| format!("{}={}", encode(name), encode(value))).collect();
        self.send(Method::POST, path, Some("application/x-www-form-urlencoded".into()), body.join("&").into_bytes())
            .await
    }

    async fn multipart(&mut self, path: &str, fields: &[(&str, &str)], files: &[File<'_>]) -> Reply {
        let content_type = format!("multipart/form-data; boundary={BOUNDARY}");
        self.send(Method::POST, path, Some(content_type), multipart(fields, files)).await
    }

    /// Whether the drawer's draft differs from the applied result.
    async fn differs(&mut self, fields: &[(&str, &str)], files: &[File<'_>]) -> bool {
        let reply = self.multipart("/api/draft", &[&[("form", "drawer")], fields].concat(), files).await;
        let value: serde_json::Value = serde_json::from_str(&reply.body).unwrap();
        value["differs"].as_bool().unwrap()
    }

    /// Build portfolio from the drawer and wait for the result.
    async fn build(&mut self, fields: &[(&str, &str)], files: &[File<'_>]) {
        let reply = self.multipart("/build", &[&[("form", "drawer")], fields].concat(), files).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
        self.wait().await;
    }

    /// Build portfolio from the landing page.
    async fn build_landing(&mut self, target: &str, cap: &str) {
        let reply = self.multipart("/build", &[("form", "landing"), ("target_vol", target), ("cap", cap)], &[]).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        self.wait().await;
    }

    /// Follow `/api/job` until the build ends.
    async fn wait(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let reply = self.get("/api/job").await;
            let value: serde_json::Value = serde_json::from_str(&reply.body).unwrap();
            if value["active"] == false {
                return;
            }
            assert!(Instant::now() < deadline, "the build did not finish");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn session_ref(&self) -> crate::session::SessionRef {
        let cookie = self.cookie.as_deref().expect("a session cookie");
        let id = cookie.split_once('=').unwrap().1;
        self.state.sessions.get(id).expect("a stored session")
    }

    fn session(&self) -> Session {
        lock(&self.session_ref()).clone()
    }

    fn applied(&self) -> Arc<Applied> {
        self.session().applied.expect("an applied result")
    }
}

/// `demo_app()`: choose Demo in the drawer (its change sets the 20% cap) and build.
async fn demo_app() -> TestApp {
    let mut app = TestApp::new(Fake::default());
    app.page().await;
    app.build(&[("market", "Demo"), ("cap", "20")], &[]).await;
    app
}

/// `csv_app()`: upload prices (and benchmarks) and build.
async fn csv_app(prices: &Frame, benchmarks: Option<&Frame>) -> TestApp {
    let mut app = TestApp::new(Fake::default());
    app.page().await;
    let mut files = vec![("prices_upload", "portfolio.csv", csv(prices))];
    if let Some(benchmarks) = benchmarks {
        files.push(("benchmark_upload", "benchmarks.csv", csv(benchmarks)));
    }
    app.build(&[("market", "Upload CSV")], &files).await;
    app
}

// ----------------------------------------------------------------------------- HTML helpers

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#x27;", "'").replace("&amp;", "&")
}

/// The element with this id, from its start tag to the matching end tag.
fn element<'a>(html: &'a str, id: &str) -> &'a str {
    let at = html.find(&format!(r#"id="{id}""#)).unwrap_or_else(|| panic!("no element {id}"));
    let start = html[..at].rfind('<').unwrap();
    let tag: String = html[start + 1..].chars().take_while(char::is_ascii_alphanumeric).collect();
    let (open, close) = (format!("<{tag}"), format!("</{tag}>"));
    let (mut depth, mut position) = (0, start);
    loop {
        let next_close = html[position..].find(&close).map(|i| i + position).expect("a closing tag");
        match html[position..].find(&open).map(|i| i + position) {
            Some(next_open) if next_open < next_close => {
                depth += 1;
                position = next_open + open.len();
            }
            _ => {
                depth -= 1;
                position = next_close + close.len();
                if depth == 0 {
                    return &html[start..position];
                }
            }
        }
    }
}

fn has_element(html: &str, id: &str) -> bool {
    html.contains(&format!(r#"id="{id}""#))
}

/// The TeX source of every display equation, in page order.
fn equations(html: &str) -> Vec<String> {
    html.split(r#"<div class="pl-latex">"#).skip(1).map(|part| unescape(part.split("</div>").next().unwrap())).collect()
}

/// The `value` of the input with this id.
fn value(html: &str, id: &str) -> String {
    let at = html.find(&format!(r#"id="{id}""#)).unwrap_or_else(|| panic!("no input {id}"));
    let tag_end = at + html[at..].find('>').unwrap();
    let tag_start = html[..at].rfind('<').unwrap();
    let tag = &html[tag_start..tag_end];
    let value = tag.split(r#"value=""#).nth(1).expect("a value attribute");
    unescape(value.split('"').next().unwrap())
}

/// The checked option of a radio group.
fn checked(html: &str, name: &str) -> String {
    let marker = format!(r#"name="{name}" value=""#);
    html.split(&marker)
        .skip(1)
        .find(|rest| rest.split('>').next().unwrap().ends_with(" checked"))
        .map(|rest| unescape(rest.split('"').next().unwrap()))
        .unwrap_or_else(|| panic!("no checked {name}"))
}

fn differs_shown(drawer: &str) -> bool {
    !element(drawer, "pl-differs")
        .starts_with(r#"<div class="pl-alert pl-alert-info" role="status" id="pl-differs" hidden"#)
}

// ----------------------------------------------------------------------------- landing

#[tokio::test(flavor = "multi_thread")]
async fn combined_default_and_risk_inputs_do_no_work_before_submit() {
    let mut app = TestApp::new(Fake::universe());
    let page = app.page().await;
    assert!(page.contains(r#"data-page="landing""#));
    assert!(element(&page, "build").contains("Build portfolio"));
    assert!(element(&page, "edit_setup").contains("Settings"));
    assert!(!page.contains("try_demo"));
    assert_eq!(equations(&page).len(), 3);
    assert_eq!(page.matches("Efficient Frontier Analysis for US Equities").count(), 2, "page title and heading");
    assert!(page.contains("Constrained Mean–Variance Optimization"));
    assert!(!page.contains(r#"class="pl-wordmark""#));
    for definition in [
        "sample mean returns",
        "annual covariance matrix",
        "initial weight vector",
        "fraction in stock",
        "optimal weights",
        "vector of ones",
        "transpose",
        "starting cash",
        "entry observation",
        "fee per unit bought",
        "sum across stocks",
    ] {
        assert!(page.contains(definition), "{definition}");
    }
    assert!(page.contains(r"$\boldsymbol{A=252}:$ observations per year."));
    assert!(!page.contains("Choose a risk budget"));
    assert_eq!(value(&page, "preview_target_vol"), "20.00");
    assert_eq!(value(&page, "preview_cap"), "10");
    let session = app.session();
    assert_eq!(session.setup.market, Market::Combined);
    assert!(session.applied.is_none() && session.job.is_none());
    assert!(!page.contains(r#"id="export""#));
    app.form("/api/preview", &[("target_vol", "18"), ("cap", "12")]).await;
    let drawer = app.drawer().await;
    assert_eq!(checked(&drawer, "market"), "S&P 500 + Nasdaq-100");
    let options: Vec<String> = drawer
        .split(r#"name="market" value=""#)
        .skip(1)
        .map(|rest| unescape(rest.split('"').next().unwrap()))
        .collect();
    assert_eq!(
        options,
        ["S&P 500 + Nasdaq-100", "Nasdaq-100", "My tickers", "Original 60 holdings", "Upload CSV", "Demo"]
    );
    assert_eq!((value(&drawer, "target_vol"), value(&drawer, "cap")), ("18.00".to_string(), "12".to_string()));
    assert!(!app.differs(&[("history", "10 years")], &[]).await, "no result yet");
    assert_eq!(app.fake.calls(), ["qqq"], "opening the app or editing a draft must not download or fit");
    assert_eq!(app.state.caches.selection.computed(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn prebuild_target_and_cap_reach_the_real_selection() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.form("/api/preview", &[("target_vol", "18"), ("cap", "12")]).await;
    app.build_landing("18", "12").await;
    let session = app.session();
    assert!(session.build_error.is_none(), "{:?}", session.build_error);
    let applied = session.applied.unwrap();
    assert_eq!(applied.metadata.f64("target_volatility"), 0.18);
    assert_eq!(applied.selection.settings.f64("max_weight"), 0.12);
    assert_eq!(applied.selection.settings.f64("target_volatility"), 0.18);
    assert_eq!(app.fake.downloads(), ["universe S&P 500 + Nasdaq-100 2021-10-30 2026-10-30"]);
    assert_eq!(*applied.prices, universe_fixture().prices);
    assert_eq!(app.state.caches.selection.computed(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn qqq_default_is_dated_and_does_not_overwrite_user_edits() {
    let fake = Fake {
        qqq: Ok(QqqReference { volatility: 0.198350420824, last_return_date: day("2026-10-16") }),
        ..Fake::universe()
    };
    let mut app = TestApp::new(fake);
    let page = app.page().await;
    assert_eq!(value(&page, "preview_target_vol"), "19.84");
    let controls = element(&page, "landing_controls");
    assert!(controls.contains("Nasdaq-100 proxy · QQQ 19.84%") && controls.contains("One year through 16 Oct 2026."));
    app.form("/api/preview", &[("target_vol", "17.25"), ("cap", "10")]).await;
    assert_eq!(value(&app.drawer().await, "target_vol"), "17.25");
    // Cancel only closes the drawer in the browser; the landing keeps the edit.
    assert_eq!(value(&app.page().await, "preview_target_vol"), "17.25");
    assert_eq!(app.fake.calls(), ["qqq"]);
    app.build_landing("19.84", "10").await;
    let applied = app.applied();
    assert!((applied.selection.settings.f64("target_volatility") - 0.1984).abs() < 1e-12);
    let page = app.page().await;
    assert!(equations(&page).iter().any(|tex| tex.contains(r"b=0.1984\;(19.84\%)")), "{:?}", equations(&page));
}

#[tokio::test(flavor = "multi_thread")]
async fn unavailable_qqq_reference_keeps_manual_controls_and_math() {
    let mut app = TestApp::new(Fake { qqq: Err(io_error!("Network is unreachable")), ..Fake::default() });
    let page = app.page().await;
    assert_eq!(equations(&page).len(), 3);
    assert!(page.contains("manual fallback, not a current Nasdaq-100 estimate"));
    assert_eq!(value(&page, "preview_target_vol"), "20.00");
    app.form("/api/preview", &[("target_vol", "23"), ("cap", "10")]).await;
    assert_eq!(app.session().setup.target_vol, 23.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_panicking_qqq_reference_falls_back_to_the_manual_budget() {
    let mut app = TestApp::new(Fake { qqq_panics: true, ..Fake::default() });
    let page = app.page().await;
    assert!(page.contains("manual fallback, not a current Nasdaq-100 estimate"));
    assert!(app.session().risk_reference_unavailable);
}

#[tokio::test(flavor = "multi_thread")]
async fn out_of_range_limits_keep_the_stored_values() {
    let mut app = TestApp::new(Fake::default());
    app.page().await;
    app.form("/api/preview", &[("target_vol", "0.5"), ("cap", "150")]).await;
    let setup = app.session().setup;
    assert_eq!((setup.target_vol, setup.cap), (20.0, 10.0));
}

// ----------------------------------------------------------------------------- builds and results

#[tokio::test(flavor = "multi_thread")]
async fn demo_build_is_offline_with_an_explicit_feasible_cap() {
    let app = demo_app().await;
    let session = app.session();
    assert!(session.build_error.is_none(), "{:?}", session.build_error);
    assert_eq!(app.fake.downloads(), Vec::<String>::new());
    assert!(app.fake.profile_requests().is_empty(), "synthetic studies have no company profiles");
    assert_eq!((session.setup.market, session.setup.cap), (Market::Demo, 20.0));
    let study = &session.applied.as_ref().unwrap().selection;
    assert_eq!(study.settings.f64("max_weight"), 0.2);
    assert_eq!(study.settings.f64("target_volatility"), 0.2);
    assert!(study.benchmark_prices.is_none() && !study.supported);
    let weights = &study.latest[&study.selected].weights;
    assert!(weights.max() <= 0.2 + 1e-6);
    assert!((weights.sum() - 1.0).abs() < 1e-9);
    assert_eq!(study.settings.str("holding_policy"), "buy_and_hold");
    assert!(!study.settings.contains_key("rebalance_every"));
    for name in study.test.allocations.keys() {
        assert_eq!(study.test.allocations[name].nrows(), 1);
        assert_eq!(study.test.trades[name].nrows(), 1);
    }
    let initial = study.test.allocations[&study.selected].row(0);
    let end = study.test.holdings[&study.selected].f64s("end_weight");
    assert!(initial.iter().zip(&end).any(|(a, b)| (a - b).abs() > 1e-9), "weights drift after the purchase");
    let mut app = app;
    let page = app.page().await;
    assert!(element(&page, "tile_holdings").contains("Demo · synthetic prices. These are not market results."));
}

#[tokio::test(flavor = "multi_thread")]
async fn demo_holdings_name_simulated_assets() {
    let mut app = demo_app().await;
    let page = app.page().await;
    let holdings = element(&page, "tile_holdings");
    assert!(holdings.contains("Simulated ") && holdings.contains("not an actual company"));
}

#[tokio::test(flavor = "multi_thread")]
async fn overview_orders_its_tiles_and_keeps_details_closed() {
    let mut app = demo_app().await;
    let page = app.page().await;
    assert!(page.contains(r#"data-page="overview""#));
    let positions: Vec<usize> =
        ["tile_holdings", "tile_fitted_frontier", "pl-market", "tile_out_of_sample", "tile_final_test"]
            .iter()
            .map(|id| page.find(&format!(r#"id="{id}""#)).unwrap())
            .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(!page.contains("<details open") && !page.contains(r#"<details class="pl-expander" open"#));
    assert!(page.contains("These weights use all supplied prices"));
    assert!(page.contains("They form the selected point on the frontier below"));
    assert!(page.contains("the replay below uses a separate initial allocation"));
    let nav = element(&page, "localnav");
    assert!(nav.contains(">Overview</button>") && nav.contains(">Data &amp; method</button>"));
    let id = app.applied().id;
    assert!(nav.contains(&format!(r#"href="/export?result={id}""#)) && nav.contains("portfolio-lab.zip"));
    for removed in ["risk_profile", "comparison_method", "key_low", "open_historical", "run_legacy"] {
        assert!(!page.contains(removed), "{removed}");
    }
    assert!(page.contains(r#"<span class="pl-keys">Keys: <kbd>1</kbd> overview"#));
    let study = app.applied().selection.clone();
    assert_eq!(study.latest.keys().collect::<Vec<_>>(), [SELECTED]);
}

#[tokio::test(flavor = "multi_thread")]
async fn overview_shows_one_test_and_separate_latest_weights() {
    let mut app = demo_app().await;
    let page = app.page().await;
    let order = [
        "Latest fitted holdings",
        "Latest fitted frontier",
        "In sample versus out of sample",
        "Historical replay · buy and hold",
        "What drove the result?",
        "Fit and replay use different holdings",
    ];
    let positions: Vec<usize> = order.iter().map(|text| page.find(text).unwrap_or_else(|| panic!("{text}"))).collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(!page.contains("annual volatility budget.</h1>") && !page.contains(r#"class="pl-context""#));
    assert!(page.contains("The latest fit includes the replay dates"));
    assert!(!page.contains("First historical fit"));
    assert!(page.contains("Historical volatility") && page.contains("ordinary sample covariance"));
    assert_eq!(page.matches(r#"class="plotly-graph-div""#).count(), 3);
    assert!(!page.to_lowercase().contains("validation") && !page.to_lowercase().contains("blend"));
    let expander = page.split("All weights, including negligible allocations").nth(1).unwrap();
    assert!(!expander.split("</details>").next().unwrap().contains("pl-holdings-grid"));
    let study = app.applied().selection.clone();
    let metrics = element(&page, "tile_final_test");
    for label in [
        "Net arithmetic mean / year",
        "Net compound growth / year (CAGR)",
        "Realized volatility / year",
        "Maximum drawdown",
    ] {
        assert!(metrics.contains(label), "{label}");
    }
    assert!(metrics.contains("Stock portfolio · buy and hold"));
    let annual_mean = study.test.metrics.f64_at(SELECTED, "annual_mean");
    assert!(metrics.contains(&efficient_frontier::pyfmt::format(annual_mean, ".2%")));
}

#[tokio::test(flavor = "multi_thread")]
async fn mathematical_definitions_use_rendered_latex_and_applied_limits() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("18", "12").await;
    let page = app.page().await;
    assert!(!page.contains("open_mathematics") && !page.contains("Mathematics ·"));
    let overview = equations(&page);
    let joined = overview.join("\n");
    // Adjacent raw strings must not turn spacing into unknown KaTeX commands.
    for command in [r"\qquad", r"\quad"] {
        for (position, _) in joined.match_indices(command) {
            let next = joined[position + command.len()..].chars().next();
            assert!(!next.is_some_and(|c| c.is_ascii_alphabetic()), "{command} runs into a letter");
        }
    }
    for expected in [
        r"\widehat\mu=A\bar r",
        r"\widehat\Sigma=\frac{A}{m-1}",
        r"b=0.18\;(18.00\%)",
        r"h=0.12\;(12\%)",
        r"V_e=\frac{V_{\mathrm{start}}}{1+c}",
        r"c=0.001\;(0.10\%)",
        r"V_t&=V_e\sum_i w_i g_{i,t}",
        r"a_{i,t}=\frac{w_i g_{i,t}}{\sum_j w_j g_{j,t}}",
        r"\frac{A}{N-1}",
        r"\mathrm{CAGR}",
        r"\mathrm{MDD}",
    ] {
        assert!(joined.contains(expected), "{expected}");
    }
    assert!(!page.split(r#"<div class="pl-latex">"#).next().unwrap().contains(r"\widehat\mu"));
    assert!(page.contains("Neither limit triggers a trade"));
    assert!(page.contains("no later refits, trades"));
    assert!(page.contains("including the entry-fee interval"));
    let reply = app.form("/view", &[("view", "Data & method")]).await;
    assert_eq!((reply.status, reply.headers[header::LOCATION].to_str().unwrap()), (StatusCode::SEE_OTHER, "/"));
    let method = app.page().await;
    assert!(method.contains(r#"data-page="method""#));
    let mut position = 0;
    for equation in equations(&method) {
        position =
            overview[position..].iter().position(|tex| *tex == equation).expect("an overview equation") + position + 1;
    }
    assert!(!has_element(&method, "math_out_of_sample") && !has_element(&method, "math_diagnosis"));
}

#[tokio::test(flavor = "multi_thread")]
async fn out_of_sample_section_precedes_replay_and_matches_walk_forward() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("20", "10").await;
    let page = app.page().await;
    let tile = element(&page, "tile_out_of_sample");
    assert!(page.find(r#"id="tile_out_of_sample""#).unwrap() < page.find("pl-chart-replay").unwrap());
    assert!(
        tile.contains("In sample versus out of sample") && tile.contains("each fit uses only earlier stock prices")
    );
    assert!(tile.contains("Out of sample, the stock portfolio averaged"));
    assert!(
        tile.contains("Window 3 uses the same fit as the replay") && tile.contains("measured from the purchase close")
    );
    assert!(tile.contains("Equal weight in sample") && tile.contains("Equal weight out of sample"));
    let math = element(tile, "math_out_of_sample");
    let equations = equations(math).join("\n");
    assert!(equations.contains(r"\mu^{\rm in}_k=\frac{A}{s_k-1}\sum_{t=2}^{s_k}"));
    assert!(equations.contains(r"\mu^{\rm out}_k=\frac{A}{e_k-s_k-1}\sum_{t=s_k+2}^{e_k}"));
    assert!(equations.contains(r"|C|=5") && equations.contains(r"\sum_{k\in C}"));
    assert!(!page.contains("This portfolio was dominated") && !page.contains("No reference dominates"));
}

#[tokio::test(flavor = "multi_thread")]
async fn short_history_explains_missing_walk_forward() {
    let prices = first_rows(&market_prices(), 20);
    let benchmarks = benchmark_prices(&prices.index);
    let mut app = csv_app(&prices, Some(&benchmarks)).await;
    assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
    let page = app.page().await;
    let tile = element(&page, "tile_out_of_sample");
    assert!(tile.contains("at least 3 fit returns and 3 test returns"));
    assert!(!tile.contains("plotly-graph-div"));
    assert!(app.applied().out_of_sample.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn short_history_export_says_why_out_of_sample_is_missing() {
    let prices = first_rows(&market_prices(), 20);
    let benchmarks = benchmark_prices(&prices.index);
    let mut app = csv_app(&prices, Some(&benchmarks)).await;
    let page = app.page().await;
    assert!(element(&page, "tile_final_test").contains("plotly-graph-div"));
    let reply = app.get(&format!("/export?result={}", app.applied().id)).await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.headers[header::CONTENT_TYPE], "application/zip");
    assert!(reply.body.contains("metadata.json") && !reply.body.contains("out_of_sample/windows.csv"));
}

#[tokio::test(flavor = "multi_thread")]
async fn short_daily_windows_are_described_in_less_than_a_month() {
    let prices = first_rows(&market_prices(), 64);
    let benchmarks = benchmark_prices(&prices.index);
    let mut app = csv_app(&prices, Some(&benchmarks)).await;
    let page = app.page().await;
    let tile = element(&page, "tile_out_of_sample");
    assert!(tile.contains("returns, less than one month") && !tile.contains("0 months"));
}

#[tokio::test(flavor = "multi_thread")]
async fn qqq_default_budget_is_named_as_later_information() {
    let app = TestApp::new(Fake::universe());
    let mut app = app;
    app.page().await;
    app.build_landing("20.00", "10").await;
    let session = app.session();
    let applied = session.applied.as_ref().unwrap();
    let windows = &applied.out_of_sample.as_ref().unwrap().windows;
    let note = budget_note(&session, &applied.selection, windows);
    assert!(
        note.starts_with(
            "The volatility budget equals QQQ's volatility over the year through 16 Oct 2026. That year ends after every walk-forward fit"
        ),
        "{note}"
    );
    assert!(!note.contains("overlaps"));
    let mut manual = TestApp::new(Fake::universe());
    manual.page().await;
    manual.build_landing("18", "10").await;
    let session = manual.session();
    let applied = session.applied.as_ref().unwrap();
    assert_eq!(budget_note(&session, &applied.selection, &applied.out_of_sample.as_ref().unwrap().windows), "");
}

#[tokio::test(flavor = "multi_thread")]
async fn qqq_default_budget_note_is_in_the_walk_forward_caption() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("20.00", "10").await;
    let page = app.page().await;
    assert!(element(&page, "math_out_of_sample").contains("The volatility budget equals QQQ&#x27;s volatility"));
}

#[tokio::test(flavor = "multi_thread")]
async fn visible_replay_diagnosis_reconciles_the_saved_initial_allocation() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("20", "10").await;
    let page = app.page().await;
    let replay = element(&page, "tile_final_test");
    let diagnosis = element(replay, "math_diagnosis");
    assert!(replay.find("pl-chart-replay").unwrap() < replay.find(r#"id="math_diagnosis""#).unwrap());
    let rendered = equations(diagnosis).join("\n");
    assert!(rendered.contains(r"\text{stock means}") && rendered.contains(r"\text{entry fee}"));
    let study = app.applied().selection.clone();
    let mean = study.test.metrics.f64_at(SELECTED, "annual_mean");
    assert!(rendered.contains(&efficient_frontier::pyfmt::format(mean, ".2%").replace('%', r"\%")));
    for name in study.test.metrics.index_labels().iter().filter(|name| *name != SELECTED) {
        assert!(diagnosis.contains(&efficient_frontier::pyfmt::escape(name)), "{name}");
    }
    assert!(diagnosis.contains("accounting explanation"));
}

#[tokio::test(flavor = "multi_thread")]
async fn combined_coverage_and_latest_holdings_preserve_every_input() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("20", "10").await;
    let session = app.session();
    assert!(session.build_error.is_none(), "{:?}", session.build_error);
    let applied = session.applied.clone().unwrap();
    let study = &applied.selection;
    let page = app.page().await;
    let holdings = element(&page, "tile_holdings");
    let mut requested = app.fake.profile_requests();
    requested.sort();
    let mut expected = crate::build::displayed_weights(study).labels;
    expected.extend(crate::build::replay_symbols(study));
    expected.sort();
    expected.dedup();
    assert_eq!(requested, expected);
    // Profiled symbols take the profile name; the others need `story::company_names`.
    for symbol in &expected {
        assert!(holdings.contains(&format!("<td>{symbol}</td><td>Company {symbol}</td>")), "{symbol}");
    }
    let replay = element(&page, "tile_replay_drivers");
    for symbol in crate::build::replay_symbols(study) {
        assert!(replay.contains(&format!("Provides products and services for {symbol}.")), "{symbol}");
    }
    assert_eq!(session.coverage.as_deref(), Some(&universe_fixture().coverage));
    let records = applied.metadata.get("universe_coverage").and_then(Json::as_list).unwrap();
    assert_eq!(records.last().and_then(Json::as_dict).unwrap().str("reason"), "Incomplete prices");
    assert!(page.contains("not a historical reconstruction"));
    assert!(holdings.contains("Available company descriptions are current Yahoo Finance profiles"));
}

#[tokio::test(flavor = "multi_thread")]
async fn combined_holdings_cards_show_company_profiles() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("20", "10").await;
    let page = app.page().await;
    let study = app.applied().selection.clone();
    let displayed = crate::build::displayed_weights(&study);
    let holdings = element(&page, "tile_holdings");
    for symbol in &app.applied().prices.columns {
        assert!(holdings.contains(&format!("<td>{symbol}</td><td>Company {symbol}</td>")), "{symbol}");
    }
    assert_eq!(holdings.matches(r#"<article class="pl-holding">"#).count(), displayed.len());
    for (symbol, weight) in displayed.iter() {
        assert!(holdings.contains(&format!(">Company {symbol}</a></h4>")));
        assert!(holdings.contains(&format!(r#"<span class="pl-holding-symbol">{symbol}</span>"#)));
        let amount = efficient_frontier::pyfmt::format(weight, ".2%");
        assert!(holdings.contains(&format!(r#"<strong class="pl-holding-weight">{amount}</strong>"#)));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn unavailable_company_profiles_keep_results_visible_without_source_claims() {
    for stalled in [false, true] {
        let gate = Arc::new(Gate::default());
        let profiles = if stalled { Profiles::Stalled(gate.clone()) } else { Profiles::Empty };
        let mut app = TestApp::with(Fake { profiles, ..Fake::universe() }, |state| {
            state.profile_timeout = Duration::from_millis(200);
        });
        app.page().await;
        let started = Instant::now();
        app.build_landing("20", "10").await;
        assert!(started.elapsed() < Duration::from_secs(60));
        assert!(app.session().build_error.is_none());
        let page = app.page().await;
        let holdings = element(&page, "tile_holdings");
        assert!(!holdings.contains("Yahoo Finance profiles") && !holdings.contains("open its source"));
        assert!(!holdings.contains("<a "));
        assert!(
            element(&page, "tile_replay_drivers").contains("Company description unavailable from the data provider.")
        );
        assert!(has_element(&page, "math_fit") && has_element(&page, "math_replay"));
        gate.release();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn partial_market_prices_keep_available_comparison_and_name_missing_etf() {
    let mut universe = universe_fixture();
    universe.benchmarks = universe.benchmarks.map(|frame| frame.select(&["SPY"]).unwrap());
    let mut app = TestApp::new(Fake { universe: Some(universe), ..Fake::default() });
    app.page().await;
    app.build_landing("20", "10").await;
    let applied = app.applied();
    let study = &applied.selection;
    assert_eq!(study.benchmark_prices.as_ref().unwrap().columns, ["SPY"]);
    assert!(!study.supported);
    assert_eq!(study.test.equity.columns, ["Historical-mean portfolio", "Equal-weight stocks", "S&P 500 (SPY)"]);
    assert!(study.reasons.iter().any(|reason| reason.contains("unavailable for QQQ")));
    assert_eq!(*applied.prices, universe_fixture().prices);
}

#[tokio::test(flavor = "multi_thread")]
async fn partial_market_prices_are_named_on_data_and_method() {
    let mut universe = universe_fixture();
    universe.benchmarks = universe.benchmarks.map(|frame| frame.select(&["SPY"]).unwrap());
    let mut app = TestApp::new(Fake { universe: Some(universe), ..Fake::default() });
    app.page().await;
    app.build_landing("20", "10").await;
    app.form("/view", &[("view", "Data & method")]).await;
    assert!(app.page().await.contains("unavailable for QQQ"));
}

#[tokio::test(flavor = "multi_thread")]
async fn repeated_identical_build_uses_cached_inputs_and_selection() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    app.build_landing("20", "10").await;
    let caches = &app.state.caches;
    assert_eq!((caches.selection.computed(), caches.frontier.computed(), caches.out_of_sample.computed()), (1, 1, 1));
    let equity = app.applied().selection.test.equity.clone();
    app.build(&[], &[]).await;
    assert!(app.session().build_error.is_none());
    let caches = &app.state.caches;
    assert_eq!((caches.selection.computed(), caches.frontier.computed(), caches.out_of_sample.computed()), (1, 1, 1));
    assert_eq!(app.fake.downloads().len(), 1);
    app.form("/view", &[("view", "Data & method")]).await;
    app.page().await;
    app.form("/view", &[("view", "Portfolio")]).await;
    app.page().await;
    assert_eq!(app.state.caches.frontier.computed(), 1);
    assert_eq!(app.applied().selection.test.equity, equity);
    assert_eq!(app.session().coverage.as_deref(), Some(&universe_fixture().coverage));
}

// ----------------------------------------------------------------------------- drawer drafts

#[tokio::test(flavor = "multi_thread")]
async fn draft_cancel_and_reopen_preserve_applied_risk_and_prices() {
    let mut app = demo_app().await;
    let original = app.applied();
    assert!(app.differs(&[("market", "Demo"), ("target_vol", "25"), ("cap", "30")], &[]).await);
    assert!(Arc::ptr_eq(&app.applied(), &original));
    assert_eq!(app.applied().metadata.f64("target_volatility"), 0.2);
    // Cancel closes the drawer in the browser without a request; reopening reads the applied setup.
    let drawer = app.drawer().await;
    assert_eq!((value(&drawer, "target_vol"), value(&drawer, "cap")), ("20.00".to_string(), "20".to_string()));
    assert!(!differs_shown(&drawer));
    let setup = app.session().setup;
    assert_eq!((setup.target_vol, setup.cap), (20.0, 20.0));
    assert!(!app.differs(&[("market", "Demo"), ("target_vol", "20"), ("cap", "20")], &[]).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn submit_applies_limits_and_fixed_assumptions() {
    let mut app = demo_app().await;
    app.build(&[("target_vol", "25"), ("cap", "30")], &[]).await;
    let applied = app.applied();
    let metadata = &applied.metadata;
    let values: Vec<f64> = ["target_volatility", "max_weight", "risk_free_rate", "train_fraction", "shrinkage"]
        .iter()
        .map(|key| metadata.f64(key))
        .collect();
    assert_eq!(values, [0.25, 0.3, 0.02, 0.7, 0.0]);
    let settings = &applied.selection.settings;
    assert_eq!(settings.str("holding_policy"), "buy_and_hold");
    assert!(!settings.contains_key("rebalance_every"));
    assert_eq!(settings.f64("cost_bps"), 10.0);
    let drawer = app.drawer().await;
    assert_eq!((value(&drawer, "target_vol"), value(&drawer, "cap")), ("25.00".to_string(), "30".to_string()));
    assert!(!differs_shown(&drawer));
}

#[tokio::test(flavor = "multi_thread")]
async fn custom_tickers_and_dates_download_only_on_submit() {
    let prices = market_prices();
    let fake =
        Fake { prices: Some(Ok(prices.clone())), benchmarks: Some(benchmark_prices(&prices.index)), ..Fake::default() };
    let mut app = TestApp::new(fake);
    app.page().await;
    let fields = [
        ("market", "My tickers"),
        ("history", "Custom dates"),
        ("ticker_text", "aapl, MSFT brk.b aapl"),
        ("start", "2020-01-01"),
        ("end", "2024-01-01"),
    ];
    assert!(!app.differs(&fields, &[]).await);
    assert!(app.fake.downloads().is_empty());
    app.build(&fields, &[]).await;
    assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
    assert_eq!(app.fake.downloads()[0], "prices AAPL,MSFT,BRK-B 2020-01-01 2024-01-01");
    assert_eq!(app.applied().metadata.str("benchmark_source"), "Yahoo Finance");
    let drawer = app.drawer().await;
    assert_eq!(checked(&drawer, "market"), "My tickers");
    assert_eq!(checked(&drawer, "history"), "Custom dates");
    assert!(app.differs(&[("market", "Original 60 holdings")], &[]).await);
    assert_eq!(app.fake.downloads().len(), 2, "a draft never downloads");
}

#[tokio::test(flavor = "multi_thread")]
async fn original_60_holdings_use_current_symbols() {
    let prices = market_prices();
    let fake =
        Fake { prices: Some(Ok(prices.clone())), benchmarks: Some(benchmark_prices(&prices.index)), ..Fake::default() };
    let mut app = TestApp::new(fake);
    app.page().await;
    app.build(&[("market", "Original 60 holdings")], &[]).await;
    let request = app.fake.downloads().into_iter().find(|call| call.starts_with("prices ")).unwrap();
    let symbols: Vec<&str> = request.split(' ').nth(1).unwrap().split(',').collect();
    assert_eq!(symbols.len(), 60);
    assert!(symbols.contains(&"MRSH") && !symbols.contains(&"MMC"));
}

#[tokio::test(flavor = "multi_thread")]
async fn csv_file_bytes_survive_reopen_and_cancel_then_explicit_removal() {
    let prices = market_prices();
    let benchmarks = benchmark_prices(&prices.index);
    let mut app = csv_app(&prices, Some(&benchmarks)).await;
    assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
    let setup = app.session().setup;
    let original = app.applied();
    let drawer = app.drawer().await;
    assert!(drawer.contains("Using portfolio.csv. Upload a file to replace it."));
    assert!(drawer.contains("Using benchmarks.csv. Upload a file to replace it."));
    assert!(app.differs(&[("market", "Upload CSV"), ("clear_prices_file", "1")], &[]).await);
    // Cancel: nothing changed on the server.
    assert!(Arc::ptr_eq(&app.applied(), &original));
    assert_eq!(app.session().setup.prices_file, setup.prices_file);
    assert_eq!(app.session().setup.benchmark_file, setup.benchmark_file);
    app.build(&[("market", "Upload CSV"), ("clear_benchmark_file", "1")], &[]).await;
    let session = app.session();
    assert!(session.build_error.is_none(), "{:?}", session.build_error);
    assert_eq!(session.setup.prices_file, setup.prices_file);
    assert_eq!(session.setup.benchmark_file, None);
    assert!(session.applied.unwrap().selection.benchmark_prices.is_none());
    assert!(app.fake.downloads().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn same_filename_with_new_bytes_marks_pending_without_changing_inputs() {
    let prices = market_prices();
    let mut app = csv_app(&prices, Some(&benchmark_prices(&prices.index))).await;
    let original = app.applied();
    let applied_inputs = original.inputs.clone();
    let mut changed = prices.clone();
    let last = changed.nrows() - 1;
    changed.values[(last, 0)] *= 1.05;
    let files = [("prices_upload", "portfolio.csv", csv(&changed))];
    assert!(app.differs(&[("market", "Upload CSV")], &files).await);
    assert!(Arc::ptr_eq(&app.applied(), &original) && app.applied().inputs == applied_inputs);
    assert_eq!(*app.applied().prices, prices);
    app.build(&[("market", "Upload CSV")], &files).await;
    assert!(app.session().build_error.is_none());
    assert_eq!(*app.applied().prices, changed);
    assert_ne!(app.applied().inputs.prices, applied_inputs.prices);
    assert!(app.fake.downloads().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn csv_download_requires_explicit_submit_and_uploaded_benchmarks_win() {
    let prices = market_prices();
    let benchmarks = benchmark_prices(&prices.index);
    let mut app = TestApp::new(Fake { benchmarks: Some(benchmarks.clone()), ..Fake::default() });
    app.page().await;
    let files = [("prices_upload", "portfolio.csv", csv(&prices))];
    app.build(&[("market", "Upload CSV")], &files).await;
    assert!(app.fake.downloads().is_empty() && app.applied().selection.benchmark_prices.is_none());
    assert!(app.differs(&[("market", "Upload CSV"), ("download_benchmarks", "true")], &[]).await);
    assert!(app.fake.downloads().is_empty() && !app.applied().inputs.download_benchmarks);
    app.build(&[("market", "Upload CSV"), ("download_benchmarks", "true")], &[]).await;
    assert_eq!(app.fake.downloads(), ["benchmarks 240"]);
    assert_eq!(app.applied().selection.benchmark_prices.as_ref().unwrap(), &benchmarks);
    let doubled = Frame::new(benchmarks.index.clone(), benchmarks.columns.clone(), &benchmarks.values * 2.0);
    app.build(&[("market", "Upload CSV")], &[("benchmark_upload", "benchmarks.csv", csv(&doubled))]).await;
    assert_eq!(app.fake.downloads().len(), 1);
    assert_eq!(app.applied().metadata.str("benchmark_source"), "CSV: benchmarks.csv");
    assert_eq!(app.applied().selection.benchmark_prices.as_ref().unwrap(), &doubled);
}

#[tokio::test(flavor = "multi_thread")]
async fn csv_comparison_columns_never_become_stock_holdings() {
    let prices = market_prices();
    let benchmarks = benchmark_prices(&prices.index);
    for symbols in [vec!["SPY"], vec!["QQQ"], vec!["SPY", "QQQ"]] {
        let comparison = benchmarks.select(&symbols).unwrap();
        let mut app = csv_app(&join(&prices, &comparison), None).await;
        assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
        let applied = app.applied();
        assert_eq!(*applied.prices, prices);
        assert_eq!(applied.selection.benchmark_prices.as_ref().unwrap(), &comparison);
        assert_eq!(applied.selection.latest[SELECTED].weights.labels, prices.columns);
        let excluded: Vec<&str> = applied
            .metadata
            .get("excluded_candidate_symbols")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .filter_map(Json::as_str)
            .collect();
        assert_eq!(excluded, symbols);
        assert!(element(&app.page().await, "tile_fitted_frontier").contains("excluded from candidate holdings"));
        app.build(&[("market", "Upload CSV"), ("download_benchmarks", "true")], &[]).await;
        assert!(app.session().build_error.is_none() && app.fake.downloads().is_empty());
        assert_eq!(app.applied().metadata.str("benchmark_source"), format!("Input columns: {}", symbols.join(", ")));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn csv_comparison_controls_do_not_put_etfs_back_in_candidate_universe() {
    let prices = market_prices();
    let benchmarks = benchmark_prices(&prices.index);
    for comparison in ["disabled", "other_currency", "override"] {
        let mut app = csv_app(&join(&prices, &benchmarks), None).await;
        let override_prices =
            Frame::new(benchmarks.index.clone(), vec!["QQQ".into()], benchmarks.select(&["QQQ"]).unwrap().values * 2.0);
        match comparison {
            "disabled" => app.build(&[("market", "Upload CSV"), ("compare_market", "false")], &[]).await,
            "other_currency" => app.build(&[("market", "Upload CSV"), ("price_currency", "Other currency")], &[]).await,
            _ => {
                app.build(&[("market", "Upload CSV")], &[("benchmark_upload", "override.csv", csv(&override_prices))])
                    .await
            }
        }
        assert!(app.session().build_error.is_none() && app.fake.downloads().is_empty(), "{comparison}");
        let applied = app.applied();
        assert_eq!(*applied.prices, prices);
        if comparison == "override" {
            assert_eq!(applied.selection.benchmark_prices.as_ref().unwrap(), &override_prices);
            assert_eq!(applied.metadata.str("benchmark_source"), "CSV: override.csv");
        } else {
            assert!(applied.selection.benchmark_prices.is_none(), "{comparison}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn csv_with_only_comparison_etfs_fails_actionably() {
    let prices = market_prices();
    let mut app = csv_app(&benchmark_prices(&prices.index), None).await;
    let session = app.session();
    assert!(session.applied.is_none());
    let error = session.build_error.unwrap();
    assert!(error.message.to_lowercase().contains("stock") && error.message.contains("SPY"));
    let page = app.page().await;
    assert!(page.contains(r#"data-page="error""#));
    assert!(page.contains("The portfolio needs stock prices."));
    assert!(element(&page, "fix_setup").contains("Change the study"));
    assert!(app.fake.downloads().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn single_stock_after_comparison_split_requires_explicit_full_holding_cap() {
    let prices = market_prices();
    let stock = prices.select(&["STOCK_00"]).unwrap();
    let combined = join(&stock, &benchmark_prices(&prices.index).select(&["SPY"]).unwrap());
    let mut app = csv_app(&combined, None).await;
    assert!(app.session().build_error.is_some());
    app.build(&[("market", "Upload CSV"), ("cap", "100")], &[]).await;
    assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
    let weights = &app.applied().selection.latest[SELECTED].weights;
    assert_eq!(weights.labels, ["STOCK_00"]);
    assert!((weights.values[0] - 1.0).abs() < 1e-9);
}

#[tokio::test(flavor = "multi_thread")]
async fn unavailable_benchmarks_clear_previous_comparison_without_changing_assets() {
    let prices = market_prices();
    let benchmarks = benchmark_prices(&prices.index);
    for change in ["disable", "currency", "missing prices"] {
        let mut app = csv_app(&prices, Some(&benchmarks)).await;
        assert!(app.applied().selection.benchmark_prices.is_some());
        match change {
            "disable" => app.build(&[("market", "Upload CSV"), ("compare_market", "false")], &[]).await,
            "currency" => app.build(&[("market", "Upload CSV"), ("price_currency", "Other currency")], &[]).await,
            _ => {
                let incomplete = benchmarks.rows(1, benchmarks.nrows());
                app.build(&[("market", "Upload CSV")], &[("benchmark_upload", "incomplete.csv", csv(&incomplete))])
                    .await;
            }
        }
        let applied = app.applied();
        assert!(app.session().build_error.is_none(), "{change}");
        assert!(applied.selection.benchmark_prices.is_none() && !applied.selection.supported);
        assert!(applied.benchmark_prices.is_none());
        assert!(!applied.metadata.str("benchmarks_note").is_empty());
        assert_eq!(*applied.prices, prices);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn short_uploaded_history_shows_actual_fit_replay_and_frontier_windows() {
    let prices = first_rows(&market_prices(), 48);
    let mut app = TestApp::new(Fake::default());
    app.page().await;
    let files = [("prices_upload", "short-history.csv", csv(&prices))];
    app.build(&[("market", "Upload CSV"), ("frequency", "Custom"), ("periods_per_year", "48.5")], &files).await;
    assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
    let settings = app.applied().selection.settings.clone();
    assert_eq!(settings.f64("periods_per_year"), 48.5);
    let split = ((prices.nrows() - 1) as f64 * 0.7).floor() as usize;
    let shown = |i: usize| crate::render::format::day(prices.index[i]);
    let page = app.page().await;
    let summary = element(&page, "replay_dates");
    let windows: Vec<&str> = summary.split("<article>").skip(1).collect();
    assert_eq!(windows.len(), 3);
    assert!(windows[0].contains("Initial fit · first 70%"));
    assert!(windows[0].contains(&format!("{} to {}", shown(1), shown(split))));
    assert!(windows[0].contains(&format!("{split} returns choose the replay holdings")));
    assert!(windows[1].contains("Replay · final 30%"));
    assert!(windows[1].contains(&format!("{} to {}", shown(split + 1), shown(prices.nrows() - 1))));
    assert!(
        windows[1].contains(&format!("{} later returns form the replay chart and table", prices.nrows() - 1 - split))
    );
    assert!(windows[2].contains("Latest frontier · full history"));
    assert!(windows[2].contains(&format!("{} to {}", shown(1), shown(prices.nrows() - 1))));
    assert!(windows[2].contains("including the replay dates"));
    assert!(!summary.contains(&shown(0)));
    let replay = element(&page, "tile_final_test");
    assert!(replay.contains("split uses observations, not a fixed one-year test"));
    assert!(replay.contains("Annualized return and volatility describe units, not the chart’s duration"));
    let frontier = element(&page, "tile_fitted_frontier");
    assert!(frontier.contains(&format!(
        "<strong>Full-history fit · {} to {}.</strong>",
        shown(1),
        shown(prices.nrows() - 1)
    )));
    assert!(frontier.contains("includes the replay dates"));
}

#[tokio::test(flavor = "multi_thread")]
async fn csv_frequency_reaches_later_test_without_resampling() {
    let market = market_prices();
    let mut months = Vec::new();
    let mut date = day("2000-01-31");
    for _ in 0..market.nrows() {
        months.push(date);
        date = (date + chrono::Days::new(1)).checked_add_months(chrono::Months::new(1)).unwrap() - chrono::Days::new(1);
    }
    let prices = Frame::new(months, market.columns.clone(), market.values.clone());
    for (frequency, periods) in [("Monthly", 12.0), ("Custom", 48.5)] {
        let mut app = TestApp::new(Fake::default());
        app.page().await;
        let files = [("prices_upload", "monthly.csv", csv(&prices))];
        app.build(&[("market", "Upload CSV"), ("frequency", frequency), ("periods_per_year", "48.5")], &files).await;
        assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
        let applied = app.applied();
        assert_eq!(applied.metadata.get("periods_per_year").and_then(Json::as_f64), Some(periods));
        assert_eq!(applied.selection.settings.f64("periods_per_year"), periods);
        assert_eq!(applied.selection.test.settings.f64("periods_per_year"), periods);
        let page = app.page().await;
        for key in ["math_fit", "math_replay"] {
            let equations = equations(element(&page, key)).join("\n");
            assert!(equations.contains(&format!("A={}", efficient_frontier::pyfmt::g(periods))), "{key}");
            assert!(!equations.contains("A=252"));
        }
        assert_eq!(*applied.prices, prices);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn export_is_deferred_and_captures_completed_selection_not_draft() {
    let mut app = demo_app().await;
    assert!(app.differs(&[("target_vol", "25")], &[]).await);
    let reply = app.get(&format!("/export?result={}", app.applied().id)).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
    assert_eq!(reply.headers[header::CONTENT_DISPOSITION], "attachment; filename=\"portfolio-lab.zip\"");
    for name in
        ["metadata.json", "latest/frontier.csv", "out_of_sample/windows.csv", "test/equity.csv", "latest_weights.csv"]
    {
        assert!(reply.body.contains(name), "{name}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn export_requires_a_completed_study() {
    let mut app = TestApp::new(Fake::default());
    let reply = app.get("/export").await;
    assert_eq!(reply.status, StatusCode::NOT_FOUND);
    assert!(reply.body.contains("Build a portfolio first."));
}

// ----------------------------------------------------------------------------- failures

#[tokio::test(flavor = "multi_thread")]
async fn infeasible_rebuild_clears_stale_results_and_preserves_failed_settings() {
    let mut app = demo_app().await;
    app.build(&[("cap", "5")], &[]).await;
    let session = app.session();
    assert!(session.applied.is_none());
    assert_eq!(session.build_error.as_ref().unwrap().kind, crate::build::ErrorKind::Infeasible);
    let page = app.page().await;
    assert!(page.contains("These settings have no solution."));
    assert!(page.contains("A 5% limit needs at least 20 assets."));
    assert!(!page.contains(r#"id="export""#));
    assert!(element(&page, "fix_setup").contains("Change the study"));
    let drawer = app.drawer().await;
    assert_eq!((value(&drawer, "cap"), value(&drawer, "target_vol")), ("5".to_string(), "20.00".to_string()));
    app.build(&[("cap", "20")], &[]).await;
    assert!(app.session().build_error.is_none() && app.session().applied.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn infeasible_volatility_target_does_not_relax_the_limits() {
    let mut app = TestApp::new(Fake::default());
    app.page().await;
    app.form("/api/preview", &[("target_vol", "1"), ("cap", "10")]).await;
    app.build(&[("market", "Demo"), ("cap", "20")], &[]).await;
    let session = app.session();
    assert!(session.applied.is_none() && session.build_error.is_some());
    let page = app.page().await;
    assert!(
        page.contains("These settings have no solution.") && page.contains("No constraint is relaxed automatically.")
    );
    assert_eq!((session.setup.target_vol, session.setup.cap), (1.0, 20.0));
    let drawer = app.drawer().await;
    assert_eq!((value(&drawer, "target_vol"), value(&drawer, "cap")), ("1.00".to_string(), "20".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_csv_clears_stale_result_without_network() {
    let mut app = demo_app().await;
    app.build(&[("market", "Upload CSV")], &[]).await;
    let session = app.session();
    assert!(session.applied.is_none());
    assert!(session.build_error.as_ref().unwrap().message.contains("Choose an adjusted-price CSV"));
    let page = app.page().await;
    assert!(page.contains("The study needs a price file.") && !page.contains(r#"id="export""#));
    assert!(app.fake.downloads().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn download_error_names_its_cause_and_keeps_a_useful_action() {
    let cases = [
        (
            value_error!(
                "Yahoo returned no prices for: ZZZQQX. Check these symbols and their available history, or upload a CSV."
            ),
            "Yahoo Finance had no prices for these symbols.",
            "fix_setup",
        ),
        (io_error!("Network is unreachable"), "We could not reach Yahoo Finance.", "build"),
    ];
    for (failure, headline, action) in cases {
        let message = failure.message().to_string();
        let mut app = TestApp::new(Fake { prices: Some(Err(failure)), ..Fake::default() });
        app.page().await;
        app.build(&[("market", "My tickers"), ("ticker_text", "ZZZQQX")], &[]).await;
        let page = app.page().await;
        assert!(page.contains(&efficient_frontier::pyfmt::escape(&message)), "{message}");
        assert!(page.contains(headline) && has_element(&page, action), "{headline}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_panicking_download_becomes_an_error_page_and_the_server_keeps_working() {
    let mut app = TestApp::new(Fake { universe_panics: true, ..Fake::default() });
    app.page().await;
    app.build_landing("20", "10").await;
    let error = app.session().build_error.expect("a build error");
    assert_eq!(error.kind, crate::build::ErrorKind::Other);
    assert!(error.message.contains("not yet implemented"), "{}", error.message);
    let page = app.page().await;
    assert!(page.contains("The study could not run.") && has_element(&page, "build"));
    assert!(page.contains("Try again"));
    app.build(&[("market", "Demo"), ("cap", "20")], &[]).await;
    assert!(app.session().applied.is_some(), "the next build still works");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_work_card_reports_progress_and_cancel_keeps_the_previous_result() {
    let gate = Arc::new(Gate::default());
    let mut app = TestApp::new(Fake { universe_gate: Some(gate.clone()), ..Fake::universe() });
    app.page().await;
    app.build(&[("market", "Demo"), ("cap", "20")], &[]).await;
    let previous = app.applied();
    let reply = app.multipart("/build", &[("form", "drawer"), ("market", "Nasdaq-100"), ("cap", "10")], &[]).await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER);
    let deadline = Instant::now() + Duration::from_secs(30);
    let progress = loop {
        let value: serde_json::Value = serde_json::from_str(&app.get("/api/job").await.body).unwrap();
        if value["bar"]["text"] == "Checked 3 of 13 securities." {
            break value;
        }
        assert!(Instant::now() < deadline, "no progress: {value}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert_eq!(progress["step"], "Load the selected prices");
    assert!((progress["bar"]["value"].as_f64().unwrap() - 3.0 / 13.0).abs() < 1e-12);
    let page = app.page().await;
    assert!(page.contains("Building your portfolio") && page.contains("Load the selected prices"));
    assert!(!page.contains(r#"id="localnav""#), "the nav bar hides during a build");
    let reply = app.form("/cancel", &[]).await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER);
    assert!(app.session().job.is_none());
    assert!(Arc::ptr_eq(&app.applied(), &previous), "an applied result is replaced only after success");
    gate.release();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(Arc::ptr_eq(&app.applied(), &previous), "the cancelled result is discarded");
    assert!(app.page().await.contains(r#"data-page="overview""#));
}

// ----------------------------------------------------------------------------- old results and navigation

/// The same result, recorded by another selection model revision.
fn with_revision(applied: &Applied, revision: &str) -> Applied {
    let mut metadata: Dict = applied.metadata.clone();
    metadata.insert("selection_model_revision", revision);
    Applied {
        id: applied.id,
        prices: applied.prices.clone(),
        metadata,
        selection: applied.selection.clone(),
        frontier: applied.frontier.clone(),
        out_of_sample: applied.out_of_sample.clone(),
        profiles: applied.profiles.clone(),
        inputs: applied.inputs.clone(),
        benchmark_prices: applied.benchmark_prices.clone(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn old_study_waits_for_intentional_migration_and_retains_input_prices() {
    let mut app = demo_app().await;
    let prices = app.applied().prices.clone();
    let old = with_revision(&app.applied(), "stock_mean_buy_hold_v2");
    lock(&app.session_ref()).applied = Some(Arc::new(old));
    let page = app.page().await;
    assert!(page.contains(r#"data-page="stale""#));
    assert!(page.contains("This saved study uses a previous selection model."));
    assert!(!page.contains(r#"id="export""#) && !page.contains("plotly-graph-div") && equations(&page).is_empty());
    assert!(element(&page, "migrate_selection").contains("Set risk budget and rebuild"));
    assert_eq!(app.state.caches.selection.computed(), 1);
    let drawer = app.drawer().await;
    assert_eq!(checked(&drawer, "market"), "Demo");
    assert_eq!((value(&drawer, "target_vol"), value(&drawer, "cap")), ("20.00".to_string(), "20".to_string()));
    assert_eq!(app.applied().prices, prices);
    let old = app.applied().id;
    assert_eq!(app.get(&format!("/export?result={old}")).await.status, StatusCode::NOT_FOUND);
    app.build(&[], &[]).await;
    assert!(app.applied().is_current());
    assert!(app.page().await.contains(r#"id="export""#));
}

#[tokio::test(flavor = "multi_thread")]
async fn data_method_is_passive_and_shows_only_the_completed_model() {
    let mut app = demo_app().await;
    let study = app.applied().selection.clone();
    app.form("/view", &[("view", "Data & method")]).await;
    let page = app.page().await;
    assert!(page.contains(r#"<h1 class="pl-page">Data &amp; method</h1>"#) && page.contains("One stock model"));
    assert!(!page.contains("run_legacy") && !page.contains("research_topic"));
    let method = element(&page, "tile_method");
    assert!(method.contains("0.10%") && method.contains("2% risk-free rate"));
    let data = element(&page, "tile_data");
    assert!(data.contains("Price eligibility at each fit") && data.contains("Download prices.csv"));
    assert!(data.contains("Fixed list. The study keeps the same input assets"));
    for header in ["fit_date", "max_unchanged_intervals", "Return observations", "Largest holding", "Fitted through"] {
        assert!(page.contains(header), "{header}");
    }
    for header in ["Turnover", "Fee", "Value before", "Value after", "Fit from", "Fit to", "P&amp;L / initial capital"]
    {
        assert!(page.contains(&format!(">{header}</th>")), "{header}");
    }
    let trades = &study.test.trades[SELECTED];
    let fee = trades.f64s("cost")[0] * 10_000.0;
    assert!(page.contains(&efficient_frontier::pyfmt::format(fee, ",.2f")));
    let id = app.applied().id;
    assert!(data.contains(&format!(r#"data-src="/api/prices?result={id}""#)));
    let csv = app.get(&format!("/download/prices.csv?result={id}")).await;
    assert_eq!(csv.headers[header::CONTENT_DISPOSITION], "attachment; filename=\"prices.csv\"");
    assert!(csv.body.starts_with("Date,DEMO_"));
    let table: serde_json::Value =
        serde_json::from_str(&app.get(&format!("/api/prices?result={id}")).await.body).unwrap();
    let prices = app.applied().prices.clone();
    assert_eq!(table["index"][0], crate::render::format::day(prices.index[0]));
    assert_eq!(table["columns"].as_array().unwrap().len(), prices.ncols());
    assert_eq!(table["rows"].as_array().unwrap().len(), prices.nrows());
    assert_eq!(table["rows"][0][0], crate::render::format::number(prices.get(0, 0)));
    assert_eq!(app.state.caches.selection.computed(), 1);
    assert!(Arc::ptr_eq(&app.applied().selection, &study));
    app.form("/view", &[("view", "Portfolio")]).await;
    assert!(app.page().await.contains(r#"data-page="overview""#));
    assert_eq!(app.state.caches.selection.computed(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn data_method_names_the_comparisons_and_checks() {
    let mut app = demo_app().await;
    app.form("/view", &[("view", "Data & method")]).await;
    let page = app.page().await;
    assert!(page.contains("buy once and hold") && page.contains("not checks of optimizer accuracy"));
    assert!(element(&page, "tile_data").contains("Market note"));
}

#[tokio::test(flavor = "multi_thread")]
async fn navigation_shortcuts_preserve_selection() {
    let mut app = demo_app().await;
    let selection = app.applied().selection.clone();
    let reply = app.form("/view", &[("view", "Data & method"), ("anchor", "")]).await;
    assert_eq!(reply.headers[header::LOCATION], "/");
    assert_eq!(app.session().view, crate::session::View::DataMethod);
    let reply = app.form("/view", &[("view", "Portfolio"), ("anchor", "pl-market")]).await;
    assert_eq!(reply.headers[header::LOCATION], "/#pl-market");
    assert_eq!(app.session().view, crate::session::View::Portfolio);
    assert!(app.page().await.contains(r#"<div id="pl-market"></div>"#));
    let reply = app.form("/view", &[("view", "Portfolio"), ("anchor", "x\"><script>")]).await;
    assert_eq!(reply.headers[header::LOCATION], "/");
    assert!(Arc::ptr_eq(&app.applied().selection, &selection));
    let page = app.page().await;
    for key in [
        "<kbd>1</kbd> overview",
        "<kbd>2</kbd> test",
        "<kbd>4</kbd> data &amp; method",
        "<kbd>S</kbd> setup",
        "<kbd>E</kbd> export",
    ] {
        assert!(page.contains(key), "{key}");
    }
}

// ----------------------------------------------------------------------------- sessions and serving

#[tokio::test(flavor = "multi_thread")]
async fn each_browser_keeps_its_own_study() {
    let mut first = demo_app().await;
    let mut second =
        TestApp { router: first.router.clone(), state: first.state.clone(), fake: first.fake.clone(), cookie: None };
    let page = second.page().await;
    assert!(page.contains(r#"data-page="landing""#));
    assert_ne!(first.cookie, second.cookie);
    assert!(first.page().await.contains(r#"data-page="overview""#));
}

#[tokio::test(flavor = "multi_thread")]
async fn default_desktop_session_survives_long_idle_with_its_study_csv_and_settings() {
    let prices = market_prices();
    let benchmarks = benchmark_prices(&prices.index);
    let mut app = csv_app(&prices, Some(&benchmarks)).await;
    let original = app.applied();
    let setup = app.session().setup;
    let cookie = app.cookie.clone().unwrap();
    app.state.sessions.advance_clock(IDLE_LIMIT + Duration::from_secs(1));
    let preview = app.form("/api/preview", &[("target_vol", "25"), ("cap", "30")]).await;
    assert_eq!(preview.status, StatusCode::NO_CONTENT);
    assert!(!preview.headers.contains_key(header::SET_COOKIE));
    let session = app.session();
    assert_eq!((session.setup.target_vol, session.setup.cap), (25.0, 30.0));
    assert!(Arc::ptr_eq(session.applied.as_ref().unwrap(), &original));
    assert_eq!(session.setup.prices_file.as_ref().unwrap().bytes.as_ref(), csv(&prices));
    assert_eq!(session.setup.benchmark_file.as_ref().unwrap().bytes.as_ref(), csv(&benchmarks));

    app.state.sessions.advance_clock(IDLE_LIMIT + Duration::from_secs(1));
    let page = app.get("/").await;
    assert_eq!(page.status, StatusCode::OK);
    assert!(!page.headers.contains_key(header::SET_COOKIE));
    assert!(page.body.contains(r#"data-page="overview""#));
    assert_eq!(app.cookie.as_deref(), Some(cookie.as_str()));
    assert_eq!(app.state.sessions.count(), 1);
    let drawer = app.drawer().await;
    assert!(drawer.contains("Using portfolio.csv. Upload a file to replace it."));
    assert!(drawer.contains("Using benchmarks.csv. Upload a file to replace it."));
    assert_eq!(app.session().setup.prices_file, setup.prices_file);
    assert_eq!(app.session().setup.benchmark_file, setup.benchmark_file);
    assert!(app.fake.downloads().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn assets_are_served_and_other_hosts_are_refused() {
    let mut app = TestApp::new(Fake::default());
    let page = app.page().await;
    for asset in ["css/app.css", "js/app.js", "katex/katex.min.js", "katex/auto-render.min.js", "katex/katex.min.css"] {
        let url = crate::assets::url(asset);
        assert!(page.contains(&url), "{asset}");
        let reply = app.get(&url).await;
        assert_eq!(reply.status, StatusCode::OK, "{asset}");
        assert!(reply.headers[header::CACHE_CONTROL].to_str().unwrap().contains("immutable"));
    }
    assert_eq!(
        app.get("/assets/katex/fonts/KaTeX_Main-Regular.woff2").await.headers[header::CONTENT_TYPE],
        "font/woff2"
    );
    assert_eq!(app.get("/assets/missing.js").await.status, StatusCode::NOT_FOUND);
    let request = Request::builder().uri("/").header(header::HOST, "example.com").body(Body::empty()).unwrap();
    let response = app.router.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let request = Request::builder().uri("/").header(header::HOST, "127.0.0.1:8501").body(Body::empty()).unwrap();
    assert_eq!(app.router.clone().oneshot(request).await.unwrap().status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_drawer_shows_every_control_for_its_market() {
    let mut app = TestApp::new(Fake::default());
    app.page().await;
    let drawer = app.drawer().await;
    assert!(drawer.contains("<h2 id=\"pl-drawer-title\">Change the study</h2>"));
    assert!(drawer.contains("Choose the stocks and the history. Nothing changes until you select Build portfolio."));
    for text in [
        "Today’s combined members, without duplicate symbols. Prices from Yahoo Finance.",
        "Stock symbols",
        "Adjusted prices (.csv)",
        "History",
        ">Custom</span>",
        "The last day is not included.",
        "30 Oct 2021 to 29 Oct 2026",
        "30 Oct 2016 to 29 Oct 2026",
        "Annual volatility budget",
        "Holding limit",
        "At least 17% is needed.",
        "One fixed method: fit on the first 70% of return observations",
        "Data options <span class=\"pl-gray\">frequency, currency, market comparisons</span>",
        "Price frequency",
        "Observations per year",
        "Frequency scales yearly figures. It does not resample prices.",
        "Price currency",
        "SPY and QQQ comparisons need USD prices. No currency conversion occurs.",
        "Compare with the S&amp;P 500 and the Nasdaq-100",
        "Market prices (optional)",
        "Download SPY and QQQ from Yahoo Finance for this file",
        "Local comparison columns take priority over a download.",
        "Your setup differs from this result. Select Build portfolio to apply it.",
        ">Cancel</button>",
        "Build portfolio",
    ] {
        assert!(drawer.contains(text), "{text}");
    }
    // The Yahoo market shows its history; the CSV, ticker and demo blocks start hidden.
    assert!(
        drawer.contains(
            r#"<div data-when-market="S&amp;P 500 + Nasdaq-100|Nasdaq-100|My tickers|Original 60 holdings">"#
        )
    );
    assert!(drawer.contains(r#"<div data-when-market="Upload CSV" hidden>"#));
    assert!(drawer.contains(r#"<div data-when-market="My tickers" hidden>"#));
    assert!(drawer.contains(r#"<div data-when-market="Demo" hidden>"#));
    assert!(!differs_shown(&drawer));
    assert!(!drawer.contains("Reference from the saved input history"));
    let mut app = csv_app(&market_prices(), Some(&benchmark_prices(&market_prices().index))).await;
    let drawer = app.drawer().await;
    assert!(drawer.contains("Reference from the saved input history: SPY "));
    assert!(drawer.contains(r#"<div data-when-market="Upload CSV">"#));
}

// ----------------------------------------------------------------------------- review fixes

/// Poll until `done` holds, for work on a background thread.
async fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Wait until the running build reports this progress message.
async fn wait_for_progress(app: &mut TestApp, text: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let value: serde_json::Value = serde_json::from_str(&app.get("/api/job").await.body).unwrap();
        if value["bar"]["text"] == text {
            return;
        }
        assert!(Instant::now() < deadline, "no progress {text:?}: {value}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

const HOST: (&str, &str) = ("host", "127.0.0.1:8501");

#[tokio::test(flavor = "multi_thread")]
async fn cross_site_posts_and_background_reads_are_refused() {
    let mut app = TestApp::new(Fake::universe());
    app.page().await;
    let form = Some("application/x-www-form-urlencoded".to_string());
    let limits = b"target_vol=30&cap=30".to_vec();
    let body = multipart(&[("form", "landing"), ("target_vol", "20"), ("cap", "10")], &[]);
    let multipart_type = Some(format!("multipart/form-data; boundary={BOUNDARY}"));
    let refused = [
        vec![HOST, ("origin", "https://evil.example")],
        vec![HOST, ("origin", "http://localhost:8501")],
        vec![HOST, ("origin", "null")],
        vec![HOST, ("sec-fetch-site", "cross-site")],
        vec![HOST, ("sec-fetch-site", "same-site")],
        vec![HOST, ("origin", "http://127.0.0.1:8501"), ("sec-fetch-site", "cross-site")],
    ];
    for headers in &refused {
        let reply = app.send_with(Method::POST, "/build", headers, multipart_type.clone(), body.clone()).await;
        assert_eq!(reply.status, StatusCode::FORBIDDEN, "{headers:?}");
        let reply = app.send_with(Method::POST, "/api/preview", headers, form.clone(), limits.clone()).await;
        assert_eq!(reply.status, StatusCode::FORBIDDEN, "{headers:?}");
    }
    let session = app.session();
    assert!(session.job.is_none() && session.setup.cap == 10.0, "no refused request changed the study");
    assert!(app.fake.downloads().is_empty());
    let own = [HOST, ("origin", "http://127.0.0.1:8501"), ("sec-fetch-site", "same-origin")];
    let reply = app.send_with(Method::POST, "/api/preview", &own, form.clone(), limits).await;
    assert_eq!(reply.status, StatusCode::NO_CONTENT);
    assert_eq!(app.session().setup.cap, 30.0);
    let background = [HOST, ("sec-fetch-site", "cross-site"), ("sec-fetch-mode", "no-cors")];
    assert_eq!(
        app.send_with(Method::GET, "/api/job", &background, None, Vec::new()).await.status,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        app.send_with(Method::GET, "/assets/js/app.js", &background, None, Vec::new()).await.status,
        StatusCode::FORBIDDEN
    );
    let link = [HOST, ("sec-fetch-site", "cross-site"), ("sec-fetch-mode", "navigate")];
    assert_eq!(app.send_with(Method::GET, "/", &link, None, Vec::new()).await.status, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_post_without_a_known_session_creates_nothing() {
    let mut app = TestApp::new(Fake::universe());
    let body = multipart(&[("form", "landing"), ("target_vol", "20"), ("cap", "10")], &[]);
    let content_type = Some(format!("multipart/form-data; boundary={BOUNDARY}"));
    for cookie in [None, Some("pl_session=0123456789abcdef0123456789abcdef".to_string())] {
        app.cookie = cookie;
        let reply = app.send(Method::POST, "/build", content_type.clone(), body.clone()).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        assert_eq!(reply.headers[header::LOCATION], "/");
        assert!(!reply.headers.contains_key(header::SET_COOKIE));
        let reply = app.form("/api/preview", &[("target_vol", "30"), ("cap", "30")]).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
    }
    assert_eq!(app.state.sessions.count(), 0);
    assert!(app.fake.calls().is_empty(), "no session, no QQQ lookup, no build");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_session_store_removes_the_least_recently_used_browser() {
    let mut first = TestApp::with(Fake::default(), |state| state.sessions = Sessions::new(2, Some(IDLE_LIMIT)));
    let share = |app: &TestApp| TestApp {
        router: app.router.clone(),
        state: app.state.clone(),
        fake: app.fake.clone(),
        cookie: None,
    };
    first.page().await;
    first.form("/api/preview", &[("target_vol", "25"), ("cap", "10")]).await;
    let first_cookie = first.cookie.clone();
    let mut second = share(&first);
    second.page().await;
    let mut third = share(&first);
    third.page().await;
    assert_eq!(first.state.sessions.count(), 2);
    // The first browser's session left; its next page starts a new one with the defaults.
    let page = first.page().await;
    assert_ne!(first.cookie, first_cookie);
    assert_eq!(value(&page, "preview_target_vol"), "20.00");
    assert_eq!(first.state.sessions.count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_stops_the_universe_download_at_the_next_batch() {
    let gate = Arc::new(Gate::default());
    let mut app = TestApp::new(Fake { batch_gate: Some(gate.clone()), ..Fake::universe() });
    app.page().await;
    let reply = app.multipart("/build", &[("form", "landing"), ("target_vol", "20"), ("cap", "10")], &[]).await;
    assert_eq!(reply.status, StatusCode::SEE_OTHER);
    wait_for_progress(&mut app, "Checked 25 of 100 securities.").await;
    assert_eq!(app.form("/cancel", &[]).await.status, StatusCode::SEE_OTHER);
    gate.release();
    let fake = app.fake.clone();
    eventually("the download to stop", || fake.calls().iter().any(|call| call.starts_with("stopped"))).await;
    let batches: Vec<String> = app.fake.calls().into_iter().filter(|call| call.starts_with("batch")).collect();
    assert_eq!(batches, ["batch 1"]);
    assert!(app.fake.calls().contains(&"stopped before batch 2".to_string()));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(app.state.caches.universe.computed(), 0, "a cancelled download is not cached");
    assert_eq!(app.state.caches.selection.computed(), 0, "no fit after Cancel");
    assert!(app.session().applied.is_none() && app.session().build_error.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rebuild_after_cancel_waits_for_the_stopping_download() {
    let gate = Arc::new(Gate::default());
    let mut app = TestApp::new(Fake { batch_gate: Some(gate.clone()), ..Fake::universe() });
    app.page().await;
    let landing = [("form", "landing"), ("target_vol", "20"), ("cap", "10")];
    app.multipart("/build", &landing, &[]).await;
    wait_for_progress(&mut app, "Checked 25 of 100 securities.").await;
    app.form("/cancel", &[]).await;
    app.multipart("/build", &landing, &[]).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let universe_calls = |app: &TestApp| app.fake.calls().iter().filter(|call| call.starts_with("universe ")).count();
    assert_eq!(universe_calls(&app), 1, "the second build waits for the same download");
    gate.release();
    app.wait().await;
    assert!(app.session().build_error.is_none(), "{:?}", app.session().build_error);
    assert!(app.session().applied.is_some());
    assert_eq!(universe_calls(&app), 2);
    assert_eq!(app.fake.most_running.load(Ordering::SeqCst), 1, "never two downloads at once");
    let batches: Vec<String> = app.fake.calls().into_iter().filter(|call| call.starts_with("batch")).collect();
    assert_eq!(batches, ["batch 1", "batch 1", "batch 2", "batch 3", "batch 4"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn downloads_of_an_older_study_answer_409() {
    let mut app = demo_app().await;
    let older = app.applied().id;
    app.build(&[("cap", "25")], &[]).await;
    let newer = app.applied().id;
    assert_ne!(older, newer);
    for path in ["/export", "/download/prices.csv", "/api/prices"] {
        let reply = app.get(&format!("{path}?result={older}")).await;
        assert_eq!(reply.status, StatusCode::CONFLICT, "{path}");
        assert!(reply.body.contains(OLDER_STUDY), "{path}");
        assert_eq!(app.get(path).await.status, StatusCode::CONFLICT, "{path} without a result id");
        let reply = app.get(&format!("{path}?result={newer}")).await;
        assert_eq!(reply.status, StatusCode::OK, "{path}: {}", reply.body);
    }
    let page = app.page().await;
    assert!(page.contains(&format!("/export?result={newer}")));
    app.form("/view", &[("view", "Data & method")]).await;
    let page = app.page().await;
    assert!(page.contains(&format!("/download/prices.csv?result={newer}")));
    assert!(page.contains(&format!("/api/prices?result={newer}")));
}

#[tokio::test(flavor = "multi_thread")]
async fn draft_checks_compare_file_digests_without_file_bytes() {
    let prices = market_prices();
    let mut app = csv_app(&prices, Some(&benchmark_prices(&prices.index))).await;
    let applied = hex(&Sha256::digest(csv(&prices)));
    let mut changed = prices.clone();
    changed.values[(0, 0)] *= 1.01;
    let other = hex(&Sha256::digest(csv(&changed)));
    let draft = |name: &'static str, digest: String| {
        vec![
            ("market", "Upload CSV".to_string()),
            ("prices_upload_name", name.to_string()),
            ("prices_upload_sha256", digest),
        ]
    };
    for (fields, expected) in [
        (draft("portfolio.csv", applied.clone()), false),
        (draft("portfolio.csv", other), true),
        (draft("renamed.csv", applied.clone()), true),
        (draft("portfolio.csv", "not a digest".into()), false),
    ] {
        let fields: Vec<(&str, &str)> = fields.iter().map(|(name, value)| (*name, value.as_str())).collect();
        assert_eq!(app.differs(&fields, &[]).await, expected, "{fields:?}");
    }
}

// ----------------------------------------------------------------------------- the window's protocol

/// A request of the window's web view, through the bridge. A download panics.
async fn through_window(bridge: &Bridge, method: Method, path: &str, headers: &[(&str, &str)], body: Vec<u8>) -> Reply {
    match bridge.forward(webview_request(method, path, headers, body)).await {
        Forwarded::Page(response) => Reply {
            status: response.status(),
            headers: response.headers().clone(),
            body: String::from_utf8_lossy(response.body()).into_owned(),
        },
        Forwarded::Download(download) => panic!("{path} offered {} for download", download.name),
    }
}

#[tokio::test]
async fn the_window_builds_a_demo_study_and_gets_its_report_as_a_download() {
    let mut state = AppState::new(Arc::new(Fake::default()));
    state.clock = today;
    let bridge = Bridge::new(crate::routes::router(Arc::new(state)));
    let own = if cfg!(windows) { "http://app.localhost" } else { "app://localhost" };
    let load = [("sec-fetch-mode", "navigate"), ("sec-fetch-site", "none")];
    let script = [("sec-fetch-mode", "cors"), ("sec-fetch-site", "same-origin")];

    let start = through_window(&bridge, Method::GET, "/", &load, Vec::new()).await;
    assert_eq!(start.status, StatusCode::OK);
    assert!(start.body.contains("Efficient Frontier Analysis for US Equities"));
    assert!(start.headers.get(header::SET_COOKIE).is_none(), "the bridge keeps the session cookie");

    let content_type = format!("multipart/form-data; boundary={BOUNDARY}");
    let post = [
        ("sec-fetch-mode", "navigate"),
        ("sec-fetch-site", "same-origin"),
        ("origin", own),
        ("content-type", content_type.as_str()),
    ];
    let form = multipart(&[("form", "drawer"), ("market", "Demo"), ("cap", "20")], &[]);
    let posted = through_window(&bridge, Method::POST, "/build", &post, form.clone()).await;
    assert_eq!(posted.status, StatusCode::OK, "a page that loads the start page");
    assert!(posted.body.contains("url=/\""), "{}", posted.body);

    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let job = through_window(&bridge, Method::GET, "/api/job", &script, Vec::new()).await;
        if serde_json::from_str::<serde_json::Value>(&job.body).unwrap()["active"] == false {
            break;
        }
        assert!(Instant::now() < deadline, "the build did not finish");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let overview = through_window(&bridge, Method::GET, "/", &load, Vec::new()).await;
    let export = overview.body.split("href=\"/export?result=").nth(1).and_then(|rest| rest.split('"').next());
    let export = format!("/export?result={}", export.expect("an export link"));
    match bridge.forward(webview_request(Method::GET, &export, &script, Vec::new())).await {
        Forwarded::Download(download) => {
            assert_eq!(download.name, "portfolio-lab.zip");
            assert!(download.bytes.starts_with(b"PK"), "a zip file");
        }
        Forwarded::Page(response) => panic!("the export answered {}", response.status()),
    }

    let foreign = [("sec-fetch-mode", "navigate"), ("sec-fetch-site", "cross-site"), ("origin", "https://example.com")];
    let refused = through_window(&bridge, Method::POST, "/build", &[&foreign[..], &post[3..]].concat(), form).await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
}
