//! HTTP routes: pages, form posts, JSON for the browser script, downloads and assets.
//!
//! Form posts answer with "303 See Other" to `/`, so the browser shows the new
//! state with a fresh page, as a Streamlit rerun does. Requests that another
//! site makes with this browser are refused, as Streamlit's XSRF protection does.

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Extension, Form, Multipart, Query, Request, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_TYPE, HOST, ORIGIN, SET_COOKIE};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Json, Redirect, Response};
use axum::routing::{get, post};
use efficient_frontier::selection_report::selection_report_zip;
use serde::Deserialize;
use serde_json::json;

use crate::assets;
use crate::build::{self, Applied};
use crate::guard::guard;
use crate::render::html::{Alert, alert_plain};
use crate::render::{self, drawer::drawer, table::frame_json};
use crate::session::{Session, SessionRef, View, cookie_id, lock, set_cookie};
use crate::setup::{Draft, Setup, Upload, differs, read_draft, resolve};
use crate::state::AppState;

/// The limit per CSV file, Streamlit's default `server.maxUploadSize` of 200 MB.
pub const UPLOAD_LIMIT: usize = 200 * 1024 * 1024;
/// The drawer's file inputs.
const FILE_FIELDS: [&str; 2] = ["prices_upload", "benchmark_upload"];
/// The answer to a page that asks for a study that is no longer applied.
pub const OLDER_STUDY: &str = "This page shows an older study. Reload the page.";

fn multipart_body_limit(file_limit: usize) -> usize {
    // Both CSV files, plus the drawer fields and multipart headers.
    FILE_FIELDS.len() * file_limit + 64 * 1024
}

/// The session of the current request.
#[derive(Clone)]
pub struct CurrentSession(pub SessionRef);

/// The app's routes.
pub fn router(state: Arc<AppState>) -> Router {
    let app = Router::new()
        .route("/", get(index))
        .route("/drawer", get(drawer_fragment))
        .route("/build", post(start_build))
        .route("/cancel", post(cancel))
        .route("/view", post(change_view))
        .route("/api/preview", post(preview))
        .route("/api/draft", post(draft_differs))
        .route("/api/job", get(job))
        .route("/api/prices", get(prices_table))
        .route("/export", get(export))
        .route("/download/prices.csv", get(prices_csv))
        .layer(DefaultBodyLimit::max(multipart_body_limit(UPLOAD_LIMIT)))
        .route_layer(middleware::from_fn_with_state(state.clone(), with_session));
    Router::new()
        .route("/assets/{*path}", get(assets::serve))
        .merge(app)
        .layer(middleware::from_fn(same_site_only))
        .layer(middleware::from_fn(local_host))
        .with_state(state)
}

/// Find the browser's session by cookie, or create one with `init_state`'s defaults.
///
/// Only a page load or a browser script's read creates a session. A post
/// without a known session (another site's form, or a page from before a
/// restart) goes to the start page and changes nothing.
async fn with_session(State(state): State<Arc<AppState>>, mut request: Request, next: Next) -> Response {
    let existing = cookie_id(request.headers()).and_then(|id| state.sessions.get(&id));
    let (handle, new_id) = match existing {
        Some(handle) => (handle, None),
        None if !is_read(request.method()) => return Redirect::to("/").into_response(),
        None => {
            let creator = state.clone();
            // The QQQ reference can wait on the network, so create the session off the async workers.
            let session = tokio::task::spawn_blocking(move || creator.new_session())
                .await
                .unwrap_or_else(|_| Session::new(Setup::new(state.today())));
            let (id, handle) = state.sessions.insert(session);
            (handle, Some(id))
        }
    };
    request.extensions_mut().insert(CurrentSession(handle));
    let mut response = next.run(request).await;
    if let Some(id) = new_id {
        response.headers_mut().append(SET_COOKIE, set_cookie(&id));
    }
    response
}

/// Refuse requests for other host names, so that a web page cannot reach this
/// server through DNS rebinding. Requests without a host (tests) pass.
async fn local_host(request: Request, next: Next) -> Response {
    let allowed = request.headers().get(HOST).is_none_or(|value| value.to_str().is_ok_and(is_local_host));
    if !allowed {
        return (StatusCode::FORBIDDEN, "Portfolio Lab answers only requests for this computer.").into_response();
    }
    next.run(request).await
}

/// True for `localhost`, `127.0.0.1` and `[::1]`, with any port.
fn is_local_host(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or_default(),
        None => host.split(':').next().unwrap_or_default(),
    };
    matches!(name, "localhost" | "127.0.0.1" | "::1")
}

/// GET and HEAD only read; every other method can change the study.
fn is_read(method: &Method) -> bool {
    method == Method::GET || method == Method::HEAD
}

/// Refuse requests that another site makes with this browser.
async fn same_site_only(request: Request, next: Next) -> Response {
    if !from_this_site(request.method(), request.headers()) {
        return (StatusCode::FORBIDDEN, "Portfolio Lab refuses requests that other sites make.").into_response();
    }
    next.run(request).await
}

/// The cross-site policy, from the browser's `Origin` and `Sec-Fetch-*` headers.
///
/// A read (GET or HEAD) from another site passes only as a top-level navigation,
/// such as a link. Any other request passes only when `Origin` is this server's
/// own origin and `Sec-Fetch-Site` is `same-origin` or `none`. A request without
/// these headers, such as one from curl or a test, passes.
fn from_this_site(method: &Method, headers: &HeaderMap) -> bool {
    let header = |name: &str| headers.get(name).map(|value| value.to_str().unwrap_or("\u{0}"));
    let site = header("sec-fetch-site");
    if is_read(method) {
        return !matches!(site, Some("cross-site" | "same-site")) || header("sec-fetch-mode") == Some("navigate");
    }
    if let Some(origin) = header(ORIGIN.as_str()) {
        let own = header(HOST.as_str()).map(|host| format!("http://{host}"));
        if !own.is_some_and(|own| own.eq_ignore_ascii_case(origin)) {
            return false;
        }
    }
    site.is_none_or(|site| matches!(site, "same-origin" | "none"))
}

/// Run blocking or CPU-heavy work off the async workers. A panic becomes `Err`.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T, String> {
    match tokio::task::spawn_blocking(move || guard(work)).await {
        Ok(result) => result,
        Err(error) => Err(error.to_string()),
    }
}

/// An HTML page that the browser must not reuse from its cache.
fn page(status: StatusCode, html: String) -> Response {
    (status, [(CACHE_CONTROL, HeaderValue::from_static("no-store"))], Html(html)).into_response()
}

fn failure(status: StatusCode, title: &str, message: &str) -> Response {
    page(status, render::message_page(title, message))
}

/// `GET /`: the page for the session's state.
async fn index(
    State(state): State<Arc<AppState>>,
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
) -> Response {
    let session = lock(&handle).clone();
    let today = state.today();
    match blocking(move || render::page(&session, today)).await {
        Ok(html) => page(StatusCode::OK, html),
        Err(message) => failure(StatusCode::INTERNAL_SERVER_ERROR, "The page could not be shown.", &message),
    }
}

/// `GET /drawer`: the settings drawer, filled from the applied setup.
async fn drawer_fragment(
    State(state): State<Arc<AppState>>,
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
) -> Response {
    let session = lock(&handle).clone();
    let today = state.today();
    match blocking(move || drawer(&session, today)).await {
        Ok(html) => page(StatusCode::OK, html),
        Err(message) => failure(StatusCode::INTERNAL_SERVER_ERROR, "The settings could not be shown.", &message),
    }
}

/// The fields and file parts of a multipart form, before the files are hashed.
struct Posted {
    fields: Vec<(String, String)>,
    /// (field name, file name, bytes). A file input without a file is left out.
    files: Vec<(String, String, Bytes)>,
}

impl Posted {
    /// The draft of the form. Hashing the files takes time, so call this off the async workers.
    fn into_draft(self) -> Draft {
        let mut draft = Draft::default();
        draft.fields.extend(self.fields);
        for (name, file_name, bytes) in self.files {
            draft.uploads.insert(name, Upload::new(file_name, bytes));
        }
        draft
    }
}

/// Read a multipart form. The file bytes stay in the request's buffers.
async fn read_form(mut multipart: Multipart, file_limit: usize) -> Result<Posted, String> {
    let read_error = |error: axum::extract::multipart::MultipartError| {
        if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
            format!("Upload request is too large. Each CSV file must be at most {file_limit} bytes.")
        } else {
            error.body_text()
        }
    };
    let mut posted = Posted { fields: Vec::new(), files: Vec::new() };
    while let Some(field) = multipart.next_field().await.map_err(read_error)? {
        let name = field.name().unwrap_or_default().to_string();
        match field.file_name().map(str::to_string) {
            Some(file_name) => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| format!("Could not read file {file_name}. {}", read_error(error)))?;
                if bytes.len() > file_limit {
                    return Err(format!("File {file_name} exceeds the per-file limit of {file_limit} bytes."));
                }
                if !file_name.is_empty() {
                    posted.files.push((name, file_name, bytes));
                }
            }
            None => {
                let text = field.text().await.map_err(read_error)?;
                posted.fields.push((name, text));
            }
        }
    }
    Ok(posted)
}

/// Read a multipart form and hash its files off the async workers.
async fn read_draft_form(multipart: Multipart) -> Result<Draft, String> {
    let posted = read_form(multipart, UPLOAD_LIMIT).await?;
    if posted.files.is_empty() { Ok(posted.into_draft()) } else { blocking(move || posted.into_draft()).await }
}

/// Apply the landing page's two limits to the setup.
fn apply_limits(setup: &mut Setup, draft: &Draft) {
    if let Some(target) = draft.limit("target_vol") {
        setup.target_vol = target;
    }
    if let Some(cap) = draft.limit("cap") {
        setup.cap = cap;
    }
}

/// `POST /build`: apply the posted setup and start a build.
///
/// `form=drawer` applies the whole drawer and shows the overview afterwards;
/// `form=landing` applies the two landing limits; any other form ("Try again")
/// rebuilds the current setup.
async fn start_build(
    State(state): State<Arc<AppState>>,
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    multipart: Multipart,
) -> Response {
    let draft = match read_draft_form(multipart).await {
        Ok(draft) => draft,
        Err(message) => return failure(StatusCode::BAD_REQUEST, "The form could not be read.", &message),
    };
    let mut session = lock(&handle);
    match draft.field("form") {
        Some("drawer") => {
            session.setup = read_draft(&draft, &session.setup);
            session.view = View::Portfolio;
        }
        Some("landing") => apply_limits(&mut session.setup, &draft),
        _ => {}
    }
    build::start(&state, &handle, &mut session);
    Redirect::to("/").into_response()
}

/// `POST /cancel`: stop the running build; its result is discarded.
async fn cancel(Extension(CurrentSession(handle)): Extension<CurrentSession>) -> Redirect {
    if let Some(job) = lock(&handle).job.take() {
        job.cancel();
    }
    Redirect::to("/")
}

#[derive(Deserialize)]
struct ViewForm {
    view: String,
    #[serde(default)]
    anchor: String,
}

/// `POST /view`: switch between Overview and Data & method (`jump`), then scroll to an anchor.
async fn change_view(
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    Form(form): Form<ViewForm>,
) -> Redirect {
    if let Some(view) = View::parse(&form.view) {
        lock(&handle).view = view;
    }
    let anchor = form.anchor.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if form.anchor.is_empty() || !anchor { Redirect::to("/") } else { Redirect::to(&format!("/#{}", form.anchor)) }
}

#[derive(Deserialize)]
struct LimitsForm {
    #[serde(default)]
    target_vol: String,
    #[serde(default)]
    cap: String,
}

/// `POST /api/preview`: the landing page's limits, saved as they change.
async fn preview(
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    Form(form): Form<LimitsForm>,
) -> StatusCode {
    let mut draft = Draft::default();
    draft.fields.insert("target_vol".into(), form.target_vol);
    draft.fields.insert("cap".into(), form.cap);
    apply_limits(&mut lock(&handle).setup, &draft);
    StatusCode::NO_CONTENT
}

/// `POST /api/draft`: whether the drawer's draft differs from the applied result.
///
/// The browser hashes a chosen file once and sends `<field>_name` and
/// `<field>_sha256` instead of the file, so a draft check never uploads it.
async fn draft_differs(
    State(state): State<Arc<AppState>>,
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    multipart: Multipart,
) -> Response {
    let mut draft = match read_draft_form(multipart).await {
        Ok(draft) => draft,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };
    for field in FILE_FIELDS {
        if draft.uploads.contains_key(field) {
            continue;
        }
        let name = draft.field(&format!("{field}_name")).unwrap_or_default().to_string();
        let identity = draft.field(&format!("{field}_sha256")).and_then(|digest| Upload::identity(name, digest));
        if let Some(upload) = identity {
            draft.uploads.insert(field.to_string(), upload);
        }
    }
    let session = lock(&handle);
    let pending = session.applied.as_ref().is_some_and(|applied| {
        let plan = resolve(&read_draft(&draft, &session.setup), state.today());
        differs(&plan, &applied.metadata, &applied.inputs)
    });
    Json(json!({ "differs": pending })).into_response()
}

/// `GET /api/job`: the work card's step and progress, or `active: false` when no build runs.
async fn job(Extension(CurrentSession(handle)): Extension<CurrentSession>) -> Response {
    let job = lock(&handle).job.clone();
    let body = match job {
        Some(job) => {
            let progress = job.progress();
            json!({
                "active": true,
                "step": progress.step,
                "bar": progress.bar.map(|(value, text)| json!({ "value": value, "text": text })),
                "detail": progress.detail,
            })
        }
        None => json!({ "active": false }),
    };
    ([(CACHE_CONTROL, HeaderValue::from_static("no-store"))], Json(body)).into_response()
}

/// The `result` that a page puts in the URLs of its study's downloads.
#[derive(Deserialize)]
struct ResultQuery {
    result: Option<u64>,
}

/// Why a download cannot serve the page that asked for it.
enum Mismatch {
    /// The session has no result.
    Missing,
    /// The page shows another result than the applied one.
    Older,
}

/// The applied result that the page shows, if it is still applied.
fn shown_result(handle: &SessionRef, query: &ResultQuery) -> Result<Arc<Applied>, Mismatch> {
    let applied = lock(handle).applied.clone().ok_or(Mismatch::Missing)?;
    if query.result == Some(applied.id) { Ok(applied) } else { Err(Mismatch::Older) }
}

/// The message page for a download that cannot serve its page.
fn mismatch_page(mismatch: Mismatch, title: &str) -> Response {
    match mismatch {
        Mismatch::Missing => failure(StatusCode::NOT_FOUND, title, "Build a portfolio first."),
        Mismatch::Older => failure(StatusCode::CONFLICT, "This study has changed.", OLDER_STUDY),
    }
}

/// `GET /export?result=<id>`: the report zip of the applied study, computed only when asked for.
async fn export(
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    Query(query): Query<ResultQuery>,
) -> Response {
    let applied = match shown_result(&handle, &query) {
        Ok(applied) if applied.is_current() => applied,
        Ok(_) => return mismatch_page(Mismatch::Missing, "Nothing to export."),
        Err(mismatch) => return mismatch_page(mismatch, "Nothing to export."),
    };
    let zip = blocking(move || {
        selection_report_zip(
            &applied.selection,
            &applied.prices,
            &applied.metadata,
            applied.frontier.as_ref().ok().map(Arc::as_ref),
            applied.out_of_sample.as_ref().ok().map(Arc::as_ref),
        )
    })
    .await;
    match zip {
        Ok(Ok(bytes)) => (
            [
                (CONTENT_TYPE, HeaderValue::from_static("application/zip")),
                (CONTENT_DISPOSITION, HeaderValue::from_static("attachment; filename=\"portfolio-lab.zip\"")),
            ],
            bytes,
        )
            .into_response(),
        Ok(Err(error)) => failure(StatusCode::INTERNAL_SERVER_ERROR, "The export failed.", error.message()),
        Err(message) => failure(StatusCode::INTERNAL_SERVER_ERROR, "The export failed.", &message),
    }
}

/// `GET /download/prices.csv?result=<id>`: the input prices (`csv_text(prices, index_label="Date")`).
async fn prices_csv(
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    Query(query): Query<ResultQuery>,
) -> Response {
    let applied = match shown_result(&handle, &query) {
        Ok(applied) => applied,
        Err(mismatch) => return mismatch_page(mismatch, "No prices yet."),
    };
    match blocking(move || applied.prices.to_csv("Date")).await {
        Ok(text) => (
            [
                (CONTENT_TYPE, HeaderValue::from_static("text/csv; charset=utf-8")),
                (CONTENT_DISPOSITION, HeaderValue::from_static("attachment; filename=\"prices.csv\"")),
            ],
            text,
        )
            .into_response(),
        Err(message) => failure(StatusCode::INTERNAL_SERVER_ERROR, "The download failed.", &message),
    }
}

/// `GET /api/prices?result=<id>`: the input prices as JSON for the browser's
/// table, which shows only the visible rows. A mismatch answers with an HTML
/// message for the expander.
async fn prices_table(
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    Query(query): Query<ResultQuery>,
) -> Response {
    let message = |status: StatusCode, text: &str| {
        (status, [(CACHE_CONTROL, HeaderValue::from_static("no-store"))], Html(alert_plain(Alert::Error, text)))
            .into_response()
    };
    let applied = match shown_result(&handle, &query) {
        Ok(applied) => applied,
        Err(Mismatch::Missing) => return message(StatusCode::NOT_FOUND, "Build a portfolio first."),
        Err(Mismatch::Older) => return message(StatusCode::CONFLICT, OLDER_STUDY),
    };
    match blocking(move || frame_json(&applied.prices, "Date")).await {
        Ok(json) => (
            [
                (CONTENT_TYPE, HeaderValue::from_static("application/json")),
                (CACHE_CONTROL, HeaderValue::from_static("no-store")),
            ],
            json,
        )
            .into_response(),
        Err(text) => message(StatusCode::INTERNAL_SERVER_ERROR, &text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    const TEST_FILE_LIMIT: usize = 16;

    async fn upload_form(prices: usize, benchmark: usize) -> (StatusCode, String) {
        let app = Router::new()
            .route(
                "/",
                post(|multipart: Multipart| async {
                    match read_form(multipart, TEST_FILE_LIMIT).await {
                        Ok(posted) => Json(json!({
                            "fields": posted.fields,
                            "lengths": posted.files.iter().map(|(_, _, bytes)| bytes.len()).collect::<Vec<_>>(),
                        }))
                        .into_response(),
                        Err(message) => (StatusCode::BAD_REQUEST, message).into_response(),
                    }
                }),
            )
            .layer(DefaultBodyLimit::max(multipart_body_limit(TEST_FILE_LIMIT)));
        let mut body = String::from("--test\r\nContent-Disposition: form-data; name=\"market\"\r\n\r\nUpload CSV\r\n");
        for (field, length) in [("prices_upload", prices), ("benchmark_upload", benchmark)] {
            body.push_str(&format!(
                "--test\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{field}.csv\"\r\nContent-Type: text/csv\r\n\r\n{}\r\n",
                "x".repeat(length)
            ));
        }
        body.push_str("--test--\r\n");
        let request = axum::http::Request::builder()
            .method(Method::POST)
            .uri("/")
            .header(CONTENT_TYPE, "multipart/form-data; boundary=test")
            .body(Body::from(body))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn csv_uploads_accept_both_files_at_each_file_limit_with_form_overhead() {
        for (prices, benchmark) in [(0, 0), (TEST_FILE_LIMIT - 1, TEST_FILE_LIMIT), (TEST_FILE_LIMIT, TEST_FILE_LIMIT)]
        {
            let (status, body) = upload_form(prices, benchmark).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let posted: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(posted["lengths"], json!([prices, benchmark]));
            assert_eq!(posted["fields"], json!([["market", "Upload CSV"]]));
        }
        assert_eq!(UPLOAD_LIMIT, 200 * 1024 * 1024);
        assert_eq!(multipart_body_limit(UPLOAD_LIMIT), 2 * UPLOAD_LIMIT + 64 * 1024);
    }

    #[tokio::test]
    async fn csv_uploads_reject_either_file_one_byte_over_its_limit() {
        for (prices, benchmark, field) in
            [(TEST_FILE_LIMIT + 1, 0, "prices_upload"), (0, TEST_FILE_LIMIT + 1, "benchmark_upload")]
        {
            let (status, body) = upload_form(prices, benchmark).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert!(body.contains(&format!("{field}.csv")), "{body}");
            assert!(body.contains("per-file limit of 16 bytes"), "{body}");
        }
    }

    #[tokio::test]
    async fn oversized_requests_explain_the_per_file_limit() {
        for (prices, benchmark) in [(70_000, 0), (0, 70_000)] {
            let (status, body) = upload_form(prices, benchmark).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert!(body.contains("Each CSV file must be at most 16 bytes"), "{body}");
        }
    }

    #[test]
    fn only_local_host_names_pass() {
        for host in ["localhost", "localhost:8501", "127.0.0.1:8501", "[::1]:8501", "[::1]"] {
            assert!(is_local_host(host), "{host}");
        }
        for host in ["example.com", "evil.localhost.example:8501", "192.168.1.5:8501", "[::2]:8501"] {
            assert!(!is_local_host(host), "{host}");
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        pairs.iter().map(|(name, value)| (name.parse().unwrap(), value.parse().unwrap())).collect()
    }

    #[test]
    fn the_cross_site_policy_follows_origin_and_fetch_metadata() {
        let own = [("host", "127.0.0.1:8501")];
        let post = |extra: &[(&str, &str)]| from_this_site(&Method::POST, &headers(&[&own[..], extra].concat()));
        let get = |extra: &[(&str, &str)]| from_this_site(&Method::GET, &headers(&[&own[..], extra].concat()));
        assert!(post(&[]), "curl and tests send neither header");
        assert!(post(&[("origin", "http://127.0.0.1:8501"), ("sec-fetch-site", "same-origin")]));
        assert!(post(&[("origin", "HTTP://127.0.0.1:8501")]));
        assert!(post(&[("sec-fetch-site", "none")]));
        assert!(!post(&[("origin", "https://example.com")]));
        assert!(!post(&[("origin", "http://localhost:8501")]), "another origin of this computer");
        assert!(!post(&[("origin", "null")]));
        assert!(!post(&[("sec-fetch-site", "cross-site")]));
        assert!(!post(&[("sec-fetch-site", "same-site")]));
        assert!(!from_this_site(&Method::POST, &headers(&[("origin", "http://127.0.0.1:8501")])), "no host to compare");
        assert!(get(&[]));
        assert!(get(&[("sec-fetch-site", "cross-site"), ("sec-fetch-mode", "navigate")]), "a link from another site");
        assert!(!get(&[("sec-fetch-site", "cross-site"), ("sec-fetch-mode", "no-cors")]), "an image or script");
        assert!(!get(&[("sec-fetch-site", "same-site"), ("sec-fetch-mode", "cors")]), "a page on another port");
        assert!(get(&[("sec-fetch-site", "same-origin"), ("sec-fetch-mode", "cors")]));
    }
}
