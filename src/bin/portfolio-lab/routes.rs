//! HTTP routes: pages, form posts, JSON for the browser script, downloads and assets.
//!
//! Form posts answer with "303 See Other" to `/`, so the browser shows the new
//! state with a fresh page, as a Streamlit rerun does.

use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Extension, Form, Multipart, Request, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_TYPE, HOST, SET_COOKIE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Json, Redirect, Response};
use axum::routing::{get, post};
use efficient_frontier::selection_report::selection_report_zip;
use serde::Deserialize;
use serde_json::json;

use crate::assets;
use crate::build;
use crate::guard::guard;
use crate::render::{self, drawer::drawer, format::number, table::frame};
use crate::session::{Session, SessionRef, View, cookie_id, lock, set_cookie};
use crate::setup::{Draft, Setup, Upload, differs, read_draft, resolve};
use crate::state::AppState;

/// The largest accepted upload, Streamlit's default `server.maxUploadSize` of 200 MB.
pub const UPLOAD_LIMIT: usize = 200 * 1024 * 1024;

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
        .route("/export", get(export))
        .route("/download/prices.csv", get(prices_csv))
        .route("/fragment/prices", get(prices_fragment))
        .layer(DefaultBodyLimit::max(UPLOAD_LIMIT))
        .route_layer(middleware::from_fn_with_state(state.clone(), with_session));
    Router::new()
        .route("/assets/{*path}", get(assets::serve))
        .merge(app)
        .layer(middleware::from_fn(local_host))
        .with_state(state)
}

/// Find the browser's session by cookie, or create one with `init_state`'s defaults.
async fn with_session(State(state): State<Arc<AppState>>, mut request: Request, next: Next) -> Response {
    let existing = cookie_id(request.headers()).and_then(|id| state.sessions.get(&id));
    let (handle, new_id) = match existing {
        Some(handle) => (handle, None),
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

/// The fields and files of a multipart form. A file input without a file is ignored.
async fn read_form(mut multipart: Multipart) -> Result<Draft, String> {
    let mut draft = Draft::default();
    while let Some(field) = multipart.next_field().await.map_err(|error| error.body_text())? {
        let name = field.name().unwrap_or_default().to_string();
        match field.file_name().map(str::to_string) {
            Some(file_name) => {
                let bytes = field.bytes().await.map_err(|error| error.body_text())?;
                if !file_name.is_empty() {
                    draft.uploads.insert(name, Upload::new(file_name, bytes.to_vec()));
                }
            }
            None => {
                let text = field.text().await.map_err(|error| error.body_text())?;
                draft.fields.insert(name, text);
            }
        }
    }
    Ok(draft)
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
    let draft = match read_form(multipart).await {
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

/// `POST /cancel`: stop waiting for the running build; its result is discarded.
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
async fn draft_differs(
    State(state): State<Arc<AppState>>,
    Extension(CurrentSession(handle)): Extension<CurrentSession>,
    multipart: Multipart,
) -> Response {
    let draft = match read_form(multipart).await {
        Ok(draft) => draft,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };
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

/// `GET /export`: the report zip of the applied study, computed only when asked for.
async fn export(Extension(CurrentSession(handle)): Extension<CurrentSession>) -> Response {
    let applied = lock(&handle).applied.clone().filter(|applied| applied.is_current());
    let Some(applied) = applied else {
        return failure(StatusCode::NOT_FOUND, "Nothing to export.", "Build a portfolio first.");
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

/// `GET /download/prices.csv`: the input prices (`csv_text(prices, index_label="Date")`).
async fn prices_csv(Extension(CurrentSession(handle)): Extension<CurrentSession>) -> Response {
    let Some(applied) = lock(&handle).applied.clone() else {
        return failure(StatusCode::NOT_FOUND, "No prices yet.", "Build a portfolio first.");
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

/// `GET /fragment/prices`: the input prices table, loaded when its expander opens.
async fn prices_fragment(Extension(CurrentSession(handle)): Extension<CurrentSession>) -> Response {
    let Some(applied) = lock(&handle).applied.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match blocking(move || frame(&applied.prices, "Date", &number).html()).await {
        Ok(html) => page(StatusCode::OK, html),
        Err(message) => page(StatusCode::INTERNAL_SERVER_ERROR, render::html::failed(&message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_local_host_names_pass() {
        for host in ["localhost", "localhost:8501", "127.0.0.1:8501", "[::1]:8501", "[::1]"] {
            assert!(is_local_host(host), "{host}");
        }
        for host in ["example.com", "evil.localhost.example:8501", "192.168.1.5:8501", "[::2]:8501"] {
            assert!(!is_local_host(host), "{host}");
        }
    }
}
