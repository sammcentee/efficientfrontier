//! The `app` protocol: the window's requests go to the app's routes inside this program.
//!
//! The web view gives each request of the window to the protocol handler, which
//! runs the routes of [`crate::routes`] without a network port. A browser and a
//! web server did three more things between them, and the bridge does them here:
//!
//! - It keeps the session cookie, because web views do not keep the cookies of
//!   a custom protocol on every platform.
//! - It answers a redirected page load with a page that loads the new address,
//!   because web views do not follow redirects from a custom protocol on every
//!   platform. A script's request follows the redirect here, as `fetch` does.
//! - It gives a file download to the app, which asks where to save the file.

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::header::{
    ACCEPT, CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_TYPE, COOKIE, HOST, LOCATION, ORIGIN, SET_COOKIE,
};
use axum::http::{HeaderMap, HeaderValue, Request, Response, StatusCode, Uri};
use efficient_frontier::pyfmt::escape;
use tauri::Url;
use tokio::sync::Mutex;
use tower::ServiceExt;

use crate::session::COOKIE_NAME;

/// The protocol of the window's pages.
pub const SCHEME: &str = "app";
/// The most redirects that a script's request follows, as a guard against a loop.
const MAX_REDIRECTS: usize = 5;

/// The address of the start page. WebView2 on Windows serves a custom protocol
/// as `http://<scheme>.localhost`; the other systems use `<scheme>://localhost`.
pub fn start_url() -> Url {
    let url = if cfg!(windows) { "http://app.localhost/" } else { "app://localhost/" };
    Url::parse(url).expect("a valid URL")
}

/// True for the address of one of the app's pages.
pub fn is_app_url(url: &Url) -> bool {
    let start = start_url();
    url.scheme() == start.scheme() && url.host_str() == start.host_str() && url.port() == start.port()
}

/// The `Origin` header of the app's pages.
fn is_own_origin(origin: &str) -> bool {
    let start = start_url();
    origin.eq_ignore_ascii_case(&format!("{}://{}", start.scheme(), start.host_str().unwrap_or_default()))
}

/// A file that a route offers for download.
pub struct Download {
    /// The file name that the route suggests.
    pub name: String,
    pub bytes: Vec<u8>,
}

/// The bridge's result for one request.
pub enum Forwarded {
    /// The response for the web view.
    Page(Response<Vec<u8>>),
    /// A file to save. The web view then gets [`no_content`].
    Download(Download),
}

/// The routes, and the session cookie that a browser kept.
pub struct Bridge {
    router: Router,
    /// The `Cookie` header of the window's session, after the first page created the session.
    cookie: Mutex<Option<HeaderValue>>,
}

impl Bridge {
    pub fn new(router: Router) -> Self {
        Self { router, cookie: Mutex::new(None) }
    }

    /// Answer one request of the web view.
    pub async fn forward(&self, mut request: Request<Vec<u8>>) -> Forwarded {
        let navigation = is_navigation(request.headers());
        prepare(&mut request);
        let mut response = self.send(request).await;
        for _ in 0..MAX_REDIRECTS {
            let Some(location) = redirect_target(&response) else { break };
            if navigation {
                return Forwarded::Page(refresh_page(&location));
            }
            // After 301, 302 and 303, `fetch` asks for the new address with GET.
            let target = location.split('#').next().unwrap_or("/");
            let mut next = Request::new(Vec::new());
            *next.uri_mut() = target.parse().unwrap_or_else(|_| Uri::from_static("/"));
            prepare(&mut next);
            response = self.send(next).await;
        }
        match attachment_name(&response) {
            Some(name) => Forwarded::Download(Download { name, bytes: response.into_body() }),
            None => Forwarded::Page(response),
        }
    }

    /// Run a request through the routes with the window's session cookie.
    async fn send(&self, mut request: Request<Vec<u8>>) -> Response<Vec<u8>> {
        let mut stored = self.cookie.lock().await;
        request.headers_mut().remove(COOKIE);
        match stored.clone() {
            Some(cookie) => {
                // With a session, requests run side by side.
                drop(stored);
                request.headers_mut().insert(COOKIE, cookie);
                let (response, renewed) = self.call(request).await;
                if renewed.is_some() {
                    *self.cookie.lock().await = renewed;
                }
                response
            }
            None => {
                // The first page creates the session. Other requests wait for it,
                // so that the window has one session.
                let (response, created) = self.call(request).await;
                if created.is_some() {
                    *stored = created;
                }
                response
            }
        }
    }

    /// Run a request through the routes. Also returns the `Cookie` header of a new session.
    async fn call(&self, request: Request<Vec<u8>>) -> (Response<Vec<u8>>, Option<HeaderValue>) {
        let Ok(response) = self.router.clone().oneshot(request.map(Body::from)).await;
        let (mut parts, body) = response.into_parts();
        let cookie = session_cookie(&parts.headers);
        parts.headers.remove(SET_COOKIE);
        match to_bytes(body, usize::MAX).await {
            Ok(bytes) => (Response::from_parts(parts, bytes.to_vec()), cookie),
            Err(error) => (text(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string()), cookie),
        }
    }
}

/// Make a request of the web view look like a browser's request to the app's server.
fn prepare(request: &mut Request<Vec<u8>>) {
    // The routes need only the path and the query.
    let target = request.uri().path_and_query().map_or("/", |target| target.as_str()).to_string();
    *request.uri_mut() = target.parse().unwrap_or_else(|_| Uri::from_static("/"));
    let headers = request.headers_mut();
    headers.insert(HOST, HeaderValue::from_static("localhost"));
    // The routes accept a post only from their own origin. The app's pages get
    // that origin; any other origin stays, so the routes refuse it.
    if headers.get(ORIGIN).and_then(|value| value.to_str().ok()).is_some_and(is_own_origin) {
        headers.insert(ORIGIN, HeaderValue::from_static("http://localhost"));
    }
}

/// True for a page load (a link, a form post or a reload), false for a script's request.
fn is_navigation(headers: &HeaderMap) -> bool {
    let value = |name: &str| headers.get(name).and_then(|value| value.to_str().ok());
    match value("sec-fetch-mode") {
        Some(mode) => mode == "navigate",
        // Without fetch metadata, only a page load asks for HTML first.
        None => value(ACCEPT.as_str()).is_some_and(|accept| accept.starts_with("text/html")),
    }
}

/// The `Cookie` header for the session that a response creates.
fn session_cookie(headers: &HeaderMap) -> Option<HeaderValue> {
    headers
        .get_all(SET_COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .filter_map(|value| value.split(';').next())
        .find(|pair| pair.trim().split_once('=').is_some_and(|(name, _)| name == COOKIE_NAME))
        .and_then(|pair| HeaderValue::from_str(pair.trim()).ok())
}

/// The address of a redirect after which a browser loads the new address with GET.
fn redirect_target(response: &Response<Vec<u8>>) -> Option<String> {
    if !matches!(response.status().as_u16(), 301 | 302 | 303) {
        return None;
    }
    response.headers().get(LOCATION)?.to_str().ok().map(str::to_string)
}

/// The file name of a download (`Content-Disposition: attachment; filename="…"`).
fn attachment_name(response: &Response<Vec<u8>>) -> Option<String> {
    if !response.status().is_success() {
        return None;
    }
    let value = response.headers().get(CONTENT_DISPOSITION)?.to_str().ok()?;
    let parameters = value.trim().strip_prefix("attachment")?;
    let name = parameters.split(';').find_map(|part| part.trim().strip_prefix("filename="))?;
    Some(name.trim_matches('"').to_string()).filter(|name| !name.is_empty())
}

/// A page that loads `location`. The canvas color avoids a white flash between the two pages.
fn refresh_page(location: &str) -> Response<Vec<u8>> {
    let html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"0; url={}\"><title>Portfolio Lab</title><style>html{{background:#f5f5f7}}</style></head></html>",
        escape(location)
    );
    let mut response = Response::new(html.into_bytes());
    let headers = response.headers_mut();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

/// The response after a download: the page stays as it is.
pub fn no_content() -> Response<Vec<u8>> {
    let mut response = Response::new(Vec::new());
    *response.status_mut() = StatusCode::NO_CONTENT;
    response
}

fn text(status: StatusCode, message: &str) -> Response<Vec<u8>> {
    let mut response = Response::new(message.as_bytes().to_vec());
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    response
}

/// A request as the web view gives it: an absolute address on the app's protocol.
#[cfg(test)]
pub fn webview_request(
    method: axum::http::Method,
    path: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> Request<Vec<u8>> {
    let mut request = Request::builder().method(method).uri(format!("{SCHEME}://localhost{path}"));
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    request.body(body).expect("a valid request")
}

#[cfg(test)]
mod tests {
    use axum::http::Method;
    use axum::routing::{get, post};

    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        pairs.iter().map(|(name, value)| (name.parse().unwrap(), value.parse().unwrap())).collect()
    }

    #[test]
    fn only_the_app_pages_are_app_addresses() {
        assert!(is_app_url(&start_url()));
        assert!(is_app_url(&start_url().join("/assets/js/app.js?v=1#top").unwrap()));
        for url in
            ["https://finance.yahoo.com/quote/AAPL", "http://localhost/", "file:///tmp/prices.csv", "about:blank"]
        {
            assert!(!is_app_url(&Url::parse(url).unwrap()), "{url}");
        }
        if cfg!(windows) {
            assert!(!is_app_url(&Url::parse("app://localhost/").unwrap()));
        } else {
            assert!(!is_app_url(&Url::parse("http://app.localhost/").unwrap()), "a real address on this computer");
        }
    }

    #[test]
    fn requests_get_the_servers_host_and_origin() {
        let own = if cfg!(windows) { "http://app.localhost" } else { "app://localhost" };
        let mut request = webview_request(Method::POST, "/view?x=1", &[("origin", own)], Vec::new());
        prepare(&mut request);
        assert_eq!(request.uri(), "/view?x=1");
        assert_eq!(request.headers()[HOST], "localhost");
        assert_eq!(request.headers()[ORIGIN], "http://localhost");
        let mut foreign = webview_request(Method::POST, "/build", &[("origin", "https://example.com")], Vec::new());
        prepare(&mut foreign);
        assert_eq!(foreign.headers()[ORIGIN], "https://example.com", "the routes refuse it");
    }

    #[test]
    fn page_loads_are_told_from_script_requests() {
        assert!(is_navigation(&headers(&[("sec-fetch-mode", "navigate")])));
        assert!(!is_navigation(&headers(&[("sec-fetch-mode", "cors"), ("accept", "text/html")])));
        assert!(is_navigation(&headers(&[("accept", "text/html,application/xhtml+xml,*/*;q=0.8")])));
        assert!(!is_navigation(&headers(&[("accept", "*/*")])));
        assert!(!is_navigation(&HeaderMap::new()));
    }

    #[test]
    fn downloads_are_found_by_their_attachment_name() {
        let response = |status: u16, disposition: &str| {
            let mut response = Response::new(b"data".to_vec());
            *response.status_mut() = StatusCode::from_u16(status).unwrap();
            response.headers_mut().insert(CONTENT_DISPOSITION, disposition.parse().unwrap());
            response
        };
        assert_eq!(
            attachment_name(&response(200, "attachment; filename=\"portfolio-lab.zip\"")).unwrap(),
            "portfolio-lab.zip"
        );
        assert_eq!(attachment_name(&response(200, "attachment; filename=prices.csv")).unwrap(), "prices.csv");
        assert!(attachment_name(&response(200, "inline; filename=\"page.html\"")).is_none());
        assert!(attachment_name(&response(409, "attachment; filename=\"portfolio-lab.zip\"")).is_none());
        assert!(attachment_name(&Response::new(Vec::new())).is_none());
    }

    #[test]
    fn a_redirect_page_loads_the_escaped_address() {
        let page = refresh_page("/#pl-market");
        let html = String::from_utf8(page.body().clone()).unwrap();
        assert!(html.contains("content=\"0; url=/#pl-market\""), "{html}");
        assert!(String::from_utf8(refresh_page("/\"><x").into_body()).unwrap().contains("url=/&quot;&gt;&lt;x"));
    }

    /// Routes that act as the app's: the first page sets the session cookie, a
    /// post redirects, and a download is an attachment.
    fn routes() -> Router {
        let page = |headers: HeaderMap| async move {
            let cookie = headers.get(COOKIE).map(|value| value.to_str().unwrap().to_string());
            let mut response = Response::new(Body::from(format!("cookie={}", cookie.as_deref().unwrap_or("none"))));
            if cookie.is_none() {
                response.headers_mut().insert(SET_COOKIE, "pl_session=abc; Path=/; HttpOnly".parse().unwrap());
            }
            response
        };
        Router::new()
            .route("/", get(page))
            .route("/view", post(|| async { axum::response::Redirect::to("/#pl-market") }))
            .route(
                "/export",
                get(|| async { ([(CONTENT_DISPOSITION, "attachment; filename=\"portfolio-lab.zip\"")], "zip bytes") }),
            )
    }

    fn page(forwarded: Forwarded) -> Response<Vec<u8>> {
        match forwarded {
            Forwarded::Page(response) => response,
            Forwarded::Download(download) => panic!("a download of {}", download.name),
        }
    }

    fn body(response: &Response<Vec<u8>>) -> String {
        String::from_utf8(response.body().clone()).unwrap()
    }

    #[tokio::test]
    async fn the_bridge_keeps_the_session_cookie() {
        let bridge = Bridge::new(routes());
        let first = page(bridge.forward(webview_request(Method::GET, "/", &[], Vec::new())).await);
        assert_eq!(body(&first), "cookie=none");
        assert!(first.headers().get(SET_COOKIE).is_none(), "the bridge keeps the cookie, not the web view");
        let stale = [("cookie", "pl_session=old")];
        let second = page(bridge.forward(webview_request(Method::GET, "/", &stale, Vec::new())).await);
        assert_eq!(body(&second), "cookie=pl_session=abc");
    }

    #[tokio::test]
    async fn a_redirected_page_load_becomes_a_page_that_loads_the_new_address() {
        let bridge = Bridge::new(routes());
        let navigate = [("sec-fetch-mode", "navigate")];
        let response = page(bridge.forward(webview_request(Method::POST, "/view", &navigate, Vec::new())).await);
        assert_eq!(response.status(), StatusCode::OK);
        assert!(body(&response).contains("url=/#pl-market"));
    }

    #[tokio::test]
    async fn a_script_follows_a_redirect() {
        let bridge = Bridge::new(routes());
        let fetch = [("sec-fetch-mode", "cors")];
        let response = page(bridge.forward(webview_request(Method::POST, "/view", &fetch, Vec::new())).await);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body(&response), "cookie=none");
    }

    #[tokio::test]
    async fn an_attachment_is_a_download() {
        let bridge = Bridge::new(routes());
        match bridge.forward(webview_request(Method::GET, "/export", &[], Vec::new())).await {
            Forwarded::Download(download) => {
                assert_eq!(download.name, "portfolio-lab.zip");
                assert_eq!(download.bytes, b"zip bytes");
            }
            Forwarded::Page(response) => panic!("a page with status {}", response.status()),
        }
        assert_eq!(no_content().status(), StatusCode::NO_CONTENT);
    }
}
