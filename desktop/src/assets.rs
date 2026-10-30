//! Static files compiled into the binary: CSS, the browser script, plotly.js and KaTeX.
//!
//! Every asset URL carries a digest of its content, so browsers can keep a file
//! for a year and still fetch a new version after an update. No file comes from
//! a network service.

use std::collections::HashMap;
use std::sync::LazyLock;

use axum::extract::Path;
use axum::http::StatusCode;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use efficient_frontier::{plot, style};
use sha2::{Digest, Sha256};

use crate::setup::hex;

/// One static file.
pub struct Asset {
    pub path: &'static str,
    pub content_type: &'static str,
    pub body: &'static [u8],
}

const CSS: &str = "text/css; charset=utf-8";
const JS: &str = "text/javascript; charset=utf-8";
const WOFF2: &str = "font/woff2";

macro_rules! file {
    ($path:literal, $type:expr) => {
        Asset { path: $path, content_type: $type, body: include_bytes!(concat!("../../assets/", $path)) }
    };
}

/// Every asset. plotly.js and the shared CSS come from the library, which embeds them already.
pub static ASSETS: &[Asset] = &[
    Asset { path: "css/tokens.css", content_type: CSS, body: style::TOKENS_CSS.as_bytes() },
    Asset { path: "css/story.css", content_type: CSS, body: style::STORY_CSS.as_bytes() },
    file!("css/app.css", CSS),
    file!("js/app.js", JS),
    Asset { path: "plotly/plotly.min.js", content_type: JS, body: plot::PLOTLY_JS.as_bytes() },
    file!("katex/katex.min.js", JS),
    file!("katex/auto-render.min.js", JS),
    file!("katex/katex.min.css", CSS),
    file!("katex/fonts/KaTeX_AMS-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Caligraphic-Bold.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Caligraphic-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Fraktur-Bold.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Fraktur-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Main-Bold.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Main-BoldItalic.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Main-Italic.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Main-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Math-BoldItalic.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Math-Italic.woff2", WOFF2),
    file!("katex/fonts/KaTeX_SansSerif-Bold.woff2", WOFF2),
    file!("katex/fonts/KaTeX_SansSerif-Italic.woff2", WOFF2),
    file!("katex/fonts/KaTeX_SansSerif-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Script-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Size1-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Size2-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Size3-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Size4-Regular.woff2", WOFF2),
    file!("katex/fonts/KaTeX_Typewriter-Regular.woff2", WOFF2),
];

/// The page icon (`:material/donut_large:`), drawn as a ring.
pub const FAVICON: &str = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='8' fill='none' stroke='%231d1d1f' stroke-width='4' stroke-dasharray='40 10.3' transform='rotate(-80 12 12)'/%3E%3C/svg%3E";

static VERSIONS: LazyLock<HashMap<&'static str, String>> =
    LazyLock::new(|| ASSETS.iter().map(|asset| (asset.path, hex(&Sha256::digest(asset.body)[..6]))).collect());

/// The versioned URL of an asset.
pub fn url(path: &str) -> String {
    let version = VERSIONS.get(path).unwrap_or_else(|| panic!("unknown asset {path}"));
    format!("/assets/{path}?v={version}")
}

/// `GET /assets/{*path}`.
pub async fn serve(Path(path): Path<String>) -> Response {
    match ASSETS.iter().find(|asset| asset.path == path) {
        Some(asset) => {
            ([(CONTENT_TYPE, asset.content_type), (CACHE_CONTROL, "public, max-age=31536000, immutable")], asset.body)
                .into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_asset_has_a_versioned_url_and_content() {
        for asset in ASSETS {
            assert!(!asset.body.is_empty(), "{}", asset.path);
            assert!(url(asset.path).starts_with(&format!("/assets/{}?v=", asset.path)));
        }
        let css = std::str::from_utf8(ASSETS.iter().find(|a| a.path == "katex/katex.min.css").unwrap().body).unwrap();
        for font in css.split("url(fonts/").skip(1).filter_map(|rest| rest.split(')').next()) {
            if font.ends_with(".woff2") {
                assert!(ASSETS.iter().any(|asset| asset.path == format!("katex/fonts/{font}")), "{font}");
            }
        }
    }
}
