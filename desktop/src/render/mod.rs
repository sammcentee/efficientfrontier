//! Server-rendered pages: the port of the page half of `app.py`, the Streamlit
//! app of the Python release (commit 6c6df6e).
//!
//! [`page`] chooses the view as the bottom of `app.py` does: the work card while
//! a build runs, the stale-model notice, the landing or error page, then the
//! overview or Data & method. Report functions of the library render their own
//! HTML fragments; [`html::guarded`] shows a failed fragment in place.

pub mod drawer;
pub mod format;
pub mod html;
pub mod landing;
pub mod method;
pub mod overview;
pub mod table;
pub mod widgets;

use efficient_frontier::frame::Date;

use crate::assets;
use crate::build::{Applied, Job};
use crate::session::{Session, View};
use crate::theme;
use html::{Alert, alert, esc, icon};

/// `st.set_page_config(page_title=...)`.
pub const PAGE_TITLE: &str = "Efficient Frontier Analysis for US Equities";

/// `backtest_report.HOLDINGS`: holding table columns and labels, in display order.
pub const HOLDING_LABELS: [(&str, &str); 6] = [
    ("pnl_contribution", "P&L / initial capital"),
    ("average_weight", "Average weight"),
    ("end_weight", "End weight"),
    ("selection_frequency", "Selection frequency"),
    ("max_target_weight", "Maximum target weight"),
    ("max_realized_weight", "Maximum realized weight"),
];

/// The label of a holdings column.
pub fn holding_label(name: &str) -> Option<String> {
    Some(HOLDING_LABELS.iter().find(|(key, _)| *key == name).map_or(name, |(_, label)| label).to_string())
}

/// The whole page for one session.
pub fn page(session: &Session, today: Date) -> String {
    let applied = session.applied.as_deref();
    let busy = session.job.is_some();
    let current = applied.is_some_and(Applied::is_current);
    let mut body = String::new();
    if let Some(applied) = applied.filter(|_| !busy) {
        body.push_str(&nav(session.view, current.then_some(applied.id)));
    }
    let (kind, main) = match (&session.job, applied) {
        (Some(job), _) => ("work", work(job)),
        (None, Some(_)) if !current => ("stale", stale()),
        (None, None) => landing::welcome_or_error(session, today),
        (None, Some(applied)) => match session.view {
            View::Portfolio => ("overview", overview::overview(session, applied)),
            View::DataMethod => ("method", method::data_method(session, applied)),
        },
    };
    body.push_str(&format!(r#"<main class="pl-main" id="pl-top">{main}</main>"#));
    let ready = applied.is_some() && !busy;
    document(&body, kind, kind == "overview", ready)
}

/// The HTML document around a page body.
pub fn document(body: &str, kind: &str, charts: bool, ready: bool) -> String {
    let plotly = if charts {
        format!(r#"<script src="{}"></script>"#, assets::url("plotly/plotly.min.js"))
    } else {
        String::new()
    };
    format!(
        r#"<!doctype html>
<html {html}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PAGE_TITLE}</title>
{theme}
<link rel="icon" href="{favicon}">
<link rel="stylesheet" href="{katex_css}">
<link rel="stylesheet" href="{tokens}">
<link rel="stylesheet" href="{app_css}">
<link rel="stylesheet" href="{story}">
<script defer src="{katex}"></script>
<script defer src="{auto_render}"></script>
<script defer src="{app_js}"></script>
{plotly}
</head>
<body class="pl-app" data-page="{kind}" data-ready="{ready}">
{body}
<div id="pl-drawer-root"></div>
</body>
</html>
"#,
        html = theme::html_attributes(),
        theme = theme::head_script(),
        favicon = assets::FAVICON,
        katex_css = assets::url("katex/katex.min.css"),
        tokens = assets::url("css/tokens.css"),
        app_css = assets::url("css/app.css"),
        story = assets::url("css/story.css"),
        katex = assets::url("katex/katex.min.js"),
        auto_render = assets::url("katex/auto-render.min.js"),
        app_js = assets::url("js/app.js"),
        ready = u8::from(ready),
    )
}

/// The Settings button with its shortcut, in a button style such as `pl-btn-outline`,
/// and the theme button after it.
pub fn settings_button(style: &str) -> String {
    format!(
        r#"<button type="button" class="pl-btn {style}" id="edit_setup" data-open-setup>{}<span>Settings</span><kbd>S</kbd></button>{}"#,
        icon::TUNE,
        theme::toggle_button()
    )
}

/// The sticky bar: wordmark, view switch, Settings, the theme button, and Export when `export` names a current result.
fn nav(view: View, export: Option<u64>) -> String {
    let segments: String = View::ALL
        .iter()
        .map(|option| {
            format!(
                r#"<button type="submit" name="view" value="{}" aria-pressed="{}">{}</button>"#,
                esc(option.key()),
                option == &view,
                esc(option.label())
            )
        })
        .collect();
    let export = export
        .map(|id| {
            format!(
                r#"<a class="pl-btn pl-btn-primary" id="export" href="/export?result={id}" download="portfolio-lab.zip" data-export title="One zip file. The report opens offline and prints to PDF.">{}<span>Export</span><kbd>E</kbd></a>"#,
                icon::SHARE
            )
        })
        .unwrap_or_default();
    format!(
        r#"<header class="pl-nav" id="localnav"><div class="pl-nav-inner"><span class="pl-wordmark">{PAGE_TITLE}</span><form class="pl-view-switch" method="post" action="/view" role="group" aria-label="View">{segments}</form>{}{export}</div></header>"#,
        settings_button("pl-btn-outline")
    )
}

/// `hero_work`: the build steps, the download progress and Cancel.
fn work(job: &Job) -> String {
    let progress = job.progress();
    let (hidden, width, text) = match &progress.bar {
        Some((fraction, text)) => ("", fraction * 100.0, text.as_str()),
        None => (" hidden", 0.0, ""),
    };
    let (detail_hidden, detail) = match progress.detail {
        Some(text) => ("", text),
        None => (" hidden", ""),
    };
    format!(
        r#"<section class="pl-hero" id="hero" data-poll="/api/job"><h1 class="pl-display pl-working">Building your portfolio</h1><div class="pl-alert pl-alert-info" role="status" id="pl-step"><p>{step}</p></div><div class="pl-progress" id="work_progress"{hidden}><p class="pl-progress-text">{text}</p><div class="pl-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{now:.0}"><i style="width:{width:.1}%"></i></div></div><p class="pl-work-detail" id="pl-detail"{detail_hidden}><span class="pl-spinner" aria-hidden="true"></span><span>{detail}</span></p><form method="post" action="/cancel"><button type="submit" class="pl-btn pl-btn-tertiary" id="cancel_build">Cancel</button></form></section>"#,
        step = esc(progress.step),
        text = esc(text),
        now = width,
    )
}

/// The notice for a result of a previous selection model.
fn stale() -> String {
    alert(
        Alert::Info,
        "This saved study uses a previous selection model. Your input prices are retained. \
         Rebuild for buy and hold: one initial purchase, no later rebalancing. The previous results are hidden.",
    ) + r#"<button type="button" class="pl-btn pl-btn-secondary" id="migrate_selection" data-open-setup>Set risk budget and rebuild</button>"#
}

/// A short page for a request that failed, such as a missing export.
pub fn message_page(title: &str, message: &str) -> String {
    let body = format!(
        r#"<main class="pl-main" id="pl-top"><section class="pl-hero"><h1 class="pl-display">{}</h1>{}<p><a class="pl-btn pl-btn-secondary" href="/">Back to the study</a></p></section></main>"#,
        esc(title),
        html::alert_plain(Alert::Error, message)
    );
    document(&body, "message", false, false)
}
