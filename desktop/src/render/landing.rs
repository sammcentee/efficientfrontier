//! The landing page and the error page (`welcome_or_error`).

use efficient_frontier::frame::{Cell, Date, Table};
use efficient_frontier::pyfmt;
use efficient_frontier::story::{self, SetupContext};

use super::format::day;
use super::html::{Alert, PREVIEW_HTML, alert_plain, caption, esc, guarded, latex, md, title_html};
use super::settings_button;
use super::table::{cell, table};
use super::widgets::NumberInput;
use crate::build::{BuildError, ErrorKind};
use crate::session::Session;
use crate::setup::{Market, Setup, resolve};

/// The three landing equations: (container id, TeX, legend Markdown). `@A@` is the
/// number of observations per year.
const LANDING_MATH: [(&str, &str, &str); 3] = [
    (
        "landing_estimates",
        concat!(r"\widehat\mu=A\bar r,\qquad ", r"\widehat\Sigma=A\operatorname{Cov}_{\mathrm{sample}}(r)"),
        concat!(
            r"$\boldsymbol{r}:$ vector of stock returns; $\boldsymbol{\bar r}:$ sample mean returns; ",
            r"$\boldsymbol{A=@A@}:$ observations per year. ",
            r"$\boldsymbol{\widehat\mu}:$ fitted annual returns; $\boldsymbol{\widehat\Sigma}:$ annual covariance matrix; ",
            r"$\boldsymbol{\operatorname{Cov}_{\mathrm{sample}}}:$ sample covariance.",
        ),
    ),
    (
        "landing_optimization",
        concat!(
            r"\begin{aligned}",
            r"w^*&=\underset{w}{\operatorname{arg\,max}}\;\widehat\mu^{\mathsf T}w\\[4pt]",
            r"\text{subject to}\quad&\mathbf1^{\mathsf T}w=1,\quad 0\le w_i\le h,\\",
            r"&\sqrt{w^{\mathsf T}\widehat\Sigma w}\le b",
            r"\end{aligned}",
        ),
        concat!(
            r"$\boldsymbol{w}:$ initial weight vector; $\boldsymbol{w_i}:$ fraction in stock $\boldsymbol{i}$; $\boldsymbol{w^*}:$ optimal weights. ",
            r"$\boldsymbol{b}:$ fitted annual volatility budget; $\boldsymbol{h}:$ initial holding limit; ",
            r"$\boldsymbol{\mathbf1}:$ vector of ones; $\boldsymbol{\mathsf T}:$ transpose; ",
            r"$\boldsymbol{\operatorname{arg\,max}}:$ weights that maximize fitted return.",
        ),
    ),
    (
        "landing_growth",
        concat!(r"V_t=\frac{V_{\mathrm{start}}}{1+c}", r"\sum_i w_i\frac{p_{i,t}}{p_{i,e}}\qquad(t\ge e)"),
        concat!(
            r"$\boldsymbol{V_t}:$ value at observation $\boldsymbol{t}$; $\boldsymbol{V_{\mathrm{start}}}:$ starting cash; ",
            r"$\boldsymbol{p_{i,t}}:$ adjusted price of stock $\boldsymbol{i}$ at $\boldsymbol{t}$; $\boldsymbol{e}:$ entry observation. ",
            r"$\boldsymbol{c=0.10\%}:$ fee per unit bought; $\boldsymbol{\sum_i}:$ sum across stocks.",
        ),
    ),
];

/// The landing page, or the error page after a failed build.
pub fn welcome_or_error(session: &Session, today: Date) -> (&'static str, String) {
    match &session.build_error {
        None => ("landing", welcome(session, today)),
        Some(error) => ("error", error_page(session, error)),
    }
}

/// `welcome_mathematics`: the estimates, the optimization and the replay growth.
fn welcome_mathematics(setup: &Setup, today: Date) -> String {
    let periods = pyfmt::g(resolve(setup, today).settings.periods_per_year);
    let groups: String = LANDING_MATH
        .iter()
        .map(|(id, tex, legend)| {
            format!(
                r#"<div class="pl-eq-group" id="{id}">{}<div class="pl-legend">{}</div></div>"#,
                latex(tex),
                md(&legend.replace("@A@", &periods))
            )
        })
        .collect();
    format!(r#"<div class="pl-landing-math" id="landing_math">{groups}</div>"#)
}

fn welcome(session: &Session, today: Date) -> String {
    let setup = &session.setup;
    let target = NumberInput {
        id: "preview_target_vol",
        name: "target_vol",
        label: "Annual volatility budget",
        value: setup.target_vol,
        format: ".2f",
        step: 0.1,
        min: Some(1.0),
        max: Some(100.0),
        percent: true,
        help: Some("b: the maximum fitted annual standard deviation at the initial allocation."),
    };
    let reference = if let Some(reference) = &session.qqq_reference {
        caption(&format!(
            "Nasdaq-100 proxy · QQQ {}\n\nOne year through {}. Daily sample volatility × √252. \
             This is a separate window from the portfolio fit.",
            pyfmt::format(reference.volatility, ".2%"),
            day(reference.last_return_date)
        ))
    } else if session.risk_reference_unavailable {
        caption(
            "QQQ volatility is unavailable. The 20% starting value is a manual fallback, not a current Nasdaq-100 estimate.",
        )
    } else {
        String::new()
    };
    let cap = NumberInput {
        id: "preview_cap",
        name: "cap",
        label: "Holding limit",
        value: setup.cap,
        format: "g",
        step: 1.0,
        help: Some("h: the maximum initial weight in one stock. A 10% cap requires at least ten stocks."),
        ..target
    };
    format!(
        r#"<section class="pl-landing" id="landing"><h1 class="pl-display">Efficient Frontier Analysis for US Equities<span class="pl-method-title">Constrained Mean–Variance Optimization</span></h1><div class="pl-landing-grid">{math}<form class="pl-landing-controls" id="landing_controls" method="post" action="/build" enctype="multipart/form-data" data-preview><input type="hidden" name="form" value="landing">{target}{reference}{cap}</form></div>{PREVIEW_HTML}<div class="pl-landing-actions" id="landing_actions"><button type="submit" form="landing_controls" class="pl-btn pl-btn-build" id="build" data-build><span>Build portfolio</span><kbd data-mod-key>Ctrl + Enter</kbd></button>{settings}</div></section>"#,
        math = welcome_mathematics(setup, today),
        target = target.html(),
        cap = cap.html(),
        settings = settings_button("pl-btn-tertiary"),
    )
}

/// The applied setup in the form of `story.context_setup_html`.
pub fn setup_context(setup: &Setup) -> SetupContext {
    SetupContext {
        market: setup.market.label().to_string(),
        prices_file: setup.prices_file.as_ref().map(|file| file.name.clone()),
        history: setup.history.label().to_string(),
        start: setup.start,
        end: setup.end,
    }
}

/// `COVERAGE_COLUMNS`: the coverage table without the industry.
pub fn coverage_label(name: &str) -> Option<String> {
    let label = match name {
        "industry" => return None,
        "symbol" => "Symbol",
        "name" => "Company",
        "status" => "Status",
        "reason" => "Reason",
        "observations" => "Price rows",
        "first_date" => "First date",
        "last_date" => "Last date",
        other => other,
    };
    Some(label.to_string())
}

/// The member coverage of a universe download.
pub fn coverage_table(coverage: &Table) -> String {
    table(coverage, None, &coverage_label, &|_, value: &Cell| cell(value)).html()
}

/// The hero of a failed build: what failed and the useful next action.
fn error_page(session: &Session, error: &BuildError) -> String {
    let setup = &session.setup;
    let context = guarded(|| story::context_setup_html(&setup_context(setup), None));
    let mut hero = format!(
        r#"<div class="pl-context-line">{context}{}</div><h1 class="pl-display">{}</h1>{}"#,
        settings_button("pl-btn-outline"),
        esc(error.kind.headline()),
        alert_plain(Alert::Error, &error.message)
    );
    if error.kind == ErrorKind::Infeasible {
        let cap = setup.cap;
        let need = if cap > 0.0 {
            let noun = if matches!(setup.market, Market::Demo | Market::UploadCsv) { "assets" } else { "stocks" };
            format!("A {}% limit needs at least {} {noun}. ", pyfmt::g(cap), (100.0 / cap - 1e-9).ceil())
        } else {
            String::new()
        };
        hero.push_str(&format!(
            r#"<p class="pl-lead pl-center">{need}Review the holding limit and annual volatility budget in Change the study. No constraint is relaxed automatically.</p>"#
        ));
    }
    hero.push_str(&try_buttons(error.kind.needs_change()));
    let mut html = format!(r#"<section class="pl-hero" id="hero">{hero}</section>"#);
    if let Some(coverage) = &session.coverage {
        html.push_str(&format!(
            r#"<section class="pl-tile" id="tile_missing">{}{}</section>"#,
            title_html("Which stocks were missing", ""),
            coverage_table(coverage)
        ));
    }
    html
}

/// `try_buttons`: "Change the study" when a retry would fail the same way, else "Try again".
fn try_buttons(fix: bool) -> String {
    let button = if fix {
        format!(
            r#"<button type="button" class="pl-btn pl-btn-primary pl-btn-large" id="fix_setup" data-open-setup>{}<span>Change the study</span></button>"#,
            super::html::icon::TUNE
        )
    } else {
        r#"<form method="post" action="/build" enctype="multipart/form-data"><input type="hidden" name="form" value="retry"><button type="submit" class="pl-btn pl-btn-build" id="build" data-build><span>Try again</span><kbd data-mod-key>Ctrl + Enter</kbd></button></form>"#.to_string()
    };
    format!(r#"<div class="pl-actions">{button}</div>"#)
}
