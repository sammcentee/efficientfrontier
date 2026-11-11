//! Small HTML pieces in place of Streamlit elements.
//!
//! Text that the app writes with `st.markdown`, `st.write` or `st.caption` goes
//! through [`md`], a Markdown subset: paragraphs, `**bold**`, `### ` headings,
//! numbered lists and inline `$...$` mathematics. KaTeX renders the mathematics
//! in the browser. Data text, such as error messages and notes, is escaped only.

use efficient_frontier::plot::{Figure, script_json};
use efficient_frontier::pyfmt;
use efficient_frontier::style::plotly_config;
use serde_json::{Value, json};

use crate::guard::guard;

/// `html.escape(text)`.
pub fn esc(text: &str) -> String {
    pyfmt::escape(text)
}

/// `title_html(title, sub)`.
pub fn title_html(title: &str, sub: &str) -> String {
    let mut html = format!(r#"<h3 class="pl-title">{}</h3>"#, esc(title));
    if !sub.is_empty() {
        html.push_str(&format!(r#"<p class="pl-sub">{}</p>"#, esc(sub)));
    }
    html
}

/// `stat_strip(stats)`: (number, label HTML, note) per statistic.
pub fn stat_strip(stats: &[(String, &str, &str)]) -> String {
    let items: String = stats
        .iter()
        .map(|(number, label, sub)| {
            format!(
                r#"<div class="pl-stat"><span class="pl-num">{}</span><span class="pl-lbl">{label}</span><small>{}</small></div>"#,
                esc(number),
                esc(sub)
            )
        })
        .collect();
    format!(r#"<div class="pl-stats">{items}</div>"#)
}

/// Bold markers to `<strong>`, in escaped text.
fn bold(text: &str) -> String {
    let parts: Vec<&str> = text.split("**").collect();
    if parts.len().is_multiple_of(2) {
        return text.to_string();
    }
    parts
        .iter()
        .enumerate()
        .map(|(position, part)| if position % 2 == 1 { format!("<strong>{part}</strong>") } else { part.to_string() })
        .collect()
}

/// Inline Markdown: escaped text with bold markers, and `$...$` kept for KaTeX.
/// Returns the HTML and whether it contains mathematics.
fn inline(text: &str) -> (String, bool) {
    let parts: Vec<&str> = text.split('$').collect();
    if parts.len() < 3 || parts.len().is_multiple_of(2) {
        return (bold(&esc(text)), false);
    }
    let html = parts
        .iter()
        .enumerate()
        .map(|(position, part)| if position % 2 == 1 { format!("${}$", esc(part)) } else { bold(&esc(part)) })
        .collect();
    (html, true)
}

/// The class attribute for a block, with the KaTeX marker when it has mathematics.
fn class(base: &str, math: bool) -> String {
    match (base.is_empty(), math) {
        (true, false) => String::new(),
        (true, true) => r#" class="pl-inline-math""#.to_string(),
        (false, false) => format!(r#" class="{base}""#),
        (false, true) => format!(r#" class="{base} pl-inline-math""#),
    }
}

/// The ordered-list item text of a line such as `1. Text`.
fn list_item(line: &str) -> Option<&str> {
    let (number, rest) = line.split_once(". ")?;
    (!number.is_empty() && number.chars().all(|c| c.is_ascii_digit())).then_some(rest)
}

/// The Markdown subset of `st.markdown`, as block HTML.
pub fn md(text: &str) -> String {
    let mut html = String::new();
    for block in text.split("\n\n").filter(|block| !block.trim().is_empty()) {
        let lines: Vec<&str> = block.lines().collect();
        if lines.iter().all(|line| list_item(line).is_some()) {
            let items: Vec<(String, bool)> = lines.iter().filter_map(|line| list_item(line)).map(inline).collect();
            let math = items.iter().any(|(_, math)| *math);
            html.push_str(&format!("<ol{}>", class("", math)));
            for (item, _) in items {
                html.push_str(&format!("<li>{item}</li>"));
            }
            html.push_str("</ol>");
        } else if let Some(heading) = block.strip_prefix("### ") {
            let (text, math) = inline(heading);
            html.push_str(&format!("<h3{}>{text}</h3>", class("pl-md-heading", math)));
        } else {
            let (text, math) = inline(&lines.join(" "));
            html.push_str(&format!("<p{}>{text}</p>", class("", math)));
        }
    }
    html
}

/// `st.caption(text)`.
pub fn caption(text: &str) -> String {
    format!(r#"<div class="pl-caption">{}</div>"#, md(text))
}

/// Escaped text in paragraphs, for data such as notes and messages.
pub fn plain(text: &str) -> String {
    text.split("\n\n").map(|paragraph| format!("<p>{}</p>", esc(paragraph))).collect()
}

/// A plain-text caption, for data text.
pub fn plain_caption(text: &str) -> String {
    format!(r#"<div class="pl-caption">{}</div>"#, plain(text))
}

/// The kinds of `st.info`, `st.warning` and `st.error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alert {
    Info,
    Warning,
    Error,
}

/// An alert box around block HTML.
pub fn alert_html(kind: Alert, body: &str) -> String {
    let (name, role) = match kind {
        Alert::Info => ("info", "status"),
        Alert::Warning => ("warning", "status"),
        Alert::Error => ("error", "alert"),
    };
    format!(r#"<div class="pl-alert pl-alert-{name}" role="{role}">{body}</div>"#)
}

/// An alert with app text in Markdown.
pub fn alert(kind: Alert, text: &str) -> String {
    alert_html(kind, &md(text))
}

/// An alert with data text, escaped only.
pub fn alert_plain(kind: Alert, text: &str) -> String {
    alert_html(kind, &plain(text))
}

/// `st.latex(tex)`: KaTeX renders the escaped source in display mode.
pub fn latex(tex: &str) -> String {
    format!(r#"<div class="pl-latex">{}</div>"#, esc(tex))
}

/// `st.expander(label)`: a closed disclosure row.
pub fn expander(label_html: &str, body: &str) -> String {
    format!(
        r#"<details class="pl-expander"><summary><span class="pl-expander-label">{label_html}</span><span class="pl-expander-icon" aria-hidden="true">{}</span></summary><div class="pl-expander-body">{body}</div></details>"#,
        icon::CHEVRON
    )
}

/// `st.plotly_chart(figure, theme=None, config=PLOTLY_CONFIG)`.
///
/// The figure goes into the page as data, in the light colors of the reports.
/// `app.js` draws it in the colors of the page's theme, and again after a change of theme.
pub fn chart(figure: &Figure, id: &str) -> String {
    let height = match figure.layout.get("height").and_then(Value::as_f64) {
        Some(height) => format!("height:{}px; width:100%;", pyfmt::g(height)),
        None => "height:100%; width:100%;".to_string(),
    };
    let source = script_json(&json!({
        "data": figure.data, "layout": figure.full_layout(), "config": plotly_config(),
    }));
    format!(
        r#"<div class="pl-chart"><div id="{id}" class="plotly-graph-div" style="{height}"></div><script type="application/json" data-chart="{id}">{source}</script></div>"#
    )
}

/// The message in place of a section that failed, as Streamlit shows an exception.
pub fn failed(message: &str) -> String {
    alert_html(Alert::Error, &format!("<p><strong>This section could not be shown.</strong> {}</p>", esc(message)))
}

/// HTML from a report function, or [`failed`] if it panics.
pub fn guarded(work: impl FnOnce() -> String) -> String {
    guard(work).unwrap_or_else(|message| failed(&message))
}

/// `KEY_LEGEND`.
pub const KEY_LEGEND: &str = r#"<span class="pl-keys">Keys: <kbd>1</kbd> overview · <kbd>2</kbd> test · <kbd>4</kbd> data &amp; method · <kbd>S</kbd> setup · <kbd>E</kbd> export</span>"#;

/// `PREVIEW_HTML`.
pub const PREVIEW_HTML: &str = concat!(
    r#"<ol class="pl-preview"><li><b>Set two limits.</b><span>Set annual volatility and the initial holding cap.</span></li>"#,
    r#"<li><b>Buy once and hold.</b><span>Fit on the first 70% of returns; hold through the final 30%.</span></li>"#,
    r#"<li><b>Inspect the latest weights.</b><span>Fit all history separately. These weights do not produce the replay.</span></li></ol>"#,
);

/// Inline icons, drawn for this app.
pub mod icon {
    pub const TUNE: &str = r#"<svg class="pl-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/></svg>"#;
    pub const SHARE: &str = r#"<svg class="pl-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12M8 7l4-4 4 4M7 10H6a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1h-1"/></svg>"#;
    pub const HELP: &str = r#"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 1 1 3.4 2.4c-.6.3-1 .8-1 1.5v.6" stroke-linecap="round"/><circle cx="12" cy="17" r="1" fill="currentColor" stroke="none"/></svg>"#;
    pub const CHEVRON: &str = r#"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>"#;
    pub const CLOSE: &str = r#"<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>"#;
}

/// A help icon with a tooltip, as the `help` argument of a widget.
pub fn help(text: &str) -> String {
    format!(
        r#"<span class="pl-help" tabindex="0" role="img" aria-label="{0}" data-tip="{0}">{1}</span>"#,
        esc(text),
        icon::HELP
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_subset_keeps_math_and_escapes_text() {
        assert_eq!(md("**Bold** & plain"), "<p><strong>Bold</strong> &amp; plain</p>");
        assert_eq!(
            md(r"$\boldsymbol{A=252}:$ observations per year; **pp:** points"),
            r#"<p class="pl-inline-math">$\boldsymbol{A=252}:$ observations per year; <strong>pp:</strong> points</p>"#
        );
        assert_eq!(
            md("1. **One.** First\n2. **Two.** Second"),
            "<ol><li><strong>One.</strong> First</li><li><strong>Two.</strong> Second</li></ol>"
        );
        assert_eq!(md("### Heading"), r#"<h3 class="pl-md-heading">Heading</h3>"#);
        assert_eq!(md("First.\n\nSecond <b>"), "<p>First.</p><p>Second &lt;b&gt;</p>");
        assert_eq!(md("Costs $5 only"), "<p>Costs $5 only</p>", "a single dollar sign is text");
    }

    #[test]
    fn a_chart_is_data_for_the_page_script() {
        let mut figure = Figure::with_trace(json!({"name": "</script><b>", "line": {"color": "#0071e3"}}));
        figure.update_layout(json!({"height": 360, "font": {"color": "#6e6e73"}}));
        figure.add_vline(json!(0.2), json!({"dash": "dot"}), None);
        let html = chart(&figure, "pl-chart-replay");
        assert!(html.starts_with(
            r#"<div class="pl-chart"><div id="pl-chart-replay" class="plotly-graph-div" style="height:360px; width:100%;"></div><script type="application/json" data-chart="pl-chart-replay">{"#
        ));
        assert!(!html.contains("Plotly.newPlot") && !html.contains("</script><b>"), "{html}");
        let source =
            html.split_once(r#"data-chart="pl-chart-replay">"#).unwrap().1.strip_suffix("</script></div>").unwrap();
        let source: Value = serde_json::from_str(source).expect("JSON for the page script");
        assert_eq!(source["data"][0]["name"], "</script><b>");
        assert_eq!(source["data"][0]["line"]["color"], "#0071e3", "the server sends the light colors");
        assert_eq!(source["layout"]["shapes"][0]["x0"], 0.2);
        assert_eq!(source["config"], plotly_config());
        assert!(chart(&Figure::new(), "c").contains(r#"style="height:100%; width:100%;""#));
    }

    #[test]
    fn strips_titles_and_alerts_match_the_python_markup() {
        assert_eq!(title_html("A & B", ""), r#"<h3 class="pl-title">A &amp; B</h3>"#);
        assert!(title_html("T", "sub").ends_with(r#"<p class="pl-sub">sub</p>"#));
        let strip = stat_strip(&[("12.00%".into(), "In sample", "Average")]);
        assert_eq!(
            strip,
            r#"<div class="pl-stats"><div class="pl-stat"><span class="pl-num">12.00%</span><span class="pl-lbl">In sample</span><small>Average</small></div></div>"#
        );
        assert!(alert_plain(Alert::Error, "Missing <x>").contains("Missing &lt;x&gt;"));
        assert!(latex(r"w^*&=1").contains(r"w^*&amp;=1"));
        assert!(guarded(|| todo!("story")).contains("not yet implemented: story"));
    }
}
