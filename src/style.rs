//! Shared visual styles for the app, charts, and reports.

use serde_json::{Value, json};

use crate::plot::Figure;

pub const FONT: &str = "-apple-system, BlinkMacSystemFont, \"SF Pro Text\", \"Segoe UI Variable Text\", \"Segoe UI\", Inter, Roboto, \"Helvetica Neue\", Arial, sans-serif";
pub const INK: &str = "#1d1d1f";
pub const INK_2: &str = "#424245";
pub const INK_3: &str = "#6e6e73";
pub const INK_4: &str = "#86868b";
pub const GRID: &str = "#e8e8ed";
pub const REST: &str = "#c7c7cc";
pub const ACCENT: &str = "#0071e3";
pub const FILL: &str = "#f2f2f4";
pub const CANVAS: &str = "#f5f5f7";

/// Colors by portfolio or profile name.
pub const COLORS: [(&str, &str); 7] = [
    ("Low", "#5e9eea"),
    ("Medium", "#0071e3"),
    ("High", "#0a3f8f"),
    ("Extreme", "#0a3f8f"),
    ("Minimum volatility", "#5e9eea"),
    ("Equal weight", "#0071e3"),
    ("Maximum Sharpe", "#0a3f8f"),
];
/// Colors by benchmark display name.
pub const BENCHMARK_COLORS: [(&str, &str); 2] = [("S&P 500 (SPY)", "#8e8e93"), ("Nasdaq-100 (QQQ)", "#545458")];

/// `COLORS.get(name, default)`.
pub fn color(name: &str, default: &'static str) -> &'static str {
    COLORS.iter().find(|(key, _)| *key == name).map_or(default, |(_, value)| value)
}

/// `BENCHMARK_COLORS.get(name, default)`.
pub fn benchmark_color(name: &str, default: &'static str) -> &'static str {
    BENCHMARK_COLORS.iter().find(|(key, _)| *key == name).map_or(default, |(_, value)| value)
}

/// The Plotly configuration of every chart.
pub fn plotly_config() -> Value {
    json!({"displayModeBar": false, "displaylogo": false, "scrollZoom": false, "doubleClick": false, "responsive": true})
}

pub const TOKENS_CSS: &str = include_str!("../assets/css/tokens.css");
pub const STORY_CSS: &str = include_str!("../assets/css/story.css");
pub const REPORT_ONLY_CSS: &str = include_str!("../assets/css/report.css");

/// The CSS of exported reports: tokens, report layout, then story components.
pub fn report_css() -> String {
    format!("{TOKENS_CSS}{REPORT_ONLY_CSS}{STORY_CSS}")
}

/// Show the engine profile "Extreme" as "Highest". Data keys do not change.
pub fn display_label(name: &str) -> String {
    if name == "Extreme" { "Highest".into() } else { name.replace(" · Extreme", " · Highest") }
}

/// Options of [`style_chart`], with the Python defaults.
#[derive(Clone, Copy, Debug)]
pub struct ChartStyle<'a> {
    pub title: Option<&'a str>,
    pub height: u32,
    pub hovermode: &'a str,
    pub story: bool,
}

impl Default for ChartStyle<'_> {
    fn default() -> Self {
        Self { title: None, height: 420, hovermode: "closest", story: false }
    }
}

/// Apply the Portfolio Lab theme. Chart data does not change.
pub fn style_chart(figure: &mut Figure, style: ChartStyle) {
    let ChartStyle { title, height, hovermode, story } = style;
    let title = match title {
        Some(text) if !text.is_empty() && !story => json!({
            "text": text, "x": 0, "xref": "paper", "xanchor": "left", "y": 1, "yref": "container",
            "yanchor": "top", "pad": {"t": 6}, "font": {"family": FONT, "size": 15, "color": INK},
        }),
        _ => json!({"text": ""}),
    };
    let mut legend = if story {
        json!({"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom"})
    } else {
        json!({"orientation": "h", "x": 0, "xanchor": "left", "y": -0.16, "yanchor": "top", "maxheight": 80})
    };
    crate::plot::merge(
        &mut legend,
        &json!({"font": {"size": 12, "color": INK_2}, "title": {"text": ""}, "bgcolor": "rgba(0,0,0,0)"}),
    );
    let top = if story {
        48
    } else if title["text"] != "" {
        44
    } else {
        16
    };
    figure.update_layout(json!({
        "template": {}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "title": title,
        "font": {"family": FONT, "size": 12, "color": INK_3},
        "colorway": [ACCENT, "#545458", "#8e8e93"],
        "margin": {"l": 8, "r": if story { 64 } else { 16 }, "t": top, "b": if story { 8 } else { 88 }},
        "legend": legend,
        "hovermode": hovermode, "hoverdistance": 24, "spikedistance": -1,
        "hoverlabel": {"bgcolor": INK, "bordercolor": INK, "align": "left",
                       "font": {"family": FONT, "size": 13, "color": "#ffffff"}},
        "height": height, "dragmode": false, "barcornerradius": 4, "bargap": 0.35,
    }));
    let axis = json!({
        "showline": false, "zeroline": false, "ticks": "", "automargin": true, "fixedrange": true,
        "tickfont": {"size": 12, "color": INK_3}, "title": {"font": {"size": 12, "color": INK_3}},
    });
    figure.update_xaxes(axis.clone());
    figure.update_xaxes(json!({
        "showgrid": false, "showspikes": hovermode == "x unified", "spikemode": "across",
        "spikesnap": "cursor", "spikethickness": 1, "spikecolor": INK_4, "spikedash": "solid",
    }));
    figure.update_yaxes(axis);
    figure.update_yaxes(json!({"showgrid": true, "gridcolor": GRID, "gridwidth": 1, "nticks": 6}));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_style_sets_title_axes_and_height() {
        let mut figure = Figure::new();
        style_chart(&mut figure, ChartStyle { title: Some("Classic frontier"), ..Default::default() });
        assert_eq!(figure.layout["title"]["text"], "Classic frontier");
        assert_eq!(figure.layout["margin"]["t"], 44);
        assert_eq!(figure.layout["height"], 420);
        assert_eq!(figure.layout["yaxis"]["nticks"], 6);
        assert_eq!(figure.layout["xaxis"]["showspikes"], false);
        assert_eq!(display_label("Rolling window · Extreme"), "Rolling window · Highest");
        assert!(report_css().starts_with(TOKENS_CSS));
    }
}
