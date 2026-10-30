//! Plotly figures as JSON, in place of `plotly.graph_objects`.
//!
//! A [`Figure`] holds trace and layout objects. Updates merge nested objects as
//! plotly.py does. [`Figure::to_html`] writes the same `<div>` and
//! `Plotly.newPlot` call as `fig.to_html(full_html=False, include_plotlyjs=False)`;
//! the page or report supplies plotly.js once.

use serde_json::{Map, Value, json};

/// The bundled plotly.js (MIT), the same file that plotly.py 7.1.0 embeds.
pub const PLOTLY_JS: &str = include_str!("../assets/plotly/plotly.min.js");
/// The plotly.js license text for report notices.
pub const PLOTLY_JS_LICENSE: &str = include_str!("../assets/plotly/LICENSE.txt");

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Figure {
    pub data: Vec<Value>,
    pub layout: Value,
    /// Layout shapes and annotations added by `add_vline` and similar helpers.
    shapes: Vec<Value>,
    annotations: Vec<Value>,
}

/// Merge `patch` into `target`. Nested objects merge; other values replace.
pub fn merge(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(target), Value::Object(patch)) => {
            for (key, value) in patch {
                match target.get_mut(key) {
                    Some(existing) if existing.is_object() && value.is_object() => merge(existing, value),
                    _ => {
                        target.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (target, patch) => *target = patch.clone(),
    }
}

impl Figure {
    pub fn new() -> Self {
        Self { data: Vec::new(), layout: Value::Object(Map::new()), shapes: Vec::new(), annotations: Vec::new() }
    }

    /// A figure with one trace.
    pub fn with_trace(trace: Value) -> Self {
        let mut figure = Self::new();
        figure.add_trace(trace);
        figure
    }

    /// Add a trace. A trace without a `type` is a scatter trace.
    pub fn add_trace(&mut self, mut trace: Value) -> &mut Self {
        if trace.get("type").is_none() {
            merge(&mut trace, &json!({"type": "scatter"}));
        }
        self.data.push(trace);
        self
    }

    pub fn update_layout(&mut self, patch: Value) -> &mut Self {
        merge(&mut self.layout, &patch);
        self
    }

    /// Update every x axis (`xaxis`, `xaxis2`, ...). Creates `xaxis` when none exists.
    pub fn update_xaxes(&mut self, patch: Value) -> &mut Self {
        self.update_axes("xaxis", patch)
    }

    /// Update every y axis (`yaxis`, `yaxis2`, ...). Creates `yaxis` when none exists.
    pub fn update_yaxes(&mut self, patch: Value) -> &mut Self {
        self.update_axes("yaxis", patch)
    }

    fn update_axes(&mut self, prefix: &str, patch: Value) -> &mut Self {
        let layout = self.layout.as_object_mut().expect("layout is an object");
        let mut names: Vec<String> = layout
            .keys()
            .filter(|key| key.starts_with(prefix) && key[prefix.len()..].chars().all(|c| c.is_ascii_digit()))
            .cloned()
            .collect();
        if names.is_empty() {
            names.push(prefix.to_string());
        }
        for name in names {
            let axis = layout.entry(name).or_insert_with(|| Value::Object(Map::new()));
            merge(axis, &patch);
        }
        self
    }

    /// Update every trace, or only the traces whose fields match `selector`.
    pub fn update_traces(&mut self, patch: Value, selector: Option<Value>) -> &mut Self {
        for trace in &mut self.data {
            let selected = match &selector {
                Some(Value::Object(fields)) => fields.iter().all(|(key, value)| trace.get(key) == Some(value)),
                _ => true,
            };
            if selected {
                merge(trace, &patch);
            }
        }
        self
    }

    pub fn add_annotation(&mut self, annotation: Value) -> &mut Self {
        self.annotations.push(annotation);
        self
    }

    pub fn add_shape(&mut self, shape: Value) -> &mut Self {
        self.shapes.push(shape);
        self
    }

    /// A vertical line across the plot, as plotly.py's `add_vline`.
    /// `line` holds line properties such as `{"dash": "dot", "color": "#6e6e73"}`.
    pub fn add_vline(&mut self, x: Value, line: Value, annotation: Option<(&str, &str)>) -> &mut Self {
        self.add_shape(
            json!({"type": "line", "x0": x, "x1": x, "xref": "x", "y0": 0, "y1": 1, "yref": "y domain", "line": line}),
        );
        if let Some((text, position)) = annotation {
            let (y, yanchor) = if position.contains("bottom") { (0, "bottom") } else { (1, "top") };
            let xanchor = if position.contains("left") {
                "right"
            } else if position.contains("right") {
                "left"
            } else {
                "center"
            };
            self.add_annotation(json!({
                "x": x, "xref": "x", "y": y, "yref": "y domain", "text": text, "showarrow": false,
                "xanchor": xanchor, "yanchor": yanchor,
            }));
        }
        self
    }

    /// A horizontal line across the plot, as plotly.py's `add_hline`.
    pub fn add_hline(&mut self, y: Value, line: Value, annotation: Option<(&str, &str)>) -> &mut Self {
        self.add_shape(
            json!({"type": "line", "y0": y, "y1": y, "yref": "y", "x0": 0, "x1": 1, "xref": "x domain", "line": line}),
        );
        if let Some((text, position)) = annotation {
            let (x, xanchor) = if position.contains("left") { (0, "left") } else { (1, "right") };
            let yanchor = if position.contains("bottom") { "top" } else { "bottom" };
            self.add_annotation(json!({
                "y": y, "yref": "y", "x": x, "xref": "x domain", "text": text, "showarrow": false,
                "xanchor": xanchor, "yanchor": yanchor,
            }));
        }
        self
    }

    /// A shaded vertical band, as plotly.py's `add_vrect`. `properties` adds fill and line settings.
    pub fn add_vrect(&mut self, x0: Value, x1: Value, properties: Value) -> &mut Self {
        let mut shape = json!({"type": "rect", "x0": x0, "x1": x1, "xref": "x", "y0": 0, "y1": 1, "yref": "y domain"});
        merge(&mut shape, &properties);
        self.add_shape(shape)
    }

    /// The complete layout, with added shapes and annotations after any set directly.
    pub fn full_layout(&self) -> Value {
        let mut layout = self.layout.clone();
        let object = layout.as_object_mut().expect("layout is an object");
        for (key, extra) in [("shapes", &self.shapes), ("annotations", &self.annotations)] {
            if !extra.is_empty() {
                let entry = object.entry(key).or_insert_with(|| Value::Array(Vec::new()));
                if let Value::Array(items) = entry {
                    items.extend(extra.iter().cloned());
                }
            }
        }
        layout
    }

    /// `{"data": [...], "layout": {...}}`.
    pub fn to_json(&self) -> Value {
        json!({"data": self.data, "layout": self.full_layout()})
    }

    /// The figure `<div>` and its `Plotly.newPlot` script, without plotly.js.
    pub fn to_html(&self, div_id: &str, config: &Value) -> String {
        let height = match self.layout.get("height").and_then(Value::as_f64) {
            Some(height) => format!("height:{}px; width:100%;", crate::pyfmt::g(height)),
            None => "height:100%; width:100%;".to_string(),
        };
        let data = script_json(&Value::Array(self.data.clone()));
        let layout = script_json(&self.full_layout());
        let config = script_json(config);
        format!(
            "<div>                        <div id=\"{div_id}\" class=\"plotly-graph-div\" style=\"{height}\"></div>            \
             <script type=\"text/javascript\">                window.PLOTLYENV=window.PLOTLYENV || {{}};                                    \
             if (document.getElementById(\"{div_id}\")) {{                    Plotly.newPlot(                        \"{div_id}\",                        \
             {data},                        {layout},                        {config}                    )                }};            </script>        </div>"
        )
    }
}

/// JSON that is safe inside a `<script>` element.
pub fn script_json(value: &Value) -> String {
    serde_json::to_string(value)
        .expect("figure JSON")
        .replace("</", "<\\/")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_merge_nested_objects() {
        let mut figure = Figure::with_trace(json!({"x": [1, 2], "y": [3, 4], "line": {"width": 2}}));
        figure.update_layout(json!({"legend": {"x": 0, "font": {"size": 12}}}));
        figure.update_layout(json!({"legend": {"y": -0.24}}));
        figure.update_traces(json!({"line": {"color": "red"}}), None);
        figure.update_yaxes(json!({"tickformat": ".0%"}));
        assert_eq!(figure.layout["legend"], json!({"x": 0, "font": {"size": 12}, "y": -0.24}));
        assert_eq!(figure.data[0]["line"], json!({"width": 2, "color": "red"}));
        assert_eq!(figure.data[0]["type"], "scatter");
        assert_eq!(figure.layout["yaxis"]["tickformat"], ".0%");
    }

    #[test]
    fn html_contains_a_safe_newplot_call() {
        let mut figure = Figure::with_trace(json!({"name": "</script><b>"}));
        figure.update_layout(json!({"height": 420}));
        figure.add_vline(json!(0.2), json!({"dash": "dot"}), Some(("Volatility budget", "top left")));
        let html = figure.to_html("chart-1", &json!({"responsive": true}));
        assert!(html.contains("style=\"height:420px; width:100%;\""));
        assert!(html.contains("Plotly.newPlot("));
        assert!(!html.contains("</script><b>"));
        assert_eq!(figure.full_layout()["shapes"][0]["x0"], 0.2);
        assert_eq!(figure.full_layout()["annotations"][0]["xanchor"], "right");
    }
}
