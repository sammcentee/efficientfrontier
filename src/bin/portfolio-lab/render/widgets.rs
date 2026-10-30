//! Form controls in place of Streamlit widgets.
//!
//! The browser submits form values; a control without a value keeps the stored
//! setting on the server. Checkboxes send `true` or `false` through a hidden
//! field, so a disabled checkbox still keeps its value, as in Streamlit.

use chrono::Months;
use efficient_frontier::frame::{Date, iso};
use efficient_frontier::pyfmt;

use super::html::{esc, help};

/// A field label with an optional help icon.
pub fn label(for_id: &str, text: &str, help_text: Option<&str>) -> String {
    format!(
        r#"<label class="pl-label" for="{for_id}">{}{}</label>"#,
        esc(text),
        help_text.map(help).unwrap_or_default()
    )
}

/// `st.number_input`.
pub struct NumberInput<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub label: &'a str,
    pub value: f64,
    /// A Python format specification of the shown value, such as `.2f` or `g`.
    pub format: &'a str,
    /// The step of the minus and plus buttons and the arrow keys.
    pub step: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Show a percent sign after the value.
    pub percent: bool,
    pub help: Option<&'a str>,
}

impl NumberInput<'_> {
    pub fn html(&self) -> String {
        let bound = |name: &str, value: Option<f64>| {
            value.map(|v| format!(r#" {name}="{}""#, pyfmt::repr(v))).unwrap_or_default()
        };
        format!(
            r#"<div class="pl-field" id="field_{id}">{label}<div class="pl-number"><input type="number" id="{id}" name="{name}" value="{value}" step="any"{min}{max} inputmode="decimal" required data-step="{step}" data-format="{format}">{unit}<button type="button" class="pl-step" data-step-by="-1" tabindex="-1" aria-label="Decrease">−</button><button type="button" class="pl-step" data-step-by="1" tabindex="-1" aria-label="Increase">+</button></div></div>"#,
            id = self.id,
            label = label(self.id, self.label, self.help),
            unit = if self.percent { r#"<span class="pl-unit" aria-hidden="true">%</span>"# } else { "" },
            name = self.name,
            value = pyfmt::format(self.value, self.format),
            min = bound("min", self.min),
            max = bound("max", self.max),
            step = pyfmt::repr(self.step),
            format = self.format,
        )
    }
}

/// `st.selectbox`.
pub fn select(id: &str, text: &str, options: &[&str], selected: &str, help_text: Option<&str>) -> String {
    let options: String = options
        .iter()
        .map(|option| {
            format!(
                r#"<option value="{0}"{1}>{0}</option>"#,
                esc(option),
                if *option == selected { " selected" } else { "" }
            )
        })
        .collect();
    format!(
        r#"<div class="pl-field" id="field_{id}">{}<div class="pl-select"><select id="{id}" name="{id}">{options}</select></div></div>"#,
        label(id, text, help_text)
    )
}

/// `st.checkbox`, with a hidden field that carries its value.
pub fn checkbox(id: &str, text: &str, checked: bool, disabled: bool) -> String {
    format!(
        r#"<div class="pl-field" id="field_{id}"><label class="pl-check"><input type="checkbox" id="{id}" data-mirror="{id}"{}{}><span>{}</span></label><input type="hidden" name="{id}" value="{checked}"></div>"#,
        if checked { " checked" } else { "" },
        if disabled { " disabled" } else { "" },
        esc(text)
    )
}

/// `st.radio` with captions, as a list of rows.
pub fn radio_list(name: &str, legend: &str, options: &[(&str, &str)], selected: &str) -> String {
    let rows: String = options
        .iter()
        .map(|(value, caption)| {
            format!(
                r#"<label class="pl-radio"><input type="radio" name="{name}" value="{0}"{1}><span class="pl-radio-text"><span>{0}</span><small>{2}</small></span></label>"#,
                esc(value),
                if *value == selected { " checked" } else { "" },
                esc(caption)
            )
        })
        .collect();
    format!(
        r#"<fieldset class="pl-field" id="field_{name}"><legend class="pl-label">{}</legend><div class="pl-radio-list">{rows}</div></fieldset>"#,
        esc(legend)
    )
}

/// `st.segmented_control(required=True)`: (value, text) per segment.
pub fn segmented(name: &str, legend: &str, options: &[(&str, &str)], selected: &str) -> String {
    let segments: String = options
        .iter()
        .map(|(value, text)| {
            format!(
                r#"<label><input type="radio" name="{name}" value="{}"{}><span>{}</span></label>"#,
                esc(value),
                if *value == selected { " checked" } else { "" },
                esc(text)
            )
        })
        .collect();
    format!(
        r#"<fieldset class="pl-field" id="field_{name}"><legend class="pl-label">{}</legend><div class="pl-segmented">{segments}</div></fieldset>"#,
        esc(legend)
    )
}

/// A date input (`st.date_input`). Streamlit allows ten years on either side of the default.
pub fn date_input(id: &str, text: &str, value: Date, help_text: Option<&str>) -> String {
    let years = Months::new(120);
    let min = value.checked_sub_months(years).map(iso).unwrap_or_default();
    let max = value.checked_add_months(years).map(iso).unwrap_or_default();
    format!(
        r#"<div class="pl-field" id="field_{id}">{}<input type="date" class="pl-date" id="{id}" name="{id}" value="{}" min="{min}" max="{max}" required></div>"#,
        label(id, text, help_text),
        iso(value)
    )
}

/// `st.file_uploader(type=["csv"])` with its stored file: "Using …" and Remove.
pub struct FileSlot<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub caption: Option<&'a str>,
    pub help: Option<&'a str>,
    /// The name of the stored file, if any.
    pub stored: Option<&'a str>,
    /// The hidden field that removes the stored file.
    pub clear: &'a str,
}

impl FileSlot<'_> {
    pub fn html(&self) -> String {
        let caption = self
            .caption
            .map(|text| format!(r#"<div class="pl-caption"><p>{}</p></div>"#, esc(text)))
            .unwrap_or_default();
        let stored = self
            .stored
            .map(|name| {
                format!(
                    r#"<div class="pl-stored" data-stored><div class="pl-caption"><p>Using {}. Upload a file to replace it.</p></div><button type="button" class="pl-btn pl-btn-tertiary" id="{}" data-remove="{}">Remove</button></div>"#,
                    esc(name),
                    self.clear,
                    self.clear
                )
            })
            .unwrap_or_default();
        format!(
            r#"<div class="pl-field pl-file" id="field_{id}">{label}<label class="pl-dropzone"><input type="file" id="{id}" name="{id}" accept=".csv,text/csv"><span class="pl-dropzone-text"><b data-file-name>Drag and drop file here</b><small>Limit 200MB per file • CSV</small></span><span class="pl-btn pl-btn-secondary pl-browse">Browse files</span></label>{caption}{stored}</div>"#,
            id = self.id,
            label = label(self.id, self.label, self.help),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_inputs_show_the_python_format() {
        let input = NumberInput {
            id: "cap",
            name: "cap",
            label: "Holding limit",
            value: 12.0,
            format: "g",
            step: 1.0,
            min: Some(1.0),
            max: Some(100.0),
            percent: true,
            help: None,
        };
        let html = input.html();
        assert!(
            html.contains(r#"<input type="number" id="cap" name="cap" value="12" step="any" min="1.0" max="100.0""#)
        );
        assert!(html.contains(r#"<span class="pl-unit" aria-hidden="true">%</span>"#));
        let budget = NumberInput { value: 19.84, format: ".2f", step: 0.1, ..input };
        assert!(budget.html().contains(r#"value="19.84""#) && budget.html().contains(r#"data-step="0.1""#));
    }

    #[test]
    fn checkboxes_carry_their_value_when_disabled() {
        let html = checkbox("compare_market", "Compare", true, true);
        assert!(
            html.contains(
                r#"<input type="checkbox" id="compare_market" data-mirror="compare_market" checked disabled>"#
            )
        );
        assert!(html.contains(r#"<input type="hidden" name="compare_market" value="true">"#));
    }
}
