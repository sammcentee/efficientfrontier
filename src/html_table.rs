//! HTML tables in the format of `pandas.DataFrame.to_html` (pandas 3.0.6).
//!
//! The output keeps the pandas layout: two-space indentation, a right-aligned
//! header row, an optional row of index names, `rowspan` cells for repeated
//! outer labels of a multi-level index, and the pandas value formats. Floats
//! use six decimals per column with common trailing zeros removed, or
//! scientific notation when a value is too small or too large. Escaping
//! replaces only `&`, `<` and `>`, as pandas does.

use crate::frame::{Cell, Table, iso};
use crate::pyfmt;

/// Options of [`to_html`]. [`Default`] is the pandas default:
/// `index=True, escape=True, border=1, classes=None`.
#[derive(Clone, Copy, Debug)]
pub struct HtmlOptions<'a> {
    /// Write the index as `<th>` cells.
    pub index: bool,
    /// Replace `&`, `<` and `>` in every cell.
    pub escape: bool,
    /// The `border` attribute. pandas omits the attribute for `0`.
    pub border: u32,
    /// Classes after `dataframe`, separated by spaces.
    pub classes: Option<&'a str>,
    /// pandas `float_format`: formats every value of a float column; NaN stays `NaN`.
    pub float_format: Option<fn(f64) -> String>,
}

impl Default for HtmlOptions<'_> {
    fn default() -> Self {
        Self { index: true, escape: true, border: 1, classes: None, float_format: None }
    }
}

impl HtmlOptions<'static> {
    /// `escape=True, border=0, classes="data"`: the options of every report table.
    pub const REPORT: Self = Self { index: true, escape: true, border: 0, classes: Some("data"), float_format: None };
    /// [`HtmlOptions::REPORT`] with `index=False`.
    pub const REPORT_NO_INDEX: Self = Self { index: false, ..Self::REPORT };
}

/// `frame.to_html(escape=True, border=0, classes="data")`.
pub fn data_table(table: &Table) -> String {
    to_html(table, &HtmlOptions::REPORT)
}

/// `frame.to_html(...)` with the given options.
pub fn to_html(table: &Table, options: &HtmlOptions) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut classes = vec!["dataframe"];
    if let Some(extra) = options.classes {
        classes.extend(extra.split_whitespace());
    }
    let border = if options.border == 0 { String::new() } else { format!(" border=\"{}\"", options.border) };
    lines.push(format!("<table{border} class=\"{}\">", classes.join(" ")));

    let cell = |text: &str| -> String {
        let text = if options.escape { escape(text) } else { text.to_string() };
        text.trim().to_string()
    };
    let levels = if options.index { table.index.len() } else { 0 };

    lines.push("  <thead>".into());
    lines.push("    <tr style=\"text-align: right;\">".into());
    for level in 0..levels {
        let label = if level + 1 == levels { table.columns_name.as_deref().unwrap_or("") } else { "" };
        lines.push(format!("      <th>{}</th>", cell(label)));
    }
    for column in &table.columns {
        lines.push(format!("      <th>{}</th>", cell(column)));
    }
    lines.push("    </tr>".into());
    if options.index && table.index_names.iter().any(Option::is_some) {
        lines.push("    <tr>".into());
        for name in &table.index_names {
            lines.push(format!("      <th>{}</th>", cell(name.as_deref().unwrap_or(""))));
        }
        for _ in &table.columns {
            lines.push("      <th></th>".into());
        }
        lines.push("    </tr>".into());
    }
    lines.push("  </thead>".into());

    let values: Vec<Vec<String>> = table
        .data
        .iter()
        .map(|column| match options.float_format {
            Some(format) if kind(column) == Kind::Float => column
                .iter()
                .map(|cell| match cell.as_f64() {
                    Some(value) if !value.is_nan() => format(value),
                    _ => "NaN".to_string(),
                })
                .collect(),
            _ => format_column(column, options.index),
        })
        .collect();
    let labels: Vec<Vec<String>> = table.index.iter().map(|level| format_column(level, false)).collect();
    let spans = if levels > 1 { row_spans(&labels) } else { Vec::new() };
    lines.push("  <tbody>".into());
    for i in 0..table.nrows() {
        lines.push("    <tr>".into());
        for (level, level_labels) in labels.iter().enumerate().take(levels) {
            let label = cell(&level_labels[i]);
            match spans.get(level).map_or(1, |level_spans| level_spans[i]) {
                0 => {}
                1 => lines.push(format!("      <th>{label}</th>")),
                span => lines.push(format!("      <th rowspan=\"{span}\" valign=\"top\">{label}</th>")),
            }
        }
        for column in &values {
            lines.push(format!("      <td>{}</td>", cell(&column[i])));
        }
        lines.push("    </tr>".into());
    }
    lines.push("  </tbody>".into());
    lines.push("</table>".into());
    lines.join("\n")
}

/// The pandas HTML escape: `&`, `<` and `>` only.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Tabs and line breaks show as `\t`, `\r` and `\n`, as in `pprint_thing`.
fn printable(text: &str) -> String {
    text.replace('\t', "\\t").replace('\r', "\\r").replace('\n', "\\n")
}

/// The row span of each label of a sparsified multi-level index: the number of
/// rows that the label covers, or zero for a label hidden under a span above it.
/// The innermost level never merges, as in pandas `sparsify_labels`.
fn row_spans(labels: &[Vec<String>]) -> Vec<Vec<usize>> {
    let levels = labels.len();
    let rows = labels.first().map_or(0, Vec::len);
    let mut spans = vec![vec![1; rows]; levels];
    for i in 1..rows {
        for level in 0..levels - 1 {
            if labels[level][i] != labels[level][i - 1] {
                break;
            }
            spans[level][i] = 0;
        }
    }
    for level_spans in spans.iter_mut().take(levels.saturating_sub(1)) {
        let mut start = 0;
        for i in 1..=rows {
            if i == rows || level_spans[i] != 0 {
                level_spans[start] = i - start;
                start = i;
            }
        }
    }
    spans
}

/// The pandas dtype that a column of cells would have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Float,
    Int,
    Bool,
    Date,
    Text,
    Object,
}

fn kind(cells: &[Cell]) -> Kind {
    let all = |test: fn(&Cell) -> bool| cells.iter().all(test);
    let null = |cell: &Cell| matches!(cell, Cell::Null);
    if cells.is_empty() || all(null) {
        Kind::Object
    } else if all(|cell| matches!(cell, Cell::Int(_))) {
        Kind::Int
    } else if all(|cell| matches!(cell, Cell::Int(_) | Cell::Float(_) | Cell::Null)) {
        Kind::Float
    } else if all(|cell| matches!(cell, Cell::Bool(_))) {
        Kind::Bool
    } else if all(|cell| matches!(cell, Cell::Date(_) | Cell::Null)) {
        Kind::Date
    } else if all(|cell| matches!(cell, Cell::Str(_) | Cell::Null)) {
        Kind::Text
    } else {
        Kind::Object
    }
}

/// The display text of each cell of one column, before escaping.
///
/// `leading_space` is pandas' `leading_space=index`. It changes the length test
/// that chooses scientific notation for floats.
pub fn format_column(cells: &[Cell], leading_space: bool) -> Vec<String> {
    match kind(cells) {
        Kind::Float => format_floats(
            &cells.iter().map(|cell| cell.as_f64().unwrap_or(f64::NAN)).collect::<Vec<_>>(),
            leading_space,
        ),
        Kind::Int | Kind::Bool => cells.iter().map(Cell::py_str).collect(),
        Kind::Date => cells.iter().map(|cell| cell.as_date().map_or_else(|| "NaT".to_string(), iso)).collect(),
        Kind::Text => cells
            .iter()
            .map(|cell| match cell {
                Cell::Str(text) => printable(text),
                _ => "NaN".to_string(),
            })
            .collect(),
        Kind::Object => cells.iter().map(object_text).collect(),
    }
}

/// One value of an object column.
fn object_text(cell: &Cell) -> String {
    match cell {
        Cell::Null => "None".into(),
        Cell::Float(value) if value.is_nan() => "NaN".into(),
        Cell::Float(value) => {
            let text = pyfmt::format(*value, " .6f");
            let text = text.trim_end_matches('0');
            if text.ends_with('.') { format!("{text}0") } else { text.to_string() }
        }
        Cell::Str(text) => printable(text),
        Cell::Date(date) => format!("{} 00:00:00", iso(*date)),
        other => other.py_str(),
    }
}

/// pandas `FloatArrayFormatter` with `display.precision` 6.
pub fn format_floats(values: &[f64], leading_space: bool) -> Vec<String> {
    let spec = |kind: char| if leading_space { format!(" .6{kind}") } else { format!(".6{kind}") };
    let with = |spec: &str| -> Vec<String> {
        trim_zeros(
            values
                .iter()
                .map(|value| if value.is_nan() { "NaN".into() } else { pyfmt::format(*value, spec) })
                .collect(),
        )
    };
    let fixed = with(&spec('f'));
    let too_long = fixed.iter().map(|text| text.chars().count()).max().unwrap_or(0) > 12;
    let large = values.iter().any(|value| value.abs() > 1e6);
    let small = values.iter().any(|value| value.abs() < 1e-6 && value.abs() > 0.0);
    if small || (too_long && large) { with(&spec('e')) } else { fixed }
}

/// `_trim_zeros_float`: remove trailing zeros equally from every decimal number, keeping one decimal.
fn trim_zeros(mut values: Vec<String>) -> Vec<String> {
    let decimal = |text: &str| {
        let body = text.trim_start();
        let body = body.strip_prefix(['+', '-']).unwrap_or(body);
        match body.split_once('.') {
            Some((integer, fraction)) => {
                !integer.is_empty()
                    && integer.chars().all(|c| c.is_ascii_digit())
                    && fraction.chars().all(|c| c.is_ascii_digit())
            }
            None => false,
        }
    };
    loop {
        let numbers: Vec<&String> = values.iter().filter(|text| decimal(text)).collect();
        if numbers.is_empty() || !numbers.iter().all(|text| text.ends_with('0')) {
            break;
        }
        for text in values.iter_mut().filter(|text| decimal(text)) {
            text.pop();
        }
    }
    for text in values.iter_mut().filter(|text| decimal(text) && text.ends_with('.')) {
        text.push('0');
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::cells;

    fn body_cells(html: &str) -> Vec<String> {
        html.lines()
            .map(str::trim)
            .filter(|line| line.starts_with("<td") || line.starts_with("<th>") || line.starts_with("<th "))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn float_columns_follow_the_pandas_formatter() {
        let cases: [(&[f64], &[&str]); 8] = [
            (&[1.0, 2.0], &["1.0", "2.0"]),
            (&[0.1, 0.25], &["0.10", "0.25"]),
            (&[0.123456789, 12.5, -3.0], &["0.123457", "12.500000", "-3.000000"]),
            (&[1e-7, 0.5], &["1.000000e-07", "5.000000e-01"]),
            (&[1e16, 0.5], &["1.000000e+16", "5.000000e-01"]),
            (&[123456789.123, 0.5], &["1.234568e+08", "5.000000e-01"]),
            (&[f64::INFINITY, f64::NEG_INFINITY, 1.0], &["inf", "-inf", "1.0"]),
            (&[1234567.891, f64::NAN], &["1234567.891", "NaN"]),
        ];
        for (values, expected) in cases {
            let shown: Vec<String> = format_floats(values, true).iter().map(|text| text.trim().to_string()).collect();
            assert_eq!(shown, expected, "{values:?}");
        }
        assert_eq!(format_floats(&[-0.0, 0.0], true), ["-0.0", " 0.0"]);
        // The leading space of an indexed table counts toward the length limit.
        assert_eq!(format_floats(&[12345678.123], false), ["12345678.123"]);
        assert_eq!(format_floats(&[12345678.123], true), [" 1.234568e+07"]);
    }

    #[test]
    fn mixed_columns_follow_pandas_dtypes() {
        let object =
            vec![Cell::from("a"), Cell::Float(1.5), Cell::Int(2), Cell::Bool(true), Cell::Null, Cell::Float(f64::NAN)];
        assert_eq!(format_column(&object, true), ["a", " 1.5", "2", "True", "None", "NaN"]);
        assert_eq!(format_column(&[Cell::Int(1), Cell::Null], true), [" 1.0", "NaN"]);
        assert_eq!(format_column(&[Cell::from("x"), Cell::Null], true), ["x", "NaN"]);
        let dates = [Cell::Date(chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()), Cell::Null];
        assert_eq!(format_column(&dates, true), ["2024-01-02", "NaT"]);
        assert_eq!(format_column(&[Cell::Bool(true), Cell::Null], true), ["True", "None"]);
        assert_eq!(format_column(&[Cell::from("a\tb\nc")], true), ["a\\tb\\nc"]);
    }

    #[test]
    fn index_names_and_columns_name_add_header_cells() {
        let mut table = Table::labelled(Some("Window"), &["r1", "<r2>"])
            .with("a", cells(&[1.5, f64::NAN]))
            .with("b", vec![Cell::from("x<y"), Cell::from("\"q\"")]);
        table.columns_name = Some("cols".into());
        let html = data_table(&table);
        assert!(html.starts_with(
            "<table class=\"dataframe data\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th>cols</th>"
        ));
        assert_eq!(
            body_cells(&html),
            [
                "<th>cols</th>",
                "<th>a</th>",
                "<th>b</th>",
                "<th>Window</th>",
                "<th></th>",
                "<th></th>",
                "<th>r1</th>",
                "<td>1.5</td>",
                "<td>x&lt;y</td>",
                "<th>&lt;r2&gt;</th>",
                "<td>NaN</td>",
                "<td>\"q\"</td>",
            ]
        );
        let plain = to_html(&Table::range(1).with("x", cells(&[1_i64])), &HtmlOptions::default());
        assert!(plain.starts_with("<table border=\"1\" class=\"dataframe\">"));
        let no_index = to_html(&table, &HtmlOptions::REPORT_NO_INDEX);
        assert_eq!(
            body_cells(&no_index),
            ["<th>a</th>", "<th>b</th>", "<td>1.5</td>", "<td>x&lt;y</td>", "<td>NaN</td>", "<td>\"q\"</td>"]
        );
    }

    #[test]
    fn multi_index_rows_merge_repeated_outer_labels() {
        let table = Table::multi(
            vec![None, Some("P".into()), Some("Q".into())],
            vec![cells(&[1_i64, 1, 1, 2]), cells(&["A", "A", "B", "B"]), cells(&["x", "y", "y", "y"])],
        )
        .with("v", cells(&[1_i64, 2, 3, 4]));
        let expected = "<table class=\"dataframe data\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th></th>\n      <th></th>\n      <th>v</th>\n    </tr>\n    <tr>\n      <th></th>\n      <th>P</th>\n      <th>Q</th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th rowspan=\"3\" valign=\"top\">1</th>\n      <th rowspan=\"2\" valign=\"top\">A</th>\n      <th>x</th>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>y</th>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>B</th>\n      <th>y</th>\n      <td>3</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <th>B</th>\n      <th>y</th>\n      <td>4</td>\n    </tr>\n  </tbody>\n</table>";
        assert_eq!(data_table(&table), expected);
    }
}
