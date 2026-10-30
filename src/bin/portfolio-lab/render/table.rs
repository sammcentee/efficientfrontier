//! Formatted HTML tables in place of `st.dataframe`.

use efficient_frontier::frame::{Cell, Frame, Table, iso};

use super::format::{day, number};
use super::html::esc;

/// A table ready to render: header labels and rows of cell HTML.
pub struct Grid {
    head: Vec<String>,
    rows: Vec<Vec<String>>,
    right: Vec<bool>,
    /// Leading columns that label the rows.
    index_columns: usize,
}

impl Grid {
    /// An empty grid. `head` holds escaped header labels.
    pub fn new(head: Vec<String>, index_columns: usize) -> Self {
        let right = vec![false; head.len()];
        Self { head, rows: Vec::new(), right, index_columns }
    }

    pub fn push(&mut self, row: Vec<String>) {
        debug_assert_eq!(row.len(), self.head.len(), "row width must match the header");
        self.rows.push(row);
    }

    /// Right-align these columns, as numbers.
    pub fn right_align(mut self, columns: impl IntoIterator<Item = usize>) -> Self {
        for column in columns {
            self.right[column] = true;
        }
        self
    }

    pub fn html(&self) -> String {
        let align = |column: usize| if self.right[column] { r#" class="pl-r""# } else { "" };
        let mut html = String::from(r#"<div class="pl-table-wrap"><table class="pl-table"><thead><tr>"#);
        for (column, label) in self.head.iter().enumerate() {
            html.push_str(&format!(r#"<th scope="col"{}>{label}</th>"#, align(column)));
        }
        html.push_str("</tr></thead><tbody>");
        for row in &self.rows {
            html.push_str("<tr>");
            for (column, value) in row.iter().enumerate() {
                let tag = if column < self.index_columns { r#"th scope="row""# } else { "td" };
                let close = if column < self.index_columns { "th" } else { "td" };
                html.push_str(&format!("<{tag}{}>{value}</{close}>", align(column)));
            }
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table></div>");
        html
    }
}

/// How `st.dataframe` shows an unformatted value.
pub fn cell(value: &Cell) -> String {
    match value {
        Cell::Null => String::new(),
        Cell::Bool(checked) => format!(
            r#"<input type="checkbox" disabled{} aria-label="{checked}">"#,
            if *checked { " checked" } else { "" }
        ),
        Cell::Int(number) => number.to_string(),
        Cell::Float(value) => number(*value),
        Cell::Str(text) => esc(text),
        Cell::Date(date) => iso(*date),
    }
}

fn is_number(value: &Cell) -> bool {
    matches!(value, Cell::Int(_) | Cell::Float(_))
}

/// True when every present value of a column is a number.
fn numeric(values: &[Cell]) -> bool {
    values.iter().any(is_number) && values.iter().all(|value| value.is_null() || is_number(value))
}

/// A [`Table`] with column labels and formats.
///
/// `index` holds one header label per index level, or `None` to hide the index.
/// `label` returns a column's header, or `None` to hide the column. `format`
/// returns a cell's HTML; it receives `"_index"` for index cells.
pub fn table(
    table: &Table,
    index: Option<&[&str]>,
    label: &dyn Fn(&str) -> Option<String>,
    format: &dyn Fn(&str, &Cell) -> String,
) -> Grid {
    let levels = if index.is_some() { table.index.len() } else { 0 };
    let columns: Vec<(usize, String)> =
        table.columns.iter().enumerate().filter_map(|(j, name)| label(name).map(|text| (j, text))).collect();
    let mut head: Vec<String> =
        (0..levels).map(|level| esc(index.and_then(|labels| labels.get(level)).copied().unwrap_or(""))).collect();
    head.extend(columns.iter().map(|(_, text)| esc(text)));
    let mut right: Vec<usize> = (0..levels).filter(|level| numeric(&table.index[*level])).collect();
    right.extend(columns.iter().enumerate().filter(|(_, (j, _))| numeric(&table.data[*j])).map(|(k, _)| levels + k));
    let mut grid = Grid::new(head, levels).right_align(right);
    for i in 0..table.nrows() {
        let mut row: Vec<String> = (0..levels).map(|level| format("_index", &table.index[level][i])).collect();
        row.extend(columns.iter().map(|(j, _)| format(&table.columns[*j], &table.data[*j][i])));
        grid.push(row);
    }
    grid
}

/// Keep every column under its own name.
pub fn same_label(name: &str) -> Option<String> {
    Some(name.to_string())
}

/// Format every cell as `st.dataframe` does without a format.
pub fn default_format(_: &str, value: &Cell) -> String {
    cell(value)
}

/// A dated [`Frame`]: dates as "2 Oct 2026" under `index_label`, values through `format`.
pub fn frame(frame: &Frame, index_label: &str, format: &dyn Fn(f64) -> String) -> Grid {
    let mut head = vec![esc(index_label)];
    head.extend(frame.columns.iter().map(|column| esc(column)));
    let mut grid = Grid::new(head, 1).right_align(1..=frame.ncols());
    for (i, date) in frame.index.iter().enumerate() {
        let mut row = vec![day(*date)];
        row.extend((0..frame.ncols()).map(|j| esc(&format(frame.get(i, j)))));
        grid.push(row);
    }
    grid
}

/// A dated [`Frame`] as JSON for the browser's table: dates as "2 Oct 2026" and
/// values in the default `st.dataframe` format, as text. The browser shows only
/// the visible rows, so a large price history does not stop the page.
pub fn frame_json(frame: &Frame, index_label: &str) -> String {
    let text = |value: &str| serde_json::to_string(value).expect("JSON text");
    let mut json = String::with_capacity(frame.nrows() * (frame.ncols() + 1) * 10 + 256);
    json.push_str(&format!(r#"{{"index_label":{},"columns":["#, text(index_label)));
    json.push_str(&frame.columns.iter().map(|column| text(column)).collect::<Vec<_>>().join(","));
    json.push_str(r#"],"index":["#);
    json.push_str(&frame.index.iter().map(|date| text(&day(*date))).collect::<Vec<_>>().join(","));
    json.push_str(r#"],"rows":["#);
    for i in 0..frame.nrows() {
        if i > 0 {
            json.push(',');
        }
        json.push('[');
        for j in 0..frame.ncols() {
            if j > 0 {
                json.push(',');
            }
            // A formatted number has only digits, a sign and a point: no JSON escape is needed.
            json.push('"');
            json.push_str(&number(frame.get(i, j)));
            json.push('"');
        }
        json.push(']');
    }
    json.push_str("]}");
    json
}

#[cfg(test)]
mod tests {
    use super::*;
    use efficient_frontier::frame::{Date, cells};

    #[test]
    fn tables_show_labels_formats_and_hidden_columns() {
        let data = Table::labelled(Some("symbol"), &["A", "B<"])
            .with("weight", cells(&[0.5, 0.25]))
            .with("hidden", cells(&["x", "y"]))
            .with("held", cells(&[true, false]));
        let label =
            |name: &str| (name != "hidden").then(|| if name == "weight" { "Weight".into() } else { name.into() });
        let format = |name: &str, value: &Cell| match (name, value.as_f64()) {
            ("weight", Some(weight)) => format!("{:.0}%", weight * 100.0),
            _ => cell(value),
        };
        let html = table(&data, Some(&["Symbol"]), &label, &format).html();
        assert!(html.contains(
            r#"<th scope="col">Symbol</th><th scope="col" class="pl-r">Weight</th><th scope="col">held</th>"#
        ));
        assert!(html.contains(r#"<th scope="row">B&lt;</th><td class="pl-r">25%</td>"#));
        assert!(!html.contains("hidden"));
        assert!(html.contains(r#"<input type="checkbox" disabled checked aria-label="true">"#));
        let hidden = table(&data, None, &same_label, &default_format).html();
        assert!(!hidden.contains("Symbol") && hidden.contains(r#"<td class="pl-r">0.5</td>"#));
    }

    #[test]
    fn frames_become_json_with_display_text() {
        let prices = Frame::from_columns(
            vec![Date::from_ymd_opt(2021, 1, 4).unwrap(), Date::from_ymd_opt(2021, 1, 5).unwrap()],
            vec![("A\"B".into(), vec![99.54586, -0.5]), ("C".into(), vec![1255.0, f64::NAN])],
        );
        let json = frame_json(&prices, "Date");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["index_label"], "Date");
        assert_eq!(value["columns"], serde_json::json!(["A\"B", "C"]));
        assert_eq!(value["index"], serde_json::json!(["4 Jan 2021", "5 Jan 2021"]));
        assert_eq!(value["rows"], serde_json::json!([["99.5459", "1255"], ["-0.5", ""]]));
    }

    #[test]
    fn frames_show_dates_as_days() {
        let prices =
            Frame::from_columns(vec![Date::from_ymd_opt(2021, 1, 4).unwrap()], vec![("A".into(), vec![99.54586])]);
        let html = frame(&prices, "Date", &number).html();
        assert!(html.contains(r#"<th scope="row">4 Jan 2021</th><td class="pl-r">99.5459</td>"#));
    }
}
