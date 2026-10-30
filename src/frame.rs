//! Tables of prices, paths and results.
//!
//! [`Frame`] holds numbers by date, such as prices, returns and equity paths.
//! [`Table`] holds mixed results with a labelled index, such as metrics,
//! holdings and trades. Both write CSV text in the format of the Python
//! release (`pandas.DataFrame.to_csv` through `presentation.csv_text`).

use chrono::NaiveDate;
use nalgebra::{DMatrix, DVector};

use crate::pyfmt;

pub type Date = NaiveDate;

/// `YYYY-MM-DD`.
pub fn iso(date: Date) -> String {
    date.format("%Y-%m-%d").to_string()
}

/// Numbers by date: rows are observations, columns are assets or paths.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub index: Vec<Date>,
    pub columns: Vec<String>,
    /// Shape `(index.len(), columns.len())`.
    pub values: DMatrix<f64>,
}

impl Frame {
    pub fn new(index: Vec<Date>, columns: Vec<String>, values: DMatrix<f64>) -> Self {
        assert_eq!(values.nrows(), index.len(), "frame rows must match the index");
        assert_eq!(values.ncols(), columns.len(), "frame columns must match the labels");
        Self { index, columns, values }
    }

    /// Build a frame from named columns of equal length.
    pub fn from_columns(index: Vec<Date>, columns: Vec<(String, Vec<f64>)>) -> Self {
        let rows = index.len();
        let mut values = DMatrix::zeros(rows, columns.len());
        let mut labels = Vec::with_capacity(columns.len());
        for (j, (label, column)) in columns.into_iter().enumerate() {
            assert_eq!(column.len(), rows, "column {label} must match the index");
            values.column_mut(j).copy_from_slice(&column);
            labels.push(label);
        }
        Self::new(index, labels, values)
    }

    /// Build a frame from rows of equal length.
    pub fn from_rows(index: Vec<Date>, columns: Vec<String>, rows: &[Vec<f64>]) -> Self {
        assert_eq!(rows.len(), index.len(), "rows must match the index");
        let mut values = DMatrix::zeros(rows.len(), columns.len());
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), columns.len(), "row length must match the columns");
            for (j, value) in row.iter().enumerate() {
                values[(i, j)] = *value;
            }
        }
        Self::new(index, columns, values)
    }

    pub fn nrows(&self) -> usize {
        self.index.len()
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty() || self.columns.is_empty()
    }

    pub fn position(&self, column: &str) -> Option<usize> {
        self.columns.iter().position(|label| label == column)
    }

    /// A copy of column `j`.
    pub fn col(&self, j: usize) -> Vec<f64> {
        self.values.column(j).iter().copied().collect()
    }

    /// A copy of the named column, if it exists.
    pub fn column(&self, name: &str) -> Option<Vec<f64>> {
        self.position(name).map(|j| self.col(j))
    }

    /// A copy of row `i`.
    pub fn row(&self, i: usize) -> Vec<f64> {
        self.values.row(i).iter().copied().collect()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[(i, j)]
    }

    /// `iloc[start:end]` on rows.
    pub fn rows(&self, start: usize, end: usize) -> Frame {
        let end = end.min(self.nrows());
        let start = start.min(end);
        Frame::new(
            self.index[start..end].to_vec(),
            self.columns.clone(),
            self.values.rows(start, end - start).into_owned(),
        )
    }

    /// `loc[:date]`: every row on or before `date`.
    pub fn rows_through(&self, date: Date) -> Frame {
        let end = self.index.partition_point(|value| *value <= date);
        self.rows(0, end)
    }

    /// Columns at the given positions, in that order.
    pub fn select_positions(&self, positions: &[usize]) -> Frame {
        let mut values = DMatrix::zeros(self.nrows(), positions.len());
        for (target, source) in positions.iter().enumerate() {
            values.column_mut(target).copy_from(&self.values.column(*source));
        }
        Frame::new(self.index.clone(), positions.iter().map(|j| self.columns[*j].clone()).collect(), values)
    }

    /// Named columns in the given order. `None` if one is missing.
    pub fn select<S: AsRef<str>>(&self, names: &[S]) -> Option<Frame> {
        let positions = names.iter().map(|name| self.position(name.as_ref())).collect::<Option<Vec<_>>>()?;
        Some(self.select_positions(&positions))
    }

    /// Every column except the named ones, in the original order.
    pub fn drop_columns<S: AsRef<str>>(&self, names: &[S]) -> Frame {
        let positions: Vec<usize> =
            (0..self.ncols()).filter(|j| !names.iter().any(|name| name.as_ref() == self.columns[*j])).collect();
        self.select_positions(&positions)
    }

    /// `pct_change(fill_method=None).iloc[1:]`: simple returns from row 1 on.
    pub fn pct_change(&self) -> Frame {
        if self.nrows() < 2 {
            return Frame::new(Vec::new(), self.columns.clone(), DMatrix::zeros(0, self.ncols()));
        }
        let rows = self.nrows() - 1;
        let mut values = DMatrix::zeros(rows, self.ncols());
        for j in 0..self.ncols() {
            for i in 0..rows {
                values[(i, j)] = self.values[(i + 1, j)] / self.values[(i, j)] - 1.0;
            }
        }
        Frame::new(self.index[1..].to_vec(), self.columns.clone(), values)
    }

    /// Column means (`DataFrame.mean()`), as a vector in column order.
    pub fn means(&self) -> DVector<f64> {
        DVector::from_iterator(self.ncols(), (0..self.ncols()).map(|j| mean(self.values.column(j).as_slice())))
    }

    /// Sample covariance with `ddof=1` (`DataFrame.cov()`).
    pub fn covariance(&self) -> DMatrix<f64> {
        covariance(&self.values)
    }

    /// CSV text with the index first, as `csv_text(frame.rename_axis(label))`.
    pub fn to_csv(&self, index_label: &str) -> String {
        let mut text = String::new();
        let mut header = vec![csv_field(&text_label(index_label))];
        header.extend(self.columns.iter().map(|column| csv_field(&text_label(column))));
        text.push_str(&header.join(","));
        text.push_str("\r\n");
        for (i, date) in self.index.iter().enumerate() {
            text.push_str(&iso(*date));
            for j in 0..self.ncols() {
                text.push(',');
                text.push_str(&float_text(self.values[(i, j)]));
            }
            text.push_str("\r\n");
        }
        text
    }
}

/// A numeric vector with labels, such as weights by asset.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    pub labels: Vec<String>,
    pub values: Vec<f64>,
}

impl Series {
    pub fn new(labels: Vec<String>, values: Vec<f64>) -> Self {
        assert_eq!(labels.len(), values.len(), "series labels must match the values");
        Self { labels, values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn position(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|name| name == label)
    }

    pub fn get(&self, label: &str) -> Option<f64> {
        self.position(label).map(|i| self.values[i])
    }

    /// `reindex(labels, fill_value=fill)`.
    pub fn reindex<S: AsRef<str>>(&self, labels: &[S], fill: f64) -> Series {
        Series::new(
            labels.iter().map(|label| label.as_ref().to_string()).collect(),
            labels.iter().map(|label| self.get(label.as_ref()).unwrap_or(fill)).collect(),
        )
    }

    /// Positions in stable descending order of value (`sort_values(ascending=False, kind="stable")`).
    pub fn descending_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_by(|a, b| self.values[*b].partial_cmp(&self.values[*a]).unwrap_or(std::cmp::Ordering::Equal));
        order
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.labels.iter().map(String::as_str).zip(self.values.iter().copied())
    }

    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }

    pub fn max(&self) -> f64 {
        self.values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// A square or rectangular matrix with row and column labels, such as a covariance matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledMatrix {
    pub index: Vec<String>,
    pub columns: Vec<String>,
    pub values: DMatrix<f64>,
}

impl LabeledMatrix {
    pub fn new(index: Vec<String>, columns: Vec<String>, values: DMatrix<f64>) -> Self {
        assert_eq!(values.nrows(), index.len(), "matrix rows must match the index");
        assert_eq!(values.ncols(), columns.len(), "matrix columns must match the labels");
        Self { index, columns, values }
    }

    /// A square matrix with the same labels on both axes.
    pub fn square(labels: Vec<String>, values: DMatrix<f64>) -> Self {
        Self::new(labels.clone(), labels, values)
    }
}

/// One table value. `Null` is a missing value (`None` or `NaT`).
#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Date(Date),
}

impl Cell {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Cell::Float(value) => Some(*value),
            Cell::Int(value) => Some(*value as f64),
            Cell::Bool(value) => Some(f64::from(u8::from(*value))),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Cell::Str(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_date(&self) -> Option<Date> {
        match self {
            Cell::Date(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Cell::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Cell::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Cell::Null) || matches!(self, Cell::Float(value) if value.is_nan())
    }

    /// Python `str(value)` of the stored value, with dates as `YYYY-MM-DD`.
    pub fn py_str(&self) -> String {
        match self {
            Cell::Null => "None".into(),
            Cell::Bool(value) => if *value { "True" } else { "False" }.into(),
            Cell::Int(value) => value.to_string(),
            Cell::Float(value) => pyfmt::repr(*value),
            Cell::Str(value) => value.clone(),
            Cell::Date(value) => iso(*value),
        }
    }

    /// The CSV text of the value, before quoting. Missing values are empty.
    pub fn csv_text(&self) -> String {
        match self {
            Cell::Null => String::new(),
            Cell::Float(value) => float_text(*value),
            Cell::Str(value) => text_label(value),
            other => other.py_str(),
        }
    }
}

impl From<f64> for Cell {
    fn from(value: f64) -> Self {
        Cell::Float(value)
    }
}

impl From<i64> for Cell {
    fn from(value: i64) -> Self {
        Cell::Int(value)
    }
}

impl From<usize> for Cell {
    fn from(value: usize) -> Self {
        Cell::Int(value as i64)
    }
}

impl From<bool> for Cell {
    fn from(value: bool) -> Self {
        Cell::Bool(value)
    }
}

impl From<&str> for Cell {
    fn from(value: &str) -> Self {
        Cell::Str(value.to_string())
    }
}

impl From<String> for Cell {
    fn from(value: String) -> Self {
        Cell::Str(value)
    }
}

impl From<Date> for Cell {
    fn from(value: Date) -> Self {
        Cell::Date(value)
    }
}

impl<T: Into<Cell>> From<Option<T>> for Cell {
    fn from(value: Option<T>) -> Self {
        value.map_or(Cell::Null, Into::into)
    }
}

/// Convert a slice of values to cells.
pub fn cells<T: Clone + Into<Cell>>(values: &[T]) -> Vec<Cell> {
    values.iter().cloned().map(Into::into).collect()
}

/// A labelled table of mixed values, like a pandas DataFrame.
///
/// The index has one or more levels. A default `RangeIndex` is one unnamed
/// level of integers from zero. Columns are stored by column.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub index_names: Vec<Option<String>>,
    /// `index[level][row]`.
    pub index: Vec<Vec<Cell>>,
    /// The name of the column axis. pandas shows it in HTML, not in CSV.
    pub columns_name: Option<String>,
    pub columns: Vec<String>,
    /// `data[column][row]`.
    pub data: Vec<Vec<Cell>>,
}

impl Table {
    /// A table with one index level and no columns yet.
    pub fn new(index_name: Option<&str>, index: Vec<Cell>) -> Self {
        Self {
            index_names: vec![index_name.map(str::to_string)],
            index: vec![index],
            columns_name: None,
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    /// A table with a default `RangeIndex` of `rows` rows.
    pub fn range(rows: usize) -> Self {
        Self::new(None, (0..rows as i64).map(Cell::Int).collect())
    }

    /// A table with a multi-level index.
    pub fn multi(index_names: Vec<Option<String>>, index: Vec<Vec<Cell>>) -> Self {
        assert_eq!(index_names.len(), index.len(), "each index level needs a name slot");
        assert!(!index.is_empty(), "a table needs at least one index level");
        let rows = index[0].len();
        assert!(index.iter().all(|level| level.len() == rows), "index levels must have equal length");
        Self { index_names, index, columns_name: None, columns: Vec::new(), data: Vec::new() }
    }

    /// A table of text labels, one row per label.
    pub fn labelled<S: AsRef<str>>(index_name: Option<&str>, labels: &[S]) -> Self {
        Self::new(index_name, labels.iter().map(|label| Cell::Str(label.as_ref().to_string())).collect())
    }

    /// A date-indexed table that copies a numeric frame.
    pub fn from_frame(frame: &Frame, index_name: Option<&str>) -> Self {
        let mut table = Self::new(index_name, frame.index.iter().map(|date| Cell::Date(*date)).collect());
        for (j, column) in frame.columns.iter().enumerate() {
            table.push(column, frame.values.column(j).iter().map(|value| Cell::Float(*value)).collect());
        }
        table
    }

    pub fn nrows(&self) -> usize {
        self.index.first().map_or(0, Vec::len)
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nrows() == 0 || self.ncols() == 0
    }

    /// Add a column. Replaces an existing column with the same name in place.
    pub fn push(&mut self, name: impl Into<String>, values: Vec<Cell>) {
        let name = name.into();
        assert_eq!(values.len(), self.nrows(), "column {name} must match the index length");
        match self.position(&name) {
            Some(j) => self.data[j] = values,
            None => {
                self.columns.push(name);
                self.data.push(values);
            }
        }
    }

    /// Builder form of [`Table::push`].
    pub fn with(mut self, name: impl Into<String>, values: Vec<Cell>) -> Self {
        self.push(name, values);
        self
    }

    pub fn position(&self, column: &str) -> Option<usize> {
        self.columns.iter().position(|name| name == column)
    }

    /// The named column. Panics when it is missing.
    pub fn column(&self, name: &str) -> &[Cell] {
        let j = self.position(name).unwrap_or_else(|| panic!("table column {name:?} is missing"));
        &self.data[j]
    }

    /// The named column as numbers. Missing values become NaN.
    pub fn f64s(&self, name: &str) -> Vec<f64> {
        self.column(name).iter().map(|cell| cell.as_f64().unwrap_or(f64::NAN)).collect()
    }

    /// The first row whose first index level is the text `label`.
    pub fn row_of(&self, label: &str) -> Option<usize> {
        self.index.first()?.iter().position(|cell| cell.as_str() == Some(label))
    }

    /// The labels of the first index level as text.
    pub fn index_labels(&self) -> Vec<String> {
        self.index.first().map_or_else(Vec::new, |level| level.iter().map(Cell::py_str).collect())
    }

    /// The value in the row labelled `row` and the named column. Panics when either is missing.
    pub fn cell(&self, row: &str, column: &str) -> &Cell {
        let i = self.row_of(row).unwrap_or_else(|| panic!("table row {row:?} is missing"));
        &self.column(column)[i]
    }

    /// The number in the row labelled `row` and the named column.
    pub fn f64_at(&self, row: &str, column: &str) -> f64 {
        self.cell(row, column).as_f64().unwrap_or(f64::NAN)
    }

    /// The rows at the given positions, in that order.
    pub fn take_rows(&self, rows: &[usize]) -> Table {
        let pick = |values: &Vec<Cell>| rows.iter().map(|i| values[*i].clone()).collect::<Vec<_>>();
        Table {
            index_names: self.index_names.clone(),
            index: self.index.iter().map(pick).collect(),
            columns_name: self.columns_name.clone(),
            columns: self.columns.clone(),
            data: self.data.iter().map(pick).collect(),
        }
    }

    /// The named columns, in the given order. Panics when one is missing.
    pub fn select<S: AsRef<str>>(&self, names: &[S]) -> Table {
        let mut table = Table {
            index_names: self.index_names.clone(),
            index: self.index.clone(),
            columns_name: self.columns_name.clone(),
            columns: Vec::new(),
            data: Vec::new(),
        };
        for name in names {
            table.push(name.as_ref(), self.column(name.as_ref()).to_vec());
        }
        table
    }

    /// Rename columns with `(old, new)` pairs. Unknown names are ignored, as in pandas.
    pub fn rename_columns(&mut self, pairs: &[(&str, &str)]) {
        for column in &mut self.columns {
            if let Some((_, new)) = pairs.iter().find(|(old, _)| old == column) {
                *column = new.to_string();
            }
        }
    }

    /// Rename first-level index labels with `(old, new)` pairs.
    pub fn rename_index(&mut self, pairs: &[(&str, &str)]) {
        if let Some(level) = self.index.first_mut() {
            for cell in level {
                if let Cell::Str(label) = cell
                    && let Some((_, new)) = pairs.iter().find(|(old, _)| old == label)
                {
                    *label = new.to_string();
                }
            }
        }
    }

    /// Python `csv_text(table, index_label)`: CSV with the index first, CRLF line ends,
    /// and an apostrophe before formula-like text.
    pub fn to_csv(&self, index_label: Option<&str>) -> String {
        let mut text = String::new();
        let mut header: Vec<String> = match index_label {
            Some(label) => vec![text_label(label)],
            None => self.index_names.iter().map(|name| name.as_deref().map(text_label).unwrap_or_default()).collect(),
        };
        header.extend(self.columns.iter().map(|column| text_label(column)));
        write_csv_row(&header, &mut text);
        for i in 0..self.nrows() {
            let mut row: Vec<String> = self.index.iter().map(|level| level[i].csv_text()).collect();
            row.extend(self.data.iter().map(|column| column[i].csv_text()));
            write_csv_row(&row, &mut text);
        }
        text
    }
}

fn write_csv_row(fields: &[String], text: &mut String) {
    if fields.len() == 1 && fields[0].is_empty() {
        text.push_str("\"\"");
    } else {
        text.push_str(&fields.iter().map(|field| csv_field(field)).collect::<Vec<_>>().join(","));
    }
    text.push_str("\r\n");
}

/// Quote a CSV field as Python's `csv` module does with `QUOTE_MINIMAL`.
pub fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Prefix formula-like text with an apostrophe so spreadsheets keep it as text.
pub fn text_label(value: &str) -> String {
    if value.trim_start().starts_with(['=', '+', '-', '@']) { format!("'{value}") } else { value.to_string() }
}

/// pandas CSV text for a float: `repr`, or empty for NaN.
pub fn float_text(value: f64) -> String {
    if value.is_nan() { String::new() } else { pyfmt::repr(value) }
}

/// Arithmetic mean. NaN for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Variance with `ddof` delta degrees of freedom. NaN when `len <= ddof`.
pub fn var(values: &[f64], ddof: usize) -> f64 {
    if values.len() <= ddof {
        return f64::NAN;
    }
    let center = mean(values);
    values.iter().map(|value| (value - center).powi(2)).sum::<f64>() / (values.len() - ddof) as f64
}

/// Standard deviation with `ddof` delta degrees of freedom.
pub fn std(values: &[f64], ddof: usize) -> f64 {
    var(values, ddof).sqrt()
}

/// Sample covariance of the columns, with `ddof=1`.
pub fn covariance(values: &DMatrix<f64>) -> DMatrix<f64> {
    let rows = values.nrows();
    let columns = values.ncols();
    if rows < 2 {
        return DMatrix::from_element(columns, columns, f64::NAN);
    }
    let mut centered = values.clone();
    for j in 0..columns {
        let center = mean(values.column(j).as_slice());
        centered.column_mut(j).add_scalar_mut(-center);
    }
    let mut result = centered.transpose() * &centered / (rows - 1) as f64;
    // Make the result exactly symmetric, as pandas computes each pair once.
    for i in 0..columns {
        for j in 0..i {
            result[(j, i)] = result[(i, j)];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(text: &str) -> Date {
        NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn returns_and_slices_follow_pandas() {
        let frame = Frame::from_columns(
            vec![day("2020-01-01"), day("2020-01-02"), day("2020-01-03")],
            vec![("A".into(), vec![100.0, 110.0, 99.0]), ("B".into(), vec![10.0, 10.0, 12.0])],
        );
        let returns = frame.pct_change();
        assert_eq!(returns.index, vec![day("2020-01-02"), day("2020-01-03")]);
        assert!((returns.get(0, 0) - 0.1).abs() < 1e-15);
        assert!((returns.get(1, 0) + 0.1).abs() < 1e-15);
        assert_eq!(frame.rows_through(day("2020-01-02")).nrows(), 2);
        assert_eq!(frame.select(&["B"]).unwrap().col(0), vec![10.0, 10.0, 12.0]);
        assert_eq!(frame.drop_columns(&["A"]).columns, vec!["B".to_string()]);
    }

    #[test]
    fn covariance_matches_sample_formula() {
        let values = DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0, 4.0]);
        let result = covariance(&values);
        assert!((result[(0, 0)] - var(&[1.0, 2.0, 3.0, 4.0], 1)).abs() < 1e-15);
        assert!((result[(0, 1)] - 5.0 / 3.0).abs() < 1e-15);
        assert_eq!(result[(0, 1)], result[(1, 0)]);
    }

    #[test]
    fn csv_text_escapes_formula_labels_and_quotes_fields() {
        let table = Table::labelled(Some("ticker"), &["=HYPERLINK()", "A,B"])
            .with("weight", cells(&[0.5, f64::NAN]))
            .with("note", vec![Cell::from("-x"), Cell::Null])
            .with("count", cells(&[1_i64, 2]));
        assert_eq!(table.to_csv(None), "ticker,weight,note,count\r\n'=HYPERLINK(),0.5,'-x,1\r\n\"A,B\",,,2\r\n");
        let frame = Frame::from_columns(vec![day("2020-01-01")], vec![("@X".into(), vec![1e-7])]);
        assert_eq!(frame.to_csv("Date"), "Date,'@X\r\n2020-01-01,1e-07\r\n");
        assert_eq!(Table::range(2).with("x", cells(&[true, false])).to_csv(None), ",x\r\n0,True\r\n1,False\r\n");
    }
}
