//! Shared helpers for parity tests against golden results of the Python release.
//!
//! Golden JSON files store tables in pandas "split" form: `index_names`,
//! `index`, `columns` and `data`. Dates are ISO text and NaN is `null`.

#![allow(dead_code)]

use std::path::PathBuf;

use efficient_frontier::frame::{Cell, Frame, Table, iso};
use efficient_frontier::pyfmt::{Dict, Json};
use serde_json::{Value, json};

pub fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden")
}

/// A golden JSON file.
pub fn fixture(name: &str) -> Value {
    let path = golden_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden JSON")
}

/// Prices from a golden CSV, loaded with the app's own CSV reader.
pub fn prices(name: &str) -> Frame {
    efficient_frontier::data::load_csv_path(&golden_dir().join(name)).expect("golden prices")
}

pub fn cell_json(cell: &Cell) -> Value {
    match cell {
        Cell::Null => Value::Null,
        Cell::Bool(value) => json!(value),
        Cell::Int(value) => json!(value),
        Cell::Float(value) => {
            if value.is_nan() {
                Value::Null
            } else {
                json!(value)
            }
        }
        Cell::Str(value) => json!(value),
        Cell::Date(value) => json!(iso(*value)),
    }
}

/// A table in pandas split form.
pub fn table_json(table: &Table) -> Value {
    let rows = table.nrows();
    let index: Vec<Value> = (0..rows)
        .map(|i| {
            if table.index.len() == 1 {
                cell_json(&table.index[0][i])
            } else {
                Value::Array(table.index.iter().map(|level| cell_json(&level[i])).collect())
            }
        })
        .collect();
    json!({
        "index_names": table.index_names,
        "index": index,
        "columns": table.columns,
        "data": (0..rows).map(|i| table.data.iter().map(|column| cell_json(&column[i])).collect::<Vec<_>>()).collect::<Vec<_>>(),
    })
}

/// A date frame in pandas split form.
pub fn frame_json(frame: &Frame) -> Value {
    json!({
        "index": frame.index.iter().map(|date| iso(*date)).collect::<Vec<_>>(),
        "columns": frame.columns,
        "data": (0..frame.nrows()).map(|i| frame.row(i).iter().map(|v| if v.is_nan() { Value::Null } else { json!(v) }).collect::<Vec<_>>()).collect::<Vec<_>>(),
    })
}

pub fn dict_json(dict: &Dict) -> Value {
    json_value(&Json::Dict(dict.clone()))
}

pub fn json_value(value: &Json) -> Value {
    match value {
        Json::Null => Value::Null,
        Json::Bool(value) => json!(value),
        Json::Int(value) => json!(value),
        Json::Float(value) => {
            if value.is_nan() {
                Value::Null
            } else {
                json!(value)
            }
        }
        Json::Str(value) => json!(value),
        Json::List(values) => Value::Array(values.iter().map(json_value).collect()),
        Json::Dict(values) => Value::Object(values.iter().map(|(k, v)| (k.to_string(), json_value(v))).collect()),
    }
}

/// Compare two JSON values. Numbers may differ by `tolerance`; everything else
/// must match exactly. Returns the largest numeric difference.
pub fn assert_close(actual: &Value, expected: &Value, tolerance: f64, path: &str) -> f64 {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            let difference = (a - e).abs();
            assert!(difference <= tolerance, "{path}: {a} vs expected {e} (difference {difference:e})");
            difference
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: length");
            a.iter()
                .zip(e)
                .enumerate()
                .map(|(i, (x, y))| assert_close(x, y, tolerance, &format!("{path}[{i}]")))
                .fold(0.0, f64::max)
        }
        (Value::Object(a), Value::Object(e)) => {
            let mut keys_a: Vec<&String> = a.keys().collect();
            let mut keys_e: Vec<&String> = e.keys().collect();
            keys_a.sort();
            keys_e.sort();
            assert_eq!(keys_a, keys_e, "{path}: keys");
            e.iter().map(|(k, v)| assert_close(&a[k], v, tolerance, &format!("{path}.{k}"))).fold(0.0, f64::max)
        }
        _ => {
            assert_eq!(actual, expected, "{path}");
            0.0
        }
    }
}
