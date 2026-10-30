//! Explicit, complete price observations for portfolio research.

use std::collections::HashSet;
use std::path::Path;

use chrono::{DateTime, NaiveDateTime};

use crate::error::Result;
use crate::frame::{Date, Frame};
use crate::value_error;

pub const BENCHMARK_SYMBOLS: [&str; 2] = ["SPY", "QQQ"];

const DEMO_PRICES: &str = include_str!("../assets/demo/demo_prices.csv");

/// Normalize one Yahoo symbol. Explicit aliases convert class-share periods.
pub fn normalize_ticker(symbol: &str) -> String {
    let symbol = symbol.trim().to_uppercase();
    match symbol.as_str() {
        "BRK.A" => "BRK-A".into(),
        "BRK.B" => "BRK-B".into(),
        "BF.A" => "BF-A".into(),
        "BF.B" => "BF-B".into(),
        _ => symbol,
    }
}

/// Normalize Yahoo symbols and remove duplicates. Preserve exchange suffixes and input order.
pub fn parse_tickers(text: &str) -> Vec<String> {
    let mut symbols: Vec<String> = Vec::new();
    for token in text.trim().split(|c: char| c == ',' || c.is_whitespace()).filter(|token| !token.is_empty()) {
        let symbol = normalize_ticker(token);
        if !symbols.contains(&symbol) {
            symbols.push(symbol);
        }
    }
    symbols
}

/// Flag long unchanged-price runs in a complete price frame without altering it.
pub fn stale_price_notes(prices: &Frame) -> Vec<String> {
    let mut notes = Vec::new();
    for (j, symbol) in prices.columns.iter().enumerate() {
        let (mut run, mut longest) = (0, 0);
        for i in 1..prices.nrows() {
            run = if prices.get(i, j) == prices.get(i - 1, j) { run + 1 } else { 0 };
            longest = longest.max(run);
        }
        if longest >= 20 {
            notes.push(format!(
                "{symbol}: {longest} consecutive price intervals had unchanged closes. \
                 Flat stretches can understate measured volatility. Check for stale or illiquid quotes \
                 before interpreting fitted risk. These prices remain in the study."
            ));
        }
    }
    notes
}

/// One parsed date format. `Zoned` keeps the local wall time of an offset timestamp.
#[derive(Clone, Copy)]
enum DateFormat {
    Date(&'static str),
    DateTime(&'static str),
    Zoned,
    Month,
}

const DATE_FORMATS: [DateFormat; 12] = [
    DateFormat::Date("%Y-%m-%d"),
    DateFormat::DateTime("%Y-%m-%d %H:%M:%S"),
    DateFormat::DateTime("%Y-%m-%d %H:%M"),
    DateFormat::DateTime("%Y-%m-%dT%H:%M:%S"),
    DateFormat::DateTime("%Y-%m-%d %H:%M:%S%.f"),
    DateFormat::Zoned,
    DateFormat::Date("%Y/%m/%d"),
    DateFormat::Date("%m/%d/%Y"),
    DateFormat::Date("%d/%m/%Y"),
    DateFormat::Date("%Y%m%d"),
    DateFormat::Date("%b %d, %Y"),
    DateFormat::Month,
];

fn parse_with(value: &str, format: DateFormat) -> Option<Date> {
    match format {
        DateFormat::Date(pattern) => Date::parse_from_str(value, pattern).ok(),
        DateFormat::DateTime(pattern) => NaiveDateTime::parse_from_str(value, pattern).ok().map(|stamp| stamp.date()),
        DateFormat::Zoned => DateTime::parse_from_rfc3339(value)
            .or_else(|_| DateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%:z"))
            .ok()
            .map(|stamp| stamp.naive_local().date()),
        DateFormat::Month => Date::parse_from_str(&format!("{value}-01"), "%Y-%m-%d").ok(),
    }
}

/// Parse dates as pandas does: infer one format from the first value, then
/// apply it to every row. Times are dropped and offsets keep their local date.
pub fn parse_price_dates(values: &[String]) -> Result<Vec<Date>> {
    let values: Vec<&str> = values.iter().map(|value| value.trim()).collect();
    let present: Vec<&str> = values.iter().copied().filter(|value| !value.is_empty()).collect();
    let invalid = || value_error!("Every row must have a valid date.");
    let mut dates = Vec::with_capacity(values.len());
    if let Some(first) = present.first() {
        let format = DATE_FORMATS.iter().find(|format| parse_with(first, **format).is_some()).ok_or_else(invalid)?;
        for value in &present {
            dates.push(parse_with(value, *format).ok_or_else(invalid)?);
        }
    }
    if present.len() != values.len() {
        return Err(value_error!("Every row must have a valid date; blank dates are not allowed."));
    }
    let unique: HashSet<&Date> = dates.iter().collect();
    if unique.len() != dates.len() {
        return Err(value_error!("Duplicate dates are not allowed; provide one price per asset per day."));
    }
    Ok(dates)
}

/// Parse a price cell as `pandas.to_numeric`: an empty cell is missing.
fn parse_number(cell: &str) -> Option<f64> {
    if cell.is_empty() {
        return Some(f64::NAN);
    }
    let text = cell.trim();
    let unsigned = text.trim_start_matches(['+', '-']).to_ascii_lowercase();
    if text.is_empty() || unsigned == "nan" {
        return None;
    }
    text.parse::<f64>().ok()
}

fn check_columns(columns: &[String]) -> Result<Vec<String>> {
    let columns: Vec<String> = columns.iter().map(|column| column.trim().to_string()).collect();
    if columns.is_empty() {
        return Err(value_error!("Provide at least one asset."));
    }
    let unique: HashSet<&String> = columns.iter().collect();
    if columns.iter().any(String::is_empty) || unique.len() != columns.len() {
        return Err(value_error!("Asset column names must be nonempty and unique."));
    }
    Ok(columns)
}

/// Check numeric prices, then sort them by date. Never fill gaps or discard an asset or row.
fn finish_prices(index: Vec<Date>, columns: Vec<String>, rows: Vec<Vec<f64>>) -> Result<Frame> {
    let frame = Frame::from_rows(index, columns, &rows);
    let missing: Vec<&str> = (0..frame.ncols())
        .filter(|j| frame.values.column(*j).iter().any(|value| value.is_nan()))
        .map(|j| frame.columns[j].as_str())
        .collect();
    if !missing.is_empty() {
        return Err(value_error!(
            "Missing prices for: {}. Choose a common date range with complete \
             history or correct the CSV; missing prices are never filled.",
            missing.join(", ")
        ));
    }
    if frame.values.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(value_error!("All prices must be finite and strictly positive."));
    }
    let mut order: Vec<usize> = (0..frame.nrows()).collect();
    order.sort_by_key(|i| frame.index[*i]);
    Ok(Frame::from_rows(
        order.iter().map(|i| frame.index[*i]).collect(),
        frame.columns.clone(),
        &order.iter().map(|i| frame.row(*i)).collect::<Vec<_>>(),
    ))
}

/// Validate text prices from a file: dates, then at least five rows, then numbers.
pub fn validate_text_prices(dates: &[String], columns: &[String], rows: &[Vec<String>]) -> Result<Frame> {
    let columns = check_columns(columns)?;
    let index = parse_price_dates(dates)?;
    if index.len() < 5 {
        return Err(value_error!("Provide at least 5 price observations for every asset."));
    }
    let values = rows
        .iter()
        .map(|row| row.iter().map(|cell| parse_number(cell)).collect::<Option<Vec<_>>>())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| value_error!("All prices must be numeric values."))?;
    finish_prices(index, columns, values)
}

/// Return numeric prices in date order. Never fill gaps or discard an asset or row.
pub fn validate_prices(prices: &Frame) -> Result<Frame> {
    let columns = check_columns(&prices.columns)?;
    let unique: HashSet<&Date> = prices.index.iter().collect();
    if unique.len() != prices.nrows() {
        return Err(value_error!("Duplicate dates are not allowed; provide one price per asset per day."));
    }
    if prices.nrows() < 5 {
        return Err(value_error!("Provide at least 5 price observations for every asset."));
    }
    finish_prices(prices.index.clone(), columns, (0..prices.nrows()).map(|i| prices.row(i)).collect())
}

/// Read CSV text as Python's `csv.reader` does: a blank line is an empty row,
/// and a malformed quote is an error.
pub fn read_csv_rows(text: &str) -> std::result::Result<Vec<Vec<String>>, String> {
    #[derive(PartialEq)]
    enum State {
        StartRecord,
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
    }
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut state = State::StartRecord;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let newline = c == '\n' || c == '\r';
        if newline && c == '\r' && chars.peek() == Some(&'\n') {
            chars.next();
        }
        match state {
            State::StartRecord | State::StartField if newline => {
                if state == State::StartField {
                    row.push(std::mem::take(&mut field));
                }
                rows.push(std::mem::take(&mut row));
                state = State::StartRecord;
            }
            State::StartRecord | State::StartField => match c {
                '"' => state = State::InQuoted,
                ',' => {
                    row.push(String::new());
                    state = State::StartField;
                }
                _ => {
                    field.push(c);
                    state = State::InField;
                }
            },
            State::InField => match c {
                _ if newline => {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    state = State::StartRecord;
                }
                ',' => {
                    row.push(std::mem::take(&mut field));
                    state = State::StartField;
                }
                _ => field.push(c),
            },
            State::InQuoted => match c {
                '"' => state = State::QuoteInQuoted,
                '\r' => field.push_str(if newline { "\r\n" } else { "\r" }),
                _ => field.push(c),
            },
            State::QuoteInQuoted => match c {
                '"' => {
                    field.push('"');
                    state = State::InQuoted;
                }
                ',' => {
                    row.push(std::mem::take(&mut field));
                    state = State::StartField;
                }
                _ if newline => {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                    state = State::StartRecord;
                }
                _ => return Err("',' expected after '\"'".into()),
            },
        }
    }
    match state {
        State::InQuoted => return Err("unexpected end of data".into()),
        State::StartRecord => {}
        _ => {
            row.push(field);
            rows.push(row);
        }
    }
    Ok(rows)
}

/// Load daily, weekly, or monthly prices from CSV bytes with Date as the first column.
pub fn load_csv(content: &[u8]) -> Result<Frame> {
    let unreadable = || value_error!("Upload a nonempty UTF-8 CSV with Date as the first column.");
    let text = std::str::from_utf8(content).map_err(|_| unreadable())?;
    let text = text.trim_start_matches('\u{feff}');
    let mut rows = read_csv_rows(text).map_err(|_| unreadable())?;
    if rows.is_empty() {
        return Err(unreadable());
    }
    let headers: Vec<String> = rows.remove(0).iter().map(|header| header.trim().to_string()).collect();
    if headers.is_empty() || headers[0].to_lowercase() != "date" {
        return Err(value_error!("The first CSV column must be named Date."));
    }
    let folded: HashSet<String> = headers.iter().map(|header| header.to_lowercase()).collect();
    if headers.iter().any(String::is_empty) || folded.len() != headers.len() {
        return Err(value_error!("CSV column headers must be nonempty and unique; duplicate headers are invalid."));
    }
    if rows.iter().any(|row| row.len() != headers.len()) {
        return Err(value_error!("Every CSV row must have the same number of columns as its header."));
    }
    let dates: Vec<String> = rows.iter().map(|row| row[0].clone()).collect();
    let cells: Vec<Vec<String>> = rows.iter().map(|row| row[1..].to_vec()).collect();
    validate_text_prices(&dates, &headers[1..], &cells)
}

/// Load a CSV file with [`load_csv`].
pub fn load_csv_path(path: &Path) -> Result<Frame> {
    let content =
        std::fs::read(path).map_err(|_| value_error!("Upload a nonempty UTF-8 CSV with Date as the first column."))?;
    load_csv(&content)
}

fn validate_benchmark_dates(expected_dates: &[Date]) -> Result<()> {
    if expected_dates.len() < 5 || expected_dates.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(value_error!(
            "Benchmark comparison requires at least 5 unique dates in increasing order, without times or a timezone."
        ));
    }
    Ok(())
}

/// Require complete SPY and QQQ prices for exactly the expected dates.
pub fn validate_benchmark_prices(benchmarks: &Frame, expected_dates: &[Date]) -> Result<Frame> {
    validate_benchmark_dates(expected_dates)?;
    let mut prices = validate_prices(benchmarks)?;
    prices.columns = prices.columns.iter().map(|symbol| normalize_ticker(symbol)).collect();
    let unique: HashSet<&str> = prices.columns.iter().map(String::as_str).collect();
    let expected: HashSet<&str> = BENCHMARK_SYMBOLS.into_iter().collect();
    if unique.len() != prices.ncols() || unique != expected {
        return Err(value_error!("Benchmark columns must contain SPY and QQQ exactly once, with no other assets."));
    }
    if prices.index != expected_dates {
        return Err(value_error!(
            "Benchmark dates must exactly match the asset dates. The comparison does not fill or remove dates."
        ));
    }
    Ok(prices.select(&BENCHMARK_SYMBOLS).expect("validated benchmark columns"))
}

/// Reproducible synthetic prices, with no external data or fallback.
///
/// The values are the exact output of the Python release's seeded generator
/// (`numpy.random.default_rng(212)`), bundled so the demo stays identical.
pub fn demo_prices() -> Frame {
    load_csv(DEMO_PRICES.as_bytes()).expect("the bundled demo prices are valid")
}

/// Fetch Yahoo adjusted closes before date selection and price validation.
///
/// Columns follow the requested symbol order and can contain NaN. `end` is exclusive.
pub fn download_adjusted_close(tickers: &[String], start: Date, end: Date, allow_incomplete: bool) -> Result<Frame> {
    let _ = (tickers, start, end, allow_incomplete);
    todo!("network: Yahoo adjusted closes")
}

/// Fetch Yahoo adjusted close prices; an unavailable asset is an error.
pub fn download_prices(tickers: &[String], start: Date, end: Date) -> Result<Frame> {
    validate_prices(&download_adjusted_close(tickers, start, end, false)?)
}

/// Fetch daily adjusted prices and select each exact requested observation.
///
/// The selection excludes extra market dates. A missing requested date is an error.
/// This permits weekly or monthly samples without interpolation or date changes.
pub fn download_benchmarks(expected_dates: &[Date]) -> Result<Frame> {
    let _ = expected_dates;
    todo!("network: Yahoo SPY and QQQ on exact dates")
}

/// Read the historical list in `spy_holdings.ods`, optionally with known current Yahoo symbols.
pub fn original_tickers(path: Option<&Path>, current_symbols: bool) -> Result<Vec<String>> {
    let _ = (path, current_symbols);
    todo!("read spy_holdings.ods")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tickers_are_normalized_deduplicated_and_ordered() {
        assert_eq!(parse_tickers(" aapl, BRK.B\nvod.l  aapl,,msft "), ["AAPL", "BRK-B", "VOD.L", "MSFT"]);
    }

    #[test]
    fn csv_loading_sorts_dates_and_rejects_gaps() {
        let text = "\u{feff}Date, A ,B\r\n2020-01-03,1,2\r\n2020-01-02,1.5,2\r\n2020-01-06,1,2\r\n2020-01-07,1,2\r\n2020-01-08,1,2\r\n";
        let frame = load_csv(text.as_bytes()).unwrap();
        assert_eq!(frame.columns, ["A", "B"]);
        assert_eq!(frame.index[0], Date::from_ymd_opt(2020, 1, 2).unwrap());
        assert_eq!(frame.get(0, 0), 1.5);
        let missing = text.replace("2020-01-06,1,2", "2020-01-06,,2");
        assert_eq!(
            load_csv(missing.as_bytes()).unwrap_err().message(),
            "Missing prices for: A. Choose a common date range with complete history or correct the CSV; missing prices are never filled."
        );
        let blank_line = format!("{text}\n");
        assert_eq!(
            load_csv(blank_line.as_bytes()).unwrap_err().message(),
            "Every CSV row must have the same number of columns as its header."
        );
        assert_eq!(load_csv(b"Day,A\n").unwrap_err().message(), "The first CSV column must be named Date.");
        assert_eq!(
            load_csv(b"Date,A,a\n").unwrap_err().message(),
            "CSV column headers must be nonempty and unique; duplicate headers are invalid."
        );
        let text_value = text.replace("2020-01-06,1,2", "2020-01-06,nan,2");
        assert_eq!(load_csv(text_value.as_bytes()).unwrap_err().message(), "All prices must be numeric values.");
        let negative = text.replace("2020-01-06,1,2", "2020-01-06,-1,2");
        assert_eq!(
            load_csv(negative.as_bytes()).unwrap_err().message(),
            "All prices must be finite and strictly positive."
        );
    }

    #[test]
    fn dates_follow_the_first_value_format() {
        let values = |items: &[&str]| items.iter().map(|item| item.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_price_dates(&values(&["01/02/2020", "01/03/2020"])).unwrap()[0],
            Date::from_ymd_opt(2020, 1, 2).unwrap()
        );
        assert_eq!(
            parse_price_dates(&values(&["13/01/2020", "14/01/2020"])).unwrap()[1],
            Date::from_ymd_opt(2020, 1, 14).unwrap()
        );
        assert_eq!(
            parse_price_dates(&values(&["2020-01-02T23:30:00-05:00"])).unwrap()[0],
            Date::from_ymd_opt(2020, 1, 2).unwrap()
        );
        assert!(parse_price_dates(&values(&["2020-01-02", "2020-01-03 05:00"])).is_err());
        assert_eq!(
            parse_price_dates(&values(&["2020-01-02", ""])).unwrap_err().message(),
            "Every row must have a valid date; blank dates are not allowed."
        );
        assert_eq!(
            parse_price_dates(&values(&["2020-01-02", "2020-01-02"])).unwrap_err().message(),
            "Duplicate dates are not allowed; provide one price per asset per day."
        );
    }

    #[test]
    fn demo_prices_match_the_python_release() {
        let prices = demo_prices();
        assert_eq!((prices.nrows(), prices.ncols()), (1000, 6));
        assert_eq!(prices.get(0, 0), 99.54586087604353);
        assert_eq!(prices.index.last().unwrap(), &Date::from_ymd_opt(2023, 10, 31).unwrap());
    }

    #[test]
    fn csv_reader_follows_python_quoting() {
        assert_eq!(read_csv_rows("a,\"b,\"\"c\"\"\"\n\nd").unwrap(), vec![vec!["a", "b,\"c\""], vec![], vec!["d"]]);
        assert!(read_csv_rows("\"a\"b").is_err());
        assert!(read_csv_rows("\"a").is_err());
    }
}
