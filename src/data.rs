//! Explicit, complete price observations for portfolio research.

use std::collections::HashSet;
use std::io::{Cursor, Read};
use std::path::Path;

use chrono::{DateTime, Datelike, Days, NaiveDateTime};

use crate::error::Result;
use crate::frame::{Date, Frame, iso};
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

/// One date layout. `Zoned` keeps the local wall time of an offset timestamp.
#[derive(Clone, Debug)]
enum DateFormat {
    Date(String),
    DateTime(String),
    Zoned,
    Month(String),
    Year,
}

/// Date layouts in pandas' order of preference: ISO first, then month-first
/// before day-first for numeric dates, then month names.
/// Chrono's `%B` accepts full and short month names, so May cannot select a short-only layout.
const DATE_PATTERNS: [&str; 22] = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%m-%d-%y",
    "%d-%m-%y",
    "%m.%d.%Y",
    "%d.%m.%Y",
    "%d-%B-%Y",
    "%d-%B-%y",
    "%d %B %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%B-%d-%Y",
    "%Y-%B-%d",
    "%d %B %y",
];

/// Time layouts after a date, such as Excel's `1/13/2020 0:00`.
const TIME_PATTERNS: [&str; 9] = [
    " %H:%M:%S",
    " %H:%M",
    "T%H:%M:%S",
    "T%H:%M",
    " %H:%M:%S%.f",
    "T%H:%M:%S%.f",
    " %I:%M %p",
    " %I:%M:%S %p",
    " %I %p",
];

static DATE_FORMATS: std::sync::LazyLock<Vec<DateFormat>> = std::sync::LazyLock::new(|| {
    let mut formats: Vec<DateFormat> = Vec::new();
    for date in DATE_PATTERNS {
        formats.push(DateFormat::Date(date.to_string()));
        formats.extend(TIME_PATTERNS.iter().map(|time| DateFormat::DateTime(format!("{date}{time}"))));
        if date == "%Y-%m-%d" {
            formats.push(DateFormat::Zoned);
        }
    }
    formats.extend(
        ["%Y-%m", "%m/%Y", "%m-%Y", "%B %Y", "%B-%Y", "%Y/%m"].map(|pattern| DateFormat::Month(pattern.to_string())),
    );
    formats.push(DateFormat::Year);
    formats
});

/// The date of `value` in one layout, and its UTC offset in seconds for a zoned timestamp.
fn parse_with(value: &str, format: &DateFormat) -> Option<(Date, Option<i32>)> {
    // chrono's %Y also reads two digits; leave those to the %y layouts.
    let full_year = |pattern: &str, date: Date| !pattern.contains("%Y") || date.year() >= 1000;
    match format {
        DateFormat::Date(pattern) => {
            Date::parse_from_str(value, pattern).ok().filter(|date| full_year(pattern, *date)).map(|date| (date, None))
        }
        DateFormat::DateTime(pattern) => NaiveDateTime::parse_from_str(value, pattern)
            .ok()
            .map(|stamp| stamp.date())
            .filter(|date| full_year(pattern, *date))
            .map(|date| (date, None)),
        DateFormat::Zoned => DateTime::parse_from_rfc3339(value)
            .or_else(|_| DateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%:z"))
            .or_else(|_| DateTime::parse_from_str(value, "%Y-%m-%dT%H:%M%:z"))
            .ok()
            .map(|stamp| (stamp.naive_local().date(), Some(stamp.offset().local_minus_utc()))),
        DateFormat::Month(pattern) => Date::parse_from_str(&format!("{value}-01"), &format!("{pattern}-%d"))
            .ok()
            .filter(|date| full_year(pattern, *date))
            .map(|date| (date, None)),
        DateFormat::Year => (value.len() == 4 && value.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| Date::from_ymd_opt(value.parse().ok()?, 1, 1))
            .flatten()
            .map(|date| (date, None)),
    }
}

/// Parse dates as pandas does: infer one layout from the first value, then
/// apply it to every row. Times are dropped and offsets keep their local date.
/// Mixed UTC offsets are invalid, as in pandas.
pub fn parse_price_dates(values: &[String]) -> Result<Vec<Date>> {
    let values: Vec<&str> = values.iter().map(|value| value.trim()).collect();
    let present: Vec<&str> = values.iter().copied().filter(|value| !value.is_empty()).collect();
    let invalid = || value_error!("Every row must have a valid date.");
    let mut dates = Vec::with_capacity(values.len());
    if let Some(first) = present.first() {
        let format = DATE_FORMATS.iter().find(|format| parse_with(first, format).is_some()).ok_or_else(invalid)?;
        let mut offsets = HashSet::new();
        for value in &present {
            let (date, offset) = parse_with(value, format).ok_or_else(invalid)?;
            offsets.insert(offset);
            dates.push(date);
        }
        if offsets.len() > 1 {
            return Err(invalid());
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

/// Python's `str.casefold()` for header comparisons: lower case plus the full
/// case foldings that change length or differ from lower case.
pub(crate) fn casefold(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for character in text.to_lowercase().chars() {
        match character {
            'ß' => folded.push_str("ss"),
            'ς' => folded.push('σ'),
            'ﬀ' => folded.push_str("ff"),
            'ﬁ' => folded.push_str("fi"),
            'ﬂ' => folded.push_str("fl"),
            'ﬃ' => folded.push_str("ffi"),
            'ﬄ' => folded.push_str("ffl"),
            'ﬅ' | 'ﬆ' => folded.push_str("st"),
            'ŉ' => folded.push_str("ʼn"),
            other => folded.push(other),
        }
    }
    folded
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
        let crlf = c == '\r' && chars.peek() == Some(&'\n');
        if crlf {
            chars.next();
        }
        // Python reads lines that end at "\n", so a carriage return outside quotes must end the line.
        if c == '\r' && !crlf && chars.peek().is_some() && !matches!(state, State::InQuoted) {
            return Err("new-line character seen in unquoted field".into());
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
                '\r' => field.push_str(if crlf { "\r\n" } else { "\r" }),
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
    if headers.is_empty() || casefold(&headers[0]) != "date" {
        return Err(value_error!("The first CSV column must be named Date."));
    }
    let folded: HashSet<String> = headers.iter().map(|header| casefold(header)).collect();
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
    let symbols = parse_tickers(&tickers.join(" "));
    if symbols.is_empty() {
        return Err(value_error!("Enter at least one ticker."));
    }
    if start >= end {
        return Err(value_error!("Choose valid start and end dates, with the start before the end."));
    }
    // Like yfinance, a failed symbol has no rows, so even a network failure arrives here as missing prices.
    let prices = crate::net::yahoo_adjusted_closes(&symbols, start, end);
    if prices.nrows() == 0 {
        return Err(value_error!("Yahoo returned no prices. Check the tickers and date range, or upload a CSV."));
    }
    let missing: Vec<&str> = symbols
        .iter()
        .enumerate()
        .filter(|(j, _)| prices.values.column(*j).iter().all(|value| value.is_nan()))
        .map(|(_, symbol)| symbol.as_str())
        .collect();
    if !missing.is_empty() && !allow_incomplete {
        return Err(value_error!(
            "Yahoo returned no prices for: {}. Check these symbols and their available history, or upload a CSV.",
            missing.join(", ")
        ));
    }
    Ok(prices)
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
    validate_benchmark_dates(expected_dates)?;
    let symbols = BENCHMARK_SYMBOLS.map(String::from);
    let (first, last) = (expected_dates[0], expected_dates[expected_dates.len() - 1]);
    let prices = download_adjusted_close(&symbols, first, last + Days::new(1), false)?;
    let missing: Vec<&Date> = expected_dates.iter().filter(|date| prices.index.binary_search(date).is_err()).collect();
    if !missing.is_empty() {
        let dates: Vec<String> = missing.iter().take(3).map(|date| iso(**date)).collect();
        return Err(value_error!(
            "Yahoo benchmarks are missing {} requested dates, including: {}.",
            missing.len(),
            dates.join(", ")
        ));
    }
    let rows: Vec<Vec<f64>> = expected_dates
        .iter()
        .map(|date| prices.row(prices.index.binary_search(date).expect("every expected date is present")))
        .collect();
    validate_benchmark_prices(&Frame::from_rows(expected_dates.to_vec(), prices.columns.clone(), &rows), expected_dates)
}

/// The historical list, bundled so that binaries work from any directory.
const ORIGINAL_HOLDINGS: &[u8] = include_bytes!("../spy_holdings.ods");
const TABLE_NAMESPACE: &str = "urn:oasis:names:tc:opendocument:xmlns:table:1.0";

/// Read the historical list in `spy_holdings.ods`, optionally with known current Yahoo symbols.
///
/// `None` reads the bundled copy of the repository file.
pub fn original_tickers(path: Option<&Path>, current_symbols: bool) -> Result<Vec<String>> {
    let unreadable = || value_error!("The original spy_holdings.ods ticker list could not be read.");
    let cells = match path {
        Some(path) => ods_first_cells(&std::fs::read(path).map_err(|_| unreadable())?),
        None => ods_first_cells(ORIGINAL_HOLDINGS),
    }
    .ok_or_else(unreadable)?;
    let symbols = cells.get(1..).unwrap_or_default();
    if cells.first().map(String::as_str) != Some("Tickers")
        || !symbols.iter().all(|ticker| !ticker.is_empty() && ticker.chars().all(|c| c.is_ascii_uppercase()))
    {
        return Err(value_error!("The original holdings file must have a Tickers column containing stock symbols."));
    }
    let mut tickers: Vec<String> = Vec::new();
    for ticker in symbols {
        if !tickers.contains(ticker) {
            tickers.push(ticker.clone());
        }
    }
    if tickers.is_empty() {
        return Err(value_error!("The original holdings file contains no tickers."));
    }
    if current_symbols {
        // Marsh changed its NYSE symbol on 2026-01-14; the ODS stays historical.
        // Source: https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html
        for ticker in &mut tickers {
            if ticker == "MMC" {
                *ticker = "MRSH".into();
            }
        }
    }
    Ok(tickers)
}

/// The nonblank text of the first cell in each table row of an ODS file.
fn ods_first_cells(content: &[u8]) -> Option<Vec<String>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(content)).ok()?;
    let mut xml = String::new();
    archive.by_name("content.xml").ok()?.read_to_string(&mut xml).ok()?;
    let document = roxmltree::Document::parse(&xml).ok()?;
    let mut cells = Vec::new();
    for row in document.descendants().filter(|node| node.has_tag_name((TABLE_NAMESPACE, "table-row"))) {
        if let Some(cell) = row.children().find(|node| node.has_tag_name((TABLE_NAMESPACE, "table-cell"))) {
            let text: String =
                cell.descendants().filter(|node| node.is_text()).filter_map(|node| node.text()).collect();
            let value = crate::net::py_strip(&text);
            if !value.is_empty() {
                cells.push(value.to_string());
            }
        }
    }
    Some(cells)
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
    fn csv_loading_preserves_monthly_dates_and_prices() {
        for first_month in 1..=12 {
            let first = Date::from_ymd_opt(2019, first_month, 1).unwrap();
            let dates: Vec<Date> = (0..13).map(|i| first + chrono::Months::new(i)).collect();
            let expected = Frame::from_rows(
                dates.clone(),
                vec!["AAA".into(), "BBB".into()],
                &(0..13).map(|i| vec![100.0 + i as f64, 200.0 + i as f64]).collect::<Vec<_>>(),
            );
            for pattern in ["%m/%Y", "%-m/%Y", "%m-%Y", "%b %Y", "%b-%Y", "%B %Y", "%B-%Y", "%Y/%m", "%Y-%m"] {
                let mut csv = String::from("Date,AAA,BBB\n");
                for i in [0, 3, 1, 12, 5, 2, 4, 6, 7, 8, 9, 10, 11] {
                    csv.push_str(&format!("{},{},{}\n", dates[i].format(pattern), 100 + i, 200 + i));
                }
                assert_eq!(load_csv(csv.as_bytes()).unwrap(), expected, "{first_month}: {pattern}");
            }
        }
    }

    #[test]
    fn month_names_accept_short_and_full_spellings_in_one_layout() {
        for pattern in [
            "%B %Y",
            "%B-%Y",
            "%d-%B-%Y",
            "%d-%B-%y",
            "%d %B %Y",
            "%B %d %Y",
            "%B %d, %Y",
            "%B-%d-%Y",
            "%Y-%B-%d",
            "%d %B %y",
            "%B %d, %Y %H:%M:%S",
            "%d-%B-%Y %I:%M %p",
        ] {
            for first_month in [1, 5, 12] {
                let first_day = if pattern.contains("%d") { 20 } else { 1 };
                let first = Date::from_ymd_opt(2019, first_month, first_day).unwrap();
                let second = Date::from_ymd_opt(2019, first_month, 1).unwrap() + chrono::Months::new(1);
                for (first_name, second_name) in [("%b", "%b"), ("%b", "%B"), ("%B", "%b"), ("%B", "%B")] {
                    let values = [
                        first.and_hms_opt(23, 30, 0).unwrap().format(&pattern.replace("%B", first_name)).to_string(),
                        second.and_hms_opt(0, 0, 0).unwrap().format(&pattern.replace("%B", second_name)).to_string(),
                    ];
                    assert_eq!(parse_price_dates(&values).unwrap(), [first, second], "{values:?}");
                }
            }
        }
    }

    #[test]
    fn month_names_reject_invalid_dates_and_mixed_layouts() {
        for (first, invalid) in [
            ("May 2019", "June-2019"),
            ("May-2019", "June 2019"),
            ("May 2019", "Januarx 2020"),
            ("May 2019", "June 2019-01"),
            ("May 20, 2020", "June 31, 2020"),
            ("May 20, 2020", "February 29, 2019"),
            ("May 20, 2020", "Januarx 01, 2020"),
            ("May 20, 2020", "June 01 2020"),
            ("May 20, 2020", "01 June 2020"),
            ("May 20, 2020", "June 01, 2020 00:00:00"),
            ("May 20, 2020 00:00:00", "June 01, 2020"),
        ] {
            let values = [first.to_string(), invalid.to_string()];
            assert_eq!(
                parse_price_dates(&values).unwrap_err().message(),
                "Every row must have a valid date.",
                "{values:?}"
            );
        }
    }

    #[test]
    fn csv_loading_rejects_invalid_or_mixed_monthly_dates() {
        for (first, invalid) in [
            ("01/2020", "13/2020"),
            ("1/2020", "0/2020"),
            ("01-2020", "13-2020"),
            ("Jan 2020", "Jax 2020"),
            ("Jan-2020", "Foo-2020"),
            ("January 2020", "Januarx 2020"),
            ("2020/01", "2020/13"),
            ("2020-01", "2020-00"),
            ("01/2020", "02-2020"),
            ("01-2020", "02/2020"),
            ("Jan 2020", "Feb-2020"),
            ("Jan-2020", "Feb 2020"),
            ("2020/01", "2020-02"),
            ("2020-01", "2020/02"),
            ("2020-01", "2020-02-01"),
            ("2020-01-01", "2020-02"),
            ("2020-01-01", "2020-02-30"),
        ] {
            let csv = format!("Date,A\n{first},100\n{invalid},101\n");
            assert_eq!(
                load_csv(csv.as_bytes()).unwrap_err().message(),
                "Every row must have a valid date.",
                "{first}, {invalid}"
            );
        }
        let csv = b"Date,A\n,100\nJan 2020,101\nFeb-2020,102\n";
        assert_eq!(load_csv(csv).unwrap_err().message(), "Every row must have a valid date.");
        let csv = b"Date,A\n,100\nJan 2020,101\nFeb 2020,102\n";
        assert_eq!(
            load_csv(csv).unwrap_err().message(),
            "Every row must have a valid date; blank dates are not allowed."
        );
    }

    #[test]
    fn csv_loading_retains_existing_date_formats() {
        let expected = Frame::from_rows(
            (13..=17).map(|day| Date::from_ymd_opt(2020, 1, day).unwrap()).collect(),
            vec!["A".into()],
            &(13..=17).map(|day| vec![100.0 + day as f64]).collect::<Vec<_>>(),
        );
        for pattern in [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y%m%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%B %d, %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S-05:00",
        ] {
            let mut csv = String::from("Date,A\n");
            for day in [15, 13, 17, 14, 16] {
                let date = Date::from_ymd_opt(2020, 1, day).unwrap().and_hms_opt(23, 30, 0).unwrap();
                csv.push_str(&format!("\"{}\",{}\n", date.format(pattern), 100 + day));
            }
            assert_eq!(load_csv(csv.as_bytes()).unwrap(), expected, "{pattern}");
        }
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
        let day = |y, m, d| Date::from_ymd_opt(y, m, d).unwrap();
        for (first, second) in [
            ("13-Jan-2020", "14-Jan-2020"),
            ("13-Jan-20", "14-Jan-20"),
            ("13-01-2020", "14-01-2020"),
            ("13/01/2020 00:00:00", "14/01/2020 00:00:00"),
            ("1/13/2020 0:00", "1/14/2020 0:00"),
            ("01/13/2020 12:00 AM", "01/14/2020 12:00 AM"),
            ("2020-01-13T00:00", "2020-01-14T00:00"),
            ("2020.01.13", "2020.01.14"),
            ("13 Jan 2020", "14 Jan 2020"),
            ("Jan 13 2020", "Jan 14 2020"),
            ("1/13/20", "1/14/20"),
            ("January 13, 2020", "January 14, 2020"),
        ] {
            assert_eq!(
                parse_price_dates(&values(&[first, second])).unwrap(),
                [day(2020, 1, 13), day(2020, 1, 14)],
                "{first}"
            );
        }
        // pandas reads ambiguous numeric dates month first, and a bare year as 1 January.
        assert_eq!(parse_price_dates(&values(&["01-02-2020", "01-03-2020"])).unwrap()[0], day(2020, 1, 2));
        assert_eq!(parse_price_dates(&values(&["02.01.2020"])).unwrap()[0], day(2020, 2, 1));
        assert_eq!(parse_price_dates(&values(&["2020", "2021"])).unwrap()[1], day(2021, 1, 1));
        assert!(parse_price_dates(&values(&["2020-01-02T10:30:00+01:00", "2020-01-03T10:30:00+02:00"])).is_err());
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
        assert!(read_csv_rows("Date,A\r2020-01-01,1\r").is_err());
        assert!(read_csv_rows("Date,A\r\n2020-01-01,1\r").is_ok());
        assert_eq!(casefold("STRASSE"), casefold("straße"));
        assert_eq!(read_csv_rows("\"a\rb\",\"c\r\nd\"\n").unwrap(), vec![vec!["a\rb", "c\r\nd"]]);
    }
}
