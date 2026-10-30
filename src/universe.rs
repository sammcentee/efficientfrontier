//! Current stock universes and explicit price coverage for each security.
//!
//! Port of `efficient_frontier/universe.py`: the Nasdaq-100 list from Nasdaq,
//! the SPY stock holdings from State Street as an S&P 500 proxy, their union,
//! and complete member prices from Yahoo Finance with a coverage record for
//! every member.

use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read, Seek};
use std::time::Duration;

use chrono::Utc;
use roxmltree::{Document, Node};
use serde_json::Value;
use zip::ZipArchive;

use crate::data::{BENCHMARK_SYMBOLS, download_adjusted_close, parse_tickers, validate_prices};
use crate::error::Result;
use crate::frame::{Cell, Date, Frame, Table, iso};
use crate::net::{self, py_strip};
use crate::pyfmt::{Dict, Json};
use crate::{dict, value_error};

pub const NASDAQ100_URL: &str = "https://api.nasdaq.com/api/quote/list-type/nasdaq100";
pub const SPY_HOLDINGS_URL: &str =
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx";
pub const SPY_PRODUCT_URL: &str = "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy";
pub const UNIVERSE_LIMITATION: &str = "This study uses current Nasdaq-100 members, not the members at each historical \
    date. It excludes securities without complete prices for the selected period. Both choices can favor survivors \
    and distort historical results. This is not a historical reconstruction of the Nasdaq-100.";
pub const SPY_LIMITATION: &str = "The S&P 500 stock list uses current SPY equity holdings published by State Street \
    as a proxy, not a certified index constituent list. Cash, funds, derivatives and non-stock rights are excluded. \
    This study uses today's stock universe and requires complete prices for the selected period. Both choices can \
    favor survivors and distort historical results. This is not a historical reconstruction of either index.";

/// The `urlopen` timeout of the Python release.
const PROVIDER_TIMEOUT: Duration = Duration::from_secs(20);
const BATCH_SIZE: usize = 25;
const SHEET_NAMESPACE: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const SPY_HEADER: [&str; 8] =
    ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"];
const SPY_COLUMNS: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];

/// A progress callback that receives `(completed, total, message)`.
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize, &str);
/// A replacement for [`download_adjusted_close`] that receives `(symbols, start, end, allow_incomplete)`.
pub type Download<'a> = &'a mut dyn FnMut(&[String], Date, Date, bool) -> Result<Frame>;

/// One current member of a stock universe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    /// The Yahoo symbol; share classes use a hyphen, as in `BRK-B`.
    pub symbol: String,
    pub name: String,
    /// The provider's sector, or `Unknown`.
    pub industry: String,
}

/// A current member list with its source, date, limitation and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct UniverseSnapshot {
    pub members: Vec<Member>,
    pub source_url: String,
    pub source_date: String,
    pub retrieved_at: String,
    pub name: String,
    pub limitation: String,
    pub source_records: Vec<Dict>,
}

impl UniverseSnapshot {
    /// A snapshot with the default Nasdaq-100 name and limitation and no source records.
    pub fn new(
        members: Vec<Member>,
        source_url: impl Into<String>,
        source_date: impl Into<String>,
        retrieved_at: impl Into<String>,
    ) -> Self {
        Self {
            members,
            source_url: source_url.into(),
            source_date: source_date.into(),
            retrieved_at: retrieved_at.into(),
            name: "Current Nasdaq-100 members".into(),
            limitation: UNIVERSE_LIMITATION.into(),
            source_records: Vec::new(),
        }
    }
}

/// Complete member prices on the market calendar, with the coverage of every member.
#[derive(Clone, Debug, PartialEq)]
pub struct UniversePrices {
    /// Included members, in member order, on the SPY and QQQ dates.
    pub prices: Frame,
    /// One row per member: `symbol`, `name`, `industry`, `status`, `reason`,
    /// `observations`, `first_date` and `last_date`, with pandas column types.
    pub coverage: Table,
    /// Validated SPY and QQQ prices; their dates are the market calendar.
    pub benchmarks: Frame,
    pub metadata: Dict,
}

/// Read the complete current list from Nasdaq. Keep each share class.
pub fn fetch_nasdaq100() -> Result<UniverseSnapshot> {
    let headers = [("User-Agent", "Mozilla/5.0"), ("Accept", "application/json")];
    let (members, source_date) =
        read_url(NASDAQ100_URL, &headers).and_then(|body| nasdaq_members(&body)).ok_or_else(|| {
            value_error!(
                "The current Nasdaq-100 list is unavailable or incomplete. \
                 Retry later, choose your own stocks, or upload a price CSV."
            )
        })?;
    let retrieved_at = net::utc_now_iso();
    let source_date = iso(source_date);
    let record = dict! {
        "name" => "Nasdaq-100 constituents", "url" => NASDAQ100_URL, "date" => source_date.as_str(),
        "retrieved_at" => retrieved_at.as_str(), "stock_count" => members.len(),
    };
    Ok(UniverseSnapshot {
        source_records: vec![record],
        ..UniverseSnapshot::new(members, NASDAQ100_URL, source_date, retrieved_at)
    })
}

/// The validated members and the provider date of a Nasdaq response.
fn nasdaq_members(body: &[u8]) -> Option<(Vec<Member>, Date)> {
    let payload: Value = serde_json::from_slice(body).ok()?;
    let data = payload.get("data")?;
    let rows = data.get("data")?.get("rows")?;
    if payload.get("status")?.get("rCode")?.as_f64() != Some(200.0) {
        return None;
    }
    let rows = rows.as_array()?;
    if rows.len() < 100 || python_int(data.get("totalrecords")?)? != rows.len() as i64 {
        return None;
    }
    let mut members = Vec::with_capacity(rows.len());
    for row in rows {
        let (symbol, name) = (row.get("symbol")?.as_str()?, row.get("companyName")?.as_str()?);
        if !is_symbol(symbol) || py_strip(name).is_empty() {
            return None;
        }
        let sector = match row.get("sector").filter(|sector| net::truthy(sector)) {
            Some(sector) => net::python_value(sector).py_str(),
            None => "Unknown".into(),
        };
        let industry = match py_strip(&sector) {
            "" => "Unknown",
            industry => industry,
        };
        members.push(Member {
            symbol: parse_tickers(symbol).into_iter().next()?,
            name: py_strip(name).to_string(),
            industry: industry.to_string(),
        });
    }
    if !unique_symbols(&members) {
        return None;
    }
    let source_date = net::provider_date(data.get("date")?.as_str()?)?;
    (source_date <= Utc::now().date_naive()).then_some((members, source_date))
}

/// Read SPY stocks as an S&P 500 proxy, with an independent fund-count check.
pub fn fetch_spy_holdings() -> Result<UniverseSnapshot> {
    let holdings = spy_holdings().ok_or_else(|| {
        value_error!(
            "The current SPY stock holdings are unavailable or incomplete. \
             Retry later, choose Nasdaq-100 or your own stocks, or upload a price CSV."
        )
    })?;
    let retrieved_at = net::utc_now_iso();
    let record = dict! {
        "name" => "State Street SPY holdings (S&P 500 proxy)", "url" => SPY_HOLDINGS_URL,
        "date" => holdings.source_date.as_str(), "retrieved_at" => retrieved_at.as_str(),
        "stock_count" => holdings.members.len(), "reported_fund_holdings" => holdings.reported_count,
        "count_source_url" => SPY_PRODUCT_URL, "excluded_holdings" => holdings.excluded,
    };
    Ok(UniverseSnapshot {
        name: "Current S&P 500 stocks (SPY holdings proxy)".into(),
        limitation: SPY_LIMITATION.into(),
        source_records: vec![record],
        ..UniverseSnapshot::new(holdings.members, SPY_HOLDINGS_URL, holdings.source_date, retrieved_at)
    })
}

/// Validated SPY stock holdings, the excluded holdings, and the published fund count.
struct SpyHoldings {
    members: Vec<Member>,
    source_date: String,
    excluded: Vec<Json>,
    reported_count: usize,
}

fn spy_holdings() -> Option<SpyHoldings> {
    let headers = [("User-Agent", "Mozilla/5.0")];
    let rows = spy_worksheet(&read_url(SPY_HOLDINGS_URL, &headers)?)?;
    let metadata: HashMap<Option<&str>, &str> =
        rows.iter().take(3).map(|row| (row.get("A").map(String::as_str), cell(row, "B"))).collect();
    let field = |key: &str| metadata.get(&Some(key)).copied();
    if field("Ticker Symbol:") != Some("SPY") || !names_sp500_trust(field("Fund Name:").unwrap_or("")) {
        return None;
    }
    let dated = field("Holdings:").unwrap_or("").strip_prefix("As of ").filter(|date| is_one_line(date))?;
    let source_date = net::provider_date(dated).filter(|date| *date <= Utc::now().date_naive())?;
    let source_date = iso(source_date);
    let header_rows: Vec<usize> = (0..rows.len())
        .filter(|i| {
            SPY_COLUMNS
                .iter()
                .zip(SPY_HEADER)
                .all(|(column, label)| rows[*i].get(*column).is_some_and(|text| text == label))
        })
        .collect();
    let [header] = header_rows[..] else { return None };
    let (mut members, mut excluded, mut weights, mut cash_count) = (Vec::new(), Vec::new(), Vec::new(), 0);
    for row in &rows[header + 1..] {
        if SPY_COLUMNS[1..].iter().all(|column| cell(row, column).is_empty()) {
            continue; // The provider places disclosure text below the holdings table.
        }
        let (name, symbol, identifier) = (cell(row, "A"), cell(row, "B"), cell(row, "C"));
        let (weight, shares) = (python_float(cell(row, "E"))?, python_float(cell(row, "G"))?);
        if name.is_empty()
            || !weight.is_finite()
            || !shares.is_finite()
            || row.get("H").map(String::as_str) != Some("USD")
        {
            return None;
        }
        weights.push(weight);
        let reason = if ["US DOLLAR", "CASH", "CASH USD"].contains(&name.to_uppercase().as_str()) {
            cash_count += 1;
            "Cash holding"
        } else if identifier.contains("CVR")
            || has_word(name, &["CVR", "CONTINGENT VALUE", "WARRANT", "RIGHT", "RIGHTS"])
        {
            "Non-stock right or warrant"
        } else if BENCHMARK_SYMBOLS.contains(&symbol)
            || has_word(name, &["ETF", "ETN", "FUTURE", "FUTURES", "SWAP", "OPTION", "OPTIONS", "MONEY MARKET"])
        {
            "Fund or derivative holding"
        } else {
            ""
        };
        if !reason.is_empty() {
            excluded.push(Json::Dict(
                dict! { "symbol" => symbol, "name" => name, "identifier" => identifier, "reason" => reason },
            ));
            continue;
        }
        if !is_symbol(symbol)
            || !is_code(identifier, 9, 0)
            || !is_code(cell(row, "D"), 6, 1)
            || weight <= 0.0
            || shares <= 0.0
        {
            return None;
        }
        let sector = cell(row, "F");
        members.push(Member {
            symbol: parse_tickers(symbol).into_iter().next()?,
            name: name.to_string(),
            industry: if sector.is_empty() || sector == "-" { "Unknown" } else { sector }.to_string(),
        });
    }
    if !(500..=525).contains(&members.len()) || !unique_symbols(&members) {
        return None;
    }
    if !(95.0..=105.0).contains(&python_sum(&weights)) {
        return None;
    }
    let page = read_url(SPY_PRODUCT_URL, &headers)?;
    let (count_date, reported_count) = spy_reported_count(std::str::from_utf8(&page).ok()?)?;
    if count_date != source_date || reported_count != weights.len() - cash_count {
        return None;
    }
    Some(SpyHoldings { members, source_date, excluded, reported_count })
}

/// One worksheet row: the stripped text of each cell by column letters.
type SheetRow = HashMap<String, String>;

fn cell<'a>(row: &'a SheetRow, column: &str) -> &'a str {
    row.get(column).map_or("", String::as_str)
}

/// Read the provider's single-sheet XLSX without an Excel dependency.
fn spy_worksheet(content: &[u8]) -> Option<Vec<SheetRow>> {
    let mut archive = ZipArchive::new(Cursor::new(content)).ok()?;
    let mut strings = Vec::new();
    if archive.index_for_name("xl/sharedStrings.xml").is_some() {
        let xml = zip_text(&mut archive, "xl/sharedStrings.xml")?;
        let document = Document::parse(&xml).ok()?;
        strings = document.root_element().children().filter(Node::is_element).map(all_text).collect();
    }
    let xml = zip_text(&mut archive, "xl/worksheets/sheet1.xml")?;
    let sheet = Document::parse(&xml).ok()?;
    let mut rows = Vec::new();
    let sheet_data = sheet.root_element().children().filter(|node| node.has_tag_name((SHEET_NAMESPACE, "sheetData")));
    for row in sheet_data.flat_map(|data| data.children().filter(|node| node.has_tag_name((SHEET_NAMESPACE, "row")))) {
        let mut cells = SheetRow::new();
        for cell in row.children().filter(Node::is_element) {
            let reference = cell.attribute("r").unwrap_or("");
            if !is_cell_reference(reference) {
                return None;
            }
            let value = cell.children().find(|node| node.has_tag_name((SHEET_NAMESPACE, "v")));
            // ElementTree's `.text`: empty without a `<v>`, `None` for an empty `<v>`.
            let mut text = value.map_or(Some(String::new()), |value| value.text().map(str::to_string));
            match cell.attribute("t") {
                Some("s") => {
                    let index = usize::try_from(python_int_text(text.as_deref()?)?).ok()?;
                    text = Some(strings.get(index)?.clone());
                }
                Some("inlineStr") => text = Some(all_text(cell)),
                _ => {}
            }
            let column: String = reference.chars().filter(|c| !c.is_ascii_digit()).collect();
            cells.insert(column, py_strip(text.as_deref().unwrap_or("")).to_string());
        }
        rows.push(cells);
    }
    Some(rows)
}

fn zip_text<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Option<String> {
    let mut text = String::new();
    archive.by_name(name).ok()?.read_to_string(&mut text).ok()?;
    Some(text)
}

/// ElementTree's `"".join(element.itertext())`.
fn all_text(node: Node) -> String {
    node.descendants().filter(Node::is_text).filter_map(|text| text.text()).collect()
}

/// `[A-Z]+[1-9][0-9]*`, a worksheet cell reference.
fn is_cell_reference(reference: &str) -> bool {
    let letters = reference.bytes().take_while(u8::is_ascii_uppercase).count();
    let digits = &reference.as_bytes()[letters..];
    letters > 0
        && digits.first().is_some_and(|first| (b'1'..=b'9').contains(first))
        && digits.iter().all(u8::is_ascii_digit)
}

/// Read the dated fund count, distinct from the index count on the same page.
fn spy_reported_count(document: &str) -> Option<(String, usize)> {
    for section in sections(document) {
        let text = net::single_spaces(&net::unescape(&strip_tags(section)));
        if let Some((date, count)) = fund_count(&text) {
            return Some((iso(net::provider_date(date)?), count.parse().ok()?));
        }
    }
    None
}

/// The contents of each `<section ...>...</section>`, as Python's
/// `re.findall(r"<section\b[^>]*>(.*?)</section>", document, flags=re.S)` finds them.
fn sections(document: &str) -> Vec<&str> {
    const CLOSE: &str = "</section>";
    let mut found = Vec::new();
    let mut position = 0;
    while let Some(offset) = document[position..].find("<section") {
        let start = position + offset;
        let name_end = start + "<section".len();
        let boundary = document[name_end..].chars().next().is_none_or(|next| !is_word_character(next));
        let content_start = document[name_end..].find('>').map(|close| name_end + close + 1);
        match content_start.filter(|_| boundary) {
            Some(content_start) => match document[content_start..].find(CLOSE) {
                Some(length) => {
                    found.push(&document[content_start..content_start + length]);
                    position = content_start + length + CLOSE.len();
                }
                None => break,
            },
            None => position = start + 1,
        }
    }
    found
}

/// Python's `re.sub(r"<[^>]+>", " ", text)`: each tag becomes a space.
fn strip_tags(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        output.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find('>') {
            Some(close) if close > 0 => {
                output.push(' ');
                rest = &after[close + 1..];
            }
            _ => {
                output.push('<');
                rest = after;
            }
        }
    }
    output.push_str(rest);
    output
}

/// The date and count of `Fund Characteristics as of (\w+ \d{1,2} \d{4}).*?Number of Holdings (\d+)`.
fn fund_count(text: &str) -> Option<(&str, &str)> {
    const LABEL: &str = "Fund Characteristics as of ";
    const COUNT: &str = "Number of Holdings ";
    text.match_indices(LABEL).find_map(|(start, _)| {
        let date_start = start + LABEL.len();
        let date_end = date_start + dated_words(&text[date_start..])?;
        let rest = &text[date_end..];
        let count = rest.match_indices(COUNT).find_map(|(position, _)| {
            let digits = &rest[position + COUNT.len()..];
            let length = digits.bytes().take_while(u8::is_ascii_digit).count();
            (length > 0).then(|| &digits[..length])
        })?;
        Some((&text[date_start..date_end], count))
    })
}

/// The length of `\w+ \d{1,2} \d{4}` at the start of `text`, as in `Oct 29 2026`.
fn dated_words(text: &str) -> Option<usize> {
    let word = text.find(|c: char| !is_word_character(c)).unwrap_or(text.len());
    let rest = text[word..].strip_prefix(' ').filter(|_| word > 0)?;
    let day = rest.bytes().take_while(u8::is_ascii_digit).count();
    let rest = rest[day..].strip_prefix(' ').filter(|_| (1..=2).contains(&day))?;
    (rest.bytes().take_while(u8::is_ascii_digit).count() >= 4).then(|| text.len() - rest.len() + 4)
}

/// Python's `\w` for one character.
fn is_word_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// Python `re.search(r"\b(WORD|...)\b", text, re.I)` for ASCII words.
fn has_word(text: &str, words: &[&str]) -> bool {
    let characters: Vec<char> = text.chars().collect();
    words.iter().any(|word| {
        let word: Vec<char> = word.chars().collect();
        characters.windows(word.len()).enumerate().any(|(start, window)| {
            window.iter().zip(&word).all(|(left, right)| left.eq_ignore_ascii_case(right))
                && (start == 0 || !is_word_character(characters[start - 1]))
                && characters.get(start + word.len()).is_none_or(|next| !is_word_character(*next))
        })
    })
}

/// `S&P 500.*ETF Trust`, searched within one line.
fn names_sp500_trust(text: &str) -> bool {
    text.match_indices("S&P 500").any(|(start, matched)| {
        let rest = &text[start + matched.len()..];
        rest.find("ETF Trust").is_some_and(|end| !rest[..end].contains('\n'))
    })
}

/// Python's `.+` within `re.fullmatch`: one or more characters without a line feed.
fn is_one_line(text: &str) -> bool {
    !text.is_empty() && !text.contains('\n')
}

/// `[A-Z][A-Z0-9.-]*`, a provider stock symbol.
fn is_symbol(text: &str) -> bool {
    let mut characters = text.chars();
    characters.next().is_some_and(|first| first.is_ascii_uppercase())
        && characters.all(|c| is_upper_alphanumeric(c) || c == '.' || c == '-')
}

fn is_upper_alphanumeric(character: char) -> bool {
    character.is_ascii_uppercase() || character.is_ascii_digit()
}

/// `[A-Z0-9]{letters}\d{digits}`, such as a CUSIP identifier or a SEDOL.
fn is_code(text: &str, letters: usize, digits: usize) -> bool {
    let characters: Vec<char> = text.chars().collect();
    characters.len() == letters + digits
        && characters[..letters].iter().all(|c| is_upper_alphanumeric(*c))
        && characters[letters..].iter().all(char::is_ascii_digit)
}

fn unique_symbols(members: &[Member]) -> bool {
    members.iter().map(|member| member.symbol.as_str()).collect::<HashSet<_>>().len() == members.len()
}

/// Python `int(value)` for a JSON number, boolean or text.
fn python_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => {
            number.as_i64().or_else(|| number.as_f64().filter(|x| x.is_finite()).map(|x| x.trunc() as i64))
        }
        Value::String(text) => python_int_text(text),
        Value::Bool(flag) => Some(i64::from(*flag)),
        _ => None,
    }
}

fn python_int_text(text: &str) -> Option<i64> {
    py_strip(text).parse().ok()
}

/// Python `float(text)`.
fn python_float(text: &str) -> Option<f64> {
    py_strip(text).parse().ok()
}

/// Python's built-in `sum` of floats, which compensates rounding errors (Neumaier).
fn python_sum(values: &[f64]) -> f64 {
    let (mut high, mut low) = (0.0_f64, 0.0_f64);
    for value in values {
        let total = high + value;
        low += if high.abs() >= value.abs() { (high - total) + value } else { (value - total) + high };
        high = total;
    }
    if low != 0.0 && low.is_finite() { high + low } else { high }
}

/// GET a provider URL as `urlopen` does: an error status is a failure.
fn read_url(url: &str, headers: &[(&str, &str)]) -> Option<Vec<u8>> {
    let response = net::get(url, &[], headers, PROVIDER_TIMEOUT).ok()?;
    (response.status < 400).then_some(response.body)
}

/// Combine both complete sources; retain share classes and deduplicate overlap.
pub fn fetch_sp500_nasdaq100() -> Result<UniverseSnapshot> {
    let nasdaq = fetch_nasdaq100()?;
    let spy = fetch_spy_holdings()?;
    let mut seen = HashSet::new();
    let mut members = Vec::new();
    for member in nasdaq.members.iter().chain(&spy.members) {
        let symbol = parse_tickers(&member.symbol).swap_remove(0);
        if seen.insert(symbol.clone()) {
            members.push(Member { symbol, ..member.clone() });
        }
    }
    let source_date = nasdaq.source_date.min(spy.source_date);
    Ok(UniverseSnapshot {
        name: "Current S&P 500 (SPY holdings proxy) + Nasdaq-100 stocks".into(),
        limitation: SPY_LIMITATION.into(),
        source_records: [nasdaq.source_records, spy.source_records].concat(),
        ..UniverseSnapshot::new(members, SPY_HOLDINGS_URL, source_date, net::utc_now_iso())
    })
}

/// Fetch all members in batches. Report every exclusion without shorter dates.
///
/// SPY and QQQ provide the common daily calendar. No missing value receives a fill.
/// The optional callback receives `(completed, total, message)`.
pub fn download_universe_prices(
    snapshot: &UniverseSnapshot,
    start: Date,
    end: Date,
    progress: Option<Progress<'_>>,
) -> Result<UniversePrices> {
    download_universe_prices_with(snapshot, start, end, progress, &mut |symbols, start, end, allow_incomplete| {
        download_adjusted_close(symbols, start, end, allow_incomplete)
    })
}

/// [`download_universe_prices`] with `download` in place of
/// [`download_adjusted_close`]. The benchmarks use it with complete prices
/// required, as `download_prices` does.
pub fn download_universe_prices_with(
    snapshot: &UniverseSnapshot,
    start: Date,
    end: Date,
    mut progress: Option<Progress<'_>>,
    download: Download<'_>,
) -> Result<UniversePrices> {
    let symbols: Vec<String> = snapshot.members.iter().map(|member| member.symbol.clone()).collect();
    let total = symbols.len();
    if total == 0 || !unique_symbols(&snapshot.members) {
        return Err(value_error!("The universe must contain unique security symbols."));
    }
    let mut report = |done: usize, message: String| {
        if let Some(progress) = progress.as_mut() {
            progress(done, total, &message);
        }
    };
    report(0, "Read market dates from SPY and QQQ.".into());
    let benchmarks = validate_prices(&download(&BENCHMARK_SYMBOLS.map(String::from), start, end, false)?)?;
    let mut coverage = Coverage { snapshot, expected: &benchmarks.index, accepted: Vec::new(), rows: Vec::new() };
    for (number, batch) in symbols.chunks(BATCH_SIZE).enumerate() {
        let offset = number * BATCH_SIZE;
        report(offset, format!("Download securities {} to {} of {total}.", offset + 1, offset + batch.len()));
        match download(batch, start, end, true) {
            Ok(prices) => {
                for symbol in batch {
                    let member = prices.select(&[symbol]).expect("the download returns every requested symbol");
                    coverage.record(symbol, Ok(member));
                    report(coverage.rows.len(), format!("Checked {} of {total} securities.", coverage.rows.len()));
                }
            }
            // Retry whole-batch download failures separately from member coverage.
            Err(error) if error.is_value() => {
                for symbol in batch {
                    let individual = download(std::slice::from_ref(symbol), start, end, true);
                    if let Err(error) = &individual
                        && !error.is_value()
                    {
                        return Err(error.clone());
                    }
                    coverage.record(symbol, individual);
                    report(coverage.rows.len(), format!("Checked {} of {total} securities.", coverage.rows.len()));
                }
            }
            Err(error) => return Err(error),
        }
    }
    let included = coverage.accepted.len();
    let table = coverage_table(&coverage.rows);
    let records: Vec<Json> = coverage.rows.iter().map(|row| Json::Dict(row.record())).collect();
    let prices = Frame::from_columns(benchmarks.index.clone(), coverage.accepted);
    let metadata = dict! {
        "universe" => snapshot.name.as_str(),
        "universe_source_url" => snapshot.source_url.as_str(),
        "universe_source_date" => snapshot.source_date.as_str(),
        "universe_retrieved_at" => snapshot.retrieved_at.as_str(),
        "universe_sources" => snapshot.source_records.iter().cloned().map(Json::Dict).collect::<Vec<_>>(),
        "universe_requested" => total,
        "universe_included" => included,
        "universe_excluded" => total - included,
        "universe_calendar" => "Yahoo adjusted SPY and QQQ dates",
        "universe_start" => iso(benchmarks.index[0]),
        "universe_end" => iso(benchmarks.index[benchmarks.nrows() - 1]),
        "universe_eligibility" => "Complete positive adjusted prices on every market date in the selected period.",
        "universe_limitation" => snapshot.limitation.as_str(),
        "universe_coverage" => records,
    };
    Ok(UniversePrices { prices, coverage: table, benchmarks, metadata })
}

/// The coverage decision for one member.
struct CoverageRow {
    symbol: String,
    name: String,
    industry: String,
    reason: String,
    observations: Option<usize>,
    first_date: Option<String>,
    last_date: Option<String>,
}

impl CoverageRow {
    fn status(&self) -> &'static str {
        if self.reason.is_empty() { "included" } else { "excluded" }
    }

    fn record(&self) -> Dict {
        dict! {
            "symbol" => self.symbol.as_str(), "name" => self.name.as_str(), "industry" => self.industry.as_str(),
            "status" => self.status(), "reason" => self.reason.as_str(), "observations" => self.observations,
            "first_date" => self.first_date.clone(), "last_date" => self.last_date.clone(),
        }
    }
}

/// Member decisions and the accepted price columns, in member order.
struct Coverage<'a> {
    snapshot: &'a UniverseSnapshot,
    expected: &'a [Date],
    accepted: Vec<(String, Vec<f64>)>,
    rows: Vec<CoverageRow>,
}

impl Coverage<'_> {
    fn record(&mut self, symbol: &str, download: Result<Frame>) {
        let member = self.snapshot.members.iter().find(|member| member.symbol == symbol).expect("a snapshot member");
        let mut row = CoverageRow {
            symbol: symbol.to_string(),
            name: member.name.clone(),
            industry: member.industry.clone(),
            reason: String::new(),
            observations: None,
            first_date: None,
            last_date: None,
        };
        match download {
            Err(error) => row.reason = error.message().to_string(),
            Ok(prices) => {
                // Inspect each member's actual dates, then require the full market calendar below.
                let kept: Vec<usize> =
                    (0..prices.nrows()).filter(|i| prices.row(*i).iter().any(|value| !value.is_nan())).collect();
                let prices = Frame::from_rows(
                    kept.iter().map(|i| prices.index[*i]).collect(),
                    prices.columns.clone(),
                    &kept.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
                );
                row.observations = Some(prices.nrows());
                match validate_prices(&prices) {
                    Err(_) if prices.nrows() == 0 => row.reason = format!("Yahoo returned no prices for: {symbol}."),
                    Err(error) => row.reason = error.message().to_string(),
                    Ok(prices) => {
                        row.first_date = Some(iso(prices.index[0]));
                        row.last_date = Some(iso(prices.index[prices.nrows() - 1]));
                        if prices.index != self.expected {
                            let missing = self.expected.iter().filter(|date| prices.index.binary_search(date).is_err());
                            let extra = prices.index.iter().filter(|date| self.expected.binary_search(date).is_err());
                            row.reason = format!(
                                "Price dates differ from market dates: {} missing, {} extra.",
                                missing.count(),
                                extra.count()
                            );
                        } else {
                            self.accepted.push((symbol.to_string(), prices.col(0)));
                        }
                    }
                }
            }
        }
        self.rows.push(row);
    }
}

/// `pandas.DataFrame(records)`: whole numbers with a missing value become floats.
fn coverage_table(rows: &[CoverageRow]) -> Table {
    let text = |value: &Option<String>| value.clone().map_or(Cell::Null, Cell::Str);
    let all_counted = rows.iter().all(|row| row.observations.is_some());
    let none_counted = rows.iter().all(|row| row.observations.is_none());
    let observations = rows
        .iter()
        .map(|row| match row.observations {
            Some(count) if all_counted => Cell::Int(count as i64),
            Some(count) => Cell::Float(count as f64),
            None if none_counted => Cell::Null,
            None => Cell::Float(f64::NAN),
        })
        .collect();
    Table::range(rows.len())
        .with("symbol", rows.iter().map(|row| Cell::from(row.symbol.as_str())).collect())
        .with("name", rows.iter().map(|row| Cell::from(row.name.as_str())).collect())
        .with("industry", rows.iter().map(|row| Cell::from(row.industry.as_str())).collect())
        .with("status", rows.iter().map(|row| Cell::from(row.status())).collect())
        .with("reason", rows.iter().map(|row| Cell::from(row.reason.as_str())).collect())
        .with("observations", observations)
        .with("first_date", rows.iter().map(|row| text(&row.first_date)).collect())
        .with("last_date", rows.iter().map(|row| text(&row.last_date)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_parsing_matches_python_patterns() {
        // Expected values from Python's `re` with the patterns of `universe.py`.
        let sections_cases: [(&str, &[&str]); 9] = [
            ("<section>a</section>", &["a"]),
            ("<section class='x'>b</section><section>c</section>", &["b", "c"]),
            ("<sections>no</section>", &[]),
            ("<section", &[]),
            ("<section>open", &[]),
            ("<section>a<section>b</section>c</section>", &["a<section>b"]),
            ("<SECTION>x</SECTION>", &[]),
            ("x<section\n data=1>multi\nline</section>y", &["multi\nline"]),
            ("<section_x>z</section>", &[]),
        ];
        for (document, expected) in sections_cases {
            assert_eq!(sections(document), expected, "{document:?}");
        }
        let tags =
            [("a<b>c", "a c"), ("a<>b", "a<>b"), ("a<b", "a<b"), ("x<a<b>c>d", "x c>d"), ("<p>x</p><br/>", " x  ")];
        for (text, expected) in tags {
            assert_eq!(strip_tags(text), expected, "{text:?}");
        }
        let counts = [
            (
                "Fund Characteristics as of Oct 29 2026 x Number of Holdings 504 y Number of Holdings 7",
                Some(("Oct 29 2026", "504")),
            ),
            (
                "Fund Characteristics as of Oct 2 2026 Number of Holdings x Number of Holdings 12",
                Some(("Oct 2 2026", "12")),
            ),
            ("Fund Characteristics as of Oct 123 2026 Number of Holdings 5", None),
            ("Fund Characteristics as of Oct29 2026 Number of Holdings 5", None),
            ("Fund Characteristics as of Oct 29 20261 Number of Holdings 5", Some(("Oct 29 2026", "5"))),
            (
                "Fund Characteristics as of x Fund Characteristics as of Sep 30 2026 Number of Holdings 501",
                Some(("Sep 30 2026", "501")),
            ),
            ("Fund Characteristics as of Oct 29 2026", None),
        ];
        for (text, expected) in counts {
            assert_eq!(fund_count(text), expected, "{text:?}");
        }
    }

    #[test]
    fn holding_names_match_python_word_patterns() {
        let right = ["CVR", "CONTINGENT VALUE", "WARRANT", "RIGHT", "RIGHTS"];
        let fund = ["ETF", "ETN", "FUTURE", "FUTURES", "SWAP", "OPTION", "OPTIONS", "MONEY MARKET"];
        let cases = [
            ("TPG INC CVR", true, false),
            ("ACME RIGHTS", true, false),
            ("COPYRIGHTS INC", false, false),
            ("Contingent Value Rts", true, false),
            ("contingent value", true, false),
            ("WARRANTS", false, false),
            ("S&P 500 FUTURE", false, true),
            ("OPTIONSXPRESS", false, false),
            ("MONEY  MARKET", false, false),
            ("ETF_X", false, false),
            ("X-ETF", false, true),
            ("\u{c9}TF", false, false),
            ("SWAPS", false, false),
        ];
        for (name, is_right, is_fund) in cases {
            assert_eq!((has_word(name, &right), has_word(name, &fund)), (is_right, is_fund), "{name:?}");
        }
        let trusts = [
            ("State Street\u{ae} SPDR\u{ae} S&P 500\u{ae} ETF Trust", true),
            ("S&P 500\nETF Trust", false),
            ("ETF Trust S&P 500", false),
            ("S&P 500 ETF Trus", false),
        ];
        for (name, expected) in trusts {
            assert_eq!(names_sp500_trust(name), expected, "{name:?}");
        }
        let references = [
            ("A1", true),
            ("AB10", true),
            ("a1", false),
            ("A0", false),
            ("A01", false),
            ("1A", false),
            ("A", false),
            ("A1B", false),
        ];
        for (reference, expected) in references {
            assert_eq!(is_cell_reference(reference), expected, "{reference:?}");
        }
        assert!(is_code("436CVR021", 9, 0) && is_code("B0YBKJ7", 6, 1));
        assert!(!is_code("-", 6, 1) && !is_code("B0YBKJé", 6, 1) && !is_code("12345678", 9, 0));
    }

    #[test]
    fn weight_sum_matches_python_sum() {
        let mut weights = vec![99.5 / 501.0; 501];
        weights.extend([0.5, 0.000003]);
        // Python 3.14 `sum(weights)` prints 100.000003; plain addition differs in the last digits.
        assert_eq!(python_sum(&weights), 100.000003);
        assert_ne!(weights.iter().sum::<f64>(), 100.000003);
    }
}
