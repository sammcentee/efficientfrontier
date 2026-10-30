//! Network access for the provider modules.
//!
//! One HTTP client serves Yahoo Finance, Nasdaq and State Street. Yahoo price
//! history follows `yfinance.download(..., auto_adjust=True, keepna=True)` of
//! the Python release, and Yahoo profile requests carry a session cookie and
//! crumb as yfinance sends them. [`with_fake_network`] replaces the network in
//! offline tests. The text helpers read provider text with the rules of
//! Python's `str.strip`, `str.split`, `html.unescape` and `pandas.Timestamp`.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use chrono::{DateTime, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc};
use chrono_tz::Tz;
use serde_json::Value;

use crate::frame::{Date, Frame};
use crate::pyfmt::{self, Json};

/// The browser identity of yfinance's plain `requests` session.
const YAHOO_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const YAHOO_HEADERS: [(&str, &str); 2] = [
    ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    ("Accept-Language", "en-US,en;q=0.5"),
];
const YAHOO_COOKIE_URL: &str = "https://fc.yahoo.com";
const YAHOO_CRUMB_URL: &str = "https://query1.finance.yahoo.com/v1/test/getcrumb";
const YAHOO_SESSION_TIMEOUT: Duration = Duration::from_secs(30);
/// The Yahoo chart endpoint. The symbol follows as one path segment.
pub const CHART_URL: &str = "https://query2.finance.yahoo.com/v8/finance/chart/";
/// The timeout of `yfinance.download`.
const CHART_TIMEOUT: Duration = Duration::from_secs(10);

/// One GET request.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The URL without a query string.
    pub url: String,
    /// Query parameters in request order, before percent-encoding.
    pub query: Vec<(String, String)>,
    /// Request headers. The client adds a browser `User-Agent` when none is given.
    pub headers: Vec<(String, String)>,
    /// The time limit for the complete request.
    pub timeout: Duration,
}

impl Request {
    /// The value of a query parameter.
    pub fn param(&self, name: &str) -> Option<&str> {
        self.query.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }

    /// The value of a request header. Header names ignore case.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(key, _)| key.eq_ignore_ascii_case(name)).map(|(_, value)| value.as_str())
    }
}

/// A complete response with any HTTP status.
#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self { status, body: body.into() }
    }
}

/// A response, or the text of a transport failure such as a refused connection or a timeout.
pub type Reply = std::result::Result<Response, String>;

type Responder = Box<dyn FnMut(&Request) -> Reply>;

thread_local! {
    static FAKE: RefCell<Option<Responder>> = const { RefCell::new(None) };
    static FAKE_CRUMB: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Run `body` with `respond` in place of the network on this thread.
///
/// Every request of the provider modules goes to `respond`, which returns a
/// recorded or synthetic reply. The Yahoo session starts without a crumb.
/// The previous state returns after `body`, also after a panic.
pub fn with_fake_network<T>(respond: impl FnMut(&Request) -> Reply + 'static, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<Responder>, Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FAKE.with(|fake| *fake.borrow_mut() = self.0.take());
            FAKE_CRUMB.with(|crumb| *crumb.borrow_mut() = self.1.take());
        }
    }
    let _restore =
        Restore(FAKE.with(|fake| fake.replace(Some(Box::new(respond)))), FAKE_CRUMB.with(|crumb| crumb.replace(None)));
    body()
}

fn is_fake() -> bool {
    FAKE.with(|fake| fake.borrow().is_some())
}

fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder().http_status_as_error(false).user_agent(YAHOO_USER_AGENT).build().into()
    })
}

/// Send one GET request through the shared client, or to the fake network of this thread.
///
/// Redirects are followed. Any HTTP status is a response; only a transport failure is an error.
pub(crate) fn get(url: &str, query: &[(&str, &str)], headers: &[(&str, &str)], timeout: Duration) -> Reply {
    let request = Request {
        url: url.to_string(),
        query: query.iter().map(|(name, value)| (name.to_string(), value.to_string())).collect(),
        headers: headers.iter().map(|(name, value)| (name.to_string(), value.to_string())).collect(),
        timeout,
    };
    if let Some(reply) = FAKE.with(|fake| fake.borrow_mut().as_mut().map(|respond| respond(&request))) {
        return reply;
    }
    let mut call = agent().get(url);
    for (name, value) in query {
        call = call.query(*name, *value);
    }
    for (name, value) in headers {
        call = call.header(*name, *value);
    }
    let mut response = call.config().timeout_global(Some(timeout)).build().call().map_err(|error| error.to_string())?;
    let status = response.status().as_u16();
    let body = response.body_mut().read_to_vec().map_err(|error| error.to_string())?;
    Ok(Response { status, body })
}

/// The crumb of the real Yahoo session. A fake network keeps its own.
static CRUMB: Mutex<Option<String>> = Mutex::new(None);

/// The session crumb, renewed on request or when none is cached.
fn yahoo_crumb(renew: bool) -> Option<String> {
    let update = |slot: &mut Option<String>| {
        if renew || slot.is_none() {
            *slot = new_crumb();
        }
        slot.clone()
    };
    if is_fake() {
        FAKE_CRUMB.with(|crumb| update(&mut crumb.borrow_mut()))
    } else {
        update(&mut CRUMB.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// A new session cookie and crumb, as yfinance's basic cookie strategy obtains them.
fn new_crumb() -> Option<String> {
    // Yahoo answers 404 but sets the session cookie. The crumb request decides success.
    let _ = get(YAHOO_COOKIE_URL, &[], &YAHOO_HEADERS, YAHOO_SESSION_TIMEOUT);
    let response = get(YAHOO_CRUMB_URL, &[], &YAHOO_HEADERS, YAHOO_SESSION_TIMEOUT).ok()?;
    let crumb = String::from_utf8_lossy(&response.body).into_owned();
    let refused = response.status == 429 || crumb.contains("Too Many Requests") || crumb.contains("<html>");
    (!refused && !crumb.is_empty()).then_some(crumb)
}

/// GET Yahoo JSON with the session crumb, as `YfData.get_raw_json` does.
///
/// An error status renews the cookie and crumb for one more attempt. A second
/// error status, a transport failure or invalid JSON is an error.
pub(crate) fn yahoo_json(url: &str, query: &[(&str, &str)], timeout: Duration) -> std::result::Result<Value, String> {
    let send = |crumb: Option<String>| {
        let mut query = query.to_vec();
        if let Some(crumb) = &crumb {
            query.push(("crumb", crumb));
        }
        get(url, &query, &YAHOO_HEADERS, timeout)
    };
    let mut response = send(yahoo_crumb(false))?;
    if response.status >= 400 {
        response = send(yahoo_crumb(true))?;
    }
    if response.status >= 400 {
        return Err(format!("HTTP status {}", response.status));
    }
    serde_json::from_slice(&response.body).map_err(|error| error.to_string())
}

/// Yahoo adjusted daily closes, as the Python release reads them with
/// `yfinance.download(symbols, start, end, auto_adjust=True, actions=False,
/// threads=False, keepna=True, interval="1d")["Close"]`.
///
/// Each symbol has its own requests, in order. A symbol that fails has no
/// rows, because yfinance keeps an empty table for a failed download. Rows
/// are the sorted union of the symbol dates; a missing value is NaN. `end` is
/// exclusive. Columns follow `symbols`.
pub(crate) fn yahoo_adjusted_closes(symbols: &[String], start: Date, end: Date) -> Frame {
    let histories: Vec<Vec<(Date, f64)>> = symbols.iter().map(|symbol| symbol_history(symbol, start, end)).collect();
    let mut index: Vec<Date> = histories.iter().flatten().map(|(date, _)| *date).collect();
    index.sort();
    index.dedup();
    let columns = symbols
        .iter()
        .zip(&histories)
        .map(|(symbol, history)| {
            let mut column = vec![f64::NAN; index.len()];
            for (date, close) in history {
                let row = index.binary_search(date).expect("the union holds every symbol date");
                column[row] = *close;
            }
            (symbol.clone(), column)
        })
        .collect();
    Frame::from_columns(index, columns)
}

/// One symbol's dated adjusted closes, or no rows when the download fails.
///
/// yfinance requests midnight to midnight in the exchange timezone. The
/// first request assumes New York; a response from another exchange repeats
/// the request with that exchange's timezone.
fn symbol_history(symbol: &str, start: Date, end: Date) -> Vec<(Date, f64)> {
    let new_york = chrono_tz::America::New_York;
    let fetch = |zone: Tz| -> Option<(Value, Tz, i64)> {
        let (period1, period2) = (local_midnight(zone, start)?, local_midnight(zone, end)?);
        let chart = chart_json(symbol, period1, period2)?;
        let exchange = exchange_timezone(&chart)?;
        Some((chart, exchange, period2))
    };
    let mut fetched = fetch(new_york);
    if let Some((_, exchange, _)) = &fetched
        && *exchange != new_york
    {
        fetched = fetch(*exchange);
    }
    fetched.and_then(|(chart, exchange, period2)| chart_rows(&chart, exchange, period2)).unwrap_or_default()
}

/// Midnight of `date` in `zone` as Unix seconds. pandas refuses a midnight
/// that a clock change skips or repeats, and so does this.
fn local_midnight(zone: Tz, date: Date) -> Option<i64> {
    zone.from_local_datetime(&date.and_time(NaiveTime::MIN)).single().map(|time| time.timestamp())
}

/// The chart response as a JSON object, or `None` when yfinance records a failed download.
///
/// yfinance also sends its session crumb, which the chart endpoint does not need; this request omits it.
fn chart_json(symbol: &str, period1: i64, period2: i64) -> Option<Value> {
    let url = format!("{CHART_URL}{}", quote(symbol));
    let (period1, period2) = (period1.to_string(), period2.to_string());
    let query = [
        ("period1", period1.as_str()),
        ("period2", period2.as_str()),
        ("interval", "1d"),
        ("includePrePost", "False"),
        ("events", "div,splits,capitalGains"),
    ];
    let send = || get(&url, &query, &YAHOO_HEADERS, CHART_TIMEOUT).ok();
    let mut response = send()?;
    if response.status >= 400 {
        // yfinance repeats an error response once with a renewed session; a second 429 is a rate limit.
        response = send()?;
        if response.status == 429 {
            return None;
        }
    }
    let text = String::from_utf8(response.body).ok()?;
    if text.contains("Will be right back") {
        return None;
    }
    serde_json::from_str::<Value>(&text).ok().filter(Value::is_object)
}

/// The exchange timezone of a chart result. A name outside the timezone database fails, as in yfinance.
fn exchange_timezone(chart: &Value) -> Option<Tz> {
    chart.pointer("/chart/result/0/meta/exchangeTimezoneName")?.as_str()?.parse().ok()
}

/// Dated adjusted closes from one chart response, as yfinance's
/// `PriceHistory.history` builds `Close` for a daily interval with
/// `auto_adjust=True`, `keepna=True` and `repair=False`. `None` is a failed
/// download. `period2` is the exclusive end in Unix seconds.
///
/// Close is the adjusted close, or the close when Yahoo omits adjusted
/// closes. Rows with a missing close stay as NaN. yfinance drops a final row
/// at or after the end, adds a NaN row for each event date after the last
/// price row, keeps the later of two final rows on the same date, and then
/// keeps the first row of each date.
fn chart_rows(chart: &Value, zone: Tz, period2: i64) -> Option<Vec<(Date, f64)>> {
    // The failure checks of `history`, in order.
    if chart.get("status_code").is_some() {
        return None;
    }
    let body = chart.get("chart").filter(|body| truthy(body))?;
    if truthy(body.get("error")?) {
        return None;
    }
    let result = body.get("result")?.as_array()?.first()?;
    let indicators = result.get("indicators")?;
    let quote = indicators.get("quote")?.get(0)?;
    if !truthy(quote) {
        return None;
    }
    let meta = result.get("meta")?;
    let instrument = meta.get("instrumentType")?;
    meta.get("currency")?;
    // `parse_quotes`: every column must have one value per timestamp.
    let times: Vec<i64> = result.get("timestamp")?.as_array()?.iter().map(Value::as_i64).collect::<Option<_>>()?;
    let column = |values: &Value| -> Option<Vec<f64>> {
        let values = values.as_array().filter(|values| values.len() == times.len())?;
        values.iter().map(|value| if value.is_null() { Some(f64::NAN) } else { value.as_f64() }).collect()
    };
    for name in ["volume", "open", "low", "high"] {
        column(quote.get(name)?)?;
    }
    let closes = match indicators.get("adjclose") {
        Some(adjusted) => column(adjusted.get(0)?.get("adjclose")?)?,
        None => column(quote.get("close")?)?,
    };
    column(quote.get("close")?)?;
    // `parse_actions` reads all three event kinds; capital gains count for funds only.
    let events = ["dividends", "splits", "capitalGains"]
        .into_iter()
        .map(|kind| event_times(result, kind))
        .collect::<Option<Vec<Vec<i64>>>>()?;
    let fund = matches!(instrument.as_str(), Some("ETF" | "MUTUALFUND"));

    let mut quotes: Vec<(i64, f64)> = times.into_iter().zip(closes).collect();
    quotes.sort_by_key(|(time, _)| *time);
    // Yahoo can append the latest price after the requested end.
    if quotes.last().is_some_and(|(time, _)| *time >= period2) {
        quotes.pop();
    }
    let mut rows: Vec<(Date, f64)> =
        quotes.iter().map(|(time, close)| Some((bar_date(*time, zone)?, *close))).collect::<Option<_>>()?;
    rows.sort_by_key(|(date, _)| *date);
    // `safe_merge_dfs`: an event after the last price row and before the end adds a row of missing prices.
    if let Some(&(mut last, _)) = rows.last() {
        for times in events.iter().take(if fund { 3 } else { 2 }) {
            let mut later: Vec<Date> = times
                .iter()
                .filter(|time| **time < period2)
                .map(|time| local_date(*time, zone))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|date| *date > last)
                .collect();
            later.sort();
            later.dedup();
            if let Some(latest) = later.last() {
                last = *latest;
            }
            rows.extend(later.into_iter().map(|date| (date, f64::NAN)));
        }
    }
    // Yahoo can return today's bar twice; the later row is current.
    let count = rows.len();
    if count > 1 && rows[count - 1].0 == rows[count - 2].0 {
        rows.remove(count - 2);
    }
    let mut seen = HashSet::new();
    rows.retain(|(date, _)| seen.insert(*date));
    Some(rows)
}

/// Event times of one kind, from the `date` field of each event.
fn event_times(result: &Value, kind: &str) -> Option<Vec<i64>> {
    match result.get("events").and_then(|events| events.get(kind)) {
        None => Some(Vec::new()),
        Some(events) => events.as_object()?.values().map(|event| event.get("date")?.as_i64()).collect(),
    }
}

/// The exchange date of a price bar. Like yfinance, a bar at 22:00 or 23:00 on
/// the hour belongs to the next day: Yahoo shifts some midnight bars near clock changes.
fn bar_date(time: i64, zone: Tz) -> Option<Date> {
    let local = DateTime::from_timestamp(time, 0)?.with_timezone(&zone);
    let shift = match (local.hour(), local.minute()) {
        (22, 0) => 2,
        (23, 0) => 1,
        _ => 0,
    };
    local_date(time + shift * 3600, zone)
}

fn local_date(time: i64, zone: Tz) -> Option<Date> {
    Some(DateTime::from_timestamp(time, 0)?.with_timezone(&zone).date_naive())
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
    }
}

/// A JSON value with Python types, for `str(value)` and `repr(value)`.
pub(crate) fn python_value(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(flag) => Json::Bool(*flag),
        Value::Number(number) => match number.as_i64() {
            Some(integer) => Json::Int(integer),
            None => Json::Float(number.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(text) => Json::Str(text.clone()),
        Value::Array(items) => Json::List(items.iter().map(python_value).collect()),
        Value::Object(items) => {
            Json::Dict(pyfmt::Dict(items.iter().map(|(key, item)| (key.clone(), python_value(item))).collect()))
        }
    }
}

/// Python `str.isspace` for one character: Unicode whitespace and the four ASCII separators.
pub(crate) fn is_python_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// Python `str.strip()`.
pub(crate) fn py_strip(text: &str) -> &str {
    text.trim_matches(is_python_space)
}

/// Python `" ".join(text.split())`: words separated by single spaces.
pub(crate) fn single_spaces(text: &str) -> String {
    text.split(is_python_space).filter(|word| !word.is_empty()).collect::<Vec<_>>().join(" ")
}

/// Python `urllib.parse.quote(text, safe="")`: percent-encode every byte except ASCII letters, digits and `_.-~`.
pub(crate) fn quote(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"_.-~".contains(&byte) {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// `datetime.now(timezone.utc).isoformat()`: microseconds unless zero, then `+00:00`.
pub(crate) fn utc_now_iso() -> String {
    let now = Utc::now();
    let seconds = now.format("%Y-%m-%dT%H:%M:%S");
    match now.timestamp_subsec_micros() {
        0 => format!("{seconds}+00:00"),
        micros => format!("{seconds}.{micros:06}+00:00"),
    }
}

/// The calendar date of a provider date text, as `pandas.Timestamp(text).date()` reads it.
///
/// The providers write ISO dates, `Oct 1, 2026` with an optional time such as
/// `2:27 PM`, `01-Oct-2026` and `Oct 1 2026`. Month names can be short or full.
pub(crate) fn provider_date(text: &str) -> Option<Date> {
    const DATES: [&str; 4] = ["%Y-%m-%d", "%B %d, %Y", "%d-%B-%Y", "%B %d %Y"];
    const TIMES: [&str; 5] = [" %H:%M", " %H:%M:%S", " %I:%M %p", " %I:%M:%S %p", "T%H:%M:%S"];
    let text = py_strip(text);
    DATES.iter().find_map(|date| {
        Date::parse_from_str(text, date).ok().or_else(|| {
            TIMES
                .iter()
                .find_map(|time| NaiveDateTime::parse_from_str(text, &format!("{date}{time}")).ok())
                .map(|stamp| stamp.date())
        })
    })
}

/// Python `html.unescape` for numeric references and the HTML 4 entity names.
///
/// Named references take their HTML 5 values, and the legacy names also
/// match without a semicolon, as in Python. Names that only HTML 5 defines,
/// other than `apos`, stay as text.
pub(crate) fn unescape(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        output.push_str(&rest[..position]);
        let after = &rest[position + 1..];
        match character_reference(after) {
            Some((replacement, used)) => {
                output.push_str(&replacement);
                rest = &after[used..];
            }
            None => {
                output.push('&');
                rest = after;
            }
        }
    }
    output.push_str(rest);
    output
}

/// The replacement for the reference after an `&`, and the bytes that the reference uses.
fn character_reference(text: &str) -> Option<(String, usize)> {
    if let Some(number) = text.strip_prefix('#') {
        let (digits, radix, offset) = match number.strip_prefix(['x', 'X']) {
            Some(hex) => (hex.bytes().take_while(u8::is_ascii_hexdigit).count(), 16, 2),
            None => (number.bytes().take_while(u8::is_ascii_digit).count(), 10, 1),
        };
        if digits == 0 {
            return None;
        }
        let value = u32::from_str_radix(&text[offset..offset + digits], radix).unwrap_or(u32::MAX);
        let used = offset + digits + usize::from(text[offset + digits..].starts_with(';'));
        return Some((numeric_reference(value), used));
    }
    // A name has 1 to 32 characters other than tab, line feed, form feed, space, `<`, `&`, `#` and `;`.
    let mut end = 0;
    for (count, (position, character)) in text.char_indices().enumerate() {
        if count == 32 || matches!(character, '\t' | '\n' | '\u{c}' | ' ' | '<' | '&' | '#' | ';') {
            break;
        }
        end = position + character.len_utf8();
    }
    if end == 0 {
        return None;
    }
    let semicolon = text[end..].starts_with(';');
    let reference = &text[..end + usize::from(semicolon)];
    let name = &text[..end];
    if (semicolon || LEGACY_ENTITIES.contains(&name))
        && let Some(value) = entity(name)
    {
        return Some((value.to_string(), reference.len()));
    }
    // The longest shorter legacy name at the start, with at least two characters, and the rest as text.
    let starts: Vec<usize> = reference.char_indices().map(|(position, _)| position).collect();
    (2..starts.len()).rev().find_map(|length| {
        let prefix = &reference[..starts[length]];
        let value = entity(prefix).filter(|_| LEGACY_ENTITIES.contains(&prefix))?;
        Some((format!("{value}{}", &reference[starts[length]..]), reference.len()))
    })
}

fn entity(name: &str) -> Option<char> {
    ENTITIES.binary_search_by_key(&name, |(key, _)| key).ok().map(|position| ENTITIES[position].1)
}

/// Python's replacements for invalid numeric references: Windows-1252 text and removed control characters.
fn numeric_reference(value: u32) -> String {
    const WINDOWS_1252: [char; 32] = [
        '\u{20ac}', '\u{81}', '\u{201a}', '\u{192}', '\u{201e}', '\u{2026}', '\u{2020}', '\u{2021}', '\u{2c6}',
        '\u{2030}', '\u{160}', '\u{2039}', '\u{152}', '\u{8d}', '\u{17d}', '\u{8f}', '\u{90}', '\u{2018}', '\u{2019}',
        '\u{201c}', '\u{201d}', '\u{2022}', '\u{2013}', '\u{2014}', '\u{2dc}', '\u{2122}', '\u{161}', '\u{203a}',
        '\u{153}', '\u{9d}', '\u{17e}', '\u{178}',
    ];
    match value {
        0 => "\u{fffd}".into(),
        0x0d => "\r".into(),
        0x80..=0x9f => WINDOWS_1252[(value - 0x80) as usize].to_string(),
        0xd800..=0xdfff => "\u{fffd}".into(),
        _ if value > 0x10ffff => "\u{fffd}".into(),
        0x1..=0x8 | 0xb | 0xe..=0x1f | 0x7f | 0xfdd0..=0xfdef => String::new(),
        _ if value & 0xfffe == 0xfffe => String::new(),
        _ => char::from_u32(value).map(String::from).unwrap_or_default(),
    }
}

/// HTML 5 values of the HTML 4 entity names, `apos`, and the upper-case legacy names, sorted by name.
#[rustfmt::skip]
const ENTITIES: [(&str, char); 259] = [
    ("AElig", '\u{c6}'), ("AMP", '\u{26}'), ("Aacute", '\u{c1}'), ("Acirc", '\u{c2}'), ("Agrave", '\u{c0}'),
    ("Alpha", '\u{391}'), ("Aring", '\u{c5}'), ("Atilde", '\u{c3}'), ("Auml", '\u{c4}'), ("Beta", '\u{392}'),
    ("COPY", '\u{a9}'), ("Ccedil", '\u{c7}'), ("Chi", '\u{3a7}'), ("Dagger", '\u{2021}'), ("Delta", '\u{394}'),
    ("ETH", '\u{d0}'), ("Eacute", '\u{c9}'), ("Ecirc", '\u{ca}'), ("Egrave", '\u{c8}'), ("Epsilon", '\u{395}'),
    ("Eta", '\u{397}'), ("Euml", '\u{cb}'), ("GT", '\u{3e}'), ("Gamma", '\u{393}'), ("Iacute", '\u{cd}'),
    ("Icirc", '\u{ce}'), ("Igrave", '\u{cc}'), ("Iota", '\u{399}'), ("Iuml", '\u{cf}'), ("Kappa", '\u{39a}'),
    ("LT", '\u{3c}'), ("Lambda", '\u{39b}'), ("Mu", '\u{39c}'), ("Ntilde", '\u{d1}'), ("Nu", '\u{39d}'),
    ("OElig", '\u{152}'), ("Oacute", '\u{d3}'), ("Ocirc", '\u{d4}'), ("Ograve", '\u{d2}'), ("Omega", '\u{3a9}'),
    ("Omicron", '\u{39f}'), ("Oslash", '\u{d8}'), ("Otilde", '\u{d5}'), ("Ouml", '\u{d6}'), ("Phi", '\u{3a6}'),
    ("Pi", '\u{3a0}'), ("Prime", '\u{2033}'), ("Psi", '\u{3a8}'), ("QUOT", '\u{22}'), ("REG", '\u{ae}'),
    ("Rho", '\u{3a1}'), ("Scaron", '\u{160}'), ("Sigma", '\u{3a3}'), ("THORN", '\u{de}'), ("Tau", '\u{3a4}'),
    ("Theta", '\u{398}'), ("Uacute", '\u{da}'), ("Ucirc", '\u{db}'), ("Ugrave", '\u{d9}'), ("Upsilon", '\u{3a5}'),
    ("Uuml", '\u{dc}'), ("Xi", '\u{39e}'), ("Yacute", '\u{dd}'), ("Yuml", '\u{178}'), ("Zeta", '\u{396}'),
    ("aacute", '\u{e1}'), ("acirc", '\u{e2}'), ("acute", '\u{b4}'), ("aelig", '\u{e6}'), ("agrave", '\u{e0}'),
    ("alefsym", '\u{2135}'), ("alpha", '\u{3b1}'), ("amp", '\u{26}'), ("and", '\u{2227}'), ("ang", '\u{2220}'),
    ("apos", '\u{27}'), ("aring", '\u{e5}'), ("asymp", '\u{2248}'), ("atilde", '\u{e3}'), ("auml", '\u{e4}'),
    ("bdquo", '\u{201e}'), ("beta", '\u{3b2}'), ("brvbar", '\u{a6}'), ("bull", '\u{2022}'), ("cap", '\u{2229}'),
    ("ccedil", '\u{e7}'), ("cedil", '\u{b8}'), ("cent", '\u{a2}'), ("chi", '\u{3c7}'), ("circ", '\u{2c6}'),
    ("clubs", '\u{2663}'), ("cong", '\u{2245}'), ("copy", '\u{a9}'), ("crarr", '\u{21b5}'), ("cup", '\u{222a}'),
    ("curren", '\u{a4}'), ("dArr", '\u{21d3}'), ("dagger", '\u{2020}'), ("darr", '\u{2193}'), ("deg", '\u{b0}'),
    ("delta", '\u{3b4}'), ("diams", '\u{2666}'), ("divide", '\u{f7}'), ("eacute", '\u{e9}'), ("ecirc", '\u{ea}'),
    ("egrave", '\u{e8}'), ("empty", '\u{2205}'), ("emsp", '\u{2003}'), ("ensp", '\u{2002}'), ("epsilon", '\u{3b5}'),
    ("equiv", '\u{2261}'), ("eta", '\u{3b7}'), ("eth", '\u{f0}'), ("euml", '\u{eb}'), ("euro", '\u{20ac}'),
    ("exist", '\u{2203}'), ("fnof", '\u{192}'), ("forall", '\u{2200}'), ("frac12", '\u{bd}'), ("frac14", '\u{bc}'),
    ("frac34", '\u{be}'), ("frasl", '\u{2044}'), ("gamma", '\u{3b3}'), ("ge", '\u{2265}'), ("gt", '\u{3e}'),
    ("hArr", '\u{21d4}'), ("harr", '\u{2194}'), ("hearts", '\u{2665}'), ("hellip", '\u{2026}'), ("iacute", '\u{ed}'),
    ("icirc", '\u{ee}'), ("iexcl", '\u{a1}'), ("igrave", '\u{ec}'), ("image", '\u{2111}'), ("infin", '\u{221e}'),
    ("int", '\u{222b}'), ("iota", '\u{3b9}'), ("iquest", '\u{bf}'), ("isin", '\u{2208}'), ("iuml", '\u{ef}'),
    ("kappa", '\u{3ba}'), ("lArr", '\u{21d0}'), ("lambda", '\u{3bb}'), ("lang", '\u{27e8}'), ("laquo", '\u{ab}'),
    ("larr", '\u{2190}'), ("lceil", '\u{2308}'), ("ldquo", '\u{201c}'), ("le", '\u{2264}'), ("lfloor", '\u{230a}'),
    ("lowast", '\u{2217}'), ("loz", '\u{25ca}'), ("lrm", '\u{200e}'), ("lsaquo", '\u{2039}'), ("lsquo", '\u{2018}'),
    ("lt", '\u{3c}'), ("macr", '\u{af}'), ("mdash", '\u{2014}'), ("micro", '\u{b5}'), ("middot", '\u{b7}'),
    ("minus", '\u{2212}'), ("mu", '\u{3bc}'), ("nabla", '\u{2207}'), ("nbsp", '\u{a0}'), ("ndash", '\u{2013}'),
    ("ne", '\u{2260}'), ("ni", '\u{220b}'), ("not", '\u{ac}'), ("notin", '\u{2209}'), ("nsub", '\u{2284}'),
    ("ntilde", '\u{f1}'), ("nu", '\u{3bd}'), ("oacute", '\u{f3}'), ("ocirc", '\u{f4}'), ("oelig", '\u{153}'),
    ("ograve", '\u{f2}'), ("oline", '\u{203e}'), ("omega", '\u{3c9}'), ("omicron", '\u{3bf}'), ("oplus", '\u{2295}'),
    ("or", '\u{2228}'), ("ordf", '\u{aa}'), ("ordm", '\u{ba}'), ("oslash", '\u{f8}'), ("otilde", '\u{f5}'),
    ("otimes", '\u{2297}'), ("ouml", '\u{f6}'), ("para", '\u{b6}'), ("part", '\u{2202}'), ("permil", '\u{2030}'),
    ("perp", '\u{22a5}'), ("phi", '\u{3c6}'), ("pi", '\u{3c0}'), ("piv", '\u{3d6}'), ("plusmn", '\u{b1}'),
    ("pound", '\u{a3}'), ("prime", '\u{2032}'), ("prod", '\u{220f}'), ("prop", '\u{221d}'), ("psi", '\u{3c8}'),
    ("quot", '\u{22}'), ("rArr", '\u{21d2}'), ("radic", '\u{221a}'), ("rang", '\u{27e9}'), ("raquo", '\u{bb}'),
    ("rarr", '\u{2192}'), ("rceil", '\u{2309}'), ("rdquo", '\u{201d}'), ("real", '\u{211c}'), ("reg", '\u{ae}'),
    ("rfloor", '\u{230b}'), ("rho", '\u{3c1}'), ("rlm", '\u{200f}'), ("rsaquo", '\u{203a}'), ("rsquo", '\u{2019}'),
    ("sbquo", '\u{201a}'), ("scaron", '\u{161}'), ("sdot", '\u{22c5}'), ("sect", '\u{a7}'), ("shy", '\u{ad}'),
    ("sigma", '\u{3c3}'), ("sigmaf", '\u{3c2}'), ("sim", '\u{223c}'), ("spades", '\u{2660}'), ("sub", '\u{2282}'),
    ("sube", '\u{2286}'), ("sum", '\u{2211}'), ("sup", '\u{2283}'), ("sup1", '\u{b9}'), ("sup2", '\u{b2}'),
    ("sup3", '\u{b3}'), ("supe", '\u{2287}'), ("szlig", '\u{df}'), ("tau", '\u{3c4}'), ("there4", '\u{2234}'),
    ("theta", '\u{3b8}'), ("thetasym", '\u{3d1}'), ("thinsp", '\u{2009}'), ("thorn", '\u{fe}'), ("tilde", '\u{2dc}'),
    ("times", '\u{d7}'), ("trade", '\u{2122}'), ("uArr", '\u{21d1}'), ("uacute", '\u{fa}'), ("uarr", '\u{2191}'),
    ("ucirc", '\u{fb}'), ("ugrave", '\u{f9}'), ("uml", '\u{a8}'), ("upsih", '\u{3d2}'), ("upsilon", '\u{3c5}'),
    ("uuml", '\u{fc}'), ("weierp", '\u{2118}'), ("xi", '\u{3be}'), ("yacute", '\u{fd}'), ("yen", '\u{a5}'),
    ("yuml", '\u{ff}'), ("zeta", '\u{3b6}'), ("zwj", '\u{200d}'), ("zwnj", '\u{200c}'),
];

/// The names that also match without a semicolon.
const LEGACY_ENTITIES: [&str; 106] = [
    "AElig", "AMP", "Aacute", "Acirc", "Agrave", "Aring", "Atilde", "Auml", "COPY", "Ccedil", "ETH", "Eacute", "Ecirc",
    "Egrave", "Euml", "GT", "Iacute", "Icirc", "Igrave", "Iuml", "LT", "Ntilde", "Oacute", "Ocirc", "Ograve", "Oslash",
    "Otilde", "Ouml", "QUOT", "REG", "THORN", "Uacute", "Ucirc", "Ugrave", "Uuml", "Yacute", "aacute", "acirc",
    "acute", "aelig", "agrave", "amp", "aring", "atilde", "auml", "brvbar", "ccedil", "cedil", "cent", "copy",
    "curren", "deg", "divide", "eacute", "ecirc", "egrave", "eth", "euml", "frac12", "frac14", "frac34", "gt",
    "iacute", "icirc", "iexcl", "igrave", "iquest", "iuml", "laquo", "lt", "macr", "micro", "middot", "nbsp", "not",
    "ntilde", "oacute", "ocirc", "ograve", "ordf", "ordm", "oslash", "otilde", "ouml", "para", "plusmn", "pound",
    "quot", "raquo", "reg", "sect", "shy", "sup1", "sup2", "sup3", "szlig", "thorn", "times", "uacute", "ucirc",
    "ugrave", "uml", "uuml", "yacute", "yen", "yuml",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescape_matches_python_html_unescape() {
        // Expected values from Python 3.14 `html.unescape`.
        let cases = [
            ("plain", "plain"),
            ("&amp;", "&"),
            ("&amp", "&"),
            ("&AMP;", "&"),
            ("&AMP", "&"),
            ("&Amp;", "&Amp;"),
            ("&ampfoo", "&foo"),
            ("&ampfoo;", "&foo;"),
            ("&notin;", "\u{2209}"),
            ("&notit;", "\u{ac}it;"),
            ("&notin", "\u{ac}in"),
            ("&lang;", "\u{27e8}"),
            ("&rang;", "\u{27e9}"),
            ("&apos;", "'"),
            ("&#39;", "'"),
            ("&#x27;", "'"),
            ("&#X41;", "A"),
            ("&#x41", "A"),
            ("&#65", "A"),
            ("&#;", "&#;"),
            ("&#x;", "&#x;"),
            ("&#xg;", "&#xg;"),
            ("&#0;", "\u{fffd}"),
            ("&#13;", "\r"),
            ("&#128;", "\u{20ac}"),
            ("&#129;", "\u{81}"),
            ("&#150;", "\u{2013}"),
            ("&#x110000;", "\u{fffd}"),
            ("&#xD800;", "\u{fffd}"),
            ("&#1;", ""),
            ("&#11;", ""),
            ("&#x7f;", ""),
            ("&#xFDD0;", ""),
            ("&#xFFFE;", ""),
            ("&#x1FFFF;", ""),
            ("&#99999999999999999999;", "\u{fffd}"),
            ("&copy 2020", "\u{a9} 2020"),
            ("&COPY;", "\u{a9}"),
            ("&sup1;", "\u{b9}"),
            ("&sup12", "\u{b9}2"),
            ("& amp;", "& amp;"),
            ("&&amp;", "&&"),
            ("a&b", "a&b"),
            ("&;", "&;"),
            ("&nbsp;x", "\u{a0}x"),
            ("&nbsp", "\u{a0}"),
            ("&eacute;t&eacute;", "\u{e9}t\u{e9}"),
            ("&aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;", "&aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;"),
            ("&ampamp;", "&amp;"),
            ("Caf&eacute; &#x2014; tools&#39;", "Caf\u{e9} \u{2014} tools'"),
            ("&#x2014", "\u{2014}"),
            ("&frac12;&frac12", "\u{bd}\u{bd}"),
        ];
        for (text, expected) in cases {
            assert_eq!(unescape(text), expected, "{text:?}");
        }
        assert!(ENTITIES.windows(2).all(|pair| pair[0].0 < pair[1].0), "the table is sorted for binary search");
        assert!(LEGACY_ENTITIES.iter().all(|name| entity(name).is_some()));
    }

    #[test]
    fn whitespace_follows_python_strings() {
        // Expected values from Python's `str.strip()` and `" ".join(str.split())`.
        let cases = [
            ("  a \t b\n", "a \t b", "a b"),
            ("\u{1c} a \u{1f}", "a", "a"),
            ("\u{a0}x\u{2003}y\u{3000}", "x\u{2003}y", "x y"),
            ("a\u{200b}b", "a\u{200b}b", "a\u{200b}b"),
            ("\u{feff}a", "\u{feff}a", "\u{feff}a"),
        ];
        for (text, stripped, joined) in cases {
            assert_eq!(py_strip(text), stripped, "{text:?}");
            assert_eq!(single_spaces(text), joined, "{text:?}");
        }
    }

    #[test]
    fn provider_dates_follow_pandas() {
        // Expected values from `pandas.Timestamp(text).date()`; None is NaT or an error.
        let cases = [
            ("Oct 1, 2026", Some("2026-10-01")),
            ("Oct 30, 2026 2:27 PM", Some("2026-10-30")),
            ("October 30, 2026", Some("2026-10-30")),
            ("2026-11-01", Some("2026-11-01")),
            ("01-Oct-2026", Some("2026-10-01")),
            ("Oct 29 2026", Some("2026-10-29")),
            (" Sep 30 2026 ", Some("2026-09-30")),
            ("2026-10-30T14:27:00", Some("2026-10-30")),
            ("Oct 30, 2026 14:27", Some("2026-10-30")),
            ("not a date", None),
            ("", None),
        ];
        for (text, expected) in cases {
            assert_eq!(provider_date(text).map(|date| date.to_string()), expected.map(String::from), "{text:?}");
        }
    }

    #[test]
    fn quote_matches_python_urllib() {
        assert_eq!(quote("BRK-B"), "BRK-B");
        assert_eq!(quote("^GSPC"), "%5EGSPC");
        assert_eq!(quote("EURUSD=X"), "EURUSD%3DX");
        assert_eq!(quote("A B/é"), "A%20B%2F%C3%A9");
    }

    #[test]
    fn truthiness_follows_python() {
        let falsy = ["null", "false", "0", "0.0", "\"\"", "[]", "{}"];
        let truthy_values = ["true", "1", "-0.5", "\"x\"", "[0]", "{\"a\": null}"];
        assert!(falsy.iter().all(|text| !truthy(&serde_json::from_str(text).unwrap())));
        assert!(truthy_values.iter().all(|text| truthy(&serde_json::from_str(text).unwrap())));
        assert_eq!(
            python_value(&serde_json::json!({"a": [1, 2.5, true, null]})).py_str(),
            "{'a': [1, 2.5, True, None]}"
        );
    }
}
