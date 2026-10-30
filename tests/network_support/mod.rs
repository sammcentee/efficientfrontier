//! A recording fake network and synthetic provider responses for the offline network tests.
//!
//! The synthetic Yahoo charts follow the structure of the recorded response
//! in `tests/fixtures/network/chart.json`, with generated prices.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use chrono::{Datelike, NaiveDate, NaiveTime, TimeZone, Weekday};
use chrono_tz::Tz;
use efficient_frontier::frame::{Date, Frame};
use efficient_frontier::net::{CHART_URL, Reply, Request, Response, with_fake_network};
use serde_json::{Value, json};

pub fn day(text: &str) -> Date {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
}

pub fn strings<S: AsRef<str>>(names: &[S]) -> Vec<String> {
    names.iter().map(|name| name.as_ref().to_string()).collect()
}

/// `pandas.bdate_range(start, periods=count)`.
pub fn business_days(start: &str, count: usize) -> Vec<Date> {
    let mut dates = Vec::with_capacity(count);
    let mut date = day(start);
    while dates.len() < count {
        if !matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
            dates.push(date);
        }
        date = date.succ_opt().unwrap();
    }
    dates
}

/// `price_table` of the Python tests: column `j` holds `100 + j`, `101 + j`, ... in date order.
pub fn price_table<S: AsRef<str>>(symbols: &[S], dates: &[Date]) -> Frame {
    Frame::from_columns(
        dates.to_vec(),
        symbols
            .iter()
            .enumerate()
            .map(|(j, symbol)| (symbol.as_ref().to_string(), (0..dates.len()).map(|i| (100 + i + j) as f64).collect()))
            .collect(),
    )
}

/// The bytes of a file in `tests/fixtures/network`.
pub fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/network").join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// Run `body` with `respond` in place of the network; return the result and every request in order.
pub fn offline<T>(mut respond: impl FnMut(&Request) -> Reply + 'static, body: impl FnOnce() -> T) -> (T, Vec<Request>) {
    let calls: Rc<RefCell<Vec<Request>>> = Rc::default();
    let seen = Rc::clone(&calls);
    let result = with_fake_network(
        move |request| {
            seen.borrow_mut().push(request.clone());
            respond(request)
        },
        body,
    );
    let calls = calls.borrow().clone();
    (result, calls)
}

pub fn ok(body: impl Into<Vec<u8>>) -> Reply {
    Ok(Response::new(200, body))
}

pub fn status(code: u16, body: impl Into<Vec<u8>>) -> Reply {
    Ok(Response::new(code, body))
}

/// The symbol of a Yahoo chart request.
pub fn chart_symbol(request: &Request) -> Option<String> {
    let encoded = request.url.strip_prefix(CHART_URL)?;
    let mut bytes = Vec::new();
    let mut rest = encoded.as_bytes();
    while let Some((&byte, tail)) = rest.split_first() {
        if byte == b'%' {
            bytes.push(u8::from_str_radix(std::str::from_utf8(&tail[..2]).unwrap(), 16).unwrap());
            rest = &tail[2..];
        } else {
            bytes.push(byte);
            rest = tail;
        }
    }
    Some(String::from_utf8(bytes).unwrap())
}

/// The symbols of all chart requests, in order.
pub fn chart_symbols(calls: &[Request]) -> Vec<String> {
    calls.iter().filter_map(chart_symbol).collect()
}

/// Midnight of a date in a timezone, as Unix seconds.
pub fn midnight(zone: Tz, date: &str) -> i64 {
    zone.from_local_datetime(&day(date).and_time(NaiveTime::MIN)).unwrap().timestamp()
}

/// A local time in a timezone, as Unix seconds.
pub fn local_time(zone: Tz, date: Date, hour: u32, minute: u32) -> i64 {
    zone.from_local_datetime(&date.and_hms_opt(hour, minute, 0).unwrap()).unwrap().timestamp()
}

/// One bar of a synthetic chart.
#[derive(Clone, Debug)]
pub struct Bar {
    pub time: i64,
    pub close: Option<f64>,
    pub adjusted: Option<f64>,
}

/// A synthetic Yahoo chart for one symbol.
#[derive(Clone, Debug)]
pub struct Chart {
    pub symbol: String,
    pub zone: Tz,
    pub instrument: &'static str,
    pub bars: Vec<Bar>,
    /// Whether the response has adjusted closes.
    pub adjusted: bool,
    pub dividends: Vec<i64>,
    pub capital_gains: Vec<i64>,
    /// Bars that Yahoo returns whatever the window, such as the latest price.
    pub late: Vec<Bar>,
}

impl Chart {
    /// Equity bars at the 09:30 open in `zone`. Each close is 1% above the adjusted close.
    pub fn new(symbol: &str, zone: Tz, dates: &[Date], adjusted: &[Option<f64>]) -> Self {
        let bars = dates
            .iter()
            .zip(adjusted)
            .map(|(date, adjusted)| Bar {
                time: local_time(zone, *date, 9, 30),
                close: adjusted.map(|value| value * 1.01),
                adjusted: *adjusted,
            })
            .collect();
        Self {
            symbol: symbol.to_string(),
            zone,
            instrument: "EQUITY",
            bars,
            adjusted: true,
            dividends: Vec::new(),
            capital_gains: Vec::new(),
            late: Vec::new(),
        }
    }

    /// The bars of one frame column in New York; NaN is a missing price.
    pub fn from_column(frame: &Frame, symbol: &str) -> Self {
        let values: Vec<Option<f64>> =
            frame.column(symbol).unwrap().into_iter().map(|value| (!value.is_nan()).then_some(value)).collect();
        Self::new(symbol, chrono_tz::America::New_York, &frame.index, &values)
    }

    /// Set the adjusted close of the bar on `date`.
    pub fn set(&mut self, date: Date, adjusted: Option<f64>) {
        let time = local_time(self.zone, date, 9, 30);
        let bar = self.bars.iter_mut().find(|bar| bar.time == time).expect("a bar on the date");
        bar.adjusted = adjusted;
        bar.close = adjusted.map(|value| value * 1.01);
    }

    /// Remove the bar on `date`.
    pub fn remove(&mut self, date: Date) {
        let time = local_time(self.zone, date, 9, 30);
        self.bars.retain(|bar| bar.time != time);
    }

    /// The response body for one request window. As Yahoo does, the window includes both ends.
    pub fn body(&self, period1: i64, period2: i64) -> String {
        let window = |time: &i64| (period1..=period2).contains(time);
        let bars: Vec<&Bar> = self.bars.iter().filter(|bar| window(&bar.time)).chain(&self.late).collect();
        let column = |value: fn(&Bar) -> Option<f64>| bars.iter().map(|bar| value(bar)).collect::<Vec<_>>();
        let quote = if bars.is_empty() {
            json!({})
        } else {
            let close = column(|bar| bar.close);
            json!({
                "high": column(|bar| bar.close.map(|value| value + 1.0)),
                "open": column(|bar| bar.close.map(|value| value - 0.5)),
                "volume": bars.iter().map(|bar| bar.close.map(|_| 1_000_000)).collect::<Vec<_>>(),
                "low": column(|bar| bar.close.map(|value| value - 1.0)),
                "close": close,
            })
        };
        let mut indicators = json!({ "quote": [quote] });
        if self.adjusted {
            indicators["adjclose"] =
                if bars.is_empty() { json!([{}]) } else { json!([{ "adjclose": column(|bar| bar.adjusted) }]) };
        }
        // The current offset; the reader takes dates from the timezone name, as yfinance does.
        let first = self.bars.first().map_or(0, |bar| bar.time);
        let offset =
            self.zone.offset_from_utc_datetime(&chrono::DateTime::from_timestamp(first, 0).unwrap().naive_utc());
        let mut result = json!({
            "meta": {
                "currency": "USD", "symbol": self.symbol, "exchangeName": "NMS", "fullExchangeName": "NasdaqGS",
                "instrumentType": self.instrument, "firstTradeDate": 946909800, "regularMarketTime": 1792675800,
                "hasPrePostMarketData": true,
                "gmtoffset": chrono::Offset::fix(&offset).local_minus_utc(), "timezone": "EDT",
                "exchangeTimezoneName": self.zone.name(), "regularMarketPrice": 103.0, "priceHint": 2,
                "dataGranularity": "1d", "range": "",
                "validRanges": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            },
            "indicators": indicators,
        });
        if !bars.is_empty() {
            result["timestamp"] = json!(bars.iter().map(|bar| bar.time).collect::<Vec<_>>());
        }
        let events = |times: &[i64]| -> Value {
            times
                .iter()
                .filter(|time| window(time))
                .map(|time| (time.to_string(), json!({ "amount": 0.5, "date": time })))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let (dividends, gains) = (events(&self.dividends), events(&self.capital_gains));
        if dividends.as_object().is_some_and(|map| !map.is_empty())
            || gains.as_object().is_some_and(|map| !map.is_empty())
        {
            result["events"] = json!({ "dividends": dividends, "capitalGains": gains });
        }
        json!({ "chart": { "result": [result], "error": null } }).to_string()
    }
}

/// Charts for every column of a frame, in New York.
pub fn charts(frame: &Frame) -> HashMap<String, Chart> {
    frame.columns.iter().map(|symbol| (symbol.clone(), Chart::from_column(frame, symbol))).collect()
}

/// Answer chart requests as Yahoo does: the bars in the window, or 404 for an unknown symbol.
pub fn yahoo(charts: HashMap<String, Chart>) -> impl FnMut(&Request) -> Reply + 'static {
    move |request| {
        let symbol = chart_symbol(request).expect("a Yahoo chart request");
        let window = |name: &str| request.param(name).unwrap().parse::<i64>().unwrap();
        match charts.get(&symbol) {
            Some(chart) => ok(chart.body(window("period1"), window("period2"))),
            None => status(404, fixture("chart_not_found.json")),
        }
    }
}
