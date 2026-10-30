//! Yahoo prices, SPY and QQQ benchmarks, and the original ticker list.
//!
//! Port of the network and ODS parts of `tests/test_data.py`. A fake network
//! answers with synthetic Yahoo chart responses in place of the monkeypatched
//! `yfinance.download`, so each test also covers how responses are read. The
//! Python cases "no_close" and "flat" test pandas table shapes that only a
//! monkeypatched download returns; they have no equivalent here.

mod network_support;

use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

use chrono_tz::{America, Asia, Europe, Tz};
use efficient_frontier::data::{
    demo_prices, download_adjusted_close, download_benchmarks, download_prices, original_tickers,
};
use efficient_frontier::frame::{Date, Frame};
use efficient_frontier::net::Request;
use network_support::*;

const START: &str = "2020-01-01";
const END: &str = "2021-01-01";

/// The first 120 demo prices as AAA and BBB.
fn prices() -> Frame {
    let mut frame = demo_prices().rows(0, 120).select(&["DEMO_EQUITY", "DEMO_TECH"]).unwrap();
    frame.columns = strings(&["AAA", "BBB"]);
    frame
}

fn benchmark_prices() -> Frame {
    let mut frame = prices();
    frame.columns = strings(&["SPY", "QQQ"]);
    frame
}

fn select_rows(frame: &Frame, dates: &[Date]) -> Frame {
    let rows: Vec<Vec<f64>> =
        dates.iter().map(|date| frame.row(frame.index.iter().position(|value| value == date).unwrap())).collect();
    Frame::from_rows(dates.to_vec(), frame.columns.clone(), &rows)
}

fn same_values(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
}

#[test]
fn download_selects_adjusted_close_in_requested_order() {
    for symbols in [vec!["AAA"], vec!["BBB", "AAA"]] {
        let prices = prices();
        let (result, calls) =
            offline(yahoo(charts(&prices)), || download_prices(&strings(&symbols), day(START), day(END)));
        // Each synthetic close is 1% above the adjusted close, so equality shows the adjusted series.
        assert_eq!(result.unwrap(), prices.select(&symbols).unwrap());
        assert_eq!(chart_symbols(&calls), symbols);
        for call in &calls {
            let query: Vec<(&str, &str)> =
                call.query.iter().map(|(name, value)| (name.as_str(), value.as_str())).collect();
            let (period1, period2) =
                (midnight(America::New_York, START).to_string(), midnight(America::New_York, END).to_string());
            assert_eq!(
                query,
                [
                    ("period1", period1.as_str()),
                    ("period2", period2.as_str()),
                    ("interval", "1d"),
                    ("includePrePost", "False"),
                    ("events", "div,splits,capitalGains"),
                ]
            );
            assert_eq!(call.timeout, Duration::from_secs(10));
        }
    }
}

#[test]
fn download_preserves_exchange_suffixes_and_normalizes_share_classes() {
    let zones: [(&str, Tz); 6] = [
        ("VOD.L", Europe::London),
        ("BMW.DE", Europe::Berlin),
        ("IWDA.AS", Europe::Amsterdam),
        ("7203.T", Asia::Tokyo),
        ("BRK-B", America::New_York),
        ("BF-B", America::New_York),
    ];
    let prices = prices();
    let values: Vec<Option<f64>> = prices.col(0).into_iter().map(Some).collect();
    let charts: HashMap<String, Chart> = zones
        .iter()
        .map(|(symbol, zone)| (symbol.to_string(), Chart::new(symbol, *zone, &prices.index, &values)))
        .collect();
    let request = strings(&["vod.l", "bmw.de", "iwda.as", "7203.t", "brk.b", "bf.b", "VOD.L"]);
    let (result, calls) = offline(yahoo(charts), || download_prices(&request, day(START), day(END)));
    let result = result.unwrap();
    assert_eq!(result.columns, zones.map(|(symbol, _)| symbol));
    assert_eq!(result.index, prices.index);
    for j in 0..result.ncols() {
        assert_eq!(result.col(j), prices.col(0));
    }
    // yfinance asks for midnight to midnight at the exchange: other exchanges repeat the first request.
    let mut expected = Vec::new();
    for (symbol, zone) in zones {
        expected.push((symbol.to_string(), midnight(America::New_York, START)));
        if zone != America::New_York {
            expected.push((symbol.to_string(), midnight(zone, START)));
        }
    }
    let requested: Vec<(String, i64)> = calls
        .iter()
        .map(|call| (chart_symbol(call).unwrap(), call.param("period1").unwrap().parse().unwrap()))
        .collect();
    assert_eq!(requested, expected);
}

#[test]
fn download_rejects_an_empty_universe_before_fetching() {
    for symbols in [vec![], vec!["", " "]] {
        let (result, calls) = offline(
            |_| panic!("An empty universe must be rejected before downloading prices."),
            || download_prices(&strings(&symbols), day(START), day(END)),
        );
        assert_eq!(result.unwrap_err().message(), "Enter at least one ticker.");
        assert!(calls.is_empty());
    }
}

#[test]
fn download_requires_the_start_before_the_end() {
    for (start, end) in [(START, START), (END, START)] {
        let (result, calls) = offline(
            |_| panic!("An invalid range must not reach Yahoo."),
            || download_prices(&strings(&["AAA"]), day(start), day(end)),
        );
        assert_eq!(result.unwrap_err().message(), "Choose valid start and end dates, with the start before the end.");
        assert!(calls.is_empty());
    }
}

#[test]
fn download_failures_never_fall_back_or_drop_assets() {
    let no_prices = "Yahoo returned no prices. Check the tickers and date range, or upload a CSV.";
    let no_bbb = "Yahoo returned no prices for: BBB. Check these symbols and their available history, or upload a CSV.";
    let cases = [
        ("empty", no_prices),
        ("absent", no_bbb),
        ("all_missing", no_bbb),
        ("down", no_bbb),
        (
            "partial",
            "Missing prices for: BBB. Choose a common date range with complete history or correct the CSV; \
             missing prices are never filled.",
        ),
        // yfinance records a failed request as an empty table, so an offline download has no prices.
        ("network", no_prices),
    ];
    for (failure, message) in cases {
        let prices = prices();
        let mut charts = charts(&prices);
        match failure {
            "empty" => charts.values_mut().for_each(|chart| chart.bars.clear()),
            "absent" => drop(charts.remove("BBB")),
            "all_missing" => prices.index.iter().for_each(|date| charts.get_mut("BBB").unwrap().set(*date, None)),
            "partial" => charts.get_mut("BBB").unwrap().set(prices.index[5], None),
            _ => {}
        }
        let mut yahoo = yahoo(charts);
        let respond = move |request: &Request| match failure {
            "network" => Err("connection refused".to_string()),
            "down" if chart_symbol(request).as_deref() == Some("BBB") => ok("<html>Will be right back</html>"),
            _ => yahoo(request),
        };
        let (result, _) = offline(respond, || download_prices(&strings(&["AAA", "BBB"]), day(START), day(END)));
        assert_eq!(result.unwrap_err().message(), message, "{failure}");
    }
}

#[test]
fn an_incomplete_download_keeps_every_requested_column() {
    let dates = business_days("2026-10-05", 5);
    let mut charts = HashMap::new();
    charts.insert("AAA".to_string(), Chart::new("AAA", America::New_York, &dates, &[Some(10.0); 5]));
    let gap = [Some(20.0), None, Some(21.0), Some(22.0), Some(23.0)];
    charts.insert("CCC".to_string(), Chart::new("CCC", America::New_York, &dates, &gap));
    let request = strings(&["AAA", "BBB", "CCC"]);
    let (result, calls) =
        offline(yahoo(charts), || download_adjusted_close(&request, day("2026-10-05"), day("2026-10-12"), true));
    let result = result.unwrap();
    assert_eq!(result.columns, request);
    assert_eq!(result.index, dates);
    assert!(result.col(1).iter().all(|value| value.is_nan()));
    assert!(same_values(&result.col(2), &[20.0, f64::NAN, 21.0, 22.0, 23.0]));
    // An error response is repeated once, as yfinance repeats it with a renewed session.
    assert_eq!(chart_symbols(&calls), ["AAA", "BBB", "BBB", "CCC"]);
}

#[test]
fn missing_adjusted_closes_fall_back_to_closes() {
    let dates = business_days("2026-10-05", 5);
    let mut chart =
        Chart::new("AAA", America::New_York, &dates, &[Some(10.0), Some(11.0), Some(12.0), Some(13.0), Some(14.0)]);
    chart.adjusted = false;
    let closes: Vec<f64> = chart.bars.iter().map(|bar| bar.close.unwrap()).collect();
    let (result, _) = offline(yahoo(HashMap::from([("AAA".to_string(), chart)])), || {
        download_prices(&strings(&["AAA"]), day("2026-10-05"), day("2026-10-12"))
    });
    assert_eq!(result.unwrap().col(0), closes);
}

#[test]
fn a_late_price_after_the_end_is_dropped() {
    let dates = business_days("2026-10-05", 5);
    let mut chart =
        Chart::new("AAA", America::New_York, &dates, &[Some(10.0), Some(11.0), Some(12.0), Some(13.0), Some(14.0)]);
    // Yahoo can append the latest price even after the requested end.
    chart.late.push(Bar {
        time: local_time(America::New_York, day("2026-10-09"), 10, 15),
        close: Some(99.0),
        adjusted: Some(99.0),
    });
    let (result, _) = offline(yahoo(HashMap::from([("AAA".to_string(), chart)])), || {
        download_adjusted_close(&strings(&["AAA"]), day("2026-10-05"), day("2026-10-09"), false)
    });
    let result = result.unwrap();
    assert_eq!(result.index, &dates[..4]);
    assert_eq!(result.col(0), [10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn a_second_row_on_the_last_date_replaces_the_first() {
    let dates = business_days("2026-10-05", 5);
    let mut chart =
        Chart::new("AAA", America::New_York, &dates, &[Some(10.0), Some(11.0), Some(12.0), Some(13.0), Some(14.0)]);
    chart.late.push(Bar {
        time: local_time(America::New_York, dates[4], 15, 45),
        close: Some(14.5),
        adjusted: Some(14.5),
    });
    let (result, _) = offline(yahoo(HashMap::from([("AAA".to_string(), chart)])), || {
        download_prices(&strings(&["AAA"]), day("2026-10-05"), day("2026-10-10"))
    });
    let result = result.unwrap();
    assert_eq!(result.index, dates);
    assert_eq!(result.col(0), [10.0, 11.0, 12.0, 13.0, 14.5]);
}

#[test]
fn earlier_rows_on_one_date_keep_the_first() {
    let dates = business_days("2026-10-05", 5);
    let mut chart =
        Chart::new("AAA", America::New_York, &dates, &[Some(10.0), Some(11.0), Some(12.0), Some(13.0), Some(14.0)]);
    let repeat = Bar { time: local_time(America::New_York, dates[1], 15, 0), close: Some(99.0), adjusted: Some(99.0) };
    chart.bars.insert(2, repeat);
    let (result, _) = offline(yahoo(HashMap::from([("AAA".to_string(), chart)])), || {
        download_prices(&strings(&["AAA"]), day("2026-10-05"), day("2026-10-10"))
    });
    let result = result.unwrap();
    assert_eq!(result.index, dates);
    assert_eq!(result.col(0), [10.0, 11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn an_event_after_the_last_price_adds_a_missing_row() {
    let dates = business_days("2026-10-05", 4);
    let window = (day("2026-10-05"), day("2026-10-12"));
    let event = local_time(America::New_York, day("2026-10-09"), 9, 30);
    let inside = local_time(America::New_York, dates[1], 9, 30);
    for (instrument, kind, added) in
        [("EQUITY", "dividend", true), ("EQUITY", "capital gain", false), ("ETF", "capital gain", true)]
    {
        let mut chart = Chart::new("AAA", America::New_York, &dates, &[Some(10.0), Some(11.0), Some(12.0), Some(13.0)]);
        chart.instrument = instrument;
        let events = if kind == "dividend" { &mut chart.dividends } else { &mut chart.capital_gains };
        events.extend([inside, event]);
        let charts = HashMap::from([("AAA".to_string(), chart)]);
        let (result, _) =
            offline(yahoo(charts.clone()), || download_adjusted_close(&strings(&["AAA"]), window.0, window.1, false));
        let result = result.unwrap();
        let label = format!("{instrument} {kind}");
        if added {
            // keepna=True keeps the row of missing prices that yfinance adds for the event.
            assert_eq!(result.index.last(), Some(&day("2026-10-09")), "{label}");
            assert!(same_values(&result.col(0), &[10.0, 11.0, 12.0, 13.0, f64::NAN]), "{label}");
            let (validated, _) = offline(yahoo(charts), || download_prices(&strings(&["AAA"]), window.0, window.1));
            assert!(validated.unwrap_err().message().starts_with("Missing prices for: AAA."), "{label}");
        } else {
            assert_eq!(result.index, dates, "{label}");
        }
    }
}

#[test]
fn bars_stamped_just_before_midnight_belong_to_the_next_day() {
    // A London bar at 22:00 or 23:00 local time on the hour is Yahoo's midnight bar of the next day.
    let dates = business_days("2026-10-05", 5);
    let mut chart = Chart::new("FX", Europe::London, &dates, &[Some(1.0), Some(1.1), Some(1.2), Some(1.3), Some(1.4)]);
    for bar in &mut chart.bars {
        bar.time -= 9 * 3600 + 30 * 60;
    }
    chart.bars[2].time -= 3600;
    chart.bars[3].time -= 2 * 3600;
    let (result, calls) = offline(yahoo(HashMap::from([("FX".to_string(), chart)])), || {
        download_prices(&strings(&["FX"]), day("2026-10-05"), day("2026-10-10"))
    });
    let result = result.unwrap();
    assert_eq!(result.index, dates);
    assert_eq!(result.col(0), [1.0, 1.1, 1.2, 1.3, 1.4]);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].param("period1"), Some(midnight(Europe::London, "2026-10-05").to_string().as_str()));
}

#[test]
fn a_skipped_exchange_midnight_fails_as_in_pandas() {
    // Chile moves its clocks from midnight to 01:00 on 2026-09-06, so that midnight does not exist.
    let dates = business_days("2026-09-07", 5);
    let chart = Chart::new("SQM-B.SN", America::Santiago, &dates, &[Some(1.0); 5]);
    let (result, calls) = offline(yahoo(HashMap::from([("SQM-B.SN".to_string(), chart)])), || {
        download_prices(&strings(&["SQM-B.SN"]), day("2026-09-06"), day("2026-09-14"))
    });
    assert_eq!(
        result.unwrap_err().message(),
        "Yahoo returned no prices. Check the tickers and date range, or upload a CSV."
    );
    assert_eq!(calls.len(), 1);
}

#[test]
fn recorded_chart_structure_is_read() {
    let respond = |request: &Request| {
        assert_eq!(chart_symbol(request).as_deref(), Some("EXMPL"));
        ok(fixture("chart.json"))
    };
    let (result, _) =
        offline(respond, || download_adjusted_close(&strings(&["EXMPL"]), day("2026-10-16"), day("2026-10-23"), true));
    let result = result.unwrap();
    assert_eq!(result.index, ["2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"].map(day));
    assert!(same_values(&result.col(0), &[99.5, 101.25, 102.5, f64::NAN, 103.0]));
}

#[test]
fn benchmark_expected_dates_are_never_changed_or_fetched_when_invalid() {
    let dates = benchmark_prices().index;
    let reversed: Vec<Date> = dates.iter().rev().copied().collect();
    let duplicate: Vec<Date> = dates.iter().chain(&dates[..1]).copied().collect();
    for expected in [dates[..4].to_vec(), reversed, duplicate] {
        let original = expected.clone();
        let (result, calls) = offline(
            |_| panic!("Invalid expected dates must fail before a network request."),
            || download_benchmarks(&expected),
        );
        assert_eq!(
            result.unwrap_err().message(),
            "Benchmark comparison requires at least 5 unique dates in increasing order, without times or a timezone."
        );
        assert!(calls.is_empty());
        assert_eq!(expected, original);
    }
}

#[test]
fn benchmark_download_selects_every_requested_date_without_filling() {
    for stride in [1, 5, 21] {
        let table = benchmark_prices();
        let expected_dates: Vec<Date> = table.index.iter().step_by(stride).take(5).copied().collect();
        // A price outside the requested observations must not affect the comparison.
        let excluded = *table.index.iter().find(|date| !expected_dates.contains(date)).unwrap();
        let mut charts = charts(&table);
        charts.get_mut("SPY").unwrap().set(excluded, None);
        let (result, calls) = offline(yahoo(charts), || download_benchmarks(&expected_dates));
        assert_eq!(result.unwrap(), select_rows(&table, &expected_dates), "stride {stride}");
        assert_eq!(chart_symbols(&calls), ["SPY", "QQQ"]);
        let after_last = (expected_dates[4] + chrono::Days::new(1)).to_string();
        for call in &calls {
            assert_eq!(
                call.param("period1"),
                Some(midnight(America::New_York, &expected_dates[0].to_string()).to_string().as_str())
            );
            assert_eq!(call.param("period2"), Some(midnight(America::New_York, &after_last).to_string().as_str()));
            assert_eq!(call.param("interval"), Some("1d"));
        }
    }
}

#[test]
fn benchmark_download_never_drops_an_expected_date_or_symbol() {
    let cases = [
        ("date", "Yahoo benchmarks are missing 1 requested dates, including: 2020-01-15."),
        (
            "price",
            "Missing prices for: QQQ. Choose a common date range with complete history or correct the CSV; \
             missing prices are never filled.",
        ),
        (
            "symbol",
            "Yahoo returned no prices for: QQQ. Check these symbols and their available history, or upload a CSV.",
        ),
    ];
    for (failure, message) in cases {
        let table = benchmark_prices();
        let expected_dates: Vec<Date> = table.index.iter().step_by(5).take(5).copied().collect();
        let mut charts = charts(&table);
        match failure {
            "date" => charts.values_mut().for_each(|chart| chart.remove(expected_dates[2])),
            "price" => charts.get_mut("QQQ").unwrap().set(expected_dates[2], None),
            _ => drop(charts.remove("QQQ")),
        }
        let (result, _) = offline(yahoo(charts), || download_benchmarks(&expected_dates));
        assert_eq!(result.unwrap_err().message(), message, "{failure}");
    }
}

#[test]
fn original_tickers_reads_bundled_ods() {
    let tickers = original_tickers(None, false).unwrap();
    let unique: std::collections::HashSet<&String> = tickers.iter().collect();
    assert_eq!((tickers.len(), unique.len()), (60, 60));
    assert_eq!(tickers[..3], ["AAPL", "NVDA", "AMZN"]);
    assert!(tickers.iter().all(|ticker| ticker.chars().all(|c| c.is_ascii_uppercase())));
}

#[test]
fn original_tickers_reads_an_explicit_path() {
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("holdings.ods");
    std::fs::copy(concat!(env!("CARGO_MANIFEST_DIR"), "/spy_holdings.ods"), &path).unwrap();
    assert_eq!(original_tickers(Some(&path), false).unwrap(), original_tickers(None, false).unwrap());
}

#[test]
fn original_tickers_missing_file_is_actionable() {
    let folder = tempfile::tempdir().unwrap();
    let error = original_tickers(Some(&folder.path().join("absent.ods")), false).unwrap_err();
    assert!(error.is_value());
    assert_eq!(error.message(), "The original spy_holdings.ods ticker list could not be read.");
}

#[test]
fn original_preset_updates_renamed_symbol_without_changing_historical_list() {
    let historical = original_tickers(None, false).unwrap();
    let current = original_tickers(None, true).unwrap();
    assert!(historical.contains(&"MMC".to_string()) && !historical.contains(&"MRSH".to_string()));
    assert!(current.contains(&"MRSH".to_string()) && !current.contains(&"MMC".to_string()));
    assert_eq!((current.len(), historical.len()), (60, 60));
    let renamed: Vec<String> =
        historical.iter().map(|ticker| if ticker == "MMC" { "MRSH".to_string() } else { ticker.clone() }).collect();
    assert_eq!(current, renamed);
}

/// An ODS file whose first table column holds `cells`, one row each.
fn ods(path: &std::path::Path, content: Option<String>) {
    let mut archive = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    archive.start_file("mimetype", zip::write::SimpleFileOptions::default()).unwrap();
    archive.write_all(b"application/vnd.oasis.opendocument.spreadsheet").unwrap();
    if let Some(content) = content {
        archive.start_file("content.xml", zip::write::SimpleFileOptions::default()).unwrap();
        archive.write_all(content.as_bytes()).unwrap();
    }
    archive.finish().unwrap();
}

fn table_xml(cells: &[&str]) -> String {
    let rows: String = cells
        .iter()
        .map(|cell| format!("<table:table-row><table:table-cell><text:p>{cell}</text:p></table:table-cell><table:table-cell><text:p>x</text:p></table:table-cell></table:table-row>"))
        .collect();
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><office:document-content \
         xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\" \
         xmlns:table=\"urn:oasis:names:tc:opendocument:xmlns:table:1.0\" \
         xmlns:text=\"urn:oasis:names:tc:opendocument:xmlns:text:1.0\"><office:body><office:spreadsheet>\
         <table:table table:name=\"Sheet1\">{rows}</table:table></office:spreadsheet></office:body></office:document-content>"
    )
}

/// The content.xml of an ODS file, if any, and the tickers or error message that it gives.
type OdsCase = (Option<String>, Result<Vec<&'static str>, &'static str>);

#[test]
fn original_tickers_require_a_tickers_column_of_stock_symbols() {
    let unreadable = "The original spy_holdings.ods ticker list could not be read.";
    let invalid = "The original holdings file must have a Tickers column containing stock symbols.";
    let cases: [OdsCase; 6] = [
        (Some(table_xml(&["Tickers", " MSFT ", "", "AAPL", "MSFT"])), Ok(vec!["MSFT", "AAPL"])),
        (Some(table_xml(&["Tickers", "AAPL", "brk"])), Err(invalid)),
        (Some(table_xml(&["Symbols", "AAPL"])), Err(invalid)),
        (Some(table_xml(&["Tickers"])), Err("The original holdings file contains no tickers.")),
        (Some("<office:document-content>".to_string()), Err(unreadable)),
        (None, Err(unreadable)),
    ];
    let folder = tempfile::tempdir().unwrap();
    for (number, (content, expected)) in cases.into_iter().enumerate() {
        let path = folder.path().join(format!("case{number}.ods"));
        ods(&path, content);
        let result = original_tickers(Some(&path), false);
        match expected {
            Ok(tickers) => assert_eq!(result.unwrap(), tickers, "case {number}"),
            Err(message) => assert_eq!(result.unwrap_err().message(), message, "case {number}"),
        }
    }
    let path = folder.path().join("text.ods");
    std::fs::write(&path, "not a zip archive").unwrap();
    assert_eq!(original_tickers(Some(&path), false).unwrap_err().message(), unreadable);
}
