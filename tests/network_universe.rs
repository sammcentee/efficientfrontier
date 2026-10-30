//! Current universes from Nasdaq and State Street, and member price coverage.
//!
//! Port of `tests/test_universe.py`. A fake network answers in place of the
//! monkeypatched `urlopen` and `yfinance.download`. Tests that monkeypatched
//! `_download_adjusted_close` use [`download_universe_prices_with`].

mod network_support;

use std::collections::HashSet;
use std::io::Write;
use std::time::Duration;

use chrono::{DateTime, Days, Utc};
use efficient_frontier::error::{Error, Result};
use efficient_frontier::frame::{Cell, Date, Frame, Table};
use efficient_frontier::net::{Reply, Request};
use efficient_frontier::pyfmt::{Json, escape};
use efficient_frontier::universe::{
    Member, NASDAQ100_URL, SPY_HOLDINGS_URL, SPY_PRODUCT_URL, UniverseSnapshot, download_universe_prices,
    download_universe_prices_with, fetch_nasdaq100, fetch_sp500_nasdaq100, fetch_spy_holdings,
};
use efficient_frontier::value_error;
use network_support::*;
use serde_json::{Value, json};

fn payload() -> Value {
    let symbols: Vec<String> =
        ["GOOG", "GOOGL", "BRK.B"].into_iter().map(String::from).chain((0..98).map(|i| format!("STOCK{i}"))).collect();
    nasdaq_payload(&symbols, "Oct 1, 2026")
}

fn nasdaq_payload(symbols: &[String], date: &str) -> Value {
    let rows: Vec<Value> =
        symbols.iter().map(|s| json!({"symbol": s, "companyName": format!("{s} Company"), "sector": ""})).collect();
    json!({
        "status": {"rCode": 200},
        "data": {"date": date, "totalrecords": symbols.len(), "data": {"rows": rows}},
    })
}

/// Answer every request with one body.
fn serve(body: impl Into<Vec<u8>>) -> impl FnMut(&Request) -> Reply + 'static {
    let body = body.into();
    move |_| ok(body.clone())
}

fn symbols_of(members: &[Member]) -> Vec<&str> {
    members.iter().map(|member| member.symbol.as_str()).collect()
}

#[test]
fn fetch_uses_official_complete_list_and_preserves_share_classes() {
    let (result, calls) = offline(serve(payload().to_string()), fetch_nasdaq100);
    let result = result.unwrap();
    let requested: Vec<(&str, Duration)> = calls.iter().map(|call| (call.url.as_str(), call.timeout)).collect();
    assert_eq!(requested, [(NASDAQ100_URL, Duration::from_secs(20))]);
    assert_eq!(calls[0].header("User-Agent"), Some("Mozilla/5.0"));
    assert_eq!(calls[0].header("Accept"), Some("application/json"));
    assert_eq!(symbols_of(&result.members)[..3], ["GOOG", "GOOGL", "BRK-B"]);
    assert_eq!(result.members.len(), 101);
    assert_eq!(result.source_date, "2026-10-01");
    assert!(DateTime::parse_from_rfc3339(&result.retrieved_at).is_ok());
    assert!(result.retrieved_at.ends_with("+00:00"));
    assert!(result.members.iter().all(|member| member.industry == "Unknown"));
    let record = &result.source_records[0];
    assert_eq!(record.keys().collect::<Vec<_>>(), ["name", "url", "date", "retrieved_at", "stock_count"]);
    assert_eq!(record.get("stock_count"), Some(&Json::Int(101)));
    assert_eq!(result.name, "Current Nasdaq-100 members");
}

#[test]
fn fetch_rejects_incomplete_or_invalid_membership() {
    let failures = [
        "truncated",
        "too_short",
        "status",
        "duplicate",
        "alias_duplicate",
        "missing_name",
        "invalid_symbol",
        "missing_date",
        "future_date",
        "null_data",
        "wrong_rows",
        "missing_total",
    ];
    for failure in failures {
        let mut malformed = payload();
        let rows = malformed["data"]["data"]["rows"].as_array_mut().unwrap();
        match failure {
            "truncated" => drop(rows.pop()),
            "too_short" => {
                rows.truncate(5);
                malformed["data"]["totalrecords"] = json!(5);
            }
            "status" => malformed["status"]["rCode"] = json!(500),
            "duplicate" => rows[1]["symbol"] = rows[0]["symbol"].clone(),
            "alias_duplicate" => rows[1]["symbol"] = json!("BRK-B"),
            "missing_name" => rows[1]["companyName"] = json!(" "),
            "invalid_symbol" => rows[1]["symbol"] = json!("BAD SYMBOL"),
            "missing_date" => malformed["data"]["date"] = Value::Null,
            "future_date" => {
                malformed["data"]["date"] = json!((Utc::now() + chrono::Duration::days(2)).date_naive().to_string())
            }
            "null_data" => malformed["data"] = Value::Null,
            "wrong_rows" => malformed["data"]["data"]["rows"] = json!({}),
            _ => drop(malformed["data"].as_object_mut().unwrap().remove("totalrecords")),
        }
        let (result, _) = offline(serve(malformed.to_string()), fetch_nasdaq100);
        let error = result.unwrap_err();
        assert!(error.is_value(), "{failure}");
        assert_eq!(
            error.message(),
            "The current Nasdaq-100 list is unavailable or incomplete. Retry later, choose your own stocks, or upload a price CSV.",
            "{failure}"
        );
    }
}

#[test]
fn fetch_failure_has_no_silent_fallback() {
    let failures: [Reply; 4] =
        [Err("offline".into()), Err("timeout".into()), status(503, "Service Unavailable"), ok("not JSON")];
    for failure in failures {
        let (result, _) = offline(move |_| failure.clone(), fetch_nasdaq100);
        assert!(result.unwrap_err().message().contains("choose your own stocks"));
    }
}

#[test]
fn recorded_nasdaq_structure_is_read() {
    let (result, _) = offline(serve(fixture("nasdaq100.json")), fetch_nasdaq100);
    let result = result.unwrap();
    assert_eq!(result.members.len(), 101);
    assert_eq!(result.source_date, "2026-10-02");
    assert_eq!(
        result.members[..2],
        [
            Member { symbol: "BRK-B".into(), name: "BRK.B Example Common Stock".into(), industry: "Technology".into() },
            Member { symbol: "GOOG".into(), name: "GOOG Example Common Stock".into(), industry: "Unknown".into() },
        ]
    );
}

fn snapshot<S: AsRef<str>>(symbols: &[S]) -> UniverseSnapshot {
    let members = symbols
        .iter()
        .map(|symbol| Member {
            symbol: symbol.as_ref().to_string(),
            name: symbol.as_ref().to_string(),
            industry: "Unknown".into(),
        })
        .collect();
    UniverseSnapshot::new(members, NASDAQ100_URL, "2026-10-01", "2026-10-04T12:00:00+00:00")
}

fn dates() -> Vec<Date> {
    business_days("2020-01-01", 10)
}

const START: &str = "2020-01-01";
const END: &str = "2021-01-01";

/// The coverage row of a symbol.
fn row(coverage: &Table, symbol: &str) -> usize {
    coverage.column("symbol").iter().position(|cell| cell.as_str() == Some(symbol)).unwrap()
}

fn coverage_text<'a>(coverage: &'a Table, symbol: &str, column: &str) -> &'a str {
    coverage.column(column)[row(coverage, symbol)].as_str().unwrap()
}

#[test]
fn incomplete_members_do_not_redownload_complete_batch_peers() {
    let symbols: Vec<String> = (0..100).map(|i| format!("STOCK{i}")).collect();
    let excluded = ["STOCK0", "STOCK25", "STOCK50", "STOCK75"];
    let everything: Vec<String> = strings(&["SPY", "QQQ"]).into_iter().chain(symbols.clone()).collect();
    let mut charts = charts(&price_table(&everything, &dates()));
    for date in &dates()[..3] {
        charts.get_mut("STOCK0").unwrap().set(*date, None);
    }
    charts.get_mut("STOCK25").unwrap().set(dates()[4], None);
    charts.remove("STOCK50");
    charts.get_mut("STOCK75").unwrap().set(dates()[4], Some(0.0));
    let (result, calls) =
        offline(yahoo(charts), || download_universe_prices(&snapshot(&symbols), day(START), day(END), None));
    let result = result.unwrap();
    // One request per member; only the 404 of the unknown symbol is repeated, as yfinance does.
    let mut expected = everything.clone();
    expected.insert(expected.iter().position(|symbol| symbol == "STOCK50").unwrap(), "STOCK50".into());
    assert_eq!(chart_symbols(&calls), expected);
    let included: Vec<&String> = symbols.iter().filter(|symbol| !excluded.contains(&symbol.as_str())).collect();
    assert_eq!(result.prices.columns.iter().collect::<Vec<_>>(), included);
    assert_eq!(result.prices.index, result.benchmarks.index);
    let coverage = &result.coverage;
    assert!(excluded.iter().all(|symbol| coverage_text(coverage, symbol, "status") == "excluded"));
    assert_eq!(
        coverage_text(coverage, "STOCK0", "reason"),
        "Price dates differ from market dates: 3 missing, 0 extra."
    );
    assert_eq!(
        coverage_text(coverage, "STOCK25", "reason"),
        "Price dates differ from market dates: 1 missing, 0 extra."
    );
    assert_eq!(coverage_text(coverage, "STOCK50", "reason"), "Yahoo returned no prices for: STOCK50.");
    assert_eq!(coverage_text(coverage, "STOCK75", "reason"), "All prices must be finite and strictly positive.");
    assert_eq!(coverage.column("observations")[row(coverage, "STOCK0")], Cell::Int(7));
    assert_eq!(coverage_text(coverage, "STOCK0", "first_date"), "2020-01-06");
    assert_eq!(result.metadata.int("universe_included"), 96);
    assert_eq!(result.metadata.int("universe_excluded"), 4);
}

#[test]
fn all_missing_member_is_excluded_without_an_individual_request() {
    let mut charts = charts(&price_table(&["SPY", "QQQ", "COMPLETE", "MISSING"], &dates()));
    for date in dates() {
        charts.get_mut("MISSING").unwrap().set(date, None);
    }
    let (result, calls) = offline(yahoo(charts), || {
        download_universe_prices(&snapshot(&["COMPLETE", "MISSING"]), day(START), day(END), None)
    });
    let result = result.unwrap();
    assert_eq!(chart_symbols(&calls), ["SPY", "QQQ", "COMPLETE", "MISSING"]);
    assert_eq!(result.prices.columns, ["COMPLETE"]);
    assert_eq!(coverage_text(&result.coverage, "MISSING", "status"), "excluded");
    assert!(coverage_text(&result.coverage, "MISSING", "reason").contains("no prices"));
}

#[test]
fn whole_batch_failure_retries_individual_requests() {
    let symbols = ["AAA", "BBB", "CCC"];
    for failure in ["network", "format"] {
        let mut yahoo = yahoo(charts(&price_table(&["SPY", "QQQ", "AAA", "BBB", "CCC"], &dates())));
        let mut failed = HashSet::new();
        // The first request of each member fails, so the whole batch fails before the individual retries.
        let respond = move |request: &Request| {
            let symbol = chart_symbol(request).unwrap();
            if symbols.contains(&symbol.as_str()) && failed.insert(symbol) {
                return if failure == "network" {
                    Err("Batch request failed.".into())
                } else {
                    ok("<html>Unexpected</html>")
                };
            }
            yahoo(request)
        };
        let (result, calls) =
            offline(respond, || download_universe_prices(&snapshot(&symbols), day(START), day(END), None));
        let result = result.unwrap();
        assert_eq!(chart_symbols(&calls), ["SPY", "QQQ", "AAA", "BBB", "CCC", "AAA", "BBB", "CCC"], "{failure}");
        assert_eq!(result.prices.columns, symbols);
        assert!(result.coverage.column("status").iter().all(|status| status.as_str() == Some("included")));
    }
}

/// A recording replacement for `_download_adjusted_close` that returns `price_table` frames.
fn table_download(calls: &mut Vec<Vec<String>>) -> impl FnMut(&[String], Date, Date, bool) -> Result<Frame> + '_ {
    move |symbols: &[String], _: Date, _: Date, _: bool| {
        calls.push(symbols.to_vec());
        Ok(price_table(symbols, &dates()))
    }
}

#[test]
fn download_batches_all_securities_in_order_without_ranking() {
    let symbols: Vec<String> = (0..101).map(|i| format!("STOCK{i}")).collect();
    let (mut calls, mut updates) = (Vec::new(), Vec::new());
    let mut progress = |done: usize, total: usize, message: &str| updates.push((done, total, message.to_string()));
    let result = download_universe_prices_with(
        &snapshot(&symbols),
        day(START),
        day(END),
        Some(&mut progress),
        &mut table_download(&mut calls),
    )
    .unwrap();
    assert_eq!(calls[0], ["SPY", "QQQ"]);
    assert_eq!(calls[1..].iter().map(Vec::len).collect::<Vec<_>>(), [25, 25, 25, 25, 1]);
    assert_eq!(result.prices.columns, symbols);
    let listed: Vec<&str> = result.coverage.column("symbol").iter().map(|cell| cell.as_str().unwrap()).collect();
    assert_eq!(listed, symbols);
    assert!(result.coverage.column("status").iter().all(|status| status.as_str() == Some("included")));
    assert_eq!(result.benchmarks.columns, ["SPY", "QQQ"]);
    assert_eq!(result.prices.index, result.benchmarks.index);
    let message = |done: usize, text: &str| (done, 101, text.to_string());
    assert_eq!(
        updates[..4],
        [
            message(0, "Read market dates from SPY and QQQ."),
            message(0, "Download securities 1 to 25 of 101."),
            message(1, "Checked 1 of 101 securities."),
            message(2, "Checked 2 of 101 securities."),
        ]
    );
    assert_eq!(
        updates[updates.len() - 2..],
        [message(100, "Download securities 101 to 101 of 101."), message(101, "Checked 101 of 101 securities.")]
    );
    assert_eq!(result.metadata.int("universe_requested"), 101);
    assert_eq!(result.metadata.int("universe_included"), 101);
    assert_eq!(result.metadata.int("universe_excluded"), 0);
    assert!(result.metadata.str("universe_limitation").contains("survivors"));
    assert_eq!(
        result.metadata.keys().collect::<Vec<_>>(),
        [
            "universe",
            "universe_source_url",
            "universe_source_date",
            "universe_retrieved_at",
            "universe_sources",
            "universe_requested",
            "universe_included",
            "universe_excluded",
            "universe_calendar",
            "universe_start",
            "universe_end",
            "universe_eligibility",
            "universe_limitation",
            "universe_coverage",
        ]
    );
    assert_eq!(result.metadata.str("universe_start"), "2020-01-01");
    assert_eq!(result.metadata.str("universe_end"), "2020-01-14");
    assert_eq!(result.metadata.str("universe_calendar"), "Yahoo adjusted SPY and QQQ dates");
}

#[test]
fn failed_batch_retries_each_member_and_keeps_coverage() {
    let symbols = ["COMPLETE", "SHORT", "GAP", "UNAVAILABLE", "EXTRA"];
    let mut calls = Vec::new();
    let mut download = |requested: &[String], _: Date, _: Date, _: bool| -> Result<Frame> {
        calls.push(requested.to_vec());
        if requested == ["SPY", "QQQ"] {
            return Ok(price_table(requested, &dates()));
        }
        if requested.len() > 1 {
            return Err(value_error!("Incomplete batch."));
        }
        let prices = price_table(requested, &dates());
        Ok(match requested[0].as_str() {
            "UNAVAILABLE" => return Err(value_error!("Yahoo returned no prices for UNAVAILABLE.")),
            "SHORT" => prices.rows(3, 10),
            "GAP" => {
                let kept: Vec<usize> = (0..10).filter(|i| *i != 4).collect();
                Frame::from_rows(
                    kept.iter().map(|i| prices.index[*i]).collect(),
                    prices.columns.clone(),
                    &kept.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
                )
            }
            "EXTRA" => {
                let mut index = prices.index.clone();
                index.push(day("2020-01-18"));
                let mut rows: Vec<Vec<f64>> = (0..10).map(|i| prices.row(i)).collect();
                rows.push(vec![100.0]);
                Frame::from_rows(index, prices.columns.clone(), &rows)
            }
            _ => prices,
        })
    };
    let result = download_universe_prices_with(&snapshot(&symbols), day(START), day(END), None, &mut download).unwrap();
    assert_eq!(calls[2..], symbols.map(|symbol| vec![symbol.to_string()]));
    assert_eq!(result.prices.columns, ["COMPLETE"]);
    assert_eq!(result.prices.nrows(), 10);
    let coverage = &result.coverage;
    assert_eq!(coverage_text(coverage, "SHORT", "reason"), "Price dates differ from market dates: 3 missing, 0 extra.");
    assert_eq!(coverage_text(coverage, "GAP", "reason"), "Price dates differ from market dates: 1 missing, 0 extra.");
    assert_eq!(coverage_text(coverage, "EXTRA", "reason"), "Price dates differ from market dates: 0 missing, 1 extra.");
    assert!(coverage_text(coverage, "UNAVAILABLE", "reason").contains("no prices"));
    // As in pandas, a count column with a missing value holds floats.
    let observations = coverage.column("observations");
    assert_eq!(observations[row(coverage, "SHORT")], Cell::Float(7.0));
    assert!(observations[row(coverage, "UNAVAILABLE")].is_null());
    assert!(coverage.column("first_date")[row(coverage, "UNAVAILABLE")].is_null());
    assert_eq!(result.metadata.int("universe_included"), 1);
    assert_eq!(result.metadata.int("universe_excluded"), 4);
    let records = result.metadata.get("universe_coverage").unwrap().as_list().unwrap();
    assert_eq!(records.len(), 5);
    let unavailable = records[3].as_dict().unwrap();
    assert_eq!(unavailable.get("observations"), Some(&Json::Null));
    assert_eq!(records[1].as_dict().unwrap().get("observations"), Some(&Json::Int(7)));
    // The metadata stays valid strict JSON, without NaN.
    let text = Json::Dict(result.metadata.clone()).dumps(None);
    assert!(!text.contains("NaN"));
    serde_json::from_str::<Value>(&text).unwrap();
}

#[test]
fn unavailable_universe_preserves_exclusion_details() {
    let mut download = |symbols: &[String], _: Date, _: Date, _: bool| -> Result<Frame> {
        if symbols == ["SPY", "QQQ"] {
            Ok(price_table(symbols, &dates()))
        } else {
            Err(value_error!("Yahoo returned no prices."))
        }
    };
    let result =
        download_universe_prices_with(&snapshot(&["AAA", "BBB"]), day(START), day(END), None, &mut download).unwrap();
    assert_eq!((result.prices.nrows(), result.prices.ncols()), (10, 0));
    let statuses: Vec<&str> = result.coverage.column("status").iter().map(|cell| cell.as_str().unwrap()).collect();
    assert_eq!(statuses, ["excluded", "excluded"]);
    assert_eq!(result.metadata.int("universe_included"), 0);
}

#[test]
fn benchmark_failure_stops_before_stock_requests() {
    let mut calls = Vec::new();
    let mut download = |symbols: &[String], _: Date, _: Date, _: bool| -> Result<Frame> {
        calls.push(symbols.to_vec());
        Err(value_error!("Yahoo unavailable."))
    };
    let error = download_universe_prices_with(&snapshot(&["AAA", "BBB"]), day(START), day(END), None, &mut download)
        .unwrap_err();
    assert_eq!(error, Error::Value("Yahoo unavailable.".into()));
    assert_eq!(calls, [["SPY", "QQQ"]]);
}

#[test]
fn download_requires_nonempty_unique_members() {
    for symbols in [vec![], vec!["AAA", "AAA"]] {
        let mut download = |_: &[String], _: Date, _: Date, _: bool| -> Result<Frame> {
            panic!("An invalid universe must not reach Yahoo.")
        };
        let error =
            download_universe_prices_with(&snapshot(&symbols), day(START), day(END), None, &mut download).unwrap_err();
        assert_eq!(error.message(), "The universe must contain unique security symbols.");
    }
}

#[test]
fn downloader_preserves_snapshot_specific_provenance_and_limitation() {
    let original = snapshot(&["AAA", "BBB"]);
    let record = efficient_frontier::dict! { "url" => "https://example.com/holdings", "date" => "2026-10-01" };
    let custom = UniverseSnapshot {
        name: "Combined stocks".into(),
        limitation: "Current holdings proxy".into(),
        source_records: vec![record.clone()],
        ..UniverseSnapshot::new(original.members, "https://example.com/holdings", "2026-10-01", original.retrieved_at)
    };
    let mut calls = Vec::new();
    let result =
        download_universe_prices_with(&custom, day(START), day(END), None, &mut table_download(&mut calls)).unwrap();
    assert_eq!(result.metadata.str("universe"), "Combined stocks");
    assert_eq!(result.metadata.str("universe_source_url"), "https://example.com/holdings");
    assert_eq!(result.metadata.get("universe_sources"), Some(&Json::List(vec![Json::Dict(record)])));
    assert_eq!(result.metadata.str("universe_limitation"), "Current holdings proxy");
    assert_eq!(result.prices.index, result.benchmarks.index);
}

/// One worksheet cell value: text, or a number written as Python's `str(value)`.
#[derive(Clone, Debug)]
enum Item {
    Text(String),
    Number(String),
}

fn text(value: &str) -> Item {
    Item::Text(value.into())
}

fn number(value: f64) -> Item {
    Item::Number(efficient_frontier::pyfmt::repr(value))
}

fn integer(value: i64) -> Item {
    Item::Number(value.to_string())
}

fn spy_rows(stock_count: usize) -> Vec<Vec<Item>> {
    let symbols: Vec<String> = ["GOOG", "GOOGL", "BRK.B", "BF.B"]
        .into_iter()
        .map(String::from)
        .chain((0..stock_count - 4).map(|i| format!("STOCK{i}")))
        .collect();
    let mut rows = vec![
        vec![text("Fund Name:"), text("State Street® SPDR® S&P 500® ETF Trust")],
        vec![text("Ticker Symbol:"), text("SPY")],
        vec![text("Holdings:"), text("As of 01-Oct-2026")],
        vec![],
        ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"]
            .map(text)
            .to_vec(),
    ];
    for (i, symbol) in symbols.iter().enumerate() {
        rows.push(vec![
            text(&format!("{symbol} Company")),
            text(symbol),
            text(&format!("{i:09}")),
            text(&format!("{i:07}")),
            number(99.5 / stock_count as f64),
            text("-"),
            integer(1000),
            text("USD"),
        ]);
    }
    rows.push(vec![
        text("US DOLLAR"),
        text("-"),
        text("999USDZ92"),
        text("-"),
        number(0.5),
        text("-"),
        integer(5000),
        text("USD"),
    ]);
    rows.push(vec![
        text("TPG INC"),
        text("2602335D"),
        text("436CVR021"),
        text("-"),
        number(0.000003),
        text("-"),
        integer(500),
        text("USD"),
    ]);
    rows.push(vec![]);
    rows.push(vec![text("Disclosure text, not a holding.")]);
    rows
}

/// A single-sheet workbook with inline or shared strings, as `spy_workbook` of the Python tests writes it.
fn spy_workbook(rows: &[Vec<Item>], shared: bool) -> Vec<u8> {
    const NAMESPACE: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
    let mut sheet = format!("<worksheet xmlns=\"{NAMESPACE}\"><sheetData>");
    let mut strings = Vec::new();
    for (i, values) in rows.iter().enumerate() {
        let number = i + 1;
        sheet.push_str(&format!("<row r=\"{number}\">"));
        for (column, value) in "ABCDEFGH".chars().zip(values) {
            let reference = format!("{column}{number}");
            match value {
                Item::Number(value) => sheet.push_str(&format!("<c r=\"{reference}\"><v>{value}</v></c>")),
                Item::Text(value) if shared => {
                    sheet.push_str(&format!("<c r=\"{reference}\" t=\"s\"><v>{}</v></c>", strings.len()));
                    strings.push(format!("<si><t>{}</t></si>", escape(value)));
                }
                Item::Text(value) => sheet
                    .push_str(&format!("<c r=\"{reference}\" t=\"inlineStr\"><is><t>{}</t></is></c>", escape(value))),
            }
        }
        sheet.push_str("</row>");
    }
    sheet.push_str("</sheetData></worksheet>");
    let mut output = std::io::Cursor::new(Vec::new());
    let mut archive = zip::ZipWriter::new(&mut output);
    let options = zip::write::SimpleFileOptions::default();
    archive.start_file("xl/worksheets/sheet1.xml", options).unwrap();
    archive.write_all(sheet.as_bytes()).unwrap();
    if shared {
        archive.start_file("xl/sharedStrings.xml", options).unwrap();
        archive.write_all(format!("<sst xmlns=\"{NAMESPACE}\">{}</sst>", strings.concat()).as_bytes()).unwrap();
    }
    archive.finish().unwrap();
    output.into_inner()
}

fn spy_page(count: usize, date: &str) -> String {
    format!(
        "<section><h2>Fund Characteristics <span>as of {date}</span></h2>\
         <table><tr><th>Number of Holdings</th><td>{count}</td></tr></table></section>\
         <section><h2>Index Characteristics</h2><th>Number of Holdings</th><td>999</td></section>"
    )
}

/// Serve the SPY workbook and the product page.
fn mock_spy(rows: &[Vec<Item>], page: String, shared: bool) -> impl FnMut(&Request) -> Reply + 'static {
    let workbook = spy_workbook(rows, shared);
    move |request| match request.url.as_str() {
        SPY_HOLDINGS_URL => ok(workbook.clone()),
        SPY_PRODUCT_URL => ok(page.clone()),
        other => panic!("unexpected request {other}"),
    }
}

#[test]
fn spy_uses_primary_dated_full_holdings_and_preserves_share_classes() {
    for shared in [false, true] {
        let (result, calls) =
            offline(mock_spy(&spy_rows(500), spy_page(501, "Oct 1 2026"), shared), fetch_spy_holdings);
        let result = result.unwrap();
        let requested: Vec<(&str, Duration)> = calls.iter().map(|call| (call.url.as_str(), call.timeout)).collect();
        assert_eq!(
            requested,
            [(SPY_HOLDINGS_URL, Duration::from_secs(20)), (SPY_PRODUCT_URL, Duration::from_secs(20))]
        );
        assert!(calls.iter().all(|call| call.header("User-Agent") == Some("Mozilla/5.0")));
        assert_eq!(result.members.len(), 500);
        assert_eq!(symbols_of(&result.members)[..4], ["GOOG", "GOOGL", "BRK-B", "BF-B"]);
        assert!(result.members.iter().all(|member| member.industry == "Unknown"));
        assert_eq!(result.source_date, "2026-10-01");
        assert!(result.name.contains("proxy") && result.limitation.contains("not a certified"));
        let source = &result.source_records[0];
        assert_eq!(source.get("reported_fund_holdings"), Some(&Json::Int(501)));
        assert_eq!(source.get("stock_count"), Some(&Json::Int(500)));
        let reasons: Vec<String> = source
            .get("excluded_holdings")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|holding| holding.as_dict().unwrap().str("reason").to_string())
            .collect();
        assert_eq!(reasons, ["Cash holding", "Non-stock right or warrant"]);
        assert!(!Json::Dict(source.clone()).dumps(None).contains("NaN"));
    }
}

#[test]
fn spy_excludes_funds_and_derivatives_from_stock_candidates() {
    for (name, symbol) in
        [("SPDR S&P 500 ETF TRUST", "SPY"), ("S&P 500 FUTURE", "ESZ6"), ("Treasury MONEY MARKET", "CASHXX")]
    {
        let mut rows = spy_rows(500);
        let position = rows.len() - 2;
        rows.insert(
            position,
            vec![
                text(name),
                text(symbol),
                text("123456789"),
                text("1234567"),
                number(0.001),
                text("-"),
                integer(1000),
                text("USD"),
            ],
        );
        let (result, _) = offline(mock_spy(&rows, spy_page(502, "Oct 1 2026"), false), fetch_spy_holdings);
        let result = result.unwrap();
        assert_eq!(result.members.len(), 500);
        assert!(!symbols_of(&result.members).contains(&symbol));
        let excluded = result.source_records[0].get("excluded_holdings").unwrap().as_list().unwrap();
        assert_eq!(excluded.last().unwrap().as_dict().unwrap().str("reason"), "Fund or derivative holding");
    }
}

#[test]
fn spy_rejects_wrong_fund_invalid_rows_and_incomplete_snapshots() {
    let failures = [
        "fund_ticker",
        "fund_name",
        "missing_date",
        "future_date",
        "header",
        "currency",
        "symbol",
        "identifier",
        "sedol",
        "weight_nan",
        "weight_zero",
        "shares_nan",
        "duplicate",
        "alias_duplicate",
        "truncated",
        "count",
        "count_date",
    ];
    // The unchanged snapshot is valid, so each failure below comes from its one change.
    let (baseline, _) = offline(mock_spy(&spy_rows(501), spy_page(502, "Oct 1 2026"), false), fetch_spy_holdings);
    assert_eq!(baseline.unwrap().members.len(), 501);
    for failure in failures {
        let mut rows = spy_rows(501);
        let (mut count, mut date) = (502, "Oct 1 2026");
        match failure {
            "fund_ticker" => rows[1][1] = text("QQQ"),
            "fund_name" => rows[0][1] = text("Some other fund"),
            "missing_date" => rows[2][1] = text(""),
            "future_date" => rows[2][1] = text("As of 01-Oct-2080"),
            "header" => rows[4][0] = text("Unexpected header"),
            "currency" => rows[5][7] = text("EUR"),
            "symbol" => rows[5][1] = text("INVALID SYMBOL"),
            "identifier" => rows[5][2] = text(""),
            "sedol" => rows[5][3] = text("-"),
            "weight_nan" => rows[5][4] = number(f64::NAN),
            "weight_zero" => rows[5][4] = integer(0),
            "shares_nan" => rows[5][6] = number(f64::NAN),
            "duplicate" => rows[6][1] = rows[5][1].clone(),
            "alias_duplicate" => rows[5][1] = text("BRK-B"),
            // Still 500 stocks: the separate published count detects this truncation.
            "truncated" => drop(rows.remove(10)),
            "count" => count += 1,
            _ => date = "Sep 30 2026",
        }
        let (result, _) = offline(mock_spy(&rows, spy_page(count, date), false), fetch_spy_holdings);
        assert_eq!(
            result.unwrap_err().message(),
            "The current SPY stock holdings are unavailable or incomplete. \
             Retry later, choose Nasdaq-100 or your own stocks, or upload a price CSV.",
            "{failure}"
        );
    }
}

#[test]
fn spy_rejects_bad_download_without_fallback() {
    for response in ["not a workbook", "<html>Access denied</html>"] {
        let (result, _) = offline(serve(response), fetch_spy_holdings);
        assert!(result.unwrap_err().message().contains("choose Nasdaq-100 or your own stocks"));
    }
}

#[test]
fn recorded_spy_page_structure_is_read() {
    let page = String::from_utf8(fixture("spy_product_page.html")).unwrap();
    let (result, _) = offline(mock_spy(&spy_rows(500), page, true), fetch_spy_holdings);
    let result = result.unwrap();
    assert_eq!(result.source_records[0].get("reported_fund_holdings"), Some(&Json::Int(501)));
    assert_eq!(result.source_date, "2026-10-01");
}

/// Serve the Nasdaq list and the SPY holdings.
fn mock_sources(nasdaq: Reply, rows: &[Vec<Item>]) -> impl FnMut(&Request) -> Reply + 'static {
    let mut spy = mock_spy(rows, spy_page(501, "Oct 1 2026"), false);
    move |request| if request.url == NASDAQ100_URL { nasdaq.clone() } else { spy(request) }
}

#[test]
fn union_combines_sources_without_dropping_share_classes_or_hiding_dates() {
    let symbols: Vec<String> = ["GOOG", "GOOGL", "BRK.B", "TECHONLY"]
        .into_iter()
        .map(String::from)
        .chain((0..97).map(|i| format!("STOCK{i}")))
        .collect();
    let nasdaq = ok(nasdaq_payload(&symbols, "Oct 2, 2026").to_string());
    let (result, calls) = offline(mock_sources(nasdaq, &spy_rows(500)), fetch_sp500_nasdaq100);
    let result = result.unwrap();
    assert_eq!(
        calls.iter().map(|call| call.url.as_str()).collect::<Vec<_>>(),
        [NASDAQ100_URL, SPY_HOLDINGS_URL, SPY_PRODUCT_URL]
    );
    assert_eq!(result.members.len(), 501);
    let listed = symbols_of(&result.members);
    assert_eq!(listed.iter().collect::<HashSet<_>>().len(), listed.len());
    assert_eq!(listed[..5], ["GOOG", "GOOGL", "BRK-B", "TECHONLY", "STOCK0"]);
    // SPY members follow the Nasdaq members, without the overlap.
    assert_eq!(listed[101], "BF-B");
    assert_eq!(result.source_date, "2026-10-01");
    let dates: Vec<&str> = result.source_records.iter().map(|record| record.str("date")).collect();
    assert_eq!(dates, ["2026-10-02", "2026-10-01"]);
    assert!(result.name.contains("S&P 500") && result.name.contains("Nasdaq-100"));
    assert_eq!(result.source_url, SPY_HOLDINGS_URL);
}

#[test]
fn union_requires_both_sources_without_partial_fallback() {
    let nasdaq = || ok(payload().to_string());
    let (result, _) = offline(mock_sources(status(500, "error"), &spy_rows(500)), fetch_sp500_nasdaq100);
    assert!(result.unwrap_err().message().starts_with("The current Nasdaq-100 list is unavailable"));
    let mut broken = spy_rows(500);
    broken[1][1] = text("QQQ");
    let (result, _) = offline(mock_sources(nasdaq(), &broken), fetch_sp500_nasdaq100);
    assert!(result.unwrap_err().message().starts_with("The current SPY stock holdings are unavailable"));
}

#[test]
fn every_member_date_gap_is_counted_once() {
    // Dates on both sides of the market calendar count as missing or extra, never as both.
    let mut download = |symbols: &[String], _: Date, _: Date, _: bool| -> Result<Frame> {
        if symbols.len() == 2 {
            return Ok(price_table(symbols, &dates()));
        }
        let shifted: Vec<Date> = dates().iter().map(|date| *date + Days::new(1)).collect();
        Ok(price_table(symbols, &shifted))
    };
    let result = download_universe_prices_with(&snapshot(&["AAA"]), day(START), day(END), None, &mut download).unwrap();
    let reason = coverage_text(&result.coverage, "AAA", "reason").to_string();
    let shifted: HashSet<Date> = dates().iter().map(|date| *date + Days::new(1)).collect();
    let missing = dates().iter().filter(|date| !shifted.contains(date)).count();
    assert_eq!(reason, format!("Price dates differ from market dates: {missing} missing, {missing} extra."));
}
