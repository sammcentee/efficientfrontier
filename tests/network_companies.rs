//! Current company names and business sentences from Yahoo Finance.
//!
//! Port of `tests/test_companies.py`. A fake network answers the Yahoo
//! session requests and the quoteSummary request in place of the
//! monkeypatched `YfData.get_raw_json`.

mod network_support;

use std::time::Duration;

use efficient_frontier::companies::fetch_company_profile;
use efficient_frontier::net::{Reply, Request};
use efficient_frontier::story::Company;
use network_support::*;
use serde_json::{Value, json};

const COOKIE_URL: &str = "https://fc.yahoo.com";
const CRUMB_URL: &str = "https://query1.finance.yahoo.com/v1/test/getcrumb";

/// A Yahoo session: the cookie page, then crumbs in order, then `respond` for quoteSummary requests.
fn yahoo_session(
    crumbs: &[&str],
    mut respond: impl FnMut(&Request) -> Reply + 'static,
) -> impl FnMut(&Request) -> Reply {
    let mut crumbs: Vec<String> = crumbs.iter().rev().map(|crumb| crumb.to_string()).collect();
    move |request| match request.url.as_str() {
        COOKIE_URL => status(404, "<html>Not found</html>"),
        CRUMB_URL => ok(crumbs.pop().expect("a crumb")),
        _ => respond(request),
    }
}

fn summary(price: Value, profile: Value) -> Reply {
    ok(json!({ "quoteSummary": { "result": [{ "price": price, "assetProfile": profile }] } }).to_string())
}

fn profile_of(price: Value, profile: Value, symbol: &str) -> Option<Company> {
    let response = summary(price, profile);
    offline(yahoo_session(&["crumb-1"], move |_| response.clone()), || fetch_company_profile(symbol)).0
}

#[test]
fn profile_uses_bounded_narrow_lookup_and_preserves_share_class() {
    let response = summary(
        json!({ "longName": "Berkshire Hathaway Inc.", "shortName": "Berkshire" }),
        json!({ "longBusinessSummary":
            "Berkshire Hathaway Inc. operates insurance and other businesses worldwide. It has several subsidiaries." }),
    );
    let (profile, calls) =
        offline(yahoo_session(&["crumb-1"], move |_| response.clone()), || fetch_company_profile("brk.b"));
    assert_eq!(
        profile,
        Some(Company {
            name: "Berkshire Hathaway Inc.".into(),
            description: "Berkshire Hathaway Inc. operates insurance and other businesses worldwide.".into(),
            source_url: "https://finance.yahoo.com/quote/BRK-B/profile/".into(),
        })
    );
    let urls: Vec<&str> = calls.iter().map(|call| call.url.as_str()).collect();
    assert_eq!(urls, [COOKIE_URL, CRUMB_URL, "https://query2.finance.yahoo.com/v10/finance/quoteSummary/BRK-B"]);
    let query: Vec<(&str, &str)> = calls[2].query.iter().map(|(name, value)| (name.as_str(), value.as_str())).collect();
    assert_eq!(
        query,
        [("modules", "price,assetProfile"), ("formatted", "false"), ("symbol", "BRK-B"), ("crumb", "crumb-1")]
    );
    assert_eq!(calls[2].timeout, Duration::from_secs(8));
}

#[test]
fn description_keeps_one_clean_business_sentence() {
    let cases = [
        ("Acme Corp. Designs products worldwide. It also sells services.", "Acme Corp. Designs products worldwide."),
        ("Acme operates in the U.S. and Canada. Its products include tools.", "Acme operates in the U.S. and Canada."),
        ("Acme makes tools! It sells them worldwide.", "Acme makes tools!"),
        ("Acme makes tools", "Acme makes tools"),
        ("  Acme makes\n tools &amp; parts.   It sells them worldwide. ", "Acme makes tools & parts."),
        // Abbreviations need a word boundary, and only an upper-case letter starts a sentence.
        ("Acme operates in 3 U.S. States. It sells tools.", "Acme operates in 3 U.S. States."),
        ("AcmeCo. Makes tools. Really.", "AcmeCo."),
        ("Acme sells tools. inc. It grows.", "Acme sells tools. inc."),
        ("Caf&eacute; &#x2014; tools&#39; maker&ampsome. It grows.", "Café — tools' maker&some."),
    ];
    for (summary, expected) in cases {
        let profile = profile_of(Value::Null, json!({ "longBusinessSummary": summary }), "ACME").unwrap();
        assert_eq!(profile.description, expected, "{summary:?}");
    }
}

#[test]
fn missing_description_is_not_invented_from_sector() {
    let profile = profile_of(json!({ "shortName": "Acme" }), json!({ "sector": "Technology" }), "ACME");
    assert_eq!(
        profile,
        Some(Company {
            name: "Acme".into(),
            description: String::new(),
            source_url: "https://finance.yahoo.com/quote/ACME/profile/".into(),
        })
    );
}

#[test]
fn missing_or_invalid_profile_is_unavailable() {
    let responses = [
        json!({}),
        json!({ "quoteSummary": { "result": null } }),
        json!({ "quoteSummary": { "result": [] } }),
        json!({ "quoteSummary": { "result": [{}] } }),
        json!({ "quoteSummary": { "result": [{ "price": { "longName": 3 }, "assetProfile": [] }] } }),
        json!({ "quoteSummary": { "result": [{ "price": ["Acme"], "assetProfile": {} }] } }),
    ];
    for response in responses {
        let body = response.to_string();
        let (profile, _) =
            offline(yahoo_session(&["crumb-1"], move |_| ok(body.clone())), || fetch_company_profile("ACME"));
        assert_eq!(profile, None, "{response}");
    }
}

#[test]
fn network_failure_is_optional_metadata_only() {
    let (profile, _) = offline(|_| Err("Provider did not respond".into()), || fetch_company_profile("ACME"));
    assert_eq!(profile, None);
    let (profile, _) =
        offline(yahoo_session(&["crumb-1", "crumb-2"], |_| status(500, "error")), || fetch_company_profile("ACME"));
    assert_eq!(profile, None);
}

#[test]
fn missing_or_multiple_symbols_do_not_fetch() {
    for symbol in ["", "MSFT NVDA"] {
        let (profile, calls) =
            offline(|_| panic!("No request without exactly one symbol."), || fetch_company_profile(symbol));
        assert_eq!(profile, None);
        assert!(calls.is_empty());
    }
}

#[test]
fn a_refused_crumb_is_renewed_once_and_a_good_crumb_is_reused() {
    let mut answers = vec![summary(json!({ "longName": "Acme Inc." }), Value::Null); 2];
    answers.push(status(
        401,
        r#"{"finance":{"result":null,"error":{"code":"Unauthorized","description":"Invalid Crumb"}}}"#,
    ));
    let respond = move |_: &Request| answers.pop().unwrap();
    let (profiles, calls) = offline(yahoo_session(&["old", "new"], respond), || {
        [fetch_company_profile("ACME"), fetch_company_profile("ACME")]
    });
    assert!(profiles.iter().all(|profile| profile.as_ref().is_some_and(|profile| profile.name == "Acme Inc.")));
    let requests: Vec<(&str, Option<&str>)> =
        calls.iter().map(|call| (call.url.as_str(), call.param("crumb"))).collect();
    let quote = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/ACME";
    assert_eq!(
        requests,
        [
            (COOKIE_URL, None),
            (CRUMB_URL, None),
            (quote, Some("old")),
            (COOKIE_URL, None),
            (CRUMB_URL, None),
            (quote, Some("new")),
            (quote, Some("new")),
        ]
    );
}

#[test]
fn symbols_are_escaped_in_the_request_and_source_urls() {
    let response = summary(json!({ "longName": "S&P 500 Index" }), Value::Null);
    let (profile, calls) =
        offline(yahoo_session(&["crumb-1"], move |_| response.clone()), || fetch_company_profile("^gspc"));
    assert_eq!(profile.unwrap().source_url, "https://finance.yahoo.com/quote/%5EGSPC/profile/");
    assert_eq!(calls[2].url, "https://query2.finance.yahoo.com/v10/finance/quoteSummary/%5EGSPC");
    assert_eq!(calls[2].param("symbol"), Some("^GSPC"));
}

#[test]
fn recorded_quote_summary_structure_is_read() {
    let body = fixture("quote_summary.json");
    let (profile, _) =
        offline(yahoo_session(&["crumb-1"], move |_| ok(body.clone())), || fetch_company_profile("EXMPL"));
    assert_eq!(
        profile,
        Some(Company {
            name: "Example Holdings Inc.".into(),
            description: "Example Holdings Inc. designs example tools & parts for customers in the U.S. and Canada."
                .into(),
            source_url: "https://finance.yahoo.com/quote/EXMPL/profile/".into(),
        })
    );
}
