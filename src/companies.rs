//! Current company identities for display, separate from portfolio price data.
//!
//! Port of `efficient_frontier/companies.py`.

use std::time::Duration;

use serde_json::{Map, Value};

use crate::data::parse_tickers;
use crate::net::{self, quote};
use crate::story::Company;

const QUOTE_SUMMARY_URL: &str = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/";
/// A short limit, because a profile is optional metadata.
const PROFILE_TIMEOUT: Duration = Duration::from_secs(8);

fn text(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).map(|value| net::single_spaces(&net::unescape(value))).unwrap_or_default()
}

/// Keep the provider's first sentence without cutting company abbreviations.
fn business_sentence(summary: &str) -> &str {
    let characters: Vec<(usize, char)> = summary.char_indices().collect();
    let mut i = 1;
    while i < characters.len() {
        // A sentence ends at `.`, `!` or `?` before whitespace and an upper-case letter.
        let start = i;
        while i < characters.len() && net::is_python_space(characters[i].1) {
            i += 1;
        }
        let ends = i > start
            && matches!(characters[start - 1].1, '.' | '!' | '?')
            && characters.get(i).is_some_and(|(_, next)| next.is_ascii_uppercase());
        if ends {
            let sentence = &summary[..characters[start].0];
            if !ends_with_abbreviation(sentence) {
                return sentence;
            }
        }
        i = i.max(start + 1);
    }
    summary
}

/// `\b(?:Inc|Corp|Ltd|Co|S\.A|N\.V|U\.S|U\.K)\.$`
fn ends_with_abbreviation(sentence: &str) -> bool {
    const ABBREVIATIONS: [&str; 8] = ["Inc.", "Corp.", "Ltd.", "Co.", "S.A.", "N.V.", "U.S.", "U.K."];
    ABBREVIATIONS.iter().any(|abbreviation| {
        sentence.strip_suffix(abbreviation).is_some_and(|before| {
            before.chars().next_back().is_none_or(|previous| !(previous.is_alphanumeric() || previous == '_'))
        })
    })
}

/// A JSON value as a Python mapping in `result.get(key) or {}`: missing and
/// false values are empty, and any other value that is not an object fails.
fn mapping(value: Option<&Value>) -> Option<Option<&Map<String, Value>>> {
    match value {
        Some(value) if net::truthy(value) => value.as_object().map(Some),
        _ => Some(None),
    }
}

/// The name and business sentence of a quoteSummary response, or `None` for an invalid response.
fn profile_text(response: &Value) -> Option<(String, String)> {
    let result = response.get("quoteSummary")?.get("result")?.get(0)?.as_object()?;
    let price = mapping(result.get("price"))?;
    let profile = mapping(result.get("assetProfile"))?;
    let field = |map: Option<&Map<String, Value>>, key: &str| text(map.and_then(|map| map.get(key)));
    let mut name = field(price, "longName");
    if name.is_empty() {
        name = field(price, "shortName");
    }
    let description = business_sentence(&field(profile, "longBusinessSummary")).to_string();
    Some((name, description))
}

/// Return the current Yahoo name, one business sentence and the source URL, or `None`.
///
/// Missing fields stay empty. A failed profile never changes the portfolio.
/// Cache this optional metadata at the UI boundary. The request carries the
/// Yahoo session cookie and crumb and an explicit 8-second timeout, and asks
/// only for the price and profile modules.
pub fn fetch_company_profile(symbol: &str) -> Option<Company> {
    let symbols = parse_tickers(symbol);
    let [ticker] = &symbols[..] else { return None };
    let escaped = quote(ticker);
    let response = net::yahoo_json(
        &format!("{QUOTE_SUMMARY_URL}{escaped}"),
        &[("modules", "price,assetProfile"), ("formatted", "false"), ("symbol", ticker)],
        PROFILE_TIMEOUT,
    )
    .ok()?;
    let (name, description) = profile_text(&response)?;
    if name.is_empty() && description.is_empty() {
        return None;
    }
    Some(Company { name, description, source_url: format!("https://finance.yahoo.com/quote/{escaped}/profile/") })
}
