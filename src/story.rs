//! Plain-language answers, HTML fragments and story chart helpers shared by the app and the report.
//!
//! Port of `efficient_frontier/story.py` in the Python release (commit 6c6df6e).
//! The web app calls the public functions below; keep these signatures.
//!
//! Every function is pure: it reads engine results and returns text, HTML or a
//! chart change. HTML fragments use the `.pl-*` classes in the story CSS and
//! escape every data string.

use chrono::{Datelike, Months};
use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::backtest::BacktestStudy;
use crate::backtest_report::{ChartOptions, backtest_chart};
use crate::benchmark_report::{evidence_chart, np_mean, np_sum};
use crate::benchmarks::BenchmarkComparison;
use crate::error::Result;
use crate::evidence::EvidenceAnalysis;
use crate::frame::{Cell, Date, Frame, Series, Table, iso};
use crate::metrics::path_returns;
use crate::plot::Figure;
use crate::profiles::ProfileAnalysis;
use crate::pyfmt::{self, Dict, Json, escape};
use crate::style::{ACCENT, CANVAS, ChartStyle, INK_2, INK_3, display_label, style_chart};

/// Display names of the engine risk levels.
pub const LEVEL_LABELS: &[(&str, &str)] = &[("Low", "Low"), ("Medium", "Medium"), ("Extreme", "Highest")];
pub const VIEW_LABELS: [(&str, &str); 2] = [("Portfolio", "Overview"), ("Research", "Research")];
/// Plain names of the backtest methods.
pub const RULE_LABELS: &[(&str, &str)] = &[
    ("Expanding window", "Refit on all past prices"),
    ("Rolling window", "Refit on recent prices"),
    ("Fixed rebalance", "Keep the first mix"),
    ("Buy and hold", "Buy once, then hold"),
];
/// Plain names of the overview chart modes.
pub const CHART_LABELS: &[(&str, &str)] =
    &[("Growth", "Growth"), ("Drawdown", "Falls from peak"), ("Relative to markets", "Versus markets")];
/// Short names of the research topics.
pub const TOPIC_LABELS: &[(&str, &str)] = &[
    ("Efficient frontiers", "Frontier"),
    ("Risk breakdown", "Risk"),
    ("Data & coverage", "Data"),
    ("All backtests", "All tests"),
    ("Methodology", "Method"),
];
/// Plain labels of study settings.
pub const SETTING_LABELS: &[(&str, &str)] = &[
    ("source", "Source"),
    ("universe", "Stocks"),
    ("price_currency", "Price currency"),
    ("compare_market", "Market comparison"),
    ("train_fraction", "Prices for the first fit"),
    ("risk_free_rate", "Risk-free rate"),
    ("max_weight", "Largest holding allowed"),
    ("target_volatility", "Annual volatility budget"),
    ("shrinkage", "Covariance shrinkage"),
    ("periods_per_year", "Observations per year"),
    ("currency_assumption", "Currency assumption"),
    ("requested_start", "From"),
    ("requested_end_exclusive", "Until (not included)"),
    ("universe_source_url", "Index list source"),
    ("universe_source_date", "Index list date"),
    ("universe_retrieved_at", "Index list retrieved"),
    ("universe_requested", "Securities checked"),
    ("universe_included", "Securities used"),
    ("universe_excluded", "Securities left out"),
    ("universe_limitation", "Survivorship note"),
    ("benchmark_source", "Market price source"),
    ("benchmarks_note", "Market note"),
    ("evidence_note", "Evidence note"),
    ("input_file", "Price file"),
    ("universe_calendar", "Market calendar"),
    ("universe_start", "Index prices from"),
    ("universe_end", "Index prices to"),
    ("universe_eligibility", "Eligibility"),
    ("training_start", "First fit from"),
    ("training_end", "First fit to"),
    ("training_observations", "Returns in the first fit"),
    ("holdout_start", "Holdout from"),
    ("holdout_end", "Holdout to"),
    ("holdout_observations", "Returns in the holdout"),
    ("assets", "Assets"),
    ("holdout_strategy", "Holdout rule"),
    ("distributions", "Distributions"),
    ("estimation", "Estimates"),
    ("exclusions", "Not included"),
    ("rebalance_every", "Trade every (price rows)"),
    ("rolling_window", "Recent history for refits (return observations)"),
    ("cost_bps", "Trading cost"),
    ("split", "Last price row of the first fit"),
    ("include_profiles", "Risk levels tested"),
    ("trading_days_per_year", "Trading days per year"),
    ("cash_interest_rate", "Interest on cash"),
    ("train_start", "First fit from"),
    ("train_end", "First fit to"),
    ("test_start", "Test from"),
    ("test_end", "Test to"),
    ("initial_execution", "First trade"),
    ("execution", "Trade timing"),
    ("cost_convention", "Cost rule"),
    ("return_objective", "Return objective"),
    ("selection_method", "Portfolio selection"),
];
/// Plain labels of the evidence statistics.
pub const STAT_LABELS: &[(&str, &str)] = &[
    ("CAGR difference", "Growth gap"),
    ("Annual mean advantage", "Average gap"),
    ("Mean CI lower", "95% range, low"),
    ("Mean CI upper", "95% range, high"),
    ("Mean p-value", "p (one test)"),
    ("Mean Holm p-value", "Adjusted p (Holm)"),
    ("Benchmark beta", "Beta"),
    ("Annual alpha", "Alpha"),
    ("Windows ahead", "Stretches ahead"),
    ("Inference observations", "Paired observations"),
    ("Historical evidence", "Result"),
];
/// Definitions of the words in the glossary.
pub const WORDS: &[(&str, &str)] = &[
    (
        "95% range",
        "A 95% confidence interval for the annual arithmetic mean return gap, not the compound growth (CAGR) gap. \
         A range that crosses zero includes no difference. It allows for returns that cluster in time (Newey–West).",
    ),
    (
        "Average gap",
        "The average difference per return observation between your rule and the market, scaled to a year. \
         It differs from the growth gap, which compounds the whole path.",
    ),
    (
        "Points",
        "Percentage points. If your rule grew 9.4% and the market grew 28.6%, the gap is 19.2 points, not 19.2%.",
    ),
];
pub const METHOD_MD: &str = "\n\
### What the model does\n\
It chooses weights for all eligible assets. Weights are zero or more and add up to 100%.\n\
Today's holdings use all of the selected history to estimate annual arithmetic mean returns and covariance.\n\
The frontier finds the least volatile mix for each target mean return. These fitted means are not compound growth (CAGR) or forecasts.\n\
\n\
### What the risk levels mean\n\
**Low** is the mix with the least modelled volatility. **Medium** is the sampled frontier point nearest to the midpoint\n\
between Low and Highest volatility. **Highest** maximises the fitted annual arithmetic mean return, with the least variance among equal solutions.\n\
The levels are relative to the selected assets. They are not absolute limits or guarantees. Exports call Highest \"Extreme\".\n\
Volatility measures how much past returns moved. It does not measure every form of risk.\n\
\n\
### Covariance shrinkage\n\
Covariance describes how returns move together. The estimates can be noisy.\n\
At 0%, the model uses the sample covariance. At 100%, it ignores correlations and keeps each asset's own variance.\n\
The default 10% reduces the cross-asset terms by 10%. This is a manual setting, not Ledoit–Wolf.\n\
\n\
### The market test\n\
**Buy once, then hold** (Buy and hold) invests once. **Keep the first mix** (Fixed rebalance) trades back to the first targets.\n\
**Refit on all past prices** (Expanding window) refits with all earlier returns. **Refit on recent prices** (Rolling window) uses a fixed recent window.\n\
Each fit selects the same mean-based risk levels from earlier returns. Each trade uses information to the previous close\n\
and executes at the next close. The first interval stays in cash. The later test reports realised compound growth (CAGR).\n\
Fees apply to each amount bought or sold. There is no final sale. Taxes and currency conversion are not included.\n\
\n\
### Reading the evidence\n\
SPY and QQQ are ETF stand-ins, not raw index series. They use the same dates and fee rate as the rule.\n\
The realised CAGR comparison shows which rule grew more in this sample. The confidence ranges and adjusted tests concern\n\
annual arithmetic mean return gaps, not CAGR gaps. An unclear mean gap does not establish equal performance or rule out future outperformance.\n\
The frontier does not promise to beat either market.\n\
The Holm adjustment covers every test in this report. It cannot correct earlier searches that you did before.\n\
Today's members, the full-history requirement and repeated changes after review can all distort a backtest.\n\
Save the rules and the report. Then judge them on later data that you have not seen.\n";
/// Four placeholder rows while a result loads.
pub const SKELETON_HTML: &str = concat!(
    "<div class=\"pl-skel-rows\" aria-hidden=\"true\">",
    "<div class=\"pl-skel-row\"><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span></div>",
    "<div class=\"pl-skel-row\"><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span></div>",
    "<div class=\"pl-skel-row\"><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span></div>",
    "<div class=\"pl-skel-row\"><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span><span class=\"pl-skel\"></span></div>",
    "</div>",
);

/// The adjective of each engine risk level.
pub const LEVEL: &[(&str, &str)] = &[("Low", "low"), ("Medium", "medium"), ("Extreme", "highest")];
/// Each market as a plural phrase and a singular noun.
pub const NOUN: &[(&str, (&str, &str))] = &[
    ("S&P 500 + Nasdaq-100", ("S&P 500 and Nasdaq-100 stocks", "stock")),
    ("Nasdaq-100", ("Nasdaq-100 stocks", "stock")),
    ("Original 60 holdings", ("stocks from your original 60", "stock")),
    ("My tickers", ("of your tickers", "stock")),
    ("Upload CSV", ("assets from your file", "asset")),
    ("Demo", ("simulated assets", "asset")),
];
/// Where each market's prices come from.
pub const PLACE: &[(&str, &str)] = &[
    ("S&P 500 + Nasdaq-100", "the S&P 500 and Nasdaq-100"),
    ("Nasdaq-100", "the Nasdaq-100"),
    ("My tickers", "your tickers"),
    ("Original 60 holdings", "your original 60 holdings"),
    ("Upload CSV", "your CSV file"),
    ("Demo", "the demo data"),
];
/// Markets that download Yahoo prices.
pub const YAHOO: [&str; 4] = ["S&P 500 + Nasdaq-100", "Nasdaq-100", "My tickers", "Original 60 holdings"];
/// The smallest weight that counts as a holding.
pub const HELD: f64 = 0.0005;
pub const INFO_SVG: &str = "<svg viewBox=\"0 0 18 18\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.5\" aria-hidden=\"true\">\
<circle cx=\"9\" cy=\"9\" r=\"7.25\"/><path d=\"M9 8v5\" stroke-linecap=\"round\"/>\
<circle cx=\"9\" cy=\"5.5\" r=\".9\" fill=\"currentColor\" stroke=\"none\"/></svg>";

/// Legal and security-type suffixes that short company names drop, one at a time.
/// `Class X` and `Series X` with one capital letter also qualify.
const NAME_WORDS: [&str; 23] = [
    "Common Stock",
    "Common Shares",
    "Capital Stock",
    "Ordinary Shares",
    "American Depositary Shares",
    "New York Registry Shares",
    "Subordinate Voting Shares",
    "Inc",
    "Inc.",
    "Incorporated",
    "Corporation",
    "Corp",
    "Corp.",
    "Company",
    "plc",
    "PLC",
    "N.V.",
    "Ltd",
    "Ltd.",
    "Limited",
    "S.A.",
    "AG",
    "SE",
];

/// One company card: name, one business sentence, and the profile URL (empty when unknown).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Company {
    pub name: String,
    pub description: String,
    pub source_url: String,
}

/// The applied setup that a context line describes.
#[derive(Clone, Debug, PartialEq)]
pub struct SetupContext {
    /// One of the app markets, such as "Nasdaq-100" or "Upload CSV".
    pub market: String,
    /// The uploaded price file name, if any.
    pub prices_file: Option<String>,
    /// "5 years", "10 years" or "Custom dates".
    pub history: String,
    pub start: Date,
    /// Exclusive end date.
    pub end: Date,
}

// ----------------------------------------------------------------------------- formatting helpers

/// `pairs.get(key)` for the constant label tables.
pub fn lookup<'a>(pairs: &[(&str, &'a str)], key: &str) -> Option<&'a str> {
    pairs.iter().find(|(name, _)| *name == key).map(|(_, value)| *value)
}

fn growth(value: f64) -> String {
    format!("{} {}", if value < 0.0 { "lost" } else { "grew" }, pct(value.abs(), 1, false))
}

fn minus(text: &str) -> String {
    text.replace('-', "−")
}

/// Format a fraction as a percent with a true minus sign.
pub fn pct(value: f64, digits: usize, plus: bool) -> String {
    minus(&pyfmt::format(value, &format!("{}.{digits}%", if plus { "+" } else { "" })))
}

/// Show a date as "2 Oct 2026".
pub fn date_text(date: Date) -> String {
    format!("{} {}", date.day(), date.format("%b %Y"))
}

/// Return "S&P 500" for "S&P 500 (SPY)": drop one final bracket note and the space before it.
pub fn market_name(name: &str) -> String {
    let Some(body) = name.strip_suffix(')') else { return name.to_string() };
    let segment = body.rfind(')').map_or(0, |position| position + 1);
    match body[segment..].find('(') {
        Some(open) => name[..segment + open].trim_end().to_string(),
        None => name.to_string(),
    }
}

fn noun(universe: &str) -> &'static str {
    NOUN.iter().find(|(name, _)| *name == universe).map_or("asset", |(_, (_, single))| single)
}

fn noun_phrase(universe: &str) -> &'static str {
    NOUN.iter().find(|(name, _)| *name == universe).map_or("assets", |(_, (phrase, _))| phrase)
}

/// Return "these 92 stocks", or "this one asset" when there is only one.
pub fn these_assets(count: usize, universe: &str) -> String {
    if count == 1 { format!("this one {}", noun(universe)) } else { format!("these {count} {}s", noun(universe)) }
}

/// Escape text and keep each hyphenated word on one line.
pub fn nb(text: &str) -> String {
    let mut output = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, output: &mut String| {
        let characters: Vec<char> = word.chars().collect();
        let hyphenated = (1..characters.len().saturating_sub(1)).any(|i| characters[i] == '-');
        if hyphenated {
            output.push_str(&format!("<span class=\"pl-nb\">{word}</span>"));
        } else {
            output.push_str(word);
        }
        word.clear();
    };
    for character in escape(text).chars() {
        if character.is_whitespace() {
            flush(&mut word, &mut output);
            output.push(character);
        } else {
            word.push(character);
        }
    }
    flush(&mut word, &mut output);
    output
}

/// Weights of at least [`HELD`], largest first in a stable order.
fn held(weights: &Series) -> Series {
    let order = weights.descending_order();
    let kept: Vec<usize> = order.into_iter().filter(|i| weights.values[*i] >= HELD).collect();
    Series::new(
        kept.iter().map(|i| weights.labels[*i].clone()).collect(),
        kept.iter().map(|i| weights.values[*i]).collect(),
    )
}

/// The first label with the largest value, as `Series.idxmax()`.
fn idxmax(weights: &Series) -> Option<usize> {
    let largest = weights.max();
    weights.values.iter().position(|value| *value == largest)
}

/// Whether `rest` (the text after a leading whitespace run) is one suffix word and trailing spaces.
fn suffix_word(rest: &str) -> bool {
    let word = rest.trim_end();
    let class = |prefix: &str| {
        word.strip_prefix(prefix)
            .is_some_and(|letter| letter.len() == 1 && letter.chars().all(|c| c.is_ascii_uppercase()))
    };
    NAME_WORDS.contains(&word) || class("Class ") || class("Series ")
}

/// The start of the first match of the name-suffix pattern, which must run to the end of `text`.
fn name_suffix(text: &str) -> Option<usize> {
    let only_space = |rest: &str| rest.chars().all(char::is_whitespace);
    text.char_indices().map(|(i, _)| i).find(|&i| {
        let suffix = &text[i..];
        let body = suffix.trim_start();
        let comma = body.strip_prefix(',').is_some_and(only_space);
        let bracket = body
            .strip_prefix('(')
            .and_then(|inner| inner.find(')').map(|close| &inner[close + 1..]))
            .is_some_and(only_space);
        let word = suffix.starts_with(char::is_whitespace) && suffix_word(body);
        comma || bracket || word
    })
}

/// Drop the security type, share class, bracket notes and legal suffixes for display.
/// Exported files keep the raw names.
pub fn short_company_name(name: &str) -> String {
    let mut text = name.trim().to_string();
    loop {
        let shorter = match name_suffix(&text) {
            Some(start) => text[..start].to_string(),
            None => text.clone(),
        };
        if shorter == text || shorter.trim().is_empty() {
            return text;
        }
        text = shorter;
    }
}

/// Show an engine strategy such as "Expanding window · Extreme" as "Refit on all past prices · Highest".
pub fn strategy_label(name: &str) -> String {
    match name.split_once(" · ") {
        None => display_label(name),
        Some((method, policy)) => format!(
            "{} · {}",
            lookup(RULE_LABELS, method).unwrap_or(method),
            lookup(LEVEL_LABELS, policy).unwrap_or(policy)
        ),
    }
}

/// Python `str.capitalize()`: the first character in upper case and the rest in lower case.
pub fn capitalize(text: &str) -> String {
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().chain(characters.flat_map(char::to_lowercase)).collect(),
        None => String::new(),
    }
}

pub fn setting_label(key: &str) -> String {
    lookup(SETTING_LABELS, key).map_or_else(|| capitalize(&key.replace('_', " ")), str::to_string)
}

/// A study setting as display text. Fractions show as percents. CSV and JSON files keep the raw values.
pub fn setting_text(key: &str, value: &Json) -> String {
    if let Json::Bool(flag) = value {
        return if *flag { "Yes" } else { "No" }.into();
    }
    if let Some(number) = value.as_f64() {
        match key {
            "train_fraction" | "shrinkage" => return format!("{}%", pyfmt::g(number * 100.0)),
            "risk_free_rate" | "cash_interest_rate" => return pyfmt::pct(number, 2),
            "max_weight" => {
                return if number >= 1.0 { "No limit".into() } else { format!("{}%", pyfmt::g(number * 100.0)) };
            }
            "cost_bps" => {
                return format!("{} basis points ({}%)", pyfmt::g(number), pyfmt::format(number / 100.0, ".2f"));
            }
            "periods_per_year" | "trading_days_per_year" => return pyfmt::g(number),
            _ => {}
        }
    }
    match (key, value.as_str()) {
        ("return_objective", Some("historical_mean")) => return "Annual historical arithmetic mean".into(),
        ("return_objective", Some("worst_window")) => {
            return "Lowest annual mean across three historical windows".into();
        }
        ("selection_method", Some("historical_mean_frontier")) => {
            return "Historical mean and variance frontier".into();
        }
        _ => {}
    }
    match value {
        Json::List(items) => items.iter().map(Json::py_str).collect::<Vec<_>>().join(", "),
        other => other.py_str(),
    }
}

/// Map each symbol to its short company name from a coverage table with `symbol` and `name` columns.
pub fn company_names(coverage: Option<&Table>) -> IndexMap<String, String> {
    let mut names = IndexMap::new();
    let Some(frame) = coverage else { return names };
    if frame.is_empty() || frame.position("symbol").is_none() || frame.position("name").is_none() {
        return names;
    }
    let text = |cell: &Cell| if cell.is_null() { "nan".to_string() } else { cell.py_str() };
    for (symbol, name) in frame.column("symbol").iter().zip(frame.column("name")) {
        names.insert(text(symbol), short_company_name(&text(name)));
    }
    names
}

/// Show company identities beside the saved weights, without rounding holdings away.
///
/// A symbol without a company shows the symbol as its name, with no description.
pub fn holdings_cards_html(weights: &Series, companies: &IndexMap<String, Company>) -> String {
    let mut cards = String::new();
    for (symbol, weight) in weights.iter() {
        let fallback = Company { name: symbol.to_string(), ..Company::default() };
        let company = companies.get(symbol).unwrap_or(&fallback);
        let name = escape(&company.name);
        let title = if company.source_url.is_empty() {
            name
        } else {
            format!(
                "<a href=\"{}\" target=\"_blank\" rel=\"noopener noreferrer\">{name}</a>",
                escape(&company.source_url)
            )
        };
        let amount = escape(&if weight >= 0.0001 { pyfmt::pct(weight, 2) } else { "<0.01%".into() });
        cards.push_str(&format!(
            "<article class=\"pl-holding\"><div class=\"pl-holding-head\"><div><h4>{title}</h4>\
             <span class=\"pl-holding-symbol\">{}</span></div>\
             <strong class=\"pl-holding-weight\">{amount}</strong></div><p>{}</p></article>",
            escape(symbol),
            escape(&company.description)
        ));
    }
    format!("<div class=\"pl-holdings-grid\">{cards}</div>")
}

fn setting_date(settings: &Dict, key: &str) -> Date {
    Date::parse_from_str(settings.str(key), "%Y-%m-%d").unwrap_or_else(|_| panic!("setting {key:?} is an ISO date"))
}

/// Length of the held-back test period in months.
pub fn months_tested(study: &BacktestStudy) -> i64 {
    let days = (setting_date(&study.settings, "test_end") - setting_date(&study.settings, "test_start")).num_days();
    (days as f64 / 30.44).round_ties_even() as i64
}

// ----------------------------------------------------------------------------- headlines and context

/// Answer "What would I hold?" in one sentence.
pub fn headline_text(weights: &Series, profile: &str, universe: &str) -> String {
    let held = held(weights);
    let level = lookup(LEVEL, profile).unwrap_or(profile);
    if held.max() >= 0.5 {
        return format!("The {level}-risk portfolio puts {} in one {}.", pyfmt::pct(held.max(), 1), noun(universe));
    }
    format!("A {level}-risk portfolio of {} {}.", held.len(), noun_phrase(universe))
}

/// The headline as escaped HTML. "Nasdaq-100 stocks" and hyphenated words do not break.
pub fn headline_html(weights: &Series, profile: &str, universe: &str) -> String {
    let text = headline_text(weights, profile, universe);
    text.split("Nasdaq-100 stocks").map(nb).collect::<Vec<_>>().join("<span class=\"pl-nb\">Nasdaq-100 stocks</span>")
}

pub fn welcome_headline_html(market: &str) -> String {
    nb(&format!("Find three portfolios in {}.", lookup(PLACE, market).unwrap_or("your data")))
}

pub fn work_headline_html(market: &str, rerun: bool) -> String {
    if rerun {
        return nb("Updating your portfolios.");
    }
    nb(&format!("Finding portfolios in {}.", lookup(PLACE, market).unwrap_or("your data")))
}

/// Python truthiness of a metadata value.
pub fn truthy(value: Option<&Json>) -> bool {
    match value {
        None | Some(Json::Null) => false,
        Some(Json::Bool(flag)) => *flag,
        Some(Json::Int(number)) => *number != 0,
        Some(Json::Float(number)) => *number != 0.0,
        Some(Json::Str(text)) => !text.is_empty(),
        Some(Json::List(items)) => !items.is_empty(),
        Some(Json::Dict(items)) => !items.is_empty(),
    }
}

/// Return "5 years" or "10 years" when the Yahoo request used a preset, else `None`.
fn history(metadata: &Dict) -> Option<String> {
    let (start, end) = (metadata.get("requested_start"), metadata.get("requested_end_exclusive"));
    if !truthy(start) || !truthy(end) {
        return None;
    }
    let parse = |value: Option<&Json>| Date::parse_from_str(&value?.py_str(), "%Y-%m-%d").ok();
    let (start, end) = (parse(start)?, parse(end)?);
    [5, 10]
        .into_iter()
        .find(|years| end.checked_sub_months(Months::new(12 * years)) == Some(start))
        .map(|years| format!("{years} years"))
}

fn context(parts: &[String], extra: &str) -> String {
    format!("<p class=\"pl-context\">{}{extra}</p>", parts.join("<span class=\"pl-sep\">·</span>"))
}

/// The context line for a result, for example "Nasdaq-100 · 92 of 101 stocks · 5 years to 2 Oct 2026".
pub fn context_result_html(metadata: &Dict, prices: &Frame) -> String {
    let universe = metadata.get("universe").map_or_else(String::new, Json::py_str);
    let first = date_text(prices.index[0]);
    let last = date_text(*prices.index.last().expect("prices have dates"));
    let history = if YAHOO.contains(&universe.as_str()) { history(metadata) } else { None };
    let period = match history {
        Some(history) => format!("<span>{history}<span class=\"hide-m\"> to {last}</span></span>"),
        None => format!("<span>{first} to {last}</span>"),
    };
    let columns = prices.ncols();
    let parts = if universe == "Nasdaq-100" || universe == "S&P 500 + Nasdaq-100" {
        let count = |key: &str| metadata.get(key).map_or_else(|| columns.to_string(), Json::py_str);
        let count = format!("{} of {} stocks", count("universe_included"), count("universe_requested"));
        vec![format!("<span>{}</span>", escape(&universe)), format!("<span>{}</span>", escape(&count)), period]
    } else {
        let label = match metadata.get("input_file") {
            Some(file) if universe == "Upload CSV" && truthy(Some(file)) => format!("CSV: {}", file.py_str()),
            _ => universe.clone(),
        };
        let count = format!("{columns} {}{}", noun(&universe), if columns == 1 { "" } else { "s" });
        vec![format!("<span>{}</span>", escape(&label)), format!("<span>{count}</span>"), period]
    };
    let source = metadata.get("source").map_or_else(String::new, Json::py_str).to_lowercase();
    let synthetic = source.contains("demo") || source.contains("synthetic");
    context(&parts, if synthetic { " <span class=\"pl-badge\">Simulated</span>" } else { "" })
}

/// The context line before a result, for example "Nasdaq-100 · 5 years".
pub fn context_setup_html(setup: &SetupContext, badge: Option<&str>) -> String {
    let market = setup.market.as_str();
    let label = match (market, &setup.prices_file) {
        ("Upload CSV", Some(file)) => format!("CSV: {file}"),
        _ => market.to_string(),
    };
    let mut parts = vec![format!("<span>{}</span>", escape(&label))];
    if YAHOO.contains(&market) {
        let history = if setup.history == "Custom dates" {
            let last = setup.end.pred_opt().expect("the end date has a previous day");
            format!("{} to {}", date_text(setup.start), date_text(last))
        } else {
            setup.history.clone()
        };
        parts.push(format!("<span>{}</span>", escape(&history)));
    }
    let extra = match badge {
        Some(badge) if !badge.is_empty() => format!(" <span class=\"pl-badge\">{}</span>", escape(badge)),
        _ => String::new(),
    };
    context(&parts, &extra)
}

// ----------------------------------------------------------------------------- holdings

/// The proposed holdings and one bar row per position. More than 12 rows end in one rest row.
pub fn list_html(weights: &Series, names: Option<&IndexMap<String, String>>) -> String {
    let held = held(weights);
    let top = held.values[0];
    let shown = if held.len() <= 12 { held.len() } else { 10 };
    let mut rows = String::new();
    for index in 0..shown {
        let (symbol, weight) = (&held.labels[index], held.values[index]);
        let label = match names.and_then(|names| names.get(symbol)).filter(|name| !name.is_empty()) {
            Some(name) => format!("<b>{}</b><small>{}</small>", escape(name), escape(symbol)),
            None => format!("<b>{}</b>", escape(symbol)),
        };
        rows.push_str(&format!(
            "<li class=\"pl-row\" style=\"--i:{index}\"><span class=\"pl-name\">{label}</span>\
             <span class=\"pl-bar\"><i style=\"width:{}%\"></i></span><span class=\"pl-pct\">{}</span></li>",
            pyfmt::format(weight / top * 100.0, ".1f"),
            pct(weight, 1, false)
        ));
    }
    if held.len() > 12 {
        let rest = 1.0 - np_sum(&held.values[..shown]);
        rows.push_str(&format!(
            "<li class=\"pl-row is-rest\" style=\"--i:10\"><span class=\"pl-name\"><b>{} more holdings</b>\
             <small>The largest of these is {}.</small></span>\
             <span class=\"pl-bar\"><i style=\"width:{}%\"></i></span><span class=\"pl-pct\">{}</span></li>",
            held.len() - 10,
            pct(held.values[10], 1, false),
            pyfmt::format((rest / top).min(1.0) * 100.0, ".1f"),
            pct(rest, 1, false)
        ));
    }
    format!(
        "<div class=\"pl-head\"><h3 class=\"pl-title\">Proposed holdings</h3><small>Share of the portfolio</small></div>\
         <ul class=\"pl-list\">{rows}</ul>"
    )
}

/// A concentration note when one holding is 50% or more. Empty otherwise.
pub fn callout_html(weights: &Series, universe: &str, names: Option<&IndexMap<String, String>>) -> String {
    let largest_weight = weights.max();
    if largest_weight < 0.5 {
        return String::new();
    }
    let largest = idxmax(weights).map_or("", |i| weights.labels[i].as_str());
    let name = names.and_then(|names| names.get(largest)).map_or(largest, String::as_str);
    let title = if largest_weight >= 0.8 {
        format!("Almost all of it is {name}.")
    } else {
        format!("At least half of it is {name}.")
    };
    let kind = if noun(universe) == "stock" { "company" } else { "asset" };
    format!(
        "<div class=\"pl-callout\"><span class=\"pl-glyph\" aria-hidden=\"true\">!</span><p><b>{}</b>\
         This is a large allocation to one {kind}. To spread the risk, select Change, \
         then Model assumptions, and set a holding limit.</p></div>",
        escape(&title)
    )
}

/// The names of the given symbols, or the symbol itself.
fn name_of<'a>(names: Option<&'a IndexMap<String, String>>, symbol: &'a str) -> &'a str {
    names.and_then(|names| names.get(symbol)).map_or(symbol, String::as_str)
}

fn profile_value(latest: &ProfileAnalysis, profile: &str, column: &str) -> f64 {
    latest.summary.f64_at(profile, column)
}

/// Fitted mean return, volatility and largest holding, divided by hairlines.
pub fn stats_html(
    latest: &ProfileAnalysis,
    profile: &str,
    universe: &str,
    names: Option<&IndexMap<String, String>>,
) -> String {
    let _ = universe;
    let portfolio = &latest.portfolios[profile];
    let weights = &portfolio.weights;
    let largest = idxmax(weights).map_or("", |i| weights.labels[i].as_str());
    let stats = [
        (
            pct(profile_value(latest, profile, "expected_return"), 1, false),
            "Fitted annual mean",
            "Full-history estimate before fees",
        ),
        (pct(portfolio.volatility, 1, false), "Model volatility", "Yearly, from the fitted covariance"),
        (pct(weights.max(), 1, false), "Largest holding", name_of(names, largest)),
    ];
    let cells: String = stats
        .iter()
        .map(|(number, label, sub)| {
            format!(
                "<div class=\"pl-stat\"><span class=\"pl-num\">{number}</span><span class=\"pl-lbl\">{label}</span><small>{}</small></div>",
                escape(sub)
            )
        })
        .collect();
    format!("<div class=\"pl-stats\">{cells}</div>")
}

/// The three paragraphs of "Why these stocks?".
pub fn why_text(latest: &ProfileAnalysis, profile: &str, universe: &str) -> Vec<String> {
    let noun_plural = format!("{}s", noun(universe));
    let years: f64 = pyfmt::format(np_sum(&latest.windows.f64s("years")), ".1f").parse().unwrap_or(f64::NAN);
    let count = latest.portfolios[profile].weights.len();
    let mixes = if count > 1 {
        format!("The model tries weighted mixes of all {count} {noun_plural}.")
    } else {
        format!("The model has one {} to hold.", noun(universe))
    };
    vec![
        format!(
            "{mixes} It uses all {} years to estimate annual arithmetic mean returns and volatility. \
             The frontier finds the least volatile mix for each target mean return.",
            pyfmt::g(years)
        ),
        format!(
            "This mix has a fitted annual arithmetic mean of {}. \
             It uses the same prices that selected today's holdings. It is not compound growth (CAGR) or a forecast. \
             The next section fits the rule using earlier prices and measures realised growth on later prices.",
            pct(profile_value(latest, profile, "expected_return"), 1, false)
        ),
        format!(
            "Low has the least fitted volatility. Medium is the sampled frontier point nearest to the midpoint between Low and Highest volatility. \
             Highest maximises the fitted annual mean return. These levels are relative to the selected {noun_plural}; \
             Highest uses no borrowing and does not maximise every measure of risk."
        ),
    ]
}

/// The "How the three levels compare" table with its footnote.
pub fn levels_ledger_html(
    latest: &ProfileAnalysis,
    selected: &str,
    names: Option<&IndexMap<String, String>>,
    universe: &str,
) -> String {
    let count_label = if noun(universe) == "stock" { "Stocks held" } else { "Assets held" };
    let mut rows = String::new();
    for profile in latest.summary.index_labels() {
        let weights = &latest.portfolios[&profile].weights;
        let largest = idxmax(weights).map_or("", |i| weights.labels[i].as_str());
        let ramp = match profile.as_str() {
            "Low" => "var(--pl-risk-low)",
            "Medium" => "var(--pl-risk-med)",
            "Extreme" => "var(--pl-risk-high)",
            _ => "var(--pl-rest)",
        };
        let css = if profile == selected { " class=\"on\"" } else { "" };
        rows.push_str(&format!(
            "<tr{css}><td><i class=\"pl-ldot\" style=\"background:{ramp}\"></i>{}</td><td>{}</td>\
             <td class=\"hide-m\">{} {}</td><td>{}</td><td>{}</td></tr>",
            escape(&display_label(&profile)),
            pct(profile_value(latest, &profile, "volatility"), 1, false),
            escape(name_of(names, largest)),
            pct(weights.max(), 1, false),
            held(weights).len(),
            pct(profile_value(latest, &profile, "expected_return"), 1, false)
        ));
    }
    format!(
        "<table class=\"pl-ledger\"><thead><tr><th>Level</th><th>Volatility</th><th class=\"hide-m\">Largest holding</th>\
         <th>{count_label}</th><th>Historical mean*</th></tr></thead><tbody>{rows}</tbody></table>\
         <p class=\"pl-small\">*Annual arithmetic mean fitted to all selected prices. This is not later realised \
         compound growth (CAGR) or a forecast.</p>"
    )
}

// ----------------------------------------------------------------------------- market test

/// `numpy.round(value * 10_000, 2) != 0`: the amount shows at two decimals per 10,000.
fn visible_amount(value: f64) -> bool {
    (value * 10_000.0 * 100.0).round_ties_even() != 0.0
}

/// The benchmarks with backtest paths, or `None`.
fn matched(benchmarks: Option<&BenchmarkComparison>) -> Option<&BenchmarkComparison> {
    benchmarks.filter(|benchmarks| benchmarks.backtest_equity.is_some())
}

/// The backtest paths and metrics of matched benchmarks.
fn market_paths(benchmarks: &BenchmarkComparison) -> (&Frame, &Table) {
    (
        benchmarks.backtest_equity.as_ref().expect("matched benchmarks have paths"),
        benchmarks.backtest_metrics.as_ref().expect("matched benchmarks have metrics"),
    )
}

/// Python `round(value * 1000)`: growth compared at a tenth of a percentage point.
fn tenths(value: f64) -> f64 {
    (value * 1000.0).round_ties_even()
}

/// Compare CAGR at 0.1 percentage point, without implying equal risk.
pub fn market_answer(study: &BacktestStudy, selected: &str, benchmarks: Option<&BenchmarkComparison>) -> String {
    let cagr = study.metrics.f64_at(selected, "cagr");
    let Some(markets) = matched(benchmarks) else {
        return format!("It {} a year.", growth(cagr));
    };
    let comparison_note = " This compares compound growth (CAGR), without matching volatility.";
    let (_, metrics) = market_paths(markets);
    let mut outcomes: Vec<(String, &str)> = Vec::new();
    for (name, theirs) in metrics.index_labels().iter().zip(metrics.f64s("cagr")) {
        let (mine, theirs) = (tenths(cagr), tenths(theirs));
        let outcome = if mine > theirs {
            "beat"
        } else if mine < theirs {
            "trailed"
        } else {
            "matched"
        };
        let name = market_name(name);
        match outcomes.iter_mut().find(|(known, _)| *known == name) {
            Some(entry) => entry.1 = outcome,
            None => outcomes.push((name, outcome)),
        }
    }
    if outcomes.len() == 1 {
        let (name, outcome) = &outcomes[0];
        return format!("It {outcome} the {name} in this test.{comparison_note}");
    }
    if outcomes.iter().all(|(_, outcome)| *outcome == outcomes[0].1) {
        let answer = match outcomes[0].1 {
            "trailed" => "No. It trailed both markets.",
            "beat" => "Yes. It beat both markets in this test.",
            _ => "It matched both markets in this test.",
        };
        return format!("{answer}{comparison_note}");
    }
    let rank = |outcome: &str| match outcome {
        "beat" => 0,
        "matched" => 1,
        _ => 2,
    };
    let mut ordered: Vec<(&str, &str)> = outcomes.iter().map(|(name, outcome)| (*outcome, name.as_str())).collect();
    ordered.sort_by_key(|(outcome, _)| rank(outcome));
    let ((better, first), (worse, second)) = (ordered[0], ordered[1]);
    format!("Partly. It {better} the {first} but {worse} the {second}.{comparison_note}")
}

/// The annual arithmetic mean of a path's net returns.
fn path_mean(path: &[f64], periods: f64) -> f64 {
    np_mean(&path_returns(path)) * periods
}

/// Compare the requested rule with fixed alternatives, without choosing a winner.
pub fn decision_panel_html(study: &BacktestStudy, selected: &str, benchmarks: Option<&BenchmarkComparison>) -> String {
    let (method, policy) = selected.split_once(" · ").unwrap_or((selected, ""));
    let in_study = |name: &str| study.equity.position(name).is_some() && study.metrics.row_of(name).is_some();
    let available = in_study(selected);
    let markets = matched(benchmarks).filter(|markets| market_paths(markets).0.index == study.equity.index);
    let mut rows: Vec<(String, String, &Frame, &Table)> =
        ["Low", "Medium", "Extreme", "Maximum Sharpe", "Equal weight"]
            .iter()
            .map(|name| (format!("{method} · {name}"), display_label(name)))
            .filter(|(name, _)| in_study(name))
            .map(|(name, label)| (name, label, &study.equity, &study.metrics))
            .collect();
    let mut etf_count = 0;
    if let Some(markets) = markets {
        let (equity, metrics) = market_paths(markets);
        for ticker in ["SPY", "QQQ"] {
            let found = metrics
                .index_labels()
                .into_iter()
                .find(|name| name == ticker || name.ends_with(&format!("({ticker})")));
            if let Some(name) = found.filter(|name| equity.position(name).is_some()) {
                rows.push((name.clone(), name, equity, metrics));
                etf_count += 1;
            }
        }
    }
    let periods = study.settings.f64("periods_per_year");
    let table_rows: String = rows
        .iter()
        .map(|(name, label, equity, metrics)| {
            let mean = path_mean(&equity.column(name).expect("a listed path"), periods);
            let values = [
                mean,
                metrics.f64_at(name, "cagr"),
                metrics.f64_at(name, "volatility"),
                metrics.f64_at(name, "max_drawdown"),
            ];
            let css = if name == selected { " class=\"on\"" } else { "" };
            let label = format!("{}{}", escape(label), if name == selected { " · Selected" } else { "" });
            let cells: String = values.iter().map(|value| format!("<td>{}</td>", pct(*value, 1, false))).collect();
            format!("<tr{css}><th scope=\"row\">{label}</th>{cells}</tr>")
        })
        .collect();
    let (verdict, implication) = if available {
        let verdict = market_answer(study, selected, markets);
        let mut implication = "";
        if let Some(markets) = markets {
            let metrics = market_paths(markets).1;
            let cagr = tenths(study.metrics.f64_at(selected, "cagr"));
            if etf_count == metrics.nrows()
                && etf_count == 2
                && metrics.f64s("cagr").iter().all(|value| cagr < tenths(*value))
            {
                implication = " For compound growth, this test gives no reason to prefer this rule to either ETF.";
            }
        }
        (verdict, implication)
    } else {
        (
            format!("{} was not backtested. Its result is unavailable.", strategy_label(selected)),
            " The table shows the available comparisons; no other rule replaces the selection.",
        )
    };
    let purpose = match policy {
        "Low" => "Low targets the least fitted volatility.",
        "Medium" => {
            "Medium selects the sampled frontier point nearest to the midpoint of Low and Highest fitted volatility. \
             Medium is not the market average and can be riskier than both ETFs."
        }
        "Extreme" => {
            "Highest maximises the fitted historical arithmetic mean return. Without a holding cap, it can allocate to one asset."
        }
        "Maximum Sharpe" => "Maximum Sharpe targets fitted excess return per unit of volatility.",
        "Equal weight" => "Equal weight assigns the same target weight to each asset.",
        _ => "",
    };
    let dates = &study.equity.index;
    let missing = if markets.is_none() {
        "<p>Matching ETF results are unavailable for these test dates.</p>"
    } else if etf_count < 2 {
        "<p>One or both ETF comparisons are unavailable for these test dates.</p>"
    } else {
        ""
    };
    format!(
        "<h3 class=\"pl-title\">Compare rules and ETFs</h3>\
         <p class=\"pl-sub\">Selected: {}. Test returns: {} to {}. Baseline: {}. Rule: {} ({}).</p>\
         <p class=\"pl-verdict\">{}{}</p>{missing}\
         <div class=\"scroll\" style=\"overflow-x:auto\"><table class=\"pl-ledger\"><thead><tr><th>Rule / ETF</th>\
         <th>Test mean / year</th><th>Growth / year (CAGR)</th><th>Volatility / year</th><th>Maximum drawdown</th>\
         </tr></thead><tbody>{table_rows}</tbody></table></div>\
         <p class=\"pl-small\">Test mean is the arithmetic mean of net portfolio returns, including the entry interval and fees, \
         scaled to a year. CAGR compounds the same net path. Rows follow a fixed order, not a performance ranking.</p>\
         <p>{} These are model rules, not recommendations. Risk differences are shown in the table.</p>\
         <p>Latest proposed weights fit all selected prices. This replay uses earlier fits, so its holdings and weights \
         can differ from the latest proposal.</p>",
        escape(&strategy_label(selected)),
        date_text(dates[1]),
        date_text(dates[dates.len() - 1]),
        date_text(dates[0]),
        escape(lookup(RULE_LABELS, method).unwrap_or(method)),
        escape(method),
        escape(&verdict),
        escape(implication),
        escape(purpose),
    )
}

/// Show gains and losses from the replay and reconcile all holdings less fees.
pub fn result_drivers_html(study: &BacktestStudy, selected: &str) -> String {
    let heading = "<h3 class=\"pl-title\">What drove the result?</h3>";
    let (Some(holdings), Some(row)) = (study.holdings.get(selected), study.metrics.row_of(selected)) else {
        return format!(
            "{heading}<p>Holding contributions for {} are unavailable.</p>",
            escape(&strategy_label(selected))
        );
    };
    let names = holdings.index_labels();
    let contributions = holdings.f64s("pnl_contribution");
    let visible: Vec<usize> = (0..contributions.len()).filter(|i| visible_amount(contributions[*i])).collect();
    let ordered = |test: fn(f64) -> bool, descending: bool| {
        let mut chosen: Vec<usize> = visible.iter().copied().filter(|i| test(contributions[*i])).collect();
        chosen.sort_by(|a, b| {
            let order = contributions[*a].partial_cmp(&contributions[*b]).unwrap_or(std::cmp::Ordering::Equal);
            if descending { order.reverse() } else { order }
        });
        chosen.truncate(3);
        chosen
    };
    let shown: Vec<usize> =
        ordered(|value| value > 0.0, true).into_iter().chain(ordered(|value| value < 0.0, false)).collect();
    let mut amounts: Vec<(String, f64)> = shown.iter().map(|i| (names[*i].clone(), contributions[*i])).collect();
    // pandas skips NaN by summing it as zero in place.
    let rest: Vec<f64> = (0..contributions.len())
        .filter(|i| !shown.contains(i))
        .map(|i| if contributions[i].is_nan() { 0.0 } else { contributions[i] })
        .collect();
    let other = np_sum(&rest);
    if pyfmt::format((other * 10_000.0).abs(), ".2f") != "0.00" {
        amounts.push(("Other holdings".into(), other));
    }
    let metric = |name: &str| study.metrics.column(name)[row].as_f64().unwrap_or(f64::NAN);
    amounts.push(("Trading fees".into(), -metric("total_cost")));
    amounts.push(("Net change".into(), metric("total_return")));
    let rows: String = amounts
        .iter()
        .map(|(name, value)| {
            let amount =
                if *value == 0.0 { "0.00".to_string() } else { minus(&pyfmt::format(value * 10_000.0, "+,.2f")) };
            format!("<tr><th scope=\"row\">{}</th><td>{amount}</td></tr>", escape(name))
        })
        .collect();
    let note = if !visible.is_empty() {
        "Up to three largest gains and three largest losses are shown, with gains first. \
         Remaining contributions are combined. Amounts that round to zero are not listed separately."
    } else if !contributions.iter().any(|value| *value != 0.0 && !value.is_nan()) {
        "No holding contributed a gain or loss in this test."
    } else {
        "No holding contribution is visible at this precision."
    };
    format!(
        "{heading}<p class=\"pl-sub\">{}. {note}</p>\
         <table class=\"pl-ledger\"><thead><tr><th>Holding / cost</th><th>Gain or loss per initial 10,000</th></tr></thead>\
         <tbody>{rows}</tbody></table>\
         <p class=\"pl-small\">These contributions come from holdings in the historical replay, not the latest proposed weights. \
         All holding contributions less trading fees equal the net change. Amounts use initial capital as the denominator.</p>",
        escape(&strategy_label(selected))
    )
}

/// The lead under the market answer: what was replayed, how often it traded and what it paid.
pub fn market_lead(study: &BacktestStudy, selected: &str) -> String {
    let (method, policy) = selected.split_once(" · ").unwrap_or((selected, ""));
    let level = lookup(LEVEL, policy).map_or_else(|| policy.to_lowercase().replace(' ', "-"), str::to_string);
    let settings = &study.settings;
    let every = settings.f64("rebalance_every").trunc() as i64;
    let cost = settings.f64("cost_bps") / 100.0;
    let periods = settings.get("periods_per_year").and_then(Json::as_f64).unwrap_or(252.0);
    let unit = if periods == 252.0 { "trading days" } else { "price rows" };
    let rolling = settings.get("rolling_window").map_or_else(|| "None".to_string(), Json::py_str);
    let rule = match method {
        "Expanding window" => format!(
            "{} it refitted using all earlier returns.",
            if every == 21 && unit == "trading days" {
                "About once a month".to_string()
            } else {
                format!("Every {every} {unit}")
            }
        ),
        "Rolling window" => {
            format!("Every {every} {unit} it refitted on the most recent {rolling} return observations.")
        }
        "Fixed rebalance" => {
            format!("It kept the mix from the first fit. Every {every} {unit} it traded back to that mix.")
        }
        "Buy and hold" => "It bought the mix from the first fit once and then held it.".to_string(),
        _ => String::new(),
    };
    let selection = if lookup(LEVEL, policy).is_some() {
        " Each fit selected the risk level from a frontier of annual historical arithmetic mean returns and volatility."
    } else {
        ""
    };
    format!(
        "We replayed the {level}-risk rule over the last {} months. {rule}{selection} \
         Fits used returns through the prior close; targets took effect at the following close. \
         The first interval stayed in cash. It paid {}% on each amount bought or sold. \
         The result is later realised compound growth (CAGR), not the fitted mean return.",
        months_tested(study),
        pyfmt::format(cost, ".2f")
    )
}

/// Kicker, big numbers, realised result and the must-read note for the result tile.
///
/// `evidence` is not shown here; the evidence section explains the uncertainty.
pub fn result_tile_html(
    study: &BacktestStudy,
    selected: &str,
    benchmarks: Option<&BenchmarkComparison>,
    evidence: Option<&EvidenceAnalysis>,
    metadata: &Dict,
) -> String {
    let _ = evidence;
    let markets = matched(benchmarks);
    let equity = &study.equity;
    let last = |frame: &Frame, name: &str| *frame.column(name).expect("a known path").last().expect("a path value");
    let (total, cagr) = (study.metrics.f64_at(selected, "total_return"), study.metrics.f64_at(selected, "cagr"));
    let mut columns = vec![("rule", "Your rule".to_string(), last(equity, selected), total, cagr)];
    if let Some(markets) = markets {
        let (paths, metrics) = market_paths(markets);
        for name in metrics.index_labels() {
            let key = if name.contains("SPY") { "spy" } else { "qqq" };
            columns.push((
                key,
                market_name(&name),
                last(paths, &name),
                metrics.f64_at(&name, "total_return"),
                metrics.f64_at(&name, "cagr"),
            ));
        }
    }
    let numbers: String = columns
        .iter()
        .map(|(key, label, value, total, cagr)| {
            format!(
                "<div class=\"pl-bignum\"><span class=\"pl-key\"><i class=\"{key}\"></i>{}</span>\
                 <span class=\"pl-num\">{}</span><small><span>{}</span>\
                 <span class=\"pl-sep\"> · </span><span>{} a year</span></small></div>",
                escape(label),
                pyfmt::format(value * 10_000.0, ",.0f"),
                pct(*total, 1, true),
                pct(*cagr, 1, false)
            )
        })
        .collect();
    let verdict = if markets.is_some() {
        let others: Vec<String> = columns[1..]
            .iter()
            .map(|(_, label, _, _, cagr)| format!("{} {} a year", escape(label), growth(*cagr)))
            .collect();
        format!("It {} a year. <span>The {}.</span>", growth(cagr), others.join(" and the "))
    } else {
        format!("In total, it {} {}.", if total >= 0.0 { "gained" } else { "lost" }, pct(total.abs(), 1, false))
    };
    let months = months_tested(study);
    let (bold, why) = if metadata.get("universe").and_then(Json::as_str) == Some("Nasdaq-100") {
        (
            if months >= 36 { "A test on today’s winners." } else { "A short test on today’s winners." },
            "The stock list is today’s Nasdaq-100, so companies that left the index are missing. \
             That tends to flatter past results.",
        )
    } else {
        (
            if months >= 36 { "A fixed list." } else { "A short test on a fixed list." },
            "The asset list is fixed for the whole period. That can favour survivors.",
        )
    };
    let held_back = format!(
        "Only the last {months} months ({} to {}) were held back for this test.",
        date_text(setting_date(&study.settings, "test_start")),
        date_text(setting_date(&study.settings, "test_end"))
    );
    format!(
        "<span class=\"pl-kicker\">10,000 on {} became, by {}</span>\
         <div class=\"pl-bignums\">{numbers}</div><p class=\"pl-verdict\">{verdict}</p>\
         <p class=\"pl-mustread\">{INFO_SVG}<span><b>{bold}</b> {held_back} {why}</span></p>",
        date_text(equity.index[0]),
        date_text(equity.index[equity.nrows() - 1])
    )
}

/// The cash-first and entry-cost note under the market chart.
pub fn chart_footnote(study: &BacktestStudy, benchmarks: Option<&BenchmarkComparison>) -> String {
    let start = date_text(study.equity.index[0]);
    let entry = date_text(setting_date(&study.settings, "initial_execution"));
    if matched(benchmarks).is_none() {
        return format!(
            "It starts with 10,000 in cash on {start} and invests at the close on {entry}. Taxes are not included."
        );
    }
    format!(
        "All three start with 10,000 in cash on {start} and invest at the close on {entry}. The two markets pay the same \
         {}% entry cost. Taxes are not included.",
        pyfmt::format(study.settings.f64("cost_bps") / 100.0, ".2f")
    )
}

/// A section that opens with its question as the eyebrow and its answer as the headline.
pub fn section_header_html(anchor: &str, eyebrow: &str, answer: &str, lead: &str) -> String {
    let lead_html = if lead.is_empty() { String::new() } else { format!("<p class=\"pl-lead\">{}</p>", escape(lead)) };
    format!(
        "<section class=\"pl-section\" id=\"{}\"><p class=\"pl-eyebrow\">{}</p>\
         <h2 class=\"pl-answer\" tabindex=\"-1\">{}</h2>{lead_html}</section>",
        escape(anchor),
        escape(eyebrow),
        nb(answer)
    )
}

// ----------------------------------------------------------------------------- evidence

/// The plain category of an evidence status.
fn category(status: &str) -> &'static str {
    match status {
        "No clear statistical advantage" => "unclear",
        "Historical mean underperformance" => "behind",
        "Historical mean advantage" | "Historical mean advantage and alpha" => "ahead",
        "Historical alpha" => "alpha",
        "Insufficient history" => "insufficient",
        _ => "unavailable",
    }
}

/// The rows of one strategy in the evidence summary, as positions.
fn strategy_rows(evidence: &EvidenceAnalysis, strategy: &str) -> Vec<usize> {
    (0..evidence.summary.nrows()).filter(|i| evidence.summary.index[0][*i].as_str() == Some(strategy)).collect()
}

/// `(market, category, cagr_difference)` for each benchmark of the selected strategy.
fn statuses(evidence: &EvidenceAnalysis, selected: &str) -> Vec<(String, &'static str, f64)> {
    let summary = &evidence.summary;
    let status = summary.column("status");
    let gaps = summary.f64s("cagr_difference");
    strategy_rows(evidence, selected)
        .into_iter()
        .map(|i| (market_name(&summary.index[1][i].py_str()), category(&status[i].py_str()), gaps[i]))
        .collect()
}

/// Answer "How sure can we be?".
pub fn evidence_answer(evidence: Option<&EvidenceAnalysis>, selected: &str) -> String {
    let Some(evidence) = evidence else { return "The uncertainty could not be estimated.".into() };
    let statuses = statuses(evidence, selected);
    let categories: Vec<&str> = unique_categories(&statuses);
    if categories.len() == 1 {
        return match categories[0] {
            "unclear" => "No statistically clear difference in mean return.",
            "behind" => "Lower mean returns than both markets, in this sample.",
            "ahead" => "Higher mean returns than both markets, in this sample.",
            "alpha" => "Some evidence of alpha. No clear lead in average return.",
            "insufficient" => "Too little history to judge.",
            _ => "The uncertainty could not be estimated.",
        }
        .into();
    }
    statuses
        .iter()
        .map(|(name, category, _)| match *category {
            "unclear" => format!("No clear mean-return difference against the {name}."),
            "behind" => format!("Lower mean returns than the {name}, in this sample."),
            "ahead" => format!("Higher mean returns than the {name}, in this sample."),
            "alpha" => format!("Some evidence of alpha against the {name}."),
            "insufficient" => format!("Too little history against the {name}."),
            _ => format!("No estimate against the {name}."),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn unique_categories(statuses: &[(String, &'static str, f64)]) -> Vec<&'static str> {
    let mut categories: Vec<&'static str> = Vec::new();
    for (_, category, _) in statuses {
        if !categories.contains(category) {
            categories.push(category);
        }
    }
    categories
}

/// Separate realised compound growth from uncertainty in arithmetic mean returns.
pub fn evidence_lead(
    evidence: Option<&EvidenceAnalysis>,
    selected: &str,
    months: i64,
    metadata: Option<&Dict>,
) -> String {
    let note = |default: &str| {
        metadata.and_then(|metadata| metadata.get("evidence_note")).map_or_else(|| default.to_string(), Json::py_str)
    };
    let Some(evidence) = evidence else { return note("") };
    let statuses = statuses(evidence, selected);
    let tests = evidence.settings.get("multiple_testing_tests").map_or_else(|| "0".to_string(), Json::py_str);
    let directions: Vec<&str> = statuses
        .iter()
        .map(|(_, _, gap)| {
            if *gap > 0.0 {
                "led"
            } else if *gap < 0.0 {
                "trailed"
            } else {
                "matched"
            }
        })
        .collect();
    let comparison = if directions.iter().all(|direction| *direction == directions[0]) {
        format!("{} both markets", directions[0])
    } else {
        directions
            .iter()
            .zip(&statuses)
            .map(|(direction, (name, _, _))| format!("{direction} the {name}"))
            .collect::<Vec<_>>()
            .join(" and ")
    };
    let mut growth = format!("The rule {comparison} on realised compound growth (CAGR) in this sample.");
    if months < 36 {
        growth.push_str(&format!(" {months} months is a short test."));
    }
    let distinction =
        "The confidence ranges and adjusted tests concern annual arithmetic mean return gaps, not CAGR gaps.";
    let categories = unique_categories(&statuses);
    let inference = if categories.len() > 1 {
        format!("The statistical evidence differs by market after adjustment for all {tests} tests.")
    } else {
        match categories[0] {
            "unclear" => format!(
                "Neither annual mean return gap is statistically clear after allowing for returns that cluster in time \
                 and all {tests} tests. This does not establish equal performance."
            ),
            "behind" => format!(
                "The annual mean return gaps are statistically negative after adjustment for all {tests} tests. \
                 This describes the sample, not future performance."
            ),
            "ahead" => format!(
                "The annual mean return gaps are statistically positive after adjustment for all {tests} tests. \
                 This describes the sample, not future performance."
            ),
            "alpha" => "Alpha adjusts for exposure to one market. It does not prove skill.".to_string(),
            "insufficient" => format!(
                "The tests need at least {} paired return observations.",
                evidence.settings.get("inference_min_observations").map_or_else(|| "60".to_string(), Json::py_str)
            ),
            _ => note("The mean-return uncertainty could not be estimated."),
        }
    };
    format!("{growth} {distinction} {inference}")
}

/// One line on statistical clarity for the result tile.
pub fn clarity_text(evidence: Option<&EvidenceAnalysis>, selected: &str) -> String {
    let unknown = "The statistics could not be estimated for this test.";
    let Some(evidence) = evidence else { return unknown.into() };
    let statuses = statuses(evidence, selected);
    if statuses.iter().any(|(_, category, _)| matches!(*category, "insufficient" | "unavailable")) {
        return unknown.into();
    }
    let clear: Vec<&String> = statuses
        .iter()
        .filter(|(_, category, _)| matches!(*category, "ahead" | "behind"))
        .map(|(name, _, _)| name)
        .collect();
    if clear.is_empty() {
        return "Neither annual mean return gap is statistically clear. This does not test the CAGR gaps. See How sure can we be."
            .into();
    }
    if clear.len() == statuses.len() {
        return "Both annual mean return gaps are statistically clear in this sample. This does not test the CAGR gaps. See How sure can we be."
            .into();
    }
    format!(
        "Only the annual mean return gap to the {} is statistically clear in this sample. \
         This does not test the CAGR gaps. See How sure can we be.",
        clear[0]
    )
}

/// One decimal with a sign and a true minus. Zero has no sign.
fn signed(value: f64) -> String {
    if pyfmt::format(value.abs(), ".1f") == "0.0" { "0.0".into() } else { minus(&pyfmt::format(value, "+.1f")) }
}

/// Python `round(value, 6)`, then `+g` with a true minus. Zero has no sign.
fn tick(value: f64) -> String {
    let value: f64 = pyfmt::format(value, ".6f").parse().unwrap_or(value);
    if value == 0.0 { "0".into() } else { minus(&pyfmt::format(value, "+g")) }
}

/// Padded axis limits on whole steps, with 4 to 6 ticks.
fn interval_axis(values: &[f64]) -> (f64, f64, f64) {
    let low = values.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.0);
    let span = if high - low != 0.0 { high - low } else { 1.0 };
    let (low, high) = (low - span * 0.1, high + span * 0.1);
    let preferred = [5.0, 10.0, 20.0, 25.0, 50.0];
    let mut others: Vec<f64> = Vec::new();
    for k in -2..5 {
        for m in [1.0, 2.0, 2.5, 5.0] {
            let step: f64 = m * 10f64.powf(f64::from(k));
            if !preferred.contains(&step) && !others.contains(&step) {
                others.push(step);
            }
        }
    }
    others.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut best: Option<(f64, f64, f64, i64)> = None;
    for step in preferred.into_iter().chain(others) {
        let (first, last) = ((low / step).floor() * step, (high / step).ceil() * step);
        let count = ((last - first) / step).round_ties_even() as i64 + 1;
        if (4..=6).contains(&count) {
            return (first, last, step);
        }
        if best.is_none_or(|(_, _, _, known)| (count - 5).abs() < (known - 5).abs()) {
            best = Some((first, last, step, count));
        }
    }
    let (first, last, step, _) = best.expect("a step");
    (first, last, step)
}

/// Average yearly gap with its 95% range against each market, as an HTML forest plot.
pub fn interval_html(evidence: &EvidenceAnalysis, strategy: &str) -> String {
    let summary = &evidence.summary;
    let column = |name: &str| summary.f64s(name);
    let (advantage, low_ci, high_ci) =
        (column("annual_advantage"), column("advantage_ci_low"), column("advantage_ci_high"));
    let data: Vec<(String, f64, f64, f64)> = strategy_rows(evidence, strategy)
        .into_iter()
        .map(|i| {
            (market_name(&summary.index[1][i].py_str()), advantage[i] * 100.0, low_ci[i] * 100.0, high_ci[i] * 100.0)
        })
        .collect();
    let mut values: Vec<f64> =
        data.iter().flat_map(|(_, a, b, c)| [*a, *b, *c]).filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        values.push(0.0);
    }
    let (low, high, step) = interval_axis(&values);
    let position = |value: f64| (value - low) / (high - low) * 100.0;
    let percent = |value: f64| pyfmt::format(value, ".2f");
    let zero = position(0.0);
    let mut body = String::new();
    for (name, estimate, ci_low, ci_high) in &data {
        let has_range = ci_low.is_finite() && ci_high.is_finite();
        let span = if has_range {
            format!(
                "<i class=\"f-ci\" style=\"left:{}%;width:{}%\"></i>",
                percent(position(*ci_low)),
                percent(position(*ci_high) - position(*ci_low))
            )
        } else {
            String::new()
        };
        let dot = if estimate.is_finite() {
            format!("<i class=\"f-dot\" style=\"left:{}%\"></i>", percent(position(*estimate)))
        } else {
            String::new()
        };
        let detail = if has_range {
            format!("range {} to {}", signed(*ci_low), signed(*ci_high))
        } else {
            "no range for this test".into()
        };
        body.push_str(&format!(
            "<div class=\"f-row\"><div class=\"f-label\">vs {}</div>\
             <div class=\"f-value\"><b>{}</b> points<small>{detail}</small></div>\
             <div class=\"f-plot\"><i class=\"f-zero\" style=\"left:{}%\"></i>{span}{dot}</div></div>",
            escape(name),
            if estimate.is_finite() { signed(*estimate) } else { "—".into() },
            percent(zero)
        ));
    }
    let count = ((high - low) / step).round_ties_even() as i64 + 1;
    let ticks: String = (0..count)
        .map(|i| {
            let value = low + i as f64 * step;
            format!("<span style=\"left:{}%\">{}</span>", percent(position(value)), tick(value))
        })
        .collect();
    format!(
        "<div class=\"pl-forest\"><div class=\"f-sides\"><div></div><div class=\"mid\">\
         <span class=\"l\" style=\"width:{}%\">← Rule behind</span><span class=\"r\">Rule ahead →</span></div></div>\
         {body}<div class=\"f-axis\"><div></div><div class=\"ticks\">{ticks}</div></div></div>",
        percent(zero)
    )
}

/// The evidence windows of one strategy.
fn strategy_windows(evidence: &EvidenceAnalysis, strategy: &str) -> Table {
    let strategies = evidence.windows.column("strategy");
    let rows: Vec<usize> = (0..strategies.len()).filter(|i| strategies[*i].as_str() == Some(strategy)).collect();
    evidence.windows.take_rows(&rows)
}

/// The large "1 of 3" summary for the stretch tile.
pub fn stretch_summary_html(evidence: &EvidenceAnalysis, strategy: &str) -> String {
    let rows = strategy_rows(evidence, strategy);
    let total = evidence.settings.get("window_count").map_or_else(|| "3".to_string(), Json::py_str);
    let ahead = evidence.summary.column("windows_ahead");
    let counts: Vec<i64> = rows.iter().map(|i| ahead[*i].as_f64().unwrap_or(f64::NAN).trunc() as i64).collect();
    let (number, label) = if counts.iter().all(|count| *count == counts[0]) {
        (format!("{} of {total}", counts[0]), "stretches ahead of each market".to_string())
    } else {
        let names: Vec<String> =
            rows.iter().map(|i| format!("the {}", market_name(&evidence.summary.index[1][*i].py_str()))).collect();
        (
            counts.iter().map(|count| format!("{count} of {total}")).collect::<Vec<_>>().join(" · "),
            format!("stretches ahead of {}", names.join(" · ")),
        )
    };
    format!(
        "<div class=\"pl-stat\"><span class=\"pl-num\">{number}</span><span class=\"pl-lbl\">{}</span></div>",
        escape(&label)
    )
}

/// A window date as "Apr 2025".
fn month_text(cell: &Cell) -> String {
    let date = cell.as_date().or_else(|| Date::parse_from_str(&cell.py_str(), "%Y-%m-%d").ok());
    date.map_or_else(|| cell.py_str(), |date| date.format("%b %Y").to_string())
}

/// Change against each market in each of the three test stretches.
pub fn stretch_table_html(evidence: &EvidenceAnalysis, strategy: &str) -> String {
    let windows = strategy_windows(evidence, strategy);
    let benchmarks = windows.column("benchmark");
    let mut markets: Vec<String> = Vec::new();
    for cell in benchmarks {
        let name = cell.py_str();
        if !markets.contains(&name) {
            markets.push(name);
        }
    }
    let rows_of = |market: &str| (0..windows.nrows()).filter(|i| benchmarks[*i].py_str() == market).collect::<Vec<_>>();
    let (starts, ends) = (windows.column("start"), windows.column("end"));
    let head: String = markets
        .first()
        .map(|first| rows_of(first))
        .unwrap_or_default()
        .iter()
        .map(|i| format!("<th>{} to {}</th>", month_text(&starts[*i]), month_text(&ends[*i])))
        .collect();
    let changes = windows.f64s("relative_return");
    let body: String = markets
        .iter()
        .map(|market| {
            let cells: String = rows_of(market)
                .iter()
                .map(|i| {
                    let change = changes[*i];
                    let outcome = if change > 0.0 {
                        "ahead"
                    } else if change < 0.0 {
                        "behind"
                    } else {
                        "level"
                    };
                    format!(
                        "<td><i class=\"w-dot{}\" aria-hidden=\"true\"></i>{}<small>{outcome}</small></td>",
                        if change > 0.0 { " on" } else { "" },
                        pct(change, 1, true)
                    )
                })
                .collect();
            format!("<tr><td>{}</td>{cells}</tr>", escape(&market_name(market)))
        })
        .collect();
    format!("<table class=\"pl-wtable\"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>")
}

// ----------------------------------------------------------------------------- fine print and work card

/// The four caveats as HTML paragraph content. `coverage_place` says where the coverage table is.
pub fn fine_print_items(metadata: &Dict, prices: &Frame, coverage_place: &str) -> Vec<String> {
    let universe = metadata.get("universe").map_or_else(String::new, Json::py_str);
    let count = prices.ncols();
    let mut items = vec![
        "<b>Research, not advice.</b> Every number comes from past prices. None of it is a forecast.".to_string(),
        format!(
            "<b>Relative risk.</b> Low, Medium and Highest compare mixes of {} only. They are not absolute risk ratings.",
            these_assets(count, &universe)
        ),
    ];
    if universe == "Nasdaq-100" {
        let value = |key: &str, default: String| metadata.get(key).map_or(default, Json::py_str);
        items.push(format!(
            "<b>Survivors only.</b> The list is today’s Nasdaq-100. {} of {} stocks were left out for missing prices. See {}.",
            escape(&value("universe_excluded", "0".into())),
            escape(&value("universe_requested", count.to_string())),
            escape(coverage_place)
        ));
    } else {
        items.push(
            "<b>Fixed list.</b> The study keeps the same assets for the whole period. That can favour survivors."
                .into(),
        );
    }
    let source = metadata.get("source").map_or_else(String::new, Json::py_str);
    let currency = if metadata.get("price_currency").is_some_and(|value| value.as_str() != Some("USD")) {
        "You declared a common price currency. Market comparisons are off, because they need USD prices. \
         No currency conversion occurs."
    } else if source.to_lowercase().contains("synthetic") || source == "Demo" {
        "The demo prices are simulated. No currency conversion occurs."
    } else {
        "You declared USD prices. The app does not check or convert currencies."
    };
    items.push(format!("<b>Currency.</b> {currency}"));
    items
}

pub fn fine_print_html(metadata: &Dict, prices: &Frame) -> String {
    let items: String =
        fine_print_items(metadata, prices, "Research › Data").iter().map(|item| format!("<p>{item}</p>")).collect();
    format!("<section class=\"pl-fine\"><h3>The fine print, in plain words</h3>{items}</section>")
}

/// The work timeline. Steps before `step` are done, `step` is running, later steps wait. Use 5 for all done.
pub fn steps_html(market: &str, step: usize, compare: bool, detail: Option<&str>) -> String {
    let second = match market {
        "Upload CSV" => "Read your price file",
        "Demo" => "Make the simulated prices",
        _ => "Download daily prices from Yahoo Finance",
    };
    let fourth = if compare && market != "Demo" {
        "Replay the test period against the S&P 500 and the Nasdaq-100"
    } else {
        "Replay the test period"
    };
    let mut labels =
        vec![(1, "Read today’s Nasdaq-100 list"), (2, second), (3, "Find the three portfolios"), (4, fourth)];
    if market != "Nasdaq-100" {
        labels.remove(0);
    }
    let items: String = labels
        .iter()
        .map(|(number, label)| {
            let state = match number.cmp(&step) {
                std::cmp::Ordering::Less => "done",
                std::cmp::Ordering::Equal => "now",
                std::cmp::Ordering::Greater => "next",
            };
            let sub = match detail {
                Some(detail) if !detail.is_empty() && *number == 2 && state == "now" => {
                    format!("<small>{}</small>", escape(detail))
                }
                _ => String::new(),
            };
            format!(
                "<li class=\"{state}\"><i aria-hidden=\"true\">{}</i><span>{}{sub}</span></li>",
                if state == "done" { "✓" } else { "" },
                escape(label)
            )
        })
        .collect();
    format!("<ol class=\"pl-steps\">{items}</ol>")
}

// ----------------------------------------------------------------------------- story chart

/// A date or text cell as a chart coordinate.
fn coordinate(cell: &Cell) -> Value {
    match cell {
        Cell::Date(date) => Value::String(iso(*date)),
        other => Value::String(other.py_str()),
    }
}

/// Shade test stretches 1 and 3 and label all three. Bands go after the base chart's own shapes.
///
/// `windows` has one row per stretch with `start` and `end` columns.
pub fn add_test_windows(figure: &mut Figure, windows: &Table) {
    let starts = windows.column("start");
    let ends = windows.column("end");
    for (position, (start, end)) in starts.iter().zip(ends).enumerate() {
        let number = position + 1;
        if number == 1 || number == 3 {
            figure.add_vrect(
                coordinate(start),
                coordinate(end),
                json!({"fillcolor": CANVAS, "opacity": 1, "layer": "below", "line": {"width": 0}}),
            );
        }
        figure.add_annotation(json!({
            "text": format!("Stretch {number}"), "xref": "x", "x": coordinate(start), "xanchor": "left", "xshift": 4,
            "yref": "paper", "y": 1, "yanchor": "top", "font": {"size": 11, "color": INK_3}, "showarrow": false,
        }));
    }
}

/// Write each line's last value at the right edge, pushed apart so the labels do not overlap.
pub fn add_end_labels(figure: &mut Figure, format: &dyn Fn(f64) -> String) {
    let numbers = |trace: &Value| -> Vec<f64> {
        trace["y"]
            .as_array()
            .map_or_else(Vec::new, |values| values.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
    };
    let lines: Vec<(Vec<f64>, String)> = figure
        .data
        .iter()
        .filter(|trace| trace["mode"] == "lines" && trace["y"].as_array().is_some_and(|values| !values.is_empty()))
        .map(|trace| (numbers(trace), trace["line"]["color"].as_str().unwrap_or("None").to_string()))
        .collect();
    if lines.is_empty() {
        return;
    }
    let values: Vec<f64> =
        lines.iter().flat_map(|(values, _)| values.iter().copied()).filter(|v| !v.is_nan()).collect();
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let nonzero = |value: f64| if value != 0.0 { Some(value) } else { None };
    let pad = nonzero((high - low) * 0.06).or_else(|| nonzero(high.abs() * 0.06)).unwrap_or(1.0);
    let (low, high) = (low - pad, high + pad);
    figure.update_yaxes(json!({"range": [low, high]}));
    let gap = (high - low) * 0.07;
    let mut labels: Vec<(f64, &String)> =
        lines.iter().map(|(values, color)| (values[values.len() - 1], color)).collect();
    labels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut previous: Option<f64> = None;
    for (value, color) in labels {
        let position = previous.map_or(value, |previous| value.max(previous + gap));
        previous = Some(position);
        figure.add_annotation(json!({
            "text": format!("<span style=\"color:{color}\">●</span> {}", format(value)), "xref": "paper", "x": 1,
            "xanchor": "left", "xshift": 6, "yref": "y", "y": position, "font": {"size": 12, "color": INK_2},
            "showarrow": false,
        }));
    }
    if let Some(previous) = previous
        && previous > high
    {
        figure.update_yaxes(json!({"range": [low, previous + pad]}));
    }
}

/// The overview market chart: growth, falls from peak or the ratio to each market, with test stretches.
///
/// `mode` is "Growth", "Drawdown" or "Relative to markets".
pub fn story_chart(
    study: &BacktestStudy,
    selected: &str,
    mode: &str,
    benchmarks: Option<&BenchmarkComparison>,
    evidence: Option<&EvidenceAnalysis>,
) -> Result<Figure> {
    let markets = matched(benchmarks);
    let relative = mode == "Relative to markets" && markets.is_some() && evidence.is_some();
    let drawdown = mode == "Drawdown";
    let mut figure = if let (true, Some(evidence)) = (relative, evidence) {
        let mut figure = evidence_chart(evidence, selected)?;
        figure.update_traces(json!({"hovertemplate": "%{fullData.name}: %{y:.2f}×<extra></extra>"}), None);
        figure
    } else {
        let strategies = [selected];
        let options = ChartOptions {
            drawdown,
            strategies: Some(&strategies),
            benchmark_equity: markets.and_then(|markets| markets.backtest_equity.as_ref()),
            ..Default::default()
        };
        let mut figure = backtest_chart(study, &options)?;
        figure.data[0]["name"] = Value::from("Your rule");
        figure.data[0]["line"] = json!({"color": ACCENT, "width": 2.5});
        if drawdown {
            crate::plot::merge(&mut figure.data[0], &json!({"fill": "tozeroy", "fillcolor": "rgba(0,113,227,.10)"}));
        }
        for trace in figure.data.iter_mut().skip(1) {
            trace["line"]["width"] = json!(1.75);
        }
        let measure = if drawdown { "%{y:.1%}" } else { "%{y:,.0f}" };
        figure.update_traces(json!({"hovertemplate": format!("%{{fullData.name}}: {measure}<extra></extra>")}), None);
        figure
    };
    style_chart(&mut figure, ChartStyle { title: None, height: 320, hovermode: "x unified", story: true });
    figure.update_xaxes(json!({"tickformat": "%b %Y", "hoverformat": "%-d %b %Y", "nticks": 6}));
    let tickformat = if relative {
        ".2f"
    } else if drawdown {
        ".0%"
    } else {
        "~s"
    };
    figure.update_yaxes(json!({"tickformat": tickformat, "ticksuffix": if relative { "×" } else { "" }}));
    if let Some(evidence) = evidence {
        let windows = strategy_windows(evidence, selected);
        if windows.nrows() > 0 {
            let first = windows.column("benchmark")[0].clone();
            let rows: Vec<usize> = (0..windows.nrows()).filter(|i| windows.column("benchmark")[*i] == first).collect();
            add_test_windows(&mut figure, &windows.take_rows(&rows));
            if drawdown {
                // Room above the 0% line for the stretch labels. It is less than one tick step, so no tick shows above 0%.
                let low = figure
                    .data
                    .iter()
                    .flat_map(|trace| trace["y"].as_array().cloned().unwrap_or_default())
                    .filter_map(|value| value.as_f64())
                    .fold(f64::INFINITY, f64::min);
                if low < 0.0 {
                    figure.update_yaxes(json!({"range": [low * 1.06, -low * 0.12]}));
                }
            }
        }
    }
    if !drawdown {
        if relative {
            add_end_labels(&mut figure, &|value| format!("{}×", pyfmt::format(value, ".2f")));
        } else {
            add_end_labels(&mut figure, &|value| pyfmt::format(value, ",.0f"));
        }
    }
    Ok(figure)
}

/// The glossary in Markdown for `k` tests, for example `glossary_md(96)`. `k = "the"` reads "all the tests".
pub fn glossary_md(k: impl std::fmt::Display) -> String {
    let word = |key: &str| lookup(WORDS, key).unwrap_or_default();
    format!(
        "### Words used here\n**95% range.** {}\n\n**Average gap.** {}\n\n**Points.** {}\n\n\
         **Adjusted p.** How surprising the gap would be if there were no real difference, after we allow for all {k} tests \
         in this report (Holm). It is not the chance that the result was luck.\n",
        word("95% range"),
        word("Average gap"),
        word("Points")
    )
}
