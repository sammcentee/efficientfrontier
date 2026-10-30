//! Plain-language answers, HTML fragments and story chart helpers shared by the app and the report.
//!
//! Port of `efficient_frontier/story.py`. Owner: reports agent.
//! The web app calls the public functions below; keep these signatures.
//!
//! Every function is pure: it reads engine results and returns text, HTML or a
//! chart change. HTML fragments use the `.pl-*` classes in the story CSS and
//! escape every data string.
//!
//! Wave 2 ports the functions that need the legacy profile, benchmark and
//! evidence types: `stats_html`, `why_text`, `levels_ledger_html` (profiles),
//! `market_answer`, `decision_panel_html`, `result_tile_html`, `chart_footnote`
//! (benchmarks), `evidence_answer`, `evidence_lead`, `clarity_text`,
//! `interval_html`, `stretch_summary_html`, `stretch_table_html` (evidence), and
//! `story_chart` (benchmarks and evidence).

use chrono::{Datelike, Months};
use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::backtest::BacktestStudy;
use crate::benchmark_report::np_sum;
use crate::frame::{Cell, Date, Frame, Series, Table, iso};
use crate::plot::Figure;
use crate::pyfmt::{self, Dict, Json, escape};
use crate::style::{CANVAS, INK_2, INK_3, display_label};

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
fn capitalize(text: &str) -> String {
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

// ----------------------------------------------------------------------------- market test

/// `numpy.round(value * 10_000, 2) != 0`: the amount shows at two decimals per 10,000.
fn visible_amount(value: f64) -> bool {
    (value * 10_000.0 * 100.0).round_ties_even() != 0.0
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
