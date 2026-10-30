//! Plain-language story components: labels, cards, context lines and charts.
//!
//! Port of `efficient_frontier/story.py`. Owner: reports agent.
//! The web app calls the public functions below; keep these signatures.

use indexmap::IndexMap;

use crate::backtest::BacktestStudy;
use crate::frame::{Date, Series, Table};
use crate::pyfmt::Json;

pub const VIEW_LABELS: [(&str, &str); 2] = [("Portfolio", "Overview"), ("Research", "Research")];

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

pub fn setting_label(key: &str) -> String {
    let _ = key;
    todo!("story")
}

/// A study setting as display text. Fractions show as percents.
pub fn setting_text(key: &str, value: &Json) -> String {
    let _ = (key, value);
    todo!("story")
}

/// Map each symbol to its short company name from a coverage table with `symbol` and `name` columns.
pub fn company_names(coverage: Option<&Table>) -> IndexMap<String, String> {
    let _ = coverage;
    todo!("story")
}

pub fn holdings_cards_html(weights: &Series, companies: &IndexMap<String, Company>) -> String {
    let _ = (weights, companies);
    todo!("story")
}

/// The context line before a result, for example "Nasdaq-100 · 5 years".
pub fn context_setup_html(setup: &SetupContext, badge: Option<&str>) -> String {
    let _ = (setup, badge);
    todo!("story")
}

pub fn result_drivers_html(study: &BacktestStudy, selected: &str) -> String {
    let _ = (study, selected);
    todo!("story")
}
