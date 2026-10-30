//! The study setup: the drawer controls, their defaults, and the study that a setup requests.
//!
//! Port of the setup half of `app.py`, the Streamlit app of the Python release
//! (commit 6c6df6e): `MARKETS`, `FREQUENCIES`, `DEFAULT_SETUP`, `FIXED_ASSUMPTIONS`,
//! `resolve`, `differs` and `read_draft`. Everything here is pure, so the unit
//! tests below cover it without a server.

use std::collections::HashMap;

use axum::body::Bytes;
use chrono::Months;
use efficient_frontier::frame::{Date, iso};
use efficient_frontier::pyfmt::{Dict, Json};
use sha2::{Digest, Sha256};

/// The default "My tickers" text.
pub const DEFAULT_TICKERS: &str = "AAPL, MSFT, AMZN, GOOGL, NVDA, META, JPM, XOM, JNJ, PG, KO, UNH";
/// `DEFAULT_TEST["cost_bps"]`: the entry fee in basis points.
pub const COST_BPS: f64 = 10.0;
/// `FIXED_ASSUMPTIONS`: percent of returns in the initial fit, covariance shrinkage and risk-free rate.
pub const TRAIN_PCT: f64 = 70.0;
pub const SHRINK_PCT: f64 = 0.0;
pub const RISK_FREE_PCT: f64 = 2.0;
/// The lowest and highest volatility budget and holding limit, in percent.
pub const LIMIT_RANGE: (f64, f64) = (1.0, 100.0);
/// `metadata["currency_assumption"]`.
pub const CURRENCY_ASSUMPTION: &str =
    "The user declares a common price currency. No verification or currency conversion occurs.";

/// The six stock sources of the drawer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Market {
    Combined,
    Nasdaq100,
    MyTickers,
    Original60,
    UploadCsv,
    Demo,
}

impl Market {
    pub const ALL: [Market; 6] =
        [Market::Combined, Market::Nasdaq100, Market::MyTickers, Market::Original60, Market::UploadCsv, Market::Demo];

    /// The option text, which is also the form value and the `universe` metadata value.
    pub fn label(self) -> &'static str {
        match self {
            Market::Combined => "S&P 500 + Nasdaq-100",
            Market::Nasdaq100 => "Nasdaq-100",
            Market::MyTickers => "My tickers",
            Market::Original60 => "Original 60 holdings",
            Market::UploadCsv => "Upload CSV",
            Market::Demo => "Demo",
        }
    }

    /// `MARKET_CAPTIONS`.
    pub fn caption(self) -> &'static str {
        match self {
            Market::Combined => "Today’s combined members, without duplicate symbols. Prices from Yahoo Finance.",
            Market::Nasdaq100 => "Today's 101 members. Prices from Yahoo Finance.",
            Market::MyTickers => "Type Yahoo Finance symbols.",
            Market::Original60 => "Your original fixed list. Marsh uses its current MRSH symbol, formerly MMC.",
            Market::UploadCsv => "Your own adjusted prices. Stays offline.",
            Market::Demo => "Six simulated assets. Works offline.",
        }
    }

    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|market| market.label() == label)
    }

    /// A current index universe with a coverage table.
    pub fn is_universe(self) -> bool {
        matches!(self, Market::Combined | Market::Nasdaq100)
    }

    /// The markets whose prices come from Yahoo Finance and use the history control.
    pub fn uses_yahoo(self) -> bool {
        matches!(self, Market::Combined | Market::Nasdaq100 | Market::MyTickers | Market::Original60)
    }
}

/// The history presets of the Yahoo markets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum History {
    FiveYears,
    TenYears,
    Custom,
}

impl History {
    pub const ALL: [History; 3] = [History::FiveYears, History::TenYears, History::Custom];

    /// The stored value and form value.
    pub fn label(self) -> &'static str {
        match self {
            History::FiveYears => "5 years",
            History::TenYears => "10 years",
            History::Custom => "Custom dates",
        }
    }

    /// The segment text (`format_func`).
    pub fn short_label(self) -> &'static str {
        match self {
            History::Custom => "Custom",
            other => other.label(),
        }
    }

    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|history| history.label() == label)
    }

    pub fn years(self) -> Option<u32> {
        match self {
            History::FiveYears => Some(5),
            History::TenYears => Some(10),
            History::Custom => None,
        }
    }
}

/// `FREQUENCIES`: the price frequency of an uploaded CSV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frequency {
    DailyMarket,
    DailyCalendar,
    Weekly,
    Monthly,
    Custom,
}

impl Frequency {
    pub const ALL: [Frequency; 5] =
        [Frequency::DailyMarket, Frequency::DailyCalendar, Frequency::Weekly, Frequency::Monthly, Frequency::Custom];

    pub fn label(self) -> &'static str {
        match self {
            Frequency::DailyMarket => "Daily market prices",
            Frequency::DailyCalendar => "Daily calendar prices",
            Frequency::Weekly => "Weekly",
            Frequency::Monthly => "Monthly",
            Frequency::Custom => "Custom",
        }
    }

    /// Observations per year, or `None` for a custom value.
    pub fn periods(self) -> Option<i64> {
        match self {
            Frequency::DailyMarket => Some(252),
            Frequency::DailyCalendar => Some(365),
            Frequency::Weekly => Some(52),
            Frequency::Monthly => Some(12),
            Frequency::Custom => None,
        }
    }

    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|frequency| frequency.label() == label)
    }
}

/// The declared price currency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Currency {
    Usd,
    Other,
}

impl Currency {
    pub const ALL: [Currency; 2] = [Currency::Usd, Currency::Other];

    pub fn label(self) -> &'static str {
        match self {
            Currency::Usd => "USD",
            Currency::Other => "Other currency",
        }
    }

    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|currency| currency.label() == label)
    }
}

/// An uploaded CSV file: its name, its bytes and their SHA-256 digest.
///
/// The bytes share the request's buffer, so a stored file is never copied.
#[derive(Clone, Debug)]
pub struct Upload {
    pub name: String,
    pub bytes: Bytes,
    pub sha256: String,
}

impl Upload {
    /// A file with its digest. Hashing a large file takes time: call this off the async workers.
    pub fn new(name: impl Into<String>, bytes: impl Into<Bytes>) -> Self {
        let bytes = bytes.into();
        let sha256 = hex(&Sha256::digest(&bytes));
        Self { name: name.into(), bytes, sha256 }
    }

    /// The identity of a file that the browser hashed, without its bytes. Only a
    /// draft comparison uses it; a build always receives the file itself.
    pub fn identity(name: impl Into<String>, sha256: &str) -> Option<Self> {
        let valid = sha256.len() == 64 && sha256.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
        valid.then(|| Self { name: name.into(), bytes: Bytes::new(), sha256: sha256.to_string() })
    }
}

impl PartialEq for Upload {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.sha256 == other.sha256
    }
}

/// Lowercase hexadecimal text of bytes.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One study setup (`DEFAULT_SETUP`). The fixed assumptions are constants, so they are not fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Setup {
    pub market: Market,
    pub history: History,
    pub ticker_text: String,
    pub start: Date,
    /// Exclusive end of a custom Yahoo request.
    pub end: Date,
    /// Holding limit, percent.
    pub cap: f64,
    /// Annual volatility budget, percent.
    pub target_vol: f64,
    pub frequency: Frequency,
    pub periods_per_year: f64,
    pub price_currency: Currency,
    pub compare_market: bool,
    pub download_benchmarks: bool,
    pub prices_file: Option<Upload>,
    pub benchmark_file: Option<Upload>,
}

impl Setup {
    /// `DEFAULT_SETUP` on the given day.
    pub fn new(today: Date) -> Self {
        Self {
            market: Market::Combined,
            history: History::FiveYears,
            ticker_text: DEFAULT_TICKERS.to_string(),
            start: Date::from_ymd_opt(2020, 1, 1).expect("valid date"),
            end: today,
            cap: 10.0,
            target_vol: 20.0,
            frequency: Frequency::DailyMarket,
            periods_per_year: 252.0,
            price_currency: Currency::Usd,
            compare_market: true,
            download_benchmarks: false,
            prices_file: None,
            benchmark_file: None,
        }
    }
}

/// Where the prices of a study come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Demo,
    UploadCsv,
    Yahoo,
}

impl Source {
    /// The `source` metadata value.
    pub fn label(self) -> &'static str {
        match self {
            Source::Demo => "Demo · synthetic",
            Source::UploadCsv => "Upload CSV",
            Source::Yahoo => "Yahoo Finance",
        }
    }
}

/// The model settings that a study passes to `run_selection` and records in its metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSettings {
    pub train_fraction: f64,
    pub risk_free_rate: f64,
    pub max_weight: f64,
    pub shrinkage: f64,
    pub periods_per_year: f64,
    /// A CSV frequency preset is an integer in Python, so the metadata keeps it as one.
    pub periods_integer: bool,
}

impl ModelSettings {
    /// The settings in metadata order, as Python values.
    pub fn entries(&self) -> [(&'static str, Json); 5] {
        let periods = if self.periods_integer {
            Json::Int(self.periods_per_year as i64)
        } else {
            Json::Float(self.periods_per_year)
        };
        [
            ("train_fraction", Json::Float(self.train_fraction)),
            ("risk_free_rate", Json::Float(self.risk_free_rate)),
            ("max_weight", Json::Float(self.max_weight)),
            ("shrinkage", Json::Float(self.shrinkage)),
            ("periods_per_year", periods),
        ]
    }
}

/// The inputs that identify an applied result besides its metadata: tickers, the
/// download choice and each file as (name, SHA-256).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputChoices {
    pub tickers: String,
    pub download_benchmarks: bool,
    pub prices: Option<(String, String)>,
    pub benchmarks: Option<(String, String)>,
}

/// The source, dates, settings and input choices that a setup requests (`resolve`).
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub market: Market,
    pub source: Source,
    pub start: Date,
    /// Exclusive end date.
    pub end: Date,
    pub settings: ModelSettings,
    pub price_currency: Currency,
    pub compare_market: bool,
    /// A fraction, not a percent.
    pub target_volatility: f64,
    pub download_for_csv: bool,
    pub ticker_text: String,
    pub prices_file: Option<Upload>,
    pub benchmark_file: Option<Upload>,
    pub input_choices: InputChoices,
}

/// `pd.Timestamp(end) - pd.DateOffset(years=years)`: the same day, or the month end after February 29.
pub fn years_before(end: Date, years: u32) -> Date {
    end.checked_sub_months(Months::new(12 * years)).expect("date in range")
}

/// The study that a setup requests on `today`.
pub fn resolve(setup: &Setup, today: Date) -> Plan {
    let market = setup.market;
    let source = match market {
        Market::Demo => Source::Demo,
        Market::UploadCsv => Source::UploadCsv,
        _ => Source::Yahoo,
    };
    let csv = source == Source::UploadCsv;
    let (mut start, mut end) = (Date::from_ymd_opt(2020, 1, 1).expect("valid date"), today);
    if source == Source::Yahoo {
        match setup.history.years() {
            Some(years) => start = years_before(end, years),
            None => (start, end) = (setup.start, setup.end),
        }
    }
    let (periods_per_year, periods_integer) = match (csv, setup.frequency.periods()) {
        (true, Some(periods)) => (periods as f64, true),
        (true, None) => (setup.periods_per_year, false),
        (false, _) => (252.0, false),
    };
    let settings = ModelSettings {
        train_fraction: TRAIN_PCT / 100.0,
        risk_free_rate: RISK_FREE_PCT / 100.0,
        max_weight: setup.cap / 100.0,
        shrinkage: SHRINK_PCT / 100.0,
        periods_per_year,
        periods_integer,
    };
    let ticker_text = if market == Market::MyTickers { setup.ticker_text.clone() } else { String::new() };
    let prices_file = if csv { setup.prices_file.clone() } else { None };
    let benchmark_file = if csv { setup.benchmark_file.clone() } else { None };
    let download_for_csv = csv && setup.download_benchmarks;
    let identity = |upload: &Option<Upload>| upload.as_ref().map(|file| (file.name.clone(), file.sha256.clone()));
    let input_choices = InputChoices {
        tickers: ticker_text.trim().to_string(),
        download_benchmarks: download_for_csv,
        prices: identity(&prices_file),
        benchmarks: identity(&benchmark_file),
    };
    Plan {
        market,
        source,
        start,
        end,
        settings,
        price_currency: setup.price_currency,
        compare_market: setup.compare_market,
        target_volatility: setup.target_vol / 100.0,
        download_for_csv,
        ticker_text,
        prices_file,
        benchmark_file,
        input_choices,
    }
}

/// True when a resolved setup would not reproduce the applied result.
pub fn differs(plan: &Plan, metadata: &Dict, applied_inputs: &InputChoices) -> bool {
    let text = |key: &str| metadata.get(key).and_then(Json::as_str);
    let number = |key: &str| metadata.get(key).and_then(Json::as_f64);
    text("universe") != Some(plan.market.label())
        || applied_inputs != &plan.input_choices
        || plan.settings.entries().iter().any(|(key, value)| number(key) != value.as_f64())
        || number("target_volatility") != Some(plan.target_volatility)
        || text("price_currency") != Some(plan.price_currency.label())
        || metadata.get("compare_market") != Some(&Json::Bool(plan.compare_market))
        || plan.source == Source::Yahoo
            && (text("requested_start") != Some(iso(plan.start).as_str())
                || text("requested_end_exclusive") != Some(iso(plan.end).as_str()))
}

/// The fields and files of a posted drawer form. A missing field is a hidden
/// control, which keeps the stored value, as a Streamlit widget that did not render.
#[derive(Clone, Debug, Default)]
pub struct Draft {
    pub fields: HashMap<String, String>,
    pub uploads: HashMap<String, Upload>,
}

impl Draft {
    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }

    /// A percent limit in `LIMIT_RANGE`, or `None` for a missing or invalid value.
    pub fn limit(&self, name: &str) -> Option<f64> {
        let value: f64 = self.field(name)?.trim().parse().ok()?;
        (value.is_finite() && (LIMIT_RANGE.0..=LIMIT_RANGE.1).contains(&value)).then_some(value)
    }

    fn date(&self, name: &str) -> Option<Date> {
        Date::parse_from_str(self.field(name)?.trim(), "%Y-%m-%d").ok()
    }

    fn flag(&self, name: &str) -> Option<bool> {
        match self.field(name)? {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }
}

/// A new setup from the drawer (`read_draft`). Files are a new upload, else the stored file unless removed.
pub fn read_draft(draft: &Draft, setup: &Setup) -> Setup {
    let mut next = setup.clone();
    if let Some(market) = draft.field("market").and_then(Market::parse) {
        next.market = market;
    }
    if let Some(history) = draft.field("history").and_then(History::parse) {
        next.history = history;
    }
    if let Some(text) = draft.field("ticker_text") {
        next.ticker_text = text.to_string();
    }
    if let Some(start) = draft.date("start") {
        next.start = start;
    }
    if let Some(end) = draft.date("end") {
        next.end = end;
    }
    if let Some(cap) = draft.limit("cap") {
        next.cap = cap;
    }
    if let Some(target) = draft.limit("target_vol") {
        next.target_vol = target;
    }
    if let Some(frequency) = draft.field("frequency").and_then(Frequency::parse) {
        next.frequency = frequency;
    }
    if let Some(periods) = draft.field("periods_per_year").and_then(|text| text.trim().parse::<f64>().ok())
        && periods.is_finite()
        && periods >= 1.0
    {
        next.periods_per_year = periods;
    }
    if let Some(currency) = draft.field("price_currency").and_then(Currency::parse) {
        next.price_currency = currency;
    }
    if let Some(compare) = draft.flag("compare_market") {
        next.compare_market = compare;
    }
    if let Some(download) = draft.flag("download_benchmarks") {
        next.download_benchmarks = download;
    }
    for (slot, upload, clear) in [
        (&mut next.prices_file, "prices_upload", "clear_prices_file"),
        (&mut next.benchmark_file, "benchmark_upload", "clear_benchmark_file"),
    ] {
        if let Some(file) = draft.uploads.get(upload) {
            *slot = Some(file.clone());
        } else if draft.field(clear) == Some("1") {
            *slot = None;
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use efficient_frontier::dict;

    fn day(text: &str) -> Date {
        Date::parse_from_str(text, "%Y-%m-%d").unwrap()
    }

    fn applied_metadata(plan: &Plan) -> Dict {
        let mut metadata = dict! {
            "source" => plan.source.label(), "universe" => plan.market.label(),
            "price_currency" => plan.price_currency.label(), "compare_market" => plan.compare_market,
            "target_volatility" => plan.target_volatility,
        };
        for (key, value) in plan.settings.entries() {
            metadata.insert(key, value);
        }
        if plan.source == Source::Yahoo {
            metadata.insert("requested_start", iso(plan.start));
            metadata.insert("requested_end_exclusive", iso(plan.end));
        }
        metadata
    }

    #[test]
    fn presets_subtract_calendar_years_and_custom_dates_pass_through() {
        let today = day("2026-10-30");
        let mut setup = Setup::new(today);
        let plan = resolve(&setup, today);
        assert_eq!((plan.source, plan.start, plan.end), (Source::Yahoo, day("2021-10-30"), today));
        assert_eq!(plan.settings.periods_per_year, 252.0);
        assert_eq!(
            (plan.settings.train_fraction, plan.settings.risk_free_rate, plan.settings.shrinkage),
            (0.7, 0.02, 0.0)
        );
        assert_eq!(plan.settings.max_weight, 0.1);
        assert_eq!(plan.target_volatility, 0.2);
        setup.history = History::TenYears;
        assert_eq!(resolve(&setup, today).start, day("2016-10-30"));
        assert_eq!(years_before(day("2024-02-29"), 1), day("2023-02-28"));
        setup.history = History::Custom;
        setup.end = day("2024-01-01");
        let plan = resolve(&setup, today);
        assert_eq!((plan.start, plan.end), (day("2020-01-01"), day("2024-01-01")));
        assert_eq!(plan.input_choices.tickers, "", "only My tickers sends its text");
    }

    #[test]
    fn csv_frequency_and_files_reach_the_plan_only_for_upload_csv() {
        let today = day("2026-10-30");
        let mut setup = Setup::new(today);
        setup.prices_file = Some(Upload::new("portfolio.csv", b"Date,A\n".as_slice()));
        setup.download_benchmarks = true;
        setup.frequency = Frequency::Monthly;
        let plan = resolve(&setup, today);
        assert_eq!(
            (plan.prices_file.clone(), plan.download_for_csv, plan.settings.periods_per_year),
            (None, false, 252.0)
        );
        setup.market = Market::UploadCsv;
        let plan = resolve(&setup, today);
        assert_eq!(plan.source, Source::UploadCsv);
        assert_eq!((plan.settings.periods_per_year, plan.settings.periods_integer), (12.0, true));
        assert_eq!(plan.settings.entries()[4].1, Json::Int(12));
        assert!(plan.download_for_csv);
        let (name, digest) = plan.input_choices.prices.clone().unwrap();
        assert_eq!(name, "portfolio.csv");
        assert_eq!(digest, hex(&Sha256::digest(b"Date,A\n")));
        setup.frequency = Frequency::Custom;
        setup.periods_per_year = 48.5;
        assert_eq!(resolve(&setup, today).settings.entries()[4].1, Json::Float(48.5));
        assert_eq!(resolve(&setup, today).start, day("2020-01-01"));
    }

    #[test]
    fn differs_detects_every_applied_input() {
        let today = day("2026-10-30");
        let setup = Setup::new(today);
        let plan = resolve(&setup, today);
        let metadata = applied_metadata(&plan);
        let inputs = plan.input_choices.clone();
        assert!(!differs(&plan, &metadata, &inputs));
        let changed = |edit: &dyn Fn(&mut Setup)| {
            let mut draft = setup.clone();
            edit(&mut draft);
            differs(&resolve(&draft, today), &metadata, &inputs)
        };
        assert!(changed(&|s| s.target_vol = 25.0));
        assert!(changed(&|s| s.cap = 30.0));
        assert!(changed(&|s| s.market = Market::Nasdaq100));
        assert!(changed(&|s| s.history = History::TenYears));
        assert!(changed(&|s| s.price_currency = Currency::Other));
        assert!(changed(&|s| s.compare_market = false));
        assert!(!changed(&|s| s.ticker_text = "MSFT".into()), "tickers matter only for My tickers");
        assert!(!changed(&|s| s.frequency = Frequency::Weekly), "frequency matters only for CSV");
        // The next day moves a preset history window.
        assert!(differs(&resolve(&setup, day("2026-10-31")), &metadata, &inputs));
    }

    #[test]
    fn same_file_name_with_new_bytes_differs() {
        let today = day("2026-10-30");
        let mut setup = Setup::new(today);
        setup.market = Market::UploadCsv;
        setup.prices_file = Some(Upload::new("portfolio.csv", b"one".as_slice()));
        let plan = resolve(&setup, today);
        let metadata = applied_metadata(&plan);
        setup.prices_file = Some(Upload::new("portfolio.csv", b"two".as_slice()));
        assert!(differs(&resolve(&setup, today), &metadata, &plan.input_choices));
    }

    #[test]
    fn draft_keeps_hidden_controls_and_files_unless_removed() {
        let today = day("2026-10-30");
        let mut setup = Setup::new(today);
        setup.prices_file = Some(Upload::new("portfolio.csv", b"prices".as_slice()));
        setup.benchmark_file = Some(Upload::new("benchmarks.csv", b"spy".as_slice()));
        let mut draft = Draft::default();
        for (key, value) in
            [("market", "Upload CSV"), ("cap", "30"), ("target_vol", "250"), ("clear_benchmark_file", "1")]
        {
            draft.fields.insert(key.into(), value.into());
        }
        let next = read_draft(&draft, &setup);
        assert_eq!(next.market, Market::UploadCsv);
        assert_eq!(next.cap, 30.0);
        assert_eq!(next.target_vol, 20.0, "an out-of-range budget keeps the stored value");
        assert_eq!(next.ticker_text, DEFAULT_TICKERS);
        assert_eq!(next.prices_file, setup.prices_file);
        assert_eq!(next.benchmark_file, None);
        draft.uploads.insert("prices_upload".into(), Upload::new("new.csv", b"new".as_slice()));
        draft.fields.insert("clear_prices_file".into(), "1".into());
        assert_eq!(read_draft(&draft, &setup).prices_file.unwrap().name, "new.csv", "a new upload wins over removal");
        draft.fields.insert("compare_market".into(), "false".into());
        draft.fields.insert("periods_per_year".into(), "0.5".into());
        let next = read_draft(&draft, &setup);
        assert!(!next.compare_market);
        assert_eq!(next.periods_per_year, 252.0, "observations per year must be at least one");
    }
}
