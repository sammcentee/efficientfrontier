//! One study build in the background: the port of `build`, `error_kind` and the work card.
//!
//! A build runs on its own thread. The session keeps the [`Job`]; the work card
//! polls its [`Progress`]. Cancel drops the job from the session, so its result
//! is discarded, and the job stops at its next step: a universe download stops
//! at its next batch of symbols. The applied result changes only when a build
//! succeeds; a failure clears it and keeps the setup, as in `app.py`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use efficient_frontier::data::{demo_prices, load_csv, parse_tickers};
use efficient_frontier::error::{Error, Result};
use efficient_frontier::frame::{Frame, Series, Table, iso};
use efficient_frontier::out_of_sample::{OutOfSampleStudy, run_default_out_of_sample};
use efficient_frontier::pyfmt::{Dict, Json};
use efficient_frontier::selection::{
    MODEL_REVISION, SelectionFrontier, SelectionOptions, SelectionStudy, run_selection, selection_frontier,
    split_stock_prices, validate_selection_benchmarks,
};
use efficient_frontier::{dict, value_error};
use indexmap::IndexMap;

use crate::cache::{dates_key, selection_key};
use crate::guard::guard;
use crate::session::{Session, SessionRef, lock};
use crate::setup::{COST_BPS, CURRENCY_ASSUMPTION, Currency, InputChoices, Market, Plan, Source, resolve};
use crate::sources::{Cancelled, CompanyProfile, Sources, UniversePrices};
use crate::state::AppState;

/// The work card steps; `mark(n)` shows step `n`.
pub const STEPS: [&str; 4] = [
    "Read the selected universe",
    "Load the selected prices",
    "Prepare the risk budget",
    "Fit one stock model, then replay buy and hold",
];
/// Points on the latest fitted frontier (`selection_frontier`'s default).
pub const FRONTIER_POINTS: usize = 30;
/// How often a wait for company profiles looks for a cancelled build.
const CANCEL_POLL: Duration = Duration::from_millis(100);
/// Weights at or below this level are not holdings.
pub const HELD: f64 = 1e-6;

/// What the work card shows.
#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    /// One of [`STEPS`].
    pub step: &'static str,
    /// The universe download bar: fraction done and its message.
    pub bar: Option<(f64, String)>,
    /// A spinner message for the work after the fit.
    pub detail: Option<&'static str>,
}

/// One running build.
#[derive(Debug)]
pub struct Job {
    pub id: u64,
    cancelled: AtomicBool,
    progress: Mutex<Progress>,
}

impl Job {
    pub fn new() -> Arc<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Arc::new(Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            cancelled: AtomicBool::new(false),
            progress: Mutex::new(Progress { step: STEPS[0], bar: None, detail: None }),
        })
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub fn progress(&self) -> Progress {
        lock(&self.progress).clone()
    }

    /// Show step `number` (1 to 4).
    fn mark(&self, number: usize) {
        lock(&self.progress).step = STEPS[number - 1];
    }

    fn bar(&self, fraction: f64, message: impl Into<String>) {
        lock(&self.progress).bar = Some((fraction.clamp(0.0, 1.0), message.into()));
    }

    fn detail(&self, message: &'static str) {
        lock(&self.progress).detail = Some(message);
    }
}

/// A successful study and everything its pages and export show.
#[derive(Debug)]
pub struct Applied {
    /// Identifies this result in the URLs of its downloads, so a page of an older
    /// study cannot download a newer one by mistake.
    pub id: u64,
    /// Stock prices, without comparison ETF columns.
    pub prices: Arc<Frame>,
    pub metadata: Dict,
    pub selection: Arc<SelectionStudy>,
    /// `calculate_frontier`: the latest fitted frontier, or the reason it failed.
    pub frontier: std::result::Result<Arc<SelectionFrontier>, String>,
    /// `calculate_out_of_sample`: the walk-forward tests, or the reason they are unavailable.
    pub out_of_sample: std::result::Result<Arc<OutOfSampleStudy>, String>,
    /// Current company profiles of the shown holdings, for Yahoo studies only.
    pub profiles: Arc<IndexMap<String, CompanyProfile>>,
    /// `applied_inputs`.
    pub inputs: InputChoices,
    /// The comparison prices that the study used.
    pub benchmark_prices: Option<Arc<Frame>>,
}

impl Applied {
    /// A new result identifier.
    pub fn next_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    /// The result uses the current selection model (`current_selection`).
    pub fn is_current(&self) -> bool {
        self.metadata.get("selection_model_revision").and_then(Json::as_str) == Some(MODEL_REVISION)
    }
}

/// The failure classes of the error page. A retry helps only `Yahoo` and `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Yahoo,
    NoPrices,
    Infeasible,
    NoFile,
    NoStocks,
    Other,
}

impl ErrorKind {
    pub fn headline(self) -> &'static str {
        match self {
            ErrorKind::Yahoo => "We could not reach Yahoo Finance.",
            ErrorKind::NoPrices => "Yahoo Finance had no prices for these symbols.",
            ErrorKind::Infeasible => "These settings have no solution.",
            ErrorKind::NoFile => "The study needs a price file.",
            ErrorKind::NoStocks => "The portfolio needs stock prices.",
            ErrorKind::Other => "The study could not run.",
        }
    }

    /// A retry would fail the same way, so the primary action opens the drawer.
    pub fn needs_change(self) -> bool {
        matches!(self, ErrorKind::NoPrices | ErrorKind::Infeasible | ErrorKind::NoFile | ErrorKind::NoStocks)
    }
}

/// Name a failure (`error_kind`). `network` marks an I/O error, Python's `OSError`.
pub fn error_kind(message: &str, network: bool) -> ErrorKind {
    let lower = message.to_lowercase();
    if network || message.starts_with("Yahoo price download failed") {
        ErrorKind::Yahoo
    } else if message.starts_with("Yahoo returned no prices") {
        ErrorKind::NoPrices
    } else if ["infeasible", "risk budget", "below the feasible minimum"].iter().any(|phrase| lower.contains(phrase)) {
        ErrorKind::Infeasible
    } else if message == "Choose an adjusted-price CSV first." {
        ErrorKind::NoFile
    } else if message.starts_with("Provide stock price columns") {
        ErrorKind::NoStocks
    } else {
        ErrorKind::Other
    }
}

/// `build_error` and `build_error_kind`.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildError {
    pub message: String,
    pub kind: ErrorKind,
}

impl BuildError {
    fn from_error(error: &Error) -> Self {
        Self { message: error.message().to_string(), kind: error_kind(error.message(), error.is_io()) }
    }
}

enum Outcome {
    Applied(Box<Applied>, Option<Arc<Table>>),
    Failed(BuildError, Option<Arc<Table>>),
    Cancelled,
}

/// Start a build of the session's setup (`start_build`). A running build is cancelled.
pub fn start(state: &Arc<AppState>, handle: &SessionRef, session: &mut Session) {
    let plan = resolve(&session.setup, state.today());
    if let Some(previous) = session.job.take() {
        previous.cancel();
    }
    session.build_error = None;
    let job = Job::new();
    session.job = Some(job.clone());
    let (state, handle) = (state.clone(), handle.clone());
    let spawned = std::thread::Builder::new().name("portfolio-build".into()).spawn(move || {
        let outcome = guard(|| run(&state, &plan, &job))
            .unwrap_or_else(|message| Outcome::Failed(BuildError { message, kind: ErrorKind::Other }, None));
        finish(&handle, &job, outcome);
    });
    if let Err(error) = spawned {
        // As any failed build: clear the result and show why.
        session.job = None;
        session.applied = None;
        session.build_error =
            Some(BuildError { message: format!("The build could not start: {error}"), kind: ErrorKind::Other });
    }
}

/// Apply a finished build, unless the session cancelled or replaced it.
fn finish(handle: &SessionRef, job: &Job, outcome: Outcome) {
    let mut session = lock(handle);
    if session.job.as_ref().is_none_or(|current| current.id != job.id) {
        return;
    }
    session.job = None;
    match outcome {
        Outcome::Applied(applied, coverage) => {
            session.applied = Some(Arc::from(applied));
            session.coverage = coverage;
            session.build_error = None;
        }
        Outcome::Failed(error, coverage) => {
            session.applied = None;
            session.coverage = coverage;
            session.build_error = Some(error);
        }
        Outcome::Cancelled => {}
    }
}

/// The prices and study of a successful fit.
struct Fitted {
    prices: Frame,
    benchmark_prices: Option<Frame>,
    study: Arc<SelectionStudy>,
    key: crate::cache::Key,
}

/// Load the inputs and build one selection study, then the views that its pages need.
fn run(state: &AppState, plan: &Plan, job: &Job) -> Outcome {
    let mut metadata = dict! {
        "source" => plan.source.label(),
        "universe" => plan.market.label(),
        "price_currency" => plan.price_currency.label(),
        "compare_market" => plan.compare_market,
        "target_volatility" => plan.target_volatility,
    };
    for (key, value) in plan.settings.entries() {
        metadata.insert(key, value);
    }
    metadata.insert("currency_assumption", CURRENCY_ASSUMPTION);
    let mut loaded = None;
    let fitted = match fit(state, plan, job, &mut metadata, &mut loaded) {
        Ok(Some(fitted)) => fitted,
        Ok(None) => return Outcome::Cancelled,
        Err(_) if job.is_cancelled() => return Outcome::Cancelled,
        Err(error) => {
            let coverage = loaded.map(|universe: Arc<UniversePrices>| Arc::new(universe.coverage.clone()));
            return Outcome::Failed(BuildError::from_error(&error), coverage);
        }
    };
    metadata.insert("selection_model_revision", MODEL_REVISION);
    let Fitted { prices, benchmark_prices, study, key } = fitted;
    if job.is_cancelled() {
        return Outcome::Cancelled;
    }
    // The overview computes these while it renders; here they are ready before it opens.
    let frontier = guard(|| {
        state.caches.frontier.get_or_try(key, || selection_frontier(&study, &prices, FRONTIER_POINTS).map(Arc::new))
    })
    .and_then(|result| result.map_err(|error| error.message().to_string()));
    if job.is_cancelled() {
        return Outcome::Cancelled;
    }
    job.detail("Run the out-of-sample tests");
    let out_of_sample = guard(|| {
        state.caches.out_of_sample.get_or_try(key, || run_default_out_of_sample(&study, &prices).map(Arc::new))
    })
    .and_then(|result| result.map_err(|error| error.message().to_string()));
    if job.is_cancelled() {
        return Outcome::Cancelled;
    }
    let profiles = if plan.source == Source::Yahoo {
        job.detail("Load company descriptions");
        let symbols = profile_symbols(&study);
        let cancelled = || job.is_cancelled();
        let fetched = state.caches.profiles.get_or_try(symbols.clone(), || {
            fetch_company_profiles(&state.sources, &symbols, state.profile_timeout, &cancelled).map(Arc::new).ok_or(())
        });
        match fetched {
            Ok(profiles) => profiles,
            Err(()) => return Outcome::Cancelled,
        }
    } else {
        Arc::new(IndexMap::new())
    };
    let coverage = loaded.map(|universe| Arc::new(universe.coverage.clone()));
    let applied = Applied {
        id: Applied::next_id(),
        prices: Arc::new(prices),
        metadata,
        selection: study,
        frontier,
        out_of_sample,
        profiles,
        inputs: plan.input_choices.clone(),
        benchmark_prices: benchmark_prices.map(Arc::new),
    };
    Outcome::Applied(Box::new(applied), coverage)
}

/// The steps of `build` up to the selection. `Ok(None)` means the build was cancelled.
fn fit(
    state: &AppState,
    plan: &Plan,
    job: &Job,
    metadata: &mut Dict,
    loaded: &mut Option<Arc<UniversePrices>>,
) -> Result<Option<Fitted>> {
    let mut universe_benchmarks = None;
    let prices = match plan.source {
        Source::Yahoo => {
            metadata.insert("requested_start", iso(plan.start));
            metadata.insert("requested_end_exclusive", iso(plan.end));
            if plan.market.is_universe() {
                job.mark(2);
                let universe = fetch_universe(state, plan, job)?;
                *loaded = Some(universe.clone());
                metadata.update(&universe.metadata);
                metadata.insert("universe", plan.market.label());
                universe_benchmarks = universe.benchmarks.clone();
                if universe.prices.ncols() < 2 {
                    return Err(value_error!(
                        "Fewer than two selected securities have complete prices. Review coverage below or choose a shorter history."
                    ));
                }
                universe.prices.clone()
            } else {
                let tickers = if plan.market == Market::Original60 {
                    state.sources.original_tickers()?
                } else {
                    parse_tickers(&plan.ticker_text)
                };
                let key = (tickers.clone(), plan.start, plan.end);
                let frame = state
                    .caches
                    .prices
                    .get_or_try(key, || state.sources.prices(&tickers, plan.start, plan.end).map(Arc::new))?;
                frame.as_ref().clone()
            }
        }
        Source::Demo => demo_prices(),
        Source::UploadCsv => {
            let Some(file) = &plan.prices_file else {
                return Err(value_error!("Choose an adjusted-price CSV first."));
            };
            let prices = load_csv(&file.bytes)?;
            metadata.insert("input_file", file.name.clone());
            prices
        }
    };
    if job.is_cancelled() {
        return Ok(None);
    }
    job.mark(3);
    job.mark(4);
    let (prices, embedded) = split_stock_prices(&prices)?;
    if let Some(embedded) = &embedded {
        metadata.insert("excluded_candidate_symbols", Json::from(embedded.columns.clone()));
        metadata.insert(
            "candidate_note",
            format!("Comparison ETFs excluded from candidate holdings: {}.", embedded.columns.join(", ")),
        );
    }
    let benchmark_prices = comparison_prices(state, plan, metadata, &prices, embedded, universe_benchmarks);
    if job.is_cancelled() {
        return Ok(None);
    }
    let key = selection_key(&prices, benchmark_prices.as_ref(), &plan.settings, plan.target_volatility, MODEL_REVISION);
    let options = SelectionOptions {
        target_volatility: plan.target_volatility,
        max_weight: plan.settings.max_weight,
        train_fraction: plan.settings.train_fraction,
        cost_bps: COST_BPS,
        shrinkage: plan.settings.shrinkage,
        periods_per_year: plan.settings.periods_per_year,
        risk_free_rate: plan.settings.risk_free_rate,
    };
    let study = state
        .caches
        .selection
        .get_or_try(key, || run_selection(&prices, benchmark_prices.as_ref(), &options).map(Arc::new))?;
    Ok(Some(Fitted { prices, benchmark_prices, study, key }))
}

/// `fetch_universe`: the cached universe download, with the work card's progress bar.
fn fetch_universe(state: &AppState, plan: &Plan, job: &Job) -> Result<Arc<UniversePrices>> {
    state.caches.universe.get_or_try((plan.market, plan.start, plan.end), || {
        job.bar(0.0, format!("Read the current {} list", plan.market.label()));
        let mut progress = |done: usize, total: usize, message: &str| {
            job.bar(done as f64 / total.max(1) as f64, message);
        };
        let cancelled = || job.is_cancelled();
        state.sources.universe_prices(plan.market, plan.start, plan.end, &mut progress, &cancelled).map(Arc::new)
    })
}

/// The comparison prices of a study, or none with a note that says why.
///
/// The precedence is: an uploaded market file, SPY/QQQ columns in the price
/// input, an explicit Yahoo download for a CSV, the universe download's ETFs,
/// then a Yahoo download for Yahoo studies. A comparison failure never stops
/// the study; it becomes `benchmarks_note`.
fn comparison_prices(
    state: &AppState,
    plan: &Plan,
    metadata: &mut Dict,
    prices: &Frame,
    embedded: Option<Frame>,
    universe: Option<Frame>,
) -> Option<Frame> {
    let note = |metadata: &mut Dict, text: &str| metadata.insert("benchmarks_note", text);
    if plan.market == Market::Demo {
        note(metadata, "Synthetic prices have no real-market benchmark comparison.");
        return None;
    }
    if !plan.compare_market {
        note(metadata, "Market comparisons are off. To turn them on, select Change, then Data options.");
        return None;
    }
    if plan.price_currency != Currency::Usd {
        note(metadata, "Market comparisons require USD prices. No currency conversion occurs.");
        return None;
    }
    let choose = || -> Result<Option<Frame>> {
        let download = || -> Result<Frame> {
            let frame = state
                .caches
                .benchmarks
                .get_or_try(dates_key(&prices.index), || state.sources.benchmarks(&prices.index).map(Arc::new))?;
            Ok(frame.as_ref().clone())
        };
        let candidate = if let Some(file) = &plan.benchmark_file {
            let frame = load_csv(&file.bytes)?;
            metadata.insert("benchmark_source", format!("CSV: {}", file.name));
            Some(frame)
        } else if let Some(frame) = embedded {
            metadata.insert("benchmark_source", format!("Input columns: {}", frame.columns.join(", ")));
            Some(frame)
        } else if plan.download_for_csv {
            let frame = download()?;
            metadata.insert("benchmark_source", "Yahoo Finance");
            Some(frame)
        } else if let Some(frame) = universe {
            metadata.insert("benchmark_source", "Yahoo Finance");
            Some(frame)
        } else if plan.source == Source::Yahoo {
            let frame = download()?;
            metadata.insert("benchmark_source", "Yahoo Finance");
            Some(frame)
        } else {
            note(
                metadata,
                "Add a market prices file, include SPY and QQQ columns, or turn on the Yahoo download. \
                 Select Change, then Data options.",
            );
            None
        };
        candidate.map(|frame| validate_selection_benchmarks(&frame, &prices.index)).transpose()
    };
    match choose() {
        Ok(frame) => frame,
        Err(error) => {
            metadata.insert("benchmarks_note", format!("Market comparison unavailable: {}", error.message()));
            None
        }
    }
}

/// The latest fitted weights above [`HELD`], largest first (stable for ties).
pub fn displayed_weights(study: &SelectionStudy) -> Series {
    let weights = &study.latest[&study.selected].weights;
    let order: Vec<usize> = weights.descending_order().into_iter().filter(|i| weights.values[*i] > HELD).collect();
    Series::new(
        order.iter().map(|i| weights.labels[*i].clone()).collect(),
        order.iter().map(|i| weights.values[*i]).collect(),
    )
}

/// The symbols that the replay bought: initial weights above [`HELD`].
pub fn replay_symbols(study: &SelectionStudy) -> Vec<String> {
    let allocation = &study.test.allocations[&study.selected];
    let initial = allocation.row(0);
    allocation
        .columns
        .iter()
        .zip(initial)
        .filter(|(_, weight)| *weight > HELD)
        .map(|(symbol, _)| symbol.clone())
        .collect()
}

/// The symbols whose company profiles the overview shows, sorted.
fn profile_symbols(study: &SelectionStudy) -> Vec<String> {
    let mut symbols: Vec<String> = displayed_weights(study).labels;
    symbols.extend(replay_symbols(study));
    symbols.sort();
    symbols.dedup();
    symbols
}

/// `fetch_company_profiles`: four workers, one shared deadline.
///
/// Profiles that miss the deadline are left out and the remaining symbols are
/// not requested. A worker that is still waiting finishes in the background and
/// its answer is ignored. A failed profile is empty; it never stops the study.
/// `None` when the build is cancelled, so that no partial set is cached.
pub fn fetch_company_profiles(
    sources: &Arc<dyn Sources>,
    symbols: &[String],
    timeout: Duration,
    cancelled: Cancelled,
) -> Option<IndexMap<String, CompanyProfile>> {
    let queue = Arc::new(Mutex::new(symbols.iter().cloned().collect::<VecDeque<_>>()));
    let stop = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    for _ in 0..symbols.len().min(4) {
        let (queue, stop, sender, sources) = (queue.clone(), stop.clone(), sender.clone(), sources.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let Some(symbol) = lock(&queue).pop_front() else { break };
                let profile = guard(|| sources.company_profile(&symbol)).unwrap_or_default();
                if sender.send((symbol, profile)).is_err() {
                    break;
                }
            }
        });
    }
    drop(sender);
    let deadline = Instant::now() + timeout;
    let mut profiles = IndexMap::new();
    while profiles.len() < symbols.len() && !cancelled() {
        let wait = deadline.saturating_duration_since(Instant::now()).min(CANCEL_POLL);
        match receiver.recv_timeout(wait) {
            Ok((symbol, profile)) => {
                profiles.insert(symbol, profile);
            }
            Err(RecvTimeoutError::Timeout) if Instant::now() < deadline => {}
            Err(_) => break,
        }
    }
    stop.store(true, Ordering::Relaxed);
    (!cancelled()).then_some(profiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_kinds_follow_the_message() {
        let cases = [
            ("Network is unreachable", true, ErrorKind::Yahoo),
            ("Yahoo price download failed: timeout", false, ErrorKind::Yahoo),
            ("Yahoo returned no prices for: ZZZQQX. Check these symbols.", false, ErrorKind::NoPrices),
            (
                "2023-01-05: requested annual volatility 1.00% is below the feasible minimum 9.00%",
                false,
                ErrorKind::Infeasible,
            ),
            ("2023-01-05: the 5.0% position cap is infeasible for 6 eligible stocks", false, ErrorKind::Infeasible),
            ("Choose an adjusted-price CSV first.", false, ErrorKind::NoFile),
            ("Provide stock price columns as well as SPY or QQQ.", false, ErrorKind::NoStocks),
            ("Missing prices for: A.", false, ErrorKind::Other),
        ];
        for (message, network, kind) in cases {
            assert_eq!(error_kind(message, network), kind, "{message}");
        }
        assert!(
            ErrorKind::Infeasible.needs_change()
                && !ErrorKind::Yahoo.needs_change()
                && !ErrorKind::Other.needs_change()
        );
        assert_eq!(ErrorKind::NoFile.headline(), "The study needs a price file.");
    }

    #[test]
    fn jobs_report_steps_and_progress() {
        let job = Job::new();
        assert_eq!(job.progress().step, "Read the selected universe");
        job.mark(4);
        job.bar(2.0, "All checked");
        job.detail("Run the out-of-sample tests");
        let progress = job.progress();
        assert_eq!(progress.step, "Fit one stock model, then replay buy and hold");
        assert_eq!(progress.bar, Some((1.0, "All checked".to_string())));
        assert_eq!(progress.detail, Some("Run the out-of-sample tests"));
        assert!(!job.is_cancelled());
        job.cancel();
        assert!(job.is_cancelled());
        assert_ne!(Job::new().id, job.id);
    }
}
