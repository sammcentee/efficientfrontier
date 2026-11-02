//! Rolling portfolio tests with dated membership and explicit price gaps.

use std::collections::HashSet;
use std::path::Path;

use indexmap::IndexMap;
use nalgebra::DMatrix;

use crate::backtest::{BacktestStudy, PORTFOLIO_NAMES, after_cost};
use crate::core::{TRADING_DAYS, estimates, optimize};
use crate::data::{parse_price_dates, read_csv_rows};
use crate::dict;
use crate::error::Result;
use crate::frame::{Cell, Date, Frame, Table, iso, mean};
use crate::membership::{MembershipHistory, is_space, load_membership_csv};
use crate::metrics::{performance_metrics, validate_periods_per_year};
use crate::value_error;

const DEMO_PRICES: &str = include_str!("../assets/demo/historical_demo_prices.csv");

/// A rolling-window study with dated membership.
///
/// `coverage` has a default index and the columns `date`, `symbol`, `status`
/// and `reason`: one eligibility decision for each member at each trade.
/// `cash_weights` holds the closing cash proportion of each strategy on each
/// date, including the initial cash balance.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalStudy {
    pub study: BacktestStudy,
    pub coverage: Table,
    pub cash_weights: Frame,
}

/// Keep missing observations. Validate each supplied price without filling gaps.
pub fn validate_historical_prices(prices: &Frame) -> Result<Frame> {
    if prices.nrows() < 5 || prices.ncols() == 0 {
        return Err(value_error!("Provide a price table with at least five dates and one asset."));
    }
    let columns: Vec<String> = prices.columns.iter().map(|name| name.trim_matches(is_space).to_string()).collect();
    let folded: HashSet<String> = columns.iter().map(|name| crate::data::casefold(name)).collect();
    if columns.iter().any(String::is_empty) || folded.len() != columns.len() {
        return Err(value_error!("Price column names must be nonempty and unique."));
    }
    if prices.index.iter().collect::<HashSet<_>>().len() != prices.nrows() {
        return Err(value_error!("Duplicate dates are not allowed; provide one price per asset per day."));
    }
    if prices.values.iter().any(|value| !value.is_nan() && (!value.is_finite() || *value <= 0.0)) {
        return Err(value_error!("Each supplied price must be finite and strictly positive."));
    }
    let mut order: Vec<usize> = (0..prices.nrows()).collect();
    order.sort_by_key(|i| prices.index[*i]);
    Ok(Frame::from_rows(
        order.iter().map(|i| prices.index[*i]).collect(),
        columns,
        &order.iter().map(|i| prices.row(*i)).collect::<Vec<_>>(),
    ))
}

/// Parse a price cell as `pandas.to_numeric` does after blank cells become NaN.
///
/// Numbers can have surrounding ASCII whitespace. Infinity words must match
/// exactly, and NaN text is invalid. `None` is an invalid cell.
fn parse_price(cell: &str) -> Option<f64> {
    if cell.chars().all(is_space) {
        return Some(f64::NAN);
    }
    let unsigned = cell.strip_prefix(['+', '-']).unwrap_or(cell);
    if unsigned.eq_ignore_ascii_case("inf") || unsigned.eq_ignore_ascii_case("infinity") {
        return Some(if cell.starts_with('-') { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    let text = cell.trim_matches([' ', '\t', '\n', '\u{0b}', '\u{0c}', '\r']);
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    if !digits.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
        return None;
    }
    text.parse().ok()
}

/// Read Date,ASSET1,ASSET2,... with blank cells for unavailable prices.
pub fn load_historical_prices(content: &[u8]) -> Result<Frame> {
    let unreadable = || value_error!("Provide a nonempty UTF-8 price CSV with Date as the first column.");
    let text = std::str::from_utf8(content).map_err(|_| unreadable())?;
    let mut rows = read_csv_rows(text.trim_start_matches('\u{feff}')).map_err(|_| unreadable())?;
    if rows.is_empty() {
        return Err(unreadable());
    }
    let headers: Vec<String> = rows.remove(0).iter().map(|name| name.trim_matches(is_space).to_string()).collect();
    if headers.first().is_none_or(|name| crate::data::casefold(name) != "date") {
        return Err(value_error!("The first price CSV column must be named Date."));
    }
    let folded: HashSet<String> = headers.iter().map(|name| crate::data::casefold(name)).collect();
    if headers.iter().any(String::is_empty) || folded.len() != headers.len() {
        return Err(value_error!("Price CSV column headers must be nonempty and unique."));
    }
    if rows.iter().any(|row| row.len() != headers.len()) {
        return Err(value_error!("Every price CSV row must have the same number of columns as its header."));
    }
    // The checks of validate_historical_prices, in its order, on text cells.
    if rows.len() < 5 || headers.len() < 2 {
        return Err(value_error!("Provide a price table with at least five dates and one asset."));
    }
    let index = parse_price_dates(&rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>())?;
    let values = rows
        .iter()
        .map(|row| row[1..].iter().map(|cell| parse_price(cell)).collect::<Option<Vec<_>>>())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| value_error!("Prices must be numeric values or blank cells."))?;
    validate_historical_prices(&Frame::from_rows(index, headers[1..].to_vec(), &values))
}

/// Load a price CSV file with [`load_historical_prices`].
pub fn load_historical_prices_path(path: &Path) -> Result<Frame> {
    let content = std::fs::read(path)
        .map_err(|_| value_error!("Provide a nonempty UTF-8 price CSV with Date as the first column."))?;
    load_historical_prices(&content)
}

/// Options for [`run_historical_backtests`], with the Python defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoricalOptions {
    pub train_fraction: f64,
    /// `None` uses the initial training length.
    pub rolling_window: Option<usize>,
    pub rebalance_every: usize,
    pub risk_free_rate: f64,
    pub max_weight: f64,
    pub shrinkage: f64,
    pub cost_bps: f64,
    pub periods_per_year: f64,
}

impl Default for HistoricalOptions {
    fn default() -> Self {
        Self {
            train_fraction: 0.7,
            rolling_window: Some(126),
            rebalance_every: 21,
            risk_free_rate: 0.02,
            max_weight: 1.0,
            shrinkage: 0.1,
            cost_bps: 10.0,
            periods_per_year: TRADING_DAYS,
        }
    }
}

/// One strategy's trade at a scheduled rebalance.
struct Trade {
    turnover: f64,
    cost: f64,
    nav_before: f64,
    nav_after: f64,
    cash_weight: f64,
}

/// One scheduled rebalance of all strategies.
struct Rebalance {
    date: Date,
    /// Target weights by strategy, over every label.
    targets: Vec<Vec<f64>>,
    trades: Vec<Trade>,
    train_start: Date,
    train_end: Date,
    expected_members: usize,
    eligible_members: usize,
}

/// Use the execution-date roster and a complete window through the prior close.
///
/// Former members leave at the next scheduled trade. Missing prices for held
/// assets or new targets stop the test. An empty or cap-infeasible universe
/// moves to cash. Cash earns zero interest. No final liquidation is assumed.
pub fn run_historical_backtests(
    prices: &Frame,
    membership: &MembershipHistory,
    options: &HistoricalOptions,
) -> Result<HistoricalStudy> {
    let prices = validate_historical_prices(prices)?;
    let periods_per_year = validate_periods_per_year(options.periods_per_year)?;
    let HistoricalOptions { train_fraction, rebalance_every, risk_free_rate, max_weight, shrinkage, cost_bps, .. } =
        *options;
    for (name, value) in [
        ("train_fraction", train_fraction),
        ("risk_free_rate", risk_free_rate),
        ("max_weight", max_weight),
        ("shrinkage", shrinkage),
        ("cost_bps", cost_bps),
    ] {
        if !value.is_finite() {
            return Err(value_error!("{name} must be a finite number."));
        }
    }
    if !(0.0 < train_fraction && train_fraction < 1.0) {
        return Err(value_error!("train_fraction must be between zero and one."));
    }
    if !(0.0 < max_weight && max_weight <= 1.0) {
        return Err(value_error!("max_weight must be greater than zero and at most one."));
    }
    if !(0.0..=1.0).contains(&shrinkage) {
        return Err(value_error!("shrinkage must be between zero and one."));
    }
    if !(0.0..10_000.0).contains(&cost_bps) {
        return Err(value_error!("cost_bps must be nonnegative and less than 10000."));
    }
    if rebalance_every < 1 {
        return Err(value_error!("rebalance_every must be a positive integer."));
    }
    let n = prices.nrows();
    let split = ((n - 1) as f64 * train_fraction).floor() as usize;
    if split < 2 || n - 1 - split < 2 {
        return Err(value_error!("At least two training returns and two holdout returns are required."));
    }
    let rolling_window = options.rolling_window.unwrap_or(split);
    if !(2..=split).contains(&rolling_window) {
        return Err(value_error!("rolling_window must be an integer between two and the initial training length."));
    }
    let dates = &prices.index;
    membership.at(dates[split])?;
    membership.at(dates[n - 1])?;
    // Include missing members in the exported holdings and target tables too.
    let mut labels = prices.columns.clone();
    for symbol in membership.symbols() {
        if !labels.contains(&symbol) {
            labels.push(symbol);
        }
    }
    let width = labels.len();
    let columns: Vec<Option<usize>> = labels.iter().map(|label| prices.position(label)).collect();
    let values = DMatrix::from_fn(n, width, |i, k| columns[k].map_or(f64::NAN, |j| prices.get(i, j)));
    let strategies: Vec<String> = PORTFOLIO_NAMES.iter().map(|name| format!("Rolling window · {name}")).collect();
    let count = strategies.len();
    let mut positions = vec![vec![0.0; width]; count];
    let mut contributions = positions.clone();
    let mut exposure = positions.clone();
    let mut maximum_weights = positions.clone();
    let mut cash = vec![1.0; count];
    let mut nav = vec![1.0; count];
    let mut paths = vec![nav.clone()];
    let mut cash_paths = vec![cash.clone()];
    let mut rebalances: Vec<Rebalance> = Vec::new();
    let mut coverage: Vec<[String; 4]> = Vec::new();
    let mut warnings = membership.limitations().to_vec();
    let mut fallbacks = vec![0_i64; count];
    let rate = cost_bps / 10_000.0;
    for i in split + 1..n {
        for row in 0..count {
            for k in 0..width {
                exposure[row][k] += positions[row][k] / nav[row];
            }
        }
        let held: Vec<bool> = (0..width).map(|k| positions.iter().any(|position| position[k] > 0.0)).collect();
        let unavailable: Vec<&str> = (0..width)
            .filter(|k| held[*k] && (!values[(i, *k)].is_finite() || !values[(i - 1, *k)].is_finite()))
            .map(|k| labels[k].as_str())
            .collect();
        if !unavailable.is_empty() {
            return Err(value_error!(
                "Missing price for held asset on {}: {}. \
                 Supply verified prices or corporate-action proceeds; prices are never filled.",
                iso(dates[i]),
                unavailable.join(", ")
            ));
        }
        for row in 0..count {
            for k in (0..width).filter(|k| held[*k]) {
                let profit = positions[row][k] * (values[(i, k)] / values[(i - 1, k)] - 1.0);
                contributions[row][k] += profit;
                positions[row][k] += profit;
            }
            nav[row] = positions[row].iter().sum::<f64>() + cash[row];
            for k in 0..width {
                maximum_weights[row][k] = maximum_weights[row][k].max(positions[row][k] / nav[row]);
            }
        }
        if i < n - 1 && (i - split - 1).is_multiple_of(rebalance_every) {
            let window = i - rolling_window - 1;
            let sample_prices = prices.rows(window, i);
            let roster = membership.at(dates[i])?;
            let hazards = membership.identity_exclusions(dates[window], Some(dates[i]))?;
            let mut eligible: Vec<&str> = Vec::new();
            for symbol in roster {
                let reason = if let Some(reason) = hazards.get(symbol) {
                    reason.clone()
                } else if let Some(j) = sample_prices.position(symbol) {
                    if sample_prices.values.column(j).iter().any(|value| value.is_nan()) {
                        format!("Need {} complete prior prices; no gap filling.", rolling_window + 1)
                    } else {
                        eligible.push(symbol);
                        String::new()
                    }
                } else {
                    "No price column for this member.".to_string()
                };
                let status = if reason.is_empty() { "eligible" } else { "excluded" };
                coverage.push([iso(dates[i]), symbol.clone(), status.to_string(), reason]);
            }
            let mut targets = vec![vec![0.0; width]; count];
            let mut fallback = false;
            if eligible.is_empty() || eligible.len() as f64 * max_weight < 1.0 - 1e-12 {
                warnings.push(format!(
                    "{}: {} eligible members cannot meet the holding cap. \
                     All strategies move to cash until the next scheduled trade.",
                    iso(dates[i]),
                    eligible.len()
                ));
            } else {
                let sample = sample_prices.select(&eligible).expect("eligible members have price columns").pct_change();
                let (mean_returns, covariance, _) = estimates(&sample, periods_per_year, shrinkage);
                let optimization = optimize(&mean_returns, &covariance, risk_free_rate, max_weight, 1)?;
                let portfolios = &optimization.portfolios;
                fallback = !portfolios.contains_key("Maximum Sharpe");
                for (row, name) in PORTFOLIO_NAMES.iter().enumerate() {
                    let chosen = portfolios.get(*name).unwrap_or(&portfolios["Minimum volatility"]);
                    targets[row] = chosen.weights.reindex(&labels, 0.0).values;
                }
                if fallback {
                    warnings.push(format!(
                        "{}: Maximum Sharpe uses Minimum volatility because no feasible \
                         portfolio has positive expected excess return.",
                        iso(dates[i])
                    ));
                }
                warnings.extend(
                    optimization
                        .warnings
                        .iter()
                        .filter(|message| !message.starts_with("Maximum Sharpe omitted:"))
                        .map(|message| format!("{}: {message}", iso(dates[i]))),
                );
            }
            let missing: Vec<&str> = (0..width)
                .filter(|k| targets.iter().any(|target| target[*k] > 0.0) && !values[(i, *k)].is_finite())
                .map(|k| labels[k].as_str())
                .collect();
            if !missing.is_empty() {
                return Err(value_error!(
                    "Missing execution price on {}: {}. \
                     Targets use prior prices; a missing execution price cannot remove a selected member.",
                    iso(dates[i]),
                    missing.join(", ")
                ));
            }
            let mut trades = Vec::with_capacity(count);
            for row in 0..count {
                let target = &targets[row];
                let after = after_cost(nav[row], &positions[row], target, rate);
                let traded: f64 = target.iter().zip(&positions[row]).map(|(t, p)| (after * t - p).abs()).sum();
                cash[row] = if target.iter().any(|weight| *weight != 0.0) { 0.0 } else { after };
                trades.push(Trade {
                    turnover: traded / nav[row],
                    cost: nav[row] - after,
                    nav_before: nav[row],
                    nav_after: after,
                    cash_weight: cash[row] / after,
                });
                positions[row] = target.iter().map(|weight| after * weight).collect();
                nav[row] = after;
            }
            fallbacks[1] += i64::from(fallback);
            for row in 0..count {
                for k in 0..width {
                    maximum_weights[row][k] = maximum_weights[row][k].max(positions[row][k] / nav[row]);
                }
            }
            rebalances.push(Rebalance {
                date: dates[i],
                targets,
                trades,
                train_start: sample_prices.index[1],
                train_end: dates[i - 1],
                expected_members: roster.len(),
                eligible_members: eligible.len(),
            });
        }
        paths.push(nav.clone());
        cash_paths.push(cash.iter().zip(&nav).map(|(c, v)| c / v).collect());
    }

    let index = dates[split..].to_vec();
    let equity = Frame::from_rows(index.clone(), strategies.clone(), &paths);
    let cash_weights = Frame::from_rows(index, strategies.clone(), &cash_paths);
    let trade_dates: Vec<Date> = rebalances.iter().map(|event| event.date).collect();
    let floats = |values: Vec<f64>| values.into_iter().map(Cell::Float).collect::<Vec<_>>();
    let (mut allocations, mut trades, mut holdings) = (IndexMap::new(), IndexMap::new(), IndexMap::new());
    let mut metric_rows: Vec<Vec<Cell>> = Vec::new();
    for (row, key) in strategies.iter().enumerate() {
        let allocation = Frame::from_rows(
            trade_dates.clone(),
            labels.clone(),
            &rebalances.iter().map(|event| event.targets[row].clone()).collect::<Vec<_>>(),
        );
        let field = |value: fn(&Trade) -> f64| rebalances.iter().map(|event| value(&event.trades[row])).collect();
        let turnover: Vec<f64> = field(|trade| trade.turnover);
        let cost: Vec<f64> = field(|trade| trade.cost);
        let table = Table::new(Some("Date"), trade_dates.iter().map(|date| Cell::Date(*date)).collect())
            .with("turnover", floats(turnover.clone()))
            .with("cost", floats(cost.clone()))
            .with("nav_before", floats(field(|trade| trade.nav_before)))
            .with("nav_after", floats(field(|trade| trade.nav_after)))
            .with("train_start", rebalances.iter().map(|event| Cell::Str(iso(event.train_start))).collect())
            .with("train_end", rebalances.iter().map(|event| Cell::Str(iso(event.train_end))).collect())
            .with("expected_members", rebalances.iter().map(|event| Cell::from(event.expected_members)).collect())
            .with("eligible_members", rebalances.iter().map(|event| Cell::from(event.eligible_members)).collect())
            .with("cash_weight", floats(field(|trade| trade.cash_weight)));
        let selection_frequency: Vec<f64> = (0..width)
            .map(|k| {
                mean(&allocation.col(k).iter().map(|weight| f64::from(u8::from(*weight > 1e-6))).collect::<Vec<_>>())
            })
            .collect();
        let max_target_weight: Vec<f64> =
            (0..width).map(|k| allocation.col(k).into_iter().fold(f64::NAN, f64::max)).collect();
        let held = Table::labelled(None, &labels)
            .with("average_weight", floats(exposure[row].iter().map(|value| value / (n - split - 1) as f64).collect()))
            .with("end_weight", floats(positions[row].iter().map(|value| value / nav[row]).collect()))
            .with("selection_frequency", floats(selection_frequency))
            .with("max_target_weight", floats(max_target_weight))
            .with("max_realized_weight", floats(maximum_weights[row].clone()))
            .with("pnl_contribution", floats(contributions[row].clone()));
        let path: Vec<f64> = paths.iter().map(|values| values[row]).collect();
        let mut metrics: Vec<Cell> = performance_metrics(&path, risk_free_rate, periods_per_year)
            .fields()
            .iter()
            .map(|(_, value)| Cell::Float(*value))
            .collect();
        metrics.push(Cell::Float(turnover.iter().sum()));
        metrics.push(Cell::Float(cost.iter().sum()));
        metrics.push(Cell::from(trade_dates.len()));
        metrics.push(Cell::Int(fallbacks[row]));
        metric_rows.push(metrics);
        allocations.insert(key.clone(), allocation);
        trades.insert(key.clone(), table);
        holdings.insert(key.clone(), held);
    }
    let metric_names = crate::metrics::PERFORMANCE_FIELDS.iter().copied().chain([
        "total_turnover",
        "total_cost",
        "rebalance_count",
        "fallback_count",
    ]);
    let mut metrics = Table::labelled(Some("strategy"), &strategies);
    for (k, name) in metric_names.enumerate() {
        metrics.push(name, metric_rows.iter().map(|row| row[k].clone()).collect());
    }
    let excluded_dates: HashSet<&String> =
        coverage.iter().filter(|decision| decision[2] == "excluded").map(|decision| &decision[0]).collect();
    if !excluded_dates.is_empty() {
        warnings.push(format!(
            "{} allocation dates exclude at least one historical member. Review coverage.csv. \
             Incomplete prices and membership records can still cause survivorship bias.",
            excluded_dates.len()
        ));
    }
    let coverage_column = |k: usize| coverage.iter().map(|decision| Cell::Str(decision[k].clone())).collect();
    let coverage_table = Table::range(coverage.len())
        .with("date", coverage_column(0))
        .with("symbol", coverage_column(1))
        .with("status", coverage_column(2))
        .with("reason", coverage_column(3));
    let settings = dict! {
        "historical_membership" => true,
        "train_fraction" => train_fraction,
        "split" => split,
        "rolling_window" => rolling_window,
        "rebalance_every" => rebalance_every,
        "risk_free_rate" => risk_free_rate,
        "max_weight" => max_weight,
        "shrinkage" => shrinkage,
        "cost_bps" => cost_bps,
        "periods_per_year" => periods_per_year,
        "cash_interest_rate" => 0.0,
        "train_start" => iso(dates[split - rolling_window + 1]),
        "train_end" => iso(dates[split]),
        "test_start" => iso(dates[split + 1]),
        "test_end" => iso(dates[n - 1]),
        "initial_execution" => iso(dates[split + 1]),
        "include_profiles" => false,
        "execution" => "Use the effective roster on each trade date; fit through the prior close; execute at the next close. \
                        The first evaluation session is cash. Membership publication timestamps are not verified.",
        "departure_policy" => "Sell former members at the next scheduled trade, not on the membership change date.",
        "eligibility" => "Require a complete prior price window and no known identity ambiguity. \
                          Never inspect future prices for eligibility.",
        "cash_policy" => "Move fully to zero-interest cash when eligible members cannot meet the holding cap.",
        "cost_convention" => "Basis points per absolute dollar bought or sold; entry included; no final liquidation.",
        "membership" => membership.metadata(),
    };
    Ok(HistoricalStudy {
        study: BacktestStudy { equity, metrics, allocations, holdings, trades, warnings, settings },
        coverage: coverage_table,
        cash_weights,
    })
}

/// A reproducible synthetic study with entry, departure, reentry and a short price history.
///
/// The prices are the exact output of the Python release's seeded generator
/// (`numpy.random.default_rng(1942)`), bundled so the demo stays identical.
pub fn historical_demo() -> (Frame, MembershipHistory) {
    let prices = load_historical_prices(DEMO_PRICES.as_bytes()).expect("the bundled historical demo prices are valid");
    let dates = &prices.index;
    let changes = [
        (0, "ALPHA,BETA,GAMMA"),
        (265, "ALPHA,GAMMA,NEWCO"),
        (286, "ALPHA,BETA,GAMMA,NEWCO"),
        (310, "ALPHA,BETA,NEWCO"),
    ];
    let mut snapshots = String::from("date,tickers\n");
    for (row, tickers) in changes {
        snapshots.push_str(&format!("{},\"{tickers}\"\n", iso(dates[row])));
    }
    let coverage_end = iso(dates[dates.len() - 1]);
    let membership = load_membership_csv(
        snapshots.as_bytes(),
        coverage_end.as_str(),
        Some("Synthetic demo with dated membership changes"),
    )
    .expect("the demo membership is valid");
    (prices, membership)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(text: &str) -> Date {
        Date::parse_from_str(text, "%Y-%m-%d").unwrap()
    }

    fn frame(columns: &[&str], rows: &[Vec<f64>]) -> Frame {
        let start = day("2021-01-04");
        let index = (0..rows.len() as u64).map(|i| start + chrono::Days::new(i)).collect();
        Frame::from_rows(index, columns.iter().map(|name| name.to_string()).collect(), rows)
    }

    fn csv(cell: &str) -> String {
        format!("Date,A,B\n2021-01-04,{cell},2\n2021-01-05,1,2\n2021-01-06,1,2\n2021-01-07,1,2\n2021-01-08,1,2\n")
    }

    #[test]
    fn missing_prices_stay_missing_and_dates_are_sorted() {
        let mut prices = frame(&[" A ", "B"], &vec![vec![1.0, f64::NAN]; 6]);
        prices.index.swap(0, 5);
        prices.values[(0, 0)] = 2.0;
        let result = validate_historical_prices(&prices).unwrap();
        assert_eq!(result.columns, ["A", "B"]);
        assert_eq!(result.index[0], day("2021-01-04"));
        assert_eq!(result.index[5], day("2021-01-09"));
        assert_eq!(result.get(5, 0), 2.0);
        assert!(result.col(1).iter().all(|value| value.is_nan()));
    }

    #[test]
    fn invalid_frames_fail_with_python_messages() {
        let rows = vec![vec![1.0, 2.0]; 5];
        let message = |prices: &Frame| validate_historical_prices(prices).unwrap_err().message().to_string();
        assert_eq!(
            message(&frame(&["A", "B"], &rows[..4])),
            "Provide a price table with at least five dates and one asset."
        );
        assert_eq!(message(&frame(&["A", "a"], &rows)), "Price column names must be nonempty and unique.");
        assert_eq!(message(&frame(&["A", " "], &rows)), "Price column names must be nonempty and unique.");
        let mut repeated = frame(&["A", "B"], &rows);
        repeated.index[1] = repeated.index[0];
        assert_eq!(message(&repeated), "Duplicate dates are not allowed; provide one price per asset per day.");
        for value in [0.0, -1.0, f64::INFINITY] {
            let mut invalid = frame(&["A", "B"], &rows);
            invalid.values[(2, 1)] = value;
            assert_eq!(message(&invalid), "Each supplied price must be finite and strictly positive.");
        }
    }

    /// Python also rejects boolean and complex objects in a frame. Here only
    /// CSV text can hold them.
    #[test]
    fn invalid_price_values_fail() {
        for cell in ["True", "(2+1j)", "-1", "0", "inf", "bad", "nan"] {
            let error = load_historical_prices(csv(cell).as_bytes()).unwrap_err();
            assert!(error.message().to_lowercase().contains("price"), "{cell}: {error}");
        }
        let message = |cell: &str| load_historical_prices(csv(cell).as_bytes()).unwrap_err().message().to_string();
        assert_eq!(message("bad"), "Prices must be numeric values or blank cells.");
        assert_eq!(message("inf"), "Each supplied price must be finite and strictly positive.");
    }

    #[test]
    fn cells_parse_as_pandas_to_numeric() {
        assert_eq!(parse_price(" 1.5\t"), Some(1.5));
        assert_eq!(parse_price("+.5"), Some(0.5));
        assert_eq!(parse_price("1E+0005"), Some(1e5));
        assert_eq!(parse_price("-Infinity"), Some(f64::NEG_INFINITY));
        assert!(parse_price(" \u{a0}\u{1c}").unwrap().is_nan());
        for invalid in ["nan", " inf", "1_000", "0x1A", "1,000", "\u{a0}1", "e1", "."] {
            assert_eq!(parse_price(invalid), None, "{invalid:?}");
        }
    }

    #[test]
    fn bad_csv_structure_fails() {
        let cases = [
            ("", "Provide a nonempty UTF-8 price CSV with Date as the first column."),
            ("day,A\n2021-01-01,1\n", "The first price CSV column must be named Date."),
            ("Date,A,A\n2021-01-01,1,2\n", "Price CSV column headers must be nonempty and unique."),
            ("Date,A\n2021-01-01,1,2\n", "Every price CSV row must have the same number of columns as its header."),
        ];
        for (text, message) in cases {
            assert_eq!(load_historical_prices(text.as_bytes()).unwrap_err().message(), message, "{text:?}");
        }
        assert!(load_historical_prices(b"Date,A\n2021-01-04,\xff\n").is_err());
        let missing = Path::new("/nonexistent/historical prices.csv");
        assert_eq!(
            load_historical_prices_path(missing).unwrap_err().message(),
            "Provide a nonempty UTF-8 price CSV with Date as the first column."
        );
    }

    #[test]
    fn demo_matches_the_python_release() {
        let (prices, membership) = historical_demo();
        assert_eq!((prices.nrows(), prices.ncols()), (360, 4));
        assert_eq!(prices.columns, ["ALPHA", "BETA", "GAMMA", "NEWCO"]);
        assert_eq!(prices.get(0, 0), 99.43356017119692);
        assert!(prices.col(3)[..170].iter().all(|value| value.is_nan()));
        assert!(prices.col(3)[170..].iter().all(|value| value.is_finite()));
        let dates: Vec<String> = membership.snapshots().iter().map(|(date, _)| iso(*date)).collect();
        assert_eq!(dates, ["2020-01-02", "2021-01-07", "2021-02-05", "2021-03-11"]);
        assert_eq!(membership.snapshots()[3].1, ["ALPHA", "BETA", "NEWCO"]);
        assert_eq!(membership.coverage_end(), day("2021-05-19"));
        assert_eq!(membership.source(), "Synthetic demo with dated membership changes");
    }
}
