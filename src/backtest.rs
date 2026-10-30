//! Chronological portfolio backtests with delayed execution and explicit costs.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;

use crate::core::{Optimization, TRADING_DAYS, estimates, optimize, validate_price_frame};
use crate::error::Result;
use crate::frame::{Cell, Frame, Table, iso, mean};
use crate::metrics::{performance_metrics, validate_periods_per_year};
use crate::profiles::{ProfileOptions, build_profiles};
use crate::pyfmt::Dict;
use crate::{dict, value_error};

pub const METHOD_NAMES: [&str; 4] = ["Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window"];
pub const PORTFOLIO_NAMES: [&str; 3] = ["Minimum volatility", "Maximum Sharpe", "Equal weight"];
pub use crate::profiles::PROFILE_NAMES;

/// Every path of one study, with its trades and holdings.
///
/// `metrics` has a `strategy` index. `allocations` are target weights by trade
/// date. `holdings` have a symbol index and `trades` have a trade-date index.
#[derive(Clone, Debug, PartialEq)]
pub struct BacktestStudy {
    pub equity: Frame,
    pub metrics: Table,
    pub allocations: IndexMap<String, Frame>,
    pub holdings: IndexMap<String, Table>,
    pub trades: IndexMap<String, Table>,
    pub warnings: Vec<String>,
    pub settings: Dict,
}

/// Solve the self-financing rebalance, charging both buys and sells.
pub fn after_cost(nav: f64, positions: &[f64], target: &[f64], rate: f64) -> f64 {
    if rate == 0.0 {
        return nav;
    }
    let (mut low, mut high) = (0.0, nav);
    for _ in 0..64 {
        let value = (low + high) / 2.0;
        let traded: f64 = target.iter().zip(positions).map(|(t, p)| (value * t - p).abs()).sum();
        if value + rate * traded > nav {
            high = value;
        } else {
            low = value;
        }
    }
    (low + high) / 2.0
}

/// Options for [`run_backtests`], with the Python defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BacktestOptions {
    pub train_fraction: f64,
    pub risk_free_rate: f64,
    pub max_weight: f64,
    pub shrinkage: f64,
    pub rebalance_every: usize,
    /// `None` uses the initial training length.
    pub rolling_window: Option<usize>,
    pub cost_bps: f64,
    pub periods_per_year: f64,
    pub include_profiles: bool,
}

impl Default for BacktestOptions {
    fn default() -> Self {
        Self {
            train_fraction: 0.7,
            risk_free_rate: 0.02,
            max_weight: 1.0,
            shrinkage: 0.1,
            rebalance_every: 21,
            rolling_window: None,
            cost_bps: 10.0,
            periods_per_year: TRADING_DAYS,
            include_profiles: false,
        }
    }
}

/// Compare four execution methods for long-only portfolio targets.
///
/// Fit through one close and trade at the following close. The first out-of-
/// sample session is cash earning zero interest. Refit windows exclude execution-
/// day returns. Costs are basis points per absolute dollar traded, including
/// entry and both legs of later trades. There is no terminal liquidation.
///
/// Holding contribution is dollar profit in initial-capital units before costs;
/// contributions minus total cost reconcile to the portfolio's total return.
/// Optional Low, Medium, and Extreme historical-mean profiles use only each past fit window.
///
/// Paths are keyed `"{method} · {portfolio}"`, in method order, then portfolio order.
/// `equity` starts at the last training close. `metrics` has the
/// [`crate::metrics::PERFORMANCE_FIELDS`] columns, then `total_turnover`, `total_cost`,
/// and the integers `rebalance_count` and `fallback_count`. `holdings` has the
/// columns `average_weight`, `end_weight`, `selection_frequency`, `max_target_weight`,
/// `max_realized_weight`, `pnl_contribution`. `trades` has the columns `turnover`,
/// `cost`, `nav_before`, `nav_after`, and the ISO dates `train_start` and `train_end`
/// of the first and last fit return.
pub fn run_backtests(prices: &Frame, options: &BacktestOptions) -> Result<BacktestStudy> {
    validate_price_frame(prices)?;
    let periods_per_year = validate_periods_per_year(options.periods_per_year)?;
    let BacktestOptions {
        train_fraction,
        risk_free_rate,
        max_weight,
        shrinkage,
        rebalance_every,
        rolling_window,
        cost_bps,
        include_profiles,
        ..
    } = *options;
    if !train_fraction.is_finite() || !(train_fraction > 0.0 && train_fraction < 1.0) {
        return Err(value_error!("train_fraction must be between zero and one."));
    }
    if !shrinkage.is_finite() || !(0.0..=1.0).contains(&shrinkage) {
        return Err(value_error!("shrinkage must be between zero and one."));
    }
    if !risk_free_rate.is_finite() {
        return Err(value_error!("risk_free_rate must be finite."));
    }
    if !max_weight.is_finite() || !(max_weight > 0.0 && max_weight <= 1.0) {
        return Err(value_error!("max_weight must be greater than zero and at most one."));
    }
    if prices.ncols() as f64 * max_weight < 1.0 - 1e-12 {
        return Err(value_error!("The position cap is infeasible: asset count * max_weight < 1."));
    }
    if rebalance_every < 1 {
        return Err(value_error!("rebalance_every must be a positive integer."));
    }
    if !cost_bps.is_finite() || !(0.0..10_000.0).contains(&cost_bps) {
        return Err(value_error!("cost_bps must be finite, nonnegative and less than 10000."));
    }
    let returns = prices.pct_change();
    let split = (returns.nrows() as f64 * train_fraction).floor() as usize;
    if split < 2 || returns.nrows() - split < 2 {
        return Err(value_error!("At least two training returns and two holdout returns are required."));
    }
    let rolling_window = rolling_window.unwrap_or(split);
    if !(2..=split).contains(&rolling_window) {
        return Err(value_error!("rolling_window must be an integer between two and the initial training length."));
    }
    if include_profiles && (split < 6 || rolling_window < 6) {
        return Err(value_error!(
            "Profile backtests require at least six training returns in both the initial and rolling windows."
        ));
    }

    let labels = &prices.columns;
    let dates = &prices.index;
    let n = labels.len();
    let mut portfolio_names: Vec<&str> = PORTFOLIO_NAMES.to_vec();
    if include_profiles {
        portfolio_names.extend(PROFILE_NAMES);
    }
    let execution_rows: Vec<usize> = (split + 1..prices.nrows() - 1).step_by(rebalance_every).collect();
    let profile_options =
        ProfileOptions { risk_free_rate, max_weight, shrinkage, periods_per_year, ..Default::default() };
    // Only weights and messages are retained, not each dense covariance.
    let fit = |start: usize, end: usize| -> Result<Fit> {
        let sample = returns.rows(start, end);
        let (mean_returns, covariance, _) = estimates(&sample, periods_per_year, shrinkage);
        let Optimization { portfolios, warnings: messages, .. } =
            optimize(&mean_returns, &covariance, risk_free_rate, max_weight, 1)?;
        let fallback = !portfolios.contains_key("Maximum Sharpe");
        let mut targets: HashMap<&'static str, Vec<f64>> = PORTFOLIO_NAMES
            .iter()
            .map(|name| {
                let source = if portfolios.contains_key(*name) { *name } else { "Minimum volatility" };
                (*name, portfolios[source].weights.values.clone())
            })
            .collect();
        let mut profile_messages = Vec::new();
        if include_profiles {
            let profiles = build_profiles(&prices.rows(start, end + 1), &profile_options)?;
            for name in PROFILE_NAMES {
                targets.insert(name, profiles.portfolios[name].weights.reindex(labels, f64::NAN).values);
            }
            profile_messages = profiles.warnings;
        }
        Ok(Fit { targets, fallback, messages, profile_messages })
    };

    let mut fits: HashMap<(usize, usize), Fit> = HashMap::new();
    let mut equity_columns = Vec::new();
    let mut allocations = IndexMap::new();
    let mut holdings = IndexMap::new();
    let mut trades = IndexMap::new();
    let mut metric_names = Vec::new();
    let mut metric_rows: Vec<Vec<Cell>> = Vec::new();
    let mut warnings = Vec::new();
    let mut seen_profile_warnings = HashSet::new();
    let cost_rate = cost_bps / 10_000.0;
    for method in METHOD_NAMES {
        let rows = if method == "Buy and hold" { vec![split + 1] } else { execution_rows.clone() };
        // The fit window `(start, end)` of each execution row.
        let mut schedule: HashMap<usize, (usize, usize)> = HashMap::new();
        for i in rows {
            let end = if matches!(method, "Buy and hold" | "Fixed rebalance") { split } else { i - 1 };
            let start = if method == "Rolling window" { end.saturating_sub(rolling_window) } else { 0 };
            if let Entry::Vacant(entry) = fits.entry((start, end)) {
                entry.insert(fit(start, end)?);
            }
            let result = &fits[&(start, end)];
            schedule.insert(i, (start, end));
            let date = iso(dates[i]);
            if result.fallback {
                warnings.push(format!(
                    "{date} · {method}: Maximum Sharpe uses Minimum volatility \
                     because no feasible portfolio has positive expected excess return."
                ));
            }
            for message in &result.messages {
                if !message.starts_with("Maximum Sharpe omitted:") {
                    warnings.push(format!("{date} · {method}: {message}"));
                }
            }
            for message in &result.profile_messages {
                if seen_profile_warnings.insert(message.clone()) {
                    warnings.push(format!("{date} · {method} · Profiles: {message}"));
                }
            }
        }

        for portfolio in &portfolio_names {
            let key = format!("{method} · {portfolio}");
            let mut positions = vec![0.0; n];
            let mut contribution = vec![0.0; n];
            let mut exposure = vec![0.0; n];
            let mut max_realized_weight = vec![0.0_f64; n];
            let (mut cash, mut nav) = (1.0, 1.0);
            let mut path = vec![nav];
            let mut targets_used: Vec<Vec<f64>> = Vec::new();
            let mut trade_values: Vec<[f64; 4]> = Vec::new();
            let mut trade_windows: Vec<(String, String)> = Vec::new();
            let mut trade_dates = Vec::new();
            let mut fallbacks = 0;
            for (i, date) in dates.iter().enumerate().skip(split + 1) {
                for (k, position) in positions.iter_mut().enumerate() {
                    exposure[k] += *position / nav;
                    let profit = *position * returns.get(i - 1, k);
                    contribution[k] += profit;
                    *position += profit;
                }
                nav = positions.iter().sum::<f64>() + cash;
                for (highest, position) in max_realized_weight.iter_mut().zip(&positions) {
                    *highest = highest.max(position / nav);
                }
                if let Some(&(start, end)) = schedule.get(&i) {
                    let result = &fits[&(start, end)];
                    let target = &result.targets[portfolio];
                    let after = after_cost(nav, &positions, target, cost_rate);
                    let traded: f64 = target.iter().zip(&positions).map(|(t, p)| (after * t - p).abs()).sum();
                    trade_values.push([traded / nav, nav - after, nav, after]);
                    trade_windows.push((iso(returns.index[start]), iso(returns.index[end - 1])));
                    trade_dates.push(*date);
                    targets_used.push(target.clone());
                    fallbacks += i64::from(result.fallback && *portfolio == "Maximum Sharpe");
                    positions = target.iter().map(|weight| after * weight).collect();
                    cash = 0.0;
                    nav = after;
                    for (highest, position) in max_realized_weight.iter_mut().zip(&positions) {
                        *highest = highest.max(position / nav);
                    }
                }
                path.push(nav);
            }

            let floats = |values: Vec<f64>| values.into_iter().map(Cell::Float).collect::<Vec<_>>();
            let allocation = Frame::from_rows(trade_dates.clone(), labels.clone(), &targets_used);
            let trade_table = Table::new(None, trade_dates.iter().map(|date| Cell::Date(*date)).collect())
                .with("turnover", floats(trade_values.iter().map(|v| v[0]).collect()))
                .with("cost", floats(trade_values.iter().map(|v| v[1]).collect()))
                .with("nav_before", floats(trade_values.iter().map(|v| v[2]).collect()))
                .with("nav_after", floats(trade_values.iter().map(|v| v[3]).collect()))
                .with("train_start", trade_windows.iter().map(|(start, _)| Cell::from(start.as_str())).collect())
                .with("train_end", trade_windows.iter().map(|(_, end)| Cell::from(end.as_str())).collect());
            let held_periods = (prices.nrows() - split - 1) as f64;
            let selection_frequency = (0..n)
                .map(|j| mean(&allocation.col(j).iter().map(|w| f64::from(u8::from(*w > 1e-6))).collect::<Vec<_>>()))
                .collect();
            let max_target_weight =
                (0..n).map(|j| allocation.col(j).into_iter().fold(f64::NEG_INFINITY, f64::max)).collect();
            let held = Table::labelled(None, labels)
                .with("average_weight", floats(exposure.iter().map(|e| e / held_periods).collect()))
                .with("end_weight", floats(positions.iter().map(|p| p / nav).collect()))
                .with("selection_frequency", floats(selection_frequency))
                .with("max_target_weight", floats(max_target_weight))
                .with("max_realized_weight", floats(max_realized_weight))
                .with("pnl_contribution", floats(contribution));
            let mut row: Vec<Cell> = performance_metrics(&path, risk_free_rate, periods_per_year)
                .fields()
                .iter()
                .map(|(_, value)| Cell::Float(*value))
                .collect();
            row.push(Cell::Float(trade_values.iter().map(|v| v[0]).sum()));
            row.push(Cell::Float(trade_values.iter().map(|v| v[1]).sum()));
            row.push(Cell::from(trade_values.len()));
            row.push(Cell::Int(fallbacks));
            metric_rows.push(row);
            metric_names.push(key.clone());
            equity_columns.push((key.clone(), path));
            allocations.insert(key.clone(), allocation);
            holdings.insert(key.clone(), held);
            trades.insert(key, trade_table);
        }
    }

    let mut metrics = Table::labelled(Some("strategy"), &metric_names);
    let columns = crate::metrics::PERFORMANCE_FIELDS.into_iter().chain([
        "total_turnover",
        "total_cost",
        "rebalance_count",
        "fallback_count",
    ]);
    for (k, column) in columns.enumerate() {
        metrics.push(column, metric_rows.iter().map(|row| row[k].clone()).collect());
    }
    let last = prices.nrows() - 1;
    let settings = dict! {
        "train_fraction" => train_fraction,
        "risk_free_rate" => risk_free_rate,
        "max_weight" => max_weight,
        "shrinkage" => shrinkage,
        "rebalance_every" => rebalance_every,
        "rolling_window" => rolling_window,
        "cost_bps" => cost_bps,
        "split" => split,
        "include_profiles" => include_profiles,
        "profile_return_objective" => include_profiles.then_some("historical_mean"),
        "periods_per_year" => periods_per_year,
        "trading_days_per_year" => periods_per_year,
        "cash_interest_rate" => 0.0,
        "train_start" => iso(returns.index[0]),
        "train_end" => iso(dates[split]),
        "test_start" => iso(dates[split + 1]),
        "test_end" => iso(dates[last]),
        "initial_execution" => iso(dates[split + 1]),
        "execution" => "Prior-close information; execute at the following close; first session in cash.",
        "cost_convention" => "Basis points per absolute dollar bought or sold; entry included; no final liquidation.",
    };
    Ok(BacktestStudy {
        equity: Frame::from_columns(dates[split..].to_vec(), equity_columns),
        metrics,
        allocations,
        holdings,
        trades,
        warnings,
        settings,
    })
}

/// The targets and messages of one fit window.
struct Fit {
    targets: HashMap<&'static str, Vec<f64>>,
    fallback: bool,
    messages: Vec<String>,
    profile_messages: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_solve_is_self_financing() {
        let positions = [0.0, 0.0];
        let target = [0.6, 0.4];
        let after = after_cost(1.0, &positions, &target, 0.001);
        assert!((after - 1.0 / 1.001).abs() < 1e-12);
        assert_eq!(after_cost(1.0, &positions, &target, 0.0), 1.0);
    }
}
