//! Chronological portfolio backtests with delayed execution and explicit costs.

use indexmap::IndexMap;

use crate::error::Result;
use crate::frame::{Frame, Table};
use crate::pyfmt::Dict;

pub const METHOD_NAMES: [&str; 4] = ["Buy and hold", "Fixed rebalance", "Expanding window", "Rolling window"];
pub const PORTFOLIO_NAMES: [&str; 3] = ["Minimum volatility", "Maximum Sharpe", "Equal weight"];
pub const PROFILE_NAMES: [&str; 3] = ["Low", "Medium", "Extreme"];

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
            periods_per_year: crate::core::TRADING_DAYS,
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
pub fn run_backtests(prices: &Frame, options: &BacktestOptions) -> Result<BacktestStudy> {
    let _ = (prices, options);
    todo!("legacy backtests: port efficient_frontier/backtest.py run_backtests")
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
