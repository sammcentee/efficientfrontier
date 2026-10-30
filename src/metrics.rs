//! Shared annualization and realized portfolio performance measures.

use crate::error::Result;
use crate::frame::{Cell, mean, std};
use crate::value_error;

/// The metric names in export order.
pub const PERFORMANCE_FIELDS: [&str; 7] =
    ["total_return", "cagr", "volatility", "sharpe", "max_drawdown", "sortino", "calmar"];

pub fn validate_periods_per_year(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(value_error!("periods_per_year must be a finite positive number."));
    }
    Ok(value)
}

/// Realized measures of one equity path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Performance {
    pub total_return: f64,
    pub cagr: f64,
    pub volatility: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub sortino: f64,
    pub calmar: f64,
}

impl Performance {
    /// `(name, value)` pairs in export order.
    pub fn fields(&self) -> [(&'static str, f64); 7] {
        [
            ("total_return", self.total_return),
            ("cagr", self.cagr),
            ("volatility", self.volatility),
            ("sharpe", self.sharpe),
            ("max_drawdown", self.max_drawdown),
            ("sortino", self.sortino),
            ("calmar", self.calmar),
        ]
    }

    /// The fields as table cells in export order.
    pub fn cells(&self) -> Vec<(&'static str, Cell)> {
        self.fields().into_iter().map(|(name, value)| (name, Cell::Float(value))).collect()
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.fields().into_iter().find(|(field, _)| *field == name).map(|(_, value)| value)
    }
}

/// Simple returns of a path, from its second value on.
pub fn path_returns(equity: &[f64]) -> Vec<f64> {
    equity.windows(2).map(|pair| pair[1] / pair[0] - 1.0).collect()
}

/// Measure a positive equity path that includes its initial value.
///
/// Sortino uses the annual arithmetic excess return divided by annual downside
/// deviation. Its period target is `risk_free_rate / periods_per_year`, and its
/// denominator includes all observations. Calmar uses CAGR / absolute drawdown.
pub fn performance_metrics(equity: &[f64], risk_free_rate: f64, periods_per_year: f64) -> Performance {
    let returns = path_returns(equity);
    let growth = equity[equity.len() - 1] / equity[0];
    let cagr = growth.powf(periods_per_year / returns.len() as f64) - 1.0;
    let volatility = std(&returns, 1) * periods_per_year.sqrt();
    let excess = mean(&returns) * periods_per_year - risk_free_rate;
    let target = risk_free_rate / periods_per_year;
    let shortfall: Vec<f64> = returns.iter().map(|value| (value - target).min(0.0).powi(2)).collect();
    let downside = (mean(&shortfall) * periods_per_year).sqrt();
    let mut peak = f64::NEG_INFINITY;
    let mut drawdown = f64::INFINITY;
    for value in equity {
        peak = peak.max(*value);
        drawdown = drawdown.min(value / peak - 1.0);
    }
    Performance {
        total_return: growth - 1.0,
        cagr,
        volatility,
        sharpe: if volatility > 1e-12 { excess / volatility } else { f64::NAN },
        max_drawdown: drawdown,
        sortino: if downside > 0.0 { excess / downside } else { f64::NAN },
        calmar: if drawdown < 0.0 { cagr / drawdown.abs() } else { f64::NAN },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_follow_their_definitions() {
        let equity = [1.0, 1.1, 0.99, 1.089];
        let result = performance_metrics(&equity, 0.02, 252.0);
        assert!((result.total_return - 0.089).abs() < 1e-12);
        assert!((result.max_drawdown + 0.1).abs() < 1e-12);
        let returns = [0.1, -0.1, 0.1];
        let volatility = std(&returns, 1) * 252f64.sqrt();
        assert!((result.volatility - volatility).abs() < 1e-12);
        assert!((result.sharpe - (mean(&returns) * 252.0 - 0.02) / volatility).abs() < 1e-12);
        assert!((result.cagr - (1.089f64.powf(84.0) - 1.0)).abs() < 1e-6 * result.cagr.abs());
    }

    #[test]
    fn ratios_are_undefined_without_risk_or_losses() {
        let result = performance_metrics(&[1.0, 1.0, 1.0], 0.0, 252.0);
        assert!(result.sharpe.is_nan());
        assert!(result.sortino.is_nan());
        assert!(result.calmar.is_nan());
        assert_eq!(result.max_drawdown, 0.0);
    }

    #[test]
    fn invalid_annualization_is_rejected() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(validate_periods_per_year(value).is_err());
        }
        assert_eq!(validate_periods_per_year(52.0).unwrap(), 52.0);
    }
}
