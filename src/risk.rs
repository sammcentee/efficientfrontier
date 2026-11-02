//! Portfolio concentration and contributions to estimated risk.

use nalgebra::{DMatrix, DVector};

use crate::core::{Analysis, dot, quadratic_form};
use crate::frame::{Cell, Table};

/// The training covariance and each portfolio's weights, both in the order of
/// `analysis.mean_returns`, as pandas aligns them with `.loc[labels]`.
///
/// Panics when a label is missing, where pandas raises `KeyError`.
fn aligned(analysis: &Analysis) -> (DMatrix<f64>, Vec<(&str, Vec<f64>)>) {
    let labels = &analysis.mean_returns.labels;
    let positions = |names: &[String]| -> Vec<usize> {
        labels
            .iter()
            .map(|label| {
                names.iter().position(|name| name == label).unwrap_or_else(|| panic!("asset {label:?} is missing"))
            })
            .collect()
    };
    let rows = positions(&analysis.covariance.index);
    let columns = positions(&analysis.covariance.columns);
    let covariance =
        DMatrix::from_fn(labels.len(), labels.len(), |i, j| analysis.covariance.values[(rows[i], columns[j])]);
    let portfolios = analysis
        .portfolios
        .iter()
        .map(|(name, portfolio)| {
            let weights = labels
                .iter()
                .map(|label| portfolio.weights.get(label).unwrap_or_else(|| panic!("weight of {label:?} is missing")))
                .collect();
            (name.as_str(), weights)
        })
        .collect();
    (covariance, portfolios)
}

/// Return concentration and diversification measures from the training model.
///
/// Effective holdings equal 1 / sum(weight ** 2). The diversification ratio
/// divides the weighted sum of asset volatilities by the portfolio volatility.
/// A zero portfolio variance gives an undefined diversification ratio.
///
/// The table has a `portfolio` index and the columns `max_weight`,
/// `effective_holdings` and `diversification_ratio`.
pub fn risk_summary(analysis: &Analysis) -> Table {
    let (covariance, portfolios) = aligned(analysis);
    let asset_volatility: Vec<f64> = covariance.diagonal().iter().map(|variance| variance.max(0.0).sqrt()).collect();
    let names: Vec<&str> = portfolios.iter().map(|(name, _)| *name).collect();
    let mut rows = Vec::with_capacity(portfolios.len());
    for (_, weights) in &portfolios {
        let variance = quadratic_form(weights, &covariance);
        rows.push([
            weights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            1.0 / dot(weights, weights),
            if variance > 0.0 { dot(weights, &asset_volatility) / variance.sqrt() } else { f64::NAN },
        ]);
    }
    let column = |k: usize| rows.iter().map(|row| Cell::Float(row[k])).collect();
    Table::labelled(Some("portfolio"), &names)
        .with("max_weight", column(0))
        .with("effective_holdings", column(1))
        .with("diversification_ratio", column(2))
}

/// Return each asset's share of portfolio variance from the training model.
///
/// Shares equal w_i * (covariance @ w)_i / portfolio variance and sum to one.
/// Shares can be negative. Zero portfolio variance gives undefined shares.
///
/// The table has an unnamed asset index in the order of `analysis.mean_returns`
/// and one column per portfolio.
pub fn risk_contributions(analysis: &Analysis) -> Table {
    let (covariance, portfolios) = aligned(analysis);
    let mut table = Table::labelled(None, &analysis.mean_returns.labels);
    for (name, weights) in portfolios {
        let marginal = &covariance * DVector::from_column_slice(&weights);
        let contributions: Vec<f64> = weights.iter().zip(marginal.iter()).map(|(weight, risk)| weight * risk).collect();
        let variance: f64 = contributions.iter().sum();
        table.push(
            name,
            contributions
                .iter()
                .map(|value| Cell::Float(if variance > 0.0 { value / variance } else { f64::NAN }))
                .collect(),
        );
    }
    table
}
