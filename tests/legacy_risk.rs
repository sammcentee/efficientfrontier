//! Port of `tests/test_risk.py`.

mod legacy_common;

use efficient_frontier::core::{Analysis, AnalyzeOptions, Frontier, Portfolio, analyze};
use efficient_frontier::frame::{Frame, LabeledMatrix, Series, Table};
use efficient_frontier::risk::{risk_contributions, risk_summary};
use indexmap::IndexMap;
use legacy_common::{assert_allclose, business_days, date, numbers, test_data};
use nalgebra::{DMatrix, DVector};

/// An analysis that holds only one portfolio and its training covariance.
fn model(covariance: &[f64], weights: &[f64]) -> Analysis {
    let n = weights.len();
    let labels: Vec<String> = (1..=n).map(|i| format!("Asset {i}")).collect();
    let matrix = DMatrix::from_row_slice(n, n, covariance);
    let w = DVector::from_column_slice(weights);
    let volatility = (w.transpose() * &matrix * &w)[(0, 0)].sqrt();
    let portfolio = Portfolio {
        name: "Portfolio".into(),
        weights: Series::new(labels.clone(), weights.to_vec()),
        expected_return: 0.1,
        volatility,
        sharpe: 0.5,
    };
    let empty = || Frame::new(Vec::new(), labels.clone(), DMatrix::zeros(0, n));
    Analysis {
        portfolios: IndexMap::from([(portfolio.name.clone(), portfolio)]),
        frontier: Frontier {
            labels: labels.clone(),
            expected_return: Vec::new(),
            volatility: Vec::new(),
            sharpe: Vec::new(),
            weights: Vec::new(),
        },
        train_returns: empty(),
        test_returns: empty(),
        equity: Frame::new(Vec::new(), Vec::new(), DMatrix::zeros(0, 0)),
        holdout_metrics: IndexMap::new(),
        mean_returns: Series::new(labels.clone(), vec![0.1; n]),
        covariance: LabeledMatrix::square(labels, matrix),
        warnings: Vec::new(),
        periods_per_year: 252.0,
        risk_free_rate: 0.02,
    }
}

fn row(table: &Table, label: &str) -> Vec<f64> {
    let i = table.row_of(label).unwrap();
    table.data.iter().map(|column| column[i].as_f64().unwrap()).collect()
}

/// Every value of a numeric table, row by row.
fn values(table: &Table) -> Vec<f64> {
    (0..table.nrows()).flat_map(|i| table.data.iter().map(move |column| column[i].as_f64().unwrap())).collect()
}

#[test]
fn diagonal_covariance_has_analytic_risk_shares_and_diversification() {
    let analysis = model(&[0.04, 0.0, 0.0, 0.09], &[0.6, 0.4]);
    assert_allclose(&risk_contributions(&analysis).f64s("Portfolio"), &[0.5, 0.5], 0.0, 1e-7);
    let summary = risk_summary(&analysis);
    assert_eq!(summary.columns, ["max_weight", "effective_holdings", "diversification_ratio"]);
    assert_eq!(summary.index_names, [Some("portfolio".to_string())]);
    assert_allclose(&row(&summary, "Portfolio"), &[0.6, 1.0 / 0.52, 2f64.sqrt()], 1e-12, 1e-6);
}

#[test]
fn risk_shares_reconcile_for_each_optimized_portfolio() {
    let returns: Vec<Vec<f64>> = test_data("risk_returns").as_array().unwrap().iter().map(numbers).collect();
    let names = ["Stocks", "Bonds", "Gold"];
    let columns = returns
        .iter()
        .zip(names)
        .map(|(column, name)| {
            let mut growth = 1.0;
            let path = column
                .iter()
                .map(|value| {
                    growth *= 1.0 + value;
                    100.0 * growth
                })
                .collect();
            (name.to_string(), path)
        })
        .collect();
    let prices = Frame::from_columns(business_days(date(2020, 1, 1), 120), columns);
    let analysis = analyze(&prices, &AnalyzeOptions { frontier_points: 3, ..Default::default() }).unwrap();
    let contributions = risk_contributions(&analysis);
    assert_eq!(contributions.columns, analysis.portfolios.keys().cloned().collect::<Vec<_>>());
    assert_eq!(risk_summary(&analysis).index_labels(), contributions.columns);
    for (name, portfolio) in &analysis.portfolios {
        let shares = contributions.f64s(name);
        assert!((shares.iter().sum::<f64>() - 1.0).abs() <= 1e-12, "{name}");
        let marginal = &analysis.covariance.values * DVector::from_column_slice(&portfolio.weights.values);
        let expected: Vec<f64> = portfolio.weights.values.iter().zip(marginal.iter()).map(|(w, m)| w * m).collect();
        let reconstructed: Vec<f64> = shares.iter().map(|share| share * portfolio.volatility.powi(2)).collect();
        assert_allclose(&reconstructed, &expected, 1e-12, 1e-7);
    }
}

#[test]
fn hedge_has_negative_risk_share() {
    let analysis = model(&[0.04, -0.01, -0.01, 0.01], &[0.8, 0.2]);
    let shares = risk_contributions(&analysis).f64s("Portfolio");
    assert_allclose(&shares, &[20.0 / 19.0, -1.0 / 19.0], 0.0, 1e-7);
    assert!((shares.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
}

#[test]
fn single_asset_has_unit_risk_share_and_diversification() {
    let analysis = model(&[0.04], &[1.0]);
    assert_allclose(&values(&risk_contributions(&analysis)), &[1.0], 0.0, 1e-7);
    assert_allclose(&values(&risk_summary(&analysis)), &[1.0, 1.0, 1.0], 0.0, 1e-7);
}

#[test]
fn zero_portfolio_variance_has_undefined_risk_measures() {
    for covariance in [[0.0; 4], [0.04, -0.04, -0.04, 0.04]] {
        let analysis = model(&covariance, &[0.5, 0.5]);
        assert!(values(&risk_contributions(&analysis)).iter().all(|value| value.is_nan()));
        let summary = row(&risk_summary(&analysis), "Portfolio");
        assert!(summary[2].is_nan());
        assert!((summary[0] - 0.5).abs() <= 1e-12);
        assert!((summary[1] - 2.0).abs() <= 1e-12);
    }
}

#[test]
fn covariance_and_weights_align_by_asset_label() {
    let mut analysis = model(&[0.04, 0.0, 0.0, 0.09], &[0.6, 0.4]);
    let expected_summary = risk_summary(&analysis);
    let expected_contributions = risk_contributions(&analysis);
    // Reverse the covariance rows only, and the weights.
    let covariance = &analysis.covariance;
    analysis.covariance = LabeledMatrix::new(
        covariance.index.iter().rev().cloned().collect(),
        covariance.columns.clone(),
        DMatrix::from_fn(2, 2, |i, j| covariance.values[(1 - i, j)]),
    );
    let weights = &analysis.portfolios["Portfolio"].weights;
    let reversed =
        Series::new(weights.labels.iter().rev().cloned().collect(), weights.values.iter().rev().copied().collect());
    analysis.portfolios["Portfolio"].weights = reversed;
    assert_eq!(risk_summary(&analysis), expected_summary);
    assert_eq!(risk_contributions(&analysis), expected_contributions);
    let mean = &analysis.mean_returns;
    analysis.mean_returns =
        Series::new(mean.labels.iter().rev().cloned().collect(), mean.values.iter().rev().copied().collect());
    assert_eq!(risk_contributions(&analysis), expected_contributions.take_rows(&[1, 0]));
}

#[test]
fn risk_measures_do_not_change_with_covariance_scale() {
    let mut analysis = model(&[0.04, 0.0, 0.0, 0.09], &[0.6, 0.4]);
    let expected_summary = risk_summary(&analysis);
    let expected_contributions = risk_contributions(&analysis);
    analysis.covariance.values *= 1e-16;
    assert_allclose(&values(&risk_summary(&analysis)), &values(&expected_summary), 0.0, 1e-5);
    assert_allclose(&values(&risk_contributions(&analysis)), &values(&expected_contributions), 0.0, 1e-5);
}
