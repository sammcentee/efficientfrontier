import numpy as np
import pandas as pd
import pytest

from efficient_frontier.core import Analysis, Portfolio, analyze
from efficient_frontier.risk import risk_contributions, risk_summary


def model(covariance, weights):
    labels = pd.Index([f"Asset {index + 1}" for index in range(len(weights))])
    weights = pd.Series(weights, index=labels)
    covariance = pd.DataFrame(covariance, index=labels, columns=labels, dtype=float)
    portfolio = Portfolio("Portfolio", weights, 0.1, np.sqrt(weights @ covariance @ weights), 0.5)
    return Analysis(
        portfolios={portfolio.name: portfolio},
        frontier=pd.DataFrame(),
        frontier_weights=pd.DataFrame(),
        train_returns=pd.DataFrame(),
        test_returns=pd.DataFrame(),
        equity=pd.DataFrame(),
        holdout_metrics=pd.DataFrame(),
        mean_returns=pd.Series(0.1, index=labels),
        covariance=covariance,
        warnings=[],
    )


def test_diagonal_covariance_has_analytic_risk_shares_and_diversification():
    analysis = model(np.diag([0.04, 0.09]), [0.6, 0.4])
    np.testing.assert_allclose(risk_contributions(analysis)["Portfolio"], [0.5, 0.5])
    summary = risk_summary(analysis).loc["Portfolio"]
    assert summary.max_weight == pytest.approx(0.6)
    assert summary.effective_holdings == pytest.approx(1 / 0.52)
    assert summary.diversification_ratio == pytest.approx(np.sqrt(2))


def test_risk_shares_reconcile_for_each_optimized_portfolio():
    rng = np.random.default_rng(42)
    returns = rng.normal([0.0004, 0.0002, 0.0003], [0.015, 0.006, 0.009], (120, 3))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        columns=["Stocks", "Bonds", "Gold"],
        index=pd.bdate_range("2020-01-01", periods=len(returns)),
    )
    analysis = analyze(prices, frontier_points=3)
    contributions = risk_contributions(analysis)
    np.testing.assert_allclose(contributions.sum(), 1, atol=1e-12)
    assert list(contributions.columns) == list(analysis.portfolios)
    assert list(risk_summary(analysis).index) == list(analysis.portfolios)
    for name, portfolio in analysis.portfolios.items():
        marginal_variance = analysis.covariance @ portfolio.weights
        reconstructed = contributions[name] * portfolio.volatility ** 2
        np.testing.assert_allclose(reconstructed, portfolio.weights * marginal_variance, atol=1e-12)


def test_hedge_has_negative_risk_share():
    analysis = model([[0.04, -0.01], [-0.01, 0.01]], [0.8, 0.2])
    contributions = risk_contributions(analysis)["Portfolio"]
    np.testing.assert_allclose(contributions, [20 / 19, -1 / 19])
    assert contributions.sum() == pytest.approx(1)


def test_single_asset_has_unit_risk_share_and_diversification():
    analysis = model([[0.04]], [1.0])
    np.testing.assert_allclose(risk_contributions(analysis), [[1.0]])
    np.testing.assert_allclose(risk_summary(analysis), [[1.0, 1.0, 1.0]])


@pytest.mark.parametrize("covariance", [np.zeros((2, 2)), [[0.04, -0.04], [-0.04, 0.04]]])
def test_zero_portfolio_variance_has_undefined_risk_measures(covariance):
    analysis = model(covariance, [0.5, 0.5])
    assert risk_contributions(analysis).isna().all().all()
    summary = risk_summary(analysis).loc["Portfolio"]
    assert np.isnan(summary.diversification_ratio)
    assert summary.max_weight == pytest.approx(0.5)
    assert summary.effective_holdings == pytest.approx(2)


def test_covariance_and_weights_align_by_asset_label():
    analysis = model(np.diag([0.04, 0.09]), [0.6, 0.4])
    expected_summary = risk_summary(analysis)
    expected_contributions = risk_contributions(analysis)
    analysis.covariance = analysis.covariance.iloc[::-1, :]
    analysis.portfolios["Portfolio"].weights = analysis.portfolios["Portfolio"].weights.iloc[::-1]
    pd.testing.assert_frame_equal(risk_summary(analysis), expected_summary)
    pd.testing.assert_frame_equal(risk_contributions(analysis), expected_contributions)
    analysis.mean_returns = analysis.mean_returns.iloc[::-1]
    pd.testing.assert_frame_equal(risk_contributions(analysis), expected_contributions.iloc[::-1])


def test_risk_measures_do_not_change_with_covariance_scale():
    analysis = model(np.diag([0.04, 0.09]), [0.6, 0.4])
    expected_summary = risk_summary(analysis)
    expected_contributions = risk_contributions(analysis)
    analysis.covariance *= 1e-16
    pd.testing.assert_frame_equal(risk_summary(analysis), expected_summary)
    pd.testing.assert_frame_equal(risk_contributions(analysis), expected_contributions)
