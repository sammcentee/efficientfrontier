import numpy as np
import pandas as pd
import pytest

from efficient_frontier.core import analyze, optimize


def model():
    names = ["A", "B"]
    return pd.Series([0.1, 0.2], index=names), pd.DataFrame(np.diag([0.04, 0.09]), index=names, columns=names)


def price_history():
    return pd.DataFrame(
        {"A": [100, 110, 99, 108, 100, 80, 100], "B": [100, 105, 102, 106, 100, 120, 100]},
        index=pd.bdate_range("2020-01-01", periods=7),
        dtype=float,
    )


def test_analytic_minimum_variance_and_maximum_sharpe():
    mu, covariance = model()
    portfolios, frontier, _, _ = optimize(mu, covariance, risk_free_rate=0.02, frontier_points=7)
    minimum = portfolios["Minimum volatility"]
    np.testing.assert_allclose(minimum.weights, [9 / 13, 4 / 13], atol=2e-5)
    assert minimum.volatility == pytest.approx(np.sqrt(0.04 * 0.09 / 0.13), abs=1e-8)
    maximum = portfolios["Maximum Sharpe"]
    np.testing.assert_allclose(maximum.weights, [0.5, 0.5], atol=2e-5)
    assert maximum.sharpe == pytest.approx(0.13 / np.sqrt(0.0325), abs=1e-8)
    assert len(frontier) == 7
    assert frontier.iloc[-1].expected_return == pytest.approx(0.2, abs=1e-6)


def test_position_cap_and_frontier_are_feasible_and_efficient():
    mu, covariance = model()
    portfolios, frontier, weights, _ = optimize(mu, covariance, max_weight=0.6, frontier_points=9)
    np.testing.assert_allclose(portfolios["Minimum volatility"].weights, [0.6, 0.4], atol=2e-5)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-8)
    assert weights.to_numpy().min() >= -1e-8
    assert weights.to_numpy().max() <= 0.600001
    np.testing.assert_allclose(frontier.expected_return, np.linspace(0.14, 0.16, 9), atol=2e-6)
    expected_b = (frontier.expected_return - 0.1) / 0.1
    np.testing.assert_allclose(weights.B, expected_b, atol=1e-8)
    np.testing.assert_allclose(frontier.volatility, np.sqrt(0.04 * (1 - expected_b) ** 2 + 0.09 * expected_b ** 2))
    assert np.all(np.diff(frontier.volatility) >= -1e-8)


def test_infeasible_cap_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        optimize(*model(), max_weight=0.4)


def test_no_positive_excess_return_omits_maximum_sharpe():
    portfolios, _, _, warnings = optimize(*model(), risk_free_rate=0.3, frontier_points=3)
    assert "Maximum Sharpe" not in portfolios
    assert any("positive expected excess return" in warning for warning in warnings)


def test_maximum_sharpe_handles_small_positive_excess_returns():
    mu, covariance = model()
    mu[:] = 0.1
    portfolios, frontier, _, _ = optimize(mu, covariance, risk_free_rate=0.1 - 1e-9, frontier_points=3)
    maximum = portfolios["Maximum Sharpe"]
    np.testing.assert_allclose(maximum.weights, [9 / 13, 4 / 13], atol=2e-5)
    assert maximum.sharpe > 0
    assert len(frontier) == 1


def test_uniform_covariance_scaling_preserves_optimal_weights():
    mu, covariance = model()
    normal, _, _, _ = optimize(mu, covariance, frontier_points=3)
    small, _, _, _ = optimize(mu, covariance * 1e-8, frontier_points=3)
    for name in normal:
        np.testing.assert_allclose(normal[name].weights, small[name].weights, atol=2e-5)
        assert small[name].volatility == pytest.approx(normal[name].volatility * 1e-4, rel=1e-7)


def test_covariance_alignment_uses_labels():
    mu, covariance = model()
    portfolios, _, _, _ = optimize(mu, covariance.loc[["B", "A"], ["B", "A"]], frontier_points=3)
    np.testing.assert_allclose(portfolios["Minimum volatility"].weights, [9 / 13, 4 / 13], atol=2e-5)


def test_invalid_covariance_is_rejected():
    mu, covariance = model()
    covariance.loc["A", "A"] = -0.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        optimize(mu, covariance)


def test_equal_returns_and_singular_covariance_work():
    mu, covariance = model()
    mu[:] = 0.1
    covariance.loc[:, :] = 0.04
    portfolios, frontier, _, _ = optimize(mu, covariance, frontier_points=3)
    assert len(frontier) == 1
    assert portfolios["Minimum volatility"].volatility == pytest.approx(0.2)
    assert portfolios["Maximum Sharpe"].sharpe == pytest.approx(0.4)


def test_zero_volatility_has_undefined_sharpe():
    mu, covariance = model()
    covariance.loc[:, :] = 0
    portfolios, _, _, warnings = optimize(mu, covariance, frontier_points=3)
    assert np.isnan(portfolios["Minimum volatility"].sharpe)
    assert any("undefined" in warning for warning in warnings)


def test_training_arithmetic_means_and_diagonal_shrinkage():
    prices = price_history()
    result = analyze(prices, train_fraction=2 / 3, shrinkage=0.25, frontier_points=3)
    expected_returns = prices.pct_change(fill_method=None).iloc[1:5]
    pd.testing.assert_series_equal(result.mean_returns, expected_returns.mean() * 252)
    sample = expected_returns.cov() * 252
    expected = sample * 0.75
    for name in expected.columns:
        expected.loc[name, name] = sample.loc[name, name]
    pd.testing.assert_frame_equal(result.covariance, expected)
    assert result.train_returns.index.max() < result.test_returns.index.min()


def test_holdout_does_not_change_fitted_weights():
    prices = price_history()
    first = analyze(prices, train_fraction=2 / 3, frontier_points=3)
    prices.iloc[5:, 0] *= 2
    second = analyze(prices, train_fraction=2 / 3, frontier_points=3)
    pd.testing.assert_series_equal(first.mean_returns, second.mean_returns)
    pd.testing.assert_frame_equal(first.covariance, second.covariance)
    for name in first.portfolios:
        np.testing.assert_allclose(first.portfolios[name].weights, second.portfolios[name].weights)


def test_holdout_is_buy_and_hold_and_includes_boundary_return():
    prices = price_history()
    result = analyze(prices, train_fraction=2 / 3, frontier_points=3)
    # Equal shares begin at A=B=100. After 80/120 then 100/100, wealth stays one.
    # Rebalancing daily would incorrectly produce a 4.17% gain on the second day.
    np.testing.assert_allclose(result.equity["Equal weight"], [1, 1, 1])
    assert result.equity.index[0] == prices.index[4]
    assert result.test_returns.iloc[0].A == pytest.approx(-0.2)
    assert result.holdout_metrics.loc["Equal weight", "total_return"] == pytest.approx(0)
    assert result.holdout_metrics.loc["Equal weight", "cagr"] == pytest.approx(0)


def test_drawdown_includes_initial_capital():
    prices = price_history()
    prices.iloc[5] = [80, 80]
    prices.iloc[6] = [90, 90]
    result = analyze(prices, train_fraction=2 / 3, frontier_points=3)
    assert result.holdout_metrics.loc["Equal weight", "max_drawdown"] == pytest.approx(-0.2)
    assert result.holdout_metrics.loc["Equal weight", "total_return"] == pytest.approx(-0.1)


@pytest.mark.parametrize("value", [np.nan, np.inf, 0, -1])
def test_invalid_prices_are_rejected(value):
    prices = price_history()
    prices.iloc[2, 0] = value
    with pytest.raises(ValueError, match="finite and strictly positive"):
        analyze(prices)
