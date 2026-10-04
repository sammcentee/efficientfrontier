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


def test_large_universe_uses_all_128_assets_with_analytic_optimal_weights():
    names = [f"ASSET{i:03d}" for i in range(128)]
    mu = pd.Series(np.linspace(0.08, 0.20, len(names)), index=names)
    variances = np.linspace(0.01, 0.09, len(names))
    covariance = pd.DataFrame(np.diag(variances), index=names, columns=names)
    portfolios, frontier, weights, _ = optimize(mu, covariance, frontier_points=3)

    minimum = portfolios["Minimum volatility"]
    inverse_variance = 1 / variances
    np.testing.assert_allclose(minimum.weights, inverse_variance / inverse_variance.sum(), atol=2e-6)
    tangency = (mu.to_numpy() - 0.02) / variances
    np.testing.assert_allclose(portfolios["Maximum Sharpe"].weights, tangency / tangency.sum(), atol=2e-6)
    assert (minimum.weights > 1e-4).sum() == 128
    assert list(weights.columns) == names
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-8)
    assert len(frontier) == 3
    assert frontier.iloc[-1].expected_return == pytest.approx(mu.max(), abs=1e-6)


def test_capped_portfolios_can_require_more_than_40_holdings():
    names = [f"ASSET{i:03d}" for i in range(64)]
    mu = pd.Series(0.10, index=names)
    variances = np.repeat([0.01, 0.09], 32)
    covariance = pd.DataFrame(np.diag(variances), index=names, columns=names)
    portfolios, _, weights, _ = optimize(mu, covariance, max_weight=1 / 48, frontier_points=3)

    # The cheaper-risk half hits its cap; the remainder is spread equally.
    expected = np.repeat([1 / 48, 1 / 96], 32)
    for name in ("Minimum volatility", "Maximum Sharpe"):
        np.testing.assert_allclose(portfolios[name].weights, expected, atol=2e-6)
        assert (portfolios[name].weights > 1e-4).sum() == 64
    assert weights.to_numpy().max() <= 1 / 48 + 1e-6


@pytest.mark.parametrize("risk_free_rate", [0.02, 0.15])
def test_single_asset_has_its_own_return_and_risk(risk_free_rate):
    mu = pd.Series({"ONLY": 0.12})
    covariance = pd.DataFrame([[0.04]], index=mu.index, columns=mu.index)
    portfolios, frontier, weights, warnings = optimize(mu, covariance, risk_free_rate=risk_free_rate)

    for portfolio in portfolios.values():
        np.testing.assert_allclose(portfolio.weights, [1])
        assert portfolio.expected_return == pytest.approx(0.12)
        assert portfolio.volatility == pytest.approx(0.2)
        assert portfolio.sharpe == pytest.approx((0.12 - risk_free_rate) / 0.2)
    assert len(frontier) == 1
    np.testing.assert_allclose(weights, [[1]])
    assert ("Maximum Sharpe" in portfolios) == (risk_free_rate < 0.12)
    assert bool(warnings) == (risk_free_rate > 0.12)


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


def test_indefinite_covariance_with_positive_diagonal_is_rejected():
    mu, covariance = model()
    covariance.loc["A", "B"] = covariance.loc["B", "A"] = 0.1
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


@pytest.mark.parametrize("periods_per_year", [12, 252, 365])
def test_holdout_does_not_change_fitted_weights(periods_per_year):
    prices = price_history()
    first = analyze(prices, train_fraction=2 / 3, frontier_points=3, periods_per_year=periods_per_year)
    prices.iloc[5:, 0] *= 2
    second = analyze(prices, train_fraction=2 / 3, frontier_points=3, periods_per_year=periods_per_year)
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


@pytest.mark.parametrize("frequency,periods_per_year", [("MS", 12), ("W", 52), ("D", 365.25)])
def test_selected_periods_scale_estimates_and_realized_metrics(frequency, periods_per_year):
    prices = price_history()[["A"]]
    prices.index = pd.date_range("2020-01-01", periods=len(prices), freq=frequency)
    prices.iloc[-2:, 0] = [90, 108]
    result = analyze(prices, train_fraction=2 / 3, risk_free_rate=0.12,
                     frontier_points=2, periods_per_year=periods_per_year)
    train = prices.pct_change(fill_method=None).iloc[1:5]
    pd.testing.assert_series_equal(result.mean_returns, train.mean() * periods_per_year)
    pd.testing.assert_frame_equal(result.covariance, train.cov() * periods_per_year)
    assert result.periods_per_year == periods_per_year
    row = result.holdout_metrics.loc["Equal weight"]
    annual_excess = 0.05 * periods_per_year - 0.12
    volatility = 0.3 * np.sqrt(periods_per_year / 2)
    downside = (0.1 + 0.12 / periods_per_year) * np.sqrt(periods_per_year / 2)
    assert row.total_return == pytest.approx(0.08)
    assert row.cagr == pytest.approx(1.08 ** (periods_per_year / 2) - 1)
    assert row.volatility == pytest.approx(volatility)
    assert row.sharpe == pytest.approx(annual_excess / volatility)
    assert row.sortino == pytest.approx(annual_excess / downside)
    assert row.max_drawdown == pytest.approx(-0.1)
    assert row.calmar == pytest.approx(row.cagr / 0.1)


def test_default_periods_match_explicit_trading_year():
    implicit = analyze(price_history(), frontier_points=3)
    explicit = analyze(price_history(), frontier_points=3, periods_per_year=252)
    assert implicit.periods_per_year == 252
    for name in ("mean_returns", "covariance", "frontier", "equity", "holdout_metrics"):
        first, second = getattr(implicit, name), getattr(explicit, name)
        if isinstance(first, pd.Series):
            pd.testing.assert_series_equal(first, second)
        else:
            pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("periods_per_year", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(True), "12", None, 12j])
def test_invalid_annualization_is_rejected(periods_per_year):
    with pytest.raises(ValueError, match="periods_per_year"):
        analyze(price_history(), periods_per_year=periods_per_year)


@pytest.mark.parametrize("kind", ["boolean", "complex", "object_boolean", "object_complex"])
def test_api_rejects_boolean_and_complex_prices(kind):
    prices = price_history()
    if kind == "boolean":
        prices = prices > 0
    elif kind == "complex":
        prices = prices.astype(complex) + 1j
    else:
        prices = prices.astype(object)
        prices.iloc[0, 0] = True if kind == "object_boolean" else 100 + 1j
    with pytest.raises(ValueError, match="Boolean or complex"):
        analyze(prices)


@pytest.mark.parametrize("holdout", [[110, 121], [100, 100]])
def test_sortino_and_calmar_are_undefined_without_downside(holdout):
    prices = price_history()[["A"]]
    prices.iloc[-2:, 0] = holdout
    result = analyze(prices, train_fraction=2 / 3, risk_free_rate=0, periods_per_year=12)
    row = result.holdout_metrics.loc["Equal weight"]
    assert np.isnan(row.sortino)
    assert np.isnan(row.calmar)
    if holdout == [100, 100]:
        assert np.isnan(row.sharpe)


def test_downside_ratios_keep_the_sign_of_losses_when_volatility_is_zero():
    prices = price_history()[["A"]]
    prices.iloc[-2:, 0] = [90, 81]
    result = analyze(prices, train_fraction=2 / 3, risk_free_rate=0, periods_per_year=12)
    row = result.holdout_metrics.loc["Equal weight"]
    assert np.isnan(row.sharpe)
    assert row.sortino == pytest.approx(-np.sqrt(12))
    assert row.calmar == pytest.approx((0.81 ** 6 - 1) / 0.19)


def test_short_and_underdetermined_training_samples_warn_without_removing_assets():
    prices = price_history().iloc[:, [0, 1, 0, 1]].copy()
    prices.columns = ["A", "B", "C", "D"]
    result = analyze(prices, train_fraction=2 / 3, frontier_points=2)
    assert any("fewer than 30" in message for message in result.warnings)
    assert any("singular before shrinkage" in message for message in result.warnings)
    assert result.mean_returns.index.tolist() == ["A", "B", "C", "D"]


def test_sufficient_training_sample_does_not_warn_about_sample_size():
    rng = np.random.default_rng(81)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, (70, 2)), axis=0)),
                          index=pd.bdate_range("2020-01-01", periods=70), columns=["A", "B"])
    result = analyze(prices, frontier_points=2)
    assert not any("fewer than 30" in message or "singular before shrinkage" in message
                   for message in result.warnings)
