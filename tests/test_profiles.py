import numpy as np
import pandas as pd
import pytest

from efficient_frontier.profiles import build_profiles


def prices_from_returns(returns, columns=None):
    values = np.asarray(returns, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return pd.DataFrame(
        100 * np.vstack([np.ones(values.shape[1]), np.cumprod(1 + values, axis=0)]),
        columns=columns,
        index=pd.date_range("2020-01-31", periods=len(values) + 1, freq="ME"),
    )


def regime_prices():
    steady = 0.01 + np.tile([-0.015, 0.015], 9)
    transient = np.repeat([0.03, -0.02, 0.04], 6) + np.tile([-0.001, 0.001], 9)
    return prices_from_returns(np.column_stack([steady, transient]), ["Steady", "Transient"])


def test_analytic_profiles_prefer_consistent_windows_over_the_full_sample_winner():
    prices = regime_prices()
    result = build_profiles(prices, shrinkage=1, periods_per_year=12)
    variance = prices.pct_change(fill_method=None).iloc[1:].var()
    low = np.array([variance.Transient, variance.Steady]) / variance.sum()
    extreme = np.array([1.0, 0.0])
    np.testing.assert_allclose(result.portfolios["Low"].weights, low, atol=2e-6)
    np.testing.assert_allclose(result.portfolios["Extreme"].weights, extreme, atol=2e-6)
    np.testing.assert_allclose(result.portfolios["Medium"].weights, (low + extreme) / 2, atol=2e-6)
    assert result.summary.loc["Extreme", "worst_window_return"] == pytest.approx(0.12, abs=1e-8)
    assert result.summary.loc["Extreme", "expected_return"] < result.summary.loc["Low", "expected_return"]
    assert np.all(np.diff(result.frontier.volatility) >= -1e-9)
    assert np.all(np.diff(result.frontier.worst_window_return) >= -1e-9)
    np.testing.assert_allclose(
        result.frontier.worst_window_return,
        np.linspace(result.frontier.worst_window_return.iloc[0], result.frontier.worst_window_return.iloc[-1], 21),
        atol=2e-7,
    )
    assert list(result.portfolios) == ["Low", "Medium", "Extreme"]
    assert list(result.summary.index) == list(result.portfolios)


def test_position_cap_applies_to_every_frontier_point():
    result = build_profiles(regime_prices(), max_weight=0.9, shrinkage=1, periods_per_year=12)
    np.testing.assert_allclose(result.frontier_weights.sum(axis=1), 1, atol=1e-12)
    assert result.frontier_weights.to_numpy().min() >= 0
    assert result.frontier_weights.to_numpy().max() <= 0.900001
    np.testing.assert_allclose(result.portfolios["Extreme"].weights, [0.9, 0.1], atol=2e-6)
    assert np.all(np.diff(result.frontier.volatility) >= -1e-9)


def test_binding_cap_gives_identical_profiles_with_a_warning():
    result = build_profiles(regime_prices(), max_weight=0.5, frontier_points=3)
    np.testing.assert_allclose(result.frontier_weights, 0.5, atol=1e-9)
    assert any("no distinct risk levels" in warning for warning in result.warnings)


def test_windows_cover_every_return_once_and_include_the_latest_date():
    prices = regime_prices()
    prices.loc[prices.index[-1] + pd.offsets.MonthEnd()] = prices.iloc[-1] * [1.02, 0.96]
    result = build_profiles(prices, periods_per_year=12, frontier_points=3)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    assert list(result.windows.observations) == [7, 6, 6]
    assert result.observations == 19
    assert result.as_of == prices.index[-1].date().isoformat()
    for number, positions in enumerate(np.array_split(np.arange(len(returns)), 3), start=1):
        chunk = returns.iloc[positions]
        row = result.windows.loc[number]
        assert row.start == chunk.index[0].date().isoformat()
        assert row.end == chunk.index[-1].date().isoformat()
        assert row.years == pytest.approx(len(chunk) / 12)
        for name, portfolio in result.portfolios.items():
            assert result.window_returns.loc[number, name] == pytest.approx(chunk.mean() @ portfolio.weights * 12)
    np.testing.assert_allclose(result.window_returns.min(), result.summary.worst_window_return)


def test_price_units_and_asset_order_do_not_change_profiles_or_mutate_input():
    prices = regime_prices()
    original = prices.copy(deep=True)
    reference = build_profiles(prices, frontier_points=3)
    rescaled = prices.mul(pd.Series({"Steady": 0.01, "Transient": 1000})).iloc[:, ::-1]
    reordered = build_profiles(rescaled, frontier_points=3)
    pd.testing.assert_frame_equal(prices, original)
    assert list(reordered.frontier_weights.columns) == ["Transient", "Steady"]
    for name in reference.portfolios:
        np.testing.assert_allclose(
            reference.portfolios[name].weights,
            reordered.portfolios[name].weights.loc[prices.columns],
            atol=2e-6,
        )
    pd.testing.assert_frame_equal(reference.summary, reordered.summary, atol=1e-6, rtol=1e-6)


def test_six_returns_and_one_asset_are_valid():
    prices = prices_from_returns([0.01, -0.02, 0.01, 0.02, 0.03, -0.01], ["Only"])
    result = build_profiles(prices, periods_per_year=12)
    np.testing.assert_allclose(result.frontier_weights, 1)
    assert len(result.frontier) == 21
    assert list(result.windows.observations) == [2, 2, 2]
    assert result.summary.effective_holdings.eq(1).all()
    assert any("no distinct risk levels" in warning for warning in result.warnings)
    assert any("less than one year" in warning for warning in result.warnings)


def test_zero_covariance_preserves_undefined_sharpe_and_flat_profiles():
    result = build_profiles(prices_from_returns(np.zeros((9, 2)), ["A", "B"]), frontier_points=3)
    assert result.frontier.volatility.eq(0).all()
    assert result.frontier.worst_window_return.eq(0).all()
    assert result.frontier.sharpe.isna().all()
    np.testing.assert_allclose(result.frontier_weights, np.tile(result.frontier_weights.iloc[0], (3, 1)))
    assert any("no distinct risk levels" in warning for warning in result.warnings)
    assert any("No feasible portfolio" in warning for warning in result.warnings)


def test_non_daily_annualization_scales_risk_and_return_without_changing_weights():
    prices = regime_prices()
    monthly = build_profiles(prices, periods_per_year=12, frontier_points=3)
    daily = build_profiles(prices, periods_per_year=252, frontier_points=3)
    np.testing.assert_allclose(daily.frontier_weights, monthly.frontier_weights, atol=1e-6)
    np.testing.assert_allclose(daily.summary.expected_return, monthly.summary.expected_return * 21, rtol=1e-5)
    np.testing.assert_allclose(daily.summary.worst_window_return, monthly.summary.worst_window_return * 21, rtol=1e-5)
    np.testing.assert_allclose(daily.summary.volatility, monthly.summary.volatility * np.sqrt(21), rtol=1e-5)
    assert monthly.settings["periods_per_year"] == 12
    assert monthly.settings["window_count"] == 3
    assert monthly.settings["selection_method"] == "worst_window_return_frontier"


def test_three_and_twenty_one_points_select_the_same_profiles():
    prices = regime_prices()
    short = build_profiles(prices, frontier_points=3)
    full = build_profiles(prices, frontier_points=21)
    for name in short.portfolios:
        np.testing.assert_allclose(short.portfolios[name].weights, full.portfolios[name].weights, atol=2e-6)


@pytest.mark.parametrize("return_scale", [1.0, 1e-6])
def test_extreme_profile_minimizes_variance_when_the_best_return_has_multiple_solutions(return_scale):
    first = 0.01 + np.tile([-0.005, 0.005, -0.005, 0.005], 3)
    second = 0.01 + np.tile([-0.015, -0.015, 0.015, 0.015], 3)
    third = np.repeat([-0.02, 0.02, 0.02], 4)
    prices = prices_from_returns(np.column_stack([first, second, third]) * return_scale, ["A", "B", "C"])
    result = build_profiles(prices, shrinkage=1, periods_per_year=12, frontier_points=3)
    np.testing.assert_allclose(result.portfolios["Extreme"].weights, [0.9, 0.1, 0], atol=3e-6)
    assert result.summary.loc["Extreme", "worst_window_return"] == pytest.approx(0.12 * return_scale, rel=1e-6)


def test_tiny_returns_preserve_the_frontier_weights():
    prices = regime_prices()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    tiny = prices_from_returns(returns.to_numpy() * 1e-6, prices.columns)
    reference = build_profiles(prices, frontier_points=3)
    result = build_profiles(tiny, frontier_points=3)
    np.testing.assert_allclose(result.frontier_weights, reference.frontier_weights, atol=3e-6)
    np.testing.assert_allclose(result.summary.volatility, reference.summary.volatility * 1e-6, rtol=1e-5)


def test_nonpositive_worst_window_returns_remain_feasible_with_a_warning():
    prices = regime_prices()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    falling = prices_from_returns(returns.to_numpy() - 0.1, prices.columns)
    result = build_profiles(falling, periods_per_year=12, frontier_points=3)
    assert result.summary.worst_window_return.lt(0).all()
    np.testing.assert_allclose(result.portfolios["Extreme"].weights, [1, 0], atol=2e-6)
    assert any("No feasible portfolio" in warning for warning in result.warnings)


def test_approximate_solver_status_is_visible_in_the_result(monkeypatch):
    from efficient_frontier import profiles

    original = profiles._solve

    def approximate_solve(problem):
        original(problem)
        problem._status = "optimal_inaccurate"

    monkeypatch.setattr(profiles, "_solve", approximate_solve)
    result = build_profiles(regime_prices(), frontier_points=3)
    messages = [message for message in result.warnings if "limited numerical accuracy" in message]
    assert len(messages) == 1
    assert "optimum can be approximate" in messages[0]
    np.testing.assert_allclose(result.frontier_weights.sum(axis=1), 1)


@pytest.mark.parametrize("options, message", [
    ({"max_weight": 0}, "max_weight"),
    ({"max_weight": 1.1}, "max_weight"),
    ({"max_weight": 0.4}, "infeasible"),
    ({"shrinkage": -0.1}, "shrinkage"),
    ({"shrinkage": float("nan")}, "shrinkage"),
    ({"risk_free_rate": float("inf")}, "risk_free_rate"),
    ({"periods_per_year": 0}, "periods_per_year"),
    ({"periods_per_year": True}, "periods_per_year"),
    ({"frontier_points": 2}, "odd integer"),
    ({"frontier_points": 4}, "odd integer"),
    ({"frontier_points": True}, "odd integer"),
    ({"frontier_points": 3.0}, "odd integer"),
])
def test_invalid_settings_fail_with_a_clear_error(options, message):
    with pytest.raises(ValueError, match=message):
        build_profiles(regime_prices(), **options)


def test_profiles_reject_less_than_six_returns():
    with pytest.raises(ValueError, match="at least six returns"):
        build_profiles(regime_prices().iloc[:6])


def test_profiles_reject_missing_prices_without_filling_them():
    prices = regime_prices()
    prices.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="resolve missing data"):
        build_profiles(prices)


def test_profiles_reject_dates_out_of_order():
    with pytest.raises(ValueError, match="increasing"):
        build_profiles(regime_prices().iloc[::-1])
