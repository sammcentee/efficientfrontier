from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from efficient_frontier import backtest
from efficient_frontier.backtest import METHOD_NAMES, run_backtests


def history():
    return pd.DataFrame(
        {"A": [100, 103, 101, 100, 100, 200, 100, 100],
         "B": [100, 101, 104, 100, 100, 100, 100, 100]},
        index=pd.bdate_range("2020-01-01", periods=8), dtype=float,
    )


def test_buy_and_hold_drift_and_daily_rebalance_have_analytic_paths():
    study = run_backtests(history(), train_fraction=0.5, rebalance_every=1, cost_bps=0)
    buy = "Buy and hold · Equal weight"
    fixed = "Fixed rebalance · Equal weight"
    np.testing.assert_allclose(study.equity[buy], [1, 1, 1.5, 1, 1])
    np.testing.assert_allclose(study.equity[fixed], [1, 1, 1.5, 1.125, 1.125])
    np.testing.assert_allclose(study.holdings[buy].average_weight, [5 / 12, 1 / 3])
    np.testing.assert_allclose(study.holdings[buy].end_weight, [0.5, 0.5])
    np.testing.assert_allclose(study.holdings[fixed].pnl_contribution, [0.125, 0])
    assert study.metrics.loc[buy, "rebalance_count"] == 1
    assert study.metrics.loc[fixed, "rebalance_count"] == 3
    assert study.metrics.loc[fixed, "total_turnover"] == pytest.approx(1 + 1 / 3 + 1 / 3)
    assert study.allocations[fixed].index[-1] == history().index[-2]
    assert len(study.equity.columns) == 12


def test_entry_cost_and_rebalances_conserve_wealth_and_contributions():
    study = run_backtests(history(), train_fraction=0.5, rebalance_every=1, cost_bps=100)
    assert (study.equity.iloc[0] == 1).all()
    np.testing.assert_allclose(study.equity.iloc[1], 1 / 1.01)
    for key, trades in study.trades.items():
        np.testing.assert_allclose(trades.nav_before - trades.nav_after, trades.cost, atol=1e-14)
        np.testing.assert_allclose(trades.cost, 0.01 * trades.turnover * trades.nav_before, atol=1e-14)
        assert study.metrics.loc[key, "total_cost"] == pytest.approx(trades.cost.sum())
        assert study.metrics.loc[key, "total_return"] == pytest.approx(
            study.holdings[key].pnl_contribution.sum() - trades.cost.sum(), abs=1e-13,
        )
        assert study.metrics.loc[key, "max_drawdown"] <= -(1 - 1 / 1.01)


def test_execution_day_jump_is_not_earned_before_initial_entry():
    prices = history()
    prices.iloc[4, 0] = 200
    study = run_backtests(prices, train_fraction=0.5, rebalance_every=2, cost_bps=0)
    np.testing.assert_allclose(study.equity["Buy and hold · Equal weight"], [1, 1, 1, 0.75, 0.75])
    assert study.allocations["Fixed rebalance · Equal weight"].index.tolist() == [
        prices.index[4], prices.index[6],
    ]


def test_realized_concentration_includes_drift_before_rebalancing():
    study = run_backtests(history(), train_fraction=0.5, rebalance_every=1, max_weight=0.5, cost_bps=0)
    buy = study.holdings["Buy and hold · Equal weight"]
    fixed = study.holdings["Fixed rebalance · Equal weight"]
    np.testing.assert_allclose(buy.max_target_weight, [0.5, 0.5])
    np.testing.assert_allclose(buy.max_realized_weight, [2 / 3, 0.5])
    np.testing.assert_allclose(fixed.max_realized_weight, [2 / 3, 2 / 3])
    assert study.settings["trading_days_per_year"] == 252
    assert study.settings["cash_interest_rate"] == 0


def test_full_switch_charges_both_sell_and_buy_legs(monkeypatch):
    def changing_optimizer(mean, covariance, *args, **kwargs):
        target = pd.Series([1.0, 0.0] if mean.A < 10 else [0.0, 1.0], index=mean.index)
        portfolios = {name: SimpleNamespace(weights=target) for name in backtest.PORTFOLIO_NAMES}
        return portfolios, pd.DataFrame(), pd.DataFrame(), []

    monkeypatch.setattr(backtest, "optimize", changing_optimizer)
    study = run_backtests(history(), train_fraction=0.5, rebalance_every=1, cost_bps=100)
    key = "Expanding window · Minimum volatility"
    # The jump at close 5 is first usable by a trade at close 6.
    np.testing.assert_allclose(study.allocations[key], [[1, 0], [1, 0], [0, 1]])
    switch = study.trades[key].iloc[-1]
    assert switch.nav_after == pytest.approx(switch.nav_before * 0.99 / 1.01)
    assert switch.cost == pytest.approx(0.01 * (switch.nav_before + switch.nav_after))


@pytest.mark.parametrize("periods_per_year", [12, 252, 365])
def test_future_changes_do_not_affect_prior_equity_or_targets(periods_per_year):
    prices = history()
    first = run_backtests(prices, train_fraction=0.5, rebalance_every=1, periods_per_year=periods_per_year)
    prices.iloc[5:, 0] *= 1.4
    second = run_backtests(prices, train_fraction=0.5, rebalance_every=1, periods_per_year=periods_per_year)
    pd.testing.assert_frame_equal(first.equity.iloc[:2], second.equity.iloc[:2])
    for key in first.allocations:
        # Even the close-5 trade cannot observe the changed close-5 price.
        pd.testing.assert_frame_equal(
            first.allocations[key].loc[:prices.index[5]],
            second.allocations[key].loc[:prices.index[5]],
        )


@pytest.mark.parametrize("periods_per_year", [12, 52, 252])
def test_training_windows_end_before_execution_and_rolling_is_bounded(monkeypatch, periods_per_year):
    calls = []

    def recording_optimizer(mean, covariance, *args, **kwargs):
        calls.append((mean.copy(), covariance.copy()))
        assert kwargs["frontier_points"] == 2
        weights = pd.Series([0.5, 0.5], index=mean.index)
        portfolios = {name: SimpleNamespace(weights=weights) for name in backtest.PORTFOLIO_NAMES}
        return portfolios, pd.DataFrame(), pd.DataFrame(), []

    monkeypatch.setattr(backtest, "optimize", recording_optimizer)
    prices = history()
    study = run_backtests(prices, train_fraction=0.5, rebalance_every=1, rolling_window=2,
                          periods_per_year=periods_per_year)
    returns = prices.pct_change().iloc[1:]
    # Shared initial fit plus expanding windows; rolling never includes execution day's return.
    expected_windows = [(0, 3), (0, 4), (0, 5), (1, 3), (2, 4), (3, 5)]
    assert len(calls) == len(expected_windows)
    for (mean, covariance), (start, end) in zip(calls, expected_windows):
        sample = returns.iloc[start:end]
        pd.testing.assert_series_equal(mean, sample.mean() * periods_per_year)
        expected_covariance = sample.cov() * periods_per_year
        expected_covariance.iloc[0, 1] *= 0.9
        expected_covariance.iloc[1, 0] *= 0.9
        pd.testing.assert_frame_equal(covariance, expected_covariance)
    for key, trades in study.trades.items():
        assert (pd.to_datetime(trades.train_end).to_numpy() < trades.index.to_numpy()).all()
        if key.startswith("Rolling window"):
            assert trades.train_start.iloc[0] == str(returns.index[1].date())
    assert study.settings["rolling_window"] == 2
    assert study.settings["split"] == 3


def test_equal_weight_refits_match_fixed_rebalancing():
    study = run_backtests(history(), train_fraction=0.5, rebalance_every=1)
    fixed = "Fixed rebalance · Equal weight"
    for method in ("Expanding window", "Rolling window"):
        key = f"{method} · Equal weight"
        np.testing.assert_allclose(study.equity[key], study.equity[fixed])
        pd.testing.assert_frame_equal(study.allocations[key], study.allocations[fixed])
        pd.testing.assert_frame_equal(study.holdings[key], study.holdings[fixed])


def test_single_asset_only_entry_cost_and_correct_exposure():
    prices = history()[["A"]]
    study = run_backtests(prices, train_fraction=0.5, rebalance_every=1, cost_bps=25)
    expected = np.array([1, 1 / 1.0025, 2 / 1.0025, 1 / 1.0025, 1 / 1.0025])
    for key in study.equity:
        np.testing.assert_allclose(study.equity[key], expected)
        assert study.holdings[key].average_weight.iloc[0] == pytest.approx(0.75)
        assert study.holdings[key].end_weight.iloc[0] == pytest.approx(1)
        assert study.holdings[key].selection_frequency.iloc[0] == pytest.approx(1)
        assert study.metrics.loc[key, "total_turnover"] == pytest.approx(1 / 1.0025)


def test_nonpositive_excess_falls_back_without_dropping_curves():
    study = run_backtests(history(), train_fraction=0.5, rebalance_every=1, risk_free_rate=1000)
    for method in METHOD_NAMES:
        minimum, maximum = f"{method} · Minimum volatility", f"{method} · Maximum Sharpe"
        np.testing.assert_allclose(study.equity[minimum], study.equity[maximum])
        assert study.metrics.loc[maximum, "fallback_count"] == study.metrics.loc[maximum, "rebalance_count"]
        assert study.metrics.loc[minimum, "fallback_count"] == 0
    assert any("2020-01-07 · Buy and hold: Maximum Sharpe uses Minimum volatility" in w for w in study.warnings)
    assert not any("omitted" in warning for warning in study.warnings)


@pytest.mark.parametrize("kwargs,match", [
    ({"train_fraction": 1}, "train_fraction"),
    ({"train_fraction": 0.01}, "two training"),
    ({"risk_free_rate": np.nan}, "risk_free_rate"),
    ({"shrinkage": -0.1}, "shrinkage"),
    ({"max_weight": 0.4}, "infeasible"),
    ({"max_weight": 1.1}, "max_weight"),
    ({"rebalance_every": 0}, "rebalance_every"),
    ({"rebalance_every": True}, "rebalance_every"),
    ({"rebalance_every": 1.5}, "rebalance_every"),
    ({"rolling_window": 1}, "rolling_window"),
    ({"rolling_window": 4}, "rolling_window"),
    ({"rolling_window": True}, "rolling_window"),
    ({"cost_bps": -1}, "cost_bps"),
    ({"cost_bps": np.inf}, "cost_bps"),
    ({"cost_bps": 10000}, "cost_bps"),
])
def test_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_backtests(history(), **{"train_fraction": 0.5, **kwargs})


@pytest.mark.parametrize("change,match", [
    (lambda frame: frame.set_axis(["A", "A"], axis=1), "unique asset"),
    (lambda frame: frame.iloc[::-1], "increasing"),
    (lambda frame: frame.reset_index(drop=True), "dates"),
    (lambda frame: frame.assign(A=np.nan), "finite and strictly positive"),
    (lambda frame: frame.assign(A=0), "finite and strictly positive"),
    (lambda frame: frame.iloc[:0], "nonempty"),
])
def test_invalid_prices(change, match):
    with pytest.raises(ValueError, match=match):
        run_backtests(change(history()))


@pytest.mark.parametrize("frequency,periods_per_year", [("MS", 12), ("W", 52)])
def test_periodic_backtest_metrics_use_selected_annualization(frequency, periods_per_year):
    prices = history()[["A"]]
    prices.index = pd.date_range("2020-01-01", periods=len(prices), freq=frequency)
    prices.iloc[5:, 0] = [90, 108, 108]
    study = run_backtests(prices, train_fraction=0.5, risk_free_rate=0.12, cost_bps=0,
                          periods_per_year=periods_per_year)
    annual_excess = 0.025 * periods_per_year - 0.12
    downside = np.sqrt(((0.1 + 0.12 / periods_per_year) ** 2
                        + 2 * (0.12 / periods_per_year) ** 2) * periods_per_year / 4)
    for _, row in study.metrics.iterrows():
        assert row.total_return == pytest.approx(0.08)
        assert row.cagr == pytest.approx(1.08 ** (periods_per_year / 4) - 1)
        assert row.volatility == pytest.approx(np.sqrt(0.0475 / 3 * periods_per_year))
        assert row.sortino == pytest.approx(annual_excess / downside)
        assert row.calmar == pytest.approx(row.cagr / 0.1)
    assert study.settings["periods_per_year"] == periods_per_year
    assert study.settings["trading_days_per_year"] == periods_per_year


def test_backtest_default_periods_match_explicit_trading_year():
    implicit = run_backtests(history(), train_fraction=0.5)
    explicit = run_backtests(history(), train_fraction=0.5, periods_per_year=252)
    pd.testing.assert_frame_equal(implicit.equity, explicit.equity)
    pd.testing.assert_frame_equal(implicit.metrics, explicit.metrics)
    assert implicit.settings == explicit.settings


@pytest.mark.parametrize("periods_per_year", [0, -1, np.nan, np.inf, True, np.bool_(True), "12", None, 12j])
def test_backtest_rejects_invalid_annualization(periods_per_year):
    with pytest.raises(ValueError, match="periods_per_year"):
        run_backtests(history(), periods_per_year=periods_per_year)


@pytest.mark.parametrize("kind", ["boolean", "complex", "object_boolean", "object_complex"])
def test_backtest_rejects_boolean_and_complex_prices(kind):
    prices = history()
    if kind == "boolean":
        prices = prices > 0
    elif kind == "complex":
        prices = prices.astype(complex) + 1j
    else:
        prices = prices.astype(object)
        prices.iloc[0, 0] = True if kind == "object_boolean" else 100 + 1j
    with pytest.raises(ValueError, match="Boolean or complex"):
        run_backtests(prices)


def test_backtest_ratios_are_undefined_for_flat_zero_cost_equity():
    prices = history().copy()
    prices.iloc[4:] = 100
    study = run_backtests(prices, train_fraction=0.5, risk_free_rate=0, cost_bps=0, periods_per_year=12)
    for column in ("sharpe", "sortino", "calmar"):
        assert study.metrics[column].isna().all()


def profile_history():
    rng = np.random.default_rng(304)
    returns = rng.normal([0.01, 0.006], [0.03, 0.015], size=(20, 2))
    prices = np.vstack([np.ones(2), np.cumprod(1 + returns, axis=0)]) * 100
    return pd.DataFrame(prices, index=pd.date_range("2020-01-01", periods=21, freq="MS"), columns=["A", "B"])


def test_optional_profiles_preserve_original_strategies_and_reconcile_costs():
    prices = profile_history()
    settings = dict(train_fraction=0.6, rolling_window=8, rebalance_every=2,
                    risk_free_rate=0.02, periods_per_year=12, cost_bps=25)
    original = run_backtests(prices, **settings)
    extended = run_backtests(prices, **settings, include_profiles=True)
    assert len(original.equity.columns) == 12
    assert len(extended.equity.columns) == 24
    assert original.settings["include_profiles"] is False
    assert extended.settings["include_profiles"] is True
    pd.testing.assert_frame_equal(original.equity, extended.equity.loc[:, original.equity.columns])
    pd.testing.assert_frame_equal(original.metrics, extended.metrics.loc[original.metrics.index])
    for method in METHOD_NAMES:
        minimum, low = f"{method} · Minimum volatility", f"{method} · Low"
        np.testing.assert_allclose(extended.allocations[low], extended.allocations[minimum], atol=2e-5)
        for profile in ("Low", "Medium", "Extreme"):
            key = f"{method} · {profile}"
            trades = extended.trades[key]
            np.testing.assert_allclose(trades.cost, 0.0025 * trades.turnover * trades.nav_before, atol=1e-13)
            assert extended.metrics.loc[key, "total_return"] == pytest.approx(
                extended.holdings[key].pnl_contribution.sum() - trades.cost.sum(), abs=1e-13,
            )
            assert extended.metrics.loc[key, "fallback_count"] == 0


def test_profile_refits_never_observe_execution_day_or_future_prices():
    prices = profile_history()
    settings = dict(train_fraction=0.6, rolling_window=8, rebalance_every=2,
                    periods_per_year=12, include_profiles=True)
    first = run_backtests(prices, **settings)
    prices.iloc[15:, 0] *= 1.4
    second = run_backtests(prices, **settings)
    pd.testing.assert_frame_equal(first.equity.loc[:prices.index[14]], second.equity.loc[:prices.index[14]])
    for method in METHOD_NAMES:
        for profile in ("Low", "Medium", "Extreme"):
            key = f"{method} · {profile}"
            pd.testing.assert_frame_equal(first.allocations[key].loc[:prices.index[15]],
                                          second.allocations[key].loc[:prices.index[15]])


def test_profile_fit_slices_match_expanding_and_rolling_return_windows(monkeypatch):
    prices = profile_history()
    calls = []

    def record_profiles(window, **kwargs):
        calls.append((window.copy(), kwargs))
        weights = pd.Series([0.5, 0.5], index=window.columns)
        return SimpleNamespace(
            portfolios={name: SimpleNamespace(weights=weights) for name in ("Low", "Medium", "Extreme")},
            warnings=["Each historical window has less than one year of observations."],
        )

    monkeypatch.setattr(backtest, "build_profiles", record_profiles)
    settings = dict(train_fraction=0.6, rolling_window=8, rebalance_every=2, risk_free_rate=0.03,
                    periods_per_year=12, max_weight=0.7, shrinkage=0.2, include_profiles=True)
    study = run_backtests(prices, **settings)
    expected_windows = [(0, 12), (0, 14), (0, 16), (0, 18), (4, 12), (6, 14), (8, 16), (10, 18)]
    assert len(calls) == len(expected_windows)
    for (window, kwargs), (start, end) in zip(calls, expected_windows):
        pd.testing.assert_frame_equal(window, prices.iloc[start:end + 1])
        assert kwargs == dict(risk_free_rate=0.03, periods_per_year=12, max_weight=0.7,
                              shrinkage=0.2, frontier_points=3)
    assert len([message for message in study.warnings if "less than one year" in message]) == 1
    for trades in study.trades.values():
        assert (pd.to_datetime(trades.train_end).to_numpy() < trades.index.to_numpy()).all()


@pytest.mark.parametrize("settings", [
    {"train_fraction": 0.25},
    {"train_fraction": 0.6, "rolling_window": 5},
])
def test_profiles_require_six_returns_in_each_fit_window(settings):
    with pytest.raises(ValueError, match="at least six training returns"):
        run_backtests(profile_history(), include_profiles=True, **settings)


def test_profiles_accept_six_initial_training_returns():
    study = run_backtests(profile_history().iloc[:11], train_fraction=0.6, rolling_window=6,
                          periods_per_year=12, include_profiles=True)
    assert study.settings["split"] == 6
    assert len(study.equity.columns) == 24
