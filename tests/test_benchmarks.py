from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.backtest import run_backtests
from efficient_frontier.benchmarks import compare_benchmarks
from efficient_frontier.core import analyze
from efficient_frontier.profiles import build_profiles


NAMES = ["S&P 500 (SPY)", "Nasdaq-100 (QQQ)"]


def histories(frequency="MS"):
    dates = pd.date_range("2020-01-01", periods=11, freq=frequency)
    assets = pd.DataFrame({
        "A": [100, 102, 99, 104, 101, 103, 105, 107, 104, 109, 111],
        "B": [100, 101, 100, 102, 103, 101, 104, 103, 106, 105, 108],
    }, index=dates, dtype=float)
    benchmarks = pd.DataFrame({
        "SPY": [100, 102, 101, 103, 105, 104, 100, 200, 220, 198, 210],
        "QQQ": [100, 103, 102, 104, 100, 107, 100, 80, 88, 92, 84],
    }, index=dates, dtype=float)
    return assets, benchmarks


def test_holdout_and_backtests_use_their_own_entry_timing_and_costs():
    assets, prices = histories()
    settings = dict(train_fraction=0.6, risk_free_rate=0.03, periods_per_year=12)
    analysis = analyze(assets, **settings)
    study = run_backtests(assets, **settings, cost_bps=100)
    result = compare_benchmarks(prices, assets, analysis, study, risk_free_rate=0.03)
    after_entry = 1 / 1.01
    np.testing.assert_allclose(result.holdout_equity[NAMES[0]], [1, 2, 2.2, 1.98, 2.1])
    np.testing.assert_allclose(result.holdout_equity[NAMES[1]], [1, 0.8, 0.88, 0.92, 0.84])
    np.testing.assert_allclose(result.backtest_equity[NAMES[0]],
                               [1, after_entry, 1.1 * after_entry, 0.99 * after_entry, 1.05 * after_entry])
    np.testing.assert_allclose(result.backtest_equity[NAMES[1]],
                               [1, after_entry, 1.1 * after_entry, 1.15 * after_entry, 1.05 * after_entry])
    for _, row in result.backtest_metrics.iterrows():
        assert row.total_turnover == pytest.approx(after_entry)
        assert row.total_cost == pytest.approx(0.01 / 1.01)
        assert row.total_return == pytest.approx(1.05 / 1.01 - 1)
        assert row.rebalance_count == 1
        assert row.fallback_count == 0
    assert result.holdout_equity.index.equals(analysis.equity.index)
    assert result.backtest_equity.index.equals(study.equity.index)
    assert result.settings["holdout_cost_bps"] == 0
    assert result.settings["backtest_cost_bps"] == 100


@pytest.mark.parametrize("frequency,periods", [("MS", 12), ("W-FRI", 52), ("D", 365)])
def test_benchmarks_match_independent_single_asset_engine_results(frequency, periods):
    assets, prices = histories(frequency)
    settings = dict(train_fraction=0.6, risk_free_rate=0.07, periods_per_year=periods)
    analysis = analyze(assets, **settings)
    study = run_backtests(assets, **settings, cost_bps=40)
    result = compare_benchmarks(prices, assets, analysis, study, risk_free_rate=0.07)
    for symbol, name in zip(prices.columns, NAMES):
        single_analysis = analyze(prices[[symbol]], **settings)
        single_study = run_backtests(prices[[symbol]], **settings, cost_bps=40)
        np.testing.assert_allclose(result.holdout_equity[name], single_analysis.equity["Equal weight"])
        np.testing.assert_allclose(result.backtest_equity[name], single_study.equity["Buy and hold · Equal weight"])
        np.testing.assert_allclose(result.holdout_metrics.loc[name],
                                   single_analysis.holdout_metrics.loc["Equal weight"], equal_nan=True)
        np.testing.assert_allclose(result.backtest_metrics.loc[name],
                                   single_study.metrics.loc["Buy and hold · Equal weight"], equal_nan=True)
    assert result.settings["periods_per_year"] == periods


def test_estimates_use_the_same_training_dates_and_latest_windows():
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6, periods_per_year=12, risk_free_rate=0.08, shrinkage=1)
    latest = build_profiles(assets, periods_per_year=12, risk_free_rate=0.08)
    result = compare_benchmarks(prices, assets, analysis, latest_profiles=latest, risk_free_rate=0.08)
    returns = prices.to_numpy()[1:] / prices.to_numpy()[:-1] - 1
    for number, name in enumerate(NAMES):
        train = returns[:6, number]
        mean, volatility = np.mean(train) * 12, np.std(train, ddof=1) * np.sqrt(12)
        row = result.training_estimates.loc[name]
        assert row.expected_return == pytest.approx(mean)
        assert row.volatility == pytest.approx(volatility)
        assert row.sharpe == pytest.approx((mean - 0.08) / volatility)
        latest_row = result.latest_estimates.loc[name]
        assert latest_row.expected_return == pytest.approx(np.mean(returns[:, number]) * 12)
        assert latest_row.volatility == pytest.approx(np.std(returns[:, number], ddof=1) * np.sqrt(12))
        assert latest_row.worst_window_return == pytest.approx(
            min(np.mean(chunk) * 12 for chunk in np.array_split(returns[:, number], 3)),
        )
    assert result.backtest_equity is None
    assert result.backtest_metrics is None


def test_missing_optional_results_do_not_create_backtests_or_window_estimates():
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6)
    result = compare_benchmarks(prices, assets, analysis, risk_free_rate=0.02)
    assert result.backtest_equity is None
    assert result.backtest_metrics is None
    assert result.latest_estimates.worst_window_return.isna().all()


@pytest.mark.parametrize("change", [
    lambda frame: frame.drop(frame.index[2]),
    lambda frame: pd.concat([frame, frame.iloc[[-1]].set_axis([frame.index[-1] + pd.Timedelta(days=1)])]),
])
def test_benchmarks_reject_missing_or_extra_full_input_dates(change):
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6)
    with pytest.raises(ValueError, match="date|Date"):
        compare_benchmarks(change(prices), assets, analysis, risk_free_rate=0.02)


def test_benchmarks_do_not_mutate_or_expand_the_original_analysis():
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6, periods_per_year=12, max_weight=0.6)
    study = run_backtests(assets, train_fraction=0.6, periods_per_year=12, max_weight=0.6, include_profiles=True)
    latest = build_profiles(assets, periods_per_year=12, max_weight=0.6)
    original = deepcopy((assets, prices, analysis, study, latest))
    result = compare_benchmarks(prices, assets, analysis, study, latest, risk_free_rate=0.02)
    pd.testing.assert_frame_equal(assets, original[0])
    pd.testing.assert_frame_equal(prices, original[1])
    for field in ("equity", "frontier", "frontier_weights", "holdout_metrics", "covariance"):
        pd.testing.assert_frame_equal(getattr(analysis, field), getattr(original[2], field))
    for field in ("equity", "metrics"):
        pd.testing.assert_frame_equal(getattr(study, field), getattr(original[3], field))
    pd.testing.assert_frame_equal(latest.summary, original[4].summary)
    assert analysis.mean_returns.index.tolist() == ["A", "B"]
    assert len(study.equity.columns) == 24
    assert study.settings == original[3].settings
    assert result.prices.columns.tolist() == ["SPY", "QQQ"]


def test_constant_benchmark_prices_have_undefined_training_and_holdout_sharpe():
    assets, prices = histories()
    prices.loc[:, :] = 100
    analysis = analyze(assets, train_fraction=0.6, periods_per_year=12)
    result = compare_benchmarks(prices, assets, analysis, risk_free_rate=0.02)
    assert result.training_estimates.sharpe.isna().all()
    assert result.latest_estimates.sharpe.isna().all()
    assert result.holdout_metrics.sharpe.isna().all()


@pytest.mark.parametrize("risk_free_rate", [np.nan, np.inf, -np.inf])
def test_benchmarks_reject_nonfinite_risk_free_rates(risk_free_rate):
    assets, prices = histories()
    with pytest.raises(ValueError, match="risk_free_rate"):
        compare_benchmarks(prices, assets, analyze(assets), risk_free_rate=risk_free_rate)


def test_benchmark_comparison_rejects_a_study_with_different_evaluation_dates():
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6)
    study = run_backtests(assets, train_fraction=0.5)
    with pytest.raises(ValueError, match="same evaluation dates"):
        compare_benchmarks(prices, assets, analysis, study, risk_free_rate=0.02)


def test_benchmark_comparison_rejects_inconsistent_annualization():
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6, periods_per_year=12)
    study = run_backtests(assets, train_fraction=0.6, periods_per_year=252)
    with pytest.raises(ValueError, match="periods_per_year"):
        compare_benchmarks(prices, assets, analysis, study, risk_free_rate=0.02)


def test_standalone_benchmarks_reject_a_rate_that_differs_from_analysis():
    assets, prices = histories()
    analysis = analyze(assets, risk_free_rate=0.07)
    assert analysis.risk_free_rate == 0.07
    with pytest.raises(ValueError, match="Analysis and benchmarks must use the same risk_free_rate"):
        compare_benchmarks(prices, assets, analysis, risk_free_rate=0.02)


def test_standalone_benchmarks_match_portfolio_metrics_with_a_nondefault_rate():
    _, prices = histories()
    assets = prices[["SPY"]].rename(columns={"SPY": "A"})
    analysis = analyze(assets, risk_free_rate=0.07, periods_per_year=12)
    result = compare_benchmarks(prices, assets, analysis, risk_free_rate=0.07)
    np.testing.assert_allclose(
        result.holdout_metrics.loc[NAMES[0]], analysis.holdout_metrics.loc["Equal weight"], equal_nan=True,
    )


@pytest.mark.parametrize("result_type", ["study", "latest_profiles"])
def test_benchmark_comparison_rejects_inconsistent_risk_free_rates(result_type):
    assets, prices = histories()
    analysis = analyze(assets, train_fraction=0.6, risk_free_rate=0.02)
    if result_type == "study":
        extra = run_backtests(assets, train_fraction=0.6, risk_free_rate=0.03)
    else:
        extra = build_profiles(assets, risk_free_rate=0.03)
    with pytest.raises(ValueError, match="risk_free_rate"):
        compare_benchmarks(prices, assets, analysis, **{result_type: extra}, risk_free_rate=0.02)
