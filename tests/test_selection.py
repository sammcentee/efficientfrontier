from dataclasses import FrozenInstanceError, asdict
import json

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.selection import (
    BASELINE, BENCHMARK_NAMES, CANDIDATE_NAMES, RISK_MATCHED_NAMES, SELECTED,
    _fit_targets, run_selection,
)


def prices_from_returns(values, columns=None):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        100 * np.vstack([np.ones(values.shape[1]), np.cumprod(1 + values, axis=0)]),
        index=pd.bdate_range("2020-01-01", periods=len(values) + 1),
        columns=columns or [f"STOCK{i}" for i in range(values.shape[1])],
    )


def stock_prices(count=96, assets=12):
    rng = np.random.default_rng(2036)
    values = rng.normal(np.linspace(0.0003, 0.002, assets), np.linspace(0.003, 0.012, assets), (count, assets))
    return prices_from_returns(values)


def changing_prices():
    alternating = np.tile([-0.001, 0.001], 20)
    a = np.r_[np.full(20, .01), np.full(10, .02), np.full(10, -.03)] + alternating
    b = np.r_[np.full(30, .002), np.full(10, .01)] - alternating
    stocks = prices_from_returns(np.column_stack([a, b]), ["A", "B"])
    market = prices_from_returns(np.column_stack([.002 + alternating, .003 - alternating]), ["SPY", "QQQ"])
    return stocks, market


def test_mean_endpoint_has_analytic_cap_and_baseline_is_equal_weight_when_feasible():
    values = np.column_stack([
        .005 + np.tile([-.002, .002], 12),
        .02 + np.tile([-.004, -.004, .004, .004], 6),
    ])
    prices = prices_from_returns(values, ["A", "B"])
    baseline, maximum, mean, covariance, coverage, _ = _fit_targets(prices, .5, .7, 1, 12)
    np.testing.assert_allclose(baseline, [.5, .5], atol=1e-7)
    np.testing.assert_allclose(maximum, [.3, .7], atol=1e-7)
    np.testing.assert_allclose(mean, [.06, .24], atol=1e-12)
    np.testing.assert_allclose(covariance, np.diag(values.var(axis=0, ddof=1) * 12), atol=1e-12)
    assert coverage.eligible.all()
    for alpha in CANDIDATE_NAMES:
        weights = (1 - alpha) * baseline + alpha * maximum
        assert weights.sum() == pytest.approx(1)
        assert weights.min() >= 0 and weights.max() <= .7000001
        assert np.sqrt(weights @ covariance @ weights) <= .5 + 1e-7


def test_lower_risk_target_changes_the_feasible_baseline_without_cash_or_cap_relaxation():
    values = np.column_stack([
        .002 + np.tile([-.01, .01], 12),
        .02 + np.tile([-.10, -.10, .10, .10], 6),
    ])
    prices = prices_from_returns(values, ["A", "B"])
    baseline, maximum, _, covariance, _, _ = _fit_targets(prices, .10, .9, 1, 12)
    variance = values.var(axis=0, ddof=1) * 12
    # A two-asset diagonal covariance gives a quadratic upper bound on B.
    roots = np.roots([variance.sum(), -2 * variance[0], variance[0] - .10 ** 2])
    b = roots.max()
    np.testing.assert_allclose(baseline, [1 - b, b], atol=3e-6)
    np.testing.assert_allclose(maximum, baseline, atol=3e-6)
    assert np.sqrt(baseline @ covariance @ baseline) == pytest.approx(.10, abs=1e-7)
    assert baseline.sum() == pytest.approx(1)


def test_infeasible_minimum_risk_and_too_few_stocks_fail_explicitly():
    prices = stock_prices(40, 12)
    with pytest.raises(ValueError, match="below the feasible minimum"):
        run_selection(prices, target_volatility=.0001)
    with pytest.raises(ValueError, match="position cap is infeasible"):
        run_selection(prices.iloc[:, :9])


def test_default_cap_and_allocation_risk_hold_at_every_prior_fit():
    prices = stock_prices()
    result = run_selection(prices, target_volatility=.2, rebalance_every=7)
    for study in (result.validation, result.test):
        for name in CANDIDATE_NAMES.values():
            weights = study.allocations[name]
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_allclose(weights.CASH, 0, atol=1e-12)
            assert weights.drop(columns="CASH").to_numpy().min() >= 0
            assert weights.drop(columns="CASH").to_numpy().max() <= .1000001
            for execution, row in weights.iterrows():
                returns = prices.loc[prices.index < execution].pct_change(fill_method=None).iloc[1:]
                covariance = returns.cov().to_numpy() * 252
                covariance = .9 * covariance + .1 * np.diag(np.diag(covariance))
                w = row.drop("CASH").to_numpy()
                assert np.sqrt(w @ covariance @ w) <= .2 + 1e-7
    assert result.latest[SELECTED].volatility <= .2 + 1e-7


def test_final_test_never_reselects_a_validation_winner_and_failure_is_visible():
    prices, benchmarks = changing_prices()
    options = dict(train_fraction=.75, target_volatility=.5, max_weight=1,
                   periods_per_year=6, cost_bps=10)
    result = run_selection(prices, benchmarks, **options)
    assert result.policy.blend == 1
    assert result.policy.validation_passed
    assert not result.supported
    assert any("later test" in reason for reason in result.reasons)
    pd.testing.assert_series_equal(result.test.equity[SELECTED],
                                   result.test.equity[CANDIDATE_NAMES[1]].rename(SELECTED))
    altered = prices.copy()
    altered.iloc[31:, 0] *= 3
    changed = run_selection(altered, benchmarks, **options)
    assert changed.policy == result.policy
    pd.testing.assert_frame_equal(result.validation.equity, changed.validation.equity)
    for name in result.test.allocations:
        pd.testing.assert_frame_equal(result.test.allocations[name].iloc[:1], changed.test.allocations[name].iloc[:1])
    with pytest.raises(FrozenInstanceError):
        result.policy.blend = .5


def test_validation_ties_choose_baseline_and_missing_markets_never_pass_gate():
    values = .002 + np.tile([-.005, .005], 24)
    prices = prices_from_returns(np.column_stack([values, values]))
    result = run_selection(prices, max_weight=.6, target_volatility=.2)
    assert result.policy.blend == 0
    assert not result.policy.validation_passed and not result.supported
    assert any("baseline" in reason for reason in result.reasons)
    assert any("benchmark prices are unavailable" in reason for reason in result.reasons)
    assert list(result.latest) == [SELECTED, BASELINE]
    json.dumps(asdict(result.policy))
    json.dumps(result.settings)


def test_benchmark_cash_uses_only_prior_volatility_and_charges_only_security_trades():
    prices = stock_prices(80)
    market_returns = np.tile([-.04, .05], 40)
    benchmarks = prices_from_returns(np.column_stack([market_returns, market_returns / 2]), ["SPY", "QQQ"])
    result = run_selection(prices, benchmarks, target_volatility=.12, rebalance_every=7, cost_bps=25)
    for study in (result.validation, result.test):
        for symbol in benchmarks:
            key = RISK_MATCHED_NAMES[symbol]
            weights = study.allocations[key]
            for execution, allocation in weights.iterrows():
                history = benchmarks.loc[benchmarks.index < execution, symbol].pct_change(fill_method=None).iloc[1:]
                expected = min(1, .12 / (history.std(ddof=1) * np.sqrt(252)))
                assert allocation[symbol] == pytest.approx(expected)
                assert allocation.CASH == pytest.approx(1 - expected)
            initial = weights.iloc[0, 0]
            assert study.equity[key].iloc[1] == pytest.approx(1 / (1 + .0025 * initial))
            assert study.trades[key].cost.iloc[0] == pytest.approx(.0025 * initial / (1 + .0025 * initial))
            raw = BENCHMARK_NAMES[symbol]
            assert len(study.trades[raw]) == 1
            np.testing.assert_allclose(study.allocations[raw].iloc[0], [1, 0])
    pd.testing.assert_frame_equal(result.benchmark_prices, benchmarks)


def test_all_equity_paths_replay_from_targets_and_reconcile_security_profit_less_cost():
    prices, benchmarks = changing_prices()
    result = run_selection(prices, benchmarks, target_volatility=.5, max_weight=1,
                           periods_per_year=12, train_fraction=.75, rebalance_every=3, cost_bps=35)
    for study in (result.validation, result.test):
        for name, weights in study.allocations.items():
            assets = [column for column in weights if column != "CASH"]
            source = benchmarks if assets[0] in ("SPY", "QQQ") else prices
            shares = pd.Series(0., index=assets)
            cash = 1.
            replay = [1.]
            for day in study.equity.index[1:]:
                nav = float(shares @ source.loc[day, assets] + cash)
                if day in weights.index:
                    after = nav - study.trades[name].loc[day, "cost"]
                    shares = after * weights.loc[day, assets] / source.loc[day, assets]
                    cash = after * weights.loc[day, "CASH"]
                    nav = after
                replay.append(nav)
            np.testing.assert_allclose(study.equity[name], replay, atol=2e-12)
            trades = study.trades[name]
            np.testing.assert_allclose(trades.cost, .0035 * trades.turnover * trades.nav_before, atol=2e-13)
            assert study.metrics.loc[name, "total_return"] == pytest.approx(
                study.holdings[name].pnl_contribution.sum() - trades.cost.sum(), abs=2e-12,
            )
            expected_mean = study.equity[name].pct_change(fill_method=None).iloc[1:].mean() * 12
            assert study.metrics.loc[name, "annual_mean"] == pytest.approx(expected_mean)
            assert (pd.to_datetime(trades.train_end) < trades.index).all()


def test_long_unchanged_price_screen_is_past_only_and_preserves_raw_labels():
    prices = stock_prices(96, 12)
    prices.iloc[:26, 0] = 100
    original = prices.copy()
    result = run_selection(prices, target_volatility=.2)
    excluded = result.eligibility[result.eligibility.symbol == prices.columns[0]]
    assert not excluded.eligible.any()
    assert excluded.max_unchanged_intervals.ge(25).all()
    assert excluded.reason.str.contains("unchanged price intervals").all()
    for study in (result.validation, result.test):
        assert any("long unchanged-price screen excluded" in warning for warning in study.warnings)
        for name in CANDIDATE_NAMES.values():
            assert study.allocations[name][prices.columns[0]].eq(0).all()
    assert result.latest[SELECTED].weights[prices.columns[0]] == 0
    pd.testing.assert_frame_equal(prices, original)
    # Future stale observations must not change a fit made before they occur.
    past = prices.iloc[:40]
    prior = _fit_targets(past, .2, .1, .1, 252)
    changed = prices.copy()
    changed.iloc[40:, 1] = changed.iloc[39, 1]
    same_prior = _fit_targets(changed.iloc[:40], .2, .1, .1, 252)
    pd.testing.assert_series_equal(prior[0], same_prior[0])
    assert prior[-2].loc[1, "eligible"]
    later = _fit_targets(changed, .2, .1, .1, 252)
    assert not later[-2].loc[1, "eligible"]


def test_newly_stale_held_stock_does_not_assume_a_verified_sale():
    prices = stock_prices(120, 12)
    prices.iloc[90:, 0] = prices.iloc[89, 0]
    with pytest.raises(ValueError, match="sale at the repeated quote is not verified"):
        run_selection(prices, target_volatility=.3, rebalance_every=3)


@pytest.mark.parametrize("options, match", [
    ({"target_volatility": 0}, "target_volatility"),
    ({"target_volatility": np.inf}, "target_volatility"),
    ({"target_volatility": True}, "target_volatility"),
    ({"max_weight": 0}, "max_weight"),
    ({"max_weight": 1.1}, "max_weight"),
    ({"train_fraction": 1}, "train_fraction"),
    ({"train_fraction": .001}, "at least two returns"),
    ({"shrinkage": -1}, "shrinkage"),
    ({"risk_free_rate": np.nan}, "risk_free_rate"),
    ({"cost_bps": -1}, "cost_bps"),
    ({"cost_bps": 10000}, "cost_bps"),
    ({"rebalance_every": True}, "rebalance_every"),
    ({"rebalance_every": 1.5}, "rebalance_every"),
    ({"periods_per_year": 0}, "periods_per_year"),
])
def test_invalid_settings_fail_before_selection(options, match):
    with pytest.raises(ValueError, match=match):
        run_selection(stock_prices(20), **options)


def test_benchmarks_need_matching_dates_and_supported_names():
    prices = stock_prices(40)
    benchmarks = prices.iloc[:, :2].copy()
    with pytest.raises(ValueError, match="SPY and/or QQQ"):
        run_selection(prices, benchmarks)
    benchmarks.columns = ["SPY", "QQQ"]
    with pytest.raises(ValueError, match="exactly the stock price dates"):
        run_selection(prices, benchmarks.iloc[1:])
    missing = prices.copy()
    missing.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="resolve missing data"):
        run_selection(missing)


def test_one_available_market_reference_is_explicit_and_has_fixed_order():
    prices, benchmarks = changing_prices()
    result = run_selection(prices, benchmarks[["QQQ"]], max_weight=1, target_volatility=.5,
                           periods_per_year=12, train_fraction=.75)
    expected = [SELECTED, *CANDIDATE_NAMES.values(), BENCHMARK_NAMES["QQQ"], RISK_MATCHED_NAMES["QQQ"]]
    assert list(result.validation.equity) == expected
    assert list(result.test.metrics.index) == expected
    assert not result.policy.validation_passed
    assert any("SPY" in reason and "unavailable" in reason for reason in result.reasons)


def test_support_requires_both_market_references_and_one_year_in_each_evaluation_period():
    alternating = np.tile([-.001, .001], 40)
    stocks = prices_from_returns(np.column_stack([.01 + alternating, .002 - alternating]))
    markets = prices_from_returns(np.column_stack([.002 + alternating, .003 - alternating]), ["SPY", "QQQ"])
    supported = run_selection(stocks, markets, max_weight=.8, target_volatility=.2, periods_per_year=12)
    assert supported.policy.blend == 1
    assert supported.policy.validation_passed and supported.supported
    assert not supported.reasons
    short = run_selection(stocks.iloc[:21], markets.iloc[:21], max_weight=.8,
                          target_volatility=.2, periods_per_year=12)
    assert short.policy.blend == 1
    assert not short.supported
    assert any("Earlier validation" in reason and "evidence floor" in reason for reason in short.reasons)
    assert any("later test" in reason and "evidence floor" in reason for reason in short.reasons)


def test_validation_risk_breach_blocks_support_even_when_later_returns_and_risk_pass():
    a = np.r_[.01 + np.tile([-.001, .001], 30), np.tile([.24, -.15], 15),
              .02 + np.tile([-.001, .001], 15)]
    b = .002 + np.tile([-.001, .001], 60)
    stocks = prices_from_returns(np.column_stack([a, b]))
    markets = prices_from_returns(np.column_stack([b / 2, b]), ["SPY", "QQQ"])
    result = run_selection(stocks, markets, target_volatility=.25, max_weight=1,
                           train_fraction=.75, periods_per_year=12, rebalance_every=1000)
    assert result.policy.blend == 1  # The risk gate does not reselect a different blend.
    assert result.validation.metrics.loc[SELECTED, "volatility"] > .25
    assert result.test.metrics.loc[SELECTED, "volatility"] < .25
    for comparator in (BASELINE, *RISK_MATCHED_NAMES.values()):
        assert result.test.metrics.loc[SELECTED, "cagr"] > result.test.metrics.loc[comparator, "cagr"]
    assert not result.policy.validation_passed and not result.supported
    assert result.reasons == ["Realized annual volatility in earlier validation exceeded the requested target."]
