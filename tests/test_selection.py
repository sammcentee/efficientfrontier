import copy
from dataclasses import fields
import json

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.selection import (
    BASELINE, BENCHMARK_NAMES, MODEL_REVISION, SELECTED, SelectionStudy,
    _fit_targets, run_selection, selection_fit_estimates,
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


def test_mean_objective_has_analytic_cap_and_equal_weight_comparator():
    values = np.column_stack([
        .005 + np.tile([-.002, .002], 12),
        .02 + np.tile([-.004, -.004, .004, .004], 6),
    ])
    prices = prices_from_returns(values, ["A", "B"])
    selected, equal, mean, covariance, coverage, _ = _fit_targets(prices, .5, .7, 1, 12)
    np.testing.assert_allclose(selected, [.3, .7], atol=1e-7)
    np.testing.assert_allclose(equal, [.5, .5], atol=1e-7)
    np.testing.assert_allclose(mean, [.06, .24], atol=1e-12)
    np.testing.assert_allclose(covariance, np.diag(values.var(axis=0, ddof=1) * 12), atol=1e-12)
    assert coverage.eligible.all()
    assert selected.sum() == pytest.approx(1)
    assert selected.min() >= 0 and selected.max() <= .7000001
    assert np.sqrt(selected @ covariance @ selected) <= .5 + 1e-7


def test_binding_risk_limit_maximizes_mean_and_does_not_modify_equal_weights():
    values = np.column_stack([
        .002 + np.tile([-.01, .01], 12),
        .02 + np.tile([-.10, -.10, .10, .10], 6),
    ])
    prices = prices_from_returns(values, ["A", "B"])
    selected, equal, mean, covariance, _, _ = _fit_targets(prices, .10, .9, 1, 12)
    variance = values.var(axis=0, ddof=1) * 12
    # A two-asset diagonal covariance gives an independent upper bound on B.
    roots = np.roots([variance.sum(), -2 * variance[0], variance[0] - .10 ** 2])
    b = roots.max()
    np.testing.assert_allclose(selected, [1 - b, b], atol=3e-6)
    assert np.sqrt(selected @ covariance @ selected) == pytest.approx(.10, abs=1e-7)
    assert selected.sum() == pytest.approx(1)
    alternatives = np.column_stack([1 - np.linspace(.1, .9, 1001), np.linspace(.1, .9, 1001)])
    feasible = alternatives[(alternatives ** 2 @ variance) <= .10 ** 2]
    assert float(mean @ selected) >= float(np.max(feasible @ mean)) - 1e-8
    np.testing.assert_allclose(equal, [.5, .5], atol=1e-12)
    assert np.sqrt(equal @ covariance @ equal) > .10


def test_negative_historical_means_still_choose_the_maximum_fully_invested_mean():
    values = np.column_stack([
        -.005 + np.tile([-.002, .002], 12),
        -.02 + np.tile([-.004, -.004, .004, .004], 6),
    ])
    selected, _, mean, _, _, _ = _fit_targets(prices_from_returns(values), .5, .7, .1, 12)
    np.testing.assert_allclose(selected, [.7, .3], atol=1e-7)
    assert float(mean @ selected) < 0


def test_infeasible_minimum_risk_and_too_few_stocks_fail_explicitly():
    prices = stock_prices(40, 12)
    with pytest.raises(ValueError, match="below the feasible minimum"):
        run_selection(prices, target_volatility=.0001)
    with pytest.raises(ValueError, match="position cap is infeasible"):
        run_selection(prices.iloc[:, :9])


def test_default_cap_and_allocation_risk_hold_at_every_prior_fit():
    prices = stock_prices()
    result = run_selection(prices, target_volatility=.2, rebalance_every=7)
    weights = result.test.allocations[SELECTED]
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
    equal = result.test.allocations[BASELINE]
    np.testing.assert_allclose(equal.drop(columns="CASH"), 1 / len(prices.columns), atol=1e-12)
    pd.testing.assert_index_equal(equal.index, weights.index)
    assert result.latest[SELECTED].volatility <= .2 + 1e-7


def test_one_split_uses_all_initial_training_returns_and_records_fit_dates():
    prices = stock_prices()
    result = run_selection(prices, rebalance_every=7)
    split = int(np.floor((len(prices) - 1) * .7))
    assert result.settings["train_fraction"] == .7
    assert result.settings["initial_training_returns"] == split
    assert result.settings["test_returns"] == len(prices) - 1 - split
    assert result.settings["initial_training_start"] == str(prices.index[1].date())
    assert result.settings["initial_training_end"] == str(prices.index[split].date())
    assert result.settings["test_start"] == str(prices.index[split + 1].date())
    assert result.settings["test_end"] == str(prices.index[-1].date())
    pd.testing.assert_index_equal(result.test.equity.index, prices.index[split:])
    executions = list(range(split + 1, len(prices) - 1, 7))
    pd.testing.assert_index_equal(result.test.allocations[SELECTED].index, prices.index[executions])
    assert result.settings["fit_dates"] == [str(prices.index[i].date()) for i in [*[j - 1 for j in executions], len(prices) - 1]]
    assert result.settings["latest_as_of"] == str(prices.index[-1].date())
    assert result.test.settings["split"] == split
    assert result.test.settings["initial_execution"] == str(prices.index[split + 1].date())


@pytest.mark.parametrize("shrinkage", [0., .1, 1.])
def test_fit_estimates_match_covariance_quadratic_forms_and_saved_latest(shrinkage):
    prices = stock_prices()
    result = run_selection(prices, target_volatility=.2, rebalance_every=7, shrinkage=shrinkage)
    estimates = selection_fit_estimates(result, prices)
    assert list(estimates.columns) == [
        "phase", "execution_date", "observations", "fitted_mean", "model_volatility",
        "historical_volatility", "max_weight",
    ]
    assert estimates.index.name == "fit_date" and estimates.index.is_unique
    assert list(estimates.index) == list(pd.to_datetime(result.settings["fit_dates"]))
    assert estimates.iloc[0].phase == "First historical fit"
    assert estimates.iloc[1:-1].phase.eq("Historical refit").all()
    assert estimates.iloc[-1].phase == "Latest fit"
    assert pd.isna(estimates.iloc[-1].execution_date)
    assert estimates.iloc[0].observations == result.settings["initial_training_returns"]
    assert estimates.iloc[-1].observations == len(prices) - 1
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(estimates.execution_date.iloc[:-1]).rename(None),
        result.test.allocations[SELECTED].index,
    )
    for fit_date, row in estimates.iterrows():
        returns = prices.loc[:fit_date].pct_change(fill_method=None).iloc[1:]
        sample = returns.cov() * result.settings["periods_per_year"]
        covariance = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
        target = (result.latest[SELECTED].weights if row.phase == "Latest fit"
                  else result.test.allocations[SELECTED].loc[row.execution_date])
        weights = target.reindex(prices.columns)
        expected_mean = float((returns.mean() * result.settings["periods_per_year"]) @ weights)
        assert row.fitted_mean == pytest.approx(expected_mean, abs=1e-12)
        assert row.model_volatility == pytest.approx(np.sqrt(weights @ covariance @ weights), abs=1e-12)
        assert row.historical_volatility == pytest.approx(np.sqrt(weights @ sample @ weights), abs=1e-12)
        assert row.max_weight == pytest.approx(weights.max(), abs=1e-12)
        assert row.model_volatility <= result.settings["target_volatility"] + 1e-7
    assert estimates.iloc[-1].fitted_mean == pytest.approx(result.latest[SELECTED].expected_return, abs=1e-12)
    assert estimates.iloc[-1].model_volatility == pytest.approx(result.latest[SELECTED].volatility, abs=1e-12)


def test_fit_estimates_use_prior_prices_without_future_contamination():
    prices = stock_prices()
    result = run_selection(prices, target_volatility=.2, rebalance_every=7)
    original = selection_fit_estimates(result, prices)
    altered = prices.copy()
    cutoff = original.index[1]
    future = altered.index > cutoff
    held_stock = result.latest[SELECTED].weights.idxmax()
    altered.loc[future, held_stock] *= np.linspace(1.1, 1.8, future.sum())
    changed = selection_fit_estimates(result, altered)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])
    assert changed.iloc[-1].fitted_mean != pytest.approx(original.iloc[-1].fitted_mean, abs=1e-8)


def test_fit_estimates_never_solve_or_mutate_inputs(monkeypatch):
    prices = stock_prices()
    result = run_selection(prices, rebalance_every=7)
    original_prices, original_result = prices.copy(deep=True), copy.deepcopy(result)

    def unexpected_solve(*args, **kwargs):
        pytest.fail("Fit estimates must use saved weights without another optimization.")

    monkeypatch.setattr("efficient_frontier.selection._fit_targets", unexpected_solve)
    monkeypatch.setattr("efficient_frontier.selection._solve", unexpected_solve)
    selection_fit_estimates(result, prices)
    pd.testing.assert_frame_equal(prices, original_prices)
    assert result.settings == original_result.settings
    assert result.test.settings == original_result.test.settings
    pd.testing.assert_frame_equal(result.eligibility, original_result.eligibility)
    for name in ("equity", "metrics"):
        pd.testing.assert_frame_equal(getattr(result.test, name), getattr(original_result.test, name))
    for name in ("allocations", "holdings", "trades"):
        for strategy, frame in getattr(result.test, name).items():
            pd.testing.assert_frame_equal(frame, getattr(original_result.test, name)[strategy])
    pd.testing.assert_series_equal(result.latest[SELECTED].weights, original_result.latest[SELECTED].weights)


def test_future_prices_do_not_change_earlier_allocations_or_replay():
    prices = stock_prices()
    options = dict(target_volatility=.5, rebalance_every=7)
    result = run_selection(prices, **options)
    altered = prices.copy()
    altered.iloc[80:, 0] *= np.linspace(1.1, 1.8, len(prices) - 80)
    changed = run_selection(altered, **options)
    cutoff = prices.index[80]
    for name in result.test.allocations:
        original = result.test.allocations[name]
        updated = changed.test.allocations[name]
        pd.testing.assert_frame_equal(original.loc[original.index <= cutoff], updated.loc[updated.index <= cutoff])
    pd.testing.assert_frame_equal(result.test.equity.loc[result.test.equity.index < cutoff],
                                  changed.test.equity.loc[changed.test.equity.index < cutoff])
    assert not np.allclose(result.latest[SELECTED].weights, changed.latest[SELECTED].weights)


def test_one_stock_method_has_no_policy_validation_or_alias_duplicates():
    prices, benchmarks = changing_prices()
    result = run_selection(prices, benchmarks, max_weight=1, target_volatility=.5, periods_per_year=6)
    assert [field.name for field in fields(SelectionStudy)] == [
        "test", "latest", "selected", "supported", "reasons", "settings", "warnings", "benchmark_prices", "eligibility",
    ]
    expected = [SELECTED, BASELINE, *BENCHMARK_NAMES.values()]
    assert list(result.test.equity) == expected
    assert list(result.test.metrics.index) == expected
    assert list(result.latest) == [SELECTED]
    assert result.selected == SELECTED == "Historical-mean portfolio"
    assert BASELINE == "Equal-weight stocks"
    assert result.settings["model_revision"] == MODEL_REVISION == "single_mean_v1"
    assert result.test.settings["model_revision"] == MODEL_REVISION
    assert not any("blend" in key or "validation" in key for key in result.settings)
    assert "arithmetic mean" in result.settings["model_objective"]
    assert "realized volatility can exceed" in result.settings["risk_constraint"]
    assert "does not measure statistical significance" in result.settings["evidence_floor_policy"]
    assert any("does not guarantee future" in warning for warning in result.warnings)
    json.dumps(result.settings)


def test_equal_mean_ties_remain_feasible_and_missing_markets_do_not_pass_check():
    values = .002 + np.tile([-.005, .005], 24)
    prices = prices_from_returns(np.column_stack([values, values]))
    result = run_selection(prices, max_weight=.6, target_volatility=.2)
    assert not result.supported
    assert any("benchmark prices are unavailable" in reason for reason in result.reasons)
    assert list(result.test.equity) == [SELECTED, BASELINE]
    assert result.latest[SELECTED].weights.max() <= .6000001
    assert result.latest[SELECTED].weights.sum() == pytest.approx(1)


def test_raw_benchmarks_buy_once_after_cash_interval_and_pay_security_costs():
    prices = stock_prices(80)
    market_returns = np.tile([-.04, .05], 40)
    benchmarks = prices_from_returns(np.column_stack([market_returns, market_returns / 2]), ["SPY", "QQQ"])
    result = run_selection(prices, benchmarks, target_volatility=.12, rebalance_every=7, cost_bps=25)
    study = result.test
    for name in study.equity:
        assert study.equity[name].iloc[0] == 1
        assert study.equity[name].iloc[1] == pytest.approx(1 / (1 + .0025))
        assert study.trades[name].cost.iloc[0] == pytest.approx(.0025 / (1 + .0025))
    for symbol in benchmarks:
        key = BENCHMARK_NAMES[symbol]
        assert len(study.trades[key]) == 1
        np.testing.assert_allclose(study.allocations[key].iloc[0], [1, 0])
        purchase_date = study.trades[key].index[0]
        expected_end = benchmarks[symbol].iloc[-1] / benchmarks.loc[purchase_date, symbol] / 1.0025
        assert study.equity[key].iloc[-1] == pytest.approx(expected_end)
    pd.testing.assert_frame_equal(result.benchmark_prices, benchmarks)


def test_all_equity_paths_replay_from_targets_and_reconcile_security_profit_less_cost():
    prices, benchmarks = changing_prices()
    result = run_selection(prices, benchmarks, target_volatility=.5, max_weight=1,
                           periods_per_year=12, train_fraction=.75, rebalance_every=3, cost_bps=35)
    study = result.test
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
    assert any("long unchanged-price screen excluded" in warning for warning in result.test.warnings)
    for name in (SELECTED, BASELINE):
        assert result.test.allocations[name][prices.columns[0]].eq(0).all()
    np.testing.assert_allclose(result.test.allocations[BASELINE][list(prices.columns[1:])], 1 / 11, atol=1e-12)
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
    ({"train_fraction": .999}, "at least two returns"),
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
    expected = [SELECTED, BASELINE, BENCHMARK_NAMES["QQQ"]]
    assert list(result.test.equity) == expected
    assert list(result.test.metrics.index) == expected
    assert not result.supported
    assert any("SPY" in reason and "unavailable" in reason for reason in result.reasons)
    reordered = run_selection(prices, benchmarks[["QQQ", "SPY"]], max_weight=1, target_volatility=.5)
    assert list(reordered.test.equity) == [SELECTED, BASELINE, *BENCHMARK_NAMES.values()]


def test_historical_check_requires_one_declared_year_of_later_returns():
    alternating = np.tile([-.001, .001], 40)
    stocks = prices_from_returns(np.column_stack([.01 + alternating, .002 - alternating]))
    markets = prices_from_returns(np.column_stack([.002 + alternating, .003 - alternating]), ["SPY", "QQQ"])
    supported = run_selection(stocks, markets, max_weight=.8, target_volatility=.2, periods_per_year=12)
    assert supported.supported
    assert not supported.reasons
    short = run_selection(stocks.iloc[:21], markets.iloc[:21], max_weight=.8,
                          target_volatility=.2, periods_per_year=12)
    assert not short.supported
    assert any("later test" in reason and "evidence floor" in reason for reason in short.reasons)


def test_realized_risk_above_the_budget_fails_the_historical_check():
    a = np.r_[.01 + np.tile([-.001, .001], 30), .03 + np.tile([-.2, .2], 30)]
    b = .002 + np.tile([-.001, .001], 60)
    stocks = prices_from_returns(np.column_stack([a, b]))
    markets = prices_from_returns(np.column_stack([b / 2, b]), ["SPY", "QQQ"])
    result = run_selection(stocks, markets, target_volatility=.25, max_weight=1,
                           train_fraction=.5, periods_per_year=12, rebalance_every=1000)
    assert result.test.metrics.loc[SELECTED, "volatility"] > .25
    assert not result.supported
    assert "Realized annual volatility in the later test exceeded the requested target." in result.reasons


def test_equal_weight_dominance_fails_the_historical_check():
    prices, benchmarks = changing_prices()
    result = run_selection(prices, benchmarks, target_volatility=.5, max_weight=1,
                           periods_per_year=6, train_fraction=.75, rebalance_every=1000)
    candidate = result.test.metrics.loc[SELECTED]
    equal = result.test.metrics.loc[BASELINE]
    assert equal.cagr > candidate.cagr
    assert equal.volatility <= candidate.volatility
    assert not result.supported
    assert any(reason.startswith(BASELINE) and "higher compound growth" in reason for reason in result.reasons)


@pytest.mark.parametrize("market_noise, dominated", [(.003, True), (.06, False)])
def test_raw_etf_dominance_fails_check_but_a_higher_risk_return_tradeoff_does_not(market_noise, dominated):
    sign = np.tile([-1., 1.], 80)
    a = .02 + sign * .02
    b = .002 - sign * .001
    stocks = prices_from_returns(np.column_stack([a, b]))
    market = .03 + sign * market_noise
    markets = prices_from_returns(np.column_stack([market, market]), ["SPY", "QQQ"])
    result = run_selection(stocks, markets, target_volatility=.25, max_weight=1,
                           train_fraction=.75, periods_per_year=12, rebalance_every=1000)
    candidate = result.test.metrics.loc[SELECTED]
    assert candidate.volatility < .25
    assert candidate.cagr > result.test.metrics.loc[BASELINE, "cagr"]
    for comparator in BENCHMARK_NAMES.values():
        raw = result.test.metrics.loc[comparator]
        assert raw.cagr > candidate.cagr
        assert bool(raw.volatility <= candidate.volatility) is dominated
    assert result.supported is not dominated
    if dominated:
        assert len(result.reasons) == 2
        assert all("higher compound growth and no more realized volatility" in reason for reason in result.reasons)
    else:
        assert not result.reasons
