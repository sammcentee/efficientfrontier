from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.index_rules import run_index_rules
from research.exploratory_core_momentum import FUND_CORE, STOCK_CORE, run_core_momentum


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(7525)
    dates = pd.bdate_range("2015-10-12", periods=650)
    values = rng.normal(.0003, .015, (649, 120))
    stocks = pd.DataFrame(100 * np.vstack([np.ones(120), np.cumprod(1 + values, axis=0)]),
                          index=dates, columns=[f"STOCK{i:03d}" for i in range(120)])
    values = rng.normal(.0004, [.018, .025, .03], (649, 3))
    funds = pd.DataFrame(100 * np.vstack([np.ones(3), np.cumprod(1 + values, axis=0)]),
                         index=dates, columns=["SPY", "QQQ", "SPMO"])
    raw = run_index_rules(stocks, funds[["SPY", "QQQ"]])
    return stocks, funds, raw


def test_stock_core_reuses_saved_targets_at_fixed_75_25_weights(inputs):
    stocks, funds, raw = inputs
    study = run_core_momentum(stocks, funds[["SPY", "QQQ"]], raw)
    assert list(study.equity) == [STOCK_CORE, "SPY", "QQQ"]
    targets = study.allocations[STOCK_CORE]
    np.testing.assert_allclose(targets.SPY, .75)
    np.testing.assert_allclose(targets.CASH, 0)
    np.testing.assert_allclose(targets[stocks.columns], .25 * raw.allocations["momentum_50"][stocks.columns])
    assert targets[stocks.columns].gt(0).sum(axis=1).eq(50).all()
    np.testing.assert_allclose(targets[stocks.columns].sum(axis=1), .25)
    pd.testing.assert_series_equal(study.trades[STOCK_CORE].signal_date, raw.trades["momentum_50"].signal_date)
    assert study.settings["exploratory_after_prior_results"]
    assert study.settings["core_weight"] == .75
    assert study.settings["momentum_weight"] == .25


def test_fund_proxy_has_the_same_warmup_and_exact_monthly_ratio(inputs):
    _, funds, _ = inputs
    study = run_core_momentum(funds, funds[["SPY", "QQQ"]])
    assert study.equity.index[0] == pd.Timestamp("2016-10-31")
    assert study.equity.index[1] == pd.Timestamp("2016-11-01")
    np.testing.assert_allclose(study.allocations[FUND_CORE][["SPY", "SPMO"]],
                               np.tile([.75, .25], (len(study.allocations[FUND_CORE]), 1)))
    assert "differs" in study.settings["momentum_source"]
    for execution, signal in study.trades[FUND_CORE].signal_date.items():
        assert funds.index[funds.index.get_loc(signal) + 1] == execution
        assert signal.to_period("M") < execution.to_period("M")


@pytest.mark.parametrize("stock_mode", [False, True])
def test_candidate_overlay_uses_combined_past_risk_and_keeps_benchmarks_buy_and_hold(inputs, stock_mode):
    stocks, funds, saved = inputs
    prices, momentum = (stocks, saved) if stock_mode else (funds, None)
    name = STOCK_CORE if stock_mode else FUND_CORE
    study = run_core_momentum(prices, funds[["SPY", "QQQ"]], momentum, target_volatility=.1)
    raw = run_core_momentum(prices, funds[["SPY", "QQQ"]], momentum)
    source = pd.concat([stocks, funds[["SPY"]]], axis=1) if stock_mode else funds[["SPY", "SPMO"]]
    for date, row in study.allocations[name].iterrows():
        desired = raw.allocations[name].loc[date].drop("CASH")
        signal = study.trades[name].loc[date, "signal_date"]
        history = source.loc[:signal].iloc[-127:].pct_change(fill_method=None).iloc[1:]
        volatility = float((history @ desired).std(ddof=1) * np.sqrt(252))
        fraction = min(1, .1 / volatility)
        np.testing.assert_allclose(row.drop("CASH"), fraction * desired, atol=1e-12)
        assert row.CASH == pytest.approx(1 - fraction)
        assert row.sum() == pytest.approx(1)
    for name in ["SPY", "QQQ"]:
        assert len(study.trades[name]) == 1
        np.testing.assert_allclose(study.allocations[name].iloc[0], [1, 0])
        pd.testing.assert_series_equal(study.equity[name], raw.equity[name])


def test_future_changes_do_not_refit_saved_weights_or_change_earlier_overlay(inputs):
    stocks, funds, saved = inputs
    original = run_core_momentum(stocks, funds[["SPY", "QQQ"]], saved, target_volatility=.1)
    cutoff = original.trades[STOCK_CORE].index[2]
    changed = stocks.copy()
    changed.loc[changed.index > cutoff, changed.columns[:50]] *= 1.8
    new_funds = funds.copy()
    new_funds.loc[new_funds.index > cutoff, "SPY"] *= .8
    altered = run_core_momentum(changed, new_funds[["SPY", "QQQ"]], saved, target_volatility=.1)
    pd.testing.assert_frame_equal(original.equity.loc[:cutoff], altered.equity.loc[:cutoff])
    pd.testing.assert_frame_equal(original.allocations[STOCK_CORE].loc[:cutoff],
                                  altered.allocations[STOCK_CORE].loc[:cutoff])
    raw = run_core_momentum(changed, new_funds[["SPY", "QQQ"]], saved)
    np.testing.assert_allclose(raw.allocations[STOCK_CORE][stocks.columns],
                               .25 * saved.allocations["momentum_50"][stocks.columns])


@pytest.mark.parametrize("stock_mode", [False, True])
def test_core_lag_and_all_security_fees_reconcile(inputs, stock_mode):
    stocks, funds, saved = inputs
    prices, momentum = (stocks, saved) if stock_mode else (funds, None)
    name = STOCK_CORE if stock_mode else FUND_CORE
    study = run_core_momentum(prices, funds[["SPY", "QQQ"]], momentum, cost_bps=25)
    assert study.equity.iloc[0].eq(1).all()
    np.testing.assert_allclose(study.equity.iloc[1], 1 / 1.0025)
    sources = {name: pd.concat([stocks, funds[["SPY"]]], axis=1) if stock_mode else funds[["SPY", "SPMO"]],
               "SPY": funds[["SPY"]], "QQQ": funds[["QQQ"]]}
    for key, source in sources.items():
        shares = pd.Series(0., index=source.columns)
        cash = 1.
        for date in study.equity.index[1:]:
            before = float(shares @ source.loc[date] + cash)
            if date in study.trades[key].index:
                trade = study.trades[key].loc[date]
                weights = study.allocations[key].loc[date]
                after_positions = trade.nav_after * weights.drop("CASH")
                fee = .0025 * float((after_positions - shares * source.loc[date]).abs().sum())
                assert fee == pytest.approx(trade.cost, abs=1e-12)
                assert before == pytest.approx(trade.nav_before)
                shares = after_positions / source.loc[date]
                cash = trade.nav_after * weights.CASH
            assert float(shares @ source.loc[date] + cash) == pytest.approx(study.equity.loc[date, key])


def test_saved_noncausal_dates_and_wrapped_allocations_are_rejected(inputs):
    stocks, funds, saved = inputs
    trades = {key: value.copy() for key, value in saved.trades.items()}
    trades["momentum_50"].loc[:, "signal_date"] = trades["momentum_50"].index
    with pytest.raises(ValueError, match="next close"):
        run_core_momentum(stocks, funds[["SPY", "QQQ"]], replace(saved, trades=trades))
    with pytest.raises(ValueError, match="raw momentum"):
        run_core_momentum(stocks, funds[["SPY", "QQQ"]],
                          replace(saved, settings={**saved.settings, "target_volatility": .1}))
