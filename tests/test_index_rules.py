import numpy as np
import pandas as pd
import pytest

from efficient_frontier.index_rules import (
    BENCHMARKS, FUNDS, STOCK_RULES, TREND_RULE, _capped_inverse_volatility,
    _signals, _stock_targets, run_fund_rules, run_index_rules,
)


@pytest.fixture
def stocks():
    rng = np.random.default_rng(20261018)
    dates = pd.bdate_range("2018-12-31", periods=650)
    values = rng.normal(np.linspace(-.0002, .0012, 120), np.linspace(.004, .025, 120), (649, 120))
    values += rng.normal(0, .008, (649, 1))
    return pd.DataFrame(100 * np.vstack([np.ones(120), np.cumprod(1 + values, axis=0)]),
                        index=dates, columns=[f"STOCK{i:03d}" for i in range(120)])


@pytest.fixture
def funds(stocks):
    rng = np.random.default_rng(1018)
    values = rng.normal(.0005, np.linspace(.01, .025, len(FUNDS)), (len(stocks) - 1, len(FUNDS)))
    return pd.DataFrame(100 * np.vstack([np.ones(len(FUNDS)), np.cumprod(1 + values, axis=0)]),
                        index=stocks.index, columns=FUNDS)


def test_fixed_stock_rules_have_exact_counts_caps_and_next_month_execution(stocks, funds):
    study = run_index_rules(stocks, funds[list(BENCHMARKS)])
    assert list(study.equity) == [*STOCK_RULES, *BENCHMARKS]
    for name in STOCK_RULES:
        weights = study.allocations[name].drop(columns="CASH")
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        assert (study.allocations[name].CASH.abs() < 1e-12).all()
        assert (weights >= 0).all().all()
        if "momentum" in name:
            assert weights.gt(0).sum(axis=1).eq(50).all()
            np.testing.assert_allclose(weights.to_numpy()[weights.to_numpy() > 0], .02)
        if name == "low_volatility_100":
            assert weights.gt(0).sum(axis=1).eq(100).all()
        if "volatility" in name:
            assert weights.max().max() <= .1 + 1e-12
        trades = study.trades[name]
        for execution, signal in trades.signal_date.items():
            assert stocks.index[stocks.index.get_loc(signal) + 1] == execution
            assert signal.to_period("M") < execution.to_period("M")
            assert execution < stocks.index[-1]
    assert len(study.trades["SPY"]) == 1
    assert len(study.trades["QQQ"]) == 1
    signal = study.trades["low_volatility_100"].iloc[0].signal_date
    volatility = stocks.loc[:signal].iloc[-253:].pct_change().iloc[1:].std(ddof=1)
    lowest = volatility.sort_values().head(100).index
    allocation = study.allocations["low_volatility_100"].iloc[0].drop("CASH")
    assert set(allocation[allocation > 0].index) == set(lowest)


def test_inverse_weights_redistribute_the_capped_share():
    volatility = pd.Series([.001, *np.linspace(1, 2, 19)], index=[f"A{i}" for i in range(20)])
    weights = _capped_inverse_volatility(volatility)
    assert weights.iloc[0] == pytest.approx(.1)
    np.testing.assert_allclose(weights.iloc[1:], .9 * (1 / volatility.iloc[1:]) / (1 / volatility.iloc[1:]).sum())
    assert weights.sum() == pytest.approx(1)


def test_momentum_uses_exact_calendar_window_and_ignores_latest_month(stocks):
    ends, signals = _signals(stocks, None)
    signal, month = signals[0]
    targets, _ = _stock_targets(stocks, signal, ends, month)
    begin, end = ends[month - 12], ends[month - 1]
    momentum = stocks.iloc[end] / stocks.iloc[begin] - 1
    volatility = stocks.iloc[begin:end + 1].pct_change().iloc[1:].std(ddof=1)
    for name, score in (("momentum_50", momentum), ("risk_adjusted_momentum_50", momentum / volatility)):
        expected = score.sort_index().sort_values(ascending=False, kind="stable").head(50).index
        assert set(stocks.columns[targets[name] > 0]) == set(expected)
    changed = stocks.copy()
    changed.iloc[end + 1:signal + 1, 0] *= 5
    altered, _ = _stock_targets(changed, signal, ends, month)
    np.testing.assert_array_equal(targets["momentum_50"], altered["momentum_50"])
    np.testing.assert_array_equal(targets["risk_adjusted_momentum_50"], altered["risk_adjusted_momentum_50"])


def test_future_prices_cannot_change_prior_signals_or_realized_history(stocks, funds):
    original = run_index_rules(stocks, funds[list(BENCHMARKS)], target_volatility=.1)
    cutoff = original.trades["equal_weight"].index[3]
    changed = stocks.copy()
    changed.loc[changed.index > cutoff, changed.columns[:60]] *= 1.7
    changed_funds = funds.copy()
    changed_funds.loc[changed_funds.index > cutoff] *= .8
    altered = run_index_rules(changed, changed_funds[list(BENCHMARKS)], target_volatility=.1)
    pd.testing.assert_frame_equal(original.equity.loc[:cutoff], altered.equity.loc[:cutoff])
    for name in original.allocations:
        pd.testing.assert_frame_equal(original.allocations[name].loc[:cutoff], altered.allocations[name].loc[:cutoff])


def test_first_interval_is_cash_and_raw_funds_drift_without_rebalance(funds):
    initial = run_fund_rules(funds, cost_bps=10)
    first = initial.trades["SPY"].index[0]
    changed = funds.copy()
    changed.loc[first, "SPY"] *= 2
    study = run_fund_rules(changed, cost_bps=10)
    assert study.equity.iloc[0].eq(1).all()
    assert study.equity.loc[first, "SPY"] == pytest.approx(1 / 1.001)
    expected = changed.SPY.loc[first:] / changed.loc[first, "SPY"] / 1.001
    np.testing.assert_allclose(study.equity.SPY.loc[first:], expected)
    assert len(study.trades["SPY"]) == 1


@pytest.mark.parametrize("overlay", [None, .1])
def test_positions_cash_and_fees_reconcile_at_each_trade(stocks, funds, overlay):
    study = run_index_rules(stocks, funds[list(BENCHMARKS)], cost_bps=25, target_volatility=overlay)
    for name in study.equity:
        prices = stocks if name in STOCK_RULES else funds[[name]]
        dates = study.equity.index
        positions = np.zeros(len(prices.columns))
        cash = 1.
        for previous, day in zip(dates[:-1], dates[1:]):
            positions *= (prices.loc[day] / prices.loc[previous]).to_numpy()
            before = float(positions.sum() + cash)
            if day in study.trades[name].index:
                trade = study.trades[name].loc[day]
                target = study.allocations[name].loc[day]
                after = float(trade.nav_after)
                next_positions = after * target.drop("CASH").to_numpy()
                fee = .0025 * np.abs(next_positions - positions).sum()
                assert before == pytest.approx(trade.nav_before)
                assert fee == pytest.approx(trade.cost, abs=1e-12)
                assert after + fee == pytest.approx(before)
                positions, cash = next_positions, after * target.CASH
            assert positions.sum() + cash == pytest.approx(study.equity.loc[day, name])
        assert (study.holdings[name].pnl_contribution.sum() - study.metrics.loc[name, "total_cost"]
                == pytest.approx(study.metrics.loc[name, "total_return"], abs=1e-12))


def test_overlay_uses_covariance_of_desired_weights_and_holds_cash(stocks, funds):
    raw = run_index_rules(stocks, funds[list(BENCHMARKS)], cost_bps=0)
    wrapped = run_index_rules(stocks, funds[list(BENCHMARKS)], cost_bps=0, target_volatility=.05)
    for name in wrapped.allocations:
        prices = stocks if name in STOCK_RULES else funds[[name]]
        for day, allocation in wrapped.allocations[name].iterrows():
            signal = wrapped.trades[name].loc[day, "signal_date"]
            weights = raw.allocations[name].loc[day].drop("CASH") if name in STOCK_RULES else pd.Series(1., index=[name])
            history = prices.loc[:signal].iloc[-127:].pct_change().iloc[1:]
            volatility = float((history @ weights).std(ddof=1) * np.sqrt(252))
            expected = weights * min(1, .05 / volatility)
            np.testing.assert_allclose(allocation.drop("CASH"), expected, atol=1e-12)
            assert allocation.CASH == pytest.approx(1 - expected.sum())
            assert allocation.sum() == pytest.approx(1)
            assert allocation.CASH >= -1e-12
    assert wrapped.allocations["QQQ"].CASH.gt(0).all()


def test_qqq_trend_uses_last_ten_completed_month_ends_and_next_close(funds):
    study = run_fund_rules(funds, cost_bps=0)
    for execution, signal in study.trades[TREND_RULE].signal_date.items():
        ends = funds.loc[:signal, "QQQ"].groupby(funds.loc[:signal].index.to_period("M")).last().iloc[-10:]
        expected = float(ends.iloc[-1] > ends.mean())
        assert study.allocations[TREND_RULE].loc[execution, "QQQ"] == expected
        assert study.allocations[TREND_RULE].loc[execution, "CASH"] == 1 - expected
    cutoff = study.trades[TREND_RULE].index[2]
    changed = funds.copy()
    changed.loc[cutoff:, "QQQ"] *= 2
    altered = run_fund_rules(changed, cost_bps=0)
    pd.testing.assert_frame_equal(study.allocations[TREND_RULE].loc[:cutoff], altered.allocations[TREND_RULE].loc[:cutoff])


def test_high_volatility_budget_never_borrows_or_exceeds_full_investment(funds):
    raw = run_fund_rules(funds, cost_bps=0)
    wrapped = run_fund_rules(funds, cost_bps=0, target_volatility=1.)
    np.testing.assert_allclose(raw.equity, wrapped.equity, atol=1e-12)
    for name in FUNDS:
        np.testing.assert_allclose(wrapped.allocations[name][name], 1)
        np.testing.assert_allclose(wrapped.allocations[name].CASH, 0)


def test_stock_screen_uses_entire_prior_history_and_blocks_unverified_sales(stocks, funds):
    changed = stocks.copy()
    changed.iloc[:21, 0] = changed.iloc[0, 0]
    screened = run_index_rules(changed, funds[list(BENCHMARKS)])
    assert screened.allocations["equal_weight"].iloc[:, 0].eq(0).all()
    original = run_index_rules(stocks, funds[list(BENCHMARKS)])
    first = stocks.index.get_loc(original.trades["equal_weight"].index[0])
    changed = stocks.copy()
    changed.iloc[first:first + 32, 0] = changed.iloc[first, 0]
    with pytest.raises(ValueError, match="held stock triggered"):
        run_index_rules(changed, funds[list(BENCHMARKS)])


def test_start_waits_for_next_eligible_execution_and_never_trades_last_row(funds):
    start = pd.Timestamp("2020-05-15")
    study = run_fund_rules(funds, start=start, target_volatility=.1)
    assert study.equity.index[1] == pd.Timestamp("2020-06-01")
    end = study.trades["SPY"].index[2]
    truncated = run_fund_rules(funds.loc[:end], start=start, target_volatility=.1)
    assert end not in truncated.trades["SPY"].index


@pytest.mark.parametrize("cost,target", [(-1, None), (10000, None), (True, None), (10, 0), (10, float("nan"))])
def test_invalid_cost_or_risk_parameters_fail(funds, cost, target):
    with pytest.raises(ValueError):
        run_fund_rules(funds, cost_bps=cost, target_volatility=target)


def test_incomplete_inputs_and_insufficient_history_fail(stocks, funds):
    with pytest.raises(ValueError, match="100 eligible"):
        run_index_rules(stocks.iloc[:, :99], funds[list(BENCHMARKS)])
    with pytest.raises(ValueError, match="exactly the stock dates"):
        run_index_rules(stocks, funds[list(BENCHMARKS)].iloc[1:])
    with pytest.raises(ValueError, match="13 month ends"):
        run_fund_rules(funds.iloc[:253])
    with pytest.raises(ValueError, match="Fund prices must contain"):
        run_fund_rules(funds.drop(columns="QUAL"))
