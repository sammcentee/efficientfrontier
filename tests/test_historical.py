from io import StringIO

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.backtest import run_backtests
from efficient_frontier.historical import (
    historical_demo, load_historical_prices, run_historical_backtests, validate_historical_prices,
)
from efficient_frontier.membership import load_membership_csv


EQUAL = "Rolling window · Equal weight"


@pytest.fixture
def prices():
    rng = np.random.default_rng(913)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.001, .015, (45, 3)), axis=0)),
                        index=pd.bdate_range("2021-01-04", periods=45, name="Date"), columns=["A", "B", "C"])


def history(prices, changes=None):
    changes = changes or [(0, "A,B"), (26, "A,C,MISSING"), (32, "A,B,C")]
    table = pd.DataFrame({"date": [str(prices.index[row].date()) for row, _ in changes],
                          "tickers": [symbols for _, symbols in changes]})
    return load_membership_csv(StringIO(table.to_csv(index=False)), coverage_end=str(prices.index[-1].date()))


def run(prices, membership=None, **kwargs):
    options = dict(train_fraction=.5, rolling_window=5, rebalance_every=3, cost_bps=10.)
    return run_historical_backtests(prices, membership or history(prices), **(options | kwargs))


def test_addition_departure_and_reentry_apply_at_trade_date(prices):
    result = run(prices)
    weights = result.allocations[EQUAL]
    assert weights.loc[prices.index[23], "C"] == 0
    assert weights.loc[prices.index[26], "C"] == .5
    assert weights.loc[prices.index[26], "B"] == 0
    assert weights.loc[prices.index[32], "B"] == pytest.approx(1 / 3)
    assert (weights.MISSING == 0).all()
    missing = result.coverage[result.coverage.symbol == "MISSING"]
    assert set(missing.status) == {"excluded"}
    assert missing.reason.str.contains("No price column").all()
    assert any("survivorship bias" in note for note in result.warnings)


def test_effective_change_between_trades_sells_at_next_scheduled_trade(prices):
    membership = history(prices, [(0, "A,B"), (24, "A,C")])
    result = run(prices, membership, cost_bps=0)
    assert result.allocations[EQUAL].loc[prices.index[23], "B"] == .5
    assert result.allocations[EQUAL].loc[prices.index[26], "B"] == 0
    expected = .5 * prices.loc[prices.index[25], ["A", "B"]] / prices.loc[prices.index[23], ["A", "B"]]
    assert result.equity.loc[prices.index[25], EQUAL] == pytest.approx(expected.sum())


def test_short_history_becomes_eligible_using_only_prior_window(prices):
    prices.loc[prices.index[:24], "C"] = np.nan
    result = run(prices)
    assert result.allocations[EQUAL].loc[prices.index[26], "C"] == 0
    assert result.allocations[EQUAL].loc[prices.index[29], "C"] == 0
    assert result.allocations[EQUAL].loc[prices.index[32], "C"] == pytest.approx(1 / 3)
    assert "complete prior prices" in result.coverage.query("symbol == 'C'").iloc[0].reason
    assert np.isfinite(result.equity.to_numpy()).all()


def test_future_prices_and_future_rosters_do_not_change_prior_targets(prices):
    original = run(prices)
    altered = prices.copy()
    altered.loc[prices.index[33]:] *= [3, .2, 7]
    future_roster = history(prices, [(0, "A,B"), (26, "A,C,MISSING"), (32, "A,B,C"), (33, "A,C")])
    changed = run(altered, future_roster)
    for name in original.allocations:
        pd.testing.assert_frame_equal(original.allocations[name].loc[:prices.index[32]],
                                       changed.allocations[name].loc[:prices.index[32]], atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(original.equity.loc[:prices.index[32]], changed.equity.loc[:prices.index[32]])


def test_execution_day_prices_cannot_change_that_days_targets_or_earn_first_return(prices):
    membership = history(prices, [(0, "A,B,C")])
    original = run(prices, membership)
    altered = prices.copy()
    altered.iloc[23] *= [10, .5, 2]
    changed = run(altered, membership)
    for name in original.allocations:
        pd.testing.assert_series_equal(original.allocations[name].iloc[0], changed.allocations[name].iloc[0])
        assert changed.equity[name].iloc[1] == pytest.approx(1 / 1.001)
        trades = changed.trades[name]
        assert (pd.to_datetime(trades.train_end) < trades.index).all()


def test_future_price_gap_cannot_silently_exclude_a_survivor(prices):
    prices.loc[prices.index[40], "C"] = np.nan
    with pytest.raises(ValueError, match="Missing price for held asset.*C"):
        run(prices)


def test_missing_execution_price_fails_instead_of_excluding_selected_new_member(prices):
    prices.loc[prices.index[26], "C"] = np.nan
    with pytest.raises(ValueError, match="Missing execution price.*C"):
        run(prices)


def test_missing_departed_price_is_not_a_free_liquidation(prices):
    membership = history(prices, [(0, "A,B"), (24, "A,C")])
    prices.loc[prices.index[24], "B"] = np.nan
    with pytest.raises(ValueError, match="Missing price for held asset.*B"):
        run(prices, membership)


def test_cash_exit_reentry_fees_and_contributions_reconcile(prices):
    membership = history(prices, [(0, "A,B"), (26, "A"), (32, "A,B")])
    result = run(prices, membership, max_weight=.5, cost_bps=100)
    for name in result.equity:
        trades = result.trades[name]
        exit_trade = trades.loc[prices.index[26]]
        entry_trade = trades.loc[prices.index[32]]
        assert exit_trade.nav_after == pytest.approx(exit_trade.nav_before * .99)
        assert exit_trade.turnover == pytest.approx(1.)
        assert entry_trade.nav_after == pytest.approx(entry_trade.nav_before / 1.01)
        assert (result.cash_weights.loc[prices.index[26]:prices.index[31], name] == 1).all()
        assert result.allocations[name].loc[prices.index[26]].sum() == 0
        assert result.cash_weights.loc[prices.index[32], name] == 0
        assert result.holdings[name].pnl_contribution.sum() - trades.cost.sum() == pytest.approx(
            result.metrics.loc[name, "total_return"], abs=1e-12)
        assert result.holdings[name].end_weight.sum() + result.cash_weights[name].iloc[-1] == pytest.approx(1.)
    assert any("cash" in message for message in result.warnings)


def test_empty_price_coverage_remains_cash_with_complete_audit(prices):
    membership = history(prices, [(0, "UNPRICED")])
    result = run(prices, membership)
    assert (result.equity == 1).all().all()
    assert (result.cash_weights == 1).all().all()
    assert (result.coverage.status == "excluded").all()
    assert (result.metrics.total_cost == 0).all()
    for weights in result.allocations.values():
        assert (weights == 0).all().all()


def test_one_eligible_member_is_valid_when_cap_is_one(prices):
    result = run(prices, history(prices, [(0, "A")]))
    for weights in result.allocations.values():
        np.testing.assert_allclose(weights.A, 1)


def test_fixed_roster_matches_existing_rolling_engine(prices):
    membership = history(prices, [(0, "A,B,C")])
    options = dict(train_fraction=.5, rolling_window=5, rebalance_every=3, cost_bps=37, max_weight=.5,
                   periods_per_year=12, shrinkage=.3, risk_free_rate=.01)
    historical = run(prices, membership, **options)
    existing = run_backtests(prices, **options)
    pd.testing.assert_frame_equal(historical.equity, existing.equity[historical.equity.columns])
    pd.testing.assert_frame_equal(historical.metrics, existing.metrics.loc[historical.metrics.index])
    for name in historical.equity:
        pd.testing.assert_frame_equal(historical.holdings[name], existing.holdings[name], check_names=False)
        pd.testing.assert_frame_equal(historical.allocations[name], existing.allocations[name], check_names=False)


def test_identity_hazard_in_training_range_blocks_otherwise_current_safe_symbol(prices, monkeypatch):
    membership = history(prices, [(0, "A,B,C")])
    calls = []

    def hazards(start, *, through):
        calls.append((start, through))
        return {"C": "Ticker identifies another security before this date."} if start <= prices.index[21] else {}

    monkeypatch.setattr(type(membership), "identity_exclusions", lambda self, start, *, through: hazards(start, through=through))
    result = run(prices, membership)
    assert calls[0] == (prices.index[17], prices.index[23])
    assert result.allocations[EQUAL].loc[prices.index[23], "C"] == 0
    assert result.allocations[EQUAL].loc[prices.index[29], "C"] == pytest.approx(1 / 3)
    assert "another security" in result.coverage.query("symbol == 'C'").iloc[0].reason


def test_membership_coverage_cannot_be_extended_by_last_snapshot(prices):
    membership = load_membership_csv(StringIO('date,tickers\n2021-01-04,"A,B"\n'), coverage_end="2021-02-15")
    with pytest.raises(ValueError, match="coverage"):
        run(prices, membership)


@pytest.mark.parametrize("key,value", [
    ("train_fraction", 1), ("train_fraction", True), ("rolling_window", 1), ("rolling_window", 23),
    ("rolling_window", True), ("rebalance_every", 0), ("rebalance_every", 1.5),
    ("max_weight", 0), ("max_weight", float("inf")), ("shrinkage", -1), ("cost_bps", 10000),
    ("risk_free_rate", float("nan")), ("periods_per_year", 0),
])
def test_invalid_settings_fail(prices, key, value):
    with pytest.raises(ValueError, match=key):
        run(prices, **{key: value})


@pytest.mark.parametrize("value", [True, complex(2, 1), -1, 0, float("inf"), "bad"])
def test_invalid_price_values_fail(prices, value):
    prices = prices.astype(object)
    prices.iloc[0, 0] = value
    with pytest.raises(ValueError, match="price|Price"):
        validate_historical_prices(prices)


def test_sparse_csv_roundtrip_preserves_missing_rows(prices):
    prices.iloc[:10, 2] = np.nan
    result = load_historical_prices(StringIO(prices.to_csv()))
    pd.testing.assert_frame_equal(result, prices, check_freq=False)


@pytest.mark.parametrize("text", [
    "", "day,A\n2021-01-01,1\n", "Date,A,A\n2021-01-01,1,2\n", "Date,A\n2021-01-01,1,2\n",
])
def test_bad_csv_structure_fails(text):
    with pytest.raises(ValueError):
        load_historical_prices(StringIO(text))


def test_demo_exercises_eligibility_changes_and_reconciles():
    prices, membership = historical_demo()
    result = run_historical_backtests(prices, membership)
    weights = result.allocations[EQUAL]
    assert (weights.NEWCO == 0).any() and (weights.NEWCO > 0).any()
    assert (weights.BETA == 0).any() and (weights.BETA > 0).any()
    assert (weights.GAMMA == 0).any() and (weights.GAMMA > 0).any()
    for name in result.equity:
        assert result.holdings[name].pnl_contribution.sum() - result.metrics.loc[name, "total_cost"] == pytest.approx(
            result.metrics.loc[name, "total_return"], abs=1e-12)
