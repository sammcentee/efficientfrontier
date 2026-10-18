import numpy as np
import pandas as pd
import pytest

from research.reproduce_r_frontier import compare_r_frontier, fit_at_risk, r_reference_estimates


def test_original_r_flat_price_score_is_positive_despite_zero_returns():
    prices = pd.DataFrame({"A": np.full(1006, 100.0)}, index=pd.bdate_range("2020-01-02", periods=1006))
    mean, covariance, legacy = r_reference_estimates(prices)
    assert mean.A == 0
    assert covariance.loc["A", "A"] == 0
    assert legacy.A == pytest.approx(252 / 1006)


def test_corrected_mean_and_sample_covariance_match_explicit_return_formulas():
    prices = pd.DataFrame({"A": [100, 110, 99], "B": [100, 100, 120]},
                          index=pd.bdate_range("2020-01-01", periods=3))
    mean, covariance, legacy = r_reference_estimates(prices)
    np.testing.assert_allclose(mean, [0, 25.2], atol=1e-12)
    np.testing.assert_allclose(covariance, [[5.04, -5.04], [-5.04, 5.04]], atol=1e-12)
    np.testing.assert_allclose(legacy, [83.16, 100.8], atol=1e-12)


@pytest.mark.parametrize("cap", [1.0, .6])
def test_reference_solver_matches_analytic_two_stock_frontier_and_global_bound(cap):
    mean = pd.Series([.1, .2], index=["A", "B"])
    covariance = pd.DataFrame(np.diag([.01, .04]), index=mean.index, columns=mean.index)
    ceiling = .13
    # If B has weight x, variance is .01*(1-x)^2 + .04*x^2.
    root = (.02 + np.sqrt(.02 ** 2 - 4 * .05 * (.01 - ceiling ** 2))) / .10
    expected_b = min(cap, root)
    weights, certificate = fit_at_risk(mean, covariance, ceiling, cap)
    np.testing.assert_allclose(weights, [1 - expected_b, expected_b], atol=2e-7)
    exact_mean = .1 + .1 * expected_b
    assert certificate["fitted_stock_mean"] == pytest.approx(exact_mean, abs=1e-8)
    assert certificate["global_upper_bound_mean"] >= exact_mean - 1e-8
    assert certificate["global_upper_bound_mean"] <= exact_mean + 1e-6
    assert certificate["fitted_stock_risk"] <= ceiling + 1e-8


def test_reference_rejects_an_impossible_holding_cap():
    mean = pd.Series([.1, .2], index=["A", "B"])
    covariance = pd.DataFrame(np.eye(2), index=mean.index, columns=mean.index)
    with pytest.raises(ValueError, match="holding cap is infeasible"):
        fit_at_risk(mean, covariance, .3, .1)


def test_benchmark_dates_cannot_differ_or_etfs_enter_the_stock_universe():
    index = pd.bdate_range("2020-01-01", periods=3)
    stocks = pd.DataFrame({"A": [100, 102, 101]}, index=index)
    benchmarks = pd.DataFrame({"SPY": [100, 101, 100], "QQQ": [100, 102, 100]}, index=index)
    with pytest.raises(ValueError, match="exactly the stock price dates"):
        compare_r_frontier(stocks, benchmarks.iloc[1:])
    with pytest.raises(ValueError, match="cannot be eligible stock holdings"):
        compare_r_frontier(stocks.rename(columns={"A": "SPY"}), benchmarks)
