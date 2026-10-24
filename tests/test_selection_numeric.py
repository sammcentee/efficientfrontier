import numpy as np
import pandas as pd
import pytest

from efficient_frontier.selection import _fit_targets


@pytest.mark.parametrize("noise", [0, 1e-8, 1e-6])
@pytest.mark.parametrize("shrinkage", [0, .1])
def test_near_constant_returns_keep_the_maximum_mean_cap_solution(noise, shrinkage):
    returns = np.random.default_rng(0).normal([.0001, .0002], noise, (40, 2))
    prices = pd.DataFrame(
        100 * np.vstack([np.ones(2), np.cumprod(1 + returns, axis=0)]),
        index=pd.bdate_range("2020-01-01", periods=41), columns=["A", "B"],
    )
    weights, _, mean, covariance, _, _ = _fit_targets(prices, .2, .7, shrinkage, 252)

    # With a redundant risk bound, the larger-mean asset must reach its cap.
    np.testing.assert_allclose(weights, [.3, .7], atol=1e-7)
    assert float(mean @ weights) == pytest.approx(float(mean @ np.array([.3, .7])), abs=1e-10)
    assert np.sqrt(weights @ covariance @ weights) <= .2


def test_small_but_binding_variance_constraint_is_retained():
    returns = np.column_stack([
        .0001 + np.tile([-.0001, .0001], 20),
        .0002 + np.tile([-.001, -.001, .001, .001], 10),
    ])
    prices = pd.DataFrame(
        100 * np.vstack([np.ones(2), np.cumprod(1 + returns, axis=0)]),
        index=pd.bdate_range("2020-01-01", periods=41), columns=["A", "B"],
    )
    weights, _, _, covariance, _, _ = _fit_targets(prices, .005, .9, 1, 252)
    variances = np.diag(covariance)
    upper_weight = np.roots([variances.sum(), -2 * variances[0], variances[0] - .005 ** 2]).max()

    np.testing.assert_allclose(weights, [1 - upper_weight, upper_weight], atol=1e-7)
    assert weights.B < .9
    assert np.sqrt(weights @ covariance @ weights) == pytest.approx(.005, abs=1e-9)
