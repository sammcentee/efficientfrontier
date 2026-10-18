"""Risk tie-breaks must preserve the maximum-mean objective."""

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.selection import _fit_targets


def price_frame(returns):
    return pd.DataFrame(
        100 * np.vstack([np.ones(returns.shape[1]), np.cumprod(1 + returns, axis=0)]),
        columns=list("ABC")[:returns.shape[1]],
        index=pd.bdate_range("2020-01-01", periods=len(returns) + 1),
    )


def tied_returns():
    return np.column_stack([
        .001 + np.tile([-.005, .005], 40),
        .001 + np.tile([-.03, -.03, .03, .03], 20),
    ])


@pytest.mark.parametrize("risk_ceiling", [.5, .08])
def test_equal_means_choose_analytic_minimum_risk_even_near_the_risk_boundary(risk_ceiling):
    prices = price_frame(tied_returns())
    selected, equal, mean, covariance, _, _ = _fit_targets(prices, risk_ceiling, 1., 0., 252.)
    expected = np.array([36 / 37, 1 / 37])
    np.testing.assert_allclose(selected, expected, atol=2e-7)
    np.testing.assert_allclose(equal, [.5, .5], atol=1e-12)
    assert float(mean @ selected) == pytest.approx(.252, abs=1e-12)
    assert np.sqrt(selected @ covariance @ selected) == pytest.approx(.078786555, abs=1e-9)
    assert np.sqrt(equal @ covariance @ equal) > .24


def test_tied_top_returns_use_the_cap_and_do_not_mix_in_the_lower_mean_stock():
    values = tied_returns() + .001
    values = np.column_stack([values, .001 + np.tile([-.008, .008, .008, -.008], 20)])
    selected, _, mean, _, _, _ = _fit_targets(price_frame(values), .5, .7, 0., 252.)
    np.testing.assert_allclose(selected, [.7, .3, 0], atol=1e-7)
    assert mean @ selected >= mean.max() - 2e-8 * np.abs(mean).max()


def test_distinct_return_optimum_does_not_add_a_second_solve(monkeypatch):
    from efficient_frontier import selection

    original = selection._solve
    calls = []

    def count(problem):
        calls.append(problem)
        original(problem)

    monkeypatch.setattr(selection, "_solve", count)
    values = tied_returns()
    values[:, 1] += .001
    selected, _, _, _, _, _ = _fit_targets(price_frame(values), .5, .7, 0., 252.)
    np.testing.assert_allclose(selected, [.3, .7], atol=1e-7)
    assert len(calls) == 1


def test_active_risk_bound_below_a_tied_return_ceiling_keeps_primary_optimum(monkeypatch):
    from efficient_frontier import selection

    original = selection._solve
    calls = []

    def count(problem):
        calls.append(problem)
        original(problem)

    monkeypatch.setattr(selection, "_solve", count)
    values = np.column_stack([
        .001 + np.tile([-.005, .005], 40),
        .002 + np.tile([-.03, -.03, .03, .03], 20),
        .002 + np.tile([-.03, -.03, .03, .03], 20),
    ])
    selected, _, mean, covariance, _, _ = _fit_targets(price_frame(values), .10, 1., 0., 252.)
    assert len(calls) == 1
    # B and C have identical returns. Their combined optimum solves a quadratic.
    variance_a, variance_b = covariance.iloc[0, 0], covariance.iloc[1, 1]
    b = np.roots([variance_a + variance_b, -2 * variance_a, variance_a - .10 ** 2]).max()
    assert selected.B + selected.C == pytest.approx(b, abs=2e-7)
    assert np.sqrt(selected @ covariance @ selected) == pytest.approx(.10, abs=1e-7)
    assert mean @ selected == pytest.approx(mean.A * (1 - b) + mean.B * b, abs=1e-7)
