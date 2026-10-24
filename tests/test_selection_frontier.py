import numpy as np
import pandas as pd
import pytest

from efficient_frontier.selection import (
    BENCHMARK_NAMES, SELECTED, run_selection, selection_frontier,
    split_stock_prices, validate_selection_benchmarks,
)


def prices():
    rng = np.random.default_rng(129)
    changes = rng.normal([.0004, .0012, .0007, .0005], [.008, .016, .01, .012], (80, 4))
    return pd.DataFrame(100 * np.vstack([np.ones(4), np.cumprod(1 + changes, axis=0)]),
                        index=pd.bdate_range('2020-01-01', periods=81), columns=['A', 'B', 'spy', 'QQQ'])


def test_embedded_benchmarks_are_separate_copies_with_the_original_calendar():
    original = prices()
    before = original.copy()
    stocks, references = split_stock_prices(original)
    assert list(stocks) == ['A', 'B']
    assert list(references) == ['SPY', 'QQQ']
    pd.testing.assert_index_equal(stocks.index, original.index)
    references.iloc[0, 0] = 999
    stocks.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(original, before)
    one, ref = split_stock_prices(original[['A', 'spy']])
    assert list(one) == ['A'] and list(ref) == ['SPY']
    with pytest.raises(ValueError, match='Provide stock price columns'):
        split_stock_prices(original[['spy', 'QQQ']])
    with pytest.raises(ValueError, match='comparison benchmarks'):
        run_selection(original, max_weight=.5)


def test_partial_benchmark_validation_rejects_duplicate_labels_and_calendar_changes():
    frame = prices()
    assert list(validate_selection_benchmarks(frame[['spy']], frame.index)) == ['SPY']
    with pytest.raises(ValueError, match='exactly the stock price dates'):
        validate_selection_benchmarks(frame[['QQQ']].iloc[1:], frame.index)
    with pytest.raises(ValueError, match='SPY and/or QQQ'):
        validate_selection_benchmarks(frame[['A']], frame.index)
    duplicate = frame[['spy', 'QQQ']].rename(columns={'QQQ': 'SPY'})
    with pytest.raises(ValueError, match='SPY and/or QQQ'):
        validate_selection_benchmarks(duplicate, frame.index)


@pytest.mark.parametrize('shrinkage', [0., .1])
def test_frontier_metrics_and_benchmarks_use_the_same_fit_dates(shrinkage):
    stocks, references = split_stock_prices(prices())
    study = run_selection(stocks, references, max_weight=.8, target_volatility=.25, shrinkage=shrinkage)
    curve, weights, comparison, _ = selection_frontier(study, stocks, frontier_points=7)
    returns = stocks.pct_change().iloc[1:]
    sample = returns.cov().to_numpy() * 252
    model = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    w = weights.to_numpy()
    assert list(weights) == ['A', 'B']
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert w.min() >= 0 and w.max() <= .8 + 1e-6
    np.testing.assert_allclose(curve.fitted_mean, w @ (returns.mean() * 252))
    np.testing.assert_allclose(curve.model_volatility, np.sqrt(np.einsum('ij,jk,ik->i', w, model, w)))
    np.testing.assert_allclose(curve.historical_volatility, np.sqrt(np.einsum('ij,jk,ik->i', w, sample, w)))
    assert comparison.loc[SELECTED, 'fitted_mean'] == pytest.approx(study.latest[SELECTED].expected_return)
    assert comparison.loc[SELECTED, 'model_volatility'] == pytest.approx(study.latest[SELECTED].volatility)
    for symbol, name in BENCHMARK_NAMES.items():
        daily = references[symbol].pct_change().iloc[1:]
        assert comparison.loc[name, 'fitted_mean'] == pytest.approx(daily.mean() * 252)
        assert comparison.loc[name, 'historical_volatility'] == pytest.approx(daily.std() * np.sqrt(252))


def test_default_selection_risk_matches_the_portfolios_sample_volatility():
    stocks, _ = split_stock_prices(prices())
    study = run_selection(stocks, max_weight=.8, target_volatility=.25)
    portfolio = study.latest[SELECTED]
    daily = stocks.pct_change().iloc[1:] @ portfolio.weights
    assert portfolio.volatility == pytest.approx(daily.std() * np.sqrt(252))
    with pytest.raises(ValueError, match='saved latest fit date'):
        selection_frontier(study, stocks.iloc[:-1])
