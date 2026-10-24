import numpy as np
import pandas as pd
import pytest

from efficient_frontier.index_comparison import normalized_metrics, paired_normalized_bootstrap


def returns_frame(values, names=("Rule", "SPY", "QQQ")):
    return pd.DataFrame(values, index=pd.bdate_range("2020-01-01", periods=len(values)), columns=names)


def test_common_volatility_and_scale_invariance_for_proportional_paths():
    base = np.array([-.02, .01, .03, -.01, .005])
    returns = returns_frame(np.column_stack([base, 2 * base, 3 * base]))
    result = normalized_metrics(returns, periods=12)
    np.testing.assert_allclose(result.scale, [1, .5, 1 / 3])
    np.testing.assert_allclose(result.normalized_volatility, base.std(ddof=1) * np.sqrt(12))
    for metric in ("mean", "cagr", "volatility", "max_drawdown"):
        np.testing.assert_allclose(result[f"normalized_{metric}"], result.loc["Rule", f"raw_{metric}"])
    assert (result.scale <= 1).all()


def test_metrics_use_actual_returns_and_include_initial_loss_in_drawdown():
    returns = returns_frame([[-.1, -.2, -.05], [.1, .2, .05]])
    result = normalized_metrics(returns, periods=2)
    assert result.loc["Rule", "raw_mean"] == pytest.approx(0)
    assert result.loc["Rule", "raw_cagr"] == pytest.approx(.9 * 1.1 - 1)
    assert result.loc["Rule", "raw_max_drawdown"] == pytest.approx(-.1)
    assert result.loc["Rule", "normalized_cagr"] == pytest.approx(.95 * 1.05 - 1)
    assert result.loc["Rule", "normalized_max_drawdown"] == pytest.approx(-.05)


@pytest.mark.parametrize("change, error", [
    (lambda r: r.assign(SPY=.001), "Zero-volatility"),
    (lambda r: r.assign(SPY=np.nan), "same dates"),
    (lambda r: r.assign(SPY=-1), "above -100%"),
    (lambda r: r.iloc[::-1], "dates"),
    (lambda r: r.set_axis([r.index[0]] * len(r)), "dates"),
    (lambda r: r.iloc[:1], "two synchronized"),
])
def test_invalid_or_unsynchronized_paths_fail_explicitly(change, error):
    returns = returns_frame(np.random.default_rng(3).normal(0, .01, (12, 3)))
    with pytest.raises(ValueError, match=error):
        normalized_metrics(change(returns))


def test_misaligned_series_cannot_silently_drop_dates():
    returns = returns_frame(np.random.default_rng(4).normal(0, .01, (12, 3)))
    misaligned = pd.concat([returns.Rule.iloc[1:], returns.SPY, returns.QQQ], axis=1, sort=True)
    with pytest.raises(ValueError, match="same dates"):
        paired_normalized_bootstrap(misaligned, repetitions=10)


def test_paired_bootstrap_preserves_proportional_paths_and_seeded_repeatability():
    base = np.random.default_rng(12).normal(.001, .01, 100)
    returns = returns_frame(np.column_stack([base, 2 * base, 3 * base]))
    options = dict(repetitions=130, block_lengths=(10, 21), seed=41)
    first = paired_normalized_bootstrap(returns, **options)
    second = paired_normalized_bootstrap(returns, **options)
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_allclose(first.estimate, 0, atol=1e-14)
    np.testing.assert_allclose(first.lower_bound, 0, atol=1e-14)
    assert set(first.block_length) == {10, 21}
    assert first.family_size.eq(2).all()
    assert first.confidence_kind.str.contains("not a future probability", regex=False).all()


def test_bootstrap_covers_every_candidate_and_benchmark_in_one_family():
    rng = np.random.default_rng(153)
    values = rng.normal(0, [.01, .00001, .02, .04], (120, 4)) + [.0005, .00001, .001, .003]
    returns = returns_frame(values, ("A", "SPY", "QQQ", "B"))
    options = dict(repetitions=256, block_lengths=(21,), seed=51)
    all_results = paired_normalized_bootstrap(returns, **options)
    subset = paired_normalized_bootstrap(returns.drop(columns="B"), **options)
    assert set(zip(all_results.candidate, all_results.benchmark)) == {
        ("A", "SPY"), ("A", "QQQ"), ("B", "SPY"), ("B", "QQQ"),
    }
    assert all_results.family_size.eq(4).all()
    assert all_results.simultaneous_critical.nunique() == 1
    assert all_results.simultaneous_critical.iloc[0] >= subset.simultaneous_critical.iloc[0]
    metrics = normalized_metrics(returns)
    for row in all_results.itertuples():
        expected = metrics.loc[row.candidate, "normalized_mean"] - metrics.loc[row.benchmark, "normalized_mean"]
        assert row.estimate == pytest.approx(expected)
        assert row.lower_bound == pytest.approx(expected - row.simultaneous_critical)


def test_bootstrap_block_order_does_not_change_a_blocks_result():
    returns = returns_frame(np.random.default_rng(7).normal(0, .01, (80, 3)))
    first = paired_normalized_bootstrap(returns, repetitions=128, block_lengths=(10, 21))
    second = paired_normalized_bootstrap(returns, repetitions=128, block_lengths=(21, 10))
    keys = ["block_length", "candidate", "benchmark"]
    pd.testing.assert_frame_equal(first.sort_values(keys).reset_index(drop=True),
                                  second.sort_values(keys).reset_index(drop=True))


def test_bootstrap_matches_sequential_blocks_and_per_sample_normalization():
    values = np.random.default_rng(18).normal([.002, .001, .003], [.02, .01, .015], (40, 3))
    returns = returns_frame(values)
    repetitions, seed, block = 12, 65, 10
    rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
    starts = rng.integers(len(values), size=(repetitions, len(values)))
    restart = rng.random((repetitions, len(values))) < 1 / block
    metrics = normalized_metrics(returns)
    observed = metrics.loc["Rule", "normalized_mean"] - metrics.loc[["SPY", "QQQ"], "normalized_mean"]
    errors = []
    for number in range(repetitions):
        indices = [starts[number, 0]]
        for position in range(1, len(values)):
            indices.append(starts[number, position] if restart[number, position]
                           else (indices[-1] + 1) % len(values))
        sample = normalized_metrics(returns_frame(values[indices]))
        gaps = sample.loc["Rule", "normalized_mean"] - sample.loc[["SPY", "QQQ"], "normalized_mean"]
        errors.append((gaps - observed).max())
    expected = max(0, np.quantile(errors, .95))
    result = paired_normalized_bootstrap(returns, repetitions=repetitions, seed=seed, block_lengths=(block,))
    np.testing.assert_allclose(result.simultaneous_critical, expected, atol=1e-14)


@pytest.mark.parametrize("options, error", [
    ({"benchmarks": ("Missing",)}, "benchmark names"),
    ({"benchmarks": ("Rule", "SPY", "QQQ")}, "At least one candidate"),
    ({"repetitions": 1}, "repetitions"),
    ({"block_lengths": (0,)}, "Block lengths"),
    ({"block_lengths": (10, 10)}, "Block lengths"),
    ({"seed": -1}, "seed"),
    ({"confidence": 1}, "confidence"),
])
def test_invalid_bootstrap_settings_fail(options, error):
    returns = returns_frame(np.random.default_rng(9).normal(0, .01, (80, 3)))
    with pytest.raises(ValueError, match=error):
        paired_normalized_bootstrap(returns, **options)
