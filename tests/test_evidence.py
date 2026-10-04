from math import erfc, sqrt

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.evidence import _holm_adjust, analyze_evidence


def equity(returns, name, initial=1.0):
    returns = np.asarray(returns, dtype=float)
    return pd.DataFrame(
        {name: initial * np.r_[1.0, np.cumprod(1 + returns)]},
        index=pd.bdate_range("2020-01-01", periods=len(returns) + 1),
    )


def sample(count=180):
    rng = np.random.default_rng(100)
    benchmark = rng.normal(0.0004, 0.008, count)
    residual = rng.normal(0, 0.002, count)
    for i in range(1, count):
        residual[i] += 0.65 * residual[i - 1]
    strategy = 0.0005 + 1.2 * benchmark + residual
    return equity(strategy, "Strategy"), equity(benchmark, "SPY")


def test_hac_matches_independent_dense_kernel_reference():
    strategy, benchmark = sample()
    result = analyze_evidence(strategy, benchmark, delayed_entry=False)
    row = result.summary.loc[("Strategy", "SPY")]
    y = strategy.pct_change(fill_method=None).iloc[1:, 0].to_numpy()
    x = benchmark.pct_change(fill_method=None).iloc[1:, 0].to_numpy()
    count = len(y)
    lags = int(np.floor(4 * (count / 100) ** (2 / 9)))
    distance = np.abs(np.subtract.outer(np.arange(count), np.arange(count)))
    kernel = np.maximum(1 - distance / (lags + 1), 0)
    active = y - x
    demeaned = active - active.mean()
    advantage_error = sqrt(float(demeaned @ kernel @ demeaned) / count ** 2 * count / (count - 1))
    design = np.column_stack([np.ones(count), x - 0.02 / 252])
    coefficients = np.linalg.lstsq(design, y - 0.02 / 252, rcond=None)[0]
    residual = y - 0.02 / 252 - design @ coefficients
    inverse = np.linalg.inv(design.T @ design)
    covariance = inverse @ design.T @ (np.outer(residual, residual) * kernel) @ design @ inverse
    covariance *= count / (count - 2)
    alpha_error = sqrt(covariance[0, 0])
    assert row.annual_advantage == pytest.approx(active.mean() * 252)
    assert row.beta == pytest.approx(coefficients[1])
    assert row.alpha == pytest.approx(coefficients[0] * 252)
    assert row.advantage_ci_low == pytest.approx((active.mean() - 1.959963984540054 * advantage_error) * 252)
    assert row.alpha_ci_high == pytest.approx((coefficients[0] + 1.959963984540054 * alpha_error) * 252)
    assert row.advantage_pvalue == pytest.approx(erfc(abs(active.mean() / advantage_error) / sqrt(2)))
    assert row.alpha_pvalue == pytest.approx(erfc(abs(coefficients[0] / alpha_error) / sqrt(2)))
    assert result.settings["hac_lags"] == lags
    assert row.tracking_error == pytest.approx(active.std(ddof=1) * sqrt(252))
    assert row.information_ratio == pytest.approx(active.mean() / active.std(ddof=1) * sqrt(252))
    assert advantage_error > active.std(ddof=1) / sqrt(count)


def test_holm_uses_the_full_family_and_retains_undefined_tests():
    np.testing.assert_allclose(_holm_adjust(np.array([0.01, 0.04, 0.03, np.nan])), [0.04, 0.09, 0.09, np.nan])
    np.testing.assert_allclose(_holm_adjust(np.array([0.02, 0.02, 0.8])), [0.06, 0.06, 0.8])


def test_all_strategies_and_benchmarks_share_one_fixed_test_family():
    strategy, benchmark = sample()
    strategy["Clone"] = benchmark.SPY
    benchmark["QQQ"] = benchmark.SPY
    result = analyze_evidence(strategy, benchmark, delayed_entry=False)
    assert result.settings["multiple_testing_tests"] == 8
    original = result.summary[["advantage_pvalue", "alpha_pvalue"]].to_numpy().reshape(-1)
    pvalues = [float(value) if np.isfinite(value) else 1.0 for value in original]
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    expected, maximum = [None] * len(order), 0
    for rank, position in enumerate(order):
        maximum = max(maximum, min(1, pvalues[position] * (len(order) - rank)))
        expected[position] = maximum if np.isfinite(original[position]) else np.nan
    actual = result.summary[["advantage_pvalue_adjusted", "alpha_pvalue_adjusted"]].to_numpy().reshape(-1)
    np.testing.assert_allclose(actual, expected)
    reordered = analyze_evidence(strategy.iloc[:, ::-1], benchmark.iloc[:, ::-1], delayed_entry=False)
    pd.testing.assert_frame_equal(result.summary.sort_index(), reordered.summary.sort_index())


def test_entry_exclusion_changes_inference_but_keeps_costs_in_full_path_results():
    strategy, benchmark = sample()
    strategy_returns = np.r_[-0.2, strategy.pct_change(fill_method=None).iloc[1:, 0]]
    benchmark_returns = np.r_[-0.01, benchmark.pct_change(fill_method=None).iloc[1:, 0]]
    strategy = equity(strategy_returns, "Strategy")
    benchmark = equity(benchmark_returns, "SPY")
    delayed = analyze_evidence(strategy, benchmark)
    immediate = analyze_evidence(strategy, benchmark, delayed_entry=False)
    row = delayed.summary.iloc[0]
    assert row.observations == 180
    assert immediate.summary.iloc[0].observations == 181
    assert row.annual_advantage == pytest.approx((strategy_returns[1:] - benchmark_returns[1:]).mean() * 252)
    assert row.annual_advantage > immediate.summary.iloc[0].annual_advantage
    assert row.total_return_difference == pytest.approx(np.prod(1 + strategy_returns) - np.prod(1 + benchmark_returns))
    assert row.cagr_difference == immediate.summary.iloc[0].cagr_difference
    pd.testing.assert_frame_equal(delayed.windows, immediate.windows)
    pd.testing.assert_frame_equal(delayed.relative_equity, immediate.relative_equity)
    assert delayed.settings["inference_start"] == strategy.index[2].date().isoformat()


def test_window_dates_cover_all_returns_once_and_relative_returns_compound():
    strategy, benchmark = sample(19)
    result = analyze_evidence(strategy, benchmark, delayed_entry=False, periods_per_year=12)
    assert list(result.windows.observations) == [7, 6, 6]
    sreturns = strategy.pct_change(fill_method=None).iloc[1:, 0]
    breturns = benchmark.pct_change(fill_method=None).iloc[1:, 0]
    for row, positions in zip(result.windows.itertuples(), np.array_split(np.arange(19), 3)):
        expected = np.prod(1 + sreturns.iloc[positions]) / np.prod(1 + breturns.iloc[positions]) - 1
        assert row.relative_return == pytest.approx(expected)
        assert row.annual_advantage == pytest.approx((sreturns.iloc[positions] - breturns.iloc[positions]).mean() * 12)
        assert row.start == sreturns.index[positions[0]].date().isoformat()
        assert row.end == sreturns.index[positions[-1]].date().isoformat()
    assert result.summary.iloc[0].windows_ahead == (result.windows.relative_return > 0).sum()
    assert np.prod(1 + result.windows.relative_return) == pytest.approx(result.relative_equity.iloc[-1, 0])


def test_initial_capital_and_labels_are_preserved_without_input_changes():
    strategy, benchmark = sample()
    original_strategy, original_benchmark = strategy.copy(), benchmark.copy()
    reference = analyze_evidence(strategy, benchmark)
    scaled = analyze_evidence(strategy * 100, benchmark * 500)
    pd.testing.assert_frame_equal(strategy, original_strategy)
    pd.testing.assert_frame_equal(benchmark, original_benchmark)
    pd.testing.assert_frame_equal(reference.summary, scaled.summary, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(reference.relative_equity, scaled.relative_equity, atol=1e-12, rtol=1e-12)
    assert scaled.relative_equity.iloc[0, 0] == 1
    assert scaled.summary.index.names == ["strategy", "benchmark"]
    assert scaled.relative_equity.columns.names == ["strategy", "benchmark"]


@pytest.mark.parametrize("count", [59, 60])
def test_inference_requires_sixty_paired_observations(count):
    result = analyze_evidence(*sample(count), delayed_entry=False)
    row = result.summary.iloc[0]
    assert row.observations == count
    assert np.isfinite(row.beta)
    assert np.isfinite(row.alpha)
    assert np.isfinite(row.advantage_pvalue) == (count >= 60)
    assert np.isfinite(row.alpha_pvalue) == (count >= 60)
    if count < 60:
        assert row.status == "Insufficient history"
        assert "60 paired observations" in row.advantage_note


def test_one_return_keeps_descriptive_results_with_no_post_entry_inference():
    result = analyze_evidence(equity([0.01], "Strategy"), equity([0.02], "SPY"))
    row = result.summary.iloc[0]
    assert row.observations == 0
    assert row.total_return_difference == pytest.approx(-0.01)
    assert np.isnan(row.annual_advantage)
    assert np.isnan(row.alpha_pvalue)
    assert len(result.windows) == 1
    assert result.settings["inference_start"] is None


def test_identical_paths_do_not_create_certainty_from_zero_residuals():
    _, benchmark = sample()
    result = analyze_evidence(benchmark.rename(columns={"SPY": "Strategy"}), benchmark)
    row = result.summary.iloc[0]
    assert row.total_return_difference == 0
    assert row.beta == pytest.approx(1)
    assert row.alpha == pytest.approx(0, abs=1e-12)
    assert row.tracking_error == 0
    assert np.isnan(row.information_ratio)
    assert np.isnan(row.advantage_pvalue)
    assert np.isnan(row.alpha_pvalue)
    assert row.status == "Uncertainty unavailable"
    np.testing.assert_allclose(result.relative_equity, 1)


def test_constant_benchmark_does_not_create_identifiable_alpha():
    strategy, _ = sample()
    result = analyze_evidence(strategy, equity(np.zeros(180), "Cash"), delayed_entry=False)
    row = result.summary.iloc[0]
    assert np.isnan(row.beta)
    assert np.isnan(row.alpha)
    assert np.isnan(row.alpha_pvalue)
    assert np.isfinite(row.advantage_pvalue)
    assert "Benchmark variation" in row.alpha_note
    assert result.settings["multiple_testing_tests"] == 2


@pytest.mark.parametrize("advantage, status", [(0.002, "Historical mean advantage and alpha"), (-0.002, "Historical mean underperformance")])
def test_strong_historical_signal_gets_a_directional_status(advantage, status):
    rng = np.random.default_rng(42)
    benchmark = rng.normal(0.0004, 0.008, 180)
    strategy = benchmark + advantage + rng.normal(0, 0.0003, 180)
    result = analyze_evidence(equity(strategy, "Strategy"), equity(benchmark, "SPY"), delayed_entry=False)
    assert result.summary.iloc[0].status == status
    assert result.summary.iloc[0].advantage_pvalue_adjusted < 0.05
    assert any("future outperformance" in warning for warning in result.warnings)


def test_higher_market_exposure_is_not_reported_as_significant_alpha():
    rng = np.random.default_rng(42)
    benchmark = rng.normal(0.003, 0.003, 180)
    result = analyze_evidence(equity(2 * benchmark, "Strategy"), equity(benchmark, "SPY"), risk_free_rate=0, delayed_entry=False)
    row = result.summary.iloc[0]
    assert row.beta == pytest.approx(2)
    assert row.alpha == pytest.approx(0, abs=1e-12)
    assert np.isnan(row.alpha_pvalue)
    assert row.status == "Historical mean advantage"


def test_annualization_scales_estimates_and_intervals_but_not_inference():
    strategy, benchmark = sample()
    monthly = analyze_evidence(strategy, benchmark, periods_per_year=12, risk_free_rate=0)
    daily = analyze_evidence(strategy, benchmark, periods_per_year=252, risk_free_rate=0)
    for column in ("annual_advantage", "alpha", "advantage_ci_low", "alpha_ci_high"):
        np.testing.assert_allclose(daily.summary[column], monthly.summary[column] * 21)
    np.testing.assert_allclose(daily.summary.tracking_error, monthly.summary.tracking_error * sqrt(21))
    np.testing.assert_allclose(daily.summary.information_ratio, monthly.summary.information_ratio * sqrt(21))
    for column in ("beta", "advantage_pvalue_adjusted", "alpha_pvalue_adjusted"):
        np.testing.assert_allclose(daily.summary[column], monthly.summary[column])


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_equity_values_are_rejected(value):
    strategy, benchmark = sample()
    strategy.iloc[3, 0] = value
    with pytest.raises(ValueError, match="finite and strictly positive"):
        analyze_evidence(strategy, benchmark)


@pytest.mark.parametrize("value", [True, np.bool_(False), 1 + 0j, np.complex128(1 + 2j)])
def test_object_equity_rejects_boolean_and_complex_scalars(value):
    strategy, benchmark = sample()
    strategy = strategy.astype(object)
    strategy.iloc[2, 0] = value
    with pytest.raises(ValueError, match="Boolean or complex"):
        analyze_evidence(strategy, benchmark)


def test_mismatched_dates_are_rejected_without_silent_alignment():
    strategy, benchmark = sample()
    with pytest.raises(ValueError, match="exactly the same dates"):
        analyze_evidence(strategy.iloc[1:], benchmark)


def test_unsorted_dates_are_rejected():
    strategy, benchmark = sample()
    with pytest.raises(ValueError, match="increasing"):
        analyze_evidence(strategy.iloc[::-1], benchmark.iloc[::-1])


def test_duplicate_names_are_rejected():
    strategy, benchmark = sample()
    with pytest.raises(ValueError, match="unique column names"):
        analyze_evidence(pd.concat([strategy, strategy], axis=1), benchmark)


@pytest.mark.parametrize("settings, message", [
    ({"periods_per_year": 0}, "periods_per_year"),
    ({"risk_free_rate": np.inf}, "risk_free_rate"),
    ({"risk_free_rate": True}, "risk_free_rate"),
    ({"delayed_entry": 1}, "delayed_entry"),
])
def test_invalid_settings_are_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        analyze_evidence(*sample(), **settings)
