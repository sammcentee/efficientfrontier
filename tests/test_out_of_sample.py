from dataclasses import replace
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from efficient_frontier import out_of_sample
from efficient_frontier.out_of_sample import portfolio_comparison, run_out_of_sample, sample_values, window_bounds
from efficient_frontier.selection import BASELINE, BENCHMARK_NAMES, SELECTED, _fit_targets, run_selection
from efficient_frontier.selection_report import (
    out_of_sample_chart, out_of_sample_findings, out_of_sample_frame, out_of_sample_html, out_of_sample_paths_frame,
    out_of_sample_text, selection_report_html, selection_report_zip,
)


@pytest.fixture
def market():
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2021-01-04", periods=320, name="Date")
    returns = rng.normal(.0005, .01, (len(dates), 12)) + rng.normal(0, .005, (len(dates), 1))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                          columns=[f"S{number:02d}" for number in range(12)])
    benchmarks = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal([.0004, .0006], [.009, .013], (len(dates), 2)), axis=0)),
                              index=dates, columns=["SPY", "QQQ"])
    return prices, benchmarks


@pytest.fixture
def study(market):
    prices, benchmarks = market
    return run_selection(prices, benchmark_prices=benchmarks, target_volatility=.2, max_weight=.15, cost_bps=25)


def hold(prices, weights, start, end, periods):
    """Independent fixed units bought at close start+1 and held to close end, measured from the purchase close."""
    units = weights / prices.iloc[start + 1]
    values = pd.Series([float((units * prices.iloc[t]).sum()) for t in range(start + 1, end + 1)])
    returns = values.pct_change().iloc[1:]
    growth = values.iloc[-1] / values.iloc[0]
    return {"annual_mean": returns.mean() * periods, "volatility": returns.std(ddof=1) * np.sqrt(periods),
            "total_return": growth - 1, "cagr": growth ** (periods / len(returns)) - 1,
            "max_drawdown": (values / values.cummax() - 1).min()}


@pytest.mark.parametrize("count", [29, 47, 251, 1255])
def test_windows_are_consecutive_do_not_overlap_and_cover_the_final_half(count):
    bounds = window_bounds(count)
    assert len(bounds) == 5
    assert bounds[0][0] == count // 2 and bounds[-1][1] == count
    for (_, end), (split, _) in zip(bounds, bounds[1:]):
        assert split == end
    assert all(end - split >= 3 for split, end in bounds)


@pytest.mark.parametrize("count", [3, 9, 14, 28])
def test_too_short_history_raises_an_explicit_reason(count):
    with pytest.raises(ValueError, match="at least 3 fit returns and 3 test returns"):
        window_bounds(count)


def test_each_window_fits_only_earlier_prices(study, market):
    prices, _ = market
    result = run_out_of_sample(study, prices)
    for number, (split, _) in enumerate(window_bounds(len(prices) - 1), 1):
        direct = _fit_targets(prices.iloc[:split + 1], .2, .15, 0.0, 252)[0]
        pd.testing.assert_series_equal(result.weights.loc[number], direct, check_names=False)
        assert result.windows.loc[number, "fit_end"] == prices.index[split]
        assert result.windows.loc[number, "test_start"] == prices.index[split + 1]


def test_later_prices_cannot_change_an_earlier_window(study, market):
    prices, benchmarks = market
    result = run_out_of_sample(study, prices)
    split, end = window_bounds(len(prices) - 1)[2]
    changed, changed_benchmarks = prices.copy(), benchmarks.copy()
    rng = np.random.default_rng(7)
    changed.iloc[end + 1:] *= rng.uniform(.5, 1.5, changed.iloc[end + 1:].shape)
    changed_benchmarks.iloc[end + 1:] *= 3
    # Reuse the saved settings; only the prices differ.
    altered = run_out_of_sample(replace(study, benchmark_prices=changed_benchmarks), changed)
    for number in (1, 2, 3):
        pd.testing.assert_frame_equal(altered.metrics.loc[number], result.metrics.loc[number])
        pd.testing.assert_series_equal(altered.weights.loc[number], result.weights.loc[number])
    # Prices after the fit, but inside the test window, change only the out-of-sample side.
    inside = prices.copy()
    inside.iloc[split + 1:end + 1] *= 1.1 ** np.arange(1, end - split + 1)[:, None] / 1.1
    moved = run_out_of_sample(study, inside)
    pd.testing.assert_series_equal(moved.weights.loc[3], result.weights.loc[3])
    pd.testing.assert_frame_equal(moved.metrics.loc[(3, "in_sample")], result.metrics.loc[(3, "in_sample")])
    assert not np.allclose(moved.metrics.loc[(3, "out_of_sample"), "annual_mean"],
                           result.metrics.loc[(3, "out_of_sample"), "annual_mean"])


def test_both_sides_match_independent_buy_and_hold_from_the_purchase_close(study, market):
    prices, benchmarks = market
    result = run_out_of_sample(study, prices)
    periods = 252
    for number, (split, end) in enumerate(window_bounds(len(prices) - 1), 1):
        weights = result.weights.loc[number]
        for sample, (start, stop) in (("in_sample", (0, split)), ("out_of_sample", (split, end))):
            expected = hold(prices, weights, start, stop, periods)
            actual = result.metrics.loc[(number, sample, SELECTED)]
            for key, value in expected.items():
                assert actual[key] == pytest.approx(value, abs=1e-12), (number, sample, key)
            equal = pd.Series(1 / len(prices.columns), index=prices.columns)
            baseline = hold(prices, equal, start, stop, periods)
            assert result.metrics.loc[(number, sample, BASELINE), "annual_mean"] == pytest.approx(
                baseline["annual_mean"], abs=1e-12)
            for symbol, name in BENCHMARK_NAMES.items():
                reference = hold(benchmarks[[symbol]], pd.Series([1.0], index=[symbol]), start, stop, periods)
                assert result.metrics.loc[(number, sample, name), "annual_mean"] == pytest.approx(
                    reference["annual_mean"], abs=1e-12)


def test_fee_and_cash_interval_do_not_bias_the_short_window(market):
    """With identical drift on both sides, a high fee must not open a gap between in and out of sample."""
    prices, _ = market
    dates = prices.index
    steady = pd.DataFrame(100 * 1.001 ** np.arange(len(dates))[:, None] * np.ones((1, 3)) * [1, 1.0001, .9999] ** np.arange(len(dates))[:, None],
                          index=dates, columns=["A", "B", "C"])
    base = run_selection(steady, target_volatility=.5, max_weight=.5, cost_bps=500)
    result = run_out_of_sample(base, steady)
    means = portfolio_comparison(result)
    np.testing.assert_allclose(means.in_sample, means.out_of_sample, rtol=1e-3)


def test_comparison_frame_and_findings_use_completed_windows(study, market):
    prices, _ = market
    result = run_out_of_sample(study, prices)
    means = portfolio_comparison(result)
    assert list(means.columns[:3]) == ["in_sample", "out_of_sample", "change"]
    np.testing.assert_allclose(means.change, means.out_of_sample - means.in_sample)
    findings = out_of_sample_findings(result)
    assert findings["completed"] == 5
    assert findings["in_sample"] == pytest.approx(means.in_sample.mean())
    assert findings["out_of_sample"] == pytest.approx(means.out_of_sample.mean())
    assert findings["lower"] == int((means.change < 0).sum())
    equal_in = sample_values(result, "in_sample")[BASELINE]
    assert findings["margin_in_sample"] == pytest.approx((means.in_sample - equal_in).mean())
    assert findings["margin_out_of_sample"] == pytest.approx((means.out_of_sample - means[BASELINE]).mean())
    assert set(findings["higher_than"]) == {BASELINE, *BENCHMARK_NAMES.values()}
    headline, lines = out_of_sample_text(findings)
    for value in (means.out_of_sample.mean(), means.in_sample.mean()):
        assert f"{value:.1%}".replace("-", "−") in headline
    assert f"in {findings['lower']} of 5 windows" in lines[0]
    assert "average margin over them" in lines[1]
    assert ", and Nasdaq-100 (QQQ) in" in lines[2]
    frame = out_of_sample_frame(result)
    assert list(frame.index) == [1, 2, 3, 4, 5]
    assert list(frame.columns) == ["Test dates", "In sample", "Out of sample", "Change",
                                   "Equal weight in sample", "Equal weight out of sample"]
    np.testing.assert_allclose(frame["Equal weight in sample"], equal_in)
    paths = out_of_sample_paths_frame(result)
    assert len(paths) == 5 * 4
    assert list(paths.loc[1].index) == ["Stock portfolio", "Equal-weight stocks", *BENCHMARK_NAMES.values()]
    np.testing.assert_allclose(paths.xs("Stock portfolio", level="Path")["Change"], means.change)


def test_chart_hover_names_the_dates_each_marker_measures(study, market):
    prices, _ = market
    result = run_out_of_sample(study, prices)
    traces = {trace.name: trace for trace in out_of_sample_chart(result).data if trace.name}
    inside, outside = traces["Stock portfolio · in sample"], traces["Stock portfolio · out of sample"]
    assert "fit dates" in inside.hovertemplate and "test dates" in outside.hovertemplate
    first = result.windows.loc[1]
    assert inside.customdata[0][1] == f"{first.fit_start.date()} to {first.fit_end.date()}"
    assert outside.customdata[0][1] == f"{first.test_start.date()} to {first.test_end.date()}"


@pytest.mark.parametrize("references,expected", [
    ({BASELINE: 2}, "than equal-weight stocks in 2 of 5."),
    ({BASELINE: 2, BENCHMARK_NAMES["QQQ"]: 1}, "than equal-weight stocks in 2 of 5 and Nasdaq-100 (QQQ) in 1 of 5."),
])
def test_reference_sentence_lists_one_or_two_paths_without_a_serial_comma(references, expected):
    findings = {"completed": 5, "in_sample": .1, "out_of_sample": -.02, "change": -.12, "lower": 4,
                "margin_in_sample": .03, "margin_out_of_sample": -.01, "higher_than": references, "test_returns": 125}
    headline, lines = out_of_sample_text(findings)
    assert "−2.0%" in headline
    assert lines[-1].endswith(expected)
    assert "+3.00 pp a year in sample and −1.00 pp out of sample" in lines[1]


def test_infeasible_window_keeps_its_dates_and_reason(monkeypatch, study, market):
    prices, _ = market
    original, calls = out_of_sample._fit_targets, []

    def first_fails(*args, **kwargs):
        calls.append(len(args[0]))
        if len(calls) == 1:
            raise ValueError("requested annual volatility 1.00% is below the feasible minimum")
        return original(*args, **kwargs)

    monkeypatch.setattr(out_of_sample, "_fit_targets", first_fails)
    result = run_out_of_sample(study, prices)
    assert "below the feasible minimum" in result.windows.loc[1, "status"]
    assert (result.windows.loc[2:, "status"] == "").all()
    assert 1 not in result.metrics.index.get_level_values("window")
    assert list(portfolio_comparison(result).index) == [2, 3, 4, 5]
    assert out_of_sample_findings(result)["completed"] == 4
    assert "Window 1 has no result. requested annual volatility" in out_of_sample_html(result)


def test_all_windows_infeasible_reports_no_result(monkeypatch, study, market):
    prices, _ = market

    def always_fails(*args, **kwargs):
        raise ValueError("The 15.0% position cap is infeasible")

    monkeypatch.setattr(out_of_sample, "_fit_targets", always_fails)
    result = run_out_of_sample(study, prices)
    assert result.metrics.empty and result.weights.empty
    assert out_of_sample_findings(result) is None
    assert portfolio_comparison(result).empty
    assert "No window had a feasible fit" in out_of_sample_html(result)


def test_export_with_no_feasible_window_keeps_dates_and_reasons(monkeypatch, study, market):
    prices, _ = market

    def always_fails(*args, **kwargs):
        raise ValueError("The 15.0% position cap is infeasible")

    monkeypatch.setattr(out_of_sample, "_fit_targets", always_fails)
    result = run_out_of_sample(study, prices)
    archive_bytes = selection_report_zip(study, prices, {"source": "Synthetic"}, out_of_sample=result)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = [name for name in archive.namelist() if name.startswith("out_of_sample/")]
        assert names == ["out_of_sample/windows.csv"]
        windows = pd.read_csv(archive.open("out_of_sample/windows.csv"), index_col=0)
        assert windows.status.str.contains("position cap is infeasible").all()
        assert "the reason it has no result" in archive.read("report.html").decode()


def test_walk_forward_fit_notes_and_eligibility_are_kept(study, market):
    prices, _ = market
    stale = prices.copy()
    stale.iloc[:40, 0] = stale.iloc[0, 0]
    result = run_out_of_sample(study, stale)
    assert result.notes and all(note.startswith("Window ") for note in result.notes)
    assert any("unchanged-price screen excluded S00" in note for note in result.notes)
    excluded = result.eligibility.loc[result.eligibility.symbol == "S00"]
    assert sorted(excluded.window) == [1, 2, 3, 4, 5] and not excluded.eligible.any()
    assert (result.weights["S00"] == 0).all()
    assert "unchanged-price screen excluded S00" in out_of_sample_html(result)


def test_missing_benchmarks_keep_stock_and_equal_weight_results(market):
    prices, _ = market
    plain = run_selection(prices, target_volatility=.2, max_weight=.15)
    result = run_out_of_sample(plain, prices)
    assert set(result.metrics.index.get_level_values("strategy")) == {SELECTED, BASELINE}
    assert set(out_of_sample_findings(result)["higher_than"]) == {BASELINE}


def test_report_and_zip_include_walk_forward_results(study, market):
    prices, _ = market
    metadata = {"source": "Synthetic adjusted prices", "universe": "Fixture stocks"}
    result = run_out_of_sample(study, prices)
    archive_bytes = selection_report_zip(study, prices, metadata)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        assert {"out_of_sample/windows.csv", "out_of_sample/metrics.csv", "out_of_sample/weights.csv",
                "out_of_sample/eligibility.csv"} <= names
        eligibility = pd.read_csv(archive.open("out_of_sample/eligibility.csv"))
        assert sorted(eligibility.window.unique()) == [1, 2, 3, 4, 5]
        metrics = pd.read_csv(archive.open("out_of_sample/metrics.csv"), index_col=[0, 1, 2])
        pd.testing.assert_frame_equal(metrics, result.metrics, check_exact=False, rtol=1e-12)
        details = json.loads(archive.read("metadata.json"))
        assert details["out_of_sample"]["windows"] == 5 and details["out_of_sample"]["notes"] == result.notes
        document = archive.read("report.html").decode()
    section = document.split('id="out-of-sample"', 1)[1].split("</section>", 1)[0]
    headline, _ = out_of_sample_text(out_of_sample_findings(result))
    assert headline in section
    assert document.index('id="out-of-sample"') < document.index('id="test"')


def test_report_explains_a_history_too_short_for_walk_forward(study, market):
    prices, benchmarks = market
    short, short_benchmarks = prices.iloc[:16], benchmarks.iloc[:16]
    short_study = run_selection(short, benchmark_prices=short_benchmarks, target_volatility=.5, max_weight=.15)
    document = selection_report_html(short_study, short, {"source": "Synthetic", "universe": "Fixture"})
    assert "at least 3 fit returns and 3 test returns" in document
    assert "There are no out_of_sample files" in document
    with zipfile.ZipFile(io.BytesIO(selection_report_zip(short_study, short, {"source": "Synthetic"}))) as archive:
        assert not any(name.startswith("out_of_sample/") for name in archive.namelist())
        assert "unavailable" in json.loads(archive.read("metadata.json"))["out_of_sample"]
