"""Golden report outputs of the Python release for the Rust parity tests.

Run from a checkout of the Python release (commit 6c6df6e) with its
requirements installed, after generate_selection.py:

    python tests/fixtures/golden/generate_reports.py

Each case runs the stock model on golden input prices and writes
reports_<case>.zip beside this script. The archive holds:

- archive/<name>: every file of selection_report_zip. report.html keeps its
  text, but "[plotly.js]" replaces the bundled plotly.js source.
- names.json: the member names of the report ZIP, in order.
- snippets.json: app-facing HTML and text, and small tables in pandas split form.
- charts.json: app-facing figures from fig.to_plotly_json(), with plain lists,
  dates as YYYY-MM-DD and no template.

The walk case uses the market fixture of tests/test_out_of_sample.py and writes
it as reports_walk_prices.csv and reports_walk_benchmarks.csv.

The legacy cases write reports_legacy_<case>.zip from presentation.report_zip on
the legacy market fixture of generate_legacy.py and on the demo prices. The
historical case writes reports_historical_demo.zip from the historical demo.
The same layout applies; the story and benchmark snippets join snippets.json.
"""

import base64
import datetime
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from plotly.offline import get_plotlyjs  # noqa: E402

from efficient_frontier import out_of_sample as walk_forward  # noqa: E402
from efficient_frontier import selection_report as report  # noqa: E402
from efficient_frontier import presentation  # noqa: E402
from efficient_frontier import story  # noqa: E402
from efficient_frontier.backtest import run_backtests  # noqa: E402
from efficient_frontier.backtest_report import backtest_chart, backtest_html, findings_markdown  # noqa: E402
from efficient_frontier.benchmark_report import benchmark_html, benchmark_markdown, evidence_chart  # noqa: E402
from efficient_frontier.benchmarks import compare_benchmarks  # noqa: E402
from efficient_frontier.core import analyze  # noqa: E402
from efficient_frontier.data import demo_prices  # noqa: E402
from efficient_frontier.evidence import analyze_evidence  # noqa: E402
from efficient_frontier.historical import historical_demo, run_historical_backtests  # noqa: E402
from efficient_frontier.historical_report import build_historical_report, historical_report_zip  # noqa: E402
from efficient_frontier.profiles import build_profiles  # noqa: E402
from efficient_frontier.data import validate_prices  # noqa: E402
from efficient_frontier.out_of_sample import run_out_of_sample  # noqa: E402
from efficient_frontier.selection import run_selection, selection_frontier  # noqa: E402

HERE = Path(__file__).resolve().parent
DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})T00:00:00(\.0+)?$")

COVERAGE = [
    {"symbol": "S00", "name": "Alpha Inc. Common Stock", "industry": "Technology", "status": "included", "reason": "",
     "observations": 701, "first_date": "2019-01-01", "last_date": "2021-09-08"},
    {"symbol": "S01", "name": "Beta & Sons, Inc. Class A", "industry": "Retail", "status": "included", "reason": "",
     "observations": 701, "first_date": "2019-01-01", "last_date": "2021-09-08"},
    {"symbol": "OLD", "name": "=Old Holdings PLC", "industry": "Utilities <old>", "status": "excluded",
     "reason": "Yahoo returned no prices for: OLD.", "observations": None, "first_date": None, "last_date": None},
]

CASES = {
    "demo": {
        "prices": "selection_demo_prices.csv", "benchmarks": None,
        "options": {"target_volatility": .2, "max_weight": .2},
        "metadata": {"source": "Demo · synthetic", "universe": "Demo", "price_currency": "USD", "compare_market": False,
                     "benchmarks_note": "Synthetic prices have no real-market benchmark comparison.",
                     "selection_model_revision": "stock_mean_buy_hold_v3"},
    },
    "market": {
        "prices": "selection_market_prices.csv", "benchmarks": "selection_market_benchmarks.csv",
        "options": {"target_volatility": .18, "max_weight": .1},
        "metadata": {"source": "Yahoo Finance", "universe": "Nasdaq-100", "price_currency": "USD", "compare_market": True,
                     "requested_start": "2019-01-01", "requested_end_exclusive": "2021-09-09",
                     "universe_sources": ["https://example.invalid/nasdaq-100", "<b>backup list</b>"],
                     "universe_requested": 42, "universe_included": 40, "universe_excluded": 2,
                     "universe_limitation": "Current members only; former members are missing.",
                     "benchmark_source": "Yahoo Finance", "notes": {"weights": [1, 2.5, None, True], "ok": "é"},
                     "universe_coverage": COVERAGE, "selection_model_revision": "stock_mean_buy_hold_v3"},
    },
    "stale": {
        "prices": "selection_stale_prices.csv", "benchmarks": "selection_stale_benchmarks.csv",
        "options": {"target_volatility": .25, "max_weight": .15, "shrinkage": .2, "cost_bps": 25.0},
        "metadata": {"source": "Upload CSV", "universe": "Upload CSV", "input_file": "prices <stale>.csv",
                     "price_currency": "USD", "benchmark_source": "Input columns: SPY, QQQ",
                     "selection_model_revision": "stock_mean_buy_hold_v3"},
    },
    "walk": {
        "prices": "reports_walk_prices.csv", "benchmarks": "reports_walk_benchmarks.csv",
        "options": {"target_volatility": .2, "max_weight": .15, "cost_bps": 25.0},
        "metadata": {"source": "Synthetic adjusted prices", "universe": "Fixture stocks"},
        "first_window_fails": "requested annual volatility 1.00% is below the feasible minimum",
    },
    "short": {
        "prices": "reports_walk_prices.csv", "benchmarks": "reports_walk_benchmarks.csv", "rows": 16,
        "options": {"target_volatility": .5, "max_weight": .15},
        "metadata": {"source": "Synthetic", "universe": "Fixture"},
    },
}


def walk_market():
    """The market fixture of tests/test_out_of_sample.py."""
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2021-01-04", periods=320, name="Date")
    returns = rng.normal(.0005, .01, (len(dates), 12)) + rng.normal(0, .005, (len(dates), 1))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                          columns=[f"S{number:02d}" for number in range(12)])
    benchmarks = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal([.0004, .0006], [.009, .013], (len(dates), 2)), axis=0)),
                              index=dates, columns=["SPY", "QQQ"])
    return prices, benchmarks


def value(item):
    """A JSON value: plain lists, dates as YYYY-MM-DD, NaN as null."""
    if isinstance(item, dict) and set(item) in ({"dtype", "bdata"}, {"dtype", "bdata", "shape"}):
        array = np.frombuffer(base64.b64decode(item["bdata"]), dtype=np.dtype(item["dtype"]))
        if "shape" in item:
            array = array.reshape([int(size) for size in str(item["shape"]).split(",")])
        return value(array)
    if isinstance(item, dict):
        return {str(key): value(entry) for key, entry in item.items()}
    if isinstance(item, (list, tuple)):
        return [value(entry) for entry in item]
    if isinstance(item, np.ndarray):
        return [value(entry) for entry in item.tolist()]
    if isinstance(item, (pd.Timestamp, np.datetime64, datetime.datetime)):
        stamp = pd.Timestamp(item)
        return None if pd.isna(stamp) else stamp.date().isoformat()
    if item is None or item is pd.NaT:
        return None
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, np.integer):
        return int(item)
    if isinstance(item, (float, np.floating)):
        return None if math.isnan(item) else float(item)
    if isinstance(item, str):
        match = DATE.match(item)
        return match.group(1) if match else item
    return item


def split(frame):
    index = frame.index
    labels = [value(list(row)) for row in index] if isinstance(index, pd.MultiIndex) else [value(x) for x in index]
    return {
        "index_names": [value(name) for name in index.names],
        "index": labels,
        "columns": [str(column) for column in frame.columns],
        "data": [[value(x) for x in row] for row in frame.itertuples(index=False, name=None)],
    }


def figure(fig):
    data = fig.to_plotly_json()
    data["layout"].pop("template", None)
    return value(data)


def load(name):
    """Golden prices with correctly rounded numbers, as the Rust CSV reader reads them.

    The release's load_csv parses numbers with pandas.to_numeric, which can be one
    unit in the last place away from the nearest float. Both versions must use
    the same inputs here.
    """
    return validate_prices(pd.read_csv(HERE / name, index_col=0, dtype=str).apply(
        lambda column: column.map(float)))


def prices_for(case):
    prices = load(case["prices"])
    benchmarks = load(case["benchmarks"]) if case["benchmarks"] else None
    rows = case.get("rows")
    if rows:
        prices = prices.iloc[:rows]
        benchmarks = benchmarks.iloc[:rows] if benchmarks is not None else None
    return prices, benchmarks


def walk_result(study, prices, failure):
    """Run the walk-forward test. With a failure message, the first window fit fails."""
    if failure is None:
        try:
            return run_out_of_sample(study, prices), None
        except ValueError as exc:
            return None, str(exc)
    original, calls = walk_forward._fit_targets, []

    def first_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError(failure)
        return original(*args, **kwargs)

    walk_forward._fit_targets = first_fails
    try:
        return run_out_of_sample(study, prices), None
    finally:
        walk_forward._fit_targets = original


def write_golden(path, archive_bytes, snippets, charts):
    """Write the archive members with plotly.js replaced, and the snippets and charts."""
    plotly = get_plotlyjs()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source, \
            zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as golden:
        names = source.namelist()
        for member in names:
            text = source.read(member).decode("utf-8")
            if member == "report.html":
                assert text.count(plotly) == 1
                text = text.replace(plotly, "[plotly.js]")
            golden.writestr(f"archive/{member}", text)
        golden.writestr("names.json", json.dumps(names, indent=1))
        golden.writestr("snippets.json", json.dumps(snippets, indent=1, ensure_ascii=False))
        golden.writestr("charts.json", json.dumps(charts, indent=1, ensure_ascii=False))
    path.write_bytes(output.getvalue())


LEGACY_SETTINGS = {"train_fraction": 0.6, "risk_free_rate": 0.03, "max_weight": 0.4}


def legacy_full():
    """The legacy CLI with --backtests and benchmark prices: every optional part."""
    market, market_benchmarks = load("legacy_market_prices.csv"), load("legacy_market_benchmark_prices.csv")
    analysis = analyze(market, **LEGACY_SETTINGS)
    study = run_backtests(market, **LEGACY_SETTINGS, rebalance_every=20, cost_bps=15.0, include_profiles=True)
    latest = build_profiles(market, risk_free_rate=0.03, max_weight=0.4)
    benchmarks = compare_benchmarks(market_benchmarks, market, analysis, study, latest, risk_free_rate=0.03)
    evidence = analyze_evidence(study.equity, benchmarks.backtest_equity, risk_free_rate=0.03, periods_per_year=252.0,
                                delayed_entry=True)
    metadata = {"source": "CSV: legacy_market_prices.csv", "price_currency": "USD",
                "currency_assumption": "The user declares all asset prices in USD. No currency conversion occurs.",
                **LEGACY_SETTINGS, "benchmark_source": "CSV: legacy <benchmarks>.csv", "notes": ["é", 1]}
    return market, analysis, study, latest, benchmarks, evidence, metadata


def legacy_plain():
    """A holdout-only analysis with benchmarks and holdout evidence, without backtests or profiles."""
    market, market_benchmarks = load("legacy_market_prices.csv"), load("legacy_market_benchmark_prices.csv")
    analysis = analyze(market, **LEGACY_SETTINGS)
    benchmarks = compare_benchmarks(market_benchmarks, market, analysis, risk_free_rate=0.03)
    evidence = analyze_evidence(analysis.equity, benchmarks.holdout_equity, risk_free_rate=0.03, periods_per_year=252.0,
                                delayed_entry=False)
    metadata = {"source": "CSV: legacy_market_prices.csv", "price_currency": "USD", **LEGACY_SETTINGS,
                "benchmark_source": "CSV: legacy_market_benchmark_prices.csv",
                "universe_coverage": COVERAGE}
    return market, analysis, None, None, benchmarks, evidence, metadata


def legacy_demo():
    """The demo with latest profiles and no market comparison."""
    prices = demo_prices()
    analysis = analyze(prices)
    latest = build_profiles(prices)
    metadata = {"source": "Demo · synthetic", "universe": "Demo", "price_currency": "USD",
                "benchmarks_note": "Synthetic demo data has no real market benchmark comparison."}
    return prices, analysis, None, latest, None, None, metadata


def legacy_case(name, build, **view):
    prices, analysis, study, latest, benchmarks, evidence, metadata = build()
    archive = presentation.report_zip(analysis, prices, metadata, study, latest, benchmarks, evidence, **view)
    charts = {
        "frontier_chart": figure(presentation.frontier_chart(
            analysis, benchmarks.training_estimates if benchmarks is not None else None, selected="Medium")),
        "holdout_chart": figure(presentation.holdout_chart(
            analysis, benchmarks.holdout_equity if benchmarks is not None else None)),
    }
    snippets = {}
    if latest is not None:
        charts["latest_profile_chart"] = figure(presentation.latest_profile_chart(
            latest, benchmarks.latest_estimates if benchmarks is not None else None, selected="Extreme"))
        snippets["why_text"] = story.why_text(latest, "Medium", "Nasdaq-100")
        snippets["stats_html"] = story.stats_html(latest, "Low", "Upload CSV", {"A00": "<Name>"})
        snippets["levels_ledger_html"] = story.levels_ledger_html(latest, "Extreme", None, "Demo")
    if benchmarks is not None:
        snippets["benchmark_html"] = benchmark_html(benchmarks, evidence, backtest=study is not None)
        snippets["benchmark_markdown"] = benchmark_markdown(benchmarks, evidence, backtest=study is not None)
    if study is not None:
        selected = "Expanding window · Medium"
        for mode in ("Growth", "Drawdown", "Relative to markets"):
            charts[f"story_chart {mode}"] = figure(story.story_chart(study, selected, mode, benchmarks, evidence))
        charts["evidence_chart"] = figure(evidence_chart(evidence, selected))
        snippets.update({
            "decision_panel_html": story.decision_panel_html(study, "Rolling window · Low", benchmarks),
            "market_answer": story.market_answer(study, selected, benchmarks),
            "market_lead": story.market_lead(study, selected),
            "result_tile_html": story.result_tile_html(study, selected, benchmarks, evidence, {"universe": "Nasdaq-100"}),
            "chart_footnote": story.chart_footnote(study, benchmarks),
            "evidence_answer": story.evidence_answer(evidence, selected),
            "evidence_lead": story.evidence_lead(evidence, selected, story.months_tested(study), {"evidence_note": "x"}),
            "clarity_text": story.clarity_text(evidence, selected),
            "interval_html": story.interval_html(evidence, selected),
            "stretch_summary_html": story.stretch_summary_html(evidence, selected),
            "stretch_table_html": story.stretch_table_html(evidence, selected),
            "backtest_html": backtest_html(study, benchmarks, evidence, focus="Rolling window · Extreme"),
            "findings_markdown": findings_markdown(study, "Golden", benchmarks, evidence),
        })
    write_golden(HERE / f"reports_legacy_{name}.zip", archive, snippets, charts)


def historical_case():
    prices, membership = historical_demo()
    study = run_historical_backtests(prices, membership)
    archive = historical_report_zip(build_historical_report(prices, membership, study))
    write_golden(HERE / "reports_historical_demo.zip", archive, {}, {})


def run_case(name, case):
    prices, benchmarks = prices_for(case)
    metadata = case["metadata"]
    study = run_selection(prices, benchmarks, **case["options"])
    frontier_data = selection_frontier(study, prices)
    result, error = walk_result(study, prices, case.get("first_window_fails"))
    archive_bytes = report.selection_report_zip(study, prices, metadata, frontier_data=frontier_data, out_of_sample=result)
    plotly = get_plotlyjs()
    snippets = {
        "selection_comparison_html": report.selection_comparison_html(study, prices),
        "selection_replay_findings_html": report.selection_replay_findings_html(study.test),
        "selection_replay_verdict_html": report.selection_replay_verdict_html(study.test),
        "selection_frontier_summary_html": report.selection_frontier_summary_html(study, prices, frontier_data),
        "selection_summary_html": report.selection_summary_html(study, metadata),
        "selection_benchmark_note": report.selection_benchmark_note(study),
        "selection_fit_summary_html": report.selection_fit_summary_html(study, prices),
        "out_of_sample_html": report.out_of_sample_html(result, error),
        "result_drivers_html": story.result_drivers_html(study.test, study.selected),
        "context_result_html": story.context_result_html(metadata, prices),
        "fine_print_html": story.fine_print_html(metadata, prices),
        "backtest_html": backtest_html(study.test),
        "findings_markdown": findings_markdown(study.test, "Golden <source>"),
        "settings": [[story.setting_label(key), story.setting_text(key, entry)] for key, entry in study.settings.items()],
        "selection_metrics_frame": split(report.selection_metrics_frame(study.test)),
        "selection_replay_dominance": split(report.selection_replay_dominance(study.test)),
        "selection_replay_return_bridge": value(report.selection_replay_return_bridge(study, prices).to_dict()),
    }
    charts = {
        "selection_chart": figure(report.selection_chart(study.test)),
        "selection_frontier_chart": figure(report.selection_frontier_chart(frontier_data, study)),
        "backtest_chart_drawdown": figure(backtest_chart(study.test, drawdown=True)),
    }
    findings = report.out_of_sample_findings(result) if result is not None else None
    if findings is not None:
        headline, lines = report.out_of_sample_text(findings)
        snippets["out_of_sample_findings"] = value(findings)
        snippets["out_of_sample_text"] = [headline, lines]
        snippets["out_of_sample_frame"] = split(report.out_of_sample_frame(result))
        snippets["out_of_sample_paths_frame"] = split(report.out_of_sample_paths_frame(result))
        charts["out_of_sample_chart"] = figure(report.out_of_sample_chart(result))
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source, \
            zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as golden:
        names = source.namelist()
        for member in names:
            text = source.read(member).decode("utf-8")
            if member == "report.html":
                assert text.count(plotly) == 1
                text = text.replace(plotly, "[plotly.js]")
            golden.writestr(f"archive/{member}", text)
        golden.writestr("names.json", json.dumps(names, indent=1))
        golden.writestr("snippets.json", json.dumps(snippets, indent=1, ensure_ascii=False))
        golden.writestr("charts.json", json.dumps(charts, indent=1, ensure_ascii=False))
    (HERE / f"reports_{name}.zip").write_bytes(output.getvalue())


def main():
    prices, benchmarks = walk_market()
    prices.to_csv(HERE / "reports_walk_prices.csv", lineterminator="\n")
    benchmarks.to_csv(HERE / "reports_walk_benchmarks.csv", lineterminator="\n")
    for name, case in CASES.items():
        run_case(name, case)
    legacy_case("full", legacy_full, selected_profile="Extreme", selected_method="Rolling window")
    legacy_case("plain", legacy_plain)
    legacy_case("demo", legacy_demo)
    historical_case()


if __name__ == "__main__":
    main()
