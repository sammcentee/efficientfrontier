"""Offline evidence and reproducible exports for the single portfolio model."""

import html
import io
import json
import zipfile

import pandas as pd
from plotly.offline import get_plotlyjs

from .backtest_report import backtest_chart
from .data import stale_price_notes
from .presentation import PLOTLY_JS_LICENSE, csv_text
from .selection import BASELINE, BENCHMARK_NAMES, SELECTED
from .style import ACCENT, BENCHMARK_COLORS, INK_3, PLOTLY_CONFIG, REPORT_CSS


SELECTION_METRIC_LABELS = {
    "annual_mean": "Net arithmetic mean / year",
    "cagr": "Net compound growth / year (CAGR)",
    "volatility": "Realized volatility / year",
    "max_drawdown": "Maximum drawdown",
    "total_return": "Net total return",
    "total_cost": "Fees / initial capital",
}
HISTORY_NOTE = (
    "The same model rule applies throughout the later test. Historical data reviewed earlier can introduce research hindsight. "
    "A chronological split alone is not fresh prospective evidence. These descriptive checks do not prove future outperformance."
)


def selection_metrics_frame(backteststudy) -> pd.DataFrame:
    """Keep the engine's fixed strategy order and exact realized metrics."""
    return backteststudy.metrics.loc[:, list(SELECTION_METRIC_LABELS)].copy()


def selection_chart(study):
    """Identify every comparison path in the app and exported report."""
    figure = backtest_chart(study, focus=SELECTED)
    colors = {SELECTED: ACCENT, BASELINE: INK_3, **BENCHMARK_COLORS}
    dashes = {BASELINE: "dash", BENCHMARK_NAMES["SPY"]: "dot"}
    for name, trace in zip(study.equity.columns, figure.data):
        trace.update(showlegend=True, line={"color": colors.get(name, INK_3),
                                           "dash": dashes.get(name, "solid")})
    return figure


def selection_summary_html(study, metadata: dict) -> str:
    """Describe the one model and its later historical check."""
    heading = "Later test: checks passed" if study.supported else "Later test: checks not met"
    reasons = list(dict.fromkeys(study.reasons))
    notes = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reasons)
    benchmark_note = metadata.get("benchmarks_note", "")
    return (
        f'<h2>{heading}</h2><p>Annual volatility budget: {study.settings["target_volatility"]:.1%}. '
        f'Largest holding: {study.settings["max_weight"]:.1%}. '
        'The model maximizes the fitted historical mean within the volatility budget and holding limit.</p>'
        '<p>The budget limits estimated volatility. Future volatility and losses can exceed it. '
        'The later test checks the same rule against equal-weight stocks, SPY, and QQQ.</p>'
        '<p>A pass requires both ETFs and at least one year of test observations. '
        'Realized test volatility must stay within the budget. No reference can have higher CAGR with equal or lower volatility.</p>'
        + (f'<ul>{notes}</ul>' if notes else '')
        + (f'<p>{html.escape(str(benchmark_note))}</p>' if benchmark_note else '')
        + f'<p class="source">{HISTORY_NOTE}</p>'
    )


def _percent_table(frame: pd.DataFrame) -> str:
    return frame.map(lambda value: "—" if pd.isna(value) else f"{value:.2%}").to_html(
        escape=True, border=0, classes="data",
    )


def _latest_frames(study) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame({name: portfolio.weights for name, portfolio in study.latest.items()}).rename_axis("symbol")
    estimates = pd.DataFrame.from_dict({
        name: {"expected_return": portfolio.expected_return, "volatility": portfolio.volatility, "sharpe": portfolio.sharpe}
        for name, portfolio in study.latest.items()
    }, orient="index").rename_axis("strategy")
    return weights, estimates


def _details(study, prices: pd.DataFrame, metadata: dict) -> dict:
    result_files = {
        f"test/{number:02d}": str(name)
        for number, name in enumerate(study.test.equity.columns, 1)
    }
    warnings = list(dict.fromkeys([
        *study.warnings, *study.test.warnings, *stale_price_notes(prices),
    ]))
    return {
        **metadata, "study": "Single historical-mean portfolio", "settings": study.settings,
        "selected": study.selected, "supported": study.supported,
        "reasons": study.reasons, "warnings": warnings, "interpretation": HISTORY_NOTE,
        "price_start": str(prices.index[0].date()), "price_end": str(prices.index[-1].date()),
        "price_symbols": list(prices.columns),
        "test": study.test.settings, "result_files": result_files,
    }


def selection_report_html(study, prices: pd.DataFrame, metadata: dict) -> str:
    """Show one later test and the latest fitted stock weights."""
    details = _details(study, prices, metadata)
    result = study.test
    metrics = selection_metrics_frame(result).rename(columns=SELECTION_METRIC_LABELS)
    chart = selection_chart(result).to_html(
        full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
    )
    test_section = (
        '<section id="test"><h2>Later historical test</h2>'
        f'<p>Baseline: {result.equity.index[0].date()}. '
        f'Evaluation returns: {result.equity.index[1].date()} to {result.equity.index[-1].date()}.</p>'
        '<p>All rows use the same dates and fee rate. Equal-weight stocks restore equal targets at each scheduled trade. '
        'SPY and QQQ buy once and hold. These references have no volatility budget.</p>'
        '<p>Arithmetic means average the net period returns, including entry fees. CAGR compounds the same returns. '
        'The table keeps a fixed order and does not choose a model from the results.</p>'
        f'<div class="scroll">{_percent_table(metrics)}</div><div class="chart">{chart}</div></section>'
    )
    weights, estimates = _latest_frames(study)
    latest = (
        '<p>These weights fit all supplied history through the final price date. The later test used earlier fitted weights. '
        'Latest fitted means are annual arithmetic estimates, not realized CAGR or forecasts.</p>'
        f'<div class="scroll">{_percent_table(weights)}</div>'
        f'<div class="scroll">{_percent_table(estimates[["expected_return", "volatility"]].rename(columns={"expected_return": "Fitted arithmetic mean / year", "volatility": "Fitted volatility / year"}))}</div>'
    )
    if not study.supported:
        latest = '<details><summary>Experimental latest holdings</summary>' + latest + '</details>'
    notes = "".join(f'<li>{html.escape(str(note))}</li>' for note in details["warnings"])
    source_rows = "".join(
        f'<tr><th>{html.escape(str(key).replace("_", " "))}</th>'
        f'<td>{html.escape(json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list, tuple)) else str(value))}</td></tr>'
        for key, value in metadata.items() if key != "universe_coverage"
    )
    coverage = (
        '<details><summary>Universe coverage</summary><div class="scroll">'
        + pd.DataFrame(metadata["universe_coverage"]).to_html(index=False, escape=True, border=0, classes="data") + '</div></details>'
        if metadata.get("universe_coverage") else ''
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Lab · Historical-mean portfolio</title><style>{REPORT_CSS}</style><script>{get_plotlyjs()}</script></head>
<body><main><p class="eyebrow">Historical research</p><h1>One portfolio, one model</h1>
<button onclick="window.print()">Print / save PDF</button>
<section id="selection-decision">{selection_summary_html(study, metadata)}</section>
{test_section}
<section id="latest-selection"><h2>Latest fitted stock weights · {prices.index[-1].date()}</h2>{latest}</section>
<section id="selection-assumptions"><h2>Method and limits</h2>
<p>The model maximizes historical arithmetic mean return under a fitted volatility budget and a per-stock holding limit.
Stock weights are long-only and sum to one. Each scheduled refit uses all available earlier returns.</p>
<p>Initial training share: {study.settings['train_fraction']:.0%}.
Refit every {study.settings['rebalance_every']} observations.
Fee rate: {study.settings['cost_bps']:g} basis points ({study.settings['cost_bps'] / 10000:.2%}) per amount bought or sold.
Covariance shrinkage: {study.settings['shrinkage']:.0%}.
Annualization: {study.settings['periods_per_year']:g} observations per year.</p>
<p>Each allocation uses prices through the close before execution. The first evaluation interval stays in cash.
Trading costs apply to securities bought or sold, including entry, with no terminal sale. Cash earns zero interest.
Equal-weight stocks use the same refit schedule. SPY and QQQ buy once and hold.
Taxes, currency conversion, liquidity, and future membership changes are not modeled.</p>
<p>Past-only eligibility checks can exclude long unchanged-price runs at a fit. eligibility.csv records each decision.
The original input prices and excluded asset columns remain in the export.</p>
<p>Current constituent lists and complete-history requirements introduce selection and survivorship bias.
The later test is a descriptive historical check, not proof of future outperformance.</p>
{'<aside><h3>Data and model notes</h3><ul>' + notes + '</ul></aside>' if notes else ''}</section>
<section id="selection-sources"><h2>Source and reproduction</h2><div class="scroll"><table class="metadata">{source_rows}</table></div>{coverage}
<p>metadata.json records the model revision, settings, input dates, source details, test reasons, and every numbered result file.
The test exports include every path, allocation, trade, and holding.</p></section>
<footer>This report works offline.<details><summary>Third-party notice: Plotly.js (MIT)</summary>
<pre>{html.escape(PLOTLY_JS_LICENSE)}</pre></details></footer></main></body></html>'''


def selection_report_zip(study, prices: pd.DataFrame, metadata: dict) -> bytes:
    """Export exact inputs, model settings, and every later-test result."""
    details = _details(study, prices, metadata)
    weights, estimates = _latest_frames(study)
    frames = {"prices.csv": prices.rename_axis("Date"), "latest_weights.csv": weights, "latest_estimates.csv": estimates}
    if getattr(study, "benchmark_prices", None) is not None:
        frames["benchmark_prices.csv"] = study.benchmark_prices.rename_axis("Date")
    if getattr(study, "eligibility", None) is not None:
        frames["eligibility.csv"] = study.eligibility
    if metadata.get("universe_coverage"):
        frames["universe_coverage.csv"] = pd.DataFrame(metadata["universe_coverage"])
    result = study.test
    frames["test/metrics.csv"] = result.metrics.rename_axis("strategy")
    frames["test/equity.csv"] = result.equity.rename_axis("date")
    for number, name in enumerate(result.equity.columns, 1):
        for kind in ("allocations", "holdings", "trades"):
            frames[f"test/{number:02d}_{kind}.csv"] = getattr(result, kind)[name].rename_axis("symbol" if kind == "holdings" else "date")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html", selection_report_html(study, prices, metadata))
        archive.writestr("metadata.json", json.dumps(details, indent=2, default=str, ensure_ascii=False))
        archive.writestr("THIRD_PARTY_NOTICES.txt", PLOTLY_JS_LICENSE)
        for name, frame in frames.items():
            archive.writestr(name, csv_text(frame))
    return output.getvalue()
