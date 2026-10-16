"""Offline evidence and reproducible exports for a portfolio risk target."""

from dataclasses import asdict
import html
import io
import json
import zipfile

import pandas as pd
from plotly.offline import get_plotlyjs

from .backtest_report import backtest_chart
from .data import stale_price_notes
from .presentation import PLOTLY_JS_LICENSE, csv_text
from .style import PLOTLY_CONFIG, REPORT_CSS


SELECTION_METRIC_LABELS = {
    "annual_mean": "Net arithmetic mean / year",
    "cagr": "Net compound growth / year (CAGR)",
    "volatility": "Realized volatility / year",
    "max_drawdown": "Maximum drawdown",
    "total_return": "Net total return",
    "total_cost": "Fees / initial capital",
}
HISTORY_NOTE = (
    "The later test was not used to choose the blend. Historical data reviewed earlier can introduce research hindsight; "
    "a chronological split alone is not unseen prospective evidence. The support screen is descriptive and is not an investment recommendation."
)


def selection_metrics_frame(backteststudy) -> pd.DataFrame:
    """Keep the engine's fixed strategy order and exact realized metrics."""
    return backteststudy.metrics.loc[:, list(SELECTION_METRIC_LABELS)].copy()


def selection_summary_html(study, metadata: dict) -> str:
    """Describe the frozen selection and support decision without a winner ranking."""
    heading = "Passed the descriptive support screen" if study.supported else "No supported stock selection"
    reasons = list(dict.fromkeys([*study.policy.reasons, *study.reasons]))
    notes = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reasons)
    benchmark_note = metadata.get("benchmarks_note", "")
    return (
        f'<h2>{heading}</h2><p>Annual volatility target: {study.settings["target_volatility"]:.1%}. '
        f'Chosen historical-mean blend: {study.policy.blend:.1%}. '
        f'Validation screen: {"passed" if study.policy.validation_passed else "not passed"}.</p>'
        '<p>The blend is fixed after validation. The later test cannot choose a different blend. '
        'A volatility target is an estimate, not a limit on future losses or realized volatility.</p>'
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
        f"{phase}/{number:02d}": str(name)
        for phase, result in (("validation", study.validation), ("test", study.test))
        for number, name in enumerate(result.equity.columns, 1)
    }
    warnings = list(dict.fromkeys([
        *study.warnings, *study.validation.warnings, *study.test.warnings, *stale_price_notes(prices),
    ]))
    return {
        **metadata, "study": "Single-portfolio risk-target selection", "settings": study.settings,
        "policy": asdict(study.policy), "selected": study.selected, "supported": study.supported,
        "reasons": study.reasons, "warnings": warnings, "interpretation": HISTORY_NOTE,
        "price_start": str(prices.index[0].date()), "price_end": str(prices.index[-1].date()),
        "price_symbols": list(prices.columns), "validation": study.validation.settings,
        "test": study.test.settings, "result_files": result_files,
    }


def selection_report_html(study, prices: pd.DataFrame, metadata: dict) -> str:
    """Show the decision, both evaluation periods, and experimental fitted weights."""
    details = _details(study, prices, metadata)
    sections = []
    for phase, title, result in (
        ("validation", "Validation: choose the blend", study.validation),
        ("test", "Later test (not used to choose the blend)", study.test),
    ):
        metrics = selection_metrics_frame(result).rename(columns=SELECTION_METRIC_LABELS)
        chart = backtest_chart(result, focus=study.selected).to_html(
            full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
        )
        sections.append(
            f'<section id="{phase}"><h2>{title}</h2>'
            f'<p>Baseline: {result.equity.index[0].date()}. '
            f'Evaluation returns: {result.equity.index[1].date()} to {result.equity.index[-1].date()}.</p>'
            '<p>The table keeps a fixed strategy order. It does not rank winners. All rows use the same evaluation dates. '
            'Arithmetic means use net period returns, including entry fees; CAGR compounds the same paths.</p>'
            f'<div class="scroll">{_percent_table(metrics)}</div><div class="chart">{chart}</div></section>'
        )
    weights, estimates = _latest_frames(study)
    latest = (
        '<p>These weights fit all supplied history through the final price date. The validation and later tests used earlier fitted weights. '
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
<title>Portfolio Lab · Risk-target selection</title><style>{REPORT_CSS}</style><script>{get_plotlyjs()}</script></head>
<body><main><p class="eyebrow">Historical research</p><h1>One portfolio, one risk target</h1>
<button onclick="window.print()">Print / save PDF</button>
<section id="selection-decision">{selection_summary_html(study, metadata)}</section>
{''.join(sections)}
<section id="latest-selection"><h2>Latest fitted stock weights · {prices.index[-1].date()}</h2>{latest}</section>
<section id="selection-assumptions"><h2>Method and limits</h2>
<p>Each allocation uses prices available before its execution close. The first evaluation interval stays in cash.
Trading costs apply to securities bought or sold, including entry, with no terminal sale. Cash earns zero interest.
Risk-matched ETF paths can hold cash to meet their estimated risk target; raw ETF paths buy once and hold.
Taxes, currency conversion, liquidity, and future membership changes are not modeled.</p>
<p>Past-only eligibility checks can exclude long unchanged-price runs at a fit. eligibility.csv records each decision.
The original input prices and excluded asset columns remain in the export.</p>
<p>Current constituent lists and complete-history requirements introduce selection and survivorship bias.
The later test is a descriptive historical check, not proof of future outperformance.</p>
{'<aside><h3>Data and model notes</h3><ul>' + notes + '</ul></aside>' if notes else ''}</section>
<section id="selection-sources"><h2>Source and reproduction</h2><div class="scroll"><table class="metadata">{source_rows}</table></div>{coverage}
<p>metadata.json records settings, input dates, source details, policy, support reasons, and every numbered result file.
policy.json records the blend fixed after validation. Each phase includes every path, allocation, trade, and holding.</p></section>
<footer>This report works offline.<details><summary>Third-party notice: Plotly.js (MIT)</summary>
<pre>{html.escape(PLOTLY_JS_LICENSE)}</pre></details></footer></main></body></html>'''


def selection_report_zip(study, prices: pd.DataFrame, metadata: dict) -> bytes:
    """Export exact inputs, fixed policy, and every validation and later-test result."""
    details = _details(study, prices, metadata)
    weights, estimates = _latest_frames(study)
    frames = {"prices.csv": prices.rename_axis("Date"), "latest_weights.csv": weights, "latest_estimates.csv": estimates}
    if getattr(study, "benchmark_prices", None) is not None:
        frames["benchmark_prices.csv"] = study.benchmark_prices.rename_axis("Date")
    if getattr(study, "eligibility", None) is not None:
        frames["eligibility.csv"] = study.eligibility
    if metadata.get("universe_coverage"):
        frames["universe_coverage.csv"] = pd.DataFrame(metadata["universe_coverage"])
    for phase, result in (("validation", study.validation), ("test", study.test)):
        frames[f"{phase}/metrics.csv"] = result.metrics.rename_axis("strategy")
        frames[f"{phase}/equity.csv"] = result.equity.rename_axis("date")
        for number, name in enumerate(result.equity.columns, 1):
            for kind in ("allocations", "holdings", "trades"):
                frames[f"{phase}/{number:02d}_{kind}.csv"] = getattr(result, kind)[name].rename_axis("symbol" if kind == "holdings" else "date")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html", selection_report_html(study, prices, metadata))
        archive.writestr("metadata.json", json.dumps(details, indent=2, default=str, ensure_ascii=False))
        archive.writestr("policy.json", json.dumps(asdict(study.policy), indent=2, ensure_ascii=False))
        archive.writestr("THIRD_PARTY_NOTICES.txt", PLOTLY_JS_LICENSE)
        for name, frame in frames.items():
            archive.writestr(name, csv_text(frame))
    return output.getvalue()
