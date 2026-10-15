"""Portable inputs, results and offline charts for historical membership studies."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path
import zipfile
from typing import Any

import pandas as pd
from plotly.offline import get_plotlyjs

from .backtest_report import HOLDINGS, METRICS, _formatted, backtest_chart
from .presentation import PLOTLY_JS_LICENSE, csv_text
from .style import PLOTLY_CONFIG, REPORT_CSS


ASSUMPTIONS = (
    "The three portfolios refit at each scheduled trade. Each fit uses the configured rolling "
    "window of returns through the prior close. New targets execute at the following close and "
    "earn no execution-day return. The first evaluation interval stays in cash.",
    "Each trade uses the membership effective on its execution date. Membership publication "
    "timestamps are not verified. Removed companies remain held until "
    "the next scheduled rebalance. An eligible asset needs the full training window of prices. "
    "The coverage file records exclusions at each trade; the study does not require every company "
    "to have prices for the full study period.",
    "Missing prices for a held asset or an execution stop the study. Prices are never filled. "
    "Cash earns zero interest. If the eligible assets cannot form a portfolio within the holding "
    "limit, the portfolio stays in cash for that allocation.",
    "Trading costs apply to purchases and sales, including entry. No final liquidation is charged. "
    "Holding P&L contributions less fees reconcile to net total return. Gross turnover counts both "
    "purchases and sales divided by portfolio value before each trade. Taxes, currency conversion "
    "and liquidity constraints are excluded. Adjusted prices must use one currency basis.",
    "Historical membership reduces one source of survivorship bias. It does not guarantee complete "
    "history: community records can omit changes, and price providers can omit delisted companies "
    "or reuse symbols. Read the source metadata and coverage exclusions before interpreting results.",
    "To rerun a study with bundled history, keep --historical-membership with the same bundled data "
    "revision, exported prices and settings. membership.csv alone does not preserve identity "
    "exclusions. metadata.json records those rules, but loading the snapshots with --membership-csv "
    "treats the symbols as user-supplied stable identifiers and does not restore the rules.",
    "These are retrospective comparisons, not forecasts. Choosing a strategy after seeing its "
    "results can overfit the evaluated period.",
)


def _metadata_table(values: dict) -> str:
    rows = []
    for key, value in values.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        rows.append(f"<tr><th>{html.escape(str(key).replace('_', ' ').capitalize())}</th>"
                    f"<td>{html.escape(str(value))}</td></tr>")
    return '<div class="scroll"><table class="metadata">' + "".join(rows) + "</table></div>"


def build_historical_report(prices: pd.DataFrame, membership: Any, study: Any) -> dict[str, bytes]:
    """Return the report, input copies and all numeric results as portable files."""
    membership_frame = pd.DataFrame({
        "tickers": [",".join(symbols) for symbols in membership.snapshots],
    }, index=membership.snapshots.index)
    membership_frame.index.name = "date"
    strategy_files = {f"backtests/{i:02d}": str(name) for i, name in enumerate(study.equity.columns, 1)}
    source = {
        **membership.metadata,
        "coverage_start": str(pd.Timestamp(membership.coverage_start).date()),
        "coverage_end": str(pd.Timestamp(membership.coverage_end).date()),
    }
    details = {
        "study": "Historical membership rolling-window backtest",
        "membership": source,
        "price_symbols": list(prices.columns),
        "price_start": str(prices.index[0].date()),
        "price_end": str(prices.index[-1].date()),
        "settings": study.settings,
        "warnings": study.warnings,
        "backtest_files": strategy_files,
        "assumptions": list(ASSUMPTIONS),
    }
    frames = {
        "prices.csv": prices.rename_axis("Date"), "membership.csv": membership_frame,
        "equity.csv": study.equity.rename_axis("date"),
        "metrics.csv": study.metrics.rename_axis("strategy"),
        "cash_weights.csv": study.cash_weights.rename_axis("date"),
        "coverage.csv": study.coverage.set_index("date"),
    }
    holdings_sections = []
    for stem, name in strategy_files.items():
        frames[f"{stem}_allocations.csv"] = study.allocations[name].rename_axis("date")
        frames[f"{stem}_holdings.csv"] = study.holdings[name].rename_axis("symbol")
        frames[f"{stem}_trades.csv"] = study.trades[name].rename_axis("date")
        holdings_sections.append(
            f"<details><summary>{html.escape(name)}</summary><div class=\"scroll\">"
            + _formatted(study.holdings[name], HOLDINGS).to_html(escape=True, border=0, classes="data")
            + "</div></details>"
        )
    files = {name: csv_text(frame).encode("utf-8") for name, frame in frames.items()}
    files["metadata.json"] = json.dumps(details, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    notices = "Plotly.js (embedded in report.html)\n\n" + PLOTLY_JS_LICENSE
    if "LICENSE.n100tickers" in str(source.get("license", "")):
        membership_license = Path(__file__).with_name("membership_data").joinpath("LICENSE.n100tickers").read_bytes()
        files["LICENSE.n100tickers"] = membership_license
        notices += "\n\nn100tickers membership data\n\n" + membership_license.decode("utf-8")
    files["THIRD_PARTY_NOTICES.txt"] = notices.encode("utf-8")
    charts = "".join(
        '<div class="chart">' + backtest_chart(study, drawdown=drawdown).to_html(
            full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
        ) + "</div>" for drawdown in (False, True)
    )
    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in study.warnings)
    assumptions = "".join(f"<p>{html.escape(note)}</p>" for note in ASSUMPTIONS)
    dates = study.equity.index
    metrics = _formatted(study.metrics, METRICS).to_html(escape=True, border=0, classes="data")
    coverage = study.coverage.to_html(index=False, escape=True, border=0, classes="data")
    cash = (study.cash_weights.iloc[-1].mul(100).map(lambda value: f"{value:.2f}%")
            .to_frame("Closing cash weight").to_html(escape=True, border=0, classes="data"))
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Lab · Historical membership</title>
<style>{REPORT_CSS}</style><script>{get_plotlyjs()}</script></head><body><main>
<p class="eyebrow">Historical research</p><h1>Historical membership</h1>
<p>Three rolling-window portfolios use dated membership and the price history available before each trade.</p>
<p>Portfolio baseline: {dates[0].date()}. Evaluation returns: {dates[1].date()} to {dates[-1].date()}.
Charts start with 10,000 units and include modeled trading costs.</p>
<section><h2>Membership source and coverage</h2>{_metadata_table(source)}
<p>Read the source limits alongside the results. Dated membership does not establish a complete historical reconstruction.</p></section>
<section><h2>Portfolio results</h2>{charts}<div class="scroll">{metrics}</div>
<h3>Cash at the final close</h3><div class="scroll">{cash}</div>
<p>cash_weights.csv contains daily closing cash proportions, including the initial cash balance.</p></section>
<section><h2>Holdings</h2><p>Asset P&amp;L contributions use initial capital as the denominator.
Fees are accounted for separately. Allocations, trades and all holdings are included in the CSV files.</p>
{''.join(holdings_sections)}</section>
<section><h2>Settings</h2>{_metadata_table(study.settings)}<h2>Assumptions</h2>{assumptions}
<p>CAGR and volatility use {study.settings.get('periods_per_year', 252):g} observations per year.
An em dash marks an undefined ratio.</p>
<details><summary>Eligibility at each trade</summary><div class="scroll">{coverage}</div></details></section>
{f'<aside><h2>Study notes</h2><ul>{notes}</ul></aside>' if notes else ''}
<footer>This report works offline. metadata.json maps numbered result files to portfolio names.
<details><summary>Third-party notices</summary><pre>{html.escape(notices)}</pre></details></footer>
</main></body></html>'''
    files["report.html"] = document.encode("utf-8")
    return files


def historical_report_zip(files: dict[str, bytes]) -> bytes:
    """Package files returned by build_historical_report without changing them."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()
