"""Charts and portable reports for an efficient-frontier analysis."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path
import zipfile
from typing import Any

import pandas as pd
import plotly.graph_objects as go


COLORS = {"Minimum volatility": "#40d4be", "Maximum Sharpe": "#ffcb77", "Equal weight": "#aab7ff"}
PLOTLY_JS_LICENSE = (Path(__file__).parent / "third_party" / "plotly.js.LICENSE.txt").read_text(encoding="utf-8")


def _style(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title=title, template="plotly_dark", paper_bgcolor="#111c2e", plot_bgcolor="#111c2e",
        font={"family": "Arial, sans-serif", "color": "#e6edf7"},
        margin={"l": 65, "r": 25, "t": 65, "b": 60},
        legend={"orientation": "h", "y": -0.22}, height=470,
        hovermode="closest",
    )
    figure.update_xaxes(gridcolor="#26344a")
    figure.update_yaxes(gridcolor="#26344a")
    return figure


def frontier_chart(analysis: Any) -> go.Figure:
    """Show annualized historical training estimates, not future returns."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=analysis.frontier["volatility"], y=analysis.frontier["expected_return"],
        mode="lines", name="Efficient frontier", line={"color": "#40d4be", "width": 3},
        hovertemplate="Annual volatility: %{x:.2%}<br>Historical expected return: %{y:.2%}<extra>%{fullData.name}</extra>",
    ))
    for name, portfolio in analysis.portfolios.items():
        figure.add_trace(go.Scatter(
            x=[portfolio.volatility], y=[portfolio.expected_return], mode="markers", name=name,
            marker={"size": 13, "color": COLORS.get(name, "#f3a7da"), "line": {"width": 2, "color": "#111c2e"}},
            hovertemplate="Annual volatility: %{x:.2%}<br>Historical expected return: %{y:.2%}<extra>%{fullData.name}</extra>",
        ))
    _style(figure, "Training period · historical efficient frontier")
    figure.update_xaxes(title="Annual volatility", tickformat=".1%", rangemode="tozero")
    figure.update_yaxes(title="Historical expected annual return", tickformat=".1%")
    return figure


def holdout_chart(analysis: Any) -> go.Figure:
    """Plot the realized buy-and-hold value of an initial 10,000 units."""
    figure = go.Figure()
    for name in analysis.equity.columns:
        figure.add_trace(go.Scatter(
            x=analysis.equity.index, y=analysis.equity[name] * 10_000, name=name,
            mode="lines", line={"width": 2.5, "color": COLORS.get(name, "#f3a7da")},
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio value: %{y:,.2f}<extra>%{fullData.name}</extra>",
        ))
    _style(figure, "Holdout period · realized buy-and-hold performance")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title="Value of 10,000 initial units", tickformat=",.0f")
    return figure


def weights_frame(analysis: Any) -> pd.DataFrame:
    frame = pd.DataFrame({name: portfolio.weights for name, portfolio in analysis.portfolios.items()})
    frame.index.name = "ticker"
    return frame


def _metadata(analysis: Any, metadata: dict) -> dict:
    result = dict(metadata)
    for label, returns in (("training", analysis.train_returns), ("holdout", analysis.test_returns)):
        result[f"{label}_start"] = str(returns.index[0].date()) if len(returns) else None
        result[f"{label}_end"] = str(returns.index[-1].date()) if len(returns) else None
        result[f"{label}_observations"] = len(returns)
    result.update(
        assets=list(analysis.train_returns.columns), annualization_days=252,
        holdout_strategy="Allocate once at the split, then hold adjusted-price exposures; no cross-asset rebalancing, and weights drift.",
        distributions="Reflected in the data provider's adjusted-price series, not accumulated as separate cash.",
        estimation="Historical arithmetic mean returns and covariance estimated from training data only.",
        exclusions="No trading costs, taxes, or currency conversion (FX).",
    )
    return result


def _table(frame: pd.DataFrame, percent_columns: list | None = None) -> str:
    display = frame.copy()
    for column in display.columns:
        if percent_columns is None or column in percent_columns:
            display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
        else:
            display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    return display.to_html(escape=True, border=0, classes="data", na_rep="—")


def report_html(analysis: Any, metadata: dict) -> str:
    """Return an offline HTML report with a single embedded Plotly bundle."""
    details = _metadata(analysis, metadata)
    source = str(details.get("source", "Unspecified source"))
    source_label = "Synthetic demonstration data" if any(word in source.lower() for word in ("demo", "synthetic")) else "Data source"
    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' ').capitalize())}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in details.items()
    )
    warnings = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in analysis.warnings)
    warnings_html = f'<aside><h2>Analysis notes</h2><ul>{warnings}</ul></aside>' if warnings else ""
    estimates = pd.DataFrame.from_dict({
        name: {"Historical expected annual return": p.expected_return, "Annual volatility": p.volatility, "Estimated Sharpe": p.sharpe}
        for name, p in analysis.portfolios.items()
    }, orient="index")
    holdout = analysis.holdout_metrics.rename(columns={
        "total_return": "Total return", "cagr": "CAGR", "volatility": "Annual volatility",
        "sharpe": "Realized Sharpe", "max_drawdown": "Maximum drawdown",
    })
    chart_options = {"responsive": True, "displaylogo": False}
    frontier = frontier_chart(analysis).to_html(full_html=False, include_plotlyjs=True, config=chart_options)
    holdout_plot = holdout_chart(analysis).to_html(full_html=False, include_plotlyjs=False, config=chart_options)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Efficient Frontier · Research Report</title>
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#091321;color:#e6edf7;font:16px/1.6 Arial,sans-serif}}
main{{max-width:1150px;margin:auto;padding:40px 24px}}h1{{font-size:36px;line-height:1.2}}h2{{font-size:23px;margin-top:32px}}
p{{max-width:950px}}.muted{{color:#adbbce}}.source,aside{{padding:16px 20px;background:#172a3b;border-left:4px solid #40d4be}}
.chart{{background:#111c2e;border-radius:12px;margin:24px 0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border-bottom:1px solid #26344a;padding:10px 12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.metadata th{{width:30%;text-align:left}}.metadata td{{text-align:left;overflow-wrap:anywhere}}footer{{margin-top:32px;color:#adbbce}}pre{{white-space:pre-wrap}}
</style></head><body><main>
<p class="muted">PORTFOLIO RESEARCH / REPRODUCIBLE ANALYSIS</p><h1>Efficient Frontier</h1>
<p class="source"><strong>{source_label}:</strong> {html.escape(source)}</p>
<p>Weights are selected using the training period, then evaluated on later, held-out observations.
Training estimates describe historical data. They are not predictions or guarantees of future performance.</p>
{warnings_html}
<h2>Training estimates</h2><p class="muted">Arithmetic mean returns and covariance are annualized using 252 trading days.
Covariance shrinkage is applied using training data only. The configured risk-free rate is used for Sharpe ratios.</p>
<div class="chart">{frontier}</div>
<div class="scroll">{_table(estimates, ["Historical expected annual return", "Annual volatility"])}</div>
<h2>Portfolio allocations</h2><p class="muted">These are the initial weights at the training/holdout split.</p>
<div class="scroll">{_table(weights_frame(analysis))}</div>
<h2>Realized holdout results</h2><p class="muted">Each portfolio starts with 10,000 units, allocates once at the split,
and then holds adjusted-price exposures. There is no subsequent trading between assets; portfolio weights drift with prices.
Distributions follow the provider's price adjustments rather than accumulating as separate cash.
CAGR is annualized using 252 trading days.</p><div class="chart">{holdout_plot}</div>
<div class="scroll">{_table(holdout, [column for column in holdout.columns if column != "Realized Sharpe"])}</div>
<h2>Data and assumptions</h2><div class="scroll"><table class="metadata">{metadata_rows}</table></div>
<footer>No trading costs, taxes, or currency conversion (FX) are included. Price series must share a consistent currency basis.
This report is a historical research tool and does not predict investment outcomes. Charts work offline.
<details><summary>Third-party notice: Plotly.js (MIT)</summary><pre>{html.escape(PLOTLY_JS_LICENSE)}</pre></details></footer>
</main></body></html>'''


def report_zip(analysis: Any, prices: pd.DataFrame, metadata: dict) -> bytes:
    """Bundle the report, inputs, and numeric results for offline inspection."""
    output = io.BytesIO()
    frames = {
        "weights.csv": weights_frame(analysis), "frontier.csv": analysis.frontier,
        "frontier_weights.csv": analysis.frontier_weights, "holdout_metrics.csv": analysis.holdout_metrics,
        "holdout_curve.csv": analysis.equity, "prices.csv": prices,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("report.html", report_html(analysis, metadata))
        bundle.writestr("THIRD_PARTY_NOTICES.txt", "Plotly.js (embedded in report.html)\n\n" + PLOTLY_JS_LICENSE)
        for name, frame in frames.items():
            bundle.writestr(name, frame.to_csv())
        bundle.writestr("metadata.json", json.dumps(_metadata(analysis, metadata), indent=2, default=str))
    return output.getvalue()
