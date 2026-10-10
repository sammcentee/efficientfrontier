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
from plotly.offline import get_plotlyjs

from .backtest_report import backtest_html, findings_markdown
from .benchmark_report import add_benchmark_estimates, add_benchmark_paths, benchmark_html
from .risk import risk_contributions, risk_summary
from .style import COLORS, style_chart


PLOTLY_JS_LICENSE = (Path(__file__).parent / "third_party" / "plotly.js.LICENSE.txt").read_text(encoding="utf-8")


def frontier_chart(analysis: Any, benchmark_estimates=None) -> go.Figure:
    """Show annualized historical training estimates, not future returns."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=analysis.frontier["volatility"], y=analysis.frontier["expected_return"],
        mode="lines", name="Efficient frontier", line={"color": "#7896b5", "width": 3},
        hovertemplate="Annual volatility: %{x:.2%}<br>Historical expected return: %{y:.2%}<extra>%{fullData.name}</extra>",
    ))
    for name, portfolio in analysis.portfolios.items():
        figure.add_trace(go.Scatter(
            x=[portfolio.volatility], y=[portfolio.expected_return], mode="markers", name=name,
            marker={"size": 13, "color": COLORS.get(name, "#737b85"), "line": {"width": 2, "color": "#ffffff"}},
            hovertemplate="Annual volatility: %{x:.2%}<br>Historical expected return: %{y:.2%}<extra>%{fullData.name}</extra>",
        ))
    add_benchmark_estimates(figure, benchmark_estimates, "expected_return")
    style_chart(figure, "Efficient frontier")
    figure.update_xaxes(title="Annual volatility", tickformat=".1%", rangemode="tozero")
    figure.update_yaxes(title="Expected annual return", tickformat=".1%")
    return figure


def holdout_chart(analysis: Any, benchmark_equity=None) -> go.Figure:
    """Plot the realized buy-and-hold value of an initial 10,000 units."""
    figure = go.Figure()
    for name in analysis.equity.columns:
        figure.add_trace(go.Scatter(
            x=analysis.equity.index, y=analysis.equity[name] * 10_000, name=name,
            mode="lines", line={"width": 2.5, "color": COLORS.get(name, "#737b85")},
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio value: %{y:,.2f}<extra>%{fullData.name}</extra>",
        ))
    add_benchmark_paths(figure, benchmark_equity)
    style_chart(figure, "Holdout performance", hovermode="x unified")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title="Value (initial 10,000)", tickformat=",.0f")
    return figure


def latest_profile_chart(profiles: Any, benchmark_estimates=None) -> go.Figure:
    """Plot the lowest historical window mean against estimated volatility."""
    figure = go.Figure(go.Scatter(
        x=profiles.frontier["volatility"], y=profiles.frontier["worst_window_return"],
        mode="lines", name="Window frontier", line={"color": "#7896b5", "width": 3},
        hovertemplate="Annual volatility: %{x:.2%}<br>Lowest annual window mean: %{y:.2%}<extra>%{fullData.name}</extra>",
    ))
    for name, row in profiles.summary.iterrows():
        figure.add_trace(go.Scatter(
            x=[row["volatility"]], y=[row["worst_window_return"]], mode="markers", name=html.escape(str(name)),
            marker={"size": 13, "color": COLORS.get(name, "#737b85"), "line": {"width": 2, "color": "#ffffff"}},
            hovertemplate="Annual volatility: %{x:.2%}<br>Lowest annual window mean: %{y:.2%}<extra>%{fullData.name}</extra>",
        ))
    add_benchmark_estimates(figure, benchmark_estimates, "worst_window_return")
    style_chart(figure, "Latest risk profiles")
    figure.update_xaxes(title="Annual volatility", tickformat=".1%", rangemode="tozero")
    figure.update_yaxes(title="Lowest annual window mean", tickformat=".1%")
    return figure


def weights_frame(analysis: Any) -> pd.DataFrame:
    frame = pd.DataFrame({name: portfolio.weights for name, portfolio in analysis.portfolios.items()})
    frame.index.name = "ticker"
    return frame


def csv_text(frame: pd.DataFrame, index_label=None) -> str:
    """Export numeric data while escaping formula-like labels and text cells."""
    def text_label(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    def text_axis(axis):
        if isinstance(axis, pd.MultiIndex):
            return axis.map(lambda values: tuple(text_label(value) for value in values)).set_names(
                [text_label(name) for name in axis.names]
            )
        return axis.map(text_label).rename(text_label(axis.name))

    display = frame.copy(deep=False)
    display.index = text_axis(frame.index)
    display.columns = text_axis(frame.columns)
    for column in display.select_dtypes(include=["object", "string"]).columns:
        display[column] = display[column].map(text_label)
    return display.to_csv(index_label=text_label(index_label), lineterminator="\r\n")


def _metadata(analysis: Any, metadata: dict, study=None, latest_profiles=None, benchmarks=None, evidence=None) -> dict:
    result = dict(metadata)
    for label, returns in (("training", analysis.train_returns), ("holdout", analysis.test_returns)):
        result[f"{label}_start"] = str(returns.index[0].date()) if len(returns) else None
        result[f"{label}_end"] = str(returns.index[-1].date()) if len(returns) else None
        result[f"{label}_observations"] = len(returns)
    result.update(
        assets=list(analysis.train_returns.columns), periods_per_year=analysis.periods_per_year,
        annualization_days=analysis.periods_per_year,  # Legacy metadata key.
        holdout_strategy="Allocate once at the split, then hold adjusted-price exposures; no cross-asset rebalancing, and weights drift.",
        distributions="Reflected in the data provider's adjusted-price series, not accumulated as separate cash.",
        estimation="Historical arithmetic mean returns and covariance estimated from training data only.",
        exclusions="No trading costs, taxes, or currency conversion (FX).",
    )
    if study is not None:
        result["backtests"] = study.settings
        result["backtest_warnings"] = study.warnings
        result["backtest_files"] = {f"backtests/{i:02d}": name for i, name in enumerate(study.equity.columns, 1)}
        result["exclusions"] = "The original holdout excludes trading costs; the backtest comparison deducts the selected fees. Both exclude taxes and FX."
    if latest_profiles is not None:
        result["latest_profiles"] = {
            "as_of": latest_profiles.as_of,
            "observations": latest_profiles.observations,
            "settings": latest_profiles.settings,
            "windows": latest_profiles.windows.rename_axis("window").reset_index().to_dict(orient="records"),
            "warnings": latest_profiles.warnings,
            "interpretation": "Fits all supplied history through the last input date. Window means are in-sample estimates, not forecasts or holdout results.",
        }
    if benchmarks is not None:
        result["benchmarks"] = benchmarks.settings
    if evidence is not None:
        result["evidence"] = {**evidence.settings, "warnings": evidence.warnings}
    return result


def _table(frame: pd.DataFrame, percent_columns: list | None = None) -> str:
    display = frame.copy()
    for column in display.columns:
        if percent_columns is None or column in percent_columns:
            display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
        else:
            display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    return display.to_html(escape=True, border=0, classes="data", na_rep="—")


def _latest_profiles_html(profiles: Any, benchmark_estimates=None) -> str:
    summary = profiles.summary.rename(columns={
        "expected_return": "Annual mean return", "volatility": "Annual volatility",
        "sharpe": "Estimated Sharpe", "worst_window_return": "Lowest annual window mean",
        "max_weight": "Largest weight", "effective_holdings": "Effective holdings",
    })
    windows = profiles.windows.rename(columns={
        "start": "Start", "end": "End", "observations": "Observations", "years": "Years of observations",
    }).rename_axis("Window")
    window_returns = profiles.window_returns.rename_axis("Window")
    warnings = "".join(f"<li>{html.escape(str(note))}</li>" for note in profiles.warnings)
    chart = latest_profile_chart(profiles, benchmark_estimates).to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False},
    )
    return (
        '<section id="latest-profiles">'
        f'<h2>Latest model holdings as of {html.escape(str(profiles.as_of))}</h2>'
        '<p class="source">These model weights use all supplied history through the last input date. '
        'The date above is the last price observation, not a live quote.</p>'
        '<p>Low, Medium, and Extreme are relative risk levels within this frontier. They are not universal risk ratings. '
        'The three historical windows are part of the fit. Their means are in-sample estimates, not forecasts or holdout results.</p>'
        + (f'<aside><h3>Profile notes</h3><ul>{warnings}</ul></aside>' if warnings else '')
        + '<h3>Latest weights</h3>'
        f'<div class="scroll">{_table(weights_frame(profiles))}</div>'
        '<h3>Profile estimates</h3>'
        f'<div class="scroll">{_table(summary, ["Annual mean return", "Annual volatility", "Lowest annual window mean", "Largest weight"])}</div>'
        '<p class="muted">The model minimizes estimated variance at a return target that applies to every historical window. '
        'Low uses the minimum-variance point. Extreme uses the highest feasible target for the lowest window mean. '
        'Medium targets the midpoint of this return range, not the midpoint of volatility.</p>'
        f'<div class="chart">{chart}</div>'
        '<h3>Historical windows</h3>'
        f'<p class="muted">The model divides {profiles.observations} return observations into three consecutive windows without overlap. '
        f'Annual arithmetic means use {profiles.settings["periods_per_year"]:g} observations per year. '
        'The means assume fixed weights each period and exclude fees and taxes.</p>'
        f'<div class="scroll">{windows.to_html(escape=True, border=0, classes="data", float_format=lambda value: f"{value:.2f}")}</div>'
        '<h3>Annual mean return in each window</h3>'
        f'<div class="scroll">{_table(window_returns)}</div>'
        '</section>'
    )


def report_html(analysis: Any, metadata: dict, study=None, latest_profiles=None, benchmarks=None, evidence=None) -> str:
    """Return an offline HTML report with a single embedded Plotly bundle."""
    details = _metadata(analysis, metadata, study, latest_profiles, benchmarks, evidence)
    source = str(details.get("source", "Unspecified source"))
    source_label = "Synthetic demonstration data" if any(word in source.lower() for word in ("demo", "synthetic")) else "Data source"
    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' ').capitalize())}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in details.items() if key not in ("backtests", "backtest_warnings", "backtest_files", "annualization_days", "latest_profiles", "benchmarks", "evidence", "universe_coverage")
    )
    coverage_html = ""
    if details.get("universe_coverage"):
        coverage = pd.DataFrame(details["universe_coverage"])
        coverage_html = ('<details><summary>Universe coverage</summary><div class="scroll">'
                         + coverage.to_html(index=False, escape=True, border=0, classes="data")
                         + '</div></details>')
    warnings = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in analysis.warnings)
    warnings_html = f'<aside><h2>Analysis notes</h2><ul>{warnings}</ul></aside>' if warnings else ""
    estimates = pd.DataFrame.from_dict({
        name: {"Historical expected annual return": p.expected_return, "Annual volatility": p.volatility, "Estimated Sharpe": p.sharpe}
        for name, p in analysis.portfolios.items()
    }, orient="index")
    holdout = analysis.holdout_metrics.rename(columns={
        "total_return": "Total return", "cagr": "CAGR", "volatility": "Annual volatility",
        "sharpe": "Realized Sharpe", "max_drawdown": "Maximum drawdown",
        "sortino": "Sortino", "calmar": "Calmar",
    })
    risk = risk_summary(analysis).rename(columns={
        "max_weight": "Largest weight", "effective_holdings": "Effective holdings",
        "diversification_ratio": "Diversification ratio",
    })
    chart_options = {"responsive": True, "displaylogo": False}
    frontier = frontier_chart(analysis, benchmarks.training_estimates if benchmarks is not None else None).to_html(
        full_html=False, include_plotlyjs=False, config=chart_options,
    )
    holdout_plot = holdout_chart(analysis, benchmarks.holdout_equity if benchmarks is not None else None).to_html(
        full_html=False, include_plotlyjs=False, config=chart_options,
    )
    comparison_html = backtest_html(study, benchmarks, evidence) if study is not None else (
        benchmark_html(benchmarks, evidence) if benchmarks is not None else ""
    )
    profiles_html = _latest_profiles_html(
        latest_profiles, benchmarks.latest_estimates if benchmarks is not None else None,
    ) if latest_profiles is not None else ""
    comparison_notes = "".join(
        f'<p class="source">{html.escape(str(details[key]))}</p>'
        for key in ("benchmarks_note", "evidence_note") if details.get(key)
    )
    cost_note = ("The original holdout excludes trading costs. The backtest comparison deducts the selected fees."
                 if study is not None else "No trading costs are included.")
    introduction = ("The backtest comparison updates portfolios using only information available before each trade. "
                    "The original fixed-allocation holdout is shown separately."
                    if study is not None else "The original holdout selects weights from the training period, then evaluates them on later observations.")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Efficient Frontier · Research Report</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f5f7;color:#1d1d1f;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1150px;margin:auto;padding:48px 24px}}h1{{font-size:36px;line-height:1.2;letter-spacing:-.04em}}h2{{font-size:23px;margin-top:36px;letter-spacing:-.025em}}
p{{max-width:950px}}.muted{{color:#62666e}}.source,aside{{padding:16px 20px;background:#fff;border:1px solid #e6e7eb;border-radius:12px}}
.chart{{background:#fff;border:1px solid #e6e7eb;border-radius:16px;margin:24px 0;overflow:hidden}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border-bottom:1px solid #e6e7eb;padding:10px 12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.metadata th{{width:30%;text-align:left}}.metadata td{{text-align:left;overflow-wrap:anywhere}}footer{{margin-top:32px;color:#62666e}}pre{{white-space:pre-wrap}}summary{{cursor:pointer;padding:12px 0}}
button{{padding:10px 16px;cursor:pointer;background:#1764c0;color:white;border:0;border-radius:10px;font:inherit}}@media print{{
@page{{size:A4 landscape;margin:12mm}}:root{{color-scheme:light}}body{{background:white;color:#172337;font-size:10pt}}
main{{max-width:none;padding:0}}h1{{font-size:25pt}}h2{{break-after:avoid}}.source,aside{{background:#eef4f7}}
.muted,footer{{color:#35445a}}.chart{{break-inside:avoid;background:white}}.scroll{{overflow:visible}}
table{{font-size:8pt}}th,td{{padding:5px}}tr{{break-inside:avoid}}thead{{display:table-header-group}}
button,.modebar{{display:none!important}}details{{display:none}}.metadata{{font-size:8pt}}
}}
@media screen and (max-width:600px){{main{{padding:24px 12px}}h1{{font-size:30px}}h2{{font-size:21px}}
.source,aside{{padding:12px}}th,td{{padding:8px}}}}
</style><script>{get_plotlyjs()}</script></head><body><main>
<p class="muted">PORTFOLIO RESEARCH / REPRODUCIBLE ANALYSIS</p><h1>Efficient Frontier</h1>
<button onclick="window.print()">Print / save PDF</button>
<p class="source"><strong>{source_label}:</strong> {html.escape(source)}</p>
{comparison_notes}
{profiles_html}
<p>{introduction}
Training estimates describe historical data. They are not predictions or guarantees of future performance.</p>
{warnings_html}
{comparison_html}
<h2>Training estimates</h2><p class="muted">Arithmetic mean returns and covariance use {analysis.periods_per_year:g} observations per year.
The analysis applies covariance shrinkage to training data only. Sharpe ratios use the configured risk-free rate.</p>
<div class="chart">{frontier}</div>
<div class="scroll">{_table(estimates, ["Historical expected annual return", "Annual volatility"])}</div>
<h2>Portfolio allocations</h2><p class="muted">These are the initial weights at the training/holdout split.</p>
<div class="scroll">{_table(weights_frame(analysis))}</div>
<h2>Concentration and risk</h2><p class="muted">These measures use the initial weights and the estimated training covariance.
Effective holdings equals one divided by the sum of squared weights.
The diversification ratio divides weighted asset volatility by portfolio volatility.</p>
<div class="scroll">{_table(risk, ["Largest weight"])}</div>
<h3>Share of portfolio variance</h3><p class="muted">Each asset contributes this share of total portfolio variance.
The shares sum to 100% when portfolio variance is positive. Negative shares identify assets that reduce total variance through covariance.
An em dash marks undefined risk shares and diversification ratios when portfolio variance is zero.</p>
<div class="scroll">{_table(risk_contributions(analysis))}</div>
<h2>Realized holdout results</h2><p class="muted">Each portfolio starts with 10,000 units, allocates once at the split,
and then holds adjusted-price exposures. There is no subsequent trading between assets; portfolio weights drift with prices.
Distributions follow the provider's price adjustments rather than accumulating as separate cash.
CAGR and volatility use {analysis.periods_per_year:g} observations per year.</p>
<p class="muted">Sortino measures excess return relative to downside deviation. Calmar divides CAGR by the absolute maximum drawdown.
An em dash marks a ratio with a zero denominator.</p><div class="chart">{holdout_plot}</div>
<div class="scroll">{_table(holdout, ["Total return", "CAGR", "Annual volatility", "Maximum drawdown"])}</div>
<h2>Data and assumptions</h2><div class="scroll"><table class="metadata">{metadata_rows}</table></div>{coverage_html}
<footer>{cost_note} Taxes and currency conversion (FX) are excluded. Price series must share a consistent currency basis.
This report is a historical research tool and does not predict investment outcomes. Charts work offline.
<details><summary>Third-party notice: Plotly.js (MIT)</summary><pre>{html.escape(PLOTLY_JS_LICENSE)}</pre></details></footer>
</main></body></html>'''


def report_zip(analysis: Any, prices: pd.DataFrame, metadata: dict, study=None, latest_profiles=None, benchmarks=None, evidence=None) -> bytes:
    """Bundle the report, inputs, and numeric results for offline inspection."""
    output = io.BytesIO()
    frames = {
        "weights.csv": weights_frame(analysis), "frontier.csv": analysis.frontier,
        "frontier_weights.csv": analysis.frontier_weights, "holdout_metrics.csv": analysis.holdout_metrics,
        "holdout_curve.csv": analysis.equity, "prices.csv": prices,
        "risk_summary.csv": risk_summary(analysis), "risk_contributions.csv": risk_contributions(analysis),
    }
    if metadata.get("universe_coverage"):
        frames["universe_coverage.csv"] = pd.DataFrame(metadata["universe_coverage"])
    if study is not None:
        frames.update({"backtest_metrics.csv": study.metrics, "backtest_curve.csv": study.equity})
        for i, name in enumerate(study.equity.columns, 1):
            frames[f"backtests/{i:02d}_holdings.csv"] = study.holdings[name]
            frames[f"backtests/{i:02d}_allocations.csv"] = study.allocations[name]
            frames[f"backtests/{i:02d}_trades.csv"] = study.trades[name]
    if latest_profiles is not None:
        frames.update({
            "latest_profile_summary.csv": latest_profiles.summary,
            "latest_profile_weights.csv": weights_frame(latest_profiles),
            "latest_profile_windows.csv": latest_profiles.windows,
            "latest_profile_window_returns.csv": latest_profiles.window_returns,
            "latest_profile_frontier.csv": latest_profiles.frontier,
            "latest_profile_frontier_weights.csv": latest_profiles.frontier_weights,
        })
    if benchmarks is not None:
        frames.update({
            "benchmark_prices.csv": benchmarks.prices,
            "benchmark_holdout_metrics.csv": benchmarks.holdout_metrics,
            "benchmark_holdout_curve.csv": benchmarks.holdout_equity,
            "benchmark_training_estimates.csv": benchmarks.training_estimates,
            "benchmark_latest_estimates.csv": benchmarks.latest_estimates,
        })
        if benchmarks.backtest_equity is not None:
            frames.update({"benchmark_backtest_metrics.csv": benchmarks.backtest_metrics,
                           "benchmark_backtest_curve.csv": benchmarks.backtest_equity})
    if evidence is not None:
        frames.update({"evidence_summary.csv": evidence.summary, "evidence_windows.csv": evidence.windows,
                       "evidence_relative_curve.csv": evidence.relative_equity})
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("report.html", report_html(analysis, metadata, study, latest_profiles, benchmarks, evidence))
        if study is not None:
            bundle.writestr("findings.md", findings_markdown(study, str(metadata.get("source", "Unspecified source")), benchmarks, evidence))
        bundle.writestr("THIRD_PARTY_NOTICES.txt", "Plotly.js (embedded in report.html)\n\n" + PLOTLY_JS_LICENSE)
        for name, frame in frames.items():
            bundle.writestr(name, csv_text(frame, index_label="Date" if name in ("prices.csv", "benchmark_prices.csv") else None))
        bundle.writestr("metadata.json", json.dumps(_metadata(analysis, metadata, study, latest_profiles, benchmarks, evidence), indent=2, default=str))
    return output.getvalue()
