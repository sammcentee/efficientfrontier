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

from . import story
from .backtest_report import backtest_html, findings_markdown
from .benchmark_report import add_benchmark_estimates, add_benchmark_paths, benchmark_html
from .risk import risk_contributions, risk_summary
from .style import COLORS, INK, INK_2, PLOTLY_CONFIG, REPORT_CSS, REST, display_label, style_chart


PLOTLY_JS_LICENSE = (Path(__file__).parent / "third_party" / "plotly.js.LICENSE.txt").read_text(encoding="utf-8")


def _point(name, x, y, hover, selected=False) -> go.Scatter:
    """A labelled portfolio point. The selected point is larger with an ink ring."""
    return go.Scatter(
        x=[x], y=[y], mode="markers+text", name=html.escape(display_label(name)), text=[html.escape(display_label(name))],
        textposition="top center", textfont={"size": 12, "color": INK_2}, cliponaxis=False,
        marker={"size": 18 if selected else 13, "color": COLORS.get(name, REST),
                "line": {"width": 2.5, "color": INK} if selected else {"width": 2, "color": "#ffffff"}},
        hovertemplate=hover,
    )


def frontier_chart(analysis: Any, benchmark_estimates=None) -> go.Figure:
    """Show annualized historical training estimates, not future returns."""
    hover = "Annual volatility: %{x:.2%}<br>Historical expected return: %{y:.2%}<extra>%{fullData.name}</extra>"
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=analysis.frontier["volatility"], y=analysis.frontier["expected_return"],
        mode="lines", name="Efficient frontier", line={"color": REST, "width": 2.5}, hovertemplate=hover,
    ))
    for name, portfolio in analysis.portfolios.items():
        figure.add_trace(_point(name, portfolio.volatility, portfolio.expected_return, hover))
    add_benchmark_estimates(figure, benchmark_estimates, "expected_return")
    style_chart(figure, "Classic frontier")
    figure.update_layout(legend_y=-0.24)
    figure.update_xaxes(title="Volatility, yearly", tickformat=".0%", rangemode="tozero")
    figure.update_yaxes(title="Average return, yearly (first fit)", tickformat=".0%")
    return figure


def holdout_chart(analysis: Any, benchmark_equity=None) -> go.Figure:
    """Plot the realized buy-and-hold value of an initial 10,000 units."""
    figure = go.Figure()
    for name in analysis.equity.columns:
        figure.add_trace(go.Scatter(
            x=analysis.equity.index, y=analysis.equity[name] * 10_000, name=name,
            mode="lines", line={"width": 2, "color": COLORS.get(name, REST)},
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio value: %{y:,.0f}<extra>%{fullData.name}</extra>",
        ))
    add_benchmark_paths(figure, benchmark_equity)
    style_chart(figure, "Classic holdout", height=380, hovermode="x unified")
    figure.update_yaxes(tickformat="~s")
    return figure


def latest_profile_chart(profiles: Any, benchmark_estimates=None, selected=None) -> go.Figure:
    """Plot the lowest historical window mean against estimated volatility. The selected level is ringed."""
    hover = "Annual volatility: %{x:.2%}<br>Lowest annual window mean: %{y:.2%}<extra>%{fullData.name}</extra>"
    figure = go.Figure(go.Scatter(
        x=profiles.frontier["volatility"], y=profiles.frontier["worst_window_return"],
        mode="lines", name="Window frontier", line={"color": REST, "width": 2.5}, hovertemplate=hover,
    ))
    for name, row in profiles.summary.iterrows():
        figure.add_trace(_point(name, row["volatility"], row["worst_window_return"], hover, selected=name == selected))
    add_benchmark_estimates(figure, benchmark_estimates, "worst_window_return")
    style_chart(figure, "Risk frontier")
    figure.update_layout(legend_y=-0.24)
    figure.update_xaxes(title="Volatility, yearly", tickformat=".0%", rangemode="tozero")
    figure.update_yaxes(title="Weakest past stretch, yearly", tickformat=".0%")
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


def _latest_profiles_html(profiles: Any, benchmark_estimates=None, selected: str = "Medium") -> str:
    summary = profiles.summary.rename(index=display_label, columns={
        "expected_return": "Annual mean return", "volatility": "Annual volatility",
        "sharpe": "Estimated Sharpe", "worst_window_return": "Lowest annual window mean",
        "max_weight": "Largest weight", "effective_holdings": "Effective holdings",
    })
    windows = profiles.windows.rename(columns={
        "start": "Start", "end": "End", "observations": "Observations", "years": "Years of observations",
    }).rename_axis("Window")
    window_returns = profiles.window_returns.rename(columns=display_label).rename_axis("Window")
    weights = weights_frame(profiles)
    weights = weights.loc[(weights >= 0.00005).any(axis=1)].sort_values(selected, ascending=False, kind="stable")
    warnings = "".join(f"<li>{html.escape(str(note))}</li>" for note in profiles.warnings)
    chart = latest_profile_chart(profiles, benchmark_estimates, selected=selected).to_html(
        full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
    )
    as_of = pd.Timestamp(profiles.as_of)
    return (
        '<section id="latest-profiles">'
        f'<h2>Latest model holdings as of {as_of.day} {as_of:%b %Y}</h2>'
        '<p class="source">These model weights use all supplied history through the last input date. '
        'The date above is the last price observation, not a live quote.</p>'
        '<p>Low, Medium, and Highest are relative risk levels within this frontier. They are not universal risk ratings. '
        'The three historical windows are part of the fit. Their means are in-sample estimates, not forecasts or holdout results.</p>'
        + (f'<aside><h3>Profile notes</h3><ul>{warnings}</ul></aside>' if warnings else '')
        + f'<h3>Latest weights</h3><p class="muted">Sorted by {html.escape(display_label(selected))}. '
        'Rows under 0.005% in all three levels are hidden here. latest_profile_weights.csv has every row.</p>'
        f'<div class="scroll">{_table(weights.rename(columns=display_label))}</div>'
        '<h3>Profile estimates</h3>'
        f'<div class="scroll">{_table(summary, ["Annual mean return", "Annual volatility", "Lowest annual window mean", "Largest weight"])}</div>'
        '<p class="muted">The model minimizes estimated variance at a return target that applies to every historical window. '
        'Low uses the minimum-variance point. Highest uses the highest feasible target for the lowest window mean. '
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


def _summary_html(analysis: Any, metadata: dict, prices, study, latest_profiles, benchmarks, evidence,
                  profile: str, method: str) -> str:
    """Answers first: what the selected level holds, how its rule did against the markets and how sure we can be."""
    universe = str(metadata.get("universe", ""))
    names = story.company_names(metadata.get("universe_coverage"))
    parts = ['<section id="summary">']
    if prices is not None:
        parts.append(story.context_result_html(metadata, prices))
    if latest_profiles is not None:
        weights = latest_profiles.portfolios[profile].weights
        parts.append(f"<h2>{story.headline_html(weights, profile, universe)}</h2>")
    else:
        parts.append("<h2>There is not enough history to set risk levels.</h2>")
    items = story.fine_print_items(metadata, prices if prices is not None else analysis.train_returns,
                                   coverage_place="Universe coverage in Data and assumptions")
    parts.append('<aside class="pl-read"><h3>Read this first</h3>' + "".join(f"<p>{item}</p>" for item in items) + "</aside>")
    if latest_profiles is not None:
        parts.append(f'<div class="tile">{story.stats_html(latest_profiles, profile, universe, names)}'
                     f'{story.list_html(weights, names)}</div>')
        parts.append('<div class="tile"><div class="pl-head"><h3 class="pl-title">How the three levels compare</h3></div>'
                     f'{story.levels_ledger_html(latest_profiles, profile, names, universe)}</div>')
    if study is not None:
        selected = f"{method} · {profile}"
        if selected not in study.equity:
            selected = f"{method} · Minimum volatility"
        markets = benchmarks is not None and benchmarks.backtest_equity is not None
        if evidence is not None and selected not in evidence.summary.index.get_level_values("strategy"):
            evidence = None
        chart = story.story_chart(study, selected, "Growth", benchmarks, evidence).to_html(
            full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
        months = story.months_tested(study)
        if markets:
            answer, lead = story.evidence_answer(evidence, selected), story.evidence_lead(evidence, selected, months, metadata)
        else:
            answer = "There is no market to compare with."
            lead = "The statistics compare the rule with the S&P 500 and the Nasdaq-100. This study has no market prices."
        parts += [
            '<p class="pl-eyebrow">Would this rule have beaten the market?</p>',
            f'<h3 class="pl-answer">{html.escape(story.market_answer(study, selected, benchmarks))}</h3>',
            f'<p class="pl-lead">{html.escape(story.market_lead(study, selected))}</p>',
            f'<div class="tile">{story.result_tile_html(study, selected, benchmarks, evidence, metadata)}</div>',
            f'<div class="tile"><h3 class="pl-title">Growth of 10,000</h3><div class="chart">{chart}</div>'
            f'<p class="pl-small">{html.escape(story.chart_footnote(study, benchmarks))}</p></div>',
            '<p class="pl-eyebrow">How sure can we be?</p>',
            f'<h3 class="pl-answer">{html.escape(answer)}</h3>',
            f'<p class="pl-lead">{html.escape(lead)}</p>' if lead else "",
        ]
        if markets and evidence is not None:
            parts.append(
                '<div class="pl-cols"><div class="tile"><h3 class="pl-title">Average yearly gap, with 95% range</h3>'
                '<p class="pl-sub">Your rule minus each market, in percentage points. This is an arithmetic average, '
                f'so it differs from the growth gap above.</p>{story.interval_html(evidence, selected)}</div>'
                f'<div class="tile"><h3 class="pl-title">Stretch by stretch</h3>{story.stretch_summary_html(evidence, selected)}'
                '<p class="pl-sub">The test period, split into three equal stretches. These differ from the three stretches '
                f'that fitted today’s holdings.</p>{story.stretch_table_html(evidence, selected)}</div></div>'
                f'<p class="pl-small">These tests describe the last {months} months. They cannot tell you how the rule will do next.</p>')
    return "".join(parts) + "</section>"


def report_html(analysis: Any, metadata: dict, study=None, latest_profiles=None, benchmarks=None, evidence=None, *,
                selected_profile=None, selected_method=None, prices=None) -> str:
    """Return an offline HTML report with a single embedded Plotly bundle.

    The summary shows the selected risk level (default Medium) and rule (default Expanding window) first.
    Prices are optional and only supply the dates and counts in the summary context line.
    """
    profile, method = selected_profile or "Medium", selected_method or "Expanding window"
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
    frontier = frontier_chart(analysis, benchmarks.training_estimates if benchmarks is not None else None).to_html(
        full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
    )
    holdout_plot = holdout_chart(analysis, benchmarks.holdout_equity if benchmarks is not None else None).to_html(
        full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
    )
    focus = None
    if study is not None:
        focus = f"{method} · {profile}" if f"{method} · {profile}" in study.equity else f"{method} · Minimum volatility"
    comparison_html = backtest_html(study, benchmarks, evidence, focus=focus) if study is not None else (
        benchmark_html(benchmarks, evidence) if benchmarks is not None else ""
    )
    profiles_html = _latest_profiles_html(
        latest_profiles, benchmarks.latest_estimates if benchmarks is not None else None, profile,
    ) if latest_profiles is not None else ""
    summary_html = _summary_html(analysis, metadata, prices if prices is not None and not prices.empty else None, study,
                                 latest_profiles, benchmarks, evidence, profile, method)
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
<title>Portfolio Lab · Research report</title>
<style>{REPORT_CSS}</style><script>{get_plotlyjs()}</script></head><body><main>
<p class="eyebrow">Historical research, not a forecast</p><h1>Portfolio Lab · Research report</h1>
<button onclick="window.print()">Print / save PDF</button>
<p class="source"><strong>{source_label}:</strong> {html.escape(source)}</p>
{comparison_notes}
{summary_html}
<hr><p class="eyebrow">Appendix: every result</p>
{profiles_html}
<section id="classic-model"><p>{introduction}
Training estimates describe historical data. They are not predictions or guarantees of future performance.</p>
{warnings_html}</section>
{comparison_html}
<section id="training"><h2>Training estimates</h2><p class="muted">Arithmetic mean returns and covariance use {analysis.periods_per_year:g} observations per year.
The analysis applies covariance shrinkage to training data only. Sharpe ratios use the configured risk-free rate.</p>
<div class="chart">{frontier}</div>
<div class="scroll">{_table(estimates, ["Historical expected annual return", "Annual volatility"])}</div></section>
<section id="allocations"><h2>Portfolio allocations</h2><p class="muted">These are the initial weights at the training/holdout split.</p>
<div class="scroll">{_table(weights_frame(analysis))}</div></section>
<section id="risk"><h2>Concentration and risk</h2><p class="muted">These measures use the initial weights and the estimated training covariance.
Effective holdings equals one divided by the sum of squared weights.
The diversification ratio divides weighted asset volatility by portfolio volatility.</p>
<div class="scroll">{_table(risk, ["Largest weight"])}</div>
<h3>Share of portfolio variance</h3><p class="muted">Each asset contributes this share of total portfolio variance.
The shares sum to 100% when portfolio variance is positive. Negative shares identify assets that reduce total variance through covariance.
An em dash marks undefined risk shares and diversification ratios when portfolio variance is zero.</p>
<div class="scroll">{_table(risk_contributions(analysis))}</div></section>
<section id="holdout"><h2>Realized holdout results</h2><p class="muted">Each portfolio starts with 10,000 units, allocates once at the split,
and then holds adjusted-price exposures. There is no subsequent trading between assets; portfolio weights drift with prices.
Distributions follow the provider's price adjustments rather than accumulating as separate cash.
CAGR and volatility use {analysis.periods_per_year:g} observations per year.</p>
<p class="muted">Sortino measures excess return relative to downside deviation. Calmar divides CAGR by the absolute maximum drawdown.
An em dash marks a ratio with a zero denominator.</p><div class="chart">{holdout_plot}</div>
<div class="scroll">{_table(holdout, ["Total return", "CAGR", "Annual volatility", "Maximum drawdown"])}</div></section>
<section id="data"><h2>Data and assumptions</h2><p class="muted">Files call the Highest level “Extreme”.</p>
<div class="scroll"><table class="metadata">{metadata_rows}</table></div>{coverage_html}</section>
<footer>{cost_note} Taxes and currency conversion (FX) are excluded. Price series must share a consistent currency basis.
This report is a historical research tool and does not predict investment outcomes. Charts work offline.
<details><summary>Third-party notice: Plotly.js (MIT)</summary><pre>{html.escape(PLOTLY_JS_LICENSE)}</pre></details></footer>
</main></body></html>'''


def report_zip(analysis: Any, prices: pd.DataFrame, metadata: dict, study=None, latest_profiles=None, benchmarks=None, evidence=None,
               *, selected_profile=None, selected_method=None) -> bytes:
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
        bundle.writestr("report.html", report_html(analysis, metadata, study, latest_profiles, benchmarks, evidence,
                                                   selected_profile=selected_profile, selected_method=selected_method,
                                                   prices=prices))
        if study is not None:
            bundle.writestr("findings.md", findings_markdown(study, str(metadata.get("source", "Unspecified source")), benchmarks, evidence))
        bundle.writestr("THIRD_PARTY_NOTICES.txt", "Plotly.js (embedded in report.html)\n\n" + PLOTLY_JS_LICENSE)
        for name, frame in frames.items():
            bundle.writestr(name, csv_text(frame, index_label="Date" if name in ("prices.csv", "benchmark_prices.csv") else None))
        details = _metadata(analysis, metadata, study, latest_profiles, benchmarks, evidence)
        if selected_profile is not None or selected_method is not None:
            details["report_view"] = {"risk_level": selected_profile or "Medium", "rule": selected_method or "Expanding window"}
        bundle.writestr("metadata.json", json.dumps(details, indent=2, default=str))
    return output.getvalue()
