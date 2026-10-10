"""Benchmark comparisons and uncertainty in historical portfolio results."""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .style import BENCHMARK_COLORS, style_chart

EVIDENCE_LABELS = {
    "total_return_difference": "Total return difference", "cagr_difference": "CAGR difference",
    "annual_advantage": "Annual mean advantage", "advantage_ci_low": "Mean CI lower", "advantage_ci_high": "Mean CI upper",
    "advantage_pvalue": "Mean p-value", "advantage_pvalue_adjusted": "Mean Holm p-value",
    "beta": "Benchmark beta", "alpha": "Annual alpha", "alpha_ci_low": "Alpha CI lower", "alpha_ci_high": "Alpha CI upper",
    "alpha_pvalue": "Alpha p-value", "alpha_pvalue_adjusted": "Alpha Holm p-value",
    "information_ratio": "Information ratio", "tracking_error": "Annual tracking error",
    "windows_ahead": "Windows ahead", "observations": "Inference observations", "status": "Historical evidence",
    "advantage_note": "Mean inference note", "alpha_note": "Alpha inference note",
}


def add_benchmark_paths(figure: go.Figure, equity: pd.DataFrame | None, drawdown: bool = False) -> None:
    if equity is None:
        return
    for name, path in equity.items():
        values = path / path.cummax() - 1 if drawdown else path * 10_000
        figure.add_trace(go.Scatter(
            x=equity.index, y=values, name=html.escape(str(name)), mode="lines",
            line={"color": BENCHMARK_COLORS.get(name, "#737b85"), "width": 2, "dash": "longdash"},
            hovertemplate="%{x|%Y-%m-%d}<br>" + ("Drawdown: %{y:.2%}" if drawdown else "Benchmark value: %{y:,.2f}")
            + "<extra>%{fullData.name}</extra>",
        ))


def add_benchmark_estimates(figure: go.Figure, estimates: pd.DataFrame | None, objective: str) -> None:
    if estimates is None:
        return
    for name, row in estimates.iterrows():
        figure.add_trace(go.Scatter(
            x=[row["volatility"]], y=[row[objective]], name=html.escape(str(name)), mode="markers",
            marker={"symbol": "diamond", "size": 11, "color": BENCHMARK_COLORS.get(name, "#737b85")},
            hovertemplate="Annual volatility: %{x:.2%}<br>"
            + ("Lowest annual window mean" if objective == "worst_window_return" else "Historical expected annual return")
            + ": %{y:.2%}<extra>%{fullData.name}</extra>",
        ))


def default_evidence_strategy(evidence: Any) -> str | None:
    """Choose a fixed default without reference to realized performance."""
    names = list(evidence.summary.index.get_level_values("strategy").unique())
    for name in ("Expanding window · Medium", "Minimum volatility", "Expanding window · Minimum volatility"):
        if name in names:
            return name
    return names[0] if names else None


def evidence_chart(evidence: Any, strategy: str) -> go.Figure:
    """Plot portfolio wealth divided by benchmark wealth, with a baseline of one."""
    if strategy not in evidence.relative_equity.columns.get_level_values("strategy"):
        raise ValueError(f"Unknown evidence strategy: {strategy}")
    paths = evidence.relative_equity.xs(strategy, axis=1, level="strategy")
    figure = go.Figure()
    for benchmark, path in paths.items():
        figure.add_trace(go.Scatter(
            x=paths.index, y=path, name=html.escape(str(benchmark)), mode="lines",
            line={"color": BENCHMARK_COLORS.get(benchmark, "#737b85"), "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>Relative wealth: %{y:.3f}×<extra>%{fullData.name}</extra>",
        ))
    figure.add_hline(y=1, line_dash="dot", line_color="#b7bcc4")
    style_chart(figure, "Relative performance", hovermode="x unified")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title="Portfolio / benchmark", tickformat=".2f")
    return figure


def evidence_summary_frame(evidence: Any) -> pd.DataFrame:
    """Format estimates and p-values without changes to the numeric evidence."""
    frame = evidence.summary.copy()
    for column in frame:
        def formatted(value):
            if pd.isna(value):
                return "—"
            if column in ("status", "advantage_note", "alpha_note"):
                return str(value)
            if "pvalue" in column:
                return f"{value:.4g}"
            if column == "windows_ahead":
                return f"{int(value)}/{evidence.settings.get('window_count', 3)}"
            if column == "observations":
                return str(int(value))
            if column in ("beta", "information_ratio"):
                return f"{value:.3f}"
            if column in ("total_return_difference", "cagr_difference", "annual_advantage", "advantage_ci_low", "advantage_ci_high"):
                return f"{value * 100:+.2f} pp"
            return f"{value:.2%}"
        frame[column] = frame[column].map(formatted)
    return frame.rename(columns=EVIDENCE_LABELS)


def _evidence_notes(evidence: Any) -> list[str]:
    settings = evidence.settings
    notes = [
        "Relative wealth divides portfolio value by benchmark value after both start at one. A value above one means the portfolio is ahead.",
        "Annual mean advantage is the arithmetic mean return difference times observations per year. It is not the CAGR difference.",
        f"Inference requires at least {settings.get('inference_min_observations', 60)} return observations. "
        f"The {settings.get('confidence_level', .95):.0%} HAC confidence intervals are pointwise, not simultaneous bands.",
        f"HAC uses {settings.get('hac_lags', 0)} lags for serial correlation and unequal return variance. "
        f"Holm correction covers {settings.get('multiple_testing_tests', 0)} tests across all strategy/benchmark pairs and both mean-advantage and alpha hypotheses.",
        f"Inference uses {settings.get('inference_observations', 0)} paired returns from {settings.get('inference_start')} to {settings.get('inference_end')}.",
        "The p-values do not measure the probability of future outperformance. These estimates describe the observed sample and use large-sample approximations.",
        "Alpha uses a separate excess-return regression for each benchmark. Beta measures only exposure to that benchmark. Positive alpha does not establish investment skill.",
        f"The {settings.get('window_count', 3)} chronological windows describe consistency within the evaluated sample. "
        "They are not independent replications of the strategy.",
    ]
    if settings.get("delayed_entry", False):
        notes.append("Inference omits the first evaluation session and its entry fee. Total returns, relative wealth, and window results retain entry costs.")
    return notes


def _window_frame(windows: pd.DataFrame) -> pd.DataFrame:
    display = windows.copy()
    display["relative_return"] = display["relative_return"].map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
    display["annual_advantage"] = display["annual_advantage"].map(lambda value: "—" if pd.isna(value) else f"{value * 100:+.2f} pp")
    return display.rename(columns={"strategy": "Strategy", "benchmark": "Benchmark", "window": "Window", "start": "Start",
                                   "end": "End", "observations": "Observations", "relative_return": "Relative wealth change",
                                   "annual_advantage": "Annual mean advantage"})


def _benchmark_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    labels = {"total_return": "Total return", "cagr": "CAGR", "volatility": "Annual volatility", "sharpe": "Sharpe",
              "sortino": "Sortino", "calmar": "Calmar", "max_drawdown": "Maximum drawdown", "total_cost": "Fees / initial capital"}
    result = frame[[column for column in labels if column in frame]].copy()
    for column in result:
        result[column] = result[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}" if column in ("sharpe", "sortino", "calmar") else f"{value:.2%}"
        )
    return result.rename(columns=labels)


def _comparison_parts(benchmarks: Any, backtest: bool) -> tuple[pd.DataFrame, list[str]]:
    equity = benchmarks.backtest_equity if backtest else benchmarks.holdout_equity
    metrics = benchmarks.backtest_metrics if backtest else benchmarks.holdout_metrics
    notes = [
        f"Evaluation baseline: {equity.index[0].date()}. Last observation: {equity.index[-1].date()}. "
        "Benchmark paths use the same dates as the portfolio paths.",
        "SPY and QQQ are USD ETF proxies for the S&P 500 and Nasdaq-100. They are not index series.",
        "Benchmark markers use the same dates and return objective as each frontier chart. The independent ETF references do not follow portfolio weight limits.",
        ("Backtest benchmarks stay in cash for the first interval, pay the same entry fee rate as the portfolios, then hold. No terminal sale occurs."
         if backtest else "The original holdout and its benchmarks allocate at the initial close. Both exclude fees."),
    ]
    return _benchmark_metrics(metrics), notes


def benchmark_html(benchmarks: Any, evidence=None, backtest: bool = False) -> str:
    """Return report sections without another Plotly bundle."""
    metrics, notes = _comparison_parts(benchmarks, backtest)
    document = '<section id="benchmarks"><h2>Market benchmarks</h2>'
    document += "".join(f"<p>{html.escape(note)}</p>" for note in notes)
    document += f'<div class="scroll">{metrics.to_html(escape=True, border=0, classes="data")}</div>'
    if evidence is not None:
        strategy = default_evidence_strategy(evidence)
        summary = evidence_summary_frame(evidence)
        document += '<h2>Evidence against benchmarks</h2>'
        document += "".join(f'<p class="muted">{html.escape(note)}</p>' for note in _evidence_notes(evidence))
        if strategy is not None:
            document += f'<h3>{html.escape(str(strategy))}</h3><p>This default does not depend on realized returns.</p>'
            document += f'<div class="scroll">{summary.xs(strategy, level="strategy").to_html(escape=True, border=0, classes="data")}</div>'
            chart = evidence_chart(evidence, strategy).to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})
            document += f'<div class="chart">{chart}</div>'
            windows = _window_frame(evidence.windows.loc[evidence.windows.strategy == strategy].drop(columns="strategy"))
            document += '<h3>Results in each window</h3>'
            document += f'<div class="scroll">{windows.to_html(index=False, escape=True, border=0, classes="data")}</div>'
        document += '<details><summary>All strategy comparisons</summary>'
        document += f'<div class="scroll">{summary.to_html(escape=True, border=0, classes="data")}</div></details>'
        windows = _window_frame(evidence.windows)
        document += '<details><summary>Historical windows for every comparison</summary>'
        document += f'<div class="scroll">{windows.to_html(index=False, escape=True, border=0, classes="data")}</div></details>'
        if evidence.warnings:
            document += '<aside><h3>Evidence notes</h3><ul>' + "".join(f'<li>{html.escape(str(note))}</li>' for note in evidence.warnings) + '</ul></aside>'
    return document + '</section>'


def _markdown(value: Any) -> str:
    text = html.escape(str(value)).replace("\\", "\\\\").replace("\n", " ").replace("\r", " ")
    return re.sub(r"([|`*_\[\]])", r"\\\1", text)


def _markdown_table(frame: pd.DataFrame) -> str:
    frame = frame.reset_index()
    lines = ["| " + " | ".join(_markdown(column) for column in frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(_markdown(cell) for cell in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def benchmark_markdown(benchmarks: Any, evidence=None, backtest: bool = False) -> str:
    metrics, notes = _comparison_parts(benchmarks, backtest)
    document = "## Market benchmarks\n\n" + "\n\n".join(_markdown(note) for note in notes) + "\n\n" + _markdown_table(metrics) + "\n"
    if evidence is not None:
        document += "\n## Evidence against benchmarks\n\n" + "\n\n".join(_markdown(note) for note in _evidence_notes(evidence)) + "\n\n"
        strategy = default_evidence_strategy(evidence)
        if strategy is not None:
            document += f"Default comparison: {_markdown(strategy)}. This choice does not depend on realized returns.\n\n"
        document += _markdown_table(evidence_summary_frame(evidence)) + "\n"
        document += "\n### Historical windows\n\n" + _markdown_table(_window_frame(evidence.windows).set_index(["Strategy", "Benchmark", "Window"])) + "\n"
        if evidence.warnings:
            document += "\n" + "\n".join(f"- {_markdown(note)}" for note in evidence.warnings) + "\n"
    return document
