"""Descriptive findings and portable views of a backtesting study."""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .benchmark_report import add_benchmark_paths, benchmark_html, benchmark_markdown
from .style import ACCENT, COLORS, GRID, INK_2, PLOTLY_CONFIG, REST, display_label, style_chart


METRICS = {
    "total_return": ("Net total return", "percent"),
    "cagr": ("CAGR", "percent"),
    "volatility": ("Annual volatility", "percent"),
    "sharpe": ("Sharpe", "number"),
    "sortino": ("Sortino", "number"),
    "calmar": ("Calmar", "number"),
    "max_drawdown": ("Maximum drawdown", "percent"),
    "total_turnover": ("Gross turnover", "multiple"),
    "total_cost": ("Fees / initial capital", "percent"),
    "rebalance_count": ("Allocation events (incl. entry)", "count"),
    "fallback_count": ("Sharpe fallbacks", "count"),
}
HOLDINGS = {
    "pnl_contribution": ("P&L / initial capital", "percent"),
    "average_weight": ("Average weight", "percent"),
    "end_weight": ("End weight", "percent"),
    "selection_frequency": ("Selection frequency", "percent"),
    "max_target_weight": ("Maximum target weight", "percent"),
    "max_realized_weight": ("Maximum realized weight", "percent"),
}
BEST_IN_HINDSIGHT = "Best in hindsight. These results were picked after the test. They are not a recommendation."
ASSUMPTIONS = (
    "Buy and hold allocates once. Fixed rebalance restores the original fitted targets at "
    "each scheduled trade. Expanding window refits using all prior observations; rolling "
    "window refits using only the most recent configured number of returns. Rebalance "
    "intervals count observed sessions, not calendar months. Fixed-rebalance, expanding-window "
    "and rolling-window equal-weight baselines coincide because they use the same targets and schedule.",
    "All methods use the same evaluation dates and fixed input universe. "
    "Targets use information through the prior close and execute at the following close. "
    "The first evaluation session is held in cash earning zero interest. Weights drift "
    "between trades; no execution-day return is earned using the new target.",
    "Trading fees are deducted from portfolio wealth. Gross turnover counts both purchases "
    "and sales, each divided by portfolio value before that trade and summed across events. "
    "Entry is included and terminal liquidation is excluded. Fees are shown as a share of "
    "initial capital. Asset P&L contributions, less "
    "fees, reconcile to the net total return. The fee total excludes foregone compounding, so it "
    "does not measure the full performance difference from a zero-fee run. Taxes, FX conversion and liquidity constraints "
    "are not modeled.",
    "Average weight uses exposure at the start of each evaluation session, including the "
    "initial cash session. Selection frequency is the share of allocation events with an "
    "asset target above 0.0001%; it is not the fraction of observations held.",
    "Prices must be adjusted and share a common currency basis. "
    "A fixed universe can introduce survivorship bias.",
    "These findings describe the evaluated sample. The highest-return strategy is selected "
    "after evaluation; it is not a forecast or a recommendation to hold its assets. Repeated "
    "strategy selection using these results can overfit the evaluation period.",
)


def _assumptions(study: Any) -> tuple[str, ...]:
    periods = study.settings.get("periods_per_year", 252)
    profile_notes = (
        "Low, Medium, and Highest use a frontier with a return target in each of three historical windows. "
        "These are relative risk levels within each fit, not universal risk ratings. "
        "Profile backtests fit only the history available before each allocation. "
        "The latest model holdings use all supplied history and remain separate from these backtests and the original holdout.",
    ) if study.settings.get("include_profiles", False) else ()
    return ASSUMPTIONS + profile_notes + (
        f"CAGR and volatility use {periods:g} observations per year. "
        "Sortino measures excess return relative to downside deviation. "
        "Calmar divides CAGR by the absolute maximum drawdown. "
        "An em dash marks a ratio with a zero denominator.",
    )


def _formatted(frame: pd.DataFrame, columns: dict) -> pd.DataFrame:
    display = pd.DataFrame(index=frame.index)
    for column, (label, kind) in columns.items():
        def format_value(value):
            if pd.isna(value):
                return "—"
            if kind == "percent":
                return f"{value:.2%}"
            if kind == "count":
                return str(int(value))
            return f"{value:.2f}" + ("×" if kind == "multiple" else "")

        display[label] = frame[column].map(format_value)
    return display


def _top_holdings(study: Any, strategy: str) -> pd.DataFrame:
    holdings = study.holdings[strategy]
    holdings = holdings.loc[holdings["max_target_weight"] > 1e-6]
    order = holdings["pnl_contribution"].abs().sort_values(ascending=False, kind="stable").index
    return holdings.loc[order[:10]]


def findings(study: Any) -> list[str]:
    """Describe measured outcomes without treating selected winners as forecasts."""
    metrics = study.metrics
    best = metrics["total_return"].idxmax()
    row = metrics.loc[best]
    method = str(best).split(" · ", 1)[0]
    benchmark = f"{method} · Equal weight"
    outcome = f"Highest net return in this sample: {best}, {row['total_return']:+.2%}."
    if benchmark == best:
        outcome += " The equal-weight comparison itself had the highest net return."
    elif benchmark in metrics.index:
        difference = (row["total_return"] - metrics.loc[benchmark, "total_return"]) * 100
        direction = "above" if difference >= 0 else "below"
        outcome += f" This was {abs(difference):.2f} percentage points {direction} {benchmark}."
    safest = metrics["max_drawdown"].idxmax()
    result = [
        outcome,
        f"Smallest maximum drawdown in this sample: {safest}, "
        f"{metrics.loc[safest, 'max_drawdown']:.2%} from a previous portfolio high.",
    ]
    contributions = study.holdings[best]["pnl_contribution"]
    # Two decimal places in percentage points cannot meaningfully show smaller values.
    positive = contributions[contributions >= 5e-5]
    negative = contributions[contributions <= -5e-5]
    if not positive.empty:
        asset = positive.idxmax()
        result.append(f"Largest positive holding contribution for {best}: {asset}, "
                      f"{positive.loc[asset] * 100:+.2f} percentage points of initial capital before fees.")
    else:
        result.append(f"No holding made a positive P&L contribution at the displayed precision for {best} in this sample.")
    if not negative.empty:
        asset = negative.idxmin()
        result.append(f"Largest holding detractor for {best}: {asset}, "
                      f"{negative.loc[asset] * 100:+.2f} percentage points of initial capital before fees.")
    else:
        result.append(f"No holding made a negative P&L contribution at the displayed precision for {best} in this sample.")
    result.append(
        f"Trading for {best}: gross turnover {row['total_turnover']:.2f}× "
        "(summed traded-notional / pretrade-value ratios); "
        f"modeled fees consumed {row['total_cost']:.2%} of initial capital "
        f"({row['total_cost'] * 10_000:.2f} per 10,000 initial units)."
    )
    holdings = study.holdings[best]
    largest = holdings["max_realized_weight"].idxmax()
    result.append(
        f"Largest observed closing position for {best}: {largest}, "
        f"{holdings.loc[largest, 'max_realized_weight']:.2%} of portfolio value; "
        f"its largest target was {holdings.loc[largest, 'max_target_weight']:.2%}. "
        "Price changes can move realized weights above target limits."
    )
    fallbacks = int(metrics["fallback_count"].sum())
    if fallbacks:
        result.append(f"Maximum-Sharpe fallback occurred {fallbacks} times across the strategy paths; "
                      "inspect the warnings and trade logs before interpreting that comparison.")
    return result


def backtest_chart(study: Any, drawdown: bool = False, strategies: list[str] | None = None, benchmark_equity=None,
                   focus: str | None = None) -> go.Figure:
    """Plot selected paths without changes to the complete study data.

    With a focus strategy, that path is blue and every other strategy is a thin grey line without a legend entry.
    """
    figure = go.Figure()
    names = study.equity.columns if strategies is None else strategies
    unknown = [name for name in names if name not in study.equity.columns]
    if unknown:
        raise ValueError("Unknown backtest strategies: " + ", ".join(map(str, unknown)))
    for name in names:
        policy = str(name).partition(" · ")[2]
        path = study.equity[name]
        values = path / path.cummax() - 1 if drawdown else path * 10_000
        if focus is None:
            line, legend = {"color": COLORS.get(policy, REST), "width": 2}, True
        elif name == focus:
            line, legend = {"color": ACCENT, "width": 2.5}, True
        else:
            line, legend = {"color": REST, "width": 1.25}, False
        figure.add_trace(go.Scatter(
            x=study.equity.index, y=values, name=html.escape(display_label(name)), mode="lines", line=line, showlegend=legend,
            hovertemplate="%{x|%Y-%m-%d}<br>" + ("Drawdown: %{y:.2%}" if drawdown else "Net portfolio value: %{y:,.2f}")
            + "<extra>%{fullData.name}</extra>",
        ))
    add_benchmark_paths(figure, benchmark_equity, drawdown)
    style_chart(figure, "Falls from peak" if drawdown else "Growth of 10,000", height=360 if drawdown else 420,
                hovermode="x unified")
    figure.update_yaxes(tickformat=".0%" if drawdown else "~s")
    return figure


def _focus(study: Any, focus: str | None) -> str:
    """Use the requested strategy, else the fixed default that does not depend on results."""
    if focus in study.equity.columns:
        return focus
    defaults = ("Expanding window · Medium", "Minimum volatility", "Expanding window · Minimum volatility")
    return next((name for name in defaults if name in study.equity.columns), study.equity.columns[0])


def _markdown(value: Any) -> str:
    text = html.escape(str(value)).replace("\\", "\\\\").replace("\n", " ").replace("\r", " ")
    return re.sub(r"([|`*_\[\]])", r"\\\1", text)


def _markdown_table(frame: pd.DataFrame) -> str:
    header = ["Strategy" if frame.index.name != "Asset" else "Asset", *frame.columns]
    lines = ["| " + " | ".join(_markdown(cell) for cell in header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join(_markdown(cell) for cell in [index, *row]) + " |")
    return "\n".join(lines)


def findings_markdown(study: Any, source: str, benchmarks=None, evidence=None) -> str:
    """Create a standalone findings report without a Markdown-table dependency."""
    best = study.metrics["total_return"].idxmax()
    holdings = _formatted(_top_holdings(study, best), HOLDINGS)
    holdings.index.name = "Asset"
    dates = study.equity.index
    settings = "\n".join(f"- {_markdown(key)}: {_markdown(value)}" for key, value in study.settings.items())
    notes = "\n".join(f"- {_markdown(note)}" for note in study.warnings)
    comparison = benchmark_markdown(benchmarks, evidence, backtest=True) + "\n" if benchmarks is not None else ""
    return (
        "# Backtesting findings\n\n"
        f"Source: {_markdown(source)}\n\n"
        f"Portfolio baseline: {dates[0].date()}. Evaluation returns: {dates[1].date()} to {dates[-1].date()} "
        f"({len(dates) - 1} observations).\n\n"
        + comparison
        + "## Observed findings\n\n" + BEST_IN_HINDSIGHT + "\n\n"
        + "\n".join(f"- {_markdown(display_label(note))}" for note in findings(study)) + "\n\n"
        "These are retrospective comparisons; the selected strategy is not a forecast.\n\n"
        "## All method and portfolio combinations\n\n"
        + _markdown_table(_formatted(study.metrics, METRICS).rename(index=display_label)) + "\n\n"
        f"## Holdings for {_markdown(display_label(best))}\n\n"
        "Up to 10 largest absolute P&L contributions are shown, restricted to assets targeted above 0.0001%. "
        "Contributions use initial capital "
        "as the denominator, with trading fees accounted for separately. The CSV export contains all holdings.\n\n"
        + _markdown_table(holdings) + "\n\n"
        "## Settings\n\n" + settings + "\n\n"
        "## Assumptions and interpretation\n\n" + "\n\n".join(_assumptions(study)) + "\n"
        + ("\n## Study notes\n\n" + notes + "\n" if notes else "")
    )


def backtest_html(study: Any, benchmarks=None, evidence=None, focus: str | None = None) -> str:
    """Return escaped report sections; the parent document supplies Plotly.js."""
    best = study.metrics["total_return"].idxmax()
    focus = _focus(study, focus)
    holdings = _top_holdings(study, best)
    holding_table = _formatted(holdings, HOLDINGS)
    holding_table.index.name = "Asset"
    bar_rows = holdings.sort_values("pnl_contribution")
    bars = go.Figure(go.Bar(
        x=bar_rows["pnl_contribution"], y=[html.escape(str(asset)) for asset in bar_rows.index],
        orientation="h", marker_color=[ACCENT if value >= 0 else "#86868b" for value in bar_rows["pnl_contribution"]],
        marker_line_width=0, text=bar_rows["pnl_contribution"], texttemplate="%{x:+.1%}", textposition="outside",
        textfont={"size": 12, "color": INK_2}, cliponaxis=False,
        hovertemplate="%{y}<br>P&L / initial capital: %{x:.2%}<extra></extra>",
    ))
    style_chart(bars, "Holding contributions", height=430)
    bars.update_layout(showlegend=False, margin={"l": 8, "r": 56, "t": 44, "b": 24})
    bars.update_xaxes(tickformat=".1%", showgrid=True, gridcolor=GRID)
    bars.update_yaxes(showgrid=False, dtick=1)
    chart_options = {"full_html": False, "include_plotlyjs": False, "config": PLOTLY_CONFIG}
    bullets = "".join(f"<li>{html.escape(display_label(note))}</li>" for note in findings(study))
    warnings = "".join(f"<li>{html.escape(str(note))}</li>" for note in study.warnings)
    assumptions = "".join(f"<p>{html.escape(note)}</p>" for note in _assumptions(study))
    settings = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
                       for key, value in study.settings.items())
    benchmark_equity = benchmarks.backtest_equity if benchmarks is not None else None
    comparison = benchmark_html(benchmarks, evidence, backtest=True) if benchmarks is not None else ""
    return (
        '<section id="backtesting">' + comparison + '<h2>Backtesting findings</h2>'
        f"<p>{BEST_IN_HINDSIGHT}</p>"
        f"<ul>{bullets}</ul><p class=\"muted\">These are retrospective comparisons; "
        "the selected strategy is not a forecast.</p>"
        f'<p class="muted">Blue: {html.escape(display_label(focus))}. Grey: the other strategies.</p>'
        f'<div class="chart">{backtest_chart(study, benchmark_equity=benchmark_equity, focus=focus).to_html(**chart_options)}</div>'
        f'<div class="chart">{backtest_chart(study, drawdown=True, benchmark_equity=benchmark_equity, focus=focus).to_html(**chart_options)}</div>'
        "<h2>All method and portfolio combinations</h2>"
        f'<div class="scroll">{_formatted(study.metrics, METRICS).rename(index=display_label).to_html(escape=True, border=0, classes="data")}</div>'
        f"<h2>Holdings for {html.escape(display_label(best))}</h2><p>Up to 10 largest absolute P&amp;L contributions "
        "are shown, restricted to assets targeted above 0.0001%. "
        "Contributions use initial capital as the denominator; fees are accounted for separately. "
        "The CSV export contains all holdings.</p>"
        f'<div class="chart">{bars.to_html(**chart_options)}</div>'
        f'<div class="scroll">{holding_table.to_html(escape=True, border=0, classes="data")}</div>'
        '<h2>Backtest settings</h2><div class="scroll"><table class="metadata">'
        f"{settings}</table></div><h2>Backtest assumptions</h2>{assumptions}"
        + (f"<aside><h2>Backtest notes</h2><ul>{warnings}</ul></aside>" if warnings else "")
        + "</section>"
    )
