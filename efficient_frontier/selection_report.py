"""Offline evidence and reproducible exports for the single portfolio model."""

import html
import io
import json
import zipfile

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .backtest_report import backtest_chart
from .data import stale_price_notes
from .out_of_sample import portfolio_comparison, run_out_of_sample, sample_values
from .presentation import PLOTLY_JS_LICENSE, csv_text
from .selection import BASELINE, BENCHMARK_NAMES, SELECTED, selection_fit_estimates, selection_frontier
from .style import ACCENT, BENCHMARK_COLORS, GRID, INK_3, PLOTLY_CONFIG, REPORT_CSS, REST, style_chart


SELECTION_METRIC_LABELS = {
    "annual_mean": "Net arithmetic mean / year",
    "cagr": "Net compound growth / year (CAGR)",
    "volatility": "Realized volatility / year",
    "max_drawdown": "Maximum drawdown",
    "total_return": "Net total return",
    "total_cost": "Fees / initial capital",
}
FIT_METRIC_LABELS = {
    "fitted_mean": "Fitted annual mean",
    "model_volatility": "Model volatility",
    "historical_volatility": "Historical volatility",
}
LATEST_LABEL = "Latest stock allocation"
REPLAY_LABEL = "Stock portfolio · buy and hold"
HISTORY_NOTE = (
    "The replay buys the initial allocation once and holds it. Historical data reviewed earlier can introduce research hindsight. "
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
        trace.update(name=REPLAY_LABEL if name == SELECTED else trace.name,
                     showlegend=True, line={"color": colors.get(name, INK_3),
                                           "dash": dashes.get(name, "solid")})
    return figure


def selection_frontier_chart(frontier_data, study):
    """Plot the latest stock frontier and passive ETF references on their fit dates."""
    frontier, _, comparisons, _ = frontier_data
    hover = "Annual volatility: %{x:.2%}<br>Annual arithmetic mean: %{y:.2%}<extra>%{fullData.name}</extra>"
    figure = go.Figure(go.Scatter(
        x=frontier.model_volatility, y=frontier.fitted_mean, mode="lines", name="Stock efficient frontier",
        line={"color": INK_3, "width": 2.5}, hovertemplate=hover,
    ))
    for name, row in comparisons.iterrows():
        selected = name == SELECTED
        label = LATEST_LABEL if selected else str(name)
        risk_label = "Model volatility" if selected and study.settings["shrinkage"] else "Annual volatility"
        figure.add_trace(go.Scatter(
            x=[row.model_volatility], y=[row.fitted_mean], mode="markers", name=html.escape(label),
            marker={"symbol": "circle" if selected else "diamond", "size": 15 if selected else 11,
                    "color": ACCENT if selected else BENCHMARK_COLORS.get(name, INK_3)},
            hovertemplate=risk_label + ": %{x:.2%}<br>Annual arithmetic mean: %{y:.2%}<extra>%{fullData.name}</extra>",
        ))
    if study.settings["shrinkage"]:
        figure.data[0].hovertemplate = hover.replace("Annual volatility", "Model volatility")
    figure.add_vline(x=study.settings["target_volatility"], line_dash="dot", line_color=INK_3,
                     annotation_text="Volatility budget", annotation_position="top left")
    style_chart(figure, "Latest fitted frontier · in-sample", height=420)
    figure.update_layout(legend={"maxheight": 120})
    figure.update_xaxes(title="Annual volatility", tickformat=".0%", rangemode="tozero")
    figure.update_yaxes(title="Annual arithmetic mean", tickformat=".0%")
    return figure


def selection_comparison_html(study, prices: pd.DataFrame) -> str:
    """Contrast the latest allocation with the initial buy-and-hold allocation."""
    latest = study.latest[study.selected]
    replay = study.test
    result = replay.metrics.loc[study.selected]
    return (
        '<h2>Fit and replay use different holdings</h2>'
        '<p><strong>The latest fit includes the replay dates. '
        'The replay did not use the latest holdings.</strong></p>'
        '<div class="pl-fit-replay"><article>'
        '<h3>Latest fit · in-sample</h3>'
        f'<p class="pl-fit-dates">{prices.index[1].date()} to {prices.index[-1].date()}</p>'
        f'<dl><dt>Annual arithmetic mean · before fees</dt><dd>{latest.expected_return:.2%}</dd>'
        f'<dt>Fitted model volatility</dt><dd>{latest.volatility:.2%}</dd></dl>'
        '<p>One allocation chosen using the full history. These fixed-weight return estimates describe its starting risk and return, not a buy-and-hold path.</p>'
        '</article><article><h3>Historical replay · buy and hold</h3>'
        f'<p class="pl-fit-dates">{replay.equity.index[1].date()} to {replay.equity.index[-1].date()}</p>'
        f'<dl><dt>Annual arithmetic mean · after fees</dt><dd>{result.annual_mean:.2%}</dd>'
        f'<dt>Realized volatility</dt><dd>{result.volatility:.2%}</dd></dl>'
        '<p>One initial allocation chosen before the replay. Buy once, pay the entry fee, then hold. '
        'Weights drift with prices; there are no later rebalances.</p></article></div>'
        '<p>Different dates, allocations, and fee treatment. A high fitted mean does not establish that the rule beats the benchmarks.</p>'
    )


def selection_replay_findings_html(study) -> str:
    """Compare net total returns and observed risk with each available ETF."""
    selected = study.metrics.loc[SELECTED]

    def difference(value, reference):
        points = 100 * (value - reference)
        if points == 0:
            return "the same"
        amount = f"{abs(points):.2f}"
        size = "less than 0.01 percentage point" if amount == "0.00" else f"{amount} percentage points"
        return f"{size} {'higher' if points > 0 else 'lower'}"

    rows, missing = [], []
    for symbol, name in BENCHMARK_NAMES.items():
        if name not in study.metrics.index:
            missing.append(symbol)
            continue
        reference = study.metrics.loc[name]
        rows.append(
            f'<p><strong>Versus unscaled {symbol} in this replay:</strong> net total return was '
            f'{difference(selected.total_return, reference.total_return)}, with realized annual volatility '
            f'{difference(selected.volatility, reference.volatility)}.</p>'
        )
    if missing:
        rows.append(f'<p>{" and ".join(missing)} comparison {"is" if len(missing) == 1 else "data are"} unavailable for this replay.</p>')
    return '<div class="pl-replay-findings">' + "".join(rows) + '</div>'


def selection_replay_dominance(study) -> pd.DataFrame:
    """Classify every observed mean/volatility pair against the stock portfolio.

    Retain original metric values and row order. Dominance requires one strict
    improvement and no deterioration. The portfolio's own row is equal.
    """
    comparisons = study.metrics.loc[:, ["annual_mean", "volatility"]].copy()
    selected = comparisons.loc[SELECTED]
    equal = (comparisons.annual_mean == selected.annual_mean) & (comparisons.volatility == selected.volatility)
    reference_dominates = (comparisons.annual_mean >= selected.annual_mean) & (comparisons.volatility <= selected.volatility) & ~equal
    portfolio_dominates = (comparisons.annual_mean <= selected.annual_mean) & (comparisons.volatility >= selected.volatility) & ~equal
    comparisons["classification"] = "tradeoff"
    comparisons.loc[equal, "classification"] = "equal"
    comparisons.loc[reference_dominates, "classification"] = "reference_dominates"
    comparisons.loc[portfolio_dominates, "classification"] = "portfolio_dominates"
    return comparisons


def selection_replay_return_bridge(study, prices: pd.DataFrame) -> pd.Series:
    """Reconcile initial fitted mean to replay mean in annual arithmetic units.

    Separate changed stock means, weight drift, the cash entry interval and
    its fee. Use saved initial weights, without refitting or using latest weights.
    """
    weights = study.test.allocations[study.selected].iloc[0].reindex(prices.columns, fill_value=0.0)
    training = prices.loc[:study.settings["initial_training_end"]].pct_change(fill_method=None).iloc[1:]
    replay_prices = prices.loc[study.test.equity.index]
    returns = replay_prices.pct_change(fill_method=None).iloc[1:]
    periods = study.settings["periods_per_year"]
    scale = periods / len(returns)
    positions = replay_prices.iloc[1:].div(replay_prices.iloc[1]).mul(weights)
    prior_weights = positions.div(positions.sum(axis=1), axis=0).shift(1).iloc[1:]
    drift = prior_weights.sub(weights).mul(returns.iloc[1:]).sum(axis=1).sum()
    fee = study.settings["cost_bps"] / 10_000
    return pd.Series({
        "fitted_mean": float(training.mean() @ weights * periods),
        "realized_mean": float(study.test.metrics.loc[study.selected, "annual_mean"]),
        "mean_change": float((returns.mean() - training.mean()) @ weights * periods),
        "weight_drift": float(drift * scale),
        "entry_delay": float(-(returns.iloc[0] @ weights) * scale),
        "entry_fee": float(-fee / (1 + fee) * scale),
    })


def selection_replay_verdict_html(study) -> str:
    """State observed mean/volatility dominance without changing the CAGR checks."""
    selected = study.metrics.loc[SELECTED]
    references = study.metrics.drop(index=SELECTED)
    dominates = references.loc[
        (references.annual_mean >= selected.annual_mean)
        & (references.volatility <= selected.volatility)
        & ((references.annual_mean > selected.annual_mean) | (references.volatility < selected.volatility))
    ]
    if dominates.empty:
        return ""
    name = dominates.annual_mean.idxmax()
    reference = dominates.loc[name]
    return (
        '<p class="pl-replay-verdict"><strong>'
        f'{html.escape(str(name))} dominated the stock rule on mean and volatility in this replay.</strong><br>'
        f'Annual arithmetic mean: {reference.annual_mean:.2%} versus {selected.annual_mean:.2%}. '
        f'Realized volatility: {reference.volatility:.2%} versus {selected.volatility:.2%}.</p>'
    )


def selection_frontier_summary_html(study, prices: pd.DataFrame, frontier_data) -> str:
    """Explain the frontier and show references on the same dates and return measure."""
    _, _, comparisons, warnings = frontier_data
    columns = ["fitted_mean", "historical_volatility"]
    shrinkage = study.settings["shrinkage"]
    if shrinkage:
        columns.insert(1, "model_volatility")
    display = comparisons.loc[:, columns].rename(columns=FIT_METRIC_LABELS, index={study.selected: LATEST_LABEL})
    risk_note = (
        f'Stock model volatility uses {shrinkage:.0%} covariance shrinkage; ETF volatility does not. '
        'Historical volatility uses ordinary sample covariance for every row. '
        'The frontier and budget use stock model volatility, so their risk estimates differ from the ETF markers.'
        if shrinkage else
        'Every row uses ordinary sample covariance, the same return dates, and the same annualization. '
        'The frontier and volatility budget use this same risk measure.'
    )
    notes = "".join(f'<li>{html.escape(str(warning))}</li>' for warning in warnings)
    return (
        '<h2>Latest fitted frontier · in-sample</h2>'
        f'<p>Return dates: {prices.index[1].date()} to {prices.index[-1].date()}. '
        'The same dates choose and evaluate these weights, before trading fees. '
        'This is an in-sample optimum, not a historical trading result.</p>'
        '<p>The line shows the highest fitted mean available at each volatility among the eligible stocks, '
        f'with no short selling and at most {study.settings["max_weight"]:.0%} in one stock. '
        'The selected point maximizes that mean within the volatility budget. '
        'These limits apply to the starting allocation; buy-and-hold weights and realized risk can later exceed them.</p>'
        f'<p>{risk_note}</p><div class="scroll">{_percent_table(display)}</div>'
        '<p>SPY and QQQ use the same fit dates. They are not portfolio candidates and do not follow the stock holding limit. '
        'None of the ETF positions is scaled to match the selected volatility.</p>'
        + (f'<ul>{notes}</ul>' if notes else '')
    )


def selection_summary_html(study, metadata: dict) -> str:
    """Describe the one model and its later historical check."""
    heading = "Replay: performance checks passed" if study.supported else "Replay: performance checks not met"
    reasons = list(dict.fromkeys(study.reasons))
    notes = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reasons)
    benchmark_note = metadata.get("benchmarks_note", "")
    return (
        f'<h2>{heading}</h2><p>Annual volatility budget: {study.settings["target_volatility"]:.2%}. '
        f'Largest holding: {study.settings["max_weight"]:.1%}. '
        'The model maximizes the fitted historical mean within the volatility budget and holding limit.</p>'
        '<p>The budget limits estimated volatility. Future volatility and losses can exceed it. '
        'The later test checks the same rule against equal-weight stocks, SPY, and QQQ. '
        'These are performance checks, not checks of optimizer accuracy.</p>'
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


OUT_OF_SAMPLE_TITLE = "In sample versus out of sample"
OUT_OF_SAMPLE_SYMBOLS = {BENCHMARK_NAMES["QQQ"]: "diamond", BENCHMARK_NAMES["SPY"]: "triangle-up", BASELINE: "square"}
OUT_OF_SAMPLE_COLORS = {**BENCHMARK_COLORS, BASELINE: INK_3}
PATH_LABELS = {SELECTED: "Stock portfolio", BASELINE: "Equal-weight stocks"}


def percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}".replace("-", "−")


def points(value: float) -> str:
    """Format a difference of two rates in percentage points, with a true minus sign."""
    return f"{100 * value:+.2f} pp".replace("-", "−")


def _dates(start, end) -> str:
    return f"{start.date()} to {end.date()}"


def _window_labels(result, windows) -> list[str]:
    rows = result.windows.loc[windows]
    return [f"{number} · {row.test_start:%b} ’{row.test_start:%y}–{row.test_end:%b} ’{row.test_end:%y}"
            for number, row in rows.iterrows()]


def out_of_sample_frame(result) -> pd.DataFrame:
    """One row per completed window: the stock portfolio on both sides, then equal-weight stocks on both sides."""
    means = portfolio_comparison(result)
    inside = sample_values(result, "in_sample")
    windows = result.windows.loc[means.index]
    return pd.DataFrame({
        "Test dates": [_dates(row.test_start, row.test_end) for row in windows.itertuples()],
        "In sample": means.in_sample, "Out of sample": means.out_of_sample, "Change": means.change,
        "Equal weight in sample": inside[BASELINE], "Equal weight out of sample": means[BASELINE],
    }, index=means.index).rename_axis("Window")


def out_of_sample_paths_frame(result) -> pd.DataFrame:
    """Every path in every completed window: annual mean and volatility on both sides."""
    rows = {}
    means = {sample: sample_values(result, sample) for sample in ("in_sample", "out_of_sample")}
    risk = {sample: sample_values(result, sample, "volatility") for sample in ("in_sample", "out_of_sample")}
    for window in means["in_sample"].index:
        for path in means["in_sample"].columns:
            inside, outside = means["in_sample"].loc[window, path], means["out_of_sample"].loc[window, path]
            rows[(window, PATH_LABELS.get(path, path))] = {
                "In-sample mean": inside, "Out-of-sample mean": outside, "Change": outside - inside,
                "In-sample volatility": risk["in_sample"].loc[window, path],
                "Out-of-sample volatility": risk["out_of_sample"].loc[window, path],
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["Window", "Path"])


def out_of_sample_formats(frame: pd.DataFrame) -> dict:
    """Show rates as percentages and changes in percentage points, with true minus signs."""
    return {column: points if column == "Change" else percent for column in frame.columns if column != "Test dates"}


def out_of_sample_findings(result) -> dict | None:
    """Average the completed windows and count each comparison. Return None when no window completed."""
    means = portfolio_comparison(result)
    if means.empty:
        return None
    references = means.columns.drop(["in_sample", "out_of_sample", "change"])
    equal_in = sample_values(result, "in_sample")[BASELINE]
    return {
        "completed": len(means), "windows": len(result.windows),
        "in_sample": float(means.in_sample.mean()), "out_of_sample": float(means.out_of_sample.mean()),
        "change": float(means.change.mean()), "lower": int((means.change < 0).sum()),
        "margin_in_sample": float((means.in_sample - equal_in).mean()),
        "margin_out_of_sample": float((means.out_of_sample - means[BASELINE]).mean()),
        "higher_than": {name: int((means.out_of_sample > means[name]).sum()) for name in references},
        "test_returns": int(result.windows.loc[means.index, "test_returns"].mean().round()),
    }


def _join(parts: list[str]) -> str:
    if len(parts) < 3:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def out_of_sample_text(findings: dict) -> tuple[str, list[str]]:
    """Return the headline and the supporting sentences for the walk-forward results."""
    count = findings["completed"]
    headline = (f'Out of sample, the stock portfolio averaged {percent(findings["out_of_sample"], 1)} a year. '
                f'In sample, it averaged {percent(findings["in_sample"], 1)}.')
    lines = [f'The out-of-sample mean was lower than the in-sample mean in {findings["lower"]} of {count} '
             f'window{"s" if count != 1 else ""}.',
             'Equal-weight stocks use no return estimates. The stock portfolio’s average margin over them was '
             f'{points(findings["margin_in_sample"])} a year in sample and {points(findings["margin_out_of_sample"])} out of sample.']
    parts = [f"{'equal-weight stocks' if name == BASELINE else name} in {wins} of {count}"
             for name, wins in findings["higher_than"].items()]
    lines.append("Out of sample, the stock portfolio had a higher annual mean than " + _join(parts) + ".")
    return headline, lines


def out_of_sample_chart(result) -> go.Figure:
    """Plot each window's in-sample and out-of-sample annual mean on one axis, with references out of sample."""
    means = portfolio_comparison(result)
    labels = _window_labels(result, means.index)
    windows = result.windows.loc[means.index]
    fit_dates = [_dates(row.fit_start, row.fit_end) for row in windows.itertuples()]
    test_dates = [_dates(row.test_start, row.test_end) for row in windows.itertuples()]

    def hover(kind):
        return f"Window %{{customdata[0]}} · {kind} %{{customdata[1]}}<br>%{{fullData.name}}: %{{x:.2%}}<extra></extra>"

    connectors_x, connectors_y = [], []
    for label, row in zip(labels, means.itertuples()):
        connectors_x += [row.in_sample, row.out_of_sample, None]
        connectors_y += [label, label, None]
    figure = go.Figure(go.Scatter(x=connectors_x, y=connectors_y, mode="lines", line={"color": REST, "width": 2},
                                  hoverinfo="skip", showlegend=False))
    tested = list(zip(means.index, test_dates))
    for name in (BASELINE, BENCHMARK_NAMES["SPY"], BENCHMARK_NAMES["QQQ"]):
        if name in means:
            figure.add_trace(go.Scatter(
                x=means[name], y=labels, mode="markers", name=html.escape(f"{name} · out of sample"),
                marker={"symbol": OUT_OF_SAMPLE_SYMBOLS[name], "size": 10, "color": OUT_OF_SAMPLE_COLORS[name],
                        "line": {"width": 1.5, "color": "#ffffff"}}, customdata=tested, hovertemplate=hover("test dates"),
            ))
    figure.add_trace(go.Scatter(x=means.in_sample, y=labels, mode="markers", name="Stock portfolio · in sample",
                                marker={"symbol": "circle-open", "size": 14, "color": ACCENT, "line": {"width": 2.5}},
                                customdata=list(zip(means.index, fit_dates)), hovertemplate=hover("fit dates")))
    figure.add_trace(go.Scatter(x=means.out_of_sample, y=labels, mode="markers", name="Stock portfolio · out of sample",
                                marker={"symbol": "circle", "size": 14, "color": ACCENT, "line": {"width": 2, "color": "#ffffff"}},
                                customdata=tested, hovertemplate=hover("test dates")))
    figure.add_vline(x=0, line_color=INK_3, line_width=1)
    style_chart(figure, height=190 + 58 * len(labels))
    figure.update_layout(legend={"maxheight": 120, "y": -90 / (58 * len(labels) + 86), "yanchor": "top"}, margin={"b": 130})
    figure.update_xaxes(title="Annual mean while held", tickformat=".0%", showgrid=True, gridcolor=GRID)
    figure.update_yaxes(title="Test window", type="category", autorange="reversed", showgrid=False)
    return figure


def _formatted(frame: pd.DataFrame) -> pd.DataFrame:
    formats = out_of_sample_formats(frame)
    return frame.apply(lambda column: column.map(formats[column.name]) if column.name in formats else column)


def out_of_sample_html(result, error: str | None = None) -> str:
    """Report section with the walk-forward findings, the window tables, and the chart."""
    if error:
        return f'<section id="out-of-sample"><h2>{OUT_OF_SAMPLE_TITLE}</h2><p>{html.escape(error)}</p></section>'
    findings = out_of_sample_findings(result)
    failed = result.windows.loc[result.windows.status != ""]
    notes = "".join(f'<li>Window {number} has no result. {html.escape(row.status)}</li>' for number, row in failed.iterrows())
    notes += "".join(f'<li>{html.escape(note)}</li>' for note in result.notes)
    method = (
        f'<p>Each of {len(result.windows)} windows fits the model on earlier stock prices only. '
        'The volatility budget and holding limit are the same fixed inputs for every window. It buys once at the next close '
        'and holds through the window without trades. The in-sample result buys the same weights at the start of the fit dates '
        'and holds them to the end of the fit. Both sides are measured from the purchase close, so the cash interval and the '
        'entry fee do not enter these figures. '
        f'The test windows do not overlap and cover the final {1 - result.settings["first_fit_fraction"]:.0%} of returns.</p>'
    )
    if findings is None:
        return (f'<section id="out-of-sample"><h2>{OUT_OF_SAMPLE_TITLE}</h2>{method}'
                f'<p>No window had a feasible fit.</p><ul>{notes}</ul></section>')
    headline, lines = out_of_sample_text(findings)
    chart = out_of_sample_chart(result).to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
    main = _formatted(out_of_sample_frame(result)).to_html(escape=True, border=0, classes="data")
    paths = _formatted(out_of_sample_paths_frame(result)).to_html(escape=True, border=0, classes="data")
    return (
        f'<section id="out-of-sample"><h2>{OUT_OF_SAMPLE_TITLE}</h2><p><strong>{html.escape(headline)}</strong></p>'
        + "".join(f'<p>{html.escape(line)}</p>' for line in lines) + method
        + f'<div class="chart">{chart}</div><h3>Annual mean while held</h3><div class="scroll">{main}</div>'
        + f'<h3>Every path in each window</h3><div class="scroll">{paths}</div>'
        + (f'<ul>{notes}</ul>' if notes else '')
        + f'<p>Each fit starts on {result.windows.fit_start.iloc[0].date()} and ends at the close before its test dates. '
          'Each window is short, so its annual mean is noisy. These are historical checks on dates you can already know, '
          'not forecasts.</p></section>'
    )


def selection_fit_summary_html(study, prices: pd.DataFrame) -> str:
    """Separate fitted estimates from the later results of earlier weights."""
    fits = selection_fit_estimates(study, prices).iloc[[0, -1]]
    columns = list(FIT_METRIC_LABELS) if study.settings["shrinkage"] else ["fitted_mean", "historical_volatility"]
    display = fits.loc[:, columns].rename(columns=FIT_METRIC_LABELS)
    display.index = [f"{row.phase} · {date.date()}" for date, row in fits.iterrows()]
    risk_note = (
        f'Model volatility uses a {study.settings["shrinkage"]:.0%} covariance adjustment toward individual stock variances. '
        'Historical volatility uses the ordinary sample covariance with the same dates and weights. '
        'The budget applies to model volatility.'
        if study.settings["shrinkage"] else
        'Historical volatility uses ordinary sample covariance, with no covariance adjustment. '
        'The volatility budget applies to this same measure.'
    )
    return (
        '<h2>Estimates at each fit</h2>'
        f'<p>Each fit uses returns from {prices.index[1].date()} through its date below. '
        'These estimates describe the data used for each fit. They are not forecasts.</p>'
        f'<div class="scroll">{_percent_table(display)}</div>'
        '<p>The model maximizes the fitted arithmetic mean within the model volatility budget and holding limit. '
        'It does not maximize the return-to-volatility ratio or CAGR.</p>'
        f'<p>{risk_note}</p>'
        '<p>The initial fit supplies the only purchase in the historical replay. The holdings then drift with prices. '
        'The latest weights did not produce the earlier test results.</p>'
    )


def selection_benchmark_note(study) -> str:
    """Define comparison holdings and their different risk levels."""
    return (
        'Equal-weight stocks divides money equally across eligible stocks at entry, then lets the weights drift. '
        'SPY tracks the S&P 500. QQQ tracks the Nasdaq-100. All four paths buy once and hold, '
        'with the same entry delay and fee rate. '
        'Each reference has its own volatility. None is scaled to the model volatility budget.'
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


def _out_of_sample(study, prices: pd.DataFrame, out_of_sample):
    """Return the supplied walk-forward result, or calculate it. A too-short history returns its reason."""
    if out_of_sample is not None:
        return out_of_sample, None
    try:
        return run_out_of_sample(study, prices), None
    except ValueError as exc:
        return None, str(exc)


def selection_report_html(study, prices: pd.DataFrame, metadata: dict, *, frontier_data=None, out_of_sample=None) -> str:
    """Show the latest fitted stock weights, the walk-forward tests, and one later replay."""
    details = _details(study, prices, metadata)
    if frontier_data is None:
        frontier_data = selection_frontier(study, prices)
    out_of_sample, out_of_sample_error = _out_of_sample(study, prices, out_of_sample)
    if out_of_sample is None:
        out_of_sample_files = "There are no out_of_sample files because the history is too short for the walk-forward windows.\n"
    elif out_of_sample.metrics.empty:
        out_of_sample_files = "out_of_sample/windows.csv records each walk-forward window and the reason it has no result.\n"
    else:
        out_of_sample_files = ("out_of_sample/windows.csv, metrics.csv, weights.csv, and eligibility.csv record each "
                               "walk-forward window, its results, its allocation, and its eligibility decisions.\n")
    frontier_chart = selection_frontier_chart(frontier_data, study).to_html(
        full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
    )
    result = study.test
    metrics = selection_metrics_frame(result).rename(columns=SELECTION_METRIC_LABELS, index={study.selected: REPLAY_LABEL})
    chart = selection_chart(result).to_html(
        full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG,
    )
    test_section = (
        '<section id="test"><h2>Historical replay · buy and hold</h2>'
        '<p>The rule chooses one initial allocation using earlier prices, then holds it through the replay. '
        'This path does not use the latest fitted weights.</p>'
        f'{selection_replay_verdict_html(result)}'
        f'<p>Baseline: {result.equity.index[0].date()}. '
        f'Evaluation returns: {result.equity.index[1].date()} to {result.equity.index[-1].date()}.</p>'
        '<p>All rows use the same replay dates and fee rate.</p>'
        f'<p>{selection_benchmark_note(study)}</p>'
        '<p>Arithmetic means average the net period returns, including entry fees. CAGR compounds the same returns. '
        'The table keeps a fixed order and does not choose a model from the results.</p>'
        f'<div class="scroll">{_percent_table(metrics)}</div><div class="chart">{chart}</div></section>'
    )
    weights, estimates = _latest_frames(study)
    fit_history = selection_fit_estimates(study, prices)
    for column in (*FIT_METRIC_LABELS, "max_weight"):
        fit_history[column] = fit_history[column].map(lambda value: f"{value:.2%}")
    fit_history["execution_date"] = fit_history.execution_date.dt.strftime("%Y-%m-%d").fillna("—")
    fit_history.index = fit_history.index.strftime("%Y-%m-%d")
    fit_history = fit_history.rename(columns={
        **FIT_METRIC_LABELS, "phase": "Fit", "execution_date": "Execution date",
        "observations": "Return observations", "max_weight": "Largest holding",
    }).rename_axis("Fit through")
    latest = (
        '<p>These weights fit all supplied history through the final price date. The replay bought a separate initial allocation once. '
        'Latest fitted means are annual arithmetic estimates, not realized CAGR or forecasts.</p>'
        '<p>All stocks are shown, largest weight first. Displayed weights are rounded; '
        'latest_weights.csv preserves every weight at full precision.</p>'
        f'<div class="scroll">{_percent_table(weights.sort_values(study.selected, ascending=False, kind="stable"))}</div>'
        f'<div class="scroll">{_percent_table(estimates[["expected_return", "volatility"]].rename(columns={"expected_return": "Fitted arithmetic mean / year", "volatility": "Fitted volatility / year"}))}</div>'
    )
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
<body><main><p class="eyebrow">Historical research</p><h1>One model: latest fit and historical replay</h1>
<button onclick="window.print()">Print / save PDF</button>
<section id="fit-and-replay">{selection_comparison_html(study, prices)}</section>
<section id="latest-frontier">{selection_frontier_summary_html(study, prices, frontier_data)}
<div class="chart">{frontier_chart}</div></section>
<section id="latest-selection"><h2>Latest fitted stock weights · {prices.index[-1].date()}</h2>{latest}</section>
{out_of_sample_html(out_of_sample, out_of_sample_error)}
{test_section}
<section id="selection-assumptions"><h2>Method and limits</h2>
<details id="selection-decision"><summary>Historical performance checks</summary>{selection_summary_html(study, metadata)}</details>
<details id="fit-estimates"><summary>First and latest fitted estimates</summary>{selection_fit_summary_html(study, prices)}</details>
<p>The model maximizes historical arithmetic mean return under a fitted volatility budget and a per-stock holding limit.
Stock weights are long-only and sum to one at entry. There are no later rebalances.</p>
<p>Initial training share: {study.settings['train_fraction']:.0%}.
Holding policy: buy once and hold.
Fee rate: {study.settings['cost_bps']:g} basis points ({study.settings['cost_bps'] / 10000:.2%}) per amount bought or sold.
Covariance shrinkage: {study.settings['shrinkage']:.0%}.
Annualization: {study.settings['periods_per_year']:g} observations per year.</p>
<p>The initial allocation uses prices through the close before entry. The first evaluation interval stays in cash.
The entry fee applies to the amount invested. There are no later trades or terminal sale. Cash earns zero interest.
Equal-weight stocks, SPY, and QQQ also buy once and hold. Position weights drift after entry.
Taxes, currency conversion, liquidity, and future membership changes are not modeled.</p>
<p>Past-only eligibility checks can exclude long unchanged-price runs at a fit. eligibility.csv records each decision for the initial and latest fits.
The original input prices and excluded asset columns remain in the export.</p>
<details><summary>Estimates for every saved fit</summary><div class="scroll">
{fit_history.to_html(escape=True, border=0, classes="data")}</div></details>
<p>Current constituent lists and complete-history requirements introduce selection and survivorship bias.
The later test is a descriptive historical check, not proof of future outperformance.</p>
{'<aside><h3>Data and model notes</h3><ul>' + notes + '</ul></aside>' if notes else ''}</section>
<section id="selection-sources"><h2>Source and reproduction</h2><div class="scroll"><table class="metadata">{source_rows}</table></div>{coverage}
<p>metadata.json records the model revision, settings, input dates, source details, test reasons, walk-forward settings and notes, and every numbered result file.
fit_estimates.csv records the mean and both volatility measures for every saved fit.
{out_of_sample_files}latest/frontier.csv and latest/frontier_weights.csv preserve each plotted stock mix.
latest/fitted_comparison.csv records the same-date portfolio and ETF estimates.
The test exports include every path, allocation, trade, and holding.</p></section>
<footer>This report works offline.<details><summary>Third-party notice: Plotly.js (MIT)</summary>
<pre>{html.escape(PLOTLY_JS_LICENSE)}</pre></details></footer></main></body></html>'''


def selection_report_zip(study, prices: pd.DataFrame, metadata: dict, *, frontier_data=None, out_of_sample=None) -> bytes:
    """Export exact inputs, model settings, every walk-forward window, and every later-test result."""
    details = _details(study, prices, metadata)
    if frontier_data is None:
        frontier_data = selection_frontier(study, prices)
    out_of_sample, out_of_sample_error = _out_of_sample(study, prices, out_of_sample)
    frontier, frontier_weights, fitted_comparison, frontier_warnings = frontier_data
    details["frontier_warnings"] = frontier_warnings
    weights, estimates = _latest_frames(study)
    frames = {"prices.csv": prices.rename_axis("Date"), "latest_weights.csv": weights, "latest_estimates.csv": estimates,
              "fit_estimates.csv": selection_fit_estimates(study, prices),
              "latest/frontier.csv": frontier, "latest/frontier_weights.csv": frontier_weights,
              "latest/fitted_comparison.csv": fitted_comparison}
    if getattr(study, "benchmark_prices", None) is not None:
        frames["benchmark_prices.csv"] = study.benchmark_prices.rename_axis("Date")
    if getattr(study, "eligibility", None) is not None:
        frames["eligibility.csv"] = study.eligibility
    if metadata.get("universe_coverage"):
        frames["universe_coverage.csv"] = pd.DataFrame(metadata["universe_coverage"])
    if out_of_sample is not None:
        details["out_of_sample"] = {**out_of_sample.settings, "notes": out_of_sample.notes}
        frames["out_of_sample/windows.csv"] = out_of_sample.windows
        if not out_of_sample.metrics.empty:
            frames["out_of_sample/metrics.csv"] = out_of_sample.metrics
            frames["out_of_sample/weights.csv"] = out_of_sample.weights
            frames["out_of_sample/eligibility.csv"] = out_of_sample.eligibility
    else:
        details["out_of_sample"] = {"unavailable": out_of_sample_error}
    result = study.test
    frames["test/metrics.csv"] = result.metrics.rename_axis("strategy")
    frames["test/equity.csv"] = result.equity.rename_axis("date")
    for number, name in enumerate(result.equity.columns, 1):
        for kind in ("allocations", "holdings", "trades"):
            frames[f"test/{number:02d}_{kind}.csv"] = getattr(result, kind)[name].rename_axis("symbol" if kind == "holdings" else "date")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.html", selection_report_html(study, prices, metadata, frontier_data=frontier_data,
                                                              out_of_sample=out_of_sample))
        archive.writestr("metadata.json", json.dumps(details, indent=2, default=str, ensure_ascii=False))
        archive.writestr("THIRD_PARTY_NOTICES.txt", PLOTLY_JS_LICENSE)
        for name, frame in frames.items():
            archive.writestr(name, csv_text(frame))
    return output.getvalue()
