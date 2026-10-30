//! Charts and portable reports for the legacy classic analysis.
//!
//! Port of `efficient_frontier/presentation.py`. Owner: reports agent.
//! `csv_text` is [`crate::frame::Table::to_csv`] and [`crate::frame::Frame::to_csv`].
//!
//! Charts in the exported report use fixed element ids instead of random ones:
//! `summary-chart`, `latest-profile-chart`, `frontier-chart`, `holdout-chart`,
//! and the `backtest-*` and `evidence-chart` ids of the backtest and benchmark sections.

use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::backtest::BacktestStudy;
use crate::backtest_report::{backtest_html, findings_markdown};
use crate::benchmark_report::{add_benchmark_estimates, add_benchmark_paths, benchmark_html, dates_json, values_json};
use crate::benchmarks::BenchmarkComparison;
use crate::core::{Analysis, CLASSIC_PROFILE_NAMES, Portfolio};
use crate::data::stale_price_notes;
use crate::error::Result;
use crate::evidence::EvidenceAnalysis;
use crate::frame::{Cell, Date, Frame, Table, iso};
use crate::html_table::{HtmlOptions, data_table, to_html};
use crate::metrics::PERFORMANCE_FIELDS;
use crate::plot::{Figure, PLOTLY_JS};
use crate::profiles::ProfileAnalysis;
use crate::pyfmt::{self, Dict, Json, escape};
use crate::risk::{risk_contributions, risk_summary};
use crate::selection_report::{records_table, write_zip};
use crate::story;
use crate::style::{ChartStyle, INK, INK_2, REST, color, display_label, plotly_config, report_css, style_chart};
use crate::value_error;

pub use crate::plot::PLOTLY_JS_LICENSE;

/// The optional parts of the classic report and the view it opens on.
///
/// `selected_profile` defaults to "Medium" and `selected_method` to "Expanding window".
#[derive(Clone, Copy, Debug, Default)]
pub struct ReportParts<'a> {
    pub study: Option<&'a BacktestStudy>,
    pub latest_profiles: Option<&'a ProfileAnalysis>,
    pub benchmarks: Option<&'a BenchmarkComparison>,
    pub evidence: Option<&'a EvidenceAnalysis>,
    pub selected_profile: Option<&'a str>,
    pub selected_method: Option<&'a str>,
}

/// A labelled portfolio point. The selected point is larger with an ink ring.
fn point(name: &str, x: f64, y: f64, hover: &str, selected: bool) -> Value {
    let label = escape(&display_label(name));
    json!({
        "type": "scatter", "x": values_json(&[x]), "y": values_json(&[y]), "mode": "markers+text", "name": label,
        "text": [label], "textposition": "top center", "textfont": {"size": 12, "color": INK_2}, "cliponaxis": false,
        "marker": {
            "size": if selected { 18 } else { 13 }, "color": color(name, REST),
            "line": if selected { json!({"width": 2.5, "color": INK}) } else { json!({"width": 2, "color": "#ffffff"}) },
        },
        "hovertemplate": hover,
    })
}

/// `fig.update_xaxes(title=text)` replaces the whole axis title, as plotly.py does.
fn axis_title(figure: &mut Figure, axis: &str, text: &str) {
    figure.layout[axis]["title"] = json!({"text": text});
}

/// Show annualized historical training estimates, not future returns.
pub fn frontier_chart(analysis: &Analysis, benchmark_estimates: Option<&Table>, selected: Option<&str>) -> Figure {
    let hover =
        "Model volatility, yearly: %{x:.2%}<br>Fit arithmetic mean, yearly: %{y:.2%}<extra>%{fullData.name}</extra>";
    let mut figure = Figure::with_trace(json!({
        "type": "scatter", "x": values_json(&analysis.frontier.volatility), "y": values_json(&analysis.frontier.expected_return),
        "mode": "lines", "name": "Efficient frontier", "line": {"color": REST, "width": 2.5}, "hovertemplate": hover,
    }));
    let classic = CLASSIC_PROFILE_NAMES.iter().all(|name| analysis.portfolios.contains_key(*name));
    let names: Vec<&str> =
        if classic { CLASSIC_PROFILE_NAMES.to_vec() } else { analysis.portfolios.keys().map(String::as_str).collect() };
    for name in names {
        let portfolio = &analysis.portfolios[name];
        figure.add_trace(point(name, portfolio.volatility, portfolio.expected_return, hover, selected == Some(name)));
    }
    add_benchmark_estimates(&mut figure, benchmark_estimates, "expected_return", Some("Fit arithmetic mean, yearly"));
    style_chart(&mut figure, ChartStyle { title: Some("Classic frontier"), ..Default::default() });
    figure.update_layout(json!({"legend": {"y": -0.24}}));
    axis_title(&mut figure, "xaxis", "Annual volatility");
    figure.update_xaxes(json!({"tickformat": ".0%", "rangemode": "tozero"}));
    axis_title(&mut figure, "yaxis", "Arithmetic mean, yearly (first fit)");
    figure.update_yaxes(json!({"tickformat": ".0%"}));
    figure
}

/// Plot the realized buy-and-hold value of an initial 10,000 units.
pub fn holdout_chart(analysis: &Analysis, benchmark_equity: Option<&Frame>, selected: Option<&str>) -> Result<Figure> {
    let equity = &analysis.equity;
    let names: Vec<&str> = match selected {
        Some(name) => vec![name],
        None => equity.columns.iter().map(String::as_str).collect(),
    };
    let mut figure = Figure::new();
    for name in names {
        let path = equity.column(name).ok_or_else(|| value_error!("Unknown holdout portfolio: {name}"))?;
        let values: Vec<f64> = path.iter().map(|value| value * 10_000.0).collect();
        figure.add_trace(json!({
            "type": "scatter", "x": dates_json(&equity.index), "y": values_json(&values), "name": escape(name), "mode": "lines",
            "line": {"width": 2, "color": color(name, REST)},
            "hovertemplate": "%{x|%Y-%m-%d}<br>Portfolio value: %{y:,.0f}<extra>%{fullData.name}</extra>",
        }));
    }
    add_benchmark_paths(&mut figure, benchmark_equity, false);
    style_chart(
        &mut figure,
        ChartStyle { title: Some("Classic holdout"), height: 380, hovermode: "x unified", story: false },
    );
    figure.update_yaxes(json!({"tickformat": "~s"}));
    Ok(figure)
}

/// Whether the profiles optimize the weakest of three window means.
fn window_objective(profiles: &ProfileAnalysis) -> bool {
    match profiles.settings.get("return_objective") {
        Some(objective) => objective.as_str() == Some("worst_window"),
        None => {
            profiles.settings.get("selection_method").and_then(Json::as_str) == Some("worst_window_return_frontier")
        }
    }
}

/// Plot the selected full-history fit objective against estimated volatility.
pub fn latest_profile_chart(
    profiles: &ProfileAnalysis,
    benchmark_estimates: Option<&Table>,
    selected: Option<&str>,
) -> Figure {
    let weakest = window_objective(profiles);
    let column = if weakest { "worst_window_return" } else { "expected_return" };
    let title = if weakest { "Weakest-window frontier" } else { "Historical mean frontier" };
    let label = if weakest { "Lowest annual window mean" } else { "Full-history arithmetic mean, yearly" };
    let hover = format!("Model volatility, yearly: %{{x:.2%}}<br>{label}: %{{y:.2%}}<extra>%{{fullData.name}}</extra>");
    let mut figure = Figure::with_trace(json!({
        "type": "scatter", "x": values_json(&profiles.frontier.f64s("volatility")), "y": values_json(&profiles.frontier.f64s(column)),
        "mode": "lines", "name": title, "line": {"color": REST, "width": 2.5}, "hovertemplate": hover,
    }));
    let (volatility, values) = (profiles.summary.f64s("volatility"), profiles.summary.f64s(column));
    for (i, name) in profiles.summary.index_labels().iter().enumerate() {
        figure.add_trace(point(name, volatility[i], values[i], &hover, selected == Some(name.as_str())));
    }
    add_benchmark_estimates(&mut figure, benchmark_estimates, column, Some(label));
    style_chart(&mut figure, ChartStyle { title: Some(title), ..Default::default() });
    figure.update_layout(json!({"legend": {"y": -0.24}}));
    axis_title(&mut figure, "xaxis", "Annual volatility");
    figure.update_xaxes(json!({"tickformat": ".0%", "rangemode": "tozero"}));
    axis_title(
        &mut figure,
        "yaxis",
        if weakest { "Lowest window mean, yearly" } else { "Arithmetic mean, yearly (full history)" },
    );
    figure.update_yaxes(json!({"tickformat": ".0%"}));
    figure
}

/// One column of weights per portfolio, indexed by `ticker`.
pub fn weights_frame(portfolios: &IndexMap<String, Portfolio>) -> Table {
    let mut tickers: Vec<String> = Vec::new();
    for portfolio in portfolios.values() {
        for label in &portfolio.weights.labels {
            if !tickers.contains(label) {
                tickers.push(label.clone());
            }
        }
    }
    let mut table = Table::labelled(Some("ticker"), &tickers);
    for (name, portfolio) in portfolios {
        table.push(name, tickers.iter().map(|ticker| Cell::from(portfolio.weights.get(ticker))).collect());
    }
    table
}

/// The classic holdout metrics, indexed by `portfolio`.
pub fn holdout_metrics_table(analysis: &Analysis) -> Table {
    let names: Vec<&String> = analysis.holdout_metrics.keys().collect();
    let mut table = Table::labelled(Some("portfolio"), &names);
    for field in PERFORMANCE_FIELDS {
        let values =
            analysis.holdout_metrics.values().map(|metrics| Cell::Float(metrics.get(field).unwrap_or(f64::NAN)));
        table.push(field, values.collect());
    }
    table
}

fn date_value(frame: &Frame, last: bool) -> Json {
    match (frame.index.first(), frame.index.last()) {
        (Some(first), Some(end)) => Json::from(iso(if last { *end } else { *first })),
        _ => Json::Null,
    }
}

/// The report metadata: inputs, dates, settings and every optional study's settings.
fn metadata(analysis: &Analysis, metadata: &Dict, parts: &ReportParts) -> Dict {
    let mut result = metadata.clone();
    for (label, returns) in [("training", &analysis.train_returns), ("holdout", &analysis.test_returns)] {
        result.insert(format!("{label}_start"), date_value(returns, false));
        result.insert(format!("{label}_end"), date_value(returns, true));
        result.insert(format!("{label}_observations"), returns.nrows());
    }
    result.insert("assets", analysis.train_returns.columns.clone());
    result.insert("periods_per_year", analysis.periods_per_year);
    result.insert("annualization_days", analysis.periods_per_year);
    result.insert(
        "holdout_strategy",
        "Allocate once at the split, then hold adjusted-price exposures; no cross-asset rebalancing, and weights drift.",
    );
    result.insert(
        "distributions",
        "Reflected in the data provider's adjusted-price series, not accumulated as separate cash.",
    );
    result.insert("estimation", "Historical arithmetic mean returns and covariance estimated from training data only.");
    result.insert("exclusions", "No trading costs, taxes, or currency conversion (FX).");
    if let Some(study) = parts.study {
        result.insert("backtests", study.settings.clone());
        result.insert("backtest_warnings", study.warnings.clone());
        let files = study
            .equity
            .columns
            .iter()
            .enumerate()
            .map(|(i, name)| (format!("backtests/{:02}", i + 1), Json::from(name.as_str())))
            .collect();
        result.insert("backtest_files", Dict(files));
        result.insert(
            "exclusions",
            "The original holdout excludes trading costs; the backtest comparison deducts the selected fees. Both exclude taxes and FX.",
        );
    }
    if let Some(profiles) = parts.latest_profiles {
        let objective_note = if window_objective(profiles) {
            "The weakest-window objective optimizes the lowest of three window means. "
        } else {
            "The historical-mean objective uses the full fit period. Three-window means are diagnostics, not optimization targets. "
        };
        let windows = &profiles.windows;
        let records: Vec<Json> = (0..windows.nrows())
            .map(|i| {
                let mut record = Dict::new();
                record.insert("window", cell_json(&windows.index[0][i]));
                for (column, cells) in windows.columns.iter().zip(&windows.data) {
                    record.insert(column.as_str(), cell_json(&cells[i]));
                }
                Json::Dict(record)
            })
            .collect();
        result.insert(
            "latest_profiles",
            crate::dict! {
                "as_of" => profiles.as_of.as_str(),
                "observations" => profiles.observations,
                "settings" => profiles.settings.clone(),
                "windows" => records,
                "warnings" => profiles.warnings.clone(),
                "interpretation" => format!(
                    "Fits all supplied history through the last input date. {objective_note}\
                     expected_return is the annualized arithmetic mean over the full fit period; worst_window_return is the lowest annual window mean. \
                     These are in-sample estimates, not forecasts or holdout results."
                ),
            },
        );
    }
    if let Some(benchmarks) = parts.benchmarks {
        result.insert("benchmarks", benchmarks.settings.clone());
    }
    if let Some(evidence) = parts.evidence {
        let mut settings = evidence.settings.clone();
        settings.insert("warnings", evidence.warnings.clone());
        result.insert("evidence", settings);
    }
    result
}

fn cell_json(cell: &Cell) -> Json {
    match cell {
        Cell::Null => Json::Null,
        Cell::Bool(value) => Json::Bool(*value),
        Cell::Int(value) => Json::Int(*value),
        Cell::Float(value) => Json::Float(*value),
        Cell::Str(value) => Json::Str(value.clone()),
        Cell::Date(value) => Json::Str(format!("{} 00:00:00", iso(*value))),
    }
}

/// Percent columns with two decimals, other columns with three; missing values as an em dash.
/// `percent_columns` `None` formats every column as a percent.
fn table(frame: &Table, percent_columns: Option<&[&str]>) -> String {
    let mut display = frame.clone();
    for (column, cells) in display.columns.iter().zip(display.data.iter_mut()) {
        let percent = percent_columns.is_none_or(|columns| columns.contains(&column.as_str()));
        for cell in cells.iter_mut() {
            let value = cell.as_f64().unwrap_or(f64::NAN);
            *cell = Cell::Str(if cell.is_null() {
                "—".into()
            } else if percent {
                pyfmt::pct(value, 2)
            } else {
                pyfmt::format(value, ".3f")
            });
        }
    }
    data_table(&display)
}

/// `frame.rename(columns=display_label)`.
fn display_columns(mut frame: Table) -> Table {
    for column in &mut frame.columns {
        *column = display_label(column);
    }
    frame
}

/// `frame.rename(index=display_label)` on the first index level.
fn display_index(mut frame: Table) -> Table {
    if let Some(level) = frame.index.first_mut() {
        for cell in level {
            *cell = Cell::Str(display_label(&cell.py_str()));
        }
    }
    frame
}

fn day_text(date: &str) -> String {
    Date::parse_from_str(date, "%Y-%m-%d").map_or_else(|_| date.to_string(), story::date_text)
}

fn latest_profiles_html(profiles: &ProfileAnalysis, benchmark_estimates: Option<&Table>, selected: &str) -> String {
    let weakest = window_objective(profiles);
    let mean_label = "Full-history arithmetic mean, yearly";
    let window_label = format!("Lowest annual window mean{}", if weakest { "" } else { " (diagnostic)" });
    let objective_note = if weakest {
        "The three historical windows define the weakest-window fit. "
    } else {
        "The historical-mean fit uses the full history. The three historical windows are diagnostics, not optimization targets. "
    };
    let selection_note = if weakest {
        "The model minimizes estimated variance at a return target that applies to every historical window. \
         Low uses the minimum-variance point. Highest uses the highest feasible target for the lowest window mean. \
         Medium targets the midpoint of this return range, not the midpoint of volatility."
    } else {
        "The model minimizes estimated variance at each full-history arithmetic mean return target. \
         Low uses the point with the least fitted volatility. Highest uses the highest fitted arithmetic mean return. \
         Medium uses the sampled frontier point nearest to the midpoint of Low and Highest volatility. \
         When the frontier has no volatility range, all three use the same highest-return mix."
    };
    let mut summary = display_index(profiles.summary.clone());
    summary.rename_columns(&[
        ("expected_return", mean_label),
        ("volatility", "Model volatility / year"),
        ("sharpe", "Estimated Sharpe"),
        ("worst_window_return", window_label.as_str()),
        ("max_weight", "Largest weight"),
        ("effective_holdings", "Effective holdings"),
    ]);
    let mut windows = profiles.windows.clone();
    windows.rename_columns(&[
        ("start", "Start"),
        ("end", "End"),
        ("observations", "Observations"),
        ("years", "Years of observations"),
    ]);
    windows.index_names = vec![Some("Window".into())];
    let mut window_returns = display_columns(profiles.window_returns.clone());
    window_returns.index_names = vec![Some("Window".into())];
    let weights = weights_frame(&profiles.portfolios);
    let visible: Vec<usize> = (0..weights.nrows())
        .filter(|i| weights.data.iter().any(|column| column[*i].as_f64().is_some_and(|value| value >= 0.00005)))
        .collect();
    let selected_weights = weights.f64s(selected);
    let mut order = visible.clone();
    order.sort_by(|a, b| {
        let (x, y) = (selected_weights[*a], selected_weights[*b]);
        match (x.is_nan(), y.is_nan()) {
            (false, false) => y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal),
            (nan_x, nan_y) => nan_x.cmp(&nan_y),
        }
    });
    let weights = display_columns(weights.take_rows(&order));
    let warnings: String = profiles.warnings.iter().map(|note| format!("<li>{}</li>", escape(note))).collect();
    let chart = latest_profile_chart(profiles, benchmark_estimates, Some(selected))
        .to_html("latest-profile-chart", &plotly_config());
    let windows_html = to_html(
        &windows,
        &HtmlOptions { float_format: Some(|value| pyfmt::format(value, ".2f")), ..HtmlOptions::REPORT },
    );
    let mut html = format!(
        "<section id=\"latest-profiles\"><h2>Latest model holdings as of {}</h2>\
         <p class=\"source\">These model weights use all supplied history through the last input date. \
         The date above is the last price observation, not a live quote.</p>\
         <p>Low, Medium, and Highest are relative risk levels within this frontier. They are not universal risk ratings. \
         {objective_note}Their means are in-sample estimates, not forecasts or holdout results. \
         An arithmetic mean is not compound growth (CAGR).</p>",
        day_text(&profiles.as_of)
    );
    if !warnings.is_empty() {
        html.push_str(&format!("<aside><h3>Profile notes</h3><ul>{warnings}</ul></aside>"));
    }
    html.push_str(&format!(
        "<h3>Latest weights</h3><p class=\"muted\">Sorted by {}. \
         Rows under 0.005% in all three levels are hidden here. latest_profile_weights.csv has every row.</p>\
         <div class=\"scroll\">{}</div><h3>Profile estimates</h3><div class=\"scroll\">{}</div>\
         <p class=\"muted\">{selection_note}</p><div class=\"chart\">{chart}</div>\
         <h3>Historical windows{}</h3>\
         <p class=\"muted\">The model divides {} return observations into three consecutive windows without overlap. \
         Annual arithmetic means use {} observations per year. \
         The means assume fixed weights each period and exclude fees and taxes.</p>\
         <div class=\"scroll\">{windows_html}</div><h3>Annual mean return in each window</h3>\
         <div class=\"scroll\">{}</div></section>",
        escape(&display_label(selected)),
        table(&weights, None),
        table(&summary, Some(&[mean_label, "Model volatility / year", window_label.as_str(), "Largest weight"])),
        if weakest { "" } else { " · diagnostics" },
        profiles.observations,
        pyfmt::g(profiles.settings.f64("periods_per_year")),
        table(&window_returns, None),
    ));
    html
}

/// Show measured rule outcomes before proposed holdings and the detailed selected result.
fn summary_html(
    analysis: &Analysis,
    metadata: &Dict,
    prices: Option<&Frame>,
    parts: &ReportParts,
    profile: &str,
    method: &str,
) -> Result<String> {
    let universe = metadata.get("universe").map_or_else(String::new, Json::py_str);
    let coverage = metadata.get("universe_coverage").and_then(Json::as_list).map(records_table);
    let names = story::company_names(coverage.as_ref());
    let names = Some(&names);
    let mut html = String::from("<section id=\"summary\">");
    if let Some(prices) = prices {
        html.push_str(&story::context_result_html(metadata, prices));
    }
    let items = story::fine_print_items(
        metadata,
        prices.unwrap_or(&analysis.train_returns),
        "Universe coverage in Data and assumptions",
    );
    html.push_str("<aside class=\"pl-read\"><h3>Read this first</h3>");
    for item in items {
        html.push_str(&format!("<p>{item}</p>"));
    }
    html.push_str("</aside>");
    let (study, benchmarks) = (parts.study, parts.benchmarks);
    if let Some(study) = study {
        html.push_str(&format!(
            "<div class=\"tile\" id=\"decision-context\">{}</div>",
            story::decision_panel_html(study, &format!("{method} · {profile}"), benchmarks)
        ));
    }
    match parts.latest_profiles {
        Some(profiles) => {
            let weights = &profiles.portfolios[profile].weights;
            html.push_str(&format!(
                "<p class=\"pl-eyebrow\">Latest proposed holdings · fitted through {}</p>",
                day_text(&profiles.as_of)
            ));
            html.push_str(&format!("<h2>{}</h2>", story::headline_html(weights, profile, &universe)));
            html.push_str(
                "<p class=\"source\">These weights use all selected prices. \
                 The test replays earlier fits of this rule, with different holdings.</p>",
            );
            if !window_objective(profiles) {
                html.push_str(
                    "<p class=\"source\">The primary profiles use the historical arithmetic mean return objective. \
                     Profile backtests use the same objective with only history available before each trade. \
                     Three-window estimates are diagnostics.</p>",
                );
            }
            let notes = prices.map_or_else(Vec::new, stale_price_notes);
            if !notes.is_empty() {
                let paragraphs: String = notes.iter().map(|note| format!("<p>{}</p>", escape(note))).collect();
                html.push_str(&format!("<aside class=\"pl-read\"><h3>Price checks</h3>{paragraphs}</aside>"));
            }
            html.push_str(&format!(
                "<div class=\"tile\">{}{}</div>",
                story::stats_html(profiles, profile, &universe, names),
                story::list_html(weights, names)
            ));
            html.push_str(&format!(
                "<div class=\"tile\"><div class=\"pl-head\"><h3 class=\"pl-title\">How the three levels compare</h3></div>{}</div>",
                story::levels_ledger_html(profiles, profile, names, &universe)
            ));
        }
        None => html.push_str("<h2>There is not enough history to set risk levels.</h2>"),
    }
    if let Some(study) = study {
        let mut selected = format!("{method} · {profile}");
        if study.equity.position(&selected).is_none() {
            selected = format!("{method} · Minimum volatility");
            html.push_str(&format!(
                "<p class=\"source\">The requested {} profile was not included in this backtest. \
                 The detailed result below shows {}.</p>",
                escape(&display_label(profile)),
                escape(&display_label(&selected))
            ));
        }
        let markets = benchmarks.is_some_and(|benchmarks| benchmarks.backtest_equity.is_some());
        let evidence = parts
            .evidence
            .filter(|evidence| evidence.summary.index[0].iter().any(|cell| cell.as_str() == Some(selected.as_str())));
        let chart = story::story_chart(study, &selected, "Growth", benchmarks, evidence)?
            .to_html("summary-chart", &plotly_config());
        let months = story::months_tested(study);
        let (answer, lead) = if markets {
            (
                story::evidence_answer(evidence, &selected),
                story::evidence_lead(evidence, &selected, months, Some(metadata)),
            )
        } else {
            (
                "There is no market to compare with.".to_string(),
                "The statistics compare the rule with the S&P 500 and the Nasdaq-100. This study has no market prices."
                    .to_string(),
            )
        };
        html.push_str(&format!(
            "<p class=\"pl-eyebrow\">Later compound growth compared with the ETFs</p>\
             <h3 class=\"pl-answer\">{}</h3><p class=\"pl-lead\">{}</p>\
             <div class=\"tile\">{}</div><div class=\"tile\">{}</div>\
             <div class=\"tile\"><h3 class=\"pl-title\">Growth of 10,000</h3><div class=\"chart\">{chart}</div>\
             <p class=\"pl-small\">{}</p></div>\
             <p class=\"pl-eyebrow\">How sure can we be?</p><h3 class=\"pl-answer\">{}</h3>",
            escape(&story::market_answer(study, &selected, benchmarks)),
            escape(&story::market_lead(study, &selected)),
            story::result_tile_html(study, &selected, benchmarks, evidence, metadata),
            story::result_drivers_html(study, &selected),
            escape(&story::chart_footnote(study, benchmarks)),
            escape(&answer),
        ));
        if !lead.is_empty() {
            html.push_str(&format!("<p class=\"pl-lead\">{}</p>", escape(&lead)));
        }
        if let (true, Some(evidence)) = (markets, evidence) {
            html.push_str(&format!(
                "<div class=\"pl-cols\"><div class=\"tile\"><h3 class=\"pl-title\">Average yearly gap, with 95% range</h3>\
                 <p class=\"pl-sub\">Your rule minus each market, in percentage points. This is an arithmetic average, \
                 so it differs from the growth gap above.</p>{}</div>\
                 <div class=\"tile\"><h3 class=\"pl-title\">Stretch by stretch</h3>{}\
                 <p class=\"pl-sub\">The test period, split into three equal stretches. These differ from the three historical windows \
                 shown with the latest holdings.</p>{}</div></div>\
                 <p class=\"pl-small\">These tests describe the last {months} months. They cannot tell you how the rule will do next.</p>",
                story::interval_html(evidence, &selected),
                story::stretch_summary_html(evidence, &selected),
                story::stretch_table_html(evidence, &selected),
            ));
        }
    }
    html.push_str("</section>");
    Ok(html)
}

/// Return an offline HTML report with a single embedded plotly.js bundle.
///
/// The summary compares measured rule outcomes before the selected level's proposed holdings.
/// Prices are optional and only supply the dates and counts in the summary context line.
pub fn report_html(
    analysis: &Analysis,
    metadata: &Dict,
    parts: &ReportParts,
    prices: Option<&Frame>,
) -> Result<String> {
    let profile = parts.selected_profile.unwrap_or("Medium");
    let method = parts.selected_method.unwrap_or("Expanding window");
    let details = self::metadata(analysis, metadata, parts);
    let source = details.get("source").map_or_else(|| "Unspecified source".to_string(), Json::py_str);
    let lower = source.to_lowercase();
    let source_label = if lower.contains("demo") || lower.contains("synthetic") {
        "Synthetic demonstration data"
    } else {
        "Data source"
    };
    let hidden = [
        "backtests",
        "backtest_warnings",
        "backtest_files",
        "annualization_days",
        "latest_profiles",
        "benchmarks",
        "evidence",
        "universe_coverage",
    ];
    let metadata_rows: String = details
        .iter()
        .filter(|(key, _)| !hidden.contains(key))
        .map(|(key, value)| {
            format!(
                "<tr><th>{}</th><td>{}</td></tr>",
                escape(&story::setting_label(key)),
                escape(&story::setting_text(key, value))
            )
        })
        .collect();
    let coverage_html = match details.get("universe_coverage").filter(|value| story::truthy(Some(value))) {
        Some(records) => format!(
            "<details><summary>Universe coverage</summary><div class=\"scroll\">{}</div></details>",
            to_html(&records_table(records.as_list().unwrap_or_default()), &HtmlOptions::REPORT_NO_INDEX)
        ),
        None => String::new(),
    };
    let warnings: String = analysis.warnings.iter().map(|warning| format!("<li>{}</li>", escape(warning))).collect();
    let warnings_html = if warnings.is_empty() {
        String::new()
    } else {
        format!("<aside><h2>Analysis notes</h2><ul>{warnings}</ul></aside>")
    };
    let names: Vec<&String> = analysis.portfolios.keys().collect();
    let field = |value: fn(&Portfolio) -> f64| analysis.portfolios.values().map(|p| Cell::Float(value(p))).collect();
    let estimates = Table::labelled(None, &names)
        .with("Fit arithmetic mean, yearly", field(|p| p.expected_return))
        .with("Model volatility / year", field(|p| p.volatility))
        .with("Estimated Sharpe", field(|p| p.sharpe));
    let mut holdout = holdout_metrics_table(analysis);
    holdout.rename_columns(&[
        ("total_return", "Total return"),
        ("cagr", "CAGR"),
        ("volatility", "Annual volatility"),
        ("sharpe", "Realized Sharpe"),
        ("max_drawdown", "Maximum drawdown"),
        ("sortino", "Sortino"),
        ("calmar", "Calmar"),
    ]);
    let mut risk = risk_summary(analysis);
    risk.rename_columns(&[
        ("max_weight", "Largest weight"),
        ("effective_holdings", "Effective holdings"),
        ("diversification_ratio", "Diversification ratio"),
    ]);
    let config = plotly_config();
    let benchmarks = parts.benchmarks;
    let frontier =
        frontier_chart(analysis, benchmarks.map(|b| &b.training_estimates), None).to_html("frontier-chart", &config);
    let holdout_plot =
        holdout_chart(analysis, benchmarks.map(|b| &b.holdout_equity), None)?.to_html("holdout-chart", &config);
    let comparison_html = match (parts.study, benchmarks) {
        (Some(study), _) => {
            let mut focus = format!("{method} · {profile}");
            if study.equity.position(&focus).is_none() {
                focus = format!("{method} · Minimum volatility");
            }
            backtest_html(study, benchmarks, parts.evidence, Some(&focus))
        }
        (None, Some(benchmarks)) => benchmark_html(benchmarks, parts.evidence, false),
        (None, None) => String::new(),
    };
    let profiles_html = parts.latest_profiles.map_or_else(String::new, |profiles| {
        latest_profiles_html(profiles, benchmarks.map(|b| &b.latest_estimates), profile)
    });
    let prices = prices.filter(|prices| !prices.is_empty());
    let summary = summary_html(analysis, metadata, prices, parts, profile, method)?;
    let comparison_notes: String = ["benchmarks_note", "evidence_note"]
        .iter()
        .filter_map(|key| details.get(key).filter(|value| story::truthy(Some(value))))
        .map(|value| format!("<p class=\"source\">{}</p>", escape(&value.py_str())))
        .collect();
    let has_study = parts.study.is_some();
    let cost_note = if has_study {
        "The original holdout excludes trading costs. The backtest comparison deducts the selected fees."
    } else {
        "No trading costs are included."
    };
    let introduction = if has_study {
        "The backtest comparison updates portfolios using only information available before each trade. \
         The original fixed-allocation holdout is shown separately."
    } else {
        "The original holdout selects weights from the training period, then evaluates them on later observations."
    };
    let classic_definitions = if CLASSIC_PROFILE_NAMES.iter().all(|name| analysis.portfolios.contains_key(*name)) {
        let tail = if parts.latest_profiles.is_some_and(window_objective) {
            "These levels differ from the optional full-history profiles, which optimize the weakest of three past windows.</p>"
        } else {
            "These levels use the same historical-mean objective as the full-history profiles, but fit only the initial training sample.</p>"
        };
        format!(
            "<p class=\"muted\"><strong>Classic Low, Medium and High.</strong> These levels belong to the first-fit mean-variance frontier. \
             Low uses the point with the least fitted volatility. Medium uses the sampled frontier point nearest to the midpoint \
             of Low and High volatility. High uses the endpoint with the highest fitted arithmetic mean return. \
             When the frontier has no volatility range, all three use the same highest-return mix. {tail}"
        )
    } else {
        String::new()
    };
    let text = |key: &str| details.get(key).map_or_else(|| "None".to_string(), Json::py_str);
    let periods = pyfmt::g(analysis.periods_per_year);
    Ok(format!(
        "<!doctype html>\n\
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
<title>Portfolio Lab · Research report</title>\n\
<style>{css}</style><script>{PLOTLY_JS}</script></head><body><main>\n\
<p class=\"eyebrow\">Historical research, not a forecast</p><h1>Portfolio Lab · Research report</h1>\n\
<button onclick=\"window.print()\">Print / save PDF</button>\n\
<p class=\"source\"><strong>{source_label}:</strong> {source}</p>\n\
{comparison_notes}\n\
{summary}\n\
<hr><p class=\"eyebrow\">Appendix: every result</p>\n\
{profiles_html}\n\
<section id=\"classic-model\"><p>{introduction}\n\
Training estimates describe historical data. They are not predictions or guarantees of future performance.</p>\n\
{warnings_html}</section>\n\
{comparison_html}\n\
<section id=\"training\"><h2>Training estimates</h2><p class=\"muted\">Arithmetic mean returns and covariance use {periods} observations per year.\n\
The analysis applies covariance shrinkage to training data only. Sharpe ratios use the configured risk-free rate.</p>\n\
<p class=\"muted\">Fit return dates: {training_start} to {training_end}.\n\
A higher point means a higher average return during this fit. It does not show which mix won on the later holdout dates.\n\
The arithmetic mean is not compound growth (CAGR) or a forecast.</p>\n\
{classic_definitions}\n\
<div class=\"chart\">{frontier}</div>\n\
<div class=\"scroll\">{estimates}</div></section>\n\
<section id=\"allocations\"><h2>Portfolio allocations</h2><p class=\"muted\">These are the initial weights at the training/holdout split.</p>\n\
<div class=\"scroll\">{weights}</div></section>\n\
<section id=\"risk\"><h2>Concentration and risk</h2><p class=\"muted\">These measures use the initial weights and the estimated training covariance.\n\
Effective holdings equals one divided by the sum of squared weights.\n\
The diversification ratio divides weighted asset volatility by portfolio volatility.</p>\n\
<div class=\"scroll\">{risk}</div>\n\
<h3>Share of portfolio variance</h3><p class=\"muted\">Each asset contributes this share of total portfolio variance.\n\
The shares sum to 100% when portfolio variance is positive. Negative shares identify assets that reduce total variance through covariance.\n\
An em dash marks undefined risk shares and diversification ratios when portfolio variance is zero.</p>\n\
<div class=\"scroll\">{contributions}</div></section>\n\
<section id=\"holdout\"><h2>Realized holdout results</h2><p class=\"muted\">Each portfolio starts with 10,000 units, allocates once at the split,\n\
and then holds adjusted-price exposures. There is no subsequent trading between assets; portfolio weights drift with prices.\n\
Distributions follow the provider's price adjustments rather than accumulating as separate cash.\n\
CAGR and volatility use {periods} observations per year.</p>\n\
<p class=\"muted\">Holdout return dates: {holdout_start} to {holdout_end}.\n\
The initial allocation uses the close on {first_close}.\n\
These later returns did not select the classic weights.</p>\n\
<p class=\"muted\">Sortino measures excess return relative to downside deviation. Calmar divides CAGR by the absolute maximum drawdown.\n\
An em dash marks a ratio with a zero denominator.</p><div class=\"chart\">{holdout_plot}</div>\n\
<div class=\"scroll\">{holdout_table}</div></section>\n\
<section id=\"data\"><h2>Data and assumptions</h2><p class=\"muted\">Files call the Highest level “Extreme”.</p>\n\
<div class=\"scroll\"><table class=\"metadata\">{metadata_rows}</table></div>{coverage_html}</section>\n\
<footer>{cost_note} Taxes and currency conversion (FX) are excluded. Price series must share a consistent currency basis.\n\
This report is a historical research tool and does not predict investment outcomes. Charts work offline.\n\
<details><summary>Third-party notice: Plotly.js (MIT)</summary><pre>{license}</pre></details></footer>\n\
</main></body></html>",
        css = report_css(),
        source = escape(&source),
        training_start = text("training_start"),
        training_end = text("training_end"),
        estimates = table(&estimates, Some(&["Fit arithmetic mean, yearly", "Model volatility / year"])),
        weights = table(&weights_frame(&analysis.portfolios), None),
        risk = table(&risk, Some(&["Largest weight"])),
        contributions = table(&risk_contributions(analysis), None),
        holdout_start = text("holdout_start"),
        holdout_end = text("holdout_end"),
        first_close = iso(analysis.equity.index[0]),
        holdout_table = table(&holdout, Some(&["Total return", "CAGR", "Annual volatility", "Maximum drawdown"])),
        license = escape(PLOTLY_JS_LICENSE),
    ))
}

/// `json.dumps(..., ensure_ascii=True)`: every non-ASCII character as a `\uXXXX` escape.
pub fn ascii_json(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        if character.is_ascii() {
            escaped.push(character);
        } else {
            let mut units = [0u16; 2];
            for unit in character.encode_utf16(&mut units) {
                escaped.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    escaped
}

/// Bundle the report, inputs, and numeric results for offline inspection.
pub fn report_zip(analysis: &Analysis, prices: &Frame, metadata: &Dict, parts: &ReportParts) -> Result<Vec<u8>> {
    let mut files: Vec<(String, String)> = vec![
        ("weights.csv".into(), weights_frame(&analysis.portfolios).to_csv(None)),
        ("frontier.csv".into(), analysis.frontier.points_table().to_csv(None)),
        ("frontier_weights.csv".into(), analysis.frontier.weights_table().to_csv(None)),
        ("holdout_metrics.csv".into(), holdout_metrics_table(analysis).to_csv(None)),
        ("holdout_curve.csv".into(), analysis.equity.to_csv("Date")),
        ("prices.csv".into(), prices.to_csv("Date")),
        ("risk_summary.csv".into(), risk_summary(analysis).to_csv(None)),
        ("risk_contributions.csv".into(), risk_contributions(analysis).to_csv(None)),
    ];
    if let Some(records) = metadata.get("universe_coverage").filter(|value| story::truthy(Some(value))) {
        files.push(("universe_coverage.csv".into(), records_table(records.as_list().unwrap_or_default()).to_csv(None)));
    }
    if let Some(study) = parts.study {
        files.push(("backtest_metrics.csv".into(), study.metrics.to_csv(None)));
        files.push(("backtest_curve.csv".into(), study.equity.to_csv("Date")));
        for (i, name) in study.equity.columns.iter().enumerate() {
            let stem = format!("backtests/{:02}", i + 1);
            files.push((format!("{stem}_holdings.csv"), study.holdings[name].to_csv(None)));
            files.push((format!("{stem}_allocations.csv"), study.allocations[name].to_csv("")));
            files.push((format!("{stem}_trades.csv"), study.trades[name].to_csv(None)));
        }
    }
    if let Some(profiles) = parts.latest_profiles {
        files.push(("latest_profile_summary.csv".into(), profiles.summary.to_csv(None)));
        files.push(("latest_profile_weights.csv".into(), weights_frame(&profiles.portfolios).to_csv(None)));
        files.push(("latest_profile_windows.csv".into(), profiles.windows.to_csv(None)));
        files.push(("latest_profile_window_returns.csv".into(), profiles.window_returns.to_csv(None)));
        files.push(("latest_profile_frontier.csv".into(), profiles.frontier.to_csv(None)));
        files.push(("latest_profile_frontier_weights.csv".into(), profiles.frontier_weights.to_csv(None)));
    }
    if let Some(benchmarks) = parts.benchmarks {
        files.push(("benchmark_prices.csv".into(), benchmarks.prices.to_csv("Date")));
        files.push(("benchmark_holdout_metrics.csv".into(), benchmarks.holdout_metrics.to_csv(None)));
        files.push(("benchmark_holdout_curve.csv".into(), benchmarks.holdout_equity.to_csv("Date")));
        files.push(("benchmark_training_estimates.csv".into(), benchmarks.training_estimates.to_csv(None)));
        files.push(("benchmark_latest_estimates.csv".into(), benchmarks.latest_estimates.to_csv(None)));
        if let (Some(equity), Some(metrics)) = (&benchmarks.backtest_equity, &benchmarks.backtest_metrics) {
            files.push(("benchmark_backtest_metrics.csv".into(), metrics.to_csv(None)));
            files.push(("benchmark_backtest_curve.csv".into(), equity.to_csv("Date")));
        }
    }
    if let Some(evidence) = parts.evidence {
        files.push(("evidence_summary.csv".into(), evidence.summary.to_csv(None)));
        files.push(("evidence_windows.csv".into(), evidence.windows.to_csv(None)));
        files.push(("evidence_relative_curve.csv".into(), evidence.relative_equity.to_csv(Some("Date"))));
    }
    let mut entries = vec![("report.html".to_string(), report_html(analysis, metadata, parts, Some(prices))?)];
    if let Some(study) = parts.study {
        let source = metadata.get("source").map_or_else(|| "Unspecified source".to_string(), Json::py_str);
        entries.push(("findings.md".into(), findings_markdown(study, &source, parts.benchmarks, parts.evidence)));
    }
    entries.push((
        "THIRD_PARTY_NOTICES.txt".into(),
        format!("Plotly.js (embedded in report.html)\n\n{PLOTLY_JS_LICENSE}"),
    ));
    entries.extend(files);
    let mut details = self::metadata(analysis, metadata, parts);
    if parts.selected_profile.is_some() || parts.selected_method.is_some() {
        details.insert(
            "report_view",
            crate::dict! {
                "risk_level" => parts.selected_profile.unwrap_or("Medium"),
                "rule" => parts.selected_method.unwrap_or("Expanding window"),
            },
        );
    }
    entries.push(("metadata.json".into(), ascii_json(&Json::Dict(details).dumps(Some(2)))));
    write_zip(&entries)
}
