//! The overview page (`overview`, `out_of_sample_section`, `selection_period`,
//! `selection_replay_explanation`, `budget_note`, `shared_fit_note`).

use efficient_frontier::data::stale_price_notes;
use efficient_frontier::frame::{Cell, Date, Table};
use efficient_frontier::out_of_sample::OutOfSampleStudy;
use efficient_frontier::pyfmt::{self, Json, format_int};
use efficient_frontier::selection::{SELECTED, SelectionStudy};
use efficient_frontier::selection_report::{
    OUT_OF_SAMPLE_TITLE, REPLAY_LABEL, SELECTION_METRIC_LABELS, out_of_sample_chart, out_of_sample_findings,
    out_of_sample_format, out_of_sample_frame, out_of_sample_paths_frame, out_of_sample_text, percent, points,
    selection_benchmark_note, selection_chart, selection_comparison_html, selection_frontier_chart,
    selection_frontier_summary_html, selection_metrics_frame, selection_replay_dominance,
    selection_replay_findings_html, selection_replay_return_bridge, selection_replay_verdict_html,
};
use efficient_frontier::story::{self, Company};
use indexmap::IndexMap;
use serde_json::json;

use super::format::{day, day_text, percent_column, percent_or_dash, replay_points};
use super::holding_label;
use super::html::{
    Alert, KEY_LEGEND, alert, alert_plain, caption, chart, esc, expander, failed, guarded, latex, md, plain,
    plain_caption, stat_strip, title_html,
};
use super::method::{fit_math, replay_math};
use super::table::{Grid, cell, default_format, same_label, table};
use crate::build::{Applied, displayed_weights, replay_symbols};
use crate::guard::guard;
use crate::session::Session;
use crate::state::round2;

/// A tile: `st.container(border=True, key=...)`.
fn tile(key: &str, body: &str) -> String {
    format!(r#"<section class="pl-tile" id="{key}">{body}</section>"#)
}

/// The company identity of every price column, as `companies` in `overview()`.
fn companies(session: &Session, applied: &Applied) -> (IndexMap<String, Company>, String) {
    let metadata = &applied.metadata;
    let universe = metadata.get("universe").and_then(Json::as_str).unwrap_or("");
    let (names, problem) = match guard(|| story::company_names(session.coverage.as_deref())) {
        Ok(names) => (names, String::new()),
        Err(message) => (IndexMap::new(), failed(&message)),
    };
    let mut companies = IndexMap::new();
    for symbol in &applied.prices.columns {
        let profile = applied.profiles.get(symbol).cloned().unwrap_or_default();
        let company = if universe == "Demo" {
            let label = symbol.strip_prefix("DEMO_").unwrap_or(symbol).replace('_', " ").to_lowercase();
            Company {
                name: format!("Simulated {label}"),
                description: format!("Synthetic {label} prices for the offline demonstration; not an actual company."),
                source_url: profile.source_url,
            }
        } else {
            let name = if profile.name.is_empty() {
                names.get(symbol).cloned().unwrap_or_else(|| symbol.clone())
            } else {
                profile.name
            };
            let description = if !profile.description.is_empty() {
                profile.description
            } else if universe == "Upload CSV" {
                "Company description unavailable. This uploaded file contains price data only.".to_string()
            } else {
                "Company description unavailable from the data provider.".to_string()
            };
            Company { name, description, source_url: profile.source_url }
        };
        companies.insert(symbol.clone(), company);
    }
    (companies, problem)
}

/// "Latest fitted holdings": company cards, every weight and stale-price notes.
fn holdings_tile(applied: &Applied, companies: &IndexMap<String, Company>) -> String {
    let study = &applied.selection;
    let prices = &applied.prices;
    let source = applied.metadata.get("source").and_then(Json::as_str).unwrap_or("");
    let last = *prices.index.last().expect("prices have dates");
    let mut html = title_html("Latest fitted holdings", &format!("Fitted through {}", day(last)));
    if source.contains("synthetic") {
        html.push_str(&alert(
            Alert::Info,
            "Demo · synthetic prices. These are not market results. The six-asset demo uses a 20% holding limit by default.",
        ));
    }
    html.push_str(&caption(
        "These weights use all supplied prices, including the replay dates. \
         They form the selected point on the frontier below. \
         The historical replay used a separate initial allocation.",
    ));
    let displayed = displayed_weights(study);
    html.push_str(&guarded(|| story::holdings_cards_html(&displayed, companies)));
    if applied.profiles.values().any(|profile| !profile.source_url.is_empty()) {
        html.push_str(&caption(
            "Available company descriptions are current Yahoo Finance profiles, not historical descriptions. \
             Select a company name to open its source.",
        ));
    }
    let weights = &study.latest[&study.selected].weights;
    let mut grid = Grid::new(vec!["Symbol".into(), "Company".into(), "Weight".into()], 0).right_align([2]);
    for i in weights.descending_order() {
        let symbol = &weights.labels[i];
        let name = companies.get(symbol).map_or("", |company| company.name.as_str());
        grid.push(vec![esc(symbol), esc(name), percent_column(weights.values[i])]);
    }
    html.push_str(&expander("All weights, including negligible allocations", &grid.html()));
    for note in stale_price_notes(prices) {
        html.push_str(&alert_plain(Alert::Warning, &note));
    }
    html
}

/// "Full-history fit": the latest frontier and its summary.
fn frontier_tile(applied: &Applied) -> String {
    let study = &applied.selection;
    let settings = &study.settings;
    let mut html = md(&format!(
        "**Full-history fit · {} to {}.** The selected point represents the holdings above. \
         This fit includes the replay dates; the replay below uses a separate initial allocation.",
        day_text(settings.str("initial_training_start")),
        day_text(settings.str("price_end"))
    ));
    match &applied.frontier {
        Ok(frontier) => {
            html.push_str(&guarded(|| chart(&selection_frontier_chart(frontier, study), "pl-chart-frontier")));
            html.push_str(&guarded(|| selection_frontier_summary_html(study, &applied.prices, frontier)));
        }
        Err(message) => html.push_str(&failed(message)),
    }
    if let Some(note) = applied.metadata.get("candidate_note").and_then(Json::as_str) {
        html.push_str(&alert_plain(Alert::Info, note));
    }
    html
}

/// The dates of a window column.
fn window_dates(windows: &Table, column: &str) -> Vec<Date> {
    windows.column(column).iter().filter_map(Cell::as_date).collect()
}

/// `budget_note`: say when the volatility budget is the QQQ default, whose year
/// ends after walk-forward fits.
pub fn budget_note(session: &Session, study: &SelectionStudy, windows: &Table) -> String {
    let Some(reference) = &session.qqq_reference else { return String::new() };
    if (study.settings.f64("target_volatility") - round2(reference.volatility * 100.0) / 100.0).abs() >= 1e-12 {
        return String::new();
    }
    let end = reference.last_return_date;
    let fit_ends = window_dates(windows, "fit_end");
    let (Some(first), Some(last)) = (fit_ends.iter().min(), fit_ends.iter().max()) else { return String::new() };
    if end <= *first {
        return String::new();
    }
    let which = if end > *last { "every walk-forward fit" } else { "some walk-forward fits" };
    format!(
        "The volatility budget equals QQQ's volatility over the year through {}. \
         That year ends after {which}, so those fits use one input from later dates. ",
        day(end)
    )
}

/// `shared_fit_note`: name the window whose fit is the replay's fit.
pub fn shared_fit_note(study: &SelectionStudy, windows: &Table) -> String {
    let fit_end = Date::parse_from_str(study.settings.str("initial_training_end"), "%Y-%m-%d").ok();
    let fit_ends = windows.column("fit_end");
    windows.index[0]
        .iter()
        .zip(fit_ends)
        .find(|(_, end)| end.as_date().is_some() && end.as_date() == fit_end)
        .map(|(number, _)| {
            format!(
                "Window {} uses the same fit as the replay. The replay holds it longer and does not refit.",
                number.py_str()
            )
        })
        .unwrap_or_default()
}

/// A walk-forward table with the formats of `out_of_sample_formats`.
fn out_of_sample_table(frame: &Table, index: &[&str]) -> String {
    let format = |name: &str, value: &Cell| match (out_of_sample_format(name), value.as_f64()) {
        (Some(formatter), Some(number)) if name != "_index" => formatter(number),
        _ => cell(value),
    };
    table(frame, Some(index), &same_label, &format).html()
}

/// `out_of_sample_section`: the walk-forward tests in and out of sample.
fn out_of_sample_section(session: &Session, applied: &Applied) -> String {
    let study = &applied.selection;
    let result: &OutOfSampleStudy = match &applied.out_of_sample {
        Ok(result) => result,
        Err(message) => return title_html(OUT_OF_SAMPLE_TITLE, "") + &alert_plain(Alert::Info, message),
    };
    let windows = &result.windows;
    let share = 1.0 - result.settings.f64("first_fit_fraction");
    let mut html = title_html(
        OUT_OF_SAMPLE_TITLE,
        &format!("{} walk-forward tests · each fit uses only earlier stock prices", windows.nrows()),
    );
    let findings = match guard(|| out_of_sample_findings(result)) {
        Ok(Some(findings)) => findings,
        Ok(None) => {
            html.push_str(&alert(
                Alert::Warning,
                "No window had a feasible fit, so there is no in-sample or out-of-sample result.",
            ));
            for (number, status) in windows.index[0].iter().zip(windows.column("status")) {
                html.push_str(&plain_caption(&format!("Window {}: {}", number.py_str(), status.py_str())));
            }
            return html;
        }
        Err(message) => return html + &failed(&message),
    };
    let strip = stat_strip(&[
        (percent(findings.in_sample, 2), "In sample", "Average annual mean while held"),
        (percent(findings.out_of_sample, 2), "Out of sample", "Average annual mean while held"),
        (points(findings.change), "Change", "Out of sample minus in sample"),
    ]);
    match guard(|| out_of_sample_text(&findings)) {
        Ok((headline, lines)) => {
            html.push_str(&format!(r#"<h2 class="pl-oos-title">{}</h2>"#, esc(&headline)));
            html.push_str(&strip);
            html.push_str(&md(&lines.join(" ")));
        }
        Err(message) => html.push_str(&(failed(&message) + &strip)),
    }
    html.push_str(&guarded(|| chart(&out_of_sample_chart(result), "pl-chart-oos")));
    html.push_str(&guarded(|| out_of_sample_table(&out_of_sample_frame(result), &["Window"])));
    let periods = study.settings.f64("periods_per_year");
    let months = 12.0 * findings.test_returns as f64 / periods;
    let length = if months >= 1.0 {
        let plural = if months.round_ties_even() != 1.0 { "s" } else { "" };
        format!("roughly {} month{plural}", pyfmt::format(months, ".0f"))
    } else {
        "less than one month".to_string()
    };
    let first_fit = window_dates(windows, "fit_start").first().map(|date| day(*date)).unwrap_or_default();
    html.push_str(&caption(&format!(
        "Each fit starts on {first_fit} and ends at the close before its test dates. \
         Each test window has about {} returns, {length}. Annual means from short windows are noisy. \
         Both sides are measured from the purchase close, so the cash interval and the entry fee do not enter these figures. \
         They therefore differ from the fitted mean and from the replay below, which includes its entry interval. {}",
        format_int(findings.test_returns, ","),
        shared_fit_note(study, windows)
    )));
    let paths = guarded(|| {
        let paths = out_of_sample_paths_frame(result);
        let names: Vec<String> = paths.index_names.iter().map(|name| name.clone().unwrap_or_default()).collect();
        let labels: Vec<&str> = names.iter().map(String::as_str).collect();
        out_of_sample_table(&paths, &labels)
    });
    html.push_str(&expander(
        "Every path in each window",
        &(paths
            + &caption(
                "Annual mean and realized annual volatility of each buy-and-hold path, measured from the purchase close. \
                 The budget limits fitted volatility at each fit. Realized volatility can be higher on either side.",
            )),
    ));
    for (number, status) in windows.index[0].iter().zip(windows.column("status")) {
        let status = status.py_str();
        if !status.is_empty() {
            html.push_str(&alert_plain(Alert::Warning, &format!("Window {} has no result. {status}", number.py_str())));
        }
    }
    if !result.notes.is_empty() {
        let notes: String = result.notes.iter().map(|note| plain(note)).collect();
        html.push_str(&expander("Notes from the walk-forward fits", &notes));
    }
    let math = [
        md("**How each test works**"),
        md(&format!(
            "Window k fits the model on returns 1 to sₖ only. Interval sₖ+1 stays in cash. \
             The portfolio buys at close sₖ+1, pays the entry fee, and holds to close eₖ with no later trades. \
             The next window adds those returns to its fit and repeats. \
             The {} test windows do not overlap. Together they cover the final {} of returns.",
            windows.nrows(),
            pyfmt::format(share, ".0%")
        )),
        latex(concat!(
            r"\underbrace{r_1,\;\dots,\;r_{s_k}}_{\text{fit: choose }w_k}\;\Big|\;",
            r"\underbrace{r_{s_k+1}}_{\text{cash, buy}}\;\Big|\;",
            r"\underbrace{r_{s_k+2},\;\dots,\;r_{e_k}}_{\text{test: hold }w_k}"
        )),
        md(
            "The in-sample result buys the same weights wₖ at close 1, after one cash interval, and holds them to close sₖ. \
            Both sides use the same rule. Both are measured from the purchase close.",
        ),
        latex(concat!(
            r"\mu^{\rm in}_k=\frac{A}{s_k-1}\sum_{t=2}^{s_k}R^{(k)}_t,\qquad ",
            r"\mu^{\rm out}_k=\frac{A}{e_k-s_k-1}\sum_{t=s_k+2}^{e_k}R^{(k)}_t"
        )),
        latex(&format!(
            r"\bar\Delta=\frac{{1}}{{|C|}}\sum_{{k\in C}}\left(\mu^{{\rm out}}_k-\mu^{{\rm in}}_k\right)\approx {}\;\text{{pp}},\qquad |C|={}",
            replay_points(findings.change),
            findings.completed
        )),
        md(&format!(
            concat!(
                r"$\boldsymbol{{r_t}}:$ stock returns; $\boldsymbol{{w_k}}:$ weights fitted on returns 1 to $\boldsymbol{{s_k}}$; ",
                r"$\boldsymbol{{R^{{(k)}}_t}}:$ return of $\boldsymbol{{w_k}}$ bought once and held; ",
                r"$\boldsymbol{{A={}}}:$ observations per year; ",
                r"$\boldsymbol{{C}}:$ the windows with a feasible fit; **pp:** percentage points."
            ),
            pyfmt::g(periods)
        )),
        md("**Why in-sample results tend to look better**"),
        md("The optimizer favors stocks with high past means, within the volatility budget and holding limit. \
            Part of each past mean is chance. The optimizer cannot separate chance from a lasting return, so it buys both. \
            On its own fit dates, the allocation therefore tends to look better than it does later. \
            Equal-weight stocks use no return estimates, so their change between the two sides is a rough market control. \
            Their holdings differ from the stock portfolio, so the control is approximate."),
        caption(&format!(
            "Every window uses the same input stock list. A current membership list favors surviving companies. {}\
             You can already know these dates, so these are historical checks, not forecasts.",
            budget_note(session, study, windows)
        )),
    ];
    html.push_str(&format!(r#"<div class="pl-math" id="math_out_of_sample">{}</div>"#, math.concat()));
    html
}

/// The replay metrics table: the engine rows with the replay label.
fn metrics_table(study: &efficient_frontier::backtest::BacktestStudy) -> String {
    let columns = ["annual_mean", "cagr", "volatility", "max_drawdown"];
    let mut metrics = selection_metrics_frame(study).select(&columns);
    metrics.rename_index(&[(SELECTED, REPLAY_LABEL)]);
    let label = |name: &str| {
        Some(SELECTION_METRIC_LABELS.iter().find(|(key, _)| *key == name).map_or(name, |(_, label)| label).to_string())
    };
    let format = |name: &str, value: &Cell| match value {
        Cell::Float(number) if name != "_index" => percent_or_dash(*number),
        other => cell(other),
    };
    table(&metrics, Some(&["Rule or baseline"]), &label, &format).html()
}

/// `selection_period`: the replay dates, chart, headline numbers and metrics.
fn selection_period(applied: &Applied, title: &str, benchmark_note: &str) -> String {
    let study = &applied.selection;
    let test = &study.test;
    let settings = &study.settings;
    let train = settings.f64("train_fraction");
    let mut html = title_html(title, "");
    html.push_str(&md("**The chart tests the later period. The latest frontier fits the full history.**"));
    html.push_str(&format!(
        concat!(
            r#"<div class="pl-replay-dates" id="replay_dates"><div class="pl-date-windows">"#,
            r#"<article><h3>Initial fit · first {train}</h3><p>{fit_start} to {fit_end}</p><span>{fit_returns} returns choose the replay holdings.</span></article>"#,
            r#"<article><h3>Replay · final {rest}</h3><p>{test_start} to {test_end}</p><span>{test_returns} later returns form the replay chart and table.</span></article>"#,
            r#"<article><h3>Latest frontier · full history</h3><p>{fit_start} to {price_end}</p><span>A separate fit, including the replay dates.</span></article></div></div>"#,
        ),
        train = pyfmt::format(train, ".0%"),
        rest = pyfmt::format(1.0 - train, ".0%"),
        fit_start = day_text(settings.str("initial_training_start")),
        fit_end = day_text(settings.str("initial_training_end")),
        fit_returns = format_int(settings.int("initial_training_returns"), ","),
        test_start = day_text(settings.str("test_start")),
        test_end = day_text(settings.str("test_end")),
        test_returns = format_int(settings.int("test_returns"), ","),
        price_end = day_text(settings.str("price_end")),
    ));
    html.push_str(&caption(
        "Return dates shown. The split uses observations, not a fixed one-year test. \
         Annualized return and volatility describe units, not the chart’s duration. \
         All paths below use the same replay dates and entry fee rate.",
    ));
    html.push_str(&guarded(|| {
        let mut figure = selection_chart(test);
        figure.update_layout(json!({"height": 360, "legend": {"maxheight": 120}}));
        chart(&figure, "pl-chart-replay")
    }));
    let metric = |name: &str| pyfmt::format(test.metrics.f64_at(SELECTED, name), ".2%");
    html.push_str(&stat_strip(&[
        (metric("total_return"), "Net total return", "Over the replay period"),
        (metric("volatility"), "Realized volatility", "Annualized"),
        (metric("max_drawdown"), "Maximum drawdown", "Largest fall from a peak"),
    ]));
    html.push_str(&guarded(|| selection_replay_findings_html(test)));
    html.push_str(&guarded(|| selection_replay_verdict_html(test)));
    html.push_str(&guarded(|| metrics_table(test)));
    html.push_str(&caption(&format!(
        "All rows use the same dates and fee rate. Annual mean scales the average return to one year; \
         CAGR measures compound growth. {benchmark_note}"
    )));
    html
}

/// The dominance comparison of each reference with the stock portfolio.
fn dominance(test: &efficient_frontier::backtest::BacktestStudy) -> String {
    let comparisons = selection_replay_dominance(test);
    let labels = |classification: &str| match classification {
        "reference_dominates" => "Reference dominates the stock portfolio.",
        "portfolio_dominates" => "Stock portfolio dominates this reference.",
        "equal" => "Equal observed mean and volatility.",
        _ => "Higher mean comes with higher volatility. Neither dominates.",
    };
    let mean = comparisons.f64_at(SELECTED, "annual_mean");
    let volatility = comparisons.f64_at(SELECTED, "volatility");
    let mut html = String::new();
    for (row, name) in comparisons.index_labels().iter().enumerate() {
        if name == SELECTED {
            continue;
        }
        let classification = comparisons.column("classification")[row].py_str();
        html.push_str(&md(&format!("**{name}** · {}", labels(&classification))));
        html.push_str(&latex(&format!(
            r"\begin{{aligned}}\mu_B-\mu_P&\approx {}\;\text{{pp}}\\[4pt]\sigma_B-\sigma_P&\approx {}\;\text{{pp}}\end{{aligned}}",
            replay_points(comparisons.f64s("annual_mean")[row] - mean),
            replay_points(comparisons.f64s("volatility")[row] - volatility)
        )));
    }
    html
}

/// The numbers of the replay reconciliation.
fn bridge(applied: &Applied) -> String {
    let bridge = selection_replay_return_bridge(&applied.selection, &applied.prices);
    let terms = latex(&format!(
        concat!(
            r"\begin{{aligned}}",
            r"S&\approx {}\;\text{{pp}}&&\text{{stock means}}\\[5pt]",
            r"D&\approx {}\;\text{{pp}}&&\text{{weight drift}}\\[5pt]",
            r"E&\approx {}\;\text{{pp}}&&\text{{entry delay}}\\[5pt]",
            r"F&\approx {}\;\text{{pp}}&&\text{{entry fee}}",
            r"\end{{aligned}}"
        ),
        replay_points(bridge.mean_change),
        replay_points(bridge.weight_drift),
        replay_points(bridge.entry_delay),
        replay_points(bridge.entry_fee)
    ));
    let totals = latex(
        &format!(
            r"\begin{{aligned}}w^{{\mathsf T}}\widehat\mu_{{\rm train}}&\approx {}\\[5pt]\mu_P&\approx {}\\[5pt]S+D+E+F&\approx {}\;\text{{pp}}\end{{aligned}}",
            pyfmt::format(bridge.fitted_mean, ".2%"),
            pyfmt::format(bridge.realized_mean, ".2%"),
            replay_points(bridge.realized_mean - bridge.fitted_mean)
        )
        .replace('%', r"\%"),
    );
    terms + &totals
}

/// `selection_replay_explanation`: dominance and the reconciliation to the initial fit.
fn replay_explanation(applied: &Applied) -> String {
    let study = &applied.selection;
    let periods = pyfmt::g(study.settings.f64("periods_per_year"));
    let parts = [
        md("### Compare the replay with each reference"),
        md("The chart compares compound wealth. The efficient frontier compares arithmetic mean and volatility. \
            Use the same return measure on both sides before calling a result dominated."),
        latex(concat!(
            r"\begin{aligned}G_X&=\prod_{t=1}^{N}(1+R_{X,t})-1\\[6pt]",
            r"\mu_X&=\frac{A}{N}\sum_{t=1}^{N}R_{X,t}\\[6pt]",
            r"\sigma_X^2&=\frac{A}{N-1}\sum_{t=1}^{N}(R_{X,t}-\mu_X/A)^2\end{aligned}"
        )),
        md(&format!(
            concat!(
                r"$\boldsymbol{{X}}:$ a portfolio or reference; $\boldsymbol{{R_{{X,t}}}}:$ its net return in interval ",
                r"$\boldsymbol{{t}}$; $\boldsymbol{{N}}:$ replay intervals, including entry; ",
                r"$\boldsymbol{{A={}}}:$ observations per year. ",
                r"$\boldsymbol{{G}}:$ total return; $\boldsymbol{{\mu}}:$ annual arithmetic mean; ",
                r"$\boldsymbol{{\sigma}}:$ annual sample volatility. Below, ",
                r"$\boldsymbol{{P}}$ is the stock portfolio and $\boldsymbol{{B}}$ is each reference."
            ),
            periods
        )),
        guarded(|| dominance(&study.test)),
        md("**pp** means percentage points. Differences use unrounded results. \
            Positive mean and negative volatility differences favor the reference."),
        md("### Why an optimal fit can still lose"),
        md(
            "The optimizer guarantees the highest fitted mean only for the training estimates and feasible initial weights. \
            The vectors below contain only stocks that passed the initial eligibility screen:",
        ),
        latex(concat!(
            r"\begin{aligned}\mathcal F_{\rm train}=\{v:\;&\mathbf1^{\mathsf T}v=1,\quad0\le v_i\le h,\\",
            r"&v^{\mathsf T}\widehat\Sigma_{\rm train}v\le b^2\}\end{aligned}"
        )),
        latex(concat!(
            r"\widehat\mu_{\rm train}^{\mathsf T}w^*\ge",
            r"\widehat\mu_{\rm train}^{\mathsf T}v\qquad\forall v\in\mathcal F_{\rm train}"
        )),
        md(concat!(
            r"$\boldsymbol{w^*}:$ chosen initial weights; $\boldsymbol{v}:$ an alternative allocation; ",
            r"$\boldsymbol{\mathcal F_{\rm train}}:$ feasible allocations; $\boldsymbol{\mathbf1}:$ vector of ones; ",
            r"$\boldsymbol{h}:$ initial holding cap; $\boldsymbol{b}:$ fitted volatility budget. ",
            r"$\boldsymbol{\widehat\mu_{\rm train}}$ and $\boldsymbol{\widehat\Sigma_{\rm train}}$: ",
            r"training estimates of annual stock means and covariance; $\boldsymbol{\mathsf T}:$ transpose."
        )),
        md("Later stock means and covariances differ. Buy-and-hold weights also drift. \
            The training inequality therefore gives no ordering of later realized means. \
            SPY and QQQ are outside the eligible stock allocations. Equal-weight can exceed the fitted risk budget. \
            A losing replay does not, by itself, prove an optimizer error."),
        md("**Reconcile this replay to its initial fit**"),
        md("The following identity accounts for the full difference in annual arithmetic mean. \
            It uses the saved initial allocation, not the latest full-history weights."),
        latex(concat!(
            r"\begin{aligned}\mu_P-w^{\mathsf T}\widehat\mu_{\rm train}&=S+D+E+F\\[8pt]",
            r"S&=w^{\mathsf T}(\widehat\mu_{\rm test}-\widehat\mu_{\rm train})\\[6pt]",
            r"D&=\frac{A}{N}\sum_{t=2}^{N}(a_{t-1}-w)^{\mathsf T}r_t\\[6pt]",
            r"E&=-\frac{A}{N}w^{\mathsf T}r_1\\[6pt]",
            r"F&=-\frac{A}{N}\frac{c}{1+c}\end{aligned}"
        )),
        md(concat!(
            r"$\boldsymbol{w=w^*}:$ initial weights; $\boldsymbol{r_t}:$ stock return vector; ",
            r"$\boldsymbol{a_{t-1}}:$ actual weights before return $\boldsymbol{t}$; ",
            r"$\boldsymbol{\widehat\mu_{\rm test}=(A/N)\sum_{t=1}^{N}r_t}:$ stock means over all replay intervals. ",
            r"$\boldsymbol{t=1}:$ the cash interval before entry; $\boldsymbol{c}:$ entry fee per unit bought."
        )),
        guarded(|| bridge(applied)),
        md("To verify the identity, write each held-period return as initial weights times stock returns, \
            plus the effect of weight drift. Sum those returns, substitute the cash-entry return −c/(1+c), \
            and subtract the fitted mean. The four terms above follow exactly."),
        md("Negative terms reduce the replay mean relative to its initial fit. \
            Drift and the entry delay can help or hurt. The identical entry fee cancels when comparing the \
            portfolio’s annual mean with a reference. This is an accounting explanation, \
            not proof of what caused stock returns to change."),
    ];
    format!(r#"<div class="pl-math" id="math_diagnosis">{}</div>"#, parts.concat())
}

/// "Replay holdings and contributions": the bought stocks with their companies.
fn replay_holdings(applied: &Applied, companies: &IndexMap<String, Company>) -> String {
    let study = &applied.selection;
    let holdings = &study.test.holdings[&study.selected];
    let symbols = replay_symbols(study);
    let labels = holdings.index_labels();
    let rows: Vec<usize> =
        symbols.iter().filter_map(|symbol| labels.iter().position(|label| label == symbol)).collect();
    let bought = holdings.take_rows(&rows);
    let company = |field: fn(&Company) -> &str| -> Vec<Cell> {
        bought.index_labels().iter().map(|symbol| Cell::from(companies.get(symbol).map_or("", field))).collect()
    };
    let mut shown = bought.select::<&str>(&[]);
    shown.push("Company", company(|c| c.name.as_str()));
    shown.push("What it does", company(|c| c.description.as_str()));
    for (name, values) in bought.columns.iter().zip(&bought.data) {
        shown.push(name, values.clone());
    }
    let format = |name: &str, value: &Cell| match value {
        Cell::Float(number) if bought.position(name).is_some() => pyfmt::format(*number, ".2%"),
        other => cell(other),
    };
    table(&shown, Some(&["Symbol"]), &holding_label, &format).html()
}

/// `overview()`: holdings, frontier, walk-forward tests, replay and its explanation.
pub fn overview(session: &Session, applied: &Applied) -> String {
    let study = &applied.selection;
    let metadata = &applied.metadata;
    let (companies, problem) = companies(session, applied);
    let benchmark_note = guard(|| selection_benchmark_note(study));
    let mut html = tile("tile_holdings", &(problem + &holdings_tile(applied, &companies)));
    html.push_str(&tile("tile_fitted_frontier", &frontier_tile(applied)));
    html.push_str(r#"<div id="pl-market"></div>"#);
    html.push_str(&tile("tile_out_of_sample", &out_of_sample_section(session, applied)));
    let note = match &benchmark_note {
        Ok(note) => note.as_str(),
        Err(_) => "",
    };
    let mut replay = selection_period(applied, "Historical replay · buy and hold", note);
    if let Err(message) = &benchmark_note {
        replay.push_str(&failed(message));
    }
    replay.push_str(&replay_explanation(applied));
    html.push_str(&tile("tile_final_test", &replay));
    let drivers = guarded(|| story::result_drivers_html(&study.test, &study.selected))
        + &expander("Replay holdings and contributions", &guarded(|| replay_holdings(applied, &companies)));
    html.push_str(&tile("tile_replay_drivers", &drivers));
    let fit_replay =
        guarded(|| selection_comparison_html(study, &applied.prices)) + &fit_math(study) + &replay_math(study);
    html.push_str(&tile("tile_fit_replay", &fit_replay));
    if let Some(first) = study.warnings.first() {
        html.push_str(&alert_plain(Alert::Warning, first));
        let notes: String = study.warnings.iter().map(|note| plain(note)).collect();
        let eligibility = table(&study.eligibility, None, &same_label, &default_format).html();
        html.push_str(&expander("All data and model notes", &(notes + &eligibility)));
    }
    if let Some(limitation) = metadata.get("universe_limitation").and_then(Json::as_str).filter(|text| !text.is_empty())
    {
        html.push_str(&plain_caption(limitation));
    }
    html.push_str(
        r#"<form method="post" action="/view" class="pl-actions-left"><input type="hidden" name="view" value="Data &amp; method"><button type="submit" class="pl-btn pl-btn-secondary" id="open_data_method">Open Data &amp; method</button></form>"#,
    );
    html.push_str(KEY_LEGEND);
    html
}
