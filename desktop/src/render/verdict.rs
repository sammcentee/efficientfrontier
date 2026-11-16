//! The out-of-sample verdict: does the method earn alpha, and if not, why not.
//!
//! Every sentence of the explanation depends on a number of the study, so a
//! study whose numbers differ gets a different explanation.

use efficient_frontier::alpha::{Alpha, MeanNoise, mean_noise, out_of_sample_alpha, rank_persistence};
use efficient_frontier::pyfmt::{self, Json, format_int};
use efficient_frontier::selection::{BASELINE, benchmark_name, latest_eligible_symbols};
use efficient_frontier::selection_report::{OutOfSampleFindings, join, out_of_sample_findings, percent, points};

use super::format::period_word;
use super::html::{Alert, alert, alert_plain, caption, esc, latex, md, stat_strip, title_html};
use super::overview::budget_note;
use super::table::Grid;
use crate::build::Applied;
use crate::session::Session;

/// A t-statistic at least this large in absolute value counts as distinguishable from chance.
const SIGNIFICANT: f64 = 2.0;
/// Above this R², the portfolio and equal-weight stocks have the same returns.
const SAME_RETURNS: f64 = 1.0 - 1e-9;
/// Below this many stocks, the share of real differences is a rough estimate.
const FEW_STOCKS: usize = 20;
const TITLE: &str = "Does the method beat simple alternatives?";

/// What the walk-forward tests show about the optimizer's skill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// A significant positive alpha against equal weights of the same stocks.
    Skill,
    /// No significant positive alpha against equal weights of the same stocks.
    NoSkill,
}

/// The verdict section and its verdict. No verdict when the tests cannot judge the method.
pub struct Evidence {
    pub html: String,
    pub verdict: Option<Verdict>,
}

/// The numbers that the explanation quotes.
struct Facts<'a> {
    findings: &'a OutOfSampleFindings,
    equal: &'a Alpha,
    market: Option<&'a Alpha>,
    noise: Option<MeanNoise>,
    persistence: Vec<f64>,
    universe: &'a str,
    /// The return interval: daily, weekly, monthly, or periodic.
    period: &'static str,
}

/// A rate with its sign and a true minus sign.
fn signed(value: f64) -> String {
    pyfmt::format(value, "+.1%").replace('-', "−")
}

/// A plain number with two decimals and a true minus sign.
fn fixed(value: f64) -> String {
    pyfmt::format(value, ".2f").replace('-', "−")
}

/// How the text names a reference path.
fn reference_name(name: &str) -> String {
    if name == BASELINE { "equal-weight stocks".into() } else { name.into() }
}

/// The alpha against SPY, if the study has SPY prices.
fn market_alpha(alphas: &[Alpha]) -> Option<&Alpha> {
    alphas.iter().find(|alpha| Some(alpha.reference.as_str()) == benchmark_name("SPY"))
}

fn beats_market(alphas: &[Alpha]) -> bool {
    market_alpha(alphas).is_some_and(|alpha| alpha.t_statistic() >= SIGNIFICANT)
}

/// The verdict and its headline. Skill needs a significant positive alpha against
/// equal weights of the same stocks, because SPY and QQQ hold other companies.
/// A significant loss against equal weights comes before a win against SPY.
fn judge(equal: &Alpha, alphas: &[Alpha]) -> (Verdict, String) {
    if equal.t_statistic() >= SIGNIFICANT {
        return (Verdict::Skill, "Out of sample, the method beat equal weights of the same stocks.".to_string());
    }
    let lagging: Vec<String> = alphas
        .iter()
        .filter(|alpha| alpha.t_statistic() <= -SIGNIFICANT)
        .map(|alpha| reference_name(&alpha.reference))
        .collect();
    let headline = if equal.t_statistic() <= -SIGNIFICANT || (!lagging.is_empty() && !beats_market(alphas)) {
        format!("Out of sample, the method fell behind {} after allowing for risk.", join(&lagging))
    } else if beats_market(alphas) {
        "The stocks beat SPY out of sample, but the optimizer added no reliable value.".to_string()
    } else {
        "Out of sample, the method shows no reliable alpha.".to_string()
    };
    (Verdict::NoSkill, headline)
}

/// The section of the walk-forward tests, or the reason that there is no verdict.
pub fn evidence(session: &Session, applied: &Applied) -> Evidence {
    let title = title_html(TITLE, "Alpha in the walk-forward tests, on prices that each fit did not see");
    let result = match &applied.out_of_sample {
        Ok(result) => result,
        Err(message) => {
            let html =
                title + &alert_plain(Alert::Info, &format!("This history is too short to test the method. {message}"));
            return Evidence { html, verdict: None };
        }
    };
    let alphas = out_of_sample_alpha(result);
    let (Some(findings), Some(equal)) =
        (out_of_sample_findings(result), alphas.iter().find(|alpha| alpha.reference == BASELINE))
    else {
        let html = title
            + &alert_plain(
                Alert::Warning,
                "No walk-forward window had a feasible fit, so there is no out-of-sample return to test.",
            );
        return Evidence { html, verdict: None };
    };
    let study = &applied.selection;
    let period = period_word(study.settings.f64("periods_per_year"));
    let regression = regression(session, applied, equal, findings.completed, period);
    let mut html = title;
    if equal.r_squared > SAME_RETURNS {
        html.push_str(&alert(
            Alert::Info,
            "The holding limit forces equal weights, so the portfolio has the same returns as equal-weight stocks. \
             A test of the optimizer needs weights that differ from that reference.",
        ));
        html.push_str(&alpha_table(&alphas));
        html.push_str(&regression);
        return Evidence { html, verdict: None };
    }
    let (verdict, headline) = judge(equal, &alphas);
    let market = market_alpha(&alphas);
    html.push_str(&format!(r#"<h2 class="pl-oos-title">{}</h2>"#, esc(&headline)));
    html.push_str(&md("Alpha is the part of a return that a reference does not explain. \
         The regression below removes the portfolio's exposure to each reference, \
         so the portfolio cannot win only because it took more market risk. \
         A t-statistic of about 2 or more is the usual sign that an estimate is not chance."));
    html.push_str(&strip(&findings, equal, market));
    html.push_str(&alpha_table(&alphas));
    html.push_str(&regression);
    let facts = Facts {
        findings: &findings,
        equal,
        market,
        noise: latest_noise(applied),
        persistence: rank_persistence(&applied.prices, result),
        universe: applied.metadata.get("universe").and_then(Json::as_str).unwrap_or(""),
        period,
    };
    match verdict {
        Verdict::NoSkill => {
            html.push_str(&md("### Why the test finds no reliable alpha"));
            html.push_str(&md(&numbered(&no_skill_reasons(&facts))));
            let spy = if beats_market(&alphas) {
                " Its lead over SPY therefore does not show skill in the choice of stocks."
            } else {
                ""
            };
            html.push_str(&md(&format!(
                "**Conclusion:** in this history, the optimizer's choices did not earn a reliable return \
                 beyond equal weights of the same stocks.{spy} The latest holdings near the end of this page \
                 describe the method. They are not a forecast."
            )));
        }
        Verdict::Skill => {
            html.push_str(&md("### Why this is still weak evidence"));
            html.push_str(&md(&numbered(&skill_caveats(&facts))));
            html.push_str(&md(
                "**Conclusion:** the walk-forward tests found a positive alpha estimate against equal-weight stocks. \
                 Treat it as a hypothesis to check on other data, not as proof.",
            ));
        }
    }
    Evidence { html, verdict: Some(verdict) }
}

fn strip(findings: &OutOfSampleFindings, equal: &Alpha, market: Option<&Alpha>) -> String {
    let t = |alpha: &Alpha| format!("Per year · t = {}", fixed(alpha.t_statistic()));
    let equal_note = t(equal);
    let market_note = market.map(t).unwrap_or_default();
    let edge_note = format!("In sample: {}", points(findings.margin_in_sample));
    let mut stats = vec![(signed(equal.annual), "Alpha vs equal weight", equal_note.as_str())];
    if let Some(alpha) = market {
        stats.push((signed(alpha.annual), "Alpha vs SPY", market_note.as_str()));
    }
    stats.push((points(findings.margin_out_of_sample), "Edge over equal weight", edge_note.as_str()));
    stat_strip(&stats)
}

fn alpha_table(alphas: &[Alpha]) -> String {
    let head = ["Reference", "Alpha per year", "Standard error", "t-statistic", "Beta", "R²"];
    let mut grid = Grid::new(head.iter().map(|label| esc(label)).collect(), 1).right_align(1..6);
    for alpha in alphas {
        grid.push(vec![
            esc(&alpha.reference),
            esc(&signed(alpha.annual)),
            esc(&percent(alpha.standard_error, 1)),
            esc(&fixed(alpha.t_statistic())),
            esc(&fixed(alpha.beta)),
            esc(&percent(alpha.r_squared, 0)),
        ]);
    }
    format!(r#"<div id="alpha_table">{}</div>"#, grid.html())
}

fn regression(session: &Session, applied: &Applied, equal: &Alpha, windows: usize, period: &str) -> String {
    let study = &applied.selection;
    let budget = match &applied.out_of_sample {
        Ok(result) => budget_note(session, study, &result.windows),
        Err(_) => String::new(),
    };
    let parts = [
        latex(concat!(
            r"r_{P,t}-r_f=\alpha+\beta\,(r_{B,t}-r_f)+\varepsilon_t,\qquad ",
            r"\alpha_{\rm year}=A\,\hat\alpha,\qquad t=\frac{\hat\alpha}{\operatorname{SE}(\hat\alpha)}"
        )),
        md(&format!(
            concat!(
                r"$\boldsymbol{{r_{{P,t}}}}:$ {} return of the stock portfolio on test date $\boldsymbol t$; ",
                r"$\boldsymbol{{r_{{B,t}}}}:$ the same for a reference; ",
                r"$\boldsymbol{{r_f}}:$ risk-free rate per return interval, {} a year; ",
                r"$\boldsymbol{{A={}}}:$ observations per year; $\boldsymbol\beta:$ exposure to the reference; ",
                r"$\boldsymbol{{\varepsilon_t}}:$ the part that the reference does not explain; ",
                r"$\boldsymbol{{\operatorname{{SE}}}}:$ standard error."
            ),
            period,
            percent(study.settings.f64("risk_free_rate"), 1),
            pyfmt::g(study.settings.f64("periods_per_year"))
        )),
        caption(&format!(
            "The regression joins the {} {period} returns of the {windows} walk-forward test windows, \
             from each purchase close. Each window holds the weights that its own fit chose, \
             and no stock return in the regression was known to that fit. {budget}\
             The standard errors are ordinary least squares errors. They assume independent errors with constant \
             variance and one beta for all windows, although each window holds different stocks. \
             Treat them as approximate.",
            format_int(equal.observations as i64, ",")
        )),
    ];
    parts.concat()
}

/// [`MeanNoise`] of the stocks that passed the screen of the latest fit.
fn latest_noise(applied: &Applied) -> Option<MeanNoise> {
    let study = &applied.selection;
    let prices = applied.prices.select(&latest_eligible_symbols(study))?;
    mean_noise(&prices, study.settings.f64("periods_per_year"))
}

/// Markdown lines `1. text`, `2. text`, ...
fn numbered(items: &[String]) -> String {
    items.iter().enumerate().map(|(i, item)| format!("{}. {item}", i + 1)).collect::<Vec<_>>().join("\n")
}

fn noise_reason(noise: &MeanNoise) -> String {
    let share = noise.signal_share();
    let measure = format!(
        "Across the {} eligible stocks, the fitted annual means have a standard deviation of {}. \
         Over {} years, chance alone gives each stock's mean an error of about {} against the average stock.",
        noise.stocks,
        percent(noise.spread, 1),
        pyfmt::format(noise.years, ".1f"),
        percent(noise.noise, 1)
    );
    let few = if noise.stocks < FEW_STOCKS {
        format!(" With only {} stocks, this share is a rough estimate.", noise.stocks)
    } else {
        String::new()
    };
    if share < 0.005 {
        format!(
            "**The spread in past mean returns looks like chance.** {measure} Chance alone can explain all of it.{few}"
        )
    } else if share < 0.5 {
        format!(
            "**Most of the spread in past mean returns is chance.** {measure} \
             Only about {} of the variance of the fitted means exceeds that noise.{few}",
            percent(share, 0)
        )
    } else {
        format!(
            "**Past mean returns differ by more than chance.** {measure} \
             About {} of the variance of the fitted means exceeds that noise. \
             That share is an upper limit, because a mean that changed during the history also counts.{few}",
            percent(share, 0)
        )
    }
}

fn persistence_reason(values: &[f64]) -> Option<String> {
    let average = values.iter().sum::<f64>() / values.len() as f64;
    if !average.is_finite() {
        return None;
    }
    let heading = if average < 0.1 {
        "Past winners did not stay ahead."
    } else if average < 0.3 {
        "Past winners kept only a small part of their lead."
    } else {
        "Past winners kept much of their lead."
    };
    let (low, high) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
    let range = if values.len() > 1 { format!(", from {} to {}", fixed(low), fixed(high)) } else { String::new() };
    Some(format!(
        "**{heading}** In the {} walk-forward windows, the rank correlation across stocks between the mean return \
         on the fit dates and the mean return on the later test dates averaged {}{range}. \
         A value of 0 means that the ranking of the fit told nothing about the later ranking. \
         A value of 1 means the same order.",
        values.len(),
        fixed(average)
    ))
}

fn margin_reason(findings: &OutOfSampleFindings) -> String {
    let numbers = format!(
        "The margin of the chosen weights over equal-weight stocks averaged {} a year on their fit dates \
         and {} on the later test dates.",
        points(findings.margin_in_sample),
        points(findings.margin_out_of_sample)
    );
    let fall = findings.margin_in_sample - findings.margin_out_of_sample;
    if fall > 0.0 {
        format!(
            "**The optimizer buys the largest errors.** It gives the most weight to the highest fitted means, \
             and the highest estimates usually include the most good luck. {numbers} \
             This fall of {} is the winner's curse of optimization, also called error maximization.",
            points(fall).trim_start_matches('+')
        )
    } else {
        format!(
            "**The margin did not fall.** {numbers} This history shows no winner's curse. \
             The next reason explains why the test still cannot separate the margin from chance."
        )
    }
}

fn power_reason(equal: &Alpha, period: &str) -> String {
    format!(
        "**This test cannot see a small alpha.** {} {period} returns give the alpha against equal-weight stocks \
         a standard error of {} a year, and the estimate was {}. \
         An alpha below about {} a year cannot be told apart from chance in this history.",
        format_int(equal.observations as i64, ","),
        percent(equal.standard_error, 1),
        signed(equal.annual),
        percent(2.0 * equal.standard_error, 1)
    )
}

fn market_reason(market: &Alpha, period: &str) -> Option<String> {
    (market.r_squared >= 0.5).then(|| {
        format!(
            "**Market moves drive most of the risk.** Out of sample, the portfolio's beta to SPY was {}, \
             and SPY explained {} of the variation of its {period} returns. \
             Gains that come from that market exposure are not alpha.",
            fixed(market.beta),
            percent(market.r_squared, 0)
        )
    })
}

/// How the stock list can favor stock paths, for each kind of universe. None for synthetic data.
fn universe_reason(universe: &str) -> Option<String> {
    const EITHER: &str = "That helps every stock path against SPY and QQQ. \
         Equal-weight stocks use the same list, but the bias can still favor either side of that comparison.";
    match universe {
        "Demo" => None,
        "S&P 500 + Nasdaq-100" | "Nasdaq-100" => Some(format!(
            "**The stock list favors survivors.** The list uses today's members, so it leaves out companies \
             that failed or left the index during the history. {EITHER}"
        )),
        _ => Some(format!(
            "**A fixed stock list can favor survivors.** The study keeps the same stocks for the whole history, \
             so it leaves out companies that a list made at the start would include. {EITHER}"
        )),
    }
}

fn no_skill_reasons(facts: &Facts) -> Vec<String> {
    let mut reasons: Vec<String> = facts.noise.as_ref().map(noise_reason).into_iter().collect();
    reasons.extend(persistence_reason(&facts.persistence));
    reasons.push(margin_reason(facts.findings));
    reasons.push(power_reason(facts.equal, facts.period));
    reasons.extend(facts.market.and_then(|market| market_reason(market, facts.period)));
    reasons.extend(universe_reason(facts.universe));
    reasons
}

fn skill_caveats(facts: &Facts) -> Vec<String> {
    let mut caveats = vec![
        "**One history.** The windows test one stretch of market history. Even with no skill, \
         a t-statistic of 2 or more happens about one time in 40, and more often when several settings are tried."
            .to_string(),
    ];
    caveats.extend(facts.noise.as_ref().map(noise_reason));
    caveats.extend(persistence_reason(&facts.persistence));
    caveats.extend(universe_reason(facts.universe));
    caveats.push(
        "**The settings matter.** The volatility budget and holding limit change which stocks the optimizer buys. \
         Check other settings before you rely on this result."
            .to_string(),
    );
    caveats
}

/// The note above the latest holdings, from the verdict of the tests.
pub fn holdings_note(verdict: Option<Verdict>) -> String {
    match verdict {
        Some(Verdict::NoSkill) => alert(
            Alert::Info,
            "The tests above found no reliable alpha. These weights show what the method chooses from all the history. \
             They are not evidence of future outperformance.",
        ),
        Some(Verdict::Skill) => alert(
            Alert::Info,
            "The tests above found a positive alpha estimate, within the limits that they state. \
             These weights are the method's latest choice, not a guarantee.",
        ),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPY: &str = "S&P 500 (SPY)";
    const QQQ: &str = "Nasdaq-100 (QQQ)";

    fn estimate(reference: &str, annual: f64, standard_error: f64) -> Alpha {
        Alpha { reference: reference.into(), annual, standard_error, beta: 1.0, r_squared: 0.8, observations: 1000 }
    }

    fn findings(margin_in_sample: f64, margin_out_of_sample: f64) -> OutOfSampleFindings {
        OutOfSampleFindings {
            completed: 5,
            windows: 5,
            in_sample: 0.4,
            out_of_sample: 0.1,
            change: -0.3,
            lower: 5,
            margin_in_sample,
            margin_out_of_sample,
            higher_than: Vec::new(),
            test_returns: 250,
        }
    }

    #[test]
    fn skill_needs_a_significant_alpha_against_equal_weights() {
        let alphas = [estimate(BASELINE, 0.06, 0.025), estimate(SPY, 0.01, 0.05)];
        let (verdict, headline) = judge(&alphas[0], &alphas);
        assert_eq!(verdict, Verdict::Skill);
        assert_eq!(headline, "Out of sample, the method beat equal weights of the same stocks.");
    }

    #[test]
    fn an_alpha_against_spy_alone_is_not_skill() {
        let alphas = [estimate(BASELINE, 0.02, 0.025), estimate(SPY, 0.12, 0.05)];
        let (verdict, headline) = judge(&alphas[0], &alphas);
        assert_eq!(verdict, Verdict::NoSkill);
        assert_eq!(headline, "The stocks beat SPY out of sample, but the optimizer added no reliable value.");
    }

    #[test]
    fn a_significant_loss_against_equal_weights_comes_before_a_win_against_spy() {
        let alphas = [estimate(BASELINE, -0.06, 0.025), estimate(SPY, 0.12, 0.05)];
        let (verdict, headline) = judge(&alphas[0], &alphas);
        assert_eq!(verdict, Verdict::NoSkill);
        assert_eq!(headline, "Out of sample, the method fell behind equal-weight stocks after allowing for risk.");
    }

    #[test]
    fn significant_negative_alphas_name_each_reference() {
        let alphas = [estimate(BASELINE, -0.06, 0.025), estimate(SPY, -0.02, 0.05), estimate(QQQ, -0.12, 0.05)];
        let (_, headline) = judge(&alphas[0], &alphas);
        assert_eq!(
            headline,
            "Out of sample, the method fell behind equal-weight stocks and Nasdaq-100 (QQQ) after allowing for risk."
        );
    }

    #[test]
    fn insignificant_or_undefined_alphas_show_no_reliable_alpha() {
        for equal in [estimate(BASELINE, 0.04, 0.025), estimate(BASELINE, 0.0, 0.0), estimate(BASELINE, 0.01, f64::NAN)]
        {
            let alphas = [equal.clone(), estimate(SPY, -0.03, 0.05)];
            assert_eq!(
                judge(&alphas[0], &alphas),
                (Verdict::NoSkill, "Out of sample, the method shows no reliable alpha.".to_string())
            );
        }
    }

    #[test]
    fn the_noise_reason_follows_the_share_of_real_differences() {
        let noise = MeanNoise { stocks: 400, years: 10.0, spread: 0.10, noise: 0.08 };
        let text = noise_reason(&noise);
        // 1 - 0.08² / 0.10² = 36%.
        assert!(text.starts_with("**Most of the spread in past mean returns is chance.**"), "{text}");
        assert!(text.contains("400 eligible stocks") && text.contains("10.0%") && text.contains("8.0%"));
        assert!(text.contains("Only about 36% of the variance"));
        let strong = noise_reason(&MeanNoise { noise: 0.05, ..noise.clone() });
        assert!(
            strong.starts_with("**Past mean returns differ by more than chance.**") && strong.contains("About 75%")
        );
        assert!(!strong.contains("Only"));
        let none = noise_reason(&MeanNoise { noise: 0.12, ..noise.clone() });
        assert!(none.contains("Chance alone can explain all of it."));
        let few = noise_reason(&MeanNoise { stocks: 6, ..noise });
        assert!(few.ends_with("With only 6 stocks, this share is a rough estimate."));
    }

    #[test]
    fn the_winners_curse_appears_only_when_the_margin_falls() {
        let fell = margin_reason(&findings(0.25, -0.012));
        assert!(fell.contains("+25.00 pp") && fell.contains("−1.20 pp") && fell.contains("This fall of 26.20 pp"));
        let rose = margin_reason(&findings(0.0305, 0.0408));
        assert!(rose.starts_with("**The margin did not fall.**") && !rose.contains("error maximization"));
        assert!(margin_reason(&findings(-0.02, -0.03)).contains("averaged −2.00 pp a year on their fit dates"));
    }

    #[test]
    fn reasons_quote_the_study_numbers() {
        let margins = findings(0.25, -0.012);
        let equal = estimate(BASELINE, 0.01, 0.031);
        let market = estimate(SPY, 0.02, 0.05);
        let facts = Facts {
            findings: &margins,
            equal: &equal,
            market: Some(&market),
            noise: Some(MeanNoise { stocks: 400, years: 10.0, spread: 0.10, noise: 0.08 }),
            persistence: vec![0.02, -0.05, 0.08],
            universe: "S&P 500 + Nasdaq-100",
            period: "weekly",
        };
        let reasons = no_skill_reasons(&facts);
        assert_eq!(reasons.len(), 6);
        assert!(reasons[1].starts_with("**Past winners did not stay ahead.**"));
        assert!(reasons[1].contains("In the 3 walk-forward windows"));
        assert!(reasons[1].contains("averaged 0.02, from −0.05 to 0.08"));
        assert!(
            reasons[3].contains("1,000 weekly returns") && reasons[3].contains("3.1%") && reasons[3].contains("6.2%")
        );
        assert!(reasons[3].contains("the estimate was +1.0%"));
        assert!(reasons[4].contains("beta to SPY was 1.00"));
        assert!(reasons[4].contains("explained 80% of the variation of its weekly returns"));
        assert!(reasons[5].starts_with("**The stock list favors survivors.**"));
        let weak = Alpha { r_squared: 0.1, ..estimate(SPY, 0.0, 0.05) };
        let reasons = no_skill_reasons(&Facts { market: Some(&weak), universe: "Demo", ..facts });
        assert_eq!(reasons.len(), 4, "no market reason at a low R², no survivor reason for synthetic data");
    }

    #[test]
    fn the_survivor_reason_names_the_kind_of_list() {
        assert!(universe_reason("Demo").is_none());
        assert!(universe_reason("Nasdaq-100").unwrap().contains("today's members"));
        for fixed_list in ["Upload CSV", "My tickers", "Original 60 holdings"] {
            assert!(universe_reason(fixed_list).unwrap().starts_with("**A fixed stock list can favor survivors.**"));
        }
    }
}
