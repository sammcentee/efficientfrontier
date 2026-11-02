//! Plain-language story components, ported from `tests/test_story.py`.

use chrono::NaiveDate;
use efficient_frontier::backtest::BacktestStudy;
use efficient_frontier::benchmarks::BenchmarkComparison;
use efficient_frontier::core::Portfolio;
use efficient_frontier::data::read_csv_rows;
use efficient_frontier::dict;
use efficient_frontier::evidence::{EvidenceAnalysis, RelativeEquity};
use efficient_frontier::frame::{Cell, Date, Frame, Series, Table, cells};
use efficient_frontier::metrics::performance_metrics;
use efficient_frontier::plot::Figure;
use efficient_frontier::profiles::ProfileAnalysis;
use efficient_frontier::pyfmt::{Dict, Json, escape};
use efficient_frontier::story::{self, *};
use indexmap::IndexMap;
use nalgebra::DMatrix;
use serde_json::json;

const SPY: &str = "S&P 500 (SPY)";
const QQQ: &str = "Nasdaq-100 (QQQ)";
const SELECTED: &str = "Expanding window · Medium";

fn day(text: &str) -> Date {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap()
}

fn weights(values: &[f64]) -> Series {
    Series::new((0..values.len()).map(|i| format!("A{i:02}")).collect(), values.to_vec())
}

fn names(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs.iter().map(|(symbol, name)| (symbol.to_string(), name.to_string())).collect()
}

fn study(settings: Dict) -> BacktestStudy {
    BacktestStudy {
        equity: Frame::from_columns(Vec::new(), Vec::new()),
        metrics: Table::labelled(None, &[] as &[&str]),
        allocations: IndexMap::new(),
        holdings: IndexMap::new(),
        trades: IndexMap::new(),
        warnings: Vec::new(),
        settings,
    }
}

#[test]
fn headline_names_the_level_count_and_concentration() {
    let cases: [(&[f64], &str, &str, &str); 8] = [
        (
            &[0.264, 0.162, 0.159, 0.074, 0.068, 0.067, 0.060, 0.055, 0.045, 0.024, 0.022, 0.0001],
            "Medium",
            "Nasdaq-100",
            "A medium-risk portfolio of 11 Nasdaq-100 stocks.",
        ),
        (&[0.946, 0.054], "Extreme", "Nasdaq-100", "The highest-risk portfolio puts 94.6% in one stock."),
        (&[0.5, 0.5], "Extreme", "Original 60 holdings", "The highest-risk portfolio puts 50.0% in one stock."),
        (
            &[0.4, 0.3, 0.2, 0.1],
            "Low",
            "Original 60 holdings",
            "A low-risk portfolio of 4 stocks from your original 60.",
        ),
        (&[0.4, 0.3, 0.2, 0.1, 0.0004], "Low", "My tickers", "A low-risk portfolio of 4 of your tickers."),
        (&[0.3, 0.3, 0.2, 0.2], "Medium", "Upload CSV", "A medium-risk portfolio of 4 assets from your file."),
        (&[0.3, 0.3, 0.2, 0.2], "Extreme", "Demo", "A highest-risk portfolio of 4 simulated assets."),
        (&[1.0], "Low", "Demo", "The low-risk portfolio puts 100.0% in one asset."),
    ];
    for (values, profile, universe, expected) in cases {
        assert_eq!(headline_text(&weights(values), profile, universe), expected);
    }
}

#[test]
fn headline_html_keeps_hyphenated_phrases_together() {
    assert!(
        headline_html(&weights(&[0.6, 0.4]), "Medium", "Nasdaq-100")
            .contains("<span class=\"pl-nb\">medium-risk</span>")
    );
    let html = headline_html(&weights(&[0.3, 0.3, 0.2, 0.2]), "Low", "Nasdaq-100");
    assert!(html.ends_with("<span class=\"pl-nb\">Nasdaq-100 stocks</span>."));
}

#[test]
fn short_company_name_drops_security_type_and_legal_suffixes() {
    let cases = [
        ("Vertex Pharmaceuticals Incorporated Common Stock", "Vertex Pharmaceuticals"),
        ("Alphabet Inc. Class A Common Stock", "Alphabet"),
        ("Alphabet Inc. Class C Capital Stock", "Alphabet"),
        ("Seagate Technology Holdings PLC Ordinary Shares (Ireland)", "Seagate Technology Holdings"),
        ("NVIDIA Corporation Common Stock", "NVIDIA"),
        ("O'Reilly Automotive, Inc. Common Stock", "O'Reilly Automotive"),
        ("ASML Holding N.V. New York Registry Shares", "ASML Holding"),
        ("AstraZeneca PLC American Depositary Shares", "AstraZeneca"),
        ("Ferrovial SE Ordinary Shares", "Ferrovial"),
        ("Cisco Systems, Inc. Common Stock (DE)", "Cisco Systems"),
        ("Copart, Inc. (DE) Common Stock", "Copart"),
        ("Strategy Inc Common Stock Class A", "Strategy"),
        ("Shopify Inc. Class A Subordinate Voting Shares", "Shopify"),
        ("Warner Bros. Discovery, Inc. Series A Common Stock", "Warner Bros. Discovery"),
        ("Apple", "Apple"),
        ("Company", "Company"),
    ];
    for (raw, short) in cases {
        assert_eq!(short_company_name(raw), short, "{raw}");
    }
}

#[test]
fn short_company_name_for_every_nasdaq_100_member() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/nasdaq100_names.csv");
    let rows = read_csv_rows(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(rows.len() - 1, 101);
    let wrong: Vec<(String, String)> = rows[1..]
        .iter()
        .filter(|row| short_company_name(&row[1]) != row[2])
        .map(|row| (row[1].clone(), short_company_name(&row[1])))
        .collect();
    assert!(wrong.is_empty(), "{wrong:?}");
}

#[test]
fn holdings_cards_escape_provider_text_and_keep_small_positions_visible() {
    let saved = Series::new(vec!["A&B".into(), "<TINY>".into()], vec![0.123456, 0.000009]);
    let companies: IndexMap<String, Company> = [
        (
            "A&B".to_string(),
            Company {
                name: "<Company & \"Co\">".into(),
                description: "<script>external text</script>".into(),
                source_url: "https://finance.yahoo.com/quote/A%26B/profile/".into(),
            },
        ),
        (
            "<TINY>".to_string(),
            Company { name: "Small holding".into(), description: "Makes tools.".into(), ..Default::default() },
        ),
    ]
    .into_iter()
    .collect();
    let rendered = holdings_cards_html(&saved, &companies);
    assert!(rendered.contains("&lt;Company &amp; &quot;Co&quot;&gt;"));
    assert!(rendered.contains("&lt;script&gt;external text&lt;/script&gt;") && !rendered.contains("<script>"));
    assert!(rendered.contains("A&amp;B") && rendered.contains("&lt;TINY&gt;"));
    assert!(rendered.contains("12.35%") && rendered.contains("&lt;0.01%"));
    assert!(rendered.contains("href=\"https://finance.yahoo.com/quote/A%26B/profile/\""));
}

fn market_study(cagr: f64) -> BacktestStudy {
    let mut study = study(Dict::new());
    study.metrics =
        Table::labelled(None, &[SELECTED]).with("cagr", cells(&[cagr])).with("total_return", cells(&[cagr * 1.5]));
    study
}

fn empty_table() -> Table {
    Table::labelled(None, &[] as &[&str])
}

fn markets(equity: Option<Frame>, metrics: Option<Table>) -> BenchmarkComparison {
    let empty = Frame::from_columns(Vec::new(), Vec::new());
    BenchmarkComparison {
        prices: empty.clone(),
        holdout_equity: empty,
        holdout_metrics: empty_table(),
        backtest_equity: equity,
        backtest_metrics: metrics,
        training_estimates: empty_table(),
        latest_estimates: empty_table(),
        settings: Dict::new(),
    }
}

fn market_metrics(spy: f64, qqq: f64) -> Table {
    Table::labelled(Some("benchmark"), &[SPY, QQQ])
        .with("cagr", cells(&[spy, qqq]))
        .with("total_return", cells(&[spy * 1.5, qqq * 1.5]))
}

#[test]
fn market_answer_compares_growth_at_a_tenth_of_a_point() {
    let cases = [
        (0.094, 0.286, 0.411, "No. It trailed both markets."),
        (0.30, 0.20, 0.25, "Yes. It beat both markets in this test."),
        (0.25, 0.20, 0.30, "Partly. It beat the S&P 500 but trailed the Nasdaq-100."),
        (0.25, 0.30, 0.20, "Partly. It beat the Nasdaq-100 but trailed the S&P 500."),
        (0.2004, 0.2, 0.3, "Partly. It matched the S&P 500 but trailed the Nasdaq-100."),
    ];
    for (rule, spy, qqq, expected) in cases {
        let benchmarks = markets(Some(Frame::from_columns(Vec::new(), Vec::new())), Some(market_metrics(spy, qqq)));
        assert_eq!(
            market_answer(&market_study(rule), SELECTED, Some(&benchmarks)),
            format!("{expected} This compares compound growth (CAGR), without matching volatility.")
        );
    }
}

#[test]
fn market_answer_without_markets_states_growth() {
    assert_eq!(market_answer(&market_study(0.072), SELECTED, None), "It grew 7.2% a year.");
    let no_test = markets(None, None);
    assert_eq!(market_answer(&market_study(-0.031), SELECTED, Some(&no_test)), "It lost 3.1% a year.");
}

#[test]
fn market_lead_describes_the_rule_and_its_costs() {
    let settings = dict! {
        "test_start" => "2025-04-03", "test_end" => "2026-10-02", "rebalance_every" => 21, "cost_bps" => 10.0,
        "rolling_window" => 252, "periods_per_year" => 252,
    };
    let lead = market_lead(&study(settings.clone()), SELECTED);
    for phrase in [
        "We replayed the medium-risk rule over the last 18 months.",
        "About once a month it refitted using all earlier returns.",
        "annual historical arithmetic mean returns and volatility",
        "returns through the prior close; targets took effect at the following close",
        "first interval stayed in cash",
        "0.10% on each amount bought or sold",
        "later realised compound growth (CAGR), not the fitted mean return",
    ] {
        assert!(lead.contains(phrase), "{phrase}");
    }
    let mut every = settings.clone();
    every.insert("rebalance_every", 42);
    assert!(
        market_lead(&study(every), "Rolling window · Extreme")
            .contains("Every 42 trading days it refitted on the most recent 252 return observations")
    );
    assert!(market_lead(&study(settings.clone()), "Buy and hold · Minimum volatility").contains("minimum-volatility"));
    let mut monthly = settings;
    monthly.insert("periods_per_year", 12);
    assert!(market_lead(&study(monthly), SELECTED).contains("Every 21 price rows"));
}

fn evidence(statuses: [&str; 2], gaps: [f64; 2]) -> EvidenceAnalysis {
    let summary = Table::multi(
        vec![Some("strategy".into()), Some("benchmark".into())],
        vec![cells(&[SELECTED, SELECTED]), cells(&[SPY, QQQ])],
    )
    .with("status", cells(&statuses))
    .with("cagr_difference", cells(&gaps));
    EvidenceAnalysis {
        summary,
        windows: empty_table(),
        relative_equity: RelativeEquity { index: Vec::new(), columns: Vec::new(), values: DMatrix::zeros(0, 0) },
        settings: dict! {"multiple_testing_tests" => 96},
        warnings: Vec::new(),
    }
}

const UNCLEAR: &str = "No clear statistical advantage";
const BEHIND: &str = "Historical mean underperformance";

#[test]
fn evidence_answer_for_each_status_and_mixed_results() {
    let cases = [
        ([UNCLEAR; 2], "No statistically clear difference in mean return."),
        ([BEHIND; 2], "Lower mean returns than both markets, in this sample."),
        (
            ["Historical mean advantage", "Historical mean advantage and alpha"],
            "Higher mean returns than both markets, in this sample.",
        ),
        (["Historical alpha"; 2], "Some evidence of alpha. No clear lead in average return."),
        (["Insufficient history"; 2], "Too little history to judge."),
        (["Uncertainty unavailable"; 2], "The uncertainty could not be estimated."),
        (
            [UNCLEAR, BEHIND],
            "No clear mean-return difference against the S&P 500. Lower mean returns than the Nasdaq-100, in this sample.",
        ),
        (
            ["Historical mean advantage", "Historical alpha"],
            "Higher mean returns than the S&P 500, in this sample. Some evidence of alpha against the Nasdaq-100.",
        ),
    ];
    for (statuses, answer) in cases {
        assert_eq!(evidence_answer(Some(&evidence(statuses, [-0.19, -0.32])), SELECTED), answer);
    }
}

#[test]
fn evidence_lead_and_clarity_follow_the_statuses() {
    let unclear = evidence([UNCLEAR; 2], [-0.19, -0.32]);
    let lead = evidence_lead(Some(&unclear), SELECTED, 18, None);
    for phrase in [
        "trailed both markets on realised compound growth (CAGR)",
        "18 months is a short test",
        "annual arithmetic mean return gaps, not CAGR gaps",
        "Neither annual mean return gap is statistically clear",
        "all 96 tests",
        "does not establish equal performance",
    ] {
        assert!(lead.contains(phrase), "{phrase}");
    }
    assert!(!evidence_lead(Some(&unclear), SELECTED, 40, None).contains("short test"));
    let split = evidence([UNCLEAR; 2], [0.1, -0.1]);
    assert!(
        evidence_lead(Some(&split), SELECTED, 18, None)
            .starts_with("The rule led the S&P 500 and trailed the Nasdaq-100")
    );
    let mixed = evidence([UNCLEAR, BEHIND], [-0.19, -0.32]);
    assert!(
        evidence_lead(Some(&mixed), SELECTED, 18, None)
            .contains("statistical evidence differs by market after adjustment for all 96 tests")
    );
    assert!(
        clarity_text(Some(&unclear), SELECTED).starts_with("Neither annual mean return gap is statistically clear.")
    );
    assert!(
        clarity_text(Some(&mixed), SELECTED)
            .contains("Only the annual mean return gap to the Nasdaq-100 is statistically clear")
    );
    let behind = evidence([BEHIND; 2], [-0.19, -0.32]);
    assert!(clarity_text(Some(&behind), SELECTED).starts_with("Both annual mean return gaps are statistically clear"));
    assert!(clarity_text(Some(&mixed), SELECTED).contains("does not test the CAGR gaps"));
    let unknown = "The statistics could not be estimated for this test.";
    assert_eq!(clarity_text(Some(&evidence(["Insufficient history"; 2], [0.0, 0.0])), SELECTED), unknown);
    assert_eq!(clarity_text(None, SELECTED), unknown);
}

#[test]
fn cagr_outcome_remains_separate_when_mean_return_evidence_differs() {
    let cases = [
        (BEHIND, [0.12, 0.03], "led both markets", "statistically negative"),
        ("Historical mean advantage", [-0.12, -0.03], "trailed both markets", "statistically positive"),
        (UNCLEAR, [0.12, 0.03], "led both markets", "Neither annual mean return gap is statistically clear"),
        (UNCLEAR, [0.0, 0.0], "matched both markets", "Neither annual mean return gap is statistically clear"),
    ];
    for (status, gaps, growth, mean) in cases {
        let lead = evidence_lead(Some(&evidence([status; 2], gaps)), SELECTED, 40, None);
        assert!(lead.contains(&format!("{growth} on realised compound growth (CAGR) in this sample")));
        assert!(lead.contains(mean));
        assert!(lead.contains("annual arithmetic mean return gaps, not CAGR gaps"));
        assert!(!lead.contains("cannot beat"));
    }
}

#[test]
fn interval_html_escapes_labels_and_places_zero_on_a_padded_axis() {
    let mut forest = evidence([UNCLEAR; 2], [0.0, 0.0]);
    forest.summary = Table::multi(
        vec![Some("strategy".into()), Some("benchmark".into())],
        vec![cells(&[SELECTED, SELECTED]), cells(&["<img src=x onerror=\"a()\"> (X)", QQQ])],
    )
    .with("annual_advantage", cells(&[-0.166, -0.269]))
    .with("advantage_ci_low", cells(&[-0.358, -0.546]))
    .with("advantage_ci_high", cells(&[0.025, 0.007]));
    let html = interval_html(&forest, SELECTED);
    assert!(!html.contains("<img") && html.contains("vs &lt;img src=x onerror=&quot;a()&quot;&gt;"));
    // Range -54.6 to +2.5, padded 10%, on 20-point steps: -80 to +20, so zero sits at 80%.
    assert!(html.contains("class=\"f-zero\" style=\"left:80.00%\""));
    assert!(html.contains(
        "<span class=\"l\" style=\"width:80.00%\">← Rule behind</span><span class=\"r\">Rule ahead →</span>"
    ));
    assert!(html.contains("<b>−16.6</b> points<small>range −35.8 to +2.5</small>"));
    for tick in ["−80<", "−60<", "−40<", "−20<", ">0<", "+20<"] {
        assert!(html.contains(tick), "{tick}");
    }
}

fn stretch_evidence(rows: &[(&str, &str, &str, f64)]) -> EvidenceAnalysis {
    let mut result = evidence([UNCLEAR; 2], [0.0, 0.0]);
    result.windows = Table::range(rows.len())
        .with("strategy", vec![Cell::from(SELECTED); rows.len()])
        .with("benchmark", rows.iter().map(|row| Cell::from(row.0)).collect())
        .with("start", rows.iter().map(|row| Cell::from(row.1)).collect())
        .with("end", rows.iter().map(|row| Cell::from(row.2)).collect())
        .with("relative_return", rows.iter().map(|row| Cell::from(row.3)).collect());
    result
}

#[test]
fn stretch_table_marks_ahead_and_behind_and_escapes_markets() {
    let starts = ["2025-04-04", "2025-10-03", "2026-04-06"];
    let ends = ["2025-10-02", "2026-04-02", "2026-10-02"];
    let mut rows = Vec::new();
    for (market, changes) in [(SPY, [-0.111, 0.054, -0.163]), ("<b>Index</b> (X)", [-0.172, 0.072, -0.23])] {
        for k in 0..3 {
            rows.push((market, starts[k], ends[k], changes[k]));
        }
    }
    let html = stretch_table_html(&stretch_evidence(&rows), SELECTED);
    assert!(html.contains("<th>Apr 2025 to Oct 2025</th><th>Oct 2025 to Apr 2026</th><th>Apr 2026 to Oct 2026</th>"));
    assert!(html.contains("<i class=\"w-dot on\" aria-hidden=\"true\"></i>+5.4%<small>ahead</small>"));
    assert!(html.contains("<i class=\"w-dot\" aria-hidden=\"true\"></i>−11.1%<small>behind</small>"));
    assert!(
        html.contains("<td>S&amp;P 500</td>")
            && html.contains("&lt;b&gt;Index&lt;/b&gt;")
            && !html.contains("<b>Index")
    );
}

#[test]
fn stretch_table_marks_an_exact_tie_as_level() {
    let document = stretch_table_html(&stretch_evidence(&[(SPY, "2025-01-02", "2025-04-01", 0.0)]), SELECTED);
    assert!(document.contains("<small>level</small>") && !document.contains("<small>behind</small>"));
}

#[test]
fn chart_end_labels_remain_inside_the_axis_when_paths_finish_together() {
    let mut figure = Figure::new();
    for _ in 0..3 {
        figure.add_trace(json!({"x": [0, 1], "y": [10_000, 20_000], "mode": "lines", "line": {"color": "#333333"}}));
    }
    add_end_labels(&mut figure, &|value| efficient_frontier::pyfmt::format(value, ",.0f"));
    let layout = figure.full_layout();
    let positions: Vec<f64> =
        layout["annotations"].as_array().unwrap().iter().map(|a| a["y"].as_f64().unwrap()).collect();
    assert_eq!(positions.iter().map(|p| p.to_bits()).collect::<std::collections::HashSet<_>>().len(), 3);
    let range = layout["yaxis"]["range"].as_array().unwrap();
    let (low, high) = (range[0].as_f64().unwrap(), range[1].as_f64().unwrap());
    let (min, max) = positions.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(*p), b.max(*p)));
    assert!(low < min && max < high);
    assert!(layout["annotations"].as_array().unwrap().iter().all(|a| a["text"].as_str().unwrap().ends_with("20,000")));
}

#[test]
fn list_shows_ten_rows_and_one_rest_row_beyond_twelve_holdings() {
    let mut values = vec![0.099, 0.08, 0.07, 0.065, 0.06, 0.058, 0.055, 0.052, 0.05, 0.05, 0.04];
    values.extend([0.321 / 21.0; 21]);
    values.push(0.00001);
    let html = list_html(&weights(&values), Some(&names(&[("A00", "Ferrovial <SE>")])));
    assert_eq!(html.matches("class=\"pl-row\"").count(), 10);
    assert_eq!(html.matches("class=\"pl-row is-rest\"").count(), 1);
    assert!(html.contains("<b>22 more holdings</b><small>The largest of these is 4.0%.</small>"));
    assert!(html.contains("<span class=\"pl-pct\">36.1%</span></li></ul>"));
    assert!(html.contains("<b>Ferrovial &lt;SE&gt;</b><small>A00</small>"));
    assert!(html.contains("<i style=\"width:100.0%\"></i>"));
    let small = list_html(&weights(&[0.5, 0.3, 0.2, 0.00001]), None);
    assert_eq!(small.matches("class=\"pl-row\"").count(), 3);
    assert!(!small.contains("more holdings") && small.contains("<b>A00</b></span>"));
}

fn profiles(expected_return: f64, worst: f64) -> ProfileAnalysis {
    let names = ["Low", "Medium", "Extreme"];
    let portfolio = |name: &str| Portfolio {
        name: name.into(),
        weights: Series::new(vec!["AAA".into(), "BBB".into()], vec![0.6, 0.4]),
        expected_return,
        volatility: 0.15,
        sharpe: f64::NAN,
    };
    ProfileAnalysis {
        portfolios: names.iter().map(|name| (name.to_string(), portfolio(name))).collect(),
        frontier: empty_table(),
        frontier_weights: empty_table(),
        windows: Table::range(3).with("years", cells(&[1.0, 1.0, 1.0])),
        window_returns: empty_table(),
        summary: Table::labelled(Some("profile"), &names)
            .with("expected_return", cells(&[expected_return; 3]))
            .with("worst_window_return", cells(&[worst; 3]))
            .with("volatility", cells(&[0.15; 3]))
            .with("effective_holdings", cells(&[1.0 / (0.36 + 0.16); 3])),
        as_of: "2026-10-02".into(),
        observations: 750,
        settings: Dict::new(),
        warnings: Vec::new(),
    }
}

#[test]
fn overview_uses_full_history_mean_even_when_worst_window_has_opposite_sign() {
    for (expected_return, shown) in [(0.234, "23.4%"), (-0.234, "−23.4%")] {
        let latest = profiles(expected_return, -expected_return);
        let stats = stats_html(&latest, "Medium", "Nasdaq-100", None);
        let why = why_text(&latest, "Medium", "Nasdaq-100").join(" ");
        let ledger = levels_ledger_html(&latest, "Medium", None, "Nasdaq-100");
        for text in [&stats, &why, &ledger] {
            assert!(text.contains(shown));
            assert!(!text.contains("worst stretch") && !text.contains("Weakest past stretch"));
        }
        assert!(stats.contains("Fitted annual mean") && stats.contains("Full-history estimate before fees"));
        assert!(why.contains("arithmetic mean") && ledger.contains("arithmetic mean"));
        for phrase in [
            "all 3 years",
            "same prices that selected today's holdings",
            "not compound growth (CAGR) or a forecast",
            "measures realised growth on later prices",
            "midpoint between Low and Highest volatility",
            "Highest maximises the fitted annual mean return",
        ] {
            assert!(why.contains(phrase), "{phrase}");
        }
        assert!(ledger.contains("Historical mean*") && ledger.contains("not later realised compound growth (CAGR)"));
        let changed = profiles(expected_return, 0.876);
        assert_eq!(stats_html(&changed, "Medium", "Nasdaq-100", None), stats);
        assert_eq!(why_text(&changed, "Medium", "Nasdaq-100").join(" "), why);
        assert_eq!(levels_ledger_html(&changed, "Medium", None, "Nasdaq-100"), ledger);
    }
}

#[test]
fn method_defines_mean_frontier_and_separates_fit_growth_and_uncertainty() {
    for phrase in [
        "annual arithmetic mean returns and covariance",
        "nearest to the midpoint",
        "Low and Highest volatility",
        "maximises the fitted annual arithmetic mean return",
        "Each fit selects the same mean-based risk levels from earlier returns",
        "previous close",
        "next close",
        "annual arithmetic mean return gaps, not CAGR gaps",
        "unclear mean gap does not establish equal performance or rule out future outperformance",
    ] {
        assert!(METHOD_MD.contains(phrase), "{phrase}");
    }
    assert!(!METHOD_MD.contains("worst stretch"));
    assert_eq!(setting_text("return_objective", &Json::from("historical_mean")), "Annual historical arithmetic mean");
}

fn columns(names: &[&str]) -> Frame {
    Frame::from_columns(Vec::new(), names.iter().map(|name| (name.to_string(), Vec::new())).collect())
}

#[test]
fn one_asset_reads_in_the_singular() {
    assert_eq!(these_assets(92, "Nasdaq-100"), "these 92 stocks");
    assert_eq!(these_assets(1, "Demo"), "this one asset");
    let items = fine_print_items(
        &dict! {"universe" => "Demo", "source" => "Demo · synthetic"},
        &columns(&["A"]),
        "Research › Data",
    );
    assert!(items[1].contains("compare mixes of this one asset only"));
}

#[test]
fn currency_note_describes_the_declaration_without_claiming_verification() {
    let prices = columns(&["VOD.L", "AAPL"]);
    for source in ["Yahoo Finance", "Upload CSV", "CSV: monthly-prices.csv", "Nasdaq-100 · Yahoo Finance"] {
        let metadata = dict! {"source" => source, "universe" => "My tickers", "price_currency" => "USD"};
        let items = fine_print_items(&metadata, &prices, "Research › Data");
        assert_eq!(
            items.last().unwrap(),
            "<b>Currency.</b> You declared USD prices. The app does not check or convert currencies."
        );
        let metadata = dict! {"source" => source, "universe" => "My tickers", "price_currency" => "Other currency"};
        let items = fine_print_items(&metadata, &prices, "Research › Data");
        assert!(items.last().unwrap().contains("You declared a common price currency."));
        assert!(items.last().unwrap().contains("Market comparisons are off"));
    }
}

#[test]
fn currency_note_identifies_actual_demo_prices() {
    for source in ["Demo · synthetic", "Demo"] {
        let metadata = dict! {"source" => source, "universe" => "Demo", "price_currency" => "USD"};
        let items = fine_print_items(&metadata, &columns(&["A"]), "Research › Data");
        assert_eq!(
            items.last().unwrap(),
            "<b>Currency.</b> The demo prices are simulated. No currency conversion occurs."
        );
    }
}

#[test]
fn concentration_callout_describes_allocation_without_claiming_risk_or_profit_shares() {
    let html = callout_html(&Series::new(vec!["BONDS".into(), "STOCKS".into()], vec![0.5, 0.5]), "Upload CSV", None);
    assert!(
        html.contains("At least half of it is BONDS.") && html.contains("This is a large allocation to one asset.")
    );
    assert!(!html.contains("gains and losses"));
    let spread = Series::new(vec!["BONDS".into(), "STOCKS".into(), "CASH".into()], vec![0.4, 0.3, 0.3]);
    assert_eq!(callout_html(&spread, "Upload CSV", None), "");
}

#[test]
fn result_tile_without_markets_does_not_repeat_the_answer_or_claim_a_failed_test() {
    let mut tile =
        study(dict! {"test_start" => "2025-04-03", "test_end" => "2026-10-02", "initial_execution" => "2025-04-03"});
    tile.equity =
        Frame::from_columns(vec![day("2025-04-02"), day("2026-10-02")], vec![(SELECTED.into(), vec![1.0, 1.073])]);
    tile.metrics =
        Table::labelled(None, &[SELECTED]).with("cagr", cells(&[0.061])).with("total_return", cells(&[0.073]));
    let html = result_tile_html(&tile, SELECTED, None, None, &dict! {"universe" => "Demo"});
    assert!(html.contains("<p class=\"pl-verdict\">In total, it gained 7.3%.</p>"));
    assert!(!html.contains("a year,") && !html.contains("could not be estimated") && !html.contains("pl-clarity"));
}

/// The `decision_study` fixture: five profile paths and the two ETFs on the same dates.
fn decision_study() -> (BacktestStudy, BenchmarkComparison) {
    let dates = vec![day("2024-01-02"), day("2024-01-03"), day("2024-01-04"), day("2024-01-05")];
    let paths = [
        ("Equal weight", [1.0, 0.99, 1.04, 1.08]),
        ("Extreme", [1.0, 0.99, 1.3, 1.05]),
        ("Medium", [1.0, 0.99, 0.85, 0.93]),
        ("Low", [1.0, 0.995, 1.01, 1.03]),
        ("Maximum Sharpe", [1.0, 0.99, 1.08, 1.06]),
    ];
    let equity = Frame::from_columns(
        dates.clone(),
        paths.iter().map(|(name, path)| (format!("Expanding window · {name}"), path.to_vec())).collect(),
    );
    let metrics_of = |frame: &Frame, cost: Option<f64>| {
        let mut table = Table::labelled(None, &frame.columns);
        for field in efficient_frontier::metrics::PERFORMANCE_FIELDS {
            let values = (0..frame.ncols()).map(|j| performance_metrics(&frame.col(j), 0.0, 12.0).get(field).unwrap());
            table.push(field, values.map(Cell::Float).collect());
        }
        if let Some(cost) = cost {
            table.push("total_cost", vec![Cell::Float(cost); frame.ncols()]);
        }
        table
    };
    let metrics = metrics_of(&equity, Some(0.01));
    let holdings = Table::labelled(None, &["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "UNHELD"])
        .with("pnl_contribution", cells(&[0.09, -0.14, -0.03, 0.02, 0.015, -0.01, -0.005, 0.0]));
    let mut study = study(dict! {"periods_per_year" => 12, "test_start" => "2024-01-03", "test_end" => "2024-01-05"});
    study.equity = equity;
    study.metrics = metrics;
    study.holdings.insert(SELECTED.into(), holdings);
    let market_equity = Frame::from_columns(
        dates,
        vec![(QQQ.into(), vec![1.0, 0.99, 1.04, 1.12]), (SPY.into(), vec![1.0, 0.99, 1.05, 1.10])],
    );
    let market_metrics = metrics_of(&market_equity, None);
    (study, markets(Some(market_equity), Some(market_metrics)))
}

/// Text of every cell of every table row, as in the Python helper.
fn table_rows(document: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for row in document.split("<tr").skip(1) {
        let row = &row[row.find('>').unwrap() + 1..row.find("</tr>").unwrap()];
        let mut cells = Vec::new();
        for cell in row.split("<t").skip(1) {
            let inside = &cell[cell.find('>').unwrap() + 1..];
            let inside = &inside[..inside.find("</t").unwrap_or(inside.len())];
            let mut text = String::new();
            let mut tag = false;
            for character in inside.chars() {
                match character {
                    '<' => tag = true,
                    '>' => tag = false,
                    c if !tag => text.push(c),
                    _ => {}
                }
            }
            cells.push(text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"));
        }
        rows.push(cells);
    }
    rows
}

fn one_decimal(value: f64) -> String {
    story::pct(value, 1, false)
}

#[test]
fn decision_panel_keeps_fixed_order_and_matches_net_paths_and_metrics() {
    let (mut study, benchmarks) = decision_study();
    let document = decision_panel_html(&study, SELECTED, Some(&benchmarks));
    let rows = table_rows(&document);
    assert_eq!(
        rows[0],
        ["Rule / ETF", "Test mean / year", "Growth / year (CAGR)", "Volatility / year", "Maximum drawdown"]
    );
    let labels: Vec<&str> = rows[1..].iter().map(|row| row[0].as_str()).collect();
    assert_eq!(labels, ["Low", "Medium · Selected", "Highest", "Maximum Sharpe", "Equal weight", SPY, QQQ]);
    assert_eq!(document.matches("class=\"on\"").count(), 1);
    for (cells, name) in rows[1..6].iter().zip(["Low", "Medium", "Extreme", "Maximum Sharpe", "Equal weight"]) {
        let name = format!("Expanding window · {name}");
        let path = study.equity.column(&name).unwrap();
        let mean = (1..path.len()).map(|i| path[i] / path[i - 1] - 1.0).sum::<f64>() * 12.0 / 3.0;
        let mut expected = vec![one_decimal(mean)];
        expected
            .extend(["cagr", "volatility", "max_drawdown"].map(|key| one_decimal(study.metrics.f64_at(&name, key))));
        assert_eq!(cells[1..], expected);
    }
    let spy_mean = (0.99 - 1.0 + 1.05 / 0.99 - 1.0 + 1.10 / 1.05 - 1.0) * 4.0;
    let metrics = benchmarks.backtest_metrics.as_ref().unwrap();
    let mut expected = vec![one_decimal(spy_mean)];
    expected.extend(["cagr", "volatility", "max_drawdown"].map(|key| one_decimal(metrics.f64_at(SPY, key))));
    assert_eq!(rows[rows.len() - 2][1..], expected);
    for phrase in [
        "Test returns: 3 Jan 2024 to 5 Jan 2024",
        "Baseline: 2 Jan 2024",
        "(Expanding window)",
        "including the entry interval and fees",
        "For compound growth, this test gives no reason to prefer this rule to either ETF",
        "Medium is not the market average and can be riskier than both ETFs",
        "Latest proposed weights fit all selected prices",
        "replay uses earlier fits",
    ] {
        assert!(document.contains(phrase), "{phrase}");
    }
    // Reversing the numerical ranking must not change row order or the chosen profile.
    let cagr = study.metrics.position("cagr").unwrap();
    for cell in study.metrics.data[cagr].iter_mut() {
        *cell = Cell::Float(-cell.as_f64().unwrap());
    }
    let changed = table_rows(&decision_panel_html(&study, SELECTED, Some(&benchmarks)));
    assert_eq!(
        changed.iter().map(|row| row[0].clone()).collect::<Vec<_>>(),
        rows.iter().map(|row| row[0].clone()).collect::<Vec<_>>()
    );
}

#[test]
fn decision_panel_explains_selected_rule_without_recommending_it() {
    let (study, benchmarks) = decision_study();
    for (policy, purpose) in [
        ("Low", "Low targets the least fitted volatility"),
        ("Extreme", "Without a holding cap, it can allocate to one asset"),
    ] {
        let document = decision_panel_html(&study, &format!("Expanding window · {policy}"), Some(&benchmarks));
        assert!(document.contains(purpose));
        assert!(document.contains("These are model rules, not recommendations"));
        assert!(document.contains(&format!("{} · Selected", efficient_frontier::style::display_label(policy))));
    }
}

#[test]
fn decision_panel_qualifies_cagr_verdict_when_net_arithmetic_mean_beats_etfs() {
    let (mut study, benchmarks) = decision_study();
    let j = study.equity.position(SELECTED).unwrap();
    for (i, value) in [1.0, 0.99, 0.6, 1.09].into_iter().enumerate() {
        study.equity.values[(i, j)] = value;
    }
    let metrics = performance_metrics(&study.equity.col(j), 0.0, 12.0);
    let row = study.metrics.row_of(SELECTED).unwrap();
    for (field, value) in metrics.fields() {
        let column = study.metrics.position(field).unwrap();
        study.metrics.data[column][row] = Cell::Float(value);
    }
    let document = decision_panel_html(&study, SELECTED, Some(&benchmarks));
    let rows = table_rows(&document);
    let number = |text: &str| text.trim_end_matches('%').replace('−', "-").parse::<f64>().unwrap();
    assert!(number(&rows[2][1]) > rows[rows.len() - 2..].iter().map(|row| number(&row[1])).fold(f64::MIN, f64::max));
    assert!(document.contains("No. It trailed both markets"));
    assert!(document.contains("For compound growth, this test gives no reason"));
    assert!(!document.contains("no return-based reason"));
}

#[test]
fn decision_panel_does_not_replace_a_missing_profile() {
    let (mut study, benchmarks) = decision_study();
    let other = "Expanding window · Minimum volatility";
    let j = study.equity.position(SELECTED).unwrap();
    study.equity.columns[j] = other.into();
    study.metrics.rename_index(&[(SELECTED, other)]);
    let document = decision_panel_html(&study, SELECTED, Some(&benchmarks));
    assert!(document.contains("Medium was not backtested. Its result is unavailable"));
    assert!(document.contains("no other rule replaces the selection"));
    assert!(
        !document.contains("Minimum volatility")
            && !document.contains("class=\"on\"")
            && !document.contains("No. It trailed")
    );
}

#[test]
fn decision_panel_handles_absent_or_unmatched_etfs_without_claiming_a_market_result() {
    let (study, mut benchmarks) = decision_study();
    for missing in [None, Some(markets(None, None))] {
        let document = decision_panel_html(&study, SELECTED, missing.as_ref());
        assert!(document.contains("Matching ETF results are unavailable"));
        assert!(document.contains("It lost") && document.contains("a year"));
        assert!(!document.contains("trailed both markets"));
        assert_eq!(table_rows(&document).len(), 6);
    }
    let equity = benchmarks.backtest_equity.as_mut().unwrap();
    equity.index = equity.index.iter().map(|date| date.succ_opt().unwrap()).collect();
    assert!(decision_panel_html(&study, SELECTED, Some(&benchmarks)).contains("Matching ETF results are unavailable"));
}

#[test]
fn decision_panel_marks_partial_benchmark_coverage() {
    let (study, mut benchmarks) = decision_study();
    let equity = benchmarks.backtest_equity.as_ref().unwrap().drop_columns(&[QQQ]);
    benchmarks.backtest_equity = Some(equity);
    let metrics = benchmarks.backtest_metrics.as_ref().unwrap();
    benchmarks.backtest_metrics = Some(metrics.take_rows(&[metrics.row_of(SPY).unwrap()]));
    let document = decision_panel_html(&study, SELECTED, Some(&benchmarks));
    assert!(document.contains("It trailed the S&amp;P 500 in this test"));
    assert!(document.contains("One or both ETF comparisons are unavailable"));
    assert!(!document.contains("either ETF"));
}

#[test]
fn result_drivers_show_largest_gains_and_losses_and_reconcile_fees() {
    let (study, _) = decision_study();
    let document = result_drivers_html(&study, SELECTED);
    let rows = table_rows(&document);
    let expected = [
        ["AAA", "+900.00"],
        ["DDD", "+200.00"],
        ["EEE", "+150.00"],
        ["BBB", "−1,400.00"],
        ["CCC", "−300.00"],
        ["FFF", "−100.00"],
        ["Other holdings", "−50.00"],
        ["Trading fees", "−100.00"],
        ["Net change", "−700.00"],
    ];
    assert_eq!(rows[1..], expected.map(|row| row.map(String::from).to_vec()));
    let amounts: Vec<f64> =
        rows[1..].iter().map(|row| row[1].replace([',', '+'], "").replace('−', "-").parse().unwrap()).collect();
    assert!((amounts[..amounts.len() - 1].iter().sum::<f64>() - amounts[amounts.len() - 1]).abs() < 1e-9);
    let path = study.equity.column(SELECTED).unwrap();
    assert!((amounts[amounts.len() - 1] - (path[path.len() - 1] - 1.0) * 10_000.0).abs() < 1e-9);
    assert!(!document.contains("UNHELD"));
    assert!(document.contains("historical replay, not the latest proposed weights"));
    assert!(document.contains("All holding contributions less trading fees equal the net change"));
}

#[test]
fn result_drivers_do_not_hide_losses_behind_larger_gains_or_show_rounding_residuals() {
    let (mut study, _) = decision_study();
    let labels = ["GAIN1", "GAIN2", "GAIN3", "LOSS1", "LOSS2", "LOSS3", "TINY_GAIN", "TINY_LOSS"];
    let holdings = Table::labelled(None, &labels)
        .with("pnl_contribution", cells(&[0.3, 0.2, 0.1, -0.03, -0.02, -0.01, 1e-14, -1e-14]));
    study.holdings.insert(SELECTED.into(), holdings);
    let row = study.metrics.row_of(SELECTED).unwrap();
    let column = study.metrics.position("total_return").unwrap();
    study.metrics.data[column][row] = Cell::Float(0.53);
    let document = result_drivers_html(&study, SELECTED);
    let shown: Vec<String> = table_rows(&document)[1..].iter().map(|row| row[0].clone()).collect();
    assert_eq!(shown, ["GAIN1", "GAIN2", "GAIN3", "LOSS1", "LOSS2", "LOSS3", "Trading fees", "Net change"]);
    assert!(!document.contains("TINY_") && document.contains("three largest gains and three largest losses"));
}

#[test]
fn result_drivers_escape_names_and_show_explicit_missing_or_zero_results() {
    let (mut study, _) = decision_study();
    let label = "<img src=x onerror=\"bad()\">";
    study.holdings.get_mut(SELECTED).unwrap().rename_index(&[("BBB", label)]);
    let document = result_drivers_html(&study, SELECTED);
    assert!(!document.contains(label) && document.contains(&escape(label)));
    assert!(result_drivers_html(&study, "Expanding window · Extreme").contains("are unavailable"));
    let holdings = study.holdings.get_mut(SELECTED).unwrap();
    let count = holdings.nrows();
    holdings.push("pnl_contribution", vec![Cell::Float(0.0); count]);
    let row = study.metrics.row_of(SELECTED).unwrap();
    let column = study.metrics.position("total_return").unwrap();
    study.metrics.data[column][row] = Cell::Float(-0.01);
    let document = result_drivers_html(&study, SELECTED);
    assert!(document.contains("No holding contributed a gain or loss"));
    let rows = table_rows(&document);
    assert_eq!(
        rows[1..],
        [vec!["Trading fees".to_string(), "−100.00".into()], vec!["Net change".into(), "−100.00".into()]]
    );
}

#[test]
fn result_tile_uses_loss_words_and_leaves_uncertainty_for_the_evidence_section() {
    let (study, mut benchmarks) = decision_study();
    let metrics = benchmarks.backtest_metrics.as_mut().unwrap();
    let (row, column) = (metrics.row_of(SPY).unwrap(), metrics.position("cagr").unwrap());
    metrics.data[column][row] = Cell::Float(-0.02);
    let document = result_tile_html(
        &study,
        SELECTED,
        Some(&benchmarks),
        Some(&evidence([BEHIND; 2], [0.0, 0.0])),
        &dict! {"universe" => "Demo"},
    );
    assert!(document.contains("It lost") && document.contains("S&amp;P 500 lost 2.0% a year"));
    assert!(!document.contains("grew −") && !document.contains("pl-clarity") && !document.contains("statistically"));
    assert!(!document.contains("How sure can we be"));
}

#[test]
fn combined_universe_context_keeps_history_and_coverage_counts() {
    let universe = "S&P 500 + Nasdaq-100";
    let mut dates = Vec::new();
    let mut date = day("2021-10-18");
    while date <= day("2026-10-15") {
        if chrono::Datelike::weekday(&date).number_from_monday() <= 5 {
            dates.push(date);
        }
        date = date.succ_opt().unwrap();
    }
    let count = dates.len();
    let prices = Frame::from_columns(
        dates,
        ["AAA", "BBB", "CCC"].iter().map(|name| (name.to_string(), vec![100.0; count])).collect(),
    );
    let metadata = dict! {
        "universe" => universe, "universe_included" => 3, "universe_requested" => 5,
        "requested_start" => "2021-10-16", "requested_end_exclusive" => "2026-10-16",
    };
    let result = context_result_html(&metadata, &prices);
    assert!(result.contains("S&amp;P 500 + Nasdaq-100") && result.contains("3 of 5 stocks"));
    assert!(result.contains("5 years") && result.contains("15 Oct 2026"));
    let setup = SetupContext {
        market: universe.into(),
        prices_file: None,
        history: "5 years".into(),
        start: day("2021-10-16"),
        end: day("2026-10-16"),
    };
    assert!(context_setup_html(&setup, None).contains("5 years"));
    assert_eq!(these_assets(3, universe), "these 3 stocks");
}

#[test]
fn context_setup_shows_custom_dates_files_and_badges() {
    let setup = SetupContext {
        market: "Nasdaq-100".into(),
        prices_file: None,
        history: "Custom dates".into(),
        start: day("2021-10-04"),
        end: day("2026-10-03"),
    };
    assert_eq!(
        context_setup_html(&setup, Some("Simulated <x>")),
        "<p class=\"pl-context\"><span>Nasdaq-100</span><span class=\"pl-sep\">·</span>\
         <span>4 Oct 2021 to 2 Oct 2026</span> <span class=\"pl-badge\">Simulated &lt;x&gt;</span></p>"
    );
    let upload = SetupContext { market: "Upload CSV".into(), prices_file: Some("my <prices>.csv".into()), ..setup };
    assert_eq!(
        context_setup_html(&upload, None),
        "<p class=\"pl-context\"><span>CSV: my &lt;prices&gt;.csv</span></p>"
    );
}

#[test]
fn settings_show_plain_labels_and_display_values() {
    assert_eq!(setting_label("cost_bps"), "Trading cost");
    assert_eq!(setting_label("model_revision"), "Model revision");
    assert_eq!(setting_text("cost_bps", &Json::Float(10.0)), "10 basis points (0.10%)");
    assert_eq!(setting_text("max_weight", &Json::Float(1.0)), "No limit");
    assert_eq!(setting_text("max_weight", &Json::Float(0.25)), "25%");
    assert_eq!(setting_text("include_profiles", &Json::Bool(true)), "Yes");
    assert_eq!(setting_text("fit_dates", &Json::from(vec!["2024-01-02", "2024-06-03"])), "2024-01-02, 2024-06-03");
    let coverage = Table::range(2)
        .with("symbol", cells(&["AAPL", "NVDA"]))
        .with("name", cells(&["Apple Inc. Common Stock", "NVIDIA Corporation Common Stock"]));
    let names = company_names(Some(&coverage));
    assert_eq!(names["AAPL"], "Apple");
    assert_eq!(names["NVDA"], "NVIDIA");
    assert!(company_names(None).is_empty());
    assert_eq!(VIEW_LABELS[0], ("Portfolio", "Overview"));
}
