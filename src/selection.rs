//! One historical-mean stock portfolio with a later buy-and-hold replay.

use indexmap::IndexMap;
use nalgebra::DMatrix;

use crate::backtest::{BacktestStudy, after_cost};
use crate::core::{
    Portfolio, TRADING_DAYS, dense_row, dot, estimates, ones_row, optimize, psd_factor, quadratic_form, shrink,
    solve_checked, validate_price_frame,
};
use crate::dict;
use crate::error::Result;
use crate::frame::{Cell, Date, Frame, LabeledMatrix, Series, Table, iso, mean, var};
use crate::metrics::{path_returns, performance_metrics, validate_periods_per_year};
use crate::pyfmt::{self, Dict, Json};
use crate::solver::{Program, Status};
use crate::{runtime_error, value_error};

pub const SELECTED: &str = "Historical-mean portfolio";
pub const BASELINE: &str = "Equal-weight stocks";
pub const MODEL_REVISION: &str = "stock_mean_buy_hold_v3";
/// Comparison ETFs and their display names, in fixed order.
pub const BENCHMARK_NAMES: [(&str, &str); 2] = [("SPY", "S&P 500 (SPY)"), ("QQQ", "Nasdaq-100 (QQQ)")];
pub const STALE_INTERVALS: usize = 20;

/// The display name of a comparison ETF symbol.
pub fn benchmark_name(symbol: &str) -> Option<&'static str> {
    BENCHMARK_NAMES.iter().find(|(ticker, _)| *ticker == symbol).map(|(_, name)| *name)
}

#[derive(Clone, Debug)]
pub struct SelectionStudy {
    pub test: BacktestStudy,
    pub latest: IndexMap<String, Portfolio>,
    pub selected: String,
    pub supported: bool,
    pub reasons: Vec<String>,
    pub settings: Dict,
    pub warnings: Vec<String>,
    pub benchmark_prices: Option<Frame>,
    /// Columns `fit_date`, `symbol`, `eligible`, `max_unchanged_intervals`, `reason`.
    pub eligibility: Table,
}

/// One fit of the stock model on past prices.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Selected weights over every price column; screened stocks have zero weight.
    pub selected: Series,
    /// Equal weights across eligible stocks, over every price column.
    pub equal: Series,
    /// Annual means over every price column; screened stocks are zero.
    pub mean: Series,
    /// Annual covariance over every price column; screened stocks are zero.
    pub covariance: LabeledMatrix,
    /// The dated eligibility decision for each stock.
    pub coverage: Table,
    pub warnings: Vec<String>,
}

/// Separate known benchmark ETFs from the stock candidates without a download.
pub fn split_stock_prices(prices: &Frame) -> Result<(Frame, Option<Frame>)> {
    validate_price_frame(prices)?;
    let normalized: Vec<String> = prices.columns.iter().map(|column| column.trim().to_uppercase()).collect();
    let references: Vec<usize> = (0..prices.ncols()).filter(|j| benchmark_name(&normalized[*j]).is_some()).collect();
    let stocks: Vec<usize> = (0..prices.ncols()).filter(|j| !references.contains(j)).collect();
    if stocks.is_empty() {
        return Err(value_error!(
            "Provide stock price columns as well as SPY or QQQ. The ETFs are comparison benchmarks, not portfolio holdings."
        ));
    }
    let mut benchmarks = None;
    if !references.is_empty() {
        let mut frame = prices.select_positions(&references);
        frame.columns = references.iter().map(|j| normalized[*j].clone()).collect();
        benchmarks = Some(validate_selection_benchmarks(&frame, &prices.index)?);
    }
    Ok((prices.select_positions(&stocks), benchmarks))
}

/// Accept either or both comparison ETFs, with exactly the stock price dates.
pub fn validate_selection_benchmarks(prices: &Frame, expected_dates: &[Date]) -> Result<Frame> {
    validate_price_frame(prices)?;
    let mut frame = prices.clone();
    frame.columns = frame.columns.iter().map(|column| column.trim().to_uppercase()).collect();
    let unique: std::collections::HashSet<&String> = frame.columns.iter().collect();
    if unique.len() != frame.ncols()
        || frame.index != expected_dates
        || frame.columns.iter().any(|column| benchmark_name(column).is_none())
    {
        return Err(value_error!("Benchmark prices must contain SPY and/or QQQ with exactly the stock price dates."));
    }
    let order: Vec<&str> =
        BENCHMARK_NAMES.iter().map(|(symbol, _)| *symbol).filter(|symbol| frame.position(symbol).is_some()).collect();
    Ok(frame.select(&order).expect("present benchmark columns"))
}

/// The longest run of unchanged consecutive prices in each column.
pub fn longest_unchanged(prices: &Frame) -> Vec<usize> {
    (0..prices.ncols())
        .map(|j| {
            let mut run = 0;
            let mut longest = 0;
            for i in 1..prices.nrows() {
                run = if prices.get(i, j) == prices.get(i - 1, j) { run + 1 } else { 0 };
                longest = longest.max(run);
            }
            longest
        })
        .collect()
}

/// Maximize historical mean within the cap and risk limit, using past prices.
///
/// At a tied maximum mean, minimize variance. The secondary solve permits a
/// normalized mean shortfall of 1e-8, checked within 2e-8 for numerical error.
pub fn fit_targets(
    prices: &Frame,
    target_volatility: f64,
    max_weight: f64,
    shrinkage: f64,
    periods_per_year: f64,
) -> Result<Fit> {
    let fit_date = iso(*prices.index.last().expect("prices have dates"));
    let longest = longest_unchanged(prices);
    let eligible: Vec<bool> = longest.iter().map(|run| *run < STALE_INTERVALS).collect();
    let positions: Vec<usize> = (0..prices.ncols()).filter(|j| eligible[*j]).collect();
    let n = positions.len();
    if n as f64 * max_weight < 1.0 - 1e-12 {
        return Err(value_error!(
            "{fit_date}: the {} position cap is infeasible for {n} eligible stocks after the long unchanged-price screen. \
             Supply more eligible stocks or explicitly change the cap.",
            pyfmt::format(max_weight, ".1%")
        ));
    }
    let returns = prices.select_positions(&positions).pct_change();
    let (mean_series, labelled_covariance, _) = estimates(&returns, periods_per_year, shrinkage);
    let mean_values = mean_series.values;
    let covariance = labelled_covariance.values;
    let largest = covariance.iter().fold(0.0_f64, |high, value| high.max(value.abs()));
    let scale = if largest != 0.0 { largest } else { 1.0 };
    let scaled = &covariance / scale;
    let budget = || {
        let mut program = Program::new(n).equal(ones_row(n), 1.0);
        for j in 0..n {
            program = program.at_least(vec![(j, 1.0)], 0.0).at_most(vec![(j, 1.0)], max_weight);
        }
        program
    };
    let largest_variance = covariance.diagonal().iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // A long-only portfolio's variance cannot exceed the largest asset variance.
    // Omit this redundant bound to avoid huge ratios for near-constant returns.
    let risk_limited = target_volatility.powi(2) < largest_variance;
    let mut warnings = Vec::new();

    let solve = |program: &Program, warnings: &mut Vec<String>| -> Result<Vec<f64>> {
        let solution =
            program.solve().map_err(|error| runtime_error!("Portfolio optimization failed: {}", error.message()))?;
        if solution.status.is_infeasible() {
            let minimum = solve_checked(&budget().quadratic(scaled.clone()))?;
            let floor = quadratic_form(&minimum.x, &covariance).max(0.0).sqrt();
            return Err(value_error!(
                "{fit_date}: requested annual volatility {} is below the feasible minimum {} with the {} position cap.",
                pyfmt::format(target_volatility, ".2%"),
                pyfmt::format(floor, ".2%"),
                pyfmt::format(max_weight, ".1%")
            ));
        }
        if !solution.status.is_optimal() {
            return Err(runtime_error!("Portfolio optimization failed: {}", solution.status.name()));
        }
        if solution.status == Status::OptimalInaccurate {
            warnings.push(format!("{fit_date}: the selection solver reported limited numerical accuracy."));
        }
        Ok(solution.x)
    };
    let checked = |value: &[f64]| -> Result<Vec<f64>> {
        let sum: f64 = value.iter().sum();
        let lowest = value.iter().copied().fold(f64::INFINITY, f64::min);
        let highest = value.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if value.iter().any(|v| !v.is_finite())
            || (sum - 1.0).abs() > 1e-7
            || lowest < -1e-7
            || highest > max_weight + 1e-7
        {
            return Err(runtime_error!("Selection solver returned infeasible stock weights."));
        }
        let clipped: Vec<f64> = value.iter().map(|v| v.clamp(0.0, max_weight)).collect();
        let total: f64 = clipped.iter().sum();
        let weights: Vec<f64> = clipped.iter().map(|v| v / total).collect();
        let volatility = quadratic_form(&weights, &covariance).max(0.0).sqrt();
        if volatility > target_volatility + 1e-7 {
            return Err(runtime_error!("Selection solver exceeded the requested estimated volatility."));
        }
        Ok(weights)
    };

    let mean_scale = mean_values.iter().fold(0.0_f64, |high, value| high.max(value.abs()));
    let mean_scale = if mean_scale != 0.0 { mean_scale } else { 1.0 };
    let scores: Vec<f64> = mean_values.iter().map(|value| value / mean_scale).collect();
    let mut program = budget().maximize(&scores);
    if risk_limited {
        program = program.norm_at_most(psd_factor(&scaled), (target_volatility.powi(2) / scale).sqrt());
    }
    let mut selected = checked(&solve(&program, &mut warnings)?)?;
    // Risk can only improve at the same maximum mean when the risk ceiling does
    // not prevent the capped, risk-unconstrained maximum. Check that face first.
    let mut ordered = scores.clone();
    ordered.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let greedy: Vec<f64> = (0..n).map(|k| (1.0 - max_weight * k as f64).max(0.0).min(max_weight)).collect();
    let highest_score = dot(&ordered, &greedy);
    let tolerance = 1e-8;
    let last_positive = (0..n).rfind(|k| greedy[*k] > tolerance).expect("the first greedy weight is positive");
    let marginal = ordered[last_positive];
    let tied = scores.iter().filter(|score| (*score - marginal).abs() <= tolerance).count();
    let tied_weight = 1.0 - max_weight * scores.iter().filter(|score| **score > marginal + tolerance).count() as f64;
    if tied > 1
        && tolerance < tied_weight
        && tied_weight < max_weight * tied as f64 - tolerance
        && dot(&scores, &selected) >= highest_score - tolerance
    {
        // Minimum variance itself enforces the existing risk ceiling. Omitting
        // the redundant ceiling avoids a second solve at a tangential boundary.
        let program = budget().quadratic(scaled.clone()).at_least(dense_row(&scores), highest_score - tolerance);
        selected = checked(&solve(&program, &mut warnings)?)?;
        if dot(&scores, &selected) < highest_score - 2.0 * tolerance {
            return Err(runtime_error!("Selection risk tie-break missed the maximum historical mean."));
        }
    }

    let full = |values: &[f64]| {
        let mut expanded = vec![0.0; prices.ncols()];
        for (k, j) in positions.iter().enumerate() {
            expanded[*j] = values[k];
        }
        Series::new(prices.columns.clone(), expanded)
    };
    let mut full_covariance = DMatrix::zeros(prices.ncols(), prices.ncols());
    for (a, i) in positions.iter().enumerate() {
        for (b, j) in positions.iter().enumerate() {
            full_covariance[(*i, *j)] = covariance[(a, b)];
        }
    }
    let reason = format!("At least {STALE_INTERVALS} consecutive unchanged price intervals in past data.");
    let coverage = Table::range(prices.ncols())
        .with("fit_date", vec![Cell::Str(fit_date.clone()); prices.ncols()])
        .with("symbol", prices.columns.iter().map(|symbol| Cell::Str(symbol.clone())).collect())
        .with("eligible", eligible.iter().map(|value| Cell::Bool(*value)).collect())
        .with("max_unchanged_intervals", longest.iter().map(|run| Cell::Int(*run as i64)).collect())
        .with(
            "reason",
            eligible.iter().map(|value| Cell::Str(if *value { String::new() } else { reason.clone() })).collect(),
        );
    if eligible.iter().any(|value| !value) {
        let excluded: Vec<&str> =
            (0..prices.ncols()).filter(|j| !eligible[*j]).map(|j| prices.columns[j].as_str()).collect();
        warnings.push(format!(
            "{fit_date}: the long unchanged-price screen excluded {}. Repeated prices do not prove an invalid quote.",
            excluded.join(", ")
        ));
    }
    Ok(Fit {
        selected: full(&selected),
        equal: full(&vec![1.0 / n as f64; n]),
        mean: full(&mean_values),
        covariance: LabeledMatrix::square(prices.columns.clone(), full_covariance),
        coverage,
        warnings,
    })
}

/// Buy each allocation once at the next close, then let its weights drift.
pub(crate) fn simulate(
    prices: &Frame,
    benchmark_prices: Option<&Frame>,
    start: usize,
    end: usize,
    fit: &Fit,
    settings: &Dict,
) -> BacktestStudy {
    let dates = &prices.index;
    let execution = start + 1;
    let rate = settings.f64("cost_bps") / 10_000.0;
    let risk_free_rate = settings.f64("risk_free_rate");
    let periods_per_year = settings.f64("periods_per_year");
    let stock_returns = prices.pct_change();
    let mut specifications: Vec<(String, Frame, Vec<f64>)> = vec![
        (SELECTED.to_string(), stock_returns.clone(), fit.selected.values.clone()),
        (BASELINE.to_string(), stock_returns, fit.equal.values.clone()),
    ];
    if let Some(benchmarks) = benchmark_prices {
        for symbol in &benchmarks.columns {
            let returns = benchmarks.select(&[symbol]).expect("benchmark column").pct_change();
            specifications.push((benchmark_name(symbol).expect("validated benchmark").to_string(), returns, vec![1.0]));
        }
    }
    let mut equity_columns = Vec::new();
    let mut allocations = IndexMap::new();
    let mut holdings = IndexMap::new();
    let mut trades = IndexMap::new();
    let mut names = Vec::new();
    let mut metric_rows: Vec<Vec<Cell>> = Vec::new();
    for (name, returns, target) in specifications {
        let m = returns.ncols();
        let mut positions = vec![0.0; m];
        let (mut cash, mut nav) = (1.0, 1.0);
        let mut contribution = vec![0.0; m + 1];
        let mut exposure = vec![0.0; m + 1];
        let mut maximum_weight = vec![0.0; m + 1];
        let mut path = vec![nav];
        let mut target_rows = Vec::new();
        let mut trade_dates = Vec::new();
        let mut trade_values: Vec<[f64; 4]> = Vec::new();
        let mut trade_windows: Vec<(String, String)> = Vec::new();
        let observe = |maximum_weight: &mut Vec<f64>, positions: &[f64], cash: f64, nav: f64| {
            for (k, position) in positions.iter().enumerate() {
                maximum_weight[k] = maximum_weight[k].max(position / nav);
            }
            maximum_weight[m] = maximum_weight[m].max(cash / nav);
        };
        for i in start + 1..=end {
            for (k, position) in positions.iter().enumerate() {
                exposure[k] += position / nav;
            }
            exposure[m] += cash / nav;
            for (k, position) in positions.iter_mut().enumerate() {
                let profit = *position * returns.get(i - 1, k);
                contribution[k] += profit;
                *position += profit;
            }
            nav = positions.iter().sum::<f64>() + cash;
            observe(&mut maximum_weight, &positions, cash, nav);
            if i == execution {
                let before = nav;
                let after = after_cost(before, &positions, &target, rate);
                let traded: f64 = target.iter().zip(&positions).map(|(t, p)| (after * t - p).abs()).sum();
                trade_values.push([traded / before, before - after, before, after]);
                trade_windows.push((iso(dates[1]), iso(dates[i - 1])));
                trade_dates.push(dates[i]);
                let residual = (1.0 - target.iter().sum::<f64>()).max(0.0);
                let mut row = target.clone();
                row.push(residual);
                target_rows.push(row);
                positions = target.iter().map(|t| after * t).collect();
                cash = after * residual;
                nav = after;
                observe(&mut maximum_weight, &positions, cash, nav);
            }
            path.push(nav);
        }
        let mut columns = returns.columns.clone();
        columns.push("CASH".to_string());
        let allocation = Frame::from_rows(trade_dates.clone(), columns.clone(), &target_rows);
        let trade_table = Table::new(None, trade_dates.iter().map(|date| Cell::Date(*date)).collect())
            .with("turnover", trade_values.iter().map(|v| Cell::Float(v[0])).collect())
            .with("cost", trade_values.iter().map(|v| Cell::Float(v[1])).collect())
            .with("nav_before", trade_values.iter().map(|v| Cell::Float(v[2])).collect())
            .with("nav_after", trade_values.iter().map(|v| Cell::Float(v[3])).collect())
            .with("train_start", trade_windows.iter().map(|(s, _)| Cell::Str(s.clone())).collect())
            .with("train_end", trade_windows.iter().map(|(_, e)| Cell::Str(e.clone())).collect());
        let mut end_weight: Vec<f64> = positions.iter().map(|p| p / nav).collect();
        end_weight.push(cash / nav);
        let frequency: Vec<f64> = (0..columns.len())
            .map(|j| mean(&allocation.col(j).iter().map(|w| f64::from(u8::from(*w > 1e-6))).collect::<Vec<_>>()))
            .collect();
        let maximum_target: Vec<f64> =
            (0..columns.len()).map(|j| allocation.col(j).into_iter().fold(f64::NAN, f64::max)).collect();
        let floats = |values: &[f64]| values.iter().map(|v| Cell::Float(*v)).collect::<Vec<_>>();
        let held = Table::labelled(None, &columns)
            .with("average_weight", floats(&exposure.iter().map(|e| e / (end - start) as f64).collect::<Vec<_>>()))
            .with("end_weight", floats(&end_weight))
            .with("selection_frequency", floats(&frequency))
            .with("max_target_weight", floats(&maximum_target))
            .with("max_realized_weight", floats(&maximum_weight))
            .with("pnl_contribution", floats(&contribution));
        let performance = performance_metrics(&path, risk_free_rate, periods_per_year);
        let mut row: Vec<Cell> = performance.fields().iter().map(|(_, v)| Cell::Float(*v)).collect();
        row.push(Cell::Float(mean(&path_returns(&path)) * periods_per_year));
        row.push(Cell::Float(trade_values.iter().map(|v| v[0]).sum()));
        row.push(Cell::Float(trade_values.iter().map(|v| v[1]).sum()));
        row.push(Cell::Int(trade_values.len() as i64));
        row.push(Cell::Int(0));
        row.push(Cell::Int(0));
        metric_rows.push(row);
        names.push(name.clone());
        equity_columns.push((name.clone(), path));
        allocations.insert(name.clone(), allocation);
        holdings.insert(name.clone(), held);
        trades.insert(name, trade_table);
    }
    let metric_names = [
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "sortino",
        "calmar",
        "annual_mean",
        "total_turnover",
        "total_cost",
        "trade_count",
        "rebalance_count",
        "fallback_count",
    ];
    let mut metrics = Table::labelled(Some("strategy"), &names);
    for (k, metric) in metric_names.iter().enumerate() {
        metrics.push(*metric, metric_rows.iter().map(|row| row[k].clone()).collect());
    }
    let mut study_settings = settings.clone();
    study_settings.update(&dict! {
        "split" => start,
        "train_start" => iso(dates[1]),
        "train_end" => iso(dates[start]),
        "test_start" => iso(dates[start + 1]),
        "test_end" => iso(dates[end]),
        "initial_execution" => iso(dates[start + 1]),
    });
    BacktestStudy {
        equity: Frame::from_columns(dates[start..=end].to_vec(), equity_columns),
        metrics,
        allocations,
        holdings,
        trades,
        warnings: fit.warnings.clone(),
        settings: study_settings,
    }
}

/// Settings for [`run_selection`], with the Python defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionOptions {
    pub target_volatility: f64,
    pub max_weight: f64,
    pub train_fraction: f64,
    pub cost_bps: f64,
    pub shrinkage: f64,
    pub periods_per_year: f64,
    pub risk_free_rate: f64,
}

impl Default for SelectionOptions {
    fn default() -> Self {
        Self {
            target_volatility: 0.2,
            max_weight: 0.1,
            train_fraction: 0.7,
            cost_bps: 10.0,
            shrinkage: 0.0,
            periods_per_year: TRADING_DAYS,
            risk_free_rate: 0.02,
        }
    }
}

/// Fit one historical-mean allocation and hold it through the later dates.
///
/// The initial fit uses prior closes. The first interval is zero-interest cash
/// and execution occurs at the next close. No allocation trades again.
/// Equal-weight stocks and raw SPY/QQQ funds use the same buy-and-hold policy.
/// The separate latest fit uses all prices and does not alter the replay.
///
/// The volatility target constrains estimated risk, not realized losses or risk.
/// This historical replay is not a claim of prospective validation.
pub fn run_selection(
    prices: &Frame,
    benchmark_prices: Option<&Frame>,
    options: &SelectionOptions,
) -> Result<SelectionStudy> {
    validate_price_frame(prices)?;
    if prices.columns.iter().any(|column| benchmark_name(&column.trim().to_uppercase()).is_some()) {
        return Err(value_error!(
            "SPY and QQQ are comparison benchmarks, not stock holdings. Use split_stock_prices before run_selection."
        ));
    }
    if prices.position("CASH").is_some() {
        return Err(value_error!("CASH is reserved for the zero-interest cash allocation."));
    }
    let SelectionOptions { target_volatility, max_weight, train_fraction, cost_bps, shrinkage, risk_free_rate, .. } =
        *options;
    let periods_per_year = validate_periods_per_year(options.periods_per_year)?;
    for (name, value) in [("target_volatility", target_volatility), ("max_weight", max_weight)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(value_error!("{name} must be finite and greater than zero."));
        }
    }
    if max_weight > 1.0 {
        return Err(value_error!("max_weight must be at most one."));
    }
    if !train_fraction.is_finite() || !(train_fraction > 0.0 && train_fraction < 1.0) {
        return Err(value_error!("train_fraction must be between zero and one."));
    }
    if !shrinkage.is_finite() || !(0.0..=1.0).contains(&shrinkage) {
        return Err(value_error!("shrinkage must be between zero and one."));
    }
    if !risk_free_rate.is_finite() {
        return Err(value_error!("risk_free_rate must be finite."));
    }
    if !cost_bps.is_finite() || !(0.0..10_000.0).contains(&cost_bps) {
        return Err(value_error!("cost_bps must be finite, nonnegative and less than 10000."));
    }
    let benchmark_prices =
        benchmark_prices.map(|frame| validate_selection_benchmarks(frame, &prices.index)).transpose()?;
    let count = prices.nrows() - 1;
    let initial_end = (count as f64 * train_fraction).floor() as usize;
    if initial_end.min(count - initial_end) < 2 {
        return Err(value_error!(
            "Selection needs at least two returns in each initial training and later test period."
        ));
    }
    let dates = &prices.index;
    let evidence_floor = periods_per_year.ceil() as i64;
    let mut settings = dict! {
        "model_revision" => MODEL_REVISION,
        "target_volatility" => target_volatility,
        "risk_target" => target_volatility,
        "max_weight" => max_weight,
        "train_fraction" => train_fraction,
        "holding_policy" => "buy_and_hold",
        "cost_bps" => cost_bps,
        "shrinkage" => shrinkage,
        "periods_per_year" => periods_per_year,
        "risk_free_rate" => risk_free_rate,
        "cash_interest_rate" => 0.0,
        "model_objective" => "Maximum historical arithmetic mean return within the position cap and estimated volatility ceiling.",
        "evidence_floor_returns" => evidence_floor,
        "evidence_floor_policy" => "At least one declared year of later test returns. This descriptive minimum does not measure statistical significance.",
        "price_start" => iso(dates[0]),
        "price_end" => iso(dates[count]),
        "initial_training_start" => iso(dates[1]),
        "initial_training_end" => iso(dates[initial_end]),
        "test_start" => iso(dates[initial_end + 1]),
        "test_end" => iso(dates[count]),
        "initial_training_returns" => initial_end,
        "test_returns" => count - initial_end,
        "execution" => "Prior-close information; execute at the following close; first interval in zero-interest cash.",
        "cost_convention" => "Basis points per absolute dollar bought or sold; cash transfers excluded; no final liquidation.",
        "risk_constraint" => "Annual estimated stock volatility ceiling; realized volatility can exceed the target.",
        "baseline" => "Buy equal weights across initially eligible stocks once, without a volatility constraint, then let weights drift.",
        "comparisons" => "Equal-weight stocks and raw SPY/QQQ buy-and-hold funds. Each uses the same initial delay and trading costs.",
        "screen" => format!("Exclude stocks with at least {STALE_INTERVALS} consecutive unchanged price intervals in past fit data only."),
        "test_role" => "Later buy-and-hold replay of the initial past-data fit, without rebalancing. The researcher can already know these dates.",
    };
    let initial_fit =
        fit_targets(&prices.rows(0, initial_end + 1), target_volatility, max_weight, shrinkage, periods_per_year)?;
    let mut test = simulate(prices, benchmark_prices.as_ref(), initial_end, count, &initial_fit, &settings);
    let mut reasons = Vec::new();
    let missing: Vec<&str> = BENCHMARK_NAMES
        .iter()
        .map(|(symbol, _)| *symbol)
        .filter(|symbol| benchmark_prices.as_ref().is_none_or(|frame| frame.position(symbol).is_none()))
        .collect();
    if !missing.is_empty() {
        reasons.push(format!(
            "Market benchmark prices are unavailable for {}. The historical check needs both SPY and QQQ.",
            missing.join(", ")
        ));
    }
    if ((count - initial_end) as i64) < evidence_floor {
        reasons.push(
            "The later test has less than one declared year of return intervals. The evidence floor is not met."
                .to_string(),
        );
    }
    let candidate_volatility = test.metrics.f64_at(SELECTED, "volatility");
    let candidate_cagr = test.metrics.f64_at(SELECTED, "cagr");
    if candidate_volatility > target_volatility + 1e-10 {
        reasons.push("Realized annual volatility in the later test exceeded the requested target.".to_string());
    }
    for name in test.metrics.index_labels() {
        if name == SELECTED {
            continue;
        }
        let cagr = test.metrics.f64_at(&name, "cagr");
        let volatility = test.metrics.f64_at(&name, "volatility");
        if cagr > candidate_cagr + 1e-10 && volatility <= candidate_volatility + 1e-10 {
            reasons
                .push(format!("{name} had higher compound growth and no more realized volatility in the later test."));
        }
    }
    let latest_fit = fit_targets(prices, target_volatility, max_weight, shrinkage, periods_per_year)?;
    let weights = latest_fit.selected.clone();
    let expected = dot(&latest_fit.mean.values, &weights.values);
    let volatility = quadratic_form(&weights.values, &latest_fit.covariance.values).max(0.0).sqrt();
    let mut latest = IndexMap::new();
    latest.insert(
        SELECTED.to_string(),
        Portfolio {
            name: SELECTED.to_string(),
            weights,
            expected_return: expected,
            volatility,
            sharpe: if volatility > 1e-12 { (expected - risk_free_rate) / volatility } else { f64::NAN },
        },
    );
    let mut warnings: Vec<String> = Vec::new();
    for message in initial_fit.warnings.iter().chain(&latest_fit.warnings) {
        if !warnings.contains(message) {
            warnings.push(message.clone());
        }
    }
    warnings.push(
        "Historical means and estimated volatility can change. A passed historical check does not guarantee future outperformance."
            .to_string(),
    );
    let eligibility = concat_rows(&[&initial_fit.coverage, &latest_fit.coverage]);
    settings.insert("fit_dates", Json::from(vec![iso(dates[initial_end]), iso(dates[count])]));
    settings.insert("latest_as_of", iso(dates[count]));
    test.settings.update(&settings);
    Ok(SelectionStudy {
        test,
        latest,
        selected: SELECTED.to_string(),
        supported: reasons.is_empty(),
        reasons,
        settings,
        warnings,
        benchmark_prices,
        eligibility,
    })
}

/// `pd.concat(tables, ignore_index=True)` for tables with the same columns.
pub fn concat_rows(tables: &[&Table]) -> Table {
    let rows: usize = tables.iter().map(|table| table.nrows()).sum();
    let mut result = Table::range(rows);
    if let Some(first) = tables.first() {
        for column in &first.columns {
            result.push(column, tables.iter().flat_map(|table| table.column(column).to_vec()).collect());
        }
    }
    result
}

/// The latest fitted frontier, its stock weights, and same-date references.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionFrontier {
    /// Default index; columns `fitted_mean`, `model_volatility`, `sharpe`, `historical_volatility`.
    pub frontier: Table,
    /// Default index; one column per price column.
    pub weights: Table,
    /// Index `portfolio`; columns `fitted_mean`, `model_volatility`, `historical_volatility`.
    pub comparisons: Table,
    pub warnings: Vec<String>,
}

/// Return the latest fitted frontier, its stock weights, and same-date references.
///
/// All rows use annual arithmetic means. The main model uses sample covariance.
/// Explicit research shrinkage changes stock model risk, not ETF sample risk.
/// This describes the fit period, not the returns of the later historical test.
pub fn selection_frontier(study: &SelectionStudy, prices: &Frame, frontier_points: usize) -> Result<SelectionFrontier> {
    validate_price_frame(prices)?;
    let latest_date = study.settings.str("latest_as_of");
    if iso(*prices.index.last().unwrap()) != latest_date {
        return Err(value_error!("Frontier prices must end on the saved latest fit date."));
    }
    let fit_dates = study.eligibility.column("fit_date");
    let symbols = study.eligibility.column("symbol");
    let eligible = study.eligibility.column("eligible");
    let labels: Vec<String> = (0..study.eligibility.nrows())
        .filter(|i| fit_dates[*i].as_str() == Some(latest_date) && eligible[*i].as_bool() == Some(true))
        .map(|i| symbols[i].py_str())
        .collect();
    let returns = prices
        .select(&labels)
        .ok_or_else(|| value_error!("Frontier prices must contain every eligible stock."))?
        .pct_change();
    let periods = study.settings.f64("periods_per_year");
    let (mean, _, sample) = estimates(&returns, periods, 0.0);
    let covariance = LabeledMatrix::square(labels.clone(), shrink(&sample, study.settings.f64("shrinkage")));
    let optimization = optimize(
        &mean,
        &covariance,
        study.settings.f64("risk_free_rate"),
        study.settings.f64("max_weight"),
        frontier_points,
    )?;
    let points = &optimization.frontier;
    let historical: Vec<f64> = points.weights.iter().map(|w| quadratic_form(w, &sample).max(0.0).sqrt()).collect();
    let floats = |values: &[f64]| values.iter().map(|v| Cell::Float(*v)).collect::<Vec<_>>();
    let frontier = Table::range(points.len())
        .with("fitted_mean", floats(&points.expected_return))
        .with("model_volatility", floats(&points.volatility))
        .with("sharpe", floats(&points.sharpe))
        .with("historical_volatility", floats(&historical));
    let mut weights = Table::range(points.len());
    for column in &prices.columns {
        let values = match labels.iter().position(|label| label == column) {
            Some(k) => points.weights.iter().map(|w| Cell::Float(w[k])).collect(),
            None => vec![Cell::Float(0.0); points.len()],
        };
        weights.push(column, values);
    }
    let fitted = selection_fit_estimates(study, prices);
    let last = fitted.nrows() - 1;
    let mut names = vec![SELECTED.to_string()];
    let mut rows = vec![[
        fitted.f64s("fitted_mean")[last],
        fitted.f64s("model_volatility")[last],
        fitted.f64s("historical_volatility")[last],
    ]];
    if let Some(benchmarks) = &study.benchmark_prices {
        let benchmarks = validate_selection_benchmarks(benchmarks, &prices.index)?;
        let benchmark_returns = benchmarks.pct_change();
        for (j, symbol) in benchmarks.columns.iter().enumerate() {
            let column = benchmark_returns.col(j);
            let volatility = var(&column, 1).sqrt() * periods.sqrt();
            names.push(benchmark_name(symbol).unwrap().to_string());
            rows.push([crate::frame::mean(&column) * periods, volatility, volatility]);
        }
    }
    let comparisons = Table::labelled(Some("portfolio"), &names)
        .with("fitted_mean", rows.iter().map(|r| Cell::Float(r[0])).collect())
        .with("model_volatility", rows.iter().map(|r| Cell::Float(r[1])).collect())
        .with("historical_volatility", rows.iter().map(|r| Cell::Float(r[2])).collect());
    Ok(SelectionFrontier { frontier, weights, comparisons, warnings: optimization.warnings })
}

/// Recover each saved fit's estimates from its weights and past prices only.
///
/// Historical volatility uses the sample covariance before shrinkage. Neither
/// volatility describes the later replay. No portfolio is solved again.
pub fn selection_fit_estimates(study: &SelectionStudy, prices: &Frame) -> Table {
    let allocations = &study.test.allocations[SELECTED];
    let trades = &study.test.trades[SELECTED];
    let execution = allocations.index[0];
    let trade_row = trades.index[0].iter().position(|cell| cell.as_date() == Some(execution)).expect("entry trade");
    let train_end = trades.column("train_end")[trade_row].py_str();
    let first_fit = Date::parse_from_str(&train_end, "%Y-%m-%d").expect("ISO training end");
    let initial = Series::new(allocations.columns.clone(), allocations.row(0));
    let latest = &study.latest[&study.selected].weights;
    let targets = [
        (first_fit, Some(execution), "First historical fit", initial.reindex(&prices.columns, 0.0)),
        (*prices.index.last().unwrap(), None, "Latest fit", latest.reindex(&prices.columns, 0.0)),
    ];
    let periods = study.settings.f64("periods_per_year");
    let shrinkage = study.settings.f64("shrinkage");
    let mut table = Table::new(Some("fit_date"), Vec::new());
    let mut columns: Vec<Vec<Cell>> = vec![Vec::new(); 7];
    for (fit_date, execution, phase, weights) in targets {
        let returns = prices.rows_through(fit_date).pct_change();
        let portfolio: Vec<f64> = (0..returns.nrows()).map(|i| dot(&returns.row(i), &weights.values)).collect();
        let raw_variance = var(&portfolio, 1) * periods;
        let diagonal_variance: f64 =
            (0..returns.ncols()).map(|j| var(&returns.col(j), 1) * periods * weights.values[j].powi(2)).sum();
        let model_variance = (1.0 - shrinkage) * raw_variance + shrinkage * diagonal_variance;
        table.index[0].push(Cell::Date(*returns.index.last().unwrap()));
        columns[0].push(Cell::from(phase));
        columns[1].push(Cell::from(execution));
        columns[2].push(Cell::Int(returns.nrows() as i64));
        columns[3].push(Cell::Float(mean(&portfolio) * periods));
        columns[4].push(Cell::Float(model_variance.max(0.0).sqrt()));
        columns[5].push(Cell::Float(raw_variance.max(0.0).sqrt()));
        columns[6].push(Cell::Float(weights.max()));
    }
    let names = [
        "phase",
        "execution_date",
        "observations",
        "fitted_mean",
        "model_volatility",
        "historical_volatility",
        "max_weight",
    ];
    for (name, values) in names.iter().zip(columns) {
        table.push(*name, values);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn prices_from_returns(values: &[Vec<f64>], names: &[&str]) -> Frame {
        let start = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let mut index = Vec::new();
        let mut day = start;
        while index.len() < values.len() + 1 {
            if chrono::Datelike::weekday(&day).number_from_monday() <= 5 {
                index.push(day);
            }
            day = day.succ_opt().unwrap();
        }
        let mut rows = vec![vec![100.0; names.len()]];
        for row in values {
            let previous = rows.last().unwrap().clone();
            rows.push(previous.iter().zip(row).map(|(p, r)| p * (1.0 + r)).collect());
        }
        Frame::from_rows(index, names.iter().map(|n| n.to_string()).collect(), &rows)
    }

    fn tile(pattern: &[f64], count: usize) -> Vec<f64> {
        (0..count).map(|i| pattern[i % pattern.len()]).collect()
    }

    #[test]
    fn mean_objective_has_analytic_cap_and_equal_weight_comparator() {
        let a = tile(&[-0.002, 0.002], 24).iter().map(|v| 0.005 + v).collect::<Vec<_>>();
        let b = tile(&[-0.004, -0.004, 0.004, 0.004], 24).iter().map(|v| 0.02 + v).collect::<Vec<_>>();
        let values: Vec<Vec<f64>> = (0..24).map(|i| vec![a[i], b[i]]).collect();
        let prices = prices_from_returns(&values, &["A", "B"]);
        let fit = fit_targets(&prices, 0.5, 0.7, 1.0, 12.0).unwrap();
        assert!((fit.selected.values[0] - 0.3).abs() < 1e-7);
        assert!((fit.selected.values[1] - 0.7).abs() < 1e-7);
        assert_eq!(fit.equal.values, vec![0.5, 0.5]);
        assert!((fit.mean.values[0] - 0.06).abs() < 1e-12);
        assert!((fit.mean.values[1] - 0.24).abs() < 1e-12);
    }

    #[test]
    fn binding_risk_limit_maximizes_mean() {
        let a = tile(&[-0.01, 0.01], 24).iter().map(|v| 0.002 + v).collect::<Vec<_>>();
        let b = tile(&[-0.10, -0.10, 0.10, 0.10], 24).iter().map(|v| 0.02 + v).collect::<Vec<_>>();
        let values: Vec<Vec<f64>> = (0..24).map(|i| vec![a[i], b[i]]).collect();
        let prices = prices_from_returns(&values, &["A", "B"]);
        let fit = fit_targets(&prices, 0.10, 0.9, 1.0, 12.0).unwrap();
        let variance = [var(&a, 1) * 12.0, var(&b, 1) * 12.0];
        // (1-b)^2 va + b^2 vb = 0.01, larger root.
        let (qa, qb, qc) = (variance[0] + variance[1], -2.0 * variance[0], variance[0] - 0.01);
        let root = (-qb + (qb * qb - 4.0 * qa * qc).sqrt()) / (2.0 * qa);
        assert!((fit.selected.values[1] - root).abs() < 3e-6, "{} vs {root}", fit.selected.values[1]);
        let volatility = quadratic_form(&fit.selected.values, &fit.covariance.values).sqrt();
        assert!((volatility - 0.10).abs() < 1e-7);
    }

    #[test]
    fn infeasible_budget_reports_the_feasible_minimum() {
        let a = tile(&[-0.01, 0.012], 30);
        let b = tile(&[0.011, -0.009, 0.002], 30);
        let values: Vec<Vec<f64>> = (0..30).map(|i| vec![a[i], b[i]]).collect();
        let prices = prices_from_returns(&values, &["A", "B"]);
        let error = fit_targets(&prices, 0.0001, 1.0, 0.0, 252.0).unwrap_err();
        assert!(error.message().contains("is below the feasible minimum"), "{error}");
        assert!(error.is_value());
    }

    #[test]
    fn replay_buys_once_after_one_cash_interval_and_pays_entry_cost() {
        let count = 40;
        let values: Vec<Vec<f64>> = (0..count)
            .map(|i| {
                (0..12).map(|j| 0.0005 * (j as f64 + 1.0) + 0.004 * (((i * (j + 3)) % 7) as f64 - 3.0) / 3.0).collect()
            })
            .collect();
        let names: Vec<String> = (0..12).map(|j| format!("STOCK{j}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let prices = prices_from_returns(&values, &refs);
        let study =
            run_selection(&prices, None, &SelectionOptions { target_volatility: 0.5, ..Default::default() }).unwrap();
        let equity = &study.test.equity;
        assert_eq!(equity.nrows(), count - 28 + 1);
        for j in 0..equity.ncols() {
            assert_eq!(equity.get(0, j), 1.0);
            assert!((equity.get(1, j) - 1.0 / 1.001).abs() < 1e-12);
        }
        assert!(!study.supported);
        assert!(study.reasons[0].starts_with("Market benchmark prices are unavailable for SPY, QQQ."));
        assert_eq!(study.eligibility.nrows(), 24);
        assert_eq!(study.settings.str("latest_as_of"), iso(*prices.index.last().unwrap()));
        let estimates = selection_fit_estimates(&study, &prices);
        assert_eq!(estimates.nrows(), 2);
        assert_eq!(estimates.column("phase")[1], Cell::from("Latest fit"));
        assert!(estimates.column("execution_date")[1].is_null());
        let frontier = selection_frontier(&study, &prices, 30).unwrap();
        assert_eq!(frontier.comparisons.index_labels(), vec![SELECTED.to_string()]);
        assert!(frontier.frontier.nrows() > 1);
    }
}
