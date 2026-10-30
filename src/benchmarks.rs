//! Independent market references with the same evaluation dates and costs.

use crate::backtest::BacktestStudy;
use crate::core::{Analysis, validate_price_frame};
use crate::data::validate_benchmark_prices;
use crate::dict;
use crate::error::Result;
use crate::frame::{Cell, Date, Frame, Table, iso, mean, std};
use crate::metrics::{PERFORMANCE_FIELDS, performance_metrics};
use crate::profiles::ProfileAnalysis;
use crate::pyfmt::Dict;
use crate::value_error;

/// The comparison ETFs and their display names, shared with the stock model.
pub use crate::selection::{BENCHMARK_NAMES, benchmark_name};

/// SPY and QQQ results on the dates of a classic analysis and its backtests.
///
/// Equity columns and table rows use the display names of [`BENCHMARK_NAMES`],
/// in the order SPY, QQQ.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkComparison {
    /// Validated SPY and QQQ prices on every asset date, with symbol columns.
    pub prices: Frame,
    /// Buy at the final training close without fees, on the holdout dates.
    pub holdout_equity: Frame,
    /// Index `benchmark`; the [`PERFORMANCE_FIELDS`] columns.
    pub holdout_metrics: Table,
    /// Zero-interest cash for the first interval, then entry at the next close
    /// after the backtest fee. `None` without a backtest study.
    pub backtest_equity: Option<Frame>,
    /// Index `benchmark`; the [`PERFORMANCE_FIELDS`] columns, then `total_turnover`,
    /// `total_cost`, and the integers `rebalance_count` and `fallback_count`.
    pub backtest_metrics: Option<Table>,
    /// Index `benchmark`; columns `expected_return`, `volatility`, `sharpe` on the training returns.
    pub training_estimates: Table,
    /// Index `benchmark`; columns `expected_return`, `volatility`, `sharpe` on every
    /// return, and `worst_window_return` (NaN without latest profiles).
    pub latest_estimates: Table,
    pub settings: Dict,
}

/// Display names for symbol columns.
fn display_names(symbols: &[String]) -> Vec<String> {
    symbols.iter().map(|symbol| benchmark_name(symbol).unwrap_or(symbol).to_string()).collect()
}

/// Use exact observation dates, with either original or delayed entry.
///
/// `rows` are the price positions of the evaluation dates. The original entry
/// (`delayed_fee` is `None`) buys at the first date and excludes fees. A delayed
/// entry holds cash for the first interval and buys at the second date after
/// the fee in basis points.
fn benchmark_buy_and_hold(prices: &Frame, rows: &[usize], delayed_fee: Option<f64>) -> Frame {
    let base = rows[usize::from(delayed_fee.is_some())];
    let columns = display_names(&prices.columns)
        .into_iter()
        .enumerate()
        .map(|(j, name)| {
            let start = prices.get(base, j);
            let path = rows
                .iter()
                .enumerate()
                .map(|(i, row)| match delayed_fee {
                    Some(_) if i == 0 => 1.0,
                    Some(cost_bps) => prices.get(*row, j) / start / (1.0 + cost_bps / 10_000.0),
                    None => prices.get(*row, j) / start,
                })
                .collect();
            (name, path)
        })
        .collect();
    Frame::from_columns(rows.iter().map(|row| prices.index[*row]).collect(), columns)
}

/// Annual mean, sample volatility and Sharpe ratio of each benchmark.
fn estimates(returns: &Frame, risk_free_rate: f64, periods_per_year: f64) -> Table {
    let mut rows = Vec::new();
    for j in 0..returns.ncols() {
        let column = returns.col(j);
        let expected = mean(&column) * periods_per_year;
        let volatility = std(&column, 1) * periods_per_year.sqrt();
        rows.push([
            expected,
            volatility,
            if volatility > 1e-12 { (expected - risk_free_rate) / volatility } else { f64::NAN },
        ]);
    }
    let column = |k: usize| rows.iter().map(|row| Cell::Float(row[k])).collect();
    Table::labelled(Some("benchmark"), &display_names(&returns.columns))
        .with("expected_return", column(0))
        .with("volatility", column(1))
        .with("sharpe", column(2))
}

/// Realized measures of each equity column.
fn metrics(equity: &Frame, risk_free_rate: f64, periods_per_year: f64) -> Table {
    let measured: Vec<_> =
        (0..equity.ncols()).map(|j| performance_metrics(&equity.col(j), risk_free_rate, periods_per_year)).collect();
    let mut table = Table::labelled(Some("benchmark"), &equity.columns);
    for (k, field) in PERFORMANCE_FIELDS.iter().enumerate() {
        table.push(*field, measured.iter().map(|metric| Cell::Float(metric.fields()[k].1)).collect());
    }
    table
}

/// Compare SPY and QQQ without any change to the portfolio universe.
///
/// The original holdout buys at the split close and excludes fees. Backtest
/// references buy at the next close with the study's entry fee. They earn zero
/// interest before entry and exclude terminal liquidation. Estimates use the
/// same observations as the corresponding portfolio estimates.
pub fn compare_benchmarks(
    benchmark_prices: &Frame,
    asset_prices: &Frame,
    analysis: &Analysis,
    study: Option<&BacktestStudy>,
    latest_profiles: Option<&ProfileAnalysis>,
    risk_free_rate: f64,
) -> Result<BenchmarkComparison> {
    if !risk_free_rate.is_finite() {
        return Err(value_error!("risk_free_rate must be finite."));
    }
    if analysis.risk_free_rate != risk_free_rate {
        return Err(value_error!("Analysis and benchmarks must use the same risk_free_rate."));
    }
    validate_price_frame(asset_prices)?;
    let train = analysis.train_returns.nrows();
    let expected_returns: Vec<Date> =
        analysis.train_returns.index.iter().chain(&analysis.test_returns.index).copied().collect();
    if asset_prices.index[1..] != expected_returns
        || asset_prices.index.get(train..).unwrap_or_default() != analysis.equity.index
    {
        return Err(value_error!("Asset prices and analysis must use the same observation dates."));
    }
    let prices = validate_benchmark_prices(benchmark_prices, &asset_prices.index)?;
    let periods = analysis.periods_per_year;
    let returns = prices.pct_change();
    // The return dates equal the analysis training dates followed by its test dates.
    let training_estimates = estimates(&returns.rows(0, train), risk_free_rate, periods);
    let mut latest_estimates = estimates(&returns, risk_free_rate, periods);
    let mut worst_window_return = vec![f64::NAN; returns.ncols()];
    if let Some(profiles) = latest_profiles {
        if profiles.settings.f64("periods_per_year") != periods {
            return Err(value_error!("Latest profiles and benchmarks must use the same periods_per_year."));
        }
        if profiles.settings.f64("risk_free_rate") != risk_free_rate {
            return Err(value_error!("Latest profiles and benchmarks must use the same risk_free_rate."));
        }
        if profiles.observations != returns.nrows() || profiles.as_of != iso(*prices.index.last().unwrap()) {
            return Err(value_error!("Latest profiles and benchmarks must use the same observation dates."));
        }
        let different_windows = || value_error!("Latest profiles and benchmarks must use the same historical windows.");
        worst_window_return = vec![f64::INFINITY; returns.ncols()];
        for ((start, end), observations) in profiles
            .windows
            .column("start")
            .iter()
            .zip(profiles.windows.column("end"))
            .zip(profiles.windows.column("observations"))
        {
            let date = |cell: &Cell| Date::parse_from_str(&cell.py_str(), "%Y-%m-%d").map_err(|_| different_windows());
            let (start, end) = (date(start)?, date(end)?);
            // `returns.loc[start:end]` includes both dates.
            let first = returns.index.partition_point(|day| *day < start);
            let last = returns.index.partition_point(|day| *day <= end).max(first);
            if observations.as_f64() != Some((last - first) as f64) {
                return Err(different_windows());
            }
            let sample = returns.rows(first, last);
            for (j, lowest) in worst_window_return.iter_mut().enumerate() {
                *lowest = lowest.min(mean(&sample.col(j)) * periods);
            }
        }
    }
    latest_estimates.push("worst_window_return", worst_window_return.into_iter().map(Cell::Float).collect());

    let holdout_rows: Vec<usize> = (train..prices.nrows()).collect();
    let holdout_equity = benchmark_buy_and_hold(&prices, &holdout_rows, None);
    let holdout_metrics = metrics(&holdout_equity, risk_free_rate, periods);
    let (mut backtest_equity, mut backtest_metrics, mut cost_bps) = (None, None, None);
    if let Some(study) = study {
        if study.equity.index != analysis.equity.index {
            return Err(value_error!("Backtests and benchmarks must use the same evaluation dates."));
        }
        if study.settings.f64("periods_per_year") != periods {
            return Err(value_error!("Backtests and benchmarks must use the same periods_per_year."));
        }
        if study.settings.f64("risk_free_rate") != risk_free_rate {
            return Err(value_error!("Backtests and benchmarks must use the same risk_free_rate."));
        }
        let fee = study.settings.f64("cost_bps");
        let equity = benchmark_buy_and_hold(&prices, &holdout_rows, Some(fee));
        let mut table = metrics(&equity, risk_free_rate, periods);
        let entry_value = 1.0 / (1.0 + fee / 10_000.0);
        let count = table.nrows();
        table.push("total_turnover", vec![Cell::Float(entry_value); count]);
        table.push("total_cost", vec![Cell::Float(1.0 - entry_value); count]);
        table.push("rebalance_count", vec![Cell::Int(1); count]);
        table.push("fallback_count", vec![Cell::Int(0); count]);
        backtest_equity = Some(equity);
        backtest_metrics = Some(table);
        cost_bps = Some(fee);
    }

    let mut names = Dict::new();
    for (symbol, name) in BENCHMARK_NAMES {
        names.insert(symbol, name);
    }
    let settings = dict! {
        "symbols" => prices.columns.clone(),
        "display_names" => names,
        "periods_per_year" => periods,
        "risk_free_rate" => risk_free_rate,
        "price_start" => iso(prices.index[0]),
        "price_end" => iso(*prices.index.last().unwrap()),
        "price_observations" => prices.nrows(),
        "holdout_entry" => iso(holdout_equity.index[0]),
        "holdout_cost_bps" => 0.0,
        "backtest_entry" => backtest_equity.as_ref().map(|equity: &Frame| iso(equity.index[1])),
        "backtest_cost_bps" => cost_bps,
        "alignment" => "Exact matches for every asset observation date. No fills or dropped portfolio dates.",
        "role" => "Independent buy-and-hold references. Portfolio position caps do not apply to benchmarks.",
        "backtest_execution" => "First interval in zero-interest cash. Entry at the next close. No terminal sale.",
    };
    Ok(BenchmarkComparison {
        prices,
        holdout_equity,
        holdout_metrics,
        backtest_equity,
        backtest_metrics,
        training_estimates,
        latest_estimates,
        settings,
    })
}
