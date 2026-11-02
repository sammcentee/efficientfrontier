//! Efficient frontier analysis for US equities.
//!
//! The library holds the stock model, the legacy research tools, data loading,
//! and report generation. Two programs use it: `efficient-frontier` (the CLI)
//! and `portfolio-lab` (the local web app).

pub mod backtest;
pub mod backtest_report;
pub mod benchmark_report;
pub mod benchmarks;
pub mod cli;
pub mod companies;
pub mod core;
pub mod data;
pub mod error;
pub mod evidence;
pub mod frame;
pub mod historical;
pub mod historical_report;
pub mod html_table;
pub mod membership;
pub mod metrics;
pub mod net;
pub mod out_of_sample;
pub mod plot;
pub mod presentation;
pub mod profiles;
pub mod pyfmt;
pub mod risk;
pub mod risk_reference;
pub mod selection;
pub mod selection_report;
pub mod solver;
pub mod story;
pub mod style;
pub mod universe;

pub use error::{Error, Result};
