//! Charts and portable reports for the legacy classic analysis, and shared CSV export.
//!
//! Port of `efficient_frontier/presentation.py`. `csv_text` is
//! [`crate::frame::Table::to_csv`] and [`crate::frame::Frame::to_csv`].
//! Owner: reports agent (wave 2 for `report_zip`).

pub use crate::plot::PLOTLY_JS_LICENSE;
