//! Display formats of the app that `pyfmt` does not cover.

use efficient_frontier::frame::Date;
use efficient_frontier::pyfmt;

/// `day(value)`: a date as "2 Oct 2026".
pub fn day(date: Date) -> String {
    date.format("%-d %b %Y").to_string()
}

/// [`day`] of ISO date text. Other text is returned unchanged.
pub fn day_text(text: &str) -> String {
    Date::parse_from_str(text, "%Y-%m-%d").map_or_else(|_| text.to_string(), day)
}

/// `replay_points(value)`: a rate difference in percentage points for KaTeX, with
/// a power of ten for differences below 0.005 points.
pub fn replay_points(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let scaled = 100.0 * value;
    if scaled.abs() >= 0.005 {
        return pyfmt::format(scaled, "+.2f");
    }
    let text = pyfmt::format(scaled, "+.2e");
    let (mantissa, exponent) = text.split_once('e').expect("exponent notation");
    format!("{mantissa}\\times10^{{{}}}", exponent.parse::<i32>().expect("integer exponent"))
}

/// Remove trailing fraction zeros and a trailing decimal point.
fn trim_fraction(text: String) -> String {
    if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text }
}

/// The default number format of `st.dataframe`: four decimals without trailing
/// zeros, or enough decimals to show a value below 0.0001. Missing values are empty.
pub fn number(value: f64) -> String {
    if !value.is_finite() {
        return String::new();
    }
    let decimals = if value == 0.0 || value.abs() >= 1e-4 {
        4
    } else {
        let text = format!("{value:e}");
        text.split_once('e').map_or(4, |(_, exponent)| exponent.parse::<i32>().map_or(4, |e| e.unsigned_abs())) as usize
    };
    trim_fraction(format!("{value:.decimals$}"))
}

/// `NumberColumn(format="percent")`: a percentage with at most two decimals.
pub fn percent_column(value: f64) -> String {
    if !value.is_finite() {
        return String::new();
    }
    format!("{}%", trim_fraction(format!("{:.2}", value * 100.0)))
}

/// `f"{value:.2%}"` with `na_rep="—"` for missing values.
pub fn percent_or_dash(value: f64) -> String {
    if value.is_nan() { "—".into() } else { pyfmt::format(value, ".2%") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_points_match_python() {
        assert_eq!(day(Date::from_ymd_opt(2026, 10, 2).unwrap()), "2 Oct 2026");
        assert_eq!(day_text("2021-01-04"), "4 Jan 2021");
        assert_eq!(day_text("not a date"), "not a date");
        assert_eq!(replay_points(0.0), "0");
        assert_eq!(replay_points(0.0123), "+1.23");
        assert_eq!(replay_points(-0.0317), "-3.17");
        assert_eq!(replay_points(0.00000123), "+1.23\\times10^{-4}");
        assert_eq!(replay_points(-0.0000005), "-5.00\\times10^{-5}");
    }

    #[test]
    fn table_numbers_follow_streamlit() {
        assert_eq!(number(99.54586087604353), "99.5459");
        assert_eq!(number(1255.0), "1255");
        assert_eq!(number(0.5), "0.5");
        assert_eq!(number(0.0), "0");
        assert_eq!(number(0.000012), "0.00001");
        assert_eq!(number(f64::NAN), "");
        assert_eq!(percent_column(0.1234), "12.34%");
        assert_eq!(percent_column(0.1), "10%");
        assert_eq!(percent_column(1.0), "100%");
        assert_eq!(percent_or_dash(f64::NAN), "—");
        assert_eq!(percent_or_dash(-0.0123), "-1.23%");
    }
}
