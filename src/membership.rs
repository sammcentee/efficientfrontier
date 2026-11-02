//! Dated membership snapshots with explicit coverage and security identity limits.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;

use indexmap::{IndexMap, IndexSet};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

use crate::data::read_csv_rows;
use crate::error::Result;
use crate::frame::{Date, iso};
use crate::pyfmt::{Dict, Json};
use crate::{io_error, value_error};

const SNAPSHOTS_CSV: &str = include_str!("../assets/membership/nasdaq100_membership_snapshots.csv");
const PROVENANCE_JSON: &str = include_str!("../assets/membership/provenance.json");
/// The license of the bundled n100tickers membership data.
pub const N100TICKERS_LICENSE: &str = include_str!("../assets/membership/LICENSE.n100tickers");

const DEFAULT_SOURCE: &str = "User-supplied membership CSV";
const CSV_LIMITATIONS: [&str; 2] = [
    "User-supplied full rosters and coverage dates are not independently verified.",
    "Price-column identifiers must identify the same security throughout the input history.",
];

/// Python `str.isspace()` for one character.
pub(crate) fn is_space(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// A calendar date argument: a [`Date`] or `YYYY-MM-DD` text.
///
/// Python also accepts pandas timestamps and then rejects times and time
/// zones. A [`Date`] has neither, so only text needs a check.
pub trait CalendarDate {
    /// The date, or a value error whose message starts with `name`.
    fn calendar_date(&self, name: &str) -> Result<Date>;
}

impl CalendarDate for Date {
    fn calendar_date(&self, _name: &str) -> Result<Date> {
        Ok(*self)
    }
}

impl CalendarDate for &str {
    fn calendar_date(&self, name: &str) -> Result<Date> {
        let bytes = self.as_bytes();
        let iso_pattern = bytes.len() == 10
            && bytes
                .iter()
                .enumerate()
                .all(|(i, byte)| if i == 4 || i == 7 { *byte == b'-' } else { byte.is_ascii_digit() });
        if !iso_pattern {
            return Err(value_error!("{name} must be a date in YYYY-MM-DD format."));
        }
        Date::parse_from_str(self, "%Y-%m-%d").map_err(|_| value_error!("{name} must be a valid calendar date."))
    }
}

/// `[A-Za-z0-9][A-Za-z0-9._:-]*`: a price-column identifier without spaces.
fn is_identifier(symbol: &str) -> bool {
    let mut characters = symbol.chars();
    characters.next().is_some_and(|first| first.is_ascii_alphanumeric())
        && characters.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'))
}

/// Each complete roster takes effect on its date, through the next change.
///
/// Symbols are exact price-column identifiers. No ticker renames or share-class
/// conversions occur here. Custom data must supply stable security identifiers.
#[derive(Clone, Debug, PartialEq)]
pub struct MembershipHistory {
    snapshots: Vec<(Date, Vec<String>)>,
    coverage_end: Date,
    source: String,
    limitations: Vec<String>,
    provenance: Dict,
}

impl MembershipHistory {
    /// Check the rosters, their dates, the coverage end and the source.
    ///
    /// `provenance` holds the source metadata, including any
    /// `identity_exclusions` rules.
    pub fn new<D: CalendarDate>(
        snapshots: Vec<(D, Vec<String>)>,
        coverage_end: impl CalendarDate,
        source: &str,
        limitations: Vec<String>,
        provenance: Dict,
    ) -> Result<Self> {
        if snapshots.is_empty() {
            return Err(value_error!("Membership snapshots must contain at least one full roster."));
        }
        let dates =
            snapshots.iter().map(|(date, _)| date.calendar_date("Snapshot date")).collect::<Result<Vec<_>>>()?;
        if dates.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(value_error!("Snapshot dates must be unique and in increasing order."));
        }
        for (_, roster) in &snapshots {
            if roster.is_empty() {
                return Err(value_error!("Each membership snapshot must contain a nonempty roster."));
            }
            if !roster.iter().all(|symbol| is_identifier(symbol)) {
                return Err(value_error!(
                    "Membership symbols must be nonempty price-column identifiers without spaces."
                ));
            }
            if roster.iter().collect::<HashSet<_>>().len() != roster.len() {
                return Err(value_error!("A membership snapshot contains duplicate symbols."));
            }
        }
        let coverage_end = coverage_end.calendar_date("Membership coverage end")?;
        if coverage_end < dates[dates.len() - 1] {
            return Err(value_error!("Membership coverage end must be on or after the last snapshot."));
        }
        if source.trim_matches(is_space).is_empty() {
            return Err(value_error!("Membership source must be a nonempty description."));
        }
        Ok(Self {
            snapshots: dates.into_iter().zip(snapshots.into_iter().map(|(_, roster)| roster)).collect(),
            coverage_end,
            source: source.to_string(),
            limitations,
            provenance,
        })
    }

    /// The rosters with their effective dates, in date order.
    pub fn snapshots(&self) -> &[(Date, Vec<String>)] {
        &self.snapshots
    }

    pub fn coverage_start(&self) -> Date {
        self.snapshots[0].0
    }

    pub fn coverage_end(&self) -> Date {
        self.coverage_end
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn limitations(&self) -> &[String] {
        &self.limitations
    }

    /// Return all raw identifiers in their first observed order.
    pub fn symbols(&self) -> Vec<String> {
        let symbols: IndexSet<&String> = self.snapshots.iter().flat_map(|(_, roster)| roster).collect();
        symbols.into_iter().cloned().collect()
    }

    /// The provenance, then the source, coverage, counts and limitations.
    pub fn metadata(&self) -> Dict {
        let mut metadata = self.provenance.clone();
        metadata.insert("source", self.source.as_str());
        metadata.insert("coverage_start", iso(self.coverage_start()));
        metadata.insert("coverage_end", iso(self.coverage_end));
        metadata.insert("snapshot_count", self.snapshots.len());
        metadata.insert("symbol_count", self.symbols().len());
        metadata.insert("limitations", Json::from(self.limitations.clone()));
        metadata
    }

    /// Return the last full roster effective on or before the requested date.
    pub fn at(&self, date: impl CalendarDate) -> Result<&[String]> {
        let date = date.calendar_date("Membership lookup date")?;
        if date < self.coverage_start() || date > self.coverage_end {
            return Err(value_error!(
                "Membership date {} is outside coverage {} through {}.",
                iso(date),
                iso(self.coverage_start()),
                iso(self.coverage_end)
            ));
        }
        let position = self.snapshots.partition_point(|(start, _)| *start <= date);
        Ok(&self.snapshots[position - 1].1)
    }

    /// Find known unsafe raw labels anywhere in an inclusive price interval.
    ///
    /// Price history can predate membership coverage. Exclusion end dates are
    /// exclusive. This does not certify identities absent from the rule list.
    /// `through: None` checks the single date. The result maps each symbol to
    /// its reason, in rule order.
    pub fn identity_exclusions<D: CalendarDate>(
        &self,
        date: D,
        through: Option<D>,
    ) -> Result<IndexMap<String, String>> {
        let start = date.calendar_date("Identity interval start")?;
        let end = match through {
            Some(through) => through.calendar_date("Identity interval end")?,
            None => start,
        };
        if end < start {
            return Err(value_error!("Identity interval end must be on or after its start."));
        }
        let rules = self.provenance.get("identity_exclusions").and_then(Json::as_list).unwrap_or_default();
        let mut exclusions = IndexMap::new();
        for rule in rules {
            let rule = rule.as_dict().expect("an identity exclusion rule is an object");
            let bound = |key: &str| {
                rule.get(key).and_then(Json::as_str).map(|text| {
                    Date::parse_from_str(text, "%Y-%m-%d").expect("identity exclusion dates are YYYY-MM-DD")
                })
            };
            if bound("start").is_none_or(|first| end >= first) && bound("end").is_none_or(|last| start < last) {
                exclusions.insert(rule.str("symbol").to_string(), rule.str("reason").to_string());
            }
        }
        Ok(exclusions)
    }
}

/// Read `date,tickers` rows as `pandas.read_csv(dtype=str, keep_default_na=False, engine="python")`.
fn read_snapshot_rows(content: &[u8]) -> Result<Vec<(String, String)>> {
    let unreadable = || value_error!("Membership CSV must contain date,tickers columns and full rosters.");
    let text = std::str::from_utf8(content).map_err(|_| unreadable())?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    // pandas skips blank lines, including lines with one whitespace field.
    let mut rows = read_csv_rows(text)
        .map_err(|_| unreadable())?
        .into_iter()
        .filter(|row| row.len() > 1 || row.first().is_some_and(|field| !field.trim_matches(is_space).is_empty()));
    let header = rows.next().ok_or_else(unreadable)?;
    let mut data: Vec<Vec<String>> = rows.collect();
    // pandas reads extra leading fields of the first data row as an index. A
    // first row followed by a row with both widths gives index names instead.
    let mut index_columns = 0;
    if let Some(first) = data.first() {
        if data.get(1).is_some_and(|second| second.len() == first.len() + header.len()) {
            index_columns = first.len();
            data.remove(0);
        } else {
            index_columns = first.len().saturating_sub(header.len());
        }
    }
    let width = header.len() + index_columns;
    if data.iter().any(|row| row.len() > width) {
        return Err(unreadable());
    }
    if header != ["date", "tickers"] || data.is_empty() || index_columns > 0 || data.iter().any(|row| row.len() < width)
    {
        return Err(value_error!("Membership CSV must contain date,tickers columns and at least one full roster."));
    }
    Ok(data
        .into_iter()
        .map(|row| {
            let [date, tickers]: [String; 2] = row.try_into().expect("two checked fields");
            (date, tickers)
        })
        .collect())
}

/// Load a CSV with date,tickers columns and quoted comma-separated full rosters.
///
/// Each row replaces the previous roster. The caller must declare the last date
/// covered by the source. A last observed change alone does not establish it.
pub fn load_membership_csv(
    content: &[u8],
    coverage_end: impl CalendarDate,
    source: Option<&str>,
) -> Result<MembershipHistory> {
    let rows = read_snapshot_rows(content)?;
    let snapshots = rows
        .iter()
        .map(|(date, tickers)| {
            (date.as_str(), tickers.split(',').map(|symbol| symbol.trim_matches(is_space).to_string()).collect())
        })
        .collect();
    MembershipHistory::new(
        snapshots,
        coverage_end,
        source.unwrap_or(DEFAULT_SOURCE),
        CSV_LIMITATIONS.map(String::from).to_vec(),
        Dict::new(),
    )
}

/// Load a membership CSV file with [`load_membership_csv`].
///
/// A file that cannot be read is an [`Error::Io`](crate::Error::Io) with
/// Python's `OSError` text.
pub fn load_membership_csv_path(
    path: &Path,
    coverage_end: impl CalendarDate,
    source: Option<&str>,
) -> Result<MembershipHistory> {
    let content = std::fs::read(path).map_err(|error| {
        let text = error.to_string();
        match error.raw_os_error() {
            Some(code) => {
                let description = text.strip_suffix(&format!(" (os error {code})")).unwrap_or(&text);
                let name = Json::from(path.display().to_string()).py_repr();
                io_error!("[Errno {code}] {description}: {name}")
            }
            None => io_error!("{text}"),
        }
    })?;
    load_membership_csv(&content, coverage_end, source)
}

/// Load the pinned community Nasdaq-100 reconstruction without network access.
pub fn bundled_membership() -> MembershipHistory {
    let Json::Dict(provenance) = parse_json(PROVENANCE_JSON).expect("the bundled provenance is valid JSON") else {
        panic!("the bundled provenance is a JSON object");
    };
    let history =
        load_membership_csv(SNAPSHOTS_CSV.as_bytes(), provenance.str("coverage_end"), Some(provenance.str("source")))
            .expect("the bundled membership snapshots are valid");
    let limitations = provenance
        .get("limitations")
        .and_then(Json::as_list)
        .expect("the bundled provenance lists limitations")
        .iter()
        .map(Json::py_str)
        .collect();
    MembershipHistory { limitations, provenance, ..history }
}

/// Parse JSON text as Python's `json.loads`: objects keep their key order, and
/// integers stay separate from floats.
fn parse_json(text: &str) -> serde_json::Result<Json> {
    struct Ordered(Json);
    struct OrderedVisitor;

    impl<'de> Visitor<'de> for OrderedVisitor {
        type Value = Ordered;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a JSON value")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Ordered, E> {
            Ok(Ordered(Json::Bool(value)))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Ordered, E> {
            Ok(Ordered(Json::Int(value)))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Ordered, E> {
            Ok(Ordered(i64::try_from(value).map_or(Json::Float(value as f64), Json::Int)))
        }

        fn visit_f64<E>(self, value: f64) -> Result<Ordered, E> {
            Ok(Ordered(Json::Float(value)))
        }

        fn visit_str<E>(self, value: &str) -> Result<Ordered, E> {
            Ok(Ordered(Json::Str(value.to_string())))
        }

        fn visit_unit<E>(self) -> Result<Ordered, E> {
            Ok(Ordered(Json::Null))
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut items: A) -> Result<Ordered, A::Error> {
            let mut values = Vec::new();
            while let Some(Ordered(item)) = items.next_element()? {
                values.push(item);
            }
            Ok(Ordered(Json::List(values)))
        }

        fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> Result<Ordered, A::Error> {
            let mut values = Dict::new();
            while let Some((key, Ordered(item))) = entries.next_entry::<String, Ordered>()? {
                values.insert(key, item);
            }
            Ok(Ordered(Json::Dict(values)))
        }
    }

    impl<'de> Deserialize<'de> for Ordered {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_any(OrderedVisitor)
        }
    }

    serde_json::from_str::<Ordered>(text).map(|Ordered(value)| value)
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::dict;

    const CSV: &str = "date,tickers\n2020-01-01,\"B,A\"\n2020-01-06,\"A,C\"\n2020-01-10,\"B,C\"\n";

    fn day(text: &str) -> Date {
        Date::parse_from_str(text, "%Y-%m-%d").unwrap()
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn history(coverage_end: &str) -> MembershipHistory {
        load_membership_csv(CSV.as_bytes(), coverage_end, None).unwrap()
    }

    #[test]
    fn effective_dates_are_inclusive_and_rosters_replace_each_other() {
        let history = history("2020-01-12");
        assert_eq!(history.at("2020-01-01").unwrap(), ["B", "A"]);
        assert_eq!(history.at("2020-01-05").unwrap(), ["B", "A"]);
        assert_eq!(history.at("2020-01-06").unwrap(), ["A", "C"]);
        assert_eq!(history.at("2020-01-10").unwrap(), ["B", "C"]);
        assert_eq!(history.at(day("2020-01-12")).unwrap(), ["B", "C"]);
        assert_eq!(history.symbols(), ["B", "A", "C"]);
    }

    #[test]
    fn lookup_does_not_extend_coverage() {
        let history = history("2020-01-12");
        for date in ["2019-12-31", "2020-01-13"] {
            assert!(history.at(date).unwrap_err().message().contains("outside coverage"), "{date}");
        }
        assert_eq!(
            history.at("2020-01-13").unwrap_err().message(),
            "Membership date 2020-01-13 is outside coverage 2020-01-01 through 2020-01-12."
        );
    }

    /// Python also rejects None, integers, NaT and timestamps with a time or a
    /// time zone. Those values cannot be a `Date` or text here.
    #[test]
    fn lookup_requires_unambiguous_calendar_dates() {
        let history = history("2020-01-12");
        for date in ["2020-01-01T12:00:00", "01/02/2020", "2020-02-30", "NaT", "", "2020-01-01 "] {
            let error = history.at(date).unwrap_err();
            assert!(error.is_value() && error.message().contains("date"), "{date}: {error}");
        }
        assert_eq!(
            history.at("2020-02-30").unwrap_err().message(),
            "Membership lookup date must be a valid calendar date."
        );
        assert_eq!(
            history.at("01/02/2020").unwrap_err().message(),
            "Membership lookup date must be a date in YYYY-MM-DD format."
        );
    }

    #[test]
    fn csv_rejects_invalid_or_ambiguous_snapshots() {
        let cases = [
            "",
            "date,tickers\n",
            "date,symbol\n2020-01-01,A\n",
            "date,tickers,extra\n2020-01-01,A,x\n",
            "date,tickers\n2020-01-01,\"A,A\"\n",
            "date,tickers\n2020-01-01,\"A, A\"\n",
            "date,tickers\n2020-01-01,\"A,\"\n",
            "date,tickers\n2020-01-01,\n",
            "date,tickers\n2020-01-01,\"A,B\"C\n",
            "date,tickers\n2020-01-01\n",
            "date,tickers\n2020-01-01,A\n2020-01-02\n",
            "date,tickers\n2020-01-01,A\n2020-01-01,B\n",
            "date,tickers\n2020-01-02,A\n2020-01-01,B\n",
            "date,tickers\n2020-01-01,BAD SYMBOL\n",
            "date,tickers\n2020-01-01,=A\n",
            "date,tickers\n2020-02-30,A\n",
            "date,tickers\n01/02/2020,A\n",
            "date,tickers\ngarbage,2020-01-01,A\n",
        ];
        for csv in cases {
            assert!(load_membership_csv(csv.as_bytes(), "2020-01-31", None).unwrap_err().is_value(), "{csv:?}");
        }
        assert_eq!(
            load_membership_csv(b"date,tickers\n2020-01-01,\xff\n", "2020-01-31", None).unwrap_err().message(),
            "Membership CSV must contain date,tickers columns and full rosters."
        );
    }

    /// Python also rejects a coverage end of None, which cannot be text here.
    #[test]
    fn coverage_end_is_required_and_cannot_precede_last_snapshot() {
        for coverage_end in ["2019-12-31", "2020-01-09", "2020-02-30", ""] {
            assert!(load_membership_csv(CSV.as_bytes(), coverage_end, None).is_err(), "{coverage_end:?}");
        }
        assert_eq!(
            load_membership_csv(CSV.as_bytes(), "", None).unwrap_err().message(),
            "Membership coverage end must be a date in YYYY-MM-DD format."
        );
    }

    #[test]
    fn custom_keys_are_preserved_without_yahoo_aliases() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("members.csv");
        std::fs::write(&path, "date,tickers\n2020-01-01,\"BRK.B,BRK-B,NA,OLD_DELL,security:123\"\n").unwrap();
        let history = load_membership_csv_path(&path, "2020-01-02", Some("Verified security IDs")).unwrap();
        assert_eq!(history.at("2020-01-01").unwrap(), ["BRK.B", "BRK-B", "NA", "OLD_DELL", "security:123"]);
        assert!(history.identity_exclusions("1990-01-01", Some("2030-01-01")).unwrap().is_empty());
        assert_eq!(history.source(), "Verified security IDs");
        let metadata = history.metadata();
        assert_eq!(metadata.str("coverage_end"), "2020-01-02");
        assert_eq!(metadata.int("symbol_count"), 5);
        assert!(history.limitations()[0].contains("not independently verified"));
        assert!(Json::Dict(metadata).dumps(None).contains("\"symbol_count\": 5"));
    }

    #[test]
    fn unreadable_files_report_the_python_os_error() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing.csv");
        let error = load_membership_csv_path(&path, "2020-01-02", None).unwrap_err();
        assert!(error.is_io());
        assert_eq!(error.message(), format!("[Errno 2] No such file or directory: '{}'", path.display()));
    }

    /// Python reads bytes and text uploads alike. Here every upload is bytes,
    /// so a byte-order mark and CRLF line ends must give the same snapshots.
    #[test]
    fn upload_bytes_and_text_have_the_same_snapshots() {
        let plain = history("2020-01-12");
        let marked = format!("\u{feff}{}", CSV.replace('\n', "\r\n"));
        let other = load_membership_csv(marked.as_bytes(), "2020-01-12", None).unwrap();
        assert_eq!(plain.snapshots(), other.snapshots());
    }

    /// Python also rejects a text roster instead of a list, which cannot occur
    /// here. Rust ownership copies the rosters.
    #[test]
    fn constructor_validates_direct_input() {
        let snapshots = vec![(day("2020-01-01"), strings(&["A", "B"]))];
        let history = MembershipHistory::new(snapshots.clone(), "2020-01-02", DEFAULT_SOURCE, vec![], Dict::new());
        assert_eq!(history.unwrap().at("2020-01-01").unwrap(), ["A", "B"]);
        let empty_roster = vec![(day("2020-01-01"), Vec::new())];
        let error = MembershipHistory::new(empty_roster, "2020-01-02", DEFAULT_SOURCE, vec![], Dict::new());
        assert!(error.unwrap_err().message().contains("nonempty roster"));
        let error = MembershipHistory::new(Vec::<(Date, _)>::new(), "2020-01-02", DEFAULT_SOURCE, vec![], Dict::new());
        assert!(error.unwrap_err().message().contains("at least one"));
        let error = MembershipHistory::new(snapshots, "2020-01-02", " ", vec![], Dict::new());
        assert_eq!(error.unwrap_err().message(), "Membership source must be a nonempty description.");
    }

    #[test]
    fn identity_exclusions_use_inclusive_query_and_exclusive_rule_end() {
        let rules = vec![
            Json::Dict(
                dict! {"symbol" => "A", "start" => Json::Null, "end" => "2019-01-01", "reason" => "Reused ticker"},
            ),
            Json::Dict(
                dict! {"symbol" => "B", "start" => "2019-06-01", "end" => Json::Null, "reason" => "Unresolved identity"},
            ),
        ];
        let history = MembershipHistory::new(
            vec![(day("2020-01-01"), strings(&["A", "B"]))],
            "2020-01-12",
            DEFAULT_SOURCE,
            vec![],
            dict! {"identity_exclusions" => rules},
        )
        .unwrap();
        let exclusions = |start: &str, through: Option<&str>| {
            let found = history.identity_exclusions(start, through).unwrap();
            found.into_iter().map(|(symbol, reason)| format!("{symbol}: {reason}")).collect::<Vec<_>>()
        };
        assert_eq!(exclusions("2018-01-01", None), ["A: Reused ticker"]);
        assert!(exclusions("2019-01-01", None).is_empty());
        assert_eq!(exclusions("2019-05-31", Some("2019-06-01")), ["B: Unresolved identity"]);
        assert_eq!(exclusions("2018-01-01", Some("2020-01-01")), ["A: Reused ticker", "B: Unresolved identity"]);
        let error = history.identity_exclusions("2020-01-02", Some("2020-01-01")).unwrap_err();
        assert!(error.message().contains("on or after"));
    }

    #[test]
    fn bundle_is_pinned_and_keeps_historical_labels() {
        let history = bundled_membership();
        assert_eq!(history.coverage_start(), day("2010-01-01"));
        assert_eq!(history.coverage_end(), day("2026-09-30"));
        assert_eq!(history.snapshots().len(), 119);
        assert_eq!(history.symbols().len(), 255);
        let has = |date: &str, symbol: &str| history.at(date).unwrap().iter().any(|member| member == symbol);
        assert!(has("2010-01-01", "PCLN"));
        assert!(!has("2010-01-01", "BKNG"));
        assert!(!has("2014-04-02", "GOOGL"));
        assert!(has("2014-04-03", "GOOGL"));
        let excluded = |start: &str, through: Option<&str>| history.identity_exclusions(start, through).unwrap();
        assert!(excluded("2014-04-02", None).contains_key("GOOG"));
        assert!(!excluded("2014-04-03", None).contains_key("GOOG"));
        assert!(excluded("2014-04-02", Some("2014-04-03")).contains_key("GOOG"));
        let limitations = history.limitations().join(" ");
        for symbol in ["CMCSK", "LVNTA", "LILA"] {
            assert!(limitations.contains(symbol), "{symbol}");
        }
        let digest: String =
            Sha256::digest(SNAPSHOTS_CSV.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect();
        let metadata = history.metadata();
        assert_eq!(digest, metadata.str("snapshot_sha256"));
        assert!(N100TICKERS_LICENSE.contains("Copyright (c) 2021 - 2025 Jeff McCarrell"));
        assert_eq!(metadata.str("source_commit"), "cf7c89419941316ff8523726ccba136125f2c11c");
    }

    #[test]
    fn bundle_blocks_known_reused_tickers_in_old_price_history() {
        let exclusions = bundled_membership().identity_exclusions("2010-01-01", Some("2016-01-01")).unwrap();
        for symbol in ["DELL", "ALTR", "LIFE", "SNDK", "FOX", "FOXA", "GOLD", "NWSA", "BBBY", "LMCA", "LMCK"] {
            assert!(exclusions.contains_key(symbol), "{symbol}");
        }
    }

    #[test]
    fn current_identity_needs_price_history_from_the_safe_start() {
        let history = bundled_membership();
        let cases = [
            ("GOOG", "2014-04-03"),
            ("SNDK", "2025-02-24"),
            ("DELL", "2018-12-28"),
            ("FOX", "2019-03-19"),
            ("FOXA", "2019-03-19"),
            ("NWSA", "2013-07-01"),
        ];
        for (symbol, start) in cases {
            let previous = day(start).pred_opt().unwrap();
            let end = day("2026-09-30");
            assert!(!history.identity_exclusions(day(start), Some(end)).unwrap().contains_key(symbol), "{symbol}");
            assert!(history.identity_exclusions(previous, Some(end)).unwrap().contains_key(symbol), "{symbol}");
        }
    }

    #[test]
    fn provenance_json_keeps_key_order_and_number_types() {
        let Json::Dict(value) = parse_json(r#"{"b": 1, "a": [1.0, null, true], "c": {"z": "x", "y": -2}}"#).unwrap()
        else {
            panic!("an object")
        };
        assert_eq!(value.keys().collect::<Vec<_>>(), ["b", "a", "c"]);
        assert_eq!(Json::Dict(value).dumps(None), r#"{"b": 1, "a": [1.0, null, true], "c": {"z": "x", "y": -2}}"#);
    }
}
