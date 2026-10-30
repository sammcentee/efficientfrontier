//! Text formats that match the earlier Python release.
//!
//! Reports, CSV files and JSON metadata keep the number formats of the Python
//! version, so exported files stay comparable. This module implements Python's
//! `repr(float)`, the format-specification subset that the project uses, HTML
//! escaping, and `json.dumps` with an ordered dictionary type.

use std::fmt::Write as _;

/// Python `repr(x)` (also `str(x)`) for a float: the shortest text that reads
/// back to the same value, in fixed notation for exponents from -4 to 15.
pub fn repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (negative, digits, exponent) = shortest_digits(x);
    let mut text = String::new();
    if negative {
        text.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let integer = (exponent + 1) as usize;
            if digits.len() <= integer {
                text.push_str(&digits);
                text.push_str(&"0".repeat(integer - digits.len()));
                text.push_str(".0");
            } else {
                text.push_str(&digits[..integer]);
                text.push('.');
                text.push_str(&digits[integer..]);
            }
        } else {
            text.push_str("0.");
            text.push_str(&"0".repeat((-exponent - 1) as usize));
            text.push_str(&digits);
        }
    } else {
        text.push_str(&digits[..1]);
        if digits.len() > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        text.push_str(&exponent_text(exponent));
    }
    text
}

/// Sign, significant digits, and the decimal exponent of the first digit.
fn shortest_digits(x: f64) -> (bool, String, i32) {
    let text = format!("{:e}", x.abs());
    let (mantissa, exponent) = text.split_once('e').expect("LowerExp output has an exponent");
    (x.is_sign_negative(), mantissa.replace('.', ""), exponent.parse().expect("integer exponent"))
}

/// Python's exponent suffix: a sign and at least two digits.
fn exponent_text(exponent: i32) -> String {
    format!("e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs())
}

#[derive(Default)]
struct Spec {
    sign: Option<char>,
    zero: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    kind: Option<char>,
}

fn parse_spec(spec: &str) -> Spec {
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let mut parsed = Spec::default();
    if i < chars.len() && matches!(chars[i], '+' | '-' | ' ') {
        parsed.sign = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '0' {
        parsed.zero = true;
        i += 1;
    }
    let start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i > start {
        parsed.width = chars[start..i].iter().collect::<String>().parse().unwrap_or(0);
    }
    if i < chars.len() && matches!(chars[i], ',' | '_') {
        parsed.grouping = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        parsed.precision = Some(chars[start..i].iter().collect::<String>().parse().unwrap_or(0));
    }
    if i < chars.len() {
        parsed.kind = Some(chars[i]);
        i += 1;
    }
    assert!(i == chars.len(), "unsupported format specification: {spec:?}");
    parsed
}

fn group(integer: &str, separator: char) -> String {
    let mut grouped = String::new();
    for (position, digit) in integer.chars().enumerate() {
        if position > 0 && (integer.len() - position).is_multiple_of(3) {
            grouped.push(separator);
        }
        grouped.push(digit);
    }
    grouped
}

fn finish(negative: bool, body: String, spec: &Spec) -> String {
    let sign = if negative {
        "-"
    } else {
        match spec.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    let mut body = body;
    if let Some(separator) = spec.grouping {
        let end = body.find(|c: char| !c.is_ascii_digit()).unwrap_or(body.len());
        body = group(&body[..end], separator) + &body[end..];
    }
    let length = sign.len() + body.chars().count();
    if spec.width > length {
        let padding = spec.width - length;
        if spec.zero {
            return format!("{sign}{}{body}", "0".repeat(padding));
        }
        return format!("{}{sign}{body}", " ".repeat(padding));
    }
    format!("{sign}{body}")
}

/// Python `'g'` formatting of a finite, nonnegative value.
fn general(x: f64, precision: usize) -> String {
    let precision = precision.max(1);
    if x == 0.0 {
        return "0".into();
    }
    let scientific = format!("{:.*e}", precision - 1, x);
    let (_, exponent) = scientific.split_once('e').expect("LowerExp output has an exponent");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    if -4 <= exponent && exponent < precision as i32 {
        let decimals = (precision as i32 - 1 - exponent) as usize;
        strip_zeros(&format!("{:.*}", decimals, x))
    } else {
        let (mantissa, _) = scientific.split_once('e').expect("LowerExp output has an exponent");
        strip_zeros(mantissa) + &exponent_text(exponent)
    }
}

fn strip_zeros(text: &str) -> String {
    if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text.to_string() }
}

/// Python `format(x, spec)` for a float.
///
/// The supported specification is `[sign][0][width][,|_][.precision][type]`,
/// where the type is `f`, `%`, `e`, `g`, or absent. An empty specification
/// gives `repr(x)`.
pub fn format(x: f64, spec: &str) -> String {
    let parsed = parse_spec(spec);
    if parsed.kind.is_none() && parsed.precision.is_none() {
        let text = repr(x);
        let (negative, body) = match text.strip_prefix('-') {
            Some(rest) => (true, rest.to_string()),
            None => (false, text),
        };
        return finish(negative, body, &parsed);
    }
    let kind = parsed.kind.unwrap_or('g');
    let negative = x.is_sign_negative() && !x.is_nan();
    let magnitude = x.abs();
    let suffix = if kind == '%' { "%" } else { "" };
    if !magnitude.is_finite() {
        let word = if magnitude.is_nan() { "nan" } else { "inf" };
        return finish(negative, format!("{word}{suffix}"), &parsed);
    }
    let body = match kind {
        'f' | 'F' => format!("{:.*}", parsed.precision.unwrap_or(6), magnitude),
        '%' => format!("{:.*}%", parsed.precision.unwrap_or(6), magnitude * 100.0),
        'e' => {
            let text = format!("{:.*e}", parsed.precision.unwrap_or(6), magnitude);
            let (mantissa, exponent) = text.split_once('e').expect("LowerExp output has an exponent");
            mantissa.to_string() + &exponent_text(exponent.parse().expect("integer exponent"))
        }
        'g' => general(magnitude, parsed.precision.unwrap_or(6)),
        other => panic!("unsupported float format type {other:?} in {spec:?}"),
    };
    finish(negative, body, &parsed)
}

/// Python `format(n, spec)` for an integer: `[sign][0][width][,|_][d]`.
pub fn format_int(n: i64, spec: &str) -> String {
    let parsed = parse_spec(spec);
    assert!(
        matches!(parsed.kind, None | Some('d')) && parsed.precision.is_none(),
        "unsupported integer format specification: {spec:?}"
    );
    finish(n < 0, n.unsigned_abs().to_string(), &parsed)
}

/// `f"{x:.2%}"`, the most common format in the reports.
pub fn pct(x: f64, decimals: usize) -> String {
    format(x, &format!(".{decimals}%"))
}

/// Python `f"{x:g}"`.
pub fn g(x: f64) -> String {
    format(x, "g")
}

/// Python `html.escape(text, quote=True)`.
pub fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// A JSON-like value with Python's distinctions: integers and floats stay
/// separate, and dictionaries keep insertion order.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Json>),
    Dict(Dict),
}

/// An ordered dictionary with Python `dict` semantics: an update keeps the
/// position of an existing key and appends a new key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dict(pub Vec<(String, Json)>);

impl Dict {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.0.iter().find(|(name, _)| name == key).map(|(_, value)| value)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Json> {
        self.0.iter_mut().find(|(name, _)| name == key).map(|(_, value)| value)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// `d[key] = value`.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Json>) {
        let key = key.into();
        let value = value.into();
        match self.get_mut(&key) {
            Some(existing) => *existing = value,
            None => self.0.push((key, value)),
        }
    }

    /// `d.update(other)`.
    pub fn update(&mut self, other: &Dict) {
        for (key, value) in &other.0 {
            self.insert(key.clone(), value.clone());
        }
    }

    /// `d.pop(key, None)`.
    pub fn remove(&mut self, key: &str) -> Option<Json> {
        let position = self.0.iter().position(|(name, _)| name == key)?;
        Some(self.0.remove(position).1)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(key, _)| key.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Json)> {
        self.0.iter().map(|(key, value)| (key.as_str(), value))
    }

    /// A number setting. Panics when the key is missing or not numeric,
    /// because settings are internal invariants.
    pub fn f64(&self, key: &str) -> f64 {
        self.get(key).and_then(Json::as_f64).unwrap_or_else(|| panic!("numeric setting {key:?} is missing"))
    }

    /// A text setting. Panics when the key is missing or not text.
    pub fn str(&self, key: &str) -> &str {
        self.get(key).and_then(Json::as_str).unwrap_or_else(|| panic!("text setting {key:?} is missing"))
    }

    /// An integer setting. Panics when the key is missing or not an integer.
    pub fn int(&self, key: &str) -> i64 {
        match self.get(key) {
            Some(Json::Int(value)) => *value,
            _ => panic!("integer setting {key:?} is missing"),
        }
    }

    /// A boolean setting. Panics when the key is missing or not a boolean.
    pub fn bool(&self, key: &str) -> bool {
        match self.get(key) {
            Some(Json::Bool(value)) => *value,
            _ => panic!("boolean setting {key:?} is missing"),
        }
    }
}

impl Json {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Float(value) => Some(*value),
            Json::Int(value) => Some(*value as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&Dict> {
        match self {
            Json::Dict(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Json]> {
        match self {
            Json::List(value) => Some(value),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Python `str(value)`.
    pub fn py_str(&self) -> String {
        match self {
            Json::Str(value) => value.clone(),
            other => other.py_repr(),
        }
    }

    /// Python `repr(value)`.
    pub fn py_repr(&self) -> String {
        match self {
            Json::Null => "None".into(),
            Json::Bool(value) => if *value { "True" } else { "False" }.into(),
            Json::Int(value) => value.to_string(),
            Json::Float(value) => repr(*value),
            Json::Str(value) => python_string_repr(value),
            Json::List(values) => {
                format!("[{}]", values.iter().map(Json::py_repr).collect::<Vec<_>>().join(", "))
            }
            Json::Dict(values) => format!(
                "{{{}}}",
                values
                    .iter()
                    .map(|(key, value)| format!("{}: {}", python_string_repr(key), value.py_repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// Python `json.dumps(value, indent=indent, ensure_ascii=False)`.
    pub fn dumps(&self, indent: Option<usize>) -> String {
        let mut text = String::new();
        write_json(self, indent, 0, &mut text);
        text
    }
}

fn python_string_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut escaped = String::new();
    escaped.push(quote);
    for character in text.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c == quote => {
                escaped.push('\\');
                escaped.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(escaped, "\\x{:02x}", c as u32);
            }
            c => escaped.push(c),
        }
    }
    escaped.push(quote);
    escaped
}

fn json_string(text: &str, output: &mut String) {
    output.push('"');
    for character in text.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(output, "\\u{:04x}", c as u32);
            }
            c => output.push(c),
        }
    }
    output.push('"');
}

fn json_float(value: f64) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else {
        repr(value)
    }
}

fn write_json(value: &Json, indent: Option<usize>, level: usize, output: &mut String) {
    match value {
        Json::Null => output.push_str("null"),
        Json::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Json::Int(value) => output.push_str(&value.to_string()),
        Json::Float(value) => output.push_str(&json_float(*value)),
        Json::Str(value) => json_string(value, output),
        Json::List(values) if values.is_empty() => output.push_str("[]"),
        Json::Dict(values) if values.is_empty() => output.push_str("{}"),
        Json::List(values) => {
            output.push('[');
            for (position, item) in values.iter().enumerate() {
                separator(position, indent, level + 1, output);
                write_json(item, indent, level + 1, output);
            }
            closing(indent, level, output);
            output.push(']');
        }
        Json::Dict(values) => {
            output.push('{');
            for (position, (key, item)) in values.0.iter().enumerate() {
                separator(position, indent, level + 1, output);
                json_string(key, output);
                output.push_str(": ");
                write_json(item, indent, level + 1, output);
            }
            closing(indent, level, output);
            output.push('}');
        }
    }
}

fn separator(position: usize, indent: Option<usize>, level: usize, output: &mut String) {
    match indent {
        Some(width) => {
            if position > 0 {
                output.push(',');
            }
            output.push('\n');
            output.push_str(&" ".repeat(width * level));
        }
        None => {
            if position > 0 {
                output.push_str(", ");
            }
        }
    }
}

fn closing(indent: Option<usize>, level: usize, output: &mut String) {
    if let Some(width) = indent {
        output.push('\n');
        output.push_str(&" ".repeat(width * level));
    }
}

impl From<f64> for Json {
    fn from(value: f64) -> Self {
        Json::Float(value)
    }
}

impl From<i64> for Json {
    fn from(value: i64) -> Self {
        Json::Int(value)
    }
}

impl From<i32> for Json {
    fn from(value: i32) -> Self {
        Json::Int(value as i64)
    }
}

impl From<usize> for Json {
    fn from(value: usize) -> Self {
        Json::Int(value as i64)
    }
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Json::Bool(value)
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Json::Str(value.to_string())
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Json::Str(value)
    }
}

impl From<&String> for Json {
    fn from(value: &String) -> Self {
        Json::Str(value.clone())
    }
}

impl From<Dict> for Json {
    fn from(value: Dict) -> Self {
        Json::Dict(value)
    }
}

impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(values: Vec<T>) -> Self {
        Json::List(values.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(value: Option<T>) -> Self {
        value.map_or(Json::Null, Into::into)
    }
}

/// Build a [`Dict`] in insertion order: `dict! { "a" => 1.0, "b" => "text" }`.
#[macro_export]
macro_rules! dict {
    () => { $crate::pyfmt::Dict::new() };
    ($($key:expr => $value:expr),+ $(,)?) => {{
        let mut dict = $crate::pyfmt::Dict::new();
        $(dict.insert($key, $value);)+
        dict
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.1, "0.1"),
            (1e-7, "1e-07"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (100.0, "100.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.1 + 0.2, "0.30000000000000004"),
            (-2.5, "-2.5"),
            (1.5e-7, "1.5e-07"),
            (123456789.125, "123456789.125"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (f64::NAN, "nan"),
        ];
        for (value, expected) in cases {
            assert_eq!(repr(value), expected, "{value:e}");
        }
    }

    #[test]
    fn format_specs_match_python() {
        let cases = [
            (0.123456, ".2%", "12.35%"),
            (0.125, ".0%", "12%"),
            (-0.001, ".2f", "-0.00"),
            (252.0, "g", "252"),
            (0.0001, "g", "0.0001"),
            (1e-5, "g", "1e-05"),
            (1234567.0, "g", "1.23457e+06"),
            (0.5, "+.2f", "+0.50"),
            (1234.5, ",.0f", "1,234"),
            (1234567.891, ",.2f", "1,234,567.89"),
            (-1234.0, "+,.2f", "-1,234.00"),
            (0.000123, "+.2e", "+1.23e-04"),
            (12.5, ".4g", "12.5"),
            (0.02, ".1%", "2.0%"),
            (f64::NAN, ".2%", "nan%"),
            (0.1, "", "0.1"),
            (10.0, "g", "10"),
            (0.15, "+g", "+0.15"),
            (-0.0, "g", "-0"),
        ];
        for (value, spec, expected) in cases {
            assert_eq!(format(value, spec), expected, "{value} with {spec}");
        }
        assert_eq!(format_int(7, "02d"), "07");
        assert_eq!(format_int(1234567, ","), "1,234,567");
        assert_eq!(format_int(-1234, ","), "-1,234");
    }

    #[test]
    fn json_dumps_matches_python() {
        let value = Json::Dict(dict! {
            "a" => 1.0, "b" => vec![Json::Int(1), Json::Float(f64::NAN)], "c" => Dict::new(),
            "d" => "é \"q\"\n", "e" => Json::Null, "f" => Vec::<Json>::new(),
        });
        assert_eq!(
            value.dumps(Some(2)),
            "{\n  \"a\": 1.0,\n  \"b\": [\n    1,\n    NaN\n  ],\n  \"c\": {},\n  \"d\": \"é \\\"q\\\"\\n\",\n  \"e\": null,\n  \"f\": []\n}"
        );
        assert_eq!(
            value.dumps(None),
            "{\"a\": 1.0, \"b\": [1, NaN], \"c\": {}, \"d\": \"é \\\"q\\\"\\n\", \"e\": null, \"f\": []}"
        );
        assert_eq!(Json::from(vec!["x", "y'"]).py_repr(), "['x', \"y'\"]");
    }

    #[test]
    fn dict_update_keeps_existing_positions() {
        let mut values = dict! { "a" => 1, "b" => 2 };
        values.update(&dict! { "b" => 3, "c" => 4 });
        assert_eq!(values.keys().collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!(values.int("b"), 3);
    }

    #[test]
    fn html_escape_matches_python() {
        assert_eq!(escape("<a href=\"x\">'&'</a>"), "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;");
    }
}
