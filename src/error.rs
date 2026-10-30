//! One error type for the whole crate.
//!
//! The three kinds keep the distinctions that the user interface and the CLI
//! need: `Value` is invalid input, `Runtime` is a solver or numerical failure,
//! and `Io` is a file or network failure. The message is the text that the user
//! sees, so it never contains the kind.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Invalid input or settings.
    Value(String),
    /// The optimizer or another numerical step failed.
    Runtime(String),
    /// A file, network, or provider failure.
    Io(String),
}

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Error::Value(message) | Error::Runtime(message) | Error::Io(message) => message,
        }
    }

    pub fn is_value(&self) -> bool {
        matches!(self, Error::Value(_))
    }

    pub fn is_runtime(&self) -> bool {
        matches!(self, Error::Runtime(_))
    }

    pub fn is_io(&self) -> bool {
        matches!(self, Error::Io(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `value_error!("text {}", x)` builds an [`Error::Value`].
#[macro_export]
macro_rules! value_error {
    ($($arg:tt)*) => { $crate::error::Error::Value(format!($($arg)*)) };
}

/// `runtime_error!("text {}", x)` builds an [`Error::Runtime`].
#[macro_export]
macro_rules! runtime_error {
    ($($arg:tt)*) => { $crate::error::Error::Runtime(format!($($arg)*)) };
}

/// `io_error!("text {}", x)` builds an [`Error::Io`].
#[macro_export]
macro_rules! io_error {
    ($($arg:tt)*) => { $crate::error::Error::Io(format!($($arg)*)) };
}
