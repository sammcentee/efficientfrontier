//! Panics as error messages.
//!
//! Streamlit shows an exception in place of the element that raised it, and the
//! server keeps running. The app does the same for panics: in a background
//! build, in one section of a page, or in an optional network lookup.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Run `work`; a panic becomes `Err` with its message.
pub fn guard<T>(work: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(work)).map_err(|payload| message(payload.as_ref()))
}

/// The text of a panic payload.
pub fn message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "An unexpected error occurred.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panics_become_messages() {
        assert_eq!(guard(|| 2 + 2), Ok(4));
        assert_eq!(guard(|| -> i32 { panic!("lost {}", 7) }), Err("lost 7".to_string()));
        assert_eq!(guard(|| -> i32 { todo!("story") }), Err("not yet implemented: story".to_string()));
    }
}
