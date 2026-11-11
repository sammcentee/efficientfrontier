//! The page theme: dark at first, light by choice.
//!
//! The page keeps the choice in the web view's local storage, so the app has
//! no settings file. Every page starts as the dark theme, and a short script in
//! its head changes it to the stored theme before the page is drawn. The style
//! sheet `app.css` holds the dark palette. The browser script draws the charts
//! in the colors of the theme and tells the window about a change, so that its
//! title bar matches the page.
//!
//! Exported reports keep the light palette of `tokens.css`.

use crate::render::html::esc;

/// The two themes of the app's pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Dark,
    Light,
}

/// The theme of a page before the stored choice applies.
pub const DEFAULT: Theme = Theme::Dark;
/// The local storage key of the chosen theme. `app.js` writes it.
pub const STORAGE_KEY: &str = "pl-theme";
/// The response header that names the page's theme for the window.
pub const HEADER: &str = "x-pl-theme";

impl Theme {
    /// The value of `data-theme`, of local storage and of the theme form.
    pub fn key(self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
        }
    }

    pub fn parse(key: &str) -> Option<Self> {
        [Theme::Dark, Theme::Light].into_iter().find(|theme| theme.key() == key)
    }

    /// The page background (`--pl-canvas`), as red, green and blue.
    pub fn canvas(self) -> (u8, u8, u8) {
        match self {
            Theme::Dark => (0x0e, 0x0e, 0x10),
            Theme::Light => (0xf5, 0xf5, 0xf7),
        }
    }

    /// The page background as a CSS color.
    pub fn canvas_css(self) -> String {
        let (red, green, blue) = self.canvas();
        format!("#{red:02x}{green:02x}{blue:02x}")
    }
}

/// The attributes of `<html>`: every page starts as the default theme.
pub fn html_attributes() -> String {
    format!(r#"lang="en" data-theme="{}""#, DEFAULT.key())
}

/// The script at the start of `<head>`: it applies a stored light theme before
/// the page is drawn, so a page never shows the other theme first.
pub fn head_script() -> String {
    format!(
        r#"<script>try{{if(localStorage.getItem("{STORAGE_KEY}")==="{light}")document.documentElement.dataset.theme="{light}"}}catch(e){{}}</script>"#,
        light = Theme::Light.key()
    )
}

/// The canvas of both themes for a page without the app's style sheets.
pub fn canvas_style() -> String {
    format!(
        r#"<style>html{{background:{}}}html[data-theme="{}"]{{background:{}}}</style>"#,
        DEFAULT.canvas_css(),
        Theme::Light.key(),
        Theme::Light.canvas_css()
    )
}

const SUN: &str = r#"<svg class="pl-theme-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/></svg>"#;
const MOON: &str = r#"<svg class="pl-theme-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>"#;

/// The label of the theme button on a page of `theme`. `app.js` has the same text.
pub fn toggle_label(theme: Theme) -> &'static str {
    match theme {
        Theme::Dark => "Use the light theme",
        Theme::Light => "Use the dark theme",
    }
}

/// The theme button: the sun on a dark page, the moon on a light page.
pub fn toggle_button() -> String {
    let label = esc(toggle_label(DEFAULT));
    format!(
        r#"<button type="button" class="pl-icon-btn pl-theme-toggle" id="theme_toggle" data-theme-toggle aria-label="{label}" title="{label}">{SUN}{MOON}</button>"#
    )
}

#[cfg(test)]
mod tests {
    use efficient_frontier::style::TOKENS_CSS;

    use super::*;

    const APP_CSS: &str = include_str!("../../assets/css/app.css");
    const APP_JS: &str = include_str!("../../assets/js/app.js");

    #[test]
    fn themes_have_keys_and_the_default_is_dark() {
        assert_eq!(DEFAULT, Theme::Dark);
        assert_eq!(Theme::parse("dark"), Some(Theme::Dark));
        assert_eq!(Theme::parse("light"), Some(Theme::Light));
        assert_eq!(Theme::parse("Dark"), None);
        assert_eq!(Theme::parse(""), None);
        assert_eq!(html_attributes(), r#"lang="en" data-theme="dark""#);
    }

    #[test]
    fn the_canvas_colors_are_those_of_the_style_sheets() {
        assert!(TOKENS_CSS.contains(&format!(":root{{--pl-canvas:{};", Theme::Light.canvas_css())));
        let dark = APP_CSS.split(r#":root[data-theme="dark"]{"#).nth(1).expect("the dark palette");
        let dark = dark.split('}').next().unwrap();
        assert!(dark.contains("color-scheme:dark;"));
        assert!(dark.contains(&format!("--pl-canvas:{};", Theme::Dark.canvas_css())), "{dark}");
        assert_eq!(
            canvas_style(),
            r#"<style>html{background:#0e0e10}html[data-theme="light"]{background:#f5f5f7}</style>"#
        );
    }

    /// The custom properties that a style sheet defines in the block after `selector`.
    fn defined(css: &str, selector: &str) -> Vec<String> {
        let block = css.split(selector).nth(1).and_then(|rest| rest.split('}').next()).expect("a block");
        block.split(';').filter_map(|pair| pair.trim().split_once(':')).map(|(name, _)| name.to_string()).collect()
    }

    #[test]
    fn the_dark_palette_covers_every_color_of_the_light_palette() {
        let dark = defined(APP_CSS, r#":root[data-theme="dark"]{"#);
        let app_light = defined(APP_CSS, ":root{");
        let geometry =
            ["--pl-font", "--pl-display", "--pl-ease", "--pl-ease-out", "--pl-r-tile", "--pl-pad-tile", "--pl-gap"];
        let light = defined(TOKENS_CSS, ":root{").into_iter().chain(app_light);
        for name in light.filter(|name| name.starts_with("--") && !geometry.contains(&name.as_str())) {
            assert!(dark.contains(&name), "the dark palette has no {name}");
        }
    }

    #[test]
    fn the_page_script_uses_the_same_storage_key_and_labels() {
        assert!(APP_JS.contains(&format!(r#"const THEME_KEY = "{STORAGE_KEY}";"#)));
        assert!(head_script().contains(r#"localStorage.getItem("pl-theme")==="light""#));
        for theme in [Theme::Dark, Theme::Light] {
            assert!(APP_JS.contains(toggle_label(theme)), "{}", toggle_label(theme));
        }
        let button = toggle_button();
        assert!(button.contains(r#"aria-label="Use the light theme""#) && button.contains("data-theme-toggle"));
    }
}
