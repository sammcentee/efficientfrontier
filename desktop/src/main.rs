//! Portfolio Lab: the desktop app of the efficient frontier study.
//!
//! The app shows its pages in a native window. The window's web view asks the
//! `app` protocol for each page, and the [`bridge`] answers inside this program,
//! so the app opens no network port. Pages are rendered here, with plotly.js
//! charts and KaTeX equations from files inside this binary, so the app works
//! offline.

// A release build on Windows opens no console window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod assets;
mod bridge;
mod build;
mod cache;
mod guard;
mod render;
mod routes;
mod session;
mod setup;
mod sources;
mod state;
#[cfg(test)]
mod tests;
mod theme;

use std::sync::Arc;

use axum::http::Response;
use tauri::webview::NewWindowResponse;
use tauri::window::Color;
use tauri::{AppHandle, Manager, Url, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

use crate::bridge::{Bridge, Download, Forwarded};
use crate::theme::Theme;

fn main() {
    let bridge = Arc::new(Bridge::new(routes::router(Arc::new(state::AppState::live()))));
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .register_asynchronous_uri_scheme_protocol(bridge::SCHEME, move |context, request, responder| {
            let bridge = bridge.clone();
            let app = context.app_handle().clone();
            tauri::async_runtime::spawn(async move {
                let response = match bridge.forward(request).await {
                    Forwarded::Page(response) => response,
                    Forwarded::Download(download) => save(&app, download).await,
                    Forwarded::Theme(theme) => {
                        style_window(&app, theme);
                        bridge::no_content()
                    }
                };
                responder.respond(response);
            });
        })
        .setup(|app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::CustomProtocol(bridge::start_url()))
                .title("Portfolio Lab")
                // The window starts as the pages do: dark, with no white flash before the first page.
                .theme(Some(window_theme(theme::DEFAULT)))
                .background_color(canvas(theme::DEFAULT))
                .inner_size(1280.0, 860.0)
                .min_inner_size(400.0, 500.0)
                .center()
                // The page's file fields take dropped files themselves, as in a browser.
                .disable_drag_drop_handler()
                .on_navigation(|url| bridge::is_app_url(url) || open_outside(url))
                .on_new_window(|url, _| {
                    open_outside(&url);
                    NewWindowResponse::Deny
                })
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Portfolio Lab could not start");
}

/// The window theme for a page theme.
fn window_theme(theme: Theme) -> tauri::Theme {
    match theme {
        Theme::Dark => tauri::Theme::Dark,
        Theme::Light => tauri::Theme::Light,
    }
}

/// The page background of a theme, as the window's color.
fn canvas(theme: Theme) -> Color {
    let (red, green, blue) = theme.canvas();
    Color(red, green, blue, 255)
}

/// Give the window the theme of its page: the title bar, and the color behind a loading page.
fn style_window(app: &AppHandle, theme: Theme) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.set_theme(Some(window_theme(theme)));
        let _ = window.set_background_color(Some(canvas(theme)));
    }
}

/// Open a web or mail link in the default browser instead of the app's window.
/// Returns false, so the window keeps its page.
fn open_outside(url: &Url) -> bool {
    if matches!(url.scheme(), "http" | "https" | "mailto") {
        let _ = tauri_plugin_opener::open_url(url.as_str(), None::<&str>);
    }
    false
}

/// Ask where to save a download, then write the file. An error shows a message.
/// The page gets "204 No Content" in every case, so it stays as it is.
async fn save(app: &AppHandle, download: Download) -> Response<Vec<u8>> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let mut dialog = app.dialog().file().set_file_name(&download.name);
    if let Some(window) = app.get_webview_window("main") {
        dialog = dialog.set_parent(&window);
    }
    if let Some((_, extension)) = download.name.rsplit_once('.') {
        dialog = dialog.add_filter(extension.to_uppercase(), &[extension]);
    }
    dialog.save_file(move |path| {
        let _ = sender.send(path);
    });
    let chosen = receiver.await.ok().flatten().and_then(|path| path.into_path().ok());
    if let Some(path) = chosen {
        let target = path.clone();
        let written = tokio::task::spawn_blocking(move || std::fs::write(&target, download.bytes)).await;
        let error = match written {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error.to_string()),
            Err(error) => Some(error.to_string()),
        };
        if let Some(error) = error {
            app.dialog()
                .message(format!("Portfolio Lab could not save {}.\n\n{error}", path.display()))
                .title("The file was not saved")
                .kind(MessageDialogKind::Error)
                .show(|_| {});
        }
    }
    bridge::no_content()
}
