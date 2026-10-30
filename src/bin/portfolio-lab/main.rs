//! Portfolio Lab: the local web app of the efficient frontier study.
//!
//! The server listens on 127.0.0.1 only. It renders pages on the server, with
//! plotly.js charts and KaTeX equations from files inside this binary, so the
//! app works offline. Each browser keeps its own study in memory.

mod assets;
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

use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;

use clap::Parser;

/// Efficient Frontier Analysis for US Equities, in your browser.
#[derive(Parser)]
#[command(name = "portfolio-lab", version)]
struct Args {
    /// The local port. 0 chooses a free port.
    #[arg(long, default_value_t = 8501)]
    port: u16,
    /// Do not open a browser.
    #[arg(long)]
    headless: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", args.port)).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("Portfolio Lab could not use port {}: {error}", args.port);
            eprintln!("Close the other program on that port, or start with --port <n>.");
            return ExitCode::FAILURE;
        }
    };
    let url = match listener.local_addr() {
        Ok(address) => format!("http://{address}"),
        Err(error) => {
            eprintln!("Portfolio Lab could not read its address: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("\n  Portfolio Lab is running. Press Ctrl+C to stop.\n\n  Local URL: {url}\n");
    if !args.headless {
        open_browser(&url);
    }
    let app = routes::router(Arc::new(state::AppState::live()));
    let served = axum::serve(listener, app).with_graceful_shutdown(async {
        // Without a signal handler the server runs until the process ends.
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    });
    match served.await {
        Ok(()) => {
            println!("\n  Portfolio Lab stopped.");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Portfolio Lab stopped with an error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Open the default browser at `url`, as `streamlit run` without `--server.headless`.
fn open_browser(url: &str) {
    let candidates: &[(&str, &[&str])] = if cfg!(target_os = "windows") {
        &[("cmd", &["/C", "start", ""])]
    } else if cfg!(target_os = "macos") {
        &[("open", &[])]
    } else {
        &[("xdg-open", &[]), ("wslview", &[])]
    };
    let opened = candidates.iter().any(|(program, arguments)| {
        Command::new(program)
            .args(*arguments)
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok()
    });
    if !opened {
        println!("  Open the Local URL in your browser.\n");
    }
}
