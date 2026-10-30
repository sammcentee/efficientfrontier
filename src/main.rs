//! `efficient-frontier`: the command-line analysis and report export.

fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(efficient_frontier::cli::main() as u8)
}
