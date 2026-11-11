# Contributing

Project code and documentation use the [MIT License](LICENSE). Contribute only material that you have permission to share under those terms. Preserve third-party notices.

## Development setup

Install Rust 1.90 or newer with [rustup](https://rustup.rs). The [README setup](README.md#build-from-source) gives the C compiler, the linker, and the Linux window libraries for each system. `Cargo.toml` sets the minimum Rust version. Start from a branch of `main`:

```bash
git switch main
git pull --ff-only
git switch -c your-change
cargo build
```

The library crate `efficient_frontier` is in `src/`. The `efficient-frontier` CLI starts in `src/main.rs` and uses `src/cli.rs`. The desktop app `portfolio-lab` is a second workspace member in `desktop/`. Tauri shows its pages in a native window. The routes in `desktop/src/routes.rs` make the pages, and `desktop/src/bridge.rs` gives them the window's requests without a network port. `assets/` contains the front-end files, the exact demo prices, and the bundled Nasdaq-100 membership data. `assets/css/tokens.css` and `story.css` have the light palette, which exported reports share. `assets/css/app.css` adds the app's dark palette, and `desktop/src/theme.rs` describes how a page chooses its theme. `desktop/icons/icon.svg` is the source of the app icons.

To start the desktop app from source:

```bash
cargo run --release -p portfolio-lab
```

## Checks

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo clippy -p portfolio-lab --all-targets -- -D warnings
cargo test -p portfolio-lab
cargo run --release --bin efficient-frontier -- --target-volatility .20 --max-weight .20 --output results/check
```

At the root, `cargo` builds and tests only the library and the CLI, so these steps do not need the window libraries. The `-p portfolio-lab` steps build the desktop app.

CI also checks the whole workspace with Rust 1.90. The Windows target in `.cargo/config.toml` links the C runtime into both programs. CI inspects their DLL dependencies to verify that they need no separate Visual C++ runtime installation.

The final command runs the app's model with synthetic data and exports an offline report. The CLI without `--target-volatility` runs the separate legacy analysis.

The default tests do not use the network. Fakes give recorded or synthetic replies in place of the downloads. An example is `with_fake_network` in `src/net.rs`. By default, `cargo test` ignores the live network tests. To run them, use this command:

```bash
cargo test --test network_live -- --ignored
```

An end-to-end test drives the app's window through WebDriver. It needs a release build and the system's WebDriver. On Linux, [`tauri-driver`](https://crates.io/crates/tauri-driver) starts `WebKitWebDriver`, which Debian and Ubuntu supply in `webkit2gtk-driver`, and Arch in `webkitgtk-6.0`. On Windows, the test starts the app and connects Microsoft Edge Driver to WebView2's remote debugging port. Put `msedgedriver.exe` for the installed WebView2 version on `PATH`. On Linux, run the test on a virtual screen with `xvfb-run` and `GDK_BACKEND=x11`, so that no window opens on the desktop, also under WSLg. On Linux the test also saves files through the Save dialog with `xdotool`.

```bash
cargo build --release -p portfolio-lab
cargo install tauri-driver --locked
env -u WAYLAND_DISPLAY GDK_BACKEND=x11 xvfb-run -a cargo test -p portfolio-lab --test window -- --ignored
```

On Windows, WebView2 155 ignored the `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` variable and the same policy under `HKCU`. Only a machine policy opened the port. In an administrator terminal, add the policy before the test and delete it after the test. While the policy exists, other programs on the computer can control the app through the port.

```powershell
$policy = 'HKLM\SOFTWARE\Policies\Microsoft\Edge\WebView2\AdditionalBrowserArguments'
reg add $policy /v portfolio-lab.exe /t REG_SZ /d '--remote-debugging-port=9222' /f
cargo build --release -p portfolio-lab
cargo test -p portfolio-lab --test window -- --ignored
reg delete $policy /v portfolio-lab.exe /f
```

For calculation changes, add a case with a known mathematical answer or a regression that demonstrates the error. Use fakes in place of network calls in the default tests. Verify that later prices cannot change earlier allocations. Document financial assumptions in the [method reference](docs/BACKTESTING.md).

## Parity with the Python release

`tests/fixtures/golden/` contains golden results from the Python release at commit `2e5f21c`. The `generate_*.py` scripts in that folder made these results. The Rust tests compare their results with the golden files. Weights and metrics usually agree to 1e-10. A few almost flat optima differ by approximately 1e-6, with equal objectives. The [fixture notes](tests/fixtures/golden/README.md) tell how to make a set again.

The Rust version converts each number in CSV text to the nearest floating-point value. The default parser of pandas can differ by one unit in the last place. The effect on results is approximately 1e-16.

## Pull requests

Keep changes focused and open pull requests against `main`. Describe the problem, resulting behavior, and checks. CI runs the format check, clippy, and the tests on `ubuntu-latest`. It also exports synthetic reports for the app model and the retained legacy tools. A second job builds and tests the desktop app on Ubuntu and Windows, runs the window test, and keeps the programs as artifacts.

Use synthetic data in examples and bug reports. Keep credentials, personal holdings, downloaded datasets, and generated reports out of commits. Use ignored `data/` or `private/` folders for local inputs.
