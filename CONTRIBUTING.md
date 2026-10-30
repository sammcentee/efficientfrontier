# Contributing

Project code and documentation use the [MIT License](LICENSE). Contribute only material that you have permission to share under those terms. Preserve third-party notices.

## Development setup

Install the Rust toolchain with [rustup](https://rustup.rs). The [README setup](README.md#setup) gives the C compiler and linker for each system. `Cargo.toml` sets the minimum Rust version. Start from a branch of `main`:

```bash
git switch main
git pull --ff-only
git switch -c your-change
cargo build
```

The library crate `efficient_frontier` is in `src/`. The `efficient-frontier` CLI starts in `src/main.rs` and uses `src/cli.rs`. The local web app `portfolio-lab` is in `src/bin/portfolio-lab/`. `assets/` contains the front-end files, the exact demo prices, and the bundled Nasdaq-100 membership data.

To start the app without an automatic browser launch:

```bash
bash run.sh --headless
```

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo run --release --bin efficient-frontier -- --target-volatility .20 --max-weight .20 --output results/check
```

The final command runs the app's model with synthetic data and exports an offline report. The CLI without `--target-volatility` runs the separate legacy analysis.

The default tests do not use the network. Fakes give recorded or synthetic replies in place of the downloads. An example is `with_fake_network` in `src/net.rs`. By default, `cargo test` ignores the live network tests. To run them, use this command:

```bash
cargo test --test network_live -- --ignored
```

For calculation changes, add a case with a known mathematical answer or a regression that demonstrates the error. Use fakes in place of network calls in the default tests. Verify that later prices cannot change earlier allocations. Document financial assumptions in the [method reference](docs/BACKTESTING.md).

## Parity with the Python release

`tests/fixtures/golden/` contains golden results from the Python release at commit `6c6df6e`. The `generate_*.py` scripts in that folder made these results. The Rust tests compare their results with the golden files. Weights and metrics usually agree to 1e-10. A few almost flat optima differ by approximately 1e-6, with equal objectives. The [fixture notes](tests/fixtures/golden/README.md) tell how to make a set again.

The Rust version converts each number in CSV text to the nearest floating-point value. The default parser of pandas can differ by one unit in the last place. The effect on results is approximately 1e-16.

## Pull requests

Keep changes focused and open pull requests against `main`. Describe the problem, resulting behavior, and checks. CI runs the format check, clippy, and the tests on `ubuntu-latest`. It also exports synthetic reports for the app model and the retained legacy tools.

Use synthetic data in examples and bug reports. Keep credentials, personal holdings, downloaded datasets, and generated reports out of commits. Use ignored `data/` or `private/` folders for local inputs.
