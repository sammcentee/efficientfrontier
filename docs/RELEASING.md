# Release checklist

The default branch is `main`. The owner authorized public source publication on 2026-10-03; the review is recorded in [PUBLIC_RELEASE_REVIEW.md](PUBLIC_RELEASE_REVIEW.md). Original project code and documentation are licensed under the [MIT License](../LICENSE). Source visibility does not create a versioned GitHub release or deploy an application.

## Before changing visibility

- Include the project's MIT license and preserve third-party licenses and notices. Keep the README accurate about the license scope; the project license does not grant rights to third-party software or data.
- Confirm rights to any newly added code or data. The owner confirmed publication rights for the original R script and `spy_holdings.ods` on 2026-10-03. The spreadsheet is a static list of 60 symbols, not a historical index-membership dataset.
- Review Git authorship metadata. Existing commits contain the author's personal email; changing future Git settings does not remove it from earlier commits. No history rewrite has been performed.
- Review tracked files and history for credentials and private data. The publication audit covered the four prior commits plus the release-preparation files; it is not a guarantee about later changes or every possible secret.
- Keep downloaded market prices and reports containing them out of public examples unless redistribution is permitted by the provider's terms. Use the deterministic synthetic demo for release screenshots and examples.
- Verify that the security-reporting route described in `SECURITY.md` works, and review the repository's secret-scanning and dependency-alert settings.

## Validate the release candidate

Run these commands from a fresh clone with the current stable Rust toolchain:

```bash
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo clippy --locked -p portfolio-lab --all-targets -- -D warnings
cargo test --locked -p portfolio-lab
cargo test --locked --test network_live -- --ignored
cargo run --release --locked --bin efficient-frontier -- --output results/release-check
cargo run --release --locked --bin efficient-frontier -- --target-volatility .20 --max-weight .20 --output results/release-check-model
bash run.sh
```

The live network tests need internet access. Confirm that CI passes on the intended `main` commit, including the window tests on Ubuntu and Windows. Open the programs from the CI artifacts on Windows and Linux. Confirm that the app shows the synthetic-data label and that the generated report opens offline. Test a CSV upload and download, including a repeat analysis after changing settings. Review the financial assumptions and known covariance degeneracy in the README.

If you replace a file in `assets/plotly/` or `assets/katex/`, update its license file and its version in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). If you add or remove a direct dependency in `Cargo.toml` or `desktop/Cargo.toml`, update the table of Rust dependencies in that file. Preserve notices in the standalone HTML and report ZIP.

## When the owner is ready

1. Make the repository public in GitHub settings after the checks above and the owner's explicit decision to publish.
2. Choose a release version. Write it in the `version` fields of `Cargo.toml`, `desktop/Cargo.toml`, and `desktop/tauri.conf.json`.
3. Run `cargo update --workspace`. This command writes the new version into `Cargo.lock`. The launchers use `--locked`, so they stop when `Cargo.lock` does not agree with `Cargo.toml`.
4. Do the checks above again. Commit `Cargo.toml`, `Cargo.lock`, and a record of the changes.
5. Create an annotated tag on the verified commit, for example `git tag -a v2.0.0 -m "Version 2.0.0"`. Push the tag with `git push origin v2.0.0`.
6. Create the GitHub release from that tag.
7. Check the public README, images, CI status and fresh-clone instructions without signing in.

No automation creates a version tag or a GitHub release. Steps 5 and 6 create them.

GitHub visibility and a hosted application are separate decisions. The app is a desktop program without a network port. A hosted service needs its own deployment configuration and a review of data uploads, downloads and access controls.
