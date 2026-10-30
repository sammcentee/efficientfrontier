#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "Portfolio Lab needs the Rust toolchain. Install it from https://rustup.rs, then run this launcher again."
  exit 1
fi
echo "Portfolio Lab - starting your local dashboard"
echo "The first start compiles the app. It needs internet and can take a few minutes."
exec cargo run --release --locked --bin portfolio-lab -- "$@"
