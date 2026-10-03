#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if command -v python3.14 >/dev/null 2>&1; then
  exec python3.14 launch.py "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 launch.py "$@"
else
  echo "Portfolio Lab needs Python 3.14. Install it from https://www.python.org/downloads/"
  exit 1
fi
