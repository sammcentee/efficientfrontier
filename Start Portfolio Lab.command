#!/usr/bin/env bash
cd -- "$(dirname -- "$0")" || exit 1
bash run.sh "$@"
portfolio_exit=$?
if [[ $portfolio_exit -ne 0 ]]; then
  read -r -p "Press Enter to close this window. "
fi
exit "$portfolio_exit"
