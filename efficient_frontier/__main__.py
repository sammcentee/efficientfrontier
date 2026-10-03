"""Reproducible command-line analysis and report export."""

import argparse
from pathlib import Path
import sys
from zipfile import ZipFile
from io import BytesIO

from .core import analyze
from .backtest import run_backtests
from .data import demo_prices, download_prices, load_csv, original_tickers, parse_tickers
from .presentation import report_zip


def main():
    parser = argparse.ArgumentParser(description="Build an efficient frontier and chronological holdout report.")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--csv", type=Path, help="Adjusted daily prices: Date,ASSET1,ASSET2,…")
    sources.add_argument("--tickers", help="Comma-separated Yahoo Finance symbols; no ticker-count cap")
    sources.add_argument("--original-holdings", action="store_true", help="Use the original 60 tickers from spy_holdings.ods")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", help="Yahoo end date, exclusive; defaults to today")
    parser.add_argument("--train-fraction", type=float, default=.7)
    parser.add_argument("--risk-free-rate", type=float, default=.02, help="Annual decimal, e.g. .02")
    parser.add_argument("--max-weight", type=float, default=1.0,
                        help="Optional per-asset cap as a decimal; default 1 means no additional cap")
    parser.add_argument("--shrinkage", type=float, default=.1)
    parser.add_argument("--backtests", action="store_true", help="Compare buy-and-hold, fixed rebalancing, expanding and rolling windows")
    parser.add_argument("--rebalance-every", type=int, default=21, help="Backtest trading interval in sessions (not calendar months)")
    parser.add_argument("--rolling-window", type=int, help="Rolling estimation returns; default is the initial training length")
    parser.add_argument("--cost-bps", type=float, default=10., help="Backtest fee per unit of bought or sold notional, in basis points")
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    args = parser.parse_args()
    settings = {key: getattr(args, key) for key in ("train_fraction", "risk_free_rate", "max_weight", "shrinkage")}
    try:
        if args.csv:
            prices, source = load_csv(args.csv), "CSV: " + args.csv.name
        elif args.tickers is not None or args.original_holdings:
            from datetime import date
            tickers = original_tickers(current_symbols=True) if args.original_holdings else parse_tickers(args.tickers)
            prices = download_prices(tickers, args.start, args.end or date.today().isoformat())
            source = "Yahoo Finance"
        else:
            prices, source = demo_prices(), "Demo · synthetic"
        analysis = analyze(prices, **settings)
        study = run_backtests(prices, **settings, rebalance_every=args.rebalance_every,
                              rolling_window=args.rolling_window, cost_bps=args.cost_bps) if args.backtests else None
        metadata = {"source": source, **settings}
        if source == "Yahoo Finance":
            metadata.update(requested_start=args.start, requested_end_exclusive=args.end or date.today().isoformat())
        bundle = report_zip(analysis, prices, metadata, study)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.zip").write_bytes(bundle)
        with ZipFile(BytesIO(bundle)) as archive:
            current_files = set(archive.namelist())
            previous_backtests = [args.output / name for name in
                                  ("findings.md", "backtest_metrics.csv", "backtest_curve.csv")]
            for kind in ("holdings", "allocations", "trades"):
                previous_backtests.extend((args.output / "backtests").glob(f"[0-9][0-9]_{kind}.csv"))
            for path in previous_backtests:
                if path.is_file() and path.relative_to(args.output).as_posix() not in current_files:
                    path.unlink()
            archive.extractall(args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Source: {source}")
    for warning in analysis.warnings:
        print(f"Note: {warning}")
    print(analysis.holdout_metrics.round(4).to_string())
    if study is not None:
        print("\nBacktest comparison (after the selected trading costs):")
        print(study.metrics.round(4).to_string())
        print(f"Findings: {(args.output / 'findings.md').resolve()}")
    print(f"Report: {(args.output / 'report.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
