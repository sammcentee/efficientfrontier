"""Reproducible command-line analysis and report export."""

import argparse
from datetime import date
from pathlib import Path
import shlex
import sys
from zipfile import ZipFile
from io import BytesIO

from .core import analyze
from .backtest import run_backtests
from .benchmark_report import default_evidence_strategy
from .benchmarks import compare_benchmarks
from .data import (BENCHMARK_SYMBOLS, demo_prices, download_benchmarks, download_prices,
                   load_csv, original_tickers, parse_tickers, validate_benchmark_prices)
from .evidence import analyze_evidence
from .historical import historical_demo, load_historical_prices, run_historical_backtests
from .historical_report import build_historical_report, historical_report_zip
from .membership import bundled_membership, load_membership_csv
from .presentation import report_zip
from .profiles import build_profiles
from .selection import run_selection
from .selection_report import selection_metrics_frame, selection_report_zip
from .universe import download_universe_prices, fetch_nasdaq100, fetch_sp500_nasdaq100


def write_report_bundle(bundle, output):
    """Replace generated results, including files from the other study mode."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.zip").write_bytes(bundle)
    with ZipFile(BytesIO(bundle)) as archive:
        current_files = set(archive.namelist())
        previous_backtests = [output / name for name in
                              ("weights.csv", "frontier.csv", "frontier_weights.csv", "holdout_metrics.csv",
                               "holdout_curve.csv", "risk_summary.csv", "risk_contributions.csv",
                               "membership.csv", "equity.csv", "metrics.csv", "cash_weights.csv", "coverage.csv", "LICENSE.n100tickers",
                               "findings.md", "backtest_metrics.csv", "backtest_curve.csv",
                               "latest_profile_summary.csv", "latest_profile_weights.csv", "latest_profile_windows.csv",
                               "latest_profile_window_returns.csv", "latest_profile_frontier.csv", "latest_profile_frontier_weights.csv",
                               "benchmark_prices.csv", "benchmark_holdout_metrics.csv", "benchmark_holdout_curve.csv",
                               "benchmark_training_estimates.csv", "benchmark_latest_estimates.csv",
                               "benchmark_backtest_metrics.csv", "benchmark_backtest_curve.csv",
                               "evidence_summary.csv", "evidence_windows.csv", "evidence_relative_curve.csv",
                               "universe_coverage.csv", "policy.json", "latest_weights.csv", "latest_estimates.csv",
                               "eligibility.csv")]
        for kind in ("holdings", "allocations", "trades"):
            previous_backtests.extend((output / "backtests").glob(f"[0-9][0-9]_{kind}.csv"))
            for phase in ("validation", "test"):
                previous_backtests.extend((output / phase).glob(f"[0-9][0-9]_{kind}.csv"))
        previous_backtests.extend(output / phase / name for phase in ("validation", "test")
                                  for name in ("metrics.csv", "equity.csv"))
        for path in previous_backtests:
            if path.is_file() and path.relative_to(output).as_posix() not in current_files:
                path.unlink()
        archive.extractall(output)


def historical_cli(args, settings):
    """Run the separate, offline study without imposing a fixed price universe."""
    if args.tickers is not None or args.original_holdings or args.nasdaq100:
        raise ValueError("Historical membership uses supplied --csv prices or --historical-demo; automatic price downloads are unavailable.")
    if args.benchmark_csv is not None or args.download_benchmarks:
        raise ValueError("Historical membership does not support market benchmark comparisons.")
    if args.historical_demo:
        if args.csv is not None or args.membership_end is not None:
            raise ValueError("--historical-demo supplies both prices and membership; omit --csv and --membership-end.")
        prices, membership = historical_demo()
        source = "Historical membership demo · synthetic"
    else:
        if args.csv is None:
            raise ValueError("Historical membership requires --csv with adjusted prices for stable security identities.")
        if args.membership_csv is not None:
            if args.membership_end is None:
                raise ValueError("--membership-csv requires --membership-end to declare the final covered date.")
            membership = load_membership_csv(args.membership_csv, coverage_end=args.membership_end,
                                             source="CSV: " + args.membership_csv.name)
        else:
            if args.membership_end is not None:
                raise ValueError("--membership-end applies only to --membership-csv.")
            membership = bundled_membership()
        prices = load_historical_prices(args.csv)
        source = "CSV: " + args.csv.name
    study = run_historical_backtests(prices, membership, **settings,
                                    rolling_window=args.rolling_window if args.rolling_window is not None else 126,
                                    rebalance_every=args.rebalance_every, cost_bps=args.cost_bps)
    study.settings.update(source=source, price_currency=args.currency.strip().upper(),
                          currency_assumption="All prices must share the declared currency. No currency conversion occurs.")
    files = build_historical_report(prices, membership, study)
    write_report_bundle(historical_report_zip(files), args.output)
    print(f"Source: {source}")
    print(f"Declared price currency: {args.currency.strip().upper()} (no currency conversion)")
    print(f"Membership coverage: {membership.coverage_start.date()} to {membership.coverage_end.date()}")
    for warning in study.warnings:
        print(f"Note: {warning}")
    print("\nHistorical membership · rolling-window portfolios after trading costs:")
    print(study.metrics.round(4).to_string())
    print(f"Eligibility decisions: {(args.output / 'coverage.csv').resolve()}")
    print(f"Report: {(args.output / 'report.html').resolve()}")
    return 0


def selection_cli(args, settings):
    """Run the separate risk-target selection and retain every reproducible result."""
    if args.rolling_window is not None:
        raise ValueError("--target-volatility uses expanding past-only fits; omit --rolling-window.")
    synthetic = not (args.csv is not None or args.tickers is not None or args.original_holdings
                     or args.nasdaq100 or args.sp500_nasdaq100)
    if synthetic and (args.benchmark_csv is not None or args.download_benchmarks):
        raise ValueError("Synthetic demo data cannot use real market benchmarks. Select --csv or a market source first.")
    universe_prices = None
    if args.sp500_nasdaq100 or args.nasdaq100:
        snapshot = fetch_sp500_nasdaq100() if args.sp500_nasdaq100 else fetch_nasdaq100()
        universe_prices = download_universe_prices(snapshot, args.start, args.end or date.today().isoformat())
        prices = universe_prices.prices
        source = "S&P 500 + Nasdaq-100 · Yahoo Finance" if args.sp500_nasdaq100 else "Nasdaq-100 · Yahoo Finance"
    elif args.csv is not None:
        prices, source = load_csv(args.csv), "CSV: " + args.csv.name
    elif args.tickers is not None or args.original_holdings:
        symbols = original_tickers(current_symbols=True) if args.original_holdings else parse_tickers(args.tickers)
        prices = download_prices(symbols, args.start, args.end or date.today().isoformat())
        source = "Yahoo Finance"
    else:
        prices, source = demo_prices(), "Demo · synthetic"
    currency = args.currency.strip().upper()
    metadata = dict(universe_prices.metadata) if universe_prices is not None else {}
    metadata.update(source=source, price_currency=currency,
                    currency_assumption=f"All prices are declared in {currency}. No currency conversion occurs.")
    if universe_prices is not None:
        metadata["universe_coverage"] = universe_prices.coverage.to_dict(orient="records")
    if universe_prices is not None or source == "Yahoo Finance":
        metadata.update(requested_start=args.start, requested_end_exclusive=args.end or date.today().isoformat())
    benchmark_prices = None
    if args.no_benchmarks:
        metadata["benchmarks_note"] = "Market benchmarks are disabled by --no-benchmarks; ETF advantage cannot be established."
    elif synthetic:
        metadata["benchmarks_note"] = "Synthetic demo data has no real market comparison; ETF advantage cannot be established."
    elif currency != "USD":
        metadata["benchmarks_note"] = f"Market comparisons require USD prices. Declared currency: {currency}. No currency conversion occurs."
    else:
        try:
            if args.benchmark_csv is not None:
                benchmark_prices = load_csv(args.benchmark_csv)
                metadata["benchmark_source"] = "CSV: " + args.benchmark_csv.name
            elif args.download_benchmarks:
                benchmark_prices = download_benchmarks(prices.index)
                metadata["benchmark_source"] = "Yahoo Finance"
            elif universe_prices is not None:
                benchmark_prices = universe_prices.benchmarks
                metadata["benchmark_source"] = "Yahoo Finance · universe download"
            elif set(BENCHMARK_SYMBOLS).issubset(prices.columns):
                benchmark_prices = prices.loc[:, list(BENCHMARK_SYMBOLS)]
                metadata["benchmark_source"] = "Asset columns: SPY, QQQ"
            elif source == "Yahoo Finance":
                benchmark_prices = download_benchmarks(prices.index)
                metadata["benchmark_source"] = "Yahoo Finance"
            else:
                metadata["benchmarks_note"] = "CSV analysis stays offline. Supply --benchmark-csv to establish ETF comparisons."
            if benchmark_prices is not None:
                benchmark_prices = validate_benchmark_prices(benchmark_prices, prices.index)
        except (ValueError, RuntimeError, OSError) as exc:
            benchmark_prices = None
            metadata["benchmarks_note"] = f"Market benchmarks are unavailable: {exc}"
    study = run_selection(prices, benchmark_prices, target_volatility=args.target_volatility, **settings,
                          rebalance_every=args.rebalance_every, cost_bps=args.cost_bps)
    export_directory = args.output.resolve()
    reproduce = ["python", "-m", "efficient_frontier", "--csv", str(export_directory / "prices.csv"),
                 "--target-volatility", str(args.target_volatility)]
    for key in ("train_fraction", "risk_free_rate", "max_weight", "shrinkage", "periods_per_year"):
        reproduce.extend(["--" + key.replace("_", "-"), str(settings[key])])
    reproduce.extend(["--rebalance-every", str(args.rebalance_every), "--cost-bps", str(args.cost_bps), "--currency", currency])
    reproduce.extend(["--benchmark-csv", str(export_directory / "benchmark_prices.csv")] if benchmark_prices is not None else ["--no-benchmarks"])
    reproduce.extend(["--output", str(export_directory / "reproduced")])
    metadata["reproduce_command"] = shlex.join(reproduce)
    metadata["reproduce_note"] = "Run from the project checkout with the same dependencies. Update file paths if the export folder moves."
    write_report_bundle(selection_report_zip(study, prices, metadata), args.output)
    print(f"Source: {source}")
    print(f"Annual volatility target: {args.target_volatility:.1%}; holding cap: {settings['max_weight']:.1%}")
    print("One rule: maximize historical mean return within the volatility budget and holding limit.")
    print("Historical checks passed." if study.supported else "Historical checks did not pass.")
    for note in dict.fromkeys([*study.reasons, *study.warnings]):
        print(f"Note: {note}")
    if "benchmarks_note" in metadata:
        print(f"Note: {metadata['benchmarks_note']}")
    print("\nLater historical test (one fixed rule):")
    print(selection_metrics_frame(study.test).round(4).to_string())
    print("Earlier research on these dates can introduce hindsight. A historical test alone is not unseen prospective evidence.")
    print(f"Report: {(args.output / 'report.html').resolve()}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Build portfolio frontiers, market comparisons, and historical evidence.")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--csv", type=Path, help="Adjusted prices: Date,ASSET1,ASSET2,…")
    sources.add_argument("--tickers", help="Comma-separated Yahoo Finance symbols; no ticker-count cap")
    sources.add_argument("--original-holdings", action="store_true", help="Use the original 60 tickers from spy_holdings.ods")
    sources.add_argument("--nasdaq100", action="store_true", help="Fetch current Nasdaq-100 members and retain securities with complete prices for the selected period")
    sources.add_argument("--sp500-nasdaq100", action="store_true", help="With --target-volatility, use the deduplicated S&P 500 (SPY holdings proxy) and Nasdaq-100 stock universe")
    membership_sources = parser.add_mutually_exclusive_group()
    membership_sources.add_argument("--historical-membership", action="store_true", help="Use bundled historical Nasdaq-100 membership with supplied --csv prices; stays offline")
    membership_sources.add_argument("--membership-csv", type=Path, help="Complete dated membership snapshots; requires --csv prices and --membership-end")
    membership_sources.add_argument("--historical-demo", action="store_true", help="Offline synthetic study with changing membership and sparse price history")
    parser.add_argument("--membership-end", help="Final covered date for a custom membership CSV, YYYY-MM-DD")
    benchmark_sources = parser.add_mutually_exclusive_group()
    benchmark_sources.add_argument("--benchmark-csv", type=Path, help="Offline USD benchmark prices: Date,SPY,QQQ; must cover every asset date")
    benchmark_sources.add_argument("--download-benchmarks", action="store_true", help="Download SPY and QQQ from Yahoo, including for CSV portfolios; unavailable for synthetic demo data")
    benchmark_sources.add_argument("--no-benchmarks", action="store_true", help="Skip market benchmark comparisons")
    parser.add_argument("--currency", default="USD", help="Declare the common asset-price currency; default USD. Benchmarks require USD. No currency conversion.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", help="Yahoo end date, exclusive; defaults to today")
    parser.add_argument("--train-fraction", type=float, default=.7)
    parser.add_argument("--risk-free-rate", type=float, default=.02, help="Annual decimal, e.g. .02")
    parser.add_argument("--max-weight", type=float,
                        help="Per-asset cap as a decimal; default .10 with --target-volatility, otherwise 1 (no additional cap)")
    parser.add_argument("--target-volatility", type=float,
                        help="Run single-portfolio selection with an annual volatility target as a decimal, e.g. .20 for 20 percent; omit a source flag to use the offline demo")
    parser.add_argument("--shrinkage", type=float, default=.1,
                        help="Covariance shrinkage toward zero correlations: 0 uses sample covariance; 1 keeps only individual variances")
    parser.add_argument("--periods-per-year", type=float, default=252,
                        help="Observations per year: 252 trading days, 365 calendar days, 52 weeks, or 12 months; no resampling")
    parser.add_argument("--backtests", action="store_true", help="Compare buy-and-hold, fixed rebalancing, expanding and rolling windows")
    parser.add_argument("--rebalance-every", type=int, default=21, help="Backtest trading interval in observations (not calendar months)")
    parser.add_argument("--rolling-window", type=int, help="Rolling estimation returns; default is 126 for historical membership, otherwise the initial training length")
    parser.add_argument("--cost-bps", type=float, default=10., help="Backtest fee per unit of bought or sold notional, in basis points")
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    args = parser.parse_args()
    if args.max_weight is None:
        args.max_weight = .1 if args.target_volatility is not None else 1.0
    settings = {key: getattr(args, key) for key in ("train_fraction", "risk_free_rate", "max_weight", "shrinkage", "periods_per_year")}
    try:
        currency = args.currency.strip().upper()
        if not currency:
            raise ValueError("currency must identify the common price currency, such as USD or EUR.")
        if args.historical_demo or args.historical_membership or args.membership_csv is not None:
            if args.target_volatility is not None or args.sp500_nasdaq100:
                raise ValueError("Historical membership is a separate study; omit --target-volatility and --sp500-nasdaq100.")
            return historical_cli(args, settings)
        if args.membership_end is not None:
            raise ValueError("--membership-end requires --membership-csv.")
        if args.target_volatility is not None:
            return selection_cli(args, settings)
        if args.sp500_nasdaq100:
            raise ValueError("--sp500-nasdaq100 requires --target-volatility, for example .20 for 20% annual volatility.")
        synthetic = args.csv is None and args.tickers is None and not args.original_holdings and not args.nasdaq100
        if synthetic and (args.benchmark_csv is not None or args.download_benchmarks):
            raise ValueError("Synthetic demo data cannot use real market benchmarks. Select --csv or --tickers first.")
        universe_prices = None
        if args.nasdaq100:
            universe_prices = download_universe_prices(fetch_nasdaq100(), args.start, args.end or date.today().isoformat())
            prices, source = universe_prices.prices, "Nasdaq-100 · Yahoo Finance"
            if len(prices.columns) < 2:
                raise ValueError(
                    f"Only {len(prices.columns)} of {len(universe_prices.coverage)} Nasdaq-100 securities have complete prices. "
                    f"At least two are required. {len(universe_prices.coverage) - len(prices.columns)} securities were excluded."
                )
        elif args.csv:
            prices, source = load_csv(args.csv), "CSV: " + args.csv.name
        elif args.tickers is not None or args.original_holdings:
            tickers = original_tickers(current_symbols=True) if args.original_holdings else parse_tickers(args.tickers)
            prices = download_prices(tickers, args.start, args.end or date.today().isoformat())
            source = "Yahoo Finance"
        else:
            prices, source = demo_prices(), "Demo · synthetic"
        analysis = analyze(prices, **settings)
        latest_profiles = build_profiles(
            prices, return_objective="historical_mean",
            **{key: value for key, value in settings.items() if key != "train_fraction"},
        ) if len(prices) >= 7 else None
        profile_backtests = min(len(analysis.train_returns),
                                args.rolling_window if args.rolling_window is not None else len(analysis.train_returns)) >= 6
        study = run_backtests(prices, **settings, rebalance_every=args.rebalance_every,
                              rolling_window=args.rolling_window, cost_bps=args.cost_bps,
                              include_profiles=profile_backtests) if args.backtests else None
        if study is not None and not profile_backtests:
            study.warnings.append("Risk-profile backtests need at least six returns in both the initial and rolling fit windows. Only the original targets are shown.")
        metadata = dict(universe_prices.metadata) if universe_prices is not None else {}
        metadata.update(source=source, price_currency=currency,
                        currency_assumption=f"The user declares all asset prices in {currency}. No currency conversion occurs. USD benchmark ETFs require the same currency basis.",
                        **settings)
        if universe_prices is not None:
            metadata["universe_coverage"] = universe_prices.coverage.to_dict(orient="records")
        if latest_profiles is None:
            metadata["latest_profiles_note"] = "Latest profiles need at least seven prices for three windows with two returns each."
        if source == "Yahoo Finance" or args.nasdaq100:
            metadata.update(requested_start=args.start, requested_end_exclusive=args.end or date.today().isoformat())
        benchmarks = evidence = None
        if args.no_benchmarks:
            metadata["benchmarks_note"] = "Market benchmarks are disabled by --no-benchmarks."
        elif synthetic:
            metadata["benchmarks_note"] = "Synthetic demo data has no real market benchmark comparison."
        elif currency != "USD":
            metadata["benchmarks_note"] = f"Market benchmarks require USD prices. The declared currency is {currency}, and no currency conversion occurs."
        else:
            try:
                benchmark_prices = None
                if args.benchmark_csv is not None:
                    metadata["benchmark_source"] = "CSV: " + args.benchmark_csv.name
                    benchmark_prices = load_csv(args.benchmark_csv)
                elif args.download_benchmarks:
                    metadata["benchmark_source"] = "Yahoo Finance"
                    benchmark_prices = download_benchmarks(prices.index)
                elif universe_prices is not None:
                    metadata["benchmark_source"] = "Yahoo Finance · Nasdaq-100 download"
                    benchmark_prices = universe_prices.benchmarks
                elif set(BENCHMARK_SYMBOLS).issubset(prices.columns):
                    metadata["benchmark_source"] = "Asset columns: " + ", ".join(BENCHMARK_SYMBOLS)
                    benchmark_prices = prices.loc[:, list(BENCHMARK_SYMBOLS)]
                elif source == "Yahoo Finance":
                    metadata["benchmark_source"] = "Yahoo Finance"
                    benchmark_prices = download_benchmarks(prices.index)
                else:
                    metadata["benchmarks_note"] = "CSV analysis stays offline. Supply --benchmark-csv or --download-benchmarks, or include SPY and QQQ price columns."
                if benchmark_prices is not None:
                    benchmark_prices = validate_benchmark_prices(benchmark_prices, prices.index)
                    benchmarks = compare_benchmarks(benchmark_prices, prices, analysis, study, latest_profiles,
                                                    risk_free_rate=args.risk_free_rate)
            except (ValueError, RuntimeError, OSError) as exc:
                metadata["benchmarks_note"] = f"Market benchmarks are unavailable: {exc}"
        if benchmarks is not None:
            try:
                evidence = analyze_evidence(
                    study.equity if study is not None else analysis.equity,
                    benchmarks.backtest_equity if study is not None else benchmarks.holdout_equity,
                    risk_free_rate=args.risk_free_rate, periods_per_year=args.periods_per_year,
                    delayed_entry=study is not None,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                metadata["evidence_note"] = f"Historical evidence is unavailable: {exc}"
        bundle = report_zip(analysis, prices, metadata, study, latest_profiles, benchmarks, evidence)
        write_report_bundle(bundle, args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    print(f"Source: {source}")
    print(f"Declared price currency: {currency} (no currency conversion)")
    if universe_prices is not None:
        included, requested = len(prices.columns), len(universe_prices.coverage)
        print(f"Nasdaq-100 coverage: {included} of {requested} securities included; {requested - included} excluded.")
        print(f"Membership source date: {metadata['universe_source_date']}")
        print(f"Membership source: {metadata['universe_source_url']}")
        print(f"Note: {metadata['universe_limitation']}")
        print(f"Coverage: {(args.output / 'universe_coverage.csv').resolve()}")
    if latest_profiles is not None:
        print(f"\nLatest model holdings as of {latest_profiles.as_of} "
              "(historical mean frontier; full-history fit, not an out-of-sample result):")
        print(latest_profiles.summary.round(4).to_string())
        for name, portfolio in latest_profiles.portfolios.items():
            leaders = portfolio.weights.sort_values(ascending=False).head(5)
            print(f"{name} largest allocations: " + ", ".join(f"{asset} {weight:.2%}" for asset, weight in leaders.items()))
        for warning in latest_profiles.warnings:
            print(f"Note: {warning}")
    else:
        print(f"Note: {metadata['latest_profiles_note']}")
    for warning in analysis.warnings:
        print(f"Note: {warning}")
    print("\nOriginal chronological holdout:")
    print(analysis.holdout_metrics.round(4).to_string())
    if study is not None:
        for warning in study.warnings:
            print(f"Note: {warning}")
        print("\nBacktest comparison (after the selected trading costs):")
        print(study.metrics.round(4).to_string())
        print(f"Findings: {(args.output / 'findings.md').resolve()}")
    if benchmarks is not None:
        print(f"\nMarket benchmarks ({metadata['benchmark_source']}; matched dates and entry rules):")
        benchmark_metrics = benchmarks.backtest_metrics if study is not None else benchmarks.holdout_metrics
        print(benchmark_metrics[["total_return", "cagr", "volatility", "max_drawdown"]].round(4).to_string())
    if evidence is not None:
        selected = default_evidence_strategy(evidence)
        if selected is not None:
            print(f"\nHistorical evidence for {selected} (fixed display choice, not the best result):")
            print(evidence.summary.xs(selected, level="strategy")[[
                "annual_advantage", "advantage_ci_low", "advantage_ci_high", "alpha",
                "alpha_pvalue_adjusted", "status",
            ]].round(4).to_string())
        for warning in evidence.warnings:
            print(f"Note: {warning}")
    for note in ("benchmarks_note", "evidence_note"):
        if note in metadata:
            print(f"Note: {metadata[note]}")
    print(f"Report: {(args.output / 'report.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
