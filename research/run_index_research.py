"""Run the frozen development and evaluation stages from supplied price files."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efficient_frontier.index_comparison import normalized_metrics, paired_normalized_bootstrap
from efficient_frontier.index_rules import run_fund_rules, run_index_rules


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("SPY", "QQQ")
DEVELOPMENT_END = {"stocks": "2025-04-16", "funds": "2022-12-30"}
EVALUATION_START = {"stocks": "2025-04-17", "funds": "2023-01-03"}
EVALUATION_END = "2026-10-16"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_prices(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def code_hashes():
    names = ("efficient_frontier/index_rules.py", "efficient_frontier/index_comparison.py",
             "research/index_research_protocol.json", "research/run_index_research.py",
             "efficient_frontier/backtest.py", "efficient_frontier/core.py",
             "efficient_frontier/metrics.py", "efficient_frontier/selection.py", "requirements.txt")
    return {name: digest(ROOT / name) for name in names}


def returns(study):
    return study.equity.pct_change(fill_method=None).iloc[1:]


def comparison(frame):
    result = normalized_metrics(frame)
    for benchmark in BENCHMARKS:
        result[f"mean_gap_{benchmark}"] = result.normalized_mean - result.loc[benchmark, "normalized_mean"]
        result[f"cagr_gap_{benchmark}"] = result.normalized_cagr - result.loc[benchmark, "normalized_cagr"]
    result["worst_mean_gap"] = result[[f"mean_gap_{b}" for b in BENCHMARKS]].min(axis=1)
    return result


def build(args, cost):
    stock_prices, benchmarks, fund_prices = map(read_prices, (args.stock_prices, args.benchmarks, args.fund_prices))
    studies = {}
    for group, run, inputs in (("stocks", run_index_rules, (stock_prices, benchmarks)),
                              ("funds", run_fund_rules, (fund_prices,))):
        for pipeline, target in (("raw", None), ("risk_controlled", .10)):
            studies[group, pipeline] = run(*inputs, cost_bps=cost, target_volatility=target)
    return studies


def development(args):
    if (args.output / "selection_lock.json").exists():
        raise ValueError("A selection lock already exists. Use a new output directory for a new recorded research run.")
    studies = build(args, 10)
    lock = {"created_at": datetime.now(timezone.utc).isoformat(), "code_hashes": code_hashes(),
            "input_hashes": {name: digest(getattr(args, name)) for name in ("stock_prices", "benchmarks", "fund_prices")},
            "chosen": {}, "development_end": DEVELOPMENT_END, "evaluation_start": EVALUATION_START,
            "evaluation_end": EVALUATION_END, "prior_date_review": True,
            "selection_pipeline": "risk_controlled", "selection_metric": "worst normalized annual mean gap"}
    for (group, pipeline), study in studies.items():
        frame = returns(study).loc[:DEVELOPMENT_END[group]]
        table = comparison(frame)
        table["development_turnover"] = pd.Series({
            name: trades.loc[:DEVELOPMENT_END[group], "turnover"].sum() for name, trades in study.trades.items()
        })
        table.to_csv(args.output / f"{group}_{pipeline}_development.csv")
        if pipeline == "risk_controlled":
            candidates = table.drop(index=list(BENCHMARKS)).reset_index().sort_values(
                ["worst_mean_gap", "development_turnover", "strategy"], ascending=[False, True, True], kind="stable")
            winner = candidates.iloc[0]
            lock["chosen"][group] = {"name": winner.strategy, "development_score": float(winner.worst_mean_gap),
                                      "beat_both_in_development": bool(winner.worst_mean_gap > 0),
                                      "first_return": str(frame.index[0].date()), "observations": len(frame)}
            print(f"{group}: development choice {winner.strategy}; worst equal-risk annual mean gap {winner.worst_mean_gap:.2%}")
            print(table[["raw_mean", "raw_volatility", "normalized_mean", "normalized_volatility", "worst_mean_gap"]].round(6).to_string())
    with (args.output / "studies.pkl").open("wb") as handle:
        pickle.dump(studies, handle)
    (args.output / "selection_lock.json").write_text(json.dumps(lock, indent=2))
    print("Frozen selection:", args.output / "selection_lock.json")


def evaluation(args):
    lock_path = args.output / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["code_hashes"] != code_hashes():
        raise ValueError("Research code or protocol changed after selection. Record a new development run first.")
    if lock["input_hashes"] != {name: digest(getattr(args, name)) for name in lock["input_hashes"]}:
        raise ValueError("Price inputs changed after selection.")
    # Only unpickle artifacts generated by this script in the trusted local workspace.
    with (args.output / "studies.pkl").open("rb") as handle:
        studies = pickle.load(handle)
    summary = {"selection_lock_sha256": digest(lock_path), "selection": lock["chosen"], "groups": {}}
    for (group, pipeline), study in studies.items():
        frame = returns(study).loc[EVALUATION_START[group]:EVALUATION_END]
        table = comparison(frame)
        table.to_csv(args.output / f"{group}_{pipeline}_evaluation.csv")
        frame.to_csv(args.output / f"{group}_{pipeline}_evaluation_returns.csv", index_label="Date")
        study.equity.to_csv(args.output / f"{group}_{pipeline}_full_equity.csv", index_label="Date")
        annual = []
        for year, part in frame.groupby(frame.index.year):
            annual.append(comparison(part).assign(year=year, first_date=str(part.index[0].date()),
                                                  last_date=str(part.index[-1].date())).reset_index())
        pd.concat(annual, ignore_index=True).to_csv(args.output / f"{group}_{pipeline}_yearly.csv", index=False)
        if pipeline == "risk_controlled":
            # Two fixed study families: 97.5% simultaneous bounds per family give
            # an approximate 95% familywise level under bootstrap assumptions.
            intervals = paired_normalized_bootstrap(frame, confidence=.975)
            intervals.to_csv(args.output / f"{group}_uncertainty.csv", index=False)
            winner = lock["chosen"][group]["name"]
            chosen_bounds = intervals[(intervals.candidate == winner) & (intervals.block_length == 21)]
            summary["groups"][group] = {
                "winner": winner, "first_return": str(frame.index[0].date()), "last_return": str(frame.index[-1].date()),
                "observations": len(frame), "metrics": table.loc[winner].to_dict(),
                "benchmark_metrics": {name: table.loc[name].to_dict() for name in BENCHMARKS},
                "both_mean_gaps_positive": bool(table.loc[winner, "worst_mean_gap"] > 0),
                "both_simultaneous_lower_bounds_positive": bool((chosen_bounds.lower_bound > 0).all()),
                "lower_bounds": dict(zip(chosen_bounds.benchmark, chosen_bounds.lower_bound)),
            }
            print(f"\n{group}: frozen choice {winner}; evaluation {frame.index[0].date()} to {frame.index[-1].date()}")
            print(table[["raw_mean", "raw_cagr", "raw_volatility", "normalized_mean", "normalized_cagr", "normalized_volatility", "worst_mean_gap"]].round(6).to_string())
            print(chosen_bounds[["benchmark", "estimate", "lower_bound"]].to_string(index=False))
    stress = build(args, 25)
    for (group, pipeline), study in stress.items():
        frame = returns(study).loc[EVALUATION_START[group]:EVALUATION_END]
        comparison(frame).to_csv(args.output / f"{group}_{pipeline}_cost_stress.csv")
    with (args.output / "stress_studies.pkl").open("wb") as handle:
        pickle.dump(stress, handle)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved evaluation and cost stress:", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "evaluation"))
    parser.add_argument("--stock-prices", required=True, type=Path)
    parser.add_argument("--benchmarks", required=True, type=Path)
    parser.add_argument("--fund-prices", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (development if args.stage == "development" else evaluation)(args)


if __name__ == "__main__":
    main()
