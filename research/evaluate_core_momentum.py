"""Evaluate the one declared exploratory core rule against passive benchmarks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efficient_frontier.index_comparison import paired_normalized_bootstrap
from research.exploratory_core_momentum import run_core_momentum
from research.run_index_research import comparison, returns, read_prices, code_hashes, digest, ROOT, EVALUATION_START, EVALUATION_END


def verify_original_inputs(args):
    lock = json.loads((args.original_output / 'selection_lock.json').read_text())
    if lock['code_hashes'] != code_hashes():
        raise ValueError('Original research code changed after the recorded selection.')
    if lock['input_hashes'] != {name: digest(getattr(args, name)) for name in lock['input_hashes']}:
        raise ValueError('Price inputs differ from the original research run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('original-output', 'stock-prices', 'benchmarks', 'fund-prices', 'output'):
        parser.add_argument('--' + flag, required=True, type=Path)
    args = parser.parse_args()
    verify_original_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    # Load only artifacts generated locally by the original research runner.
    with (args.original_output / 'studies.pkl').open('rb') as handle:
        original = pickle.load(handle)
    stocks, benchmarks, funds = map(read_prices, (args.stock_prices, args.benchmarks, args.fund_prices))
    studies = {}
    for group in ('stocks', 'funds'):
        prices = stocks if group == 'stocks' else funds
        refs = benchmarks if group == 'stocks' else funds[['SPY', 'QQQ']]
        momentum = original['stocks', 'raw'] if group == 'stocks' else None
        for pipeline, target in (('raw', None), ('risk_controlled', .10)):
            for cost in (10, 25):
                studies[group, pipeline, cost] = run_core_momentum(prices, refs, momentum, cost, target)
    summary = {'classification': 'Exploratory follow-up after original results; no fresh evaluation data.',
               'primary_pipeline': 'raw', 'core_weight': .75, 'momentum_weight': .25, 'groups': {}}
    for group in ('stocks', 'funds'):
        start = EVALUATION_START[group]
        pieces = []
        for pipeline in ('raw', 'risk_controlled'):
            frame = returns(original[group, pipeline]).loc[start:EVALUATION_END]
            pieces.append(frame.drop(columns=['SPY', 'QQQ']).add_suffix(' · ' + pipeline))
        pieces.append(returns(original[group, 'raw']).loc[start:EVALUATION_END, ['SPY', 'QQQ']])
        core_name = next(name for name in studies[group, 'raw', 10].equity if name not in ('SPY', 'QQQ'))
        for pipeline in ('raw', 'risk_controlled'):
            study = studies[group, pipeline, 10]
            frame = returns(study).loc[start:EVALUATION_END]
            pieces.append(frame[[core_name]].add_suffix(' · ' + pipeline))
            for cost in (10, 25):
                result = studies[group, pipeline, cost]
                evaluation_returns = returns(result).loc[start:EVALUATION_END]
                table = comparison(evaluation_returns)
                table.to_csv(args.output / f'{group}_{pipeline}_{cost}bps_comparison.csv')
                evaluation_returns.to_csv(args.output / f'{group}_{pipeline}_{cost}bps_returns.csv', index_label='Date')
            study.allocations[core_name].iloc[-1].loc[lambda x: x > 1e-10].rename('target_weight').to_csv(
                args.output / f'{group}_{pipeline}_last_target.csv', index_label='holding')
        combined = pd.concat(pieces, axis=1)
        all_metrics = comparison(combined)
        all_metrics.to_csv(args.output / f'{group}_all_trials_vs_passive.csv')
        intervals = paired_normalized_bootstrap(combined, confidence=.975)
        intervals.to_csv(args.output / f'{group}_all_trials_uncertainty.csv', index=False)
        primary = studies[group, 'raw', 10]
        frame = returns(primary).loc[start:EVALUATION_END]
        primary_table = comparison(frame)
        annual = []
        for year, part in frame.groupby(frame.index.year):
            annual.append(comparison(part).assign(year=year, first_date=str(part.index[0].date()),
                                                 last_date=str(part.index[-1].date())).reset_index())
        pd.concat(annual, ignore_index=True).to_csv(args.output / f'{group}_primary_yearly.csv', index=False)
        bounds = intervals[(intervals.candidate == core_name + ' · raw') & (intervals.block_length == 21)]
        summary['groups'][group] = {
            'primary': core_name, 'start': str(frame.index[0].date()), 'end': str(frame.index[-1].date()),
            'observations': len(frame), 'metrics': primary_table.to_dict(orient='index'),
            'compared_candidate_implementations': len(combined.columns) - 2,
            'primary_bounds_at_all_trial_common_risk': bounds.to_dict(orient='records'),
            'last_signal': str(primary.trades[core_name].iloc[-1].signal_date.date()),
            'last_execution': str(primary.allocations[core_name].index[-1].date()),
        }
        print('\n', group, 'PRIMARY FULLY INVESTED 75/25 versus PASSIVE benchmarks')
        print(primary_table[['raw_mean','raw_cagr','raw_volatility','normalized_mean','normalized_cagr','normalized_volatility','worst_mean_gap']].round(6).to_string())
        print(bounds[['benchmark','estimate','lower_bound']].to_string(index=False))
    summary['code_hashes'] = {str(p): sha256((ROOT / p).read_bytes()).hexdigest() for p in (
        Path('research/exploratory_core_momentum.py'), Path('research/exploratory_core_protocol.json'),
        Path('research/evaluate_core_momentum.py'))}
    with (args.output / 'core_studies.pkl').open('wb') as handle:
        pickle.dump(studies, handle)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
