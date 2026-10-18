from types import SimpleNamespace
import json

import pytest

from research import evaluate_core_momentum as runner


def test_follow_up_rejects_mixed_inputs_and_changed_original_code(tmp_path):
    inputs = {}
    for name in ('stock_prices', 'benchmarks', 'fund_prices'):
        path = tmp_path / f'{name}.csv'
        path.write_text('Original input.\n')
        inputs[name] = path
    lock = {'code_hashes': runner.code_hashes(),
            'input_hashes': {name: runner.digest(path) for name, path in inputs.items()}}
    lock_path = tmp_path / 'selection_lock.json'
    lock_path.write_text(json.dumps(lock))
    args = SimpleNamespace(original_output=tmp_path, **inputs)
    runner.verify_original_inputs(args)
    args.benchmarks.write_text('Different benchmark observations.\n')
    with pytest.raises(ValueError, match='Price inputs differ'):
        runner.verify_original_inputs(args)
    lock['code_hashes'] = {'old': 'hash'}
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='Original research code changed'):
        runner.verify_original_inputs(args)
