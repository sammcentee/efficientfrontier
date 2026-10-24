from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from research import run_index_research as runner


def test_development_choice_ignores_later_winner_and_locks_inputs(tmp_path, monkeypatch):
    dates = pd.bdate_range('2022-09-01', periods=150)
    common = .01 * np.sin(np.arange(len(dates)))
    frame = pd.DataFrame({'SPY': common + .0002, 'QQQ': common + .0003,
                          'steady': common + .0008, 'later_winner': common - .0005}, index=dates)
    frame.loc[frame.index >= '2023-01-03', 'later_winner'] += .2
    equity = (1 + frame).cumprod()
    study = SimpleNamespace(equity=equity,
                            trades={name: pd.DataFrame({'turnover': [1.]}, index=dates[:1]) for name in frame})
    monkeypatch.setattr(runner, 'build', lambda *args: {('funds', 'risk_controlled'): study})
    inputs = {}
    for name in ('stock_prices', 'benchmarks', 'fund_prices'):
        path = tmp_path / f'{name}.csv'
        path.write_text('A fixed input for hashing.\n')
        inputs[name] = path
    args = SimpleNamespace(output=tmp_path, **inputs)
    runner.development(args)
    lock = json.loads((tmp_path / 'selection_lock.json').read_text())
    assert lock['chosen']['funds']['name'] == 'steady'
    assert lock['chosen']['funds']['beat_both_in_development']
    assert lock['input_hashes'] == {name: runner.digest(path) for name, path in inputs.items()}
    assert lock['code_hashes'] == runner.code_hashes()
    assert not (tmp_path / 'funds_risk_controlled_evaluation.csv').exists()
    with pytest.raises(ValueError, match='selection lock already exists'):
        runner.development(args)
    args.fund_prices.write_text('Changed input.\n')
    with pytest.raises(ValueError, match='Price inputs changed'):
        runner.evaluation(args)


def test_evaluation_refuses_code_changed_after_selection(tmp_path):
    (tmp_path / 'selection_lock.json').write_text(json.dumps({'code_hashes': {'old': 'hash'}}))
    with pytest.raises(ValueError, match='code or protocol changed'):
        runner.evaluation(SimpleNamespace(output=tmp_path))
