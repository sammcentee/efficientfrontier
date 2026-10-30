"""Golden results of the Python release for the historical membership parity tests.

Run from a checkout of the Python release (commit 6c6df6e) with its
requirements installed:

    python tests/fixtures/golden/generate_historical.py

All prices are synthetic. The script writes input prices as CSV and every
result as JSON beside itself.
"""

import json
import math
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from efficient_frontier.backtest import run_backtests  # noqa: E402
from efficient_frontier.historical import (  # noqa: E402
    historical_demo, load_historical_prices, run_historical_backtests,
)
from efficient_frontier.membership import bundled_membership, load_membership_csv  # noqa: E402

HERE = Path(__file__).resolve().parent


def value(item):
    if isinstance(item, (pd.Timestamp,)):
        return None if pd.isna(item) else item.date().isoformat()
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    if isinstance(item, (np.integer,)):
        return int(item)
    if isinstance(item, (float, np.floating)):
        return None if math.isnan(item) else float(item)
    if item is None or item is pd.NaT:
        return None
    if isinstance(item, (list, tuple)):
        return [value(x) for x in item]
    if isinstance(item, dict):
        return {str(k): value(v) for k, v in item.items()}
    return item


def split(frame):
    index = frame.index
    return {
        "index_names": [value(name) for name in index.names],
        "index": [value(x) for x in index],
        "columns": [str(c) for c in frame.columns],
        "data": [[value(x) for x in row] for row in frame.itertuples(index=False, name=None)],
    }


def write(name, result):
    (HERE / f"{name}.json").write_text(json.dumps(result, indent=1))


def study_json(study):
    return {
        "equity": split(study.equity), "cash_weights": split(study.cash_weights),
        "metrics": split(study.metrics), "coverage": split(study.coverage),
        "allocations": {k: split(v) for k, v in study.allocations.items()},
        "holdings": {k: split(v) for k, v in study.holdings.items()},
        "trades": {k: split(v) for k, v in study.trades.items()},
        "warnings": study.warnings, "settings": value(study.settings),
        "settings_keys": list(study.settings), "membership_keys": list(study.settings["membership"]),
    }


def roster_csv(dates, changes):
    """Membership CSV text for (row, tickers) changes, as the Python tests build it."""
    table = pd.DataFrame({"date": [str(dates[row].date()) for row, _ in changes],
                          "tickers": [symbols for _, symbols in changes]})
    return table.to_csv(index=False)


def test_prices():
    """The rng(913) price fixture of tests/test_historical.py."""
    rng = np.random.default_rng(913)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.001, .015, (45, 3)), axis=0)),
                        index=pd.bdate_range("2021-01-04", periods=45, name="Date"), columns=["A", "B", "C"])


def bundled_prices():
    """Synthetic prices for bundled Nasdaq-100 labels around known membership and identity events."""
    dates = pd.bdate_range("2012-11-01", "2014-12-31", name="Date")
    symbols = ["AAPL", "MSFT", "INTC", "CSCO", "GOOG", "GOOGL", "NWSA", "DELL", "FOXA", "TSLA", "BMC", "XTRA"]
    rng = np.random.default_rng(2013)
    market = rng.normal(.0006, .009, len(dates))
    # A weak first quarter of 2014 adds a drawdown inside the test period.
    market[(dates >= "2014-01-01") & (dates < "2014-04-01")] -= .003
    betas = np.linspace(.6, 1.4, len(symbols))
    noise = rng.normal(0, .012, (len(dates), len(symbols)))
    prices = pd.DataFrame(50 * np.exp(np.cumsum(market[:, None] * betas + noise, axis=0)), index=dates, columns=symbols)
    prices.loc[:"2014-04-02", "GOOGL"] = np.nan
    return prices


def custom_prices():
    """Weekly synthetic prices with one late listing and one weak asset."""
    dates = pd.bdate_range("2015-01-05", periods=1100, name="Date")
    rng = np.random.default_rng(77)
    drift = np.array([.0005, .0003, -.0006, .0004, .0002])
    daily = pd.DataFrame(80 * np.exp(np.cumsum(rng.normal(drift, .013, (len(dates), 5)), axis=0)),
                         index=dates, columns=["A", "B", "C", "D", "E"])
    weekly = daily.iloc[::5].copy()
    weekly.iloc[:120, 3] = np.nan
    return weekly


def study_case(name, prices, membership, spec, **options):
    if spec["kind"] != "demo":
        prices.to_csv(HERE / f"{name}_prices.csv", lineterminator="\n")
    study = run_historical_backtests(prices, membership, **options)
    write(name, {"options": options, "membership": spec, "study": study_json(study)})


def fixed_case(prices):
    membership = load_membership_csv(StringIO(roster_csv(prices.index, [(0, "A,B,C")])),
                                     coverage_end=str(prices.index[-1].date()))
    options = dict(train_fraction=.5, rolling_window=5, rebalance_every=3, cost_bps=37., max_weight=.5,
                   periods_per_year=12., shrinkage=.3, risk_free_rate=.01)
    historical = run_historical_backtests(prices, membership, **options)
    existing = run_backtests(prices, **options)
    names = list(historical.equity.columns)
    write("historical_fixed", {
        "options": options, "changes": [[0, "A,B,C"]],
        "rolling": {
            "equity": split(existing.equity[names]), "metrics": split(existing.metrics.loc[names]),
            "allocations": {k: split(existing.allocations[k]) for k in names},
            "holdings": {k: split(existing.holdings[k]) for k in names},
        },
    })


def error_cases(prices):
    default_changes = [(0, "A,B"), (26, "A,C,MISSING"), (32, "A,B,C")]
    specs = [
        {"nan": [[40, 2]]},
        {"nan": [[26, 2]]},
        {"nan": [[24, 1]], "changes": [(0, "A,B"), (24, "A,C")]},
        {"changes": [(0, "A,B")], "coverage_end": "2021-02-15"},
        {"options": {"train_fraction": 1.0}}, {"options": {"rolling_window": 1}},
        {"options": {"rolling_window": 23}}, {"options": {"rebalance_every": 0}},
        {"options": {"max_weight": 0.0}}, {"options": {"max_weight": math.inf}},
        {"options": {"shrinkage": -1.0}}, {"options": {"cost_bps": 10000.0}},
        {"options": {"risk_free_rate": math.nan}}, {"options": {"periods_per_year": 0.0}},
        {"options": {"train_fraction": .02}}, {"options": {"train_fraction": .99}},
    ]
    cases = []
    for spec in specs:
        altered = prices.copy()
        for row, column in spec.get("nan", []):
            altered.iloc[row, column] = np.nan
        changes = spec.get("changes", default_changes)
        coverage_end = spec.get("coverage_end", str(prices.index[-1].date()))
        membership = load_membership_csv(StringIO(roster_csv(prices.index, changes)), coverage_end=coverage_end)
        options = dict(train_fraction=.5, rolling_window=5, rebalance_every=3, cost_bps=10.) | spec.get("options", {})
        try:
            run_historical_backtests(altered, membership, **options)
            raise AssertionError(f"{spec} did not fail")
        except ValueError as exc:
            cases.append({"nan": spec.get("nan", []), "changes": [list(change) for change in changes],
                          "coverage_end": coverage_end,
                          "options": {k: (None if isinstance(v, float) and math.isnan(v) else
                                          "inf" if v == math.inf else v) for k, v in options.items()},
                          "error": str(exc)})
    return cases


MEMBERSHIP_TEXTS = [
    "", "\n", "\n\n", "  \n", "date,tickers\n", "date,tickers", "date,symbol\n2020-01-01,A\n",
    "date,tickers,extra\n2020-01-01,A,x\n", 'date,tickers\n2020-01-01,"A,A"\n', 'date,tickers\n2020-01-01,"A, A"\n',
    'date,tickers\n2020-01-01,"A,"\n', "date,tickers\n2020-01-01,\n", 'date,tickers\n2020-01-01,"A,B"C\n',
    "date,tickers\n2020-01-01\n", "date,tickers\n2020-01-01,A\n2020-01-02\n", "date,tickers\n2020-01-01,A\n2020-01-01,B\n",
    "date,tickers\n2020-01-02,A\n2020-01-01,B\n", "date,tickers\n2020-01-01,BAD SYMBOL\n", "date,tickers\n2020-01-01,=A\n",
    "date,tickers\n2020-02-30,A\n", "date,tickers\n01/02/2020,A\n", "date,tickers\ngarbage,2020-01-01,A\n",
    "date,tickers\n2020-01-01,A\n2020-01-02,B,C\n", "date,tickers\n2020-01-01,A,B\n2020-01-02,B,C\n",
    "date,tickers\n2020-01-01,A,B\n2020-01-02,B\n", "date,tickers\nx\n2020-01-02,A,B\n",
    "\n\ndate,tickers\n\n2020-01-01,A\n\n", "date,tickers\n   \n2020-01-01,A\n", 'date,tickers\n""\n2020-01-01,A\n',
    "date,tickers\n,\n", "date,tickers\n2020-01-01 ,A\n", "date,tickers\n 2020-01-01,A\n",
    'date,tickers\n2020-01-01," A , B "\n', "date,tickers\r\n2020-01-01,A\r\n", "﻿date,tickers\n2020-01-01,A\n",
    '﻿"date",tickers\n2020-01-01,A\n', "﻿﻿date,tickers\n2020-01-01,A\n",
    '"date","tickers"\n"2020-01-01","A"\n', "date ,tickers\n2020-01-01,A\n", "Date,tickers\n2020-01-01,A\n",
    "date,date\n2020-01-01,A\n", 'date,tickers\n2020-01-01,"A\nB"\n', 'date,tickers\r\n2020-01-01,"A,\r\nB"\r\n',
    'date,tickers\n2020-01-01,"A\n', 'date,tickers\n2020-01-01,A"B\n', "date,tickers\n2020-01-01,a.b:c-d_e\n",
    "date,tickers\n2020-01-01,_A\n", "date,tickers\n2020-01-01,é\n", "date,tickers\n2020-01-01,NA\n",
    "date,tickers\n2020-01-01,nan\n", "date,tickers\nNaT,A\n", 'date,tickers\n2020-01-01,""\n',
    "date,tickers\n2020-01-01,A,\n", "date,tickers,\n2020-01-01,A,\n", "date,tickers\n2020-01-01,A\n,\n",
    "date,tickers\n2020-01-01,A\n,,\n", ",\n", "a\n", "date\n2020-01-01\n", 'date,tickers\n2020-01-01,"A""B"\n',
    "date,tickers\n2020-01-01,A\n\t\n2020-01-02,B\n", 'date,tickers\n2020-01-01,A\n2020-01-02,"B,C",\n',
    "date,tickers\n2020-01-01,A\n2020-02-01,B\n2020-03-01,C\n", "date,tickers\n2020-01-01,A\n2020-01-31,B\n",
    "date,tickers\n2020-01-01,A\n2020-02-01,B\n", "date,tickers\n2020-1-01,A\n", "date,tickers\n2020-01-01T00:00,A\n",
    "date,tickers\n2020-00-10,A\n", "date,tickers\n1999-12-31,\"Q, R ,S\"\n", "date,tickers\n2020-01-01,A\x00B\n",
    "date,tickers\n2020-01-01,\x1cA\x1f\n",
]

PRICE_TEXTS = [
    "", "\n", "﻿", "day,A\n2021-01-01,1\n", "Date,A,A\n2021-01-01,1,2\n", "Date,A\n2021-01-01,1,2\n",
    "Date\n2021-01-01\n", "Date,A\n2021-01-04,1\n2021-01-05,1\n",
]
PRICE_BODY = "2021-01-04,1,2\n2021-01-05,1,2\n2021-01-06,1,2\n2021-01-07,1,2\n2021-01-08,1,2\n"
PRICE_VARIANTS = [
    ("Date,A,a\n", PRICE_BODY), ("Date,A,\n", PRICE_BODY), ("Date, A ,B\n", PRICE_BODY), (" date ,A,B\n", PRICE_BODY),
    ("DATE,A,B\n", PRICE_BODY), ("Date,A,B\n", PRICE_BODY + "\n"), ("Date,A,B\n\n", PRICE_BODY),
    ("﻿﻿Date,A,B\n", PRICE_BODY), ('"Date","A","B"\n', PRICE_BODY), ("Date,=A,B\n", PRICE_BODY),
    ("Date,A,B\r\n", PRICE_BODY.replace("\n", "\r\n")), ("Date,A,B\n", PRICE_BODY.replace("-", "/")),
    ("Date,A\n", "".join(f"2021-01-0{i},\n" for i in range(4, 9))),
    ("Date,A,B\n", "".join(f"{i},1,2\n" for i in range(5))),
    ("Date,A,B\n", PRICE_BODY.replace("2021-01-04", "2021-01-09")),
    ("Date,A,B\x1c\n", PRICE_BODY),
]
PRICE_REPLACEMENTS = [
    ("2021-01-06,1,2", r) for r in [
        "", ",1,2", "2021-01-05,1,2", "garbage,1,2", "2021-01-06,,", "2021-01-06,-1,", "2021-01-06,abc,-1",
        "garbage,abc,-1", '2021-01-06,"1",2', '2021-01-06,"1"x,2', '2021-01-06,"1', "2021-01-06 00:00,1,2",
    ]
]
PRICE_CELLS = [
    "1", "1.5", " 1.5", "1.5 ", "+1.5", "-1", "0", "1e3", "1E3", "1e-3", ".5", "5.", "inf", "Infinity", "-inf",
    "+inf", "INF", "iNf", "infinity", "-Infinity", "nan", "NaN", "-nan", "NA", "N/A", "null", "None", "True", "1_000",
    "1,000", "0x1A", "1.5.2", "(1)", "1+2j", "١٢", "１２", "\t", " ", "  1  ", "1d", "1f", "$1",
    "1e", "e1", "-", "+", ".", "1e400", "1e-400", "00012", "1.0e+02", " 1", "1 ", "#N/A", "nan ", " nan",
    "Inf ", " inf", "\t1", "1\t", "\n1", "\r1", "\x0b1", "\x0c1", "\x1c1", "-0.0", "+.5", "1.e5", ".e5", "1e+",
    "1e5.5", "0x1p3", "+-1", "1E+0005", "infinit", " ", " ", "\x1c", "\x85", "1\x85", " \t ",
    "123456789012345678901234567890", "4.9e-324", "1 2", "0.1e1",
]


def outcome_membership(text):
    try:
        history = load_membership_csv(StringIO(text), "2020-01-31")
    except ValueError as exc:
        return {"error": str(exc)}
    return {"snapshots": [[str(d.date()), list(r)] for d, r in history.snapshots.items()]}


def outcome_prices(text):
    try:
        prices = load_historical_prices(StringIO(text))
    except ValueError as exc:
        return {"error": str(exc)}
    return {"prices": split(prices)}


def parsing_cases():
    membership = [{"text": text, **outcome_membership(text)} for text in MEMBERSHIP_TEXTS]
    texts = list(PRICE_TEXTS)
    texts += [header + body for header, body in PRICE_VARIANTS]
    texts += ["Date,A,B\n" + PRICE_BODY.replace(old, new) for old, new in PRICE_REPLACEMENTS]
    texts += ['Date,A\n2021-01-04,"' + cell + '"\n2021-01-05,1\n2021-01-06,1\n2021-01-07,1\n2021-01-08,1\n'
              for cell in PRICE_CELLS]
    prices = [{"text": text, **outcome_prices(text)} for text in texts]
    history = load_membership_csv(StringIO('date,tickers\n2020-01-01,"B,A"\n2020-01-06,"A,C"\n2020-01-10,"B,C"\n'),
                                  "2020-01-12")
    lookups = []
    for date in ["2019-12-31", "2020-01-01", "2020-01-05", "2020-01-06", "2020-01-09", "2020-01-10", "2020-01-12",
                 "2020-01-13", "2020-01-01T12:00:00", "01/02/2020", "2020-02-30", "NaT", "", " 2020-01-01"]:
        try:
            lookups.append({"date": date, "roster": list(history.at(date))})
        except ValueError as exc:
            lookups.append({"date": date, "error": str(exc)})
    coverage_ends = []
    for coverage_end in ["2019-12-31", "2020-01-09", "2020-01-10", "2020-02-30", "", "2020-01-10 ", "garbage"]:
        try:
            result = load_membership_csv(StringIO('date,tickers\n2020-01-01,"B,A"\n2020-01-10,"B,C"\n'), coverage_end)
            coverage_ends.append({"coverage_end": coverage_end, "metadata": value(result.metadata)})
        except ValueError as exc:
            coverage_ends.append({"coverage_end": coverage_end, "error": str(exc)})
    write("historical_parsing", {"membership": membership, "prices": prices, "lookups": lookups,
                                 "coverage_ends": coverage_ends})


def bundled_case():
    history = bundled_membership()
    queries = [("2010-01-01", "2016-01-01"), ("1990-01-01", "2030-01-01"), ("2026-09-30", None)]
    for rule in history.metadata["identity_exclusions"]:
        if rule["end"] is not None:
            end = pd.Timestamp(rule["end"])
            before = str((end - pd.Timedelta(days=1)).date())
            queries += [(before, None), (rule["end"], None), (before, rule["end"]), (rule["end"], "2026-09-30")]
    exclusions = [{"start": start, "through": through,
                   "result": list(history.identity_exclusions(start, through=through).items())}
                  for start, through in queries]
    write("historical_membership", {
        "coverage_start": str(history.coverage_start.date()), "coverage_end": str(history.coverage_end.date()),
        "source": history.source, "limitations": list(history.limitations), "symbols": list(history.symbols),
        "metadata": value(history.metadata), "metadata_keys": list(history.metadata),
        "snapshots": [[str(d.date()), list(r)] for d, r in history.snapshots.items()],
        "exclusions": exclusions,
    })


def main():
    prices, membership = historical_demo()
    study_case("historical_demo", prices, membership, {"kind": "demo"})
    study_case("historical_bundled", bundled_prices(), bundled_membership(), {"kind": "bundled"},
               train_fraction=.25, rolling_window=63, rebalance_every=42, max_weight=.25, cost_bps=15.)
    weekly = custom_prices()
    changes = [(0, "A,B,C"), (130, "A"), (150, "A,B,D,MISSING"), (175, "B,C,D,E")]
    text = roster_csv(weekly.index, changes)
    coverage_end = str(weekly.index[-1].date())
    source = "Synthetic weekly membership"
    custom = load_membership_csv(StringIO(text), coverage_end=coverage_end, source=source)
    study_case("historical_custom", weekly, custom, {"kind": "csv", "csv": text, "coverage_end": coverage_end,
                                                      "source": source},
               train_fraction=.6, rolling_window=None, rebalance_every=4, max_weight=.5, shrinkage=.3, cost_bps=0.,
               periods_per_year=52., risk_free_rate=.05)
    test = test_prices()
    test.to_csv(HERE / "historical_test_prices.csv", lineterminator="\n")
    fixed_case(test)
    write("historical_errors", {"cases": error_cases(test)})
    parsing_cases()
    bundled_case()


if __name__ == "__main__":
    main()
