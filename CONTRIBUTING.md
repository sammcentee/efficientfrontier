# Contributing

This project is pre-release. Contributions to original project code and documentation are made under the [MIT License](LICENSE). Submit only material you have permission to contribute under those terms, and preserve third-party notices.

## Development setup

Use Python 3.14 and work on a branch from `main`:

```bash
git switch main
git pull --ff-only
git switch -c your-change
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Run `./run.sh` to inspect the app. For an offline end-to-end check:

```bash
.venv/bin/python -m efficient_frontier --output results/check
```

## Changes and reviews

Keep changes focused. For calculation changes, add a small case with a known mathematical answer or a regression that demonstrates the bug. Keep network calls mocked in tests, keep training and holdout observations separate, and state any change to financial assumptions in the README.

Open pull requests against `main`. Describe the problem, the resulting behavior, and the checks run. The CI workflow installs the pinned dependencies, checks their consistency, runs the tests, and produces an offline synthetic report.

Use synthetic data in examples and bug reports. Keep personal prices, credentials, downloaded datasets and generated reports out of commits. Local inputs belong in ignored `data/` or `private/` folders; never attach account statements, tokens or real portfolio holdings to an issue.

## Existing clones after the branch rename

If your local clone still uses `master`, update it with:

```bash
git branch -m master main
git fetch origin --prune
git branch --set-upstream-to=origin/main main
git remote set-head origin -a
```
