# Contributing

Project code and documentation use the [MIT License](LICENSE). Contribute only material that you have permission to share under those terms. Preserve third-party notices.

## Development setup

Use Python 3.14 and a branch from `main`:

```bash
git switch main
git pull --ff-only
git switch -c your-change
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

On Windows, create the environment with `py -3.14 -m venv .venv`. Use `.venv\Scripts\python.exe` for subsequent Python commands.

To start the app without an automatic browser launch:

```bash
bash run.sh --server.headless=true
```

## Checks

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m efficient_frontier --target-volatility .20 --max-weight .20 --output results/check
```

The final command runs the app's model with synthetic data and exports an offline report. The CLI without `--target-volatility` runs the separate legacy analysis.

For calculation changes, add a case with a known mathematical answer or a regression that demonstrates the error. Keep network calls mocked in tests. Verify that later prices cannot change earlier allocations. Document financial assumptions in the [method reference](docs/BACKTESTING.md).

## Pull requests

Keep changes focused and open pull requests against `main`. Describe the problem, resulting behavior, and checks. CI checks dependencies, runs the tests, and exports synthetic reports for the app model and retained legacy tools.

Use synthetic data in examples and bug reports. Keep credentials, personal holdings, downloaded datasets, and generated reports out of commits. Use ignored `data/` or `private/` folders for local inputs.
