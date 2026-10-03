# Public release checklist

This repository is being prepared for a later public release. Its default branch is `main`. This preparation does not publish a release, deploy the app, or change repository visibility. Project licensing remains undecided at the owner's request.

## Before changing visibility

- Decide whether to select a project license or publish the source with licensing still undecided. Keep the README accurate; do not call it open source without an appropriate license. Dependency licenses do not license this project's code.
- Confirm the provenance and publication rights of the original R script and `spy_holdings.ods`. The spreadsheet is the original user-supplied list of 60 symbols; it is not a documented historical index-membership dataset.
- Review Git authorship metadata. Existing commits contain the author's personal email; changing future Git settings does not remove it from earlier commits. No history rewrite has been performed.
- Review tracked files and history for credentials and private data. The preparation audit found no common credential patterns in the two existing commits; it is not a guarantee about later changes or every possible secret.
- Keep downloaded market prices and reports containing them out of public examples unless redistribution is permitted by the provider's terms. Use the deterministic synthetic demo for release screenshots and examples.
- Decide how you want to receive issue reports and private vulnerability reports when public access opens. Configure those GitHub features before advertising them as available.

## Validate the release candidate

Run from a fresh clone with Python 3.14:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python -m efficient_frontier --output results/release-check
./run.sh
```

Confirm that CI passes on the intended `main` commit, the app shows the synthetic-data label, and the generated report opens offline. Test a CSV upload and download, including a repeat analysis after changing settings. Review the financial assumptions and known covariance degeneracy in the README.

If Plotly is upgraded, check the bundled Plotly.js version and refresh its license notice in `efficient_frontier/third_party/plotly.js.LICENSE.txt`. Preserve notices in the standalone HTML and report ZIP.

## When the owner is ready

1. Make the repository public in GitHub settings after the checks above and the owner's explicit decision to publish.
2. Choose a release version, record its changes, and create the matching tag and GitHub release from the verified commit. No version tag or GitHub release is created by this checklist.
3. Check the public README, images, CI status and fresh-clone instructions without signing in.

GitHub visibility and a hosted application are separate decisions. The app currently binds to localhost; public hosting needs its own deployment configuration and a review of data uploads, downloads and access controls.
