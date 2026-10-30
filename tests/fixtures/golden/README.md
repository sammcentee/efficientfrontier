# Golden results of the Python release

These files record inputs and outputs of the Python release at commit `6c6df6e`, the last commit before the Rust rewrite. The `tests/golden_*.rs` and similar parity tests compare the Rust results with them.

All prices are synthetic. The demo prices are the exact output of the Python release's seeded generator.

To regenerate a set, check out commit `6c6df6e`, install its `requirements.txt`, copy the `generate_*.py` script into `tests/fixtures/golden/` of that checkout, and run it with Python 3.14 from the checkout root. Copy the new files back into this folder.

The Rust tests allow small numeric differences. Both versions solve the same convex programs with Clarabel, so weights agree to the solver tolerance. Each test file states its tolerance and the reason for it.
