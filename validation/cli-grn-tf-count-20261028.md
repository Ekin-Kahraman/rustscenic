# GRN CLI TF-count regression, 2026-10-28

Base source: `db4dc2bfc63b71346cc70445cab424111bb3239d`.

The CLI counted every matching TF-list line in `n_tfs_used`, although the
backend selects each matching TF column once. The fix counts distinct matches
for the JSON sidecar and CLI header. The original TF list still reaches
`grn.infer` unchanged. No inference, identifier-resolution or input behavior
changes are included.

The regression uses 64 synthetic cells and five genes. `TF_B` is constant and
emits no edges, while `TF_A` emits edges to `G1` and `G2`. Both TFs are matched
inputs, so the correct count is two rather than the one TF present in the
output. The repeated list also contains missing names, whitespace, comments
and a blank line.

| TF list | Unchanged source | Fixed source |
| --- | --- | --- |
| Two unique matching names | 2 | 2 |
| Five matching entries for those same names | 5 | 2 |

Both lists produce identical TSV bytes and complete `grn_fit` telemetry.
The two-edge TSV SHA256 is
`ef915b928ae140092478d4d90e89d0706c51b6d54b1760003dd1d36c928c0afe`.
The test also checks that the parsed TF list, including duplicates and missing
names, is forwarded unchanged to the real inference function.

Controls: seed 777, 32 trees, learning rate 0.1, `max_features=1`,
`subsample=1`, depth 2, early-stop window 0 and `RAYON_NUM_THREADS=2`.

The new test failed on unchanged source specifically at `5 == 2`, after its
network and fitting-telemetry comparisons passed. It passed after the one-line
fix. The complete CLI test file then passed all seven tests with no skips;
the existing synthetic CLI cases emitted 11 warnings.

Local execution used CPython 3.13.18, NumPy 2.5.4, pandas 3.1.0 and pytest 9.2.0
on an `osx-64` Conda runtime under pre-existing Rosetta. A scoped import
bootstrap loaded Python modules from the source worktree and the unchanged
published RustScenic 0.5.1 native extension. Its SHA256 was
`b953a7b641152eeff24fa5fb42f69fe66266b487a1ee94207e813ec48e063dee`.
Module paths and before/after content hashes were checked. This was not a
rebuild of the current source extension or an additional platform qualification.

In a source-built environment, the regression can be run with:

```sh
RAYON_NUM_THREADS=2 python -m pytest tests/test_cli.py -q \
  -k test_grn_cli_counts_unique_matching_tfs_without_changing_inference
RAYON_NUM_THREADS=2 python -m pytest tests/test_cli.py -q
```

This is an unreleased source correction. Published 0.5.1 packages retain the
old counter. The check establishes software bookkeeping and output preservation
for the synthetic case, not biological validity or broader identifier support.
