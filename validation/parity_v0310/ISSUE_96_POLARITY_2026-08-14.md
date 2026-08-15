# Issue #96: activator/repressor validation

Date: 2026-08-14. This single local validation ran on an Apple M5 with macOS
26.5, Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3 and a release
extension rebuilt from the uncommitted review set based on HEAD `98e52c7`.
The corrected PBMC3k GRN contained 974,784 edges across 2,700 cells, 13,714
genes and 1,274 TFs. Its SHA-256 was
`87570e0956c9244ded26f995de4c5a82bd8246c86958bebaad8b3c4badb45ddd`;
the H5AD SHA-256 was
`6b049eced92c4bf5c54328d1f55e7a93b74f8658873c057b02ea0b675b62d42a`.
Expression was sparse and all BLAS/OpenMP pools were pinned to one; Rayon used
10 threads for the upstream GRN fit.

`rustscenic.grn.add_correlation(..., rho_threshold=0.03,
mask_dropouts=False)` completed in 0.268 seconds. The deterministic split was:

| Polarity | Edges |
| --- | ---: |
| Activating (`rho > 0.03`) | 468,913 |
| Repressing (`rho < -0.03`) | 24,166 |
| Neutral/indeterminate | 481,705 |

Building top-50, minimum-10 regulons took 0.140 seconds and produced 1,586
signed programmes: 1,274 activator and 312 repressor regulons. Neutral edges
were excluded. Names are `<TF>_activator` and `<TF>_repressor` and remain
distinct through cisTarget, peak projection, eRegulon assembly and AUCell.

For a 2,000-edge subset selected without replacement by
`numpy.random.default_rng(9600)`, correlations were recomputed with
`scipy.stats.pearsonr` from the same expression columns. All 2,000 polarity
classifications matched and maximum absolute rho difference was `5.05e-08`.
The saved sample-index array SHA-256 was
`60174901aab103a11f9d3467d9cc7243ceabd16c6b2204f7564e93325ceb6313`;
the temporary signed parquet SHA-256 was
`e534f8fa060e8277a664fb09ff944fd62a75a9392231e8f61bf87c50edaccea4`.
Rust unit tests additionally cover exact positive, negative,
constant, dropout-masked, explicit-sparse-zero and dense/sparse parity cases.

The pipeline defaults to both polarities. `activating` is available for
positive-only workflows. `unsigned` is the explicit legacy compatibility mode;
there is no silent unsigned fallback.

Focused end-to-end regression commands:

```bash
cargo test -p rustscenic-grn correlation
python -m pytest -q \
  tests/test_grn.py \
  tests/test_eregulon.py \
  tests/test_cli.py::test_add_cor_cli_emits_signed_adjacencies \
  tests/test_pipeline_integration.py::test_attribute_peaks_keeps_signed_regulon_programmes_distinct
```

The numerical cross-check loads the two hashed inputs above, calls
`add_correlation`, samples `rng.choice(len(signed), size=2000, replace=False)`
with seed 9600, and compares each selected row to `scipy.stats.pearsonr` using
the same TF and target expression columns. Generated signed parquet, index
array and JSON remained under `/tmp` and were not added to Git.
