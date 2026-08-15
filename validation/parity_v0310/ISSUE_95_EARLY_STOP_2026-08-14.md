# Issue #95: PBMC3k early-stop validation

Date: 2026-08-14. These are single-run local Apple M5 arm64 observations, not
Minerva results or publication-grade speed claims. The machine ran macOS 26.5,
Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, Rust 1.94.1 and a
release extension rebuilt from the uncommitted review set based on HEAD
`98e52c7`. Both RustScenic processes used 10 Rayon threads, one OpenMP/BLAS
thread, seed 777, a 5,000-estimator ceiling, learning rate 0.01, max-features
0.1, subsample 0.9 and depth 3. Cache state was not controlled and each mode
was measured once.

Inputs:

- `pbmc3k.h5ad`: 2,700 cells x 13,714 genes, SHA-256
  `6b049eced92c4bf5c54328d1f55e7a93b74f8658873c057b02ea0b675b62d42a`
  and MD5 `71b74599ccebc86b7de841fefee40b90`.
- `allTFs_hg38.txt`: 1,274 TFs present, SHA-256
  `3953034f84112c60d3d8ef15b0e0c8ac5fce0b40d2c7c0824c2945c70cee2523`.
- Committed arboreto output: SHA-256
  `33f501b42e008d9c0ecb23c8ef7b10c529c93019f2451429ac928a7545f2a968`.
  It was generated with pySCENIC 0.12.1 and arboreto 0.1.6. The reference
  Dockerfile now pins scikit-learn 1.7.2 because its OOB-improvement semantics
  affect stopping.

The committed full-parity JSON identifies the fixture with the current H5AD's
MD5, but the arboreto sidecar records a different H5AD SHA-256
(`2461c170...`). HDF5 serialisation can change file bytes without changing the
matrix, but the old reference does not contain a canonical matrix-content hash
to prove that explanation. This provenance inconsistency is retained as a
cluster/reference-container rerun requirement rather than silently resolved.

## Semantics reproduced

Arboreto evaluates the strict mean of exactly the last `window` entries from
scikit-learn's `oob_improvement_` and retains the tree that triggers the stop.
With modern scikit-learn, round zero is pre-fit minus post-fit loss on the first
OOB mask. Later entries are the previous round's post-fit OOB loss minus the
current round's post-fit OOB loss, so adjacent masks may differ. Thirty sampled
targets and the six issue exemplars reproduced the committed arboreto fitted
counts under scikit-learn 1.7.2. Historical scikit-learn used a same-mask
pre/post definition and does not reproduce the committed baseline.

RustScenic now exposes:

- `arboreto` (default): fixed-size in-bag samples and the modern cross-round OOB
  trailing-window monitor.
- `legacy_inbag`: the previous RustScenic Bernoulli sample and two-point in-bag
  MSE monitor.
- window zero: no early stop.

## Fitted trees and resources

| Run | Wall seconds | Peak RSS GB | Total trees | Mean / median / p95 / max | Edges |
| --- | ---: | ---: | ---: | --- | ---: |
| Current branch, `legacy_inbag` | 154.31 | 0.45 | 462,285 | 33.709 / 29 / 46 / 408 | 1,137,831 |
| Current branch, `arboreto` | 119.80 | 0.40 | 362,608 | 26.441 / 26 / 30 / 79 | 974,784 |
| Committed arboreto reference | 380.94 | not recorded | 363,178 | 26.482 / 26 / 30 / 59 | 949,452 |

In these two single observations, the correctness mode used 21.56% fewer
fitted trees, 22.36% less wall time, 11.1% less peak RSS and emitted 14.33%
fewer edges than the same build's legacy mode. Its total fitted-tree count is
570 below arboreto, a 0.16% difference; median and p95 match exactly.
Tree-builder implementations remain different, so exact edge or per-target
tree-count equality is not expected. Neither the local wall/RSS difference nor
the 380.94-second reference-stack time is used as a general speedup claim.

## GRN parity before and after

| Metric against committed arboreto | `legacy_inbag` | `arboreto` | Change |
| --- | ---: | ---: | ---: |
| Importance Spearman on fixed three-way common edges | 0.6146 | 0.6363 | +0.0217 |
| Mean within-TF Spearman on the same fixed edges | 0.6218 | 0.6325 | +0.0107 |
| Global top-1k edge Jaccard | 0.1621 | 0.2034 | +0.0413 |
| Global top-5k edge Jaccard | 0.1606 | 0.2093 | +0.0487 |
| Global top-10k edge Jaccard | 0.1964 | 0.2340 | +0.0376 |
| Global top-50k edge Jaccard | 0.3405 | 0.3583 | +0.0178 |
| Mean per-TF top-10 Jaccard | 0.2270 | 0.2302 | +0.0032 |
| Mean per-TF top-20 Jaccard | 0.2836 | 0.2915 | +0.0079 |
| Mean per-TF top-50 Jaccard | 0.3822 | 0.3855 | +0.0033 |

The controlled Spearman comparison uses the same 337,414 edges present in the
reference and both RustScenic modes. Pairwise shared-edge counts still differ
(480,236 legacy versus 444,444 corrected), so the earlier pairwise Spearman
values are not used to infer improvement. Fixed-universe rank and top-edge
agreement increased; no scientific check was relaxed.

Temporary-output SHA-256 values were: corrected parquet
`87570e0956c9244ded26f995de4c5a82bd8246c86958bebaad8b3c4badb45ddd`,
corrected metadata `ef0b8d885181e22e112c8894dc734a4c8ac46a05c41d51e4fd8f29c98a1c5aa3`,
legacy parquet `f119a3cbf9ed08c1258af564fe14092a4a4acdd880faaacc046d82f4003642b6`,
and legacy metadata `392334b60a46a6fd7c37bfbf7c623272b184fcb0292d2af6beaac1f167c22620`.

After the final audit changed fitted-tree telemetry collection from retained
per-target edge buffers to one atomic count per target, the release extension
was rebuilt and the full arboreto-mode run was repeated. It completed in
118.04 seconds at 0.46 GB reported peak RSS and reproduced the corrected
parquet SHA-256 above exactly, including 974,784 edges and 362,608 fitted
trees. Its metadata SHA-256 was
`e41c0a91f470c1d0562d793450acd1e7267b3f1fa0e6eb577c32c32b4b5ab6f8`.
The wall/RSS values are another single observation, not a performance claim;
the identical output hash is the regression proof used here.

Commands:

```bash
export RAYON_NUM_THREADS=10 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python validation/run_rustscenic_grn_pbmc3k.py \
  validation/parity_v0310/pbmc3k.h5ad \
  validation/parity_v0310/allTFs_hg38.txt \
  /tmp/grn.parquet /tmp/grn.meta.json \
  --early-stop-mode arboreto
python validation/grn_parity_v0310.py \
  /tmp/grn.parquet \
  validation/parity_v0310/pyscenic_grn_pbmc3k.parquet \
  /tmp/grn-parity.json
```

Temporary generated parquet/JSON outputs were not added to Git.
