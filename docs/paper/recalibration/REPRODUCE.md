# Reproducing the evaluation repair

This directory is an analysis workspace. It does not refit RustScenic, change a
production recipe, send a message, or publish a result. The original package and
Huangfu's return remain unchanged. The corrected analysis is post hoc because
the original outcomes were already inspected.

## Inputs and software

- Frozen package: `../RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm`.
- Returned networks: `../RustScenic-Biology-Confirm-Results-Audit-20261112/returned`.
- Raw inputs: the package's sibling `source-cache`, with source hashes in
  `inputs/*/metadata.json`. `prepare_counts.py` verifies source counts, barcodes,
  guide assignments, captures, training/evaluation separation and the exact
  original aggregate counts before exporting the 5,000-gene panels.
- R 4.5.2; Matrix 1.7-4; data.table 1.18.0; jsonlite 2.0.0; SCEPTRE 0.99.1 at
  commit `20c33349c2c5a0df66589e37f67238e67c53bcb0`. The unmodified source is in
  `sceptre-source`, and the isolated installation is in `r-library`. Full R
  session information is in `results/*/R-session.txt`.
  `R-package-versions.csv` records transitive runtime and build dependencies.
- Analysis Python: the existing `../RustScenic-Minerva-Final-20261111/qa-reference-env`
  environment (NumPy, pandas, scipy, scikit-learn, anndata and parquet support).
- Plotting Python: the existing `../RustScenic-Validated-Workflow-20261108/venv`
  environment, which includes matplotlib.
  The two `Python-*-versions.csv` files record the installed package versions.

Scripts locate both preserved packages under the parent of this directory.
When relocating them, set `RUSTSCENIC_EVIDENCE_ROOT` to their common parent;
do not mix a different package or return. `analysis_paths.py` holds this single
path setting. Prediction/input receipts are checked on every scoring run. The
original archive SHA-256 hashes are recorded in the final report.

The copy under the paper's `recalibration/` directory is a compact evidence set,
not a standalone raw-data archive. For that copy, set `RUSTSCENIC_EVIDENCE_ROOT`
to the original archive parent and adjust the interpreter paths below. Restore
or regenerate the omitted files listed in `SOURCE-MANIFEST.json` before running
the full verification. SCEPTRE must be installed from the pinned source into
`r-library` on the target platform, with the recorded dependencies; the macOS
binary library is not a portable HPC installation.

## Run order

Run from this directory. The modelling stages take hours on a small laptop.
Existing successful `discovery.csv` files can be reused for scoring; neither
raw-count preparation nor model testing needs repeating to rebuild summaries.

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
ANALYSIS_PY=../RustScenic-Minerva-Final-20261111/qa-reference-env/bin/python
PLOT_PY=../RustScenic-Validated-Workflow-20261108/venv/bin/python

"$ANALYSIS_PY" prepare_counts.py
Rscript recalibrate.R k562 calibration
Rscript recalibrate.R rpe1 calibration
Rscript validate_distinct.R k562
Rscript validate_distinct.R rpe1
# Proceed only when both CALIBRATION-STATUS.json files say passed.
Rscript recalibrate.R k562 discovery
Rscript recalibrate.R rpe1 discovery
"$ANALYSIS_PY" control_diagnostics.py
"$ANALYSIS_PY" -m unittest -v test_corrected_evaluation.py
"$ANALYSIS_PY" score_saved_networks.py
Rscript verify_bh.R
"$ANALYSIS_PY" verify_recalibration.py
"$ANALYSIS_PY" make_report.py
"$PLOT_PY" plot_calibration.py
"$PLOT_PY" plot_results.py
```

The two cohorts can run concurrently, with two SCEPTRE workers per cohort.
Limiting OpenMP/BLAS to one thread per worker avoids a fork/thread-pool deadlock.
Do not alter resampling effort or model settings in order to accelerate a rerun.

## Analysis contract

`RECALIBRATION-PLAN.md` was hashed before new calibration or network comparisons.
`CALIBRATION-AMENDMENT.md` explains the upstream deterministic sampler: the two
initial median-size runs are identical and are counted once. The distinct
validation adds 10,000 previously uncounted pairs per cohort. All negative-control
tests share cells and constructs; they are not biological replicates.

The primary response rule is cohort-wide BH q <= 0.05 and absolute estimated
log2 fold change >= 0.5. Pairwise QC requires seven nonzero cells in each group.
Missing/failed pairs are unavailable. Self-targets are excluded from network
ranking and assessed separately. Non-significant tested pairs are benchmark
nonresponders, not established biological negatives. Responses may be indirect.

Read the final report alongside `summary/label-coverage.csv`,
`summary/coverage-per-TF.csv`, `summary/positive-control-coverage.csv` and the
full threshold/subgroup tables. Do not substitute emitted-edge precision for
all-regulator accuracy, or treat a conditional TF-bootstrap interval as donor
replication or a prospective equivalence test.

The original activity diagnostic was already replayed from saved AUCell group
means. `verify_recalibration.py` checks those saved activity rows against that
replay. This is not a new per-cell AUCell fit or a new biological endpoint.

## Preservation

Keep the raw return, exact sent package, source provenance, failed adapter logs,
calibration duplicate, amendment, scripts, session information and final tables.
The large `object.rds` files are local resume aids, not essential published
results; the raw-input adapter and pinned method can regenerate them. A durable
public deposit and coauthor review remain separate, explicitly authorised steps.
