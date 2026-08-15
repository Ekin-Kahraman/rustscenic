# HPC operation

RustScenic computation is scheduler-neutral. Allocate one host, set one Rayon
pool for Rust work, and pin every BLAS/OpenMP pool to one thread. LSF launchers
under `validation/hpc/minerva/` provide Minerva-specific wrapping and evidence
collection; the Python and Rust APIs do not depend on LSF or personal paths.

## Thread contract

```bash
export RAYON_NUM_THREADS="${LSB_DJOB_NUMPROC:-8}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
```

Request `span[hosts=1]`. RustScenic is a shared-memory process, not MPI, so a
multi-host allocation wastes hosts and does not increase throughput. In GRN
thread-scaling jobs, child points may use fewer Rayon threads than the job's
allocated ceiling. Nested BLAS/OpenMP pools must remain at one.

Rebuild immediately before a preflight when testing local source. Package and
extension version strings alone cannot distinguish a stale same-version shared
library:

```bash
python -m maturin develop --release
python -m rustscenic doctor --pretty
```

## Memory model

- Sparse RNA and ATAC matrices stay sparse across the production kernels.
- GRN memory is bounded by the binned expression matrix and per-Rayon-worker
  scratch buffers. More Rayon threads increase scratch memory.
- Correlation dichotomisation operates on requested TF-target pairs without a
  dense edge-by-cell matrix. Sparse CSC inputs use column intersections.
- Motif ranking parquet/feather inputs are projected to the required features;
  do not eagerly load atlas-scale ranking databases.
- Fragment preprocessing should receive cell-called barcodes. Carrying raw
  observed/empty droplets can turn a nominal 10k experiment into hundreds of
  thousands of matrix rows.
- The integrated `rna_with_regulons.h5ad` write can require another materialised
  output copy. `SKIP_INTEGRATED_ADATA=1` is appropriate for compute profiling,
  but not for a complete end-to-end production artefact.

Start with measured pilot memory and leave headroom for the input matrices,
external ranking projection and final writes. The audited 44,222-cell cortex
report used 16.8 GB peak RSS, but its input shapes and stages are not a universal
memory formula.

## Scientific configuration

- `early_stop_mode="arboreto"` is the default. It uses the strict mean of the
  exact trailing OOB-improvement window and retains the stopping tree.
- `early_stop_mode="legacy_inbag"` preserves the historical RustScenic
  two-point in-bag rule. Use it only for an explicitly labelled legacy rerun.
- `early_stop_window=0` disables stopping. This can fit all 5,000 trees for
  every target and materially increase runtime.
- `subsample=1.0` leaves no OOB rows, so `arboreto` mode also fits to the
  estimator ceiling. Keep the arboreto default `subsample=0.9` unless that is
  an intentional full-sample experiment.
- `grn_regulon_polarities="both"` is the production pipeline default. It emits
  separate `<TF>_activator` and `<TF>_repressor` programmes using Pearson
  `rho_threshold=0.03`; neutral edges are excluded.
- `"activating"` keeps only activators. `"unsigned"` is the documented legacy
  migration path and deliberately skips dichotomisation.

The Minerva launchers default to `GRN_N_ESTIMATORS=100` for smoke/scaling
turnaround. Those runs must not be presented as the 5,000-estimator production
configuration. Use the following exact production-validation sequence after
the branch has been committed on Minerva, because `--require-clean` correctly
rejects an uncommitted validation tree:

```bash
cd /sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/repo
. /sc/arion/work/kahrae01/rustscenic/envs/rustscenic-v047/bin/activate
python -m maturin develop --release
export GRN_N_ESTIMATORS=5000
export SKIP_INTEGRATED_ADATA=0
bsub < validation/hpc/minerva/run_real_pbmc3k_full_pipeline.lsf
bsub < validation/hpc/minerva/run_real_pbmc3k_grn_scaling.lsf
```

For a cheaper harness exercise, leave `GRN_N_ESTIMATORS` unset. Every generated
benchmark JSON declares `path_policy="portable"`; repo paths are relative and
external inputs are represented by basenames plus hashes, so archived artefacts
contain no workstation or cluster absolute paths.
