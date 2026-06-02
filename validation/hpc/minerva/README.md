# Minerva benchmark launchers

These LSF files are thin launchers for reproducible RustScenic scaling runs on
Minerva. They assume the repo is checked out at:

```text
/sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/repo
```

and the Python environment is:

```text
/sc/arion/work/kahrae01/rustscenic/envs/rustscenic-v047
```

Submit from the repo root on Minerva:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}"
python validation/hpc/minerva/prepare_real_pbmc3k_data.py
python validation/hpc/minerva/preflight_minerva.py \
  --require-clean \
  --require-repo-import \
  --require-thread-pins
bsub < validation/hpc/minerva/run_real_pbmc3k_full_pipeline.lsf
bsub < validation/hpc/minerva/run_real_pbmc3k_full_pipeline_scaling.lsf
bsub < validation/hpc/minerva/run_real_pbmc3k_grn_scaling.lsf
```

The full-pipeline job writes one JSON artefact under
`/sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/results/real_pbmc3k_full_pipeline/`.
The full-pipeline scaling job writes one aggregate JSON plus one validated
child JSON per cell count under
`/sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/results/real_pbmc3k_full_pipeline_scaling/`.
The GRN scaling job writes one JSON artefact under
`/sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/results/`.

Each launcher runs the preflight first and writes a `.preflight.json` file next
to the benchmark result. Before preflight, the launcher runs
`prepare_real_pbmc3k_data.py` to download any missing 10x PBMC3k inputs and
verify their SHA-256 hashes. The preflight records `rustscenic.__file__` and
the compiled extension path, and the launchers pass `--require-repo-import` so
jobs fail before benchmarking a stale installed package. It also checks that
the compiled extension exposes the Rust kernels required by the pipeline.

The benchmark artefact records the repo commit, tracked source-clean state,
tracked-diff SHA-256 fingerprint, RustScenic version, backend capabilities,
dataset hashes, input shape, parameters, stage timings, per-stage peak RSS,
output counts and LSF environment. Use those JSON files as the source of truth
for benchmark docs. Public benchmark claims should use runs where
`repo_state.tracked_source_count` and `repo_state.untracked_source_count` are
both `0`. The LSF launchers pass `--require-clean`, so publication-grade jobs
fail immediately if source files differ from HEAD or untracked source files are
present. The launchers also validate completed result JSON before printing
`RESULT_JSON=...`, then print a compact collector table for the just-finished
artefact, so incomplete, stale-backend or dirty-source artefacts fail the job.

Re-run validation manually before copying numbers into docs:

```bash
python validation/hpc/minerva/validate_benchmark_artifact.py \
  /sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/results/*.json
```

Collect the latest validated benchmark rows after several LSF jobs:

```bash
python validation/hpc/minerva/collect_benchmark_results.py \
  --latest-per-benchmark \
  --check-output-files \
  /sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic/results/
```
