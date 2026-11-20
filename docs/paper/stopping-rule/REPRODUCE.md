# Reproducing the stopping-rule analyses

The scripts in this folder are published as they were run between 12 and
16 November 2026 on a 10-core macOS arm64 laptop, except `make_figure.py`,
which was revised for print legibility on 20 November. They are a record of the
analysis, not a packaged pipeline, and they do not run unmodified from this
repository: they find code and data through fixed folder names under their
parent directory (`HERE.parent`), which was a local analysis workspace. In the
repository that parent is `docs/paper/`. To rerun them, recreate the layout
below (or edit the path constants at the top of each script) and supply the
external inputs.

The five scripts frozen at 21:01 UTC on 12 November (`stage_confirmation.py`,
`beeline_arms.py`, `beeline_score.py`, `beeline_regulons.py`,
`reference_paired.py`) and `CONFIRMATION-PROTOCOL.md` are byte-identical to the
hashes in `CONFIRMATION-FREEZE.json`.

## Frozen dependencies, now included

`frozen-dependencies/` holds the four unchanged dependencies listed in
`CONFIRMATION-FREEZE.json`, copied from the original workspace. Each file's
SHA-256 matches the freeze record.

| File | Original location in the workspace | SHA-256 |
| --- | --- | --- |
| `prepare_inputs.py` | `RustScenic-Biological-Validation-20261028/` | `06d93e071c773984de25b6b36cd2c14f4018b962c4959e386e796f86059d1197` |
| `reference_metrics.py` | `RustScenic-Paper-Validation-20261106/rustscenic-paper-validation/` | `321991931dd6ac962351765e98b309b31114fbfcdd9c13d7648c1a5c611cafac` |
| `grn_reference.py` | `RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm/` | `a87bde99218804d7ad0c453b7032ed42e9741d9ff51b5416ec4ffea559fc4c41` |
| `recipe.py` | `RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm/` | `07a29f8476a5900052a9338f6ae29c222944015cc5010b499c04aba5517f93e7` |
| `data_io.py` (imported by `stage_bmmc.py` and `stage_hetero.py`; not part of the freeze) | `RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm/` | `53a95a8a0da580f56f10e6f5a10892e64d9aa71d894e437900abb2dab6f6b624` |

Not included: `fit_variant.py` imports `fit.py` from the 12 November package,
which in turn imports that package's `common.py`, `monitoring.py`,
`rankings.py` and `reference.py`; `evaluate_fits.py`, `diagnose_labels.py` and
`stage_hetero.py` also read that package's `inputs/` folder. Those files are not
in this repository.

## Expected layout (as run)

```
<workspace>/
  stopping-rule/                                   this folder; writes inputs/, runs/, fits/, results/, diagnosis/
  RustScenic-Biological-Validation-20261028/
    prepare_inputs.py                              = frozen-dependencies/prepare_inputs.py
    source-cache/BEELINE-data.zip, BEELINE-Networks.zip
  RustScenic-Paper-Validation-20261106/rustscenic-paper-validation/
    reference_metrics.py                           = frozen-dependencies/reference_metrics.py
  RustScenic-Biology-Confirm-20261112/
    rustscenic-biology-confirm/                    12 November package: grn_reference.py, recipe.py, data_io.py
                                                   (included), fit.py and its modules, inputs/<dataset>/ (not included)
    source-cache/ReplogleWeissman2022_K562_essential.h5ad, ReplogleWeissman2022_rpe1.h5ad
  RustScenic-Biology-Confirm-Results-Audit-20261112/returned/jobs/   returned networks of that package
  RustScenic-Biology-Recalibration-20261112/       inputs/ and results/<dataset>/corrected-pair-results.parquet
                                                   (compact copy without per-pair results in ../recalibration/)
  RustScenic-Final-Scaling-20261028/
    rustscenic-final-scaling/inputs/motifs/{human,mouse}-gene-annotations.tbl
    source-cache/bmmc-multiome.h5ad
```

`beeline_regulons.py` and `stage_bmmc.py` also read cisTarget ranking databases
from RustScenic's default cache, `Path.home() / '.cache/rustscenic/cistarget'`:
`hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather`
and `mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather`
(the published cisTarget v10 databases).

## External data

| Input | Source | Used by |
| --- | --- | --- |
| BEELINE expression and reference networks | `BEELINE-data.zip` and `BEELINE-Networks.zip`, Zenodo record 19009603 (SHA-256 pinned in `prepare_inputs.py`) | `stage_confirmation.py`, BEELINE arms and scoring |
| Bone-marrow multiome (BMMC) | GEO GSE194122 | `stage_bmmc.py`, `evaluate_bmmc.py` |
| Replogle K562 and RPE1 Perturb-seq | `ReplogleWeissman2022_K562_essential.h5ad`, `ReplogleWeissman2022_rpe1.h5ad` (source hashes in `../recalibration/inputs/*/metadata.json`) | `stage_hetero.py`, `evaluate_fits.py`, `diagnose_labels.py` |
| Motif annotations and rankings | cisTarget v10 rankings (above); gene annotation tables from the final-scaling package | `beeline_regulons.py`, `stage_bmmc.py` |

## Software

- Python 3.11.8. Analysis ran in a local virtual environment whose reference
  packages match the HPC reference environment recorded in
  `../minerva-final/returned/jobs/*/environment.json` for numpy 1.23.5,
  pandas 1.5.3, scikit-learn 1.3.2, arboreto 0.1.6, pySCENIC 0.12.1,
  anndata 0.10.5.post1, dask and distributed 2023.4.1, ctxcore 0.2.0 and
  pyarrow 15.0.0 (scipy 1.11.4 and h5py 3.16.0 locally). No lock file was
  recorded for it.
- RustScenic arms used a second local virtual environment layered on that one
  (a `.pth` file adds the reference packages) with a local build of RustScenic
  v0.6.1 (commit `e59dc76`) plus the uncommitted `paired_oob` stopping mode
  (branch `fix/paired-oob-early-stop`). The frozen macOS arm64 wheel has SHA-256
  `c3e0a90c68b6da8e2366dee7dc712333f977958d899ec7754ebe7c8195850977`
  (`CONFIRMATION-FREEZE.json`, which records only the hash of the 183-line
  diff against v0.6.1). The patch shipped for the HPC replication,
  `../stopping-rule-minerva/package/resources/KERNEL-PATCH.diff` (299 lines),
  adds CLI and pipeline options, tests and a changelog entry; its `BUILD.json`
  records that the Rust sources and `grn.py` were unchanged since the frozen
  wheel. The stopping logic is the same in RustScenic 0.7.0, so 0.7.0 with
  `max_bins=255` and either `early_stop_mode` is expected to reproduce this
  kernel (inferred from the source, not rerun on these datasets).
- Real scikit-learn 1.2.2 arm: `beeline_arms.py` unmodified, with the PyPI
  scikit-learn 1.2.2 cp311 wheel first on `PYTHONPATH` (1.2.2 has no wheels for
  Python 3.12 or later), then `compare_sklearn122.py`.
- Run Python with `-I`. Fits used 10 threads; `runs/beeline-confirmation-RUNLOG.md`
  records thread caps set during the confirmation run.

## Order

See the table in [FINDINGS.md](FINDINGS.md), section 6: label diagnosis,
training-population refits, bone marrow, BEELINE arms, the scikit-learn 1.2.2
check, scoring, confirmation staging and the figure.
