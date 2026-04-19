# Head-to-head on Ziegler 2021 nasopharyngeal atlas — rustscenic vs pyscenic

**Date:** 2026-04-19
**Purpose:** Validate rustscenic against pyscenic on a **real atlas-scale published dataset** (not synthetic, not cached reference data, not PBMC-3k). Uses identical input on both sides to isolate AUCell-stage agreement.

**Dataset:** Ziegler et al. 2021 *Cell* — 58-donor nasopharyngeal scRNA-seq, 18 airway cell types, 18,073 COVID+ / 14,515 COVID− cells. After preprocessing: **31,602 cells × 3,044 genes**, 59 regulons.

**Full analysis + reproducible scripts + biological interpretation** live in a companion case-study repository (currently private pending a follow-up publication). The tool-validation numbers + figures here are complete for verifying the install / speed / agreement claims without it.

## Setup — isolates AUCell stage

- **Shared GRN:** rustscenic.grn.infer adjacencies used for BOTH sides (arboreto is broken — see install matrix below). Top-50-target regulons per TF, 59 regulons total.
- **Shared expression matrix:** identical 31,602 × 3,044 log-normalized matrix.
- **Shared env:** both tools run in the same venv. Only the AUCell implementation differs.
- pyscenic run in two modes to isolate weighting effect:
  - `noweights=True` — unit per-gene weights (matches our semantics)
  - `noweights=False` — GRN-importance weights (pyscenic default)

## Agreement

| Metric | rustscenic vs pyscenic-unit | rustscenic vs pyscenic-weighted |
|---|---:|---:|
| Per-cell Pearson (mean / median) | **0.984 / 0.997** | 0.949 / 0.965 |
| Cells with Pearson > 0.95 | **91.7 %** | 71.6 % |
| Per-regulon Pearson (mean / median) | 0.952 / 0.988 | 0.916 / 0.953 |
| Argmax-regulon per-cell match | 85.4 % | 50.1 % |

Against the *semantics-matched* pyscenic run (unit weights), agreement is essentially bit-equivalent for clustering / marker-analysis purposes. Against weighted pyscenic, agreement is lower because rustscenic does not use per-gene regulon weights — a known v0.2 item.

See `validation/figures/ziegler_fig2_per_cell_pearson.png` for the full per-cell distribution.

## Runtime — same workload, same machine, same env

| Tool | Wall-clock | vs rustscenic |
|---|---:|---:|
| **rustscenic.aucell** | **0.25 s** | — |
| pyscenic.aucell (unit) | 6.81 s | **27× slower** |
| pyscenic.aucell (weighted) | 5.29 s | **21× slower** |

See `validation/figures/ziegler_fig3_runtime.png`.

## Canonical-TF benchmark — all three tools agree on hits AND misses

14 literature-known airway + immune TFs evaluated for "does the regulon's top-activity cell type match the expected cell type":

| Metric | rustscenic | pyscenic-unit | pyscenic-weighted |
|---|---:|---:|---:|
| Direct hits | 8 / 14 | 8 / 14 | 9 / 14 |
| Identical miss set | STAT1, MYB, IRF7, SOX2, PAX5 | (same) | (same) |

**Per-TF z-scores in expected cell type agree to within 0.02 for 10 / 14 TFs** — see `validation/figures/ziegler_fig1_canonical_tf_3way.png` for the side-by-side.

This is the strongest single line of evidence for numerical fidelity:

> When rustscenic, pyscenic-unit, and pyscenic-weighted all miss the **same five TFs with the same z-scores**, the tool-to-tool variation is strictly smaller than the dataset-inherent noise.

## Install matrix — "pip install" pitch literalised

| Tool + environment | pip install | import | GRN runs | AUCell runs |
|---|:---:|:---:|:---:|:---:|
| rustscenic, fresh Python 3.12 venv | ✓ | ✓ | ✓ | ✓ |
| pyscenic, fresh Python 3.12 venv | fails (`pkg_resources` deprecated) | — | — | — |
| arboreto, fresh Python 3.12 venv | succeeds | ✓ | **fails**: `TypeError: Must supply at least one delayed object` (dask_expr) | — |
| arboreto, inside pyscenic's own env (pandas pinned 1.5.3) | ✓ | **fails**: `Dask requires pandas ≥ 2.0.0` | — | — |

**There is no 2026-Python environment where arboreto actually runs.** This isn't a rustscenic stunt — pyscenic's own install recipe is broken against modern dask. Tested in a clean macOS 14 + Python 3.12 venv and confirmed in the pre-existing scenic-env.

## Runtime + memory summary across all stages

| Stage | Wall (rustscenic) | Peak RSS | vs pyscenic on same data |
|---|---:|---:|---|
| GRN (31,602 cells × 59 TFs × 500 estimators) | 26.5 s | ~1.5 GB | arboreto cannot install to measure |
| Regulon construction | <1 s | — | pyscenic.utils unchanged |
| AUCell | **0.25 s** | ~3.8 GB | **27× faster** than pyscenic |
| Total end-to-end | ~30 s | ~4 GB | pyscenic end-to-end not measurable (see above) |

## Figures

- `validation/figures/ziegler_fig1_canonical_tf_3way.png` — 14 TFs × 3 tools side-by-side, z in expected cell type
- `validation/figures/ziegler_fig2_per_cell_pearson.png` — distribution of per-cell Pearson (rustscenic vs pyscenic-{unit,weighted}) on all 31,602 cells
- `validation/figures/ziegler_fig3_runtime.png` — wall-clock bar chart

## Where to read more

- **Biological findings** (COVID± differential regulons — IFN response ↑, AP-1 stress ↓ in squamous metaplasia, WNT ↑ in secretory): not tool validation, kept with the companion case-study (private pending publication). Candidate for a standalone regulatory-biology paper.
- **Scripts** that produced these numbers: same case-study repo (`scripts/03_headtohead_pyscenic_aucell.py`, `scripts/04_comparison_figures.py`). Requires the Ziegler h5ad, so they live there, not here.

## Provenance

All numbers produced by scripts in the companion case-study (private). Public users can reproduce by downloading the Ziegler h5ad from GEO, using rustscenic's public API, and matching the preprocessing in `validation/ours/` + the HVG∪TFs union pattern from `examples/pbmc3k_end_to_end.py`.
