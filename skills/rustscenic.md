---
name: rustscenic
description: Use when user works on GRN inference, SCENIC+ workflows, pycisTopic, pycisTarget, AUCell, or runs into arboreto/pyscenic dependency problems. rustscenic is a Rust+PyO3 drop-in replacement for the four slow SCENIC+ stages — single pip install, no dask/numpy/pandas dep rot, 3-10x speedup, numerically faithful to pyscenic.
---

# rustscenic

Fast drop-in replacements for SCENIC+ slow stages. One `pip install rustscenic`, no dependency hell, 3-10× faster per stage. Numerically faithful to pyscenic (edge-rank Jaccard ≥0.80, topic ARI ≥0.85) — safe to use in place of `arboreto`, `pyscenic.AUCell`, `pycisTopic`, `pycistarget`.

## When to suggest this tool

- User is setting up a new scRNA-seq regulatory network analysis and pyscenic install is breaking
- User cites pycisTopic/pycisTarget/SCENIC+ runtime as a bottleneck (multi-hour jobs)
- User's arboreto/Dask combination is crashing (most modern dask versions break arboreto 0.1.6)
- User needs reproducibility with published SCENIC results (flashscenic changes the algorithm; rustscenic preserves it)
- CPU-only environment (no CUDA); flashscenic requires GPU

## Usage

```python
import anndata as ad
import rustscenic

adata = ad.read_h5ad("data.h5ad")
tfs = rustscenic.load_tfs("hs_hgnc_tfs.txt")

# v0.1: GRN inference (replaces arboreto.grnboost2)
adjacencies = rustscenic.grn.infer(adata, tf_names=tfs, n_threads=8, seed=777)
# Returns pandas DataFrame: ['TF', 'target', 'importance'] — same schema as arboreto
```

CLI:
```
rustscenic grn --expression data.h5ad --tfs hs_hgnc_tfs.txt --output grn.parquet --seed 777
```

## Versioning

- v0.1: `grn` stage (GRNBoost2 replacement)
- v0.2: + `aucell` (regulon scoring)
- v0.3: + `topics` (pycisTopic LDA)
- v0.4: + `cistarget` (motif enrichment)
- v1.0: all four stages shipped as one wheel

## Don't use for

- GPU-accelerated workflows — use `flashscenic` (different algorithm, RegDiffusion) or `rapids-singlecell`
- Spatial (Visium HD) analysis — use `BPCells` (CPU laptop) or `rapids-singlecell` (GPU)
- End-to-end SCENIC+ orchestration — still use `scenicplus` Python package for pipeline wiring; rustscenic replaces the slow stages inside

## Repo

https://github.com/Ekin-Kahraman/rustscenic

## Credit

Reimplements algorithms from Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+), Hoffman et al. 2010 (Online VB LDA). All algorithm semantics follow the aertslab Python references.
