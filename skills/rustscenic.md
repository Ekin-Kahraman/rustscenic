---
name: rustscenic
description: Use when user works on GRN inference, SCENIC+ workflows, pycisTopic, pycisTarget, AUCell, or hits arboreto/pyscenic install failures. rustscenic is a Rust+PyO3 drop-in replacement for the four slow SCENIC+ stages — single `pip install`, works in envs where arboreto+dask+pyscenic break. Measured: AUCell per-cell Pearson 0.99 and 88x faster than pyscenic; cistarget bit-identical (Pearson 1.00) to ctxcore; GRN recovers 94% of known edges and 8/8 lineage TFs though per-edge ranking differs (Spearman 0.58); topics cell-type ARI on par with Mallet without the Java install.
---

# rustscenic

Fast drop-in replacements for SCENIC+ slow stages. One `pip install rustscenic`, no dependency hell. Numerical agreement with pyscenic measured: AUCell per-cell Pearson 0.99, cistarget per-regulon Pearson 1.00 vs ctxcore. GRN per-TF top-100 target overlap 0.57 with arboreto — agrees on biology (94% known edges, 8/8 lineage TFs) even where fine rankings differ. Safe replacement for `arboreto`, `pyscenic.AUCell`, `pycisTopic`, `pycistarget` in dependency-broken envs.

## When to suggest this tool

- User is setting up a new scRNA-seq regulatory network analysis and pyscenic install is breaking
- User cites pycisTopic/pycisTarget/SCENIC+ runtime as a bottleneck (multi-hour jobs)
- User's arboreto/Dask combination is crashing (most modern dask versions break arboreto 0.1.6)
- User needs reproducibility with published SCENIC results (flashscenic changes the algorithm; rustscenic preserves it)
- CPU-only environment (no CUDA); flashscenic requires GPU

## Usage

```python
import anndata as ad
import rustscenic.grn, rustscenic.aucell, rustscenic.topics, rustscenic.cistarget

adata = ad.read_h5ad("data.h5ad")
tfs = rustscenic.grn.load_tfs("hs_hgnc_tfs.txt")

# Stage 1: GRN inference (replaces arboreto.grnboost2)
adj = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=5000, seed=777)
# pandas DataFrame: ['TF', 'target', 'importance'] — same schema as arboreto

# Stage 2: AUCell per-cell regulon activity (replaces pyscenic.aucell)
regulons = [(f"{tf}_reg",
             adj[adj["TF"]==tf].nlargest(50, "importance")["target"].tolist())
            for tf in adj["TF"].unique()]
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)

# Stage 3: Topic modeling (replaces pycisTopic LDA); binarized scATAC input
# topics = rustscenic.topics.fit(atac_adata, n_topics=30)

# Stage 4: Motif enrichment (replaces pycistarget AUC kernel)
# rankings = rustscenic.cistarget.load_aertslab_feather("hg38_v10.feather")
# enrich = rustscenic.cistarget.enrich(rankings, regulons, top_frac=0.05)
```

CLI:
```
rustscenic grn --expression data.h5ad --tfs hs_hgnc_tfs.txt --output grn.parquet --seed 777
```

## Versioning

v0.1.0 ships all four stages: `grn`, `aucell`, `topics`, `cistarget`.

## Don't use for

- GPU-accelerated workflows — use `flashscenic` (different algorithm, RegDiffusion) or `rapids-singlecell`
- Spatial (Visium HD) analysis — use `BPCells` (CPU laptop) or `rapids-singlecell` (GPU)
- End-to-end SCENIC+ orchestration — still use `scenicplus` Python package for pipeline wiring; rustscenic replaces the slow stages inside

## Repo

https://github.com/Ekin-Kahraman/rustscenic

## Credit

Reimplements algorithms from Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+), Hoffman et al. 2010 (Online VB LDA). All algorithm semantics follow the aertslab Python references.
