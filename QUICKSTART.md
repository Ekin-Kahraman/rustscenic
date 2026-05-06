# rustscenic quickstart

5-minute end-to-end: install, run all 4 SCENIC+ stages on PBMC-3k (scanpy-bundled).

## Install

```bash
pip install "rustscenic @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.4.0" anndata scanpy
```

(PyPI publication is being set up; until then, pip-install from the GitHub release tag.)

Python ≥ 3.10 required. No Java / no CUDA / no Dask. Works on macOS, Linux.

## One-shot script

```python
import scanpy as sc
import rustscenic.grn, rustscenic.aucell

# 1. Load + preprocess PBMC-3k (shipped with scanpy, zero download)
adata = sc.datasets.pbmc3k()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"PBMC-3k: {adata.shape}")  # (2700, 13714)

# 2. GRN inference — replaces arboreto.grnboost2
tfs = ["SPI1", "CEBPD", "MAFB", "CEBPB", "KLF4", "IRF8", "PAX5", "EBF1",
       "TCF7", "LEF1", "TBX21"]  # or load a full TF list
tfs = [t for t in tfs if t in adata.var_names]

grn = rustscenic.grn.infer(adata, tfs, seed=777, n_estimators=100)
print(f"GRN: {len(grn)} edges")
print(grn.nlargest(5, "importance").to_string(index=False))

# 3. Build regulons + run AUCell — replaces pyscenic.aucell (~88× faster on 10k cells)
regulons = [
    (f"{tf}_regulon", grn[grn["TF"] == tf].nlargest(50, "importance")["target"].tolist())
    for tf in tfs
]
regulons = [(name, genes) for name, genes in regulons if len(genes) >= 10]

auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
print(f"Regulon activity: {auc.shape} (cells x regulons)")
print(auc.iloc[:3, :3])
```

## Command-line equivalent

```bash
# Download aertslab TF list (once)
curl -o tfs_hg38.txt https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt

# Preprocess + save your AnnData as .h5ad first, then:
rustscenic grn \
  --expression data.h5ad \
  --tfs tfs_hg38.txt \
  --output grn.parquet \
  --seed 777

rustscenic aucell \
  --expression data.h5ad \
  --regulons grn.parquet \
  --output auc.parquet
```

## Running the full 4-stage SCENIC+ pipeline

For scATAC + RNA multiome analysis:

```python
import rustscenic.grn, rustscenic.aucell, rustscenic.topics, rustscenic.cistarget

# Stage 1: GRN on RNA
grn = rustscenic.grn.infer(rna_adata, tfs, seed=777)

# Stage 2: Regulon activity
regulons = build_regulons_from_adj(grn)  # top-N targets per TF
auc = rustscenic.aucell.score(rna_adata, regulons)

# Stage 3: Topic modeling on binarized ATAC peaks
topics_result = rustscenic.topics.fit(
    atac_adata,  # cells × peaks, binarized
    n_topics=50,
    n_passes=15,
    seed=777,
)

# Stage 4: Motif enrichment (provide aertslab ranking DB)
import rustscenic.cistarget
rankings = rustscenic.cistarget.load_aertslab_feather("hg38_screen_v10.feather")
enrichments = rustscenic.cistarget.enrich(rankings, regulons, top_frac=0.05)
```

## Speed on a 10-core laptop

| Dataset | Stage | Wall-clock |
|---|---|---|
| PBMC-3k (2700 cells × 13714 genes) | grn | 207s |
| PBMC-3k (1274 regulons) | aucell | 0.1s |
| 10x Multiome PBMC (2588 cells, all 4 stages) | full pipeline | 9.1 min |

Reference (arboreto + pyscenic + tomotopy) on 10x Multiome: 11.8 min. Reference with real pycisTopic-Mallet: 534 s for topics stage alone on 10k PBMC ATAC (measured).

## Common errors

- **"ValueError: expression has 0 peaks/genes"** — preprocess first (`sc.pp.filter_genes`).
- **"ValueError: N duplicate gene name(s) in expression matrix"** — run `adata.var_names_make_unique()`.
- **"UserWarning: all regulons dropped"** — your regulon gene symbols don't match `adata.var_names`; check species / naming (human uses uppercase, mouse uses TitleCase).

## Full pipeline on real 10x Multiome

```bash
# Download 10x Multiome PBMC 3k
curl -o fbm.h5 https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5

# See validation/validate_multiome_e2e.py for a 55-line reproducible example.
```

## Get help

- Full docs: README.md + CHANGELOG.md + [VALIDATION_SUMMARY](validation/VALIDATION_SUMMARY.md)
- Agent skill (Claude Code): auto-loads when you mention "SCENIC" or "arboreto"
- Issues: open one at https://github.com/Ekin-Kahraman/rustscenic/issues; see [docs/topic-collapse.md](docs/topic-collapse.md) for the one known algorithmic caveat (v0.2 roadmap item)
