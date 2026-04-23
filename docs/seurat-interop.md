# Using rustscenic from a Seurat / R workflow

rustscenic is Python-first, but the scRNA / scATAC community splits
roughly evenly between scanpy / AnnData and Seurat / SingleCellExperiment.
This page documents two bridges that let Seurat users run rustscenic
without re-implementing analyses in Python.

The key claim: **rustscenic's algorithms don't depend on scanpy** —
they operate on expression matrices, gene symbol lists, and regulon
gene sets. Any object that can be converted to those primitives works.

## Option A — Seurat → AnnData via `anndata2ri` (recommended)

`anndata2ri` is a battle-tested R-Python bridge specifically for the
scverse ecosystem. It converts Seurat objects to AnnData bidirectionally
and is what `scverse/anndataR` and most hybrid pipelines already use.

In R:

```r
# install.packages("BiocManager"); BiocManager::install("zellkonverter")
library(Seurat)
library(SeuratDisk)

# Export a Seurat object as an h5ad file
SaveH5Seurat(seurat_obj, filename = "pbmc.h5Seurat", overwrite = TRUE)
Convert("pbmc.h5Seurat", dest = "h5ad", overwrite = TRUE)
```

Then in Python:

```python
import anndata as ad
import rustscenic.aucell
import rustscenic.data

adata = ad.read_h5ad("pbmc.h5ad")
regulons = [
    ("SPI1_regulon", ["SPI1", "CD14", "LYZ", "CSF1R"]),
    ("PAX5_regulon", ["PAX5", "CD79A", "MS4A1"]),
]
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
# auc is a pandas DataFrame of shape (n_cells, n_regulons) — write back
# to a CSV or h5ad for import into the R session.
auc.to_csv("per_cell_regulon_activity.csv")
```

Back in R:

```r
auc <- read.csv("per_cell_regulon_activity.csv", row.names = 1)
seurat_obj <- AddMetaData(seurat_obj, auc)
# Regulons now available as seurat_obj$SPI1_regulon, seurat_obj$PAX5_regulon, etc.
```

## Option B — `reticulate` one-session bridge

For tighter integration in a single R session:

```r
library(reticulate)
use_virtualenv("path/to/venv")  # where rustscenic is installed
rs <- import("rustscenic")

# Convert Seurat → AnnData in-memory via sceasy or anndata2ri
library(sceasy)
adata <- convertFormat(seurat_obj, from = "seurat", to = "anndata")

regulons <- list(
  list("SPI1_regulon", c("SPI1", "CD14", "LYZ", "CSF1R")),
  list("PAX5_regulon", c("PAX5", "CD79A", "MS4A1"))
)
auc <- rs$aucell$score(adata, regulons, top_frac = 0.05)
# auc is a pandas DataFrame; convert back with py_to_r()
```

## What rustscenic expects from the input

Regardless of which bridge you use, the expression object needs:

- **A dense or sparse `(n_cells, n_genes)` matrix.** Log-normalised is
  strongly recommended — passing raw UMI counts emits a
  `UserWarning` from rustscenic because rankings become dominated by
  library size.
- **Gene-symbol names on the column axis.** If your Seurat object
  uses ENSEMBL IDs as feature names, rustscenic will try to find a
  gene-symbol column (`feature_name`, `gene_symbols`, `gene_name`,
  `symbol`, `Gene`) in `adata.var` and auto-swap. This is the same
  auto-detect that handles cellxgene-curated atlases.

## Gotchas

- **Seurat assays.** If the Seurat object has multiple assays (e.g.
  `RNA` + `ATAC` after integration), `SaveH5Seurat` exports the default
  assay. Switch with `DefaultAssay(seurat_obj) <- "RNA"` before exporting.
- **Layer choice.** rustscenic uses `adata.X` by default. If your
  Seurat export stores the log-normalised matrix in a named layer
  (common with `SeuratDisk`), set it with
  `adata.X = adata.layers["logcounts"]` in Python before scoring.
- **Cell barcodes.** Seurat strips the `-1` suffix 10x adds to
  barcodes by default, so cross-referencing against a 10x ATAC
  fragments file may need a suffix fix
  (`adata.obs_names = [b + "-1" for b in adata.obs_names]`).

## What isn't supported

- **Native R package.** rustscenic is not callable from R without
  going through one of the bridges above. A native R wrapper would
  require a separate crate + R package distribution; it isn't on the
  roadmap unless the Seurat user base asks.
- **SingleCellExperiment outside of AnnData conversion.** The same
  bridges work via `zellkonverter::readH5AD` → `SingleCellExperiment`,
  but the workflow is symmetric: convert, score, convert back.
