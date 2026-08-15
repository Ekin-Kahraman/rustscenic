# Quickstart

This example starts with an AnnData RNA matrix and produces a GRN plus per-cell
regulon activity scores. It is the shortest path from install to a useful
RustScenic output.

```python
import anndata as ad
import rustscenic.grn
import rustscenic.aucell
import rustscenic.data

adata = ad.read_h5ad("rna.h5ad")
tfs = rustscenic.data.tfs("hs")

grn = rustscenic.grn.infer(
    adata,
    tf_names=tfs,
    n_estimators=500,
    seed=777,
)
signed_grn = rustscenic.grn.add_correlation(grn, adata, rho_threshold=0.03)
regulons = rustscenic.grn.build_regulons(
    signed_grn,
    top_targets_per_tf=50,
    min_targets=10,
    include_repressors=True,
)

auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
auc.to_parquet("aucell.parquet")
```

## CLI

The same core stages are available from the command line:

```bash
rustscenic grn \
  --expression data.h5ad \
  --tfs tfs.txt \
  --output grn.parquet

rustscenic add-cor \
  --expression data.h5ad \
  --adjacencies grn.parquet \
  --output signed-grn.parquet

rustscenic aucell \
  --expression data.h5ad \
  --regulons signed-grn.parquet \
  --output auc.parquet
```

## End-To-End Example

The repository includes a PBMC-3k example:

```bash
pip install "rustscenic[examples]"
python examples/pbmc3k_end_to_end.py
```

For collaborators or external testers, use the
[tester quickstart](https://github.com/Ekin-Kahraman/rustscenic/blob/main/docs/tester-quickstart.md)
in the repository.
