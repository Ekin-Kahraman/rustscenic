# Quickstart

This example runs RNA GRN inference and AUCell scoring on an AnnData object.

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

regulons = [
    (
        f"{tf}_regulon",
        grn[grn["TF"] == tf].nlargest(50, "importance")["target"].tolist(),
    )
    for tf in grn["TF"].unique()
]

auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
auc.to_parquet("aucell.parquet")
```

## CLI

```bash
rustscenic grn \
  --expression data.h5ad \
  --tfs tfs.txt \
  --output grn.parquet

rustscenic aucell \
  --expression data.h5ad \
  --regulons grn.parquet \
  --output auc.parquet
```

## End-To-End Example

The repository includes a PBMC-3k example:

```bash
pip install "rustscenic[examples]"
python examples/pbmc3k_end_to_end.py
```

For collaborators or external testers, use the tester path in `docs/tester-quickstart.md` in the repository.
