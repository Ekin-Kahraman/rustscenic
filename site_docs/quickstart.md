# Quickstart

Infer a gene network and score candidate gene-set activity with **v0.5.0**.
Start with [Installation](installation.md), or
[run PBMC3k](https://github.com/Ekin-Kahraman/rustscenic/blob/main/QUICKSTART.md)
for a complete example that downloads its own data.

## Prepare your input

Use an RNA AnnData file (`rna.h5ad`) with:

- Cells as rows and genes as columns.
- Log-normalised expression in `adata.X`. If it is in a layer, set
  `adata.X = adata.layers["logcounts"]` using your layer's name.
- Gene symbols matching your TF list. Use `data.tfs("hs")` for human or
  `data.tfs("mm")` for mouse.

For Ensembl-indexed AnnData, RustScenic can resolve gene symbols from
`adata.var["feature_name"]`. For R inputs, see the
[Seurat / R guide](seurat-interop.md) for explicit RNA layer selection and a
sparse matrix round-trip.

## Python

```python
import anndata as ad
from rustscenic import aucell, data, grn

adata = ad.read_h5ad("rna.h5ad")
edges = grn.infer(
    adata, tf_names=data.tfs("hs"), n_estimators=500,
    top_targets_per_tf=50, seed=777,
)
regulons = {
    tf: group["target"].tolist()
    for tf, group in edges.groupby("TF")
    if len(group) >= 10
}
auc = aucell.score(adata, regulons, top_frac=0.05)

edges.to_parquet("grn.parquet")
auc.to_parquet("aucell.parquet")
print(edges.head())
print(auc.iloc[:3, :3])
```

`grn.parquet` contains `TF`, `target` and `importance` columns.
`aucell.parquet` contains cells × candidate gene-set activity scores,
with cell names in the index.

These candidate sets are not motif-filtered or split by correlation sign.
For signed sets and motif support, see the [API map](api.md).

## Command line

Run the RNA pipeline on the same prepared file. The `hs` shortcut uses the
bundled human TF list; use `mm` for mouse.

```bash
rustscenic pipeline --rna rna.h5ad --tfs hs --output out/
```

Results are written to `out/`. The pipeline separates candidate target sets
by correlation sign; motif filtering requires motif rankings and annotations.
For individual GRN and AUCell commands with prepared inputs, see the
[PBMC3k walkthrough](https://github.com/Ekin-Kahraman/rustscenic/blob/main/QUICKSTART.md#command-line).

## Next steps

[Motif and multiome stages](api.md) ·
[HPC operation](hpc.md) ·
[Tester guide](https://github.com/Ekin-Kahraman/rustscenic/blob/main/docs/tester-quickstart.md) ·
[Explore a saved network](demo/endothelial/index.html)
