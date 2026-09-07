# Quickstart

This example works with the published **v0.4.7** package. It starts with an
AnnData RNA matrix, infers a gene network and scores candidate gene sets in
each cell. The sets are not motif-filtered or split by correlation sign.

The development-only `add_correlation`, `build_regulons` and `add-cor` features
are planned for v0.5.0; see the [API map](api.md). They are not required below.

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
    top_targets_per_tf=50,
    seed=777,
)
regulons = {
    tf: group["target"].tolist()
    for tf, group in grn.groupby("TF")
    if len(group) >= 10
}

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

For collaborators or external testers, use the
[tester quickstart](https://github.com/Ekin-Kahraman/rustscenic/blob/main/docs/tester-quickstart.md)
in the repository.
