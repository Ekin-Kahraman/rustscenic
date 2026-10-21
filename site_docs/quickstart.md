# Quickstart

This example uses **v0.5.0**. It starts with an
AnnData RNA matrix, infers a gene network and scores candidate gene sets in
each cell. The sets are not motif-filtered or split by correlation sign.

To separate target sets by correlation sign, use `add_correlation`,
`build_regulons` or `add-cor`; see the [API map](api.md). They are not required
for the basic example below.

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

From a repository checkout, run the PBMC3k RNA example:

```bash
pip install "rustscenic[examples]"
python examples/pbmc3k_end_to_end.py
```

It downloads 10x data on first use, caches it locally and writes the network,
candidate gene sets and activity scores to `examples/out/`. It includes a
marker-based lineage sanity check; the gene sets are not motif-filtered.

For a copy-and-run PBMC3k walkthrough, see the
[repository quickstart](https://github.com/Ekin-Kahraman/rustscenic/blob/main/QUICKSTART.md).
To explore saved results without installation, open the
[endothelial-cell example](demo/endothelial/index.html).

For collaborators or external testers, use the
[tester quickstart](https://github.com/Ekin-Kahraman/rustscenic/blob/main/docs/tester-quickstart.md)
in the repository.
