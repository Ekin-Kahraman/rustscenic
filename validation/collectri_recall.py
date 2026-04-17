"""CollecTRI external gold-standard recall for rustscenic vs arboreto on PBMC-3k."""
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path

collectri = pd.read_csv("/tmp/collectri_human.tsv", sep="\t")
# Column 'source' is TF, 'target' is target gene
print(f"CollecTRI: {len(collectri)} edges, {collectri['source'].nunique()} TFs")

adata = ad.read_h5ad("/Users/ekin/rustscenic/validation/reference/data/pbmc3k.h5ad")
tfs = open("/Users/ekin/rustscenic/validation/reference/data/allTFs_hg38.txt").read().strip().splitlines()
tfs_in = set(t for t in tfs if t in set(adata.var_names))
genes_in = set(adata.var_names)

# Keep edges where both ends are in our data AND TF is in our TF list
ct_in = collectri[
    collectri["source"].isin(tfs_in)
    & collectri["target"].isin(genes_in)
]
print(f"CollecTRI edges with both ends in PBMC-3k expressed genes: {len(ct_in)}")
print(f"  TFs covered: {ct_in['source'].nunique()}")

ours = pd.read_parquet("/Users/ekin/rustscenic/validation/ours/pbmc3k_grn.parquet").rename(columns={"tf": "TF"})
arb = pd.read_parquet("/Users/ekin/rustscenic/validation/reference/data/pbmc3k_grn_full.parquet")

def recall_at_k(df, edges, k):
    # For each TF, get top-k targets; check if CollecTRI edge target is in that top-k
    tf_to_top = {}
    hits = 0
    for _, row in edges.iterrows():
        tf = row["source"]
        if tf not in tf_to_top:
            tf_to_top[tf] = set(df[df["TF"] == tf].nlargest(k, "importance")["target"].values)
        if row["target"] in tf_to_top[tf]:
            hits += 1
    return hits

total = len(ct_in)
print(f"\nRecall on {total} CollecTRI edges (external, independent):")
print(f"  {'k':>6} {'rustscenic':>14} {'arboreto':>14} {'random-chance':>15}")
for k in [10, 20, 50, 100, 500]:
    our_h = recall_at_k(ours, ct_in, k)
    arb_h = recall_at_k(arb, ct_in, k)
    random_expected = k / adata.n_vars
    print(f"  top-{k:<3} {our_h:>5}/{total} ({100*our_h/total:>4.1f}%)  "
          f"{arb_h:>5}/{total} ({100*arb_h/total:>4.1f}%)  "
          f"{100*random_expected:>5.2f}%")
