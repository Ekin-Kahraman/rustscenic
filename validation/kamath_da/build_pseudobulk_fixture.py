"""Reproducibly build the Kamath DA-neuron subtype pseudobulk fixture.

Replicates the preprocessing protocol described in Fuaad's slack message:
  1. Filter to non-disease ("normal") cells
  2. log-normalize per cell (target_sum=1e4 + log1p)
  3. Mean-aggregate by author_cell_type (10 subtypes)
  4. Filter genes with ≥1 raw count in EVERY subtype
  5. Resolve ENSEMBL IDs → HGNC symbols via var["feature_name"]
"""
from __future__ import annotations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

HERE = Path(__file__).parent
SRC = HERE / "kamath_da_neurons.h5ad"
OUT = HERE / "kamath_da_subtype_pseudobulk.h5ad"
TFS_OUT = HERE / "kamath_tfs_in_matrix.txt"
PARITY_TF_LIST = HERE.parent / "parity_v0310" / "allTFs_hg38.txt"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(
            f"Missing {SRC} — fetch via:\n"
            f"  curl -fsSL -o {SRC} "
            f"https://datasets.cellxgene.cziscience.com/a41c9e65-1abd-428b-aa0a-1d11474bfbe7.h5ad"
        )

    a = ad.read_h5ad(SRC)
    nd = a[a.obs["disease"] == "normal"].copy()
    print(f"non-disease subset: {nd.shape}")

    # log-normalize per cell
    sc.pp.normalize_total(nd, target_sum=1e4)
    sc.pp.log1p(nd)

    subtypes = sorted(nd.obs["author_cell_type"].unique())
    print(f"subtypes ({len(subtypes)}): {subtypes}")

    # Mean-aggregate log-norm per subtype
    pseudo_lognorm = np.zeros((len(subtypes), nd.n_vars), dtype=np.float32)
    for i, st in enumerate(subtypes):
        mask = (nd.obs["author_cell_type"] == st).values
        Xs = nd.X[mask]
        pseudo_lognorm[i] = (
            np.asarray(Xs.mean(axis=0)).ravel() if sp.issparse(Xs) else Xs.mean(axis=0)
        )

    # Filter genes by raw counts: ≥1 count in every subtype
    raw = a[a.obs["disease"] == "normal"].copy()
    keep = np.ones(raw.n_vars, dtype=bool)
    for st in subtypes:
        mask = (raw.obs["author_cell_type"] == st).values
        Xs = raw.X[mask]
        sums = np.asarray(Xs.sum(axis=0)).ravel() if sp.issparse(Xs) else Xs.sum(axis=0)
        keep &= sums >= 1
    print(f"genes with ≥1 raw count in every subtype: {keep.sum()} / {raw.n_vars}")

    matrix = pseudo_lognorm[:, keep]
    symbols = raw.var.loc[keep, "feature_name"].astype(str).tolist()
    ensg = raw.var_names[keep].tolist()

    # Build AnnData with HGNC symbols as var_names
    out_ad = ad.AnnData(
        X=matrix.astype(np.float32),
        obs=pd.DataFrame({"subtype": subtypes}, index=subtypes),
        var=pd.DataFrame({"ensembl": ensg, "feature_name": symbols}, index=symbols),
    )
    if out_ad.var_names.duplicated().any():
        n_dup = int(out_ad.var_names.duplicated().sum())
        print(f"WARN: {n_dup} duplicate gene symbols after ENSEMBL→HGNC; keeping first")
        out_ad = out_ad[:, ~out_ad.var_names.duplicated(keep="first")].copy()

    out_ad.var_names_make_unique()
    out_ad.write_h5ad(OUT)
    print(f"wrote {OUT}: {out_ad.shape}")

    # Filter the parity TF list to genes present in the matrix
    if PARITY_TF_LIST.exists():
        tfs = [l.strip() for l in PARITY_TF_LIST.read_text().splitlines() if l.strip()]
        present = [t for t in tfs if t in set(out_ad.var_names)]
        TFS_OUT.write_text("\n".join(present) + "\n")
        print(f"wrote {TFS_OUT}: {len(present)} TFs (of {len(tfs)} in aertslab list)")


if __name__ == "__main__":
    main()
