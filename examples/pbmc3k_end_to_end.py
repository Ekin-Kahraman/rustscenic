"""End-to-end rustscenic pipeline on PBMC-3k.

What this shows — the realistic RNA-only SCENIC workflow:

  1. Load + preprocess PBMC-3k (bundled with scanpy; zero download, zero paths)
  2. Infer a GRN with rustscenic.grn.infer (replaces arboreto.grnboost2)
  3. Build regulons from top-N targets per TF (no pyscenic required)
  4. Score per-cell regulon activity with rustscenic.aucell.score
     (replaces pyscenic.aucell.aucell)
  5. Save outputs: grn.parquet, regulons.json, auc.csv
  6. Sanity check: lineage TFs should be higher in their expected cell types

Run:
    python examples/pbmc3k_end_to_end.py

Outputs are written to examples/out/.

Runtime: ~3 minutes on an 8-core laptop with n_estimators=500.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def main() -> int:
    import anndata  # noqa: F401
    import scanpy as sc
    import rustscenic.grn
    import rustscenic.aucell

    # ---- 1. Load + preprocess PBMC-3k ----
    print("[1/6] loading PBMC-3k (10x filtered gene-bc matrix, cached)")
    mtx_dir = OUT_DIR / "filtered_gene_bc_matrices" / "hg19"
    if not mtx_dir.exists():
        import urllib.request, tarfile
        tgz = OUT_DIR / "pbmc3k.tar.gz"
        if not tgz.exists():
            url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
            print(f"      downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req) as resp, open(tgz, "wb") as fh:
                fh.write(resp.read())
        print(f"      extracting {tgz.name} ...")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(OUT_DIR)
    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    # Canonical TF list — a realistic analysis uses the full aertslab allTFs_hg38.txt
    candidate_tfs = [
        "SPI1", "CEBPB", "CEBPD", "IRF8", "MAFB",             # myeloid
        "PAX5", "EBF1", "POU2AF1", "BACH2",                   # B cell
        "TCF7", "LEF1", "ETS1", "GATA3", "RUNX3",             # T cell
        "TBX21", "EOMES",                                     # NK / Th1
        "KLF4", "ZEB2", "STAT1", "IRF1", "NFKB1",             # pan-lineage
    ]
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Restrict to top 2000 HVGs for speed — but union with TFs + canonical
    # lineage markers so we can both regulate (TFs) and sanity-check (markers).
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    marker_union = ["MS4A1", "CD79A", "CD3D", "CD3E", "CD14", "LYZ", "NKG7",
                    "GNLY", "GZMB", "FCGR3A", "CCR7", "IL7R"]
    keep = adata.var.highly_variable.copy()
    keep[adata.var_names.isin(candidate_tfs + marker_union)] = True
    adata = adata[:, keep].copy()
    # Louvain / leiden cluster labels for downstream lineage checks
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5, random_state=0, flavor="igraph",
                 n_iterations=2, directed=False)
    print(f"      shape: {adata.shape}  leiden clusters: {adata.obs['leiden'].nunique()}")

    # ---- 2. GRN inference ----
    tfs = [t for t in candidate_tfs if t in adata.var_names]
    print(f"[2/6] GRN: {len(tfs)} TFs in data, running rustscenic.grn.infer...")
    grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=500, seed=777)
    print(f"      edges: {len(grn):,}")
    grn.to_parquet(OUT_DIR / "grn.parquet")
    print(f"      saved: {OUT_DIR / 'grn.parquet'}")

    # ---- 3. Build regulons: top-50 targets per TF by importance ----
    print("[3/6] building regulons (top-50 targets per TF)")
    regulons = {}
    for tf in grn["TF"].unique():
        top = grn[grn["TF"] == tf].nlargest(50, "importance")["target"].tolist()
        if len(top) >= 10:
            regulons[f"{tf}_regulon"] = top
    print(f"      regulons: {len(regulons)} (≥10 targets)")
    with open(OUT_DIR / "regulons.json", "w") as fh:
        json.dump(regulons, fh, indent=2)

    # ---- 4. AUCell: per-cell regulon activity ----
    print("[4/6] AUCell: scoring per-cell regulon activity")
    reg_list = [(name, genes) for name, genes in regulons.items()]
    auc = rustscenic.aucell.score(adata, reg_list, top_frac=0.05)
    print(f"      activity matrix: {auc.shape}")
    auc.to_csv(OUT_DIR / "auc.csv")
    print(f"      saved: {OUT_DIR / 'auc.csv'}")

    # ---- 5. Save combined output for downstream analysis ----
    adata.obs = adata.obs.join(auc, how="left", lsuffix="", rsuffix="_auc")
    adata.write_h5ad(OUT_DIR / "pbmc3k_with_regulons.h5ad")
    print(f"[5/6] saved: {OUT_DIR / 'pbmc3k_with_regulons.h5ad'} "
          f"(original adata + per-cell regulon activity in .obs)")

    # ---- 6. Sanity check: canonical lineage TFs should be enriched in expected clusters ----
    print("[6/6] biology sanity: per-cluster mean regulon activity")
    # Mark clusters by canonical marker-based identity (crude, for demo sanity only)
    marker_genes = {
        "B": ["MS4A1", "CD79A"],
        "T": ["CD3D", "CD3E"],
        "Mono": ["CD14", "LYZ"],
        "NK": ["NKG7", "GNLY"],
    }
    cluster_means = auc.groupby(adata.obs["leiden"].values, observed=False).mean()
    # Identify clusters by dominant marker expression
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    expr_by_cluster = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    expr_by_cluster["leiden"] = adata.obs["leiden"].values
    cluster_marker_means = expr_by_cluster.groupby("leiden", observed=False).mean()
    # Z-score each marker across clusters so the label picks up *relative*
    # enrichment, not absolute expression (CD14/LYZ are high everywhere;
    # we want the cluster where they're most *distinctively* high).
    marker_z = (cluster_marker_means - cluster_marker_means.mean(axis=0)) / (cluster_marker_means.std(axis=0) + 1e-12)
    cluster_lineage = {}
    for c in marker_z.index:
        scores = {}
        for lin, genes in marker_genes.items():
            present = [g for g in genes if g in marker_z.columns]
            scores[lin] = marker_z.loc[c, present].mean() if present else -np.inf
        cluster_lineage[c] = max(scores, key=scores.get)
    print("      cluster -> assigned lineage (by marker expression):")
    for c, lin in cluster_lineage.items():
        print(f"        cluster {c}: {lin}")

    # Check that key regulons are highest in their expected lineage
    checks = [
        ("SPI1_regulon", "Mono"),   # SPI1 drives myeloid
        ("PAX5_regulon", "B"),      # PAX5 drives B cells
        ("TCF7_regulon", "T"),      # TCF7 drives T cell identity
        ("TBX21_regulon", "NK"),    # T-bet drives NK / Th1
    ]
    print(f"\n      regulon -> most-active cluster lineage (✓ = expected):")
    passed = 0
    for reg, expected_lin in checks:
        if reg not in cluster_means.columns: continue
        top_cluster = cluster_means[reg].idxmax()
        actual_lin = cluster_lineage.get(top_cluster, "?")
        ok = actual_lin == expected_lin
        passed += ok
        mark = "✓" if ok else "·"
        print(f"        {mark} {reg} top in cluster {top_cluster} ({actual_lin}); expected {expected_lin}")
    print(f"      {passed}/{sum(1 for r,_ in checks if r in cluster_means.columns)} canonical regulons in expected lineage\n")

    print(f"done. outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
