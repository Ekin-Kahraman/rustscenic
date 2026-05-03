"""V1 + V2: AUCell compatibility + cell-type discrimination.

Feed both rustscenic's and arboreto's GRN adjacencies through pyscenic.AUCell,
cluster cells, compare per-cluster regulon activity for canonical lineage TFs.

A rustscenic output that's practically useful must:
  (a) AUCell accepts it (same schema as arboreto)
  (b) Resulting regulon activities discriminate cell types
  (c) Match arboreto within biological expectation
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

# pyscenic AUCell + regulon builder
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell
from ctxcore.genesig import Regulon

ADATA_PATH = Path("/Users/ekin/projects/bio/rustscenic/validation/reference/data/pbmc3k.h5ad")
OUR_GRN = Path("/Users/ekin/projects/bio/rustscenic/validation/ours/pbmc3k_grn.parquet")
ARB_GRN = Path("/Users/ekin/projects/bio/rustscenic/validation/reference/data/pbmc3k_grn_full.parquet")

# -- load + cluster PBMC3k ---------------------------------------------------
adata = ad.read_h5ad(ADATA_PATH)
print(f"adata: {adata.shape}")

# Add HVG + PCA + neighbors + leiden for cell typing
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.leiden(adata, resolution=0.5)
print(f"leiden clusters: {adata.obs['leiden'].value_counts().to_dict()}")

# Annotate clusters by marker-gene expression (textbook PBMC markers)
MARKERS = {
    "T": ["CD3D", "CD3E", "CD3G", "IL7R"],
    "CD14_mono": ["CD14", "LYZ"],
    "FCGR3A_mono": ["FCGR3A", "MS4A7"],
    "B": ["CD79A", "CD79B", "MS4A1"],
    "NK": ["GNLY", "NKG7", "KLRD1"],
    "DC": ["FCER1A", "CST3"],
    "Platelet": ["PPBP", "PF4"],
}

def cluster_mean_expr(adata, gene):
    if gene not in adata.var_names:
        return None
    gi = list(adata.var_names).index(gene)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    return pd.DataFrame({"expr": np.asarray(X[:, gi]).flatten(),
                         "cluster": adata.obs["leiden"].values}).groupby("cluster")["expr"].mean()

annotation = {}
for cluster in adata.obs["leiden"].cat.categories:
    scores = {}
    for label, markers in MARKERS.items():
        vals = []
        for g in markers:
            m = cluster_mean_expr(adata, g)
            if m is not None:
                vals.append(m[cluster])
        scores[label] = float(np.mean(vals)) if vals else 0.0
    best = max(scores, key=scores.get)
    annotation[cluster] = best

adata.obs["cell_type"] = adata.obs["leiden"].map(annotation).astype("category")
print("\ncell type assignments:")
print(adata.obs["cell_type"].value_counts().to_string())

# -- build regulons from each tool's adjacencies -----------------------------
# Expression matrix for pyscenic (cells × genes), using adata.to_df()
ex_mtx = adata.to_df()  # pandas DataFrame, cells × genes

def build_regulons(adj_df: pd.DataFrame, name: str, n_targets: int = 50):
    adj_df = adj_df.rename(columns=str.strip)
    if "tf" in adj_df.columns:
        adj_df = adj_df.rename(columns={"tf": "TF"})
    print(f"\n-- {name}: building modules from {len(adj_df)} adjacencies")
    # Collapse to one module per TF (avoid duplicate regulon names → unstack fails)
    modules = list(modules_from_adjacencies(
        adj_df, ex_mtx,
        thresholds=(),
        top_n_targets=(n_targets,),
        top_n_regulators=(),
        min_genes=10,
        rho_mask_dropouts=False,
    ))
    # De-duplicate by TF (keep the first module per TF)
    seen = set()
    regulons = []
    for m in modules:
        tf = m.transcription_factor
        if tf in seen:
            continue
        seen.add(tf)
        # Name must be unique and not collide
        regulons.append(Regulon(
            name=f"{tf}_regulon",
            gene2weight=m.gene2weight,
            transcription_factor=tf,
            gene2occurrence={},
        ))
    print(f"   modules: {len(modules)}   unique-TF regulons: {len(regulons)}")
    return regulons

our_adj = pd.read_parquet(OUR_GRN)
arb_adj = pd.read_parquet(ARB_GRN)

# -- V1: AUCell compat -------------------------------------------------------
print("\n" + "=" * 80)
print("V1: AUCell COMPAT")
print("=" * 80)
t0 = time.monotonic()
try:
    our_reg = build_regulons(our_adj, "rustscenic")
    our_auc = aucell(ex_mtx, our_reg, num_workers=1)
    print(f"V1 rustscenic->AUCell: wall {time.monotonic()-t0:.1f}s  auc shape {our_auc.shape}  PASS")
except Exception as e:
    print(f"V1 rustscenic->AUCell FAIL: {type(e).__name__}: {e}")
    raise

t0 = time.monotonic()
arb_reg = build_regulons(arb_adj, "arboreto")
arb_auc = aucell(ex_mtx, arb_reg, num_workers=1)
print(f"V1 arboreto->AUCell:   wall {time.monotonic()-t0:.1f}s  auc shape {arb_auc.shape}  PASS")

# -- V2: cell-type discrimination -------------------------------------------
print("\n" + "=" * 80)
print("V2: CELL-TYPE DISCRIMINATION")
print("=" * 80)

# Canonical TF->lineage expectations
LINEAGE = {
    "SPI1":  {"high": ["CD14_mono", "FCGR3A_mono", "DC"], "low": ["T", "B", "NK"]},
    "CEBPD": {"high": ["CD14_mono", "FCGR3A_mono"], "low": ["T", "B"]},
    "PAX5":  {"high": ["B"], "low": ["T", "NK", "CD14_mono"]},
    "EBF1":  {"high": ["B"], "low": ["T", "CD14_mono"]},
    "TCF7":  {"high": ["T"], "low": ["CD14_mono", "B"]},
    "LEF1":  {"high": ["T"], "low": ["CD14_mono", "B"]},
    "TBX21": {"high": ["NK"], "low": ["B"]},
    "IRF8":  {"high": ["CD14_mono", "FCGR3A_mono", "DC"], "low": ["T", "B"]},
}

def lineage_score(auc_df, cell_types):
    """For each TF with a known lineage, does its regulon activity peak in the expected cell type?"""
    def tf_of(colname):
        # "SPI1_regulon" -> "SPI1"; "SPI1(+)" -> "SPI1"
        return colname.replace("_regulon", "").replace("(+)", "").replace("(-)", "").strip()
    col_to_tf = {c: tf_of(c) for c in auc_df.columns}
    scores = []
    details = []
    for tf, expect in LINEAGE.items():
        matching = [c for c, t in col_to_tf.items() if t == tf]
        if not matching:
            details.append((tf, "not_in_regulons"))
            continue
        # average across all regulon variants (pyscenic may produce multiple thresholds)
        col = matching[0]
        means = auc_df[col].groupby(cell_types).mean()
        hi_val = np.mean([means.get(ct, 0.0) for ct in expect["high"] if ct in means.index])
        lo_val = np.mean([means.get(ct, 0.0) for ct in expect["low"] if ct in means.index])
        ratio = hi_val / max(lo_val, 1e-9)
        passed = ratio > 1.5
        scores.append(1 if passed else 0)
        details.append((tf, ratio, passed, means.to_dict()))
    return scores, details

our_scores, our_det = lineage_score(our_auc, adata.obs["cell_type"])
arb_scores, arb_det = lineage_score(arb_auc, adata.obs["cell_type"])

print("\nrustscenic lineage discrimination:")
for d in our_det:
    if len(d) == 2:
        print(f"  {d[0]:8}  {d[1]}")
    else:
        tf, ratio, passed, means = d
        print(f"  {tf:8}  hi/lo ratio = {ratio:>6.2f}  {'PASS' if passed else 'FAIL'}")
print(f"  passed: {sum(our_scores)}/{len(our_scores)}")

print("\narboreto lineage discrimination:")
for d in arb_det:
    if len(d) == 2:
        print(f"  {d[0]:8}  {d[1]}")
    else:
        tf, ratio, passed, means = d
        print(f"  {tf:8}  hi/lo ratio = {ratio:>6.2f}  {'PASS' if passed else 'FAIL'}")
print(f"  passed: {sum(arb_scores)}/{len(arb_scores)}")
