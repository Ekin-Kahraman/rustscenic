"""V1+V2 on PBMC-10k: AUCell compat + cell-type discrimination on larger dataset."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell
from ctxcore.genesig import Regulon

ADATA = ad.read_h5ad("/Users/ekin/rustscenic/validation/reference/data/pbmc10k.h5ad")
OUR_GRN = Path("/Users/ekin/rustscenic/validation/ours/pbmc10k_grn_ours.parquet")
print(f"adata: {ADATA.shape}   adjacencies: {pd.read_parquet(OUR_GRN).shape}")

# cluster + annotate
sc.pp.highly_variable_genes(ADATA, n_top_genes=2000, flavor="seurat")
sc.tl.pca(ADATA, n_comps=30, mask_var="highly_variable")
sc.pp.neighbors(ADATA, n_neighbors=15)
sc.tl.leiden(ADATA, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
print(f"leiden: {ADATA.obs['leiden'].value_counts().to_dict()}")

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
    if gene not in adata.var_names: return None
    gi = list(adata.var_names).index(gene)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    return pd.DataFrame({"expr": np.asarray(X[:, gi]).flatten(),
                         "cluster": adata.obs["leiden"].values}).groupby("cluster", observed=True)["expr"].mean()
ann = {}
for c in ADATA.obs["leiden"].cat.categories:
    scores = {}
    for lab, ms in MARKERS.items():
        vals = [cluster_mean_expr(ADATA, g)[c] for g in ms if cluster_mean_expr(ADATA, g) is not None]
        scores[lab] = float(np.mean(vals)) if vals else 0.0
    ann[c] = max(scores, key=scores.get)
ADATA.obs["cell_type"] = ADATA.obs["leiden"].map(ann).astype("category")
print(f"cell_type counts: {ADATA.obs['cell_type'].value_counts().to_dict()}")

ex_mtx = ADATA.to_df()
adj = pd.read_parquet(OUR_GRN).rename(columns={"tf": "TF"})
t0 = time.monotonic()
modules = list(modules_from_adjacencies(
    adj, ex_mtx, thresholds=(), top_n_targets=(50,),
    top_n_regulators=(), min_genes=10, rho_mask_dropouts=False,
))
print(f"modules from adjacencies: {time.monotonic()-t0:.1f}s  ({len(modules)} modules)")

seen = set()
regulons = []
for m in modules:
    if m.transcription_factor in seen:
        continue
    seen.add(m.transcription_factor)
    regulons.append(Regulon(name=f"{m.transcription_factor}_regulon",
                            gene2weight=m.gene2weight,
                            transcription_factor=m.transcription_factor,
                            gene2occurrence={}))
print(f"unique-TF regulons: {len(regulons)}")

t0 = time.monotonic()
auc = aucell(ex_mtx, regulons, num_workers=1)
print(f"AUCell: {time.monotonic()-t0:.1f}s  shape={auc.shape}")

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
def tf_of(c):
    return c.replace("_regulon", "")
col_to_tf = {c: tf_of(c) for c in auc.columns}
print("\nlineage discrimination (hi/lo ratio, pass if >1.5):")
passed = 0
for tf, expect in LINEAGE.items():
    m = [c for c, t in col_to_tf.items() if t == tf]
    if not m:
        print(f"  {tf:8}  missing regulon"); continue
    col = m[0]
    means = auc[col].groupby(ADATA.obs["cell_type"], observed=True).mean()
    hi = np.mean([means.get(ct, 0) for ct in expect["high"] if ct in means.index])
    lo = np.mean([means.get(ct, 0) for ct in expect["low"] if ct in means.index])
    r = hi / max(lo, 1e-9)
    ok = r > 1.5
    passed += int(ok)
    print(f"  {tf:8}  ratio={r:>6.2f}  {'PASS' if ok else 'FAIL'}")
print(f"\npassed: {passed}/{len(LINEAGE)}")
