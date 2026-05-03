"""rustscenic.aucell on PBMC-10k — speed + lineage discrimination."""
import time
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell as py_aucell
from ctxcore.genesig import Regulon
import rustscenic
import rustscenic.aucell

ADATA = ad.read_h5ad("/Users/ekin/projects/bio/rustscenic/validation/reference/data/pbmc10k.h5ad")
print(f"adata: {ADATA.shape}")

sc.pp.highly_variable_genes(ADATA, n_top_genes=2000, flavor="seurat")
sc.tl.pca(ADATA, n_comps=30, mask_var="highly_variable")
sc.pp.neighbors(ADATA, n_neighbors=15)
sc.tl.leiden(ADATA, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)

MARKERS = {
    "T": ["CD3D", "CD3E", "CD3G", "IL7R"],
    "CD14_mono": ["CD14", "LYZ"],
    "FCGR3A_mono": ["FCGR3A", "MS4A7"],
    "B": ["CD79A", "CD79B", "MS4A1"],
    "NK": ["GNLY", "NKG7", "KLRD1"],
    "DC": ["FCER1A", "CST3"],
    "Platelet": ["PPBP", "PF4"],
}
def cme(gene, cluster):
    if gene not in ADATA.var_names: return None
    gi = list(ADATA.var_names).index(gene)
    X = ADATA.X.toarray() if hasattr(ADATA.X, "toarray") else ADATA.X
    return pd.DataFrame({"expr": np.asarray(X[:, gi]).flatten(),
                         "cluster": ADATA.obs["leiden"].values}).groupby("cluster", observed=True)["expr"].mean()[cluster]
ann = {}
for c in ADATA.obs["leiden"].cat.categories:
    scores = {l: np.mean([cme(g, c) for g in ms if cme(g, c) is not None]) for l, ms in MARKERS.items()}
    ann[c] = max(scores, key=scores.get)
ADATA.obs["cell_type"] = ADATA.obs["leiden"].map(ann).astype("category")
print(f"cell_type: {ADATA.obs['cell_type'].value_counts().to_dict()}")

ex_mtx = ADATA.to_df()
adj = pd.read_parquet("/Users/ekin/projects/bio/rustscenic/validation/ours/pbmc10k_grn_ours.parquet").rename(columns={"tf": "TF"})
t0 = time.monotonic()
modules = list(modules_from_adjacencies(adj, ex_mtx, thresholds=(), top_n_targets=(50,),
                                        top_n_regulators=(), min_genes=10, rho_mask_dropouts=False))
seen = set()
regs_rust = []
regs_py = []
for m in modules:
    tf = m.transcription_factor
    if tf in seen: continue
    seen.add(tf)
    name = f"{tf}_regulon"
    regs_rust.append((name, list(m.gene2weight.keys())))
    regs_py.append(Regulon(name=name, gene2weight=m.gene2weight, transcription_factor=tf, gene2occurrence={}))
print(f"modules built in {time.monotonic()-t0:.1f}s   {len(regs_rust)} regulons")

print("\n--- rustscenic.aucell ---")
t0 = time.monotonic()
our_auc = rustscenic.aucell.score(ADATA, regs_rust, top_frac=0.05)
print(f"  wall: {time.monotonic()-t0:.2f}s   shape={our_auc.shape}")

print("\n--- pyscenic.aucell ---")
t0 = time.monotonic()
py_auc = py_aucell(ex_mtx, regs_py, num_workers=1)
py_wall = time.monotonic() - t0
print(f"  wall: {py_wall:.2f}s   shape={py_auc.shape}")

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
def lineage_disc(auc_df, label):
    passed = 0
    print(f"\n{label} lineage discrimination:")
    for tf, ex in LINEAGE.items():
        col = f"{tf}_regulon"
        if col not in auc_df.columns: continue
        means = auc_df[col].groupby(ADATA.obs["cell_type"], observed=True).mean()
        hi = np.mean([means.get(ct, 0) for ct in ex["high"] if ct in means.index])
        lo = np.mean([means.get(ct, 0) for ct in ex["low"] if ct in means.index])
        r = hi / max(lo, 1e-9)
        ok = r > 1.5
        passed += int(ok)
        print(f"  {tf:8}  ratio={r:>6.2f}  {'PASS' if ok else 'FAIL'}")
    print(f"  passed: {passed}/{len(LINEAGE)}")
    return passed

rust_pass = lineage_disc(our_auc, "rustscenic")
py_pass = lineage_disc(py_auc, "pyscenic")
