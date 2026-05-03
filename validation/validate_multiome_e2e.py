"""End-to-end rustscenic on paired 10x Multiome: grn → aucell (RNA) +
topics (ATAC). Does regulon activity from grn correlate with ATAC topic
assignment (should — same cells, same underlying cell-type structure)?

Also: benchmark total wall-clock vs what full pyscenic pipeline would take
(arboreto 393s on PBMC-3k + pyscenic.aucell 36s on PBMC-10k → scale).

Requires: pip install "rustscenic[validation]"
(pulls scanpy, anndata, igraph, leidenalg, scikit-learn — not core deps.)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

import rustscenic, rustscenic.grn, rustscenic.aucell, rustscenic.topics

RNA = Path("/Users/ekin/projects/bio/rustscenic/validation/reference/data/multiome3k/rna.h5ad")
ATAC = Path("/Users/ekin/projects/bio/rustscenic/validation/reference/data/multiome3k/atac_binarized.h5ad")
TFS = Path("/Users/ekin/projects/bio/rustscenic/validation/reference/data/allTFs_hg38.txt")

rna = ad.read_h5ad(RNA)
atac = ad.read_h5ad(ATAC)
print(f"RNA  {rna.shape}   ATAC  {atac.shape}")

# Intersect cells present in both modalities (should be ~all of them)
shared = sorted(set(rna.obs_names) & set(atac.obs_names))
rna = rna[shared].copy()
atac = atac[shared].copy()
print(f"shared cells: {len(shared)}")

# --- stage 1: grn on RNA ---
tfs = [t for t in rustscenic.grn.load_tfs(TFS) if t in set(rna.var_names)]
print(f"\n--- grn (RNA) ---  cells={rna.n_obs}  genes={rna.n_vars}  tfs={len(tfs)}")
t0 = time.monotonic()
grn_df = rustscenic.grn.infer(rna, tfs, seed=777, n_estimators=300, early_stop_window=25)
t_grn = time.monotonic() - t0
print(f"  wall: {t_grn:.1f}s  edges: {len(grn_df)}")

# --- stage 2: aucell on RNA ---
# Build regulons from grn (top-50 targets per TF), filter to ≥10 genes
regs = []
for tf, grp in grn_df.groupby("TF"):
    top_targets = grp.nlargest(50, "importance")["target"].tolist()
    if len(top_targets) >= 10:
        regs.append((f"{tf}_regulon", top_targets))
print(f"\n--- aucell ---  regulons={len(regs)}")
t0 = time.monotonic()
auc = rustscenic.aucell.score(rna, regs, top_frac=0.05)
t_aucell = time.monotonic() - t0
print(f"  wall: {t_aucell:.1f}s  shape {auc.shape}")

# --- stage 3: topics on ATAC ---
K = 10  # fewer topics → more stable on this small dataset
print(f"\n--- topics (ATAC) ---  K={K}")
t0 = time.monotonic()
tres = rustscenic.topics.fit(
    atac, n_topics=K, n_passes=20, batch_size=256, seed=777,
    alpha=1.0/K, eta=1.0/K,
)
t_topics = time.monotonic() - t0
topic_assign = np.asarray([int(s.replace("Topic_", "")) for s in tres.cell_assignment().values])
print(f"  wall: {t_topics:.1f}s  unique top-1 topic: {len(set(topic_assign))}")

# --- cluster cells from ATAC for ground-truth cell type proxy ---
atac_norm = atac.copy()
atac_norm.X = atac_norm.X.astype(np.float32)
sc.pp.normalize_total(atac_norm); sc.pp.log1p(atac_norm)
sc.pp.highly_variable_genes(atac_norm, n_top_genes=5000)
sc.tl.pca(atac_norm, n_comps=30, mask_var="highly_variable")
sc.pp.neighbors(atac_norm, n_neighbors=15)
sc.tl.leiden(atac_norm, resolution=0.3, flavor="igraph", n_iterations=2, directed=False)
cluster = atac_norm.obs["leiden"].astype(str).values
print(f"  atac leiden clusters: {len(set(cluster))}")

# --- cross-modal check: do regulon activities discriminate the same cell clusters
# that ATAC topics do?
auc_vals = auc.values  # cells x regulons
# For each cluster, mean regulon activity; find regulons differentiating clusters
from scipy.stats import f_oneway
top_discriminative_reg = []
for r_idx, reg_name in enumerate(auc.columns):
    groups = [auc_vals[cluster == c, r_idx] for c in np.unique(cluster)]
    try:
        F, p = f_oneway(*[g for g in groups if len(g) > 1])
        top_discriminative_reg.append((reg_name, F, p))
    except Exception:
        pass
top_discriminative_reg.sort(key=lambda x: -x[1])
print(f"\ntop-10 cluster-discriminative regulons (by ANOVA F):")
for name, F, p in top_discriminative_reg[:10]:
    print(f"  {name:25s}  F={F:>7.1f}  p={p:.2e}")

# --- total pipeline timing ---
t_total = t_grn + t_aucell + t_topics
print(f"\n=== TOTAL pipeline wall-clock: {t_total:.1f}s ({t_total/60:.1f} min) ===")
print(f"  grn:    {t_grn:>6.1f}s  ({100*t_grn/t_total:.0f}%)")
print(f"  aucell: {t_aucell:>6.1f}s  ({100*t_aucell/t_total:.0f}%)")
print(f"  topics: {t_topics:>6.1f}s  ({100*t_topics/t_total:.0f}%)")
print(f"\nARI of grn-based cell-type clustering (via top regulon activity) vs ATAC leiden:")
# assign cells by top regulon
cell_by_topreg = auc.idxmax(axis=1).values
mapped = np.unique(cell_by_topreg, return_inverse=True)[1]
print(f"  {adjusted_rand_score(cluster, mapped):.4f}  ({len(set(cell_by_topreg))} unique top regulons)")
