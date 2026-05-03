"""Validate rustscenic.topics against tomotopy (C++ Gibbs LDA, pip-installable,
algorithmically the same class as pycisTopic's default Mallet Gibbs backend).

Data: 10x Multiome PBMC 3k, ATAC peak matrix binarized at nonzero,
      2598 cells x 98079 peaks, 7.5% sparsity.
Both tools run with K=30 topics, alpha=eta=1/K, same seed.
Metric: topic-assignment ARI (Hungarian-optimal relabeling not needed, ARI is
permutation-invariant). Also: top-10 peak overlap per matched topic.
"""
import time
from pathlib import Path
import numpy as np
import anndata as ad
import scipy.sparse as sp
import tomotopy as tp
from sklearn.metrics import adjusted_rand_score

import rustscenic.topics

ATAC = Path("/Users/ekin/projects/bio/rustscenic/validation/reference/data/multiome3k/atac_binarized.h5ad")
adata = ad.read_h5ad(ATAC)
print(f"ATAC: {adata.shape}  density {adata.X.nnz/(adata.shape[0]*adata.shape[1]):.4f}")

K = 30
ALPHA = 1.0 / K
ETA = 1.0 / K
SEED = 42

# --- rustscenic.topics ---
print("\n--- rustscenic.topics (online VB) ---")
t0 = time.monotonic()
ours = rustscenic.topics.fit(
    adata, n_topics=K, n_passes=15, batch_size=256, seed=SEED,
    alpha=ALPHA, eta=ETA,
)
ours_wall = time.monotonic() - t0
ours_assign = np.asarray([int(s.replace("Topic_", "")) for s in ours.cell_assignment().values])
print(f"  wall: {ours_wall:.1f}s  unique: {len(set(ours_assign))}")

# --- tomotopy (Gibbs, C++ AVX) ---
print("\n--- tomotopy (Gibbs, C++ AVX) ---")
peak_names = list(adata.var_names)
t0 = time.monotonic()
mdl = tp.LDAModel(k=K, alpha=ALPHA, eta=ETA, seed=SEED)
X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
for c in range(X.shape[0]):
    row = X.getrow(c)
    nz_peaks = row.indices
    if len(nz_peaks) == 0:
        continue
    mdl.add_doc(words=[peak_names[p] for p in nz_peaks])
# Gibbs iterations — tomotopy default 100 iterations per train() call
TOTAL_ITERS = 500  # more thorough than online VB passes count — aligns with typical cisTopic Mallet default
mdl.train(TOTAL_ITERS, workers=8)
tomo_wall = time.monotonic() - t0
tomo_assign = np.zeros(X.shape[0], dtype=int)
for i, doc in enumerate(mdl.docs):
    topic_dist = doc.get_topic_dist()
    tomo_assign[i] = int(np.argmax(topic_dist))
print(f"  wall: {tomo_wall:.1f}s  unique: {len(set(tomo_assign))}")

# --- metrics ---
ari = adjusted_rand_score(ours_assign, tomo_assign)
print(f"\nCross-tool ARI (rustscenic.topics vs tomotopy): {ari:.4f}")
print(f"Speedup (tomotopy / ours): {tomo_wall/ours_wall:.2f}x")

# Top-peak overlap: for each pair of topics (ours, tomotopy), find best matching pair
# (Hungarian-style) and report mean top-10 peak overlap on matched pairs.
ours_top = []
for k in range(K):
    row = ours.topic_peak.iloc[k].sort_values(ascending=False)
    ours_top.append(set(row.index[:10]))
tomo_top = []
for k in range(K):
    wp = mdl.get_topic_words(k, top_n=10)
    tomo_top.append(set(w for w, _ in wp))

# Best match per our topic
best_overlaps = []
for i in range(K):
    best = 0
    for j in range(K):
        ov = len(ours_top[i] & tomo_top[j])
        if ov > best:
            best = ov
    best_overlaps.append(best)
print(f"mean best top-10 peak overlap per matched topic: {np.mean(best_overlaps):.2f}/10")
print(f"   distribution: min {np.min(best_overlaps)}  median {int(np.median(best_overlaps))}  max {np.max(best_overlaps)}")

# --- cell-type agreement: cluster cells by UMAP, measure against topic assignments ---
import scanpy as sc
print("\n--- cell-type proxy (leiden on ATAC LSI) ---")
# Log-normalize the ATAC (for clustering purposes)
atac_norm = adata.copy()
atac_norm.X = atac_norm.X.astype(np.float32)
sc.pp.normalize_total(atac_norm)
sc.pp.log1p(atac_norm)
sc.pp.highly_variable_genes(atac_norm, n_top_genes=5000)
sc.tl.pca(atac_norm, n_comps=30, mask_var="highly_variable")
sc.pp.neighbors(atac_norm, n_neighbors=15)
sc.tl.leiden(atac_norm, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
cluster = atac_norm.obs["leiden"].astype(str).values
print(f"  leiden clusters: {len(set(cluster))}")

print(f"  rustscenic topics ARI vs leiden clusters: {adjusted_rand_score(cluster, ours_assign):.4f}")
print(f"  tomotopy   topics ARI vs leiden clusters: {adjusted_rand_score(cluster, tomo_assign):.4f}")
