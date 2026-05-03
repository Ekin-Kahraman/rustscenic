"""Compare outputs of reference pipeline (arboreto+pyscenic+tomotopy) vs
rustscenic on same 10x Multiome data. Fair integration comparison."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import anndata as ad
import scanpy as sc

ADATA = ad.read_h5ad("/Users/ekin/projects/bio/rustscenic/validation/reference/data/multiome3k/rna.h5ad")
ATAC = ad.read_h5ad("/Users/ekin/projects/bio/rustscenic/validation/reference/data/multiome3k/atac_binarized.h5ad")
shared = sorted(set(ADATA.obs_names) & set(ATAC.obs_names))
ADATA = ADATA[shared]; ATAC = ATAC[shared]

# Reference outputs
ref_adj = pd.read_parquet("/Users/ekin/projects/bio/rustscenic/validation/ours/reference_pipeline_multiome.adj.parquet")
ref_auc = pd.read_parquet("/Users/ekin/projects/bio/rustscenic/validation/ours/reference_pipeline_multiome.auc.parquet")

# Our outputs — re-run grn on SAME data + aucell for fair comparison
import rustscenic.grn, rustscenic.aucell
tfs = [t for t in rustscenic.grn.load_tfs("/Users/ekin/projects/bio/rustscenic/validation/reference/data/allTFs_hg38.txt")
       if t in set(ADATA.var_names)]

print(f"Ref adjacencies: {len(ref_adj)}  {ref_adj.columns.tolist()}")

# Load our grn from the earlier multiome e2e run
import glob
our_grn_files = glob.glob("/Users/ekin/projects/bio/rustscenic/validation/ours/multiome*grn*")
if not our_grn_files:
    print("running rustscenic.grn on 10x Multiome RNA...")
    import time
    t0 = time.monotonic()
    our_adj = rustscenic.grn.infer(ADATA, tfs, seed=777, n_estimators=300, early_stop_window=25)
    print(f"  grn wall: {time.monotonic()-t0:.1f}s  edges: {len(our_adj)}")
    our_adj.to_parquet("/Users/ekin/projects/bio/rustscenic/validation/ours/multiome3k_grn_ours.parquet", index=False)
else:
    our_adj = pd.read_parquet(our_grn_files[0]).rename(columns={"tf": "TF"})
    print(f"loaded our grn: {len(our_adj)} edges from {our_grn_files[0]}")

# --- Compare adjacencies ---
print("\n========== GRN comparison: rustscenic vs arboreto ==========")
# Both use TF | target | importance
ref_top = ref_adj.sort_values("importance", ascending=False).head(10000)
our_top = our_adj.sort_values("importance", ascending=False).head(10000)
ref_edges = set(zip(ref_top["TF"], ref_top["target"]))
our_edges = set(zip(our_top["TF"], our_top["target"]))
jaccard = len(ref_edges & our_edges) / len(ref_edges | our_edges)
print(f"top-10k edge Jaccard: {jaccard:.3f}")

# Per-TF top-20 target overlap
shared_tfs = set(ref_adj["TF"]) & set(our_adj["TF"])
per_tf = []
for tf in shared_tfs:
    a = set(ref_adj[ref_adj["TF"] == tf].nlargest(20, "importance")["target"])
    b = set(our_adj[our_adj["TF"] == tf].nlargest(20, "importance")["target"])
    if a and b:
        per_tf.append(len(a & b) / max(len(a), len(b)))
print(f"per-TF top-20 overlap (mean across {len(per_tf)} shared TFs): {np.mean(per_tf):.3f}")

# --- Compare regulon activity (AUCell) ---
print("\n========== Regulon activity comparison ==========")
# ref_auc is cells × regulons from pyscenic
# Build our aucell on our_adj + run on same cells
from pyscenic.utils import modules_from_adjacencies
ex_mtx = ADATA.to_df()
our_modules = list(modules_from_adjacencies(our_adj, ex_mtx, thresholds=(), top_n_targets=(50,),
                                             top_n_regulators=(), min_genes=10, rho_mask_dropouts=False))
seen = set()
our_regs = []
for m in our_modules:
    if m.transcription_factor in seen:
        continue
    seen.add(m.transcription_factor)
    our_regs.append((f"{m.transcription_factor}_regulon", list(m.gene2weight.keys())))

our_auc = rustscenic.aucell.score(ADATA, our_regs, top_frac=0.05)
print(f"ref auc shape: {ref_auc.shape}   our auc shape: {our_auc.shape}")

common = sorted(set(ref_auc.columns) & set(our_auc.columns))
rhos = []
for r in common:
    a = ref_auc[r].values
    b = our_auc[r].values
    if len(a) != len(b):
        continue
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        continue
    rhos.append(float(np.corrcoef(a, b)[0, 1]))
print(f"per-regulon Pearson correlation across {len(rhos)} common regulons:")
print(f"  mean {np.mean(rhos):.3f}  median {np.median(rhos):.3f}  min {np.min(rhos):.3f}  max {np.max(rhos):.3f}")

# --- Biological agreement: do both pipelines find the same top-lineage TFs? ---
# Cluster cells first
sc.pp.highly_variable_genes(ADATA, n_top_genes=2000, flavor="seurat")
sc.tl.pca(ADATA, n_comps=30, mask_var="highly_variable")
sc.pp.neighbors(ADATA, n_neighbors=15)
sc.tl.leiden(ADATA, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
cluster = ADATA.obs["leiden"].astype(str).values

LINEAGE_CANON = {
    "SPI1":  ["CD14", "LYZ"], "CEBPD": ["CD14", "LYZ"], "PAX5": ["CD79A", "MS4A1"],
    "EBF1":  ["CD79A", "MS4A1"], "TCF7": ["CD3D", "IL7R"], "TBX21": ["NKG7", "GNLY"],
    "IRF8":  ["CD14", "FCER1A"],
}
# Compute ANOVA F-stat for each regulon across leiden clusters
from scipy.stats import f_oneway
def top_discriminative(auc, name):
    arr = auc.values
    Fs = []
    for r_idx, reg in enumerate(auc.columns):
        groups = [arr[cluster == c, r_idx] for c in np.unique(cluster) if np.sum(cluster == c) > 1]
        try:
            F, _ = f_oneway(*groups); Fs.append((reg, F))
        except Exception:
            pass
    Fs.sort(key=lambda x: -x[1])
    return Fs[:20]

ref_top_disc = top_discriminative(ref_auc, "ref")
our_top_disc = top_discriminative(our_auc, "ours")
print("\nTop-20 cluster-discriminative regulons (both pipelines):")
print(f"  ref (arboreto+pyscenic): {[x[0].replace('_regulon','') for x in ref_top_disc[:10]]}")
print(f"  ours (rustscenic):       {[x[0].replace('_regulon','') for x in our_top_disc[:10]]}")

ref_top_set = set(x[0] for x in ref_top_disc)
our_top_set = set(x[0] for x in our_top_disc)
top_overlap = len(ref_top_set & our_top_set)
print(f"overlap in top-20 discriminative regulons: {top_overlap}/20")
