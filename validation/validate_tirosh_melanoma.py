"""Run rustscenic on Tirosh melanoma scRNA-seq. Test cancer-specific
master regulators: MITF (melanocyte), MYC (proliferation), plus immune
lineage TFs for the TME (TCF7/LEF1 T cells, PAX5 B cells, etc.).
"""
import time
import numpy as np
import pandas as pd
import anndata as ad
import rustscenic, rustscenic.grn, rustscenic.aucell

ADATA = "/Users/ekin/rustscenic/validation/reference/data/tirosh_melanoma.h5ad"
adata = ad.read_h5ad(ADATA)
print(f"Tirosh melanoma: {adata.shape}")

# Recode labels per Tirosh 2016 (malig col: 1=NON-malig, 2=malig, 0=unresolved)
# non_malignant codes: 1=T, 2=B, 3=Macrophage, 4=Endo, 5=CAF, 6=NK
labels = []
for m, nm in zip(adata.obs["malignant"], adata.obs["non_malignant"]):
    if m == "2":
        labels.append("malignant")
    elif m == "1":
        labels.append({
            "1": "T", "2": "B", "3": "Macrophage", "4": "Endo", "5": "CAF", "6": "NK",
            "0": "unknown_nonmalig",
        }.get(nm, "unknown_nonmalig"))
    else:
        labels.append("unresolved")
adata.obs["cell_type"] = pd.Categorical(labels)
print(f"cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# TF list — aertslab hg38 plus melanoma-specific
tfs_all = rustscenic.grn.load_tfs("/Users/ekin/rustscenic/validation/reference/data/allTFs_hg38.txt")
tfs_in = [t for t in tfs_all if t in set(adata.var_names)]
# Verify melanoma canonical TFs are present
CANON = ["MITF", "SOX10", "MYC", "TCF7", "LEF1", "PAX5", "EBF1", "TBX21", "EOMES",
         "CEBPD", "CEBPB", "SPI1", "IRF8", "STAT1", "STAT3", "NFKB1", "TP53"]
present = [t for t in CANON if t in tfs_in]
print(f"TFs total: {len(tfs_in)}   canonical melanoma-TME TFs present: {present}")

print("\n--- rustscenic.grn (Tirosh) ---")
t0 = time.monotonic()
grn = rustscenic.grn.infer(adata, tfs_in, seed=777, n_estimators=300, early_stop_window=25)
t_grn = time.monotonic() - t0
print(f"  wall: {t_grn:.1f}s   edges: {len(grn)}")

# Build regulons, score with aucell
regs = []
for tf, grp in grn.groupby("TF"):
    top = grp.nlargest(50, "importance")["target"].tolist()
    if len(top) >= 10:
        regs.append((f"{tf}_regulon", top))
print(f"\n--- aucell ---  {len(regs)} regulons")
t0 = time.monotonic()
auc = rustscenic.aucell.score(adata, regs, top_frac=0.05)
t_aucell = time.monotonic() - t0
print(f"  wall: {t_aucell:.1f}s   shape {auc.shape}")

# Canonical regulon activity by cell type
CANON_MAP = {
    "MITF":  "malignant",       # master regulator of melanocyte/melanoma
    "SOX10": "malignant",       # neural crest / melanocyte
    "MYC":   "malignant",       # proliferation
    "TCF7":  "T",
    "LEF1":  "T",
    "PAX5":  "B",
    "EBF1":  "B",
    "TBX21": "NK",
    "EOMES": "NK",
    "CEBPD": "Macrophage",
    "SPI1":  "Macrophage",
    "IRF8":  "Macrophage",
    "STAT1": "T",               # IFN signaling — T dominantly
}
print("\nCanonical regulon lineage check:")
correct = 0
considered = 0
for tf, expected in CANON_MAP.items():
    col = f"{tf}_regulon"
    if col not in auc.columns:
        continue
    means = auc[col].groupby(adata.obs["cell_type"], observed=True).mean().sort_values(ascending=False)
    top3 = list(means.head(3).index)
    hit = expected in top3
    considered += 1
    correct += int(hit)
    print(f"  {tf:6}  expected {expected:12s}  top-3 {top3}  {'HIT' if hit else 'miss'}")
print(f"\n{correct}/{considered} canonical TF lineages placed in their expected cell type's top-3 clusters")

# Specifically — does MITF fire highest in malignant? (melanoma-identity test)
if "MITF_regulon" in auc.columns:
    mitf_means = auc["MITF_regulon"].groupby(adata.obs["cell_type"], observed=True).mean()
    malig_val = mitf_means.get("malignant", 0)
    others = mitf_means.drop("malignant", errors="ignore")
    fold = malig_val / max(others.mean(), 1e-9)
    print(f"\nMITF regulon activity: malignant={malig_val:.3f}  non-malig mean={others.mean():.3f}  fold={fold:.2f}x")
    print(f"full per-cell-type: {mitf_means.to_dict()}")
