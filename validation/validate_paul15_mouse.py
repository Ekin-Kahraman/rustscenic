"""Cross-species validation: rustscenic.grn on Paul15 mouse hematopoiesis.

Paul et al. 2015 (Cell) — scRNA-seq of mouse bone marrow myeloid progenitors,
3461 cells x 2730 genes, with classic lineage TFs well-characterized:
  GATA1 -> erythroid (HBB, KLF1)
  MPO / ELANE -> neutrophil (CEBPA, GFI1)
  PF4 / MPL -> megakaryocyte (RUNX1)
  IRF8 / CEBPB -> monocyte

Tests: does rustscenic produce biologically-sensible mouse regulons? Does
it find known master regulators (GATA1, CEBPA, IRF8) in the top TFs?
"""
import time
import numpy as np
import pandas as pd
import scanpy as sc
import rustscenic
import rustscenic.grn
import rustscenic.aucell

adata = sc.datasets.paul15()
print(f"Paul15 raw: {adata.shape}")
# preprocess
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"Paul15 post-QC: {adata.shape}")
print(f"cell types: {adata.obs['paul15_clusters'].value_counts().to_dict()}")

# Mouse TF list — use upper-case gene symbols in Paul15 (scanpy loads as capital)
# Actually Paul15 uses mouse symbols (capitalized first letter only). Let's check:
print(f"sample gene names: {list(adata.var_names[:5])}")

# Select candidate mouse TFs — a focused known-TF list for mouse hematopoiesis
MOUSE_TFS = [
    "Gata1", "Gata2", "Klf1", "Tal1", "Fli1", "Erg",         # erythroid / megakaryocyte
    "Cebpa", "Cebpb", "Cebpe", "Gfi1", "Spi1",               # myeloid / granulocyte
    "Runx1", "Runx3", "Mpl",                                  # megakaryocyte
    "Irf8", "Mef2c", "Tcf4",                                  # monocyte / DC / progenitor
    "Ikzf1", "Ikzf2", "Bcl11a", "Pax5",                       # lymphoid
    "Myb", "Myc", "Hoxa9", "Hoxa10", "Meis1",                # stem / progenitor
    "Foxo1", "Foxo3", "Stat3", "Stat5a", "Stat5b",            # signaling
    "Jun", "Junb", "Fos", "Fosb", "Atf3",                     # AP-1
    "Zeb2", "Nfe2", "Lmo2", "Kit", "Epor",                    # additional
    "Cdk6", "Ccnd1", "Ccnd2", "Ccne1",                        # cell cycle
]
tfs_in = [t for t in MOUSE_TFS if t in set(adata.var_names)]
print(f"mouse TFs in Paul15: {len(tfs_in)}")
print(f"  present: {tfs_in}")

# --- run rustscenic.grn ---
print("\n--- rustscenic.grn (mouse) ---")
t0 = time.monotonic()
grn = rustscenic.grn.infer(adata, tfs_in, seed=777, n_estimators=300, early_stop_window=25)
wall = time.monotonic() - t0
print(f"wall: {wall:.1f}s  edges: {len(grn)}")

# Known biology: top regulators should match cell type
CANONICAL = {
    "Gata1": ["Klf1", "Hbb-b1", "Hbb-b2", "Hba-a1", "Hba-a2", "Gypa", "Alas2"],    # erythroid
    "Cebpa": ["Mpo", "Elane", "Prtn3", "Ctsg", "Csf1r", "Lyz1", "Lyz2"],             # myeloid
    "Irf8":  ["Csf1r", "Cd74", "H2-Aa", "H2-Ab1"],                                   # monocyte / MHC-II
    "Runx1": ["Mpl", "Pf4", "Itga2b", "Gp1ba"],                                       # megakaryocyte
    "Spi1":  ["Csf1r", "Lyz1", "Lyz2", "Cd74"],                                       # myeloid
}

print("\nCanonical biology check (is expected target gene in TF's top-30?):")
hits = 0
total = 0
for tf, expected in CANONICAL.items():
    if tf not in tfs_in:
        continue
    top30 = set(grn[grn["TF"] == tf].nlargest(30, "importance")["target"].values)
    for t in expected:
        if t not in adata.var_names:
            continue
        total += 1
        if t in top30:
            hits += 1
            status = "HIT"
        else:
            status = "miss"
        print(f"  {tf:8} -> {t:10}  {status}")
print(f"\n{hits}/{total} canonical mouse hematopoiesis edges in top-30")

# Cell-type discriminator: Gata1 top 5 targets in erythroid cells, etc.
print("\nPer-cell-type regulon activity (AUCell):")
regulons = []
for tf in tfs_in:
    tops = grn[grn["TF"] == tf].nlargest(30, "importance")["target"].tolist()
    if len(tops) >= 10:
        regulons.append((f"{tf}_regulon", tops))

if regulons:
    auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
    for tf_check in ["Gata1", "Cebpa", "Irf8", "Runx1"]:
        col = f"{tf_check}_regulon"
        if col not in auc.columns:
            continue
        means = auc[col].groupby(adata.obs["paul15_clusters"], observed=True).mean().sort_values(ascending=False)
        print(f"  {tf_check:8} top-5 cluster activity: {means.head(5).to_dict()}")
