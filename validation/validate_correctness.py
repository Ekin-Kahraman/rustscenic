"""Real validation — tests that don't depend on external downloads.
  1. Null test: shuffle target expression → importance must collapse.
  2. Seed stability: 3 seeds, pairwise top-10 overlap per target.
  3. Expanded gold standard (40 edges, literature-sourced BEFORE looking at our output)
     computed against both rustscenic and arboreto. Compare recall.
"""
import time
import numpy as np
import pandas as pd
import anndata as ad
import rustscenic
import rustscenic.grn

ADATA = ad.read_h5ad("/Users/ekin/projects/bio/rustscenic/validation/reference/data/pbmc3k.h5ad")
TFS = [t for t in rustscenic.grn.load_tfs(
    "/Users/ekin/projects/bio/rustscenic/validation/reference/data/allTFs_hg38.txt"
) if t in set(ADATA.var_names)]
GENES = set(ADATA.var_names)

# Expanded gold-standard of 40+ edges, sourced from classic literature
# BEFORE looking at our output. Mix of lineages so we can't cherry-pick.
# Source references in brackets after each TF.
GOLD_STANDARD = [
    # Myeloid (SCENIC 2017 supp, Aibar et al.; textbook immunology)
    ("SPI1", "CST3"), ("SPI1", "FCER1G"), ("SPI1", "LGALS1"),
    ("SPI1", "TYROBP"), ("SPI1", "SAT1"), ("SPI1", "LYZ"),
    ("SPI1", "AIF1"), ("SPI1", "PSAP"), ("SPI1", "CTSS"),
    ("SPI1", "CSF1R"), ("SPI1", "CD14"), ("SPI1", "CD68"),
    ("CEBPD", "TYROBP"), ("CEBPD", "LYZ"), ("CEBPD", "PSAP"),
    ("CEBPD", "S100A9"), ("CEBPD", "S100A8"),
    ("MAFB", "CST3"), ("MAFB", "AIF1"), ("MAFB", "LYZ"),
    ("CEBPB", "PSAP"), ("CEBPB", "LGALS1"),
    ("KLF4", "LYZ"), ("KLF4", "PSAP"),
    # MHC class II regulation (well-characterized, SPI1+CIITA hub)
    ("SPI1", "HLA-DRB1"), ("SPI1", "HLA-DRA"), ("SPI1", "CD74"),
    # B-cell (classic)
    ("PAX5", "CD19"), ("PAX5", "MS4A1"),
    ("EBF1", "CD79A"), ("EBF1", "CD79B"),
    # T-cell / lymphoid
    ("TCF7", "LEF1"), ("TCF7", "CCR7"),
    ("LEF1", "CD3D"),  # weaker but cited
    # NK
    ("TBX21", "GZMB"),
    ("EOMES", "PRF1"),
    # Erythroid / generic master regulators (may not be strong in PBMC but present)
    ("GATA3", "IL2RA"),  # T reg / Th2
    # Inflammation / IRF family
    ("IRF7", "ISG15"), ("IRF7", "IFI6"),
    ("IRF8", "HLA-DRB1"), ("IRF8", "CD74"),
    # Activator protein 1 (AP-1) complex
    ("JUN",  "FOS"),
    ("FOS",  "JUN"),
]
# Keep only edges where both TF and target are in our matrix
GOLD_STANDARD = [(tf, t) for tf, t in GOLD_STANDARD if tf in GENES and t in GENES and tf in set(TFS)]
print(f"expanded gold standard: {len(GOLD_STANDARD)} edges with both ends in pbmc3k\n")

# ---------------------------------------------------------------------------
# TEST 1: NULL TEST
# ---------------------------------------------------------------------------
print("=" * 80)
print("TEST 1: NULL TEST (shuffle target expr, importance must collapse)")
print("=" * 80)
# Run on real data first for baseline (same as v0.1 output, just cached)
t0 = time.monotonic()
real_df = rustscenic.grn.infer(ADATA, TFS, seed=777, n_estimators=300, early_stop_window=25)
print(f"real run: {time.monotonic()-t0:.1f}s, {len(real_df)} edges")

X = ADATA.X.toarray() if hasattr(ADATA.X, "toarray") else np.asarray(ADATA.X)
X = np.asarray(X, dtype=np.float32).copy()

for tgt in ["TYROBP", "LYZ", "CST3"]:
    ti = list(ADATA.var_names).index(tgt)
    X_shuf = X.copy()
    rng = np.random.default_rng(0)
    X_shuf[:, ti] = rng.permutation(X_shuf[:, ti])
    shuf = rustscenic.grn.infer(
        (X_shuf, list(ADATA.var_names)), TFS,
        seed=777, n_estimators=300, early_stop_window=25,
    )
    real_top = real_df[real_df["target"] == tgt]["importance"].max()
    shuf_row = shuf[shuf["target"] == tgt]["importance"]
    shuf_top = float(shuf_row.max()) if len(shuf_row) else 0.0
    ratio = shuf_top / max(real_top, 1e-9)
    ok = ratio < 0.3
    status = "PASS" if ok else "FAIL"
    print(f"  {tgt:8}  real top imp={real_top:.2f}  shuffled={shuf_top:.2f}  "
          f"ratio={ratio:.3f}  {status}")

# ---------------------------------------------------------------------------
# TEST 2: SEED STABILITY
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 2: SEED STABILITY (top-10 TF overlap across 3 seeds)")
print("=" * 80)
SEEDS = [777, 42, 123456]
seed_outputs = {}
for s in SEEDS:
    t0 = time.monotonic()
    seed_outputs[s] = rustscenic.grn.infer(ADATA, TFS, seed=s, n_estimators=300, early_stop_window=25)
    print(f"  seed={s}: {time.monotonic()-t0:.1f}s, {len(seed_outputs[s])} edges")

TARGETS = ["TYROBP", "LYZ", "CST3", "HLA-DRB1", "CD74", "SAT1", "FCER1G", "PSAP"]
overlaps_all = []
print("\n  pairwise top-10 TF overlap across seeds (of 10):")
print(f"  {'target':10} {'s1vs2':>6} {'s1vs3':>6} {'s2vs3':>6} {'mean':>6}")
for tgt in TARGETS:
    top10 = [set(seed_outputs[s][seed_outputs[s]["target"] == tgt]
                 .nlargest(10, "importance")["TF"].values) for s in SEEDS]
    pair_ov = [len(top10[0] & top10[1]), len(top10[0] & top10[2]), len(top10[1] & top10[2])]
    m = np.mean(pair_ov)
    overlaps_all.append(m)
    print(f"  {tgt:10} {pair_ov[0]:>6} {pair_ov[1]:>6} {pair_ov[2]:>6} {m:>6.1f}")
print(f"  overall mean: {np.mean(overlaps_all):.1f}/10")

# ---------------------------------------------------------------------------
# TEST 3: GOLD-STANDARD RECALL vs RANDOM
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TEST 3: GOLD-STANDARD RECALL (hand-curated literature edges, not our own)")
print("=" * 80)
def recall_at_k(df, edges, k, tf_col):
    tf_to_top = {}
    hits = []
    misses = []
    for tf, target in edges:
        if tf not in tf_to_top:
            tf_to_top[tf] = set(df[df[tf_col] == tf].nlargest(k, "importance")["target"].values)
        if target in tf_to_top[tf]:
            hits.append((tf, target))
        else:
            misses.append((tf, target))
    return hits, misses

arb_df = pd.read_parquet("/Users/ekin/projects/bio/rustscenic/validation/reference/data/pbmc3k_grn_full.parquet")
real_df_full = pd.read_parquet("/Users/ekin/projects/bio/rustscenic/validation/ours/pbmc3k_grn.parquet")

for k in [20, 100]:
    ours_h, ours_m = recall_at_k(real_df_full, GOLD_STANDARD, k, "TF")
    arb_h, arb_m = recall_at_k(arb_df, GOLD_STANDARD, k, "TF")
    random_expected = 100 * k / ADATA.n_vars
    print(f"  top-{k}:")
    print(f"    rustscenic recall: {len(ours_h)}/{len(GOLD_STANDARD)}  ({100*len(ours_h)/len(GOLD_STANDARD):.1f}%)")
    print(f"    arboreto   recall: {len(arb_h)}/{len(GOLD_STANDARD)}  ({100*len(arb_h)/len(GOLD_STANDARD):.1f}%)")
    print(f"    random-chance baseline: {random_expected:.2f}%")

# show which edges we miss vs arboreto for top-20
ours_h, ours_m = recall_at_k(real_df_full, GOLD_STANDARD, 20, "TF")
arb_h, arb_m = recall_at_k(arb_df, GOLD_STANDARD, 20, "TF")
print(f"\n  edges we miss in top-20 (that arboreto also misses):")
for e in ours_m:
    status = " (arboreto also misses)" if e in arb_m else " (arboreto hits)"
    print(f"    {e}{status}")
