"""Validate rustscenic.cistarget at realistic JASPAR-like scale.

800 motifs x 20000 genes — approximates aertslab's cistarget feather DB
structure. For each of 100 planted regulons (TF + target gene set), we
assign the regulon's targets to ranks 1-100 of one specific motif. The
rest of that motif's ranking is random; other motifs' rankings are random.

Expected: rustscenic.cistarget.enrich should return the planted motif as
the top-AUC motif for each planted regulon.

Runtime target: < 5s on 10-core M-chip.
"""
import time
import numpy as np
import pandas as pd
import rustscenic.cistarget

N_MOTIFS = 800
N_GENES = 20000
N_REGULONS = 100
REGULON_SIZE = 50
SEED = 42

rng = np.random.default_rng(SEED)

motif_names = [f"motif_{m:04d}" for m in range(N_MOTIFS)]
gene_names = [f"g{i:05d}" for i in range(N_GENES)]

# planted: regulon r is paired with motif r (for r < N_REGULONS)
# each regulon has REGULON_SIZE target genes
planted_pairs = []
regulons = []
for r in range(N_REGULONS):
    paired_motif = r
    # Pick REGULON_SIZE genes; these will appear at top of paired motif's ranking
    genes = rng.choice(N_GENES, size=REGULON_SIZE, replace=False)
    regulons.append((f"regulon_{r}", [gene_names[g] for g in genes]))
    planted_pairs.append((f"regulon_{r}", f"motif_{paired_motif:04d}", set(genes)))

# Build rankings: for each motif, default = random permutation of 1..N_GENES
rankings = np.zeros((N_MOTIFS, N_GENES), dtype=np.int32)
for m in range(N_MOTIFS):
    perm = rng.permutation(N_GENES)
    rankings[m, perm] = np.arange(1, N_GENES + 1)

# Planted: for each regulon's paired motif, force regulon genes to ranks 1..REGULON_SIZE
for r in range(N_REGULONS):
    _, _, genes = planted_pairs[r]
    m = r  # paired motif
    # current ranks of those genes on motif m; we'll SWAP them into top
    current_ranks = rankings[m, list(genes)]
    # the genes currently at ranks 1..REGULON_SIZE become displaced
    top_gene_indices = np.where(rankings[m] <= REGULON_SIZE)[0]
    # Swap ranks so regulon genes get 1..REGULON_SIZE
    swap_targets = [g for g in top_gene_indices if g not in genes]
    for new_gene, displaced in zip(list(genes), swap_targets):
        old_rank = rankings[m, new_gene]
        new_rank = rankings[m, displaced]
        rankings[m, new_gene] = new_rank
        rankings[m, displaced] = old_rank

rank_df = pd.DataFrame(rankings, index=motif_names, columns=gene_names)
print(f"motif ranking DB: {rank_df.shape}")

t0 = time.monotonic()
out = rustscenic.cistarget.enrich(rank_df, regulons, top_frac=0.01, auc_threshold=0.0)
wall = time.monotonic() - t0
print(f"rustscenic.cistarget.enrich: {wall:.2f}s  ({len(out)} enriched pairs returned)")

# Per-regulon, find the top-AUC motif; check if it matches the planted one
correct = 0
top1_auc_sum = 0.0
misses = []
for r, planted_motif, _ in planted_pairs:
    regulon_rows = out[out["regulon"] == r].sort_values("auc", ascending=False)
    if regulon_rows.empty:
        misses.append((r, "no output"))
        continue
    top_motif = regulon_rows.iloc[0]["motif"]
    top_auc = regulon_rows.iloc[0]["auc"]
    if top_motif == planted_motif:
        correct += 1
        top1_auc_sum += top_auc
    else:
        planted_row = regulon_rows[regulon_rows["motif"] == planted_motif]
        planted_rank = (regulon_rows["motif"].values == planted_motif).argmax() + 1
        planted_auc = planted_row.iloc[0]["auc"] if len(planted_row) else 0.0
        misses.append((r, top_motif, top_auc, planted_motif, planted_auc, planted_rank))

print(f"\n{correct}/{N_REGULONS} regulons correctly recover their planted motif at top-1")
if correct > 0:
    print(f"  mean top-1 AUC for correct: {top1_auc_sum/correct:.3f}")
if misses[:3]:
    print(f"\nFirst 3 misses:")
    for m in misses[:3]:
        print(f"  {m}")
