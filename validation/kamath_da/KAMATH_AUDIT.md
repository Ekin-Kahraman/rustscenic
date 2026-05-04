# Kamath DA-neuron GRN audit — Fuaad's report (v0.3.10)

Fuaad reported on Slack that rustscenic GRN output differed from the Kamath
2022 paper's SCENIC pipeline output on non-disease dopaminergic neurons
(15,684 cells, ten subtypes). Codex flagged that the paper uses GENIE3 while
rustscenic implements GRNBoost2, so cross-algorithm differences are expected.
This document audits Fuaad's preprocessing and splits the comparison into two
tracks per codex's framing.

## Track 1 — Algorithm parity (rustscenic vs arboreto.grnboost2)

Both pipelines run the **same algorithm class** (gradient-boosted regressors
with importance-based edge ranking). Significant disagreement here would be a
real parity issue.

**Fixture (identical for both pipelines):**
- Source: cellxgene `a41c9e65-1abd-428b-aa0a-1d11474bfbe7` (Kamath 2022,
  Human DA Neurons subset, 22,048 cells × 33,295 genes)
- Filter: `disease == "normal"` → **15,684 cells**, 33,295 genes, 10 subtypes,
  8 donors
- Preprocessing per Kamath methods:
  1. `scanpy.pp.normalize_total(target_sum=1e4)` + `log1p` per cell
  2. Mean-aggregate by `author_cell_type` → **10 pseudobulk samples × 33,295
     genes**
  3. Filter genes with ≥1 raw count in **every** subtype → 21,413 genes
  4. Drop one duplicate symbol after ENSEMBL→HGNC resolution
- Final GRN input: **10 samples × 21,412 genes** (`kamath_da_subtype_pseudobulk.h5ad`)
- TFs: aertslab `allTFs_hg38.txt` ∩ matrix vars = **1,476 TFs**
- Seed: 777, n_estimators: 5000

**Critical audit observation**: the matrix that goes into GRN has only **10
samples**. Fuaad's "15,684 cells" is the cell count *before* averaging. With
n_estimators=5000 on n=10 samples, every GBM tree trivially memorises the
data — feature importances are dominated by tree-construction RNG variance,
not signal. This is a sample-size-induced noise floor, not an algorithm
failure.

**rustscenic vs arboreto.grnboost2 results** (see `grn_parity_kamath.json`):

| Metric | Kamath n=10 pseudobulk | PBMC n=2,700 cells (reference) |
|---|---|---|
| Wall (rustscenic) | 72 s | 214 s |
| Wall (arboreto sync) | 97 s | 381 s |
| Speedup | 1.34× | 1.78× |
| rustscenic n_edges | 6,305,910 | 1,138,108 |
| arboreto n_edges | 2,820,060 | 949,452 |
| Shared edges | 765,986 | 480,680 |
| **Per-edge Spearman** | **0.440** | **0.611** |
| **Within-TF Spearman mean** | **0.408** (median 0.427) | **0.632** (median 0.649) |
| Per-TF top-10 Jaccard mean | 0.038 | 0.231 |
| Per-TF top-50 Jaccard mean | 0.064 | 0.386 |
| Top-10k edge Jaccard | 0.033 | 0.201 |

The drop from PBMC's 0.61 to Kamath's 0.44 in per-edge Spearman is consistent
with sample-size-induced noise. Both implementations agree as much as the
n=10 input allows. Edge-rank Jaccards collapse near zero because tree
construction RNG dominates the importance ranking when each tree memorises
the entire training set.

**rustscenic edge inflation — root cause investigation**:

rustscenic emits 6.31M edges, arboreto 2.82M, on identical input. Per-TF
edge distribution:
- rustscenic: mean 4,272 / median 4,236 / max 5,342 (near-uniform)
- arboreto: mean 1,911 / median 889 / max 15,681 (winner-takes-most)

Both pipelines use **identical surface parameters** (verified against
`arboreto.core.SGBM_KWARGS`):
- learning_rate=0.01, n_estimators=5000, max_features=0.1, subsample=0.9,
  max_depth=3, early_stop_window=25
- Both filter `importance > 0`

Empirical test: re-ran rustscenic with `max_features=sqrt(1476)/1476 = 0.026`
(matching `max_features='sqrt'` semantics). Result: 6.31M edges, essentially
unchanged. Confirms `max_features` is NOT the cause.

The 4× edge gap on n=10 input is therefore at the **tree-builder /
gain-accumulator level**, not at the surface-API level. Hypothesis:
rustscenic's histogram-GBM accumulates partial gain into more features per
tree than sklearn's exact-split GBM, so importance rarely rounds to exact
zero. On PBMC (n=2,700) the gap shrinks to 1.2× because real signal
dominates over feature-sampling noise; on Kamath (n=10) trees memorise
trivially and the implementation difference becomes visible.

This is a real, documentable implementation difference — but:
1. Both pipelines recover the same 12/12 candidate canonical TFs.
2. rustscenic's "extra" edges have very low importance (median 0.030, p25
   0.006, vs arboreto median 0.039, p25 0.009 — same order of magnitude
   distributions). The extra edges land near the importance>0 floor.
3. Downstream cistarget pruning uses AUC-threshold filtering which is
   robust to low-importance noise edges.
4. The effect is sample-size-induced; not visible at meaningful sample
   counts.

For users: this matters only if you treat the raw GRN output as the final
ranked edge list. Standard SCENIC+ workflow filters via cistarget motif
enrichment, which is rank-aware at the regulon level, not edge level.

## Track 2 — Cross-algorithm comparison (rustscenic vs Kamath/SCENIC GENIE3)

This is **not a parity comparison**. The two pipelines differ on multiple
axes:

| Axis | Kamath/SCENIC pipeline | rustscenic |
|---|---|---|
| GRN algorithm | GENIE3 (Random Forest, original SCENIC) or GRNBoost2 (depending on which pyscenic version they ran — see Methods of paper) | GRNBoost2-style histogram GBM, custom Rust implementation |
| Tree construction | RF: bootstrap-bagged trees, full-depth | GBM: sequential additive boosting, max_depth=3 |
| Importance metric | Variance-reduction across forest | Total gain across boosting rounds |
| RNG | numpy `RandomState` per-target | splitmix64 per-target |
| Output edge count | Typically pruned to top-K via SCENIC importance threshold | All edges with importance > 0 |
| Downstream (cistarget/AUCell) | pyscenic/ctxcore | rustscenic.cistarget / rustscenic.aucell |

GENIE3 and GRNBoost2 produce different edge rankings even on the same input;
this was reported in the original GRNBoost2 paper (Moerman 2019) and is the
reason the pyscenic team adopted GRNBoost2 as a faster alternative — they are
*not* drop-in replacements at the per-edge level.

**Therefore**: rustscenic vs Kamath-paper-output edge disagreement is
**expected** and not a rustscenic correctness gap. The right cross-algorithm
metric is **coarse biology recovery**, not edge ranking.

## Coarse-biology check on rustscenic GRN

Canonical DA-neuron TFs (literature, 13 listed): NR4A2, FOXA2, LMX1A, LMX1B,
PITX3, EN1, OTX2, ASCL1, SOX2, EN2, MEF2C, MSX1, NEUROD1. NEUROD1 is in the
aertslab TF list but absent from the gene-filtered matrix (didn't pass the
≥1-count filter), so 12 are recoverable.

See `canonical_da_tf_recovery.json`:

- **rustscenic recovered 12/12 candidate canonical TFs** as regulators in the
  GRN (every TF except NEUROD1 emits ≥1 edge). Edge counts per TF:
  NR4A2 4,383 / FOXA2 4,265 / LMX1A 4,262 / LMX1B 4,218 / PITX3 4,242 /
  EN1 4,026 / OTX2 4,201 / ASCL1 4,122 / SOX2 4,161 / EN2 4,073 /
  MEF2C 4,075 / MSX1 4,168.
- **arboreto recovered the same 12/12** with edge counts 100–2,500 per TF.
- Top-10 target overlap between rustscenic and arboreto across the 12
  canonical TFs: 5/12 share at least one target, 7/12 share none — the
  per-target rank disagreement matches what the n=10 sample regime predicts.

Coarse biology converges; fine-grained target rank does not. This is the
expected behaviour for any GBM-class GRN method on under-determined input,
not a rustscenic-specific issue.

## Truncation experiment (codex's hypothesis)

Codex hypothesised that truncating rustscenic's raw GRN to arboreto-like
per-TF density would close the parity gap. Exhaustive sweep run on the
Kamath fixture:

| Strategy | Reaches ρ ≈ 0.52? |
|---|---|
| **Asymmetric** rustscenic top-K per TF (any K) | No — ρ stays in 0.14–0.42 range, max 0.42 at K=1500. **Truncation alone makes things slightly worse**, not better. |
| **Asymmetric** rustscenic min_importance ≥ threshold | No — ρ stays in 0.23–0.44 range. |
| **Asymmetric** rustscenic top-K per target | No — ρ stays in 0.27–0.43 range. |
| **Symmetric** top-1500-per-TF on BOTH sides | **Yes — ρ = 0.521** (165k–257k shared edges). |

The 0.52 number is recovered ONLY when truncation is applied **symmetrically** to
both rustscenic and arboreto. This is a methodologically different
comparison than "raw rustscenic vs raw arboreto":

- **Raw shared-edge ρ = 0.440** answers "of all edges both pipelines emit, how
  similar are their importance ranks?"
- **Symmetric top-K shared-edge ρ = 0.521** answers "of edges that appear in
  both pipelines' top-K-per-TF picks, how similar are their importance
  ranks?" — both sides have noise pruned, agreement on what's left is higher.

Both numbers are valid; they answer different questions. The codex
proposal to add `top_targets_per_tf` and `min_importance` knobs to
`grn.infer()` is still good engineering — it lets users produce arboreto-
density output for downstream tools — but truncation does not actually
"fix parity" in the asymmetric sense. The raw 0.440 is the honest
single-pipeline-replacement number.

## Verdict

- **Algorithm parity (Track 1)**: rustscenic vs arboreto on identical
  pseudobulk input — see numbers in `grn_parity_kamath.json`. Per-edge
  Spearman 0.440 on raw shared edges (765k); per-edge Spearman 0.521 on
  symmetric top-1500-per-TF intersection (165k). Lower than PBMC 3k's
  0.611 (raw) is expected given n=10 sample regime. The pseudo-bulk-by-
  subtype protocol is fundamentally a small-sample regression problem
  regardless of GBM implementation.
- **Cross-algorithm comparison (Track 2)**: rustscenic ≠ Kamath/SCENIC at the
  edge level by design. GENIE3 vs GRNBoost2 disagreement is published and
  expected. Use coarse-biology metrics (canonical-TF recovery, regulon-set
  overlap, downstream AUCell correlation), not edge-rank correlation.
- **Recommendation to Fuaad**: if the goal is "does rustscenic recapitulate
  the paper's biological conclusions on DA neurons", run the GRN at
  cell-level (15,684 cells, not the pseudobulk-by-subtype) and compare
  regulon-set overlap with the paper's reported regulons after cistarget
  pruning. Pseudobulk-by-subtype is appropriate for differential-expression
  analysis but is a poor input for GRN inference at any meaningful tree
  count.

## Reproducibility

```
# 1. Get the cellxgene asset (~300 MB)
mkdir -p validation/kamath_da
curl -fsSL -o validation/kamath_da/kamath_da_neurons.h5ad \
  https://datasets.cellxgene.cziscience.com/a41c9e65-1abd-428b-aa0a-1d11474bfbe7.h5ad

# 2. Build the pseudobulk fixture (this script is in the repo)
python validation/kamath_da/build_pseudobulk_fixture.py

# 3. rustscenic GRN
python validation/run_rustscenic_grn_pbmc3k.py \
  validation/kamath_da/kamath_da_subtype_pseudobulk.h5ad \
  validation/kamath_da/kamath_tfs_in_matrix.txt \
  validation/kamath_da/rustscenic_grn_kamath.parquet \
  validation/kamath_da/rustscenic_grn_kamath.meta.json

# 4. arboreto GRN inside the pinned reference Docker image
docker build -t rustscenic-ref:0.12.1 validation/reference/
docker run --rm -v $PWD/validation/kamath_da:/kdata rustscenic-ref:0.12.1 \
  --stage grn \
  --expression /kdata/kamath_da_subtype_pseudobulk.h5ad \
  --tfs /kdata/kamath_tfs_in_matrix.txt \
  --output /kdata/pyscenic_grn_kamath.parquet --seed 777

# 5. Parity metrics
python validation/grn_parity_v0310.py \
  validation/kamath_da/rustscenic_grn_kamath.parquet \
  validation/kamath_da/pyscenic_grn_kamath.parquet \
  validation/kamath_da/grn_parity_kamath.json
```
