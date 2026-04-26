# Scaling benchmark

Measures how rustscenic's GRN inference and AUCell scale with cell count.
Delivers the evidence promised to Moha/Kuan: scaling behaviour at atlas scale.

## Run

```bash
# Atlas-scale source (31k distinct cells)
python validation/scaling/bench_scaling.py \
    --input /path/to/ziegler2021_nasopharyngeal.h5ad \
    --sizes 3000 10000 30000 \
    --n-estimators 300 \
    --out validation/scaling/

# Smaller source, for lighter/faster CI-style checks
python validation/scaling/bench_scaling.py \
    --input validation/reference/data/pbmc10k.h5ad \
    --sizes 1000 5000 10000 30000 \
    --n-estimators 300 \
    --out validation/scaling/
```

Each cell count runs in a fresh subprocess so `ru_maxrss` reports a clean
per-size peak rather than the cumulative peak across the whole benchmark.

## Results

### Ziegler 2021 nasopharyngeal atlas (31,602 distinct cells, 32,871 genes)

This is the reference number — real single-cell atlas data, no up-sampling.

#### Pre-fix (2026-04-21)

| n_cells | GRN (s) | AUCell (s) | peak RSS (GB) |
|---:|---:|---:|---:|
| 3,000 | 11.4 | 0.48 | 2.0 |
| 10,000 | 42.0 | 1.78 | 2.7 |
| 30,000 | 301.3 | 6.14 | 3.4 |
| 50,000 | 697.7 | 27.95 | 5.6 |

10k → 30k showed GRN 2.39× over linear — a real super-linear regime. Diagnosed
as per-split `Vec<usize>` allocation churn compounding under rayon contention.
See `crates/rustscenic-grn/src/tree.rs` history for the fix (PR #12).

#### Post-fix (2026-04-22)

With the pooled `TreeScratch::partition_bufs` allocator change:

| n_cells | GRN (s) | AUCell (s) | peak RSS (GB) |
|---:|---:|---:|---:|
| 10,000 | 39.5 | 1.67 | 2.9 |
| 30,000 | 139.2 | 6.50 | 3.3 |

| Transition | GRN pre | GRN post | AUCell post |
|---|---:|---:|---:|
| 10k → 30k | **2.39× over linear** | **1.17× over linear** | 1.30× over linear |

GRN at 30k: **2.16× faster** (301s → 139s). Super-linearity eliminated.

50k post-fix couldn't complete on this 32 GB laptop — during the re-run the
system was already using 12.4 GB of 13.3 GB swap, so any 50k+ cell run thrashed.
That's a hardware ceiling, not a fix regression. A clean 50k+ run belongs
on HPC (Hali or Minerva) where RAM headroom removes the swap confound.

### pbmc10k (11k source, up-sampled above)

| n_cells | GRN (s) | AUCell (s) | peak RSS (GB)* |
|---:|---:|---:|---:|
| 1,000 | 3.1 | 0.07 | 0.75 |
| 5,000 | 15.8 | 0.34 | 1.4 |
| 10,000 | 34.8 | 0.78 | 2.2 |
| 30,000 | 111.8 | 4.0 | 6.9 |
| 100,000 | 1,876.9 | 45.9 | 6.9* |

\* RSS above 10k in the pbmc10k table is from the older cumulative-RSS
runner and reflects peak across the whole benchmark, not per-size.

### Microglia atlas 91k (real cellxgene, 58,232 genes, 50 TFs)

This is the counterexample to the earlier "linear at atlas scale" claim.
The run uses a real 91,838-cell cellxgene microglia atlas, the first 50
bundled human TFs present in the matrix, and `n_estimators=20`.
Raw JSON: [`microglia_91k_grn_scaling.json`](microglia_91k_grn_scaling.json).

| n_cells | GRN (s) | AUCell (s) | peak RSS (GB) | GRN per-cell |
|---:|---:|---:|---:|---:|
| 5,000 | 36.7 | 0.73 | 4.4 | 7.3 ms |
| 10,000 | 94.2 | 1.55 | 4.4 | 9.4 ms |
| 20,000 | 230.3 | 4.70 | 4.4 | 11.5 ms |
| 40,000 | 681.9 | 9.89 | 5.1 | 17.0 ms |
| 80,000 | 5,478.3 | 14.29 | 7.8 | 68.5 ms |
| 91,838 | 6,590.6 | 14.98 | 7.8 | 71.8 ms |

GRN log-log slope across the full run is **1.81**. The cliff is the
40k→80k transition: 8.0× wall-clock for 2× cells, segment slope 3.01.
AUCell does not show the same cliff.

Interpretation after audit: the cliff was not explained by allocator
churn alone. The dominant issue was row-major strided target extraction:
the old loop pulled one target gene at a time from a cells × genes dense
matrix, so each target caused a cache/TLB-hostile pass with stride
`n_genes`. Target blocking now materialises 64 consecutive targets at a
time into a compact column-major buffer.

Post-scratch sanity on the same atlas, before re-running the expensive
40k/80k/91.8k points:

| n_cells | pre-scratch GRN | post-scratch GRN | speedup |
|---:|---:|---:|---:|
| 5,000 | 36.7 s | 27.7 s | 1.33× |
| 10,000 | 94.2 s | 57.6 s | 1.64× |
| 20,000 | 230.3 s | 125.7 s | 1.83× |

The 5k→20k post-scratch slope was **1.09**, but a later 40k/80k/91.8k
rerun showed scratch-only did not fix atlas scale. Target blocking is the
actual atlas fix:

| n_cells | original GRN | scratch-only GRN | target-blocked GRN | speedup vs original |
|---:|---:|---:|---:|---:|
| 5,000 | 36.7 s | 27.7 s | 30.9 s | 1.19× |
| 10,000 | 94.2 s | 57.6 s | 64.1 s | 1.47× |
| 20,000 | 230.3 s | 125.7 s | 132.4 s | 1.74× |
| 40,000 | 681.9 s | 1,140 s hot-run | 287.7 s | 2.37× |
| 80,000 | 5,478.3 s | 6,300 s hot-run | 735.3 s | 7.45× |
| 91,838 | 6,590.6 s | 8,070 s hot-run | 864.1 s | 7.63× |

Post target-blocking slope across 5k→91.8k is **1.15**. The old
40k→80k segment was slope 3.01; target blocking reduces it to 1.35.
That is a substantial fix, not perfect linearity.

## Scaling summary

| Stage | Measured range | Slope / behavior | Verdict |
|---|---:|---|
| GRN | Ziegler 10k–30k, 21 TFs | 1.17 | Near-linear in this range |
| GRN | Gland Atlas 2k–40k, 6 TFs | 1.11 | Linear narrow-TF check |
| GRN | Microglia 5k–91.8k, 50 TFs | 1.81 | Super-linear at atlas scale; 40k→80k cliff |
| GRN | Microglia 5k–20k post-scratch, 50 TFs | 1.09 | Early fix signal; not full cliff proof |
| GRN | Microglia 5k–91.8k post-target-blocking, 50 TFs | 1.15 | Atlas cliff materially fixed; still mildly super-linear |
| AUCell | Microglia 40k→91.8k | 9.9s→15.0s | Not the bottleneck |

Peak memory on Ziegler grows sub-linearly: 3× cells → 1.14× RSS.

## Root cause of the original super-linearity

`tree::build_node_rec` allocated two fresh `Vec<usize>` partition buffers
per split node. For depth-3 trees that's 14 allocations per tree ×
`n_estimators` × ~20k targets = ~84M allocations per GRN run, with
buffer sizes growing with n_cells. Modern allocators served the
allocations fast, but populating fresh pages forced page-faults whose
cost compounded under rayon worker contention at larger cell counts.

PR #12 pooled the buffers in `TreeScratch` with `take_partition_buf` /
`return_partition_buf`. DFS keeps at most 2 × max_depth buffers live,
so the pool stabilises at 6 buffers per thread and never allocates
after warm-up. Also removed a redundant `root_samples.to_vec()` copy.

## Complexity analysis

After PR #12 every per-cell loop in GRN inference is linear in
`n_cells`, but measured wall-clock can still become super-linear when
the memory system dominates. Walking the loop:

| Stage | Cost | Depends on `n_cells` as |
|---|---|---|
| Feature binning (one-time) | `O(n_cells × n_features)` | linear |
| Sub-sampling rows (per tree) | `O(n_cells)` | linear |
| Residual update (per tree) | `O(n_cells)` | linear |
| Histogram accumulation (per split, per feature) | `O(n_samples_in_node)` | linear in `n_cells` at root, ≤ linear deeper |
| Tree prediction over training rows (per tree) | `O(n_cells × max_depth)` | linear (max_depth const.) |
| Partition samples → left/right (per split) | `O(n_samples)` with zero allocation after pool warm-up | linear |
| Per-target GBM total | `O(n_estimators × n_cells)` | linear |
| Full GRN, parallel over targets | `O(n_estimators × n_targets × n_cells / n_cores)` | **linear** |

Feature count `n_features` is bounded by the TF list length (typically
20–2,000, not `n_cells`-dependent). `max_depth` is a fixed
hyperparameter (3 for GRNBoost2 defaults). `n_estimators` is a
user-set hyperparameter independent of `n_cells`. These keep the
algorithmic work linear for fixed settings, but they do not guarantee
linear wall-clock on a laptop once per-worker buffers reach multi-MB
sizes and are churned across tens of thousands of target genes.

AUCell is similarly O(n_cells × n_genes × log n_genes) — argsort
dominates, and the `log n_genes` factor does not depend on `n_cells`.
Linear in `n_cells` for any fixed gene panel.

The only paths that can produce **effective** super-linear wall-time
on a machine where the algorithm is O(n):

1. Memory pressure / paging when `n_cells × n_genes × 4` exceeds
   physical RAM (hit locally at 50k on Ziegler).
2. L3 cache misses when the per-thread working set grows past cache.
3. Allocator page-fault contention (the bug PR #12 fixed).

(1) is hardware pressure, not a semantic bug. (3) was reduced by PR #12
inside each tree, and later by worker-local GBM scratch buffers. The
microglia 91k run exposed a larger memory-layout issue: one strided
row-major pass per target gene. Target blocking fixes the biggest part
of that by copying 64 target genes at a time into column-major buffers.

## CI regression test

`tests/test_scaling_regression.py` fits GRN at four cell counts
(1k / 2k / 4k / 8k) on synthetic data every CI run, fits a log-log
slope, and fails the build if the slope exceeds **1.30**. A clean
linear run is ≈ 1.0; PR #12's result is ~1.05–1.17. Anything above
1.3 means super-linearity has crept back in and the build stops.

Runs in ~15s on CI. The point isn't to benchmark, it's to catch
regressions the next time someone touches the inner loops.

## Up-sampling caveat

For cells above the source-dataset size, `subsample()` draws with
replacement and adds σ = 0.01 Gaussian noise. This preserves sparsity
and avoids literal duplicates, but very large replication ratios
(>5×) can still degrade tree split quality. That's what happened in
the pbmc10k 100k point (×9 replication).

Use a source dataset with at least 1/3 of the target cell count, or
document up-sampling explicitly when you cross that ratio.

## Methodology

- Cells below source size: drawn without replacement.
- Cells above source size: drawn with replacement + σ = 0.01 Gaussian
  perturbation on normalised counts.
- GRN: benchmark-specific `n_estimators`; do not extrapolate small-TF
  or low-cell-count slopes to full atlas runs without measuring.
- Peak RSS: `resource.getrusage(RUSAGE_SELF).ru_maxrss` from a fresh
  subprocess per cell count.
- TF list: same 21-TF union as `examples/pbmc3k_end_to_end.py`.

## pbmc10k post-fix cross-check (2026-04-22)

Same pooled-partition-buffer build, smaller-memory source (11k cells ×
20,292 genes). Confirms the fix lands on a different source too:

| n_cells | GRN pre (s) | GRN post (s) | speedup |
|---:|---:|---:|---:|
| 30,000 | 111.8 | 80.5 | **1.39×** |

Smaller speedup than on Ziegler (2.16×) because pbmc10k pre-fix was
already close to linear at 30k (slope 1.05) — there was less
super-linearity to fix. On Ziegler (slope 2.39 pre-fix), the fix
had more to bite into.

## 50k+ follow-up

The local 32 GB laptop swaps hard above 50k cells on the Ziegler
expression matrix (~6.6 GB for 50k × 32,871 genes, doubled during TF
extraction + binning). Clean scaling numbers above 30k need HPC.

Attempted locally (2026-04-22):

- Ziegler 50k: swap filled to 14/14 GB within 60 s of load; aborted.
- pbmc10k 100k: reached the tree-fitting phase but never finished
  inside 43 minutes before abort. Memory-bound at ~5 GB RSS with
  5–7 cores active, clearly paging.

Both are hardware ceilings, not fix regressions, but the later 91k
microglia run shows that sufficient RAM alone does not guarantee
near-linear GRN wall-clock. Re-run the 50k+ curve after each GRN
allocator/cache optimisation before making atlas-scale speed claims.

Planned: re-run at 10k, 30k, 50k, 100k, 200k, 300k on Hali or Minerva
once access lands. This is the gating data point for Kuan's proposed
cellxgene sweep (hundreds of cell types × 30–100k each) — need to
measure the per-cell-type cost after the worker-local buffer reuse
patch before committing Minerva time.
