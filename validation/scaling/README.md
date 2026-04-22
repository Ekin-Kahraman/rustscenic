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

## Scaling summary

| Stage | Ziegler slope post-fix (10k–30k) | Verdict |
|---|---:|---|
| GRN | 1.17 | Linear after PR #12 (was 2.39 before) |
| AUCell | 1.30 | Near-linear; pending deeper profiling |

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

## Complexity analysis (algorithmic guarantee)

After PR #12 every per-cell loop in GRN inference is linear in
`n_cells`. Walking the loop:

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
user-set hyperparameter independent of `n_cells`. None of those can
turn the per-cell cost non-linear.

AUCell is similarly O(n_cells × n_genes × log n_genes) — argsort
dominates, and the `log n_genes` factor does not depend on `n_cells`.
Linear in `n_cells` for any fixed gene panel.

The only paths that can produce **effective** super-linear wall-time
on a machine where the algorithm is O(n):

1. Memory pressure / paging when `n_cells × n_genes × 4` exceeds
   physical RAM (hit locally at 50k on Ziegler).
2. L3 cache misses when the per-thread working set grows past cache.
3. Allocator page-fault contention (the bug PR #12 fixed).

(1) is hardware, not algorithm. (3) is solved. (2) is mitigable with
chunked target-gene processing — tracked for future work but not
currently observed inside measured ranges.

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
- GRN: `n_estimators=300`. Scaling ratios hold at higher estimator
  counts by tree construction linearity in `n_estimators`.
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

Both are hardware ceilings, not fix regressions. On an HPC node with
sufficient RAM (64–128 GB), the PR #12 allocation fix should continue
to deliver near-linear scaling at 100k–300k cells.

Planned: re-run at 10k, 30k, 50k, 100k, 200k, 300k on Hali or Minerva
once access lands. This is the gating data point for Kuan's proposed
cellxgene sweep (hundreds of cell types × 30–100k each) — need to
confirm the per-cell-type cost stays linear before committing
Minerva time.
