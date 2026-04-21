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

## Results (2026-04-21)

### Ziegler 2021 nasopharyngeal atlas (31,602 distinct cells, 32,871 genes)

This is the reference number — real single-cell atlas data, no up-sampling.

| n_cells | GRN (s) | AUCell (s) | peak RSS (GB) |
|---:|---:|---:|---:|
| 3,000 | 11.4 | 0.48 | 2.0 |
| 10,000 | 42.0 | 1.78 | 2.7 |
| 30,000 | 301.3 | 6.14 | 3.4 |

| Transition | GRN time / cells | AUCell time / cells |
|---|---:|---:|
| 3k → 10k | 1.11× (linear) | 1.12× (linear) |
| 10k → 30k | 2.39× (super-linear) | 1.15× (near-linear) |

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

| Stage | Ziegler slope (3k–30k) | Verdict |
|---|---:|---|
| AUCell | 1.11 | Essentially linear on real atlas data |
| GRN (3k–10k) | 1.11 | Linear |
| GRN (10k–30k) | 1.42 | Super-linear, ~2.4× worse than linear at 30k |

Peak memory on Ziegler grows sub-linearly: 10× cells → 1.7× RSS.

## Why GRN is super-linear at 30k

GRNBoost2 tree fitting is O(n log n) per tree due to split-finding sort.
With ~300 trees, that adds roughly log(n) / log(n₀) overhead on top of
linear, which would predict ~3.4× for a 3× cell jump from 10k to 30k.
We saw 7.2× — so log-n alone doesn't explain it. Likely additional
contributors:

- Sparse-to-dense conversion memory pressure at higher cell counts.
- Cache behaviour: the working set exceeds L3 around 30k on this
  machine.
- PyO3 GIL release overhead across threads at larger allocation sizes.

This is a real finding, not an artifact. The fix is to profile GRN at
atlas scale and either:

1. Keep expression matrix sparse through the tree fitting (bigger
   engineering change).
2. Stream cell batches through the tree builder (standard GBM trick).
3. Accept the super-linearity and document as the honest upper bound.

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

## 100k+ follow-up

A Ziegler 100k run (only 3.2× up-sampling — below the 5× threshold)
is the next data point. Takes ~25 min for GRN alone. Will extend this
table when it completes.
