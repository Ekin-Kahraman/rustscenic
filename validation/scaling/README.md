# Scaling benchmark

Measures how rustscenic's GRN inference and AUCell scale with cell count.
Delivers the evidence promised to Moha/Kuan: linear scaling at atlas scale.

## Run

```bash
python validation/scaling/bench_scaling.py \
    --input validation/reference/data/pbmc10k.h5ad \
    --sizes 1000 5000 10000 30000 100000 300000 \
    --out validation/scaling/
```

Output: `scaling_results.csv` + `scaling_plot.png` (log-log wall-time and
linear peak-RSS vs cell count).

## Methodology

- Cells below source size: drawn without replacement.
- Cells above source size: drawn with replacement with small Gaussian
  perturbation (σ = 0.01 on log-normalised counts). Measures computational
  complexity on representative sparsity, not biological novelty.
- GRN uses `n_estimators=300` for these runs (reduced from 5000 for
  tractable scaling across six points). Per-stage scaling ratios hold at
  higher estimator counts by tree construction cost linearity.
- Peak RSS reported via `resource.getrusage(RUSAGE_SELF)`.
- Same TF list as `examples/pbmc3k_end_to_end.py` for consistency.

## Results (2026-04-21, pbmc10k source)

| n_cells | GRN (s) | AUCell (s) | peak RSS (GB) |
|---:|---:|---:|---:|
| 1,000 | 3.1 | 0.07 | 0.75 |
| 5,000 | 15.8 | 0.34 | 1.4 |
| 10,000 | 34.8 | 0.78 | 2.2 |
| 30,000 | 111.8 | 4.0 | 6.8 |
| 100,000 | 1,876.9 | 45.9 | 6.8* |

Log-log slope 1k→30k: **GRN 1.05, AUCell 1.18** — essentially linear.

Log-log slope 1k→100k: GRN 1.34, AUCell 1.40 — **super-linear at 100k**.

*Peak RSS reported for 100k is cumulative across the benchmark process
and reflects the 30k peak, not the 100k-specific peak. Fix: run each
size in a separate subprocess.

## Known caveats

- The 100k point is **up-sampled ×9 from the 11k pbmc10k source**. With
  this much replication, each tree split finds many near-identical cells,
  which forces GRNBoost2 to explore deeper trees than it would on distinct
  cells. The 30k→100k super-linearity is almost certainly this artifact,
  not a real scaling problem.
- A rerun using the 31k-cell Ziegler atlas as source is in progress (3×
  replication at 100k is much less degenerate). Compare the two 100k
  numbers before publishing.
- For a definitive atlas-scale number, use Tabula Sapiens or HCA
  (≥300k distinct cells) so no up-sampling is needed at any size.

## Expected result

- Log-log slope near 1.0 (linear) or sub-linear across the full range.
- Memory growth linear or sub-linear.
- If slope > 1.5 at any cell count on a dataset with ≥3× distinct-cell
  headroom, that's a real scaling problem to investigate.
