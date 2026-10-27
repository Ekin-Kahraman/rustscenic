# 1.3 million real-cell benchmark

RustScenic achieved **3.06x faster GRN analysis and 78.7% lower peak measured
analysis memory** than arboreto's sparse Dask implementation on **1,306,127
real mouse-brain cells**, using ratios of medians across three matched seeds.
The short description is **3x faster with approximately 80% lower measured
memory on 1.3 million real cells**.

## Measurements

| Implementation | Median analysis time | Median peak analysis PSS | Completed seeds |
| --- | ---: | ---: | ---: |
| RustScenic 0.5.0 | 31m20s | 2.97 GB | 3 / 3 |
| Arboreto 0.1.6, sparse Dask | 96m00s | 13.93 GB | 3 / 3 |
| Arboreto 0.1.6, sparse multiprocessing | 92m12s | 6.52 GB | 3 / 3 |

All three paired comparisons favoured RustScenic: 3.01-3.06x faster than
the Dask implementation. The multiprocessing implementation produced the
same arboreto networks, with a separate 2.94x speedup and 54.5% lower PSS
for RustScenic. The approximately 80% memory claim is specific to Dask.

## Conditions and interpretation

- Official 10x E18 mouse-brain dataset: 1,306,127 distinct real cells,
  2,095 selected genes and 256 transcription factors.
- Same input and hardware, 16 CPUs, seeds 777, 778 and 779;
  Intel Xeon Platinum 8568Y+, Python 3.12.5, arboreto 0.1.6 and Dask 2024.4.1.
- Matrix-to-network GRN analysis on prepared RNA; preparation and input loading
  are excluded. Memory is peak sampled process-tree PSS, with decimal GB.
- 5,000-tree ceiling, learning rate 0.01, depth 3, feature fraction 0.1,
  subsample 0.9 and early-stop window 25. Methods can stop at different tree counts.
- Network rankings differ: candidate-edge Spearman 0.624-0.630 and top-50
  target-set mean Jaccard 0.402-0.404. Faster execution does not establish
  biological equivalence. No complete million-cell multiome workflow is claimed.

[Individual seed measurements](measurements.csv) · [Parameters and source hashes](summary.json) ·
[Benchmark harness](../../../validation/hpc/huangfu_atlas/METHODS.md)

The earlier 20,000-cell experiment remains a separate historical result:
[original measurements and conditions](../real-rna-20260828/benchmark-source-record.md).
