# Benchmarks

This page is the evidence behind the README claim. It keeps the marketing line
separate from the measurement detail: same inputs, fixed seed, stated hardware,
runtime, memory and output agreement.

RustScenic is benchmarked against SCENIC+ on the shared matrix-level regulatory
output path:

```text
RNA + ATAC + cistromes
  -> TF-to-gene
  -> region-to-gene
  -> eRegulons
  -> gene and region AUCell
```

This page does not benchmark raw fragment parsing, topic modelling, motif
database construction, or full workflow scheduling. Those are separate stages.

This is a practical output-path benchmark, not an algorithm-identical kernel
benchmark. RustScenic enhancer linking uses correlation over the fixed search
space. The SCENIC+ reference row uses GBM plus Pearson scoring for
region-to-gene links. Region-to-gene Jaccard below therefore means edge-set
agreement under the benchmark search space, not score-level identity.

## Setup

| Item | Value |
| --- | --- |
| Machine | Apple M5 laptop |
| RAM | 16 GB |
| OS | macOS arm64 |
| Python | RustScenic 3.13.9; SCENIC+ 3.11.8 |
| Threads | 4 CPU threads |
| Seed | 777 |
| RustScenic build | Release |
| Benchmark harness | `validation/head_to_head/bench_e2e.py` |
| Summary data | `validation/head_to_head/head_to_head_summary.json` |
| Provenance note | Raw local result JSONs are condensed into the committed summary; the ignored `validation/head_to_head/results/` directory is not required to read the public benchmark table. |

## Runtime

Rows can be sampled subsets; the shape column is the actual benchmark input.

| Dataset | Shape | RustScenic | SCENIC+ | Speedup | Peak RSS (RustScenic / SCENIC+) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic micro | 150 cells, 80 genes, 30 peaks, 3 TFs | 0.035 s | 9.45 s | 269x | 0.18 / 0.40 GB |
| Synthetic scale | 1,500 cells, 1,500 genes, 450 peaks, 12 TFs | 1.56 s | 146.5 s | 94x | 0.30 / 0.51 GB |
| PBMC3k | 1,500 cells, 2,000 genes, 3,000 peaks, 20 TFs | 1.93 s | 52.5 s | 27x | 0.84 / 0.96 GB |
| PBMC3k dense | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 4.98 s | 258.9 s | 52x | 1.21 / 1.26 GB |
| PBMC10k dense | 2,000 sampled cells, 4,000 genes, 8,000 peaks, 30 TFs | 21.5 s | 241.5 s | 11x | 2.37 / 2.63 GB |
| Mouse brain E18 | 1,500 cells, 3,000 genes, 6,000 peaks, 25 TFs | 2.82 s | 90.4 s | 32x | 1.65 / 2.10 GB |
| Human brain GEM-X | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 7.41 s | 146.0 s | 19.7x | 2.18 / 2.19 GB |

Real-data speedups in this set range from 11x to 52x. Median real-data speedup
is 27x. Peak RSS is comparable or lower in every real-data row, but the
reduction is modest: median SCENIC+ / RustScenic memory ratio is 1.15x.

For the human brain GEM-X row, including data preparation:

| Tool | Compute | Data prep | Total | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| RustScenic | 7.41 s | 4.48 s | 11.89 s | 2.18 GB |
| SCENIC+ | 145.97 s | 4.38 s | 150.36 s | 2.19 GB |

## Memory Scaling

The core E2E rows above show comparable or lower memory against SCENIC+ on
small real-data subsets. The stronger memory result comes from the atlas-scale
stage check archived in `validation/VALIDATION_SUMMARY.md`: a 100k-cell,
20,292-gene four-stage run peaked at 6.34 GB RSS, while legacy pySCENIC reports
exceed 40 GB on similar workloads.

| Workload | RustScenic peak RSS | Reference context |
| --- | ---: | --- |
| 100k cells x 20,292 genes, GRN + AUCell + topics + cisTarget | 6.34 GB | legacy pySCENIC reports exceed 40 GB on similar workloads |

## Validation

Output agreement is measured from saved benchmark signatures. Jaccard values
compare edge or TF sets. Pearson values compare per-cell AUCell vectors for
common TFs.

| Check | Synthetic micro | Human brain GEM-X |
| --- | ---: | ---: |
| TF-to-gene top-edge Jaccard | 0.988 | 0.537 |
| Region-to-gene edge-set Jaccard | 1.000 | 1.000 |
| eRegulon TF Jaccard | 1.000 | 0.840 |
| eRegulon edge Jaccard | 0.487 | 0.161 |
| Gene AUCell mean Pearson | 0.990 | 0.386 |
| Region AUCell mean Pearson | 0.970 | 0.823 |

Interpretation:

- Region-to-gene edge-set agreement is exact under the fixed search space used
  here; score-level identity is not claimed.
- Region AUCell agreement is strong on the real human brain row.
- TF-to-gene rankings are directionally aligned but not identical.
- eRegulon edges and gene AUCell are the main targets for the next parity pass.

## Interpretation

The benchmark set supports a direct message:

- RustScenic is substantially faster than SCENIC+ on the tested CPU matrix-level
  E2E workloads.
- The package runs this path without Java, dask, CUDA, or a Snakemake stack.
- Peak memory is lower or comparable in the tested real-data rows.
- The clearest current strength is faster local execution with a single modern
  Python install.

The strongest public claim today is faster, CPU-first multiome regulatory-network
analysis with tested core E2E speedups and a simpler installation path.

The next benchmark tier is aimed at larger real multiome inputs, repeated runs
and full workflow coverage, so the headline can move from core E2E performance
to broader external validation.

## Reproduce

Run RustScenic:

```bash
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python validation/head_to_head/bench_e2e.py \
  --tool rustscenic \
  --input-10x-h5 path/to/filtered_feature_bc_matrix.h5 \
  --dataset-name human_brain_gemx_10k_multiome_profile \
  --species hs \
  --n-cells 2000 \
  --n-genes 4000 \
  --n-peaks 8000 \
  --n-tfs 30 \
  --n-cpu 4 \
  --grn-estimators 5000 \
  --min-abs-corr 0.0 \
  --max-distance 1000000 \
  --save-signatures \
  --signature-top-n 50000 \
  --out validation/head_to_head/results/rustscenic.json
```

Run SCENIC+:

```bash
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python validation/head_to_head/bench_e2e.py \
  --tool scenicplus \
  --input-10x-h5 path/to/filtered_feature_bc_matrix.h5 \
  --dataset-name human_brain_gemx_10k_multiome_profile \
  --species hs \
  --n-cells 2000 \
  --n-genes 4000 \
  --n-peaks 8000 \
  --n-tfs 30 \
  --n-cpu 4 \
  --grn-estimators 5000 \
  --min-abs-corr 0.0 \
  --max-distance 1000000 \
  --save-signatures \
  --signature-top-n 50000 \
  --out validation/head_to_head/results/scenicplus.json
```

Compare signatures:

```bash
python validation/head_to_head/compare_e2e_outputs.py \
  --rust validation/head_to_head/results/rustscenic.json \
  --scenicplus validation/head_to_head/results/scenicplus.json \
  --out validation/head_to_head/results/compare.json
```

## Next Benchmarks

For the next evidence tier, repeat this benchmark on:

- more real multiome datasets;
- larger cell counts on the same command path;
- a second machine;
- repeated runs per dataset;
- full workflow runs that include fragments, topics and motif-ranking inputs.
