# IFB real-RNA GRN validation

Date: 2026-08-28. These runs used RustScenic 0.5.0 from commit
`f8f605cc2ae3d942ecf33d76872e5a6e4110bfbc` on one IFB Slurm node. They
validate the corrected arboreto early-stop mode, signed regulon construction,
AUCell output and cell-count scaling on genuine scRNA-seq counts. They do not
validate the complete SCENIC+ motif, enhancer or spatial workflow.

The artefacts record `source_dirty=true` because the two benchmark scripts
were intentionally uploaded as untracked review changes. Remote `git status
--short` contained only those scripts; the installed RustScenic extension and
Python package were built from the recorded commit.

## Input and preparation

The source was the official 10x Genomics **1.3 Million Brain Cells from E18
Mice** dataset (1,306,127 cells). The original filtered count matrix contained
27,998 feature rows, 65 duplicate symbols and 2,624,828,308 non-zero counts.

| Input | SHA-256 |
| --- | --- |
| Full 10x H5, 4,216,018,749 bytes | `255a36ee92de25cb3568faa2c27d31fe6d0db30f285c5c977be8d6245de14044` |
| Official 20k-cell H5, 61,694,953 bytes | `ecaf7203c8f3d9b043bd8c0ad3a6b3b19b39422edd9ffce5649d1936a4e4fae7` |
| Prepared matrix | `5fea5a5772ef40e78cbc6a10317d69962edd45a8525630149ca62b58eb3d7c43` |

Feature selection was fixed before scaling: genes detected in at least 20 of
the official 20k sample cells were eligible; mitochondrial and ribosomal genes
were excluded from the highly-variable set; duplicate symbols were summed;
the selected union contained 5,000 HVGs and 1,264 expressed bundled mouse TFs,
or 5,858 genes. Counts were normalised per cell to 10,000 and transformed with
`log1p`. Nested cell prefixes use a deterministic permutation with seed
`20260828`.

Preparation was deterministic: two runs produced the same prepared-file hash.
The accepted job, `1500592`, took 7m18s and its process peak was 71,491.5 MB.
This is a material one-time memory requirement caused by transforming the full
1.3-million-cell sparse matrix. It must not be conflated with GRN fit memory.

## Cell-count scaling on all 1.306 million cells

Job `1500621` used 16 CPUs, a fixed 2,095-gene profile (2,000 HVGs plus up to
256 expressed TFs), 256 TFs, a 5,000-tree ceiling, arboreto trailing-window
stopping, signed TF-target correlation, regulon construction and AUCell.

| Cells | GRN wall s | Process peak MB | Mean fitted trees | Activator / repressor regulons |
| ---: | ---: | ---: | ---: | ---: |
| 50,000 | 56.922 | 294.1 | 28.233 | 132 / 1 |
| 100,000 | 121.758 | 465.7 | 30.600 | 135 / 0 |
| 200,000 | 272.721 | 814.7 | 34.621 | 137 / 0 |
| 400,000 | 639.504 | 1,495.9 | 39.879 | 142 / 0 |
| 800,000 | 1,530.810 | 2,592.9 | 47.396 | 143 / 2 |
| 1,306,127 | 2,801.771 | 4,276.9 | 53.429 | 147 / 1 |

The full-range log-log slopes were 1.201 for GRN wall time and 0.822 for
memory. At 1,306,127 cells the GRN completed in 46m42s, retained 12,800
top-50-per-TF edges and AUCell produced a finite 1,306,127 by 148 matrix.
All points used the Rust backend, emitted non-empty finite GRNs and contained
both activating and repressing correlated edges. A zero repressor-regulon
count at some intermediate points means no negative programme passed the
regulon size threshold; it is not unsigned output.

Portable JSON SHA-256:
`e4330f0df14fd4babb305fb26030b25e3330ea4d3336668e910ec3f5f89fe7e4`.

## Broader production-shaped profile

Job `1500623` exercised all 5,858 selected genes and all 1,264 expressed TFs
with the same 5,000-tree ceiling and 16 threads.

| Cells | GRN wall s | Process peak MB | Raw / retained edges | Activator / repressor regulons |
| ---: | ---: | ---: | ---: | ---: |
| 50,000 | 643.546 | 551.9 | 543,778 / 63,200 | 350 / 8 |
| 100,000 | 1,376.303 | 972.0 | 541,476 / 63,200 | 340 / 9 |

The two-point wall ratio was 2.139 (slope 1.097) and memory ratio was 1.761
(slope 0.817). AUCell returned finite matrices of 50,000 by 358 and 100,000 by
349. Portable JSON SHA-256:
`974df293c49d0b74024e8aa4762596faae4108c4643b3825eec5b347aff7bbdb`.

## Same-node RustScenic versus arboreto

The comparison ran both tools sequentially in one Slurm allocation on the
same nested 20,000-cell prefix, 2,095 genes, 256 TFs, seed 777, 16 workers and
5,000-tree ceiling. The reference environment pins arboreto 0.1.6,
scikit-learn 1.7.2, Dask/distributed 2024.1.1, NumPy 1.26.4, pandas 2.1.4 and
SciPy 1.13.1. Physical memory is reported as proportional set size (PSS), so
fork-shared pages are divided among the arboreto workers instead of counted
once per process.

| Same-node run | RustScenic wall s | arboreto wall s | arboreto / RustScenic | RustScenic / arboreto peak PSS MB |
| --- | ---: | ---: | ---: | ---: |
| `1500622`, `cpu-node-122`, AMD EPYC 9754 | 23.344 | 47.621 | 2.040x | not recorded |
| `1509218`, `cpu-node-37`, Intel Xeon E5-2695 v3 | 19.837 | 65.964 | 3.325x | 188.9 / 995.5 |

These are two same-input, same-node observations, not a universal speedup
factor. Timed regions exclude file loading and arboreto's dense conversion,
which favours the reference. The second run used 5-second PSS sampling; a
separate 0.5-second-sampling run produced essentially the same memory and
arboreto timing, ruling out a material sampling-frequency artefact. Summed
arboreto RSS is not reported as physical memory because it counts fork-shared
pages once per process. On the accepted PSS run, RustScenic used 5.27x less
physical memory.

Fitted-tree totals differed by only 53 of 56,546 arboreto trees (0.094%),
confirming that the corrected Rust trailing-window stop behaves like the
pinned modern arboreto reference. Fine edge rankings are not identical:
87,621 edges were shared, shared-edge Spearman was 0.628, and mean top-20,
top-50 and top-100 Jaccard values were 0.373, 0.414 and 0.429. This is strong
early-stop parity and a measured throughput advantage, not proof that the
histogram GBM is biologically superior to exact scikit-learn GBM.

As a biological sanity check rather than independent ground truth, top-ten
target overlap was 8/10 for `Olig1`, 10/10 for `Olig2`, 6/10 for `Dlx1`,
7/10 for `Dlx2`, 8/10 for `Neurog2`, 7/10 for `Sox10` and 9/10 for `Mef2c`.
The retained targets included expected lineage-linked pairs such as
`Olig1`/`Olig2`, `Dlx1`/`Dlx2`, `Neurog2`/`Eomes` and oligodendrocyte genes
under `Sox10`. These observations do not replace annotated-cell or motif
validation.

Accepted portable comparison JSON SHA-256:
`6f73f4f0099793890b325e679d1b7f82afcebab978fea711329b4c5a258defbb`.
RustScenic and arboreto metadata JSON SHA-256 values were
`4aa8d5d388cb4af7854a83b9339beb503b4d4966d9c0813e84ac87c5dafd7246`
and `5cb33e01487da3e61735c34a85e9b169530f17156c585ffdb3b3ef3c2aadd0c1`.
All accepted real-RNA JSON artefacts passed a recursive scan with zero
absolute path strings.

## Reproduction command

Run on one node. Do not request multiple hosts for this non-MPI workload.

```bash
export RAYON_NUM_THREADS=16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python validation/scaling/bench_real_rna_grn_scaling.py scale \
  --prepared /shared/projects/drug_discovery/rustscenic-data/10x-1m-neurons/1M_neurons_rustscenic_grn_fixed.h5 \
  --sizes 50000 100000 200000 400000 800000 1306127 \
  --n-hvg 2000 --n-tfs 256 --threads 16 \
  --n-estimators 5000 --seed 777 \
  --out /shared/projects/drug_discovery/rustscenic-results/real-rna/real-rna-scale.json
```

## Boundary of the evidence

This validates real RNA GRN computation at the target cell count and a broader
5,858-gene profile through 100,000 cells. It does not yet prove Mohammad's
actual 1.2-million-cell spatial object, which has not been made available in
the shared project. Before a full run, benchmark a 100,000-cell nested subset
of that object with its real gene panel, sparsity, annotations and motif
rankings, then advance to 200k, 400k, 800k and full size only while measured
memory headroom remains safe.
