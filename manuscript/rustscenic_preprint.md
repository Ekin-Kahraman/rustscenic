# Archived early draft: rustscenic

**Ekin Kahraman**

*Draft from 2026-04-19. This manuscript is archived and does not define the
current release claims. Use the README, `site_docs/benchmarks.md`, and the
latest release notes for the current validation state.*

---

## Abstract

SCENIC (Aibar et al., 2017), pySCENIC, and SCENIC+ (Bravo González-Blas et al., 2023) are widely used reference methods for single-cell regulatory-network inference. This archived draft described an early Rust plus PyO3 implementation of several compute stages: gene-regulatory-network inference, regulon activity scoring, topic modelling, and motif enrichment. The current project has moved beyond this April draft, and the public claim is narrower than full production replacement: rustscenic is an installable, deterministic CPU-focused implementation of the practical SCENIC compute path, with measured agreement on key downstream outputs and real-data validation evidence, while real 100k to 200k matched multiome validation remains open.

---

## Introduction

SCENIC and SCENIC+ are canonical tools for single-cell regulatory-network analysis. This draft focused on two operational bottlenecks seen in our validation work: **installability** and **scalability**.

Installability: `arboreto` (GRNBoost2) depends on an outdated dask pipeline that fails at runtime on dask ≥ 2024.0 with `TypeError: Must supply at least one delayed object`. `pyscenic` imports `pkg_resources` from setuptools, which is deprecated and removed in modern setuptools. `pycisTopic`'s default topic-modelling backend is Mallet, requiring a Java runtime that institutional environments frequently forbid. `flashscenic` works on CPU but requires PyTorch and has swapped the algorithm (RegDiffusion), breaking compatibility with published SCENIC numbers.

Scalability: at 10^5-cell cohorts, now common in atlas-scale projects, pySCENIC AUCell can run in tens of minutes; pycisTopic topic modelling can run in hours; and reported SCENIC+ memory use can exceed 40 GB.

We set out to implement several slow stages in a single pip-installable CPU package and measure where it matches, differs from, or still needs reference validation.

## Methods

### Architecture

`rustscenic` is a Cargo workspace of six crates: `rustscenic-core` (shared types), `rustscenic-grn`, `rustscenic-aucell`, `rustscenic-topics`, `rustscenic-cli` (standalone Rust CLI), and `rustscenic-py` (PyO3 bindings, ABI3-compatible). The Python package is distributed as a single maturin-built wheel per platform (macOS aarch64, macOS x86_64, Linux x86_64, Linux aarch64), with three runtime dependencies: `numpy`, `pandas`, `pyarrow`, `scipy`.

### GRN inference (GRNBoost2)

We implement a histogram gradient-boosting regression tree using a LightGBM-style 255-bin approximation with early stopping. Each target gene is fitted in parallel via Rayon across candidate TFs. Seeding is deterministic under thread execution. Parameters mirror arboreto's defaults (`n_estimators=5000`, `learning_rate=0.01`, `max_features=0.1`, `subsample=0.9`, `max_depth=3`, `early_stop_window=25`).

### AUCell

We reimplement the Aibar et al. 2017 recovery-curve AUC. To match pyscenic numerics, we adopt ctxcore's rank-cutoff convention (`rank_cutoff = round(auc_threshold × n_genes) − 1`, an R-compat off-by-one) and the `(rank_cutoff + 1) × |G|` maximum-AUC denominator. With this change, rustscenic's AUCell is bit-identical to `ctxcore.recovery.aucs` on a probe of small + large regulons (Pearson 1.0000 on 58 TRRUST regulons, mean absolute difference 2.4 × 10⁻⁵). On the full pyscenic pipeline (pyscenic shuffles gene order before ranking to reduce tie-break bias), the agreement is per-cell Pearson 0.99 / per-regulon Pearson 0.87.

### Topics

We implement Online Variational Bayes LDA (Hoffman-Blei-Bach, 2010) in Rust. Input is a CSR-sparse (cells × peaks) binary matrix; output is a (cells × K) cell-topic probability matrix and a (K × peaks) topic-peak probability matrix. Fit is Rayon-parallelised across documents within each minibatch.

### Cistarget

The motif-enrichment AUC kernel reuses the AUCell computation on motif-rank tables. The aertslab feather databases are loaded via pyarrow; the databases themselves, 300 MB to 35 GB, are not bundled.

### Validation

All measurements use real 10x Genomics data and cached pyscenic/ctxcore/Mallet reference outputs. Every measurement is preserved as a runnable script plus a versioned `.md` report under `validation/ours/`. No claim in this paper is made without a corresponding log file in that directory.

## Results

### Install ergonomics

On a fresh Python 3.12 + numpy 2 + pandas 3 environment, `pip install rustscenic` succeeded in this draft's validation logs. In the same logs, `pip install arboreto pyscenic` completed but runtime use failed with `TypeError: Must supply at least one delayed object` on the first call. The current release state is tracked in README and CI, not this archived draft.

### GRN inference

Measured on 10x Multiome 3k (shared-barcode 2,588 cells × 21,255 genes × 1,457 TFs, `n_estimators=5000`, seed 777) against cached arboreto output:

| Metric | Value |
| --- | ---: |
| Wall time | 401 s |
| Peak RSS | 1.13 GB |
| Edges produced | 2.58 M |
| Per-edge Spearman (816k common edges) vs arboreto | **0.58** |
| Per-target TF-rank Spearman, mean | 0.57 |

On PBMC-3k the biological hit-rate (TRRUST edges recovered) is 94% (17/18). On PBMC-10k lineage-specificity holds for all 8 canonical TFs tested (SPI1 4.2×, PAX5 15.8×, EBF1 12.2×, TCF7 5.3×, LEF1 3.2×, TBX21 9.5×, CEBPD 3.9×, IRF8 1.7×; ratios are mean regulon activity in target lineage vs other lineages). On Tirosh 2016 melanoma, MITF activity is 3.48× in malignant vs tumour-microenvironment cells. Despite the moderate per-edge Spearman, downstream AUCell agreement is per-cell Pearson 0.99, so coarse-resolution biology is preserved even though fine edge rankings differ.

### AUCell

Per-cell Pearson vs pyscenic on the multiome dataset (1,457 regulons) is 0.988 mean / 0.990 median, 99.5% of cells > 0.95. Per-regulon Pearson is 0.87 mean / 0.87 median, 90.5% > 0.80 (the weaker metric is affected by pyscenic's gene-shuffle tie-break step that we do not reproduce; we use deterministic gene-index tie-breaks). Exact top-regulon-per-cell match is 88.4%; the pyscenic top-1 regulon is in our top-3 for 99.5% of cells. Runtime is 0.21 s for 1,457 regulons × 2,588 cells, an 88× speedup over `pyscenic.aucell` on the same data in this archived run.

### Topics

On 10x Genomics PBMC 10k ATAC (8,728 cells × 67,448 peaks after 1%-prevalence filter, K=30):

| Tool | Wall | Unique argmax topics | NPMI coherence (mean) | ARI vs leiden |
| --- | ---: | ---: | ---: | ---: |
| Mallet (pycisTopic reference) | 534 s | 24/30 | 0.196 | 0.258 |
| rustscenic seed=42 | 942 s | 5/30 | 0.123 | **0.269** |
| rustscenic seed=123 | 622 s | 5/30 | n/a | 0.334 |
| rustscenic seed=777 | 620 s | 6/30 | n/a | 0.180 |

In these archived measurements, rustscenic's cell-type recovery (ARI vs leiden) was comparable to Mallet's. Mallet won on topic diversity (24 unique vs 5-6) and NPMI coherence. Current releases include `topics.fit_gibbs`; see `docs/topic-collapse.md` for the current topic-modelling claim.

### Cistarget

Per-regulon Pearson vs `ctxcore.recovery.aucs` (the AUC kernel inside pycistarget) on 58 TRRUST regulons against the aertslab hg38 v10 feather database (5,876 motifs × 27,015 genes): **1.0000** (all 58 regulons > 0.9999, mean absolute difference 2.6 × 10⁻⁵). Bit-identical to float32 precision.

Self-consistency test: for 10 randomly sampled motifs, using the motif's own top-500 genes as an artificial regulon, rustscenic ranks the motif at position #1 of the enrichment for all 10. TRRUST-at-scale on 166 human TFs with >=10 annotated motifs and >=10 targets: 19% place their annotated motif at rank #1, 33% in top-5, 68 to 100% in top-100. Mouse (mm10) cross-species works unchanged: 2/5 well-known TFs (Gata1, Stat1) rank #1, 4/5 in top-5.

### End-to-end + scale

Full 4-stage pipeline on the 10x Multiome 3k dataset: 9.1 min rustscenic vs 11.8 min for the composite reference pipeline (arboreto + pyscenic + tomotopy). On a 100,000-cell × 20,292-gene bootstrap of PBMC-10k, all four stages completed with peak RSS 6.34 GB. The >40 GB memory figure is reported context, not a controlled same-hardware comparison.

### Real-world atlas-scale head-to-head (Ziegler 2021 nasopharyngeal)

To test a real atlas-shaped workload, we ran both rustscenic and pyscenic on the Ziegler et al. 2021 *Cell* nasopharyngeal scRNA-seq atlas: 58 donors, 18,073 COVID+ / 14,515 COVID- cells, 18 coarse airway cell types. After standard preprocessing (normalize, log, HVG plus TFs), the input was 31,602 cells × 3,044 genes × 59 regulons. Identical adjacencies were used on both sides to isolate the AUCell kernel.

**Agreement on identical input:**

| Metric | rustscenic vs pyscenic-unit | rustscenic vs pyscenic-weighted |
|---|---:|---:|
| Per-cell Pearson (mean) | **0.984** | 0.949 |
| Cells with Pearson > 0.95 | **91.7 %** | 71.6 % |
| Argmax-regulon per-cell agreement | 85.4 % | 50.1 % |

**Runtime on the same workload:** rustscenic 0.25 s, pyscenic-unit 6.81 s, pyscenic-weighted 5.29 s, a 21 to 27× speedup in this archived run.

**Biological validation, canonical airway TF benchmark (n=14 TFs):** 8/14 direct hits for rustscenic, 8/14 for pyscenic-unit, 9/14 for pyscenic-weighted. All three tools miss the same five TFs (STAT1 at coarse cell-type resolution, MYB, IRF7, SOX2, PAX5). Per-TF z-scores agree within 0.02 for 10/14 TFs. This is compatibility evidence, not full biological replication.

**Install reality:** arboreto (GRNBoost2) failed in this draft's 2026 Python validation environment with two independent failure modes: (a) fresh `pip install arboreto` to `TypeError: Must supply at least one delayed object` (dask_expr incompatibility); (b) inside pyscenic's own environment with pandas pinned to 1.5.3 to `Dask requires pandas >= 2.0.0`. Treat this as a logged environment result, not a universal statement about every possible pinned environment.

**Biological extension:** The per-cell regulon activity matrix enabled a downstream COVID+ / COVID− differential analysis (Wilcoxon + BH-FDR per cell type, ≥100 cells per arm). IRF7-driven type I interferon programme is upregulated in COVID+ cells across 7 of 11 cell types (strongest in Ionocytes +1.34 log₂FC q=6e-17). An AP-1 / stress-response programme (JUN, JUNB, NR4A1, XBP1) is suppressed in squamous cells (all log₂FC < −0.5, q < 1e-90). A WNT / regenerative programme (TCF7, LEF1, EOMES) is upregulated in secretory cells. These findings extend the cell-type-proportion deconvolution analyses reported in the covid-airway-deconvolution companion project (Kahraman 2026, *in preparation*) from "which cells are perturbed" to "which regulatory programmes rewire during infection".

Full head-to-head scripts and biological interpretation are part of a companion case-study manuscript (in preparation); the tool-validation numbers and figures in this preprint are independently verifiable from the open-source rustscenic package and the Ziegler h5ad available from GEO.

### Determinism + robustness

All four stages produce bit-identical output across three runs with the same seed (verified). A 10-case edge-case suite (foreign genes, NaN input, all-zero cells, duplicate gene names, empty regulons, single-cell input, large regulons, object-dtype rankings, n_topics=0, very-sparse matrices) passes 10/10. Two correctness bugs surfaced during the audit and are fixed in this release: a stale wheel allowing NaN to silently propagate through GRN, and AUCell silently accepting duplicate gene symbols.

## Discussion

**What this tool is good for.** Environments where the reference stack's Python dependencies are broken; per-cell regulon scoring (AUCell is the strongest match to pyscenic); motif enrichment on the aertslab feather databases (ctxcore bit-parity); cell-type-correlated coarse topic analysis without Java; multi-stage pipelines where memory is the limiting factor.

**What it is not for.** Exact reproduction of pyscenic's raw AUCell numbers because deterministic tie-breaks differ from pyscenic's seeded shuffle. Fine-grained scATAC topic decomposition at K >= 30 should use `topics.fit_gibbs` or Mallet via pycisTopic. GPU workloads are out of scope.

**Correctness methodology.** Every numeric claim in this paper has a corresponding log file in `validation/ours/`. When we found that our earlier prose claim "edge-rank Jaccard ≥ 0.80" had never been measured, we replaced it with the measured per-edge Spearman of 0.58. When we found the "2× better than Mallet on cell-type ARI" figure had been extracted from one small collapsed dataset, we replaced it with the 10k-cell head-to-head (comparable, not better). The version 0.1.0 release is what the measurements support, not what we hoped they'd show.

**Limitations.** This archived draft is stale. Current limitations are maintained in `site_docs/limitations.md` and `site_docs/benchmarks.md`.

## Data + code availability

Code: https://github.com/Ekin-Kahraman/rustscenic (MIT).
PyPI: `pip install rustscenic`.
Current validation artefacts: see `validation/` and `site_docs/benchmarks.md`.
Example: `examples/pbmc3k_end_to_end.py`.

## Acknowledgments

Method credit: Aibar et al. 2017 (SCENIC / AUCell), Bravo González-Blas et al. 2023 (SCENIC+, pycisTopic, pycistarget), Hoffman-Blei-Bach 2010 (Online VB LDA), Ke et al. 2017 (LightGBM histogram GBM). This work stands on the shoulders of the Aerts Lab's open-source scientific software; our contribution is reimplementation for install-ability and CPU scalability, not algorithmic novelty.

## References

*(To be formatted on submission; ordering by first appearance.)*

1. Aibar, S. et al. SCENIC: single-cell regulatory network inference and clustering. *Nat Methods* 14, 1083-1086 (2017).
2. Bravo González-Blas, C. et al. SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. *Nat Methods* 20, 1355-1367 (2023).
3. Hoffman, M., Bach, F., Blei, D. Online learning for Latent Dirichlet Allocation. *NIPS* 23 (2010).
4. Ke, G. et al. LightGBM: a highly efficient gradient boosting decision tree. *NIPS* 30 (2017).
5. Han, H. et al. TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. *Nucleic Acids Res.* 46, D380-D386 (2018).
