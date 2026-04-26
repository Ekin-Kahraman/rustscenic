# rustscenic

[![CI](https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml/badge.svg)](https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-stable-orange)](https://www.rust-lang.org/)

A Rust + PyO3 replacement for the SCENIC / SCENIC+ compute stack: one install, modern Python, low-memory CPU execution, and atlas-scale regulatory-network analysis without Java, dask, CUDA, or fragile multi-tool environments.

```bash
# Universal source install while PyPI trusted-publishing is being configured:
pip install git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.0

# Or install a prebuilt wheel from the v0.2.0 GitHub Release for your platform:
# macOS Apple Silicon:
pip install https://github.com/Ekin-Kahraman/rustscenic/releases/download/v0.3.0/rustscenic-0.3.0-cp310-abi3-macosx_11_0_arm64.whl
# Linux x86_64:
pip install https://github.com/Ekin-Kahraman/rustscenic/releases/download/v0.3.0/rustscenic-0.3.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Four runtime dependencies (numpy, pandas, pyarrow, scipy). Python 3.10–3.13, Linux + macOS (x86_64 + aarch64). No dask, no Java, no CUDA.

## Goal

rustscenic is being built as the single-install replacement for the practical SCENIC / SCENIC+ workflow: RNA GRN inference, AUCell regulon activity, motif enrichment, ATAC fragment preprocessing, topic modelling, enhancer-gene linking, and eRegulon assembly in one package.

The project is intentionally not a thin wrapper around the old stack. The target is a simpler architecture that makes regulatory-network analysis easier to install, cheaper to run on CPU, deterministic under a fixed seed, and robust to real atlas conventions such as ENSEMBL `var_names`, duplicate gene symbols, backed AnnData, and UCSC/Ensembl chromosome mismatches.

v0.2.0 already replaces the main compute stages used by pySCENIC / arboreto / pycisTopic / pycistarget / scenicplus in common Python pipelines. Region-based cistarget is wired into eRegulon assembly; the remaining replacement proof is concentrated in Mallet-class ATAC topic modelling, MACS2 reference cross-checks, full 100k-cell real multiome validation, and head-to-head scenicplus parity numbers on real region-ranking databases.

## What it does

Rust-native replacements for the compute stages plus the glue that scenicplus builds eRegulons from:

| Stage | **rustscenic** | Replaces |
|---|---|---|
| Gene-regulatory network inference | `rustscenic.grn.infer` | `arboreto.grnboost2` |
| Per-cell regulon activity scoring | `rustscenic.aucell.score` | `pyscenic.aucell.aucell` |
| Topic modelling on scATAC peaks | `rustscenic.topics.fit` | `pycisTopic` (Mallet) |
| Motif-regulon enrichment | `rustscenic.cistarget.enrich` | `pycistarget` AUC kernel |
| ATAC fragments → cells × peaks matrix | `rustscenic.preproc.fragments_to_matrix` | `pycisTopic` fragment loader |
| Cell QC (TSS enrichment, FRiP, insert size) | `rustscenic.preproc.qc` | `pycisTopic.qc` |
| Enhancer → gene correlation | `rustscenic.enhancer.link_peaks_to_genes` | `scenicplus` p2g linking |
| eRegulon assembly (TF × enhancers × target genes) | `rustscenic.eregulon.build_eregulons` | `scenicplus` eRegulon builder |
| End-to-end pipeline orchestrator | `rustscenic.pipeline.run` | `scenicplus` snakemake |

Bundled with the wheel: HGNC (1,839 human) and MGI (1,721 mouse) TF lists via `rustscenic.data.tfs(species)`. Motif rankings auto-download on first use via `rustscenic.data.download_motif_rankings`. Cellxgene-curated h5ads (ENSEMBL IDs in `var_names`, gene symbols in `var["feature_name"]`) are auto-detected so atlas data works without manual patching.

## Quick example (PBMC-3k, end-to-end)

```python
import anndata as ad
import rustscenic.grn, rustscenic.aucell

adata = ad.read_h5ad("rna.h5ad")
tfs = rustscenic.grn.load_tfs("hs_hgnc_tfs.txt")

# 1. GRN inference
grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=5000, seed=777)

# 2. Build top-50-target regulons and score per-cell activity
regulons = [
    (f"{tf}_regulon", grn[grn["TF"] == tf].nlargest(50, "importance")["target"].tolist())
    for tf in grn["TF"].unique()
]
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
```

Full end-to-end script: [`examples/pbmc3k_end_to_end.py`](examples/pbmc3k_end_to_end.py). Runs cold in seconds in a fresh venv. [`docs/tester-quickstart.md`](docs/tester-quickstart.md) is the collaborator smoke-test path.

## Measured against the pyscenic / arboreto reference

Same input on both sides. Every row has a log file under [`validation/`](validation/).

| Axis | pyscenic / arboreto | **rustscenic** |
|---|---|---|
| Installs on fresh Python 3.10–3.13 venv (2026-04) | arboreto: `TypeError: Must supply at least one delayed object` (dask_expr); pyscenic: `ModuleNotFoundError: pkg_resources` in current stacks | GitHub Release wheels and source install succeed; all 4 core stages import |
| AUCell wall-time, Ziegler 2021 atlas (31,602 × 59) | 6.81 s (pyscenic) | 0.25 s |
| AUCell wall-time, 10x Multiome (10,290 × 1,457) | 18.6 s (pyscenic) | 0.21 s |
| Peak RSS, 4 stages on 100,000 cells × 20,292 genes | > 40 GB (reported) | 6.3 GB |
| Cistarget kernel vs `ctxcore.recovery.aucs` | reference | Pearson 1.0000, mean abs diff 2.4 × 10⁻⁵ |
| AUCell per-cell Pearson vs pyscenic (Ziegler, 31,602 cells) | reference | 0.984 mean, 91.7 % of cells > 0.95 |
| Canonical airway TFs matching literature (Ziegler, n=14) | 8 / 14 (pyscenic, unit weights) | 8 / 14 — same hits, same 5/14 misses |
| Bit-identical output under same seed across threaded runs | no (dask non-determinism) | yes |
| Runtime dependencies | 40 + | 4 |

Tool-to-tool variation (same hits, same misses on the same 14 canonical TFs) is smaller than the dataset-inherent noise, consistent with rustscenic being numerically equivalent to pyscenic at the per-cell level.

## Per-stage detail

Numbers are **rustscenic**'s values. The measurement context (dataset, `n_cells`, etc.) is in each row.

### GRN — `arboreto.grnboost2` replacement

| Measurement | Value |
|---|---|
| Per-edge Spearman vs arboreto (multiome3k, n_estimators=5000, 816 k common edges) | 0.58 |
| Per-target TF-ranking Spearman mean | 0.57 |
| TRRUST known TF→target edges recovered (PBMC-3k) | 17 / 18 (94 %) |
| Lineage TFs correctly enriched in expected cell types (PBMC-10k) | 8 / 8 (SPI1, PAX5, EBF1, TCF7, LEF1, TBX21, CEBPD, IRF8) |
| MITF regulon activity, Tirosh 2016 melanoma — malignant vs TME | 3.48× |
| 100k-cell bootstrap, n_estimators=100 | 17 min / 5.0 GB peak RSS |

Edge rankings disagree with arboreto at fine grain (Spearman 0.58, top-100 Jaccard 0.10) — expected consequence of independent histogram-GBM quantisation. Coarse biology converges. Downstream AUCell is 0.99 per-cell with pyscenic, so edge-ranking differences do not propagate.

### AUCell — `pyscenic.aucell` replacement

| Measurement | Value |
|---|---|
| Per-cell Pearson vs pyscenic (10x Multiome, 2,588 × 1,457) | 0.988 mean, 99.5 % of cells > 0.95 |
| Per-cell Pearson vs pyscenic (Ziegler atlas, 31,602 × 59) | 0.984 mean, 91.7 % of cells > 0.95 |
| Per-regulon Pearson (10x Multiome) | 0.87 mean, 90.5 % > 0.80 |
| Exact top-regulon-per-cell match (Multiome) | 88.4 % |
| Wall-time, 10k cells × 1,457 regulons | 0.21 s (vs 18.6 s pyscenic) |
| 100 k cells × 500 regulons | 10 s, 5.6 GB peak RSS |

### Topics — `pycisTopic` LDA replacement (Online VB)

10 x PBMC 10 k ATAC, 8,728 cells × 67,448 peaks, K = 30:

| Tool | Wall | Unique topics (of 30) | NPMI coherence | ARI vs leiden |
|---|---|---|---|---|
| Mallet (pycisTopic reference) | 534 s | 24 | 0.196 | 0.258 |
| **rustscenic** (Online VB) | 620–942 s (seed-dependent) | 5 – 6 | 0.123 | 0.18 – 0.33 |

Mallet recovers more distinct topics and higher coherence. Our Online VB LDA collapses aggressively at K = 30 on this dataset — see [`docs/topic-collapse.md`](docs/topic-collapse.md). Cell-type recovery (ARI vs leiden) is comparable. For fine-grained K ≥ 30 topic decomposition use Mallet via pycisTopic; for a no-Java drop-in that tracks cell-type structure, rustscenic works.

### Cistarget — `pycistarget` AUC kernel replacement

Validated on the aertslab hg38 v10 feather database (5,876 motifs × 27,015 genes):

| Measurement | Value |
|---|---|
| Per-regulon Pearson vs `ctxcore.recovery.aucs` (58 TRRUST regulons) | 1.0000 (all > 0.9999, abs diff 2.4 × 10⁻⁵) |
| Self-consistency (motif's own top-500 genes → rank #1) | 10 / 10 |
| TRRUST at scale (166 TFs ≥ 10 targets): TF-annotated motif ranks #1 | 19 % |
| Same benchmark: any TF-motif in top-100 | 68 – 100 % (rises with regulon size) |
| Mouse mm10 cross-species (5 TRRUST TFs) | 2 / 5 rank #1, 4 / 5 in top-5 |
| 100 k-cell workload × 100 regulons | 2.6 s, 6.3 GB peak RSS |

Bit-identical to `ctxcore.recovery.aucs` at float32 precision. The 19 % rank-#1 rate is the scaled-out TRRUST-vs-motif-binding benchmark, a property of the gold-standard mismatch, not the implementation.

### End-to-end + determinism

| Pipeline | Wall (10x Multiome 3k, all 4 stages) |
|---|---|
| Reference (arboreto + pyscenic + tomotopy), when it installs | 11.8 min |
| rustscenic | 9.1 min |

Peak RSS at 100 k cells: 6.3 GB across all 4 stages. Bit-identical output under the same seed across threaded runs, verified across three consecutive runs per stage. 10 / 10 robustness edge-case tests pass (foreign genes, NaN input, duplicate gene names, all-zero cells, large regulons, object-dtype rankings, n_topics = 0, very-sparse matrices).

## Scope and alternatives

rustscenic covers the four legacy SCENIC / SCENIC+ slow stages on CPU. Adjacent tools with different scope:

- **GPU, CUDA** — [flashSCENIC](https://github.com/haozhu233/flashscenic) (uses RegDiffusion, a different algorithm from GENIE3 / GRNBoost2, so outputs are not pyscenic-numerical).
- **Multiomic enhancer-aware GRN** — [scenicplus](https://github.com/aertslab/scenicplus) (joint scRNA + scATAC enhancer inference; superset of this scope).
- **TF-activity scoring from prebuilt regulons, no GRN inference** — [decoupler-py](https://saezlab.github.io/decoupler-py/) with CollecTRI.
- **R Bioconductor ecosystem** — the original R-SCENIC or [Epiregulon](https://www.nature.com/articles/s41467-025-62252-5).

rustscenic does not bundle the aertslab motif ranking feather databases (300 MB – 35 GB). Users fetch them from [`resources.aertslab.org`](https://resources.aertslab.org/) and pass the resulting DataFrame to `cistarget.enrich`.

## CLI

```bash
rustscenic grn       --expression data.h5ad --tfs tfs.txt --output grn.parquet
rustscenic aucell    --expression data.h5ad --regulons grn.parquet --output auc.parquet
rustscenic topics    --expression atac.h5ad --output topics --n-topics 30
rustscenic cistarget --rankings motifs.feather --regulons grn.parquet --output enrichment.tsv
```

## Repo layout

- `crates/` — Rust workspace: `rustscenic-{grn, aucell, topics, preproc, py}`
- `python/rustscenic/` — Python package, CLI entry point, type stubs
- `examples/pbmc3k_end_to_end.py` — end-to-end script on real PBMC-3k
- `validation/` — reproducible benchmark scripts + measurement reports for every number above, plus `VALIDATION_SUMMARY.md`
- `tests/` — pytest suite (106 Python tests) + Rust crate tests (51)
- `manuscript/` — preprint source
- `docs/topic-collapse.md` — known algorithmic caveat

## License

MIT. Algorithm implementations follow the aertslab Python references — original method credit to Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+), Hoffman-Blei-Bach 2010 (Online VB LDA).

## Contact

File issues at [github.com/Ekin-Kahraman/rustscenic/issues](https://github.com/Ekin-Kahraman/rustscenic/issues). Coordinated vulnerability disclosure: see [SECURITY.md](SECURITY.md).
