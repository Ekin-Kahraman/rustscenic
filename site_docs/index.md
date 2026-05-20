# RustScenic

RustScenic is a Rust-backed Python package for SCENIC-style single-cell
regulatory-network analysis. It covers GRN inference, AUCell scoring, cisTarget
enrichment, scATAC topic modelling, enhancer-gene linking and eRegulon assembly
from one install.

It is built for local CPU execution with modern Python. The core path does not
require Java, dask, CUDA or Snakemake.

## Why Use It

| Need | RustScenic |
| --- | --- |
| Simple install | `pip install rustscenic` |
| Local execution | CPU-first Rust kernels exposed through Python |
| SCENIC-style stages | GRN, AUCell, cisTarget, topics, enhancer links, eRegulons |
| Reproducibility | Deterministic threaded runs under a fixed seed |
| Public evidence | Head-to-head benchmarks and validation artefacts in-repo |

## Evidence Snapshot

Current public release: `v0.4.7`.

| Result | Value |
| --- | ---: |
| Tested real-data core E2E speedup vs SCENIC+ | `11x` to `52x` |
| Median real-data core E2E speedup | `27x` |
| Human brain GEM-X 2k total runtime | RustScenic `11.89 s`; SCENIC+ `150.36 s` |
| Human brain GEM-X region-to-gene Jaccard | `1.000` |
| Human brain GEM-X region AUCell mean Pearson | `0.823` |
| cisTarget AUC kernel agreement vs `ctxcore.recovery.aucs` | Pearson `1.0000` |

The full benchmark matrix includes dataset shape, command path, hardware,
runtime, memory and validation metrics. Start with [Benchmarks](benchmarks.md).

## What It Replaces

| Stage | RustScenic API | Reference stack |
| --- | --- | --- |
| Gene-regulatory network inference | `rustscenic.grn.infer` | `arboreto.grnboost2` |
| Per-cell regulon activity scoring | `rustscenic.aucell.score` | `pyscenic.aucell` |
| Motif-regulon enrichment | `rustscenic.cistarget.enrich` | `ctxcore` / `pycistarget` |
| scATAC topic modelling | `rustscenic.topics.fit`, `fit_gibbs` | `pycisTopic` / Mallet |
| Fragment preprocessing and QC | `rustscenic.preproc` | `pycisTopic` preprocessing |
| Enhancer-gene linking | `rustscenic.enhancer.link_peaks_to_genes` | SCENIC+ p2g linking |
| eRegulon assembly | `rustscenic.eregulon.build_eregulons` | SCENIC+ eRegulon builder |
| Pipeline orchestration | `rustscenic.pipeline.run` | SCENIC+ workflow glue |

## Next

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [API map](api.md)
- [Benchmarks](benchmarks.md)
- [Validation](validation.md)
- [Lab adoption](adoption.md)
- [Scope](limitations.md)
