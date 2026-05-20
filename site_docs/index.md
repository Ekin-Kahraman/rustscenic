# RustScenic

RustScenic is for researchers who want SCENIC-style regulatory biology without
spending the first day debugging the stack.

It builds GRNs, scores regulons, links enhancers and assembles eRegulons from
Python, with Rust kernels behind the slow stages. The core path is one install
and runs locally on CPU without Java, dask, CUDA or Snakemake.

```bash
pip install rustscenic
```

## What It Fixes

| Pain | RustScenic |
| --- | --- |
| Fragile legacy installs | Python 3.10 to 3.13 wheels and five core runtime dependencies |
| Local CPU runs that take too long | Rust kernels for the matrix-heavy stages |
| Multiome workflow glue across many packages | One API for GRN, AUCell, motifs, topics, enhancer links and eRegulons |
| Hard-to-defend performance claims | Benchmarks include commands, hardware, runtime, memory and output checks |

## Evidence Snapshot

Current public release: `v0.4.7`.

| Result | Value |
| --- | ---: |
| Tested real-data core E2E speedup vs SCENIC+ | `11x` to `52x` |
| Human brain GEM-X 2k total runtime | RustScenic `11.89 s`; SCENIC+ `150.36 s` |
| Human brain GEM-X region-to-gene Jaccard | `1.000` |
| Human brain GEM-X region AUCell mean Pearson | `0.823` |
| cisTarget AUC kernel agreement vs `ctxcore.recovery.aucs` | Pearson `1.0000` |

The full benchmark matrix includes dataset shape, command path, hardware,
runtime, memory and validation metrics. Start with [Benchmarks](benchmarks.md).

## What It Runs

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
