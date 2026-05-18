# RustScenic

RustScenic is a Rust plus PyO3 implementation of the practical SCENIC and SCENIC+ compute path for single-cell regulatory-network analysis.

It is built for users who want one install, modern Python, deterministic CPU execution and fewer fragile dependencies than the old SCENIC stack.

## What It Replaces

| Stage | RustScenic API | Reference stack |
| --- | --- | --- |
| Gene regulatory network inference | `rustscenic.grn.infer` | `arboreto.grnboost2` |
| Per-cell regulon activity | `rustscenic.aucell.score` | `pyscenic.aucell` |
| Motif-regulon enrichment | `rustscenic.cistarget.enrich` | `ctxcore` / `pycistarget` |
| scATAC topic modelling | `rustscenic.topics.fit`, `fit_gibbs` | `pycisTopic` / Mallet |
| Fragment preprocessing | `rustscenic.preproc.fragments_to_matrix` | `pycisTopic` fragment loader |
| Enhancer-gene links | `rustscenic.enhancer.link_peaks_to_genes` | SCENIC+ p2g linking |
| eRegulon assembly | `rustscenic.eregulon.build_eregulons` | SCENIC+ eRegulon builder |
| Pipeline orchestration | `rustscenic.pipeline.run` | SCENIC+ workflow glue |

## Current Status

Current public release: `v0.4.6`.

The package has public wheels, CI across macOS and Linux, Windows x64 coverage in the release workflow, unit and integration tests, real-data validation artefacts and community validation reports. It is still alpha research software: the core API works, but the project needs more independent lab adoption and broader multi-dataset parity before it should be treated as a mature community standard.

## Best Evidence

- AUCell agreement against pySCENIC on the Ziegler airway atlas: mean per-cell Pearson `0.984`; `91.7%` of cells above `0.95`.
- Cistarget AUC kernel agreement against `ctxcore.recovery.aucs`: Pearson `1.0000`.
- Real multiome end-to-end runs on 10x PBMC, mouse brain and PBMC granulocyte datasets.
- Community reports from external users on Kamath dopaminergic neurons and 10x human brain multiome data.

For the full benchmark matrix, including commands, hardware, baseline,
runtime, memory, parity metric and biological sanity check, see
[Benchmarks](benchmarks.md).

Start with [Installation](installation.md), then run the [Quickstart](quickstart.md). If you are evaluating the tool for a lab, use [Lab Adoption](adoption.md).
