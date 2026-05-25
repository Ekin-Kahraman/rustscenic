# RustScenic

Faster, memory-efficient regulatory-network analysis for single-cell and multiome
data.

RustScenic provides Rust kernels for GRN inference, regulon activity, motif
enrichment, topic modelling, enhancer links and eRegulons through a Python API.
It is CPU-first, installable from PyPI and designed for reproducible local runs.

```bash
pip install rustscenic
```

![RustScenic evidence snapshot: built, released, benchmarked and lab-validated](assets/rustscenic-evidence.svg)

## Evidence Snapshot

| Signal | Evidence |
| --- | --- |
| Built | Cross-platform Rust and Python CI, docs build, release smoke checks and nightly real-data validation workflows. |
| Released | Current release `v0.4.7`; PyPI package with Python 3.10 to 3.13 release wheels plus source distribution. |
| Benchmarked | `11x` to `52x` faster than SCENIC+ in tested real-data core E2E rows; commands, hardware, runtime, memory and output checks are committed. |
| Memory-scaled | `6.34 GB` peak RSS on a 100k-cell four-stage scale check; legacy pySCENIC reports exceed `40 GB` on similar workloads. |
| Lab-validated | Huang Lab collaborator artefacts include a 10x human brain GEM-X full monolith run recovering `16/17` expected brain TFs. |

## Highlights

| Feature | Status |
| --- | --- |
| Tested real-data speedup | `11x` to `52x` vs SCENIC+ in core E2E rows |
| Memory scaling | `6.34 GB` peak RSS on a 100k-cell four-stage scale check; legacy pySCENIC reports exceed `40 GB` on similar workloads |
| Current release | `v0.4.7` |
| Python support | 3.10 to 3.13 |
| Core install | `pip install rustscenic` |
| Runtime model | CPU-first Rust kernels |
| Core path dependencies avoided | Java, dask, CUDA, Snakemake |
| Evidence | Controlled benchmarks plus committed collaborator real-data artefacts |

## Benchmark Snapshot

| Result | Value |
| --- | ---: |
| Human brain GEM-X 2k total runtime | RustScenic `11.89 s`; reference `150.36 s` |
| Human brain GEM-X region-to-gene edge-set Jaccard | `1.000` |
| Human brain GEM-X region AUCell mean Pearson | `0.823` |
| cisTarget AUC kernel agreement vs `ctxcore.recovery.aucs` | Pearson `1.0000` |

The full benchmark matrix includes dataset shape, command path, hardware,
runtime, memory and validation metrics. Start with [Benchmarks](benchmarks.md).

## Stage Coverage

| Stage | RustScenic API | SCENIC ecosystem stage covered |
| --- | --- | --- |
| TF-to-gene GRN | `rustscenic.grn.infer` | GRNBoost2-style regulatory-network inference |
| AUCell | `rustscenic.aucell.score` | Per-cell regulon activity scoring |
| cisTarget | `rustscenic.cistarget.enrich` | Motif enrichment and support filtering |
| Topics | `rustscenic.topics.fit`, `fit_gibbs` | scATAC topic modelling |
| ATAC preprocessing | `rustscenic.preproc` | Fragment matrix building and QC |
| Enhancer links | `rustscenic.enhancer.link_peaks_to_genes` | Peak-to-gene linking |
| eRegulons | `rustscenic.eregulon.build_eregulons` | Enhancer-linked regulon assembly |
| Orchestration | `rustscenic.pipeline.run` | Staged workflow across RNA and multiome inputs |

## Next

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [API map](api.md)
- [Benchmarks](benchmarks.md)
- [Validation](validation.md)
- [Lab adoption](adoption.md)
- [Scope](limitations.md)
