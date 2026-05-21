# RustScenic

Faster, memory-efficient regulatory-network analysis for single-cell and multiome
data.

RustScenic provides Rust kernels for GRN inference, regulon activity, motif
enrichment, topic modelling, enhancer links and eRegulons through a Python API.
It is CPU-first, installable from PyPI and designed for reproducible local runs.

```bash
pip install rustscenic
```

## Highlights

| Feature | Status |
| --- | --- |
| Python support | 3.10 to 3.13 |
| Core install | `pip install rustscenic` |
| Runtime model | CPU-first Rust kernels |
| Tested real-data speedup | `11x` to `52x` vs SCENIC+ in core E2E rows |
| Core path dependencies avoided | Java, dask, CUDA, Snakemake |
| Evidence | Commands, hardware, runtime, memory and output checks in-repo |

## Benchmark Snapshot

| Result | Value |
| --- | ---: |
| Human brain GEM-X 2k total runtime | RustScenic `11.89 s`; reference `150.36 s` |
| Human brain GEM-X region-to-gene Jaccard | `1.000` |
| Human brain GEM-X region AUCell mean Pearson | `0.823` |
| cisTarget AUC kernel agreement vs `ctxcore.recovery.aucs` | Pearson `1.0000` |

The full benchmark matrix includes dataset shape, command path, hardware,
runtime, memory and validation metrics. Start with [Benchmarks](benchmarks.md).

## API Surface

| Module | Purpose |
| --- | --- |
| `rustscenic.grn.infer` | TF-to-gene regulatory network inference |
| `rustscenic.aucell.score` | Per-cell regulon activity scoring |
| `rustscenic.cistarget.enrich` | Motif support and enrichment |
| `rustscenic.topics.fit`, `fit_gibbs` | scATAC topic modelling |
| `rustscenic.preproc` | Fragment matrix building and QC |
| `rustscenic.enhancer.link_peaks_to_genes` | Enhancer-gene linking |
| `rustscenic.eregulon.build_eregulons` | Enhancer-linked regulon assembly |
| `rustscenic.pipeline.run` | Staged workflow orchestration |

## Next

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [API map](api.md)
- [Benchmarks](benchmarks.md)
- [Validation](validation.md)
- [Lab adoption](adoption.md)
- [Scope](limitations.md)
