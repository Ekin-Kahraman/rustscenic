<h1 align="center">RustScenic</h1>

<p align="center">
  <strong>Rust-backed SCENIC-style regulatory-network analysis from one Python package.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/rustscenic/">PyPI</a> |
  <a href="https://ekin-kahraman.github.io/rustscenic/">Docs</a> |
  <a href="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml">CI</a> |
  <a href="LICENSE">MIT License</a>
</p>

RustScenic implements the practical SCENIC and SCENIC+ compute path for single-cell
regulatory-network analysis: GRN inference, AUCell scoring, cisTarget enrichment,
scATAC topic modelling, enhancer-gene linking and eRegulon assembly.

The package is built for researchers who want the workflow to install cleanly and
run locally on CPU. It uses Rust for the compute-heavy kernels and exposes a small
Python API. No Java, dask, CUDA or Snakemake stack is required for the core path.

## Install

```bash
pip install rustscenic
```

Python 3.10 to 3.13 is supported. Release wheels are published for macOS and
Linux, with Windows x64 covered by the release workflow.

## Why Use It

- **Single install**: `pip install rustscenic` gives the Python API and CLI.
- **CPU-first**: designed for laptop and workstation runs without GPU setup.
- **Rust kernels**: GRN, AUCell, cisTarget, topics, preprocessing and enhancer
  linking are implemented as native modules.
- **Measured speed**: tested real-data core E2E rows are `11x` to `52x` faster
  than SCENIC+ on the same inputs.
- **Lower or comparable memory**: peak RSS is lower in the tested real-data rows.
- **Reproducible outputs**: fixed seeds give deterministic threaded runs.

## Benchmark Snapshot

All rows below use the same matrix-level regulatory path on both tools:
TF-to-gene, region-to-gene, eRegulons, gene AUCell and region AUCell.
Hardware: Apple M5 laptop, 16 GB RAM, macOS arm64, Python 3.13.9, 4 CPU threads.

| Dataset | Shape | RustScenic | SCENIC+ | Speedup | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic micro | 150 cells, 80 genes, 30 peaks, 3 TFs | 0.035 s | 9.45 s | 269x | 0.18 / 0.40 GB |
| PBMC3k dense | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 4.98 s | 258.9 s | 52x | 1.21 / 1.26 GB |
| Mouse brain E18 | 1,500 cells, 3,000 genes, 6,000 peaks, 25 TFs | 2.82 s | 90.4 s | 32x | 1.65 / 2.10 GB |
| Human brain GEM-X | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 7.41 s | 146.0 s | 19.7x | 2.18 / 2.19 GB |

Including data preparation, the human brain GEM-X row is `11.89 s` for
RustScenic versus `150.36 s` for SCENIC+.

Full commands, hardware, validation metrics and the complete benchmark table are
in [site_docs/benchmarks.md](site_docs/benchmarks.md).

## What Ships

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

Bundled TF lists are available through `rustscenic.data.tfs("hs")` and
`rustscenic.data.tfs("mm")`. Motif ranking databases are not bundled because the
public Aerts Lab databases are large; pass downloaded rankings to
`rustscenic.cistarget.enrich`.

## Quick Start

Run the CLI pipeline:

```bash
rustscenic pipeline --rna data.h5ad --tfs tfs.txt --output out/
```

Use the Python API:

```python
import anndata as ad
import rustscenic.aucell
import rustscenic.data
import rustscenic.grn

adata = ad.read_h5ad("rna.h5ad")
tfs = rustscenic.data.tfs("hs")

grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=5000, seed=777)

regulons = [
    (f"{tf}_regulon", grn[grn["TF"] == tf].nlargest(50, "importance")["target"].tolist())
    for tf in grn["TF"].unique()
]
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
```

See [examples/pbmc3k_end_to_end.py](examples/pbmc3k_end_to_end.py) for a small
real-data RNA example.

## Validation

- cisTarget AUC kernel agreement against `ctxcore.recovery.aucs`: Pearson
  `1.0000`, mean absolute difference about `2.4e-5`.
- AUCell agreement against pySCENIC on the Ziegler 2021 airway atlas: mean
  per-cell Pearson `0.984`, with `91.7%` of cells above `0.95`.
- Human brain GEM-X SCENIC+ comparison: region-to-gene Jaccard `1.000`, region
  AUCell mean Pearson `0.823`.
- External-user reports cover Kamath dopaminergic neurons and 10x human brain
  multiome data.

Validation artefacts live under [validation/](validation/). Public benchmark
interpretation lives in [site_docs/benchmarks.md](site_docs/benchmarks.md) and
[site_docs/validation.md](site_docs/validation.md).

## Current Scope

RustScenic focuses on the CPU matrix-level SCENIC-style compute path and the
Python/CLI workflow around it. The next benchmark tier is larger real multiome
inputs, repeated runs on a second machine and full workflow coverage that starts
from raw fragments and external motif-ranking databases.

For adjacent tools:

- Use SCENIC+ when you need the full upstream reference workflow.
- Use flashSCENIC when you specifically want a GPU-oriented method.
- Use decoupler when you only need TF activity scoring from prebuilt regulons.

## Documentation

- [Installation](site_docs/installation.md)
- [Quickstart](site_docs/quickstart.md)
- [API map](site_docs/api.md)
- [Benchmarks](site_docs/benchmarks.md)
- [Validation](site_docs/validation.md)
- [Scope](site_docs/limitations.md)

## Repository Layout

- `crates/` - Rust workspace for GRN, AUCell, topics, preprocessing and PyO3 bindings
- `python/rustscenic/` - Python package, CLI entry point and type stubs
- `examples/` - runnable examples on small public datasets
- `validation/` - benchmark scripts, logs and measurement reports
- `tests/` - Python and Rust test suites
- `site_docs/` - public documentation built by MkDocs

## Citation

If you use RustScenic in a paper, report, benchmark, derivative package or lab
workflow, cite the exact release used. GitHub citation metadata is in
[CITATION.cff](CITATION.cff). Zenodo concept DOI:
[10.5281/zenodo.20246040](https://doi.org/10.5281/zenodo.20246040).

RustScenic is created and maintained by Ekin Kahraman. See [AUTHORS.md](AUTHORS.md)
for attribution.

## Contact

File issues at
[github.com/Ekin-Kahraman/rustscenic/issues](https://github.com/Ekin-Kahraman/rustscenic/issues).
