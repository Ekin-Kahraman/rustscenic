<h1 align="center"><code>rustscenic</code></h1>

<p align="center">
  <strong>Faster, memory-efficient regulatory-network analysis for single-cell and multiome data.</strong>
</p>

<p align="center">
  Rust kernels for GRN inference, regulon activity, motif enrichment, topic modelling,
  enhancer links and eRegulons. Python API. CPU-first. One install.
</p>

<p align="center">
  <a href="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml"><img alt="CI" src="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml/badge.svg"></a>
  <a href="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://pypi.org/project/rustscenic/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rustscenic"></a>
  <a href="https://doi.org/10.5281/zenodo.20246040"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20246040.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-stable-orange"></a>
</p>

<p align="center">
  <a href="https://ekin-kahraman.github.io/rustscenic/">Documentation</a> |
  <a href="site_docs/benchmarks.md">Benchmarks</a> |
  <a href="CITATION.cff">Citation</a>
</p>

## Highlights

- `pip install rustscenic`
- Python 3.10 to 3.13 release wheels
- Rust implementations for the matrix-heavy regulatory-network stages
- `11x` to `52x` faster than SCENIC+ in tested real-data core E2E rows
- Lower peak RSS than SCENIC+ in every tested real-data row
- Core path runs without Java, dask, CUDA or Snakemake
- Benchmark artefacts include commands, hardware, runtime, memory and output checks

## Installation

```bash
pip install rustscenic
```

## SCENIC+ Benchmark

Core E2E comparison on the same matrix-level path: TF-to-gene, region-to-gene,
eRegulons, gene AUCell and region AUCell.

Machine: Apple M5 laptop, 16 GB RAM, macOS arm64, Python 3.13.9, 4 CPU threads.

| Dataset | Shape | RustScenic | SCENIC+ | Speedup | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBMC3k dense | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 4.98 s | 258.9 s | 52x | 1.21 / 1.26 GB |
| Mouse brain E18 | 1,500 cells, 3,000 genes, 6,000 peaks, 25 TFs | 2.82 s | 90.4 s | 32x | 1.65 / 2.10 GB |
| Human brain GEM-X | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 7.41 s | 146.0 s | 19.7x | 2.18 / 2.19 GB |

Including data preparation, the human brain GEM-X row is `11.89 s` for
RustScenic versus `150.36 s` for SCENIC+.

Full commands, hardware, validation metrics and output signatures are in
[site_docs/benchmarks.md](site_docs/benchmarks.md).

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

## Quick Start

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

Command line:

```bash
rustscenic pipeline --rna data.h5ad --tfs tfs.txt --output out/
```

See [examples/pbmc3k_end_to_end.py](examples/pbmc3k_end_to_end.py) for a small
real-data RNA example.

## Validation

- cisTarget AUC kernel agreement against `ctxcore.recovery.aucs`: Pearson
  `1.0000`, mean absolute difference about `2.4e-5`.
- AUCell agreement against pySCENIC on the Ziegler 2021 airway atlas: mean
  per-cell Pearson `0.984`, with `91.7%` of cells above `0.95`.
- Human brain GEM-X comparison: region-to-gene Jaccard `1.000`, region AUCell
  mean Pearson `0.823`.
- External-user reports cover Kamath dopaminergic neurons and 10x human brain
  multiome data.

Validation artefacts live under [validation/](validation/). Public interpretation
lives in [site_docs/benchmarks.md](site_docs/benchmarks.md) and
[site_docs/validation.md](site_docs/validation.md).

## Documentation

- [Installation](site_docs/installation.md)
- [Quickstart](site_docs/quickstart.md)
- [API map](site_docs/api.md)
- [Benchmarks](site_docs/benchmarks.md)
- [Validation](site_docs/validation.md)
- [Scope](site_docs/limitations.md)

## Citation

If you use RustScenic in a paper, report, benchmark, derivative package or lab
workflow, cite the exact release used. GitHub citation metadata is in
[CITATION.cff](CITATION.cff). Zenodo concept DOI:
[10.5281/zenodo.20246040](https://doi.org/10.5281/zenodo.20246040).

RustScenic is created and maintained by Ekin Kahraman. See [AUTHORS.md](AUTHORS.md)
for attribution.
