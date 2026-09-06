<h1 align="center">RustScenic</h1>

<p align="center">
  <strong>Fast, memory-efficient gene-regulation analysis for single-cell data.</strong>
</p>

<p align="center">
  Infer gene networks and score their activity using RNA and chromatin-accessibility data.
  A Python package accelerated with Rust. Runs on CPUs; no GPU required.
</p>

<p align="center">
  Created and maintained by Ekin Kahraman, developed in collaboration with the
  Kuan-Lin Huang Lab at the Icahn School of Medicine at Mount Sinai.
</p>

<p align="center">
  <a href="https://ekin-kahraman.github.io/rustscenic/">Documentation</a> |
  <a href="site_docs/benchmarks.md">Benchmarks</a> |
  <a href="site_docs/validation.md">Validation</a> |
  <a href="CITATION.cff">Citation</a> |
  <a href="https://doi.org/10.5281/zenodo.20246040">Zenodo DOI</a>
</p>

<p align="center">
  <a href="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml"><img alt="CI" src="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/audit.yml/badge.svg"></a>
  <a href="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/nightly-real-data.yml"><img alt="Nightly real-data validation" src="https://github.com/Ekin-Kahraman/rustscenic/actions/workflows/nightly-real-data.yml/badge.svg"></a>
  <a href="https://pypi.org/project/rustscenic/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rustscenic"></a>
  <br>
  <a href="https://doi.org/10.5281/zenodo.20246040"><img alt="Zenodo DOI" src="https://img.shields.io/badge/DOI-Zenodo-1682d4"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/Rust-stable-orange"></a>
</p>

## Highlights

- Gene-network inference on **1.3 million mouse-brain cells** in under 47 minutes,
  with **4.28 GB** peak memory during analysis on 16 CPU cores
  ([v0.5.0 candidate benchmark](https://github.com/Ekin-Kahraman/rustscenic/blob/0c8eb00539e3860c78e452c8661cc2735c169386/validation/scaling/IFB_REAL_RNA_GRN_2026-08-28.md)).
- **3.3x faster with about 81% less peak physical memory than arboreto** in a
  controlled 20,000-cell gene-network comparison on the same hardware.
- **21.4% lower topic-model peak memory**, with unchanged output files in repeated
  mouse-brain tests ([v0.5.0 candidate memory audit](https://github.com/Ekin-Kahraman/rustscenic/blob/0c8eb00539e3860c78e452c8661cc2735c169386/validation/scaling/IFB_SCALE_2026-08-28.md#compact-gibbs-token-audit)).
- `11x` to `52x` faster than SCENIC+ for selected analysis stages on sampled real-data inputs, measured on one machine.
- Huang Lab collaborator run recovered `16/17` expected brain transcription factors in human brain data.

The first three results use the **v0.5.0 release candidate**, not the current PyPI
release. The million-cell run used prepared RNA and 2,095 selected genes;
separate full-data preparation peaked at **71.49 GB**. These measurements do not
describe a complete million-cell spatial workflow.

Current release: `v0.4.7`. Python 3.10 to 3.13; Linux, macOS and Windows.
Core analysis runs without Java, dask, CUDA or Snakemake.

## Installation

```bash
pip install rustscenic
```

## Benchmark Evidence

The SCENIC+ comparison starts from prepared matrices and measures gene-network
inference, enhancer links and activity scores. It excludes raw-data processing,
topic modelling and motif-database construction.

The tools use different methods for enhancer linking: RustScenic
uses correlation over the fixed search space, while SCENIC+ uses boosted trees plus
Pearson scoring for region-to-gene links.

Machine: Apple M5 laptop, 16 GB RAM, macOS arm64, 4 CPU threads. RustScenic
rows used Python 3.13.9; SCENIC+ reference rows used Python 3.11.8 for its
dependency stack.
Rows can be sampled subsets; the shape column is the actual benchmark input.

| Dataset | Shape | RustScenic | SCENIC+ | Speedup | Peak RSS (RustScenic / SCENIC+) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBMC3k dense | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 4.98 s | 258.9 s | 52x | 1.21 / 1.26 GB |
| PBMC10k dense | 2,000 sampled cells, 4,000 genes, 8,000 peaks, 30 TFs | 21.5 s | 241.5 s | 11x | 2.37 / 2.63 GB |
| Mouse brain E18 | 1,500 cells, 3,000 genes, 6,000 peaks, 25 TFs | 2.82 s | 90.4 s | 32x | 1.65 / 2.10 GB |
| Human brain GEM-X | 2,000 cells, 4,000 genes, 8,000 peaks, 30 TFs | 7.41 s | 146.0 s | 19.7x | 2.18 / 2.19 GB |

Including data preparation, the human brain GEM-X row is `11.89 s` for
RustScenic versus `150.36 s` for SCENIC+.

Full commands, hardware, validation metrics and output signatures are in
[site_docs/benchmarks.md](site_docs/benchmarks.md).

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

## Quick Start

This example works with the published **v0.4.7** package. It builds candidate
gene sets from network edges and scores their activity; these sets have not
been filtered for motif support or split by positive and negative correlation.

```python
import anndata as ad
import rustscenic.aucell
import rustscenic.data
import rustscenic.grn

adata = ad.read_h5ad("rna.h5ad")
tfs = rustscenic.data.tfs("hs")

grn = rustscenic.grn.infer(
    adata, tf_names=tfs, n_estimators=5000, top_targets_per_tf=50, seed=777
)
regulons = {
    tf: group["target"].tolist()
    for tf, group in grn.groupby("TF")
    if len(group) >= 10
}
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
```

Command line:

```bash
rustscenic pipeline --rna data.h5ad --tfs tfs.txt --output out/
rustscenic grn --expression rna.h5ad --tfs tfs.txt --output grn.parquet
rustscenic aucell --expression rna.h5ad --regulons grn.parquet --output aucell.parquet
rustscenic topics --expression atac.h5ad --output topics.parquet --n-topics 30
rustscenic cistarget --rankings rankings.feather --regulons regulons.tsv --output motifs.parquet
```

See [examples/pbmc3k_end_to_end.py](examples/pbmc3k_end_to_end.py) for a small
real-data RNA example.

The development branch adds `add_correlation`, `build_regulons`, and
`rustscenic add-cor`. These are planned for v0.5.0 and are **not available in
the current PyPI release**. See the [API map](site_docs/api.md) for those features.

## Validation

| Validation axis | Result |
| --- | --- |
| cisTarget kernel | Pearson `1.0000` against `ctxcore.recovery.aucs`; mean absolute difference about `2.4e-5`. |
| AUCell parity | Ziegler 2021 airway atlas mean per-cell Pearson `0.984`; `91.7%` of cells above `0.95`. |
| Human brain GEM-X benchmark | Region-to-gene Jaccard `1.000`; region AUCell mean Pearson `0.823`. |
| Collaborator analysis | A human brain RNA/chromatin workflow recovered `16/17` expected brain transcription factors. |
| Open parity targets | Gene AUCell Pearson `0.386` and eRegulon edge Jaccard `0.161` on the human brain GEM-X row. |

Validation artefacts live under [validation/](validation/). Public interpretation
lives in [site_docs/benchmarks.md](site_docs/benchmarks.md) and
[site_docs/validation.md](site_docs/validation.md).

## Current Boundaries

- The SCENIC+ speedups cover selected analysis stages, not every possible
  raw-data and motif-database workflow.
- GRN, gene AUCell and eRegulon edge agreement are not claimed to be
  bit-identical to SCENIC+; see [Benchmarks](site_docs/benchmarks.md) for the
  parity metrics.
- The development branch changes `grn.infer` to arboreto-compatible early stopping.
  Use `early_stop_mode="legacy_inbag"` only to reproduce historical
  RustScenic stopping behaviour. The pipeline defaults to separate activator
  and repressor regulons; `grn_regulon_polarities="unsigned"` is the explicit
  compatibility path.
- Complete million-cell spatial workflows and an atlas-wide CELLxGENE resource
  remain outside the validated scope.

## Documentation

- [Installation](site_docs/installation.md)
- [Quickstart](site_docs/quickstart.md)
- [API map](site_docs/api.md)
- [HPC operation](site_docs/hpc.md)
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
