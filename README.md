<p align="center">
  <img src="https://raw.githubusercontent.com/Ekin-Kahraman/rustscenic/main/site_docs/assets/branding/rustscenic-logo-256.png" width="96" height="96" alt="RustScenic logo">
</p>

<h1 align="center">RustScenic</h1>

<p align="center">
  <strong>Predict how genes are regulated from single-cell data.</strong><br>
  Fast regulatory-network analysis in Rust, with a Python API.
</p>

<p align="center">
  Created by <strong>Ekin Kahraman</strong>, developed and scaled with the
  <strong>Huang Lab at Mount Sinai</strong>.
</p>

<p align="center">
  <a href="https://ekin-kahraman.github.io/rustscenic/">Documentation</a> ·
  <a href="https://github.com/Ekin-Kahraman/rustscenic/blob/main/site_docs/benchmarks.md">Benchmarks</a> ·
  <a href="https://github.com/Ekin-Kahraman/rustscenic/tree/main/examples/cellxgene-endothelial">Try the example</a> ·
  <a href="https://doi.org/10.5281/zenodo.20246040">Cite the software</a>
</p>

- **1.3 million real cells:** GRN inference in **46m42s on 16 CPU cores**,
  using 2,095 selected genes and 256 transcription factors.
- **2.04–3.33× faster than arboreto** in matched 20,000-cell comparisons;
  **81% lower physical memory** in the measured Intel-node comparison.
- **704 human cell-ontology labels:** precomputed exploratory networks from
  CELLxGENE Census, with source attribution and reproducible methods.

## Try a network

[Download the small endothelial-cell example](https://raw.githubusercontent.com/Ekin-Kahraman/rustscenic/main/examples/cellxgene-endothelial/starter.zip), extract it and
open `explore.html`. Choose a regulator to see its predicted targets.
No installation, HPC connection or new model fitting is needed.

The example includes the complete **184,000-edge network**, **85 motif-supported
regulons**, activity summaries and source-study citations.

For Python users:

```bash
python3 example.py --tf ETS1
```

## Performance

![RustScenic real-RNA scaling and matched arboreto comparisons](docs/benchmarks/real-rna-20260828/figure.png)

The million-cell result measures GRN inference. Separate input preparation
took **7m18s** and peaked at **71.5 GB**, versus **4.28 GB** in the measured
inference run. Comparisons use matching inputs and worker counts; each is a
single observation, not a repeated-run average.

[Exact measurements and conditions](docs/benchmarks/real-rna-20260828/benchmark-source-record.md) ·
[SCENIC+ comparisons and output agreement](https://github.com/Ekin-Kahraman/rustscenic/blob/main/site_docs/benchmarks.md)

## Analyse your own data

Current release: `v0.5.0`.

```bash
pip install rustscenic
```

RustScenic supports RNA regulatory-network inference, motif enrichment,
regulon activity scoring and multiome analysis through a Python API and CLI.
See the [quickstart](https://ekin-kahraman.github.io/rustscenic/quickstart/).

## CELLxGENE-derived resource

The collection contains **137.8 million retained edges** and **51,527
motif-supported regulons** across 704 human cell-ontology labels, drawing on
565 source datasets. Each cohort uses up to 5,000 balanced cells.

All outputs passed integrity and content-consistency checks. These are
exploratory predictions; motif support and successful execution do not
establish biological accuracy. Pooled cohorts span tissues and disease
annotations, and activity scores are not directly comparable across cohorts.

**Release status:** the collection is verified and packaged; the CELLxGENE
submission enquiry has been sent. Public resource and citation links will
be added when the release is available.

## Citation and contribution

Cite [RustScenic](https://doi.org/10.5281/zenodo.20246040), the resource citation
when available, CELLxGENE and the contributing studies supplied with the data.
Software: Apache-2.0. The resource has a separate data-licence notice.

Questions, bugs and reproducible examples are welcome in
[GitHub Issues](https://github.com/Ekin-Kahraman/rustscenic/issues).
