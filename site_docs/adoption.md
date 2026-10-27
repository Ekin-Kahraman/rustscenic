# Lab Adoption

This page is for collaborators deciding whether RustScenic is worth testing on
their own data.

## When To Try It

RustScenic is worth testing if one of these problems is blocking your analysis:

- pySCENIC, arboreto, pycisTopic or pycistarget will not install cleanly on a
  modern Python stack.
- AUCell, cisTarget, enhancer linking or GRN inference is becoming a runtime or
  memory bottleneck.
- You need reproducible CPU runs with recorded seeds, thread counts and settings.
- You want a single Python package covering GRN, AUCell, motif support, topics,
  enhancer links and eRegulons.

## Public Proof Points

Use these when introducing the package to a lab or collaborator:

| Proof point | Evidence |
| --- | --- |
| Published package | [v0.5.1](https://pypi.org/project/rustscenic/0.5.1/), with prebuilt CPU packages. See [installation](installation.md) for platforms and Python qualification details. |
| Faster tested path | `11x` to `52x` faster than SCENIC+ for selected stages on sampled data, measured on one machine. |
| Memory measurement | **3× faster with approximately 80% lower peak measured memory** than sparse Dask arboreto on **1.3 million real cells**, across three matched seeds (exact: 3.06×, 78.7%). See [benchmark scope](benchmarks.md#13-million-real-cell-comparison). |
| Lab artefact | Collaborator human brain workflow recovered 16 of 17 expected brain transcription factors. |

## Minimal Adoption Test

Run one small dataset first:

```bash
pip install rustscenic
python -m rustscenic.quickstart
```

For a real-data walkthrough, [open the PBMC3k notebook](https://colab.research.google.com/github/Ekin-Kahraman/rustscenic/blob/main/examples/pbmc3k_quickstart.ipynb).
It installs the published `0.5.1` wheel, runs GRN inference and candidate
gene-set scoring, and exports results with dataset, version and settings
provenance. It demonstrates software execution; interpreting regulatory
biology requires the [additional validation steps](validation.md).

Then infer a gene network from your own AnnData object. The
[quickstart](quickstart.md) adds activity scoring:

```python
import anndata as ad
import rustscenic.data
import rustscenic.grn
import rustscenic.aucell

adata = ad.read_h5ad("your_data.h5ad")
tfs = rustscenic.data.tfs("hs")
grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=500, seed=777)
```

## Evidence To Record

For a useful adoption report, record:

- Dataset name and source.
- Number of cells, genes, peaks and transcription factors.
- RustScenic version.
- Python version and operating system.
- Command or script.
- Wall time and peak memory if available.
- Output checks: number of GRN edges, regulons, AUCell matrix shape, non-empty motif results.
- Biological checks: expected marker TFs and their expected cell types.

## What Counts As A Strong Adoption Result

A strong report does not need to prove RustScenic is better than every reference tool. It should prove:

- The tool installs.
- It runs on real data without manual source edits.
- Outputs are non-empty and shape-correct.
- Known biological controls are recovered or failures are explained.
- The command is reproducible by another person.

## Best First Dataset

Use a small real dataset before a full atlas. The most useful first adoption
run has enough cells to show biological signal, but not so many that debugging
becomes a compute problem. A good first target is 1,000 to 5,000 cells with a
known set of marker TFs.
