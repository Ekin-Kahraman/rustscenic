# Lab Adoption

This page is for collaborators deciding whether RustScenic is worth testing on
their own data.

## When To Try It

RustScenic is worth testing if one of these problems is blocking your analysis:

- pySCENIC, arboreto, pycisTopic or pycistarget will not install cleanly on a
  modern Python stack.
- AUCell, cisTarget, enhancer linking or GRN inference is becoming a runtime or
  memory bottleneck.
- You need deterministic CPU execution under a fixed seed.
- You want a single Python package covering GRN, AUCell, motif support, topics,
  enhancer links and eRegulons.

## Public Proof Points

Use these when introducing the package to a lab or collaborator:

| Proof point | Evidence |
| --- | --- |
| Released | `v0.5.0` on PyPI, with Python 3.10 to 3.13 release wheels. |
| Faster tested path | `11x` to `52x` faster than SCENIC+ on sampled real-data inputs in a single-machine output-path benchmark. |
| Historical atlas-scale check | RustScenic `v0.3.2` synthetic 100k-cell seven-stage run peaked at `7.09 GB` RSS; it is not a v0.5.0, default-parameter or reference-memory result. |
| Lab artefact | Collaborator human brain GEM-X full monolith run recovered 16 of 17 expected brain TFs. |

## Minimal Adoption Test

Run one small dataset first:

```bash
pip install rustscenic
python -m rustscenic.quickstart
```

Then run your own AnnData object through GRN plus AUCell:

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
