# 5-minute tester quickstart

For collaborators trying rustscenic on their own data. The goal of
this page is to get you from **`pip install`** to **first useful
output** in under five minutes, and to tell you exactly what to send
back if anything is off.

## Install

```bash
pip install --upgrade git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.6
```

Wheels are also at the [latest release page](https://github.com/Ekin-Kahraman/rustscenic/releases/latest)
if your network can't pull from git directly.

Requires Python 3.10–3.13. Linux + macOS only (Windows untested).
Brings five runtime deps: numpy, pandas, pyarrow, scipy, anndata.

If you are running the example or validation scripts, install the
validation extra in one line (v0.3.6+):

```bash
pip install --upgrade "rustscenic[validation] @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.6"
```

That adds the ecosystem packages used outside rustscenic core: scanpy,
anndata, igraph, leidenalg, and scikit-learn.

From a source checkout, use:

```bash
pip install -e ".[validation]"
```

For SCENIC-ecosystem parity benchmarks (pyscenic, arboreto, ctxcore), use
`[reference]`. For topic-model benchmarks (tomotopy, gensim), use `[benchmarks]`.
The canonical reproducible reference path remains the pinned Docker image at
`validation/reference/Dockerfile`.

## RNA-only smoke test (10 lines)

If you have a scRNA AnnData and a TF list, this is the smallest run
that exercises GRN + AUCell:

```python
import anndata as ad
import rustscenic.grn
import rustscenic.aucell

adata = ad.read_h5ad("your_data.h5ad")
tfs = ["SPI1", "PAX5", "TCF7"]   # or rustscenic.data.tfs("human")

grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=50)
regulons = [(tf, grn[grn.TF == tf].nlargest(20, "importance").target.tolist())
            for tf in tfs]
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)
print(auc.head(), auc.attrs["regulon_coverage"])
```

If `auc.attrs["regulon_coverage"]` shows `0/N` for any regulon,
that's a coverage warning — paste the warning text and the regulon
name back to me and I'll tell you why.

## What to expect on cellxgene-format AnnData

If `adata.var_names[0]` looks like `ENSG00000…`, rustscenic prints:

```
UserWarning: var_names look like ENSEMBL IDs (e.g. 'ENSG…'); using
`var['feature_name']` for gene-symbol matching (cellxgene/10x
convention). First three swaps: [(...), (...), (...)]
```

That's the auto-swap firing. If it doesn't fire on data you know is
ENSEMBL, that's a bug — please report.

## Datasets I've already validated against

- **Kamath et al. 2022** (cellxgene asset
  `f25a8375-1db5-49a0-9c85-b72dbe5e2a92`, OPC cells, 13,691 × 33,295).
  Validation script: `validation/kamath/validate_kamath_fix.py`. If
  you want to reproduce or compare against this baseline, run that
  script first.
- **Ziegler 2021** airway atlas (scaling benchmark, 1k → 50k cells).

If you pick a dataset I haven't tested yet, that's exactly what we
want — the gaps left in our coverage are dataset-shape-specific.

## What to send back

If the run looks wrong, please paste:
1. The exact `pip install` line you ran (so we know which version)
2. The shape: `print(adata.shape, adata.X.dtype)`
3. The first ~20 lines of `var_names` and the columns of `adata.var`
4. **All warning text** that came out of the rustscenic call
5. The output of the function — first few rows + `.shape`

Then I can usually tell you within an hour whether it's a known
class of bug, a config issue on your side, or something new for me
to fix.

For performance or biological validation runs, send one row per dataset with:

1. Rustscenic version, command, Python version, OS, CPU, and RAM
2. Cells, genes, peaks, nonzeros, and whether the matrix is sparse or dense
3. Wall time and peak RSS for each stage: GRN, AUCell, topics, cistarget, enhancer
4. Topic method, topic count, seed, and number of unique top-1 topics
5. Marker consistency, ARI/NMI where available, and top regulons per cluster

If Online VB gives few unique top-1 topics on small or sparse ATAC data, rerun
the topic stage with `topics_method="gibbs"` and the same seed before treating
topic quality as a RustScenic biology failure.

## Where to ask

Post in our Slack thread — quickest. Or open a GitHub issue at
[Ekin-Kahraman/rustscenic/issues](https://github.com/Ekin-Kahraman/rustscenic/issues)
if it's a clear repro you want tracked.
