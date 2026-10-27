# RustScenic quickstart

Infer a gene network and score candidate gene-set activity on PBMC3k with
**v0.5.0**. This RNA example downloads data on first use and caches it locally.
Runtime depends on the data, settings and CPU.

## Install

Use regular CPython **3.10–3.14** in an active environment; see
[Installation](site_docs/installation.md) for setup.

```bash
python -m pip install "rustscenic[examples]"
```

## RNA example

Run the block below in a notebook, or save it as `pbmc3k_example.py` and run
`python pbmc3k_example.py`. Internet access is needed for the first download.

The small transcription-factor list and 100-tree limit keep this demonstration
manageable. Candidate gene sets are not motif-filtered or split by correlation
sign; their activity scores do not establish causal regulation.

```python
from pathlib import Path

import scanpy as sc
import rustscenic.aucell
import rustscenic.grn

# Downloads about 5.9 MB on first use; later calls use the local cache.
adata = sc.datasets.pbmc3k()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# A short TF list for this example; use a full matching list for your study.
tfs = ["SPI1", "CEBPD", "MAFB", "CEBPB", "KLF4", "IRF8", "PAX5", "EBF1",
       "TCF7", "LEF1", "TBX21"]
tfs = [tf for tf in tfs if tf in adata.var_names]
grn = rustscenic.grn.infer(
    adata, tf_names=tfs, seed=777, n_estimators=100
)
regulons = {
    tf: group.nlargest(50, "importance")["target"].tolist()
    for tf, group in grn.groupby("TF")
    if len(group) >= 10
}
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)

adata.write_h5ad("pbmc3k.h5ad")
Path("tfs.txt").write_text("\n".join(tfs) + "\n")
grn.to_parquet("grn.parquet")
auc.to_parquet("aucell.parquet")
print(grn.nlargest(5, "importance"))
print(auc.iloc[:3, :3])
```

A successful run prints network edges and a preview of activity scores.
Files are saved in the current working directory:

| File | Contents |
| --- | --- |
| `grn.parquet` | Network edges: `TF`, `target`, `importance` |
| `aucell.parquet` | Activity scores: cells × candidate gene sets |
| `pbmc3k.h5ad` | Prepared RNA matrix |
| `tfs.txt` | Matching TF names, one per line |

For signed target sets, use `add_correlation` and `build_regulons`; see the
[API guide](site_docs/api.md). For your own AnnData file, use the
[own-data quickstart](site_docs/quickstart.md).

## Command line

With the prepared matrix and TF list above, run the same two stages:

```bash
rustscenic grn --expression pbmc3k.h5ad --tfs tfs.txt --output grn.parquet --seed 777 --n-estimators 100
rustscenic aucell --expression pbmc3k.h5ad --regulons grn.parquet --output aucell.parquet
```

The package includes species-specific TF lists through `rustscenic.data.tfs`.
You can also supply the [Aerts lab human TF list](https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt).
Match its gene identifiers to your expression matrix.

## Add chromatin and motif analysis

These are individual stages with separate inputs. A complete multiome analysis
also needs preprocessing, enhancer links and eRegulon assembly; see
[`rustscenic.pipeline.run`](site_docs/api.md) and the [scope guide](site_docs/limitations.md).

Given a prepared ATAC AnnData matrix (`atac_adata`) and a compatible motif-ranking
database, extend the RNA example with:

```python
import rustscenic.cistarget
import rustscenic.topics

# Cells × peaks, with count or binary values.
topics = rustscenic.topics.fit(atac_adata, n_topics=50, n_passes=15, seed=777)

# Ranking columns must match the gene identifiers in the candidate sets.
rankings = rustscenic.cistarget.load_aertslab_feather("hg38_screen_v10.feather")
enrichments = rustscenic.cistarget.enrich(rankings, regulons, top_frac=0.05)
```

`fit_gibbs` is also available for sparse scATAC topic modelling. Inspect topic
quality and convergence before interpreting either model. Motif-annotation
pruning requires annotation inputs; see the [API guide](site_docs/api.md).

## More examples and benchmarks

From a repository checkout:

```bash
python examples/pbmc3k_end_to_end.py
```

This RNA example downloads 10x data on first use, writes outputs to
`examples/out/`, and includes a marker-based lineage sanity check. The gene
sets are candidates, not motif-validated regulons.

For multiome inputs, [download the 10x PBMC3k matrix](https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5)
and follow the [multiome validation example](validation/validate_multiome_e2e.py).

Use the [benchmark records](site_docs/benchmarks.md) for measured runtime,
memory, hardware and comparison conditions. The [validation summary](validation/VALIDATION_SUMMARY.md)
retains earlier development evidence.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Empty expression matrix | Inspect the shape and filtering thresholds before fitting. |
| Duplicate gene names | Resolve duplicate biological identifiers before analysis; renaming alone does not combine expression values. |
| No TF or gene-set overlap | Match species and identifiers, such as gene symbols versus Ensembl IDs. |
| Few topics carry most cells | Inspect [topic-model limitations](docs/topic-collapse.md) and compare model settings. |

[Installation](site_docs/installation.md) · [HPC guide](site_docs/hpc.md) ·
[Release notes](docs/releases/v0.5.0.md) ·
[Report a problem](https://github.com/Ekin-Kahraman/rustscenic/issues/new/choose)

---

Ekin Kahraman · Mohamed El Moussaoui · Kuan-Lin Huang · Longtao Huangfu · Fuaad Siyaad

[Contributions](AUTHORS.md)
