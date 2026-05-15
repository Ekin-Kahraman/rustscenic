# API Map

## RNA Regulatory Network

```python
rustscenic.grn.infer(adata, tf_names, n_estimators=500, seed=777)
```

Returns a `pandas.DataFrame` with transcription factor, target and importance columns.

Use this when you need a GRNBoost2-style TF-target edge table.

## AUCell

```python
rustscenic.aucell.score(adata, regulons, top_frac=0.05)
```

Returns a cells by regulons activity matrix.

Use this when you already have regulons and need per-cell TF programme activity.

## cisTarget

```python
rustscenic.cistarget.enrich(rankings, regulons, nes_threshold=3.0)
```

Returns motif enrichment rows with AUC and NES values.

Use this for motif support filtering of candidate regulons.

## Topics

```python
rustscenic.topics.fit(atac_adata, n_topics=30)
rustscenic.topics.fit_gibbs(atac_adata, n_topics=30, n_threads=8)
```

Use Online VB for smaller or faster exploratory runs. Use collapsed Gibbs for higher topic diversity at larger `K`.

## ATAC Preprocessing

```python
rustscenic.preproc.fragments_to_matrix("fragments.tsv.gz", "peaks.bed")
```

Returns an AnnData peak matrix suitable for topic modelling.

## Pipeline

```python
rustscenic.pipeline.run(rna=adata, tfs=tfs, output_dir="out")
```

Use the orchestrator when you want the full staged workflow and a manifest.
