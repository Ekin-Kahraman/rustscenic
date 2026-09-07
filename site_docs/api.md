# API Map

RustScenic exposes the workflow as small Python calls. Each stage can be used
alone, or combined through `rustscenic.pipeline.run`.

**Version note:** this page describes the development branch. The correlation
helpers, `early_stop_mode`, and pipeline polarity options below are planned for
v0.5.0 and are not in the published v0.4.7 package. For an example that works
with `pip install rustscenic`, use the [quickstart](quickstart.md).

## RNA Regulatory Network

```python
rustscenic.grn.infer(
    adata,
    tf_names,
    n_estimators=500,
    early_stop_mode="arboreto",
    early_stop_window=25,
    seed=777,
)
```

Returns a `pandas.DataFrame` with transcription factor, target and importance columns.

Use this when you need a GRNBoost2-style TF-target edge table without depending
on arboreto and dask at runtime. The default monitor stops when the strict mean
of the last 25 scikit-learn-style OOB improvements is negative, including the
stopping tree, matching arboreto's trailing-window rule. Set
`early_stop_mode="legacy_inbag"` to reproduce RustScenic's historical in-bag
two-point monitor, or `early_stop_window=0` to disable stopping. With
`subsample=1.0`, arboreto mode has no OOB rows and also fits to the estimator
ceiling.

Flat target scheduling no longer blocks targets. `target_block_size` remains
accepted for API compatibility but does not change execution.

## TF-Target Correlation and Polarity

```python
signed = rustscenic.grn.add_correlation(
    grn,
    adata,
    rho_threshold=0.03,
    mask_dropouts=False,
)
regulons = rustscenic.grn.build_regulons(
    signed,
    top_targets_per_tf=50,
    min_targets=10,
    include_repressors=True,
)
```

`add_correlation` adds Pearson `rho` and `regulation`: `1` above `+0.03`,
`-1` below `-0.03`, and `0` otherwise. Constant or insufficient pairs are
neutral. Dense and sparse inputs use deterministic Rust kernels. Neutral edges
are never silently assigned to a regulon. Names are `<TF>_activator` and
`<TF>_repressor` and remain signed through cisTarget, eRegulon assembly and
AUCell.

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

Use Online VB for smaller or faster exploratory runs. Use collapsed Gibbs for
higher topic diversity at larger `K`.

## ATAC Preprocessing

```python
rustscenic.preproc.fragments_to_matrix("fragments.tsv.gz", "peaks.bed")
```

Returns an AnnData peak matrix suitable for topic modelling.

## Pipeline

```python
rustscenic.pipeline.run(
    rna=adata,
    tfs=tfs,
    output_dir="out",
    grn_early_stop_mode="arboreto",
    grn_regulon_polarities="both",
)
```

Use the orchestrator when you want the full staged workflow and a manifest.

When `region_motif_rankings` points to a parquet or feather file, the
orchestrator reads only the motif ID column plus the peak columns used by the
current run. This keeps large region-ranking databases from being loaded in
full. Peak IDs in the BED or ATAC matrix must match the ranking database
region IDs.

`grn_regulon_polarities="both"` is the correctness default. Use
`"activating"` to retain only positive TF-target programmes, or `"unsigned"`
as the documented legacy migration path. The manifest records the correlation
threshold, edge counts, early-stop mode and fitted-tree summary.
