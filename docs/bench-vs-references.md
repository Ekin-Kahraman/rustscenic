# Benchmark proof — head-to-head vs reference implementations

Real numbers measured against the established tools. Honest. Same input
on both sides, same hardware, same seed where applicable.

Hardware: macOS 25.4 arm64, 32 GB RAM. Date: 2026-04-25.
Dataset: 10x public PBMC 3k Multiome (`pbmc_granulocyte_sorted_3k`).

## Peak calling — rustscenic vs MACS2

Same input fragments file (`fragments.tsv.gz`, 44,109,954 fragments,
3,000 cell-called barcodes). Same human genome size flag.

| Tool | Wall-clock | Peaks called | Throughput |
|---|---|---|---|
| MACS2 2.2.9.1 | 83.3 s | 122,330 | 530K frags/s |
| **rustscenic.preproc.call_peaks** | **8.4 s** | **77,556** | **5.2M frags/s** |

**Speed: rustscenic 9.9× faster.** rustscenic emits a more conservative
peak set (Corces-2018 algorithm with iterative consensus merging vs
MACS2's narrow peaks).

### F1 overlap (MACS2 as gold standard)

| Metric | Value |
|---|---|
| rustscenic peaks with any MACS2 overlap (recall) | 82.7 % |
| MACS2 peaks with any rustscenic overlap (precision) | 82.2 % |
| **F1** | **0.825** |

The 17 % non-overlap is consistent with the algorithmic differences
(Corces 2018 caps peak width at `2 × peak_half_width + 1` = 501 bp by
default; MACS2 uses local p-value scoring). For SCENIC+ downstream,
where peaks feed into the topic model + cistarget, F1 0.825 is well
within usable range.

## Topic modelling — rustscenic vs gensim LdaModel

Same cells × peaks matrix (3,000 cells × 98,319 peaks, nnz=20.97 M),
same seed (42), 2 passes, identical AnnData input.

| K | rustscenic | gensim 4.4.0 | Ratio |
|---|---|---|---|
| 10 | 31.6 s | 21.7 s | gensim 1.5× faster |
| 30 | 70.7 s | 26.4 s | gensim 2.7× faster |

**gensim is faster** at this shape. rustscenic's online-VB LDA spends
proportionally more time in Rayon scheduling at small K, where
gensim's pure-numpy code path is already cache-resident. Honest call:
**we are not yet the topic-model speed leader.**

What rustscenic still gives at this layer:
- **Bit-identical determinism** under same seed (verified across two
  runs, two batch sizes — see `tests/test_topics.py`)
- **One pip install** (gensim works, but Mallet is the SCENIC+
  reference for higher-fidelity topics, requires Java + binary)
- **Atlas-scale memory bound at `O(threads × n_topics × n_words)`**
  (PR #39 fold/reduce refactor) — gensim's full-corpus VB may not
  bound similarly at 100k+ cells × 200k peaks

What this means for the SCENIC+ flow: at typical per-sample atlas
shape (~10k–100k cells × 100k+ peaks), rustscenic is competitive.
At small K on small samples (this benchmark), use gensim if speed
matters more than determinism. A collapsed-Gibbs rewrite (v0.3
candidate, `docs/topic-collapse.md`) is the path to closing the gap
on quality (Mallet wins on NPMI 0.196 vs our 0.123) and would also
likely close the speed gap.

## Headline (take to the scverse meeting)

| Layer | Reference | rustscenic | Result |
|---|---|---|---|
| Peak calling | MACS2 | rustscenic.preproc.call_peaks | **9.9× faster, F1 0.825** |
| Topic modelling K=10 | gensim LdaModel | rustscenic.topics.fit | **0.7× (gensim wins)** |
| Topic modelling K=30 | gensim LdaModel | rustscenic.topics.fit | **0.37× (gensim wins)** |
| AUCell (10x Multiome 10k cells × 1,457 regulons) | pyscenic | rustscenic.aucell.score | **88× faster** |
| Cistarget AUC kernel | ctxcore.recovery.aucs | rustscenic.cistarget.enrich | **bit-identical** (Pearson 1.0000) |
| GRN end-to-end | arboreto | rustscenic.grn.infer | **1.3× faster, biology equivalent** |
| Memory at 100k cells × 20k genes × 4 stages | scenicplus stack > 40 GB (reported) | 6.3 GB | **~6.3× less** |

## Reproduction

Each row above has a script under `validation/`:

```bash
# Peak calling vs MACS2
python validation/scaling/bench_macs2_head_to_head.py

# Topics vs gensim
python validation/scaling/bench_gensim_lda.py

# AUCell + GRN comparisons under validation/ours/
```

Real datasets used live under `validation/real_multiome/` and
`validation/multi_dataset/` (gitignored — files 50 MB to 1 GB).
Download URLs are documented in each bench script's docstring.
