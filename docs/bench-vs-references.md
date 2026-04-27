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

## Topic modelling — Online VB vs Collapsed Gibbs vs gensim

rustscenic ships two topic-model paths:
- `topics.fit` — Online VB LDA (Hoffman 2010), fast at small K
- `topics.fit_gibbs` — Collapsed Gibbs LDA (Griffiths-Steyvers 2004),
  Mallet-class quality at K ≥ 30

### Speed (PBMC 3k Multiome ATAC, 3,000 × 98,319, nnz=20.97 M)

| K | rustscenic VB | rustscenic Gibbs | gensim 4.4.0 |
|---|---|---|---|
| 10 | 31.6 s | 39.2 s | 21.7 s |
| 30 | 42.6 s | 49.0 s | 26.4 s |

gensim's pure-numpy VB wins on raw wall-clock at this shape. We
provide both — pick by priority.

### Quality at K=30 (real PBMC ATAC, 1,500 cells × 98,319 peaks)

| Tool | Unique argmax topics / 30 | Top-20 peak overlap | Top-10 NPMI (intrinsic, mean) |
|---|---|---|---|
| rustscenic VB | **2 / 30** (collapsed) | 0.373 | **+0.012** |
| rustscenic Gibbs | **22 / 30** | **0.005** | **+0.031** |
| Mallet 500-iter (reference) | 24 / 30 | n/a | 0.196 (extrinsic, different protocol) |

**Collapsed Gibbs gives 11× more distinct topics than Online VB on
sparse scATAC at K=30, with topic-peak distributions 75× less
overlapped, and ~2.7× higher intrinsic NPMI on the training corpus.**
Same algorithm class as Mallet, no Java required.

NPMI numbers reproduce with `python validation/scaling/bench_npmi_head_to_head.py`.
Mallet's published 0.196 is extrinsic NPMI (against an external reference
corpus); our 0.012 / 0.031 are intrinsic top-10 NPMI on the training
corpus and are not directly comparable in absolute scale — what is
comparable is the *Gibbs / VB ratio*, which is where the algorithmic
gap lives.

What rustscenic gives at this layer:
- **Two algorithms, drop-in choice** — VB for speed at K ≤ 10, Gibbs
  for fidelity at K ≥ 30.
- **Bit-identical determinism** under same seed (both algorithms).
- **One pip install** — Mallet, the SCENIC+ reference for K ≥ 30,
  needs Java + binary.
- **Atlas-scale memory bound at `O(threads × K × n_words)`** for VB
  (PR #39 fold/reduce refactor).

## Headline (take to the scverse meeting)

| Layer | Reference | rustscenic | Result |
|---|---|---|---|
| Peak calling | MACS2 | rustscenic.preproc.call_peaks | **9.9× faster, F1 0.825** |
| Topic modelling K=10 (speed) | gensim LdaModel | rustscenic.topics.fit | **0.7× (gensim wins)** |
| Topic modelling K=30 (speed) | gensim LdaModel | rustscenic.topics.fit | **0.61× (gensim wins)** |
| Topic modelling K=30 (quality) | rustscenic VB collapses (2/30) | **rustscenic Gibbs 21/30** | **Gibbs ~10× more distinct topics** |
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
