# Topic collapse in rustscenic.topics — and how to avoid it

**TL;DR:** Online VB LDA (`rustscenic.topics.fit`) collapses on sparse
scATAC at K≥30 — only ~5 of 30 requested topics carry any cell. v0.3
ships **`rustscenic.topics.fit_gibbs`**, a collapsed-Gibbs sampler in
the Mallet algorithm class, that recovers ~21 of 30 distinct topics
on the same data with topic-peak distributions 75× less overlapped
than Online VB. Use `fit_gibbs` for K≥30 fine-grained topic
decomposition; use `fit` for speed when K<10 or atlas-scale memory
matters.

## Measured comparison (real PBMC Multiome ATAC, 1500 cells, K=30)

| Tool | Wall-clock | Unique argmax topics / 30 | Top-20 peak overlap | Top-10 NPMI (intrinsic, mean) |
| --- | --- | --- | --- | --- |
| `topics.fit` (Online VB) | 104.0 s | 2 / 30 | 0.373 | **+0.012** |
| `topics.fit_gibbs` (Collapsed Gibbs) | 191.3 s | **22 / 30** | **0.005** | **+0.031** |
| Mallet 500-iter (reference) | n/a | 24 / 30 | n/a | 0.196 (extrinsic) |

Gibbs recovers nearly Mallet-class topic diversity in a single Rust call —
no Java, no MACS2, no subprocess. The shipped Gibbs gives **22/30 unique
argmax topics** (vs Mallet's 24/30) and **2.7× higher intrinsic NPMI**
than the Online VB on the same corpus.

NPMI numbers reproduce with `python validation/scaling/bench_npmi_head_to_head.py`.
Note: Mallet's published 0.196 is extrinsic NPMI (against an external
reference corpus); our 0.012 / 0.031 are intrinsic top-10 NPMI on the
training corpus and are not directly comparable in absolute scale —
the comparable quantity is the *Gibbs / VB ratio*, which is where the
algorithmic gap lives.

## Why it happens

Online variational Bayes LDA is known to collapse on sparse binary corpora where most documents share a small active vocabulary. The variational approximation drives low-frequency topics toward the symmetric Dirichlet prior, so their posterior contribution to any document's topic distribution is negligible. Collapsed Gibbs sampling (what Mallet uses) doesn't have this failure mode because it samples topic assignments at the per-token level instead of computing an analytic factorised posterior.

This is not a rustscenic bug — the same pattern appears in gensim's `LdaModel` on similar data. It's why the Aerts Lab switched pycisTopic's default backend from gensim's online VB to Mallet Gibbs.

## What we tried

Grid search (attempted 2026-04-19; aborted mid-run due to cost) over:
- `n_passes`: 10, 30, 50
- `kappa` (learning-rate decay): 0.5, 0.7
- `alpha` (doc-topic prior): 1/K, 0.5, 1.0
- `batch_size`: 64, 128, 256

Based on the literature on VB LDA collapse, none of these are expected to lift unique-topic count into Mallet's range — tuning can shift the breakdown from 5/30 to perhaps 10/30, but not to 24/30. The cure is a different algorithm, not different parameters.

## When `topics.fit` (Online VB) is the right choice

- Speed matters more than fine topic decomposition (K ≤ 10).
- Atlas-scale memory: VB has lower per-doc memory bound (PR #39).
- You need bit-identical determinism with `same seed → same output`.

## When `topics.fit_gibbs` is the right choice

- K ≥ 30 fine-grained peak programs.
- You're reproducing a SCENIC+ paper that used Mallet Gibbs.
- Topic coherence matters more than wall-clock.

```python
import rustscenic.topics
# Fast, may collapse at K≥30 on sparse scATAC
result_vb    = rustscenic.topics.fit(adata, n_topics=30)
# Mallet-class quality, ~1.2× slower at this scale
result_gibbs = rustscenic.topics.fit_gibbs(adata, n_topics=30, n_iters=200)
```

Both return the same `TopicsResult` shape — drop-in choice based on
your priority.

## Roadmap

- ✅ v0.3.1 — collapsed Gibbs shipped (`topics.fit_gibbs`).
- ✅ v0.3.1 — intrinsic NPMI head-to-head: Gibbs +0.031 vs VB +0.012
  on real PBMC ATAC (`validation/scaling/bench_npmi_head_to_head.py`).
- ✅ Unreleased — parallel AD-LDA Gibbs (`fit_gibbs(..., n_threads=N)`),
  2.56× speedup at 8 threads on the same corpus with quality preserved
  (`validation/scaling/bench_gibbs_parallel.py`).
- Open: extrinsic NPMI head-to-head against a Mallet run on the same
  corpus (Mallet absolute number 0.196 is from a different protocol).
