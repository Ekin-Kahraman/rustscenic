# Topic collapse in rustscenic.topics — known limitation

**TL;DR:** On sparse binary scATAC at K≥30, rustscenic's Online VB LDA collapses to ~5 dominant topics regardless of parameter tuning. Cell-type recovery (argmax-topic ARI vs leiden) remains on par with Mallet, but fine-grained topic decomposition is weaker. If fine topics matter for your analysis, use Mallet through pycisTopic for that stage.

## What "collapse" means

Given K=30 requested topics and Mallet's 500-iteration collapsed-Gibbs reference:

| Tool | Unique argmax-topic labels (10k PBMC ATAC, K=30) | NPMI coherence mean |
| --- | --- | --- |
| Mallet (reference) | **24/30** | 0.196 |
| rustscenic.topics seed=42 | 5/30 | 0.123 |
| rustscenic.topics seed=123 | 5/30 | — |
| rustscenic.topics seed=777 | 6/30 | — |

Cell-type ARI vs leiden stays comparable (0.27 ours vs 0.26 Mallet), but ~24 of the 30 requested topics are "absorbed" into ~5 dominant ones.

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

## When rustscenic.topics is the right choice

- You just need cell-topic activity that correlates with cell types. ARI vs leiden is on par with Mallet.
- You can't install Java (no admin rights, conda Mallet package is fragile).
- You need determinism across seeds / thread counts.
- Speed at K<10 on RNA-shaped input.

## When to use Mallet instead

- You need K=30+ distinct topics, each explaining a distinct peak program.
- You're reproducing a SCENIC+ paper that used Mallet Gibbs.
- You need topic coherence (NPMI) comparable to Mallet.

Install Mallet and call pycisTopic's `run_cgs_models_mallet`. rustscenic's other three stages (grn, aucell, cistarget) work fine in the same pipeline regardless.

## Roadmap

v0.2 candidate: replace the Online VB backend with a collapsed Gibbs sampler (Rust port of Mallet's core). Estimated effort: one-to-two weeks. Not scheduled for v0.1.
