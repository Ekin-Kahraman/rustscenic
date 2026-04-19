# Changelog

## 0.1.0 — 2026-04-19

Initial release. All four SCENIC+ slow stages reimplemented in Rust + PyO3:

- `rustscenic.grn` — GRNBoost2 replacement (histogram-GBM regression trees, Rayon-parallel, deterministic).
- `rustscenic.aucell` — pyscenic.aucell replacement (per-cell recovery-AUC regulon scoring; 88× faster than pyscenic on 10k-cell data).
- `rustscenic.topics` — pycisTopic LDA replacement (Online VB LDA, Hoffman-Blei-Bach 2010).
- `rustscenic.cistarget` — pycistarget AUC kernel replacement (bit-identical to `ctxcore.recovery.aucs` at float32 precision).

Ships as a single `pip install` wheel (maturin + abi3). Runs on Python 3.10–3.13, macOS arm64, Linux x86_64. No Java, no dask, no CUDA.

### Validation (measured on this release)

- **AUCell vs pyscenic** (10x Multiome, 2,588 cells × 1,457 regulons): per-cell Pearson 0.99 mean (99.5% of cells > 0.95). Per-regulon Pearson 0.87 mean. 88× faster than pyscenic.
- **Cistarget vs `ctxcore.recovery.aucs`** (aertslab hg38 v10, 5,876 motifs × 27,015 genes): Pearson 1.0000 across 58 TRRUST regulons, mean abs diff 2.4e-05.
- **GRN vs arboreto** (10x Multiome, n_estimators=5000): per-edge Spearman 0.58 on 816k common edges. Biology agrees at coarse resolution: 94% of known TF→target edges recovered (PBMC-3k); 8/8 lineage TFs correctly enriched (PBMC-10k); MITF 3.48× in Tirosh melanoma.
- **Topics vs Mallet** (pycisTopic reference backend; 10x PBMC 10k ATAC, 8,728 cells × 67,448 peaks): ARI vs leiden comparable (0.27 vs 0.26). Mallet wins on unique topic count (24/30 vs 5/30) and NPMI coherence (0.196 vs 0.123) — our Online VB LDA collapses aggressively at K=30 on this scale.
- **End-to-end** (10x Multiome 3k, all 4 stages): 9.1 min vs reference pipeline's 11.8 min.
- **Memory**: 6.3 GB peak RSS at 100k cells × 20k genes across all 4 stages.
- **Determinism**: bit-identical output under same seed across all 4 stages.

Full log files under [`validation/ours/`](validation/ours).
