# Changelog

## 0.2.0 — 2026-04-24

### Added
- **ATAC preprocessing**: `rustscenic.preproc.fragments_to_matrix`,
  `call_peaks` (Corces-2018 iterative consensus), `qc.insert_size_stats`,
  `qc.frip`, `qc.tss_enrichment` (MACS2-free, Java-free).
- **Enhancer-to-gene linking** (`rustscenic.enhancer.link_peaks_to_genes`)
  — Pearson / Spearman peak↔gene correlation, chrom-convention normalised.
- **eRegulon assembly** (`rustscenic.eregulon.build_eregulons`) — three-way
  intersection of GRN × cistarget × enhancer links.
- **End-to-end orchestrator** (`rustscenic.pipeline.run`) + bundled TF
  lists (`rustscenic.data.tfs`) + motif-rankings downloader.
- **Quickstart** — `python -m rustscenic.quickstart` runs PBMC-3k end-to-end
  with a synthetic fallback when the network is down.

### Robustness (silent-regression guards closed)
- Auto-swap ENSEMBL var_names → `var["feature_name"]` (cellxgene convention)
  — was silently scoring AUCell to zero on Kamath-class data.
- Auto-dedupe duplicate symbols (sum columns, scanpy/limma `avereps`
  convention) instead of raising a cryptic `ValueError`.
- Auto-strip versioned ENSEMBL IDs (`ENSG...7` → `ENSG...`) when no
  symbol column is present.
- UCSC vs Ensembl chrom normalisation across peak calling, FRiP,
  TSS enrichment, and enhancer→gene linking.
- Species-case mismatch diagnostic (HGNC `SPI1` vs MGI `Spi1`) with
  one-line fix hint.
- `diagnose_zero_tf_overlap` emits the actual convention mismatch
  instead of "check your conventions".
- `top_frac` validation: `(0, 1]` bounded, warn above `0.3`.
- Backed AnnData (`read_h5ad(path, backed='r')`) now materialises
  cleanly in both AUCell and GRN.
- Dict regulons (`{"R1": [...]}`) supported alongside list of tuples.
- Scenicplus `TF(+)` / `TF(-)` / `TF_activator` / `TF_extended` polarity
  suffixes stripped in eRegulon assembly.
- 6-column strand BED mis-parse detection (warns when the barcode
  column is near one-per-row).
- `build_eregulons` warns when > 50% of TFs drop from the intersection.
- `link_peaks_to_genes` warns before densifying a > 8 GiB matrix.
- `data.tfs()` accepts `"hs"` / `"human"` / `"hg38"` aliases (same for mouse).

### Validation
- **Real Kamath 2022 OPC cells** (13,691 × 33,295, cellxgene schema)
  round-trips ENSEMBL auto-swap, duplicate auto-sum, AUCell non-zero on
  every regulon, GRN recovers all requested HGNC-symbol TFs. Script:
  `validation/kamath/validate_kamath_fix.py`.

### Performance
- **GRN partition-buffer pool** eliminates per-split `Vec<usize>` churn
  that was causing super-linear scaling on 30k+ cell runs. Measured
  2.16× faster on Ziegler 30k cells, slope restored to linear (CI
  regression test enforces `O(N_cells)` slope ≤ 1.30).

### Cleanup
- Removed dead `rustscenic-cli` stub crate and unused `rustscenic-core`
  scaffold crate (four placeholder dependencies, zero imports).
- Completed PyO3 type stubs for the preproc bindings.

### Workflow
- Nightly `nightly-real-data.yml` CI runs the Kamath end-to-end
  validation weekly — catches cellxgene schema drift / URL rot.

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
- **Topics vs Mallet** (pycisTopic reference backend; 10x PBMC 10k ATAC, 8,728 cells × 67,448 peaks): ARI vs leiden comparable (0.27 vs 0.26). Mallet wins on unique topic count (24/30 vs 5/30) and NPMI coherence (0.196 vs 0.123) — our Online VB LDA collapses aggressively at K=30 on this scale. This is a known VB-LDA limitation on sparse binary scATAC (same pattern in gensim). See `docs/topic-collapse.md` for guidance on when to fall back to Mallet; v0.2 candidate is a collapsed Gibbs rewrite.
- **End-to-end** (10x Multiome 3k, all 4 stages): 9.1 min vs reference pipeline's 11.8 min.
- **Memory**: 6.3 GB peak RSS at 100k cells × 20k genes across all 4 stages.
- **Determinism**: bit-identical output under same seed across all 4 stages.

Full log files under [`validation/ours/`](validation/ours).
