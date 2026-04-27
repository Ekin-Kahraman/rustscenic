# Changelog

## 0.3.1 — 2026-04-27

### Added
- **Collapsed-Gibbs LDA** (`rustscenic.topics.fit_gibbs`) — Mallet-class
  topic model. Closes the only place rustscenic still lost to
  references on quality. Shipped after the v0.3.0 tag was cut, so this
  patch release brings the wheel artifacts in line with main.

  Real PBMC ATAC, 1,500 cells × 98k peaks, K=30:
  - Online VB: 2/30 unique argmax topics (collapsed), NPMI +0.012
  - Collapsed Gibbs: **22/30 unique argmax topics**, NPMI **+0.031**
  - Top-20 peak overlap: VB 0.373 → Gibbs 0.005 (75× more diverse)
  - Gibbs intrinsic NPMI is **2.7× higher** than VB on the same corpus

  3 Rust unit tests + 5 Python tests cover synthetic recovery,
  determinism, AnnData input, edge cases.

- **`rustscenic.topics.coherence_npmi`** — per-topic intrinsic NPMI
  metric for fitted topic models. Backs the published quality
  comparison; runs entirely in Rust. Reproduce with
  `python validation/scaling/bench_npmi_head_to_head.py`.

### Validation
- 200k synthetic GRN scaling: 9 min, slope 1.30, 8.6 GB RSS.
- Real multiome end-to-end first run on 10x PBMC 3k Multiome:
  6.2 min total wall-clock, all 6 stages connect.

### Docs
- `docs/topic-collapse.md` updated to point at the shipped
  `topics.fit_gibbs` API instead of recommending Mallet.
- `docs/bench-vs-references.md` carries the K=30 quality numbers.
- `docs/what-rustscenic-is.md` no longer lists Gibbs as a future
  candidate.

### Test counts
129 Python tests + 54 Rust tests pass.

## 0.3.0 — 2026-04-26

### Performance
- **GRN atlas-scale fix**: worker-local `GbmScratch` + 64-target column-major
  blocking. The 8× cliff at 40k→80k cells is gone. Real 91,838-cell microglia
  GRN: 110 min → 14.4 min. Full 5k→91.8k log-log slope: 1.81 → 1.15.
- **GRN binned-matrix column-major**: 10.6% wall-clock saving on real PBMC Multiome.
- **PyO3 input borrow**: ~12 GB instantaneous RSS saved at atlas scale.
- **Topics par_iter().fold().reduce()**: ~30× lower memory bound for online VB LDA.
- **Chrom × fragment loop inversions** in peak calling, TSS, matrix builder.

### Capabilities
- **Region-based cistarget**: exact eRegulon assembly when region rankings supplied.
- **Regulon specificity scores**: `rustscenic.specificity.regulon_specificity_scores`.
- **Topic candidate enhancers**: `rustscenic.specificity.candidate_enhancers_per_topic`.
- **Mouse mm10 motif rankings download path**.

### Robustness
- **Aertslab URL fix** — broken since v0.1.0 (mocked tests never caught it).
  Live HTTP smoke now runs.
- Duplicate-symbol auto-sum (scanpy/limma avereps).
- Backed AnnData support in AUCell + GRN.
- Dict regulons accepted (docstring promised, finally works).
- Versioned ENSEMBL `.N` suffix auto-strip when no symbol column.
- `top_frac` bounds + saturation warning + tiny-cutoff warning.
- 6-column strand BED detection.
- eRegulon catastrophic-drop warning + > 8 GiB densification warning.
- scenicplus polarity suffix stripping (`TF(+)`, `_extended`, `_activator`).
- Actionable zero-overlap diagnostics that name the specific mismatch.
- Repeat `pipeline.run` calls no longer crash on overlapping regulon columns.

### Validation
- Kamath 2022 OPC end-to-end on real cellxgene data.
- Multi-dataset bench: 1.2k mouse → 30k human, all coverage 100%.
- 10x PBMC 3k Multiome full pipeline run.
- 100k × 30k synthetic atlas E2E at 9.5 GB peak RSS.
- 91k microglia atlas GRN scaling — slope 1.15.
- Bench vs MACS2: 9.9× faster, F1 0.825.
- Bench vs gensim LDA: gensim still 1.5–2.7× faster at K=10/30 (documented).

### Cleanup
- Removed dead `rustscenic-cli` and `rustscenic-core` crates.
- pipeline.run goes end-to-end (preproc → topics → GRN → cistarget → enhancer → eRegulon → AUCell).
- Quickstart hardened against transient scanpy network failures.

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

## Unreleased

### Performance
- PyO3 input borrow: `grn_infer` and `aucell_score` now borrow the
  numpy buffer instead of copying it. Saves ~12 GB instantaneous RSS
  on a 100k × 30k atlas-scale input.

### Validated
- **100k cells × 30k genes synthetic atlas-scale end-to-end**:
  GRN (50 TFs, n_estimators=20) 39.9 min, AUCell (20 regulons) 2.0 min,
  peak RSS 9.5 GB. Reference scenicplus stack reports > 40 GB at
  comparable scale. No OOM, no crash.
