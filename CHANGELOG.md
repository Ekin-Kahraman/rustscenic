# Changelog

## 0.3.5 — 2026-05-01

### Release Integrity
- Repair the v0.3.4 release cut: the public `v0.3.4` tag was created
  before the version-bump commit, so GitHub Release assets were
  correctly built from that tag but incorrectly named `0.3.3` under the
  `v0.3.4` release page. v0.3.5 is the non-force-push fix: bump
  `pyproject.toml`, the Rust workspace version, `Cargo.lock`, CI smoke
  assertions, and install docs together so the tag, wheel names,
  package metadata, and README URLs all agree.

### Validation
- Re-affirmed real PBMC 3k Multiome E2E with real GENCODE v46 hg38 TSS
  coordinates: 591,022 GRN edges, 35,410 cistarget hits, 21,284
  enhancer links, 19 eRegulons, 65.1s total. Reproduce with
  `python validation/scaling/bench_real_pbmc_full_e2e.py`.

### Tests
- 144 Python + 57 Rust tests pass.

## 0.3.4 — 2026-04-30

### Fixes
- **Gene-only eRegulon bridge** (`python/rustscenic/eregulon.py`).
  The bridge was using the full GRN as the target set, producing a
  ~3.5B-row merge on real PBMC multiome that stalled the pipeline
  indefinitely. Vectorised the join and restricted the target set
  to gene-only candidates. Real PBMC multiome E2E now completes;
  previously hung at the eRegulon stage.

### Features
- **`rustscenic.data.download_gene_coords`** — closes the
  synthetic-TSS gap. GENCODE GTF (hg38 v46 / mm10 vM35),
  strand-aware TSS extraction, parquet-cached on first call.
  Real PBMC multiome E2E now validated against real gene
  coordinates instead of synthetic TSSs.

### Validation
- **Real PBMC multiome 7-stage E2E** with real gene coordinates.
  All compute stages green through eRegulon assembly. Reproduce
  with `python validation/scaling/bench_real_pbmc_full_e2e.py`.

### Robustness
- Hardened PBMC validation paths and ranking inputs against edge
  cases surfaced during real-data testing.
- Tightened scaling docs and enhancer warnings.

### Tests
- 144 Python + 57 Rust tests pass.

### Dependencies
- `rand_mt` 4.2.2 → 6.0.3 (#55)
- `thiserror` 1.0.69 → 2.0.18 (#53)

## 0.3.3 — 2026-04-27

### Performance
- **Sparse enhancer-to-gene Pearson** (`python/rustscenic/enhancer.py`).
  The previous correlation loop densified both RNA and ATAC at the
  start of the function — fine at 100k cells × 50k peaks (~20 GB
  ATAC dense, OS swap absorbed it), broken at 200k × 30k (24 GB
  dense ATAC stalled the single-threaded peak loop indefinitely).

  Fix: keep ATAC as `scipy.sparse.csc` and stream one peak column at
  a time through a new `_pearson_sparse_x_dense_Y` helper. Pearson is
  rewritten to consume only the column's nonzeros — work per peak
  goes from `O(n_cells × n_candidate_genes)` to
  `O(nnz_peak × n_candidate_genes)`. Numerically equivalent to the
  dense path within float32 noise (1e-5 tolerance, 2 new tests).

  Result on the 200k synthetic E2E (`bench_e2e_200k_synthetic.py`):
  enhancer step **74-min stall → 503s (8.4 min)** with peak RSS
  unchanged at 7.06 GB.

### Validation
- **200k synthetic full 7-stage E2E** (`e2e_200k_synthetic.json`):
  200,000 cells × 8,000 genes / 30,000 peaks × K=30. **1009s
  (16.8 min) total wall-clock at 7.44 GB peak RSS, 30/30 unique
  topics, 229,687 GRN edges, 93,750 enhancer links, 30 eRegulons,
  AUCell shape (200000, 30).** Doubles the v0.3.2 E2E proof from
  100k cells; 1.05× memory growth from 100k → 200k.

- **500k synthetic GRN** (`grn_500k.json`): 500,000 cells × 5,000
  genes × 50 TFs (n_estimators=20). **521s (8.7 min) wall-clock,
  224,966 edges, 7.25 GB peak RSS.** Extends the GRN cell-count
  scaling proof from 200k (which used 15k genes) to 500k. Apples-to-
  apples comparison with the 200k × 15k bench is constrained by
  laptop RAM — 500k × 15k dense ≈ 30 GB; 500k × 5k = 10 GB fits.
  Per-cell GRN time (3.7 ms / 50 TFs) sits in line with the 200k ×
  15k bench's 2.7 ms / 50 TFs scaled by gene density.

- **100k synthetic atlas full-pipeline E2E** (`bench_e2e_100k_synthetic.py`):
  100,000 cells × 15,000 genes / 50,000 peaks × K=30, all 7 stages of
  the rustscenic pipeline connected end-to-end (skipping fragments→matrix
  preproc, validated separately on real PBMC).

  | Stage | Wall | Output |
  |---|---|---|
  | Topics (Gibbs 8-thread, 50 iters) | 487s | 30/30 unique topics |
  | GRN (50 TFs, n_estimators=20) | 143s | 441,734 edges |
  | Regulons | 0.3s | 30 regulons |
  | Cistarget | <0.1s | 900 enrichments |
  | Enhancer→gene linking | 116s | 93,750 peak-gene links |
  | eRegulon assembly | 0.8s | 30 eRegulons |
  | AUCell | 15s | (100,000 × 30) |
  | **TOTAL** | **762s (12.7 min)** | All 7 stages |
  | **Peak RSS** | **7.09 GB** | Across all stages |

  Closes the named credibility gap from `docs/what-rustscenic-is.md`:
  "100k-cell atlas end-to-end is unmeasured for the full ATAC + RNA
  pipeline." Reference scenicplus stack reports > 40 GB at comparable
  scale; rustscenic delivers full-pipeline atlas analysis at **5.6×
  less memory and ~13 min wall-clock**.

- **100k synthetic atlas Gibbs alone** (`gibbs_100k.json`): 100,000
  cells × 50,000 peaks × K=30, n_iters=50, 8-thread AD-LDA. 544s
  (9.1 min), 27/30 unique topics, 8.38 GB peak RSS. The Gibbs-only
  curve continues 25k → 50k → 100k cleanly.

- **50k synthetic atlas Gibbs** (`bench_gibbs_50k.py`): 50,000 cells ×
  50,000 peaks × K=30, n_iters=100, 8-thread AD-LDA. **422s (7.0 min),
  30/30 unique argmax topics recovered, 5.06 GB peak RSS.** Extends the
  v0.3.2 atlas validation curve from 25k cells to 50k with quality
  preserved.

- **Real PBMC 3k Multiome E2E with real gene coordinates**
  (`bench_real_pbmc_full_e2e.py`): real RNA, real ATAC fragments, real
  aertslab hg38 gene motif rankings, and GENCODE v46 hg38 TSS
  coordinates run through preproc → Gibbs topics → GRN → cistarget →
  enhancer → eRegulon → AUCell in **65s**. Outputs: 591,022 GRN edges,
  35,410 cistarget hits, 21,284 enhancer links, and 19 eRegulons. The
  script accepts `RUSTSCENIC_GENE_COORDS` for pinned coordinate tables
  and only falls back to synthetic random TSS coordinates with explicit
  `RUSTSCENIC_ALLOW_SYNTHETIC_GENE_COORDS=1` smoke-test opt-in.

### Fixes
- `bench_gibbs_50k.py` now reports `peak_rss_gb` correctly on macOS;
  `ru_maxrss` is bytes on darwin and KB on linux — normalise per-platform.
- `pipeline.run` now accepts aertslab-style `.feather` motif ranking
  paths directly (`motifs` column as row index) instead of requiring
  callers to hand-load the DataFrame first.

## 0.3.2 — 2026-04-27

### Added
- **`rustscenic.topics.coherence_npmi(result, corpus, top_n=10)`** — per-topic
  intrinsic NPMI metric for fitted topic models, runs entirely in Rust.
  Backs the published quality comparison; reproducible with
  `python validation/scaling/bench_npmi_head_to_head.py`.
- **`pipeline.run(..., topics_method='gibbs', topics_n_iters, topics_n_threads)`**:
  the end-to-end orchestrator can now fit topics with the Mallet-class
  collapsed-Gibbs sampler instead of online VB. `topics_method='vb'`
  (default) preserves existing behaviour.

### Performance
- **Parallel collapsed-Gibbs LDA** (`rustscenic.topics.fit_gibbs(..., n_threads=N)`)
  — AD-LDA (Newman et al. 2009): documents partitioned across Rayon
  workers with thread-local `n_kw` deltas merged at sweep boundaries.
  Persistent per-thread buffers + parallel merge over topic rows.

  Real PBMC ATAC, 1500 cells × 98k peaks, K=30, 200 sweeps:
  - Serial:   214s, 22/30 unique topics, NPMI +0.031
  - 4-thread: 120s, 21/30 unique, NPMI +0.031 (**1.79×**)
  - 8-thread:  84s, 25/30 unique, NPMI +0.019 (**2.56×**)

  Synthetic atlas-scale (50k peaks, 8000 nnz/cell, K=30, 100 sweeps):
  - 3k cells:  43s → 24s (1.81× at 8 threads, 29/30 unique)
  - 10k cells: 131s → 81s (1.61× at 8 threads, 28/30 unique)
  - 25k cells: 351s → 217s (1.62× at 8 threads, 29/30 unique)

  Quality preserved across thread counts and corpus sizes.
  Bit-deterministic at fixed `n_threads`, drops to the original
  `fit` path at `n_threads=1`. Speedup plateau around 1.6× at atlas
  scale is consistent with memory-bandwidth bound on the n_kw read
  path; sparse-LDA hash-table approach is the next optimisation.

### Validation
- **NPMI head-to-head, real PBMC ATAC, K=30**: VB +0.012 vs Gibbs
  +0.031 (~2.7× higher coherence) on the same corpus quoted in
  `docs/topic-collapse.md`. Mallet's published 0.196 is extrinsic
  (different protocol) and now flagged as not directly comparable in
  absolute scale.
- **Real PBMC Multiome E2E with `topics_method='gibbs'`**: 74s total
  wall-clock through preproc → topics (Gibbs, 4-thread) → GRN →
  enhancer → AUCell.

### CI
- **Nightly maturin venv fix**: `nightly-real-data.yml` switched from
  `maturin develop --release` (requires venv) to `maturin build` +
  `pip install` of the resulting wheel. Live cellxgene Kamath OPC
  validation green again post-fix.

### Docs
- `docs/topic-collapse.md`, `docs/bench-vs-references.md`,
  `docs/what-rustscenic-is.md`, README.md updated with measured NPMI
  and parallel-Gibbs numbers; the K≥30 "outstanding for v0.4" item is
  now scoped down to extrinsic-Mallet-NPMI head-to-head (the AD-LDA
  parallel path shipped here).

### Test counts
144 Python tests + 57 Rust tests pass.

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
