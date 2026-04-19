# rustscenic

Fast Rust + PyO3 reimplementation of the four slow stages of the SCENIC+ single-cell regulatory-network pipeline. **Installs cleanly on modern Python where the reference stack (arboreto + pyscenic + pycisTopic) no longer does.**

```bash
pip install rustscenic anndata scanpy
```

→ **See [QUICKSTART.md](QUICKSTART.md)** for a 5-minute end-to-end walkthrough on PBMC-3k.

## Why

As of April 2026, on a fresh Python 3.12 + numpy 2 + pandas 3 environment:
- `arboreto.grnboost2` → runtime crash (`TypeError: Must supply at least one delayed object` — dask_expr incompatibility)
- `pyscenic.aucell` → import fails with `ModuleNotFoundError: pkg_resources`
- `pycisTopic-Mallet` → requires a Java install (multi-hour setup, flaky)
- `flashscenic` → CPU fallback works but requires PyTorch + changes the algorithm (RegDiffusion, not pyscenic-reproducible)

rustscenic installs and runs. One pip install, 3 runtime deps (numpy, pandas, pyarrow). All four SCENIC+ stages — `grn`, `aucell`, `topics`, `cistarget` — are native Rust via PyO3.

## Validation

Every number below has a log file under [`validation/ours/`](validation/ours). Measurements are from this codebase, this week. The [VALIDATION_SUMMARY.md](validation/VALIDATION_SUMMARY.md) gathers them in one place.

### Real-world atlas-scale head-to-head vs pyscenic

[`validation/ziegler_headtohead_2026-04-19.md`](validation/ziegler_headtohead_2026-04-19.md) — 31,602-cell nasopharyngeal atlas (Ziegler 2021 *Cell*), identical 59 regulons on both sides:

| | rustscenic | pyscenic-unit | pyscenic-weighted |
|---|---:|---:|---:|
| Per-cell Pearson with rustscenic | 1.000 | **0.984** | 0.949 |
| Canonical airway TF hits (of 14) | 8 | 8 | 9 |
| AUCell wall-time | **0.25 s** | 6.81 s | 5.29 s |

Both tools miss the same 5 TFs (STAT1, MYB, IRF7, SOX2, PAX5) — tool-to-tool variation is smaller than dataset-inherent noise.

- `validation/figures/ziegler_fig1_canonical_tf_3way.png` — the 3-way TF comparison
- `validation/figures/ziegler_fig2_per_cell_pearson.png` — per-cell agreement distribution
- `validation/figures/ziegler_fig3_runtime.png` — 27× speedup bar chart



### GRN — `arboreto.grnboost2` replacement

| Dataset | n_cells | Measurement | Value |
|---|---|---|---|
| Multiome 3k (shared barcodes) | 2,588 | Per-edge Spearman vs arboreto (816k common edges, n_estimators=5000) | **0.58** |
| Multiome 3k | 2,588 | Per-target TF-ranking Spearman mean | 0.57 |
| PBMC-3k | 2,698 | Known TF→target edges recovered | **94% (17/18)** |
| PBMC-10k | 10,290 | Lineage TFs correctly enriched | **8/8** (SPI1, PAX5, EBF1, TCF7, LEF1, TBX21, CEBPD, IRF8) |
| Tirosh 2016 melanoma | 4,645 | MITF regulon activity malignant vs TME | **3.48×** |
| 100k cells bootstrap | 100,000 | Wall / peak RSS (n_estimators=100, 20 TFs) | 17 min / 5.0 GB |

**Interpretation.** rustscenic's edge rankings disagree with arboreto at fine grain (Spearman 0.58, top-100 Jaccard 0.10). Both tools converge on the same biology at coarse resolution (94% known edges, 8/8 lineage TFs). Downstream AUCell is 0.99 per-cell with pyscenic — fine-edge ranking differences do not propagate to regulon activity.

### AUCell — `pyscenic.aucell` replacement

| Measurement | Value |
|---|---|
| **Per-cell Pearson vs pyscenic** (10x Multiome, 2,588 cells × 1,457 regulons) | **0.9881 mean, 99.5% of cells > 0.95** |
| Per-regulon Pearson (same data) | 0.87 mean, 90.5% > 0.80 |
| Exact top-regulon-per-cell match | 88.4% |
| pyscenic top-1 regulon in our top-3 | 99.5% |
| Speed vs pyscenic at 10k cells | **88×** (0.21s vs 18.6s) |
| 100k cells × 500 regulons | **10s, 5.6 GB peak RSS** |

### Topics — `pycisTopic` LDA replacement (Online VB)

Head-to-head vs **Mallet** (pycisTopic's reference backend) on 10x PBMC 10k ATAC, 8,728 cells × 67,448 peaks, K=30:

| Tool | Wall | Unique topics (K=30) | NPMI coherence | ARI vs leiden |
|---|---|---|---|---|
| Mallet | 534 s | **24 / 30** | **0.196** | 0.258 |
| rustscenic seed=42 | 942 s | 5 / 30 | 0.123 | 0.269 |
| rustscenic seed=123 | 622 s | 5 / 30 | — | 0.334 |

**Interpretation.** Mallet discovers ~5× more distinct topics and has 60% higher coherence. Our Online VB LDA collapses aggressively at K=30 on this dataset. Cell-type recovery (ARI vs leiden) is comparable. For fine-grained topic decomposition, prefer Mallet (if you can install it); for a no-Java drop-in that agrees with Mallet at the cell-type level, ours works. Cross-seed ARI is 0.63 mean.

### Cistarget — `pycistarget` replacement (AUC kernel)

Validated on the real aertslab hg38 v10 feather DB (5,876 motifs × 27,015 genes):

| Test | Value |
|---|---|
| **Per-regulon Pearson vs `ctxcore.recovery.aucs`** (58 regulons) | **1.0000** (all > 0.9999, mean abs diff 2.4e-5) |
| Self-consistency (motif's own top-500 genes → rank #1) | 10/10 |
| TRRUST at scale (166 TFs with ≥10 targets): **TF-annotated motif ranks #1** | 19% |
| Same, any TF-motif in top-100 | 68–100% (rises with regulon size) |
| Mouse mm10 (5 TRRUST TFs) | 2/5 rank #1, 4/5 in top-5 |
| 100k-cell workload × 100 regulons | **2.6 s, 6.3 GB peak RSS** |

**Interpretation.** Our cistarget is bit-identical to pyscenic's own `ctxcore.recovery.aucs` at float32 precision. The 19% TF-#1 rate is the honest scaled-out figure for the TRRUST-vs-motif-binding benchmark — this is a property of the benchmark, not our code. Earlier "6/8" figures in prior docs were a hand-picked subset.

### End-to-end

| Pipeline | Wall (10x Multiome 3k) |
|---|---|
| Reference (arboreto + pyscenic + tomotopy), when it installs | 11.8 min |
| **rustscenic (all 4 stages native Rust)** | **9.1 min** |

Memory ceiling at 100k cells: **6.3 GB peak RSS** across all 4 stages (pyscenic reported to exceed 40 GB on similar workloads — ~7× less).

### Determinism + robustness

- All 4 stages produce bit-identical output under same seed (verified across 3 runs).
- 10/10 edge-case tests pass (foreign genes, NaN input, duplicate gene names, all-zero cells, large regulons, etc.). See [`robustness_2026-04-18.md`](validation/ours/robustness_2026-04-18.md).

## What rustscenic does NOT claim

- **Not bit-identical to arboreto's sklearn GBR.** Different quantization + RNG tape → fine-grained edge ranks differ (Spearman 0.58). Biology converges (94% known edges).
- **Not faster than Mallet on topics.** At 10k ATAC cells, Mallet is 1.5–1.8× faster and finds more distinct topics. Our pitch is no-Java install, not speed.
- **Not faster than flashscenic on GPU.** If you have an A100 and accept a RegDiffusion algorithm swap, flashscenic is the speed play.
- **Does NOT bundle the aertslab motif ranking feather databases** (300 MB–35 GB). Users fetch from `resources.aertslab.org` and pass the DataFrame to `cistarget.enrich`.

## Full pipeline surface

```python
import anndata as ad
import rustscenic

adata = ad.read_h5ad("rna.h5ad")
tfs = rustscenic.grn.load_tfs("hs_hgnc_tfs.txt")

# Stage 1: GRN inference (arboreto.grnboost2 replacement)
grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=5000, seed=777)

# Stage 2: Regulon activity (pyscenic.aucell replacement)
from pyscenic.utils import modules_from_adjacencies
regulons = list(modules_from_adjacencies(grn, adata.to_df(), top_n_targets=(50,)))
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)

# Stage 3: Topic modeling (pycisTopic LDA replacement, Online VB)
atac = ad.read_h5ad("atac_binarized.h5ad")
topics_result = rustscenic.topics.fit(atac, n_topics=30)

# Stage 4: Motif enrichment (pycistarget AUC kernel replacement)
rankings = rustscenic.cistarget.load_aertslab_feather("hg38_...feather")
enrichments = rustscenic.cistarget.enrich(rankings, regulons, top_frac=0.05)
```

CLI equivalent:
```bash
rustscenic grn       --expression data.h5ad --tfs tfs.txt --output grn.parquet
rustscenic aucell    --expression data.h5ad --regulons grn.parquet --output auc.parquet
rustscenic topics    --expression atac.h5ad --output topics --n-topics 30
rustscenic cistarget --rankings motifs.feather --regulons grn.parquet --output enrichment.tsv
```

End-to-end example: [`examples/pbmc3k_end_to_end.py`](examples/pbmc3k_end_to_end.py).

## Repo layout

- `crates/` — Rust workspace: `rustscenic-{core,grn,aucell,topics,cistarget,cli,py}`
- `python/rustscenic/` — Python package (lazy imports, CLI entry point)
- `validation/` — reproducible benchmark scripts + validation documents for every claim above
- `docs/specs/` — design spec
- `skills/rustscenic.md` — Claude Code agent skill

## License

MIT. Algorithm implementations are independent reimplementations following the aertslab Python references — original method credit to Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+), Hoffman et al. 2010 (Online VB LDA).

## Contact

Open an issue at [github.com/Ekin-Kahraman/rustscenic](https://github.com/Ekin-Kahraman/rustscenic/issues).
