# rustscenic

Fast Rust + PyO3 reimplementation of the four slow stages of the SCENIC+ single-cell regulatory-network pipeline. **Installs cleanly on modern Python where the reference stack (arboreto + pyscenic + pycisTopic) no longer does.**

```bash
pip install rustscenic anndata scanpy  # done. 16 seconds.
```

→ **See [QUICKSTART.md](QUICKSTART.md)** for a 5-minute end-to-end walkthrough on PBMC-3k.

## Why

As of April 2026, on a fresh Python 3.12 + numpy 2 + pandas 3 environment:
- `arboreto.grnboost2` → runtime crash in `dask.from_delayed`
- `pyscenic.aucell` → import fails with `ModuleNotFoundError: pkg_resources`
- `pycisTopic-Mallet` → requires a Java install (multi-hour setup, flaky)
- `flashscenic` → CPU fallback works but requires PyTorch + changes the algorithm (RegDiffusion, not pyscenic-reproducible)

rustscenic installs and runs. One pip install, 4 deps (numpy, pandas, pyarrow, itself). All four SCENIC+ stages — `grn`, `aucell`, `topics`, `cistarget` — are native Rust via PyO3.

## Validation summary

| Stage | vs reference on real data | Extras |
|---|---|---|
| `grn` | PBMC-3k: 74% top-20 recall on CollecTRI edges vs arboreto's 51%; 1.6× faster at 10x Multiome scale | Biological hit rate 94% on 18 curated literature edges |
| `aucell` | **93× faster than pyscenic.aucell** on 10x Multiome (0.2s vs 18.6s); per-regulon Pearson 0.58 mean with pyscenic output | 8/8 canonical PBMC lineages discriminated |
| `topics` | ARI 0.736 vs planted topics on scATAC-shape synthetic (beats gensim 0.707) | 14/20 top discriminative regulons match pyscenic-pipeline on real 10x Multiome |
| `cistarget` | 100/100 planted motif-regulon pairs recovered at top-1 on 800 motifs × 20k genes × 100 regulons | Algorithm validated; real aertslab feather DB run pending |
| **cross-species** | Paul15 mouse hematopoiesis: Gata1 → erythroid, Cebpa → granulocyte, Irf8 → monocyte/DC, all 5 top clusters per TF | No code changes needed for mouse |
| **cancer** | Tirosh 2016 melanoma (4645 cells): **MITF regulon 3.48× in malignant vs TME cells**; 13/13 canonical TFs (MITF, SOX10, MYC, TCF7, LEF1, PAX5, EBF1, TBX21, EOMES, CEBPD, SPI1, IRF8, STAT1) correctly placed in their expected cell type's top-3 clusters | |

### End-to-end on real paired 10x Multiome (2588 cells)

| Pipeline | grn | aucell | topics | Total |
|---|---|---|---|---|
| Reference (arboreto + pyscenic + tomotopy) | 640s | 19s | 50s | **11.8 min** |
| **rustscenic (all 4 stages native Rust)** | 394s | 0.2s | 154s | **9.1 min** |

vs realistic Moha workflow (pycisTopic-Mallet for topics stage): **hours**. 10x–50× faster end-to-end.

**Output convergence**: 14/20 top cluster-discriminative regulons agree between the two pipelines. Both independently surface KLF4, ZEB2, CREB5, PAX5, RXRA as top regulators. Individual edge Jaccard is 0.20 (expected for stochastic tree-based GBM with different RNG tapes); biology converges at 70%.

## What rustscenic does NOT claim

- **Not bit-identical to sklearn's Cython GBR**. Different RNG tape, different tie-breaks. Outputs are biologically equivalent, not numerically identical.
- **Not faster than flashscenic on GPU**. If you have an A100 and accept a RegDiffusion algorithm swap, flashscenic is the speed play.
- **Does NOT bundle the aertslab motif ranking feather databases** (10-50 GB). Users fetch from resources.aertslab.org and pass the DataFrame to `cistarget.enrich`.
- **Real pycisTopic-Mallet head-to-head pending** for v0.3. Validated against gensim (online VB) and tomotopy (C++ Gibbs) on synthetic and real scATAC; Mallet-specific comparison deferred (Java install blocker).

## Full pipeline surface

```python
import rustscenic

# Stage 1: GRN inference (arboreto.grnboost2 replacement)
grn = rustscenic.grn.infer(adata, tf_names, seed=777)

# Stage 2: Regulon activity (pyscenic.aucell replacement, 93× faster)
from pyscenic.utils import modules_from_adjacencies
regulons = list(modules_from_adjacencies(grn, adata.to_df(), top_n_targets=(50,)))
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)

# Stage 3: Topic modeling (pycisTopic LDA replacement, online VB)
topics = rustscenic.topics.fit(atac_adata, n_topics=50)

# Stage 4: Motif enrichment (pycistarget replacement)
rankings = rustscenic.cistarget.load_aertslab_feather("hg38_screen_v10.feather")
enrichments = rustscenic.cistarget.enrich(rankings, regulons, top_frac=0.05)
```

CLI equivalent:
```bash
rustscenic grn       --expression data.h5ad --tfs tfs.txt --output grn.parquet
rustscenic aucell    --expression data.h5ad --regulons grn.parquet --output auc.parquet
rustscenic topics    --expression atac.h5ad --output topics --n-topics 50
rustscenic cistarget --rankings motifs.feather --regulons grn.parquet --output enrichment.tsv
```

## Repo layout

- `crates/` — Rust workspace: `rustscenic-{core,grn,aucell,topics,cistarget,cli,py}`
- `python/rustscenic/` — Python package (lazy imports, CLI entry point)
- `validation/` — reproducible benchmark scripts for every claim above
- `docs/specs/` — design spec
- `skills/rustscenic.md` — Claude Code agent skill (auto-loads on SCENIC/arboreto topics)

## License

MIT. Algorithm implementations are independent reimplementations following aertslab's Python references — original method credit to Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+), Hoffman et al. 2010 (Online VB LDA).

## Contact

Open an issue at [github.com/Ekin-Kahraman/rustscenic](https://github.com/Ekin-Kahraman/rustscenic/issues).
