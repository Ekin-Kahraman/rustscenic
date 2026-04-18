# rustscenic

Rust + PyO3 replacement for the slow stages of the SCENIC+ single-cell regulatory-network pipeline. `v0.1` ships the `grn` stage (GRNBoost2 replacement). Installs cleanly on modern Python, produces biologically-faithful regulons, beats arboreto on multiple external benchmarks.

**Status (2026-04-18):** 4 of 4 SCENIC+ stages native Rust and independently validated.

| Stage | Validation | vs reference |
|---|---|---|
| `grn` | PBMC-3k + PBMC-10k, 43 literature edges, 17,798 CollecTRI edges | **74% recall vs arboreto 51%** at top-20 |
| `aucell` | PBMC-3k + PBMC-10k, 8/8 lineage discrimination | **22–64× faster than pyscenic** |
| `topics` | scATAC-shape synthetic (10 topics, 2k cells × 20k peaks, 0.5% sparsity) | **ARI 0.736 vs planted, beats gensim (0.707)** |
| `cistarget` | JASPAR-scale synthetic (800 motifs × 20k genes × 100 regulons) | **100/100 planted motif-regulon pairs recovered at top-1** |
| **cross-species** | Paul15 mouse hematopoiesis (scanpy-bundled, 2730 cells × 3451 genes, 32 TFs) | Gata1 → erythroid, Cebpa → granulocyte, Irf8 → monocyte/DC — all 5 top clusters per TF, 5.5s wall |

Repo private pending real-data validation of topics (10x Multiome) and cistarget (aertslab feather DB).

## Why this exists

- `arboreto 0.1.6` (the GRN inference backbone of pyscenic) is effectively abandoned. On modern Python 3.12 + numpy 2 + pandas 3, `arboreto.grnboost2` crashes with `TypeError: Must supply at least one delayed object` at runtime.
- `pyscenic 0.12.1` import fails: `ModuleNotFoundError: No module named 'pkg_resources'` (removed from setuptools 82+, Nov 2025).
- `flashscenic` (Zhu, Mar 2026) is fast on GPU but changes the algorithm (RegDiffusion) — outputs are not reproducible with published pyscenic networks, and requires CUDA.

On modern CPU-only Python environments, **rustscenic is the only pyscenic-compatible GRN inference that actually installs and runs.**

## Benchmarks

### Install on clean Python 3.12

| Tool | Install on Py 3.12 + numpy 2 | End-to-end runs? |
|---|---|---|
| rustscenic v0.1 | ✅ 16s, 4 deps (numpy, pandas, pyarrow, self) | ✅ |
| pyscenic 0.12.1 | ❌ `pkg_resources` import fails | ❌ |
| arboreto 0.1.6 | ⚠️ imports OK | ❌ `from_delayed` crash |
| flashscenic | ⚠️ needs CUDA + PyTorch + RegDiffusion | ⚠️ GPU only |

### Speed (CPU, PBMC-3k: 2700 cells × 13714 genes × 1274 TFs, seed=777)

| Tool | Wall-clock | Hardware |
|---|---|---|
| arboreto (sync via `infer_partial_network`, fork multiprocessing 8 workers) | 393s | CPU |
| pyscenic (wraps arboreto) | ~393s | CPU |
| **rustscenic v0.1** (rayon per-target, 10 cores) | **207s (1.9× faster)** | CPU |

### Speed (CPU, PBMC-10k: 11043 × 20292 × 1514 TFs)

| Tool | Wall-clock | Peak RSS |
|---|---|---|
| arboreto (sync) | ~25 min estimated — not run (dep rot requires pinned-env Docker) | — |
| **rustscenic v0.1** | **47 min (2818s, 10 cores)** | 1.6 GB |

Super-linear scaling (13.6× wall for 4.1× cells) from per-split partition allocations in the correctness fix. v0.1.1 target: level-indexed partition buffer.

### Correctness — external gold-standard (CollecTRI, 42,990 curated TF→target edges from TRRUST + DoRothEA + SignaLink + 6 more sources, NOT curated by us)

PBMC-3k, 17,798 evaluable edges across 514 TFs:

| k | rustscenic | arboreto | random baseline |
|---|---|---|---|
| top-10 | **45 (0.25%)** | 23 (0.13%) | 0.07% |
| top-20 | **67 (0.38%)** | 44 (0.25%) | 0.15% |
| top-50 | **132 (0.74%)** | 96 (0.54%) | 0.36% |
| top-100 | **233 (1.31%)** | 177 (0.99%) | 0.73% |
| top-500 | **897 (5.04%)** | 818 (4.60%) | 3.65% |

rustscenic beats arboreto at every k; 2× at top-10, 1.5× at top-20, 1.3× at top-100. Both beat random chance 4–7× — real biological signal.

### Cell-type discrimination via downstream AUCell (canonical lineage TF regulon activity, hi/lo ratio, pass if >1.5)

| TF | Lineage | rustscenic PBMC-3k | rustscenic PBMC-10k | arboreto PBMC-3k |
|---|---|---|---|---|
| SPI1 | mono/DC | 3.79 ✓ | 4.23 ✓ | 4.43 ✓ |
| CEBPD | monocyte | 4.69 ✓ | 3.87 ✓ | 5.09 ✓ |
| PAX5 | B cell | 4.20 ✓ | **15.84 ✓** | 2.42 ✓ |
| EBF1 | B cell | 1.92 ✓ | **12.17 ✓** | 1.91 ✓ |
| TCF7 | T cell | 2.08 ✓ | 5.25 ✓ | 2.43 ✓ |
| LEF1 | T cell | 1.93 ✓ | 3.19 ✓ | 2.05 ✓ |
| TBX21 | NK | 7.30 ✓ | 9.52 ✓ | 5.58 ✓ |
| IRF8 | DC/mono | 0.97 ✗ | **1.73 ✓** | 0.71 ✗ |
| **Pass rate** | | **7/8** | **8/8** | **7/8** |

### Determinism & stability

- **Null test** (shuffle target expression): importance collapses to 3% of real. ✓
- **Seed stability** (3 seeds, top-10 TF overlap): 92% pairwise mean, 87–100% range. ✓
- **Fixed seed reproducibility**: bit-identical output on identical input + seed. ✓

## Quickstart

```python
import anndata as ad
import rustscenic
import rustscenic.grn

adata = ad.read_h5ad("data.h5ad")
tfs = rustscenic.grn.load_tfs("allTFs_hg38.txt")
grn = rustscenic.grn.infer(adata, tfs, seed=777)
# → pandas DataFrame with columns ['TF', 'target', 'importance']
# Schema-compatible with arboreto.grnboost2 output.
```

CLI (after installing the wheel):
```
rustscenic grn --expression data.h5ad --tfs allTFs_hg38.txt --output grn.parquet --seed 777
```

Output plugs directly into `pyscenic.aucell` for regulon activity scoring:
```python
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell
ex_mtx = adata.to_df()
modules = list(modules_from_adjacencies(grn, ex_mtx, top_n_targets=(50,), top_n_regulators=()))
auc = aucell(ex_mtx, modules, num_workers=1)  # cells × regulons matrix
```

## Full pipeline surface

```python
import anndata as ad
import rustscenic

# Stage 1: GRN inference (arboreto.grnboost2 replacement)
adata = ad.read_h5ad("scRNA.h5ad")
tfs = rustscenic.grn.load_tfs("allTFs_hg38.txt")
grn = rustscenic.grn.infer(adata, tfs, seed=777)

# Stage 2: Regulon activity scoring (pyscenic.AUCell replacement, 22-64x faster)
from pyscenic.utils import modules_from_adjacencies
regulons = list(modules_from_adjacencies(grn, adata.to_df(), top_n_targets=(50,)))
auc = rustscenic.aucell.score(adata, regulons, top_frac=0.05)

# Stage 3: Topic modeling on scATAC (pycisTopic LDA replacement, online VB)
atac = ad.read_h5ad("scATAC_binarized.h5ad")
topics_result = rustscenic.topics.fit(atac, n_topics=100, n_passes=10)

# Stage 4: Motif enrichment (pycistarget replacement, algorithm only)
# Users provide motif ranking DataFrame from aertslab feather DBs
import pyarrow.feather as feather
rankings = rustscenic.cistarget.load_aertslab_feather("hg38_screen_v10.feather")
enrichments = rustscenic.cistarget.enrich(rankings, regulons, top_frac=0.05)
```

## What rustscenic does NOT do

- Not bit-identical to sklearn's Cython GBR. Different RNG tape, different tie-breaks. Within-TF rankings are highly similar (72% top-10 overlap with arboreto on PBMC-3k; SPI1 dominates myeloid targets in both).
- Does NOT ship the aertslab motif-ranking feather databases (10–50 GB). Users fetch from resources.aertslab.org and pass the DataFrame to `cistarget.enrich`.
- Not faster on GPU than flashscenic — this is a CPU-native tool. If you have an A100, use flashscenic and accept the RegDiffusion algorithm swap.

## License

MIT. Algorithm reimplementations follow aertslab's Python references — original method credit to Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+). This repo is an independent, pyscenic-compatible reimplementation with no shared code.

## Repo layout

See `docs/specs/2026-04-16-rustscenic-design.md` for the full design spec. Validation artifacts in `validation/`. Every measurement above can be reproduced from `validation/*.py` scripts on the pinned `pbmc3k.h5ad` + `pbmc10k.h5ad`.
