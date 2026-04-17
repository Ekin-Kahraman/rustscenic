# rustscenic — design spec

**Date:** 2026-04-16
**Author:** Ekin Kahraman (with Claude)
**Status:** Draft — awaiting user review
**Target v1.0 completion:** ~2 weeks active Claude+user time, shipped stage-by-stage

---

## 0. One-paragraph pitch

`rustscenic` is a Rust + PyO3 reimplementation of the four slow stages of the SCENIC+ single-cell regulatory-network pipeline — GRNBoost2, AUCell, pycisTopic (LDA), and pycisTarget (motif enrichment) — shipped as a single pip-installable wheel with zero Python runtime dep rot. Each stage is a drop-in replacement for its pyscenic / aertslab counterpart, numerically faithful (edge-rank Spearman ≥0.95, topic ARI ≥0.85), and 3–10× faster. End-to-end SCENIC+ on a 50k-cell dataset goes from **8–16 hours (if it installs at all) → 1–2 hours, one pip install.**

---

## 1. Problem

### 1.1 The bottleneck Moha actually named

From the Apr 13 2026 email thread with Kuan Huang (Mount Sinai) and Mohamed El Moussaoui:

- **Moha #1 pain:** "dependency management... many tools rely on other packages behind the scenes, and this can easily break the environment... combine several tools with incompatible dependencies."
- **Moha #4 pain:** "scATAC-seq workflows are also quite slow, especially tools involving pycisTopic, pycisTarget, and SCENIC+."
- **Kuan's explicit steer:** "Maybe addressing moha's ideas is more straightforward in terms of tool dev."

### 1.2 What the profiling confirmed

Local audit (Apr 16 2026) verified both pain points as real:

- **arboreto 0.1.6 (the GRN inference backbone of pyscenic) does not install on modern Python stacks.** Broken against dask 2024+, numpy 2.x, pandas 3.x. Tried 5 dask versions — all fail. Moha's #1 pain, manifested.
- **pycisTopic repo dormant since Sept 2024** (19 months, 64 open issues, 60 about slowness/memory). Community PR #226 opened 6 days before this spec adding a `tomotopy` backend — patching the symptom, not the dep web.
- **arboreto last commit: 1 in 4 years.** pyscenic in pure maintenance mode.
- **flashscenic (Mar 2026, Hao Zhu)** replaced GRNBoost2 with a diffusion model on GPU — fast, but not numerically faithful to pyscenic and requires CUDA + PyTorch stack.

### 1.3 Runtime pain at realistic scales

Per local cProfile on PBMC-3k with 1274 TFs (see `validation/reports/2026-04-16-grnboost2-profile.md`):

| Scale | Single core | 8 cores |
|---|---|---|
| PBMC 3k (2700 × 1274 TFs) | 20 min | 2.5 min |
| 50k-cell atlas | 6 h | 45 min |
| 200k-cell SCENIC+ eGRN step | 24 h | 3 h |

Time breakdown at GRNBoost2 stage:
- 62% in sklearn Cython `Tree.build` (the actual work)
- 38% in Python overhead (sklearn validation, `inspect`, boosting-stage Python loop, numpy scratch)
- ~4% in Dask scheduling (NOT the bottleneck)

### 1.4 Why no existing tool fixes this

| Tool | What it does | Why it doesn't solve the problem |
|---|---|---|
| arboreto (aertslab) | Reference GRNBoost2 | Abandoned, broken on modern deps |
| pyscenic (aertslab) | Full SCENIC pipeline Python | Same |
| pycisTopic (aertslab) | LDA topic modeling | Dormant, Mallet-Java is slow and OOMs |
| pycistarget (aertslab) | Motif enrichment | Tied to aertslab feather rankings DB (10s GB) |
| flashscenic (hao zhu) | GPU fast SCENIC | Changes algorithm (RegDiffusion), not reproducible; requires CUDA |
| tomotopy PR #226 | C++ AVX-512 Gibbs | Unmerged; fixes one stage; still needs the full broken stack |
| rapids-singlecell | GPU scverse acceleration | Explicitly doesn't include SCENIC |
| SnapATAC2 / PeakVI | scATAC embeddings | Can't feed SCENIC+ downstream |

**Open niche:** CPU-native Rust+PyO3 drop-in that restores installability AND delivers 3–10× per stage AND preserves numerical equivalence with pyscenic. No one has shipped this.

---

## 2. Scope

### 2.1 In scope for v1.0

Four stages, one PyO3 wheel, one CLI binary:

| Stage | Replaces | Target speedup | Correctness metric |
|---|---|---|---|
| `grn` | arboreto.grnboost2 | 3–5× | Edge-rank Spearman ≥0.95 on top-10k edges |
| `aucell` | pyscenic AUCell | 2–3× | Per-cell regulon AUC r ≥0.99 |
| `topics` | pycisTopic LDA | 3–10× | Topic assignment ARI ≥0.85 (30-run mean) |
| `cistarget` | pycistarget | 3–10× | Motif-rank AUC correlation ≥0.95 |

### 2.2 Explicitly NOT in scope

These were candidates that the audit struck:

- **Adaptive-QC / Yates cancer-aware QC.** Profiling showed QC is a seconds-level problem. Not a real bottleneck. This was Ekin's original email commitment to Kuan on Apr 14 — superseded by profile data.
- **Spatial Visium HD scalability.** BPCells (CPU, bioRxiv 2025) + rapids-singlecell (GPU) already own the niche. SpaceRanger v4 (June 2025) ships native segmentation, moving the field away from 11M-spot raw analysis.
- **Visualization compounding.** matplotlib/datashader problem, not a Rust problem.
- **GPU acceleration.** flashscenic's angle. Different tool, different project.
- **End-to-end SCENIC+ orchestration.** Users still use the `scenicplus` Python package for pipeline wiring. We replace the slow stages inside it.
- **R bindings, Julia bindings, JVM bindings.** Out of scope.
- **Novel methodology** (e.g., inventing a new GRN algorithm). We are faithful to the existing algorithms — our contribution is engineering, not methodology.

### 2.3 Versioning roadmap

- **v0.1** — `grn` stage alone. ~3 days active time.
- **v0.2** — adds `aucell`. ~1 day.
- **v0.3** — adds `topics`. ~2–3 days.
- **v0.4** — adds `cistarget`. ~2–3 days.
- **v1.0** — integration, docs, PyPI release, agent skill. ~1–2 days.

Each release is independently useful; users can upgrade stage-by-stage.

---

## 3. Architecture

### 3.1 Repo layout

```
rustscenic/
├── Cargo.toml                    # workspace
├── pyproject.toml                # maturin config
├── README.md
├── crates/
│   ├── rustscenic-core/          # shared: sparse CSR/CSC, errors, PyO3 utils
│   ├── rustscenic-grn/           # v0.1: GRNBoost2 replacement
│   ├── rustscenic-aucell/        # v0.2: AUCell regulon scoring
│   ├── rustscenic-topics/        # v0.3: online VB LDA or tomotopy wrap
│   ├── rustscenic-cistarget/     # v0.4: motif rankings lookup
│   ├── rustscenic-cli/           # single binary: `rustscenic <stage>` subcommands
│   └── rustscenic-py/            # PyO3 wheel — maturin-built
├── python/rustscenic/            # Python package: re-exports + AnnData glue
├── validation/
│   ├── reference/
│   │   ├── Dockerfile            # pyscenic 0.12.1 + arboreto 0.1.6 + dask 2024.1.1 pinned
│   │   ├── datasets.sh           # fetch PBMC-10k, aertslab refs
│   │   └── run_reference.py      # generate baseline outputs
│   ├── baselines/                # git-LFS committed golden outputs (parquet)
│   ├── compare.py                # diff vs baseline, emit metrics
│   ├── benchmarks.py             # wall-clock, peak RSS, append CSV
│   └── results.csv               # append-only audit log, git-tracked
├── .github/workflows/
│   ├── audit.yml                 # per-PR: build, compare, benchmark, post
│   └── release.yml               # tag → PyPI + GitHub release
├── skills/rustscenic.md          # agent skill (RastQC-style)
└── docs/
    └── specs/
        └── 2026-04-16-rustscenic-design.md    # THIS FILE
```

### 3.2 Dependencies (Rust side)

Shared across stages:
- `pyo3 = 0.24` — Python bindings (version pinned to match rustqc/rustscrublet)
- `numpy = 0.24` — zero-copy numpy ⇔ ndarray
- `rayon = 1.10` — data parallelism
- `sprs = 0.11` — sparse CSR/CSC (already in rustscrublet)
- `ndarray = 0.16` — dense arrays
- `anyhow` / `thiserror` — errors
- `serde` + `arrow-parquet` — output serialization
- `hdf5 = 0.8` — AnnData/H5AD reading

Stage-specific:
- `grn`: gradient-boosting with **sklearn-compatible semantics** (Huber loss, depth-3 trees, sklearn feature-importance definition, `* n_estimators` denormalization per arboreto/core.py:168). **Histogram binning is an internal optimization** — semantics match sklearn, not LightGBM. ~500 lines, no external GBM crate.
- `aucell`: uses `rustscrublet/src/sparse.rs` patterns
- `topics`: `statrs` for digamma/gamma, custom online VB (Hoffman 2010 update rules)
- `cistarget`: `memmap2` for mmap'd feather rankings, `arrow2` for feather parsing

### 3.3 Python package surface

```python
import anndata as ad
import rustscenic

# Stage 1: GRN inference
adata = ad.read_h5ad("data.h5ad")
tfs = rustscenic.load_tfs("hs_hgnc_tfs.txt")
adjacencies = rustscenic.grn.infer(adata, tf_names=tfs, n_threads=8, seed=777)
# → pandas DataFrame with (TF, target, importance)

# Stage 2: AUCell
regulons = rustscenic.regulons_from_adjacencies(adjacencies, motif_db)
auc_matrix = rustscenic.aucell.score(adata, regulons)

# Stage 3: cisTopic LDA
atac_adata = ad.read_h5ad("atac.h5ad")
topics = rustscenic.topics.fit(atac_adata, n_topics=100, method="online_vb", seed=777)

# Stage 4: cisTarget
enrichments = rustscenic.cistarget.enrich(topics, feather_db="hg38_screen_v10.feather")
```

### 3.4 CLI surface

```
rustscenic grn \
  --expression data.h5ad \
  --tfs hs_hgnc_tfs.txt \
  --output grn.parquet \
  --threads 8 --seed 777

rustscenic aucell \
  --adjacencies grn.parquet \
  --expression data.h5ad \
  --output auc.parquet

rustscenic topics \
  --atac atac.h5ad \
  --n-topics 100 \
  --method online_vb \
  --output topics.parquet

rustscenic cistarget \
  --topics topics.parquet \
  --feather-db hg38_screen_v10.feather \
  --output enrichment.parquet
```

---

## 4. v0.1 detailed scope: `grn` stage

### 4.1 Algorithm

**Semantics faithful to sklearn's `GradientBoostingRegressor` as used by `arboreto.algo.grnboost2` (confirmed against `arboreto/core.py:150-213`):**

- Loss: Huber (sklearn `HuberLoss` with `alpha=0.9`; arboreto default `SGBM_KWARGS`)
- `n_estimators=5000`
- `max_depth=3`, `max_features="sqrt"`
- `learning_rate=0.01`, `subsample=0.9`
- Early stopping via arboreto's `EarlyStopMonitor` (window 25): halt when recent improvement stalls
- Importance: sklearn's per-feature Gini accumulation, **multiplied by `n_estimators`** per arboreto/core.py:168
- Output schema: `['TF', 'target', 'importance']` (capital TF), filtered to `importance > 0`, sorted descending per target

**Implementation optimizations (internal, do not change outputs modulo float precision):**
1. Histogram binning (255 buckets) — split eval O(buckets × features) not O(cells × features).
2. No GIL callbacks during fit.
3. Per-target rayon parallelism, no Dask.
4. f32 throughout; validate Spearman/Jaccard hold vs sklearn f64.
5. Zero-copy sparse CSR input via PyO3 buffer pointers.

**Seed fidelity:** numpy MT19937 drives sklearn's randomness. Replicate bit-for-bit via `rand_mt` crate in the same consumption order as sklearn (subsample mask → `max_features` column selection → tree splits). Document the RNG tape in `crates/rustscenic-grn/src/rng.rs`.

### 4.2 Inputs

- AnnData `.h5ad` OR dense TSV (cells × genes) OR 10x MEX directory
- TF list: TSV with one gene symbol per line
- Optional: `--tf-col` to read TFs from an AnnData var column

### 4.3 Outputs

- `grn.parquet`: `(TF: str, target: str, importance: f32)` — capital `TF`, importance denormalized (`* n_estimators`), filtered `importance > 0`, sorted descending per target. Exact-schema match for arboreto.
- `grn.json`: metadata — wall-clock, peak RSS, git SHA, seed, dataset SHA256, rustscenic version, per-target `n_estimators_effective` if early stopping fired.

### 4.4 Validation (real-time audit)

**Reference baseline:**
- Build Docker image with pyscenic 0.12.1 + arboreto 0.1.6 + dask 2024.1.1 + numpy 1.26 + pandas 2.1 + lightgbm 4.6 + scanpy 1.11
- Run `arboreto.grnboost2(expression_matrix, tf_names=tfs, seed=777)` on PBMC-3k
- Save output to `validation/baselines/pbmc3k_grn.parquet`
- Commit to git-LFS

**Per-commit audit (CI):**
- Build wheel via maturin → install → run `rustscenic grn --seed 777` on PBMC-3k
- `validation/compare.py --stage grn` computes:
  - **Jaccard of top-10k edge sets** — must ≥0.80
  - **Spearman on importance ranks across union of top-10k** — must ≥0.85
  - Per-TF top-100 target overlap, averaged across all TFs (not `[:50]`) — must ≥0.70
- `validation/benchmarks.py --stage grn`: wall-clock + peak RSS (via `psutil`, not `RUSAGE_CHILDREN` — the latter is cumulative across session children and pollutes sequential runs).
- Append row to `validation/results.csv`
- PR comment: `grn: Jaccard 0.832 (Δ-0.003), Spearman-union 0.891 (Δ+0.012), wall 142s (ref 620s, 4.4×)`

### 4.5 Success criteria (v0.1 hard gates)

- [ ] Jaccard top-10k edges ≥0.80, Spearman-on-union ≥0.85, per-TF top-100 overlap ≥0.70 — all vs arboreto-sync reference on PBMC-3k with matching seed
- [ ] 3× or better wall-clock single-thread vs arboreto-sync reference
- [ ] `pip install rustscenic==0.1.0` succeeds on Python **3.10/3.11/3.12/3.13** × macOS arm64/x86_64 × Linux x86_64/aarch64. **Windows deferred to v1.0** (hdf5-rs + maturin Windows matrix costs more than v0.1 warrants).
- [ ] Wheel size ≤8 MB uncompressed
- [ ] Zero Python runtime deps except numpy, pandas, pyarrow
- [ ] Agent skill present at `skills/rustscenic.md` in repo AND installed at `~/.claude/skills/rustscenic/SKILL.md`

### 4.6 Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Histogram GBM matches sklearn trees but not speed | Medium | Collapses speed to 1.6× | Still ships; install fix alone is valuable |
| Stochastic GBM output divergence at matching seed | Medium | Fails Spearman gate | Implement bit-identical RNG (numpy's MT19937 in Rust via `rand_pcg` / `rand_mt`) |
| sklearn Huber loss implementation subtleties | Low | Tree quality drift | Port Huber exactly from sklearn `_loss/losses.py` line-by-line |
| PyO3 0.24 API breakage during development | Low | Build friction | Pin exact version, match rustqc/rustscrublet working config |

---

## 5. v0.2–v0.4 scope summaries

### v0.2: `aucell` (~1 day)
- Algorithm: argsort genes per cell; recovery-curve AUC on regulons
- Reuse: `rustscrublet/src/sparse.rs` patterns
- Validation: per-cell regulon AUC correlation ≥0.99 vs pyscenic.aucell on PBMC-10k

### v0.3: `topics` (~2–3 days)
- Default: wrap tomotopy (if PR #226 merged) OR port Hoffman 2010 online VB
- Digamma/gamma via `statrs`
- Validation: topic assignment ARI ≥0.85 (30-run mean, different seeds) vs pycisTopic-Mallet on PBMC-10k Multiome
- Output format: cell-topic + topic-region matrices in parquet (pycisTopic-compatible schema)

### v0.4: `cistarget` (~2–3 days)
- mmap the aertslab feather ranking DB (10–50 GB)
- Per-motif recovery curves on ranked regions per topic
- rayon parallel across motifs
- Validation: motif-rank AUC correlation ≥0.95 vs pycistarget

---

## 6. Real-time audit architecture

### 6.1 Design principle

**Every commit must improve or preserve all of: correctness ≥ threshold, speed ≥ baseline, install ≥ matrix.** No "validate later." No "TODO: benchmark."

### 6.2 What runs on every PR

```yaml
# .github/workflows/audit.yml (abridged)
jobs:
  audit:
    strategy:
      matrix:
        os: [macos-14, ubuntu-24.04, windows-2022]
        python: ["3.9", "3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - run: cargo build --release --workspace
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test --workspace
      - run: maturin build --release
      - run: pip install target/wheels/rustscenic-*.whl
      - run: docker pull rustscenic-reference:latest
      - run: python validation/run_ours.py --all-stages
      - run: python validation/compare.py --all-stages --fail-below-threshold
      - run: python validation/benchmarks.py --all-stages
      - run: python validation/post_pr_comment.py
```

### 6.3 results.csv schema (append-only, git-tracked)

```csv
commit_sha,stage,dataset,metric_name,value,wall_clock_s,peak_rss_mb,date
<sha>,grn,pbmc3k,spearman_top10k,0.9572,142.3,480,2026-04-16T14:22:11Z
<sha>,grn,pbmc3k,exact_rank_top100,0.781,142.3,480,2026-04-16T14:22:11Z
```

### 6.4 Trajectory tracking

README renders a mermaid chart of Spearman + wall-clock over commits. Regression visible in a glance.

### 6.5 Reference environment

Docker image published to ghcr.io/ekin-kahraman/rustscenic-reference. Fixed at:
- pyscenic 0.12.1
- arboreto 0.1.6
- dask 2024.1.1
- numpy 1.26.4
- pandas 2.1.4
- lightgbm 4.6.0
- scanpy 1.11.5

This image is our ground truth. We don't upgrade it except to document divergence.

---

## 7. Release & distribution

- **crates.io:** `rustscenic-core`, `rustscenic-grn`, etc. (publish in order as stages land)
- **PyPI:** `rustscenic` — one wheel, contains all stages available at that release
- **GitHub releases:** pre-built wheels for all supported platforms
- **Agent skill:** `skills/rustscenic.md` bundled with release
- **Homebrew tap:** deferred to post-v1.0 based on adoption

---

## 8. Relationship to Ekin's existing repos

Reusable infrastructure (no code duplication):
- `rustqc` — PyO3 wiring patterns, numpy interop
- `rustscrublet` — sparse CSR ops, gene filter, normalization, PCA, HNSW-KNN patterns
- `rustcell` — CLI arg parsing, HTML report scaffolding
- `rustnn` — not directly used (brute-force KNN doesn't scale past 10k)

Reference where applicable; don't fork. If we need e.g. sparse CSR utilities across all rewrites, consider extracting to a shared `ekincore` crate post-v1.0.

---

## 9. Communication plan

### 9.1 Email to Kuan after v0.1 lands

Draft (to be sent when v0.1 is merged to main with passing audit):

> Subject: rustscenic v0.1 — following your steer
>
> Hi Kuan,
>
> Quick update. The profiling I ran on GRNBoost2 showed QC-level tools are seconds-level problems for users — adaptive-QC / Yates wouldn't move the needle. Following your suggestion to address Moha's ideas, I've scoped a Rust+PyO3 replacement for the SCENIC+ slow stages — GRNBoost2 first, with AUCell, pycisTopic, and pycisTarget to follow.
>
> v0.1 (just GRN inference) is live at github.com/Ekin-Kahraman/rustscenic. Numerical equivalence with pyscenic (edge-rank Spearman X.XX on PBMC-3k), Y× faster, one pip install with no dask/numpy/pandas dep rot.
>
> Happy to jump on a call once v0.3 (topics) lands if there's lab interest in validation on your datasets.
>
> Ekin

### 9.2 Post-v1.0

- Submit brief to bioRxiv co-authored with Kuan (if interested)
- Announce on scverse Slack, bioinformatics.stackexchange, bsky
- Add to SCENIC+ ecosystem tools page via PR to aertslab docs

---

## 10. Resolved decisions (post-audit 2026-04-17)

1. Python floor: **3.10** (scanpy 1.11 requires 3.10).
2. `grn` accepts dense ndarray, sparse CSR, and AnnData inputs.
3. Errors: `thiserror` in library crates, `anyhow` in CLI binary, `PyException` across FFI.
4. Reference baseline: **pre-rendered parquet committed to the repo**; Dockerfile reproduces it.
5. tomotopy PR #226 — decide at v0.3 based on merge status.

## 10.1 Open (deferred)

- Whether to ship `rustscenic run-all` end-to-end wrapper chaining all four stages (feature creep; add only if users ask).

---

## 11. Definition of done

**v0.1 done =** `grn` stage passes correctness + speed + install gates on all target platforms; agent skill deployed; results.csv tracking baseline.

**v1.0 done =** all four stages pass gates; PyPI `rustscenic==1.0.0` live; README benchmarks + mermaid trajectory; Kuan email sent; agent skill at `~/.claude/skills/rustscenic/SKILL.md`.

---

_End of spec._
