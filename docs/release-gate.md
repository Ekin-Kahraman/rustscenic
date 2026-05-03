# Release gate

The bar a rustscenic release must clear before a tag goes public. Every check must be green on a fresh environment for the release to be considered "publishable end-to-end" rather than "compute stages work in isolation".

## 1. Fresh-environment install matrix

Each install path must succeed in a clean Python venv with no prior rustscenic deps installed.

| Path | Install command | Verifies |
|---|---|---|
| Core | `pip install "rustscenic @ git+https://github.com/Ekin-Kahraman/rustscenic@vX.Y.Z"` | numpy, pandas, pyarrow, scipy resolve; `import rustscenic` works |
| Examples | `pip install "rustscenic[examples] @ git+...@vX.Y.Z"` | + scanpy, anndata, igraph, leidenalg |
| Validation | `pip install "rustscenic[validation] @ git+...@vX.Y.Z"` | + scikit-learn |
| Reference | `pip install "rustscenic[reference] @ git+...@vX.Y.Z"` (Linux preferred) | + pyscenic, arboreto, ctxcore |
| Benchmarks | `pip install "rustscenic[benchmarks] @ git+...@vX.Y.Z"` | + tomotopy, gensim, psutil |
| Reference Docker | `docker build -t rustscenic-ref validation/reference/` then run | Pinned 2024-stack arboreto/pyscenic for reproducible parity |

Status as of v0.3.6: Core, Examples, Validation verified. Reference + Benchmarks + Docker are pinned but not part of automated release-gate yet.

## 2. Script smoke tests

Each must run from a fresh venv after the corresponding extra is installed, with no manual intervention.

| Script | Install path | Verifies |
|---|---|---|
| `examples/pbmc3k_end_to_end.py` | `[examples]` | RNA GRN + AUCell + leiden + biology check (canonical TFs hit expected lineages) |
| `examples/atac_fragments_to_matrix.py` | core | preproc.fragments_to_matrix on synthetic data |
| Full `pipeline.run` (RNA + ATAC + motifs + enhancer + eRegulon) | `[examples]` + motif rankings | `rustscenic.cistarget.enrich`, `rustscenic.enhancer.link_peaks_to_genes`, `rustscenic.eregulon.build_eregulons` |
| `validation/validate_multiome_e2e.py` (or equivalent on real PBMC multiome) | `[validation]` + dataset | Real-data multiome end-to-end without manual debugging |

Status as of v0.3.6:
- `examples/pbmc3k_end_to_end.py`: ✅ verified live (41,338 edges, 4/4 canonical regulons in expected lineages, 13.5 s wall, fresh tmp dir)
- `examples/atac_fragments_to_matrix.py`: ✅ covered by `tests/test_preproc_python_api.py` (11 tests pass)
- Full `pipeline.run` (RNA + ATAC + motifs + enhancer + eRegulon): ✅ covered by `tests/test_full_scenicplus_smoke.py` (2 pass) + `tests/test_pipeline_integration.py` (10 pass — multiome, cellxgene-shaped RNA, gene coords, gibbs topics, region cistarget, rankings parsing)
- Real PBMC multiome end-to-end: ⚠ tests cover synthetic multiome end-to-end; first real-data attempt was Lucy's 2026-05-01 run (predated v0.3.6 packaging fix)

Aggregate as of v0.3.6: **144 Python tests pass (1 skipped) + 57 Rust inline tests pass (grn 12, topics 8, preproc 32, aucell 5). Bit-identical determinism verified live (68,565-edge GRN identical under same seed).**

## 3. Claim-vs-evidence matrix

Every concrete claim in `README.md` must map to one of: a passing test, a measurable benchmark, a logged artefact under `validation/`, or be explicitly softened to "not yet proven".

Anchor claims at v0.3.6:

| Claim | Evidence | Status |
|---|---|---|
| "Four runtime dependencies" | `pyproject.toml` core deps | ✓ proven |
| "Python 3.10–3.13, Linux + macOS x86_64+aarch64" | release wheel matrix | ✓ proven (4 wheels per release) |
| "GitHub Release wheels and source install succeed" | release.yml CI green per tag | ✓ proven on v0.3.6 |
| AUCell wall-time numbers (Ziegler, Multiome) | `validation/aucell_celltype_pbmc10k.py` log | ⚠ pre-existing logs, not regenerated per release |
| AUCell per-cell Pearson 0.984 mean | `validation/validate_aucell_pbmc10k.py` log | ⚠ same |
| GRN per-edge Spearman 0.58 vs arboreto | `validation/compare_pipelines_multiome.py` log | ⚠ requires `[reference]` install + pinned data |
| Cistarget kernel Pearson 1.0000 vs ctxcore | log file | ⚠ requires `[reference]` |
| 100k-cell bootstrap 17 min / 5 GB peak RSS | scaling logs | ⚠ requires real data |
| Bit-identical output under same seed | `crates/rustscenic-grn/src/rng.rs` + `crates/rustscenic-topics/src/gibbs.rs` inline tests + live 68,565-edge GRN reproducibility check | ✅ proven |
| End-to-end real PBMC multiome runs without hand-holding | synthetic-multiome end-to-end smoke covered by `test_full_scenicplus_smoke.py` + `test_pipeline_integration.py`; first real-data attempt was Lucy's 2026-05-01 run on v0.3.5 (predated packaging fix) | ⚠ partially proven (synthetic ✓, real-data first-run gap closed in v0.3.6 but unverified end-to-end with rerun) |

The `⚠` and `❌` rows are the publication-threshold bottleneck.

## 4. Publication threshold

The release is "publishable end-to-end" only when ALL of:

- [x] Fresh install works on every declared install path (core ✓, examples ✓, validation ✓; reference/benchmarks declared but not auto-tested)
- [x] Synthetic multiome end-to-end completes via `pipeline.run` (covered by `test_full_scenicplus_smoke.py` + `test_pipeline_integration.py`, 12 passing)
- [ ] **Real-data** PBMC multiome end-to-end completes from a fresh venv without intervention
- [ ] Memory/time table has hardware, dataset, command, version baked in alongside numbers
- [ ] SCENIC+/pySCENIC parity numbers regenerated against current pyscenic, not 2026-04-snapshot
- [x] Docs tell users exactly which install path to use (`docs/tester-quickstart.md` ✓)
- [x] Bit-identical determinism under same seed verified (live + Rust inline tests)
- [ ] Audit workflow checks each install path's smoke test on every tag push

v0.3.6 satisfies 5 of 8. Status: "compute stages individually validated, end-to-end collaborator path not yet release-gated".

## 5. What changes from v0.3.6 to a publishable release

In rough EV order:

1. **Add a real-multiome smoke test to the audit workflow.** Pull a small public 10x multiome dataset (e.g., bundled-with-scanpy or 10x example), run the full pipeline.run, assert non-empty outputs at every stage. Publish the wall/memory numbers per stage in the release notes.
2. **Regenerate the parity numbers in `validation/` per release** rather than referring to 2026-04 logs. Tag each log with the release SHA it was produced from.
3. **Wire `[reference]` install into a CI job** that runs at least one comparison script (e.g., compare_pipelines_multiome.py) so we can detect upstream pyscenic/arboreto API drift.
4. **Add an install-matrix CI job** that runs `pip install "rustscenic[<extra>]"` for each extra in a fresh container, validates imports, runs the corresponding smoke script.
5. **Cut README to claims that have green evidence rows in section 3.** Anything ⚠ or ❌ either gets backed by a regenerated log or moved into "in progress".

When all five land, the next tag (v0.4.0) gets called publishable.

## Non-goals

- Tests for the SCENIC/scenicplus reference pipelines themselves (those are external; we only test our parity against snapshots)
- Windows support (out of scope; documented in install matrix)
- GPU execution (CPU-only by design)
