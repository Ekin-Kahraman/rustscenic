# rustscenic

Rust + PyO3 reimplementation of the four slow stages of the SCENIC+ single-cell regulatory-network pipeline. Drop-in replacements for `arboreto.grnboost2`, `pyscenic.aucell`, `pycisTopic`, and `pycistarget` — one `pip install`, zero Python runtime dep rot, 3–10× faster per stage.

**Status:** scaffolding (v0.1 under active development)

## Why

The pyscenic / SCENIC+ stack is the reference for gene regulatory network inference in single-cell biology, but:

- `arboreto` (GRN inference) is effectively abandoned — 1 commit in 4 years and will not install on any modern dask/numpy/pandas combination.
- `pycisTopic` (LDA) is dormant (no commits to `main` since Sept 2024, 64 open issues, 60 about slowness).
- End-to-end SCENIC+ on a 50k-cell dataset takes 8–16 hours if the stack installs at all.

`rustscenic` makes SCENIC+ installable again, numerically faithful to pyscenic, and 3–10× faster per stage.

## Design principles

1. **Numerical faithfulness.** Every stage is validated against its pyscenic/aertslab reference on every commit (Spearman ≥0.95 for GRN edges, ARI ≥0.85 for topics, etc.).
2. **Single wheel.** One `pip install rustscenic` replaces arboreto + pyscenic stages + pycisTopic + pycistarget with zero transitive Python dependencies.
3. **Stage-by-stage shipping.** Each release is useful alone; users upgrade per stage.
4. **Real-time audit.** Every PR runs correctness diffs + benchmarks against a pinned reference Docker image, posts results to the PR, and appends to `validation/results.csv`.

## Layout

See `docs/specs/2026-04-16-rustscenic-design.md` for the full design spec.

```
crates/
  rustscenic-core/      shared sparse + PyO3 utils
  rustscenic-grn/       v0.1: GRNBoost2 replacement
  rustscenic-aucell/    v0.2: regulon AUC scoring
  rustscenic-topics/    v0.3: cisTopic LDA replacement
  rustscenic-cistarget/ v0.4: motif enrichment
  rustscenic-cli/       single binary with subcommands
  rustscenic-py/        PyO3 wheel
validation/
  reference/            pinned pyscenic Docker image
  baselines/            golden outputs (git-LFS)
  compare.py            correctness diff
  benchmarks.py         wall-clock + RSS
  results.csv           append-only audit log
```

## Credit

Reimplements algorithms from Aibar et al. 2017 (SCENIC), Bravo González-Blas et al. 2023 (SCENIC+), Subramanian et al. 2022 (DDQC), and Hoffman et al. 2010 (Online VB LDA). All algorithm definitions follow the reference Python implementations in the [aertslab](https://github.com/aertslab) organization.

## AI disclosure

Implementation assisted by Claude. Correctness validated by automated output comparison against pyscenic/aertslab references on every commit, not by manual code review alone.

## License

MIT
