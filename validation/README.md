# validation/

Reproducibility scripts + measurement artefacts behind the rustscenic README
numbers. Every claim made in the top-level README and CHANGELOG has a
corresponding script here.

## Layout

- `ziegler_headtohead_2026-04-19.md` — flagship real-atlas head-to-head
  with pyscenic on 31,602 nasopharyngeal cells (Ziegler et al. 2021).
- `VALIDATION_SUMMARY.md` — one-page index of every measured number.
- `fresh_install_proof.md` — the 2026 "what happens in a cold venv" test.
- `figures/` — the three headline figures (tool-validation only; biology
  figures live in the companion case-study).
- `reference/` — pinned `python:3.11-slim` Docker + `run_reference.py`
  for regenerating the cached arboreto/pyscenic outputs we compare against.
- `baselines/` — small committed parquets (top-10k edges) used as
  CI-smoke-test inputs to `compare.py`.
- `validate_*.py`, `*_pipeline.py`, etc. — the driver scripts.
- `compare.py` — the CLI tool CI calls.

## ⚠️ Paths are historically hardcoded

Most of the `validate_*.py` scripts hardcode absolute paths to the author's
filesystem (`/Users/ekin/rustscenic/validation/reference/data/...`). These
are the original research scripts committed for reproducibility, **not**
user-facing entry points — if you want to regenerate numbers on your own
environment, you will need to adapt the paths. The rustscenic package
itself (GitHub Release wheel / source install today, PyPI once live)
does not contain any hardcoded paths.

For user-facing pipelines, see `examples/pbmc3k_end_to_end.py` in the
repo root — that script downloads its own data and has zero hardcoded
paths.

## Regenerating the Docker reference

```bash
cd validation/reference
docker build -t rustscenic-reference .
docker run --rm -v $(pwd)/data:/data rustscenic-reference
```

This produces the pyscenic/arboreto cached outputs that the rest of the
validation scripts compare against. Pinned to `pyscenic 0.12.1` + `arboreto
0.1.6` + `dask 2024.1.1` + `distributed 2024.1.1` + `pandas 2.1.4` — the
last versions where pyscenic's own stack cohabits cleanly in a single env.
