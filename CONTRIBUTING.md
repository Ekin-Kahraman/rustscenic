# Contributing to rustscenic

Bug reports, correctness fixes, documentation and measured performance
improvements are welcome.

Ekin Kahraman · Mohamed El Moussaoui · Kuan-Lin Huang · Longtao Huangfu · Fuaad Siyaad

[Contributions](AUTHORS.md)

## Ground rules

- **Measure agreement.** Compare changed outputs with the relevant reference and record differences in `validation/`. Agreement varies by stage; see the [benchmark scope](site_docs/benchmarks.md). Algorithm changes need before/after results with inputs, versions and settings.
- **Tests before features.** New Rust code needs a `#[test]`; new Python paths need a `tests/test_*.py`. CI runs both.
- **Check reproducibility.** Hold inputs, software version, seed, thread count and settings fixed. Test repeatability and document any expected numerical variation.
- **Report invalid inputs clearly.** Validate inputs and raise an actionable error rather than returning misleading results. See `crates/rustscenic-grn/src/histogram.rs` for the NaN guard.

## Setup

```bash
git clone https://github.com/Ekin-Kahraman/rustscenic
cd rustscenic
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy pandas pyarrow scipy anndata
rustup target add <your-target-triple>  # if cross-compiling

maturin develop --release --manifest-path crates/rustscenic-py/Cargo.toml
```

After any Rust change:

```bash
cargo test --workspace --exclude rustscenic-py --release
cargo clippy --workspace --exclude rustscenic-py --all-targets -- -D warnings
maturin develop --release --manifest-path crates/rustscenic-py/Cargo.toml
pytest tests/
```

## Where things live

- `crates/rustscenic-{core,grn,aucell,topics}` — Rust algorithm crates, no Python. Write `#[test]` here.
- `crates/rustscenic-py` — PyO3 bindings. Don't add algorithm code here; keep it a thin wrapper.
- `crates/rustscenic-cli` — standalone Rust CLI binary. Separate from the Python `rustscenic` CLI.
- `python/rustscenic/` — Python package (public API + `rustscenic` CLI).
- `tests/` — pytest, exercises the Python API.
- `validation/` — reproducible scripts, measurements and output checks; `scaling/`, `head_to_head/` and `community/` cover scale tests, reference comparisons and collaborator runs. Earlier records remain in `ours/`.
- `examples/` — user-facing example scripts. End-to-end, self-contained.

## PR checklist

- [ ] `cargo test --workspace --exclude rustscenic-py --release` passes
- [ ] `cargo clippy --workspace --exclude rustscenic-py --all-targets -- -D warnings` passes
- [ ] `pytest tests/` passes
- [ ] If algorithm output changed: a new or updated record in `validation/` with before/after numbers and reproduction steps
- [ ] If you added a public function or kwarg: added to the relevant docstring + at least one test
- [ ] README / CHANGELOG updated if user-facing

## Style

- Rust: edition 2021, `rustfmt` defaults, no `#[allow(clippy::...)]` without a one-line justification comment on the same line.
- Python: PEP 8, type hints on public API, 4-space indent.
- No trailing whitespace, no tabs.
- Commit messages: lowercase imperative first line, concise explanation after a blank line. See recent commits for the pattern.

## Reporting bugs

Open an issue with:
- `pip show rustscenic` output
- Python version + OS
- A minimal reproducible example (can be synthetic; dataset-dependent crashes are welcome, but the smaller the repro, the faster it gets fixed)
- Expected vs actual output

Scientific-correctness bugs are a priority. Include the reference command and
expected result where available, and distinguish documented method differences
from regressions. See [validation](site_docs/validation.md) and
[topic-model limitations](docs/topic-collapse.md).

## Documentation

The API reference is rendered from Python source and docstrings with
mkdocstrings. Building it does not import RustScenic or require the Rust
extension, numerical dependencies or a package installation.

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

Keep the [API map](site_docs/api.md) as a workflow guide. Add new public modules
to the [reference navigation](mkdocs.yml); members and signatures are collected
from their source. Members without docstrings still appear with their signatures.

## License

By contributing, you agree that your contributions are licensed under the Apache License, Version 2.0, including the patent grant and termination terms in that licence.
