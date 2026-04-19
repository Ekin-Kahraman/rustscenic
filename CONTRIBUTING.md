# Contributing to rustscenic

Thanks for the interest. rustscenic is a small project — bug reports, correctness fixes, and measured performance improvements are especially welcome.

## Ground rules

- **Match pyscenic semantics.** A "drop-in" tool that produces different numbers than the reference is worse than a tool that doesn't exist. If you change the algorithm, show a new `validation/ours/*_2026-*.md` document that measures the change.
- **Tests before features.** New Rust code needs a `#[test]`; new Python paths need a `tests/test_*.py`. CI runs both.
- **Deterministic output.** Same input + same seed must produce bit-identical output. If a PR breaks determinism, it's not landing.
- **No silent corruption.** Prefer `panic!` or `raise` with a clear message over a plausibly-wrong output. See `crates/rustscenic-grn/src/histogram.rs` for the NaN-guard pattern.

## Setup

```bash
git clone https://github.com/Ekin-Kahraman/rustscenic
cd rustscenic
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy pandas pyarrow scipy
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
- `validation/ours/` — reproducible benchmark scripts + measurement docs. `*.md` files are the canonical record of "what this version measures".
- `examples/` — user-facing example scripts. End-to-end, self-contained.

## PR checklist

- [ ] `cargo test --workspace --exclude rustscenic-py --release` passes
- [ ] `cargo clippy --workspace --exclude rustscenic-py --all-targets -- -D warnings` passes
- [ ] `pytest tests/` passes
- [ ] If you changed an algorithm's output: new or updated `validation/ours/*_<date>.md` with before/after numbers
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

Scientific-correctness bugs (our output disagrees with pyscenic in a way that can't be explained by `docs/topic-collapse.md` or the AUCell tie-break note) are top priority.

## License

By contributing, you agree that your contributions will be licensed under the repository's MIT license.
