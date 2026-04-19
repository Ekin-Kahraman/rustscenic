# Security policy

## Supported versions

Security fixes ship on the latest 0.x release. Older 0.x versions are not patched — update.

## Reporting a vulnerability

rustscenic processes numeric matrices and gene-name strings. The attack surface is small but non-zero:

- Parsing user-supplied h5ad / parquet / feather files
- PyO3 boundary between Python and Rust
- The `rustscenic` CLI taking path arguments from user input

If you find a vulnerability — memory unsafety, arbitrary file write, crash on crafted input, supply-chain concern — please **do not** open a public issue.

Email: ekinkhrmn2005@outlook.com
Subject: `rustscenic security: <short description>`

Include:
- Version affected (`pip show rustscenic`)
- Reproduction steps
- Impact (crash, leak, RCE, etc.)
- Your GitHub handle if you want credit in the release notes

Expect a response within 7 days. Fix timeline depends on severity; typical:
- Critical (RCE, persistent data corruption) — 48 hours
- High (crash-as-DoS, wrong output for specific crafted input) — 7 days
- Moderate/low — next regular release

## Non-security issues

Correctness bugs, performance regressions, usability issues — open a public issue on GitHub instead.

## Supply chain

- Rust dependencies are pinned in `Cargo.lock`. Run `cargo audit` in CI (to be added).
- Python dependencies are loose (`numpy>=1.21`, etc.) to match the scientific-Python ecosystem; pin your own environment if you need strict reproducibility.
- Wheels are built via GitHub Actions with trusted-publisher OIDC to PyPI — no API tokens stored in the repo.

## Signing + SBOM

- maturin emits a CycloneDX SBOM with every wheel (`dist-info/sboms/`).
- No GPG signing yet. PyPI's attestations (sigstore) are the verification path once available for rustscenic.
