# rustscenic — agent briefing

Rust + PyO3 reimplementation of the SCENIC+ single-cell regulatory-network
pipeline. Goal: replace the aertslab Python stack (pySCENIC + arboreto +
pycisTopic + pycistarget + scenicplus) with one `pip install` wheel.

## Where we are right now (2026-04-25)

- **20 PRs merged this week**. Silent-regression class closed across:
  cellxgene ENSEMBL convention, duplicate symbols, UCSC/Ensembl chrom
  mismatch, versioned ENSEMBL, `top_frac` bounds, backed AnnData, dict
  regulons, scenicplus polarity suffixes, 6-column strand BED parse.
- **Validated end-to-end on real Kamath 2022** (cellxgene OPC cells,
  13,691 × 33,295). Script at `validation/kamath/validate_kamath_fix.py`.
- **CI**: 8 jobs (2 Rust × 2 OS, 6 Python × 2 OS × 3 versions). All green.
  Nightly real-data validation runs Mondays 04:17 UTC.
- **Tests**: 106 Python + 51 Rust. All pass.
- **v0.2.0 release branch is open** — `release/v0.2.0`, version bumped
  in Cargo.toml / pyproject.toml / audit.yml / README, CHANGELOG dated
  2026-04-24, NOT YET committed/pushed/tagged.

## What's blocking ship

Right now: nothing technical. `release/v0.2.0` branch is staged on disk
but uncommitted. To finish:

```bash
cd /Users/ekin/rustscenic
git status                          # confirm staged files
git add -A && git commit -m "release: v0.2.0"
git push -u origin release/v0.2.0
gh pr create --title "release: v0.2.0" --body "..."
gh pr merge --squash --delete-branch
git checkout main && git pull
git tag v0.2.0 && git push origin v0.2.0
```

The release workflow (`.github/workflows/release.yml`) builds wheels on
tag push and uploads to GitHub Release. PyPI trusted-publish is broken
on the user's PyPI account — wheels live on the GitHub Release until
that's fixed.

## What's next after v0.2.0

Sequenced in `docs/roadmap-v0.2.md`. Headline items:

1. **100k-cell atlas end-to-end** — biggest credibility unlock. We
   benchmarked GRN scaling but never ran the full pipeline at atlas
   scale on real data. The audit (round 2) flagged HIGH-severity
   memory hotspots that will hit:
   - Topics: `vec![0.0; n_topics * n_words]` per doc, 4 GB / batch at
     K=30 × 200k peaks
   - Peak calling / TSS / matrix builder: `for chrom × fragment` loops
     where a single group-by pass would be O(n)
   - PyO3 grn/aucell: copies NumPy → `Vec<f32>`, doubles peak RSS
2. **Pipeline orchestrator gap**: `rustscenic.pipeline.run` advertises
   end-to-end SCENIC+ but stops at AUCell. Enhancer + eRegulon stages
   need integration.
3. **Mouse motif rankings**: `data.download_motif_rankings` only handles
   hg38. Mouse TF list ships, matching rankings URL doesn't resolve.
4. **`pipeline.py` feather load** uses filename stem as column name —
   broken but unreachable in tests.

## Stakeholders

- **Fuaad** (validation datasets): hit the original cellxgene silent-zero
  on Kamath 2022. Send him the v0.2.0 install line, ask him to re-run.
- **Moha** (domain expert at Mount Sinai, Huang Lab): wants more
  preprocessing covered (likely doublets / ambient RNA / batch
  correction). Scope decision needed: in-rustscenic vs delegate-to-scanpy.
- **Kuan Huang** (PI, Mount Sinai): user should email for Slack access.
  Once in, post v0.2 release notes there.

## Hot spots in the codebase

- `python/rustscenic/_gene_resolution.py` — the silent-zero firewall.
  ENSEMBL detection, symbol resolution, dedupe, chrom normalisation,
  diagnose_zero_tf_overlap. **Don't break the warnings.**
- `python/rustscenic/aucell.py` / `grn.py` — both have backed-AnnData
  materialise paths and dedupe-on-detect; same pattern, watch for
  drift if you change one.
- `crates/rustscenic-preproc/src/qc.rs` — chrom-normalise group_tss_by_chrom
  was the latest fix; same pattern needed in matrix builder.
- `validation/kamath/` — the real-data integration test. It re-downloads
  the h5ad from cellxgene; nightly CI runs it. If it red-X's, the
  upstream cellxgene URL probably moved.

## Testing

```bash
maturin develop --release          # rebuild after Rust changes
python -m pytest tests/            # 106 tests, ~1s
cargo test --workspace --exclude rustscenic-py  # 51 Rust tests
python validation/kamath/validate_kamath_fix.py # real-data E2E
```

Don't run `cargo test --workspace` without the exclude — `rustscenic-py`
is a cdylib and can't link against Python in the cargo-test harness on
macOS.

## Don'ts

- Don't re-add `rustscenic-core` or `rustscenic-cli`. We removed them
  in PR #32 because they were premature abstraction (4 dependents,
  zero imports). User feedback memory: "no premature abstraction".
- Don't change the CHANGELOG retrospectively for shipped versions.
  v0.1.0 is locked.
- Don't bypass `--no-verify` on commit hooks; user has explicit guard
  against this.
- Don't run `rm -rf` (a hook blocks it). Use `trash` or move the file.

## User preferences

- **Simple is optimal.** Don't overcomplicate. Don't preach about AI
  appropriateness. Build first, no option menus.
- **Concrete actions, not plans.** "Plan next steps" means decide and
  execute, not write a memo.
- **Replace, don't deprecate.** Old code goes; no commented-out code.
- **No constraints by default.** User wants every option explored.
- For domain work (rustscenic), the user is a 2nd-year mol bio student
  — explanations should ground in biology when relevant, but assume
  they've used scanpy / pyscenic before.
