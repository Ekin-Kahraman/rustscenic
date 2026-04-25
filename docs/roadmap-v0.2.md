# Roadmap after v0.2.0

Prioritised from the two-round forward audit (docs/independent-debate.md
+ the v2 audit findings). Ordered by value-per-effort.

## Ship v0.2.0 (this sprint)

The "close the audit" release. No new features; just the robustness
work of the last two days made official.

- [x] All five PRs (#26-#31) landed, audit gaps closed
- [x] CHANGELOG updated with the 30+ silent-regression guards added
- [x] Dead code removed (`rustscenic-cli`, `rustscenic-core`)
- [x] Type stubs complete
- [x] Kamath real-data validation scripted
- [x] Nightly CI job in place
- [x] **Version bump**: atomic `0.1.0 → 0.2.0` in
      `Cargo.toml` / `pyproject.toml` / `README.md:12` / `audit.yml:57`
- [x] **Release tag** + GitHub Release with wheels
- [ ] **PyPI publish** (blocked on user-side: PyPI trusted-publisher
      config. Fallback: GitHub Release wheels + `pip install git+...`)

## Near-term (next 1-2 weeks)

Goal: kill the remaining HIGH-severity scaling gaps so "full SCENIC+
at atlas scale" is demonstrated, not claimed.

- [ ] **100k-cell end-to-end run** on a real multiome atlas. We have
      the 10x public PBMC 10k Multiome already; the equivalent 100k
      scale test is needed before v0.2 claims are defensible.
- [ ] **Topics memory**: currently allocates `vec![0.0; n_topics * n_words]`
      per doc in the batch, so 256 docs × K=30 × 200k peaks = 40 GB
      intermediate. Rewrite the batch loop to stream instead of collect.
      Unlocks actually-large ATAC pipelines.
- [ ] **Peak calling / TSS / matrix builder**: invert the
      `for each chrom × fragment` pattern to a single group-by pass.
      Current O(n_clusters × n_chroms × n_fragments) is measurable at
      atlas scale.
- [ ] **PyO3 buffer borrow**: `grn_infer` and `aucell_score` copy the
      NumPy input into a `Vec<f32>`. Use `PyArray.as_array()` to borrow
      instead — halves peak RSS on large matrices.
- [ ] **Missing tests**: `pipeline.run`, `data.tfs`, `data.download_motif_rankings`,
      `quickstart.main`. Each is advertised, each is untested.

## Medium-term (weeks 3-4)

Goal: deliver the "one call, full pipeline" promise for users who
don't want to assemble stages themselves.

- [ ] **Integrate enhancer + eRegulon into `pipeline.run`**. Currently
      the orchestrator stops at AUCell; users have to chain the
      SCENIC+-distinguishing stages by hand.
- [ ] **Tutorial notebook**: migration from pySCENIC. "I was doing X
      in pyscenic; here's the rustscenic equivalent." Drop-in for
      every stage, with a diff showing results agree.
- [ ] **Mouse motif rankings download path**: `data.download_motif_rankings`
      only handles hg38 gene-based + region-based. Add mm10 entries.
- [ ] **MACS2 cross-check**: real ENCODE PBMC multiome, our peaks
      vs MACS2 broadPeak at 50 bp IoU tolerance. Validates the
      MACS2-free claim for the README.

## Longer-term (month+)

- [ ] **Collapsed Gibbs topic model** — replaces the Online VB LDA
      that collapses aggressively on sparse binary scATAC
      (docs/topic-collapse.md). v0.2 candidate; only ship if we can
      match Mallet NPMI.
- [ ] **Windows build**: maturin likely already builds there; nobody
      has confirmed. Would extend reachable-user set meaningfully.
- [ ] **Rust `rand 0.8 → 0.9`** migration (workspace-wide). `.gen()`
      renamed to `.random()`. Not urgent but blocks Rust 1.90 edition
      flag day.
- [ ] **Seurat interop beyond the docs**: actually pipe a Seurat object
      through a 10-line wrapper. `docs/seurat-interop.md` exists but
      no exercised path.
- [ ] **Manuscript preprint** — `manuscript/rustscenic_preprint.md`
      exists; decide target venue (Bioinformatics? Nature Methods tools?)
      once 100k-scale run is in hand.

## Stakeholder-dependent

- [ ] **Moha's preprocessing bottlenecks**: he wanted more pre-matrix
      steps covered. Find out which specific steps (likely doublet
      detection, ambient RNA, batch correction) and scope which are
      in-scope for rustscenic vs "use scanpy/scVI".
- [ ] **Fuaad's next dataset pick**: he's curating validation datasets
      for Moha. When he names the next one, set up a reproducible
      validation script mirroring `validation/kamath/`.
- [ ] **Huang Lab slack**: Moha said to email Kuan for slack access.
      Once in, post v0.2 release notes and weekly progress.

## What NOT to do

- Don't expand API surface before v0.2 ships. Robustness first.
- Don't re-architect `rustscenic-core` after removing it — we had
  zero imports from it, and re-adding pre-emptively is the premature
  abstraction the user explicitly rejected.
- Don't port `link_peaks_to_genes` to Rust yet. The Python version is
  fast enough at Kamath OPC scale (2.8s for 100 regulons); Rust
  rewrite pays off only at atlas scale, which we haven't benchmarked.
- Don't chase PyPI publish before v0.2 actually works at atlas scale.
  A broken `pip install rustscenic` from PyPI is worse than a working
  `pip install git+...`.

## Today (2026-04-25)

v0.2.0 is tagged and GitHub Release wheels are published. Silent-zero
class is closed on the datasets we tested. Next concrete action:
prove a real 100k-cell multiome end-to-end run and reduce full-TF GRN
runtime for collaborator-facing workflows.
