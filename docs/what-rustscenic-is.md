# What rustscenic is

A one-page summary for collaborators and ecosystem maintainers deciding
whether rustscenic is worth integrating with.

## Thesis

rustscenic is a bet that the SCENIC / SCENIC+ workflow should not require
users to stitch together old Python, Java/Mallet, dask, MACS2 wrappers,
pycisTopic, pycistarget, scenicplus, and fragile environment pins before
they can ask a regulatory-biology question.

The intended endpoint is a **single-install CPU package** that covers the
full practical workflow from AnnData / fragments through GRN, regulon
activity, motif enrichment, ATAC preprocessing, enhancer-gene linking, and
eRegulon assembly. The reason to build it in Rust is not novelty; it is
memory predictability, deterministic execution, portable wheels, and
removing entire classes of silent failure that show up in real atlas data.

## What it is

A Rust + PyO3 replacement for the slow CPU stages in the **SCENIC /
SCENIC+** single-cell regulatory-network workflow. One GitHub
wheel/source install today, no Java, no dask, no CUDA. PyPI is pending
trusted-publisher setup. Replaces or covers:

- `arboreto` / `pyscenic.grn` (GRNBoost2 inference)
- `pyscenic.aucell` (per-cell regulon scoring)
- `pycisTopic` (LDA topic models on scATAC)
- `pycistarget` (motif enrichment AUC kernel)
- `scenicplus` eRegulon assembly mechanics
- Plus full ATAC preprocessing — fragments → matrix, MACS2-free
  iterative consensus peak calling (Corces 2018), per-cell QC (FRiP,
  TSS enrichment, insert-size).

Ships as one abi3 wheel for Python 3.10–3.13, Linux + macOS (x86_64 +
aarch64), plus source install. Four runtime deps: numpy, pandas,
pyarrow, scipy.

## Boundary

- **Not an upstream tool**: starts at the AnnData / fragments stage,
  not at FASTQ. Rob Patro's [alevin-fry](https://github.com/COMBINE-lab/alevin-fry)
  fills that layer; rustscenic is downstream of it.
- **Not a DEG tool**: differential expression is out of scope. scverse's
  JAX DEG efforts are the right place for that.
- **Not a clustering / dimensionality-reduction tool**: scanpy still
  owns that. We assume your AnnData is already log-normalised + clustered.
- **Not a pyscenic API clone at the syntax level**: the function names
  are similar but signatures are explicit, not auto-magical.

The boundary is deliberate: rustscenic should own the regulatory-network
compute path, not every surrounding single-cell method.

## Performance vs the references it replaces

Measured on this v0.2.0 release. Full numbers in `CHANGELOG.md` /
`validation/`.

| Stage | Reference | rustscenic |
|---|---|---|
| AUCell (10x Multiome 10k cells × 1,457 regulons) | 18.6 s pyscenic | 0.21 s (88×) |
| Cistarget AUC kernel (5,876 motifs × 27,015 genes) | reference | Pearson 1.0000 |
| GRN (10x Multiome) | per-edge Spearman 0.58 vs arboreto | biology-recovered: 94% known TF→target edges |
| End-to-end (10x Multiome 3k, 4 stages) | 11.8 min ref pipeline | 9.1 min |
| Peak RSS (100k cells × 20k genes, 4 stages) | > 40 GB reported | 6.3 GB |

Bit-identical output under same seed. 51 Rust tests + 106 Python tests.

## Intellectual Risk

The ambition is full replacement. The risk is not whether we can make a
demo pass; that is already done. The hard questions are:

1. **Can region-level cistarget/cistromes match scenicplus closely enough
   on real region-ranking databases?** The exact region-ranking path is
   now wired into eRegulon assembly. The remaining risk is reference
   parity on real scenicplus workloads, not missing plumbing.
2. **Can ATAC topics beat or match Mallet where users actually care: K≥30,
   sparse binary peak matrices, stable topic identities, and good
   coherence?** Online VB currently recovers cell-type structure but
   collapses fine-grained topics. This is the largest algorithmic gap.
3. **Can full-TF GRN stay fast enough at 100k+ cells without making users
   rent a large node?** The low-memory story is strong; the next target is
   full biological settings, not toy TF panels.
4. **Can peak calling agree with MACS2 / ENCODE enough for users to trust
   a no-MACS2 path?** The implementation is self-contained and passes
   synthetic/self-consistency tests; reference F1 is the missing proof.

These are engineering and algorithmic risks, not reasons to shrink the
vision. They define the v0.3/v0.4 work.

## Current Evidence

Where the implementation is weaker than the reference, or where we
haven't validated yet:

1. **Topic modelling at K ≥ 30 on scATAC** has two paths:
   `topics.fit` (Online VB LDA) collapses aggressively at K ≥ 30 on
   sparse scATAC (5/30 unique topics on PBMC 10k × 67k peaks; 2/30
   on PBMC 1500 × 98k peaks). Use `topics.fit_gibbs` (collapsed
   Gibbs, shipped v0.3.1) for fine topic decomposition: 22/30 unique
   topics on the same 1500 × 98k benchmark, mean pairwise top-20
   peak overlap 0.005 vs VB's 0.373, intrinsic top-10 NPMI +0.031
   vs VB's +0.012 (~2.7× higher coherence), at only 1.2× the
   serial wall-clock. AD-LDA parallel path (`n_threads=N`,
   shipped) gives 2.56× speedup at 8 threads on real PBMC ATAC
   with quality preserved (NPMI within sampling variance). Same
   algorithm class as Mallet, no Java required. Outstanding for
   v0.4: extrinsic NPMI head-to-head against an actual Mallet run
   on the same corpus.
2. **SCENIC+ eRegulons need real-reference parity numbers next**:
   enhancer→gene linking, region cistarget, and the assembly schema are
   tested end-to-end. The next proof point is a direct scenicplus /
   aertslab-region-ranking comparison on real multiome data.
3. **GRN per-edge agreement with arboreto** is 0.58 Spearman, not
   1.0. Coarse biology agrees (94% known TF→target edges recovered,
   8/8 lineage TFs correctly enriched), and downstream AUCell is
   0.99 per-cell Pearson with pyscenic — so fine-edge disagreement
   doesn't propagate to regulon activity. But if you're publishing
   per-edge effect sizes against an arboreto baseline, we won't
   match.
4. **MACS2 cross-check pending**: peak calling matches Corces 2018
   density-window / iterative-overlap-rejection, validated on
   synthetic recovery. We have not yet benchmarked against MACS2 on
   real ENCODE data. F1 vs MACS2 broadPeak is on the v0.3 list.
5. **100k-cell atlas end-to-end** is now measured (synthetic). Full
   7-stage pipeline (topics → GRN → regulons → cistarget → enhancer
   → eRegulon → AUCell) on 100,000 × 15,000 RNA + 100,000 × 50,000
   ATAC: **762 s (12.7 min) total wall-clock at 7.09 GB peak RSS** —
   see `validation/scaling/bench_e2e_100k_synthetic.py`. Reference
   scenicplus stack reports > 40 GB at comparable scale, so the
   memory delta is **~5.6×**. Real 100k+ multiome E2E (not synthetic)
   is the next step — synthetic demonstrates plumbing + scaling, not
   biology. The earlier 91k microglia GRN cliff was fixed by target
   blocking + worker-local scratch (5k→91.8k slope: 1.81 → 1.15).
6. **Windows build**: untested. macOS + Linux only.
7. **PyPI publish blocked**: trusted-publisher needs config on the
   maintainer's PyPI account. Until that resolves, install via
   GitHub Release wheels or `pip install git+...`.

## Robustness work

The class of bug that hits real users is "silent zero" — output
finishes without error but is structurally empty (e.g. AUCell scoring
to all zeros because regulons reference HGNC symbols but
cellxgene-curated `var_names` are ENSEMBL IDs). v0.2.0 closed 30+ of
these:

- ENSEMBL `var_names` → `feature_name` auto-swap
- Duplicate symbols auto-summed (scanpy / limma `avereps` convention)
- UCSC `chr1` vs Ensembl `1` chrom normalisation across peak calling,
  FRiP, TSS, enhancer→gene
- Versioned ENSEMBL `.N` auto-strip
- Backed AnnData materialisation
- Dict regulons supported
- scenicplus `TF(+)/(-)`, `TF_extended`, `TF_activator/repressor`
  polarity stripping
- 6-column strand BED parse detection
- `top_frac` bounds + saturation warning
- > 8 GiB densification warning
- > 50% TF-drop warning in eRegulon assembly
- Actionable zero-overlap diagnostics that name the specific
  convention mismatch (case, ENSEMBL/symbol, versioned)

Validated end-to-end on real Kamath 2022 (cellxgene OPC cells,
13,691 × 33,295). Nightly CI runs the full validation against the
live cellxgene dataset URL each Monday.

## What we're asking ecosystem partners

If you maintain a single-cell tool that overlaps with rustscenic's
scope:

- **muon / SnapATAC-2**: would you accept rustscenic as a Rust perf
  backend behind muon's ATAC functions? We match anndata conventions
  by design.
- **scenicplus**: would you accept a co-authored note positioning
  rustscenic as the speed-and-memory drop-in for the slow stages?
- **Anyone else**: what dataset shape have you seen that we haven't
  tested? Send a slice; if it breaks, we want it to break in CI.

Repo: <https://github.com/Ekin-Kahraman/rustscenic>
Latest: v0.2.0 (2026-04-25)
