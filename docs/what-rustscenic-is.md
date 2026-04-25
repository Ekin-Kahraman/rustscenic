# What rustscenic is, and what it isn't

A one-page honest summary for collaborators and ecosystem maintainers
deciding whether rustscenic is worth integrating with.

## What it is

A Rust + PyO3 reimplementation of the **SCENIC+** single-cell
regulatory-network pipeline. One `pip install`, no Java, no MACS2, no
dask, no CUDA. Replaces:

- `arboreto` / `pyscenic.grn` (GRNBoost2 inference)
- `pyscenic.aucell` (per-cell regulon scoring)
- `pycisTopic` (LDA topic models on scATAC)
- `pycistarget` (motif enrichment AUC kernel)
- `scenicplus` (eRegulon assembly)
- Plus full ATAC preprocessing — fragments → matrix, MACS2-free
  iterative consensus peak calling (Corces 2018), per-cell QC (FRiP,
  TSS enrichment, insert-size).

Ships as one wheel, abi3, Python 3.10–3.13, Linux + macOS (x86_64 +
aarch64). Four runtime deps: numpy, pandas, pyarrow, scipy.

## What it isn't

- **Not an upstream tool**: starts at the AnnData / fragments stage,
  not at FASTQ. Rob Patro's [alevin-fry](https://github.com/COMBINE-lab/alevin-fry)
  fills that layer; rustscenic is downstream of it.
- **Not a DEG tool**: differential expression is out of scope. scverse's
  JAX DEG efforts are the right place for that.
- **Not a clustering / dimensionality-reduction tool**: scanpy still
  owns that. We assume your AnnData is already log-normalised + clustered.
- **Not a pyscenic API clone at the syntax level**: the function names
  are similar but signatures are explicit, not auto-magical.

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

## Honest caveats

Where the implementation is weaker than the reference, or where we
haven't validated yet:

1. **Topic modelling at K ≥ 30 on scATAC**: our Online VB LDA
   collapses aggressively (5/30 unique topics vs Mallet's 24/30 on
   PBMC 10k × 67k peaks). NPMI coherence 0.123 vs 0.196. Same
   collapse pattern as gensim's VB. Mallet still wins for topic-count
   diversity. Documented in `docs/topic-collapse.md`. v0.3 candidate
   is a collapsed-Gibbs rewrite; until then we recommend falling back
   to Mallet for K ≥ 30 on scATAC scale.
2. **GRN per-edge agreement with arboreto** is 0.58 Spearman, not
   1.0. Coarse biology agrees (94% known TF→target edges recovered,
   8/8 lineage TFs correctly enriched), and downstream AUCell is
   0.99 per-cell Pearson with pyscenic — so fine-edge disagreement
   doesn't propagate to regulon activity. But if you're publishing
   per-edge effect sizes against an arboreto baseline, we won't
   match.
3. **MACS2 cross-check pending**: peak calling matches Corces 2018
   density-window / iterative-overlap-rejection, validated on
   synthetic recovery. We have not yet benchmarked against MACS2 on
   real ENCODE data. F1 vs MACS2 broadPeak is on the v0.3 list.
4. **100k-cell atlas end-to-end is unmeasured** for the full
   ATAC + RNA pipeline. We have GRN scaling proof to 50k (linear
   slope post-PR #12) and peak RSS at 100k (6.3 GB across 4 stages),
   but the integrated multi-modal pipeline at 100k cells is the next
   credibility test.
5. **Windows build**: untested. macOS + Linux only.
6. **PyPI publish blocked**: trusted-publisher needs config on the
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
