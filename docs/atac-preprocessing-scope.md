# ATAC preprocessing scope

Moha flagged on 2026-04-20 that the current rustscenic pipeline starts
at the cell × peak matrix. In a real scATAC workflow, the expensive,
install-heavy work happens *before* that point: parsing fragments,
calling peaks, building the matrix. Absorbing this into rustscenic
collapses most of the SCENIC+ ATAC install story into the same
GitHub wheel/source install used for the RNA stages.

This doc maps pycisTopic's preprocessing surface to what rustscenic
needs to implement, and flags which pieces are Rust-shaped and which
are better delegated.

## The pipeline pycisTopic provides today

Source: `aertslab/pycisTopic/src/pycisTopic/` on GitHub.

1. **Fragment I/O** — `fragments.py`
   - `read_fragments_to_polars_df` / `read_fragments_to_pyranges`:
     parse `fragments.tsv.gz` (10x cellranger output) into a
     polars DataFrame or a PyRanges object.
   - `read_barcodes_file_to_polars_series`: load cellranger
     `barcodes.tsv` for the cells-passing-filter list.
   - Dependencies pulled in: polars, pyarrow, pandas, pyranges.
2. **Fragment QC** — `fragments.py` + `qc.py`
   - `get_fragments_per_cb`: unique fragment counts per cell barcode.
   - `get_insert_size_distribution`: nucleosome signal (mono-/di-
     nucleosome periodicity) per cell.
   - `compute_qc_stats`: TSS enrichment, unique fragments,
     duplication rate, fraction-of-reads-in-peaks (FRiP).
   - `get_otsu_threshold`: auto-threshold QC metrics for filtering.
   - `get_cbs_passing_filter`: apply thresholds, return passing
     barcode set.
3. **Peak calling**
   - `pseudobulk_peak_calling.py`: groups barcodes by cluster label,
     shells out to **MACS2** per pseudobulk. Reintroduces a Python
     dependency (macs2 is a Python package).
   - `iterative_peak_calling.get_consensus_peaks`: Corces-2018 style
     consensus merging across pseudobulk peak sets.
4. **Cell × peak matrix** — `fragments.py`
   - `get_fragments_in_peaks`: inner-join fragments against consensus
     peaks, count per (cell, peak) pair. This is what rustscenic's
     Topics stage currently expects as input.

## What rustscenic currently takes as input

`rustscenic.topics.fit` accepts an `AnnData` with cells × peaks as its
`.X` (peaks in `.var_names`). `rustscenic.preproc` can now build that
matrix from `fragments.tsv[.gz]` plus a peak BED, compute FRiP / TSS /
insert-size QC, and call MACS2-free consensus peaks. The remaining
gap is real-data validation against MACS2 / ENCODE peak sets, not
missing API surface.

## Proposed rustscenic scope (two tiers)

### Tier 1 — Fragment I/O + QC + matrix construction (Rust-native) — shipped

Implement in Rust. These are I/O-and-counting workloads that Rust
flattens:

- **Fragment parser**: gzipped BED → in-memory fragment table. Use
  `flate2` + a hand-rolled BED splitter or `polars` Rust. No new
  Python deps; polars-rs is already usable from PyO3.
- **Per-cell fragment stats**: unique fragments, duplicates, insert
  size distribution. Pure counting over the fragment table.
- **TSS enrichment**: requires a gene annotation BED. Compute
  signal-over-background at TSS windows. Small lookup table + stream
  over fragments.
- **FRiP + cell × peak matrix**: interval-intersect fragments against
  a peak BED. Build sparse `cells × peaks` matrix directly. Needs a
  fast interval tree — `coitrees` (Rust crate, used by rust-bio) fits.
- **QC gate**: compute thresholds (Otsu or user-supplied), mark cells
  pass/fail, return filtered fragment table + barcode list.

**New Python deps:** none. Everything stays inside rustscenic's
existing four (numpy, pandas, pyarrow, scipy).

**New Rust deps:** `coitrees` (interval tree), possibly `bed-utils`.
`flate2` is already transitively available.

**Estimated effort:** 2 weeks. Roughly the same shape as
`rustscenic-aucell` was.

**Deliverable API:**

```python
import rustscenic.preproc

# Fragment QC + matrix in one shot
result = rustscenic.preproc.fragments_to_matrix(
    fragments_tsv_gz="path/to/fragments.tsv.gz",
    peaks_bed="path/to/consensus_peaks.bed",
    valid_barcodes="path/to/barcodes.tsv",  # optional
    tss_bed="path/to/tss.bed",             # for TSS enrichment
    qc=rustscenic.preproc.QCConfig(
        min_unique_fragments=1000,
        min_tss_enrichment=4.0,
        auto_threshold=True,
    ),
)
# result is an AnnData with shape (n_cells, n_peaks), sparse .X,
# QC metrics in .obs, peak info in .var.
```

### Tier 2 — Peak calling — shipped, reference cross-check pending

rustscenic now ships the self-contained Corces-2018-style
density-window / iterative consensus peak caller. It does not shell out
to MACS2. The next validation step is to benchmark its output against
MACS2 broadPeak files on real ENCODE / 10x multiome fragments.

## What stays out of scope

- Cellranger-equivalent alignment (BAM → fragments). That's the
  preprocessing *before* fragments. pycisTopic doesn't do it either.
- BAM I/O. Stick to the 10x fragments.tsv.gz contract, which is the
  standard output of every scATAC tool.
- Doublet detection for ATAC. Leave to scDblFinder / AMULET — not
  rustscenic's lane.

## Install story after this lands

Today:
```
pip install git+https://github.com/Ekin-Kahraman/rustscenic@v0.2.0
pip install pycisTopic          # breaks on modern Python
pip install macs2               # needs numpy + conda-ish
```

After v0.2:
```
pip install git+https://github.com/Ekin-Kahraman/rustscenic@v0.2.0
                                # covers 4 SCENIC+ compute stages
                                # + fragment I/O, QC, matrix build
                                # + MACS2-free consensus peak calling
```

After PyPI setup:
```
pip install rustscenic          # same package, once PyPI is live
```

## Dependencies by tier

| Tier | rustscenic Python deps | New Rust deps | External tools |
|---|---|---|---|
| Today | numpy, pandas, pyarrow, scipy | — | — |
| +T1 preproc | same | `coitrees` | — |
| +T2 peakcall | same | same + peak call | optional macs2 |

## Validation plan

For T1, benchmark against pycisTopic's outputs on a 10x multiome PBMC
dataset (same fragments, same peak set). Targets:

- **Byte-identical matrix** after joining on (cell, peak). Difference
  of zero is the pass bar.
- **QC metric agreement** to 1e-6 (float rounding only).
- **Wall-time**: aim for 3–5× faster than pycisTopic at 10k cells,
  10× at 100k.
- **Peak RSS**: at most 1.5× pycisTopic's. Going lower is a stretch.

## Open questions for Moha

1. Is 10x `fragments.tsv.gz` the only format you use, or do you also
   need snap/snap2 or custom BED formats as entry points?
2. Do you use MACS2 or iterative peak calling today?
3. Is pyranges the bottleneck in your workflow, or is it fragment
   loading, or peak calling?
4. Which genome annotations do you want TSS enrichment against —
   GENCODE, RefSeq, custom?

Answers to (1) and (2) decide whether Tier 2 is week-4 or month-6 work.
