"""Enhancer-to-gene linking — the SCENIC+ distinguishing step.

pySCENIC scores regulons as "TF → its top-N co-expressed target genes".
SCENIC+ upgrades this by **grounding regulation in chromatin**:

  TF → enhancer (motif enrichment, from cistarget)
        ↓
     enhancer → target gene (peak accessibility ↔ gene expression
                             correlated across matched cells)

The enhancer-to-gene edge is what this module produces. Combined with
the motif-to-enhancer and TF-to-target edges rustscenic already
computes, it's the raw material for eRegulons — the chromatin-aware
regulons that are scenicplus's distinguishing output.

This module requires **matched cells**: every cell in ``rna_adata.obs_names``
must also be in ``atac_adata.obs_names``. Multiome data (10x Multiome
or nf-core/multiome) is the natural input. For non-matched samples,
CCA or Harmony-style integration needs to happen upstream.

Complexity:
  O(n_peaks × max_genes_in_window × n_cells) where max_genes_in_window
  is typically 20–50. On 100k peaks × 30 candidate genes × 50k cells
  that's ~150 GFLOPs — a few seconds in numpy on a single core.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


def link_peaks_to_genes(
    rna_adata,
    atac_adata,
    gene_coords: pd.DataFrame,
    peak_coords: Optional[pd.DataFrame] = None,
    *,
    max_distance: int = 500_000,
    min_abs_corr: float = 0.1,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """Link ATAC peaks to nearby gene expression.

    For each peak, finds candidate target genes whose TSS is within
    ``max_distance`` bp of the peak center, computes the correlation of
    peak accessibility vs gene expression across the matched cells, and
    returns links whose absolute correlation exceeds ``min_abs_corr``.

    Parameters
    ----------
    rna_adata
        ``AnnData`` with log-normalised expression. Cells in
        ``obs_names`` must match ``atac_adata.obs_names``.
    atac_adata
        ``AnnData`` of cells × peaks accessibility (typically from
        ``rustscenic.preproc.fragments_to_matrix``). Peak coordinates
        may come from ``atac_adata.var`` (columns ``chrom``, ``start``,
        ``end``) or from the explicit ``peak_coords`` argument.
    gene_coords
        ``DataFrame`` with columns ``["gene", "chrom", "tss"]`` listing
        each gene's TSS position. Build from a GTF, a GENCODE BED, or
        the ``rustscenic.data`` helpers once they ship.
    peak_coords
        Optional override; ``DataFrame`` indexed by peak id with
        ``chrom``, ``start``, ``end``. Required if ``atac_adata.var``
        doesn't carry peak coordinates.
    max_distance
        Maximum bp between a peak's centre and a gene's TSS for the
        pair to be considered (default 500 kb, matching pycisTopic).
    min_abs_corr
        Minimum absolute correlation to emit a link (default 0.1).
        Increase for stricter filtering.
    method
        ``"pearson"`` (default, matches scenicplus) or ``"spearman"``.

    Returns
    -------
    pandas.DataFrame
        One row per surviving peak-gene link with columns:
        ``['peak_id', 'peak_chrom', 'peak_start', 'peak_end', 'gene',
        'gene_tss', 'distance', 'correlation']``. Sorted by
        descending ``|correlation|``.
    """
    rna_adata, atac_adata = _align_cells(rna_adata, atac_adata)
    from rustscenic._gene_resolution import resolve_gene_names

    gene_names_rna = resolve_gene_names(rna_adata, quiet=True)
    peaks = _peak_frame(atac_adata, peak_coords)
    genes = _validate_gene_coords(gene_coords)

    # Build gene_name → row_index lookup in RNA.
    gene_rna_idx = {g: i for i, g in enumerate(gene_names_rna)}
    genes_in_rna = genes[genes["gene"].isin(gene_rna_idx)].reset_index(drop=True)
    if genes_in_rna.empty:
        raise ValueError(
            "no gene_coords genes match any gene name in rna_adata — "
            "check species + symbol convention"
        )

    # Normalise chrom names on both sides so "chr1"/"1" join correctly
    # regardless of whether peak BED and gene_coords use the same
    # convention (they often don't).
    genes_in_rna = genes_in_rna.assign(
        _chrom_norm=genes_in_rna["chrom"].map(_normalise_chrom)
    )
    peaks = peaks.assign(_chrom_norm=peaks["chrom"].map(_normalise_chrom))

    # For fast lookup: group genes by normalised chrom, sort by tss.
    gene_by_chrom = {
        chrom: sub.sort_values("tss").reset_index(drop=True)
        for chrom, sub in genes_in_rna.groupby("_chrom_norm")
    }

    # Guard against the silent-zero mode: peaks on chroms the gene_coords
    # never name (or vice versa). After normalisation, the intersection
    # should be non-empty; if it is, the pipeline will produce zero links.
    overlap = set(gene_by_chrom) & set(peaks["_chrom_norm"].unique())
    if not overlap:
        import warnings
        warnings.warn(
            f"no chromosome name overlaps between peaks and gene_coords "
            f"even after UCSC/Ensembl normalisation. Peak chroms: "
            f"{sorted(set(peaks['_chrom_norm'].unique()))[:5]}; "
            f"gene_coords chroms: {sorted(gene_by_chrom)[:5]}. The "
            f"resulting link DataFrame will be empty.",
            UserWarning, stacklevel=3,
        )

    _warn_if_densification_expensive(rna_adata, atac_adata)
    rna_X = _densify(rna_adata.X).astype(np.float32, copy=False)  # (n_cells, n_genes_rna)
    atac_X = _densify(atac_adata.X).astype(np.float32, copy=False)  # (n_cells, n_peaks)

    corr_fn = _spearman_matrix if method == "spearman" else _pearson_matrix

    rows = []
    # Iterate peaks in chrom-order; batch peaks on the same chromosome
    # so we amortise the sorted-gene-TSS work and compute the correlation
    # over the slice of candidate genes in one vectorised call.
    peak_centers = ((peaks["start"].values + peaks["end"].values) // 2).astype(np.int64)
    peak_chroms_norm = peaks["_chrom_norm"].values
    peak_chroms = peaks["chrom"].values
    peak_starts = peaks["start"].values
    peak_ends = peaks["end"].values
    peak_ids = list(peaks.index)
    for chrom, gg in gene_by_chrom.items():
        # Match peaks by normalised chrom so "chr1" peaks join to "1" genes
        peak_positions = np.where(peak_chroms_norm == chrom)[0]
        if peak_positions.size == 0:
            continue
        tss = gg["tss"].values.astype(np.int64)
        gene_col_idx = np.array(
            [gene_rna_idx[g] for g in gg["gene"].values], dtype=np.int64
        )

        for i in peak_positions:
            centre = peak_centers[i]
            lo = np.searchsorted(tss, centre - max_distance)
            hi = np.searchsorted(tss, centre + max_distance + 1)
            if lo == hi:
                continue
            candidate_rna_cols = gene_col_idx[lo:hi]
            candidate_tss = tss[lo:hi]

            peak_vec = atac_X[:, i]
            rna_block = rna_X[:, candidate_rna_cols]

            corr = corr_fn(peak_vec, rna_block)
            keep = np.abs(corr) >= min_abs_corr
            if not keep.any():
                continue
            kept_genes = gg["gene"].values[lo:hi][keep]
            kept_tss = candidate_tss[keep]
            kept_corr = corr[keep]
            for g, t, c in zip(kept_genes, kept_tss, kept_corr):
                rows.append(
                    (
                        peak_ids[i],
                        chrom,
                        int(peak_starts[i]),
                        int(peak_ends[i]),
                        g,
                        int(t),
                        int(centre - t),
                        float(c),
                    )
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "peak_id", "peak_chrom", "peak_start", "peak_end",
            "gene", "gene_tss", "distance", "correlation",
        ],
    )
    df = df.sort_values("correlation", key=lambda s: -s.abs()).reset_index(drop=True)
    return df


def _align_cells(rna_adata, atac_adata):
    common = rna_adata.obs_names.intersection(atac_adata.obs_names)
    if len(common) == 0:
        raise ValueError(
            "rna_adata and atac_adata share no cell barcodes — this function "
            "requires matched multiome data. For separate scRNA + scATAC "
            "samples, integrate via CCA or scVI before calling."
        )
    if len(common) < len(rna_adata) or len(common) < len(atac_adata):
        import warnings
        warnings.warn(
            f"keeping {len(common)} cells with barcodes in both RNA "
            f"({rna_adata.n_obs}) and ATAC ({atac_adata.n_obs}) AnnDatas.",
            UserWarning, stacklevel=3,
        )
    return rna_adata[list(common)].copy(), atac_adata[list(common)].copy()


def _peak_frame(atac_adata, peak_coords) -> pd.DataFrame:
    if peak_coords is not None:
        need = {"chrom", "start", "end"}
        missing = need - set(peak_coords.columns)
        if missing:
            raise ValueError(f"peak_coords missing columns: {missing}")
        return peak_coords.loc[atac_adata.var_names]
    # Try to get from atac_adata.var
    var = atac_adata.var
    if {"chrom", "start", "end"}.issubset(var.columns):
        return var[["chrom", "start", "end"]].copy()
    # Parse "chrN:start-end" pattern from var_names
    parsed = _parse_peak_names(list(atac_adata.var_names))
    if parsed is None:
        raise ValueError(
            "atac_adata has no peak coordinates — either include chrom/start/end "
            "columns in atac_adata.var, pass peak_coords explicitly, or use "
            "`chr:start-end` formatted var_names."
        )
    parsed.index = atac_adata.var_names
    return parsed


def _parse_peak_names(names):
    """Parse peak names like ``chr1:100-200``, ``1:100-200``, ``chr1-100-200``.

    Accepts both UCSC (``chr1``) and Ensembl (``1``) chromosome conventions
    because real-world peak BEDs use either. The full chrom token is
    preserved as-is (including any ``chr`` prefix) so downstream callers
    that joined on ``chr1`` keep working; chromosome-name normalisation
    for cross-convention joining is done in ``_normalise_chrom`` at join
    time rather than here.
    """
    import re
    pat = re.compile(
        r"^((?:chr)?[\dXYMTIivxmt]+)[:\-_](\d+)[\-_](\d+)$",
        re.IGNORECASE,
    )
    rows = []
    for n in names:
        m = pat.match(str(n))
        if m is None:
            return None
        rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def _normalise_chrom(name: str) -> str:
    """Strip ``chr`` prefix and collapse mito aliases so UCSC and Ensembl
    chrom names match. Mirrors the Rust-side ``normalise_chrom``.
    """
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()
    if len(s) >= 3 and s[:3].lower() == "chr":
        s = s[3:]
    s = s.upper()
    return "MT" if s == "M" else s


def _validate_gene_coords(gene_coords: pd.DataFrame) -> pd.DataFrame:
    need = {"gene", "chrom", "tss"}
    missing = need - set(gene_coords.columns)
    if missing:
        raise ValueError(
            f"gene_coords missing columns: {missing}. Expected columns "
            f"['gene', 'chrom', 'tss'] — use a GTF → DataFrame converter "
            f"like gtfparse, or extract from biomart / GENCODE."
        )
    out = gene_coords[["gene", "chrom", "tss"]].copy()
    out["tss"] = out["tss"].astype(np.int64)
    return out


def _densify(X):
    import scipy.sparse as sp
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


# Warn the user before `link_peaks_to_genes` densifies a matrix larger
# than this (bytes, per-matrix). Module-level so tests can patch down.
_DENSIFY_WARN_BYTES = 8 * 1024**3  # 8 GiB


def _warn_if_densification_expensive(rna_adata, atac_adata) -> None:
    """Warn before materialising large float32 matrices from sparse input.

    `link_peaks_to_genes` builds dense RNA and ATAC matrices for the
    correlation loop. On 100k cells × 100k peaks that's 40 GB — a
    common surprise for users who never saw the sparse-to-dense step.
    Raise a warning so a user can interrupt before their laptop swaps
    itself to death.
    """
    import warnings

    for name, ad_obj in [("rna_adata", rna_adata), ("atac_adata", atac_adata)]:
        bytes_needed = ad_obj.n_obs * ad_obj.n_vars * 4  # float32
        if bytes_needed >= _DENSIFY_WARN_BYTES:
            gib = bytes_needed / 1024**3
            warnings.warn(
                f"link_peaks_to_genes will densify {name} to a "
                f"{ad_obj.n_obs} × {ad_obj.n_vars} float32 matrix "
                f"(~{gib:.1f} GiB). If this OOMs, subset to highly "
                f"variable peaks / genes first, or batch by chromosome.",
                UserWarning, stacklevel=3,
            )


def _pearson_matrix(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation of a vector ``x`` against every column of ``Y``.

    Returns a length-``Y.shape[1]`` numpy array.
    """
    x = x.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    x_mean = x.mean()
    Y_mean = Y.mean(axis=0)
    x_centered = x - x_mean
    Y_centered = Y - Y_mean
    x_norm = np.sqrt(np.sum(x_centered * x_centered))
    Y_norm = np.sqrt(np.sum(Y_centered * Y_centered, axis=0))
    denom = x_norm * Y_norm
    num = x_centered @ Y_centered
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, num / denom, 0.0).astype(np.float32)


def _spearman_matrix(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman correlation of ``x`` against every column of ``Y``.

    Implemented as Pearson on rank-transformed inputs (exact at integer
    rank granularity; the scipy ranker handles ties the same way).
    """
    from scipy.stats import rankdata
    x_r = rankdata(x)
    Y_r = np.empty_like(Y, dtype=np.float64)
    for j in range(Y.shape[1]):
        Y_r[:, j] = rankdata(Y[:, j])
    return _pearson_matrix(x_r, Y_r)


__all__ = ["link_peaks_to_genes"]
