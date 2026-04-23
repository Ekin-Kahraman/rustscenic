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

    # For fast lookup: group genes by chrom, sort by tss.
    gene_by_chrom = {
        chrom: sub.sort_values("tss").reset_index(drop=True)
        for chrom, sub in genes_in_rna.groupby("chrom")
    }

    rna_X = _densify(rna_adata.X).astype(np.float32, copy=False)  # (n_cells, n_genes_rna)
    atac_X = _densify(atac_adata.X).astype(np.float32, copy=False)  # (n_cells, n_peaks)

    corr_fn = _spearman_matrix if method == "spearman" else _pearson_matrix

    rows = []
    # Iterate peaks in chrom-order; batch peaks on the same chromosome
    # so we amortise the sorted-gene-TSS work and compute the correlation
    # over the slice of candidate genes in one vectorised call.
    peak_centers = ((peaks["start"].values + peaks["end"].values) // 2).astype(np.int64)
    peak_chroms = peaks["chrom"].values
    peak_starts = peaks["start"].values
    peak_ends = peaks["end"].values
    peak_ids = list(peaks.index)
    for chrom, gg in gene_by_chrom.items():
        # Positional indices of peaks on this chromosome
        peak_positions = np.where(peak_chroms == chrom)[0]
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
    """Parse peak names like ``chr1:100-200`` or ``chr1-100-200`` to a DataFrame."""
    import re
    pat = re.compile(r"^(chr[\dXYMTI]+)[:\-_](\d+)[\-_](\d+)$")
    rows = []
    for n in names:
        m = pat.match(str(n))
        if m is None:
            return None
        rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


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
