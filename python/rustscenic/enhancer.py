"""Enhancer-to-gene linking for chromatin-aware regulatory networks.

pySCENIC scores regulons as "TF → its top-N co-expressed target genes".
SCENIC+ adds chromatin context:

  TF → enhancer (motif enrichment, from cistarget)
        ↓
     enhancer → target gene (peak accessibility ↔ gene expression
                             correlated across matched cells)

The enhancer-to-gene edge is what this module produces. Combined with
the motif-to-enhancer and TF-to-target edges rustscenic already
computes, it is the raw material for eRegulons.

This module requires **matched cells**: every cell in ``rna_adata.obs_names``
must also be in ``atac_adata.obs_names``. Multiome data (10x Multiome
or nf-core/multiome) is the natural input. For non-matched samples,
CCA or Harmony-style integration needs to happen upstream.

Complexity:
  O(n_peaks × max_genes_in_window × n_cells) where max_genes_in_window
  is typically 20–50. On 100k peaks × 30 candidate genes × 50k cells
  that is about 150 GFLOPs. The Pearson path keeps the hot loop in Rust
  and emits only surviving peak-gene links back to pandas.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    enhancer_align_cell_indices as _align_cell_indices,
    enhancer_link_pearson as _enhancer_link_pearson,
    enhancer_link_pearson_sparse_rna as _enhancer_link_pearson_sparse_rna,
    enhancer_match_gene_coords_to_rna as _match_gene_coords_to_rna,
    enhancer_normalise_chrom_codes as _normalise_chrom_codes,
    enhancer_parse_peak_names as _parse_peak_names_rust,
    enhancer_prepare_gene_order as _prepare_gene_order,
)
from rustscenic._stage_utils import as_float32_array, string_list

_LINK_COLUMNS = [
    "peak_id", "peak_chrom", "peak_start", "peak_end",
    "gene", "gene_tss", "distance", "correlation",
]


def link_peaks_to_genes(
    rna_adata,
    atac_adata,
    gene_coords: pd.DataFrame,
    peak_coords: pd.DataFrame | None = None,
    *,
    max_distance: int = 500_000,
    min_abs_corr: float = 0.1,
    method: str = "pearson",
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
        Must be ``"pearson"``. This is the Rust-backed correlation path.

    Returns
    -------
    pandas.DataFrame
        One row per surviving peak-gene link with columns:
        ``['peak_id', 'peak_chrom', 'peak_start', 'peak_end', 'gene',
        'gene_tss', 'distance', 'correlation']``. Sorted by
        descending ``|correlation|``.
    """
    if method != "pearson":
        raise ValueError("enhancer linking only supports method='pearson'")

    rna_adata, atac_adata = _align_cells(rna_adata, atac_adata)
    backend_symbols = ["enhancer_align_cell_indices"]
    from rustscenic._gene_resolution import resolve_gene_names

    gene_names_rna = resolve_gene_names(rna_adata, quiet=True)
    peaks = _peak_frame(atac_adata, peak_coords)
    backend_symbols.extend(_rust_backend_symbols(peaks))
    genes = _validate_gene_coords(gene_coords)

    matched_gene_rows, source_rna_cols = _match_gene_coords_to_rna(
        string_list(gene_names_rna),
        string_list(genes["gene"]),
    )
    backend_symbols.append("enhancer_match_gene_coords_to_rna")
    matched_gene_rows = np.asarray(matched_gene_rows, dtype=np.intp)
    if matched_gene_rows.size == 0:
        raise ValueError(
            "no gene_coords genes match any gene name in rna_adata - "
            "check species + symbol convention"
        )
    genes_in_rna = genes.iloc[matched_gene_rows].reset_index(drop=True)
    source_rna_cols = np.asarray(source_rna_cols, dtype=np.int64)

    # Normalise chrom names and encode chromosomes once for the Rust Pearson
    # kernel. The vector loop runs in Rust because peak tables can be large.
    gene_chrom_norm, peak_chrom_norm, gene_chrom_codes, peak_chrom_codes = (
        _normalise_chrom_codes(
            string_list(genes_in_rna["chrom"]),
            string_list(peaks["chrom"]),
        )
    )
    backend_symbols.append("enhancer_normalise_chrom_codes")
    gene_chrom_codes = np.asarray(gene_chrom_codes, dtype=np.int32)
    peak_chrom_codes = np.asarray(peak_chrom_codes, dtype=np.int32)
    peak_chroms = np.asarray(peak_chrom_norm, dtype=object)

    # Keep ATAC sparse (CSC for fast column extraction). Pearson now calls
    # into Rust for the per-peak sparse x dense correlation loop; Python only
    # prepares AnnData/pandas metadata and assembles the output DataFrame.
    import scipy.sparse as sp

    atac_raw = atac_adata.X
    if sp.issparse(atac_raw):
        atac_csc = atac_raw.tocsc().astype(np.float32, copy=False)
    else:
        atac_csc = sp.csc_matrix(atac_raw, dtype=np.float32)
    if hasattr(atac_csc, "sum_duplicates"):
        atac_csc.sum_duplicates()
    if hasattr(atac_csc, "sort_indices"):
        atac_csc.sort_indices()

    peak_starts = peaks["start"].to_numpy(dtype=np.int64, copy=False)
    peak_ends = peaks["end"].to_numpy(dtype=np.int64, copy=False)
    peak_ids = np.asarray(peaks.index, dtype=object)

    return _link_peaks_to_genes_pearson(
        rna_adata,
        atac_csc,
        genes_in_rna,
        peak_ids,
        peak_chroms,
        gene_chrom_norm,
        peak_chrom_codes,
        peak_starts,
        peak_ends,
        gene_chrom_codes,
        source_rna_cols,
        max_distance,
        min_abs_corr,
        backend_symbols,
    )


def _link_peaks_to_genes_pearson(
    rna_adata,
    atac_csc,
    genes_in_rna: pd.DataFrame,
    peak_ids: np.ndarray,
    peak_chroms: np.ndarray,
    gene_chroms: list[str],
    peak_chrom_codes: np.ndarray,
    peak_starts: np.ndarray,
    peak_ends: np.ndarray,
    gene_chrom_codes_raw: np.ndarray,
    gene_source_cols_raw: np.ndarray,
    max_distance: int,
    min_abs_corr: float,
    backend_symbols: list[str],
) -> pd.DataFrame:
    gene_tss_raw = genes_in_rna["tss"].to_numpy(dtype=np.int64)
    gene_order, has_chrom_overlap, n_gene_columns = _prepare_gene_order(
        peak_chrom_codes,
        gene_chrom_codes_raw,
        gene_tss_raw,
        gene_source_cols_raw,
    )
    backend_symbols = backend_symbols + ["enhancer_prepare_gene_order"]
    gene_order = np.asarray(gene_order, dtype=np.intp)
    gene_chrom_codes = gene_chrom_codes_raw[gene_order]
    gene_source_cols = gene_source_cols_raw[gene_order]
    gene_names = genes_in_rna["gene"].to_numpy(dtype=object)[gene_order]
    gene_tss = gene_tss_raw[gene_order]

    if not has_chrom_overlap:
        import warnings
        warnings.warn(
            f"no chromosome name overlaps between peaks and gene_coords "
            f"even after UCSC/Ensembl normalisation. Peak chroms: "
            f"{_chrom_examples(peak_chroms)}; "
            f"gene_coords chroms: {_chrom_examples(gene_chroms)}. The "
            f"resulting link DataFrame will be empty.",
            UserWarning, stacklevel=3,
        )
        out = pd.DataFrame(columns=_LINK_COLUMNS)
        out.attrs["rust_backend"] = {"engine": "rust", "symbols": backend_symbols}
        return out

    import scipy.sparse as sp

    rna_raw = rna_adata.X
    rna_csc = None
    if sp.issparse(rna_raw):
        rna_csc = rna_raw.tocsc().astype(np.float32, copy=False)
        if hasattr(rna_csc, "sum_duplicates"):
            rna_csc.sum_duplicates()
        if hasattr(rna_csc, "sort_indices"):
            rna_csc.sort_indices()
        rna_indptr = rna_csc.indptr
        rna_indices = np.asarray(rna_csc.indices, dtype=np.int32)
        rna_data = np.asarray(rna_csc.data, dtype=np.float32)
    else:
        _warn_if_dense_rna_memory_expensive(
            rna_adata,
            n_gene_columns=int(n_gene_columns),
        )
        rna_dense = as_float32_array(rna_raw)

    atac_indptr = atac_csc.indptr
    atac_indices = np.asarray(atac_csc.indices, dtype=np.int32)
    atac_data = np.asarray(atac_csc.data, dtype=np.float32)

    gene_source_cols_u32 = gene_source_cols.astype(np.uint32, copy=False)
    if rna_csc is None:
        backend_symbol = "enhancer_link_pearson"
        peak_ix, gene_ix, distances, corr = _enhancer_link_pearson(
            rna_dense,
            atac_indptr,
            atac_indices,
            atac_data,
            peak_chrom_codes,
            peak_starts,
            peak_ends,
            gene_chrom_codes,
            gene_tss,
            gene_source_cols_u32,
            int(max_distance),
            float(min_abs_corr),
            None,
        )
    else:
        backend_symbol = "enhancer_link_pearson_sparse_rna"
        peak_ix, gene_ix, distances, corr = _enhancer_link_pearson_sparse_rna(
            rna_indptr,
            rna_indices,
            rna_data,
            int(rna_csc.shape[0]),
            int(rna_csc.shape[1]),
            atac_indptr,
            atac_indices,
            atac_data,
            peak_chrom_codes,
            peak_starts,
            peak_ends,
            gene_chrom_codes,
            gene_tss,
            gene_source_cols_u32,
            int(max_distance),
            float(min_abs_corr),
            None,
        )

    peak_ix = np.asarray(peak_ix)
    gene_ix = np.asarray(gene_ix)
    if peak_ix.size == 0:
        out = pd.DataFrame(columns=_LINK_COLUMNS)
        out.attrs["rust_backend"] = {
            "engine": "rust",
            "symbols": backend_symbols + [backend_symbol],
        }
        return out

    correlations = np.asarray(corr, dtype=np.float32)
    out = pd.DataFrame(
        {
            "peak_id": peak_ids[peak_ix],
            "peak_chrom": peak_chroms[peak_ix],
            "peak_start": peak_starts[peak_ix],
            "peak_end": peak_ends[peak_ix],
            "gene": gene_names[gene_ix],
            "gene_tss": gene_tss[gene_ix],
            "distance": np.asarray(distances),
            "correlation": correlations,
        },
        columns=_LINK_COLUMNS,
    )
    out = out.reset_index(drop=True)
    out.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": backend_symbols + [backend_symbol],
    }
    return out


def _rust_backend_symbols(obj) -> list[str]:
    backend = getattr(obj, "attrs", {}).get("rust_backend")
    if not isinstance(backend, dict):
        return []
    symbols = backend.get("symbols")
    if not isinstance(symbols, list):
        return []
    return [symbol for symbol in symbols if isinstance(symbol, str) and symbol]


def _chrom_examples(values, limit: int = 5) -> list[str]:
    """Return a tiny first-seen chromosome sample without scanning uniques."""
    out: list[str] = []
    for value in values:
        text = str(value)
        if text not in out:
            out.append(text)
            if len(out) >= limit:
                break
    return out


def _align_cells(rna_adata, atac_adata):
    rna_idx, atac_idx = _align_cell_indices(
        string_list(rna_adata.obs_names),
        string_list(atac_adata.obs_names),
    )
    rna_idx = np.asarray(rna_idx, dtype=np.intp)
    atac_idx = np.asarray(atac_idx, dtype=np.intp)
    if rna_idx.size == 0:
        raise ValueError(
            "rna_adata and atac_adata share no cell barcodes - this function "
            "requires matched multiome data. For separate scRNA + scATAC "
            "samples, integrate via CCA or scVI before calling."
        )
    if rna_idx.size < len(rna_adata) or atac_idx.size < len(atac_adata):
        import warnings
        warnings.warn(
            f"keeping {rna_idx.size} cells with barcodes in both RNA "
            f"({rna_adata.n_obs}) and ATAC ({atac_adata.n_obs}) AnnDatas.",
            UserWarning, stacklevel=3,
        )
    if (
        rna_idx.size == len(rna_adata) == len(atac_adata)
        and np.array_equal(rna_idx, np.arange(len(rna_adata), dtype=np.intp))
        and np.array_equal(atac_idx, np.arange(len(atac_adata), dtype=np.intp))
    ):
        return rna_adata, atac_adata
    return rna_adata[rna_idx], atac_adata[atac_idx]


def _peak_frame(atac_adata, peak_coords) -> pd.DataFrame:
    if peak_coords is not None:
        need = {"chrom", "start", "end"}
        missing = need - set(peak_coords.columns)
        if missing:
            raise ValueError(f"peak_coords missing columns: {missing}")
        out = peak_coords.loc[atac_adata.var_names]
        out.attrs.update(getattr(peak_coords, "attrs", {}))
        return out
    # Try to get from atac_adata.var
    var = atac_adata.var
    if {"chrom", "start", "end"}.issubset(var.columns):
        return var[["chrom", "start", "end"]].copy()
    # Parse "chrN:start-end" pattern from var_names
    parsed = _parse_peak_names(string_list(atac_adata.var_names))
    if parsed is None:
        raise ValueError(
            "atac_adata has no peak coordinates - either include chrom/start/end "
            "columns in atac_adata.var, pass peak_coords explicitly, or use "
            "`chr:start-end` formatted var_names."
        )
    parsed.index = atac_adata.var_names
    return parsed


def _parse_peak_names(names):
    """Parse peak names like ``chr1:100-200``, ``1:100-200``, ``chr1-100-200``.

    Accepts UCSC (``chr1``), Ensembl (``1``), and alt-contig conventions
    (e.g. ``KI270721.1:2090-2985`` from 10x ATAC outputs which include
    decoys + alt-contigs). The full chrom token is preserved as-is so
    downstream callers that joined on ``chr1`` keep working; chromosome-
    name normalisation for cross-convention joining is done in
    ``_normalise_chrom`` at join time rather than here.
    """
    parsed = _parse_peak_names_rust(string_list(names))
    if parsed is None:
        return None
    chroms, starts, ends = parsed
    out = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    out.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": ["enhancer_parse_peak_names"],
    }
    return out


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
            f"['gene', 'chrom', 'tss'] - use a GTF → DataFrame converter "
            f"like gtfparse, or extract from biomart / GENCODE."
        )
    out = gene_coords[["gene", "chrom", "tss"]].copy()
    out["tss"] = out["tss"].astype(np.int64)
    return out


# Warn before `link_peaks_to_genes` hands a large dense RNA matrix to Rust.
# Module-level so tests can patch down.
_DENSIFY_WARN_BYTES = 8 * 1024**3  # 8 GiB


def _warn_if_dense_rna_memory_expensive(rna_adata, n_gene_columns: int | None = None) -> None:
    """Warn before the dense RNA path touches a large matrix.

    Pearson `link_peaks_to_genes` keeps ATAC sparse (CSC). Sparse RNA also
    stays sparse; this warning is only for callers that pass dense RNA.
    """
    import warnings

    n_cols = rna_adata.n_vars if n_gene_columns is None else int(n_gene_columns)
    bytes_needed = rna_adata.n_obs * n_cols * 4
    if bytes_needed >= _DENSIFY_WARN_BYTES:
        gib = bytes_needed / 1024**3
        warnings.warn(
            f"link_peaks_to_genes will process a dense "
            f"{rna_adata.n_obs} × {n_cols} float32 RNA matrix "
            f"(~{gib:.1f} GiB). Keep RNA sparse or subset genes if "
            f"this OOMs.",
            UserWarning,
            stacklevel=3,
        )


__all__ = ["link_peaks_to_genes"]
