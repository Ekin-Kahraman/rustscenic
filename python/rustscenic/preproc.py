"""scATAC fragment preprocessing.

Public API:
    rustscenic.preproc.fragments_to_matrix(fragments_path, peaks_path) -> AnnData
    rustscenic.preproc.call_peaks(fragments_path, cluster_per_barcode, ...) -> pd.DataFrame
    rustscenic.preproc.qc.insert_size_stats(fragments_path) -> pd.DataFrame
    rustscenic.preproc.qc.frip(fragments_path, peaks_path) -> pd.Series
    rustscenic.preproc.qc.tss_enrichment(fragments_path, tss_df) -> pd.Series

All of these are Rust-native and require no extra Python dependencies
beyond the four rustscenic ships with (numpy, pandas, pyarrow, scipy).

Together they replace pycisTopic's fragment parsing, iterative peak
calling, and per-cell QC — the MACS2-free, Java-free, Python 3.10–3.13
path through rustscenic's ATAC preprocessing surface.

See `docs/atac-preprocessing-scope.md` for scope + validation plan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    preproc_call_peaks as _call_peaks,
    preproc_fragments_to_matrix as _fragments_to_matrix,
    preproc_frip as _frip,
    preproc_insert_size_stats as _insert_size_stats,
    preproc_tss_enrichment as _tss_enrichment,
)


def fragments_to_matrix(
    fragments_path: Union[str, Path],
    peaks_path: Union[str, Path],
):
    """Build a cells × peaks AnnData from fragments + peaks.

    Parameters
    ----------
    fragments_path : str or Path
        10x cellranger `fragments.tsv` or `fragments.tsv.gz`.
    peaks_path : str or Path
        Consensus peak BED (plain or .gz). Uses the first three columns;
        column 4 (if present) is stored as `peak_id`.

    Returns
    -------
    AnnData
        - `.X` is a `scipy.sparse.csr_matrix` of u32 fragment-in-peak
          counts with shape (n_cells, n_peaks).
        - `.obs_names` are cell barcodes (from the fragments file).
        - `.var_names` are peak IDs (from the peak BED, or
          `chrom:start-end` if no name column).
        - `.obs["fragments_per_cell"]` and `.obs["total_counts"]` are
          per-cell QC metrics over the full fragments file (before the
          peak intersection).

    Notes
    -----
    UCSC (`chr1`) and Ensembl (`1`) chrom conventions are matched via
    `normalise_chrom` — a peak BED in either convention joins to
    fragments in either convention. Peaks on chroms actually absent
    from the fragments file are silently dropped.
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    fragments_path = str(Path(fragments_path))
    peaks_path = str(Path(peaks_path))

    data, indices, indptr, shape, barcodes, peaks, fpc, tcc = _fragments_to_matrix(
        fragments_path, peaks_path
    )

    # Guard against the 6-column strand BED parse mode — if > 90% of
    # "barcodes" are unique (i.e. one per fragment), the barcode column
    # is almost certainly a per-row ID (peak name, gene name), not a
    # cell barcode. Column count in a 10x cellranger fragments.tsv is
    # 5: chrom start end barcode count. A 6-column strand-BED is
    # chrom start end name score strand — the barcode parse lands on
    # `name`, one per line.
    total_frags = int(np.asarray(fpc, dtype=np.uint64).sum())
    if barcodes and total_frags > 100 and len(barcodes) > 0.9 * total_frags:
        import warnings
        warnings.warn(
            f"{len(barcodes)} unique 'barcodes' parsed from {total_frags} "
            f"fragments — almost one-per-row. This usually means the file "
            f"is a 6-column strand BED (chrom, start, end, name, score, "
            f"strand) rather than a 10x cellranger fragments.tsv "
            f"(chrom, start, end, barcode, count). Convert with e.g. "
            f"awk or resave as 5-col cellranger format first.",
            UserWarning, stacklevel=2,
        )

    X = csr_matrix((data, indices, indptr), shape=shape)

    # 10x fragments.tsv contains every observed barcode, including empty
    # droplets and dead cells (often 100k+ in a "3k" sample). Most
    # downstream analysis wants only the cell-called barcodes from the
    # 10x filtered_feature_bc_matrix.h5, not the raw set. Warn if the
    # barcode count looks like a raw fragments file. Heuristic: a real
    # cell-called sample is typically < 100k cells; raw fragments are
    # often 10x+ that.
    if len(barcodes) > 100_000:
        import warnings
        median_frags = float(np.median(np.asarray(fpc, dtype=np.uint64)))
        warnings.warn(
            f"fragments_to_matrix returned {len(barcodes):,} barcodes "
            f"(median {median_frags:.0f} fragments/barcode). 10x raw "
            f"fragments.tsv contains every observed barcode including "
            f"empty droplets — most downstream analysis wants only the "
            f"cell-called barcodes. Subset by the 10x "
            f"filtered_feature_bc_matrix.h5 cell list (or by "
            f"obs['fragments_per_cell'] > some threshold) before "
            f"running topics / AUCell on this AnnData.",
            UserWarning, stacklevel=2,
        )

    obs = pd.DataFrame(
        {
            "fragments_per_cell": np.asarray(fpc, dtype=np.uint32),
            "total_counts": np.asarray(tcc, dtype=np.uint32),
        },
        index=pd.Index(list(barcodes), name="barcode"),
    )
    var = pd.DataFrame(index=pd.Index(list(peaks), name="peak"))

    return ad.AnnData(X=X, obs=obs, var=var)


def call_peaks(
    fragments_path: Union[str, Path],
    cluster_per_barcode: Union[pd.Series, np.ndarray, list[int]],
    *,
    n_clusters: Union[int, None] = None,
    window_size: int = 50,
    min_fragments_per_window: int = 3,
    quantile_threshold: float = 0.95,
    max_gap: int = 250,
    peak_half_width: int = 250,
    output_bed: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Call iterative consensus peaks from pseudobulked fragments.

    Corces-2018-style density-window peak calling: per cluster, tile the
    genome, threshold at the top-quantile of the cluster's per-window
    count distribution, merge adjacent significant windows, then union
    cluster peaks into a consensus set via greedy iterative overlap
    rejection. No MACS2, no external binaries.

    Parameters
    ----------
    fragments_path
        10x cellranger `fragments.tsv[.gz]`.
    cluster_per_barcode
        Cluster id per barcode (0-indexed). Same length as the fragments'
        unique-barcode set — i.e. as what `fragments_to_matrix(...).obs`
        produces. Use ``u32::MAX`` (2**32 - 1) to mark unassigned.
        Accepts a list, numpy array, or a pandas Series (the latter's
        index is used to align to the fragments' barcode order — the
        Rust layer will reject a mismatch loudly).
    n_clusters
        Number of distinct cluster ids. Defaults to
        ``int(max(cluster_per_barcode)) + 1``.
    window_size, min_fragments_per_window, quantile_threshold, max_gap, peak_half_width
        See ``rustscenic-preproc``'s ``PeakCallingConfig``. Defaults
        match the Corces-2018 convention (50 bp windows, top 5 %
        quantile, 250 bp max gap, 501 bp wide peaks).
    output_bed
        If set, write the called peaks as a 4-column BED
        (chrom, start, end, name) to this path.

    Returns
    -------
    pd.DataFrame
        One row per consensus peak with columns
        ``['chrom', 'start', 'end', 'name']``. ``name`` is
        ``chrom:start-end``.
    """
    fragments_path = str(Path(fragments_path))
    clusters = np.asarray(cluster_per_barcode, dtype=np.int64)
    # PyO3 receives u32; clamp negative / NaN-likes to u32::MAX.
    clusters_u32 = np.where(clusters < 0, np.uint32(0xFFFF_FFFF), clusters).astype(np.uint32)
    if n_clusters is None:
        # Max valid cluster id → n_clusters
        valid = clusters[clusters >= 0]
        n_clusters = int(valid.max()) + 1 if valid.size > 0 else 0

    chroms, starts, ends, names = _call_peaks(
        fragments_path,
        clusters_u32.tolist(),
        n_clusters,
        window_size,
        min_fragments_per_window,
        quantile_threshold,
        max_gap,
        peak_half_width,
    )
    df = pd.DataFrame(
        {
            "chrom": list(chroms),
            "start": np.asarray(starts, dtype=np.uint32),
            "end": np.asarray(ends, dtype=np.uint32),
            "name": list(names),
        }
    )
    if output_bed is not None:
        df.to_csv(output_bed, sep="\t", header=False, index=False)
    return df


class qc:
    """Per-barcode cell QC metrics.

    Invocable as a namespace class so users see the grouping in IDE
    auto-complete and the module docstring lists them at the top.
    Each method reads the fragments file once and returns a pandas
    object indexed by barcode.
    """

    @staticmethod
    def insert_size_stats(
        fragments_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Per-barcode insert-size summary.

        Returns DataFrame indexed by barcode with columns:
          - ``mean`` (float) — mean fragment length
          - ``median`` (float) — median fragment length
          - ``n_fragments`` (uint32) — count used in the stats
          - ``sub_nucleosomal`` (uint32) — fragments < 150 bp
          - ``mono_nucleosomal`` (uint32) — 150–299 bp
          - ``di_nucleosomal`` (uint32) — 300–449 bp
        """
        fragments_path = str(Path(fragments_path))
        barcodes, means, medians, counts, sub, mono, di = _insert_size_stats(
            fragments_path
        )
        return pd.DataFrame(
            {
                "mean": np.asarray(means, dtype=np.float32),
                "median": np.asarray(medians, dtype=np.float32),
                "n_fragments": np.asarray(counts, dtype=np.uint32),
                "sub_nucleosomal": np.asarray(sub, dtype=np.uint32),
                "mono_nucleosomal": np.asarray(mono, dtype=np.uint32),
                "di_nucleosomal": np.asarray(di, dtype=np.uint32),
            },
            index=pd.Index(list(barcodes), name="barcode"),
        )

    @staticmethod
    def frip(
        fragments_path: Union[str, Path],
        peaks_path: Union[str, Path],
    ) -> pd.Series:
        """Per-barcode fraction of reads in peaks.

        Returns a pandas Series indexed by barcode. UCSC vs Ensembl
        chrom conventions are normalised at join time.
        """
        fragments_path = str(Path(fragments_path))
        peaks_path = str(Path(peaks_path))
        barcodes, scores = _frip(fragments_path, peaks_path)
        return pd.Series(
            np.asarray(scores, dtype=np.float32),
            index=pd.Index(list(barcodes), name="barcode"),
            name="frip",
        )

    @staticmethod
    def tss_enrichment(
        fragments_path: Union[str, Path],
        tss: pd.DataFrame,
    ) -> pd.Series:
        """Per-barcode TSS enrichment (signal-over-background).

        Parameters
        ----------
        fragments_path
            10x fragments file.
        tss
            DataFrame with columns ``['chrom', 'position']`` listing
            transcription-start-site coordinates. Chromosome names in
            either UCSC (`chr1`) or Ensembl (`1`) convention are fine —
            they're normalised to match the fragments' chrom namespace.

        Returns
        -------
        pd.Series indexed by barcode with TSS enrichment scores (0.0 if
        no fragments overlap any TSS window).
        """
        if not {"chrom", "position"}.issubset(tss.columns):
            raise ValueError(
                "tss DataFrame must have columns 'chrom' and 'position'. "
                f"Got: {list(tss.columns)}"
            )
        fragments_path = str(Path(fragments_path))
        chroms = [str(c) for c in tss["chrom"].tolist()]
        positions = [int(p) for p in tss["position"].tolist()]
        barcodes, scores = _tss_enrichment(fragments_path, chroms, positions)
        return pd.Series(
            np.asarray(scores, dtype=np.float32),
            index=pd.Index(list(barcodes), name="barcode"),
            name="tss_enrichment",
        )


__all__ = ["fragments_to_matrix", "call_peaks", "qc"]
