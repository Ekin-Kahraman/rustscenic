"""Type stubs for the rustscenic PyO3 extension.

These reflect the real signatures in crates/rustscenic-py/src/lib.rs.
Kept in sync by convention — CI's import-smoke step will complain loudly
if these drift.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str


def grn_infer(
    expression: npt.NDArray[np.float32],
    gene_names: list[str],
    tf_names: list[str],
    n_estimators: int = 5000,
    learning_rate: float = 0.01,
    max_features: float = 0.1,
    subsample: float = 0.9,
    max_depth: int = 3,
    early_stop_window: int = 25,
    seed: int = 777,
) -> tuple[list[str], list[str], npt.NDArray[np.float32]]:
    """Infer (TF, target, importance) edges from an expression matrix.

    Returns three parallel arrays: tf names, target names, importance scores.
    """
    ...


def aucell_score(
    expression: npt.NDArray[np.float32],
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
) -> npt.NDArray[np.float32]:
    """Per-cell regulon recovery AUCs.

    Returns a (n_cells, n_regulons) f32 array of normalised AUCs in [0, 1].
    """
    ...


def topics_fit(
    row_ptr: list[int],
    col_idx: list[int],
    counts: list[float],
    n_words: int,
    n_topics: int = 50,
    alpha: float = 0.02,
    eta: float = 0.02,
    tau0: float = 64.0,
    kappa: float = 0.7,
    batch_size: int = 256,
    n_passes: int = 10,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fit Online VB LDA; return (cell_topic, topic_word) probability matrices."""
    ...


def preproc_fragments_to_matrix(
    fragments_path: str,
    peaks_path: str,
) -> tuple[
    npt.NDArray[np.uint32],   # data
    npt.NDArray[np.uint32],   # indices
    npt.NDArray[np.uint32],   # indptr
    tuple[int, int],          # shape
    list[str],                # barcodes
    list[str],                # peak ids
    npt.NDArray[np.uint32],   # fragments_per_cell
    npt.NDArray[np.uint32],   # total_counts
]:
    """Parse a 10x fragments.tsv[.gz] and peaks BED into CSR cells×peaks."""
    ...


def preproc_insert_size_stats(
    fragments_path: str,
) -> tuple[
    list[str],                # barcodes
    npt.NDArray[np.float32],  # mean
    npt.NDArray[np.float32],  # median
    npt.NDArray[np.uint32],   # n_fragments
    npt.NDArray[np.uint32],   # sub-nucleosomal
    npt.NDArray[np.uint32],   # mono-nucleosomal
    npt.NDArray[np.uint32],   # di-nucleosomal
]:
    """Per-barcode insert-size summary statistics."""
    ...


def preproc_frip(
    fragments_path: str,
    peaks_path: str,
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Per-barcode fraction of reads in peaks. Returns (barcodes, frip)."""
    ...


def preproc_tss_enrichment(
    fragments_path: str,
    tss_chroms: list[str],
    tss_positions: list[int],
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Per-barcode TSS enrichment (signal-over-background)."""
    ...


def preproc_call_peaks(
    fragments_path: str,
    cluster_per_barcode: list[int],
    n_clusters: int,
    window_size: int = 50,
    min_fragments_per_window: int = 3,
    quantile_threshold: float = 0.95,
    max_gap: int = 250,
    peak_half_width: int = 250,
) -> tuple[list[str], list[int], list[int], list[str]]:
    """Iterative consensus peak calling. Returns (chroms, starts, ends, names)."""
    ...
