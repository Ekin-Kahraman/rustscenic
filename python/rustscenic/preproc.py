"""scATAC fragment preprocessing.

Public API:
    rustscenic.preproc.fragments_to_matrix(fragments_path, peaks_path) -> AnnData

Takes a 10x cellranger `fragments.tsv[.gz]` and a consensus-peak BED,
returns an AnnData with shape (n_cells, n_peaks) — the format
`rustscenic.topics.fit` accepts.

This is the Rust-native replacement for pycisTopic's fragment-to-matrix
pipeline. No new Python dependencies (still just numpy, pandas, scipy).

See `docs/atac-preprocessing-scope.md` for scope + validation plan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    preproc_fragments_to_matrix as _fragments_to_matrix,
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
    Peaks on chromosomes not present in the fragments file are silently
    dropped. If no peak's chromosome matches any fragment, the resulting
    matrix is all-zero (but with the correct shape).
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    fragments_path = str(Path(fragments_path))
    peaks_path = str(Path(peaks_path))

    data, indices, indptr, shape, barcodes, peaks, fpc, tcc = _fragments_to_matrix(
        fragments_path, peaks_path
    )

    X = csr_matrix((data, indices, indptr), shape=shape)

    obs = pd.DataFrame(
        {
            "fragments_per_cell": np.asarray(fpc, dtype=np.uint32),
            "total_counts": np.asarray(tcc, dtype=np.uint32),
        },
        index=pd.Index(list(barcodes), name="barcode"),
    )
    var = pd.DataFrame(index=pd.Index(list(peaks), name="peak"))

    return ad.AnnData(X=X, obs=obs, var=var)


__all__ = ["fragments_to_matrix"]
