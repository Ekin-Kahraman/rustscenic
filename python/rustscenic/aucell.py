"""AUCell regulon activity scoring.

Public API:
    rustscenic.aucell.score(expression, regulons, top_frac=0.05) -> pd.DataFrame

Matches pyscenic.aucell.aucell at the algorithmic level:
  For each cell c, rank all genes by expression (descending).
  For each regulon r with gene set G_r, AUC = sum over g in G_r with rank(g) < K
  of (K - rank), normalized by K * |G_r| (clamped to [0, 1]).
  K = floor(top_frac * n_genes).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    aucell_score as _aucell_score,
    aucell_score_sparse_csr as _aucell_score_sparse_csr,
)
from rustscenic._gene_resolution import (
    dedupe_by_symbol,
    duplicate_gene_summary,
    warn_if_max_likely_unnormalized,
    warn_if_poor_coverage,
)
from rustscenic._stage_utils import (
    as_float32_array,
    as_float32_contiguous,
    prepare_regulon_indices_with_coverage,
)


def score(
    expression,
    regulons: Iterable,
    *,
    top_frac: float = 0.05,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """Compute per-cell regulon activity matrix.

    Parameters
    ----------
    expression
        AnnData, pandas DataFrame (cells × genes), or (matrix, gene_names) tuple.
        For AnnData inputs, gene names are resolved via
        ``rustscenic._gene_resolution.resolve_gene_names`` so cellxgene /
        ENSEMBL-ID-keyed datasets are matched against regulon gene symbols
        automatically.
    regulons
        Either:
          - A dict mapping regulon_name -> set/list of target-gene symbols
          - A list of pyscenic `Regulon` / `GeneSignature` objects
            (objects with `.name` and either `.genes` or `.gene2weight` attrs)
    top_frac
        Fraction of top-ranked genes per cell used as AUC cutoff (default 0.05,
        matches pyscenic).
    chunk_size
        Kept for API compatibility. Sparse SciPy/AnnData inputs are scored by
        the Rust CSR kernel without dense materialisation; dense inputs ignore
        this setting.

    Returns
    -------
    pandas.DataFrame of shape (n_cells, n_regulons). Index: cell barcodes.
    Columns: regulon names. Per-regulon gene-coverage counts are stored on
    the DataFrame's ``.attrs["regulon_coverage"]`` dict, keyed by regulon name.
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be None or a positive integer, got {chunk_size}")
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(
            f"top_frac must be in (0, 1], got {top_frac}. "
            f"top_frac=0 scores every cell to zero; top_frac>1 is invalid."
        )
    if top_frac > 0.3:
        import warnings
        warnings.warn(
            f"top_frac={top_frac} is unusually high - pyscenic defaults to "
            f"0.05 and values > 0.3 saturate every regulon near its ceiling. "
            f"Are you sure?",
            UserWarning, stacklevel=3,
        )

    X_raw, gene_names, cell_names = _coerce(expression)
    # Warn if the rank-cutoff K = floor(top_frac × n_genes) is so small
    # that no regulon's members can be inside it. Was a silent-all-zero
    # mode at very small top_frac on small gene sets.
    rank_cutoff_raw = int(round(top_frac * len(gene_names)))
    rank_cutoff = max(rank_cutoff_raw - 1, 0)
    if rank_cutoff < 1:
        import warnings
        warnings.warn(
            f"effective AUCell rank cutoff is {rank_cutoff}: "
            f"round(top_frac × n_genes) - 1 = round({top_frac} × "
            f"{len(gene_names)}) - 1, which is below 1. Every AUC will score to 0. "
            f"Increase top_frac or use a wider gene set.",
            UserWarning, stacklevel=3,
        )
    dup_count, top_dupes = duplicate_gene_summary(gene_names)
    if dup_count > 0:
        import warnings
        warnings.warn(
            f"{dup_count} duplicate gene name(s) after ENSEMBL→symbol "
            f"resolution (e.g. {top_dupes}). Summing expression across "
            f"duplicate symbols so regulon lookup stays unambiguous. "
            f"Pass the AnnData through `rustscenic._gene_resolution."
            f"dedupe_by_symbol()` upstream if you want full control.",
            UserWarning, stacklevel=3,
        )
        X_raw, gene_names = dedupe_by_symbol(X_raw, gene_names)

    reg_names, reg_gene_indices, reg_pairs, coverage, dropped_empty = (
        prepare_regulon_indices_with_coverage(gene_names, regulons)
    )
    warn_if_poor_coverage(coverage, stacklevel=3)

    if dropped_empty > 0 and not reg_names:
        import warnings
        from rustscenic._gene_resolution import diagnose_zero_tf_overlap
        # Pool all regulon genes as a stand-in for "TFs" - the diagnostic
        # works the same way (convention-mismatch is symmetric).
        all_reg_genes = [g for _, gs in reg_pairs for g in gs]
        hint = diagnose_zero_tf_overlap(all_reg_genes[:50], gene_names)
        warnings.warn(
            f"all {dropped_empty} regulons dropped - no genes overlap the "
            f"expression matrix. {hint}",
            UserWarning, stacklevel=2,
        )
    elif dropped_empty > 0:
        import warnings
        warnings.warn(
            f"{dropped_empty} of {dropped_empty + len(reg_names)} regulons dropped "
            f"(no genes overlap expression matrix).",
            UserWarning, stacklevel=2,
        )

    import scipy.sparse as sp
    if sp.issparse(X_raw):
        X_csr = X_raw.tocsr().astype(np.float32, copy=False)
        X_csr.sum_duplicates()
        X_csr.sort_indices()
        auc, expression_max = _unpack_aucell_result(_aucell_score_sparse_csr(
            X_csr.indptr,
            np.asarray(X_csr.indices, dtype=np.int32),
            np.asarray(X_csr.data, dtype=np.float32),
            int(X_csr.shape[1]),
            reg_names,
            reg_gene_indices,
            top_frac,
        ))
    else:
        X = as_float32_array(X_raw)
        auc, expression_max = _unpack_aucell_result(
            _aucell_score(X, reg_names, reg_gene_indices, top_frac)
        )
    warn_if_max_likely_unnormalized(expression_max, stacklevel=3)
    df = pd.DataFrame(np.asarray(auc), index=cell_names, columns=reg_names)
    # Store per-regulon coverage as metadata so downstream code can
    # diagnose low-overlap regulons without rerunning.
    df.attrs["regulon_coverage"] = coverage
    return df


def _coerce(expression):
    """Return (X, gene_names, cell_names) - X may be dense numpy OR scipy sparse.

    For AnnData inputs, ``gene_names`` comes from
    ``_gene_resolution.resolve_gene_names`` which auto-detects cellxgene /
    ENSEMBL-in-var_names datasets and swaps to the symbol column.
    """
    import scipy.sparse as sp
    from rustscenic._gene_resolution import resolve_gene_names
    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        # Backed AnnData (read_h5ad(..., backed='r')) exposes X as an
        # anndata `_CSRDataset` / `_CSCDataset`, neither a numpy array nor
        # a scipy.sparse matrix. Materialise it so downstream ops work.
        if not hasattr(expression, "isbacked") or not expression.isbacked:
            X = expression.X
        else:
            import warnings
            warnings.warn(
                "AnnData is backed (disk-resident). Materialising X to a "
                "sparse matrix in memory - if this OOMs, subset cells or "
                "genes before passing to rustscenic.",
                UserWarning, stacklevel=3,
            )
            X = expression.X[:]  # anndata backed → scipy.sparse
            if sp.issparse(X) and not isinstance(X, sp.csr_matrix):
                X = X.tocsr()
        if not sp.issparse(X):
            X = np.asarray(X)
        return X, resolve_gene_names(expression), list(expression.obs_names)
    if isinstance(expression, pd.DataFrame):
        return np.asarray(expression.values), list(expression.columns), list(expression.index)
    if isinstance(expression, tuple) and len(expression) == 2:
        X, gene_names = expression
        if not sp.issparse(X):
            X = np.asarray(X)
        return X, list(gene_names), list(range(X.shape[0]))
    raise TypeError("expression must be AnnData, pandas.DataFrame, or (matrix, gene_names) tuple")


def _unpack_aucell_result(result):
    """Return ``(auc_matrix, expression_max)`` from current or old test doubles."""
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None
