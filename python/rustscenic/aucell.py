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

from typing import Iterable, Union

import numpy as np
import pandas as pd

from rustscenic._rustscenic import aucell_score as _aucell_score
from rustscenic._gene_resolution import (
    dedupe_by_symbol,
    regulon_coverage,
    warn_if_likely_unnormalized,
    warn_if_poor_coverage,
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
        For sparse input, densify and score at most this many cells at a time
        (default 10_000). Keeps peak RSS to ~chunk_size × n_genes × 4 bytes
        during the densification step. Set to None or >= n_cells to densify
        everything in one go.

    Returns
    -------
    pandas.DataFrame of shape (n_cells, n_regulons). Index: cell barcodes.
    Columns: regulon names. Per-regulon gene-coverage counts are stored on
    the DataFrame's ``.attrs["regulon_coverage"]`` dict, keyed by regulon name.
    """
    X_raw, gene_names, cell_names = _coerce(expression)
    warn_if_likely_unnormalized(X_raw, stacklevel=3)

    dup_count = len(gene_names) - len(set(gene_names))
    if dup_count > 0:
        import warnings
        from collections import Counter
        top_dupes = [
            n for n, c in Counter(gene_names).most_common(3) if c > 1
        ]
        warnings.warn(
            f"{dup_count} duplicate gene name(s) after ENSEMBL→symbol "
            f"resolution (e.g. {top_dupes}). Summing expression across "
            f"duplicate symbols so regulon lookup stays unambiguous. "
            f"Pass the AnnData through `rustscenic._gene_resolution."
            f"dedupe_by_symbol()` upstream if you want full control.",
            UserWarning, stacklevel=3,
        )
        X_raw, gene_names = dedupe_by_symbol(X_raw, gene_names)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    reg_names: list[str] = []
    reg_gene_indices: list[list[int]] = []
    # Accumulate (name, genes) pairs once so we can report coverage
    # alongside the AUC matrix and also warn on poor overlap.
    reg_pairs: list[tuple[str, list[str]]] = []

    dropped_empty = 0
    for reg in regulons:
        name, genes = _coerce_regulon(reg)
        genes_list = list(genes)
        reg_pairs.append((name, genes_list))
        idx = [gene_to_idx[g] for g in genes_list if g in gene_to_idx]
        if not idx:
            dropped_empty += 1
            continue
        reg_names.append(name)
        reg_gene_indices.append(idx)

    coverage = regulon_coverage(gene_names, reg_pairs)
    warn_if_poor_coverage(coverage, stacklevel=3)

    if dropped_empty > 0 and not reg_names:
        import warnings
        warnings.warn(
            f"all {dropped_empty} regulons dropped — no genes overlap the expression "
            f"matrix. Check that regulon gene symbols match your AnnData.var_names.",
            UserWarning, stacklevel=2,
        )
    elif dropped_empty > 0:
        import warnings
        warnings.warn(
            f"{dropped_empty} of {dropped_empty + len(reg_names)} regulons dropped "
            f"(no genes overlap expression matrix).",
            UserWarning, stacklevel=2,
        )

    # Score in chunks if the input is sparse — densifying 100k x 20k is 8 GB
    # float32, and the peak during conversion is 2x that. Chunking bounds
    # instantaneous RSS to chunk_size × n_genes × 4 bytes.
    import scipy.sparse as sp
    n_cells = len(cell_names)
    if sp.issparse(X_raw):
        if chunk_size is None or chunk_size >= n_cells:
            X = X_raw.toarray()
            if X.dtype != np.float32:
                X = X.astype(np.float32, copy=False)
            X = np.ascontiguousarray(X)
            auc = _aucell_score(X, reg_names, reg_gene_indices, top_frac)
        else:
            # Dense-per-chunk; concat.
            parts = []
            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                X_chunk = X_raw[start:end].toarray()
                if X_chunk.dtype != np.float32:
                    X_chunk = X_chunk.astype(np.float32, copy=False)
                X_chunk = np.ascontiguousarray(X_chunk)
                parts.append(np.asarray(_aucell_score(X_chunk, reg_names, reg_gene_indices, top_frac)))
            auc = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(reg_names)), dtype=np.float32)
    else:
        X = np.asarray(X_raw)
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        X = np.ascontiguousarray(X)
        auc = _aucell_score(X, reg_names, reg_gene_indices, top_frac)
    df = pd.DataFrame(np.asarray(auc), index=cell_names, columns=reg_names)
    # Store per-regulon coverage as metadata so downstream code can
    # diagnose low-overlap regulons without rerunning.
    df.attrs["regulon_coverage"] = coverage
    return df


def _coerce(expression):
    """Return (X, gene_names, cell_names) — X may be dense numpy OR scipy sparse.

    For AnnData inputs, ``gene_names`` comes from
    ``_gene_resolution.resolve_gene_names`` which auto-detects cellxgene /
    ENSEMBL-in-var_names datasets and swaps to the symbol column.
    """
    import scipy.sparse as sp
    from rustscenic._gene_resolution import resolve_gene_names
    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        X = expression.X
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


def _coerce_regulon(reg):
    if isinstance(reg, tuple) and len(reg) == 2:
        name, genes = reg
        return str(name), list(genes)
    if isinstance(reg, dict):
        if "name" in reg and "genes" in reg:
            return str(reg["name"]), list(reg["genes"])
    # pyscenic Regulon / GeneSignature
    name = getattr(reg, "name", None) or getattr(reg, "transcription_factor", None)
    if hasattr(reg, "gene2weight"):
        genes = list(reg.gene2weight.keys())
    elif hasattr(reg, "genes"):
        genes = list(reg.genes)
    else:
        raise TypeError(f"cannot extract regulon genes from {type(reg).__name__}")
    if name is None:
        raise TypeError(f"regulon {reg!r} has no .name")
    return str(name), genes
