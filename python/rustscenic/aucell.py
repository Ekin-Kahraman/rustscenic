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


def score(
    expression,
    regulons: Iterable,
    *,
    top_frac: float = 0.05,
) -> pd.DataFrame:
    """Compute per-cell regulon activity matrix.

    Parameters
    ----------
    expression
        AnnData, pandas DataFrame (cells × genes), or (matrix, gene_names) tuple.
    regulons
        Either:
          - A dict mapping regulon_name -> set/list of target-gene symbols
          - A list of pyscenic `Regulon` / `GeneSignature` objects
            (objects with `.name` and either `.genes` or `.gene2weight` attrs)
    top_frac
        Fraction of top-ranked genes per cell used as AUC cutoff (default 0.05,
        matches pyscenic).

    Returns
    -------
    pandas.DataFrame of shape (n_cells, n_regulons). Index: cell barcodes.
    Columns: regulon names.
    """
    X, gene_names, cell_names = _coerce(expression)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    X = np.ascontiguousarray(X)

    dup_count = len(gene_names) - len(set(gene_names))
    if dup_count > 0:
        raise ValueError(
            f"{dup_count} duplicate gene name(s) in expression matrix — regulon "
            f"gene-to-column lookup is ambiguous. Call AnnData.var_names_make_unique() "
            f"or deduplicate upstream."
        )

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    reg_names: list[str] = []
    reg_gene_indices: list[list[int]] = []

    dropped_empty = 0
    for reg in regulons:
        name, genes = _coerce_regulon(reg)
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idx:
            dropped_empty += 1
            continue
        reg_names.append(name)
        reg_gene_indices.append(idx)

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

    auc = _aucell_score(X, reg_names, reg_gene_indices, top_frac)
    return pd.DataFrame(np.asarray(auc), index=cell_names, columns=reg_names)


def _coerce(expression):
    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        X = expression.X.toarray() if hasattr(expression.X, "toarray") else np.asarray(expression.X)
        return X, list(expression.var_names), list(expression.obs_names)
    if isinstance(expression, pd.DataFrame):
        return np.asarray(expression.values), list(expression.columns), list(expression.index)
    if isinstance(expression, tuple) and len(expression) == 2:
        X, gene_names = expression
        return np.asarray(X), list(gene_names), list(range(np.asarray(X).shape[0]))
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
