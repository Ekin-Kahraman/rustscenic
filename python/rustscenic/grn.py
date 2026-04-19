"""GRN inference (GRNBoost2 replacement).

Public API:
    rustscenic.grn.infer(adata_or_matrix, tf_names, ...) -> pd.DataFrame
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from rustscenic._rustscenic import grn_infer as _grn_infer


def infer(
    expression,
    tf_names: Iterable[str],
    *,
    n_estimators: int = 5000,
    learning_rate: float = 0.01,
    max_features: float = 0.1,
    subsample: float = 0.9,
    max_depth: int = 3,
    early_stop_window: int = 25,
    seed: int = 777,
    verbose: bool = True,
) -> pd.DataFrame:
    """Infer a gene regulatory network.

    Parameters
    ----------
    expression
        An AnnData object, a ``pandas.DataFrame`` (cells × genes), or a
        ``(matrix, gene_names)`` tuple where ``matrix`` is shape
        ``(n_cells, n_genes)`` float32/float64.
    tf_names
        Iterable of candidate transcription factor gene symbols.

    Returns
    -------
    pandas.DataFrame with columns ``['TF', 'target', 'importance']``,
    filtered to ``importance > 0``, sorted descending per target.
    Matches the schema produced by ``arboreto.algo.grnboost2``.
    """
    X, gene_names = _coerce_expression(expression)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    X = np.ascontiguousarray(X)

    # Warn on duplicate gene symbols — silent success can produce wrong results
    # when multiple columns map to the same TF name at aggregation time.
    dup_count = len(gene_names) - len(set(gene_names))
    if dup_count > 0:
        import warnings
        warnings.warn(
            f"{dup_count} duplicate gene name(s) in input; only the first match "
            f"will be used for each duplicate. Use AnnData.var_names_make_unique() "
            f"or deduplicate upstream.",
            UserWarning, stacklevel=2,
        )

    tfs_list = list(tf_names)
    if not tfs_list:
        import warnings
        warnings.warn("empty TF list — returning empty DataFrame", UserWarning, stacklevel=2)

    import sys, time
    if verbose:
        print(
            f"[rustscenic.grn] fitting GRNBoost2 — {X.shape[0]:,} cells × "
            f"{X.shape[1]:,} genes × {len(tfs_list)} TFs × "
            f"n_estimators={n_estimators} (early-stop window={early_stop_window}). "
            f"Running in parallel, this can take seconds to tens of minutes "
            f"depending on shape...",
            file=sys.stderr, flush=True,
        )
    t0 = time.monotonic()
    tfs, targets, importances = _grn_infer(
        X,
        list(gene_names),
        tfs_list,
        n_estimators,
        learning_rate,
        max_features,
        subsample,
        max_depth,
        early_stop_window,
        seed,
    )
    wall = time.monotonic() - t0
    df = pd.DataFrame({
        "TF": tfs,
        "target": targets,
        "importance": np.asarray(importances),
    })
    if verbose:
        print(
            f"[rustscenic.grn] done in {wall:.1f}s — {len(df):,} edges.",
            file=sys.stderr, flush=True,
        )
    return df


def _coerce_expression(expression):
    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        # AnnData
        X = expression.X.toarray() if hasattr(expression.X, "toarray") else np.asarray(expression.X)
        gene_names = list(expression.var_names)
        return X, gene_names
    if isinstance(expression, pd.DataFrame):
        return np.asarray(expression.values), list(expression.columns)
    if isinstance(expression, tuple) and len(expression) == 2:
        X, gene_names = expression
        return np.asarray(X), list(gene_names)
    raise TypeError(
        "expression must be AnnData, pandas.DataFrame, or (matrix, gene_names) tuple"
    )


def load_tfs(path: Union[str, Path]) -> list[str]:
    """Load a TF list (one gene symbol per line) from a text file.

    Strips whitespace (including \\r from Windows line endings) and skips
    blank lines / comment lines starting with ``#``.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out
