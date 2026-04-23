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
from rustscenic._gene_resolution import (
    dedupe_by_symbol,
    warn_if_likely_unnormalized,
)


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

    warn_if_likely_unnormalized(X, stacklevel=3)

    # Duplicate gene symbols typically come from ENSEMBL → symbol
    # resolution (multiple transcripts collapsing). Sum columns so
    # regression sees one row per gene, not silently lose data.
    dup_count = len(gene_names) - len(set(gene_names))
    if dup_count > 0:
        import warnings
        from collections import Counter
        top_dupes = [n for n, c in Counter(gene_names).most_common(3) if c > 1]
        warnings.warn(
            f"{dup_count} duplicate gene name(s) after ENSEMBL→symbol "
            f"resolution (e.g. {top_dupes}). Summing expression across "
            f"duplicate symbols so GRN inputs are unambiguous. Pass the "
            f"AnnData through `rustscenic._gene_resolution.dedupe_by_symbol()` "
            f"upstream if you want full control.",
            UserWarning, stacklevel=2,
        )
        X, gene_names = dedupe_by_symbol(X, gene_names)
        X = np.ascontiguousarray(X.astype(np.float32, copy=False))

    tfs_list = list(tf_names)
    if not tfs_list:
        import warnings
        warnings.warn("empty TF list — returning empty DataFrame", UserWarning, stacklevel=2)

    # Report TF-list / gene-list overlap. Zero overlap is the specific
    # failure mode users hit on cellxgene-curated h5ads — the TF list is
    # gene symbols, var_names are ENSEMBL IDs. `_coerce_expression` now
    # auto-resolves the convention, but the user can still pass a
    # mismatched TF list (e.g. mouse TFs against a human dataset).
    gene_set = set(gene_names)
    tfs_present = [t for t in tfs_list if t in gene_set]
    if tfs_list and not tfs_present:
        import warnings
        from rustscenic._gene_resolution import diagnose_zero_tf_overlap
        hint = diagnose_zero_tf_overlap(tfs_list, gene_names)
        warnings.warn(
            f"none of the {len(tfs_list)} supplied TFs match any gene in the "
            f"expression matrix — returning empty DataFrame. {hint}",
            UserWarning, stacklevel=2,
        )
    elif tfs_list and len(tfs_present) < 0.2 * len(tfs_list):
        import warnings
        warnings.warn(
            f"only {len(tfs_present)} of {len(tfs_list)} supplied TFs are present "
            f"in the expression matrix. GRN will fit a very narrow regulator set. "
            f"Example missing TFs: {[t for t in tfs_list if t not in gene_set][:5]}.",
            UserWarning, stacklevel=2,
        )

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
    """Return ``(X_dense, gene_names)`` from AnnData / DataFrame / tuple input.

    For AnnData, gene names come from
    :func:`rustscenic._gene_resolution.resolve_gene_names`, which
    auto-detects cellxgene-style datasets (ENSEMBL IDs in ``var_names``,
    gene symbols in ``var["feature_name"]``) and swaps to the symbol
    column so user-supplied TF lists match.
    """
    from rustscenic._gene_resolution import resolve_gene_names
    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        # AnnData
        X = expression.X.toarray() if hasattr(expression.X, "toarray") else np.asarray(expression.X)
        gene_names = resolve_gene_names(expression)
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
