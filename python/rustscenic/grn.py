"""GRN inference (GRNBoost2 replacement).

Public API:
    rustscenic.grn.infer(adata_or_matrix, tf_names, ...) -> pd.DataFrame
"""
from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    grn_correlations_dense as _grn_correlations_dense,
    grn_correlations_sparse_csc as _grn_correlations_sparse_csc,
    grn_infer as _grn_infer,
    grn_infer_sparse_csc as _grn_infer_sparse_csc,
    pipeline_candidate_regulons_from_grn as _pipeline_candidate_regulons_from_grn,
)
from rustscenic._gene_resolution import (
    dedupe_backend_symbol_for_matrix,
    dedupe_by_symbol,
    duplicate_gene_summary,
    warn_if_max_likely_unnormalized,
)
from rustscenic._stage_utils import as_float32_array, as_float32_contiguous, string_list


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
    early_stop_mode: str = "arboreto",
    top_targets_per_tf: int | None = None,
    min_importance: float | None = None,
    seed: int = 777,
    target_block_size: int | None = None,
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
    top_targets_per_tf
        If set, keep only the top-K targets per TF by importance,
        descending. Useful on under-determined inputs (n_samples ≪
        n_features) where rustscenic's histogram-GBM emits more
        near-zero-importance edges than sklearn-backed pipelines and
        downstream regulon construction wants a sparser GRN.
    min_importance
        If set, drop edges with ``importance < min_importance`` before
        returning. Cheap floor filter; combine with ``top_targets_per_tf``
        for arboreto-like edge-density behaviour.
    target_block_size
        Retained for API compatibility. The flat target scheduler no longer
        blocks targets, so this value is validated but does not affect
        execution.
    early_stop_mode
        ``"arboreto"`` (default) uses the trailing mean of out-of-bag
        improvements, matching arboreto with modern scikit-learn semantics.
        ``"legacy_inbag"`` retains RustScenic's historical two-point in-bag
        MSE rule for reproducing older RustScenic artefacts. Set
        ``early_stop_window=0`` to disable either rule. Arboreto OOB stopping
        also has no OOB rows at ``subsample=1.0`` and therefore fits to the
        estimator ceiling.

    Returns
    -------
    pandas.DataFrame with columns ``['TF', 'target', 'importance']``,
    filtered to ``importance > 0``, sorted descending per target.
    Matches the schema produced by ``arboreto.algo.grnboost2``.
    """
    import scipy.sparse as sp

    X, gene_names = _coerce_expression(expression)
    if sp.issparse(X):
        X = _sparse_float32_csc(X)
    else:
        X = as_float32_array(X)

    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    early_stop_window = int(early_stop_window)
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    learning_rate = float(learning_rate)
    max_features = float(max_features)
    subsample = float(subsample)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if not math.isfinite(max_features) or not 0 < max_features <= 1:
        raise ValueError("max_features must be finite and in (0, 1]")
    if not math.isfinite(subsample) or not 0 < subsample <= 1:
        raise ValueError("subsample must be finite and in (0, 1]")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if early_stop_window < 0:
        raise ValueError("early_stop_window must be non-negative")

    if target_block_size is None:
        target_block_size_for_rust = 0
    else:
        target_block_size_for_rust = int(target_block_size)
        if target_block_size_for_rust < 1:
            raise ValueError("target_block_size must be None or a positive integer")
    early_stop_mode = str(early_stop_mode)
    if early_stop_mode not in {"arboreto", "legacy_inbag"}:
        raise ValueError(
            "early_stop_mode must be 'arboreto' or 'legacy_inbag', "
            f"got {early_stop_mode!r}"
        )

    # Duplicate gene symbols typically come from ENSEMBL → symbol
    # resolution (multiple transcripts collapsing). Sum columns so
    # regression sees one row per gene, not silently lose data.
    dup_count, top_dupes = duplicate_gene_summary(gene_names)
    backend_symbols = ["gene_duplicate_summary"]
    if dup_count > 0:
        import warnings
        warnings.warn(
            f"{dup_count} duplicate gene name(s) after ENSEMBL→symbol "
            f"resolution (e.g. {top_dupes}). Summing expression across "
            f"duplicate symbols so GRN inputs are unambiguous. Pass the "
            f"AnnData through `rustscenic._gene_resolution.dedupe_by_symbol()` "
            f"upstream if you want full control.",
            UserWarning, stacklevel=2,
        )
        dedupe_symbol = dedupe_backend_symbol_for_matrix(X)
        if dedupe_symbol is not None:
            backend_symbols.append(dedupe_symbol)
        X, gene_names = dedupe_by_symbol(X, gene_names)
        if sp.issparse(X):
            X = _sparse_float32_csc(X)
        else:
            X = as_float32_contiguous(X)

    tfs_list = string_list(tf_names)
    if not tfs_list:
        import warnings
        warnings.warn("empty TF list - returning empty DataFrame", UserWarning, stacklevel=2)

    import sys, time
    if verbose:
        print(
            f"[rustscenic.grn] fitting GRNBoost2 - {X.shape[0]:,} cells × "
            f"{X.shape[1]:,} genes × {len(tfs_list)} TFs × "
            f"n_estimators={n_estimators} (early-stop mode={early_stop_mode}, "
            f"window={early_stop_window}). "
            f"Running in parallel, this can take seconds to tens of minutes "
            f"depending on shape...",
            file=sys.stderr, flush=True,
        )
    t0 = time.monotonic()
    top_targets_for_rust = (
        None if top_targets_per_tf is None else int(top_targets_per_tf)
    )
    common_args = (
        string_list(gene_names),
        tfs_list,
        n_estimators,
        learning_rate,
        max_features,
        subsample,
        max_depth,
        early_stop_window,
        early_stop_mode,
        seed,
        target_block_size_for_rust,
        top_targets_for_rust,
        None if min_importance is None else float(min_importance),
    )
    if sp.issparse(X):
        backend_symbol = "grn_infer_sparse_csc"
        (
            tfs,
            targets,
            importances,
            raw_n,
            tf_present_count,
            missing_tfs,
            expression_max,
            fit_summary,
        ) = _grn_infer_sparse_csc(
            X.indptr,
            np.asarray(X.indices, dtype=np.int32),
            np.asarray(X.data, dtype=np.float32),
            int(X.shape[0]),
            int(X.shape[1]),
            *common_args,
        )
    else:
        backend_symbol = "grn_infer"
        (
            tfs,
            targets,
            importances,
            raw_n,
            tf_present_count,
            missing_tfs,
            expression_max,
            fit_summary,
        ) = _grn_infer(X, *common_args)
    wall = time.monotonic() - t0
    warn_if_max_likely_unnormalized(expression_max, stacklevel=3)

    # Rust resolves TF-list / gene-list overlap in the same pass used for GRN
    # fitting, avoiding a duplicate Python set scan over large gene tables.
    if tfs_list and tf_present_count == 0:
        import warnings
        from rustscenic._gene_resolution import diagnose_zero_tf_overlap
        hint = diagnose_zero_tf_overlap(tfs_list, gene_names)
        warnings.warn(
            f"none of the {len(tfs_list)} supplied TFs match any gene in the "
            f"expression matrix - returning empty DataFrame. {hint}",
            UserWarning, stacklevel=2,
        )
    elif tfs_list and tf_present_count < 0.2 * len(tfs_list):
        import warnings
        warnings.warn(
            f"only {tf_present_count} of {len(tfs_list)} supplied TFs are present "
            f"in the expression matrix. GRN will fit a very narrow regulator set. "
            f"Example missing TFs: {missing_tfs}.",
            UserWarning, stacklevel=2,
        )

    # Under-determined-input warning. With n_samples < ~50 (or
    # n_samples / n_features < 0.01), tree-builder RNG dominates over
    # signal: edge importances are noisy across both rustscenic and
    # arboreto, and per-edge rank correlation drops sharply (verified
    # empirically on Kamath n=10 pseudobulk vs PBMC n=2,700, see
    # validation/kamath_da/KAMATH_AUDIT.md).
    n_samples = X.shape[0]
    if tf_present_count and n_samples < 50:
        import warnings
        warnings.warn(
            f"only {n_samples} samples for {X.shape[1]:,} genes and "
            f"{tf_present_count} TFs. GRN edge rankings are unstable in this "
            f"regime regardless of implementation - a GBM with n_estimators="
            f"{n_estimators} memorises the training set trivially. Consider "
            f"running on cell-level (not pseudobulk) input, or apply "
            f"`top_targets_per_tf=...` / `min_importance=...` to extract a "
            f"sparser, more comparable edge set. See "
            f"validation/kamath_da/KAMATH_AUDIT.md for the analysis behind "
            f"this guidance.",
            UserWarning, stacklevel=2,
        )
    df = pd.DataFrame({
        "TF": tfs,
        "target": targets,
        "importance": np.asarray(importances),
    })
    df.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": backend_symbols + [backend_symbol],
    }
    df.attrs["grn_fit"] = {
        "early_stop_mode": early_stop_mode,
        "early_stop_window": int(early_stop_window),
        "n_estimators_ceiling": int(n_estimators),
        "subsample": float(subsample),
        **dict(fit_summary),
    }

    if verbose:
        if raw_n != len(df):
            print(
                f"[rustscenic.grn] done in {wall:.1f}s - fit emitted {raw_n:,} "
                f"edges, returning {len(df):,} after truncation "
                f"(top_targets_per_tf={top_targets_per_tf}, "
                f"min_importance={min_importance}).",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                f"[rustscenic.grn] done in {wall:.1f}s - {len(df):,} edges.",
                file=sys.stderr, flush=True,
            )
    return df


def add_correlation(
    adjacencies: pd.DataFrame,
    expression,
    *,
    rho_threshold: float = 0.03,
    mask_dropouts: bool = False,
) -> pd.DataFrame:
    """Add TF-target Pearson correlation and SCENIC regulation polarity.

    The returned frame preserves the adjacency row order and input columns,
    then adds ``regulation`` (``1`` activator, ``-1`` repressor, ``0``
    indeterminate) and ``rho``. With ``mask_dropouts=True``, only cells where
    both the TF and target are non-zero contribute to a pair's correlation,
    matching pySCENIC's historical dropout-masked option.
    """
    required = {"TF", "target", "importance"}
    missing_columns = required - set(adjacencies.columns)
    if missing_columns:
        raise ValueError(
            f"adjacencies is missing required columns: {sorted(missing_columns)}. "
            f"Got columns: {list(adjacencies.columns)}"
        )
    rho_threshold = float(rho_threshold)
    if not np.isfinite(rho_threshold) or rho_threshold <= 0.0:
        raise ValueError(
            f"rho_threshold must be finite and greater than zero, got {rho_threshold}"
        )

    import scipy.sparse as sp

    X, gene_names = _coerce_expression(expression)
    if sp.issparse(X):
        X = _sparse_float32_csc(X)
    else:
        X = as_float32_array(X)

    backend_symbols = ["gene_duplicate_summary"]
    dup_count, top_dupes = duplicate_gene_summary(gene_names)
    if dup_count > 0:
        import warnings

        warnings.warn(
            f"{dup_count} duplicate gene name(s) after ENSEMBL→symbol "
            f"resolution (e.g. {top_dupes}). Summing expression across "
            "duplicate symbols before TF-target correlation.",
            UserWarning,
            stacklevel=2,
        )
        dedupe_symbol = dedupe_backend_symbol_for_matrix(X)
        if dedupe_symbol is not None:
            backend_symbols.append(dedupe_symbol)
        X, gene_names = dedupe_by_symbol(X, gene_names)
        if sp.issparse(X):
            X = _sparse_float32_csc(X)
        else:
            X = as_float32_contiguous(X)

    out = adjacencies.copy()
    if out.empty:
        out["regulation"] = np.asarray([], dtype=np.int8)
        out["rho"] = np.asarray([], dtype=np.float64)
        out.attrs["rust_backend"] = {
            "engine": "rust",
            "symbols": backend_symbols,
        }
        return out

    gene_index = pd.Index(gene_names)
    edge_tfs = string_list(out["TF"])
    edge_targets = string_list(out["target"])
    tf_indices = gene_index.get_indexer(pd.Index(edge_tfs))
    target_indices = gene_index.get_indexer(pd.Index(edge_targets))
    missing_mask = (tf_indices < 0) | (target_indices < 0)
    if bool(np.any(missing_mask)):
        missing_indices = np.flatnonzero(missing_mask)
        examples = [
            f"{edge_tfs[int(index)]}->{edge_targets[int(index)]}"
            for index in missing_indices[:5]
        ]
        raise ValueError(
            f"{int(np.count_nonzero(missing_mask))} adjacency edge(s) reference genes missing "
            f"from the expression matrix (examples: {examples}). Use the same "
            "gene-resolved expression matrix for GRN inference and correlation."
        )

    tf_indices = np.asarray(tf_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if sp.issparse(X):
        backend_symbol = "grn_correlations_sparse_csc"
        rhos, regulations = _grn_correlations_sparse_csc(
            X.indptr,
            np.asarray(X.indices, dtype=np.int32),
            np.asarray(X.data, dtype=np.float32),
            int(X.shape[0]),
            int(X.shape[1]),
            tf_indices,
            target_indices,
            rho_threshold,
            bool(mask_dropouts),
        )
    else:
        backend_symbol = "grn_correlations_dense"
        rhos, regulations = _grn_correlations_dense(
            X,
            tf_indices,
            target_indices,
            rho_threshold,
            bool(mask_dropouts),
        )
    regulations_array = np.asarray(regulations, dtype=np.int8)
    out["regulation"] = regulations_array
    out["rho"] = np.asarray(rhos, dtype=np.float64)
    out.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": backend_symbols + [backend_symbol],
    }
    out.attrs["correlation"] = {
        "method": "pearson",
        "rho_threshold": rho_threshold,
        "mask_dropouts": bool(mask_dropouts),
        "activating_edges": int(np.count_nonzero(regulations_array > 0)),
        "repressing_edges": int(np.count_nonzero(regulations_array < 0)),
        "indeterminate_edges": int(np.count_nonzero(regulations_array == 0)),
    }
    return out


def build_regulons(
    correlated_adjacencies: pd.DataFrame,
    *,
    top_targets_per_tf: int = 50,
    min_targets: int = 10,
    include_repressors: bool = True,
) -> dict[str, list[str]]:
    """Build deterministic activator/repressor regulons from signed edges.

    Names use ``<TF>_activator`` and ``<TF>_repressor``. Neutral edges
    (``regulation == 0``) are excluded rather than silently assigned a sign.
    """
    required = {"TF", "target", "importance", "regulation"}
    missing_columns = required - set(correlated_adjacencies.columns)
    if "regulation" in missing_columns:
        raise ValueError(
            "correlated_adjacencies must contain a 'regulation' column; call "
            "rustscenic.grn.add_correlation(...) first"
        )
    if missing_columns:
        raise ValueError(
            "correlated_adjacencies is missing required columns: "
            f"{sorted(missing_columns)}. Got columns: "
            f"{list(correlated_adjacencies.columns)}"
        )
    for value in pd.unique(correlated_adjacencies["regulation"]):
        if (
            pd.isna(value)
            or isinstance(value, (bool, np.bool_))
            or value not in {-1, 0, 1}
        ):
            raise ValueError(
                "correlated_adjacencies['regulation'] must contain only "
                f"-1, 0, or 1; got {value!r}"
            )
    top_targets_per_tf = int(top_targets_per_tf)
    min_targets = int(min_targets)
    if top_targets_per_tf < 1:
        raise ValueError("top_targets_per_tf must be at least 1")
    if min_targets < 1:
        raise ValueError("min_targets must be at least 1")

    def one_polarity(frame: pd.DataFrame, suffix: str) -> dict[str, list[str]]:
        if frame.empty:
            return {}
        names, target_lists = _pipeline_candidate_regulons_from_grn(
            string_list(frame["TF"]),
            string_list(frame["target"]),
            np.asarray(frame["importance"].array, dtype=np.float64),
            top_targets_per_tf,
            min_targets,
        )
        return {
            f"{name[:-len('_regulon')]}_{suffix}": targets
            for name, targets in zip(names, target_lists, strict=True)
        }

    activating = one_polarity(
        correlated_adjacencies[correlated_adjacencies["regulation"] > 0],
        "activator",
    )
    if not include_repressors:
        return activating
    repressing = one_polarity(
        correlated_adjacencies[correlated_adjacencies["regulation"] < 0],
        "repressor",
    )
    return {**activating, **repressing}


def _coerce_expression(expression):
    """Return ``(X, gene_names)`` from AnnData / DataFrame / tuple input.

    For AnnData, gene names come from
    :func:`rustscenic._gene_resolution.resolve_gene_names`, which
    auto-detects cellxgene-style datasets (ENSEMBL IDs in ``var_names``,
    gene symbols in ``var["feature_name"]``) and swaps to the symbol
    column so user-supplied TF lists match.
    """
    from rustscenic._gene_resolution import resolve_gene_names
    import scipy.sparse as sp

    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        # AnnData. Handle backed ('r') mode explicitly - _CSRDataset /
        # _CSCDataset doesn't have .toarray() and np.asarray() on it
        # returns a 0-d array, triggering a cryptic IndexError downstream.
        if getattr(expression, "isbacked", False):
            import warnings
            warnings.warn(
                "AnnData is backed (disk-resident). Loading X into memory "
                "for GRN; sparse inputs remain sparse, but backed arrays "
                "must still be read before fitting.",
                UserWarning, stacklevel=3,
            )
            X_raw = expression.X[:]
        else:
            X_raw = expression.X
        X = X_raw if sp.issparse(X_raw) else np.asarray(X_raw)
        gene_names = resolve_gene_names(expression)
        return X, gene_names
    if isinstance(expression, pd.DataFrame):
        return np.asarray(expression.values), string_list(expression.columns)
    if isinstance(expression, tuple) and len(expression) == 2:
        X, gene_names = expression
        if sp.issparse(X):
            return X, string_list(gene_names)
        return np.asarray(X), string_list(gene_names)
    raise TypeError(
        "expression must be AnnData, pandas.DataFrame, or (matrix, gene_names) tuple"
    )


def _sparse_float32_csc(X):
    """Return a CSC float32 matrix with duplicate entries summed."""
    X = X.tocsc(copy=False).astype(np.float32, copy=False)
    if hasattr(X, "sum_duplicates"):
        X.sum_duplicates()
    if hasattr(X, "sort_indices"):
        X.sort_indices()
    if hasattr(X, "eliminate_zeros"):
        X.eliminate_zeros()
    return X


def load_tfs(path: str | Path) -> list[str]:
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
