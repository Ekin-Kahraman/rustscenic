"""Gene-name resolution + input-validation helpers shared across modules.

Problem this solves: different upstream conventions for what appears in
``AnnData.var_names`` silently break gene-symbol-based regulon lookup.

  - scanpy-native h5ads usually put **gene symbols** in ``var_names``.
  - cellxgene-curated h5ads (schema v4+) put **ENSEMBL IDs** in
    ``var_names`` and **gene symbols** in ``var["feature_name"]``.
  - 10x h5 outputs sometimes put ENSEMBL IDs in ``var_names`` and
    gene symbols in ``var["gene_symbols"]`` (case-dependent by loader).

Without this helper, passing a cellxgene atlas through
``rustscenic.aucell.score(adata, regulons)`` silently scores to near
zero because the regulon symbols (``SOX6``, ``FOXJ1`` …) never match
the ENSEMBL IDs in ``var_names``. The AUC matrix *looks* computed;
it's just structurally empty. This is how atlas-scale runs produce
garbage.

The three helpers here close three silent-failure modes:

1. ``resolve_gene_names(adata)`` - auto-detects cellxgene /
   ENSEMBL-id-as-varname datasets and returns the best-match
   gene-symbol list, emitting a ``UserWarning`` so the switch is
   visible in logs.

2. ``regulon_coverage(gene_names, regulons)`` - reports per-regulon
   how many of the regulon's genes are actually present in the
   dataset.

3. ``warn_if_poor_coverage(coverage, threshold)`` - emits a warning
   when ≥ one regulon has lost most of its genes to the lookup. The
   existing ``all regulons dropped`` warning only fires at 0 %
   overlap, so partial losses (which still produce near-zero AUCs)
   were silent.

Plus ``warn_if_likely_unnormalized(X)`` - flags raw-count input that
should have gone through ``normalize_total`` + ``log1p`` first.
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from rustscenic._rustscenic import (
    gene_duplicate_summary as _gene_duplicate_summary,
    gene_dedupe_dense_f32 as _gene_dedupe_dense_f32,
    gene_dedupe_dense_f64 as _gene_dedupe_dense_f64,
    gene_dedupe_dense_i32 as _gene_dedupe_dense_i32,
    gene_dedupe_dense_i64 as _gene_dedupe_dense_i64,
    gene_dedupe_dense_u32 as _gene_dedupe_dense_u32,
    gene_dedupe_dense_u64 as _gene_dedupe_dense_u64,
    gene_dedupe_sparse_csc_f32 as _gene_dedupe_sparse_csc_f32,
    gene_dedupe_sparse_csc_f64 as _gene_dedupe_sparse_csc_f64,
    gene_dedupe_sparse_csc_i32 as _gene_dedupe_sparse_csc_i32,
    gene_dedupe_sparse_csc_i64 as _gene_dedupe_sparse_csc_i64,
    gene_dedupe_sparse_csc_u32 as _gene_dedupe_sparse_csc_u32,
    gene_dedupe_sparse_csc_u64 as _gene_dedupe_sparse_csc_u64,
    stage_regulon_coverage_counts as _stage_regulon_coverage_counts,
)


# ENSEMBL gene-ID pattern - covers ENSG (human), ENSMUSG (mouse),
# ENSRNOG (rat), ENSDARG (zebrafish), etc. Optional version suffix.
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{8,}(\.\d+)?$")

# Ordered list of var columns to check when var_names look like ENSEMBL IDs.
# Matches the common conventions across cellxgene, 10x Cell Ranger, and
# scanpy-loaded h5ads in practice.
_SYMBOL_COLUMN_CANDIDATES = (
    "feature_name",
    "gene_symbol",
    "gene_symbols",
    "symbol",
    "Symbol",
    "Gene",
    "gene_name",
)


def _looks_like_ensembl(names: Sequence[str], min_fraction: float = 0.8) -> bool:
    """Return True if ≥ ``min_fraction`` of the first 20 names match ENSEMBL pattern."""
    sample = list(names[:20]) if len(names) > 20 else list(names)
    if not sample:
        return False
    hits = sum(1 for n in sample if _ENSEMBL_RE.match(str(n)))
    return hits >= len(sample) * min_fraction


def resolve_gene_names(adata, *, quiet: bool = False) -> list[str]:
    """Return the best-match gene-symbol list for an AnnData.

    If ``var_names`` look like ENSEMBL IDs *and* ``adata.var`` has a
    recognised symbol column, return that column's values. Otherwise
    return ``list(adata.var_names)`` unchanged.

    Emits a ``UserWarning`` when the swap happens (suppressible with
    ``quiet=True``) so the user sees which column is being used.

    Parameters
    ----------
    adata
        An ``AnnData`` object (anything with ``.var_names`` and ``.var``).
    quiet
        If ``True``, suppress the swap notice. Intended for internal
        callers that already log it.

    Returns
    -------
    list[str] - gene names suitable for regulon lookup.
    """
    names = list(adata.var_names)
    if not _looks_like_ensembl(names):
        return names

    # var_names look like ENSEMBL IDs - look for a symbol column.
    var = getattr(adata, "var", None)
    if var is None or not hasattr(var, "columns"):
        return names

    for col in _SYMBOL_COLUMN_CANDIDATES:
        if col in var.columns:
            swapped = [str(v) for v in var[col]]
            # Sanity check: the symbol column must not itself look like
            # ENSEMBL IDs (e.g. a dataset that duplicated the ID column
            # under a different name).
            if _looks_like_ensembl(swapped):
                continue
            if not quiet:
                warnings.warn(
                    f"var_names look like ENSEMBL IDs (e.g. {names[0]!r}); using "
                    f"`var[{col!r}]` for gene-symbol matching (cellxgene/10x "
                    f"convention). First three swaps: "
                    f"{list(zip(names[:3], swapped[:3]))}.",
                    UserWarning, stacklevel=2,
                )
            return swapped

    # No symbol column found. If var_names are VERSIONED ENSEMBL, strip
    # versions - downstream regulons are almost always unversioned, so
    # the match rate goes from 0 to ~100%.
    versioned = sum(1 for n in names[:20] if "." in str(n))
    if versioned >= min(10, len(names) // 2):
        stripped = [str(n).split(".", 1)[0] for n in names]
        if not quiet:
            warnings.warn(
                f"var_names are versioned ENSEMBL IDs (e.g. {names[0]!r}) "
                f"and no gene-symbol column was found. Stripping versions "
                f"({stripped[0]!r}) so unversioned regulons can match. "
                f"Add `var['feature_name']` to preserve versions.",
                UserWarning, stacklevel=2,
            )
        return stripped
    return names


def regulon_coverage(
    gene_names: Sequence[str],
    regulons: Iterable[tuple[str, Iterable[str]]],
) -> dict[str, tuple[int, int]]:
    """Count per-regulon how many genes are found in ``gene_names``.

    Returns ``{regulon_name: (n_matched, n_total)}``. Useful both for
    emitting coverage warnings and for returning a diagnostic alongside
    the AUC matrix.
    """
    names: list[str] = []
    regulon_genes: list[list[str]] = []
    for name, genes in regulons:
        names.append(str(name))
        regulon_genes.append([str(g) for g in genes])
    matched, total = _stage_regulon_coverage_counts(
        [str(g) for g in gene_names],
        regulon_genes,
    )
    return {
        name: (int(n_matched), int(n_total))
        for name, n_matched, n_total in zip(names, matched, total, strict=True)
    }


def warn_if_poor_coverage(
    coverage: Mapping[str, tuple[int, int]],
    *,
    threshold: float = 0.5,
    stacklevel: int = 2,
) -> None:
    """Emit a ``UserWarning`` listing regulons with < ``threshold`` overlap.

    This catches the silent-near-zero case that the existing
    ``all regulons dropped`` warning misses: a regulon with 3 out of 50
    genes still runs, and still produces a floor-bound AUC score.
    """
    poor = [
        (name, matched, total)
        for name, (matched, total) in coverage.items()
        if total > 0 and matched / total < threshold
    ]
    if not poor:
        return
    examples = ", ".join(f"{n} ({m}/{t})" for n, m, t in poor[:3])
    warnings.warn(
        f"{len(poor)} of {len(coverage)} regulons have < "
        f"{int(threshold * 100)}% of their genes present in the expression "
        f"matrix (first: {examples}). Their AUCell / GRN scores will be "
        f"dominated by the floor. Common cause: ENSEMBL-ID-vs-symbol "
        f"mismatch between regulons and `var_names`; "
        f"`rustscenic._gene_resolution.resolve_gene_names(adata)` handles the "
        f"cellxgene convention automatically.",
        UserWarning, stacklevel=stacklevel,
    )


def warn_if_likely_unnormalized(X, *, max_threshold: float = 50.0, stacklevel: int = 2) -> None:
    """Warn if ``X`` looks like raw UMI counts rather than log-normalised expression.

    pyscenic / rustscenic AUCell + GRN both assume log-normalised input
    (scanpy's ``normalize_total`` + ``log1p``). Raw counts produce a
    different ranking and silently wrong regulon activities.

    Heuristic: a log1p-normalised matrix rarely exceeds ~15; raw UMI
    counts routinely hit thousands. ``max > 50`` is a conservative
    threshold between the two regimes.
    """
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            if X.nnz == 0:
                return
            max_val = float(X.max())
        else:
            arr = np.asarray(X)
            if arr.size == 0:
                return
            max_val = float(arr.max())
    except Exception:
        return  # don't fail the user's pipeline on a diagnostic

    warn_if_max_likely_unnormalized(
        max_val,
        max_threshold=max_threshold,
        stacklevel=stacklevel,
    )


def warn_if_max_likely_unnormalized(
    max_val: float | None,
    *,
    max_threshold: float = 50.0,
    stacklevel: int = 2,
) -> None:
    """Warn from a Rust-collected expression maximum without rescanning in Python."""
    if max_val is None:
        return
    try:
        max_val = float(max_val)
    except Exception:
        return
    import math
    if not math.isfinite(max_val) or max_val <= max_threshold:
        return
    warnings.warn(
        f"expression matrix max value is {max_val:.1f}; rustscenic expects "
        f"log-normalised input. If this is raw UMI counts, run "
        f"`scanpy.pp.normalize_total(adata, target_sum=1e4)` followed by "
        f"`scanpy.pp.log1p(adata)` before calling rustscenic.",
        UserWarning, stacklevel=stacklevel,
    )


def dedupe_by_symbol(X, gene_names: Sequence[str]):
    """Collapse duplicate gene-symbol columns by summing their expression.

    When an ENSEMBL → symbol swap produces duplicate symbols (e.g.
    several transcripts collapsing onto the same HGNC symbol, or
    pseudogenes sharing a parent name), downstream regulon lookup is
    ambiguous. The scanpy / limma convention is to sum the columns:
    total per-gene expression aggregated across transcripts. This
    preserves signal; the alternative (drop all but one) silently
    loses half the gene's reads.

    Returns
    -------
    (X_deduped, unique_gene_names) - X may be sparse (scipy.sparse) or
    dense numpy; the return type matches the input.
    """
    names_list = list(gene_names)
    dup_count, _ = duplicate_gene_summary(names_list)
    if dup_count == 0:
        return X, names_list

    import scipy.sparse as sp
    if not sp.issparse(X):
        arr = np.asarray(X)
        dense_kernel = _dense_dedupe_kernel(arr.dtype)
        if dense_kernel is not None:
            out, unique_names = dense_kernel(arr, names_list)
            return np.asarray(out), list(unique_names)
        raise TypeError(
            "dedupe_by_symbol only supports numeric float32/float64/int32/int64/"
            "uint32/uint64 dense arrays on the Rust-backed path"
        )

    if sp.issparse(X):
        out, rust_names = _dedupe_sparse_by_symbol(X, names_list)
        if out is not None:
            return out, rust_names
        raise TypeError(
            "dedupe_by_symbol only supports numeric float32/float64/int32/int64/"
            "uint32/uint64 sparse arrays on the Rust-backed path"
        )

    raise TypeError(
        "dedupe_by_symbol expects a dense numpy array or scipy sparse matrix"
    )


def duplicate_gene_summary(
    gene_names: Sequence[str],
    *,
    max_examples: int = 3,
) -> tuple[int, list[str]]:
    """Return duplicate gene count and example names from the Rust stage helper."""
    duplicate_count, examples = _gene_duplicate_summary(list(gene_names), int(max_examples))
    return int(duplicate_count), list(examples)


def dedupe_backend_symbol_for_matrix(X) -> str | None:
    """Return the Rust dedupe kernel symbol that ``dedupe_by_symbol`` will use."""
    import scipy.sparse as sp

    dtype = np.dtype(X.dtype)
    suffix = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(np.uint32): "u32",
        np.dtype(np.uint64): "u64",
    }.get(dtype)
    if suffix is None:
        return None
    if sp.issparse(X):
        return f"gene_dedupe_sparse_csc_{suffix}"
    return f"gene_dedupe_dense_{suffix}"


def _dedupe_sparse_by_symbol(X, gene_names: list[str]):
    import scipy.sparse as sp

    sparse_kernel = _sparse_dedupe_kernel(X.dtype)
    if sparse_kernel is None:
        return None, None

    original_format = X.getformat()
    X_csc = X.tocsc(copy=False)
    X_csc.sum_duplicates()
    X_csc.sort_indices()
    data, indices, indptr, unique_names = sparse_kernel(
        X_csc.indptr,
        X_csc.indices,
        X_csc.data,
        int(X_csc.shape[0]),
        gene_names,
    )
    out = sp.csc_matrix(
        (data, indices, indptr),
        shape=(X_csc.shape[0], len(unique_names)),
        dtype=X_csc.dtype,
    )
    if original_format != "csc":
        out = out.asformat(original_format)
    return out, list(unique_names)


def _dense_dedupe_kernel(dtype):
    dtype = np.dtype(dtype)
    return {
        np.dtype(np.float32): _gene_dedupe_dense_f32,
        np.dtype(np.float64): _gene_dedupe_dense_f64,
        np.dtype(np.int32): _gene_dedupe_dense_i32,
        np.dtype(np.int64): _gene_dedupe_dense_i64,
        np.dtype(np.uint32): _gene_dedupe_dense_u32,
        np.dtype(np.uint64): _gene_dedupe_dense_u64,
    }.get(dtype)


def _sparse_dedupe_kernel(dtype):
    dtype = np.dtype(dtype)
    return {
        np.dtype(np.float32): _gene_dedupe_sparse_csc_f32,
        np.dtype(np.float64): _gene_dedupe_sparse_csc_f64,
        np.dtype(np.int32): _gene_dedupe_sparse_csc_i32,
        np.dtype(np.int64): _gene_dedupe_sparse_csc_i64,
        np.dtype(np.uint32): _gene_dedupe_sparse_csc_u32,
        np.dtype(np.uint64): _gene_dedupe_sparse_csc_u64,
    }.get(dtype)


def diagnose_zero_tf_overlap(
    tf_names: Sequence[str], gene_names: Sequence[str]
) -> str:
    """Return an actionable hint when a TF list has zero gene overlap.

    Figures out WHICH convention mismatch is most likely - case
    (SPI1 vs Spi1), ENSEMBL vs symbol, or versioned vs unversioned
    ENSEMBL - and tells the user how to fix it. Far more useful than
    a generic "check your conventions" message.
    """
    tf_list = list(tf_names)
    gene_list = list(gene_names)
    if not tf_list or not gene_list:
        return "TF list and/or gene list is empty."

    gene_set = set(gene_list)

    # Case mismatch: TFs are uppercase, genes are titlecase (or reverse)
    title_versions = {t: t.title() if t.isupper() else None for t in tf_list[:10]}
    title_hits = sum(
        1 for t, alt in title_versions.items() if alt and alt in gene_set
    )
    if title_hits >= len(tf_list[:10]) // 2:
        return (
            f"TF list looks UPPERCASE (human HGNC) but data looks "
            f"Titlecase (mouse MGI). Try `[tf.title() for tf in tf_names]` "
            f"or pass the mouse TF list from `rustscenic.data.tfs('mouse')`."
        )
    upper_versions = {t: t.upper() if not t.isupper() else None for t in tf_list[:10]}
    upper_hits = sum(
        1 for t, alt in upper_versions.items() if alt and alt in gene_set
    )
    if upper_hits >= len(tf_list[:10]) // 2:
        return (
            f"TF list looks Titlecase (mouse MGI) but data looks "
            f"UPPERCASE (human HGNC). Try `[tf.upper() for tf in tf_names]` "
            f"or pass the human TF list from `rustscenic.data.tfs('human')`."
        )

    # ENSEMBL convention mismatch
    sample_gene = gene_list[0]
    sample_tf = tf_list[0]
    if _ENSEMBL_RE.match(str(sample_gene)) and not _ENSEMBL_RE.match(str(sample_tf)):
        return (
            f"data var_names are ENSEMBL IDs (e.g. {sample_gene!r}) but TF "
            f"list is gene symbols (e.g. {sample_tf!r}). The auto-swap "
            f"(cellxgene convention) couldn't find a `feature_name` / "
            f"`gene_symbol` column in `adata.var` - add one, or pass "
            f"ENSEMBL-IDed TF names."
        )
    if _ENSEMBL_RE.match(str(sample_tf)) and not _ENSEMBL_RE.match(str(sample_gene)):
        return (
            f"TF list is ENSEMBL IDs (e.g. {sample_tf!r}) but data is "
            f"symbols (e.g. {sample_gene!r}). Convert TFs to symbols before "
            f"calling - e.g. via biomart or `mygene.MyGeneInfo().getgenes()`."
        )

    # Versioned-vs-unversioned ENSEMBL IDs
    if _ENSEMBL_RE.match(str(sample_tf)) and _ENSEMBL_RE.match(str(sample_gene)):
        tf_has_version = "." in str(sample_tf)
        gene_has_version = "." in str(sample_gene)
        if tf_has_version != gene_has_version:
            if tf_has_version:
                return (
                    f"TFs are versioned ENSEMBL (e.g. {sample_tf!r}) but "
                    f"data is unversioned ({sample_gene!r}). Strip versions: "
                    f"`[t.split('.')[0] for t in tf_names]`."
                )
            return (
                f"TFs are unversioned ENSEMBL ({sample_tf!r}) but data is "
                f"versioned ({sample_gene!r}). Strip on the data side: "
                f"`adata.var_names = [v.split('.')[0] for v in adata.var_names]`."
            )

    # Fallback - give a concrete example so at least debugging is fast
    return (
        f"TF example: {sample_tf!r}; data example: {sample_gene!r}. "
        f"No automatic hint available - check symbol convention and species."
    )


__all__ = [
    "resolve_gene_names",
    "regulon_coverage",
    "warn_if_poor_coverage",
    "warn_if_likely_unnormalized",
    "warn_if_max_likely_unnormalized",
    "dedupe_by_symbol",
    "dedupe_backend_symbol_for_matrix",
    "duplicate_gene_summary",
    "diagnose_zero_tf_overlap",
]
