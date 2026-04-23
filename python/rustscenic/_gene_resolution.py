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

1. ``resolve_gene_names(adata)`` — auto-detects cellxgene /
   ENSEMBL-id-as-varname datasets and returns the best-match
   gene-symbol list, emitting a ``UserWarning`` so the switch is
   visible in logs.

2. ``regulon_coverage(gene_names, regulons)`` — reports per-regulon
   how many of the regulon's genes are actually present in the
   dataset.

3. ``warn_if_poor_coverage(coverage, threshold)`` — emits a warning
   when ≥ one regulon has lost most of its genes to the lookup. The
   existing ``all regulons dropped`` warning only fires at 0 %
   overlap, so partial losses (which still produce near-zero AUCs)
   were silent.

Plus ``warn_if_likely_unnormalized(X)`` — flags raw-count input that
should have gone through ``normalize_total`` + ``log1p`` first.
"""
from __future__ import annotations

import re
import warnings
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


# ENSEMBL gene-ID pattern — covers ENSG (human), ENSMUSG (mouse),
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
    list[str] — gene names suitable for regulon lookup.
    """
    names = list(adata.var_names)
    if not _looks_like_ensembl(names):
        return names

    # var_names look like ENSEMBL IDs — look for a symbol column.
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
    return names


def regulon_coverage(
    gene_names: Sequence[str],
    regulons: Iterable[Tuple[str, Iterable[str]]],
) -> dict[str, Tuple[int, int]]:
    """Count per-regulon how many genes are found in ``gene_names``.

    Returns ``{regulon_name: (n_matched, n_total)}``. Useful both for
    emitting coverage warnings and for returning a diagnostic alongside
    the AUC matrix.
    """
    gene_set = set(gene_names)
    out: dict[str, Tuple[int, int]] = {}
    for name, genes in regulons:
        g = list(genes)
        out[name] = (sum(1 for x in g if x in gene_set), len(g))
    return out


def warn_if_poor_coverage(
    coverage: Mapping[str, Tuple[int, int]],
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

    if max_val > max_threshold:
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
    (X_deduped, unique_gene_names) — X may be sparse (scipy.sparse) or
    dense numpy; the return type matches the input.
    """
    names_list = list(gene_names)
    from collections import defaultdict

    index_of: dict[str, int] = {}
    groups: list[list[int]] = []
    for i, g in enumerate(names_list):
        if g in index_of:
            groups[index_of[g]].append(i)
        else:
            index_of[g] = len(groups)
            groups.append([i])

    if len(groups) == len(names_list):
        return X, names_list

    unique_names = [names_list[g[0]] for g in groups]

    import scipy.sparse as sp
    if sp.issparse(X):
        # Build a gene × unique_gene aggregation matrix and right-multiply.
        n_genes = len(names_list)
        n_unique = len(groups)
        rows = []
        cols = []
        for dst, src_indices in enumerate(groups):
            for src in src_indices:
                rows.append(src)
                cols.append(dst)
        data = np.ones(len(rows), dtype=X.dtype)
        agg = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_genes, n_unique), dtype=X.dtype
        )
        return X @ agg, unique_names

    arr = np.asarray(X)
    out = np.zeros((arr.shape[0], len(groups)), dtype=arr.dtype)
    for dst, src_indices in enumerate(groups):
        if len(src_indices) == 1:
            out[:, dst] = arr[:, src_indices[0]]
        else:
            out[:, dst] = arr[:, src_indices].sum(axis=1)
    return out, unique_names


__all__ = [
    "resolve_gene_names",
    "regulon_coverage",
    "warn_if_poor_coverage",
    "warn_if_likely_unnormalized",
    "dedupe_by_symbol",
]
