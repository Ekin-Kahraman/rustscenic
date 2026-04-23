"""Motif enrichment scoring (pycistarget replacement, core algorithm).

pycistarget's central operation: for each regulon (a set of target genes),
compute the AUC of recovery of those genes against a motif ranking. High AUC
means the motif is enriched for that regulon. The aertslab feather-format
ranking databases (hg38_10kb_up_and_down_tss.feather etc.) provide the
per-motif gene rankings.

We reuse rustscenic's aucell core — the algorithm is mathematically identical,
just applied to motif rankings rather than per-cell expression rankings. This
module is a thin wrapper that:

  1. Accepts a motif-ranking matrix (motifs × genes, where cell[m, g] = rank
     of gene g for motif m; lower rank = stronger association).
  2. Accepts regulons (gene sets).
  3. For each (regulon, motif) pair, computes recovery AUC.
  4. Returns enriched pairs above threshold.

We do NOT bundle the feather DB reader (the aertslab databases are 10-50 GB
and require hg38/mm10 coordinates; loading them is a separate concern).
Callers can load feather files via pyarrow and pass the resulting DataFrame.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from rustscenic._rustscenic import aucell_score as _aucell_score


def enrich(
    rankings: pd.DataFrame,
    regulons: Iterable,
    *,
    top_frac: float = 0.05,
    auc_threshold: float = 0.05,
) -> pd.DataFrame:
    """Compute motif-regulon enrichment AUCs.

    Parameters
    ----------
    rankings
        DataFrame with motifs as rows (index = motif names) and genes as columns.
        Values = rank of gene for that motif (lower rank = stronger association).
        Use ``load_aertslab_feather()`` to load the aertslab feather DB in the
        correct orientation.
    regulons
        Iterable of `(name, gene_list)` tuples, or objects with `.name` + `.genes`.
    top_frac
        Fraction of top-ranked genes per motif used as AUC cutoff (default 0.05,
        matches pycisTopic/pycistarget).
    auc_threshold
        Minimum AUC to report a regulon-motif pair as enriched. Set to 0 to
        return all scores.

    Returns
    -------
    pandas.DataFrame with columns [regulon, motif, auc], sorted descending
    by AUC. Only rows where auc >= auc_threshold.
    """
    # Expect motifs as rows, genes as columns. Refuse to guess orientation —
    # a wrong guess silently produces an empty result.
    motif_names = list(rankings.index)
    gene_names = list(rankings.columns)
    if rankings.values.dtype == object:
        raise TypeError(
            "rankings DataFrame has dtype=object (likely non-numeric or "
            "wrong columns). Ensure rank values are numeric before passing."
        )
    if not np.all(np.isfinite(rankings.values)):
        raise ValueError(
            "rankings contain NaN or Inf values — motif enrichment is "
            "undefined on non-finite ranks. Load the feather file cleanly "
            "(aertslab feathers are int16) and check for upstream corruption."
        )
    # Convert rankings (lower = better) into "expression" (higher = better)
    # by negating — AUCell's recovery AUC expects descending sort by value.
    # Use -rank so smaller rank maps to larger pseudo-expression.
    scores = -rankings.values.astype(np.float32)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    reg_names: list[str] = []
    reg_gene_indices: list[list[int]] = []
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

    # Silent-zero guardrails: this is the cistarget mirror of the cellxgene
    # bug Fuaad hit on aucell. If regulon genes don't match the rankings'
    # gene columns (for example: regulons built with HGNC symbols but the
    # aertslab v10 feather indexed by ENSEMBL; or mouse regulons passed
    # against an hg38 ranking), every lookup misses and the output is
    # silently empty. Warn loudly with a diagnostic the user can act on.
    from rustscenic._gene_resolution import regulon_coverage, warn_if_poor_coverage
    coverage = regulon_coverage(gene_names, reg_pairs)
    warn_if_poor_coverage(coverage, stacklevel=3)
    if dropped_empty > 0 and not reg_names:
        import warnings
        warnings.warn(
            f"all {dropped_empty} regulons dropped — none of their genes appear "
            f"in the rankings DataFrame columns. Common causes: (1) rankings "
            f"indexed by ENSEMBL while regulons use gene symbols; (2) species "
            f"mismatch between rankings (e.g. hg38) and regulons (e.g. mouse "
            f"MGI); (3) rankings orientation swapped (motifs-in-cols vs "
            f"motifs-in-rows). First regulon genes: "
            f"{reg_pairs[0][1][:3] if reg_pairs else 'n/a'}. First 3 ranking "
            f"columns: {gene_names[:3]}.",
            UserWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=["regulon", "motif", "auc"])

    # Run the per-motif (as "cells") AUC scoring
    auc = _aucell_score(np.ascontiguousarray(scores), reg_names, reg_gene_indices, top_frac)
    # auc shape: (n_motifs, n_regulons)
    auc_df = pd.DataFrame(np.asarray(auc), index=motif_names, columns=reg_names)

    # Stack to long form, filter by threshold
    long = auc_df.stack().reset_index()
    long.columns = ["motif", "regulon", "auc"]
    long = long[long["auc"] >= auc_threshold].sort_values("auc", ascending=False).reset_index(drop=True)
    return long[["regulon", "motif", "auc"]]


def _coerce_regulon(reg):
    if isinstance(reg, tuple) and len(reg) == 2:
        name, genes = reg
        return str(name), list(genes)
    if isinstance(reg, dict):
        if "name" in reg and "genes" in reg:
            return str(reg["name"]), list(reg["genes"])
    name = getattr(reg, "name", None) or getattr(reg, "transcription_factor", None)
    if hasattr(reg, "gene2weight"):
        genes = list(reg.gene2weight.keys())
    elif hasattr(reg, "genes"):
        genes = list(reg.genes)
    else:
        raise TypeError(f"cannot extract regulon genes from {type(reg).__name__}")
    if name is None:
        raise TypeError(f"regulon has no .name")
    return str(name), genes


def load_aertslab_feather(path) -> pd.DataFrame:
    """Load an aertslab motif-ranking feather file.

    The feather file typically has `motifs` or `features` as one column and the
    rest as genes. Returns a DataFrame indexed by motif name.
    """
    import pyarrow.feather as feather
    df = feather.read_feather(path)
    # aertslab feathers have an "features" or "motifs" column
    for key in ("features", "motifs"):
        if key in df.columns:
            df = df.set_index(key)
            break
    return df
