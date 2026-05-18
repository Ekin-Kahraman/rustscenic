"""Motif enrichment scoring (pycistarget replacement, core algorithm).

pycistarget's central operation: for each regulon (a set of target genes),
compute the AUC of recovery of those genes against a motif ranking. High AUC
means the motif is enriched for that regulon. The aertslab feather-format
ranking databases (hg38_10kb_up_and_down_tss.feather etc.) provide the
per-motif gene rankings.

We reuse rustscenic's aucell core - the algorithm is mathematically identical,
just applied to motif rankings rather than per-cell expression rankings. This
module is a thin wrapper that:

  1. Accepts a motif-ranking matrix (motifs × genes, where cell[m, g] = rank
     of gene g for motif m; lower rank = stronger association).
  2. Accepts regulons (gene sets).
  3. For each (regulon, motif) pair, computes recovery AUC.
  4. Returns enriched pairs above threshold.
  5. Optionally prunes enriched motifs through motif-to-TF annotations
     to produce final TF-supported regulons.

We do NOT bundle the large aertslab ranking databases. Callers can load
feather files via pyarrow and pass the resulting DataFrame. Motif annotation
tables are separate inputs; use ``prune_regulons`` when you need
pycistarget-style motif-annotation pruning rather than candidate GRN regulons.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from rustscenic._rustscenic import aucell_score as _aucell_score


_NES_MIN_MOTIFS = 30


def enrich(
    rankings: pd.DataFrame,
    regulons: Iterable,
    *,
    top_frac: float = 0.05,
    auc_threshold: float = 0.05,
    nes_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Compute motif-regulon enrichment AUCs and NES.

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
        matches pyscenic / ctxcore gene-mode). pycistarget region-mode uses
        ``auc_threshold=0.005`` instead; pass that explicitly for region-mode
        parity.
    auc_threshold
        Minimum AUC to report a regulon-motif pair as enriched. Set to 0 to
        return all scores.
    nes_threshold
        Optional minimum NES to keep. ``None`` (default) returns all rows that
        pass ``auc_threshold`` and is backwards-compatible with v0.4.3 callers.
        Set to ``3.0`` to match the canonical pyscenic / pycistarget cutoff.
        When NES is undefined for a regulon (zero standard deviation across
        motifs, or fewer than 30 motifs), NES is recorded as ``NaN`` and a
        warning is emitted; ``NaN`` rows are dropped by any ``nes_threshold``
        filter but preserved when ``nes_threshold is None``.

    Returns
    -------
    pandas.DataFrame with columns [regulon, motif, auc, nes], sorted descending
    by AUC. Only rows where auc >= auc_threshold (and, if set, nes >= nes_threshold).
    NES is computed as the per-regulon population z-score of AUC across the
    full motif universe (matching pyscenic ``transform.py`` and pycistarget
    ``motif_enrichment_cistarget.py``).
    """
    # Expect motifs as rows, genes as columns. Refuse to guess orientation -
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
            "rankings contain NaN or Inf values - motif enrichment is "
            "undefined on non-finite ranks. Load the feather file cleanly "
            "(aertslab feathers are int16) and check for upstream corruption."
        )
    # Convert rankings (lower = better) into "expression" (higher = better)
    # by negating - AUCell's recovery AUC expects descending sort by value.
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
            f"all {dropped_empty} regulons dropped - none of their genes appear "
            f"in the rankings DataFrame columns. Common causes: (1) rankings "
            f"indexed by ENSEMBL while regulons use gene symbols; (2) species "
            f"mismatch between rankings (e.g. hg38) and regulons (e.g. mouse "
            f"MGI); (3) rankings orientation swapped (motifs-in-cols vs "
            f"motifs-in-rows). First regulon genes: "
            f"{reg_pairs[0][1][:3] if reg_pairs else 'n/a'}. First 3 ranking "
            f"columns: {gene_names[:3]}.",
            UserWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=["regulon", "motif", "auc", "nes"])

    # Run the per-motif (as "cells") AUC scoring
    auc = _aucell_score(np.ascontiguousarray(scores), reg_names, reg_gene_indices, top_frac)
    # auc shape: (n_motifs, n_regulons)
    auc_arr = np.asarray(auc)
    auc_df = pd.DataFrame(auc_arr, index=motif_names, columns=reg_names)

    # Compute per-regulon NES (population z-score across the motif universe).
    # Matches pyscenic transform.py and pycistarget motif_enrichment_cistarget.py
    # (per-regulon mean and std, ddof=0). Two guards on top of upstream:
    #   - n_motifs < _NES_MIN_MOTIFS: z-score is unreliable, emit NaN + warn.
    #   - std == 0 for a regulon: divide-by-zero, emit NaN + warn.
    # Both keep the AUC rows visible to the caller; only NES is undefined.
    nes_arr = _compute_nes(auc_arr, n_motifs=len(motif_names), reg_names=reg_names)
    nes_df = pd.DataFrame(nes_arr, index=motif_names, columns=reg_names)

    # Stack both AUC and NES to long form before any filtering, so the
    # two columns stay aligned on (motif, regulon) under the same
    # MultiIndex order.
    auc_long = auc_df.stack()
    nes_long = nes_df.stack().reindex(auc_long.index)
    long = pd.DataFrame({"auc": auc_long.values, "nes": nes_long.values})
    long["motif"] = auc_long.index.get_level_values(0)
    long["regulon"] = auc_long.index.get_level_values(1)
    long = long.reset_index(drop=True)
    long = long[long["auc"] >= auc_threshold]
    if nes_threshold is not None:
        # NaN NES rows are dropped here by design - NES is undefined for them.
        long = long[long["nes"].notna() & (long["nes"] >= nes_threshold)]
    long = long.sort_values("auc", ascending=False).reset_index(drop=True)
    return long[["regulon", "motif", "auc", "nes"]]


def _compute_nes(
    auc_arr: np.ndarray, *, n_motifs: int, reg_names: list,
) -> np.ndarray:
    """Per-regulon population z-score of AUC across the motif axis.

    Returns an array shaped like ``auc_arr`` with NaN entries wherever NES is
    undefined (small motif universe, or zero-variance regulon column).
    """
    import warnings
    nes = np.full_like(auc_arr, fill_value=np.nan, dtype=np.float32)
    if n_motifs < _NES_MIN_MOTIFS:
        warnings.warn(
            f"NES computed on only {n_motifs} motifs, below the "
            f"{_NES_MIN_MOTIFS}-motif floor where a z-score is reliable. "
            f"NES values are returned as NaN. Either pass a larger motif "
            f"ranking matrix or ignore the `nes` column for this run.",
            UserWarning, stacklevel=3,
        )
        return nes
    means = auc_arr.mean(axis=0)
    stds = auc_arr.std(axis=0, ddof=0)
    # Threshold catches both literal-zero variance and float32 accumulation
    # noise (~6e-8 for AUC in [0,1]). A real signal in AUC variance sits
    # several orders of magnitude above this, so 1e-6 is a safe floor.
    zero_var = stds < 1e-6
    if zero_var.any():
        bad = [reg_names[i] for i in np.where(zero_var)[0]]
        warnings.warn(
            f"NES undefined for {len(bad)} regulons with zero AUC variance "
            f"across the motif universe (first few: {bad[:3]}). Such regulons "
            f"score identically on every motif, which usually means the "
            f"regulon genes are not present in the rankings (check coverage) "
            f"or the rankings collapse to a constant value. NES set to NaN; "
            f"AUC rows are preserved.",
            UserWarning, stacklevel=3,
        )
    safe_stds = np.where(zero_var, 1.0, stds)
    nes_full = (auc_arr - means) / safe_stds
    nes_full[:, zero_var] = np.nan
    return nes_full.astype(np.float32)


def prune_enriched_motifs(
    enriched: pd.DataFrame,
    motif_annotations: pd.DataFrame,
    *,
    motif_col: Optional[str] = None,
    tf_col: Optional[str] = None,
    auc_threshold: Optional[float] = None,
    nes_threshold: Optional[float] = None,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Filter enriched motif rows through motif-to-TF annotations.

    This is the pycistarget-style pruning step that turns motif enrichment
    evidence into TF-supported regulons: a row survives only when the motif
    enriched for ``TF_regulon`` is annotated to that same TF.

    Parameters
    ----------
    enriched
        ``enrich`` output with columns ``['regulon', 'motif', 'auc']`` plus
        an optional ``nes`` column (present when produced by v0.4.4+ enrich).
    motif_annotations
        DataFrame mapping motifs to TF names. Common column names such as
        ``motif``, ``motif_id``, ``features`` and ``TF``, ``gene_name`` are
        auto-detected; pass ``motif_col`` / ``tf_col`` to override.
    auc_threshold
        Optional minimum AUC to keep before applying annotations. ``None``
        preserves the rows already returned by ``enrich``.
    nes_threshold
        Optional minimum NES to keep before applying annotations. Requires
        the ``enriched`` DataFrame to contain an ``nes`` column. Rows with
        NaN NES are dropped by this filter. ``3.0`` matches the canonical
        pyscenic / pycistarget cutoff.
    case_sensitive
        Match TF names case-sensitively. Default ``False`` tolerates common
        upper/lower-case differences in annotation exports while preserving
        the candidate regulon's original TF symbol in the result.

    Returns
    -------
    DataFrame containing the enriched rows that pass motif annotation support,
    plus ``tf`` and ``annotation_tf`` columns.
    """
    _require_columns(enriched, {"regulon", "motif", "auc"}, name="enriched")
    if nes_threshold is not None and "nes" not in enriched.columns:
        raise ValueError(
            "nes_threshold was set but the enriched DataFrame has no `nes` "
            "column. Re-run rustscenic.cistarget.enrich with v0.4.4+ to "
            "populate the NES column, or pass nes_threshold=None."
        )
    if enriched.empty:
        return pd.DataFrame(
            columns=list(enriched.columns) + ["tf", "annotation_tf"]
        )

    ct = enriched.copy()
    if auc_threshold is not None:
        ct = ct.loc[ct["auc"] >= auc_threshold].copy()
    if nes_threshold is not None:
        ct = ct.loc[ct["nes"].notna() & (ct["nes"] >= nes_threshold)].copy()
    if ct.empty:
        return pd.DataFrame(
            columns=list(enriched.columns) + ["tf", "annotation_tf"]
        )

    ann = _normalise_motif_annotations(
        motif_annotations,
        motif_col=motif_col,
        tf_col=tf_col,
        case_sensitive=case_sensitive,
    )

    ct["tf"] = ct["regulon"].map(_tf_from_regulon_name)
    ct["_motif_key"] = ct["motif"].astype(str)
    ct["_tf_key"] = ct["tf"].astype(str)
    if not case_sensitive:
        ct["_tf_key"] = ct["_tf_key"].str.lower()

    out = ct.merge(
        ann,
        on=["_motif_key", "_tf_key"],
        how="inner",
        sort=False,
    )
    out = out.drop(columns=["_motif_key", "_tf_key"])
    return out.reset_index(drop=True)


def prune_regulons(
    enriched: pd.DataFrame,
    regulons: Iterable,
    motif_annotations: pd.DataFrame,
    *,
    rankings: Optional[pd.DataFrame] = None,
    top_frac: float = 0.05,
    auc_threshold: Optional[float] = None,
    nes_threshold: Optional[float] = None,
    min_genes: int = 1,
    motif_col: Optional[str] = None,
    tf_col: Optional[str] = None,
    case_sensitive: bool = False,
) -> dict[str, list[str]]:
    """Create final motif-annotation-pruned regulons.

    Candidate regulons are first filtered to rows whose enriched motifs are
    annotated back to the source TF. When ``rankings`` is provided, each
    surviving regulon's targets are further restricted to genes recovered in
    the enriched motif's top-ranked window, matching the usual cistarget
    pruning semantics more closely than keeping the whole GRN top-N list.
    ``nes_threshold`` adds the canonical pyscenic / pycistarget NES filter
    on top of motif annotation matching; pass ``3.0`` for the standard cutoff.
    """
    candidate = {
        name: list(dict.fromkeys(genes))
        for name, genes in (_coerce_regulon(reg) for reg in regulons)
    }
    if not candidate:
        return {}

    pruned_motifs = prune_enriched_motifs(
        enriched,
        motif_annotations,
        motif_col=motif_col,
        tf_col=tf_col,
        auc_threshold=auc_threshold,
        nes_threshold=nes_threshold,
        case_sensitive=case_sensitive,
    )
    if pruned_motifs.empty:
        return {}

    if rankings is not None:
        _validate_rankings_for_pruning(rankings)
        rank_cutoff = max(1, int(np.ceil(top_frac * rankings.shape[1])))
    else:
        rank_cutoff = None

    kept: dict[str, set[str]] = {}
    for row in pruned_motifs.itertuples(index=False):
        regulon_name = str(getattr(row, "regulon"))
        genes = candidate.get(regulon_name)
        if not genes:
            continue
        if rankings is None:
            kept.setdefault(regulon_name, set()).update(genes)
            continue
        motif = str(getattr(row, "motif"))
        if motif not in rankings.index:
            continue
        ranks = rankings.loc[motif]
        top_genes = set(ranks.nsmallest(rank_cutoff).index)
        recovered = [
            g for g in genes
            if g in top_genes
        ]
        kept.setdefault(regulon_name, set()).update(recovered)

    out = {
        name: [g for g in candidate[name] if g in genes]
        for name, genes in kept.items()
        if len(genes) >= min_genes
    }
    return out


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


def _require_columns(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )


_REGULON_NAME_SUFFIXES = ("_regulon", "_extended", "_activator", "_repressor")
_REGULON_NAME_PARENS = ("(+)", "(-)")


def _tf_from_regulon_name(name: str) -> str:
    """Strip every recognised regulon-name suffix and polarity marker until
    the result is stable, so canonical scenicplus names (e.g.
    ``FOXP3_extended_regulon``, ``PAX5_extended(+)``) reduce to the bare TF
    symbol. The original implementation broke on the first match and left
    compound suffixes intact."""
    tf = str(name).strip()
    while True:
        prev = tf
        for suffix in _REGULON_NAME_SUFFIXES:
            if tf.endswith(suffix):
                tf = tf[: -len(suffix)].strip()
        for paren in _REGULON_NAME_PARENS:
            if tf.endswith(paren):
                tf = tf[:-3].strip()
        if tf == prev:
            return tf


def _normalise_motif_annotations(
    motif_annotations: pd.DataFrame,
    *,
    motif_col: Optional[str],
    tf_col: Optional[str],
    case_sensitive: bool,
) -> pd.DataFrame:
    if not isinstance(motif_annotations, pd.DataFrame):
        raise TypeError("motif_annotations must be a pandas DataFrame")
    if motif_annotations.empty:
        return pd.DataFrame(columns=["_motif_key", "_tf_key", "annotation_tf"])

    motif_col = motif_col or _find_annotation_column(
        motif_annotations,
        ["motif", "motifs", "motif_id", "motifid", "features", "#motif_id"],
        role="motif",
    )
    tf_col = tf_col or _find_annotation_column(
        motif_annotations,
        [
            "tf", "TF", "transcription_factor", "gene_name", "gene",
            "symbol", "tf_name", "factor",
        ],
        role="TF",
    )

    rows = []
    for rec in motif_annotations[[motif_col, tf_col]].itertuples(index=False):
        motif = str(rec[0])
        for tf in _split_annotation_tfs(rec[1]):
            key = tf if case_sensitive else tf.lower()
            rows.append((motif, key, tf))
    return (
        pd.DataFrame(rows, columns=["_motif_key", "_tf_key", "annotation_tf"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _find_annotation_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    role: str,
) -> str:
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        found = lower_to_col.get(cand.lower())
        if found is not None:
            return found
    raise ValueError(
        f"could not infer {role} column in motif_annotations. "
        f"Pass the column explicitly. Got columns: {list(df.columns)}"
    )


def _split_annotation_tfs(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    for sep in (";", ",", "|"):
        text = text.replace(sep, "/")
    return [part.strip() for part in text.split("/") if part.strip()]


def _validate_rankings_for_pruning(rankings: pd.DataFrame) -> None:
    if not isinstance(rankings, pd.DataFrame):
        raise TypeError("rankings must be a pandas DataFrame")
    if rankings.values.dtype == object:
        raise TypeError("rankings DataFrame has dtype=object")
    if not np.all(np.isfinite(rankings.values)):
        raise ValueError("rankings contain NaN or Inf values")


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
