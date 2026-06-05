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

from collections.abc import Iterable

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    cistarget_enrichment_from_projected_rankings_i16 as _cistarget_enrichment_from_projected_rankings_i16,
    cistarget_enrichment_from_projected_rankings_i32 as _cistarget_enrichment_from_projected_rankings_i32,
    cistarget_enrichment_from_projected_rankings_i64 as _cistarget_enrichment_from_projected_rankings_i64,
    cistarget_enrichment_from_rankings_i16 as _cistarget_enrichment_from_rankings_i16,
    cistarget_enrichment_from_rankings_i32 as _cistarget_enrichment_from_rankings_i32,
    cistarget_enrichment_from_rankings_i64 as _cistarget_enrichment_from_rankings_i64,
    cistarget_motif_annotation_prune_rows_filtered_f32 as _motif_annotation_prune_rows_filtered_f32,
    cistarget_motif_annotation_prune_rows_filtered_f64 as _motif_annotation_prune_rows_filtered_f64,
    cistarget_motif_annotation_prune_standard_rows_f32 as _motif_annotation_prune_standard_rows_f32,
    cistarget_motif_annotation_prune_standard_rows_f64 as _motif_annotation_prune_standard_rows_f64,
    cistarget_prune_regulon_targets_f32 as _prune_regulon_targets_f32,
    cistarget_prune_regulon_targets_f64 as _prune_regulon_targets_f64,
    cistarget_prune_regulon_targets_i16 as _prune_regulon_targets_i16,
    cistarget_prune_regulon_targets_i32 as _prune_regulon_targets_i32,
    cistarget_prune_regulon_targets_i64 as _prune_regulon_targets_i64,
    cistarget_prune_regulon_targets_projected_f32 as _prune_regulon_targets_projected_f32,
    cistarget_prune_regulon_targets_projected_f64 as _prune_regulon_targets_projected_f64,
    cistarget_prune_regulon_targets_projected_i16 as _prune_regulon_targets_projected_i16,
    cistarget_prune_regulon_targets_projected_i32 as _prune_regulon_targets_projected_i32,
    cistarget_prune_regulon_targets_projected_i64 as _prune_regulon_targets_projected_i64,
    cistarget_prune_regulon_targets_unranked as _prune_regulon_targets_unranked,
    cistarget_rankings_to_i32_f32 as _rankings_to_i32_f32,
    cistarget_rankings_to_i32_f64 as _rankings_to_i32_f64,
)
from rustscenic._stage_utils import (
    iter_regulon_pairs,
    prepare_regulon_indices,
    prepare_regulon_indices_with_coverage,
    require_columns as _require_columns,
    string_list,
    string_list_missing_empty,
    tf_from_regulon_name,
)


_NES_MIN_MOTIFS = 30
_tf_from_regulon_name = tf_from_regulon_name


def enrich(
    rankings: pd.DataFrame,
    regulons: Iterable,
    *,
    top_frac: float = 0.05,
    auc_threshold: float = 0.05,
    nes_threshold: float | None = None,
    rank_universe_size: int | None = None,
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
    rank_universe_size
        Original ranking-universe width when ``rankings`` has been projected to
        a subset of columns. Internal HPC pipeline hook; callers normally leave
        this as ``None``.

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
    motif_names = string_list(rankings.index)
    gene_names = string_list(rankings.columns)
    ranking_values = rankings.to_numpy(copy=False)
    if ranking_values.dtype == object:
        raise TypeError(
            "rankings DataFrame has dtype=object (likely non-numeric or "
            "wrong columns). Ensure rank values are numeric before passing."
        )

    reg_names, reg_gene_indices, reg_pairs, coverage, dropped_empty = (
        prepare_regulon_indices_with_coverage(gene_names, regulons)
    )

    # Silent-zero guardrails: this is the cistarget mirror of the cellxgene
    # bug Fuaad hit on aucell. If regulon genes don't match the rankings'
    # gene columns (for example: regulons built with HGNC symbols but the
    # aertslab v10 feather indexed by ENSEMBL; or mouse regulons passed
    # against an hg38 ranking), every lookup misses and the output is
    # silently empty. Warn loudly with a diagnostic the user can act on.
    from rustscenic._gene_resolution import warn_if_poor_coverage
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

    out = _enrichment_rows_from_rankings(
        ranking_values,
        motif_names,
        reg_names,
        reg_gene_indices,
        top_frac=top_frac,
        auc_threshold=auc_threshold,
        nes_threshold=nes_threshold,
        rank_universe_size=rank_universe_size,
    )
    out.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": _rankings_backend_symbols(
            ranking_values,
            rank_universe_size=rank_universe_size,
        ),
    }
    return out


def _enrichment_rows_from_rankings(
    ranking_values: np.ndarray,
    motif_names: list[str],
    reg_names: list[str],
    reg_gene_indices: list[list[int]],
    *,
    top_frac: float,
    auc_threshold: float,
    nes_threshold: float | None,
    rank_universe_size: int | None = None,
) -> pd.DataFrame:
    """Score cistarget directly from lower-is-better rankings in Rust."""
    rankings_arg, kernel, projected_universe = _rankings_kernel_arg(
        ranking_values,
        rank_universe_size=rank_universe_size,
    )
    if projected_universe is None:
        result = kernel(
            rankings_arg,
            motif_names,
            reg_names,
            reg_gene_indices,
            float(top_frac),
            float(auc_threshold),
            None if nes_threshold is None else float(nes_threshold),
            _NES_MIN_MOTIFS,
        )
    else:
        result = kernel(
            rankings_arg,
            motif_names,
            reg_names,
            reg_gene_indices,
            int(projected_universe),
            float(top_frac),
            float(auc_threshold),
            None if nes_threshold is None else float(nes_threshold),
            _NES_MIN_MOTIFS,
        )
    return _enrichment_result_to_dataframe(result, motif_names, reg_names)


def _rankings_kernel_arg(
    ranking_values: np.ndarray,
    *,
    rank_universe_size: int | None = None,
):
    """Choose the Rust ranking kernel for integer cistarget databases."""
    values = np.asarray(ranking_values)
    projected_universe = _projected_universe_arg(values, rank_universe_size)
    if projected_universe is not None:
        if values.dtype == np.int16:
            return values, _cistarget_enrichment_from_projected_rankings_i16, projected_universe
        if values.dtype == np.int32:
            return values, _cistarget_enrichment_from_projected_rankings_i32, projected_universe
        if values.dtype == np.int64:
            return values, _cistarget_enrichment_from_projected_rankings_i64, projected_universe
        if values.dtype == np.float32:
            return (
                _rankings_to_i32_f32(values),
                _cistarget_enrichment_from_projected_rankings_i32,
                projected_universe,
            )
        if values.dtype == np.float64:
            return (
                _rankings_to_i32_f64(values),
                _cistarget_enrichment_from_projected_rankings_i32,
                projected_universe,
            )
        raise TypeError(
            "rankings must contain integer rank values for cistarget enrichment"
        )
    if values.dtype == np.int16:
        return values, _cistarget_enrichment_from_rankings_i16, None
    if values.dtype == np.int32:
        return values, _cistarget_enrichment_from_rankings_i32, None
    if values.dtype == np.int64:
        return values, _cistarget_enrichment_from_rankings_i64, None
    if values.dtype == np.float32:
        return _rankings_to_i32_f32(values), _cistarget_enrichment_from_rankings_i32, None
    if values.dtype == np.float64:
        return _rankings_to_i32_f64(values), _cistarget_enrichment_from_rankings_i32, None
    raise TypeError(
        "rankings must contain integer rank values for cistarget enrichment"
    )


def _projected_universe_arg(
    ranking_values: np.ndarray,
    rank_universe_size: int | None,
) -> int | None:
    if rank_universe_size is None:
        return None
    universe = int(rank_universe_size)
    n_ranked_columns = int(np.asarray(ranking_values).shape[1])
    if universe < n_ranked_columns:
        raise ValueError(
            "rank_universe_size must be >= the number of ranking columns"
        )
    return universe if universe != n_ranked_columns else None


def _rankings_backend_symbols(
    ranking_values: np.ndarray,
    *,
    rank_universe_size: int | None = None,
) -> list[str]:
    values = np.asarray(ranking_values)
    projected = _projected_universe_arg(values, rank_universe_size) is not None
    prefix = (
        "cistarget_enrichment_from_projected_rankings"
        if projected else
        "cistarget_enrichment_from_rankings"
    )
    if values.dtype == np.int16:
        return [f"{prefix}_i16"]
    if values.dtype == np.int32:
        return [f"{prefix}_i32"]
    if values.dtype == np.int64:
        return [f"{prefix}_i64"]
    if values.dtype == np.float32:
        return [
            "cistarget_rankings_to_i32_f32",
            f"{prefix}_i32",
        ]
    if values.dtype == np.float64:
        return [
            "cistarget_rankings_to_i32_f64",
            f"{prefix}_i32",
        ]
    return []


def _enrichment_result_to_dataframe(
    result,
    motif_names: list[str],
    reg_names: list[str],
) -> pd.DataFrame:
    regulon, motif, auc, nes, zero_var_indices, too_few_motifs = result
    _warn_about_nes_issues(
        reg_names,
        zero_var_indices=np.asarray(zero_var_indices, dtype=np.uint32),
        too_few_motifs=bool(too_few_motifs),
        n_motifs=len(motif_names),
    )
    if len(regulon) == 0:
        return pd.DataFrame(columns=["regulon", "motif", "auc", "nes"])

    return pd.DataFrame({
        "regulon": list(regulon),
        "motif": list(motif),
        "auc": np.asarray(auc, dtype=np.float32),
        "nes": np.asarray(nes, dtype=np.float32),
    })


def _warn_about_nes_issues(
    reg_names: list[str],
    *,
    zero_var_indices: np.ndarray,
    too_few_motifs: bool,
    n_motifs: int,
) -> None:
    import warnings

    if too_few_motifs:
        warnings.warn(
            f"NES computed on only {n_motifs} motifs, below the "
            f"{_NES_MIN_MOTIFS}-motif floor where a z-score is reliable. "
            f"NES values are returned as NaN. Either pass a larger motif "
            f"ranking matrix or ignore the `nes` column for this run.",
            UserWarning, stacklevel=3,
        )
        return
    if zero_var_indices.size:
        bad = [reg_names[int(i)] for i in zero_var_indices[: len(reg_names)]]
        warnings.warn(
            f"NES undefined for {len(zero_var_indices)} regulons with zero AUC variance "
            f"across the motif universe (first few: {bad[:3]}). Such regulons "
            f"score identically on every motif, which usually means the "
            f"regulon genes are not present in the rankings (check coverage) "
            f"or the rankings collapse to a constant value. NES set to NaN; "
            f"AUC rows are preserved.",
            UserWarning, stacklevel=3,
        )


def prune_enriched_motifs(
    enriched: pd.DataFrame,
    motif_annotations: pd.DataFrame,
    *,
    motif_col: str | None = None,
    tf_col: str | None = None,
    auc_threshold: float | None = None,
    nes_threshold: float | None = None,
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

    if not isinstance(motif_annotations, pd.DataFrame):
        raise TypeError("motif_annotations must be a pandas DataFrame")
    if motif_annotations.empty:
        return pd.DataFrame(
            columns=list(enriched.columns) + ["tf", "annotation_tf"]
        )
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
    annotation_tfs = string_list_missing_empty(motif_annotations[tf_col])

    standard_columns = {"regulon", "motif", "auc", "nes"}
    if all(col in standard_columns for col in enriched.columns):
        auc_values, nes_values, standard_kernel = _motif_annotation_standard_kernel_arg(
            enriched["auc"],
            enriched["nes"] if "nes" in enriched.columns else None,
        )
        backend_symbol = _motif_annotation_standard_backend_symbol(standard_kernel)
        (
            regulon_values,
            motif_values,
            auc_out,
            nes_out,
            tf_values,
            annotation_tf,
        ) = standard_kernel(
            string_list(enriched["regulon"]),
            string_list(enriched["motif"]),
            auc_values,
            nes_values,
            string_list(motif_annotations[motif_col]),
            annotation_tfs,
            auc_threshold,
            nes_threshold,
            bool(case_sensitive),
        )
        if len(regulon_values) == 0:
            out = pd.DataFrame(
                columns=list(enriched.columns) + ["tf", "annotation_tf"]
            )
            out.attrs["rust_backend"] = {
                "engine": "rust",
                "symbols": [backend_symbol],
            }
            return out
        out_cols = {
            "regulon": regulon_values,
            "motif": motif_values,
            "auc": auc_out,
            "tf": list(tf_values),
            "annotation_tf": list(annotation_tf),
        }
        if nes_out is not None:
            out_cols["nes"] = nes_out
        out = pd.DataFrame(
            out_cols,
            columns=list(enriched.columns) + ["tf", "annotation_tf"],
        ).reset_index(drop=True)
        out.attrs["rust_backend"] = {
            "engine": "rust",
            "symbols": [backend_symbol],
        }
        return out

    auc_values, nes_values, kernel = _motif_annotation_prune_kernel_arg(
        enriched["auc"],
        enriched["nes"] if nes_threshold is not None else None,
    )
    backend_symbol = _motif_annotation_prune_backend_symbol(kernel)
    row_ix, tf_values, annotation_tf = kernel(
        string_list(enriched["regulon"]),
        string_list(enriched["motif"]),
        auc_values,
        nes_values,
        string_list(motif_annotations[motif_col]),
        annotation_tfs,
        auc_threshold,
        nes_threshold,
        bool(case_sensitive),
    )
    if len(row_ix) == 0:
        out = pd.DataFrame(
            columns=list(enriched.columns) + ["tf", "annotation_tf"]
        )
        out.attrs["rust_backend"] = {
            "engine": "rust",
            "symbols": [backend_symbol],
        }
        return out

    out = enriched.take(row_ix, axis=0).reset_index(drop=True)
    out["tf"] = list(tf_values)
    out["annotation_tf"] = list(annotation_tf)
    out.attrs["rust_backend"] = {
        "engine": "rust",
        "symbols": [backend_symbol],
    }
    return out


def prune_regulons(
    enriched: pd.DataFrame,
    regulons: Iterable,
    motif_annotations: pd.DataFrame,
    *,
    rankings: pd.DataFrame | None = None,
    top_frac: float = 0.05,
    auc_threshold: float | None = None,
    nes_threshold: float | None = None,
    min_genes: int = 1,
    motif_col: str | None = None,
    tf_col: str | None = None,
    case_sensitive: bool = False,
    rank_universe_size: int | None = None,
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
    candidate_pairs = list(iter_regulon_pairs(regulons))
    if not candidate_pairs:
        return {}
    candidate_names = [name for name, _ in candidate_pairs]
    candidate_genes = [genes for _, genes in candidate_pairs]

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
        rank_universe = _projected_universe_arg(
            rankings.to_numpy(copy=False),
            rank_universe_size,
        )
        n_ranked_columns = (
            int(rankings.shape[1]) if rank_universe is None else rank_universe
        )
        rank_cutoff = max(1, int(np.ceil(top_frac * n_ranked_columns)))
        ranking_values, kernel = _prune_rankings_kernel_arg(
            rankings,
            projected=rank_universe is not None,
        )
        names, genes = kernel(
            ranking_values,
            string_list(rankings.index),
            string_list(rankings.columns),
            candidate_names,
            candidate_genes,
            string_list(pruned_motifs["regulon"]),
            string_list(pruned_motifs["motif"]),
            rank_cutoff,
            int(min_genes),
        )
        return dict(zip(names, genes, strict=True))

    names, genes = _prune_regulon_targets_unranked(
        candidate_names,
        candidate_genes,
        string_list(pruned_motifs["regulon"]),
        int(min_genes),
    )
    return dict(zip(names, genes, strict=True))


def _motif_annotation_prune_kernel_arg(
    auc: pd.Series,
    nes: pd.Series | None,
):
    auc_values = auc.to_numpy(copy=False)
    nes_values = None if nes is None else nes.to_numpy(copy=False)
    if not np.issubdtype(auc_values.dtype, np.number):
        raise TypeError("enriched['auc'] must contain numeric values")
    if nes_values is not None and not np.issubdtype(nes_values.dtype, np.number):
        raise TypeError("enriched['nes'] must contain numeric values")
    if auc_values.dtype == np.float32 and (
        nes_values is None or nes_values.dtype == np.float32
    ):
        return auc_values, nes_values, _motif_annotation_prune_rows_filtered_f32
    if auc_values.dtype == np.float64 and (
        nes_values is None or nes_values.dtype == np.float64
    ):
        return auc_values, nes_values, _motif_annotation_prune_rows_filtered_f64
    return (
        auc_values.astype(np.float64, copy=False),
        None if nes_values is None else nes_values.astype(np.float64, copy=False),
        _motif_annotation_prune_rows_filtered_f64,
    )


def _motif_annotation_prune_backend_symbol(kernel) -> str:
    if kernel is _motif_annotation_prune_rows_filtered_f32:
        return "cistarget_motif_annotation_prune_rows_filtered_f32"
    if kernel is _motif_annotation_prune_rows_filtered_f64:
        return "cistarget_motif_annotation_prune_rows_filtered_f64"
    raise RuntimeError("unknown motif annotation pruning kernel")


def _motif_annotation_standard_kernel_arg(
    auc: pd.Series,
    nes: pd.Series | None,
):
    auc_values = auc.to_numpy(copy=False)
    nes_values = None if nes is None else nes.to_numpy(copy=False)
    if auc_values.dtype == np.float32 and (
        nes_values is None or nes_values.dtype == np.float32
    ):
        return auc_values, nes_values, _motif_annotation_prune_standard_rows_f32
    return (
        auc_values.astype(np.float64, copy=False),
        None if nes_values is None else nes_values.astype(np.float64, copy=False),
        _motif_annotation_prune_standard_rows_f64,
    )


def _motif_annotation_standard_backend_symbol(kernel) -> str:
    if kernel is _motif_annotation_prune_standard_rows_f32:
        return "cistarget_motif_annotation_prune_standard_rows_f32"
    if kernel is _motif_annotation_prune_standard_rows_f64:
        return "cistarget_motif_annotation_prune_standard_rows_f64"
    raise RuntimeError("unknown standard motif annotation pruning kernel")


def _prune_regulons_backend_symbols(
    rankings: pd.DataFrame | None,
    *,
    rank_universe_size: int | None = None,
) -> list[str]:
    if rankings is None:
        return ["cistarget_prune_regulon_targets_unranked"]
    if not isinstance(rankings, pd.DataFrame):
        raise TypeError("rankings must be a pandas DataFrame")
    values = rankings.to_numpy(copy=False)
    projected = _projected_universe_arg(values, rank_universe_size) is not None
    _, kernel = _prune_rankings_kernel_arg(rankings, projected=projected)
    return [_prune_rankings_backend_symbol(kernel)]


def _prune_rankings_kernel_arg(
    rankings: pd.DataFrame,
    *,
    projected: bool = False,
):
    if not isinstance(rankings, pd.DataFrame):
        raise TypeError("rankings must be a pandas DataFrame")
    values = rankings.to_numpy(copy=False)
    if values.dtype == object:
        raise TypeError("rankings DataFrame has dtype=object")
    if values.dtype == np.int16:
        return values, (
            _prune_regulon_targets_projected_i16
            if projected else _prune_regulon_targets_i16
        )
    if values.dtype == np.int32:
        return values, (
            _prune_regulon_targets_projected_i32
            if projected else _prune_regulon_targets_i32
        )
    if values.dtype == np.int64:
        return values, (
            _prune_regulon_targets_projected_i64
            if projected else _prune_regulon_targets_i64
        )
    if np.issubdtype(values.dtype, np.integer):
        if values.dtype.itemsize <= np.dtype(np.int32).itemsize and not np.issubdtype(
            values.dtype, np.unsignedinteger
        ):
            return values.astype(np.int32, copy=False), (
                _prune_regulon_targets_projected_i32
                if projected else _prune_regulon_targets_i32
            )
        if values.dtype.itemsize < np.dtype(np.int64).itemsize:
            return values.astype(np.int64, copy=False), (
                _prune_regulon_targets_projected_i64
                if projected else _prune_regulon_targets_i64
            )
        raise TypeError(
            "rankings integer dtype is wider than the supported signed int64 "
            "path; cast rankings to int64 before pruning"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("rankings must contain numeric rank values")
    if values.dtype == np.float32:
        return values, (
            _prune_regulon_targets_projected_f32
            if projected else _prune_regulon_targets_f32
        )
    if values.dtype == np.float64:
        return values, (
            _prune_regulon_targets_projected_f64
            if projected else _prune_regulon_targets_f64
        )
    return values.astype(np.float64, copy=False), (
        _prune_regulon_targets_projected_f64
        if projected else _prune_regulon_targets_f64
    )


def _prune_rankings_backend_symbol(kernel) -> str:
    if kernel is _prune_regulon_targets_i16:
        return "cistarget_prune_regulon_targets_i16"
    if kernel is _prune_regulon_targets_i32:
        return "cistarget_prune_regulon_targets_i32"
    if kernel is _prune_regulon_targets_i64:
        return "cistarget_prune_regulon_targets_i64"
    if kernel is _prune_regulon_targets_f32:
        return "cistarget_prune_regulon_targets_f32"
    if kernel is _prune_regulon_targets_f64:
        return "cistarget_prune_regulon_targets_f64"
    if kernel is _prune_regulon_targets_projected_i16:
        return "cistarget_prune_regulon_targets_projected_i16"
    if kernel is _prune_regulon_targets_projected_i32:
        return "cistarget_prune_regulon_targets_projected_i32"
    if kernel is _prune_regulon_targets_projected_i64:
        return "cistarget_prune_regulon_targets_projected_i64"
    if kernel is _prune_regulon_targets_projected_f32:
        return "cistarget_prune_regulon_targets_projected_f32"
    if kernel is _prune_regulon_targets_projected_f64:
        return "cistarget_prune_regulon_targets_projected_f64"
    raise RuntimeError("unknown regulon target pruning kernel")


def _prune_rankings_backend_symbol_for_values(values: np.ndarray) -> str:
    if values.dtype == object:
        raise TypeError("rankings DataFrame has dtype=object")
    if values.dtype == np.int16:
        return "cistarget_prune_regulon_targets_i16"
    if values.dtype == np.int32:
        return "cistarget_prune_regulon_targets_i32"
    if values.dtype == np.int64:
        return "cistarget_prune_regulon_targets_i64"
    if np.issubdtype(values.dtype, np.integer):
        if values.dtype.itemsize <= np.dtype(np.int32).itemsize and not np.issubdtype(
            values.dtype, np.unsignedinteger
        ):
            return "cistarget_prune_regulon_targets_i32"
        if values.dtype.itemsize < np.dtype(np.int64).itemsize:
            return "cistarget_prune_regulon_targets_i64"
        raise TypeError(
            "rankings integer dtype is wider than the supported signed int64 "
            "path; cast rankings to int64 before pruning"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("rankings must contain numeric rank values")
    if values.dtype == np.float32:
        return "cistarget_prune_regulon_targets_f32"
    return "cistarget_prune_regulon_targets_f64"


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
