"""eRegulon assembly - the SCENIC+ endpoint.

A classical SCENIC regulon is just a TF + its co-expressed target genes.
An **eRegulon** adds chromatin grounding:

    TF -> enhancer (motif enrichment, from cistarget)
           |
           v
        enhancer -> gene (accessibility vs expression correlation,
                          from rustscenic.enhancer)

An eRegulon is therefore a three-way intersection:

    1. The TF's enriched motifs / peaks (cistarget output)
    2. The enhancer -> gene links on those peaks (enhancer output)
    3. The TF -> gene predictions from expression (GRN output, optional)

This module produces eRegulon records from those three rustscenic
outputs. The scale-sensitive table assembly runs in Rust; Python keeps
the public dataclass API and pandas/AnnData-facing glue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    eregulon_assemble as _eregulon_assemble,
    eregulon_assemble_f32 as _eregulon_assemble_f32,
    eregulon_assemble_summary as _eregulon_assemble_summary,
    eregulon_assemble_summary_f32 as _eregulon_assemble_summary_f32,
)
from rustscenic._stage_utils import (
    require_columns as _require_columns,
)


@dataclass
class ERegulon:
    """A chromatin-aware regulon.

    Attributes
    ----------
    tf
        Transcription factor gene symbol.
    enhancers
        Peak IDs whose accessibility correlates with at least one of
        this eRegulon's target genes (the peaks that carry the TF's motif
        and are linked to a target gene by rustscenic.enhancer).
    target_genes
        Gene symbols reachable from this TF via either (a) an enhancer
        link on a TF-enriched peak, or (b) direct GRN co-expression when
        ``use_grn_union=True`` in :func:`build_eregulons`.
    n_enhancer_links
        Number of (peak, gene) edges that survive all filters. A useful
        sanity signal - eRegulons with only 1-2 supporting links are
        weak; ≥ 5 is typical for a real regulon.
    motif_auc
        Mean motif-enrichment AUC across the supporting peaks (from
        cistarget). Higher = stronger TF-motif evidence.
    """

    tf: str
    enhancers: list[str] = field(default_factory=list)
    target_genes: list[str] = field(default_factory=list)
    n_enhancer_links: int = 0
    motif_auc: float = 0.0
    # target -> set of peaks linking to it. Preserves the per-edge support
    # so eregulons_to_dataframe can emit only real (enhancer, target) pairs
    # rather than the Cartesian product of (enhancers x target_genes).
    target_to_peaks: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.target_genes)


def build_eregulons(
    grn: pd.DataFrame | None,
    cistarget: pd.DataFrame,
    enhancer_links: pd.DataFrame,
    *,
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> list[ERegulon]:
    """Assemble eRegulons by intersecting the three SCENIC+ edge types.

    Parameters
    ----------
    grn
        ``rustscenic.grn.infer`` output. Columns: ``['TF', 'target',
        'importance']``. Pass ``None`` to skip GRN filtering (accept
        any enhancer-linked gene as a target).
    cistarget
        ``rustscenic.cistarget.enrich`` output with at minimum the
        columns ``['regulon', 'motif', 'auc']`` and - critically -
        ``'peak_id'`` or ``'region_id'`` identifying which peak each
        motif was enriched in. The ``'regulon'`` column should map
        back to the source TF (either a bare TF name or ``TF_regulon``).
    enhancer_links
        ``rustscenic.enhancer.link_peaks_to_genes`` output. Required
        columns: ``['peak_id', 'gene', 'correlation']``.
    min_target_genes
        Drop eRegulons with fewer than this many surviving target
        genes. Typical scenicplus cut-off is 10; we default to 5
        because our smaller-regulon test fixtures exercise this path.
    min_enhancer_links
        Drop eRegulons with fewer than this many (peak, gene) edges.
        Catches cases where a TF motif enrichment is supported by only
        one or two peak-gene pairs (weak evidence).
    cistarget_auc_threshold
        Filter cistarget rows below this AUC before assembling. Set to
        0 to keep everything.
    use_grn_intersection
        If ``True`` (default), a target gene is kept only if the
        enhancer-linked gene is also a predicted target of the TF in
        the GRN. If ``False``, keep all enhancer-linked genes
        regardless of GRN support. Set ``False`` when you trust the
        chromatin grounding more than the GBM predictions.

    Returns
    -------
    list[ERegulon] sorted by descending ``n_enhancer_links``.
    """
    return _build_eregulon_objects(
        grn,
        cistarget,
        enhancer_links,
        min_target_genes=min_target_genes,
        min_enhancer_links=min_enhancer_links,
        cistarget_auc_threshold=cistarget_auc_threshold,
        use_grn_intersection=use_grn_intersection,
    )


def _build_eregulon_objects(
    grn: pd.DataFrame | None,
    cistarget: pd.DataFrame,
    enhancer_links: pd.DataFrame,
    *,
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> list[ERegulon]:
    """Rust-backed assembly returning public ERegulon objects."""
    _validate_build_inputs(
        grn,
        cistarget,
        enhancer_links,
        use_grn_intersection=use_grn_intersection,
    )
    peak_col = _find_peak_column(cistarget)

    if use_grn_intersection:
        grn_tfs = grn["TF"].astype(str).tolist()
        grn_targets = grn["target"].astype(str).tolist()
    else:
        grn_tfs = []
        grn_targets = []

    cistarget_aucs = _cistarget_auc_arg(cistarget["auc"])
    (
        tf,
        enhancers,
        target_genes,
        n_enhancer_links,
        motif_auc,
        target_to_peaks,
        n_input_tfs,
        n_eregulons,
    ) = _eregulon_assemble_summary_arg(cistarget_aucs)(
        grn_tfs,
        grn_targets,
        cistarget["regulon"].astype(str).tolist(),
        cistarget[peak_col].astype(str).tolist(),
        cistarget_aucs,
        enhancer_links["peak_id"].astype(str).tolist(),
        enhancer_links["gene"].astype(str).tolist(),
        enhancer_links["correlation"].to_numpy(dtype=np.float32, copy=False),
        int(min_target_genes),
        int(min_enhancer_links),
        float(cistarget_auc_threshold),
        bool(use_grn_intersection),
    )

    _warn_if_catastrophic_drop_counts(
        n_output=int(n_eregulons),
        n_input_tfs=int(n_input_tfs),
        use_grn_intersection=use_grn_intersection,
    )
    return [
        ERegulon(
            tf=str(tf_i),
            enhancers=list(enhancers_i),
            target_genes=list(targets_i),
            n_enhancer_links=int(n_links_i),
            motif_auc=float(auc_i),
            target_to_peaks={
                str(target): list(peaks)
                for target, peaks in dict(target_map).items()
            },
        )
        for tf_i, enhancers_i, targets_i, n_links_i, auc_i, target_map in zip(
            tf,
            enhancers,
            target_genes,
            np.asarray(n_enhancer_links, dtype=np.uint32),
            np.asarray(motif_auc, dtype=np.float64),
            target_to_peaks,
            strict=True,
        )
    ]


def _build_eregulons_dataframe(
    grn: pd.DataFrame | None,
    cistarget: pd.DataFrame,
    enhancer_links: pd.DataFrame,
    *,
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> pd.DataFrame:
    """Rust-backed assembly returning the final long eRegulon table."""
    _validate_build_inputs(
        grn,
        cistarget,
        enhancer_links,
        use_grn_intersection=use_grn_intersection,
    )
    peak_col = _find_peak_column(cistarget)

    if use_grn_intersection:
        grn_tfs = grn["TF"].astype(str).tolist()
        grn_targets = grn["target"].astype(str).tolist()
    else:
        grn_tfs = []
        grn_targets = []

    cistarget_aucs = _cistarget_auc_arg(cistarget["auc"])
    (
        tf,
        enhancer,
        target_gene,
        n_enhancer_links,
        motif_auc,
        n_input_tfs,
        n_eregulons,
    ) = _eregulon_assemble_arg(cistarget_aucs)(
        grn_tfs,
        grn_targets,
        cistarget["regulon"].astype(str).tolist(),
        cistarget[peak_col].astype(str).tolist(),
        cistarget_aucs,
        enhancer_links["peak_id"].astype(str).tolist(),
        enhancer_links["gene"].astype(str).tolist(),
        enhancer_links["correlation"].to_numpy(dtype=np.float32, copy=False),
        int(min_target_genes),
        int(min_enhancer_links),
        float(cistarget_auc_threshold),
        bool(use_grn_intersection),
    )

    table = pd.DataFrame(
        {
            "tf": tf,
            "enhancer": enhancer,
            "target_gene": target_gene,
            "n_enhancer_links": np.asarray(n_enhancer_links, dtype=np.uint32),
            "motif_auc": np.asarray(motif_auc, dtype=np.float64),
        },
        columns=["tf", "enhancer", "target_gene", "n_enhancer_links", "motif_auc"],
    )
    table.attrs["n_eregulons"] = int(n_eregulons)
    table.attrs["n_input_tfs"] = int(n_input_tfs)
    _warn_if_catastrophic_drop_counts(
        n_output=int(n_eregulons),
        n_input_tfs=int(n_input_tfs),
        use_grn_intersection=use_grn_intersection,
    )
    return table


def _cistarget_auc_arg(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(copy=False)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("cistarget['auc'] must contain numeric values")
    return arr.astype(np.float64, copy=False)


def _eregulon_assemble_arg(values: np.ndarray):
    return _eregulon_assemble_f32 if values.dtype == np.float32 else _eregulon_assemble


def _eregulon_assemble_summary_arg(values: np.ndarray):
    return (
        _eregulon_assemble_summary_f32
        if values.dtype == np.float32
        else _eregulon_assemble_summary
    )


def _validate_build_inputs(
    grn: pd.DataFrame | None,
    cistarget: pd.DataFrame,
    enhancer_links: pd.DataFrame,
    *,
    use_grn_intersection: bool,
) -> None:
    _require_columns(cistarget, {"regulon", "auc"}, name="cistarget")
    _find_peak_column(cistarget)
    _require_columns(enhancer_links, {"peak_id", "gene", "correlation"}, name="enhancer_links")
    if use_grn_intersection:
        if grn is None:
            raise ValueError(
                "use_grn_intersection=True but grn is None. Pass the grn "
                "DataFrame or set use_grn_intersection=False."
            )
        _require_columns(grn, {"TF", "target"}, name="grn")


def _warn_if_catastrophic_drop_counts(
    *,
    n_output: int,
    n_input_tfs: int,
    use_grn_intersection: bool,
) -> None:
    import warnings

    if n_input_tfs == 0:
        return
    if n_output >= max(1, n_input_tfs // 2):
        return
    reason = (
        "try use_grn_intersection=False - the GRN ∩ enhancer-link step "
        "dropped most TFs, which usually means GRN gene names and "
        "enhancer_links `gene` column use different conventions (symbol "
        "vs ENSEMBL)"
        if use_grn_intersection
        else (
            "check that cistarget `peak_id` values overlap "
            "enhancer_links `peak_id` values - the two sets appear to "
            "be keyed differently"
        )
    )
    warnings.warn(
        f"build_eregulons kept only {n_output} of {n_input_tfs} input "
        f"TFs. {reason}.",
        UserWarning, stacklevel=3,
    )


def eregulons_to_dataframe(eregulons: Sequence[ERegulon]) -> pd.DataFrame:
    """Flatten a list of ERegulon objects into a long-format DataFrame.

    One row per (TF, enhancer, target_gene) triple. Useful for parquet
    persistence, joining back to downstream AUCell scoring, and quick
    cross-regulon aggregation.
    """
    rows = []
    for er in eregulons:
        # Emit one row per actual (peak, target_gene) edge, not the Cartesian
        # product of (enhancers x target_genes) - n_enhancer_links is the true
        # support count and the dataframe should match it.
        for tgt, peaks in er.target_to_peaks.items():
            rows.extend(
                (er.tf, enh, tgt, er.n_enhancer_links, er.motif_auc)
                for enh in peaks
            )
    return pd.DataFrame(
        rows,
        columns=["tf", "enhancer", "target_gene", "n_enhancer_links", "motif_auc"],
    )

def _find_peak_column(ct: pd.DataFrame) -> str:
    for candidate in ("peak_id", "region_id", "peak", "region"):
        if candidate in ct.columns:
            return candidate
    raise ValueError(
        "cistarget DataFrame must carry a peak / region identifier column. "
        "Expected one of: peak_id, region_id, peak, region."
    )


__all__ = ["ERegulon", "build_eregulons", "eregulons_to_dataframe"]
