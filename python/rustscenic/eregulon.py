"""eRegulon assembly — the SCENIC+ endpoint.

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
outputs. It is pure Python pandas / dataclass bookkeeping — no
computation — because every numerical step already ran upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import pandas as pd


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
        sanity signal — eRegulons with only 1-2 supporting links are
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

    def __len__(self) -> int:
        return len(self.target_genes)


def build_eregulons(
    grn: Optional[pd.DataFrame],
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
        columns ``['regulon', 'motif', 'auc']`` and — critically —
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
    # Validate inputs
    _require_columns(cistarget, {"regulon", "auc"}, name="cistarget")
    peak_col = _find_peak_column(cistarget)
    _require_columns(enhancer_links, {"peak_id", "gene", "correlation"}, name="enhancer_links")
    if use_grn_intersection:
        if grn is None:
            raise ValueError(
                "use_grn_intersection=True but grn is None. Pass the grn "
                "DataFrame or set use_grn_intersection=False."
            )
        _require_columns(grn, {"TF", "target"}, name="grn")

    # Filter cistarget rows to passing motif enrichments.
    ct = cistarget.loc[cistarget["auc"] >= cistarget_auc_threshold].copy()
    if ct.empty:
        return []

    # Normalise regulon names to bare TF symbols. pyscenic / scenicplus
    # emit variants: "TF_regulon", "TF(+)", "TF(-)", "TF_activator",
    # "TF_repressor", "TF_extended", ...
    tf_col = ct["regulon"].astype(str)
    tf_col = tf_col.str.replace(r"_regulon$", "", regex=True)
    tf_col = tf_col.str.replace(r"_extended$", "", regex=True)
    tf_col = tf_col.str.replace(r"_(activator|repressor)$", "", regex=True)
    tf_col = tf_col.str.replace(r"\s*\([+\-]\)\s*$", "", regex=True)
    ct["tf"] = tf_col

    # GRN TF → {predicted targets}
    grn_targets: dict[str, set[str]] | None = None
    if use_grn_intersection and grn is not None:
        grn_targets = (
            grn.groupby("TF")["target"].apply(set).to_dict()
        )

    # Pre-index enhancer_links for fast per-peak lookup.
    # Each peak can map to multiple (gene, correlation) pairs.
    links_by_peak: dict[str, list[tuple[str, float]]] = {}
    for _, row in enhancer_links[["peak_id", "gene", "correlation"]].iterrows():
        links_by_peak.setdefault(str(row["peak_id"]), []).append(
            (str(row["gene"]), float(row["correlation"]))
        )

    # Assemble per TF.
    eregulons: list[ERegulon] = []
    for tf, tf_group in ct.groupby("tf"):
        tf_str = str(tf)
        peaks_for_tf = tf_group[peak_col].astype(str).unique().tolist()
        if not peaks_for_tf:
            continue

        # Find enhancer-linked targets for these peaks.
        target_to_peaks: dict[str, set[str]] = {}
        for peak in peaks_for_tf:
            for gene, corr in links_by_peak.get(peak, ()):
                # Keep only positive correlations by default — negative
                # correlation implies TF represses gene, which is a real
                # but less common case. Users can rebuild with custom
                # enhancer filtering upstream if they want negatives too.
                if corr <= 0.0:
                    continue
                target_to_peaks.setdefault(gene, set()).add(peak)

        if use_grn_intersection and grn_targets is not None:
            grn_set = grn_targets.get(tf_str, set())
            target_to_peaks = {
                g: peaks for g, peaks in target_to_peaks.items() if g in grn_set
            }

        if len(target_to_peaks) < min_target_genes:
            continue

        # Count (peak, gene) edges supporting this eRegulon.
        n_edges = sum(len(v) for v in target_to_peaks.values())
        if n_edges < min_enhancer_links:
            continue

        supporting_peaks = set()
        for peaks in target_to_peaks.values():
            supporting_peaks.update(peaks)
        motif_auc_mean = float(
            tf_group.loc[tf_group[peak_col].astype(str).isin(supporting_peaks), "auc"].mean()
        )
        if pd.isna(motif_auc_mean):
            motif_auc_mean = 0.0

        eregulons.append(
            ERegulon(
                tf=tf_str,
                enhancers=sorted(supporting_peaks),
                target_genes=sorted(target_to_peaks.keys()),
                n_enhancer_links=n_edges,
                motif_auc=motif_auc_mean,
            )
        )

    eregulons.sort(key=lambda e: (-e.n_enhancer_links, -len(e.target_genes)))

    _warn_if_catastrophic_drop(eregulons, ct, use_grn_intersection)
    return eregulons


def _warn_if_catastrophic_drop(
    eregulons: list[ERegulon],
    ct: pd.DataFrame,
    use_grn_intersection: bool,
) -> None:
    """Warn when the intersection dropped > 50% of cistarget TFs.

    A common failure mode: a strict `use_grn_intersection=True` wipes
    out most regulons because GRN and enhancer links name genes under
    different conventions (ENSEMBL vs symbol), or the peak_id keys
    didn't actually match across the three inputs. Without this warning
    the user sees an empty or tiny output and has no clue why.
    """
    import warnings

    n_input_tfs = ct["tf"].nunique()
    n_output = len(eregulons)
    if n_input_tfs == 0:
        return
    if n_output >= max(1, n_input_tfs // 2):
        return
    reason = (
        "try use_grn_intersection=False — the GRN ∩ enhancer-link step "
        "dropped most TFs, which usually means GRN gene names and "
        "enhancer_links `gene` column use different conventions (symbol "
        "vs ENSEMBL)"
        if use_grn_intersection
        else (
            "check that cistarget `peak_id` values overlap "
            "enhancer_links `peak_id` values — the two sets appear to "
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
        for enh in er.enhancers:
            for tgt in er.target_genes:
                rows.append(
                    (er.tf, enh, tgt, er.n_enhancer_links, er.motif_auc)
                )
    return pd.DataFrame(
        rows,
        columns=["tf", "enhancer", "target_gene", "n_enhancer_links", "motif_auc"],
    )


def _require_columns(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
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
