"""Regulon specificity scores (RSS) and topic-based candidate enhancers.

Two small downstream pieces SCENIC+ ships that rustscenic was missing.

Public API
    rustscenic.specificity.regulon_specificity_scores(auc, cell_groups) -> pd.DataFrame
    rustscenic.specificity.candidate_enhancers_per_topic(topic_peak, top_n=2000) -> dict

RSS measures how cluster-specific each regulon's activity is. Implemented
as Jensen-Shannon divergence between (a) the regulon's per-group mean AUC
distribution and (b) the cluster-membership distribution — the same
formula pyscenic ships in `pyscenic.binarization.regulon_specificity_scores`.

Topic candidate enhancers turn the topics-fit output (a topic-peak weight
matrix) into per-topic ranked peak lists, the bridge that pycisTopic uses
between topic modelling and motif enrichment.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd


def regulon_specificity_scores(
    auc: pd.DataFrame,
    cell_groups: Union[pd.Series, Sequence],
) -> pd.DataFrame:
    """Per-(group, regulon) Jensen-Shannon-based specificity score.

    Parameters
    ----------
    auc
        AUCell output: cells × regulons. Index = cell barcodes,
        columns = regulon names. From ``rustscenic.aucell.score``.
    cell_groups
        Cluster / cell-type label per cell (length == auc.n_obs).
        Pandas Series indexed by barcode, or any sequence aligned to
        ``auc.index``.

    Returns
    -------
    pandas.DataFrame indexed by group, columns = regulons. Values in
    [0, 1]; higher = more group-specific. Same orientation as
    ``pyscenic.binarization.regulon_specificity_scores``.
    """
    if isinstance(cell_groups, pd.Series):
        cell_groups = cell_groups.reindex(auc.index)
    cell_groups = np.asarray(cell_groups)
    if len(cell_groups) != auc.shape[0]:
        raise ValueError(
            f"cell_groups length {len(cell_groups)} must equal n_cells "
            f"{auc.shape[0]}"
        )

    groups = np.unique(cell_groups[~pd.isna(cell_groups)])
    auc_arr = auc.values.astype(np.float64)

    # Distribution Q: regulon activity averaged within each group, normalised.
    # Distribution P: per-group cell-count fraction (i.e., uniform over cells
    # in that group), broadcasting against regulons.
    rss = np.zeros((len(groups), auc.shape[1]), dtype=np.float64)
    for gi, g in enumerate(groups):
        mask = cell_groups == g
        n_in = mask.sum()
        if n_in == 0:
            continue
        # Per-regulon: how concentrated is activity within this group?
        # Q[c, r] = auc[c, r] / sum(auc[:, r])  (normalised across all cells)
        col_sums = auc_arr.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        Q = auc_arr / col_sums  # (n_cells, n_regulons), columns sum to 1
        # P[c, r] = 1/n_in if cell c in group g else 0  (column-constant)
        P = mask.astype(np.float64) / max(n_in, 1)
        # JS divergence per regulon: 0.5 * (KL(P || M) + KL(Q || M))
        # where M = 0.5 * (P + Q). Compute per-regulon (column-wise).
        for r in range(auc.shape[1]):
            p = P
            q = Q[:, r]
            m = 0.5 * (p + q)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0).sum()
                kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0).sum()
            js = 0.5 * (kl_pm + kl_qm)
            # RSS = 1 - sqrt(JS) — bounded [0, 1], higher = more specific.
            rss[gi, r] = 1.0 - float(np.sqrt(max(js, 0.0)))

    return pd.DataFrame(rss, index=groups, columns=auc.columns)


def candidate_enhancers_per_topic(
    topic_peak: Union[np.ndarray, pd.DataFrame],
    peak_names: Sequence[str] | None = None,
    top_n: int = 2_000,
) -> dict[str, list[str]]:
    """Top-ranked peaks per topic — pycisTopic's "candidate enhancers".

    Topic models on scATAC give a (topic × peak) weight matrix. Per-topic,
    the highest-weight peaks are the topic's candidate enhancers — used
    downstream to query motif enrichment region-by-region.

    Parameters
    ----------
    topic_peak
        Either a (n_topics × n_peaks) numpy array, or a DataFrame indexed
        by topic with peaks as columns. From ``rustscenic.topics.fit``.
    peak_names
        Peak ID per column when ``topic_peak`` is a numpy array. Ignored
        for DataFrame input.
    top_n
        Number of top-weighted peaks per topic.

    Returns
    -------
    dict mapping ``topic_name -> [peak_ids ranked by descending weight]``.
    """
    if isinstance(topic_peak, pd.DataFrame):
        peak_names = list(topic_peak.columns)
        topic_names = [str(t) for t in topic_peak.index]
        weights = topic_peak.values
    else:
        weights = np.asarray(topic_peak)
        if peak_names is None:
            peak_names = [f"peak_{i}" for i in range(weights.shape[1])]
        else:
            peak_names = list(peak_names)
        topic_names = [f"topic_{i}" for i in range(weights.shape[0])]

    out: dict[str, list[str]] = {}
    for ti, tname in enumerate(topic_names):
        order = np.argsort(weights[ti])[::-1][:top_n]
        out[tname] = [peak_names[i] for i in order]
    return out


__all__ = [
    "regulon_specificity_scores",
    "candidate_enhancers_per_topic",
]
