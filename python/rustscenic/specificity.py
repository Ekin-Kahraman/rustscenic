"""Regulon specificity scores (RSS) and topic-based candidate enhancers.

Two small downstream pieces SCENIC+ ships that rustscenic was missing.

Public API
    rustscenic.specificity.regulon_specificity_scores(auc, cell_groups) -> pd.DataFrame
    rustscenic.specificity.candidate_enhancers_per_topic(topic_peak, top_n=2000) -> dict

RSS measures how cluster-specific each regulon's activity is. Implemented
as Jensen-Shannon divergence between (a) the regulon's per-group mean AUC
distribution and (b) the cluster-membership distribution - the same
formula pyscenic ships in `pyscenic.binarization.regulon_specificity_scores`.

Topic candidate enhancers turn the topics-fit output (a topic-peak weight
matrix) into per-topic ranked peak lists, the bridge that pycisTopic uses
between topic modelling and motif enrichment.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    specificity_candidate_top_indices as _candidate_top_indices,
    specificity_candidate_top_indices_f32 as _candidate_top_indices_f32,
    specificity_group_codes_with_numeric_order as _specificity_group_codes_with_numeric_order,
    specificity_group_codes_with_order as _specificity_group_codes_with_order,
    specificity_rss as _specificity_rss,
    specificity_rss_f32 as _specificity_rss_f32,
)


def regulon_specificity_scores(
    auc: pd.DataFrame,
    cell_groups: pd.Series | Sequence,
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

    missing = np.asarray(pd.isna(cell_groups), dtype=bool)
    group_labels = [
        "" if is_missing else str(group)
        for group, is_missing in zip(cell_groups, missing, strict=True)
    ]
    if np.issubdtype(cell_groups.dtype, np.number):
        group_codes, group_first_indices = _specificity_group_codes_with_numeric_order(
            group_labels,
            missing,
            np.asarray(cell_groups, dtype=np.float64),
        )
    else:
        group_codes, group_first_indices = _specificity_group_codes_with_order(
            group_labels,
            missing,
        )
    group_first_indices = np.asarray(group_first_indices)
    groups = cell_groups[group_first_indices]

    auc_values, rss_kernel = _rss_kernel_arg(auc.values)
    rss = rss_kernel(
        auc_values,
        group_codes,
        len(groups),
    )
    return pd.DataFrame(np.asarray(rss), index=groups, columns=auc.columns)


def _rss_kernel_arg(values: np.ndarray):
    values = np.asarray(values)
    if values.dtype == np.float32:
        return values, _specificity_rss_f32
    if values.dtype == np.float64:
        return values, _specificity_rss
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("auc must contain numeric values")
    return values.astype(np.float64, copy=False), _specificity_rss


def candidate_enhancers_per_topic(
    topic_peak: np.ndarray | pd.DataFrame,
    peak_names: Sequence[str] | None = None,
    top_n: int = 2_000,
) -> dict[str, list[str]]:
    """Top-ranked peaks per topic - pycisTopic's "candidate enhancers".

    Topic models on scATAC give a (topic × peak) weight matrix. Per-topic,
    the highest-weight peaks are the topic's candidate enhancers - used
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

    weights_arg, top_indices_kernel = _candidate_top_indices_kernel_arg(weights)
    top_indices = np.asarray(top_indices_kernel(weights_arg, int(top_n)))
    return {
        tname: [peak_names[i] for i in top_indices[ti]]
        for ti, tname in enumerate(topic_names)
    }


def _candidate_top_indices_kernel_arg(weights: np.ndarray):
    weights = np.asarray(weights)
    if weights.dtype == np.float32:
        return weights, _candidate_top_indices_f32
    if weights.dtype == np.float64:
        return weights, _candidate_top_indices
    if not np.issubdtype(weights.dtype, np.number):
        raise TypeError("topic_peak must contain numeric values")
    return weights.astype(np.float64, copy=False), _candidate_top_indices


__all__ = [
    "regulon_specificity_scores",
    "candidate_enhancers_per_topic",
]
