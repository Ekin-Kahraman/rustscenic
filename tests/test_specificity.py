"""Regulon specificity (RSS) + topic candidate enhancers tests.

Closes layer-coverage gaps #14 (RSS) and #6 (topic-based candidate
enhancers) from the per-stage audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rustscenic.specificity import (
    candidate_enhancers_per_topic,
    regulon_specificity_scores,
)


# ---- regulon_specificity_scores -----------------------------------------


def test_rss_returns_groups_x_regulons_in_unit_interval():
    rng = np.random.default_rng(0)
    auc = pd.DataFrame(
        rng.random((100, 4)),
        index=[f"c{i}" for i in range(100)],
        columns=["TF1", "TF2", "TF3", "TF4"],
    )
    groups = ["A"] * 30 + ["B"] * 40 + ["C"] * 30
    rss = regulon_specificity_scores(auc, groups)
    assert rss.shape == (3, 4)
    assert set(rss.index) == {"A", "B", "C"}
    assert list(rss.columns) == ["TF1", "TF2", "TF3", "TF4"]
    assert ((rss.values >= 0) & (rss.values <= 1)).all()


def test_rss_picks_out_group_specific_regulon():
    """Construct a regulon active only in group A; RSS for that
    (group, regulon) cell should rank highest."""
    n = 90
    auc = pd.DataFrame(
        np.zeros((n, 2)),
        index=[f"c{i}" for i in range(n)],
        columns=["TF_specific", "TF_uniform"],
    )
    auc.iloc[:30, 0] = 0.8         # only group A active for TF_specific
    auc.iloc[:, 1] = 0.5           # uniform across all groups for TF_uniform
    groups = ["A"] * 30 + ["B"] * 30 + ["C"] * 30
    rss = regulon_specificity_scores(auc, groups)
    # TF_specific has highest score for group A
    assert rss.loc["A", "TF_specific"] > rss.loc["B", "TF_specific"]
    assert rss.loc["A", "TF_specific"] > rss.loc["C", "TF_specific"]
    # TF_specific in group A scores higher than TF_uniform in group A
    assert rss.loc["A", "TF_specific"] > rss.loc["A", "TF_uniform"]


def test_rss_rejects_length_mismatch():
    auc = pd.DataFrame(np.zeros((10, 2)), columns=["A", "B"])
    with pytest.raises(ValueError, match="cell_groups length"):
        regulon_specificity_scores(auc, ["X"] * 5)


# ---- candidate_enhancers_per_topic -----------------------------------------


def test_candidate_enhancers_returns_top_n_per_topic_dataframe_input():
    n_topics, n_peaks = 5, 50
    rng = np.random.default_rng(0)
    weights = rng.random((n_topics, n_peaks))
    peak_names = [f"peak_{i:03d}" for i in range(n_peaks)]
    topic_peak = pd.DataFrame(
        weights, index=[f"topic_{i}" for i in range(n_topics)], columns=peak_names
    )
    out = candidate_enhancers_per_topic(topic_peak, top_n=10)
    assert set(out.keys()) == {f"topic_{i}" for i in range(n_topics)}
    for topic, peaks in out.items():
        assert len(peaks) == 10
        # Peaks must be sorted by descending weight
        ti = int(topic.split("_")[1])
        for prev_peak, this_peak in zip(peaks, peaks[1:]):
            prev_idx = peak_names.index(prev_peak)
            this_idx = peak_names.index(this_peak)
            assert weights[ti, prev_idx] >= weights[ti, this_idx]


def test_candidate_enhancers_numpy_input_uses_default_names():
    rng = np.random.default_rng(0)
    out = candidate_enhancers_per_topic(rng.random((3, 20)), top_n=5)
    assert set(out.keys()) == {"topic_0", "topic_1", "topic_2"}
    for peaks in out.values():
        assert all(p.startswith("peak_") for p in peaks)
        assert len(peaks) == 5


def test_candidate_enhancers_top_n_clamps_to_n_peaks():
    rng = np.random.default_rng(0)
    out = candidate_enhancers_per_topic(rng.random((2, 10)), top_n=100)
    for peaks in out.values():
        assert len(peaks) == 10
