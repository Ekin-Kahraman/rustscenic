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
import rustscenic.specificity as specificity


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
    assert rss.attrs["rust_backend"]["symbols"] == [
        "specificity_group_codes_with_order",
        "specificity_rss",
    ]
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


def test_rss_accepts_strided_auc_without_python_contiguity_copy(monkeypatch):
    values = np.asfortranarray(np.arange(24, dtype=np.float64).reshape(6, 4))
    auc = pd.DataFrame(
        values,
        index=[f"c{i}" for i in range(6)],
        columns=[f"TF{i}" for i in range(4)],
    )

    def fake_rss(auc_arg, group_codes, n_groups):
        assert np.shares_memory(auc_arg, values)
        assert auc_arg.flags.f_contiguous
        assert not auc_arg.flags.c_contiguous
        assert n_groups == 2
        assert group_codes.tolist() == [0, 0, 0, 1, 1, 1]
        return np.zeros((2, 4), dtype=np.float64)

    monkeypatch.setattr(specificity, "_specificity_rss", fake_rss)
    out = specificity.regulon_specificity_scores(auc, ["A", "A", "A", "B", "B", "B"])

    assert out.shape == (2, 4)
    assert out.attrs["rust_backend"]["symbols"] == [
        "specificity_group_codes_with_order",
        "specificity_rss",
    ]


def test_rss_uses_float32_auc_without_upcast_copy(monkeypatch):
    values = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(6, 4))
    auc = pd.DataFrame(
        values,
        index=[f"c{i}" for i in range(6)],
        columns=[f"TF{i}" for i in range(4)],
    )

    def fake_rss(auc_arg, group_codes, n_groups):
        assert np.shares_memory(auc_arg, values)
        assert auc_arg.dtype == np.float32
        assert auc_arg.flags.f_contiguous
        assert not auc_arg.flags.c_contiguous
        assert n_groups == 2
        assert group_codes.tolist() == [0, 0, 0, 1, 1, 1]
        return np.zeros((2, 4), dtype=np.float64)

    monkeypatch.setattr(specificity, "_specificity_rss_f32", fake_rss)
    out = specificity.regulon_specificity_scores(auc, ["A", "A", "A", "B", "B", "B"])

    assert out.shape == (2, 4)
    assert out.attrs["rust_backend"]["symbols"] == [
        "specificity_group_codes_with_order",
        "specificity_rss_f32",
    ]


def test_rss_group_encoding_discovers_groups_in_rust(monkeypatch):
    auc = pd.DataFrame(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        index=["c0", "c1", "c2"],
        columns=["TF0", "TF1"],
    )
    seen = {}

    def fake_group_codes(labels, missing):
        seen["labels"] = labels
        seen["missing"] = missing
        return np.array([1, -1, 0], dtype=np.int32), np.array([2, 0], dtype=np.uint64)

    def fake_rss(auc_arg, group_codes, n_groups):
        assert n_groups == 2
        assert group_codes.tolist() == [1, -1, 0]
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    monkeypatch.setattr(specificity, "_specificity_group_codes_with_order", fake_group_codes)
    monkeypatch.setattr(specificity, "_specificity_rss", fake_rss)

    out = specificity.regulon_specificity_scores(auc, ["B", None, "A"])

    assert seen["labels"] == ["B", "", "A"]
    assert seen["missing"].tolist() == [False, True, False]
    assert list(out.index) == ["A", "B"]
    np.testing.assert_array_equal(
        out.values,
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
    )


def test_rss_numeric_group_order_uses_rust_numeric_helper(monkeypatch):
    auc = pd.DataFrame(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        columns=["TF0", "TF1"],
    )
    seen = {}

    def fake_group_codes(labels, missing, numeric_values):
        seen["labels"] = labels
        seen["missing"] = missing
        seen["numeric_values"] = numeric_values
        return np.array([1, 0, 1], dtype=np.int32), np.array([1, 0], dtype=np.uint64)

    def fake_rss(auc_arg, group_codes, n_groups):
        assert n_groups == 2
        assert group_codes.tolist() == [1, 0, 1]
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    monkeypatch.setattr(
        specificity,
        "_specificity_group_codes_with_numeric_order",
        fake_group_codes,
    )
    monkeypatch.setattr(specificity, "_specificity_rss", fake_rss)

    out = specificity.regulon_specificity_scores(auc, [10, 2, 10])

    assert seen["labels"] == ["10", "2", "10"]
    assert seen["missing"].tolist() == [False, False, False]
    np.testing.assert_array_equal(seen["numeric_values"], np.array([10.0, 2.0, 10.0]))
    assert list(out.index) == [2, 10]


def test_rss_missing_groups_are_ignored():
    auc = pd.DataFrame(
        np.array(
            [
                [1.0, 0.0],
                [0.2, 0.2],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        columns=["TF0", "TF1"],
    )

    rss = regulon_specificity_scores(auc, ["B", None, "A"])

    assert list(rss.index) == ["A", "B"]
    assert rss.shape == (2, 2)


def test_rss_preserves_numeric_group_index_values():
    auc = pd.DataFrame(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.2, 0.3],
            ],
            dtype=np.float64,
        ),
        columns=["TF0", "TF1"],
    )

    rss = regulon_specificity_scores(auc, [10, 2, 10])

    assert list(rss.index) == [2, 10]
    assert rss.index.dtype.kind in {"i", "u"}


def test_rss_strided_auc_matches_c_contiguous_result():
    values = np.arange(40, dtype=np.float64).reshape(10, 4) / 40.0
    strided = np.asfortranarray(values)
    groups = ["A"] * 4 + ["B"] * 3 + ["C"] * 3

    got = regulon_specificity_scores(
        pd.DataFrame(strided, columns=[f"TF{i}" for i in range(4)]),
        groups,
    )
    expected = regulon_specificity_scores(
        pd.DataFrame(values.copy(order="C"), columns=[f"TF{i}" for i in range(4)]),
        groups,
    )

    pd.testing.assert_frame_equal(got, expected)


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
    assert isinstance(out, dict)
    assert out.rust_backend["symbols"] == ["specificity_candidate_enhancers"]
    assert set(out.keys()) == {f"topic_{i}" for i in range(n_topics)}
    for topic, peaks in out.items():
        assert len(peaks) == 10
        # Peaks must be sorted by descending weight
        ti = int(topic.split("_")[1])
        for prev_peak, this_peak in zip(peaks, peaks[1:], strict=False):
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


def test_candidate_enhancers_accepts_strided_weights_without_python_contiguity_copy(monkeypatch):
    weights = np.asfortranarray(np.arange(30, dtype=np.float64).reshape(3, 10))
    topic_peak = pd.DataFrame(
        weights,
        index=[f"topic_{i}" for i in range(3)],
        columns=[f"peak_{i}" for i in range(10)],
    )

    def fake_candidate_enhancers(weights_arg, topic_names, peak_names, top_n):
        assert np.shares_memory(weights_arg, weights)
        assert weights_arg.flags.f_contiguous
        assert not weights_arg.flags.c_contiguous
        assert topic_names == ["topic_0", "topic_1", "topic_2"]
        assert peak_names == [f"peak_{i}" for i in range(10)]
        assert top_n == 4
        return {
            topic: ["peak_0", "peak_1", "peak_2", "peak_3"]
            for topic in topic_names
        }

    monkeypatch.setattr(specificity, "_candidate_enhancers", fake_candidate_enhancers)
    out = specificity.candidate_enhancers_per_topic(topic_peak, top_n=4)

    assert out == {
        "topic_0": ["peak_0", "peak_1", "peak_2", "peak_3"],
        "topic_1": ["peak_0", "peak_1", "peak_2", "peak_3"],
        "topic_2": ["peak_0", "peak_1", "peak_2", "peak_3"],
    }


def test_candidate_enhancers_uses_float32_weights_without_upcast_copy(monkeypatch):
    weights = np.asfortranarray(np.arange(30, dtype=np.float32).reshape(3, 10))
    topic_peak = pd.DataFrame(
        weights,
        index=[f"topic_{i}" for i in range(3)],
        columns=[f"peak_{i}" for i in range(10)],
    )

    def fake_candidate_enhancers(weights_arg, topic_names, peak_names, top_n):
        assert np.shares_memory(weights_arg, weights)
        assert weights_arg.dtype == np.float32
        assert weights_arg.flags.f_contiguous
        assert not weights_arg.flags.c_contiguous
        assert topic_names == ["topic_0", "topic_1", "topic_2"]
        assert peak_names == [f"peak_{i}" for i in range(10)]
        assert top_n == 4
        return {
            topic: ["peak_0", "peak_1", "peak_2", "peak_3"]
            for topic in topic_names
        }

    monkeypatch.setattr(specificity, "_candidate_enhancers_f32", fake_candidate_enhancers)
    out = specificity.candidate_enhancers_per_topic(topic_peak, top_n=4)

    assert out.rust_backend["symbols"] == ["specificity_candidate_enhancers_f32"]
    assert out == {
        "topic_0": ["peak_0", "peak_1", "peak_2", "peak_3"],
        "topic_1": ["peak_0", "peak_1", "peak_2", "peak_3"],
        "topic_2": ["peak_0", "peak_1", "peak_2", "peak_3"],
    }


def test_candidate_enhancers_strided_weights_match_c_contiguous_result():
    rng = np.random.default_rng(1)
    weights = rng.random((4, 25))
    peak_names = [f"peak_{i}" for i in range(weights.shape[1])]

    got = candidate_enhancers_per_topic(np.asfortranarray(weights), peak_names, top_n=7)
    expected = candidate_enhancers_per_topic(weights.copy(order="C"), peak_names, top_n=7)

    assert got == expected


def test_candidate_enhancers_top_n_clamps_to_n_peaks():
    rng = np.random.default_rng(0)
    out = candidate_enhancers_per_topic(rng.random((2, 10)), top_n=100)
    for peaks in out.values():
        assert len(peaks) == 10
