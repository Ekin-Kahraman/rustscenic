"""Tests for rustscenic.topics.fit."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import rustscenic.topics as topics


@pytest.fixture
def synthetic_atac_2_topics():
    """Cells with two distinct peak programs — LDA should find both."""
    n_cells, n_peaks = 200, 40
    # Topic A: peaks 0-9; Topic B: peaks 20-29
    X = np.zeros((n_cells, n_peaks), dtype=np.int32)
    rng = np.random.default_rng(0)
    for i in range(100):
        active = rng.choice(range(0, 10), size=6, replace=False)
        X[i, active] = 1
    for i in range(100, 200):
        active = rng.choice(range(20, 30), size=6, replace=False)
        X[i, active] = 1
    return sp.csr_matrix(X), [f"c{i}" for i in range(n_cells)], [f"p{i}" for i in range(n_peaks)]


class TestTopicsShape:
    def test_result_shapes(self, synthetic_atac_2_topics):
        X, cells, peaks = synthetic_atac_2_topics
        res = topics.fit((X, cells, peaks), n_topics=4, n_passes=3, seed=0, verbose=False)
        assert res.cell_topic.shape == (len(cells), 4)
        assert res.topic_peak.shape == (4, len(peaks))
        # Rows sum to 1 (probabilities)
        np.testing.assert_allclose(res.cell_topic.values.sum(axis=1), 1.0, atol=1e-4)
        np.testing.assert_allclose(res.topic_peak.values.sum(axis=1), 1.0, atol=1e-4)


class TestTopicsCorrectness:
    def test_separates_planted_topics(self, synthetic_atac_2_topics):
        X, cells, peaks = synthetic_atac_2_topics
        res = topics.fit((X, cells, peaks), n_topics=2, n_passes=8, seed=0, verbose=False)
        labels = res.cell_topic.values.argmax(axis=1)
        # First 100 cells should share one argmax label; last 100 cells the other.
        a_half = labels[:100]
        b_half = labels[100:]
        assert len(np.unique(a_half)) == 1, f"first group split across {np.unique(a_half)}"
        assert len(np.unique(b_half)) == 1, f"second group split across {np.unique(b_half)}"
        assert a_half[0] != b_half[0]


class TestTopicsEdgeCases:
    def test_n_topics_zero_raises(self, synthetic_atac_2_topics):
        X, cells, peaks = synthetic_atac_2_topics
        with pytest.raises(ValueError, match="n_topics"):
            topics.fit((X, cells, peaks), n_topics=0, verbose=False)

    def test_single_cell_input(self):
        X = sp.csr_matrix(np.array([[1, 1, 0, 0, 1]], dtype=np.int32))
        res = topics.fit((X, ["c0"], [f"p{i}" for i in range(5)]),
                         n_topics=2, n_passes=2, seed=0, verbose=False)
        assert res.cell_topic.shape == (1, 2)

    def test_nan_input_panics(self):
        X = sp.csr_matrix(np.array([[1.0, np.nan, 0, 0]], dtype=np.float32))
        with pytest.raises(BaseException, match=r"Na[Nn]|[Ii]nf"):
            topics.fit((X, ["c0"], ["a", "b", "c", "d"]),
                       n_topics=2, n_passes=2, seed=0, verbose=False)


class TestTopicsDeterminism:
    def test_same_seed_bit_identical(self, synthetic_atac_2_topics):
        X, cells, peaks = synthetic_atac_2_topics
        a = topics.fit((X, cells, peaks), n_topics=3, n_passes=3, seed=42, verbose=False)
        b = topics.fit((X, cells, peaks), n_topics=3, n_passes=3, seed=42, verbose=False)
        np.testing.assert_array_equal(a.cell_topic.values, b.cell_topic.values)
        np.testing.assert_array_equal(a.topic_peak.values, b.topic_peak.values)
