"""Tests for rustscenic.topics.fit."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import rustscenic.topics as topics


@pytest.fixture
def synthetic_atac_2_topics():
    """Cells with two distinct peak programs - LDA should find both."""
    n_cells, n_peaks = 200, 40
    # Topic A: peaks 0-9; Topic B: peaks 20-29
    X = np.zeros((n_cells, n_peaks), dtype=np.int32)
    rng = np.random.default_rng(0)
    for i in range(100):
        active = rng.choice(range(10), size=6, replace=False)
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
        assert res.backend_execution == {"engine": "rust", "symbols": ["topics_fit"]}
        assert res.cell_topic.attrs["rust_backend"] == res.backend_execution
        assert res.topic_peak.attrs["rust_backend"] == res.backend_execution
        # Rows sum to 1 (probabilities)
        np.testing.assert_allclose(res.cell_topic.values.sum(axis=1), 1.0, atol=1e-4)
        np.testing.assert_allclose(res.topic_peak.values.sum(axis=1), 1.0, atol=1e-4)

    def test_fit_passes_numpy_csr_buffers_to_extension(self, monkeypatch):
        X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
        cells = ["c0", "c1"]
        peaks = ["p0", "p1", "p2"]
        seen = {}

        def fake_topics_fit(row_ptr, col_idx, counts, n_words, n_topics, *_args):
            seen["row_ptr"] = row_ptr
            seen["col_idx"] = col_idx
            seen["counts"] = counts
            return (
                np.full((len(row_ptr) - 1, n_topics), 1.0 / n_topics, dtype=np.float32),
                np.full((n_topics, n_words), 1.0 / n_words, dtype=np.float32),
            )

        monkeypatch.setattr(topics, "_topics_fit", fake_topics_fit)
        res = topics.fit((X, cells, peaks), n_topics=2, n_passes=1, verbose=False)

        assert res.cell_topic.shape == (2, 2)
        assert isinstance(seen["row_ptr"], np.ndarray)
        assert isinstance(seen["col_idx"], np.ndarray)
        assert isinstance(seen["counts"], np.ndarray)
        assert seen["row_ptr"].dtype == X.indptr.dtype
        assert seen["col_idx"].dtype == np.int32
        assert seen["counts"].dtype == np.float32
        assert np.shares_memory(seen["row_ptr"], X.indptr)
        assert np.shares_memory(seen["col_idx"], X.indices)


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

    def test_cell_assignment_marks_zero_rows_missing(self):
        res = topics.TopicsResult(
            cell_topic=pd.DataFrame(
                [[0.0, 0.0], [0.2, 0.8]],
                index=["empty_cell", "active_cell"],
                columns=["Topic_0", "Topic_1"],
            ),
            topic_peak=pd.DataFrame([[0.5], [0.5]], index=["Topic_0", "Topic_1"]),
            n_topics=2,
        )

        with pytest.warns(UserWarning, match="zero or non-finite total topic weight"):
            assignment = res.cell_assignment()

        assert assignment.attrs["rust_backend"]["symbols"] == ["topics_cell_assignment"]
        assert pd.isna(assignment.loc["empty_cell"])
        assert assignment.loc["active_cell"] == "Topic_1"

    def test_cell_assignment_uses_rust_row_scanner_without_float32_copy(self, monkeypatch):
        values = np.asfortranarray(
            np.array(
                [[0.1, 0.9], [0.0, 0.0], [0.8, 0.2]],
                dtype=np.float32,
            )
        )
        res = topics.TopicsResult(
            cell_topic=pd.DataFrame(
                values,
                index=["c0", "c1", "c2"],
                columns=["Topic_0", "Topic_1"],
            ),
            topic_peak=pd.DataFrame([[0.5], [0.5]], index=["Topic_0", "Topic_1"]),
            n_topics=2,
        )

        def fake_assignment(arg):
            assert np.shares_memory(arg, values)
            assert arg.flags.f_contiguous
            assert not arg.flags.c_contiguous
            return np.array([1, -1, 0], dtype=np.int64), 2, 1

        monkeypatch.setattr(topics, "_topics_cell_assignment", fake_assignment)

        with pytest.warns(UserWarning, match="zero or non-finite total topic weight"):
            assignment = res.cell_assignment()

        assert assignment.attrs["rust_backend"]["symbols"] == ["topics_cell_assignment"]
        assert assignment.loc["c0"] == "Topic_1"
        assert pd.isna(assignment.loc["c1"])
        assert assignment.loc["c2"] == "Topic_0"

    def test_top_peaks_per_topic_uses_descending_topic_weights(self):
        res = topics.TopicsResult(
            cell_topic=pd.DataFrame([[0.5, 0.5]], columns=["Topic_0", "Topic_1"]),
            topic_peak=pd.DataFrame(
                [[0.1, 0.9, 0.2, 0.8], [0.6, 0.2, 0.7, 0.1]],
                index=["Topic_0", "Topic_1"],
                columns=["p0", "p1", "p2", "p3"],
            ),
            n_topics=2,
        )

        out = res.top_peaks_per_topic(n=2)
        assert out == {
            "Topic_0": ["p1", "p3"],
            "Topic_1": ["p2", "p0"],
        }
        assert out.rust_backend["symbols"] == ["specificity_candidate_top_indices"]

    def test_top_peaks_per_topic_delegates_to_shared_candidate_helper(self, monkeypatch):
        values = np.asfortranarray(
            np.array(
                [[0.1, 0.9, 0.2, 0.8], [0.6, 0.2, 0.7, 0.1]],
                dtype=np.float64,
            )
        )
        res = topics.TopicsResult(
            cell_topic=pd.DataFrame([[0.5, 0.5]], columns=["Topic_0", "Topic_1"]),
            topic_peak=pd.DataFrame(
                values,
                index=["Topic_0", "Topic_1"],
                columns=["p0", "p1", "p2", "p3"],
            ),
            n_topics=2,
        )
        seen = {}

        def fake_candidate_helper(topic_peak, top_n):
            seen["topic_peak"] = topic_peak
            assert top_n == 2
            out = topics.CandidateEnhancers({
                "Topic_0": ["p1", "p3"],
                "Topic_1": ["p2", "p0"],
            })
            out.rust_backend = {
                "engine": "rust",
                "symbols": ["specificity_candidate_top_indices"],
            }
            return out

        monkeypatch.setattr(topics, "_candidate_enhancers_per_topic", fake_candidate_helper)

        out = res.top_peaks_per_topic(n=2)
        assert seen["topic_peak"] is res.topic_peak
        assert out == {
            "Topic_0": ["p1", "p3"],
            "Topic_1": ["p2", "p0"],
        }
        assert out.rust_backend["symbols"] == ["specificity_candidate_top_indices"]

    def test_top_peaks_per_topic_passes_float32_weights_without_upcast(self, monkeypatch):
        values = np.asfortranarray(
            np.array(
                [[0.1, 0.9, 0.2, 0.8], [0.6, 0.2, 0.7, 0.1]],
                dtype=np.float32,
            )
        )
        res = topics.TopicsResult(
            cell_topic=pd.DataFrame([[0.5, 0.5]], columns=["Topic_0", "Topic_1"]),
            topic_peak=pd.DataFrame(
                values,
                index=["Topic_0", "Topic_1"],
                columns=["p0", "p1", "p2", "p3"],
            ),
            n_topics=2,
        )

        out = res.top_peaks_per_topic(n=2)
        assert out == {
            "Topic_0": ["p1", "p3"],
            "Topic_1": ["p2", "p0"],
        }
        assert out.rust_backend["symbols"] == ["specificity_candidate_top_indices_f32"]


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

    def test_nan_input_raises_value_error(self):
        X = sp.csr_matrix(np.array([[1.0, np.nan, 0, 0]], dtype=np.float32))
        with pytest.raises(ValueError, match="finite"):
            topics.fit((X, ["c0"], ["a", "b", "c", "d"]),
                       n_topics=2, n_passes=2, seed=0, verbose=False)

    def test_negative_input_raises_value_error(self):
        X = sp.csr_matrix(np.array([[1.0, -1.0, 0, 0]], dtype=np.float32))
        with pytest.raises(ValueError, match="non-negative"):
            topics.fit((X, ["c0"], ["a", "b", "c", "d"]),
                       n_topics=2, n_passes=2, seed=0, verbose=False)


class TestTopicsDeterminism:
    def test_same_seed_bit_identical(self, synthetic_atac_2_topics):
        X, cells, peaks = synthetic_atac_2_topics
        a = topics.fit((X, cells, peaks), n_topics=3, n_passes=3, seed=42, verbose=False)
        b = topics.fit((X, cells, peaks), n_topics=3, n_passes=3, seed=42, verbose=False)
        np.testing.assert_array_equal(a.cell_topic.values, b.cell_topic.values)
        np.testing.assert_array_equal(a.topic_peak.values, b.topic_peak.values)
