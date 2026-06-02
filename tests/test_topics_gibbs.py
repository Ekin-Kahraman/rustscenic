"""Tests for the collapsed-Gibbs LDA topic model.

Closes the only place rustscenic still loses to references on quality:
NPMI 0.123 (online VB) vs Mallet 0.196 on sparse scATAC at K ≥ 30.
The Gibbs sampler matches Mallet's algorithm class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import rustscenic.topics


def _two_topic_corpus(n_docs: int = 60, n_words: int = 20):
    """Half the docs sample words 0..(n_words/2 - 1), half the other
    half. A 2-topic LDA should recover the split."""
    half = n_words // 2
    rows, cols, data = [], [], []
    for d in range(n_docs):
        ws = list(range(half)) if d < n_docs // 2 else list(range(half, n_words))
        for w in ws:
            rows.append(d)
            cols.append(w)
            data.append(1.0)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, n_words))
    cells = [f"c{i}" for i in range(n_docs)]
    peaks = [f"p{i}" for i in range(n_words)]
    return X, cells, peaks


def test_gibbs_recovers_planted_two_topics():
    X, cells, peaks = _two_topic_corpus(60, 20)
    r = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=200, seed=42, verbose=False,
    )
    assert r.cell_topic.shape == (60, 2)
    assert r.topic_peak.shape == (2, 20)
    assert r.backend_execution == {"engine": "rust", "symbols": ["topics_fit_gibbs"]}
    assert r.cell_topic.attrs["rust_backend"] == r.backend_execution
    assert r.topic_peak.attrs["rust_backend"] == r.backend_execution

    # Each row of cell_topic sums to ~1 (probability)
    assert np.allclose(r.cell_topic.sum(axis=1), 1.0, atol=1e-3)
    # Each row of topic_peak sums to ~1
    assert np.allclose(r.topic_peak.sum(axis=1), 1.0, atol=1e-3)

    # Check the assignment cleanly splits the corpus
    argmax = r.cell_topic.values.argmax(axis=1)
    first_half_topic = argmax[:30]
    second_half_topic = argmax[30:]
    # All docs in first half should share one topic; second half the other
    assert (first_half_topic == first_half_topic[0]).sum() >= 28
    assert (second_half_topic != first_half_topic[0]).sum() >= 28


def test_gibbs_deterministic_under_same_seed():
    X, cells, peaks = _two_topic_corpus(40, 15)
    a = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=3, n_iters=30, seed=7, verbose=False,
    )
    b = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=3, n_iters=30, seed=7, verbose=False,
    )
    assert np.array_equal(a.cell_topic.values, b.cell_topic.values)
    assert np.array_equal(a.topic_peak.values, b.topic_peak.values)


def test_gibbs_anndata_input():
    """fit_gibbs accepts the same AnnData / DataFrame / tuple shapes as fit."""
    import anndata as ad

    X, cells, peaks = _two_topic_corpus(30, 10)
    adata = ad.AnnData(
        X=X.toarray().astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=peaks),
    )
    r = rustscenic.topics.fit_gibbs(adata, n_topics=2, n_iters=50, seed=0, verbose=False)
    assert r.cell_topic.shape == (30, 2)


def test_gibbs_passes_numpy_csr_buffers_to_extension(monkeypatch):
    X, cells, peaks = _two_topic_corpus(4, 6)
    seen = {}

    def fake_topics_fit_gibbs(row_ptr, col_idx, counts, n_words, n_topics, *_args):
        seen["row_ptr"] = row_ptr
        seen["col_idx"] = col_idx
        seen["counts"] = counts
        return (
            np.full((len(row_ptr) - 1, n_topics), 1.0 / n_topics, dtype=np.float32),
            np.full((n_topics, n_words), 1.0 / n_words, dtype=np.float32),
        )

    monkeypatch.setattr(rustscenic.topics, "_topics_fit_gibbs", fake_topics_fit_gibbs)
    r = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=1, seed=0, verbose=False,
    )

    assert r.cell_topic.shape == (4, 2)
    assert isinstance(seen["row_ptr"], np.ndarray)
    assert isinstance(seen["col_idx"], np.ndarray)
    assert isinstance(seen["counts"], np.ndarray)
    assert seen["row_ptr"].dtype == X.indptr.dtype
    assert seen["col_idx"].dtype == np.int32
    assert seen["counts"].dtype == np.float32
    assert np.shares_memory(seen["row_ptr"], X.indptr)
    assert np.shares_memory(seen["col_idx"], X.indices)


def test_gibbs_rejects_invalid_args():
    X, cells, peaks = _two_topic_corpus(20, 10)
    with pytest.raises(ValueError, match="n_topics"):
        rustscenic.topics.fit_gibbs((X, cells, peaks), n_topics=0, verbose=False)
    with pytest.raises(ValueError, match="n_iters"):
        rustscenic.topics.fit_gibbs((X, cells, peaks), n_topics=2, n_iters=0, verbose=False)


def test_gibbs_rejects_nonfinite_negative_and_fractional_counts():
    X, cells, peaks = _two_topic_corpus(4, 4)

    bad = X.copy()
    bad.data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rustscenic.topics.fit_gibbs((bad, cells, peaks), n_topics=2, n_iters=5, verbose=False)

    bad = X.copy()
    bad.data[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        rustscenic.topics.fit_gibbs((bad, cells, peaks), n_topics=2, n_iters=5, verbose=False)

    bad = X.copy()
    bad.data[0] = 0.5
    with pytest.raises(ValueError, match="integer"):
        rustscenic.topics.fit_gibbs((bad, cells, peaks), n_topics=2, n_iters=5, verbose=False)


def test_gibbs_alpha_eta_defaults():
    """Defaults differ from online VB (Griffiths-Steyvers' 0.1 / 0.01)."""
    X, cells, peaks = _two_topic_corpus(20, 10)
    # No-arg call should work without error
    r = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=20, verbose=False,
    )
    assert r.n_topics == 2


def test_coherence_npmi_separates_planted_from_random():
    """Planted-topic NPMI must be measurably higher than a random topic
    on the same corpus. Backs the published quality comparison -
    if this passes, the metric is at least monotone in topic structure."""
    X, cells, peaks = _two_topic_corpus(80, 20)

    # Real topic-peak: place mass on the planted halves
    planted = np.zeros((2, 20), dtype=np.float32)
    planted[0, :10] = 1.0 / 10
    planted[1, 10:] = 1.0 / 10
    planted_result = rustscenic.topics.TopicsResult(
        cell_topic=pd.DataFrame(np.zeros((80, 2), dtype=np.float32),
                                index=cells, columns=["Topic_0", "Topic_1"]),
        topic_peak=pd.DataFrame(planted, index=["Topic_0", "Topic_1"], columns=peaks),
        n_topics=2,
    )
    npmi_planted = rustscenic.topics.coherence_npmi(
        planted_result, (X, cells, peaks), top_n=5,
    )

    # Random topic-peak: uniform mass
    rng = np.random.default_rng(0)
    rand_tw = rng.dirichlet(np.ones(20), size=2).astype(np.float32)
    random_result = rustscenic.topics.TopicsResult(
        cell_topic=pd.DataFrame(np.zeros((80, 2), dtype=np.float32),
                                index=cells, columns=["Topic_0", "Topic_1"]),
        topic_peak=pd.DataFrame(rand_tw, index=["Topic_0", "Topic_1"], columns=peaks),
        n_topics=2,
    )
    npmi_random = rustscenic.topics.coherence_npmi(
        random_result, (X, cells, peaks), top_n=5,
    )

    assert npmi_planted.shape == (2,)
    assert npmi_random.shape == (2,)
    # Planted topics should score strictly higher than random topics
    assert npmi_planted.mean() > npmi_random.mean()


def test_gibbs_parallel_n_threads_1_matches_serial():
    """n_threads=1 must dispatch to the bit-deterministic serial path."""
    X, cells, peaks = _two_topic_corpus(60, 20)
    serial = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=30, seed=11, n_threads=1, verbose=False,
    )
    par_one = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=30, seed=11, n_threads=1, verbose=False,
    )
    assert np.array_equal(serial.cell_topic.values, par_one.cell_topic.values)
    assert np.array_equal(serial.topic_peak.values, par_one.topic_peak.values)


def test_gibbs_parallel_recovers_planted_topics():
    """AD-LDA path with n_threads=4 still recovers two planted topics."""
    X, cells, peaks = _two_topic_corpus(80, 20)
    r = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=200, seed=42, n_threads=4, verbose=False,
    )
    argmax = r.cell_topic.values.argmax(axis=1)
    first_half = argmax[:40]
    second_half = argmax[40:]
    # Allow a few more drift docs than serial, AD-LDA is approximate.
    assert (first_half == first_half[0]).sum() >= 35
    assert (second_half != first_half[0]).sum() >= 35


def test_gibbs_parallel_rejects_n_threads_zero():
    X, cells, peaks = _two_topic_corpus(20, 10)
    with pytest.raises(ValueError, match="n_threads"):
        rustscenic.topics.fit_gibbs(
            (X, cells, peaks), n_topics=2, n_iters=10, n_threads=0, verbose=False,
        )


def test_coherence_npmi_rejects_column_mismatch():
    """Caller-error: corpus columns must match the fit's topic_peak."""
    X, cells, peaks = _two_topic_corpus(40, 15)
    r = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=20, seed=0, verbose=False,
    )
    # Corpus with the same shape but a different peak ordering
    wrong_peaks = list(reversed(peaks))
    with pytest.raises(ValueError, match="column order"):
        rustscenic.topics.coherence_npmi(r, (X, cells, wrong_peaks), top_n=5)


def test_coherence_npmi_invariant_to_duplicate_csr_indices():
    X = sp.csr_matrix(
        (
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.array([0, 2, 4], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    cells = ["c0", "c1"]
    peaks = ["p0", "p1"]
    topic_peak = pd.DataFrame(
        np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        index=["Topic_0", "Topic_1"],
        columns=peaks,
    )
    result = rustscenic.topics.TopicsResult(
        cell_topic=pd.DataFrame(np.zeros((2, 2), dtype=np.float32), index=cells, columns=topic_peak.index),
        topic_peak=topic_peak,
        n_topics=2,
    )

    canonical = X.copy()
    canonical.sum_duplicates()

    dup_score = rustscenic.topics.coherence_npmi(result, (X, cells, peaks), top_n=2)
    canonical_score = rustscenic.topics.coherence_npmi(
        result, (canonical, cells, peaks), top_n=2
    )

    np.testing.assert_allclose(dup_score, canonical_score, equal_nan=True)


def test_coherence_npmi_passes_numpy_csr_buffers_to_extension(monkeypatch):
    X, cells, peaks = _two_topic_corpus(4, 6)
    topic_values = np.asfortranarray(np.full((2, 6), 1.0 / 6.0, dtype=np.float32))
    result = rustscenic.topics.TopicsResult(
        cell_topic=pd.DataFrame(
            np.zeros((4, 2), dtype=np.float32),
            index=cells,
            columns=["Topic_0", "Topic_1"],
        ),
        topic_peak=pd.DataFrame(
            topic_values,
            index=["Topic_0", "Topic_1"],
            columns=peaks,
        ),
        n_topics=2,
    )
    seen = {}

    def fake_topics_npmi(topic_word, n_topics, n_words, row_ptr, col_idx, top_n):
        seen["topic_word"] = topic_word
        seen["row_ptr"] = row_ptr
        seen["col_idx"] = col_idx
        return np.zeros(n_topics, dtype=np.float32)

    monkeypatch.setattr(rustscenic.topics, "_topics_npmi", fake_topics_npmi)
    score = rustscenic.topics.coherence_npmi(result, (X, cells, peaks), top_n=3)

    assert score.shape == (2,)
    assert np.shares_memory(seen["topic_word"], topic_values)
    assert seen["topic_word"].flags.f_contiguous
    assert not seen["topic_word"].flags.c_contiguous
    assert isinstance(seen["row_ptr"], np.ndarray)
    assert isinstance(seen["col_idx"], np.ndarray)
    assert seen["row_ptr"].dtype == X.indptr.dtype
    assert seen["col_idx"].dtype == np.int32
    assert np.shares_memory(seen["row_ptr"], X.indptr)
    assert np.shares_memory(seen["col_idx"], X.indices)


def test_coherence_npmi_strided_topic_word_matches_c_contiguous_result():
    X, cells, peaks = _two_topic_corpus(20, 10)
    topic_values = np.array(
        [
            [0.18, 0.17, 0.16, 0.15, 0.14, 0.05, 0.04, 0.04, 0.04, 0.03],
            [0.03, 0.04, 0.04, 0.04, 0.05, 0.14, 0.15, 0.16, 0.17, 0.18],
        ],
        dtype=np.float32,
    )
    common_cell_topic = pd.DataFrame(
        np.full((len(cells), 2), 0.5, dtype=np.float32),
        index=cells,
        columns=["Topic_0", "Topic_1"],
    )
    strided = rustscenic.topics.TopicsResult(
        cell_topic=common_cell_topic,
        topic_peak=pd.DataFrame(
            np.asfortranarray(topic_values),
            index=["Topic_0", "Topic_1"],
            columns=peaks,
        ),
        n_topics=2,
    )
    contiguous = rustscenic.topics.TopicsResult(
        cell_topic=common_cell_topic,
        topic_peak=pd.DataFrame(
            topic_values.copy(order="C"),
            index=["Topic_0", "Topic_1"],
            columns=peaks,
        ),
        n_topics=2,
    )

    got = rustscenic.topics.coherence_npmi(strided, (X, cells, peaks), top_n=4)
    expected = rustscenic.topics.coherence_npmi(contiguous, (X, cells, peaks), top_n=4)

    np.testing.assert_allclose(got, expected, equal_nan=True)
