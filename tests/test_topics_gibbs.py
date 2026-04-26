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


def test_gibbs_rejects_invalid_args():
    X, cells, peaks = _two_topic_corpus(20, 10)
    with pytest.raises(ValueError, match="n_topics"):
        rustscenic.topics.fit_gibbs((X, cells, peaks), n_topics=0, verbose=False)
    with pytest.raises(ValueError, match="n_iters"):
        rustscenic.topics.fit_gibbs((X, cells, peaks), n_topics=2, n_iters=0, verbose=False)


def test_gibbs_alpha_eta_defaults():
    """Defaults differ from online VB (Griffiths-Steyvers' 0.1 / 0.01)."""
    X, cells, peaks = _two_topic_corpus(20, 10)
    # No-arg call should work without error
    r = rustscenic.topics.fit_gibbs(
        (X, cells, peaks), n_topics=2, n_iters=20, verbose=False,
    )
    assert r.n_topics == 2
