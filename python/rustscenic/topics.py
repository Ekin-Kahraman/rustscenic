"""Topic modeling (pycisTopic LDA replacement).

Online variational Bayes LDA (Hoffman-Blei-Bach 2010) for scATAC peak-topic
modeling. Converges in tens of passes vs Gibbs's thousands of iterations.

    rustscenic.topics.fit(adata_or_sparse, n_topics=50) -> TopicsResult

Output is a `TopicsResult` namedtuple with:
    cell_topic:  (cells x topics) probability matrix (each row sums to 1)
    topic_peak:  (topics x peaks) probability matrix (each row sums to 1)

Both pycisTopic (Mallet Gibbs) and rustscenic (online VB) are probabilistic —
topic labels are permutation-free. Validation metric is topic assignment ARI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    topics_fit as _topics_fit,
    topics_fit_gibbs as _topics_fit_gibbs,
    topics_npmi as _topics_npmi,
)


@dataclass
class TopicsResult:
    cell_topic: pd.DataFrame   # (cells x topics)
    topic_peak: pd.DataFrame   # (topics x peaks)
    n_topics: int

    def cell_assignment(self) -> pd.Series:
        """Argmax topic per cell."""
        return self.cell_topic.idxmax(axis=1)

    def top_peaks_per_topic(self, n: int = 20) -> dict[str, list[str]]:
        return {
            k: list(self.topic_peak.loc[k].nlargest(n).index)
            for k in self.topic_peak.index
        }


def fit(
    expression,
    *,
    n_topics: int = 50,
    alpha: Optional[float] = None,
    eta: Optional[float] = None,
    tau0: float = 64.0,
    kappa: float = 0.7,
    batch_size: int = 256,
    n_passes: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> TopicsResult:
    """Fit LDA on a (cells × peaks) count / binarized matrix.

    Parameters
    ----------
    expression
        AnnData, pandas DataFrame, or (sparse-csr, cell_names, peak_names) tuple.
        For scATAC use binarized accessibility (1 if peak accessible in cell).
    n_topics
        Number of latent topics K. pycisTopic typical range: 50–200.
    alpha, eta
        Dirichlet priors. Default 1/K, matches pycisTopic.
    tau0, kappa
        Learning-rate schedule (Hoffman 2010).
    batch_size, n_passes
        Minibatch SGD controls.

    Returns
    -------
    TopicsResult
    """
    if not isinstance(n_topics, int) or n_topics < 1:
        raise ValueError(f"n_topics must be a positive integer, got {n_topics!r}")
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    row_ptr, col_idx, counts, n_words, cell_names, peak_names = _coerce(expression)

    if n_words == 0:
        raise ValueError("expression has 0 peaks/genes — nothing to model")

    if alpha is None:
        alpha = 1.0 / n_topics
    if eta is None:
        eta = 1.0 / n_topics

    import sys, time
    n_docs = len(row_ptr) - 1
    nnz = len(col_idx)
    if verbose:
        print(
            f"[rustscenic.topics] online-VB LDA — {n_docs:,} docs × "
            f"{n_words:,} vocab (nnz={nnz:,}), K={n_topics}, {n_passes} passes, "
            f"batch_size={batch_size}. Running in parallel...",
            file=sys.stderr, flush=True,
        )
    t0 = time.monotonic()
    ct, tw = _topics_fit(
        list(row_ptr), list(col_idx), list(counts.astype(np.float32)),
        int(n_words), int(n_topics),
        float(alpha), float(eta), float(tau0), float(kappa),
        int(batch_size), int(n_passes), int(seed),
    )
    wall = time.monotonic() - t0
    topic_names = [f"Topic_{k}" for k in range(n_topics)]
    cell_topic = pd.DataFrame(np.asarray(ct), index=cell_names, columns=topic_names)
    topic_peak = pd.DataFrame(np.asarray(tw), index=topic_names, columns=peak_names)
    unique = int(np.unique(cell_topic.values.argmax(axis=1)).size)
    if verbose:
        print(
            f"[rustscenic.topics] done in {wall:.1f}s — "
            f"{unique}/{n_topics} topics carry an argmax assignment.",
            file=sys.stderr, flush=True,
        )
    return TopicsResult(cell_topic=cell_topic, topic_peak=topic_peak, n_topics=n_topics)


def fit_gibbs(
    expression,
    *,
    n_topics: int = 50,
    alpha: Optional[float] = None,
    eta: Optional[float] = None,
    n_iters: int = 200,
    seed: int = 42,
    n_threads: int = 1,
    verbose: bool = True,
) -> TopicsResult:
    """Fit collapsed-Gibbs LDA on a (cells × peaks) count / binarized matrix.

    The Mallet-class topic model — better topic-coherence (NPMI) on
    sparse scATAC at K ≥ 30 than the default online-VB
    :func:`fit`, at the cost of thousands of iterations instead of tens
    of passes. Use this when topic quality matters more than wall-clock,
    typically for small-to-medium samples where you can afford the
    Gibbs sampling time.

    Parameters
    ----------
    expression
        AnnData, pandas DataFrame, or (sparse-csr, cell_names, peak_names) tuple.
    n_topics
        Number of latent topics K. Mallet typical range: 30–100.
    alpha, eta
        Dirichlet priors. Default 0.1 / 0.01 — Griffiths & Steyvers
        2004's "good defaults" for LDA, slightly less concentrated than
        the 1/K we use for online VB.
    n_iters
        Number of Gibbs sweeps over the corpus. 200 is a reasonable
        default for convergence on small samples; bump to 500–1000 for
        higher-quality posterior estimates.
    seed
        Random seed. Topics are stochastic — bit-identical under same
        seed (single-threaded), reproducible across runs at fixed
        ``n_threads`` for the parallel path.
    n_threads
        ``1`` (default): bit-deterministic serial sampler. ``>1``:
        AD-LDA (Newman et al. 2009) parallel sampler — partitions docs
        across threads, near-linear speedup on atlas-scale corpora at
        the cost of small cross-thread staleness within a sweep
        (perplexity gap is well within sampling variance per Newman
        2009 §4). Recommended for K ≥ 30 runs over 50k+ cells.

    Returns
    -------
    TopicsResult — same shape as :func:`fit`, columns are
    ``Topic_0 .. Topic_{K-1}``.
    """
    if not isinstance(n_topics, int) or n_topics < 1:
        raise ValueError(f"n_topics must be a positive integer, got {n_topics!r}")
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")

    row_ptr, col_idx, counts, n_words, cell_names, peak_names = _coerce(expression)
    if n_words == 0:
        raise ValueError("expression has 0 peaks/genes — nothing to model")
    if alpha is None:
        alpha = 0.1
    if eta is None:
        eta = 0.01

    import sys, time
    n_docs = len(row_ptr) - 1
    nnz = len(col_idx)
    if verbose:
        thread_label = "serial" if n_threads == 1 else f"{n_threads}-thread AD-LDA"
        print(
            f"[rustscenic.topics] collapsed-Gibbs LDA ({thread_label}) — "
            f"{n_docs:,} docs × {n_words:,} vocab (nnz={nnz:,}), K={n_topics}, "
            f"{n_iters} sweeps, alpha={alpha}, eta={eta}",
            file=sys.stderr, flush=True,
        )
    t0 = time.monotonic()
    theta, beta = _topics_fit_gibbs(
        list(row_ptr), list(col_idx), list(counts.astype(np.float32)),
        int(n_words), int(n_topics),
        float(alpha), float(eta), int(n_iters), int(seed), int(n_threads),
    )
    wall = time.monotonic() - t0
    topic_names = [f"Topic_{k}" for k in range(n_topics)]
    cell_topic = pd.DataFrame(np.asarray(theta), index=cell_names, columns=topic_names)
    topic_peak = pd.DataFrame(np.asarray(beta), index=topic_names, columns=peak_names)
    unique = int(np.unique(cell_topic.values.argmax(axis=1)).size)
    if verbose:
        print(
            f"[rustscenic.topics] Gibbs done in {wall:.1f}s — "
            f"{unique}/{n_topics} topics carry an argmax assignment.",
            file=sys.stderr, flush=True,
        )
    return TopicsResult(cell_topic=cell_topic, topic_peak=topic_peak, n_topics=n_topics)


def coherence_npmi(
    result: TopicsResult,
    corpus,
    *,
    top_n: int = 10,
) -> np.ndarray:
    """Per-topic NPMI coherence for a fitted topic model.

    Parameters
    ----------
    result
        :class:`TopicsResult` from :func:`fit` or :func:`fit_gibbs`.
    corpus
        Corpus to score against (AnnData / DataFrame / sparse-tuple,
        same shape conventions as :func:`fit`). Should have the same
        peak/word vocabulary as ``result`` — column order must match
        ``result.topic_peak.columns``.
    top_n
        Top-N peaks per topic to evaluate pairwise NPMI over. 10 is
        standard for LDA topic-coherence.

    Returns
    -------
    np.ndarray of shape (n_topics,) — mean pairwise NPMI per topic.
    Higher is better; positive values mean top-words co-occur more
    than independence would predict.
    """
    row_ptr, col_idx, _, n_words, _, peak_names = _coerce(corpus)
    if list(peak_names) != list(result.topic_peak.columns):
        raise ValueError(
            "corpus column order does not match the fit's topic_peak columns; "
            "supply the same peak/word ordering used at fit time"
        )
    tw = np.ascontiguousarray(result.topic_peak.values, dtype=np.float32)
    out = _topics_npmi(
        tw,
        int(result.n_topics),
        int(n_words),
        list(row_ptr),
        list(col_idx),
        int(top_n),
    )
    return np.asarray(out)


def _coerce(expression):
    """Return (row_ptr, col_idx, counts, n_peaks, cell_names, peak_names)."""
    import scipy.sparse as sp

    if hasattr(expression, "X") and hasattr(expression, "var_names"):
        X = expression.X
        cell_names = list(expression.obs_names)
        peak_names = list(expression.var_names)
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
    elif isinstance(expression, pd.DataFrame):
        cell_names = list(expression.index)
        peak_names = list(expression.columns)
        X = sp.csr_matrix(expression.values)
    elif isinstance(expression, tuple) and len(expression) == 3:
        X, cell_names, peak_names = expression
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        X = X.tocsr()
    else:
        raise TypeError("expression must be AnnData, DataFrame, or (sparse, cells, peaks) tuple")

    if X.nnz > np.iinfo(np.uint32).max:
        raise OverflowError(
            f"input matrix has {X.nnz} nonzeros, exceeding uint32 max "
            f"({np.iinfo(np.uint32).max}). Subset or bin the matrix first."
        )
    if X.shape[1] > np.iinfo(np.uint32).max:
        raise OverflowError(f"too many features/peaks ({X.shape[1]}) for uint32 index")
    return (
        np.asarray(X.indptr, dtype=np.int64),
        np.asarray(X.indices, dtype=np.uint32),
        np.asarray(X.data, dtype=np.float32),
        X.shape[1],
        cell_names,
        peak_names,
    )
