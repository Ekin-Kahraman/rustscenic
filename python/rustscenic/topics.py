"""Topic modeling (pycisTopic LDA replacement).

Online variational Bayes LDA (Hoffman-Blei-Bach 2010) for scATAC peak-topic
modeling. Converges in tens of passes vs Gibbs's thousands of iterations.

    rustscenic.topics.fit(adata_or_sparse, n_topics=50) -> TopicsResult

Output is a `TopicsResult` namedtuple with:
    cell_topic:  (cells x topics) probability matrix (each row sums to 1)
    topic_peak:  (topics x peaks) probability matrix (each row sums to 1)

Both pycisTopic (Mallet Gibbs) and rustscenic (online VB) are probabilistic -
topic labels are permutation-free. Validation metric is topic assignment ARI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    topics_cell_assignment as _topics_cell_assignment,
    topics_fit as _topics_fit,
    topics_fit_gibbs as _topics_fit_gibbs,
    topics_npmi as _topics_npmi,
)
from rustscenic.specificity import (
    CandidateEnhancers,
    candidate_enhancers_per_topic as _candidate_enhancers_per_topic,
)


class RustBackendArray(np.ndarray):
    """ndarray subclass carrying exact Rust backend provenance."""

    rust_backend: dict


@dataclass
class TopicsResult:
    cell_topic: pd.DataFrame   # (cells x topics)
    topic_peak: pd.DataFrame   # (topics x peaks)
    n_topics: int
    backend_execution: dict | None = None

    def cell_assignment(self) -> pd.Series:
        """Argmax topic per cell."""
        import warnings

        assignment_idx, _, n_empty = _topic_assignment_indices(self.cell_topic)
        topic_names = np.asarray(self.cell_topic.columns, dtype=object)
        values = np.empty(len(assignment_idx), dtype=object)
        valid = assignment_idx >= 0
        values[valid] = topic_names[assignment_idx[valid]]
        values[~valid] = pd.NA
        if n_empty:
            warnings.warn(
                f"{n_empty} cells have zero or non-finite total topic weight; "
                "their topic assignment is set to NA instead of Topic_0.",
                UserWarning,
                stacklevel=2,
            )
        out = pd.Series(values, index=self.cell_topic.index, dtype="object")
        out.attrs["rust_backend"] = {
            "engine": "rust",
            "symbols": ["topics_cell_assignment"],
        }
        return out

    def top_peaks_per_topic(self, n: int = 20) -> CandidateEnhancers:
        return _candidate_enhancers_per_topic(self.topic_peak, top_n=n)


def _topic_assignment_indices(cell_topic: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    values = cell_topic.values.astype(np.float32, copy=False)
    assignment_idx, active_topics, empty_cells = _topics_cell_assignment(values)
    return np.asarray(assignment_idx, dtype=np.int64), int(active_topics), int(empty_cells)


def assignment_summary(cell_topic: pd.DataFrame) -> dict[str, int]:
    """Summarise topic usage with the Rust assignment kernel.

    Returns active argmax-topic and empty-cell counts without materialising a
    Python boolean mask over the cells-by-topics matrix.
    """
    _, active_topics, empty_cells = _topic_assignment_indices(cell_topic)
    return {
        "active_argmax_topics": active_topics,
        "empty_cells": empty_cells,
    }


def _warn_on_topic_collapse(active_topics: int, n_topics: int, method: str) -> None:
    if active_topics >= (n_topics + 1) // 2:
        return
    import warnings

    recommendation = (
        "For sparse scATAC at K >= 30, retry with fit_gibbs and record a "
        "fixed n_threads value."
        if method == "vb"
        else "Increase n_iters and validate coherence before interpretation."
    )
    warnings.warn(
        f"topic model collapse: only {active_topics} of {n_topics} topics "
        f"carry any cell argmax assignment. {recommendation}",
        UserWarning,
        stacklevel=3,
    )


def fit(
    expression,
    *,
    n_topics: int = 50,
    alpha: float | None = None,
    eta: float | None = None,
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
        raise ValueError("expression has 0 peaks/genes - nothing to model")

    if alpha is None:
        alpha = 1.0 / n_topics
    if eta is None:
        eta = 1.0 / n_topics

    import sys
    import time
    n_docs = len(row_ptr) - 1
    nnz = len(col_idx)
    if verbose:
        print(
            f"[rustscenic.topics] online-VB LDA - {n_docs:,} docs × "
            f"{n_words:,} vocab (nnz={nnz:,}), K={n_topics}, {n_passes} passes, "
            f"batch_size={batch_size}. Running in parallel...",
            file=sys.stderr, flush=True,
        )
    t0 = time.monotonic()
    ct, tw = _topics_fit(
        row_ptr, col_idx, counts,
        int(n_words), int(n_topics),
        float(alpha), float(eta), float(tau0), float(kappa),
        int(batch_size), int(n_passes), int(seed),
    )
    wall = time.monotonic() - t0
    topic_names = [f"Topic_{k}" for k in range(n_topics)]
    cell_topic = pd.DataFrame(np.asarray(ct), index=cell_names, columns=topic_names)
    topic_peak = pd.DataFrame(np.asarray(tw), index=topic_names, columns=peak_names)
    backend_execution = {"engine": "rust", "symbols": ["topics_fit"]}
    cell_topic.attrs["rust_backend"] = backend_execution
    topic_peak.attrs["rust_backend"] = backend_execution
    _, unique, _ = _topic_assignment_indices(cell_topic)
    _warn_on_topic_collapse(unique, n_topics, "vb")
    if verbose:
        print(
            f"[rustscenic.topics] done in {wall:.1f}s - "
            f"{unique}/{n_topics} topics carry an argmax assignment.",
            file=sys.stderr, flush=True,
        )
    return TopicsResult(
        cell_topic=cell_topic,
        topic_peak=topic_peak,
        n_topics=n_topics,
        backend_execution=backend_execution,
    )


def fit_gibbs(
    expression,
    *,
    n_topics: int = 50,
    alpha: float | None = None,
    eta: float | None = None,
    n_iters: int = 200,
    seed: int = 42,
    n_threads: int = 1,
    verbose: bool = True,
) -> TopicsResult:
    """Fit collapsed-Gibbs LDA on a (cells × peaks) count / binarized matrix.

    The Mallet-class topic model - better topic-coherence (NPMI) on
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
        Dirichlet priors. Default 0.1 / 0.01 - Griffiths & Steyvers
        2004's "good defaults" for LDA, slightly less concentrated than
        the 1/K we use for online VB.
    n_iters
        Number of Gibbs sweeps over the corpus. 200 is a reasonable
        default for convergence on small samples; bump to 500–1000 for
        higher-quality posterior estimates.
    seed
        Random seed. Topics are stochastic - bit-identical under same
        seed (single-threaded), reproducible across runs at fixed
        ``n_threads`` for the parallel path.
    n_threads
        ``1`` (default): bit-deterministic serial sampler. ``>1``:
        AD-LDA (Newman et al. 2009) parallel sampler - partitions docs
        across threads. Results are reproducible at fixed ``seed`` and
        ``n_threads``. Changing the thread count changes the partition and
        stochastic trajectory and may reach a different posterior mode, so
        keep it fixed when comparing biological outputs. Speedup is workload
        dependent and often saturates once topic-word reads become
        memory-bandwidth bound.

    Returns
    -------
    TopicsResult - same shape as :func:`fit`, columns are
    ``Topic_0 .. Topic_{K-1}``.
    """
    if not isinstance(n_topics, int) or n_topics < 1:
        raise ValueError(f"n_topics must be a positive integer, got {n_topics!r}")
    if n_threads > 1 and n_topics > 65_535:
        raise ValueError("parallel Gibbs supports at most 65,535 topics")
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")

    row_ptr, col_idx, counts, n_words, cell_names, peak_names = _coerce(expression)
    if n_words == 0:
        raise ValueError("expression has 0 peaks/genes - nothing to model")
    if alpha is None:
        alpha = 0.1
    if eta is None:
        eta = 0.01

    import sys
    import time
    n_docs = len(row_ptr) - 1
    nnz = len(col_idx)
    if verbose:
        thread_label = "serial" if n_threads == 1 else f"{n_threads}-thread AD-LDA"
        print(
            f"[rustscenic.topics] collapsed-Gibbs LDA ({thread_label}) - "
            f"{n_docs:,} docs × {n_words:,} vocab (nnz={nnz:,}), K={n_topics}, "
            f"{n_iters} sweeps, alpha={alpha}, eta={eta}",
            file=sys.stderr, flush=True,
        )
    t0 = time.monotonic()
    theta, beta = _topics_fit_gibbs(
        row_ptr, col_idx, counts,
        int(n_words), int(n_topics),
        float(alpha), float(eta), int(n_iters), int(seed), int(n_threads),
    )
    wall = time.monotonic() - t0
    topic_names = [f"Topic_{k}" for k in range(n_topics)]
    cell_topic = pd.DataFrame(np.asarray(theta), index=cell_names, columns=topic_names)
    topic_peak = pd.DataFrame(np.asarray(beta), index=topic_names, columns=peak_names)
    backend_execution = {"engine": "rust", "symbols": ["topics_fit_gibbs"]}
    cell_topic.attrs["rust_backend"] = backend_execution
    topic_peak.attrs["rust_backend"] = backend_execution
    _, unique, _ = _topic_assignment_indices(cell_topic)
    _warn_on_topic_collapse(unique, n_topics, "gibbs")
    if verbose:
        print(
            f"[rustscenic.topics] Gibbs done in {wall:.1f}s - "
            f"{unique}/{n_topics} topics carry an argmax assignment.",
            file=sys.stderr, flush=True,
        )
    return TopicsResult(
        cell_topic=cell_topic,
        topic_peak=topic_peak,
        n_topics=n_topics,
        backend_execution=backend_execution,
    )


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
        peak/word vocabulary as ``result`` - column order must match
        ``result.topic_peak.columns``.
    top_n
        Top-N peaks per topic to evaluate pairwise NPMI over. 10 is
        standard for LDA topic-coherence.

    Returns
    -------
    np.ndarray of shape (n_topics,) - mean pairwise NPMI per topic.
    Higher is better; positive values mean top-words co-occur more
    than independence would predict.
    """
    row_ptr, col_idx, _, n_words, _, peak_names = _coerce(corpus)
    if list(peak_names) != list(result.topic_peak.columns):
        raise ValueError(
            "corpus column order does not match the fit's topic_peak columns; "
            "supply the same peak/word ordering used at fit time"
        )
    tw = result.topic_peak.values.astype(np.float32, copy=False)
    out = _topics_npmi(
        tw,
        int(result.n_topics),
        int(n_words),
        row_ptr,
        col_idx,
        int(top_n),
    )
    scores = np.asarray(out).view(RustBackendArray)
    scores.rust_backend = {"engine": "rust", "symbols": ["topics_npmi"]}
    return scores


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

    if X.shape[1] > np.iinfo(np.int32).max:
        raise OverflowError(f"too many features/peaks ({X.shape[1]}) for int32 index")
    X.sum_duplicates()
    return (
        _csr_indptr_arg(X.indptr),
        _csr_indices_arg(X.indices),
        _csr_counts_arg(X.data),
        X.shape[1],
        cell_names,
        peak_names,
    )


def _csr_indptr_arg(indptr) -> np.ndarray:
    arr = np.asarray(indptr)
    if arr.dtype not in (np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.uint64)):
        arr = arr.astype(np.int64, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _csr_indices_arg(indices) -> np.ndarray:
    arr = np.asarray(indices)
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _csr_counts_arg(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr
