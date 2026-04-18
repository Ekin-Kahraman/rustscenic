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

from rustscenic._rustscenic import topics_fit as _topics_fit


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

    ct, tw = _topics_fit(
        list(row_ptr), list(col_idx), list(counts.astype(np.float32)),
        int(n_words), int(n_topics),
        float(alpha), float(eta), float(tau0), float(kappa),
        int(batch_size), int(n_passes), int(seed),
    )
    topic_names = [f"Topic_{k}" for k in range(n_topics)]
    cell_topic = pd.DataFrame(np.asarray(ct), index=cell_names, columns=topic_names)
    topic_peak = pd.DataFrame(np.asarray(tw), index=topic_names, columns=peak_names)
    return TopicsResult(cell_topic=cell_topic, topic_peak=topic_peak, n_topics=n_topics)


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
