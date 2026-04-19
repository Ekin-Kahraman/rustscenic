"""Type stubs for the rustscenic PyO3 extension.

These reflect the real signatures in crates/rustscenic-py/src/lib.rs.
Kept in sync by convention — CI's import-smoke step will complain loudly
if these drift.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str


def grn_infer(
    expression: npt.NDArray[np.float32],
    gene_names: list[str],
    tf_names: list[str],
    n_estimators: int = 5000,
    learning_rate: float = 0.01,
    max_features: float = 0.1,
    subsample: float = 0.9,
    max_depth: int = 3,
    early_stop_window: int = 25,
    seed: int = 777,
) -> tuple[list[str], list[str], npt.NDArray[np.float32]]:
    """Infer (TF, target, importance) edges from an expression matrix.

    Returns three parallel arrays: tf names, target names, importance scores.
    """
    ...


def aucell_score(
    expression: npt.NDArray[np.float32],
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
) -> npt.NDArray[np.float32]:
    """Per-cell regulon recovery AUCs.

    Returns a (n_cells, n_regulons) f32 array of normalised AUCs in [0, 1].
    """
    ...


def topics_fit(
    row_ptr: list[int],
    col_idx: list[int],
    counts: list[float],
    n_words: int,
    n_topics: int = 50,
    alpha: float = 0.02,
    eta: float = 0.02,
    tau0: float = 64.0,
    kappa: float = 0.7,
    batch_size: int = 256,
    n_passes: int = 10,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fit Online VB LDA; return (cell_topic, topic_word) probability matrices."""
    ...
