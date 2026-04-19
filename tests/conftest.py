"""Shared fixtures for rustscenic tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_expr(rng):
    """Tiny (cells × genes) dense DataFrame for fast tests."""
    n_cells, n_genes = 60, 80
    X = rng.poisson(1.5, size=(n_cells, n_genes)).astype(np.float32)
    return pd.DataFrame(
        X,
        index=[f"c{i}" for i in range(n_cells)],
        columns=[f"g{i}" for i in range(n_genes)],
    )


@pytest.fixture
def canonical_regulons():
    return [
        ("R_small", ["g0", "g1", "g2", "g3", "g4"]),
        ("R_medium", [f"g{i}" for i in range(15)]),
    ]
