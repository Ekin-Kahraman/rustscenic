"""Seeded AUCell rank agreement, including ties at sparse zero boundaries."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from rustscenic.aucell import score
from rustscenic import _rustscenic as native


def reference(x, genes, sets, seed, fraction=.1):
    # pySCENIC: one seeded column permutation, then stable descending ranking.
    permutation = pd.Series(np.arange(len(genes))).sample(frac=1, random_state=seed).to_numpy()
    order = permutation[np.argsort(-x[:, permutation], axis=1, kind="stable")]
    ranks = np.empty_like(order)
    ranks[np.arange(len(x))[:, None], order] = np.arange(len(genes))
    cutoff = round(fraction * len(genes)) - 1
    return np.column_stack([
        np.maximum(0, cutoff - ranks[:, [genes.index(g) for g in targets]]).sum(1)
        / ((cutoff + 1) * len(targets)) for targets in sets.values()
    ])


@pytest.mark.parametrize("seed", [0, 777, 20261106])
@pytest.mark.parametrize("form", ["dense", "strided", "csr", "csc"])
def test_matches_seeded_reference_with_positive_zero_and_negative_ties(seed, form):
    rng = np.random.default_rng(41)
    x = rng.choice([-1., 0., 0., 0., 1., 1., 2.], (13, 100)).astype("float32")
    x[0] = 0
    x[1] = -1
    x[1, :2] = [3, 0]  # Ranking crosses positive, zero and negative values.
    genes = [f"g{i}" for i in range(100)]
    sets = {"positive": genes[:15], "negative": genes[20:31]}
    value = {"dense": lambda: x, "strided": lambda: np.asfortranarray(x),
             "csr": lambda: sparse.csr_matrix(x), "csc": lambda: sparse.csc_matrix(x)}[form]()
    original = value.copy()
    result = score((value, genes), sets, top_frac=.1, seed=seed)
    np.testing.assert_allclose(result.to_numpy(), reference(x, genes, sets, seed), rtol=1e-7, atol=3e-8)
    if sparse.issparse(value):
        assert (value != original).nnz == 0
    else:
        np.testing.assert_array_equal(value, original)
    assert result.attrs["aucell_seed"] == seed


def test_seeded_sparse_never_densifies_or_shuffles_matrix(monkeypatch):
    x = sparse.csr_matrix(np.random.default_rng(4).poisson(.2, (20, 100)).astype("float32"))
    def forbidden(*args, **kwargs):
        pytest.fail("Seeded AUCell copied or densified the expression matrix")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csr_matrix, "__getitem__", forbidden)
    result = score((x, [f"g{i}" for i in range(100)]), {"R": ["g0", "g1"]}, seed=3)
    assert result.shape == (20, 1)


def test_seeded_block_invariance_and_default_compatibility():
    x = sparse.csr_matrix(np.random.default_rng(4).poisson(.2, (20, 100)).astype("float32"))
    genes = [f"g{i}" for i in range(100)]
    sets = {"R": genes[:20]}
    whole = score((x, genes), sets, seed=777)
    parts = np.vstack([score((x[i:i+3], genes), sets, seed=777).to_numpy() for i in range(0, 20, 3)])
    np.testing.assert_array_equal(whole.to_numpy(), parts)
    np.testing.assert_array_equal(score((x, genes), sets), score((x, genes), sets, seed=None))
    different = score((x, genes), sets, seed=778)
    assert not np.array_equal(whole.to_numpy(), different.to_numpy())


@pytest.mark.parametrize("order", [[0], [0, 1, 2, 3, 5], [0, 1, 2, 3, 3]])
def test_invalid_backend_permutation_rejected(order):
    x = np.ones((2, 5), dtype="float32")
    csr = sparse.csr_matrix(x)
    with pytest.raises(ValueError, match="permutation"):
        native.aucell_score(x, ["R"], [[0]], .4, gene_order=order)
    with pytest.raises(ValueError, match="permutation"):
        native.aucell_score_sparse_csr(csr.indptr, csr.indices, csr.data, 5, ["R"], [[0]], .4, gene_order=order)


def test_seed_is_applied_after_duplicate_gene_resolution():
    genes = ["g0", "g0"] + [f"g{i}" for i in range(1, 100)]
    x = np.random.default_rng(5).poisson(.4, (8, len(genes))).astype("float32")
    merged = np.column_stack([x[:, 0] + x[:, 1], x[:, 2:]])
    with pytest.warns(UserWarning, match="duplicate gene"):
        result = score((sparse.csr_matrix(x), genes), {"R": ["g0", "g5"]}, seed=777)
    expected = score((merged, [f"g{i}" for i in range(100)]), {"R": ["g0", "g5"]}, seed=777)
    np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())
