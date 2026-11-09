"""Check critical statistical and rank semantics independently of the real data."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from trace import decode21, enrichment
from case_study import assess, score_sets, verify_rank_cache


def test_rank_transport_and_auc_boundary():
    values = np.array([0, 1, 9185, 9186, 9187, 1837303, 0, 2**21 - 1], dtype="<u4")
    bits = np.unpackbits(values.view(np.uint8).reshape(-1, 4), axis=1, bitorder="little")[:, :21]
    packed = np.packbits(bits.ravel(), bitorder="little").tobytes()
    np.testing.assert_array_equal(decode21(packed, 8), values)
    # Cutoff 3: rank 0 contributes 3; rank 2 contributes 1; rank 3 contributes 0.
    ranks = np.array([[0, 2, 3], [3, 3, 3], [1, 3, 3]])
    contributions = sparse.csr_matrix(np.maximum(3 - ranks, 0))
    membership = sparse.csc_matrix(np.array([[1], [1], [1]]))
    auc, nes = enrichment(contributions, membership, 3)
    np.testing.assert_allclose(auc[:, 0], [1 / 3, 0, 1 / 6])
    np.testing.assert_allclose(nes[:, 0], np.array([1, -1, 0]) * np.sqrt(1.5))


def test_lineage_missing_is_not_zero_and_donor_axes_align():
    cells = [f"c{i}" for i in range(80)]
    meta = pd.DataFrame({"donor": np.repeat(["a", "b"], 40),
                         "cell_type": ["CD14+ Mono"] * 20 + ["NK"] * 20 + ["NK"] * 20 + ["CD14+ Mono"] * 20}, index=cells)
    scores = pd.DataFrame({"SPI1": meta.cell_type.eq("CD14+ Mono").astype(float)}, index=cells).iloc[::-1]
    rows = pd.DataFrame(assess(scores, meta, "BMMC", 20000, 777, "test", "Gene activity"))
    assert rows[rows.TF.eq("SPI1")].auroc.tolist() == [1., 1.]
    assert rows[rows.TF.eq("PAX5")].status.eq("missing_regulon").all()
    assert rows[rows.TF.eq("PAX5")].auroc.isna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        assess(pd.concat([scores, scores]), meta, "BMMC", 20000, 777, "test", "Gene activity")


def test_signature_coverage_and_empty_sets():
    genes = [f"g{i}" for i in range(100)]
    contributions = sparse.csr_matrix(([4., 2.], ([0, 0], [0, 1])), shape=(1, 100))
    result = score_sets(contributions, genes, ["cell"], {
        "both": ["g0", "g1"], "poor_coverage": ["g0", "missing"], "empty": []})
    assert result.loc["cell", "both"] == pytest.approx(.6)
    assert result.loc["cell", "poor_coverage"] == 0
    assert result.loc["cell", "empty"] == 0


def test_rank_cache_verification_rejects_misaligned_values():
    genes = 100
    expression = sparse.csr_matrix(np.arange(genes, dtype=float).reshape(1, -1))
    cache = sparse.csr_matrix(([4., 3., 2., 1.], ([0] * 4, [99, 98, 97, 96])), shape=(1, genes))
    verify_rank_cache(expression, cache)
    cache.data[0] += 1
    with pytest.raises(ValueError, match="Rank cache"):
        verify_rank_cache(expression, cache)


def test_equal_integer_auc_sums_remain_exactly_tied():
    genes = [f"g{i}" for i in range(100)]
    ranks = sparse.csr_matrix(([4., 1., 1., 2., 2., 2.],
        ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])), shape=(2, 100))
    result = score_sets(ranks, genes, ["a", "b"], {"TF": genes[:3]})
    assert result.loc["a", "TF"] == result.loc["b", "TF"] == .4
