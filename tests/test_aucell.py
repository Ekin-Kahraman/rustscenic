"""Tests for rustscenic.aucell.score."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import rustscenic.aucell as aucell


class TestAucellShape:
    def test_returns_cells_by_regulons_df(self, small_expr, canonical_regulons):
        auc = aucell.score(small_expr, canonical_regulons, top_frac=0.1)
        assert auc.shape == (small_expr.shape[0], len(canonical_regulons))
        assert list(auc.columns) == [n for n, _ in canonical_regulons]
        assert list(auc.index) == list(small_expr.index)

    def test_auc_values_in_unit_interval(self, small_expr, canonical_regulons):
        auc = aucell.score(small_expr, canonical_regulons, top_frac=0.1)
        assert (auc.values >= 0).all()
        assert (auc.values <= 1.0).all()


class TestAucellCorrectness:
    def test_matches_ctxcore_on_small_case(self):
        """Bit-exact parity with ctxcore.recovery.aucs (hand-verified).

        5 genes, 1 cell, regulon = {g0} (highest-ranked), top_frac=0.4
          → rank_cutoff = round(0.4*5) - 1 = 1
          → only g0 at rank 0 passes the filter
          → auc_sum = (1 - 0) * 1 = 1
          → max_auc = (1+1) * 1 = 2
          → 1 / 2 = 0.5
        """
        df = pd.DataFrame([[10.0, 1.0, 1.0, 1.0, 1.0]], columns=list("abcde"), index=["c0"])
        out = aucell.score(df, [("R", ["a"])], top_frac=0.4)
        assert abs(out.values[0, 0] - 0.5) < 1e-6

    def test_regulon_outside_topk_gives_zero(self):
        # 10 genes, 1 cell, regulon = {g9} (lowest-ranked), top_frac=0.1 (K=1, K_ctx=0)
        # No ranks < 0, so auc_sum = 0
        df = pd.DataFrame([list(range(10, 0, -1))],
                          columns=[f"g{i}" for i in range(10)], index=["c0"],
                          dtype=np.float32)
        out = aucell.score(df, [("R", ["g9"])], top_frac=0.1)
        assert out.values[0, 0] == 0.0


class TestAucellEdgeCases:
    def test_regulon_with_no_matching_genes_is_dropped(self, small_expr):
        regs = [("gone", ["FOREIGN1", "FOREIGN2"]), ("ok", ["g0", "g1"])]
        with pytest.warns(UserWarning):
            auc = aucell.score(small_expr, regs, top_frac=0.1)
        assert list(auc.columns) == ["ok"]

    def test_duplicate_gene_names_raises_valueerror(self, rng):
        X = rng.random((20, 5)).astype(np.float32)
        df = pd.DataFrame(X, columns=["a", "b", "c", "a", "e"])  # duplicate 'a'
        with pytest.raises(ValueError, match="duplicate gene name"):
            aucell.score(df, [("R", ["a", "b"])], top_frac=0.2)

    def test_nan_input_panics(self, rng):
        X = rng.random((10, 5)).astype(np.float32)
        X[0, 0] = np.nan
        df = pd.DataFrame(X, columns=list("abcde"))
        with pytest.raises(BaseException, match=r"Na[Nn]"):
            aucell.score(df, [("R", ["a", "b"])], top_frac=0.2)

    def test_empty_regulons_list_returns_empty_df(self, small_expr):
        auc = aucell.score(small_expr, [], top_frac=0.1)
        assert auc.shape == (small_expr.shape[0], 0)


class TestAucellDeterminism:
    def test_reproducible_across_runs(self, small_expr, canonical_regulons):
        a = aucell.score(small_expr, canonical_regulons, top_frac=0.1)
        b = aucell.score(small_expr, canonical_regulons, top_frac=0.1)
        assert a.equals(b)


class TestAucellSparse:
    def test_sparse_input_same_as_dense(self, small_expr, canonical_regulons):
        """Passing an AnnData with sparse X should give identical AUCs."""
        import anndata as ad
        X_sparse = sp.csr_matrix(small_expr.values.astype(np.float32))
        a_dense = aucell.score(small_expr, canonical_regulons, top_frac=0.1)
        adata = ad.AnnData(X=X_sparse,
                           obs=pd.DataFrame(index=small_expr.index),
                           var=pd.DataFrame(index=small_expr.columns))
        a_sparse = aucell.score(adata, canonical_regulons, top_frac=0.1)
        np.testing.assert_allclose(a_dense.values, a_sparse.values, atol=1e-6)

    def test_chunking_gives_same_answer(self, small_expr, canonical_regulons):
        """chunk_size < n_cells must not change the result."""
        import anndata as ad
        X_sparse = sp.csr_matrix(small_expr.values.astype(np.float32))
        adata = ad.AnnData(X=X_sparse,
                           obs=pd.DataFrame(index=small_expr.index),
                           var=pd.DataFrame(index=small_expr.columns))
        whole = aucell.score(adata, canonical_regulons, top_frac=0.1, chunk_size=1_000_000)
        chunked = aucell.score(adata, canonical_regulons, top_frac=0.1, chunk_size=13)
        np.testing.assert_allclose(whole.values, chunked.values, atol=1e-6)
        assert list(whole.index) == list(chunked.index)
