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

    def test_equal_expression_ties_break_by_gene_order(self):
        df = pd.DataFrame([[1.0, 1.0, 1.0, 1.0, 1.0]], columns=list("abcde"), index=["c0"])
        out = aucell.score(df, [("first", ["a"]), ("last", ["e"])], top_frac=0.4)
        assert abs(out.loc["c0", "first"] - 0.5) < 1e-6
        assert out.loc["c0", "last"] == 0.0

    def test_dense_path_passes_strided_float32_without_python_contiguity_copy(self, monkeypatch):
        values = np.asfortranarray(np.arange(60, dtype=np.float32).reshape(12, 5))
        assert values.flags.f_contiguous
        assert not values.flags.c_contiguous
        df = pd.DataFrame(
            values,
            index=[f"c{i}" for i in range(values.shape[0])],
            columns=[f"g{i}" for i in range(values.shape[1])],
        )
        seen = {}

        def fake_score(expression_arg, reg_names, reg_gene_indices, top_frac):
            seen["expression"] = expression_arg
            seen["reg_names"] = reg_names
            seen["reg_gene_indices"] = reg_gene_indices
            seen["top_frac"] = top_frac
            return np.zeros((expression_arg.shape[0], len(reg_names)), dtype=np.float32)

        monkeypatch.setattr(aucell, "_aucell_score", fake_score)

        out = aucell.score(df, [("R", ["g0", "g2"])], top_frac=0.2)

        assert out.shape == (values.shape[0], 1)
        assert out.attrs["rust_backend"]["symbols"] == ["aucell_score"]
        assert np.shares_memory(seen["expression"], values)
        assert seen["expression"].flags.f_contiguous
        assert not seen["expression"].flags.c_contiguous
        assert seen["reg_names"] == ["R"]
        assert seen["reg_gene_indices"] == [[0, 2]]
        assert seen["top_frac"] == 0.2

    def test_dense_path_does_not_allocate_python_isfinite_mask(self, monkeypatch):
        values = np.ones((12, 5), dtype=np.float32)
        df = pd.DataFrame(
            values,
            index=[f"c{i}" for i in range(values.shape[0])],
            columns=[f"g{i}" for i in range(values.shape[1])],
        )

        def fail_isfinite(_values):
            raise AssertionError("dense finite validation should happen in Rust")

        def fake_score(expression_arg, reg_names, *_args):
            return np.zeros((expression_arg.shape[0], len(reg_names)), dtype=np.float32)

        monkeypatch.setattr(aucell.np, "isfinite", fail_isfinite)
        monkeypatch.setattr(aucell, "_aucell_score", fake_score)

        out = aucell.score(df, [("R", ["g0", "g2"])], top_frac=0.2)

        assert out.shape == (values.shape[0], 1)

    def test_dense_strided_float32_matches_c_contiguous_result(self):
        values = np.array(
            [
                [9.0, 1.0, 4.0, 2.0, 3.0],
                [0.0, 8.0, 7.0, 1.0, 2.0],
                [5.0, 5.0, 4.0, 3.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        )
        genes = [f"g{i}" for i in range(values.shape[1])]
        cells = [f"c{i}" for i in range(values.shape[0])]
        regulons = [("R1", ["g0", "g2"]), ("R2", ["g3", "g4"])]

        got = aucell.score(
            pd.DataFrame(np.asfortranarray(values), index=cells, columns=genes),
            regulons,
            top_frac=0.6,
        )
        expected = aucell.score(
            pd.DataFrame(values.copy(order="C"), index=cells, columns=genes),
            regulons,
            top_frac=0.6,
        )

        pd.testing.assert_frame_equal(got, expected)


class TestAucellEdgeCases:
    def test_regulon_index_preparation_runs_through_rust_helper(self, monkeypatch):
        import rustscenic._stage_utils as stage_utils

        seen = {}

        def fake_prepare(gene_names, regulon_names, regulon_genes):
            seen["gene_names"] = gene_names
            seen["regulon_names"] = regulon_names
            seen["regulon_genes"] = regulon_genes
            return ["R"], [[1, 0]], 1

        monkeypatch.setattr(stage_utils, "_stage_prepare_regulon_indices", fake_prepare)

        reg_names, reg_gene_indices, reg_pairs, dropped = stage_utils.prepare_regulon_indices(
            ["g0", "g1", "g2"],
            [("empty", ["missing"]), ("R", ["g1", "g0"])],
        )

        assert seen == {
            "gene_names": ["g0", "g1", "g2"],
            "regulon_names": ["empty", "R"],
            "regulon_genes": [["missing"], ["g1", "g0"]],
        }
        assert reg_names == ["R"]
        assert reg_gene_indices == [[1, 0]]
        assert reg_pairs == [("empty", ["missing"]), ("R", ["g1", "g0"])]
        assert dropped == 1

    def test_regulon_index_preparation_with_coverage_runs_one_rust_helper(self, monkeypatch):
        import rustscenic._stage_utils as stage_utils

        seen = {}

        def fake_prepare(gene_names, regulon_names, regulon_genes):
            seen["gene_names"] = gene_names
            seen["regulon_names"] = regulon_names
            seen["regulon_genes"] = regulon_genes
            return (
                ["R"],
                [[1, 0]],
                np.asarray([0, 2], dtype=np.uint64),
                np.asarray([1, 2], dtype=np.uint64),
                1,
            )

        monkeypatch.setattr(
            stage_utils,
            "_stage_prepare_regulon_indices_with_coverage",
            fake_prepare,
        )

        reg_names, reg_gene_indices, reg_pairs, coverage, dropped = (
            stage_utils.prepare_regulon_indices_with_coverage(
                ["g0", "g1", "g2"],
                [("empty", ["missing"]), ("R", ["g1", "g0"])],
            )
        )

        assert seen == {
            "gene_names": ["g0", "g1", "g2"],
            "regulon_names": ["empty", "R"],
            "regulon_genes": [["missing"], ["g1", "g0"]],
        }
        assert reg_names == ["R"]
        assert reg_gene_indices == [[1, 0]]
        assert reg_pairs == [("empty", ["missing"]), ("R", ["g1", "g0"])]
        assert coverage == {"empty": (0, 1), "R": (2, 2)}
        assert dropped == 1

    def test_regulon_with_no_matching_genes_is_dropped(self, small_expr):
        regs = [("gone", ["FOREIGN1", "FOREIGN2"]), ("ok", ["g0", "g1"])]
        with pytest.warns(UserWarning):
            auc = aucell.score(small_expr, regs, top_frac=0.1)
        assert list(auc.columns) == ["ok"]

    def test_duplicate_gene_names_auto_dedupes_with_warning(self, rng):
        """Duplicate symbols (common after ENSEMBL→HGNC swap) sum rather
        than raise - losing half the signal to a cryptic error is the
        silent-regression class we're here to prevent."""
        import warnings

        X = rng.random((20, 5)).astype(np.float32)
        df = pd.DataFrame(X, columns=["a", "b", "c", "a", "e"])  # duplicate 'a'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            auc = aucell.score(df, [("R", ["a", "b"])], top_frac=0.2)
        assert auc.shape == (20, 1)
        dedup_msgs = [
            w for w in caught if "duplicate gene name" in str(w.message)
        ]
        assert dedup_msgs, "auto-dedupe should emit a warning"

    def test_nan_input_raises_value_error(self, rng):
        X = rng.random((10, 5)).astype(np.float32)
        X[0, 0] = np.nan
        df = pd.DataFrame(X, columns=list("abcde"))
        with pytest.raises(ValueError, match=r"NaN|Inf"):
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

    def test_sparse_zero_tie_boundary_matches_dense(self):
        """Implicit sparse zeros must rank like dense zeros by gene order."""
        import anndata as ad
        X = np.array(
            [
                [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        genes = [f"g{i}" for i in range(X.shape[1])]
        cells = [f"c{i}" for i in range(X.shape[0])]
        regulons = {
            "early_zero": ["g1", "g2"],
            "late_zero": ["g8", "g9"],
            "positive": ["g0", "g2"],
        }
        dense = aucell.score(pd.DataFrame(X, index=cells, columns=genes), regulons, top_frac=0.5)
        adata = ad.AnnData(
            X=sp.csr_matrix(X),
            obs=pd.DataFrame(index=cells),
            var=pd.DataFrame(index=genes),
        )
        sparse = aucell.score(adata, regulons, top_frac=0.5)
        np.testing.assert_allclose(dense.values, sparse.values, atol=1e-6)

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

    def test_chunk_size_zero_raises_value_error(self, small_expr, canonical_regulons):
        import anndata as ad
        X_sparse = sp.csr_matrix(small_expr.values.astype(np.float32))
        adata = ad.AnnData(
            X=X_sparse,
            obs=pd.DataFrame(index=small_expr.index),
            var=pd.DataFrame(index=small_expr.columns),
        )
        with pytest.raises(ValueError, match="chunk_size"):
            aucell.score(adata, canonical_regulons, top_frac=0.1, chunk_size=0)

    def test_chunk_size_negative_raises_value_error(self, small_expr, canonical_regulons):
        import anndata as ad
        X_sparse = sp.csr_matrix(small_expr.values.astype(np.float32))
        adata = ad.AnnData(
            X=X_sparse,
            obs=pd.DataFrame(index=small_expr.index),
            var=pd.DataFrame(index=small_expr.columns),
        )
        with pytest.raises(ValueError, match="chunk_size"):
            aucell.score(adata, canonical_regulons, top_frac=0.1, chunk_size=-5)

    def test_sparse_path_passes_int32_indices_without_copy(self, monkeypatch):
        import anndata as ad

        X_sparse = sp.csr_matrix(
            np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        )
        seen = {}

        def fake_sparse_score(row_ptr, col_idx, data, n_genes, reg_names, *_args):
            seen["row_ptr"] = row_ptr
            seen["col_idx"] = col_idx
            seen["data"] = data
            return np.zeros((len(row_ptr) - 1, len(reg_names)), dtype=np.float32)

        monkeypatch.setattr(aucell, "_aucell_score_sparse_csr", fake_sparse_score)
        adata = ad.AnnData(
            X=X_sparse,
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )

        out = aucell.score(adata, {"R": ["g0", "g2"]}, top_frac=0.5)

        assert out.shape == (2, 1)
        assert out.attrs["rust_backend"]["symbols"] == ["aucell_score_sparse_csr"]
        assert seen["row_ptr"].dtype == X_sparse.indptr.dtype
        assert seen["col_idx"].dtype == np.int32
        assert seen["data"].dtype == np.float32
        assert np.shares_memory(seen["row_ptr"], X_sparse.indptr)
        assert np.shares_memory(seen["col_idx"], X_sparse.indices)


def test_aucell_top_frac_zero_raises():
    """top_frac=0 used to silently score every cell to zero. Now raises."""
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((10, 5)).astype(np.float32)
    df = pd.DataFrame(X, columns=list("abcde"))
    with pytest.raises(ValueError, match="top_frac"):
        aucell.score(df, [("R", ["a", "b"])], top_frac=0.0)


def test_aucell_top_frac_over_one_raises():
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((10, 5)).astype(np.float32)
    df = pd.DataFrame(X, columns=list("abcde"))
    with pytest.raises(ValueError, match="top_frac"):
        aucell.score(df, [("R", ["a", "b"])], top_frac=1.5)


def test_aucell_top_frac_high_warns():
    """Unusually high top_frac saturates regulons - warn so users notice."""
    import warnings
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((10, 5)).astype(np.float32)
    df = pd.DataFrame(X, columns=list("abcde"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aucell.score(df, [("R", ["a", "b"])], top_frac=0.5)
    msgs = [str(w.message) for w in caught]
    assert any("top_frac=0.5" in m and "unusually high" in m for m in msgs)


def test_aucell_accepts_dict_regulons():
    """Docstring advertises dict form; it must actually work."""
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((20, 5)).astype(np.float32)
    df = pd.DataFrame(X, columns=list("abcde"))
    auc = aucell.score(df, {"R1": ["a", "b"], "R2": ["c", "d"]}, top_frac=0.2)
    assert auc.shape == (20, 2)
    assert set(auc.columns) == {"R1", "R2"}


def test_aucell_backed_anndata_materialises_cleanly():
    """Backed AnnData (read_h5ad(..., backed='r')) must work without
    a cryptic IndexError - real users use this for large files."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import tempfile, os, warnings
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.random((30, 10)).astype(np.float32))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
        var=pd.DataFrame(index=list("abcdefghij")),
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "small.h5ad")
        adata.write_h5ad(path)
        backed = ad.read_h5ad(path, backed="r")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                auc = aucell.score(backed, [("R", ["a", "b"])], top_frac=0.3)
        finally:
            backed.file.close()
        assert auc.shape == (30, 1)
        assert any("backed" in str(w.message).lower() for w in caught)


def test_aucell_too_small_top_frac_warns():
    """top_frac so small that floor(top_frac × n_genes) < 1 = silent zero
    before the audit. Now warns explicitly so the user can debug."""
    import warnings
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((20, 10)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"G{i}" for i in range(10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aucell.score(df, [("R", ["G0", "G1"])], top_frac=0.05)
    msgs = [str(w.message) for w in caught]
    # 0.05 × 10 = 0.5 → rank cutoff 0
    assert any("rank cutoff" in m and "below 1" in m for m in msgs), msgs


def test_aucell_single_rank_cutoff_warns_and_scores_zero():
    import warnings

    df = pd.DataFrame([[10.0] + [0.0] * 9], columns=[f"G{i}" for i in range(10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = aucell.score(df, [("R", ["G0"])], top_frac=0.1)
    assert out.iloc[0, 0] == 0.0
    msgs = [str(w.message) for w in caught]
    assert any("rank cutoff" in m and "below 1" in m for m in msgs), msgs
