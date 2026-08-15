"""Tests for rustscenic.grn.infer."""
import numpy as np
import pandas as pd
import pytest

import rustscenic.grn as grn


class TestGrnShape:
    def test_returns_dataframe_with_expected_columns(self, small_expr):
        out = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                        n_estimators=30, verbose=False)
        assert set(out.columns) == {"TF", "target", "importance"}
        assert len(out) > 0

    def test_importance_is_nonnegative(self, small_expr):
        out = grn.infer(small_expr, tf_names=["g0", "g1"],
                        n_estimators=30, verbose=False)
        assert (out["importance"] >= 0).all()

    def test_tfs_only_from_input_list(self, small_expr):
        out = grn.infer(small_expr, tf_names=["g0", "g5", "g10"],
                        n_estimators=30, verbose=False)
        assert set(out["TF"].unique()).issubset({"g0", "g5", "g10"})


class TestGrnEdgeCases:
    def test_foreign_tf_names_are_dropped(self, small_expr):
        out = grn.infer(small_expr, tf_names=["g0", "FOREIGN", "g1"],
                        n_estimators=30, verbose=False)
        assert "FOREIGN" not in out["TF"].unique()
        assert {"g0", "g1"}.issubset(set(out["TF"].unique()))

    def test_all_foreign_tfs_returns_empty(self, small_expr):
        out = grn.infer(small_expr, tf_names=["NOT_IN_DATA1", "NOT_IN_DATA2"],
                        n_estimators=30, verbose=False)
        assert len(out) == 0

    def test_empty_tf_list_warns_and_returns_empty(self, small_expr):
        with pytest.warns(UserWarning, match="empty TF list"):
            out = grn.infer(small_expr, tf_names=[], n_estimators=30, verbose=False)
        assert len(out) == 0

    def test_nan_input_raises_value_error(self, rng):
        X = rng.random((20, 10)).astype(np.float32)
        X[0, 0] = np.nan
        df = pd.DataFrame(X, columns=[f"g{i}" for i in range(10)])
        with pytest.raises(ValueError, match=r"NaN|Inf"):
            grn.infer(df, tf_names=["g0", "g1"], n_estimators=20, verbose=False)

    def test_sparse_anndata_does_not_call_toarray(self, monkeypatch):
        import anndata as ad
        import scipy.sparse as sp

        X = sp.csc_matrix(np.ones((60, 4), dtype=np.float32))
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        )

        def fail_toarray(*_args, **_kwargs):
            raise AssertionError("sparse GRN path must not densify with toarray()")

        monkeypatch.setattr(sp.csr_matrix, "toarray", fail_toarray)
        grn.infer(adata, tf_names=["g0", "g1"], n_estimators=5, verbose=False)

    def test_sparse_grn_passes_sparse_buffers_to_rust_without_pointer_copy(self, monkeypatch):
        import anndata as ad
        import scipy.sparse as sp

        X = sp.csc_matrix(np.ones((60, 4), dtype=np.float32))
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        )
        seen = {}

        def fake_sparse(indptr, indices, data, *_args):
            seen["indptr"] = indptr
            seen["indices"] = indices
            seen["data"] = data
            seen["indptr_dtype"] = indptr.dtype
            seen["indices_dtype"] = indices.dtype
            seen["data_dtype"] = data.dtype
            return [], [], np.asarray([], dtype=np.float32), 0, 2, [], 1.0, {}

        monkeypatch.setattr(grn, "_grn_infer_sparse_csc", fake_sparse)

        out = grn.infer(adata, tf_names=["g0", "g1"], n_estimators=5, verbose=False)

        assert out.empty
        assert out.attrs["rust_backend"]["symbols"] == [
            "gene_duplicate_summary",
            "grn_infer_sparse_csc",
        ]
        assert seen["indptr_dtype"] == X.indptr.dtype
        assert seen["indices_dtype"] == np.dtype(np.int32)
        assert seen["data_dtype"] == np.dtype(np.float32)
        assert np.shares_memory(seen["indptr"], X.indptr)
        assert np.shares_memory(seen["indices"], X.indices)

    def test_sparse_grn_does_not_scan_negative_values_in_python(self, monkeypatch):
        import anndata as ad
        import scipy.sparse as sp

        X = sp.csc_matrix(np.ones((60, 4), dtype=np.float32))
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        )

        def fail_min(_values):
            raise AssertionError("sparse finite/negative validation should happen in Rust")

        def fake_sparse(_indptr, _indices, _data, *_args):
            return [], [], np.asarray([], dtype=np.float32), 0, 2, [], 1.0, {}

        monkeypatch.setattr(grn.np, "min", fail_min)
        monkeypatch.setattr(grn, "_grn_infer_sparse_csc", fake_sparse)

        out = grn.infer(adata, tf_names=["g0", "g1"], n_estimators=5, verbose=False)

        assert out.empty

    def test_sparse_integer_anndata_matches_dense_float32(self, rng):
        import anndata as ad
        import scipy.sparse as sp

        X = rng.poisson(2.0, size=(60, 12)).astype(np.int64)
        genes = [f"g{i}" for i in range(X.shape[1])]
        cells = [f"c{i}" for i in range(X.shape[0])]
        dense = pd.DataFrame(X.astype(np.float32), index=cells, columns=genes)
        adata = ad.AnnData(
            X=sp.csr_matrix(X),
            obs=pd.DataFrame(index=cells),
            var=pd.DataFrame(index=genes),
        )

        X_coerced, _ = grn._coerce_expression(adata)
        assert sp.issparse(X_coerced)

        dense_out = grn.infer(
            dense,
            tf_names=["g0", "g1", "g2"],
            n_estimators=30,
            seed=17,
            verbose=False,
        )
        sparse_out = grn.infer(
            adata,
            tf_names=["g0", "g1", "g2"],
            n_estimators=30,
            seed=17,
            verbose=False,
        )
        dense_out = dense_out.sort_values(["TF", "target"]).reset_index(drop=True)
        sparse_out = sparse_out.sort_values(["TF", "target"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(dense_out, sparse_out)

    def test_dense_grn_passes_strided_float32_without_python_contiguity_copy(self, monkeypatch, rng):
        values = np.asfortranarray(rng.random((60, 8)).astype(np.float32))
        assert values.flags.f_contiguous
        assert not values.flags.c_contiguous
        df = pd.DataFrame(
            values,
            index=[f"c{i}" for i in range(values.shape[0])],
            columns=[f"g{i}" for i in range(values.shape[1])],
            copy=False,
        )
        seen = {}

        def fake_grn(expression_arg, *_args):
            seen["expression"] = expression_arg
            return [], [], np.asarray([], dtype=np.float32), 0, 2, [], 1.0, {}

        monkeypatch.setattr(grn, "_grn_infer", fake_grn)

        out = grn.infer(df, tf_names=["g0", "g1"], n_estimators=5, verbose=False)

        assert out.empty
        assert np.shares_memory(seen["expression"], values)
        assert seen["expression"].flags.f_contiguous
        assert not seen["expression"].flags.c_contiguous

    def test_duplicate_gene_dedupe_kernel_is_recorded(self, monkeypatch):
        values = np.ones((60, 3), dtype=np.float32)
        df = pd.DataFrame(values, columns=["TF1", "g1", "TF1"])
        seen = {}

        def fake_grn(expression_arg, *_args):
            seen["shape"] = expression_arg.shape
            return [], [], np.asarray([], dtype=np.float32), 0, 1, [], 1.0, {}

        monkeypatch.setattr(grn, "_grn_infer", fake_grn)

        with pytest.warns(UserWarning, match="duplicate gene"):
            out = grn.infer(df, tf_names=["TF1"], n_estimators=5, verbose=False)

        assert seen["shape"] == (60, 2)
        assert out.attrs["rust_backend"]["symbols"] == [
            "gene_duplicate_summary",
            "gene_dedupe_dense_f32",
            "grn_infer",
        ]

    def test_dense_grn_does_not_allocate_python_isfinite_mask(self, monkeypatch, rng):
        values = rng.random((60, 8)).astype(np.float32)
        df = pd.DataFrame(
            values,
            index=[f"c{i}" for i in range(values.shape[0])],
            columns=[f"g{i}" for i in range(values.shape[1])],
        )

        def fail_isfinite(_values):
            raise AssertionError("dense finite validation should happen in Rust")

        def fake_grn(_expression_arg, *_args):
            return [], [], np.asarray([], dtype=np.float32), 0, 2, [], 1.0, {}

        monkeypatch.setattr(grn.np, "isfinite", fail_isfinite)
        monkeypatch.setattr(grn, "_grn_infer", fake_grn)

        out = grn.infer(df, tf_names=["g0", "g1"], n_estimators=5, verbose=False)

        assert out.empty

    def test_rejects_unknown_early_stop_mode(self, small_expr):
        with pytest.raises(ValueError, match="early_stop_mode"):
            grn.infer(
                small_expr,
                tf_names=["g0", "g1"],
                n_estimators=5,
                early_stop_mode="mystery",
                verbose=False,
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_estimators": 0}, "n_estimators"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"max_features": 0.0}, "max_features"),
            ({"max_features": 1.1}, "max_features"),
            ({"subsample": 0.0}, "subsample"),
            ({"subsample": 1.1}, "subsample"),
            ({"max_depth": 0}, "max_depth"),
            ({"early_stop_window": -1}, "early_stop_window"),
        ],
    )
    def test_rejects_invalid_fit_configuration(self, small_expr, kwargs, message):
        with pytest.raises(ValueError, match=message):
            grn.infer(
                small_expr,
                tf_names=["g0", "g1"],
                verbose=False,
                **kwargs,
            )

    def test_dense_grn_uses_rust_tf_overlap_metadata_for_warnings(self, monkeypatch, rng):
        values = rng.random((60, 8)).astype(np.float32)
        df = pd.DataFrame(
            values,
            index=[f"c{i}" for i in range(values.shape[0])],
            columns=[f"g{i}" for i in range(values.shape[1])],
        )

        def fake_grn(_expression_arg, *_args):
            return (
                [],
                [],
                np.asarray([], dtype=np.float32),
                0,
                1,
                ["missing_a", "missing_b"],
                1.0,
                {},
            )

        monkeypatch.setattr(grn, "_grn_infer", fake_grn)

        with pytest.warns(UserWarning, match=r"only 1 of 10 supplied TFs"):
            out = grn.infer(
                df,
                tf_names=[
                    "g0",
                    "missing_a",
                    "missing_b",
                    "missing_c",
                    "missing_d",
                    "missing_e",
                    "missing_f",
                    "missing_g",
                    "missing_h",
                    "missing_i",
                ],
                n_estimators=5,
                verbose=False,
            )

        assert out.empty

    def test_dense_strided_float32_matches_c_contiguous_result(self):
        n_cells = 64
        genes = [f"g{i}" for i in range(6)]
        values = np.zeros((n_cells, len(genes)), dtype=np.float32)
        for i in range(n_cells):
            x = i / n_cells
            values[i, 0] = x
            values[i, 1] = 1.0 - x
            values[i, 2] = 2.0 * x + 0.1
            values[i, 3] = values[i, 0] + values[i, 1]
            values[i, 4] = 0.5 * x
            values[i, 5] = 0.25

        common = {
            "tf_names": ["g0", "g1", "g2"],
            "n_estimators": 30,
            "max_features": 1.0,
            "subsample": 1.0,
            "target_block_size": 2,
            "seed": 13,
            "verbose": False,
        }
        contiguous = grn.infer(
            pd.DataFrame(values.copy(order="C"), columns=genes),
            **common,
        )
        strided = grn.infer(
            pd.DataFrame(np.asfortranarray(values), columns=genes),
            **common,
        )

        contiguous = contiguous.sort_values(["TF", "target"]).reset_index(drop=True)
        strided = strided.sort_values(["TF", "target"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(contiguous, strided)

    def test_sparse_nan_input_raises_value_error(self):
        import anndata as ad
        import scipy.sparse as sp

        X = sp.csr_matrix(np.ones((20, 6), dtype=np.float32))
        X.data[0] = np.nan
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(20)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
        )
        with pytest.raises(ValueError, match=r"NaN|Inf"):
            grn.infer(adata, tf_names=["g0", "g1"], n_estimators=5, verbose=False)

    def test_sparse_negative_input_raises_value_error(self):
        import anndata as ad
        import scipy.sparse as sp

        X = sp.csr_matrix(np.ones((20, 6), dtype=np.float32))
        X.data[0] = -1.0
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"c{i}" for i in range(20)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
        )
        with pytest.raises(ValueError, match="negative values"):
            grn.infer(adata, tf_names=["g0", "g1"], n_estimators=5, verbose=False)


class TestGrnDeterminism:
    def test_same_seed_produces_identical_output(self, small_expr):
        a = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                      n_estimators=100, seed=42, verbose=False)
        b = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                      n_estimators=100, seed=42, verbose=False)
        a_s = a.sort_values(["TF", "target"]).reset_index(drop=True)
        b_s = b.sort_values(["TF", "target"]).reset_index(drop=True)
        assert a_s.equals(b_s)

    def test_different_seed_differs(self, small_expr):
        a = grn.infer(small_expr, tf_names=["g0", "g1"],
                      n_estimators=100, seed=42, verbose=False)
        b = grn.infer(small_expr, tf_names=["g0", "g1"],
                      n_estimators=100, seed=123, verbose=False)
        a_s = a.sort_values(["TF", "target"]).reset_index(drop=True)
        b_s = b.sort_values(["TF", "target"]).reset_index(drop=True)
        assert not a_s.equals(b_s)


class TestGrnEarlyStopping:
    def test_fit_summary_records_public_mode_and_bounds(self, small_expr):
        out = grn.infer(
            small_expr,
            tf_names=["g0", "g1", "g2"],
            n_estimators=17,
            early_stop_window=3,
            early_stop_mode="arboreto",
            seed=12,
            verbose=False,
        )

        fit = out.attrs["grn_fit"]
        assert fit["early_stop_mode"] == "arboreto"
        assert fit["early_stop_window"] == 3
        assert fit["n_estimators_ceiling"] == 17
        assert fit["target_count"] == small_expr.shape[1]
        assert 1 <= fit["fitted_trees_min"] <= fit["fitted_trees_max"] <= 17

    def test_arboreto_mode_without_oob_reaches_estimator_ceiling(self, small_expr):
        out = grn.infer(
            small_expr,
            tf_names=["g0", "g1"],
            n_estimators=8,
            subsample=1.0,
            early_stop_window=2,
            early_stop_mode="arboreto",
            verbose=False,
        )

        assert out.attrs["grn_fit"]["fitted_trees_min"] == 8
        assert out.attrs["grn_fit"]["fitted_trees_max"] == 8

    def test_subsample_just_below_one_still_uses_oob_monitor(self, small_expr):
        out = grn.infer(
            small_expr,
            tf_names=["g0", "g1"],
            n_estimators=20,
            subsample=0.9999995,
            early_stop_window=1,
            early_stop_mode="arboreto",
            seed=42,
            verbose=False,
        )

        assert out.attrs["grn_fit"]["fitted_trees_min"] < 20

    def test_legacy_mode_is_explicit_and_deterministic(self, small_expr):
        kwargs = dict(
            tf_names=["g0", "g1"],
            n_estimators=30,
            early_stop_window=3,
            early_stop_mode="legacy_inbag",
            seed=42,
            verbose=False,
        )
        first = grn.infer(small_expr, **kwargs)
        second = grn.infer(small_expr, **kwargs)

        assert first.attrs["grn_fit"]["early_stop_mode"] == "legacy_inbag"
        pd.testing.assert_frame_equal(first, second)
        assert first.attrs["grn_fit"] == second.attrs["grn_fit"]


class TestGrnCorrelation:
    @staticmethod
    def expression_frame():
        return pd.DataFrame(
            {
                "TF": [0.0, 1.0, 2.0, 3.0, 4.0],
                "POS": [0.0, 2.0, 4.0, 6.0, 8.0],
                "NEG": [4.0, 3.0, 2.0, 1.0, 0.0],
                "CONST": [1.0] * 5,
            }
        )

    @staticmethod
    def adjacencies():
        return pd.DataFrame(
            {
                "TF": ["TF", "TF", "TF"],
                "target": ["POS", "NEG", "CONST"],
                "importance": [3.0, 2.0, 1.0],
            }
        )

    def test_add_correlation_classifies_both_polarities_and_neutral(self):
        out = grn.add_correlation(self.adjacencies(), self.expression_frame())

        assert list(out.columns) == ["TF", "target", "importance", "regulation", "rho"]
        np.testing.assert_allclose(out.loc[:1, "rho"], [1.0, -1.0], atol=1e-12)
        assert np.isnan(out.loc[2, "rho"])
        assert out["regulation"].tolist() == [1, -1, 0]
        assert out.attrs["correlation"] == {
            "method": "pearson",
            "rho_threshold": 0.03,
            "mask_dropouts": False,
            "activating_edges": 1,
            "repressing_edges": 1,
            "indeterminate_edges": 1,
        }

    @pytest.mark.parametrize("mask_dropouts", [False, True])
    def test_sparse_and_dense_correlation_match(self, mask_dropouts):
        import anndata as ad
        import scipy.sparse as sp

        dense = self.expression_frame()
        sparse = ad.AnnData(
            X=sp.csr_matrix(dense.to_numpy(dtype=np.float32)),
            var=pd.DataFrame(index=dense.columns),
        )
        expected = grn.add_correlation(
            self.adjacencies(), dense, mask_dropouts=mask_dropouts
        )
        actual = grn.add_correlation(
            self.adjacencies(), sparse, mask_dropouts=mask_dropouts
        )

        pd.testing.assert_frame_equal(expected, actual)
        assert "grn_correlations_sparse_csc" in actual.attrs["rust_backend"]["symbols"]

    def test_dropout_mask_ignores_explicit_sparse_zeros(self):
        import anndata as ad
        import scipy.sparse as sp

        dense = self.expression_frame()
        sparse_values = sp.csc_matrix(dense.to_numpy(dtype=np.float32))
        sparse_values[0, 0] = 0.0
        sparse_values[0, 1] = 0.0
        assert np.any(sparse_values.data == 0.0)
        sparse = ad.AnnData(
            X=sparse_values,
            var=pd.DataFrame(index=dense.columns),
        )

        expected = grn.add_correlation(
            self.adjacencies(), dense, mask_dropouts=True
        )
        actual = grn.add_correlation(
            self.adjacencies(), sparse, mask_dropouts=True
        )

        pd.testing.assert_frame_equal(expected, actual)

    def test_missing_expression_gene_is_actionable(self):
        adjacencies = self.adjacencies().copy()
        adjacencies.loc[0, "target"] = "MISSING"
        with pytest.raises(ValueError, match="missing from the expression matrix"):
            grn.add_correlation(adjacencies, self.expression_frame())

    def test_build_regulons_is_signed_deterministic_and_excludes_neutral(self):
        correlated = pd.DataFrame(
            {
                "TF": ["TF_B", "TF_A", "TF_A", "TF_B", "TF_A"],
                "target": ["b_pos", "a_neg", "a_pos", "b_neg", "neutral"],
                "importance": [3.0, 4.0, 5.0, 2.0, 100.0],
                "regulation": [1, -1, 1, -1, 0],
            }
        )

        out = grn.build_regulons(
            correlated, top_targets_per_tf=5, min_targets=1
        )

        assert out == {
            "TF_A_activator": ["a_pos"],
            "TF_B_activator": ["b_pos"],
            "TF_A_repressor": ["a_neg"],
            "TF_B_repressor": ["b_neg"],
        }
        assert "neutral" not in {gene for genes in out.values() for gene in genes}
        assert grn.build_regulons(
            correlated,
            top_targets_per_tf=5,
            min_targets=1,
            include_repressors=False,
        ) == {
            "TF_A_activator": ["a_pos"],
            "TF_B_activator": ["b_pos"],
        }

    @pytest.mark.parametrize("invalid", [2, -2, np.nan, pd.NA, True, "1"])
    def test_build_regulons_rejects_invalid_regulation(self, invalid):
        correlated = pd.DataFrame(
            {
                "TF": ["TF_A"],
                "target": ["target"],
                "importance": [1.0],
                "regulation": [invalid],
            }
        )

        with pytest.raises(ValueError, match="only -1, 0, or 1"):
            grn.build_regulons(
                correlated,
                top_targets_per_tf=1,
                min_targets=1,
            )


class TestGrnTruncationKnobs:
    """`top_targets_per_tf` and `min_importance` give users sparser output
    on under-determined inputs without rerunning the GBM."""

    def test_top_targets_per_tf_caps_per_tf_count(self, small_expr):
        full = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                         n_estimators=100, seed=42, verbose=False)
        trunc = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                          n_estimators=100, seed=42, verbose=False,
                          top_targets_per_tf=5)
        # Same fit (deterministic); the trunc result is a strict subset of full.
        assert (trunc.groupby("TF").size() <= 5).all()
        # Each TF with edges in the full output keeps its top-K by importance.
        for tf, sub in trunc.groupby("TF"):
            full_tf = full[full["TF"] == tf].sort_values(
                "importance", ascending=False
            ).head(5)
            assert set(zip(sub["target"], sub["importance"], strict=True)) == set(
                zip(full_tf["target"], full_tf["importance"], strict=True)
            )

    def test_min_importance_floor(self, small_expr):
        full = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                         n_estimators=100, seed=42, verbose=False)
        threshold = full["importance"].median()
        out = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                        n_estimators=100, seed=42, verbose=False,
                        min_importance=threshold)
        assert (out["importance"] >= threshold).all()
        assert len(out) <= len(full)

    def test_truncation_matches_pandas_reference(self, small_expr):
        full = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                         n_estimators=100, seed=42, verbose=False)
        threshold = full["importance"].quantile(0.25)
        out = grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                        n_estimators=100, seed=42, verbose=False,
                        min_importance=threshold, top_targets_per_tf=3)
        expected = (
            full[full["importance"] >= threshold]
            .sort_values("importance", ascending=False, kind="mergesort")
            .groupby("TF", sort=False, group_keys=False)
            .head(3)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)

    def test_under_determined_warning_fires(self):
        """n=10 samples × ~80 genes triggers the under-determined warning."""
        rng = np.random.default_rng(0)
        X = rng.poisson(1.0, size=(10, 80)).astype(np.float32)
        df = pd.DataFrame(X, columns=[f"g{i}" for i in range(80)])
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grn.infer(df, tf_names=["g0", "g1", "g2"],
                      n_estimators=30, verbose=False)
        msgs = [str(w.message) for w in caught]
        assert any("samples" in m and "unstable" in m for m in msgs), (
            f"expected under-determined warning, got {msgs}"
        )

    def test_no_warning_at_normal_sample_count(self, small_expr):
        """60 samples (small_expr) is above the n<50 threshold; no warning."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grn.infer(small_expr, tf_names=["g0", "g1", "g2"],
                      n_estimators=30, verbose=False)
        msgs = [str(w.message) for w in caught]
        assert not any("rankings are unstable" in m for m in msgs)


class TestGrnScalingKnobs:
    def test_target_block_size_preserves_results(self, small_expr):
        adaptive = grn.infer(
            small_expr,
            tf_names=["g0", "g1", "g2"],
            n_estimators=80,
            seed=42,
            verbose=False,
        )
        forced_single = grn.infer(
            small_expr,
            tf_names=["g0", "g1", "g2"],
            n_estimators=80,
            seed=42,
            target_block_size=1,
            verbose=False,
        )

        adaptive = adaptive.sort_values(["TF", "target"]).reset_index(drop=True)
        forced_single = forced_single.sort_values(["TF", "target"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(adaptive, forced_single)

    def test_target_block_size_rejects_non_positive_values(self, small_expr):
        with pytest.raises(ValueError, match="target_block_size"):
            grn.infer(
                small_expr,
                tf_names=["g0", "g1", "g2"],
                n_estimators=10,
                target_block_size=0,
                verbose=False,
            )


class TestGrnLoadTfs:
    def test_strips_crlf_and_comments(self, tmp_path):
        path = tmp_path / "tfs.txt"
        path.write_text("SPI1\r\n# comment line\n\nPAX5\nTCF7\n")
        tfs = grn.load_tfs(path)
        assert tfs == ["SPI1", "PAX5", "TCF7"]
