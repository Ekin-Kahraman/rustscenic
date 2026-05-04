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

    def test_nan_input_panics(self, rng):
        X = rng.random((20, 10)).astype(np.float32)
        X[0, 0] = np.nan
        df = pd.DataFrame(X, columns=[f"g{i}" for i in range(10)])
        with pytest.raises(BaseException, match=r"[Nn]a[Nn]"):
            grn.infer(df, tf_names=["g0", "g1"], n_estimators=20, verbose=False)


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
            assert set(zip(sub["target"], sub["importance"])) == set(
                zip(full_tf["target"], full_tf["importance"])
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


class TestGrnLoadTfs:
    def test_strips_crlf_and_comments(self, tmp_path):
        path = tmp_path / "tfs.txt"
        path.write_text("SPI1\r\n# comment line\n\nPAX5\nTCF7\n")
        tfs = grn.load_tfs(path)
        assert tfs == ["SPI1", "PAX5", "TCF7"]
