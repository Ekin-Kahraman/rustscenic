"""Python-side input and explicit sparse RNA exchange regression tests.

Direct AnnData construction does not validate any third-party R converter.
The separate test_seurat_interop.R exercises actual SeuratObject APIs.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

import rustscenic.aucell


def _make_adata(n_cells: int = 200, n_genes: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.gamma(2.0, 0.5, size=(n_cells, n_genes)).astype(np.float32)
    return X, [f"SYM{i}" for i in range(n_genes)], [f"cell{i}" for i in range(n_cells)]


def test_seurat_style_symbols_in_var_names():
    """Named gene-symbol expression can be scored directly."""
    X, symbols, cells = _make_adata()
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=symbols), obs=pd.DataFrame(index=cells))
    regulons = [("R1", ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"])]
    auc = rustscenic.aucell.score(adata, regulons, top_frac=0.3)
    assert auc.shape == (200, 1)
    # Non-empty activity expected on random-ish data.
    assert (auc.values > 0).any()


def test_ensembl_ids_with_symbol_metadata():
    """Explicit gene-symbol metadata gives the same AUC as symbol indices."""
    X, symbols, cells = _make_adata()
    ensembl = [f"ENSG0000011{i:04d}" for i in range(len(symbols))]
    var = pd.DataFrame({"feature_name": symbols}, index=ensembl)
    adata_cx = ad.AnnData(X=X, var=var, obs=pd.DataFrame(index=cells))

    # Symbols-in-var_names baseline
    adata_sym = ad.AnnData(X=X.copy(), var=pd.DataFrame(index=symbols), obs=pd.DataFrame(index=cells))

    regulons = [("R1", ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_cx = rustscenic.aucell.score(adata_cx, regulons, top_frac=0.3)
        auc_sym = rustscenic.aucell.score(adata_sym, regulons, top_frac=0.3)

    np.testing.assert_array_equal(auc_cx.to_numpy(), auc_sym.to_numpy())
    assert auc_cx.attrs["regulon_coverage"]["R1"] == (5, 5)


def test_explicit_log_normalised_layer_selection():
    """A caller can deliberately select a known layer, without converter guesses."""
    X_raw = (np.random.default_rng(0).poisson(5, size=(100, 40)) + 1).astype(np.float32)
    X_logcounts = np.log1p(X_raw / X_raw.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)

    symbols = [f"SYM{i}" for i in range(40)]
    adata = ad.AnnData(
        X=X_raw,
        var=pd.DataFrame(index=symbols),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(100)]),
        layers={"logcounts": X_logcounts},
    )
    adata.X = adata.layers["logcounts"].copy()
    regulons = [("R1", ["SYM0", "SYM1", "SYM2", "SYM3"])]
    auc = rustscenic.aucell.score(adata, regulons, top_frac=0.3)
    assert auc.shape == (100, 1)
    expected = rustscenic.aucell.score((X_logcounts, symbols), regulons, top_frac=0.3)
    np.testing.assert_array_equal(auc.to_numpy(), expected.to_numpy())


@pytest.fixture(scope="module")
def score_example():
    path = Path(__file__).parents[1] / "examples" / "seurat_score.py"
    spec = importlib.util.spec_from_file_location("seurat_score_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exchange(tmp_path):
    X, genes, cells = _make_adata(n_cells=8, n_genes=64)
    cells[0] = "001"
    cells[1] = "CELL_A-1"
    X[X < 0.5] = 0
    mmwrite(tmp_path / "expression.mtx", sparse.csc_matrix(X.T.astype(np.float64)), precision=17)
    (tmp_path / "genes.txt").write_text("\n".join(genes) + "\n")
    (tmp_path / "cells.txt").write_text("\n".join(cells) + "\n")
    return tmp_path, X, genes, cells


def test_sparse_exchange_preserves_values_names_and_activity(score_example, exchange):
    directory, X, genes, cells = exchange
    expression = score_example.read_export(directory)
    assert sparse.issparse(expression.X)
    np.testing.assert_array_equal(expression.X.toarray(), X)
    assert list(expression.var_names) == genes
    assert list(expression.obs_names) == cells
    regulons = {"R1(+)": genes[:4], "R2": genes[4:8]}
    actual = rustscenic.aucell.score(expression, regulons)
    direct = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
    expected = rustscenic.aucell.score(direct, regulons)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_sparse_exchange_cli_writes_named_activity_and_h5ad(score_example, exchange):
    directory, X, genes, cells = exchange
    (directory / "regulons.json").write_text(json.dumps({"R1(+)": genes[:4]}))
    output = directory / "activity.csv"
    h5ad = directory / "rna.h5ad"
    assert score_example.main([
        str(directory), str(directory / "regulons.json"), str(output), "--h5ad", str(h5ad)
    ]) == 0
    activity = pd.read_csv(
        output, index_col=0, dtype={"Unnamed: 0": str}, float_precision="round_trip"
    )
    assert list(activity.index) == cells
    assert list(activity.columns) == ["R1(+)"]
    saved = ad.read_h5ad(h5ad)
    assert sparse.issparse(saved.X)
    np.testing.assert_array_equal(saved.X.toarray(), X)
    assert list(saved.obs_names) == cells
    assert list(saved.var_names) == genes
    expected_activity = rustscenic.aucell.score(saved, {"R1(+)": genes[:4]})
    np.testing.assert_array_equal(activity.to_numpy(), expected_activity.to_numpy())


def test_sparse_exchange_rejects_wrong_orientation(score_example, exchange):
    directory, X, _, _ = exchange
    mmwrite(directory / "expression.mtx", sparse.csr_matrix(X))
    with pytest.raises(ValueError, match="must be genes by cells"):
        score_example.read_export(directory)


@pytest.mark.parametrize("filename", ["genes.txt", "cells.txt"])
@pytest.mark.parametrize("contents", ["", "same\nsame\n", "first\n\n"])
def test_sparse_exchange_rejects_ambiguous_ids(score_example, exchange, filename, contents):
    directory, _, _, _ = exchange
    (directory / filename).write_text(contents)
    with pytest.raises(ValueError, match="nonempty, unique IDs"):
        score_example.read_export(directory)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_sparse_exchange_rejects_invalid_expression(score_example, exchange, bad_value):
    directory, X, _, _ = exchange
    X[0, 0] = bad_value
    mmwrite(directory / "expression.mtx", sparse.csr_matrix(X.T))
    with pytest.raises(ValueError, match="finite, nonnegative"):
        score_example.read_export(directory)
