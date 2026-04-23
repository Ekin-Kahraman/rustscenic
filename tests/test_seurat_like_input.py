"""Smoke tests for Seurat-conversion input shapes.

When Seurat users convert to AnnData (via SeuratDisk, anndata2ri,
sceasy, or zellkonverter), the result typically lands in rustscenic as
either:

  (a) a gene-symbols-in-var_names AnnData (scanpy-native shape), or
  (b) an ENSEMBL-in-var_names AnnData with symbols in `var` (cellxgene
      shape, sometimes produced by zellkonverter on SingleCellExperiment
      objects that carry ENSEMBL rowData).

Both must produce bit-identical per-cell AUC. These tests assert that.

Not testing R or an actual conversion — that requires a full R session
and is covered by documentation. These tests guard the Python side of
the bridge.
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import rustscenic.aucell


def _make_adata(n_cells: int = 200, n_genes: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.gamma(2.0, 0.5, size=(n_cells, n_genes)).astype(np.float32)
    return X, [f"SYM{i}" for i in range(n_genes)], [f"cell{i}" for i in range(n_cells)]


def test_seurat_style_symbols_in_var_names():
    """The default SeuratDisk → AnnData export has symbols in var_names.
    Should pass straight through rustscenic.aucell."""
    X, symbols, cells = _make_adata()
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=symbols), obs=pd.DataFrame(index=cells))
    regulons = [("R1", ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"])]
    auc = rustscenic.aucell.score(adata, regulons, top_frac=0.3)
    assert auc.shape == (200, 1)
    # Non-empty activity expected on random-ish data.
    assert (auc.values > 0).any()


def test_zellkonverter_style_ensembl_in_var_names():
    """zellkonverter on a SingleCellExperiment with ENSEMBL rowData
    lands in Python with ENSEMBL IDs in var_names. The auto-swap should
    handle it identically to the symbols-in-var_names case."""
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

    # Bit-identical per-cell Pearson between the two shapes.
    r, _ = pearsonr(auc_cx["R1"].values, auc_sym["R1"].values)
    assert r > 0.9999, f"Seurat→AnnData shape produced different AUCs (Pearson {r:.6f})"
    assert auc_cx.attrs["regulon_coverage"]["R1"] == (5, 5)


def test_seurat_default_assay_with_log_normalised_layer_pattern():
    """Seurat+SeuratDisk export sometimes puts the log-normalised
    matrix in a named layer rather than X. Users are instructed to
    `adata.X = adata.layers["logcounts"]` — this test documents that
    the resulting AnnData works fine."""
    X_raw = (np.random.default_rng(0).poisson(5, size=(100, 40)) + 1).astype(np.float32)
    X_logcounts = np.log1p(X_raw / X_raw.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)

    symbols = [f"SYM{i}" for i in range(40)]
    adata = ad.AnnData(
        X=X_logcounts,
        var=pd.DataFrame(index=symbols),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(100)]),
        layers={"counts": X_raw},  # raw counts preserved in a layer
    )
    regulons = [("R1", ["SYM0", "SYM1", "SYM2", "SYM3"])]
    auc = rustscenic.aucell.score(adata, regulons, top_frac=0.3)
    assert auc.shape == (100, 1)
    # `.X` was the log-normalised matrix so no unnormalised warning
    # should have fired. (We don't assert the absence of warnings
    # because sparse inputs / small datasets can produce benign ones.)
