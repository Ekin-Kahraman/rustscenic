"""Tests for the gene-name resolution + input-validation helpers.

These guard three silent-failure modes:
  1. cellxgene-style AnnData (ENSEMBL var_names) silently underscoring
  2. partial regulon overlap producing floor-bound AUCell scores
  3. raw UMI counts passed in instead of log-normalised expression
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rustscenic._gene_resolution import (
    regulon_coverage,
    resolve_gene_names,
    warn_if_likely_unnormalized,
    warn_if_poor_coverage,
)


# ---- cellxgene convention auto-detect -------------------------------------


class _FakeVar:
    """Minimal duck-typed stand-in for AnnData.var (a DataFrame wrapper)."""

    def __init__(self, columns_dict):
        self._cols = columns_dict

    @property
    def columns(self):
        return list(self._cols.keys())

    def __getitem__(self, key):
        return self._cols[key]


class _FakeAnnData:
    def __init__(self, var_names, var_columns=None):
        self.var_names = list(var_names)
        self.var = _FakeVar(var_columns or {})


def test_non_ensembl_var_names_pass_through_unchanged():
    """A scanpy-native h5ad (symbols in var_names) must not be touched."""
    ad = _FakeAnnData(["SOX6", "FOXJ1", "TP63", "PAX5"])
    assert resolve_gene_names(ad, quiet=True) == ["SOX6", "FOXJ1", "TP63", "PAX5"]


def test_cellxgene_ensembl_swaps_to_feature_name():
    """ENSG-ids in var_names + feature_name column = auto-swap."""
    ensembl = [f"ENSG0000011{i:04d}" for i in range(20)]
    symbols = [f"SYMBOL{i}" for i in range(20)]
    ad = _FakeAnnData(ensembl, var_columns={"feature_name": symbols})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_gene_names(ad)
    assert resolved == symbols
    assert any("ENSEMBL" in str(w.message) for w in caught), \
        "expected a warning announcing the swap"


def test_mouse_ensembl_also_swaps():
    """ENSMUSG IDs should trigger the swap too (not just human ENSG)."""
    ensembl = [f"ENSMUSG0000{i:06d}" for i in range(20)]
    symbols = [f"Gene{i}" for i in range(20)]
    ad = _FakeAnnData(ensembl, var_columns={"feature_name": symbols})
    assert resolve_gene_names(ad, quiet=True) == symbols


def test_alternate_symbol_columns_checked_in_order():
    """If feature_name is absent, fall through to gene_symbols / gene_name."""
    ensembl = [f"ENSG0000011{i:04d}" for i in range(20)]
    symbols = [f"S{i}" for i in range(20)]
    ad = _FakeAnnData(ensembl, var_columns={"gene_symbols": symbols})
    assert resolve_gene_names(ad, quiet=True) == symbols


def test_ensembl_with_no_symbol_column_falls_back():
    """No known symbol column = return var_names as-is (no raise)."""
    ensembl = [f"ENSG0000011{i:04d}" for i in range(20)]
    ad = _FakeAnnData(ensembl)
    assert resolve_gene_names(ad, quiet=True) == ensembl


def test_ensembl_swap_rejects_if_symbol_column_also_ensembl():
    """Guard against a duplicated ID column being picked by accident."""
    ensembl = [f"ENSG0000011{i:04d}" for i in range(20)]
    ad = _FakeAnnData(
        ensembl,
        var_columns={"feature_name": ensembl, "gene_symbols": [f"S{i}" for i in range(20)]},
    )
    # Should skip the feature_name duplicate and fall through to gene_symbols.
    assert resolve_gene_names(ad, quiet=True) == [f"S{i}" for i in range(20)]


def test_versioned_ensembl_ids_also_match():
    """ENSEMBL IDs with .N version suffix (ENSG00000110693.5) must still trigger."""
    ensembl = [f"ENSG0000011{i:04d}.3" for i in range(20)]
    symbols = [f"SYM{i}" for i in range(20)]
    ad = _FakeAnnData(ensembl, var_columns={"feature_name": symbols})
    assert resolve_gene_names(ad, quiet=True) == symbols


# ---- regulon coverage ------------------------------------------------------


def test_regulon_coverage_counts_matches_and_totals():
    genes = ["A", "B", "C", "D", "E"]
    regulons = [
        ("R1", ["A", "B", "X"]),   # 2/3
        ("R2", ["A", "B", "C"]),   # 3/3
        ("R3", ["X", "Y", "Z"]),   # 0/3
    ]
    cov = regulon_coverage(genes, regulons)
    assert cov == {"R1": (2, 3), "R2": (3, 3), "R3": (0, 3)}


def test_warn_if_poor_coverage_fires_on_low_match():
    cov = {"R1": (2, 10), "R2": (9, 10)}  # R1 at 20%
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_poor_coverage(cov, threshold=0.5)
    assert any("R1" in str(w.message) for w in caught)


def test_warn_if_poor_coverage_silent_when_all_good():
    cov = {"R1": (8, 10), "R2": (9, 10)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_poor_coverage(cov, threshold=0.5)
    assert not caught


# ---- unnormalised-input guard ---------------------------------------------


def test_warn_if_likely_unnormalized_fires_on_raw_counts():
    X = np.array([[0, 1000, 0], [2000, 0, 500]], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_likely_unnormalized(X)
    assert any("log-normalised" in str(w.message) for w in caught)


def test_warn_if_likely_unnormalized_silent_on_log_normalised():
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 0.5, size=(100, 50)).astype(np.float32)  # max ~5-10
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_likely_unnormalized(X)
    assert not caught


def test_warn_if_likely_unnormalized_handles_sparse():
    """Shouldn't crash on scipy.sparse input."""
    import scipy.sparse as sp
    X = sp.csr_matrix(np.array([[0, 500], [1000, 0]], dtype=np.float32))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_likely_unnormalized(X)
    assert any("log-normalised" in str(w.message) for w in caught)


def test_warn_if_likely_unnormalized_handles_empty_sparse():
    """Empty sparse matrix must not crash."""
    import scipy.sparse as sp
    X = sp.csr_matrix((0, 0))
    warn_if_likely_unnormalized(X)  # just not crashing is enough


# ---- integration: aucell on a cellxgene-like AnnData ----------------------


def test_aucell_matches_regulons_after_ensembl_swap():
    """End-to-end: regulons with gene symbols must score non-zero on a
    cellxgene-style h5ad after the auto-swap kicks in."""
    import anndata as ad
    import rustscenic.aucell

    n_cells, n_genes = 60, 20
    rng = np.random.default_rng(42)
    X = rng.gamma(2.0, 0.5, size=(n_cells, n_genes)).astype(np.float32)

    ensembl = [f"ENSG0000011{i:04d}" for i in range(n_genes)]
    symbols = [f"SYM{i}" for i in range(n_genes)]
    var = pd.DataFrame({"feature_name": symbols}, index=ensembl)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Regulon references the SYMBOL space — the swap must happen for it to score.
    regulons = [("R_sym", ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = rustscenic.aucell.score(adata, regulons, top_frac=0.3)

    assert auc.shape == (n_cells, 1)
    # Without the swap, every regulon-to-column lookup misses and the
    # regulon is dropped → shape would be (n_cells, 0). This asserts the
    # swap fired and matched the symbols.
    assert auc.columns.tolist() == ["R_sym"]
    # Per-regulon coverage metadata must round-trip on .attrs.
    assert "regulon_coverage" in auc.attrs
    assert auc.attrs["regulon_coverage"]["R_sym"] == (5, 5)
