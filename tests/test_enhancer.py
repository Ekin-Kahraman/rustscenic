"""Tests for enhancer-to-gene linking.

Validates that the core SCENIC+ step works correctly on:
  (a) a synthetic multiome with known peak→gene relationships
  (b) the cellxgene/ENSEMBL-in-var_names shape (via resolver)
  (c) explicit peak_coords vs auto-parsed from var_names
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from rustscenic.enhancer import link_peaks_to_genes


def _synthetic_multiome(seed: int = 0, n_cells: int = 500):
    """Build matched RNA + ATAC where:
      peak_0 on chr1:1000-1500 IS correlated with gene GENE_A (TSS at 1_250)
      peak_1 on chr1:60000-60500 IS correlated with gene GENE_C (TSS at 60_200)
      peak_2 on chr1:200000-200500 (far from any gene) - should not link
    Other gene-peak pairs are independent noise.
    """
    rng = np.random.default_rng(seed)
    n_cells = n_cells

    # Shared latent - drives both peak_0 and GENE_A
    latent_A = rng.normal(size=n_cells)
    latent_B = rng.normal(size=n_cells)

    rna_genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]
    rna = np.column_stack([
        0.8 * latent_A + 0.2 * rng.normal(size=n_cells),  # GENE_A tracks latent_A
        rng.normal(size=n_cells),                          # GENE_B noise
        0.8 * latent_B + 0.2 * rng.normal(size=n_cells),   # GENE_C tracks latent_B
        rng.normal(size=n_cells),                          # GENE_D noise
    ]).astype(np.float32)

    peak_names = ["chr1:1000-1500", "chr1:60000-60500", "chr1:200000-200500"]
    atac = np.column_stack([
        0.8 * latent_A + 0.2 * rng.normal(size=n_cells),   # peak_0 tracks latent_A (same as GENE_A)
        0.8 * latent_B + 0.2 * rng.normal(size=n_cells),   # peak_1 tracks latent_B (same as GENE_C)
        rng.normal(size=n_cells),                          # peak_2 noise
    ]).astype(np.float32)

    cell_names = [f"cell{i}" for i in range(n_cells)]
    rna_adata = ad.AnnData(
        X=rna,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=rna_genes),
    )
    atac_adata = ad.AnnData(
        X=atac,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=peak_names),
    )
    gene_coords = pd.DataFrame({
        "gene": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "tss": [1_250, 30_000, 60_200, 10_000],
    })
    return rna_adata, atac_adata, gene_coords


def _sort_links_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            [
                "peak_id", "gene", "peak_chrom", "peak_start",
                "peak_end", "gene_tss", "distance",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _pearson_reference(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    x_centered = x - x.mean()
    Y_centered = Y - Y.mean(axis=0)
    denom = np.sqrt(np.sum(x_centered * x_centered)) * np.sqrt(
        np.sum(Y_centered * Y_centered, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, x_centered @ Y_centered / denom, 0.0).astype(
            np.float32
        )


def test_links_correlated_peak_to_correlated_gene():
    rna, atac, genes = _synthetic_multiome(seed=1)
    links = link_peaks_to_genes(rna, atac, genes, min_abs_corr=0.5)
    assert links.attrs["rust_backend"]["symbols"] == [
        "enhancer_align_cell_indices",
        "enhancer_parse_peak_names",
        "enhancer_match_gene_coords_to_rna",
        "enhancer_normalise_chrom_codes",
        "enhancer_prepare_gene_order",
        "enhancer_link_pearson",
    ]
    # Expect peak_0 ↔ GENE_A and peak_1 ↔ GENE_C above threshold
    found = set(zip(links["peak_id"], links["gene"], strict=True))
    assert ("chr1:1000-1500", "GENE_A") in found
    assert ("chr1:60000-60500", "GENE_C") in found


def test_rejects_noise_peak_with_no_nearby_gene():
    """peak_2 on chr1:200000-200500 is far from every gene (nearest is
    GENE_A at TSS 1,250, distance ~199kb - within default 500kb but
    correlation is ~0). Should fall below threshold, not get linked."""
    rna, atac, genes = _synthetic_multiome(seed=42)
    links = link_peaks_to_genes(rna, atac, genes, min_abs_corr=0.5)
    assert ("chr1:200000-200500", "GENE_A") not in set(
        zip(links["peak_id"], links["gene"], strict=True)
    )


def test_distance_filter_excludes_genes_too_far_away():
    rna, atac, genes = _synthetic_multiome(seed=1)
    # Tight distance cap - GENE_C TSS at 60,200 is too far from peak_0 at ~1,250
    links = link_peaks_to_genes(rna, atac, genes, max_distance=10_000, min_abs_corr=0.0)
    assert ("chr1:1000-1500", "GENE_C") not in set(
        zip(links["peak_id"], links["gene"], strict=True)
    )


def test_rejects_non_pearson_method():
    rna, atac, genes = _synthetic_multiome(seed=7)
    with pytest.raises(ValueError, match="method='pearson'"):
        link_peaks_to_genes(rna, atac, genes, method="spearman", min_abs_corr=0.3)


def test_cell_mismatch_raises():
    rna, atac, genes = _synthetic_multiome()
    rna2 = rna.copy()
    rna2.obs_names = [f"different_cell{i}" for i in range(rna2.n_obs)]
    with pytest.raises(ValueError, match="share no cell barcodes"):
        link_peaks_to_genes(rna2, atac, genes)


def test_partial_cell_overlap_warns_and_continues():
    rna, atac, genes = _synthetic_multiome()
    # Shrink RNA to 80% of cells - ATAC still has full set
    rna2 = rna[:400].copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        links = link_peaks_to_genes(rna2, atac, genes, min_abs_corr=0.5)
    assert any("keeping" in str(w.message) for w in caught)
    assert not links.empty


def test_align_cells_preserves_rna_order_with_atac_integer_indices():
    from rustscenic.enhancer import _align_cells

    rna = ad.AnnData(
        X=np.arange(6, dtype=np.float32).reshape(3, 2),
        obs=pd.DataFrame(index=["c2", "c0", "c_missing"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    atac = ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(4, 3),
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["p0", "p1", "p2"]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aligned_rna, aligned_atac = _align_cells(rna, atac)

    assert list(aligned_rna.obs_names) == ["c2", "c0"]
    assert list(aligned_atac.obs_names) == ["c2", "c0"]
    np.testing.assert_array_equal(aligned_rna.X, rna.X[[0, 1]])
    np.testing.assert_array_equal(aligned_atac.X, atac.X[[2, 0]])
    assert any("keeping 2 cells" in str(w.message) for w in caught)


def test_rust_chrom_code_normalisation_matches_python_helper():
    from rustscenic._rustscenic import enhancer_normalise_chrom_codes
    from rustscenic.enhancer import _chrom_examples, _normalise_chrom

    gene_chroms = ["chr1", "chrM", "  chr2  ", "GL000220.1"]
    peak_chroms = ["1", "MT", "chr2", "chrGL000220.1"]

    gene_norm, peak_norm, gene_codes, peak_codes = enhancer_normalise_chrom_codes(
        gene_chroms,
        peak_chroms,
    )

    assert gene_norm == [_normalise_chrom(chrom) for chrom in gene_chroms]
    assert peak_norm == [_normalise_chrom(chrom) for chrom in peak_chroms]
    assert gene_codes.dtype == np.int32
    assert peak_codes.dtype == np.int32
    assert gene_codes[0] == peak_codes[0]
    assert gene_codes[1] == peak_codes[1]
    assert gene_codes[2] == peak_codes[2]
    assert gene_codes[3] == peak_codes[3]
    assert _chrom_examples(["1", "1", "2", "3", "4", "5", "6"]) == [
        "1", "2", "3", "4", "5"
    ]


def test_peak_coords_override_var_lookup():
    rna, atac, genes = _synthetic_multiome(seed=1)
    # Peak var_names don't match coord format; pass explicit coords instead.
    atac2 = atac.copy()
    atac2.var_names = ["peakA", "peakB", "peakC"]
    peak_coords = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [1000, 60000, 200000],
        "end": [1500, 60500, 200500],
    }, index=["peakA", "peakB", "peakC"])
    links = link_peaks_to_genes(
        rna, atac2, genes, peak_coords=peak_coords, min_abs_corr=0.5,
    )
    assert ("peakA", "GENE_A") in set(zip(links["peak_id"], links["gene"], strict=True))
    assert "enhancer_match_peak_coords_to_atac" in links.attrs["rust_backend"]["symbols"]


def test_correlation_sign_preserved():
    """Negative correlation (accessibility inversely tracking expression)
    should be reported with a negative sign, not rejected."""
    rng = np.random.default_rng(0)
    n_cells = 500
    latent = rng.normal(size=n_cells)
    peak = (0.8 * latent + 0.2 * rng.normal(size=n_cells)).astype(np.float32)
    gene = (-0.8 * latent + 0.2 * rng.normal(size=n_cells)).astype(np.float32)
    rna = ad.AnnData(
        X=gene.reshape(-1, 1),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=["REP_GENE"]),
    )
    atac = ad.AnnData(
        X=peak.reshape(-1, 1),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=["chr1:500-600"]),
    )
    gene_coords = pd.DataFrame({
        "gene": ["REP_GENE"], "chrom": ["chr1"], "tss": [550],
    })
    links = link_peaks_to_genes(rna, atac, gene_coords, min_abs_corr=0.5)
    assert len(links) == 1
    assert links["correlation"].iloc[0] < 0


def test_link_peaks_keeps_atac_sparse_at_scale():
    """At 5000 cells × 200 peaks the sparse-path correlations should
    match the dense-path ones to within float32 noise. Guards against
    regressions where the sparse path drifts from the dense reference."""
    import scipy.sparse as sp

    rng = np.random.default_rng(7)
    n_cells = 5000
    n_genes = 4
    n_peaks = 8
    latent = rng.normal(size=n_cells)

    rna = (0.7 * latent[:, None]
           + 0.3 * rng.normal(size=(n_cells, n_genes))).astype(np.float32)
    # Sparse ATAC: same latent at first peak, noise elsewhere
    atac_dense = rng.normal(size=(n_cells, n_peaks)).astype(np.float32)
    atac_dense[:, 0] = 0.7 * latent + 0.3 * rng.normal(size=n_cells)
    # 70% sparsity
    mask = rng.random((n_cells, n_peaks)) < 0.3
    atac_dense = (atac_dense * mask).astype(np.float32)
    atac_sparse = sp.csr_matrix(atac_dense)

    rna_adata = ad.AnnData(
        X=rna,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]),
    )
    atac_adata = ad.AnnData(
        X=atac_sparse,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"chr1:{i*100}-{i*100+50}" for i in range(n_peaks)]),
    )
    gene_coords = pd.DataFrame({
        "gene": [f"G{i}" for i in range(n_genes)],
        "chrom": ["chr1"] * n_genes,
        "tss": [i * 100 + 25 for i in range(n_genes)],
    })

    links_sparse = link_peaks_to_genes(
        rna_adata, atac_adata, gene_coords, min_abs_corr=0.0,
    )

    # Force-densify the same input and compare correlations
    atac_adata_dense = atac_adata.copy()
    atac_adata_dense.X = atac_dense
    links_dense = link_peaks_to_genes(
        rna_adata, atac_adata_dense, gene_coords, min_abs_corr=0.0,
    )

    # Same peak-gene rows in both, just possibly different ordering
    sparse_pairs = links_sparse.set_index(["peak_id", "gene"])["correlation"]
    dense_pairs = links_dense.set_index(["peak_id", "gene"])["correlation"]
    assert set(sparse_pairs.index) == set(dense_pairs.index)
    aligned = sparse_pairs.reindex(dense_pairs.index)
    assert np.allclose(aligned.values, dense_pairs.values, atol=1e-4)


def test_multichrom_pearson_matches_dense_reference():
    """Chromosome batching must keep the public Pearson output unchanged."""
    import scipy.sparse as sp
    from rustscenic.enhancer import _normalise_chrom

    rng = np.random.default_rng(19)
    n_cells = 240
    genes = [f"G{i}" for i in range(6)]
    peaks = [
        "chr1:100-180", "chr2:120-220", "chr1:900-980", "chr2:850-930",
    ]
    latent_1 = rng.normal(size=n_cells)
    latent_2 = rng.normal(size=n_cells)
    rna_dense = rng.normal(size=(n_cells, len(genes))).astype(np.float32)
    rna_dense[:, 0] = 0.8 * latent_1 + 0.2 * rng.normal(size=n_cells)
    rna_dense[:, 3] = 0.8 * latent_2 + 0.2 * rng.normal(size=n_cells)
    atac_dense = rng.normal(size=(n_cells, len(peaks))).astype(np.float32)
    atac_dense[:, 0] = 0.8 * latent_1 + 0.2 * rng.normal(size=n_cells)
    atac_dense[:, 1] = 0.8 * latent_2 + 0.2 * rng.normal(size=n_cells)

    cells = [f"c{i}" for i in range(n_cells)]
    rna = ad.AnnData(
        X=sp.csr_matrix(rna_dense),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(atac_dense),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=peaks),
    )
    gene_coords = pd.DataFrame({
        "gene": genes,
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2", "chr2"],
        "tss": [140, 520, 940, 170, 560, 890],
    }).sample(frac=1.0, random_state=7).reset_index(drop=True)
    peak_coords = pd.DataFrame({
        "chrom": ["chr1", "chr2", "chr1", "chr2"],
        "start": [100, 120, 900, 850],
        "end": [180, 220, 980, 930],
    }, index=peaks)

    observed = link_peaks_to_genes(
        rna,
        atac,
        gene_coords,
        peak_coords=peak_coords,
        max_distance=500,
        min_abs_corr=0.05,
    )

    rows = []
    for peak_i, (peak_id, peak) in enumerate(peak_coords.iterrows()):
        centre = (int(peak["start"]) + int(peak["end"])) // 2
        chrom = _normalise_chrom(peak["chrom"])
        gene_block = (
            gene_coords[
                gene_coords["chrom"].map(_normalise_chrom).eq(chrom)
            ]
            .sort_values("tss", kind="mergesort")
            .reset_index(drop=True)
        )
        in_window = (
            (gene_block["tss"] >= centre - 500)
            & (gene_block["tss"] <= centre + 500)
        )
        gene_block = gene_block.loc[in_window].reset_index(drop=True)
        if gene_block.empty:
            continue
        gene_cols = [genes.index(g) for g in gene_block["gene"]]
        corr = _pearson_reference(atac_dense[:, peak_i], rna_dense[:, gene_cols])
        keep = np.abs(corr) >= 0.05
        for gene, tss, r in zip(
            gene_block["gene"].to_numpy()[keep],
            gene_block["tss"].to_numpy(dtype=np.int64)[keep],
            corr[keep],
            strict=True,
        ):
            rows.append((
                peak_id,
                chrom,
                int(peak["start"]),
                int(peak["end"]),
                gene,
                int(tss),
                int(centre - tss),
                float(r),
            ))
    expected = pd.DataFrame(rows, columns=observed.columns)

    observed = _sort_links_for_compare(observed)
    expected = _sort_links_for_compare(expected)
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_dtype=False,
        atol=1e-5,
        rtol=1e-5,
    )


def test_multichrom_sparse_rna_does_not_densify(monkeypatch):
    """Sparse RNA should stay sparse and use the Rust sparse Pearson kernel."""
    import scipy.sparse as sp

    rng = np.random.default_rng(23)
    n_cells = 80
    n_genes = 8
    gene_names = [f"G{i}" for i in range(n_genes)]
    rna = ad.AnnData(
        X=sp.csr_matrix(rng.normal(size=(n_cells, n_genes)).astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(rng.normal(size=(n_cells, 4)).astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[
            "chr1:100-200", "chr2:100-200", "chr1:500-600", "chr2:500-600",
        ]),
    )
    gene_coords = pd.DataFrame({
        "gene": gene_names,
        "chrom": ["chr1"] * 4 + ["chr2"] * 4,
        "tss": [150, 300, 450, 600, 150, 300, 450, 600],
    })

    def fail_toarray(*_args, **_kwargs):
        raise AssertionError("sparse RNA enhancer path must not densify")

    monkeypatch.setattr(sp.csr_matrix, "toarray", fail_toarray)
    links = link_peaks_to_genes(
        rna,
        atac,
        gene_coords,
        max_distance=1_000,
        min_abs_corr=0.0,
    )

    assert not links.empty


def test_gene_coordinate_setup_uses_rust_helper(monkeypatch):
    """Gene-coordinate sort/overlap setup should stay out of Python."""
    import scipy.sparse as sp
    import rustscenic.enhancer as enh

    rna = ad.AnnData(
        X=np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["G0", "G1"]),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(np.array([[1.0], [2.0]], dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["chr1:50-70"]),
    )
    gene_coords = pd.DataFrame({
        "gene": ["G1", "G0"],
        "chrom": ["chr1", "chr1"],
        "tss": [100, 50],
    })
    seen = {}

    def fake_match(rna_genes, coord_genes):
        seen["rna_genes"] = list(rna_genes)
        seen["coord_genes"] = list(coord_genes)
        return (
            np.array([0, 1], dtype=np.uint64),
            np.array([1, 0], dtype=np.int64),
        )

    def fake_prepare(peak_chrom_codes, gene_chrom_codes, gene_tss, gene_source_cols):
        seen["peak_chrom_codes"] = peak_chrom_codes.copy()
        seen["gene_chrom_codes"] = gene_chrom_codes.copy()
        seen["gene_tss"] = gene_tss.copy()
        seen["gene_source_cols"] = gene_source_cols.copy()
        return np.array([1, 0], dtype=np.uint64), True, 2

    def fake_link(
        rna_dense,
        atac_indptr,
        atac_indices,
        atac_data,
        peak_chrom_codes,
        peak_starts,
        peak_ends,
        gene_chrom_codes,
        gene_tss,
        gene_rna_cols,
        max_distance,
        min_abs_corr,
        atac_peak_cols,
    ):
        assert peak_starts.tolist() == [50]
        assert peak_ends.tolist() == [70]
        assert gene_tss.tolist() == [50, 100]
        assert gene_rna_cols.tolist() == [0, 1]
        assert max_distance == 500_000
        assert min_abs_corr == 0.1
        return (
            np.array([0], dtype=np.uint32),
            np.array([1], dtype=np.uint32),
            np.array([-40], dtype=np.int64),
            np.array([0.9], dtype=np.float32),
        )

    monkeypatch.setattr(enh, "_match_gene_coords_to_rna", fake_match)
    monkeypatch.setattr(enh, "_prepare_gene_order", fake_prepare)
    monkeypatch.setattr(enh, "_enhancer_link_pearson", fake_link)

    links = link_peaks_to_genes(rna, atac, gene_coords)

    assert seen["rna_genes"] == ["G0", "G1"]
    assert seen["coord_genes"] == ["G1", "G0"]
    assert seen["gene_tss"].tolist() == [100, 50]
    assert seen["gene_source_cols"].tolist() == [1, 0]
    assert links[["peak_id", "gene", "gene_tss", "distance", "correlation"]].to_dict(
        "records"
    ) == [
        {
            "peak_id": "chr1:50-70",
            "gene": "G1",
            "gene_tss": 100,
            "distance": -40,
            "correlation": np.float32(0.9),
        }
    ]


def test_enhancer_output_does_not_upcast_rust_index_arrays(monkeypatch):
    """Large enhancer-link outputs should not copy u32 Rust indices to int64."""
    import scipy.sparse as sp
    import rustscenic.enhancer as enh

    rna = ad.AnnData(
        X=np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["G0", "G1"]),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(np.array([[1.0], [2.0]], dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["chr1:50-70"]),
    )
    gene_coords = pd.DataFrame({
        "gene": ["G0", "G1"],
        "chrom": ["chr1", "chr1"],
        "tss": [50, 100],
    })
    peak_idx = np.array([0], dtype=np.uint32)
    gene_idx = np.array([1], dtype=np.uint32)
    copied_index_ids = {id(peak_idx), id(gene_idx)}

    def fake_link(*_args, **_kwargs):
        return (
            peak_idx,
            gene_idx,
            np.array([-40], dtype=np.int64),
            np.array([0.9], dtype=np.float32),
        )

    original_asarray = enh.np.asarray

    def guarded_asarray(value, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if id(value) in copied_index_ids and dtype is not None:
            if np.dtype(dtype) == np.dtype(np.int64):
                raise AssertionError("Rust u32 index outputs should not be copied to int64")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(enh, "_enhancer_link_pearson", fake_link)
    monkeypatch.setattr(enh.np, "asarray", guarded_asarray)

    links = link_peaks_to_genes(rna, atac, gene_coords)

    assert links[["peak_id", "gene"]].to_dict("records") == [
        {"peak_id": "chr1:50-70", "gene": "G1"}
    ]


def test_gene_coordinate_rna_matcher_preserves_rows_and_duplicate_rna_semantics():
    from rustscenic._rustscenic import enhancer_match_gene_coords_to_rna

    row_ix, source_cols = enhancer_match_gene_coords_to_rna(
        ["G0", "G1", "G1"],
        ["missing", "G1", "G0", "G1"],
    )

    np.testing.assert_array_equal(row_ix, np.array([1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(source_cols, np.array([2, 0, 2], dtype=np.int64))


def test_peak_coordinate_matcher_preserves_atac_order_and_reports_missing():
    from rustscenic._rustscenic import enhancer_match_peak_coords_to_atac

    row_ix, missing = enhancer_match_peak_coords_to_atac(
        ["p2", "missing", "p1", "p2"],
        ["p1", "p2", "p2"],
    )

    np.testing.assert_array_equal(row_ix, np.array([2, 0, 2], dtype=np.uint64))
    assert missing == ["missing"]


def test_multichrom_sparse_atac_reuses_original_csc_buffers(monkeypatch):
    """Enhancer linking should pass the original ATAC CSC buffers to Rust once."""
    import scipy.sparse as sp
    import rustscenic.enhancer as enh

    rng = np.random.default_rng(27)
    n_cells = 50
    cells = [f"c{i}" for i in range(n_cells)]
    genes = ["G1", "G2", "G3", "G4"]
    peaks = ["chr1:100-200", "chr2:100-200", "chr1:500-600", "chr2:500-600"]
    rna_matrix = sp.csc_matrix(rng.random((n_cells, len(genes))).astype(np.float32))
    atac_matrix = sp.csc_matrix(rng.random((n_cells, len(peaks))).astype(np.float32))
    rna = ad.AnnData(
        X=rna_matrix,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    atac = ad.AnnData(
        X=atac_matrix,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=peaks),
    )
    gene_coords = pd.DataFrame(
        {
            "gene": genes,
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "tss": [150, 520, 150, 520],
        }
    )
    calls = []

    def fake_sparse_rna(
        rna_indptr,
        rna_indices,
        rna_data,
        _n_cells,
        _n_rna_genes,
        atac_indptr,
        atac_indices,
        atac_data,
        *_args,
    ):
        atac_peak_cols = _args[-1]
        calls.append(
            {
                "rna_indptr_shared": np.shares_memory(rna_indptr, rna_matrix.indptr),
                "rna_indices_shared": np.shares_memory(rna_indices, rna_matrix.indices),
                "rna_data_shared": np.shares_memory(rna_data, rna_matrix.data),
                "atac_indptr_shared": np.shares_memory(atac_indptr, atac_matrix.indptr),
                "indices_shared": np.shares_memory(atac_indices, atac_matrix.indices),
                "data_shared": np.shares_memory(atac_data, atac_matrix.data),
                "rna_indptr_dtype": rna_indptr.dtype,
                "atac_indptr_dtype": atac_indptr.dtype,
                "peak_cols": None if atac_peak_cols is None else np.asarray(atac_peak_cols).copy(),
            }
        )
        return (
            np.asarray([], dtype=np.uint32),
            np.asarray([], dtype=np.uint32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
        )

    monkeypatch.setattr(enh, "_enhancer_link_pearson_sparse_rna", fake_sparse_rna)

    links = link_peaks_to_genes(
        rna,
        atac,
        gene_coords,
        max_distance=1_000,
        min_abs_corr=0.0,
    )

    assert links.empty
    assert len(calls) == 1
    assert calls[0]["rna_indptr_shared"]
    assert calls[0]["rna_indices_shared"]
    assert calls[0]["rna_data_shared"]
    assert calls[0]["atac_indptr_shared"]
    assert all(call["indices_shared"] for call in calls)
    assert all(call["data_shared"] for call in calls)
    assert calls[0]["rna_indptr_dtype"] == rna_matrix.indptr.dtype
    assert calls[0]["atac_indptr_dtype"] == atac_matrix.indptr.dtype
    assert calls[0]["peak_cols"] is None


def test_multichrom_dense_rna_reuses_original_matrix_for_linking(monkeypatch):
    """Dense RNA should be passed once, not copied into per-chrom blocks."""
    import rustscenic.enhancer as enh

    rng = np.random.default_rng(28)
    n_cells = 64
    cells = [f"c{i}" for i in range(n_cells)]
    genes = ["G1", "G2", "G3", "G4"]
    peaks = ["chr1:100-200", "chr2:100-200", "chr1:500-600", "chr2:500-600"]
    rna_matrix = rng.random((n_cells, len(genes))).astype(np.float32)
    atac_matrix = rng.random((n_cells, len(peaks))).astype(np.float32)
    rna = ad.AnnData(
        X=rna_matrix,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    atac = ad.AnnData(
        X=atac_matrix,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=peaks),
    )
    gene_coords = pd.DataFrame(
        {
            "gene": genes,
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "tss": [150, 520, 150, 520],
        }
    )
    calls = []

    def fake_dense_rna(
        rna_arg,
        _atac_indptr,
        _atac_indices,
        _atac_data,
        _peak_chrom_codes,
        _peak_starts,
        _peak_ends,
        _gene_chrom_codes,
        _gene_tss,
        gene_rna_cols,
        *_args,
    ):
        calls.append(
            {
                "shape": rna_arg.shape,
                "shares_rna": np.shares_memory(rna_arg, rna_matrix),
                "gene_cols": tuple(np.asarray(gene_rna_cols, dtype=np.uint32)),
            }
        )
        return (
            np.asarray([], dtype=np.uint32),
            np.asarray([], dtype=np.uint32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
        )

    monkeypatch.setattr(enh, "_enhancer_link_pearson", fake_dense_rna)

    links = link_peaks_to_genes(
        rna,
        atac,
        gene_coords,
        max_distance=1_000,
        min_abs_corr=0.0,
    )

    assert links.empty
    assert len(calls) == 1
    assert all(call["shape"] == rna_matrix.shape for call in calls)
    assert all(call["shares_rna"] for call in calls)
    assert calls[0]["gene_cols"] == (0, 1, 2, 3)


def test_dense_strided_rna_matches_c_contiguous_result():
    """The Rust Pearson kernel should accept dense RNA views without copying."""
    rng = np.random.default_rng(31)
    n_cells = 180
    cells = [f"c{i}" for i in range(n_cells)]
    genes = ["G0", "G1", "G2", "G3"]
    peaks = ["chr1:100-180", "chr1:300-380", "chr1:500-580"]
    base = rng.normal(size=(n_cells, len(genes) * 2)).astype(np.float32)
    rna_view = base[:, ::2]
    assert not rna_view.flags.c_contiguous
    atac_matrix = rng.normal(size=(n_cells, len(peaks))).astype(np.float32)
    atac_matrix[:, 0] = 0.7 * rna_view[:, 0] + 0.3 * rng.normal(size=n_cells)
    atac_matrix[:, 1] = -0.6 * rna_view[:, 2] + 0.4 * rng.normal(size=n_cells)

    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    rna_strided = ad.AnnData(X=rna_view, obs=obs.copy(), var=var.copy())
    rna_contig = ad.AnnData(
        X=np.ascontiguousarray(rna_view),
        obs=obs.copy(),
        var=var.copy(),
    )
    assert not rna_strided.X.flags.c_contiguous
    atac = ad.AnnData(
        X=atac_matrix,
        obs=obs.copy(),
        var=pd.DataFrame(index=peaks),
    )
    gene_coords = pd.DataFrame(
        {
            "gene": genes,
            "chrom": ["chr1"] * len(genes),
            "tss": [130, 330, 530, 730],
        }
    )

    observed = link_peaks_to_genes(
        rna_strided,
        atac,
        gene_coords,
        max_distance=1_000,
        min_abs_corr=0.0,
    )
    expected = link_peaks_to_genes(
        rna_contig,
        atac,
        gene_coords,
        max_distance=1_000,
        min_abs_corr=0.0,
    )

    pd.testing.assert_frame_equal(
        _sort_links_for_compare(observed),
        _sort_links_for_compare(expected),
        check_dtype=False,
        atol=1e-6,
        rtol=1e-6,
    )


def test_multichrom_enhancer_output_avoids_pandas_concat(monkeypatch):
    """Enhancer output assembly should not keep per-chrom DataFrames alive."""
    import scipy.sparse as sp
    import rustscenic.enhancer as enh

    rng = np.random.default_rng(29)
    n_cells = 120
    cells = [f"c{i}" for i in range(n_cells)]
    genes = ["G1", "G2", "G3", "G4"]
    peaks = ["chr1:100-200", "chr2:100-200"]
    rna = ad.AnnData(
        X=sp.csr_matrix(rng.normal(size=(n_cells, len(genes))).astype(np.float32)),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(rng.normal(size=(n_cells, len(peaks))).astype(np.float32)),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=peaks),
    )
    gene_coords = pd.DataFrame(
        {
            "gene": genes,
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "tss": [150, 180, 150, 180],
        }
    )

    def fail_concat(*_args, **_kwargs):
        raise AssertionError("enhancer output assembly should not use pd.concat")

    monkeypatch.setattr(enh.pd, "concat", fail_concat)

    links = link_peaks_to_genes(
        rna,
        atac,
        gene_coords,
        max_distance=1_000,
        min_abs_corr=0.0,
    )

    assert list(links.columns) == [
        "peak_id", "peak_chrom", "peak_start", "peak_end",
        "gene", "gene_tss", "distance", "correlation",
    ]
    assert len(links) == 4


def test_dense_rna_memory_warning_fires_when_matrix_is_huge(monkeypatch):
    """Warn users before the dense RNA path touches a very large matrix.

    We don't actually build a 10 GB matrix. We patch the threshold down and
    check the warning text includes the dataset shape.
    """
    import warnings
    import rustscenic.enhancer as enh

    rng = np.random.default_rng(0)
    n = 40
    rna = ad.AnnData(
        X=rng.normal(size=(n, 3)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    atac = ad.AnnData(
        X=rng.normal(size=(n, 2)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["chr1:100-200", "chr1:500-600"]),
    )
    gene_coords = pd.DataFrame({
        "gene": ["G1", "G2", "G3"],
        "chrom": ["chr1"] * 3,
        "tss": [150, 550, 900],
    })

    monkeypatch.setattr(enh, "_DENSIFY_WARN_BYTES", 1)  # trip immediately

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        link_peaks_to_genes(rna, atac, gene_coords, min_abs_corr=0.0)
    messages = [str(w.message) for w in caught]
    assert any("dense 40 × 3 float32 RNA matrix" in m for m in messages), (
        f"expected dense RNA memory warning, got: {messages}"
    )


def test_parse_peak_names_handles_alt_contigs():
    """10x ATAC peak names include alt-contigs like ``KI270721.1:2090-2985`` and
    ``GL000220.1:100-200``. The period in the contig token must not break parsing.
    Regression: pre-v0.3.9 the regex restricted chrom to ``[\\dXYMT...]+`` which
    rejected dotted accessions and caused pipeline.run to fail at the enhancer
    stage on raw 10x multiome output.
    """
    from rustscenic.enhancer import _parse_peak_names

    names = [
        "chr1:100-200",
        "1:300-400",
        "KI270721.1:2090-2985",
        "GL000220.1:5000-6000",
        "chrX:7000-8000",
    ]
    parsed = _parse_peak_names(names)
    assert parsed is not None
    assert list(parsed["chrom"]) == [
        "chr1", "1", "KI270721.1", "GL000220.1", "chrX",
    ]
    assert list(parsed["start"]) == [100, 300, 2090, 5000, 7000]
    assert list(parsed["end"]) == [200, 400, 2985, 6000, 8000]


def test_parse_peak_names_rejects_malformed():
    """The error-path: any unparseable name in the batch must return None
    (not silently produce a coords frame with bogus rows)."""
    from rustscenic.enhancer import _parse_peak_names

    # Mixed valid + invalid: function rejects the whole batch.
    assert _parse_peak_names(["chr1:100-200", "completely_invalid"]) is None
    # Pure garbage.
    assert _parse_peak_names(["not a peak"]) is None
    # Empty string.
    assert _parse_peak_names([""]) is None
    # Degenerate chrom tokens (regression: pre-tightening these passed).
    assert _parse_peak_names(["..._:100-200"]) is None
    assert _parse_peak_names(["._:100-200"]) is None
    assert _parse_peak_names(["_:100-200"]) is None


def test_parse_peak_names_rejects_inverted_coords():
    """start >= end is a bad coordinate window; refuse rather than coerce."""
    from rustscenic.enhancer import _parse_peak_names

    assert _parse_peak_names(["chr1:500-200"]) is None  # inverted
    assert _parse_peak_names(["chr1:100-100"]) is None  # zero width
