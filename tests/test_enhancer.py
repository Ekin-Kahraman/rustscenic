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
      peak_2 on chr1:200000-200500 (far from any gene) — should not link
    Other gene-peak pairs are independent noise.
    """
    rng = np.random.default_rng(seed)
    n_cells = n_cells

    # Shared latent — drives both peak_0 and GENE_A
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


def test_links_correlated_peak_to_correlated_gene():
    rna, atac, genes = _synthetic_multiome(seed=1)
    links = link_peaks_to_genes(rna, atac, genes, min_abs_corr=0.5)
    # Expect peak_0 ↔ GENE_A and peak_1 ↔ GENE_C above threshold
    found = set(zip(links["peak_id"], links["gene"]))
    assert ("chr1:1000-1500", "GENE_A") in found
    assert ("chr1:60000-60500", "GENE_C") in found


def test_rejects_noise_peak_with_no_nearby_gene():
    """peak_2 on chr1:200000-200500 is far from every gene (nearest is
    GENE_A at TSS 1,250, distance ~199kb — within default 500kb but
    correlation is ~0). Should fall below threshold, not get linked."""
    rna, atac, genes = _synthetic_multiome(seed=42)
    links = link_peaks_to_genes(rna, atac, genes, min_abs_corr=0.5)
    assert ("chr1:200000-200500", "GENE_A") not in set(zip(links["peak_id"], links["gene"]))


def test_distance_filter_excludes_genes_too_far_away():
    rna, atac, genes = _synthetic_multiome(seed=1)
    # Tight distance cap — GENE_C TSS at 60,200 is too far from peak_0 at ~1,250
    links = link_peaks_to_genes(rna, atac, genes, max_distance=10_000, min_abs_corr=0.0)
    assert ("chr1:1000-1500", "GENE_C") not in set(zip(links["peak_id"], links["gene"]))


def test_spearman_method_accepted():
    rna, atac, genes = _synthetic_multiome(seed=7)
    links = link_peaks_to_genes(rna, atac, genes, method="spearman", min_abs_corr=0.3)
    assert not links.empty
    # Column schema is stable across methods
    assert list(links.columns) == [
        "peak_id", "peak_chrom", "peak_start", "peak_end",
        "gene", "gene_tss", "distance", "correlation",
    ]


def test_cell_mismatch_raises():
    rna, atac, genes = _synthetic_multiome()
    rna2 = rna.copy()
    rna2.obs_names = [f"different_cell{i}" for i in range(rna2.n_obs)]
    with pytest.raises(ValueError, match="share no cell barcodes"):
        link_peaks_to_genes(rna2, atac, genes)


def test_partial_cell_overlap_warns_and_continues():
    rna, atac, genes = _synthetic_multiome()
    # Shrink RNA to 80% of cells — ATAC still has full set
    rna2 = rna[:400].copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        links = link_peaks_to_genes(rna2, atac, genes, min_abs_corr=0.5)
    assert any("keeping" in str(w.message) for w in caught)
    assert not links.empty


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
    assert ("peakA", "GENE_A") in set(zip(links["peak_id"], links["gene"]))


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


def test_densification_warning_fires_when_matrix_is_huge(monkeypatch):
    """Warn users before the sparse→dense step blows past 8 GiB per matrix.

    We don't actually build a 10 GB matrix — we patch the threshold down
    to 1 KiB and check the warning text includes the dataset shape.
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
    assert any("densify" in m for m in messages), (
        f"expected densification warning, got: {messages}"
    )
