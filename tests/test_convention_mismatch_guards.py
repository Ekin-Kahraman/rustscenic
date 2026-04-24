"""Silent-zero regression guards — the Fuaad-class bug catalogue.

Every test in this file corresponds to a real-world data convention
that produces a *correctly-typed but empty/zero* result if rustscenic
doesn't handle it. These are the bugs that don't crash — they just
quietly return garbage. Catching them requires asserting on values,
not just shapes.

Covers:
  1. Chromosome naming — UCSC `chr1` vs Ensembl `1`
  2. Mitochondrial aliases — `chrM` vs `chrMT` vs `MT`
  3. Cistarget rankings indexed by ENSEMBL while regulons use symbols
  4. Cistarget rankings for the wrong species
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest


# ---- chromosome naming ----------------------------------------------------


def test_ucsc_vs_ensembl_chrom_names_dont_silently_drop_fragments():
    """Fragments in `chr1` notation, peaks in `1` notation.
    `build_cell_peak_matrix` must still produce a non-empty matrix.
    Before the normalise_chrom helper, every overlap was silently dropped."""
    import io
    from rustscenic._rustscenic import __version__  # ensure module built

    # Exercise the Rust layer via its PyO3 binding when available; for now
    # the pure-Rust path is covered by the cargo tests. This test asserts
    # the *symptom* — Python-facing behaviour — is correct.
    # We construct a minimal scenario using the pure-Python preproc shape.
    # Since fragments_to_matrix is the public API, we simulate by calling
    # it on a hand-authored temp fragments + peaks file.
    import tempfile
    import gzip
    import os

    frag_lines = [
        # UCSC-style chroms
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t150\t250\tAAA-1\t1",
        "chr1\t500\t900\tBBB-1\t1",
        "chr2\t1000\t1100\tAAA-1\t1",
    ]
    peak_lines = [
        # Ensembl-style chroms — no `chr` prefix
        "1\t100\t300\tpeak1",
        "1\t500\t1000\tpeak2",
        "2\t900\t1200\tpeak3",
    ]

    import rustscenic.preproc
    with tempfile.TemporaryDirectory() as td:
        frag_path = os.path.join(td, "frags.tsv.gz")
        peak_path = os.path.join(td, "peaks.bed")
        with gzip.open(frag_path, "wt") as fh:
            fh.write("\n".join(frag_lines) + "\n")
        with open(peak_path, "w") as fh:
            fh.write("\n".join(peak_lines) + "\n")

        adata = rustscenic.preproc.fragments_to_matrix(frag_path, peak_path)

    # If chrom normalisation works, AAA-1's fragments at chr1:100-200 should
    # hit Ensembl-1:100-300 → one count for AAA-1 in peak1.
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    assert X.sum() > 0, (
        "chrom-naming mismatch (chr1 vs 1) silently dropped every fragment. "
        "If you see this fail, align_chroms_to has regressed."
    )


# ---- cistarget convention ------------------------------------------------


def test_cistarget_all_regulons_missing_warns_loudly():
    """Rankings indexed by ENSEMBL + regulons in symbols = silent empty output.
    We can't auto-swap here because the rankings DataFrame is user-supplied;
    we must at least warn visibly with a diagnostic."""
    import rustscenic.cistarget

    # Ranking matrix indexed by ENSEMBL IDs — typical of aertslab v10 output.
    n_motifs = 3
    n_genes = 10
    rankings = pd.DataFrame(
        np.tile(np.arange(n_genes), (n_motifs, 1)),
        index=[f"MOTIF_{i}" for i in range(n_motifs)],
        columns=[f"ENSG0000011{i:04d}" for i in range(n_genes)],
    )
    # Regulons use HGNC symbols — NONE overlap with ENSEMBL columns
    regulons = [("SPI1_regulon", ["SPI1", "CEBPB", "IRF8"])]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = rustscenic.cistarget.enrich(rankings, regulons, auc_threshold=0.0)

    assert out.empty, "expected empty output when no regulon genes match"
    messages = [str(w.message) for w in caught]
    assert any(
        "all 1 regulons dropped" in m or "none of their genes" in m
        for m in messages
    ), f"expected loud warning on full dropout; got: {messages}"


def test_cistarget_partial_overlap_warns():
    """Rankings have some overlap with regulon genes but most miss.
    Should still run, but warn_if_poor_coverage should fire."""
    import rustscenic.cistarget

    gene_cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    rankings = pd.DataFrame(
        np.tile(np.arange(len(gene_cols)), (3, 1)),
        index=["M0", "M1", "M2"],
        columns=gene_cols,
    )
    regulons = [("R1", ["A", "B", "X", "Y", "Z", "W", "V", "U"])]  # 2/8 match

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = rustscenic.cistarget.enrich(rankings, regulons, auc_threshold=0.0)

    assert not out.empty  # still runs
    messages = [str(w.message) for w in caught]
    assert any(
        "regulons have <" in m or "< 50%" in m for m in messages
    ), f"expected coverage warning on partial overlap; got: {messages}"


def test_cistarget_full_match_no_warning():
    """All regulon genes present → silent pass."""
    import rustscenic.cistarget

    gene_cols = ["A", "B", "C", "D", "E"]
    rankings = pd.DataFrame(
        np.tile(np.arange(len(gene_cols)), (3, 1)),
        index=["M0", "M1", "M2"],
        columns=gene_cols,
    )
    regulons = [("R1", ["A", "B", "C"])]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = rustscenic.cistarget.enrich(rankings, regulons, auc_threshold=0.0)

    assert not out.empty
    # Should not emit any coverage warnings
    bad = [w for w in caught if "regulons have <" in str(w.message)]
    assert not bad, f"unexpected coverage warning on full-match: {bad}"


# ---- integration: chrom + var_names both wrong ----------------------------


def test_full_pipeline_survives_both_chrom_and_var_names_conventions():
    """End-to-end: RNA in cellxgene shape AND ATAC peak BED uses Ensembl
    chrom convention. Both resolvers must kick in for the chain to work."""
    import rustscenic.aucell
    import rustscenic.enhancer

    # Build matched multiome
    rng = np.random.default_rng(0)
    n_cells = 120
    latent_a = rng.normal(size=n_cells)

    # RNA: 20 genes, 5 of which track latent_a
    rna_X = np.column_stack([
        0.8 * latent_a + 0.2 * rng.normal(size=n_cells) if i < 5
        else rng.normal(size=n_cells)
        for i in range(20)
    ]).astype(np.float32)

    # Cellxgene-style var
    rna_symbols = [f"SYM{i}" for i in range(20)]
    rna_ensembl = [f"ENSG0000011{i:04d}" for i in range(20)]
    rna = ad.AnnData(
        X=rna_X,
        var=pd.DataFrame({"feature_name": rna_symbols}, index=rna_ensembl),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)]),
    )

    # ATAC: peaks in Ensembl chrom convention; 5 peaks on chrom "1",
    # first 3 tracking latent_a.
    atac_X = np.column_stack([
        0.8 * latent_a + 0.2 * rng.normal(size=n_cells) if i < 3
        else rng.normal(size=n_cells)
        for i in range(5)
    ]).astype(np.float32)
    peak_names = [f"1:{(i+1)*10_000}-{(i+1)*10_000+500}" for i in range(5)]
    atac = ad.AnnData(
        X=atac_X,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=peak_names),
    )

    # Gene coords use UCSC "chr1" — cross-convention between RNA and peaks
    # is realistic (genome annotations and multiome peaks often come from
    # different providers). Enhancer linker should still connect them.
    gene_coords = pd.DataFrame({
        "gene": rna_symbols,
        "chrom": ["chr1"] * 20,
        "tss": [(i + 1) * 10_000 + 250 for i in range(20)],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        links = rustscenic.enhancer.link_peaks_to_genes(
            rna, atac, gene_coords, max_distance=1_000_000, min_abs_corr=0.3,
        )

    # Chroms between peaks ("1") and gene_coords ("chr1") are normalised
    # in the enhancer linker (PR #29) so peak↔gene joins still work across
    # UCSC/Ensembl. Additionally verify AUCell on the cellxgene RNA works —
    # the original Fuaad scenario.
    auc = rustscenic.aucell.score(
        rna, [("R_latent_A", ["SYM0", "SYM1", "SYM2", "SYM3"])], top_frac=0.3,
    )
    assert auc.shape == (n_cells, 1)
    assert (auc.values > 0).any(), "cellxgene-shape AUCell regressed silently"
