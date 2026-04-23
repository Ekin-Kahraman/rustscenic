"""Tests for eRegulon assembly.

Validates the three-way intersection of GRN, cistarget, and enhancer
outputs into TF × enhancers × target_genes records.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rustscenic.eregulon import (
    ERegulon,
    build_eregulons,
    eregulons_to_dataframe,
)


# ---- fixtures --------------------------------------------------------------


def _fixture_grn() -> pd.DataFrame:
    """TF → target predictions from GRN. SPI1 targets A-E, PAX5 targets F-I."""
    return pd.DataFrame(
        [
            ("SPI1", f"GENE_{g}", i * 0.1)
            for i, g in enumerate(["A", "B", "C", "D", "E", "F"])
        ] + [
            ("PAX5", f"GENE_{g}", i * 0.1)
            for i, g in enumerate(["F", "G", "H", "I", "J"])
        ],
        columns=["TF", "target", "importance"],
    )


def _fixture_cistarget() -> pd.DataFrame:
    """TF-motif enrichments. SPI1 enriched on peaks 1-3, PAX5 on peaks 4-5."""
    return pd.DataFrame(
        [
            ("SPI1_regulon", "SPI1_motif_a", "peak_1", 0.20),
            ("SPI1_regulon", "SPI1_motif_a", "peak_2", 0.18),
            ("SPI1_regulon", "SPI1_motif_a", "peak_3", 0.15),
            ("PAX5_regulon", "PAX5_motif_b", "peak_4", 0.22),
            ("PAX5_regulon", "PAX5_motif_b", "peak_5", 0.19),
            # A very-low-AUC hit that should be filtered out
            ("SPI1_regulon", "SPI1_motif_a", "peak_6", 0.01),
        ],
        columns=["regulon", "motif", "peak_id", "auc"],
    )


def _fixture_enhancer_links() -> pd.DataFrame:
    """Peak → gene links. peak_1 reaches GENE_A/B/C, peak_2 reaches GENE_D,
    peak_3 reaches GENE_E, peak_4/5 reach GENE_F/G/H/I."""
    return pd.DataFrame(
        [
            ("peak_1", "GENE_A", 0.6),
            ("peak_1", "GENE_B", 0.55),
            ("peak_1", "GENE_C", 0.5),
            ("peak_2", "GENE_D", 0.52),
            ("peak_3", "GENE_E", 0.48),
            ("peak_4", "GENE_F", 0.6),
            ("peak_4", "GENE_G", 0.45),
            ("peak_5", "GENE_H", 0.5),
            ("peak_5", "GENE_I", 0.47),
            # Negative correlation — should be dropped by default
            ("peak_1", "GENE_NEG", -0.7),
            # Link to a gene that GRN doesn't predict for SPI1 — gets dropped
            # under use_grn_intersection=True
            ("peak_1", "GENE_NOT_IN_GRN", 0.8),
        ],
        columns=["peak_id", "gene", "correlation"],
    )


# ---- happy path ------------------------------------------------------------


def test_builds_eregulons_for_both_tfs():
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        min_target_genes=3, min_enhancer_links=2,
    )
    tfs = [e.tf for e in eregs]
    assert "SPI1" in tfs
    assert "PAX5" in tfs


def test_spi1_targets_are_grn_intersect_enhancer():
    """SPI1 enhancer-linked: GENE_A, B, C, D, E, NEG, NOT_IN_GRN.
    After dropping negative-correlation and GRN-not-predicted targets:
    A, B, C, D, E survive (all in GRN's SPI1 target set)."""
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        min_target_genes=3, min_enhancer_links=2,
    )
    spi1 = next(e for e in eregs if e.tf == "SPI1")
    assert set(spi1.target_genes) == {"GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"}
    assert set(spi1.enhancers) == {"peak_1", "peak_2", "peak_3"}
    # 5 targets, 1 peak each except peak_1 which has 3 → 3+1+1 = 5 edges
    assert spi1.n_enhancer_links == 5


def test_low_auc_peak_is_filtered():
    """peak_6 at auc=0.01 should not appear in any eRegulon."""
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        cistarget_auc_threshold=0.05, min_target_genes=3,
    )
    for e in eregs:
        assert "peak_6" not in e.enhancers


def test_negative_correlation_dropped_by_default():
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        min_target_genes=3,
    )
    for e in eregs:
        assert "GENE_NEG" not in e.target_genes


def test_min_target_genes_filter():
    """Raising min_target_genes should drop eRegulons that don't meet it."""
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        min_target_genes=10,  # PAX5 has 4, SPI1 has 5 — both drop
    )
    assert eregs == []


def test_grn_intersection_off_keeps_extra_targets():
    """With use_grn_intersection=False, GENE_NOT_IN_GRN should be retained
    (it's enhancer-linked, just not GRN-predicted)."""
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        use_grn_intersection=False, min_target_genes=3,
    )
    spi1 = next(e for e in eregs if e.tf == "SPI1")
    assert "GENE_NOT_IN_GRN" in spi1.target_genes


def test_grn_none_requires_use_grn_intersection_false():
    with pytest.raises(ValueError, match="use_grn_intersection=True but grn is None"):
        build_eregulons(None, _fixture_cistarget(), _fixture_enhancer_links())


def test_grn_none_accepted_with_flag():
    eregs = build_eregulons(
        None, _fixture_cistarget(), _fixture_enhancer_links(),
        use_grn_intersection=False, min_target_genes=3,
    )
    assert len(eregs) >= 1


# ---- error handling --------------------------------------------------------


def test_missing_cistarget_columns_raises():
    bad = pd.DataFrame({"regulon": ["SPI1_regulon"], "auc": [0.2]})
    with pytest.raises(ValueError, match="peak / region identifier"):
        build_eregulons(_fixture_grn(), bad, _fixture_enhancer_links(), use_grn_intersection=False)


def test_missing_enhancer_columns_raises():
    bad = pd.DataFrame({"peak_id": ["peak_1"], "gene": ["GENE_A"]})  # no correlation
    with pytest.raises(ValueError, match="enhancer_links is missing"):
        build_eregulons(
            _fixture_grn(), _fixture_cistarget(), bad, use_grn_intersection=False,
        )


# ---- DataFrame export ------------------------------------------------------


def test_eregulons_to_dataframe_flattens():
    eregs = [
        ERegulon(
            tf="SPI1",
            enhancers=["peak_1"],
            target_genes=["GENE_A", "GENE_B"],
            n_enhancer_links=2,
            motif_auc=0.2,
        ),
    ]
    df = eregulons_to_dataframe(eregs)
    assert set(df.columns) == {"tf", "enhancer", "target_gene", "n_enhancer_links", "motif_auc"}
    # 1 TF × 1 enhancer × 2 genes = 2 rows
    assert len(df) == 2


def test_eregulons_sorted_by_edge_count_descending():
    eregs = build_eregulons(
        _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
        min_target_genes=3, min_enhancer_links=1,
    )
    n_edges = [e.n_enhancer_links for e in eregs]
    assert n_edges == sorted(n_edges, reverse=True)


def test_catastrophic_drop_emits_warning():
    """> 50% TF drop triggers a diagnostic warning so silent-empty isn't silent."""
    import warnings

    # Two TFs in cistarget (SPI1, PAX5), GRN only supports SPI1 → one survives
    # if we raise min_target_genes high enough the intersection drops both.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eregs = build_eregulons(
            _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
            min_target_genes=100, min_enhancer_links=1,
        )
    assert len(eregs) == 0
    drop_warnings = [w for w in caught if "kept only" in str(w.message)]
    assert drop_warnings, "silent zero-output should have emitted a warning"
    assert "use_grn_intersection=False" in str(drop_warnings[0].message)


def test_catastrophic_drop_quiet_when_healthy():
    """Healthy runs (≥ 50% of input TFs survive) do not spam warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eregs = build_eregulons(
            _fixture_grn(), _fixture_cistarget(), _fixture_enhancer_links(),
            min_target_genes=3, min_enhancer_links=1,
        )
    assert len(eregs) >= 1
    drop_warnings = [w for w in caught if "kept only" in str(w.message)]
    assert not drop_warnings, (
        f"healthy run should not warn about drops, got: {drop_warnings}"
    )


def test_polarity_suffix_normalised_to_bare_tf():
    """scenicplus emits `TF(+)` / `TF_activator` / `TF_extended` variants.
    The intersection step must map all back to the bare TF symbol."""
    grn = pd.DataFrame([
        ("SPI1", "GENE_A", 0.5), ("SPI1", "GENE_B", 0.4), ("SPI1", "GENE_C", 0.3),
        ("SPI1", "GENE_D", 0.2), ("SPI1", "GENE_E", 0.1),
    ], columns=["TF", "target", "importance"])
    cistarget = pd.DataFrame([
        {"regulon": "SPI1(+)", "motif": "m", "peak_id": f"peak_{i}", "auc": 0.2}
        for i in range(3)
    ] + [
        {"regulon": "SPI1_extended", "motif": "m", "peak_id": f"peak_{i}", "auc": 0.2}
        for i in range(3, 5)
    ])
    enhancer_links = pd.DataFrame([
        {"peak_id": f"peak_{i}", "gene": g, "correlation": 0.3}
        for i in range(5)
        for g in ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
    ])
    eregs = build_eregulons(
        grn, cistarget, enhancer_links,
        min_target_genes=3, min_enhancer_links=1,
    )
    # Both polarity suffixes should map to SPI1
    assert len(eregs) >= 1
    assert any(e.tf == "SPI1" for e in eregs), (
        f"polarity suffix stripping failed; TFs seen: {[e.tf for e in eregs]}"
    )
