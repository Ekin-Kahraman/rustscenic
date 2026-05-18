"""Tests for rustscenic.cistarget.enrich."""
import numpy as np
import pandas as pd
import pytest

import rustscenic.cistarget as cistarget


@pytest.fixture
def tiny_rankings():
    """10 motifs × 20 genes. Motif 0 has regulon genes at top ranks; other
    motifs are deterministically scrambled so none by chance also has g0-g4
    at low ranks."""
    rankings = pd.DataFrame(
        np.zeros((10, 20), dtype=np.int32),
        index=[f"m{i}" for i in range(10)],
        columns=[f"g{i}" for i in range(20)],
    )
    # Motif 0 - regulon genes (g0-g4) at top ranks (0-4); others shifted
    for j in range(20):
        if j < 5:
            rankings.loc["m0", f"g{j}"] = j       # regulon genes at ranks 0-4
        else:
            rankings.loc["m0", f"g{j}"] = j       # g5-g19 at ranks 5-19
    # Motifs 1-9 - regulon genes (g0-g4) at WORST ranks (15-19), others at top
    for i in range(1, 10):
        # Cycle so different motifs get different permutations but all place
        # regulon genes at the bottom.
        offset = 15 + ((i + 0) % 5)
        for j in range(20):
            if j < 5:
                rankings.loc[f"m{i}", f"g{j}"] = offset  # worst ranks 15-19
                offset = 15 + ((offset - 15 + 1) % 5)
            else:
                rankings.loc[f"m{i}", f"g{j}"] = (i * 3 + j) % 15  # 0-14
    return rankings


class TestCistargetShape:
    def test_returns_df_with_expected_cols(self, tiny_rankings):
        regs = [("R1", ["g0", "g1", "g2", "g3", "g4"])]
        out = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=0.0)
        assert set(out.columns) == {"regulon", "motif", "auc", "nes"}


class TestCistargetCorrectness:
    def test_self_consistency_motif_tops_its_own_genes(self, tiny_rankings):
        """Motif 0 has g0-g4 at ranks 0-4. A regulon of those genes should
        rank motif 0 at the top of the enrichment."""
        regs = [("R1", ["g0", "g1", "g2", "g3", "g4"])]
        out = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=0.0)
        top_motif = out.sort_values("auc", ascending=False).iloc[0]["motif"]
        assert top_motif == "m0"

    def test_prune_enriched_motifs_requires_motif_annotation_support(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "G000_regulon", "motif": "M_G000", "auc": 0.30},
                {"regulon": "G005_regulon", "motif": "M_WRONG", "auc": 0.40},
                {"regulon": "G010_regulon", "motif": "M_G005", "auc": 0.50},
            ]
        )
        annotations = pd.DataFrame(
            {
                "motif": ["M_G000", "M_WRONG", "M_G005"],
                "TF": ["G000", "OTHER", "G005"],
            }
        )

        out = cistarget.prune_enriched_motifs(enriched, annotations)

        assert out[["regulon", "motif", "tf", "annotation_tf"]].to_dict("records") == [
            {
                "regulon": "G000_regulon",
                "motif": "M_G000",
                "tf": "G000",
                "annotation_tf": "G000",
            }
        ]

    def test_prune_regulons_keeps_only_recovered_targets(self):
        enriched = pd.DataFrame(
            [{"regulon": "G000_regulon", "motif": "M_G000", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M_G000"], "TF": ["G000"]})
        rankings = pd.DataFrame(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            index=["M_G000"],
            columns=[f"g{i}" for i in range(10)],
        )
        candidates = [("G000_regulon", [f"g{i}" for i in range(5)])]

        pruned = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=rankings,
            top_frac=0.2,
        )

        assert pruned == {"G000_regulon": ["g0", "g1"]}

    @pytest.mark.parametrize(
        "regulon_name, expected_tf",
        [
            ("PAX5_regulon", "PAX5"),
            ("FOXP3_extended", "FOXP3"),
            ("FOXP3_extended_regulon", "FOXP3"),
            ("PAX5(+)", "PAX5"),
            ("PAX5_regulon(+)", "PAX5"),
            ("PAX5_extended(+)", "PAX5"),
            ("PAX5_extended_activator", "PAX5"),
            ("PAX5_extended_repressor(-)", "PAX5"),
        ],
    )
    def test_tf_from_regulon_name_strips_compound_suffixes(self, regulon_name, expected_tf):
        """``_tf_from_regulon_name`` must strip every recognised suffix and
        polarity marker, even when they appear together (canonical scenicplus
        names look like ``FOXP3_extended_regulon`` or ``PAX5_extended(+)``).

        The original implementation broke on the first match and left the
        compound suffix attached, so any signed / extended / activator regulon
        from scenicplus failed to match its motif annotations during pruning.
        """
        from rustscenic.cistarget import _tf_from_regulon_name
        assert _tf_from_regulon_name(regulon_name) == expected_tf


class TestCistargetEdgeCases:
    def test_object_dtype_rankings_rejected(self):
        bad = pd.DataFrame([["a", "b"], ["c", "d"]], index=["m1", "m2"], columns=["g1", "g2"])
        with pytest.raises(TypeError, match="dtype=object"):
            cistarget.enrich(bad, [("R", ["g1"])])

    def test_nan_rankings_rejected(self, tiny_rankings):
        bad = tiny_rankings.astype(np.float32)
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match=r"NaN|Inf|finite"):
            cistarget.enrich(bad, [("R", ["g0", "g1"])])

    def test_empty_regulons_returns_empty_df(self, tiny_rankings):
        out = cistarget.enrich(tiny_rankings, [], auc_threshold=0.0)
        assert len(out) == 0
        assert set(out.columns) == {"regulon", "motif", "auc", "nes"}

    def test_auc_threshold_filters(self, tiny_rankings):
        # Very high threshold should filter out everything
        regs = [("R1", ["g0", "g1", "g2"])]
        out = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=10.0)
        assert len(out) == 0

    def test_prune_enriched_motifs_rejects_missing_annotation_columns(self):
        enriched = pd.DataFrame(
            [{"regulon": "G000_regulon", "motif": "M_G000", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M_G000"], "not_tf": ["G000"]})
        with pytest.raises(ValueError, match="TF column"):
            cistarget.prune_enriched_motifs(enriched, annotations)


class TestCistargetDeterminism:
    def test_reproducible(self, tiny_rankings):
        regs = [("R1", ["g0", "g1", "g2", "g3", "g4"])]
        a = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=0.0)
        b = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=0.0)
        pd.testing.assert_frame_equal(
            a.sort_values(["motif", "regulon"]).reset_index(drop=True),
            b.sort_values(["motif", "regulon"]).reset_index(drop=True),
        )


class TestCistargetNES:
    """v0.4.4 NES (normalised enrichment score) tests.

    Lines up with pyscenic transform.py / pycistarget motif_enrichment_cistarget.py:
    per-regulon population z-score of AUC across the motif universe.
    """

    @pytest.fixture
    def big_synthetic_rankings(self):
        """200 motifs x 500 genes. Motif 0 has g0..g19 at top ranks; all other
        motifs assign g0..g19 to random ranks. A regulon of g0..g19 should land
        motif 0 at high NES and every other motif at low NES."""
        rng = np.random.default_rng(0)
        n_motifs, n_genes = 200, 500
        regulon_genes = list(range(20))
        rankings = np.zeros((n_motifs, n_genes), dtype=np.int32)
        for i in range(n_motifs):
            if i == 0:
                perm = regulon_genes + [g for g in range(n_genes) if g not in regulon_genes]
            else:
                perm = list(rng.permutation(n_genes))
            for rank, gene in enumerate(perm):
                rankings[i, gene] = rank
        return pd.DataFrame(
            rankings,
            index=[f"m{i}" for i in range(n_motifs)],
            columns=[f"g{i}" for i in range(n_genes)],
        )

    def test_nes_separates_true_positive_from_noise(self, big_synthetic_rankings):
        regs = [("R1", [f"g{i}" for i in range(20)])]
        out = cistarget.enrich(
            big_synthetic_rankings, regs, top_frac=0.05, auc_threshold=0.0,
        )
        # The true-positive motif m0 must land at the top NES, and the gap to
        # the noise floor must be wide enough to reflect a real signal.
        top = out.sort_values("nes", ascending=False).iloc[0]
        assert top["motif"] == "m0", (
            f"true-positive motif m0 was not at the top NES; "
            f"top row instead: {top.to_dict()}"
        )
        assert top["nes"] > 5.0, (
            f"true-positive NES should be high (z >> 3); got {top['nes']:.3f}"
        )
        # NES >= 3.0 should isolate the true positive on this synthetic
        # fixture (noise motifs sit at NES around 0 by construction).
        n_above_threshold = int((out["nes"] >= 3.0).sum())
        assert n_above_threshold == 1, (
            f"NES >= 3.0 should keep exactly 1 motif on this fixture; got "
            f"{n_above_threshold}"
        )

    def test_nes_threshold_filter_drops_noise(self, big_synthetic_rankings):
        regs = [("R1", [f"g{i}" for i in range(20)])]
        full = cistarget.enrich(
            big_synthetic_rankings, regs, top_frac=0.05, auc_threshold=0.0,
        )
        filtered = cistarget.enrich(
            big_synthetic_rankings, regs, top_frac=0.05,
            auc_threshold=0.0, nes_threshold=3.0,
        )
        assert len(filtered) == 1
        assert len(full) >= 1
        # Filter should not invent rows or change AUC values
        assert filtered.iloc[0]["motif"] == "m0"
        assert filtered.iloc[0]["auc"] == full.loc[full["motif"] == "m0", "auc"].iloc[0]

    def test_nes_nan_when_motif_universe_below_floor(self, tiny_rankings):
        """tiny_rankings has 10 motifs; below the 30-motif floor NES is NaN."""
        regs = [("R1", ["g0", "g1", "g2", "g3", "g4"])]
        with pytest.warns(UserWarning, match="30-motif floor"):
            out = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=0.0)
        assert out["nes"].isna().all(), (
            "NES must be NaN for every row when n_motifs < 30"
        )

    def test_nes_threshold_drops_nan_rows(self, tiny_rankings):
        regs = [("R1", ["g0", "g1", "g2", "g3", "g4"])]
        with pytest.warns(UserWarning, match="30-motif floor"):
            out = cistarget.enrich(
                tiny_rankings, regs, top_frac=0.3,
                auc_threshold=0.0, nes_threshold=3.0,
            )
        assert len(out) == 0, (
            "NaN NES rows must be dropped when nes_threshold is set; "
            "otherwise NaN comparisons silently slip through"
        )

    def test_nes_zero_variance_warns_and_emits_nan(self):
        """A regulon whose AUC is constant across all motifs (because no
        regulon gene appears in any motif's top window) has zero variance;
        NES is undefined."""
        n_motifs, n_genes = 50, 100
        rankings = pd.DataFrame(
            np.tile(np.arange(n_genes, dtype=np.int32), (n_motifs, 1)),
            index=[f"m{i}" for i in range(n_motifs)],
            columns=[f"g{i}" for i in range(n_genes)],
        )
        # All motifs assign identical ranks, so every regulon's AUC is
        # identical across the motif universe -> std == 0.
        regs = [("R_CONST", [f"g{i}" for i in range(10)])]
        with pytest.warns(UserWarning, match="zero AUC variance"):
            out = cistarget.enrich(rankings, regs, top_frac=0.1, auc_threshold=0.0)
        assert out["nes"].isna().all(), (
            "NES must be NaN for zero-variance regulons"
        )
        # But AUC rows are still present (we don't drop the regulon entirely)
        assert len(out) > 0

    def test_backwards_compat_nes_threshold_none_matches_v043_row_count(self, big_synthetic_rankings):
        """Default behaviour (nes_threshold=None) must keep every row that
        auc_threshold alone would have kept under v0.4.3. The NES column is
        additive and does not change the filter logic."""
        regs = [("R1", [f"g{i}" for i in range(20)])]
        out_default = cistarget.enrich(
            big_synthetic_rankings, regs, top_frac=0.05, auc_threshold=0.0,
        )
        # Without nes_threshold, NES is computed but does not filter rows.
        # Row count must equal the number of (motif, regulon) pairs the AUC
        # filter alone would keep: every pair since auc_threshold=0.0.
        n_motifs = big_synthetic_rankings.shape[0]
        assert len(out_default) == n_motifs

    def test_prune_enriched_motifs_with_nes_threshold(self):
        """prune_enriched_motifs accepts nes_threshold when the enriched
        DataFrame has an nes column."""
        enriched = pd.DataFrame(
            [
                {"regulon": "G000_regulon", "motif": "M_G000", "auc": 0.30, "nes": 5.2},
                {"regulon": "G000_regulon", "motif": "M_OTHER", "auc": 0.31, "nes": 2.0},
            ]
        )
        annotations = pd.DataFrame(
            {"motif": ["M_G000", "M_OTHER"], "TF": ["G000", "G000"]}
        )
        out = cistarget.prune_enriched_motifs(enriched, annotations, nes_threshold=3.0)
        assert len(out) == 1
        assert out.iloc[0]["motif"] == "M_G000"

    def test_prune_enriched_motifs_rejects_nes_threshold_when_no_column(self):
        """If the enriched DataFrame is from a v0.4.3 build with no NES column,
        passing nes_threshold should raise rather than silently filter on a
        missing column."""
        enriched = pd.DataFrame(
            [{"regulon": "G000_regulon", "motif": "M_G000", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M_G000"], "TF": ["G000"]})
        with pytest.raises(ValueError, match="no `nes` column"):
            cistarget.prune_enriched_motifs(enriched, annotations, nes_threshold=3.0)
