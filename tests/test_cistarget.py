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
    # Motif 0 — regulon genes (g0-g4) at top ranks (0-4); others shifted
    for j in range(20):
        if j < 5:
            rankings.loc["m0", f"g{j}"] = j       # regulon genes at ranks 0-4
        else:
            rankings.loc["m0", f"g{j}"] = j       # g5-g19 at ranks 5-19
    # Motifs 1-9 — regulon genes (g0-g4) at WORST ranks (15-19), others at top
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
        assert set(out.columns) == {"regulon", "motif", "auc"}


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
        assert set(out.columns) == {"regulon", "motif", "auc"}

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
