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
        assert out.attrs["rust_backend"] == {
            "engine": "rust",
            "symbols": ["cistarget_enrichment_from_rankings_i32"],
        }


class TestCistargetCorrectness:
    def test_self_consistency_motif_tops_its_own_genes(self, tiny_rankings):
        """Motif 0 has g0-g4 at ranks 0-4. A regulon of those genes should
        rank motif 0 at the top of the enrichment."""
        regs = [("R1", ["g0", "g1", "g2", "g3", "g4"])]
        out = cistarget.enrich(tiny_rankings, regs, top_frac=0.3, auc_threshold=0.0)
        top_motif = out.sort_values("auc", ascending=False).iloc[0]["motif"]
        assert top_motif == "m0"

    def test_enrich_accepts_strided_rankings_without_python_contiguity_copy(self):
        base = np.array(
            [
                [0, 1, 2, 5, 6, 7],
                [5, 4, 3, 2, 1, 0],
                [2, 3, 4, 0, 1, 5],
            ],
            dtype=np.int32,
        )
        values = np.asfortranarray(base)
        assert values.flags.f_contiguous
        assert not values.flags.c_contiguous

        rankings = pd.DataFrame(
            values,
            index=["m0", "m1", "m2"],
            columns=[f"g{i}" for i in range(base.shape[1])],
            copy=False,
        )
        rankings_arg, kernel, projected = cistarget._rankings_kernel_arg(
            rankings.to_numpy(copy=False)
        )

        assert np.shares_memory(rankings_arg, values)
        assert rankings_arg.flags.f_contiguous
        assert not rankings_arg.flags.c_contiguous
        assert kernel is cistarget._cistarget_enrichment_from_rankings_i32
        assert projected is None

        regs = [("R1", ["g0", "g1", "g2"]), ("R2", ["g3", "g4"])]
        got = cistarget.enrich(rankings, regs, top_frac=0.5, auc_threshold=0.0)
        expected = cistarget.enrich(
            pd.DataFrame(
                base,
                index=rankings.index,
                columns=rankings.columns,
            ),
            regs,
            top_frac=0.5,
            auc_threshold=0.0,
        )
        pd.testing.assert_frame_equal(got, expected)

    def test_enrich_does_not_allocate_isfinite_mask_for_integer_rankings(
        self,
        tiny_rankings,
        monkeypatch,
    ):
        def fail_isfinite(_values):
            raise AssertionError("integer rankings do not need np.isfinite")

        monkeypatch.setattr(cistarget.np, "isfinite", fail_isfinite)

        out = cistarget.enrich(
            tiny_rankings.astype(np.int16),
            [("R1", ["g0", "g1", "g2", "g3", "g4"])],
            top_frac=0.3,
            auc_threshold=0.0,
        )

        assert not out.empty

    def test_enrich_uses_int64_rankings_without_upcast_copy(self):
        base = np.array(
            [
                [0, 1, 2, 5, 6, 7],
                [5, 4, 3, 2, 1, 0],
                [2, 3, 4, 0, 1, 5],
            ],
            dtype=np.int64,
        )
        values = np.asfortranarray(base)
        rankings = pd.DataFrame(
            values,
            index=["m0", "m1", "m2"],
            columns=[f"g{i}" for i in range(base.shape[1])],
            copy=False,
        )
        rankings_arg, kernel, projected = cistarget._rankings_kernel_arg(
            rankings.to_numpy(copy=False)
        )

        assert rankings_arg.dtype == np.int64
        assert kernel is cistarget._cistarget_enrichment_from_rankings_i64
        assert projected is None
        assert np.shares_memory(rankings_arg, values)
        assert rankings_arg.flags.f_contiguous
        assert not rankings_arg.flags.c_contiguous

        regs = [("R1", ["g0", "g1", "g2"]), ("R2", ["g3", "g4"])]
        got = cistarget.enrich(rankings, regs, top_frac=0.5, auc_threshold=0.0)
        expected = cistarget.enrich(
            pd.DataFrame(
                base.astype(np.int32),
                index=rankings.index,
                columns=rankings.columns,
            ),
            regs,
            top_frac=0.5,
            auc_threshold=0.0,
        )
        pd.testing.assert_frame_equal(got, expected)

    def test_enrich_float_rankings_validate_and_convert_in_rust(self, monkeypatch):
        base = np.array(
            [
                [0, 1, 2, 5, 6, 7],
                [5, 4, 3, 2, 1, 0],
                [2, 3, 4, 0, 1, 5],
            ],
            dtype=np.float64,
        )
        values = np.asfortranarray(base)
        rankings = pd.DataFrame(
            values,
            index=["m0", "m1", "m2"],
            columns=[f"g{i}" for i in range(base.shape[1])],
            copy=False,
        )
        seen = {}

        def fake_to_i32(values_arg):
            seen["called"] = True
            assert np.shares_memory(values_arg, values)
            assert values_arg.flags.f_contiguous
            assert not values_arg.flags.c_contiguous
            return base.astype(np.int32)

        monkeypatch.setattr(cistarget, "_rankings_to_i32_f64", fake_to_i32)

        rankings_arg, kernel, projected = cistarget._rankings_kernel_arg(
            rankings.to_numpy(copy=False)
        )

        assert seen["called"] is True
        assert rankings_arg.dtype == np.int32
        assert kernel is cistarget._cistarget_enrichment_from_rankings_i32
        assert projected is None

    def test_projected_enrich_matches_full_rankings(self, tiny_rankings):
        regs = [
            ("R1", ["g0", "g1", "g2", "g3", "g4"]),
            ("R2", ["g10", "g11", "g12"]),
        ]
        projected = tiny_rankings[["g0", "g1", "g2", "g3", "g4", "g10", "g11", "g12"]]

        got = cistarget.enrich(
            projected,
            regs,
            top_frac=0.3,
            auc_threshold=0.0,
            rank_universe_size=tiny_rankings.shape[1],
        )
        expected = cistarget.enrich(
            tiny_rankings,
            regs,
            top_frac=0.3,
            auc_threshold=0.0,
        )

        pd.testing.assert_frame_equal(got, expected)
        assert got.attrs["rust_backend"] == {
            "engine": "rust",
            "symbols": ["cistarget_enrichment_from_projected_rankings_i32"],
        }

    def test_projected_rankings_kernel_uses_projected_symbol(self, tiny_rankings):
        values = tiny_rankings[["g0", "g1"]].to_numpy(copy=False)

        rankings_arg, kernel, projected = cistarget._rankings_kernel_arg(
            values,
            rank_universe_size=tiny_rankings.shape[1],
        )

        assert np.shares_memory(rankings_arg, values)
        assert kernel is cistarget._cistarget_enrichment_from_projected_rankings_i32
        assert projected == tiny_rankings.shape[1]

    def test_float_nan_rankings_do_not_allocate_python_isfinite_mask(
        self,
        tiny_rankings,
        monkeypatch,
    ):
        bad = tiny_rankings.astype(np.float32)
        bad.iloc[0, 0] = np.nan

        def fail_isfinite(_values):
            raise AssertionError("float ranking finite validation should run in Rust")

        monkeypatch.setattr(cistarget.np, "isfinite", fail_isfinite)

        with pytest.raises(ValueError, match=r"NaN|Inf|finite"):
            cistarget.enrich(bad, [("R", ["g0", "g1"])])

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
        assert out.attrs["rust_backend"]["symbols"] == [
            "cistarget_motif_annotation_prune_standard_rows_f64"
        ]

    def test_prune_enriched_motifs_matches_previous_pandas_merge_reference(self):
        enriched = pd.DataFrame(
            [
                {
                    "regulon": "TF1_regulon",
                    "motif": "M1",
                    "auc": 0.50,
                    "nes": 3.5,
                    "source": "keep_twice",
                },
                {
                    "regulon": "TF2_regulon",
                    "motif": "M2",
                    "auc": 0.60,
                    "nes": 4.0,
                    "source": "split_tf",
                },
                {
                    "regulon": "TF3_regulon",
                    "motif": "M3",
                    "auc": 0.10,
                    "nes": 5.0,
                    "source": "low_auc",
                },
                {
                    "regulon": "TFLOW_regulon",
                    "motif": "M4",
                    "auc": 0.70,
                    "nes": np.nan,
                    "source": "nan_nes",
                },
            ]
        )
        annotations = pd.DataFrame(
            {
                "motif_id": ["M1", "M1", "M2", "M3", "M4"],
                "gene_name": ["TF1", "tf1", "TFX;TF2", "TF3", "TFLOW"],
            }
        )

        got = cistarget.prune_enriched_motifs(
            enriched,
            annotations,
            motif_col="motif_id",
            tf_col="gene_name",
            auc_threshold=0.2,
            nes_threshold=1.0,
        )
        expected = _reference_prune_enriched_motifs_with_pandas_merge(
            enriched,
            annotations,
            motif_col="motif_id",
            tf_col="gene_name",
            auc_threshold=0.2,
            nes_threshold=1.0,
        )
        pd.testing.assert_frame_equal(got, expected, check_dtype=False)
        assert got.attrs["rust_backend"]["symbols"] == [
            "cistarget_motif_annotation_prune_rows_filtered_f64"
        ]

    def test_prune_enriched_motifs_threshold_filter_runs_in_rust_without_float32_upcast(
        self,
        monkeypatch,
    ):
        enriched = pd.DataFrame(
            {
                "regulon": ["TF1_regulon", "TF2_regulon"],
                "motif": ["M1", "M2"],
                "auc": np.asarray([0.5, 0.1], dtype=np.float32),
                "nes": np.asarray([3.5, 0.5], dtype=np.float32),
            }
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        auc_source = enriched["auc"].to_numpy(copy=False)
        nes_source = enriched["nes"].to_numpy(copy=False)

        def fake_standard_rows(
            enriched_regulons,
            enriched_motifs,
            enriched_auc,
            enriched_nes,
            annotation_motifs,
            annotation_tfs,
            auc_threshold,
            nes_threshold,
            case_sensitive,
        ):
            assert enriched_regulons == ["TF1_regulon", "TF2_regulon"]
            assert enriched_motifs == ["M1", "M2"]
            assert np.shares_memory(enriched_auc, auc_source)
            assert np.shares_memory(enriched_nes, nes_source)
            assert enriched_auc.dtype == np.float32
            assert enriched_nes.dtype == np.float32
            assert auc_threshold == 0.2
            assert nes_threshold == 1.0
            assert case_sensitive is False
            return (
                ["TF1_regulon"],
                ["M1"],
                np.asarray([0.5], dtype=np.float32),
                np.asarray([3.5], dtype=np.float32),
                ["TF1"],
                ["TF1"],
            )

        monkeypatch.setattr(
            cistarget,
            "_motif_annotation_prune_standard_rows_f32",
            fake_standard_rows,
        )

        out = cistarget.prune_enriched_motifs(
            enriched,
            annotations,
            auc_threshold=0.2,
            nes_threshold=1.0,
        )

        assert out[["regulon", "motif", "tf", "annotation_tf"]].to_dict("records") == [
            {
                "regulon": "TF1_regulon",
                "motif": "M1",
                "tf": "TF1",
                "annotation_tf": "TF1",
            }
        ]
        assert out.attrs["rust_backend"]["symbols"] == [
            "cistarget_motif_annotation_prune_standard_rows_f32"
        ]

    def test_prune_enriched_motifs_maps_missing_annotation_tfs_without_pandas_mask(
        self,
        monkeypatch,
    ):
        enriched = pd.DataFrame(
            {
                "regulon": ["TF1_regulon"],
                "motif": ["M1"],
                "auc": np.asarray([0.5], dtype=np.float32),
                "nes": np.asarray([3.5], dtype=np.float32),
            }
        )
        annotations = pd.DataFrame(
            {
                "motif": ["M0", "M1", "M2", "M3"],
                "TF": [None, "TF1", np.nan, pd.NA],
            }
        )

        def fake_standard_rows(
            enriched_regulons,
            enriched_motifs,
            enriched_auc,
            enriched_nes,
            annotation_motifs,
            annotation_tfs,
            auc_threshold,
            nes_threshold,
            case_sensitive,
        ):
            assert annotation_motifs == ["M0", "M1", "M2", "M3"]
            assert annotation_tfs == ["", "TF1", "", ""]
            return (
                list(enriched_regulons),
                list(enriched_motifs),
                np.asarray([0.5], dtype=np.float32),
                np.asarray([3.5], dtype=np.float32),
                ["TF1"],
                ["TF1"],
            )

        monkeypatch.setattr(
            cistarget,
            "_motif_annotation_prune_standard_rows_f32",
            fake_standard_rows,
        )

        out = cistarget.prune_enriched_motifs(
            enriched,
            annotations,
            auc_threshold=0.2,
            nes_threshold=1.0,
        )

        assert out[["tf", "annotation_tf"]].to_dict("records") == [
            {"tf": "TF1", "annotation_tf": "TF1"}
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

    def test_prune_regulons_rank_recovery_uses_rust_kernel(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF1_regulon", "motif": "M2", "auc": 0.31},
                {"regulon": "TF2_regulon", "motif": "M3", "auc": 0.32},
            ]
        )
        annotations = pd.DataFrame(
            {"motif": ["M1", "M2", "M3"], "TF": ["TF1", "TF1", "TF2"]}
        )
        base_rankings = pd.DataFrame(
            [
                [0, 4, 1, 5, 2],
                [5, 0, 4, 1, 2],
                [3, 4, 0, 1, 2],
            ],
            index=["M1", "M2", "M3"],
            columns=["g0", "g1", "g2", "g3", "x"],
        )
        candidates = [
            ("TF1_regulon", ["g0", "g1", "g2", "g3"]),
            ("TF2_regulon", ["g2", "missing"]),
        ]

        cases = (
            (base_rankings, "cistarget_prune_regulon_targets_i64"),
            (base_rankings.astype(float) + 0.25, "cistarget_prune_regulon_targets_f64"),
        )
        for rankings, backend_symbol in cases:
            assert cistarget._prune_regulons_backend_symbols(rankings) == [
                backend_symbol
            ]
            pruned = cistarget.prune_regulons(
                enriched,
                candidates,
                annotations,
                rankings=rankings,
                top_frac=0.4,
                min_genes=1,
            )
            assert pruned == {
                "TF1_regulon": ["g0", "g1", "g2", "g3"],
                "TF2_regulon": ["g2"],
            }

    def test_projected_prune_regulons_matches_full_rankings(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
            ]
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        full_rankings = pd.DataFrame(
            [
                [0, 1, 7, 8, 9, 2, 3, 4, 5, 6],
                [7, 8, 0, 1, 9, 2, 3, 4, 5, 6],
            ],
            index=["M1", "M2"],
            columns=[f"g{i}" for i in range(10)],
            dtype=np.int32,
        )
        projected = full_rankings[["g0", "g1", "g2", "g3", "g4"]]
        candidates = [
            ("TF1_regulon", ["g0", "g1", "g4"]),
            ("TF2_regulon", ["g2", "g3", "g4"]),
        ]

        got = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=projected,
            top_frac=0.2,
            min_genes=1,
            rank_universe_size=full_rankings.shape[1],
        )
        expected = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=full_rankings,
            top_frac=0.2,
            min_genes=1,
        )

        assert got == expected == {
            "TF1_regulon": ["g0", "g1"],
            "TF2_regulon": ["g2", "g3"],
        }
        assert cistarget._prune_regulons_backend_symbols(
            projected,
            rank_universe_size=full_rankings.shape[1],
        ) == ["cistarget_prune_regulon_targets_projected_i32"]

    def test_array_backed_projected_enrich_and_prune_match_dataframe(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
            ]
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        full_rankings = pd.DataFrame(
            [
                [0, 1, 7, 8, 9, 2, 3, 4, 5, 6],
                [7, 8, 0, 1, 9, 2, 3, 4, 5, 6],
            ],
            index=["M1", "M2"],
            columns=[f"g{i}" for i in range(10)],
            dtype=np.int32,
        )
        projected = full_rankings[["g0", "g1", "g2", "g3", "g4"]]
        candidates = [
            ("TF1_regulon", ["g0", "g1", "g4"]),
            ("TF2_regulon", ["g2", "g3", "g4"]),
        ]

        array_enriched = cistarget._enrich_from_rank_array(
            projected.to_numpy(copy=False),
            projected.index,
            projected.columns,
            candidates,
            top_frac=0.2,
            auc_threshold=0.0,
            rank_universe_size=full_rankings.shape[1],
        )
        frame_enriched = cistarget.enrich(
            projected,
            candidates,
            top_frac=0.2,
            auc_threshold=0.0,
            rank_universe_size=full_rankings.shape[1],
        )
        pd.testing.assert_frame_equal(array_enriched, frame_enriched)

        pruned_motifs = cistarget.prune_enriched_motifs(
            enriched,
            annotations,
        )
        array_pruned = cistarget._prune_regulons_from_pruned_motifs(
            pruned_motifs,
            candidates,
            ranking_values=projected.to_numpy(copy=False),
            motif_names=projected.index,
            gene_names=projected.columns,
            top_frac=0.2,
            min_genes=1,
            rank_universe_size=full_rankings.shape[1],
        )
        frame_pruned = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=projected,
            top_frac=0.2,
            min_genes=1,
            rank_universe_size=full_rankings.shape[1],
        )

        assert array_pruned == frame_pruned == {
            "TF1_regulon": ["g0", "g1"],
            "TF2_regulon": ["g2", "g3"],
        }
        assert cistarget._prune_regulons_backend_symbols_for_values(
            projected.to_numpy(copy=False),
            rank_universe_size=full_rankings.shape[1],
        ) == ["cistarget_prune_regulon_targets_projected_i32"]

    def test_prune_regulons_without_rankings_keeps_supported_candidates_in_rust(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF1_regulon", "motif": "M1b", "auc": 0.32},
                {"regulon": "TF3_regulon", "motif": "M3", "auc": 0.33},
            ]
        )
        annotations = pd.DataFrame(
            {
                "motif": ["M1", "M1b", "M2", "M3"],
                "TF": ["TF1", "TF1", "TF2", "TF3"],
            }
        )
        candidates = [
            ("TF1_regulon", ["g0", "g1", "g2"]),
            ("TF2_regulon", ["g3", "g4"]),
            ("TF3_regulon", ["drop_me"]),
        ]

        pruned = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=None,
            min_genes=2,
        )

        assert list(pruned) == ["TF2_regulon", "TF1_regulon"]
        assert pruned == {
            "TF2_regulon": ["g3", "g4"],
            "TF1_regulon": ["g0", "g1", "g2"],
        }
        assert cistarget._prune_regulons_backend_symbols(None) == [
            "cistarget_prune_regulon_targets_unranked"
        ]

    def test_prune_regulons_deduplicates_candidate_targets_in_rust(self):
        enriched = pd.DataFrame(
            [{"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M1"], "TF": ["TF1"]})
        candidates = [("TF1_regulon", ["g0", "g0", "g1", "g1"])]

        unranked = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=None,
            min_genes=2,
        )

        rankings = pd.DataFrame(
            [[0, 1, 2]],
            index=["M1"],
            columns=["g0", "g1", "g2"],
        )
        ranked = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=rankings,
            top_frac=1.0,
            min_genes=2,
        )

        assert unranked == {"TF1_regulon": ["g0", "g1"]}
        assert ranked == {"TF1_regulon": ["g0", "g1"]}

    def test_prune_regulons_duplicate_candidate_names_keep_last_value(self):
        enriched = pd.DataFrame(
            [{"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M1"], "TF": ["TF1"]})
        candidates = [
            ("TF1_regulon", ["old"]),
            ("TF1_regulon", ["g0", "g1"]),
        ]

        pruned = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=None,
            min_genes=2,
        )

        assert pruned == {"TF1_regulon": ["g0", "g1"]}

    def test_prune_regulons_without_rankings_uses_rust_helper(self, monkeypatch):
        enriched = pd.DataFrame(
            [{"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M1"], "TF": ["TF1"]})
        calls = []

        def fake_unranked(candidate_names, candidate_genes, pruned_regulons, min_genes):
            calls.append((candidate_names, candidate_genes, pruned_regulons, min_genes))
            return ["TF1_regulon"], [["g0", "g1"]]

        monkeypatch.setattr(cistarget, "_prune_regulon_targets_unranked", fake_unranked)

        pruned = cistarget.prune_regulons(
            enriched,
            [("TF1_regulon", ["g0", "g0", "g1"])],
            annotations,
            rankings=None,
            min_genes=2,
        )

        assert pruned == {"TF1_regulon": ["g0", "g1"]}
        assert calls == [
            (
                ["TF1_regulon"],
                [["g0", "g0", "g1"]],
                ["TF1_regulon"],
                2,
            )
        ]

    def test_prune_regulons_accepts_strided_rankings_without_python_contiguity_copy(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
            ]
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        base = np.array(
            [
                [0.0, 1.0, 4.0, 5.0],
                [4.0, 5.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        values = np.asfortranarray(base)
        rankings = pd.DataFrame(
            values,
            index=["M1", "M2"],
            columns=["g0", "g1", "g2", "g3"],
            copy=False,
        )
        rankings_arg, _ = cistarget._prune_rankings_kernel_arg(rankings)

        assert np.shares_memory(rankings_arg, values)
        assert rankings_arg.flags.f_contiguous
        assert not rankings_arg.flags.c_contiguous

        candidates = [
            ("TF1_regulon", ["g0", "g1", "g2"]),
            ("TF2_regulon", ["g1", "g2", "g3"]),
        ]
        got = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=rankings,
            top_frac=0.5,
            min_genes=1,
        )
        assert got == {
            "TF1_regulon": ["g0", "g1"],
            "TF2_regulon": ["g2", "g3"],
        }

    def test_prune_regulons_uses_float32_rankings_without_upcast_copy(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
            ]
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        base = np.array(
            [
                [0.0, 1.0, 4.0, 5.0],
                [4.0, 5.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        values = np.asfortranarray(base)
        rankings = pd.DataFrame(
            values,
            index=["M1", "M2"],
            columns=["g0", "g1", "g2", "g3"],
            copy=False,
        )
        rankings_arg, kernel = cistarget._prune_rankings_kernel_arg(rankings)

        assert rankings_arg.dtype == np.float32
        assert kernel is cistarget._prune_regulon_targets_f32
        assert np.shares_memory(rankings_arg, values)
        assert rankings_arg.flags.f_contiguous
        assert not rankings_arg.flags.c_contiguous

        candidates = [
            ("TF1_regulon", ["g0", "g1", "g2"]),
            ("TF2_regulon", ["g1", "g2", "g3"]),
        ]
        got = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=rankings,
            top_frac=0.5,
            min_genes=1,
        )
        assert got == {
            "TF1_regulon": ["g0", "g1"],
            "TF2_regulon": ["g2", "g3"],
        }

    def test_prune_regulons_uses_int16_rankings_without_upcast_copy(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
            ]
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        base = np.array(
            [
                [0, 1, 4, 5],
                [4, 5, 0, 1],
            ],
            dtype=np.int16,
        )
        values = np.asfortranarray(base)
        rankings = pd.DataFrame(
            values,
            index=["M1", "M2"],
            columns=["g0", "g1", "g2", "g3"],
            copy=False,
        )
        rankings_arg, kernel = cistarget._prune_rankings_kernel_arg(rankings)

        assert rankings_arg.dtype == np.int16
        assert kernel is cistarget._prune_regulon_targets_i16
        assert np.shares_memory(rankings_arg, values)
        assert rankings_arg.flags.f_contiguous
        assert not rankings_arg.flags.c_contiguous

        candidates = [
            ("TF1_regulon", ["g0", "g1", "g2"]),
            ("TF2_regulon", ["g1", "g2", "g3"]),
        ]
        got = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=rankings,
            top_frac=0.5,
            min_genes=1,
        )
        assert got == {
            "TF1_regulon": ["g0", "g1"],
            "TF2_regulon": ["g2", "g3"],
        }

    def test_prune_regulons_uses_int64_rankings_without_upcast_copy(self):
        enriched = pd.DataFrame(
            [
                {"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30},
                {"regulon": "TF2_regulon", "motif": "M2", "auc": 0.31},
            ]
        )
        annotations = pd.DataFrame({"motif": ["M1", "M2"], "TF": ["TF1", "TF2"]})
        base = np.array(
            [
                [0, 1, 4, 5],
                [4, 5, 0, 1],
            ],
            dtype=np.int64,
        )
        values = np.asfortranarray(base)
        rankings = pd.DataFrame(
            values,
            index=["M1", "M2"],
            columns=["g0", "g1", "g2", "g3"],
            copy=False,
        )
        rankings_arg, kernel = cistarget._prune_rankings_kernel_arg(rankings)

        assert rankings_arg.dtype == np.int64
        assert kernel is cistarget._prune_regulon_targets_i64
        assert np.shares_memory(rankings_arg, values)
        assert rankings_arg.flags.f_contiguous
        assert not rankings_arg.flags.c_contiguous

        candidates = [
            ("TF1_regulon", ["g0", "g1", "g2"]),
            ("TF2_regulon", ["g1", "g2", "g3"]),
        ]
        got = cistarget.prune_regulons(
            enriched,
            candidates,
            annotations,
            rankings=rankings,
            top_frac=0.5,
            min_genes=1,
        )
        assert got == {
            "TF1_regulon": ["g0", "g1"],
            "TF2_regulon": ["g2", "g3"],
        }

    def test_prune_rankings_arg_does_not_allocate_isfinite_mask_for_integer_rankings(
        self,
        monkeypatch,
    ):
        rankings = pd.DataFrame(
            np.arange(12, dtype=np.int32).reshape(2, 6),
            index=["m0", "m1"],
            columns=[f"g{i}" for i in range(6)],
        )

        def fail_isfinite(_values):
            raise AssertionError("integer rankings do not need np.isfinite")

        monkeypatch.setattr(cistarget.np, "isfinite", fail_isfinite)

        rankings_arg, kernel = cistarget._prune_rankings_kernel_arg(rankings)

        assert rankings_arg.dtype == np.int32
        assert kernel is cistarget._prune_regulon_targets_i32

    def test_prune_rankings_arg_unsigned_small_int_promotes_without_python_range_scan(self):
        rankings = pd.DataFrame(
            np.arange(12, dtype=np.uint32).reshape(2, 6),
            index=["m0", "m1"],
            columns=[f"g{i}" for i in range(6)],
        )

        rankings_arg, kernel = cistarget._prune_rankings_kernel_arg(rankings)

        assert rankings_arg.dtype == np.int64
        assert kernel is cistarget._prune_regulon_targets_i64

    def test_prune_rankings_arg_rejects_uint64_without_python_range_scan(self):
        rankings = pd.DataFrame(
            np.arange(12, dtype=np.uint64).reshape(2, 6),
            index=["m0", "m1"],
            columns=[f"g{i}" for i in range(6)],
        )

        with pytest.raises(TypeError, match="signed int64"):
            cistarget._prune_rankings_kernel_arg(rankings)

    def test_prune_rankings_float_nan_validation_runs_in_rust(
        self,
        monkeypatch,
    ):
        enriched = pd.DataFrame(
            [{"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M1"], "TF": ["TF1"]})
        rankings = pd.DataFrame(
            [[0.0, np.nan]],
            index=["M1"],
            columns=["g0", "g1"],
            dtype=np.float64,
        )

        def fail_isfinite(_values):
            raise AssertionError("float64 pruning should validate in Rust")

        monkeypatch.setattr(cistarget.np, "isfinite", fail_isfinite)

        with pytest.raises(ValueError, match="NaN or Inf"):
            cistarget.prune_regulons(
                enriched,
                [("TF1_regulon", ["g0", "g1"])],
                annotations,
                rankings=rankings,
                top_frac=1.0,
                min_genes=1,
            )

    def test_prune_rankings_float16_nan_validation_runs_in_rust(
        self,
        monkeypatch,
    ):
        enriched = pd.DataFrame(
            [{"regulon": "TF1_regulon", "motif": "M1", "auc": 0.30}]
        )
        annotations = pd.DataFrame({"motif": ["M1"], "TF": ["TF1"]})
        rankings = pd.DataFrame(
            np.array([[0.0, np.nan]], dtype=np.float16),
            index=["M1"],
            columns=["g0", "g1"],
        )

        def fail_isfinite(_values):
            raise AssertionError("float16 pruning should validate after Rust dispatch")

        monkeypatch.setattr(cistarget.np, "isfinite", fail_isfinite)

        with pytest.raises(ValueError, match="NaN or Inf"):
            cistarget.prune_regulons(
                enriched,
                [("TF1_regulon", ["g0", "g1"])],
                annotations,
                rankings=rankings,
                top_frac=1.0,
                min_genes=1,
            )

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

    def test_all_dropped_regulons_preserve_rust_metadata(self, tiny_rankings):
        out = cistarget.enrich(
            tiny_rankings,
            [("R_missing", ["not_in_rankings"])],
            auc_threshold=0.0,
        )

        assert len(out) == 0
        assert set(out.columns) == {"regulon", "motif", "auc", "nes"}
        assert out.attrs["rust_backend"] == {
            "engine": "rust",
            "symbols": ["stage_prepare_regulon_indices_with_coverage"],
        }

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


def _reference_enrich_with_pandas_stack(
    rankings,
    regulons,
    *,
    top_frac,
    auc_threshold,
    nes_threshold,
):
    from rustscenic._stage_utils import prepare_regulon_indices
    from rustscenic._rustscenic import aucell_score as _aucell_score

    motif_names = list(rankings.index)
    gene_names = list(rankings.columns)
    scores = -rankings.values.astype(np.float32)
    reg_names, reg_gene_indices, _, _ = prepare_regulon_indices(gene_names, regulons)
    auc, _expression_max = _aucell_score(
        np.ascontiguousarray(scores),
        reg_names,
        reg_gene_indices,
        top_frac,
    )
    auc_arr = np.asarray(auc)
    auc_df = pd.DataFrame(auc_arr, index=motif_names, columns=reg_names)
    nes_arr = _reference_compute_nes(
        auc_arr,
        n_motifs=len(motif_names),
    )
    nes_df = pd.DataFrame(nes_arr, index=motif_names, columns=reg_names)
    auc_long = auc_df.stack()
    nes_long = nes_df.stack().reindex(auc_long.index)
    long = pd.DataFrame({"auc": auc_long.values, "nes": nes_long.values})
    long["motif"] = auc_long.index.get_level_values(0)
    long["regulon"] = auc_long.index.get_level_values(1)
    long = long.reset_index(drop=True)
    long = long[long["auc"] >= auc_threshold]
    if nes_threshold is not None:
        long = long[long["nes"].notna() & (long["nes"] >= nes_threshold)]
    long = long.sort_values("auc", ascending=False).reset_index(drop=True)
    return long[["regulon", "motif", "auc", "nes"]]


def _reference_compute_nes(auc_arr, *, n_motifs):
    nes = np.full_like(auc_arr, fill_value=np.nan, dtype=np.float32)
    if n_motifs < cistarget._NES_MIN_MOTIFS:
        return nes
    means = auc_arr.mean(axis=0)
    stds = auc_arr.std(axis=0, ddof=0)
    zero_var = stds < 1e-6
    safe_stds = np.where(zero_var, 1.0, stds)
    nes_full = (auc_arr - means) / safe_stds
    nes_full[:, zero_var] = np.nan
    return nes_full.astype(np.float32)


def _sort_enrichment_for_compare(df):
    return (
        df.sort_values(["regulon", "motif", "auc", "nes"], kind="mergesort")
        .reset_index(drop=True)
    )


def _reference_prune_enriched_motifs_with_pandas_merge(
    enriched,
    motif_annotations,
    *,
    motif_col=None,
    tf_col=None,
    auc_threshold=None,
    nes_threshold=None,
    case_sensitive=False,
):
    cistarget._require_columns(enriched, {"regulon", "motif", "auc"}, name="enriched")
    if enriched.empty:
        return pd.DataFrame(columns=list(enriched.columns) + ["tf", "annotation_tf"])

    ct = enriched.copy()
    if auc_threshold is not None:
        ct = ct.loc[ct["auc"] >= auc_threshold].copy()
    if nes_threshold is not None:
        ct = ct.loc[ct["nes"].notna() & (ct["nes"] >= nes_threshold)].copy()
    if ct.empty:
        return pd.DataFrame(columns=list(enriched.columns) + ["tf", "annotation_tf"])

    ann = _reference_normalise_motif_annotations(
        motif_annotations,
        motif_col=motif_col,
        tf_col=tf_col,
        case_sensitive=case_sensitive,
    )

    ct["tf"] = ct["regulon"].map(cistarget._tf_from_regulon_name)
    ct["_motif_key"] = ct["motif"].astype(str)
    ct["_tf_key"] = ct["tf"].astype(str)
    if not case_sensitive:
        ct["_tf_key"] = ct["_tf_key"].str.lower()

    out = ct.merge(
        ann,
        on=["_motif_key", "_tf_key"],
        how="inner",
        sort=False,
    )
    out = out.drop(columns=["_motif_key", "_tf_key"])
    return out.reset_index(drop=True)


def _reference_normalise_motif_annotations(
    motif_annotations,
    *,
    motif_col,
    tf_col,
    case_sensitive,
):
    motif_col = motif_col or cistarget._find_annotation_column(
        motif_annotations,
        ["motif", "motifs", "motif_id", "motifid", "features", "#motif_id"],
        role="motif",
    )
    tf_col = tf_col or cistarget._find_annotation_column(
        motif_annotations,
        [
            "tf", "TF", "transcription_factor", "gene_name", "gene",
            "symbol", "tf_name", "factor",
        ],
        role="TF",
    )
    rows = []
    for rec in motif_annotations[[motif_col, tf_col]].itertuples(index=False):
        if pd.isna(rec[1]):
            continue
        motif = str(rec[0])
        text = str(rec[1])
        for sep in (";", ",", "|"):
            text = text.replace(sep, "/")
        for tf in (part.strip() for part in text.split("/") if part.strip()):
            key = tf if case_sensitive else tf.lower()
            rows.append((motif, key, tf))
    return (
        pd.DataFrame(rows, columns=["_motif_key", "_tf_key", "annotation_tf"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("nes_threshold", [None, 0.25])
def test_enrich_matches_previous_pandas_stack_reference(nes_threshold):
    """The memory-lean long-form assembly must preserve cistarget rows."""
    rng = np.random.default_rng(41)
    n_motifs, n_genes = 80, 140
    rankings = np.empty((n_motifs, n_genes), dtype=np.int32)
    for motif_i in range(n_motifs):
        perm = rng.permutation(n_genes)
        rankings[motif_i, perm] = np.arange(n_genes, dtype=np.int32)
    rankings = pd.DataFrame(
        rankings,
        index=[f"m{i}" for i in range(n_motifs)],
        columns=[f"g{i}" for i in range(n_genes)],
    )
    regulons = [
        ("R_A", [f"g{i}" for i in range(30)]),
        ("R_B", [f"g{i}" for i in range(20, 70, 2)]),
        ("R_C", [f"g{i}" for i in range(75, 120)]),
    ]

    got = cistarget.enrich(
        rankings,
        regulons,
        top_frac=0.08,
        auc_threshold=0.01,
        nes_threshold=nes_threshold,
    )
    expected = _reference_enrich_with_pandas_stack(
        rankings,
        regulons,
        top_frac=0.08,
        auc_threshold=0.01,
        nes_threshold=nes_threshold,
    )

    pd.testing.assert_frame_equal(
        _sort_enrichment_for_compare(got),
        _sort_enrichment_for_compare(expected),
        check_dtype=False,
        atol=1e-7,
        rtol=1e-7,
    )


def test_enrich_rejects_float_rankings():
    rankings = pd.DataFrame(
        np.arange(12, dtype=np.float32).reshape(2, 6) + 0.5,
        index=["m0", "m1"],
        columns=[f"g{i}" for i in range(6)],
    )
    with pytest.raises(TypeError, match="integer rank"):
        cistarget.enrich(rankings, [("R_A", ["g0", "g1"])], auc_threshold=0.0)


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
