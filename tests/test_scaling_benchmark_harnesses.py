from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from validation.scaling import bench_synthetic_grn_curve as grn_curve
from validation.scaling import bench_real_rna_grn_scaling as real_rna
from validation.scaling.bench_real_rna_grn_head_to_head import (
    ProcessTreePeakRSS,
    fitted_tree_summary,
    top_k_jaccard,
)
from validation.scaling.bench_e2e_100k_synthetic import (
    _build_synth_multiome,
    _topic_normalisation_checks,
)
from validation.scaling.bench_real_atac_topics_scaling import _thread_comparisons
from validation.scaling.bench_synthetic_grn_curve import synthetic_expression


def test_synthetic_multiome_builder_constructs_bounded_csr() -> None:
    n_cells = 12
    n_peaks = 2_000
    rna, atac, gene_coords, tfs, rankings = _build_synth_multiome(
        n_cells=n_cells,
        n_genes=1_000,
        n_peaks=n_peaks,
        n_programmes=30,
        seed=42,
    )

    assert rna.shape == (n_cells, 1_000)
    assert atac.shape == (n_cells, n_peaks)
    assert sp.isspmatrix_csr(atac.X)
    assert atac.X.indices.dtype == np.int32
    assert atac.X.nnz <= n_cells * n_peaks
    assert np.all(atac.X.data == 1.0)
    assert gene_coords.shape[0] == 1_000
    assert len(tfs) == 30
    assert rankings.shape == (30, 1_000)


def test_topic_normalisation_allows_only_float32_accumulation_error() -> None:
    cell_topic = np.full((3, 30), 1.0 / 30, dtype=np.float32)
    topic_peak = np.full((30, 50_000), 1.0 / 50_000, dtype=np.float32)

    checks, cell_error, peak_error = _topic_normalisation_checks(
        cell_topic, topic_peak
    )

    assert all(checks.values())
    assert cell_error < 1e-5
    assert peak_error < 2e-3

    topic_peak[0] *= 0.99
    checks, _, peak_error = _topic_normalisation_checks(cell_topic, topic_peak)
    assert not checks["topic_peak_rows_normalised"]
    assert peak_error > 2e-3


def test_atac_thread_comparison_is_topic_label_invariant() -> None:
    baseline = {
        "method": "vb",
        "n_cells": 4,
        "threads": 4,
        "fit_wall_s": 10.0,
        "assignments": [0, 0, 1, 1],
        "top_peak_indices": [list(range(10)), list(range(10, 20))],
    }
    relabelled = {
        "method": "vb",
        "n_cells": 4,
        "threads": 8,
        "fit_wall_s": 5.0,
        "assignments": [1, 1, 0, 0],
        "top_peak_indices": [list(range(10, 20)), list(range(10))],
    }

    comparisons = _thread_comparisons([baseline, relabelled], n_cells=4)

    assert comparisons[1]["speedup_vs_baseline"] == 2.0
    assert comparisons[1]["efficiency_vs_baseline"] == 1.0
    assert comparisons[1]["assignment_ari"] == 1.0
    assert comparisons[1]["matched_top10_peak_overlap_mean"] == 1.0


def test_grn_scaling_inputs_are_nested_row_prefixes() -> None:
    small, small_genes, small_tfs = synthetic_expression(
        n_cells=20,
        n_genes=40,
        n_tfs=5,
        n_programmes=4,
        seed=777,
    )
    large, large_genes, large_tfs = synthetic_expression(
        n_cells=40,
        n_genes=40,
        n_tfs=5,
        n_programmes=4,
        seed=777,
    )

    np.testing.assert_array_equal(small, large[:20])
    assert small_genes == large_genes
    assert small_tfs == large_tfs


def test_grn_scaling_child_does_not_change_seed_with_cell_count(monkeypatch) -> None:
    seen: list[tuple[int, int]] = []

    def fake_expression(*, n_cells, n_genes, n_tfs, n_programmes, seed):
        seen.append((n_cells, seed))
        return np.ones((n_cells, n_genes), dtype=np.float32), ["TF", "G1"], ["TF"]

    def fake_infer(*_args, **_kwargs):
        result = pd.DataFrame(
            {"TF": ["TF"], "target": ["G1"], "importance": [1.0]}
        )
        result.attrs["rust_backend"] = {"engine": "rust"}
        return result

    monkeypatch.setattr(grn_curve, "synthetic_expression", fake_expression)
    monkeypatch.setattr("rustscenic.grn.infer", fake_infer)
    for n_cells in (20, 40):
        args = grn_curve.parse_args(
            [
                "--run-one",
                str(n_cells),
                "--n-genes",
                "2",
                "--n-tfs",
                "1",
                "--seed",
                "777",
            ]
        )
        grn_curve.run_one(args)

    assert seen == [(20, 777), (40, 777)]


def _write_old_10x_h5(path, cells_genes: sp.csr_matrix, genes: list[str]) -> None:
    import h5py

    gene_cell = cells_genes.T.tocsc()
    with h5py.File(path, "w") as handle:
        group = handle.create_group("mm10")
        group.create_dataset("data", data=gene_cell.data)
        group.create_dataset("indices", data=gene_cell.indices)
        group.create_dataset("indptr", data=gene_cell.indptr)
        group.create_dataset("shape", data=gene_cell.shape)
        group.create_dataset("gene_names", data=np.asarray(genes, dtype="S"))
        group.create_dataset(
            "barcodes",
            data=np.asarray([f"cell-{index}" for index in range(cells_genes.shape[0])], dtype="S"),
        )


def test_real_rna_preparation_makes_nested_normalised_prefixes(tmp_path) -> None:
    genes = ["Gata1", "Spi1", "Pax5", "GeneA", "GeneB", "GeneC", "Rpl1", "mt-Nd1"]
    values = np.asarray(
        [
            [4, 1, 0, 7, 0, 2, 1, 1],
            [2, 0, 3, 0, 8, 1, 2, 0],
            [1, 4, 1, 2, 0, 5, 0, 1],
            [3, 2, 0, 1, 4, 0, 1, 0],
            [0, 3, 2, 5, 1, 1, 0, 2],
            [5, 1, 4, 0, 2, 3, 1, 0],
            [1, 2, 0, 4, 3, 1, 0, 1],
            [2, 1, 5, 1, 0, 4, 2, 0],
            [4, 0, 2, 3, 1, 2, 0, 1],
            [1, 3, 1, 0, 5, 2, 1, 0],
            [3, 2, 4, 1, 0, 1, 0, 2],
            [2, 4, 0, 2, 3, 1, 1, 0],
        ],
        dtype=np.uint16,
    )
    source = tmp_path / "source.h5"
    sample = tmp_path / "sample.h5"
    prepared = tmp_path / "prepared.h5"
    _write_old_10x_h5(source, sp.csr_matrix(values), genes)
    _write_old_10x_h5(sample, sp.csr_matrix(values), genes)

    args = real_rna.parse_args(
        [
            "prepare",
            "--source",
            str(source),
            "--feature-sample",
            str(sample),
            "--out",
            str(prepared),
            "--prepare-hvg",
            "4",
            "--min-detected-cells",
            "1",
            "--cell-order-seed",
            "42",
        ]
    )
    assert real_rna.prepare(args) == 0

    small, small_genes, small_tfs, small_indices = real_rna.load_prepared_prefix(
        prepared, n_cells=5, n_hvg=3, n_tfs=2
    )
    large, large_genes, large_tfs, large_indices = real_rna.load_prepared_prefix(
        prepared, n_cells=10, n_hvg=3, n_tfs=2
    )

    assert small_genes == large_genes
    assert small_tfs == large_tfs
    np.testing.assert_array_equal(small_indices, large_indices[:5])
    np.testing.assert_allclose(small.toarray(), large[:5].toarray())
    assert np.isfinite(large.data).all()
    assert large.data.min() >= 0


def test_real_rna_head_to_head_metrics_use_complete_target_sums() -> None:
    left = pd.DataFrame(
        {
            "TF": ["A", "B", "A", "B"],
            "target": ["X", "X", "Y", "Y"],
            "importance": [10.0, 15.0, 12.0, 18.0],
        }
    )
    right = pd.DataFrame(
        {
            "TF": ["A", "B", "A", "B"],
            "target": ["X", "X", "Y", "Y"],
            "importance": [12.0, 16.0, 15.0, 15.0],
        }
    )

    summary = fitted_tree_summary(left)
    assert summary == {
        "target_count": 2,
        "total": 55,
        "mean": 27.5,
        "median": 27.5,
        "min": 25,
        "max": 30,
    }
    assert top_k_jaccard(left, right, k=1) == {
        "k": 1,
        "shared_tfs": 2,
        "mean": 0.5,
        "median": 0.5,
    }


def test_real_rna_head_to_head_samples_physical_memory() -> None:
    with ProcessTreePeakRSS(interval_s=0.01) as memory:
        allocation = np.ones(1_000_000, dtype=np.float64)

    assert allocation.sum() == 1_000_000
    assert memory.peak_rss_mb > 0
    assert memory.peak_pss_mb > 0
    assert memory.peak_uss_mb > 0
