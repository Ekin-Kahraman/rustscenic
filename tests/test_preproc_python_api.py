"""End-to-end tests for the rustscenic.preproc Python API.

Validates that the Rust layer's peak calling + cell QC is reachable
from Python, and that results are sensible on synthetic fragment data
with known structure.
"""
from __future__ import annotations

import gzip
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import rustscenic.preproc


def _write_fragments(lines: list[str]) -> tuple[str, object]:
    """Write synthetic fragments to a .tsv.gz temp file; return (path, tmpdir).
    The tmpdir must be held by the caller until done."""
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, "fragments.tsv.gz")
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return path, td


def _write_peaks_bed(lines: list[str]) -> tuple[str, object]:
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, "peaks.bed")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path, td


# ---- insert_size_stats ----------------------------------------------------


def test_insert_size_stats_end_to_end():
    frag_path, td = _write_fragments([
        "chr1\t100\t200\tAAA-1\t1",     # 100 bp, sub-nucleosomal
        "chr1\t150\t250\tAAA-1\t1",     # 100 bp, sub
        "chr1\t500\t900\tAAA-1\t1",     # 400 bp, di-nucleosomal
        "chr1\t600\t620\tBBB-1\t1",     # 20 bp, sub
    ])
    with td:
        stats = rustscenic.preproc.qc.insert_size_stats(frag_path)
    assert isinstance(stats, pd.DataFrame)
    assert stats.attrs["rust_backend"]["symbols"] == ["preproc_insert_size_stats"]
    assert set(stats.columns) == {
        "mean", "median", "n_fragments",
        "sub_nucleosomal", "mono_nucleosomal", "di_nucleosomal",
    }
    assert stats.loc["AAA-1", "n_fragments"] == 3
    assert abs(stats.loc["AAA-1", "mean"] - 200.0) < 1e-4
    assert stats.loc["AAA-1", "sub_nucleosomal"] == 2
    assert stats.loc["AAA-1", "di_nucleosomal"] == 1


# ---- frip ----------------------------------------------------------------


def test_frip_end_to_end():
    frag_path, td_f = _write_fragments([
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t150\t250\tAAA-1\t1",
        "chr1\t500\t900\tAAA-1\t1",
        "chr1\t600\t620\tBBB-1\t1",
        "chr2\t1000\t1100\tBBB-1\t1",
    ])
    peaks_path, td_p = _write_peaks_bed([
        "chr1\t100\t300\tpeak1",
        "chr1\t500\t1000\tpeak2",
    ])
    with td_f, td_p:
        scores = rustscenic.preproc.qc.frip(frag_path, peaks_path)
    assert isinstance(scores, pd.Series)
    assert scores.attrs["rust_backend"]["symbols"] == ["preproc_frip"]
    # AAA-1: all 3 fragments hit a peak
    assert abs(scores["AAA-1"] - 1.0) < 1e-4
    # BBB-1: 1 of 2 hits
    assert abs(scores["BBB-1"] - 0.5) < 1e-4


def test_frip_normalises_ucsc_vs_ensembl_chroms():
    """fragments in chr1 convention, peaks BED in 1 convention - must still
    produce non-zero FRiP after PR #24's chrom normalisation."""
    frag_path, td_f = _write_fragments([
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t500\t900\tAAA-1\t1",
    ])
    peaks_path, td_p = _write_peaks_bed([
        "1\t100\t300\tpeak1",      # Ensembl chrom
        "1\t500\t1000\tpeak2",
    ])
    with td_f, td_p:
        scores = rustscenic.preproc.qc.frip(frag_path, peaks_path)
    assert scores["AAA-1"] > 0, \
        "UCSC/Ensembl chrom mismatch dropped every fragment (regression)"


# ---- tss_enrichment ------------------------------------------------------


def test_tss_enrichment_end_to_end():
    # Put 10 AAA-1 fragments tightly around a TSS and 10 BBB-1 fragments
    # far away; AAA should score high, BBB should score near zero.
    lines = []
    for i in range(10):
        lines.append(f"chr1\t{10_000 - 30 + i * 3}\t{10_050 + i * 3}\tAAA-1\t1")
    for i in range(10):
        lines.append(f"chr1\t{100_000 + i * 50}\t{100_050 + i * 50}\tBBB-1\t1")
    frag_path, td = _write_fragments(lines)
    tss = pd.DataFrame({"chrom": ["chr1"], "position": [10_000]})
    with td:
        scores = rustscenic.preproc.qc.tss_enrichment(frag_path, tss)
    assert scores.attrs["rust_backend"]["symbols"] == ["preproc_tss_enrichment"]
    assert scores["AAA-1"] > 5.0
    assert scores["BBB-1"] < 1.0


def test_tss_enrichment_missing_columns_raises():
    frag_path, td = _write_fragments(["chr1\t100\t200\tA-1\t1"])
    with td, pytest.raises(ValueError, match="chrom.*position"):
        rustscenic.preproc.qc.tss_enrichment(
            frag_path,
            pd.DataFrame({"chromosome": ["chr1"], "position": [150]}),
        )


# ---- call_peaks ----------------------------------------------------------


def test_call_peaks_recovers_synthetic_peaks():
    """Two dense 40-fragment clusters at chr1:10_000 and chr1:50_000 on
    top of diffuse noise. The peak caller must produce at least one
    peak near each."""
    lines = []
    for i in range(40):
        lines.append(f"chr1\t{10_000 + i * 2}\t{10_080 + i * 2}\tAAA-1\t1")
    for i in range(40):
        lines.append(f"chr1\t{50_000 + i * 2}\t{50_080 + i * 2}\tAAA-1\t1")
    # diffuse noise
    for i in range(50):
        start = i * 1973 + 100
        lines.append(f"chr1\t{start}\t{start + 80}\tAAA-1\t1")

    frag_path, td = _write_fragments(lines)
    with td:
        peaks = rustscenic.preproc.call_peaks(
            frag_path, cluster_per_barcode=[0], n_clusters=1,
        )
    assert isinstance(peaks, pd.DataFrame)
    assert peaks.attrs["rust_backend"]["symbols"] == ["preproc_call_peaks"]
    assert set(peaks.columns) >= {"chrom", "start", "end", "name"}
    assert not peaks.empty
    # Recovery tolerance: peak_half_width=250 by default → peaks 501 bp wide
    hits_10k = ((peaks["start"] < 10_500) & (peaks["end"] > 9_500)).any()
    hits_50k = ((peaks["start"] < 50_500) & (peaks["end"] > 49_500)).any()
    assert hits_10k, f"peak near 10_000 not recovered: {peaks}"
    assert hits_50k, f"peak near 50_000 not recovered: {peaks}"


def test_call_peaks_derives_n_clusters_in_rust_when_omitted():
    lines = [f"chr1\t{1000 + i * 5}\t{1080 + i * 5}\tAAA-1\t1" for i in range(30)]
    frag_path, td = _write_fragments(lines)
    with td:
        peaks = rustscenic.preproc.call_peaks(
            frag_path,
            cluster_per_barcode=np.array([1], dtype=np.uint32),
        )
    assert not peaks.empty


def test_call_peaks_writes_bed_when_requested():
    lines = [f"chr1\t{1000 + i * 5}\t{1080 + i * 5}\tAAA-1\t1" for i in range(30)]
    frag_path, td_f = _write_fragments(lines)
    with td_f, tempfile.TemporaryDirectory() as td_out:
        out_path = os.path.join(td_out, "peaks.bed")
        peaks = rustscenic.preproc.call_peaks(
            frag_path,
            cluster_per_barcode=[0], n_clusters=1,
            output_bed=out_path,
        )
        assert os.path.exists(out_path)
        with open(out_path) as fh:
            content = fh.read().strip().splitlines()
        assert len(content) == len(peaks)
        # Each BED line is 4 tab-separated fields
        assert all(len(line.split("\t")) == 4 for line in content)


def test_call_peaks_wrong_cluster_length_raises():
    frag_path, td = _write_fragments([
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t200\t300\tBBB-1\t1",
    ])
    with td, pytest.raises(RuntimeError, match="cluster_per_barcode has length"):
        rustscenic.preproc.call_peaks(
            frag_path,
            cluster_per_barcode=[0, 0, 0],  # 3, but only 2 barcodes
            n_clusters=1,
        )


def test_call_peaks_aligns_series_by_barcode_index(tmp_path, monkeypatch):
    captured = {}

    def fake_call_peaks(
        fragments_path,
        cluster_per_barcode,
        n_clusters,
        window_size,
        min_fragments_per_window,
        quantile_threshold,
        max_gap,
        peak_half_width,
        cluster_barcodes=None,
    ):
        captured["clusters"] = cluster_per_barcode
        captured["cluster_barcodes"] = cluster_barcodes
        captured["n_clusters"] = n_clusters
        return [], np.array([], dtype=np.uint32), np.array([], dtype=np.uint32), []

    monkeypatch.setattr(rustscenic.preproc, "_call_peaks", fake_call_peaks)
    monkeypatch.setattr(
        rustscenic.preproc,
        "_insert_size_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Series alignment should happen inside preproc_call_peaks")
        ),
    )

    frag_path = tmp_path / "fragments.tsv.gz"
    frag_path.write_text("")
    clusters = pd.Series([1, 0], index=["BBB-1", "AAA-1"])

    out = rustscenic.preproc.call_peaks(frag_path, clusters)

    assert out.empty
    assert isinstance(captured["clusters"], np.ndarray)
    assert captured["clusters"].dtype == np.int64
    assert captured["clusters"].tolist() == [1, 0]
    assert captured["cluster_barcodes"] == ["BBB-1", "AAA-1"]
    assert captured["n_clusters"] is None


def test_call_peaks_passes_uint32_array_without_list_expansion_or_python_cluster_scan(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_call_peaks(
        fragments_path,
        cluster_per_barcode,
        n_clusters,
        window_size,
        min_fragments_per_window,
        quantile_threshold,
        max_gap,
        peak_half_width,
        cluster_barcodes=None,
    ):
        captured["clusters"] = cluster_per_barcode
        captured["cluster_barcodes"] = cluster_barcodes
        captured["n_clusters"] = n_clusters
        return [], np.array([], dtype=np.uint32), np.array([], dtype=np.uint32), []

    monkeypatch.setattr(rustscenic.preproc, "_call_peaks", fake_call_peaks)

    def fail_max(*_args, **_kwargs):
        raise AssertionError("n_clusters derivation should happen in Rust")

    monkeypatch.setattr(rustscenic.preproc.np, "max", fail_max)

    frag_path = tmp_path / "fragments.tsv.gz"
    frag_path.write_text("")
    clusters = np.array([0, 1], dtype=np.uint32)

    out = rustscenic.preproc.call_peaks(frag_path, clusters)

    assert out.empty
    assert captured["clusters"].dtype == np.uint32
    assert np.shares_memory(captured["clusters"], clusters)
    assert captured["clusters"].tolist() == [0, 1]
    assert captured["cluster_barcodes"] is None
    assert captured["n_clusters"] is None


def test_call_peaks_normalises_negative_cluster_labels_in_rust():
    lines = [
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t200\t300\tBBB-1\t1",
    ]
    frag_path, td = _write_fragments(lines)
    with td:
        peaks = rustscenic.preproc.call_peaks(
            frag_path,
            cluster_per_barcode=np.array([0, -1], dtype=np.int64),
            n_clusters=1,
        )
    assert isinstance(peaks, pd.DataFrame)


def test_call_peaks_series_aligns_by_barcode_index_inside_rust():
    lines = []
    for i in range(40):
        lines.append(f"chr1\t{10_000 + i * 2}\t{10_080 + i * 2}\tAAA-1\t1")
    for i in range(40):
        lines.append(f"chr1\t{50_000 + i * 2}\t{50_080 + i * 2}\tBBB-1\t1")

    frag_path, td = _write_fragments(lines)
    clusters = pd.Series([0, -1], index=["BBB-1", "AAA-1"])
    with td:
        peaks = rustscenic.preproc.call_peaks(
            frag_path,
            clusters,
            n_clusters=1,
        )

    hits_10k = ((peaks["start"] < 10_500) & (peaks["end"] > 9_500)).any()
    hits_50k = ((peaks["start"] < 50_500) & (peaks["end"] > 49_500)).any()
    assert hits_50k, f"reversed Series index was not aligned in Rust: {peaks}"
    assert not hits_10k, f"unassigned AAA-1 fragments should not produce a peak: {peaks}"


def test_call_peaks_series_missing_barcode_raises():
    frag_path, td = _write_fragments([
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t200\t300\tBBB-1\t1",
    ])
    with pytest.raises(ValueError, match="missing fragment barcodes"):
        with td:
            rustscenic.preproc.call_peaks(
                frag_path,
                pd.Series([0], index=["AAA-1"]),
            )


# ---- TSS chrom-normalisation regression -----------------------------------


def test_tss_enrichment_chrom_convention_mismatch_no_longer_silent_zero():
    """Fragments in `chr1` + TSS in `1` (Ensembl) must still register -
    exact same class as the Fuaad cellxgene silent-zero, one layer
    down in the Rust preproc core."""
    # 10 fragments around position 10_000 on chr1 (UCSC-style)
    frag_lines = [
        f"chr1\t{10_000 + i * 3}\t{10_080 + i * 3}\tAAA-1\t1"
        for i in range(10)
    ]
    frag_path, td = _write_fragments(frag_lines)
    # TSS in Ensembl chrom convention - should still match
    tss = pd.DataFrame({"chrom": ["1"], "position": [10_000]})
    with td:
        scores = rustscenic.preproc.qc.tss_enrichment(frag_path, tss)
    assert scores["AAA-1"] > 0, (
        "chrom convention mismatch silently dropped every TSS "
        "(regression of the normalise_chrom fix)"
    )


def test_fragments_to_matrix_warns_on_6column_strand_bed():
    """6-col strand BED (chrom start end name score strand) passed in place
    of a 5-col cellranger fragments.tsv would silently treat peak-names
    as cell barcodes. Detect via near-one-per-row 'barcodes' and warn."""
    # Fake a 6-col BED where col 4 is a per-row feature ID (not a barcode)
    import warnings
    lines = [f"chr1\t{1000 + i * 100}\t{1080 + i * 100}\tfeature_{i}\t1" for i in range(200)]
    frag_path, td_f = _write_fragments(lines)
    # Need a peaks file - any one, even empty
    peaks_path, td_p = _write_peaks_bed(["chr1\t1000\t20000\tpeak"])
    with td_f, td_p, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adata = rustscenic.preproc.fragments_to_matrix(frag_path, peaks_path)
    assert adata.uns["rust_backend"]["symbols"] == ["preproc_fragments_to_matrix"]
    msgs = [str(w.message) for w in caught]
    assert any("one-per-row" in m or "strand BED" in m for m in msgs), (
        f"one-per-row barcode pattern should have warned, got: {msgs}"
    )


def test_fragments_to_matrix_warns_on_unfiltered_barcodes():
    """Raw 10x fragments.tsv has 100k+ barcodes (every observed). Warn
    so users know to subset to cell-called barcodes before downstream."""
    import warnings
    # 110_000 barcodes, 1 fragment each - looks like a raw fragments file
    lines = [f"chr1\t{i*100}\t{i*100 + 100}\tBC{i:08d}\t1" for i in range(110_000)]
    frag_path, td = _write_fragments(lines)
    peaks_path, td_p = _write_peaks_bed(["chr1\t0\t20000000\tpeak"])
    with td, td_p, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = rustscenic.preproc.fragments_to_matrix(frag_path, peaks_path)
    msgs = [str(w.message) for w in caught]
    assert any("cell-called barcodes" in m for m in msgs), msgs


def test_fragments_to_matrix_cell_barcode_filter_builds_only_requested_rows():
    frag_path, td = _write_fragments([
        "chr1\t100\t150\tAAA-1\t1",
        "chr1\t120\t170\tBBB-1\t7",
        "chr1\t520\t580\tBBB-1\t2",
        "chr1\t520\t580\tEMPTY-1\t1",
    ])
    peaks_path, td_p = _write_peaks_bed([
        "chr1\t90\t200\tpeak1",
        "chr1\t500\t600\tpeak2",
    ])

    with td, td_p:
        adata = rustscenic.preproc.fragments_to_matrix(
            frag_path,
            peaks_path,
            cell_barcodes=["MISSING-1", "BBB-1"],
        )

    assert list(adata.obs_names) == ["BBB-1"]
    assert list(adata.var_names) == ["peak1", "peak2"]
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[1, 1]], dtype=np.uint32))
    assert adata.obs.loc["BBB-1", "fragments_per_cell"] == 2
    assert adata.obs.loc["BBB-1", "total_counts"] == 9
    assert adata.uns["cell_barcode_filter"] == {"requested": 2, "matched": 1}
