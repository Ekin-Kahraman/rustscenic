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
    # AAA-1: all 3 fragments hit a peak
    assert abs(scores["AAA-1"] - 1.0) < 1e-4
    # BBB-1: 1 of 2 hits
    assert abs(scores["BBB-1"] - 0.5) < 1e-4


def test_frip_normalises_ucsc_vs_ensembl_chroms():
    """fragments in chr1 convention, peaks BED in 1 convention — must still
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
    assert set(peaks.columns) >= {"chrom", "start", "end", "name"}
    assert not peaks.empty
    # Recovery tolerance: peak_half_width=250 by default → peaks 501 bp wide
    hits_10k = ((peaks["start"] < 10_500) & (peaks["end"] > 9_500)).any()
    hits_50k = ((peaks["start"] < 50_500) & (peaks["end"] > 49_500)).any()
    assert hits_10k, f"peak near 10_000 not recovered: {peaks}"
    assert hits_50k, f"peak near 50_000 not recovered: {peaks}"


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
        content = open(out_path).read().strip().splitlines()
        assert len(content) == len(peaks)
        # Each BED line is 4 tab-separated fields
        assert all(len(line.split("\t")) == 4 for line in content)


def test_call_peaks_wrong_cluster_length_raises():
    frag_path, td = _write_fragments([
        "chr1\t100\t200\tAAA-1\t1",
        "chr1\t200\t300\tBBB-1\t1",
    ])
    with td:
        with pytest.raises(RuntimeError, match="cluster_per_barcode has length"):
            rustscenic.preproc.call_peaks(
                frag_path,
                cluster_per_barcode=[0, 0, 0],  # 3, but only 2 barcodes
                n_clusters=1,
            )
