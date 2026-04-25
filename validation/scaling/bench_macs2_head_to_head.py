"""Head-to-head: rustscenic.preproc.call_peaks vs MACS2 on real fragments.

Reproducible peak-calling benchmark used to back the F1 0.825 + 9.9×
speed claims in `docs/bench-vs-references.md`.

Inputs (under validation/real_multiome/, gitignored):
  fragments.tsv.gz    10x public PBMC 3k Multiome ATAC fragments
  peaks.bed           (not used here — we re-call peaks)

How to set up MACS2 (it doesn't pip-install on Python 3.10+):
  conda create -n macs2_bench python=3.9 macs2 -c bioconda -c conda-forge -y

Then run this script with both envs active in turn:
  source .../macs2_bench/bin/activate
  python <this script> --tool macs2

  source .../rustscenic_env/bin/activate
  python <this script> --tool rustscenic
"""
from __future__ import annotations

import argparse
import bisect
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent / "real_multiome"


def call_rustscenic():
    import warnings

    import numpy as np

    import rustscenic.preproc

    fragments = HERE / "fragments.tsv.gz"
    if not fragments.exists():
        sys.exit(f"missing input: {fragments}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        is_stats = rustscenic.preproc.qc.insert_size_stats(str(fragments))
        rng = np.random.default_rng(0)
        top = is_stats["n_fragments"].nlargest(3000).index.tolist()
        top_set = set(top)
        clusters = [
            int(rng.integers(0, 3)) if bc in top_set else 0xFFFFFFFF
            for bc in is_stats.index
        ]
        t0 = time.monotonic()
        peaks = rustscenic.preproc.call_peaks(
            str(fragments), cluster_per_barcode=clusters, n_clusters=3,
        )
        elapsed = time.monotonic() - t0
    print(f"rustscenic: {elapsed:.1f}s, {len(peaks):,} peaks")
    out_path = HERE / "rustscenic_peaks.bed"
    peaks.to_csv(out_path, sep="\t", header=False, index=False)
    print(f"  saved → {out_path}")
    return peaks


def call_macs2():
    """Invoke macs2 callpeak as a subprocess (must be on PATH)."""
    import shutil
    import subprocess

    if shutil.which("macs2") is None:
        sys.exit("macs2 not on PATH — install via conda first")

    fragments = HERE / "fragments.tsv.gz"
    out_dir = Path("/tmp/macs2_bench")
    out_dir.mkdir(exist_ok=True)

    t0 = time.monotonic()
    subprocess.run(
        [
            "macs2", "callpeak",
            "-t", str(fragments),
            "-f", "BED",
            "-g", "hs",
            "-n", "pbmc3k",
            "--nomodel",
            "--shift", "-75", "--extsize", "150",
            "--keep-dup", "all", "--call-summits",
            "-q", "0.01",
            "--outdir", str(out_dir),
        ],
        check=True,
    )
    elapsed = time.monotonic() - t0
    narrowpeak = out_dir / "pbmc3k_peaks.narrowPeak"
    n_peaks = sum(1 for _ in open(narrowpeak))
    print(f"macs2: {elapsed:.1f}s, {n_peaks:,} narrow peaks → {narrowpeak}")
    return narrowpeak


def overlap_fraction(a, b):
    """Fraction of `a` intervals that have any overlap with a `b` interval."""
    matched = 0
    for chrom in a["chrom"].unique():
        a_ch = a[a["chrom"] == chrom].sort_values("start").reset_index(drop=True)
        b_ch = b[b["chrom"] == chrom].sort_values("start").reset_index(drop=True)
        if b_ch.empty:
            continue
        b_starts = b_ch["start"].values
        b_ends = b_ch["end"].values
        for _, row in a_ch.iterrows():
            j = bisect.bisect_right(b_ends, int(row["start"]))
            if j < len(b_ch) and b_starts[j] < int(row["end"]):
                matched += 1
    return matched / max(len(a), 1)


def f1():
    """Compare both peak sets after both tools have run."""
    import pandas as pd

    rs = pd.read_csv(
        HERE / "rustscenic_peaks.bed", sep="\t", header=None,
        names=["chrom", "start", "end", "name"],
    )
    macs = pd.read_csv(
        Path("/tmp/macs2_bench/pbmc3k_peaks.narrowPeak"),
        sep="\t", header=None,
        names=["chrom", "start", "end", "name", "score", "strand",
               "signalValue", "pValue", "qValue", "peak"],
    )
    print(f"\nrustscenic peaks: {len(rs):,}")
    print(f"MACS2 peaks:      {len(macs):,}")

    recall = overlap_fraction(rs, macs)
    precision = overlap_fraction(macs, rs)
    f1_score = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else 0
    )
    print(f"recall:    {recall*100:.1f}%")
    print(f"precision: {precision*100:.1f}%")
    print(f"F1:        {f1_score:.3f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["rustscenic", "macs2", "f1"], default="f1")
    args = p.parse_args()
    if args.tool == "rustscenic":
        call_rustscenic()
    elif args.tool == "macs2":
        call_macs2()
    elif args.tool == "f1":
        f1()
    return 0


if __name__ == "__main__":
    sys.exit(main())
