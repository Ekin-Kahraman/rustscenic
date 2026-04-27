"""Push synthetic Gibbs scaling to 50k cells.

The shipped v0.3.2 atlas bench (`bench_gibbs_parallel_synth_atlas.py`)
proves AD-LDA quality + speedup hold up to 25k cells × K=30. This script
runs the same kernel at 50k cells × K=30 to extend the proof, focused
on the parallel path only (n_threads=8, since serial would take ~12 min).

Reports:
- wall-clock at 8 threads
- unique argmax topics / 30
- top-10 NPMI mean (intrinsic, on training corpus)
- peak resident set size (rusage)

Setup:
  python validation/scaling/bench_gibbs_50k.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def build_synth_atac(n_cells, n_peaks, nnz_per_cell, n_planted_topics, seed):
    rng = np.random.default_rng(seed)
    block_size = n_peaks // n_planted_topics
    rows, cols = [], []
    for c in range(n_cells):
        cluster = c % n_planted_topics
        n_block = int(0.7 * nnz_per_cell)
        n_other = nnz_per_cell - n_block
        block_start = cluster * block_size
        block_peaks = rng.integers(block_start, block_start + block_size, size=n_block)
        other_peaks = rng.integers(0, n_peaks, size=n_other)
        peaks = np.unique(np.concatenate([block_peaks, other_peaks]))
        rows.extend([c] * peaks.size)
        cols.extend(peaks.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_peaks))
    return X, [f"c{i}" for i in range(n_cells)], [f"p{i}" for i in range(n_peaks)]


def main() -> int:
    import warnings
    import rustscenic.topics

    n_cells = 50_000
    n_peaks = 50_000
    nnz_per_cell = 8_000
    K = 30
    n_iters = 100
    n_threads = 8

    print(f"corpus: {n_cells:,} cells × {n_peaks:,} peaks, "
          f"~{nnz_per_cell:,} nnz/cell", flush=True)
    print(f"K={K}, n_iters={n_iters}, n_threads={n_threads}\n", flush=True)

    print("building synthetic corpus...", flush=True)
    t0 = time.monotonic()
    X, cells, peaks = build_synth_atac(
        n_cells=n_cells, n_peaks=n_peaks, nnz_per_cell=nnz_per_cell,
        n_planted_topics=K, seed=42,
    )
    print(f"  built in {time.monotonic() - t0:.1f}s, nnz={X.nnz:,}", flush=True)

    print("\nfitting AD-LDA Gibbs...", flush=True)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rustscenic.topics.fit_gibbs(
            (X, cells, peaks),
            n_topics=K, n_iters=n_iters, seed=42,
            n_threads=n_threads, verbose=False,
        )
    wall = time.monotonic() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KB. Normalise to GB on both.
    if sys.platform == "darwin":
        rss_gb = rss_after / (1024 ** 3)
    else:
        rss_gb = rss_after / (1024 ** 2)

    unique = int(np.unique(r.cell_topic.values.argmax(axis=1)).size)
    npmi = rustscenic.topics.coherence_npmi(r, (X, cells, peaks), top_n=10)
    npmi_mean = float(np.nanmean(npmi))
    npmi_median = float(np.nanmedian(npmi))

    print(f"  wall-clock:      {wall:.1f}s  ({wall/60:.1f} min)", flush=True)
    print(f"  unique topics:   {unique}/{K}", flush=True)
    print(f"  NPMI mean:       {npmi_mean:+.4f}", flush=True)
    print(f"  NPMI median:     {npmi_median:+.4f}", flush=True)
    print(f"  peak RSS:        {rss_gb:.2f} GB", flush=True)

    record = {
        "n_cells": n_cells,
        "n_peaks": n_peaks,
        "nnz_per_cell": nnz_per_cell,
        "K": K,
        "n_iters": n_iters,
        "n_threads": n_threads,
        "wall_s": round(wall, 1),
        "unique_argmax_topics": unique,
        "npmi_mean": npmi_mean,
        "npmi_median": npmi_median,
        "peak_rss_gb": round(rss_gb, 2),
    }
    out = Path(__file__).parent / "gibbs_50k.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
