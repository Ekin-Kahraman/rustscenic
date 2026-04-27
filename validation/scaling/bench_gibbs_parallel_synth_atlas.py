"""Atlas-scale synthetic-corpus benchmark for the parallel AD-LDA Gibbs path.

Real cell-called PBMC ATAC tops out at ~2.6k cells. To prove the AD-LDA
speedup persists at atlas scale, generate a synthetic sparse binary
peak corpus with the same density as real ATAC (~10k nnz / cell at 100k
peaks), at increasing cell counts.

Reports per (n_cells, n_threads):
- wall-clock
- unique argmax topics (sanity that AD-LDA quality holds at scale)

Setup:
  python validation/scaling/bench_gibbs_parallel_synth_atlas.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def build_synth_atac(
    n_cells: int,
    n_peaks: int,
    nnz_per_cell: int,
    n_planted_topics: int,
    seed: int,
):
    """Plant `n_planted_topics` cell-clusters: cells in cluster k draw peaks
    from peak-block k with probability 0.7 and from anywhere with 0.3."""
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
    cells = [f"c{i}" for i in range(n_cells)]
    peaks = [f"p{i}" for i in range(n_peaks)]
    return X, cells, peaks


def main() -> int:
    import warnings
    import rustscenic.topics

    K = 30
    n_iters = 100  # halve from 200 to cap total runtime
    n_peaks = 50_000
    nnz_per_cell = 8_000

    sizes = [3_000, 10_000, 25_000]
    threads = [1, 4, 8]

    print(f"K={K}, n_iters={n_iters}, n_peaks={n_peaks}, nnz/cell={nnz_per_cell}", flush=True)
    print(f"{'n_cells':>8s} | {'threads':>7s} | {'wall-clock':>11s} | {'unique':>9s} | speedup", flush=True)
    print("-" * 65, flush=True)

    results: dict = {"K": K, "n_iters": n_iters, "n_peaks": n_peaks, "runs": []}

    for n_cells in sizes:
        X, cells, peaks = build_synth_atac(
            n_cells=n_cells, n_peaks=n_peaks, nnz_per_cell=nnz_per_cell,
            n_planted_topics=K, seed=42,
        )
        serial_wall: float | None = None
        for nt in threads:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.monotonic()
                r = rustscenic.topics.fit_gibbs(
                    (X, cells, peaks),
                    n_topics=K, n_iters=n_iters, seed=42, n_threads=nt,
                    verbose=False,
                )
                wall = time.monotonic() - t0
            unique = int(np.unique(r.cell_topic.values.argmax(axis=1)).size)
            if nt == 1:
                serial_wall = wall
                speedup_str = "1.00× (ref)"
            else:
                speedup_str = f"{serial_wall / wall:.2f}×"
            print(
                f"{n_cells:>8d} | {nt:>7d} | {wall:>9.1f}s  | {unique:>3d}/{K:<3d}  | {speedup_str}",
                flush=True,
            )
            results["runs"].append({
                "n_cells": n_cells,
                "n_threads": nt,
                "wall_s": round(wall, 1),
                "unique_argmax_topics": unique,
            })
        print()

    save = Path(__file__).parent / "gibbs_parallel_synth_atlas.json"
    save.write_text(json.dumps(results, indent=2))
    print(f"results → {save}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
