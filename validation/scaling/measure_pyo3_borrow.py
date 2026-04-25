"""Measure peak RSS savings from the PyO3 input-borrow change.

Before this PR: `grn_infer` and `aucell_score` did
`arr.as_standard_layout().iter().copied().collect()` — always allocates
a duplicate of the input matrix on the Rust side.

After: borrow the numpy buffer directly when it's C-contiguous (which
the Python wrapper guarantees), with a single-copy fallback only for
non-standard-layout inputs.

Expected impact on peak RSS at 100k cells × 30k genes (12 GB f32 input):
roughly a 12 GB reduction in instantaneous RSS during the
allow_threads window.
"""
from __future__ import annotations

import os
import resource
import sys
import time

import numpy as np
import pandas as pd


def main() -> int:
    rng = np.random.default_rng(0)
    # Sized so the input matrix is meaningful but the run finishes in
    # minutes on a laptop. 30k cells × 5k genes = 600 MB at f32 — large
    # enough that a duplicate copy shows up in RSS.
    n_cells, n_genes = 30_000, 5_000
    print(f"building synthetic {n_cells} × {n_genes} f32 matrix...")
    X = rng.lognormal(mean=0.5, sigma=0.7, size=(n_cells, n_genes)).astype(np.float32)
    X = np.ascontiguousarray(X)
    gene_names = [f"G{i:05d}" for i in range(n_genes)]
    tfs = [f"G{i:05d}" for i in range(20)]

    import rustscenic.grn as grn

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.monotonic()
    result = grn.infer(
        (X, gene_names),
        tf_names=tfs,
        n_estimators=30,
        seed=0,
        verbose=False,
    )
    elapsed = time.monotonic() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    if sys.platform == "darwin":
        rss_after_mb = rss_after / 1024 / 1024
    else:
        rss_after_mb = rss_after / 1024

    input_mb = X.nbytes / 1024 / 1024

    print(f"  input matrix: {input_mb:.0f} MB f32")
    print(f"  edges emitted: {len(result):,}")
    print(f"  GRN wall-time: {elapsed:.1f} s")
    print(f"  process peak RSS after run: {rss_after_mb:.0f} MB")
    print(f"  ratio peak / input size: {rss_after_mb / input_mb:.2f}×")
    print()
    print(f"  Pre-borrow expected: ratio ~3×–4× (numpy + duplicate copy + GBM scratch)")
    print(f"  Post-borrow expected: ratio ~2×–2.5× (numpy + GBM scratch only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
