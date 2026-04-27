"""Wall-clock + quality benchmark: serial vs parallel collapsed-Gibbs LDA.

Validates the AD-LDA path (Newman et al. 2009) introduced for atlas-scale
K ≥ 30 runs. Same corpus as docs/topic-collapse.md (real PBMC Multiome
ATAC, 1500 cells × 98k peaks), so wall-clock here is directly comparable
to the published 191s / 22-of-30-topics serial baseline.

Reports per-thread-count:
- wall-clock
- unique argmax topics (sanity for AD-LDA quality preservation)
- intrinsic top-10 NPMI mean (quality regression check)

Setup:
  python validation/scaling/bench_gibbs_parallel.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent / "real_multiome"


def main() -> int:
    import warnings
    import anndata
    import rustscenic.topics

    h5_path = HERE / "out_full_atac" / "atac_cells_by_peaks.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atac = anndata.read_h5ad(h5_path)
        per_cell = np.asarray(atac.X.sum(axis=1)).ravel()
        keep = np.argsort(per_cell)[::-1][:1500]
        atac = atac[keep].copy()

    print(f"corpus: {atac.shape}, nnz={atac.X.nnz:,}", flush=True)
    K = 30
    print(f"K={K}, n_iters=200, alpha=0.1, eta=0.01\n", flush=True)
    print(f"{'n_threads':>10s} | {'wall-clock':>11s} | {'unique':>10s} | {'NPMI mean':>10s} | speedup", flush=True)
    print("-" * 75, flush=True)

    results: dict = {"corpus_shape": list(atac.shape), "K": K, "runs": []}
    serial_wall: float | None = None

    for nt in (1, 2, 4, 8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.monotonic()
            r = rustscenic.topics.fit_gibbs(
                atac, n_topics=K, n_iters=200, seed=42, n_threads=nt, verbose=False,
            )
            wall = time.monotonic() - t0
        unique = int(np.unique(r.cell_topic.values.argmax(axis=1)).size)
        npmi = rustscenic.topics.coherence_npmi(r, atac, top_n=10)
        npmi_mean = float(np.nanmean(npmi))

        if serial_wall is None:
            serial_wall = wall
            speedup_str = "1.00× (ref)"
        else:
            speedup_str = f"{serial_wall / wall:.2f}×"
        print(
            f"{nt:>10d} | {wall:>9.1f}s  | {unique:>3d}/{K:<3d}    | {npmi_mean:+.4f}    | {speedup_str}",
            flush=True,
        )
        results["runs"].append({
            "n_threads": nt,
            "wall_s": round(wall, 1),
            "unique_argmax_topics": unique,
            "npmi_mean": npmi_mean,
            "npmi_median": float(np.nanmedian(npmi)),
        })

    save = HERE.parent / "scaling" / "gibbs_parallel.json"
    save.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {save}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
