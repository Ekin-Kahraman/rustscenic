"""NPMI head-to-head: rustscenic.topics.fit (VB) vs fit_gibbs (collapsed Gibbs).

Closes the topic-quality story for v0.3.1: backs the published claim that
collapsed Gibbs delivers Mallet-class topic coherence on sparse scATAC at
K=30, where Online VB is known to collapse.

Runs on the same 1500-cell × 98k-peak PBMC Multiome ATAC slice quoted in
docs/bench-vs-references.md and docs/topic-collapse.md, so the numbers are
directly comparable.

Metric: per-topic mean pairwise NPMI over top-10 peaks, averaged over
topics, evaluated on the training corpus (intrinsic coherence, the
Mimno-Wallach 2011 protocol).

Setup:
  # rustscenic already installed
  python validation/scaling/bench_npmi_head_to_head.py
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
        # Subset to top 1500 cells by total fragments — same protocol as
        # docs/topic-collapse.md (1500 × 98k peaks, K=30).
        per_cell = np.asarray(atac.X.sum(axis=1)).ravel()
        keep = np.argsort(per_cell)[::-1][:1500]
        atac = atac[keep].copy()

    print(f"corpus: {atac.shape}, nnz={atac.X.nnz:,}", flush=True)
    K = 30
    print(f"\nK={K}, top_n=10 peaks per topic, intrinsic NPMI on training corpus", flush=True)
    print("-" * 70, flush=True)

    out: dict = {"corpus_shape": list(atac.shape), "nnz": int(atac.X.nnz), "K": K}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.monotonic()
        vb = rustscenic.topics.fit(atac, n_topics=K, n_passes=10, seed=42, verbose=False)
        vb_t = time.monotonic() - t0
    vb_unique = int(np.unique(vb.cell_topic.values.argmax(axis=1)).size)
    npmi_vb = rustscenic.topics.coherence_npmi(vb, atac, top_n=10)
    out["vb"] = {
        "wall_s": round(vb_t, 1),
        "unique_argmax_topics": vb_unique,
        "npmi_per_topic": [float(x) for x in npmi_vb],
        "npmi_mean": float(np.nanmean(npmi_vb)),
        "npmi_median": float(np.nanmedian(npmi_vb)),
    }
    print(
        f"  VB:    {vb_t:6.1f}s  unique={vb_unique:>2d}/{K}  "
        f"NPMI mean={np.nanmean(npmi_vb):+.4f}  median={np.nanmedian(npmi_vb):+.4f}",
        flush=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.monotonic()
        gb = rustscenic.topics.fit_gibbs(atac, n_topics=K, n_iters=200, seed=42, verbose=False)
        gb_t = time.monotonic() - t0
    gb_unique = int(np.unique(gb.cell_topic.values.argmax(axis=1)).size)
    npmi_gb = rustscenic.topics.coherence_npmi(gb, atac, top_n=10)
    out["gibbs"] = {
        "wall_s": round(gb_t, 1),
        "unique_argmax_topics": gb_unique,
        "npmi_per_topic": [float(x) for x in npmi_gb],
        "npmi_mean": float(np.nanmean(npmi_gb)),
        "npmi_median": float(np.nanmedian(npmi_gb)),
    }
    print(
        f"  Gibbs: {gb_t:6.1f}s  unique={gb_unique:>2d}/{K}  "
        f"NPMI mean={np.nanmean(npmi_gb):+.4f}  median={np.nanmedian(npmi_gb):+.4f}",
        flush=True,
    )

    print("\nMallet 500-iter reference (from `docs/bench-vs-references.md`):", flush=True)
    print("  Mallet:    n/a   unique=24/30  NPMI mean ~0.196 (extrinsic)", flush=True)

    delta_unique = gb_unique - vb_unique
    delta_npmi = float(np.nanmean(npmi_gb)) - float(np.nanmean(npmi_vb))
    print(
        f"\nGibbs vs VB:  +{delta_unique} unique topics, "
        f"NPMI Δ={delta_npmi:+.4f} ({'Gibbs wins' if delta_npmi > 0 else 'VB wins'})",
        flush=True,
    )

    out["delta_unique_gibbs_minus_vb"] = delta_unique
    out["delta_npmi_gibbs_minus_vb"] = delta_npmi

    save = HERE.parent / "scaling" / "npmi_head_to_head.json"
    save.write_text(json.dumps(out, indent=2))
    print(f"\nresults written to {save}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
