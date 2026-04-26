"""GRN scaling curve on real cellxgene data.

Multi-dataset bench (PR #44) flagged a super-linear bump 2.7k → 13.7k:
9.6× time for 3.9× compute. This script measures the curve in 5
points so we can see exactly where it breaks and quantify the win
from any fix.

Subsamples the same real dataset at multiple cell counts. Same TFs,
same n_estimators. Reports per-call wall-clock and log-log slope.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.parent / "multi_dataset"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=HERE / "gland_atlas_50k.h5ad",
        help="source h5ad; default is the local Gland Atlas fixture",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2_000, 5_000, 10_000, 20_000, 40_000],
        help="cell counts to subsample without replacement",
    )
    p.add_argument("--n-estimators", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    # By default scan a curated 50-TF list (production-realistic); pass
    # `--tfs short` to use the 6-TF check that backs PR #50's slope claim.
    p.add_argument("--tfs", choices=["short", "fifty"], default="fifty")
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "grn_scaling_curve.json",
        help="where to write JSON results",
    )
    args = p.parse_args()

    h5ad_path = args.input
    if not h5ad_path.exists():
        sys.exit(f"missing: {h5ad_path}")

    print(f"loading {h5ad_path.name}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  shape: {adata.shape}")
    print(f"  X dtype: {adata.X.dtype}")
    print(f"  X max: {float(adata.X.max() if not hasattr(adata.X, 'data') else adata.X.data.max()):.1f}")

    if args.tfs == "short":
        candidate_tfs = ["SPI1", "PAX5", "TCF7", "CEBPB", "GATA3", "FOXP3"]
    else:
        import rustscenic.data
        candidate_tfs = rustscenic.data.tfs("human")[:50]

    rng = np.random.default_rng(args.seed)
    cell_counts = args.sizes
    results = []
    import rustscenic.grn

    for n in cell_counts:
        if n > adata.n_obs:
            continue
        sel = rng.choice(adata.n_obs, n, replace=False)
        sub = adata[sel].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.monotonic()
            grn = rustscenic.grn.infer(
                sub,
                tf_names=candidate_tfs,
                n_estimators=args.n_estimators,
                seed=args.seed,
                verbose=False,
            )
            dt = time.monotonic() - t0
        # Estimate "compute units": cells × genes × tfs × n_estimators (constant TFs/n_est here).
        compute_units = n * sub.n_vars * len(set(grn["TF"].unique())) * args.n_estimators
        results.append({
            "n_cells": n,
            "n_genes": sub.n_vars,
            "n_tfs": len(set(grn["TF"].unique())),
            "wall_s": round(dt, 2),
            "edges": len(grn),
            "n_estimators": args.n_estimators,
            "compute_units": compute_units,
        })
        print(
            f"  {n:>6} cells × {sub.n_vars} genes × "
            f"{len(set(grn['TF'].unique()))} TFs:"
            f" {dt:6.1f}s  ({len(grn):,} edges)"
        )

    # Compute time-vs-cells slope (log-log)
    if len(results) >= 2:
        cells = np.array([r["n_cells"] for r in results], dtype=np.float64)
        wall = np.array([r["wall_s"] for r in results], dtype=np.float64)
        log_cells = np.log(cells)
        log_wall = np.log(wall)
        slope = np.polyfit(log_cells, log_wall, 1)[0]
        print(f"\n  log-log slope (time vs cells): {slope:.2f}")
        print(f"  perfectly linear would be ~1.0; super-linear if > 1.2")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(h5ad_path),
        "tfs": args.tfs,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "results": results,
    }
    if len(results) >= 2:
        payload["grn_slope"] = float(slope)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
