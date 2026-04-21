"""Scaling benchmark for rustscenic.

Measures wall-time and peak memory for the GRN + AUCell stages across
cell counts from ~1k to the largest size that fits in memory on the
current machine. Output is a CSV and a log-log plot that shows whether
runtime scales linearly with cell count.

Each size is run in a fresh subprocess so peak RSS is clean per-size
rather than cumulative across the benchmark.

Usage:
    python validation/scaling/bench_scaling.py \\
        --input validation/reference/data/pbmc10k.h5ad \\
        --sizes 1000 5000 10000 30000 100000 300000 \\
        --n-estimators 300 \\
        --out validation/scaling/

For cell counts larger than the source dataset, cells are up-sampled
with replacement and a small Gaussian perturbation on normalised counts
so the expression profile is still non-degenerate. This measures
computational complexity on representative sparsity, not identical cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def peak_rss_mb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    factor = 1 << 20 if os.uname().sysname == "Darwin" else 1 << 10
    return rusage.ru_maxrss / factor


def subsample(adata, n_cells: int, seed: int = 0):
    import anndata as ad

    rng = np.random.default_rng(seed)
    n_source = adata.n_obs
    if n_cells <= n_source:
        idx = rng.choice(n_source, size=n_cells, replace=False)
        return adata[idx].copy()
    idx = rng.choice(n_source, size=n_cells, replace=True)
    sub = adata[idx].copy()
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    noise = rng.normal(0.0, 0.01, size=X.shape).astype(X.dtype)
    X = np.clip(X + noise, 0, None)
    sub = ad.AnnData(X=X, obs=sub.obs.reset_index(drop=True), var=sub.var)
    return sub


DEFAULT_TFS = [
    "SPI1", "CEBPB", "CEBPD", "IRF8", "MAFB",
    "PAX5", "EBF1", "POU2AF1", "BACH2",
    "TCF7", "LEF1", "ETS1", "GATA3", "RUNX3",
    "TBX21", "EOMES",
    "KLF4", "ZEB2", "STAT1", "IRF1", "NFKB1",
]


def run_one_size(input_path: str, n_cells: int, n_estimators: int,
                 tfs_path: str | None) -> dict:
    """Run the benchmark for a single cell count. Prints one JSON line."""
    import anndata as ad
    import scanpy as sc

    import rustscenic.aucell
    import rustscenic.grn

    adata = ad.read_h5ad(input_path)
    if adata.X.max() > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if tfs_path and Path(tfs_path).exists():
        candidates = [
            ln.strip() for ln in Path(tfs_path).read_text().splitlines()
            if ln.strip()
        ]
    else:
        candidates = DEFAULT_TFS
    tfs = [t for t in candidates if t in adata.var_names]

    sub = subsample(adata, n_cells)

    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        sub, tf_names=tfs, n_estimators=n_estimators, seed=777
    )
    t_grn = time.perf_counter() - t0

    regulons = []
    for tf in grn["TF"].unique():
        top = grn[grn["TF"] == tf].nlargest(50, "importance")["target"].tolist()
        if len(top) >= 10:
            regulons.append((f"{tf}_regulon", top))

    t0 = time.perf_counter()
    auc = rustscenic.aucell.score(sub, regulons, top_frac=0.05)
    t_auc = time.perf_counter() - t0

    rss = peak_rss_mb()

    return {
        "n_cells": n_cells,
        "n_regulons": len(regulons),
        "n_edges": len(grn),
        "t_grn_s": t_grn,
        "t_auc_s": t_auc,
        "rss_peak_mb": rss,
        "auc_rows": int(auc.shape[0]),
        "auc_cols": int(auc.shape[1]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1000, 5000, 10000, 30000, 100000])
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tfs", type=Path, default=None)
    p.add_argument("--_run-one", type=int, default=None,
                   help="internal: run one size and print JSON")
    args = p.parse_args()

    if args._run_one is not None:
        result = run_one_size(
            str(args.input), args._run_one, args.n_estimators,
            str(args.tfs) if args.tfs else None,
        )
        print("__RESULT__", json.dumps(result), flush=True)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.sizes:
        print(f"\n=== n_cells = {n:,} (subprocess) ===", flush=True)
        cmd = [
            sys.executable, __file__,
            "--input", str(args.input),
            "--sizes", str(n),
            "--n-estimators", str(args.n_estimators),
            "--out", str(args.out),
            "--_run-one", str(n),
        ]
        if args.tfs:
            cmd += ["--tfs", str(args.tfs)]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  FAILED at n={n}: {e.stderr[-500:]}", flush=True)
            break
        for line in completed.stdout.splitlines():
            print(f"  child: {line}", flush=True)
            if line.startswith("__RESULT__"):
                result = json.loads(line.split(" ", 1)[1])
                rows.append(result)
                for k, v in result.items():
                    mark = f"{v:,.3f}" if isinstance(v, float) else f"{v:,}"
                    print(f"  {k}: {mark}", flush=True)

    if not rows:
        print("no results collected", flush=True)
        return 1

    csv_path = args.out / "scaling_results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}", flush=True)

    try:
        plot_results(rows, args.out / "scaling_plot.png")
        print(f"wrote {args.out / 'scaling_plot.png'}", flush=True)
    except Exception as e:
        print(f"plot skipped ({e})", flush=True)

    return 0


def plot_results(rows, path: Path):
    import matplotlib.pyplot as plt

    n = np.array([r["n_cells"] for r in rows], dtype=float)
    t_grn = np.array([r["t_grn_s"] for r in rows])
    t_auc = np.array([r["t_auc_s"] for r in rows])
    rss = np.array([r["rss_peak_mb"] for r in rows]) / 1024.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(n, t_grn, "o-", label="GRN inference",
             color="#2980b9", linewidth=2, markersize=8)
    ax1.plot(n, t_auc, "s-", label="AUCell",
             color="#27ae60", linewidth=2, markersize=8)
    ref = t_auc[0] * (n / n[0])
    ax1.plot(n, ref, "k--", alpha=0.4, label="linear reference (slope=1)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("cells"); ax1.set_ylabel("wall-time (s)")
    ax1.set_title("rustscenic wall-time vs cell count")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(n, rss, "o-", color="#c0392b", linewidth=2, markersize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("cells"); ax2.set_ylabel("peak RSS (GB)")
    ax2.set_title("rustscenic peak memory vs cell count")
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
