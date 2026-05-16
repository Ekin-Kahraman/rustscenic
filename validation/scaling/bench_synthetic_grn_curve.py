"""Synthetic high-cell-count GRN scaling curve.

This benchmark is for answering one narrow question: with genes, TFs and
estimators held constant, does GRN wall-clock scale roughly linearly as cell
count rises?

It deliberately avoids private real-data inputs. Each size runs in a fresh
subprocess so peak RSS is per-size rather than cumulative across the whole
curve.

Examples:
    python validation/scaling/bench_synthetic_grn_curve.py

    python validation/scaling/bench_synthetic_grn_curve.py \\
        --sizes 25000 50000 100000 200000 \\
        --n-genes 500 \\
        --n-tfs 30 \\
        --n-estimators 20 \\
        --target-block-size 32 \\
        --max-slope 1.30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUT = Path(__file__).with_name("synthetic_grn_scaling_curve.json")


def peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process().memory_info().rss / (1024**2)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024**2)
    return rss / 1024


def synthetic_expression(
    *,
    n_cells: int,
    n_genes: int,
    n_tfs: int,
    n_programmes: int,
    seed: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a dense, log-normalised-looking expression matrix.

    The programme signal keeps the data non-degenerate while leaving cell count
    as the only changing dimension in the scaling curve.
    """
    if n_tfs > n_genes:
        raise ValueError("n_tfs must be <= n_genes")
    if n_programmes < 1:
        raise ValueError("n_programmes must be >= 1")

    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=2.0, scale=0.5, size=(n_cells, n_genes)).astype(np.float32)
    clusters = rng.integers(0, n_programmes, size=n_cells)
    block = max(1, min(n_genes // n_programmes, 25))
    for programme in range(n_programmes):
        rows = np.flatnonzero(clusters == programme)
        if rows.size == 0:
            continue
        start = (programme * block) % n_genes
        cols = np.arange(start, min(start + block, n_genes))
        signal = rng.normal(loc=1.5, scale=0.2, size=(rows.size, cols.size))
        x[np.ix_(rows, cols)] += signal.astype(np.float32)
    np.clip(x, 0.0, None, out=x)

    genes = [f"G{i:05d}" for i in range(n_genes)]
    tfs = genes[:n_tfs]
    return x, genes, tfs


def log_log_slope(points: list[dict[str, Any]], value_key: str) -> float:
    xs = [math.log(float(row["n_cells"])) for row in points]
    ys = [math.log(float(row[value_key])) for row in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def segment_slopes(points: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for left, right in zip(points, points[1:]):
        slope = (
            math.log(float(right[value_key]) / float(left[value_key]))
            / math.log(float(right["n_cells"]) / float(left["n_cells"]))
        )
        out.append(
            {
                "from_cells": left["n_cells"],
                "to_cells": right["n_cells"],
                "slope": round(slope, 3),
            }
        )
    return out


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    import rustscenic.grn

    x, genes, tfs = synthetic_expression(
        n_cells=args.run_one,
        n_genes=args.n_genes,
        n_tfs=args.n_tfs,
        n_programmes=args.n_programmes,
        seed=args.seed + args.run_one,
    )

    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        (x, genes),
        tf_names=tfs,
        n_estimators=args.n_estimators,
        target_block_size=args.target_block_size,
        seed=args.seed,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    rss = peak_rss_mb()

    return {
        "n_cells": args.run_one,
        "n_genes": args.n_genes,
        "n_tfs": args.n_tfs,
        "n_estimators": args.n_estimators,
        "target_block_size": args.target_block_size,
        "wall_s": round(wall, 3),
        "edges": int(len(grn)),
        "peak_rss_mb": round(rss, 1) if rss is not None else None,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 25_000, 50_000])
    parser.add_argument("--n-genes", type=int, default=300)
    parser.add_argument("--n-tfs", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument(
        "--target-block-size",
        type=int,
        default=None,
        help="target window width for GRN; omit for the adaptive default",
    )
    parser.add_argument("--n-programmes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-slope",
        type=float,
        default=None,
        help="fail if the overall wall-time slope exceeds this threshold",
    )
    parser.add_argument("--run-one", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.run_one is not None:
        print("__RESULT__ " + json.dumps(run_one(args), sort_keys=True), flush=True)
        return 0

    if len(args.sizes) < 2:
        raise SystemExit("--sizes must include at least two cell counts")

    rows: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for size in args.sizes:
        print(f"\n=== n_cells={size:,} ===", flush=True)
        cmd = [
            sys.executable,
            str(script),
            "--run-one",
            str(size),
            "--n-genes",
            str(args.n_genes),
            "--n-tfs",
            str(args.n_tfs),
            "--n-estimators",
            str(args.n_estimators),
            "--n-programmes",
            str(args.n_programmes),
            "--seed",
            str(args.seed),
        ]
        if args.target_block_size is not None:
            cmd.extend(["--target-block-size", str(args.target_block_size)])
        env = dict(os.environ)
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode
        for line in completed.stdout.splitlines():
            print(f"  child: {line}", flush=True)
            if line.startswith("__RESULT__ "):
                rows.append(json.loads(line.split(" ", 1)[1]))

    wall_slope = log_log_slope(rows, "wall_s")
    payload = {
        "benchmark": "synthetic_grn_scaling_curve",
        "params": {
            "sizes": args.sizes,
            "n_genes": args.n_genes,
            "n_tfs": args.n_tfs,
            "n_estimators": args.n_estimators,
            "target_block_size": args.target_block_size,
            "n_programmes": args.n_programmes,
            "seed": args.seed,
        },
        "results": rows,
        "wall_slope": round(wall_slope, 3),
        "segment_slopes": segment_slopes(rows, "wall_s"),
        "interpretation": (
            "near-linear" if wall_slope <= 1.2 else "mildly super-linear"
            if wall_slope <= 1.5 else "super-linear"
        ),
    }
    if args.max_slope is not None:
        payload["max_slope"] = args.max_slope
        payload["passed"] = wall_slope <= args.max_slope

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwall-time log-log slope: {wall_slope:.3f}")
    print(f"wrote {args.out}")

    if args.max_slope is not None and wall_slope > args.max_slope:
        print(
            f"slope {wall_slope:.3f} exceeds --max-slope {args.max_slope:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
