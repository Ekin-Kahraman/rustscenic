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
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUT = Path(__file__).with_name("synthetic_grn_scaling_curve.json")
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path | None:
    explicit = os.environ.get("RUSTSCENIC_REPO_DIR")
    candidates = [Path(explicit)] if explicit else []
    candidates.extend([REPO_ROOT, Path.cwd()])
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate.resolve()
    return None


def git_output(args: list[str]) -> str | None:
    root = repo_root()
    if root is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_state() -> dict[str, Any]:
    status = git_output(["status", "--short", "--untracked-files=all"])
    return {
        "commit": os.environ.get("RUSTSCENIC_SOURCE_SHA")
        or git_output(["rev-parse", "HEAD"]),
        "source_dirty": None if status is None else bool(status),
    }


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

    # Independent RNG streams make every smaller point a strict row-prefix of
    # every larger point.  Cell count is therefore the only changing input.
    rng = np.random.default_rng(seed)
    cluster_rng = np.random.default_rng(seed + 1)
    x = rng.gamma(shape=2.0, scale=0.5, size=(n_cells, n_genes)).astype(np.float32)
    clusters = cluster_rng.integers(0, n_programmes, size=n_cells)
    block = max(1, min(n_genes // n_programmes, 25))
    for programme in range(n_programmes):
        rows = np.flatnonzero(clusters == programme)
        if rows.size == 0:
            continue
        start = (programme * block) % n_genes
        cols = np.arange(start, min(start + block, n_genes))
        signal_rng = np.random.default_rng(seed + 2 + programme)
        signal = signal_rng.normal(loc=1.5, scale=0.2, size=(rows.size, cols.size))
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
        seed=args.seed,
    )

    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        (x, genes),
        tf_names=tfs,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_features=args.max_features,
        subsample=args.subsample,
        max_depth=args.max_depth,
        early_stop_window=args.early_stop_window,
        early_stop_mode=args.early_stop_mode,
        target_block_size=args.target_block_size,
        seed=args.seed,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    rss = peak_rss_mb()

    if grn.empty or not np.isfinite(grn["importance"].to_numpy()).all():
        raise AssertionError("GRN must contain finite, non-empty importance values")
    canonical = grn[["TF", "target", "importance"]].copy()
    canonical["importance"] = canonical["importance"].round(7)
    canonical = canonical.sort_values(["TF", "target"], kind="stable")
    signature = hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()

    return {
        "n_cells": args.run_one,
        "n_genes": args.n_genes,
        "n_tfs": args.n_tfs,
        "threads": args.threads,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_features": args.max_features,
        "subsample": args.subsample,
        "max_depth": args.max_depth,
        "early_stop_window": args.early_stop_window,
        "early_stop_mode": args.early_stop_mode,
        "target_block_size": args.target_block_size,
        "wall_s": round(wall, 3),
        "edges": int(len(grn)),
        "importance_sum": float(grn["importance"].sum()),
        "output_sha256": signature,
        "grn_fit": dict(grn.attrs.get("grn_fit", {})),
        "backend_execution": dict(grn.attrs.get("rust_backend", {})),
        "peak_rss_mb": round(rss, 1) if rss is not None else None,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 25_000, 50_000])
    parser.add_argument("--n-genes", type=int, default=300)
    parser.add_argument("--n-tfs", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-features", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--early-stop-window", type=int, default=25)
    parser.add_argument(
        "--early-stop-mode",
        choices=("arboreto", "legacy_inbag"),
        default="arboreto",
    )
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

    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.n_genes < 1 or args.n_tfs < 1 or args.n_tfs > args.n_genes:
        raise SystemExit("require 1 <= --n-tfs <= --n-genes")
    if args.n_estimators < 1 or args.max_depth < 1:
        raise SystemExit("--n-estimators and --max-depth must be positive")
    if args.early_stop_window < 0:
        raise SystemExit("--early-stop-window must be non-negative")
    if args.learning_rate <= 0 or not 0 < args.max_features <= 1:
        raise SystemExit("invalid learning-rate or max-features")
    if not 0 < args.subsample <= 1:
        raise SystemExit("--subsample must be in (0, 1]")

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
            "--threads",
            str(args.threads),
            "--learning-rate",
            str(args.learning_rate),
            "--max-features",
            str(args.max_features),
            "--subsample",
            str(args.subsample),
            "--max-depth",
            str(args.max_depth),
            "--early-stop-window",
            str(args.early_stop_window),
            "--early-stop-mode",
            args.early_stop_mode,
            "--n-programmes",
            str(args.n_programmes),
            "--seed",
            str(args.seed),
        ]
        if args.target_block_size is not None:
            cmd.extend(["--target-block-size", str(args.target_block_size)])
        env = dict(os.environ)
        env.update(
            {
                "RAYON_NUM_THREADS": str(args.threads),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
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
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": (
            "Cell-count scaling with a fixed synthetic expression schema; "
            "not biological validation or a full-gene-count runtime claim."
        ),
        "rustscenic": {"version": version("rustscenic"), **source_state()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "thread_policy": {
                "rayon": args.threads,
                "blas_openmp": 1,
            },
        },
        "path_policy": "portable",
        "params": {
            "sizes": args.sizes,
            "n_genes": args.n_genes,
            "n_tfs": args.n_tfs,
            "n_estimators": args.n_estimators,
            "threads": args.threads,
            "learning_rate": args.learning_rate,
            "max_features": args.max_features,
            "subsample": args.subsample,
            "max_depth": args.max_depth,
            "early_stop_window": args.early_stop_window,
            "early_stop_mode": args.early_stop_mode,
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
        "correctness_checks": {
            "all_runs_non_empty": all(row["edges"] > 0 for row in rows),
            "all_importances_finite": all(
                math.isfinite(float(row["importance_sum"])) for row in rows
            ),
            "rust_backend_every_run": all(
                row["backend_execution"].get("engine") == "rust" for row in rows
            ),
        },
    }
    if not all(payload["correctness_checks"].values()):
        raise AssertionError(f"correctness checks failed: {payload['correctness_checks']}")
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
