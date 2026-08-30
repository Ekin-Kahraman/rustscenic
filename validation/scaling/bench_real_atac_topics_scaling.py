"""Real-data scATAC preprocessing and topic-model scaling benchmark.

The coordinator builds a cell-called peak matrix once from 10x fragments,
then runs each topic-model point in a fresh child process.  Fresh processes
are required because Rayon reads ``RAYON_NUM_THREADS`` when its global pool is
first initialised.  The resulting JSON contains no absolute paths.

Example
-------
python validation/scaling/bench_real_atac_topics_scaling.py \
  --dataset-name 10x-human-brain-gemx-10k \
  --rna-10x-h5 /data/filtered_feature_bc_matrix.h5 \
  --fragments /data/atac_fragments.tsv.gz \
  --peaks /data/atac_peaks.bed \
  --work-dir /scratch/rustscenic-atac \
  --out /results/atac-scaling.json \
  --cell-counts 2500 5000 10000 \
  --thread-cells 5000 --thread-counts 4 8 16 \
  --gibbs-cells 5000 --gibbs-threads 16 --require-clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


THREAD_ENV_KEYS = (
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--rna-10x-h5", type=Path, required=True)
    p.add_argument("--fragments", type=Path, required=True)
    p.add_argument("--peaks", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo-dir", type=Path, required=True)
    p.add_argument("--cell-counts", type=int, nargs="+", default=[2500, 5000, 10000])
    p.add_argument("--thread-cells", type=int, default=5000)
    p.add_argument("--thread-counts", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--skip-vb", action="store_true")
    p.add_argument("--n-topics", type=int, default=30)
    p.add_argument("--n-passes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gibbs-cells", type=int, default=0)
    p.add_argument("--gibbs-threads", type=int, default=16)
    p.add_argument("--gibbs-iters", type=int, default=50)
    p.add_argument("--gibbs-cell-counts", type=int, nargs="*", default=[])
    p.add_argument("--gibbs-thread-cells", type=int, default=0)
    p.add_argument("--gibbs-thread-counts", type=int, nargs="*", default=[])
    p.add_argument(
        "--reuse-matrix",
        type=Path,
        help="reuse a previously validated cell-called, binarized sparse H5AD",
    )
    p.add_argument("--require-clean", action="store_true")
    p.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--matrix", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--method", choices=("vb", "gibbs"), help=argparse.SUPPRESS)
    p.add_argument("--n-cells", type=int, help=argparse.SUPPRESS)
    p.add_argument("--threads", type=int, help=argparse.SUPPRESS)
    p.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    return p


def _rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3) if sys.platform == "darwin" else rss / (1024**2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: Any) -> None:
    def json_default(item: Any) -> Any:
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    tmp.replace(path)


def _top_peak_indices(topic_peak, n: int = 10) -> list[list[int]]:
    import numpy as np

    values = topic_peak.to_numpy(dtype=np.float32, copy=False)
    out: list[list[int]] = []
    for row in values:
        order = np.argsort(row, kind="stable")[-n:][::-1]
        out.append([int(i) for i in order])
    return out


def _run_one(args: argparse.Namespace) -> int:
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    import rustscenic.topics

    if not args.matrix or not args.child_out or not args.method:
        raise SystemExit("--run-one requires --matrix, --child-out and --method")
    if not args.n_cells or not args.threads:
        raise SystemExit("--run-one requires positive --n-cells and --threads")

    t0 = time.perf_counter()
    atac = ad.read_h5ad(args.matrix)
    if args.n_cells > atac.n_obs:
        raise ValueError(f"requested {args.n_cells} cells, matrix has {atac.n_obs}")
    atac = atac[: args.n_cells].copy()
    if not sp.issparse(atac.X):
        raise TypeError("ATAC benchmark matrix must remain sparse")
    atac.X = atac.X.tocsr().astype(np.float32, copy=False)
    load_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    if args.method == "vb":
        result = rustscenic.topics.fit(
            atac,
            n_topics=args.n_topics,
            n_passes=args.n_passes,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=False,
        )
        kernel = "topics_fit"
        iterations = args.n_passes
    else:
        result = rustscenic.topics.fit_gibbs(
            atac,
            n_topics=args.n_topics,
            n_iters=args.gibbs_iters,
            n_threads=args.threads,
            seed=args.seed,
            verbose=False,
        )
        kernel = "topics_fit_gibbs"
        iterations = args.gibbs_iters
    fit_wall = time.perf_counter() - t0

    cell_topic = result.cell_topic.to_numpy(dtype=np.float32, copy=False)
    topic_peak = result.topic_peak.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(cell_topic).all() or not np.isfinite(topic_peak).all():
        raise AssertionError("topic outputs contain non-finite values")
    assignments = np.argmax(cell_topic, axis=1).astype(np.int32)
    empty_cells = int(np.count_nonzero(cell_topic.sum(axis=1) <= 0.0))
    top_peaks = _top_peak_indices(result.topic_peak)

    t0 = time.perf_counter()
    npmi = np.asarray(rustscenic.topics.coherence_npmi(result, atac, top_n=10))
    npmi_wall = time.perf_counter() - t0

    record = {
        "method": args.method,
        "kernel": kernel,
        "n_cells": int(atac.n_obs),
        "n_peaks": int(atac.n_vars),
        "nnz": int(atac.X.nnz),
        "threads": int(args.threads),
        "n_topics": int(args.n_topics),
        "iterations": int(iterations),
        "seed": int(args.seed),
        "load_wall_s": round(load_wall, 3),
        "fit_wall_s": round(fit_wall, 3),
        "npmi_wall_s": round(npmi_wall, 3),
        "peak_rss_gb": round(_rss_gb(), 3),
        "unique_argmax_topics": int(np.unique(assignments).size),
        "empty_cells": empty_cells,
        "cell_topic_row_sum_max_abs_error": float(
            np.max(np.abs(cell_topic.sum(axis=1) - 1.0))
        ),
        "topic_peak_row_sum_max_abs_error": float(
            np.max(np.abs(topic_peak.sum(axis=1) - 1.0))
        ),
        "npmi_mean": float(np.nanmean(npmi)),
        "npmi_median": float(np.nanmedian(npmi)),
        "assignment_sha256": hashlib.sha256(assignments.tobytes()).hexdigest(),
        "assignments": assignments.tolist(),
        "top_peak_indices": top_peaks,
        "backend_execution": result.backend_execution,
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
    }
    _atomic_json(args.child_out, record)
    print(json.dumps({k: v for k, v in record.items() if k not in {"assignments", "top_peak_indices"}}, sort_keys=True))
    return 0


def _thread_comparisons(
    runs: list[dict[str, Any]], n_cells: int, method: str = "vb"
) -> list[dict[str, Any]]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score

    selected = sorted(
        (r for r in runs if r["method"] == method and r["n_cells"] == n_cells),
        key=lambda r: r["threads"],
    )
    if not selected:
        return []
    baseline = selected[0]
    baseline_assignments = np.asarray(baseline["assignments"], dtype=np.int32)
    baseline_top = baseline["top_peak_indices"]
    out = []
    for run in selected:
        assignments = np.asarray(run["assignments"], dtype=np.int32)
        overlap = np.zeros((len(baseline_top), len(run["top_peak_indices"])), dtype=np.float64)
        for i, left in enumerate(baseline_top):
            left_set = set(left)
            for j, right in enumerate(run["top_peak_indices"]):
                overlap[i, j] = len(left_set.intersection(right)) / 10.0
        rows, cols = linear_sum_assignment(-overlap)
        out.append(
            {
                "method": method,
                "baseline_threads": int(baseline["threads"]),
                "threads": int(run["threads"]),
                "speedup_vs_baseline": round(baseline["fit_wall_s"] / run["fit_wall_s"], 3),
                "efficiency_vs_baseline": round(
                    (baseline["fit_wall_s"] / run["fit_wall_s"])
                    / (run["threads"] / baseline["threads"]),
                    3,
                ),
                "assignment_ari": float(adjusted_rand_score(baseline_assignments, assignments)),
                "matched_top10_peak_overlap_mean": float(overlap[rows, cols].mean()),
            }
        )
    return out


def _coordinator(args: argparse.Namespace) -> int:
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    import rustscenic
    import rustscenic.preproc

    for path in (args.rna_10x_h5, args.fragments, args.peaks):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.repo_dir = args.repo_dir.resolve()
    commit = _git(args.repo_dir, "rev-parse", "HEAD")
    status = _git(args.repo_dir, "status", "--porcelain")
    if args.require_clean and status:
        raise RuntimeError("--require-clean requested but repository is dirty")

    counts = sorted(set(args.cell_counts))
    threads = sorted(set(args.thread_counts))
    gibbs_counts = sorted(set(args.gibbs_cell_counts))
    gibbs_threads = sorted(set(args.gibbs_thread_counts))
    positive_values = counts + threads + gibbs_counts + gibbs_threads
    if any(n < 1 for n in positive_values):
        raise ValueError("cell and thread counts must be positive")
    requested_cells = []
    if not args.skip_vb:
        requested_cells.extend(counts + [args.thread_cells])
    requested_cells.extend(gibbs_counts)
    if args.gibbs_cells:
        requested_cells.append(args.gibbs_cells)
    if args.gibbs_thread_cells:
        requested_cells.append(args.gibbs_thread_cells)
    if not requested_cells:
        raise ValueError("no VB or Gibbs benchmark points were requested")
    max_cells = max(requested_cells)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.reuse_matrix or (args.work_dir / "atac_cell_called_binarized.h5ad")

    input_hashes = {
        "rna_10x_h5": {"name": args.rna_10x_h5.name, "sha256": _sha256(args.rna_10x_h5)},
        "fragments": {"name": args.fragments.name, "sha256": _sha256(args.fragments)},
        "peaks": {"name": args.peaks.name, "sha256": _sha256(args.peaks)},
    }

    t0 = time.perf_counter()
    if args.reuse_matrix:
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        atac = ad.read_h5ad(matrix_path)
        if not sp.issparse(atac.X):
            raise TypeError("reused ATAC benchmark matrix must be sparse")
        atac.X = atac.X.tocsr()
        if atac.n_obs < max_cells:
            raise AssertionError(
                f"reused matrix has {atac.n_obs} rows; {max_cells} are required"
            )
        if not np.all(atac.X.data == 1.0):
            raise AssertionError("reused ATAC benchmark matrix is not binarized")
        preproc_backend = atac.uns.get("rust_backend")
        reused = True
    else:
        rna = sc.read_10x_h5(args.rna_10x_h5, gex_only=True)
        barcodes = list(rna.obs_names[:max_cells])
        del rna
        atac = rustscenic.preproc.fragments_to_matrix(
            args.fragments,
            args.peaks,
            cell_barcodes=barcodes,
        )
        if not sp.issparse(atac.X):
            raise TypeError("fragments_to_matrix returned a dense matrix")
        atac.X = atac.X.tocsr().astype("float32", copy=False)
        atac.X.data.fill(1.0)
        atac.X.eliminate_zeros()
        atac.X.sort_indices()
        if atac.n_obs != max_cells:
            raise AssertionError(f"expected {max_cells} cell-called rows, got {atac.n_obs}")
        atac.write_h5ad(matrix_path, compression="lzf")
        preproc_backend = atac.uns.get("rust_backend")
        reused = False
    preproc_wall = time.perf_counter() - t0
    preproc = {
        "wall_s": round(preproc_wall, 3),
        "peak_rss_gb": round(_rss_gb(), 3),
        "shape": [int(atac.n_obs), int(atac.n_vars)],
        "nnz": int(atac.X.nnz),
        "dtype": str(atac.X.dtype),
        "format": "csr",
        "binarized": True,
        "reused": reused,
        "matrix_name": matrix_path.name,
        "backend_execution": preproc_backend,
    }
    del atac

    points: list[tuple[str, int, int]] = []
    if not args.skip_vb:
        for n_cells in counts:
            points.append(("vb", n_cells, max(threads)))
        for n_threads in threads:
            points.append(("vb", args.thread_cells, n_threads))
    if args.gibbs_cells:
        points.append(("gibbs", args.gibbs_cells, args.gibbs_threads))
    gibbs_scale_threads = gibbs_threads or [args.gibbs_threads]
    for n_cells in gibbs_counts:
        points.append(("gibbs", n_cells, max(gibbs_scale_threads)))
    if args.gibbs_thread_cells:
        for n_threads in gibbs_scale_threads:
            points.append(("gibbs", args.gibbs_thread_cells, n_threads))
    points = list(dict.fromkeys(points))

    runs: list[dict[str, Any]] = []
    for method, n_cells, n_threads in points:
        child_out = args.work_dir / f"{method}_{n_cells}cells_{n_threads}threads.json"
        env = os.environ.copy()
        env.update(
            {
                "RAYON_NUM_THREADS": str(n_threads),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-one",
            "--dataset-name", args.dataset_name,
            "--rna-10x-h5", str(args.rna_10x_h5),
            "--fragments", str(args.fragments),
            "--peaks", str(args.peaks),
            "--work-dir", str(args.work_dir),
            "--out", str(args.out),
            "--repo-dir", str(args.repo_dir),
            "--matrix", str(matrix_path),
            "--method", method,
            "--n-cells", str(n_cells),
            "--threads", str(n_threads),
            "--n-topics", str(args.n_topics),
            "--n-passes", str(args.n_passes),
            "--batch-size", str(args.batch_size),
            "--gibbs-iters", str(args.gibbs_iters),
            "--seed", str(args.seed),
            "--child-out", str(child_out),
        ]
        subprocess.run(command, check=True, env=env)
        runs.append(json.loads(child_out.read_text()))

    comparisons: list[dict[str, Any]] = []
    if not args.skip_vb:
        comparisons.extend(_thread_comparisons(runs, args.thread_cells, "vb"))
    if args.gibbs_thread_cells:
        comparisons.extend(
            _thread_comparisons(runs, args.gibbs_thread_cells, "gibbs")
        )
    for run in runs:
        run.pop("assignments", None)
        run.pop("top_peak_indices", None)

    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    artifact = {
        "benchmark": "real_atac_preprocessing_and_topics_scaling",
        "harness_sha256": _sha256(Path(__file__)),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "Real-data execution, memory, thread-scaling and topic-output invariants; not biological cell-type validation.",
        "dataset": {"name": args.dataset_name, "inputs": input_hashes},
        "rustscenic": {"version": version("rustscenic"), "commit": commit, "source_dirty": bool(status)},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": cpu,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cache_condition": "All inputs were read once for SHA-256 before preprocessing; filesystem page cache may therefore be warm.",
        },
        "params": {
            "cell_counts": counts,
            "thread_cells": args.thread_cells,
            "thread_counts": threads,
            "skip_vb": args.skip_vb,
            "n_topics": args.n_topics,
            "n_passes": args.n_passes,
            "batch_size": args.batch_size,
            "gibbs_cells": args.gibbs_cells,
            "gibbs_threads": args.gibbs_threads,
            "gibbs_iters": args.gibbs_iters,
            "gibbs_cell_counts": gibbs_counts,
            "gibbs_thread_cells": args.gibbs_thread_cells,
            "gibbs_thread_counts": gibbs_threads,
            "seed": args.seed,
        },
        "preprocessing": preproc,
        "runs": runs,
        "thread_comparisons": comparisons,
        "path_policy": "portable",
        "invocation": {"python": "python", "script": Path(__file__).name},
    }
    _atomic_json(args.out, artifact)
    print(json.dumps({"status": "ok", "out": args.out.name, "runs": len(runs)}, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    return _run_one(args) if args.run_one else _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
