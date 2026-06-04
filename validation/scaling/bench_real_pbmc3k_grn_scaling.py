from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.backend_requirements import backend_capabilities
from validation.python_hot_paths import hot_path_state
from validation.repo_cleanliness import repo_state_from_git_outputs


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def repo_state() -> dict[str, Any]:
    """Return enough git state to decide whether a benchmark is release-grade."""
    commit = _git_output(["rev-parse", "HEAD"])
    tracked_status = _git_output(["status", "--short", "--untracked-files=no"]) or ""
    untracked_status = _git_output(["status", "--short", "--untracked-files=all"]) or ""
    tracked_diff = _git_output(["diff", "HEAD", "--binary", "--no-ext-diff"]) or ""
    return repo_state_from_git_outputs(
        commit=commit,
        tracked_status=tracked_status,
        untracked_status=untracked_status,
        tracked_diff=tracked_diff,
    )


def _path_under(path: str | None, root: Path) -> bool | None:
    if not path:
        return None
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def runtime_import_state() -> dict[str, Any]:
    """Record the actual Python package and extension imported by this run."""
    import rustscenic

    package_version = getattr(rustscenic, "__version__", None)
    try:
        import rustscenic._rustscenic as ext
    except Exception as exc:  # pragma: no cover - benchmark provenance path
        extension_file = None
        extension_version = None
        extension_error = repr(exc)
    else:
        extension_file = getattr(ext, "__file__", None)
        extension_version = getattr(ext, "__version__", None)
        extension_error = None

    package_file = getattr(rustscenic, "__file__", None)
    return {
        "package_version": package_version,
        "extension_version": extension_version,
        "package_file": package_file,
        "package_under_repo": _path_under(package_file, REPO_ROOT),
        "extension_file": extension_file,
        "extension_under_repo": _path_under(extension_file, REPO_ROOT),
        "extension_error": extension_error,
    }


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024**3)
    return float(rss) / (1024**2)


def configure_thread_env(threads: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def load_pbmc_rna(data_dir: Path):
    import rustscenic.data

    rna_h5 = data_dir / "pbmc_3k_filtered_feature_bc_matrix.h5"
    rna = sc.read_10x_h5(rna_h5)
    rna.var_names_make_unique()
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    all_tfs = rustscenic.data.tfs(species="hs")
    present = set(rna.var_names)
    tfs = [tf for tf in all_tfs if tf in present]
    return rna, tfs


def subset_cells(rna, n_cells: int, seed: int):
    if n_cells >= rna.n_obs:
        return rna.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(rna.n_obs)
    keep = np.sort(order[:n_cells])
    return rna[keep].copy()


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    import rustscenic.grn

    t_load = time.perf_counter()
    rna, tfs = load_pbmc_rna(args.data_dir)
    rna_sub = subset_cells(rna, args.n_cells, args.seed)
    load_wall = time.perf_counter() - t_load

    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        rna_sub,
        tfs,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_features=args.max_features,
        subsample=args.subsample,
        max_depth=args.max_depth,
        early_stop_window=args.early_stop_window,
        target_block_size=args.target_block_size,
        seed=args.seed,
        verbose=False,
    )
    grn_wall = time.perf_counter() - t0

    out = {
        "dataset": "10x_pbmc_unsorted_3k",
        "run_kind": args.run_kind,
        "n_cells": int(rna_sub.n_obs),
        "n_genes": int(rna_sub.n_vars),
        "n_tfs": int(len(tfs)),
        "threads": int(args.threads),
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_features": float(args.max_features),
        "subsample": float(args.subsample),
        "max_depth": int(args.max_depth),
        "early_stop_window": int(args.early_stop_window),
        "target_block_size": args.target_block_size,
        "load_qc_wall_s": round(load_wall, 3),
        "grn_wall_s": round(grn_wall, 3),
        "edges": int(len(grn)),
        "peak_rss_gb": round(peak_rss_gb(), 3),
        "env": {
            "python": platform.python_version(),
            "rustscenic": version("rustscenic"),
            "scanpy": version("scanpy"),
            "anndata": version("anndata"),
            "host": platform.node(),
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "lsf_project": os.environ.get("RUSTSCENIC_LSF_PROJECT"),
            "lsf_requested_queue": os.environ.get("RUSTSCENIC_LSF_QUEUE"),
            "lsf_requested_cores": os.environ.get("RUSTSCENIC_LSF_CORES"),
            "lsf_requested_mem_mb": os.environ.get("RUSTSCENIC_LSF_MEM_MB"),
            "lsf_requested_walltime": os.environ.get("RUSTSCENIC_LSF_WALLTIME"),
            "repo_state": repo_state(),
            "runtime_import": runtime_import_state(),
            "backend_capabilities": backend_capabilities(),
            "python_hot_paths": hot_path_state(),
        },
    }
    return out


def log_log_slope(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    usable = [row for row in rows if row.get(x_key) and row.get(y_key)]
    if len(usable) < 2:
        return None
    xs = [math.log(float(row[x_key])) for row in usable]
    ys = [math.log(float(row[y_key])) for row in usable]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / den


def speedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_threads = sorted(rows, key=lambda row: row["threads"])
    baseline = next((row for row in by_threads if row["threads"] == 1), None)
    if baseline is None:
        baseline = by_threads[0] if by_threads else None
    if baseline is None:
        return []
    base_wall = float(baseline["grn_wall_s"])
    return [
        {
            "threads": row["threads"],
            "wall_s": row["grn_wall_s"],
            "speedup_vs_baseline": round(base_wall / float(row["grn_wall_s"]), 3),
            "efficiency_vs_baseline": round(
                (base_wall / float(row["grn_wall_s"])) / (row["threads"] / baseline["threads"]),
                3,
            ),
        }
        for row in by_threads
    ]


def child_cmd(args: argparse.Namespace, *, n_cells: int, threads: int, run_kind: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        "--run-kind",
        run_kind,
        "--data-dir",
        str(args.data_dir),
        "--n-cells",
        str(n_cells),
        "--threads",
        str(threads),
        "--n-estimators",
        str(args.n_estimators),
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
        "--seed",
        str(args.seed),
    ]
    if args.target_block_size is not None:
        cmd.extend(["--target-block-size", str(args.target_block_size)])
    return cmd


def run_child(args: argparse.Namespace, *, n_cells: int, threads: int, run_kind: str) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "RAYON_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = subprocess.run(
        child_cmd(args, n_cells=n_cells, threads=threads, run_kind=run_kind),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"child failed for n_cells={n_cells}, threads={threads}, "
            f"run_kind={run_kind}, code={completed.returncode}"
        )
    result_line = None
    for line in completed.stdout.splitlines():
        if line.startswith("__RESULT__ "):
            result_line = line.split(" ", 1)[1]
    if result_line is None:
        raise RuntimeError("child did not emit __RESULT__ line")
    return json.loads(result_line)


def coordinator(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    state = repo_state()
    if args.require_clean and state["source_dirty"]:
        raise SystemExit(
            "source files are dirty; commit tracked changes and add or remove "
            "untracked source files before running a publication-grade "
            "benchmark. Use without --require-clean only for explicit "
            "local-build profiling."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset_rows: list[dict[str, Any]] = []
    thread_rows: list[dict[str, Any]] = []

    for n_cells in args.subset_sizes:
        print(f"\n=== subset scaling: cells={n_cells:,}, threads={args.subset_threads} ===", flush=True)
        subset_rows.append(
            run_child(args, n_cells=n_cells, threads=args.subset_threads, run_kind="subset_scaling")
        )
        args.out.write_text(json.dumps({"partial_subset_scaling": subset_rows}, indent=2))

    for threads in args.thread_counts:
        print(f"\n=== thread scaling: cells={args.thread_cells:,}, threads={threads} ===", flush=True)
        thread_rows.append(
            run_child(args, n_cells=args.thread_cells, threads=threads, run_kind="thread_scaling")
        )
        args.out.write_text(
            json.dumps(
                {
                    "partial_subset_scaling": subset_rows,
                    "partial_thread_scaling": thread_rows,
                },
                indent=2,
            )
        )

    payload = {
        "benchmark": "real_pbmc3k_grn_scaling",
        "dataset": "10x PBMC unsorted 3k multiome RNA post-QC",
        "repo_state": state,
        "runtime_import": runtime_import_state(),
        "backend_capabilities": backend_capabilities(),
        "python_hot_paths": hot_path_state(),
        "rustscenic": version("rustscenic"),
        "params": {
            "subset_sizes": args.subset_sizes,
            "subset_threads": args.subset_threads,
            "thread_cells": args.thread_cells,
            "thread_counts": args.thread_counts,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_features": args.max_features,
            "subsample": args.subsample,
            "max_depth": args.max_depth,
            "early_stop_window": args.early_stop_window,
            "target_block_size": args.target_block_size,
            "seed": args.seed,
        },
        "subset_scaling": subset_rows,
        "subset_wall_slope_vs_cells": None if (s := log_log_slope(subset_rows, "n_cells", "grn_wall_s")) is None else round(s, 3),
        "subset_memory_slope_vs_cells": None if (s := log_log_slope(subset_rows, "n_cells", "peak_rss_gb")) is None else round(s, 3),
        "thread_scaling": thread_rows,
        "thread_speedups": speedup_rows(thread_rows),
        "interpretation_notes": [
            "This isolates real-data GRN. It does not include topics, cistarget, enhancer, eRegulon or AUCell.",
            "Genes and TFs are held fixed after full PBMC RNA QC; only cells or Rayon threads vary.",
            "Each data point runs in a fresh Python subprocess so Rayon thread count and peak RSS are isolated.",
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"--{name.replace('_', '-')} must be positive")


def _require_positive_sorted_unique(name: str, values: list[int]) -> None:
    flag = f"--{name.replace('_', '-')}"
    if any(value <= 0 for value in values):
        raise SystemExit(f"{flag} must be positive integers")
    if values != sorted(values):
        raise SystemExit(f"{flag} must be sorted ascending")
    if len(set(values)) != len(values):
        raise SystemExit(f"{flag} must not contain duplicates")


def validate_args(args: argparse.Namespace) -> None:
    _require_positive_sorted_unique("subset_sizes", args.subset_sizes)
    _require_positive_sorted_unique("thread_counts", args.thread_counts)
    for name in (
        "subset_threads",
        "thread_cells",
        "n_estimators",
        "max_depth",
        "early_stop_window",
        "n_cells",
        "threads",
    ):
        _require_positive_int(name, int(getattr(args, name)))
    if args.target_block_size is not None:
        _require_positive_int("target_block_size", int(args.target_block_size))
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if not (0 < args.max_features <= 1):
        raise SystemExit("--max-features must be in (0, 1]")
    if not (0 < args.subsample <= 1):
        raise SystemExit("--subsample must be in (0, 1]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("real_pbmc3k_grn_scaling.json"))
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[500, 1000, 2000, 2767])
    parser.add_argument("--subset-threads", type=int, default=4)
    parser.add_argument("--thread-cells", type=int, default=1000)
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-features", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--early-stop-window", type=int, default=25)
    parser.add_argument("--target-block-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail if tracked files differ from HEAD; use for publication-grade runs",
    )
    parser.add_argument("--run-one", action="store_true")
    parser.add_argument("--run-kind", default="manual")
    parser.add_argument("--n-cells", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.run_one:
        configure_thread_env(args.threads)
        print("__RESULT__ " + json.dumps(run_one(args), sort_keys=True), flush=True)
        return 0

    payload = coordinator(args)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
