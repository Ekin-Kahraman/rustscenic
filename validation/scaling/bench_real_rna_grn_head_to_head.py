"""Same-input RustScenic versus arboreto GRN benchmark on real 10x RNA.

Run both tools sequentially in the same Slurm allocation, using their pinned
environments, then compare the two Parquet outputs. The arboreto path uses a
fork pool because arboreto 0.1.6's Dask graph is incompatible with current
Dask. Fork keeps one read-only dense expression matrix shared across workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.scaling.bench_real_rna_grn_scaling import (  # noqa: E402
    DATASET_NAME,
    DATASET_URL,
    canonical_grn_sha256,
    load_prepared_prefix,
    prepared_metadata,
)


_REFERENCE_X = None
_REFERENCE_TF_MATRIX = None
_REFERENCE_TF_NAMES = None
_REFERENCE_GENES = None
_REFERENCE_KWARGS = None
_REFERENCE_SEED = None


class ProcessTreePeakRSS:
    """Sample aggregate RSS for this process and its children."""

    def __init__(self, interval_s: float = 5.0) -> None:
        self.interval_s = interval_s
        self.peak_rss_mb = 0.0
        self.peak_pss_mb = 0.0
        self.peak_uss_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        import psutil

        process = psutil.Process()
        members = [process]
        try:
            members.extend(process.children(recursive=True))
        except psutil.Error:
            pass
        rss_total = 0
        pss_total = 0
        uss_total = 0
        for member in members:
            try:
                rss = member.memory_info().rss
                full = member.memory_full_info()
            except psutil.Error:
                continue
            rss_total += rss
            pss_total += getattr(full, "pss", rss)
            uss_total += getattr(full, "uss", rss)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_total / (1024**2))
        self.peak_pss_mb = max(self.peak_pss_mb, pss_total / (1024**2))
        self.peak_uss_mb = max(self.peak_uss_mb, uss_total / (1024**2))

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def __enter__(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._sample_once()
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


def _reference_initializer(x, tf_matrix, tf_names, genes, kwargs, seed) -> None:
    global _REFERENCE_X, _REFERENCE_TF_MATRIX, _REFERENCE_TF_NAMES
    global _REFERENCE_GENES, _REFERENCE_KWARGS, _REFERENCE_SEED
    _REFERENCE_X = x
    _REFERENCE_TF_MATRIX = tf_matrix
    _REFERENCE_TF_NAMES = tf_names
    _REFERENCE_GENES = genes
    _REFERENCE_KWARGS = kwargs
    _REFERENCE_SEED = seed


def _reference_one(target_index: int):
    from arboreto.core import infer_partial_network

    return infer_partial_network(
        regressor_type="GBM",
        regressor_kwargs=_REFERENCE_KWARGS,
        tf_matrix=_REFERENCE_TF_MATRIX,
        tf_matrix_gene_names=_REFERENCE_TF_NAMES,
        target_gene_name=_REFERENCE_GENES[target_index],
        target_gene_expression=_REFERENCE_X[:, target_index],
        include_meta=False,
        early_stop_window_length=25,
        seed=_REFERENCE_SEED,
    )


def package_versions(names: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def fitted_tree_summary(frame: pd.DataFrame) -> dict[str, Any]:
    # Arboreto and RustScenic both normalise per-target gains, then multiply
    # them by the fitted tree count. Rounding the per-target importance sum
    # therefore recovers fitted trees for a complete, untruncated adjacency.
    trees = np.rint(frame.groupby("target", sort=False)["importance"].sum()).astype(
        np.int64
    )
    return {
        "target_count": int(len(trees)),
        "total": int(trees.sum()),
        "mean": float(trees.mean()),
        "median": float(np.median(trees)),
        "min": int(trees.min()),
        "max": int(trees.max()),
    }


def run_rustscenic(matrix, genes, tfs, args) -> tuple[pd.DataFrame, dict[str, Any]]:
    import rustscenic.grn

    started = time.perf_counter()
    with ProcessTreePeakRSS() as memory:
        frame = rustscenic.grn.infer(
            (matrix, genes),
            tf_names=tfs,
            n_estimators=args.n_estimators,
            learning_rate=0.01,
            max_features=0.1,
            subsample=0.9,
            max_depth=3,
            early_stop_window=25,
            early_stop_mode="arboreto",
            seed=args.seed,
            verbose=True,
        )
    return frame, {
        "wall_s": round(time.perf_counter() - started, 3),
        "process_tree_peak_rss_mb": round(memory.peak_rss_mb, 1),
        "process_tree_peak_pss_mb": round(memory.peak_pss_mb, 1),
        "process_tree_peak_uss_mb": round(memory.peak_uss_mb, 1),
        "fit": dict(frame.attrs.get("grn_fit", {})),
    }


def run_arboreto(matrix, genes, tfs, args) -> tuple[pd.DataFrame, dict[str, Any]]:
    from arboreto.core import SGBM_KWARGS, to_tf_matrix

    dense = np.ascontiguousarray(matrix.toarray(), dtype=np.float32)
    tf_matrix, tf_names = to_tf_matrix(dense, genes, tfs)
    kwargs = dict(SGBM_KWARGS)
    kwargs["n_estimators"] = args.n_estimators
    started = time.perf_counter()
    results = []
    context = get_context("fork")
    with context.Pool(
        args.threads,
        initializer=_reference_initializer,
        initargs=(dense, tf_matrix, tf_names, genes, kwargs, args.seed),
    ) as pool:
        with ProcessTreePeakRSS() as memory:
            for completed, frame in enumerate(
                pool.imap_unordered(_reference_one, range(len(genes)), chunksize=8),
                start=1,
            ):
                results.append(frame)
                if completed % 100 == 0 or completed == len(genes):
                    elapsed = time.perf_counter() - started
                    print(
                        f"arboreto targets {completed}/{len(genes)} "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )
    output = pd.concat(results, ignore_index=True)
    return output, {
        "wall_s": round(time.perf_counter() - started, 3),
        "process_tree_peak_rss_mb": round(memory.peak_rss_mb, 1),
        "process_tree_peak_pss_mb": round(memory.peak_pss_mb, 1),
        "process_tree_peak_uss_mb": round(memory.peak_uss_mb, 1),
        "fit": fitted_tree_summary(output),
    }


def run(args: argparse.Namespace) -> int:
    metadata = prepared_metadata(args.prepared)
    matrix, genes, tfs, source_indices = load_prepared_prefix(
        args.prepared,
        n_cells=args.n_cells,
        n_hvg=args.n_hvg,
        n_tfs=args.n_tfs,
    )
    if args.tool == "rustscenic":
        output, measurements = run_rustscenic(matrix, genes, tfs, args)
        packages = package_versions(["rustscenic", "numpy", "pandas", "scipy"])
    else:
        output, measurements = run_arboreto(matrix, genes, tfs, args)
        packages = package_versions(
            ["arboreto", "scikit-learn", "dask", "distributed", "numpy", "pandas"]
        )
    if output.empty or not np.isfinite(output["importance"]).all():
        raise AssertionError("GRN output must contain finite, non-empty importances")

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.out_parquet, index=False)
    result = {
        "benchmark": "real_rna_grn_head_to_head",
        "tool": args.tool,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": DATASET_NAME,
            "url": DATASET_URL,
            "source_sha256": metadata["source_sha256"],
            "feature_sample_sha256": metadata["feature_sample_sha256"],
            "normalisation": metadata["normalisation"],
            "cell_order_seed": metadata["cell_order_seed"],
        },
        "params": {
            "n_cells": args.n_cells,
            "n_genes": int(matrix.shape[1]),
            "n_tfs": len(tfs),
            "n_hvg_requested": args.n_hvg,
            "n_tfs_requested": args.n_tfs,
            "n_estimators": args.n_estimators,
            "threads": args.threads,
            "seed": args.seed,
            "early_stop_window": 25,
            "subsample": 0.9,
        },
        "environment": {
            "packages": packages,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "thread_policy": {"target_workers": args.threads, "blas_openmp": 1},
        },
        "measurements": measurements,
        "output": {
            "edges": int(len(output)),
            "tfs": int(output["TF"].nunique()),
            "targets": int(output["target"].nunique()),
            "canonical_sha256": canonical_grn_sha256(output),
            "parquet_file": args.out_parquet.name,
            "parquet_sha256": hashlib.sha256(args.out_parquet.read_bytes()).hexdigest(),
            "fitted_trees_recovered": fitted_tree_summary(output),
        },
        "input_signatures": {
            "cell_prefix_sha256": hashlib.sha256(source_indices.tobytes()).hexdigest(),
            "gene_profile_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        },
        "path_policy": "portable",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("__RESULT__ " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def top_k_jaccard(
    left: pd.DataFrame, right: pd.DataFrame, *, k: int
) -> dict[str, float | int]:
    left_groups = {
        tf: set(group.nlargest(k, "importance")["target"])
        for tf, group in left.groupby("TF", sort=False)
    }
    right_groups = {
        tf: set(group.nlargest(k, "importance")["target"])
        for tf, group in right.groupby("TF", sort=False)
    }
    scores = []
    for tf in sorted(left_groups.keys() & right_groups.keys()):
        union = left_groups[tf] | right_groups[tf]
        scores.append(len(left_groups[tf] & right_groups[tf]) / len(union))
    return {
        "k": k,
        "shared_tfs": len(scores),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
    }


def compare(args: argparse.Namespace) -> int:
    rust_meta = json.loads(args.rust_json.read_text())
    arb_meta = json.loads(args.arboreto_json.read_text())
    if rust_meta["input_signatures"] != arb_meta["input_signatures"]:
        raise AssertionError("head-to-head inputs do not have identical signatures")
    if rust_meta["params"] != arb_meta["params"]:
        raise AssertionError("head-to-head parameters differ")
    rust = pd.read_parquet(args.rust_parquet)
    arboreto = pd.read_parquet(args.arboreto_parquet)
    shared = rust.merge(
        arboreto,
        on=["TF", "target"],
        suffixes=("_rust", "_arboreto"),
        validate="one_to_one",
    )
    fixed_shared_spearman = float(
        shared["importance_rust"].corr(shared["importance_arboreto"], method="spearman")
    )
    rust_wall = float(rust_meta["measurements"]["wall_s"])
    arb_wall = float(arb_meta["measurements"]["wall_s"])
    payload = {
        "benchmark": "real_rna_grn_head_to_head_comparison",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": rust_meta["dataset"],
        "params": rust_meta["params"],
        "input_signatures": rust_meta["input_signatures"],
        "same_slurm_job": (
            rust_meta["environment"]["slurm_job_id"]
            == arb_meta["environment"]["slurm_job_id"]
        ),
        "rustscenic": rust_meta,
        "arboreto": arb_meta,
        "comparison": {
            "arboreto_over_rustscenic_wall_ratio": arb_wall / rust_wall,
            "shared_edges": int(len(shared)),
            "shared_edge_fraction_of_rust": float(len(shared) / len(rust)),
            "shared_edge_fraction_of_arboreto": float(len(shared) / len(arboreto)),
            "fixed_shared_edge_spearman": fixed_shared_spearman,
            "spearman_scope": (
                "intersection fixed for this two-tool comparison; absent edges are not ranked"
            ),
            "top_20_jaccard": top_k_jaccard(rust, arboreto, k=20),
            "top_50_jaccard": top_k_jaccard(rust, arboreto, k=50),
            "top_100_jaccard": top_k_jaccard(rust, arboreto, k=100),
        },
        "path_policy": "portable",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--tool", choices=("rustscenic", "arboreto"), required=True)
    run_parser.add_argument("--prepared", type=Path, required=True)
    run_parser.add_argument("--n-cells", type=int, default=20_000)
    run_parser.add_argument("--n-hvg", type=int, default=500)
    run_parser.add_argument("--n-tfs", type=int, default=64)
    run_parser.add_argument("--threads", type=int, default=16)
    run_parser.add_argument("--n-estimators", type=int, default=5_000)
    run_parser.add_argument("--seed", type=int, default=777)
    run_parser.add_argument("--out-json", type=Path, required=True)
    run_parser.add_argument("--out-parquet", type=Path, required=True)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--rust-json", type=Path, required=True)
    compare_parser.add_argument("--arboreto-json", type=Path, required=True)
    compare_parser.add_argument("--rust-parquet", type=Path, required=True)
    compare_parser.add_argument("--arboreto-parquet", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        if min(args.n_cells, args.n_hvg, args.n_tfs, args.threads, args.n_estimators) < 1:
            raise SystemExit("cell, feature, thread and estimator counts must be positive")
        return run(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
