"""Run real multiome full-pipeline scaling as isolated child jobs.

This coordinator is for HPC batch runs where we need full E2E scaling
numbers, not only isolated GRN timing. Each cell-count point invokes
``bench_real_multiome_pipeline.py`` in a fresh Python process, validates the
child JSON artefact, then writes one aggregate JSON for plotting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.backend_requirements import backend_capabilities
from validation.hpc.minerva.validate_benchmark_artifact import validate_record
from validation.python_hot_paths import hot_path_state
from validation.scaling.bench_real_multiome_pipeline import (
    DEFAULT_SUMMARY_MAX_ROWS,
    benchmark_env,
    repo_state,
    runtime_import_state,
)


CHILD_SCRIPT = REPO_ROOT / "validation/scaling/bench_real_multiome_pipeline.py"


def invocation_state(argv: list[str] | None) -> dict[str, Any]:
    script = Path(__file__).resolve()
    args = [str(value) for value in (sys.argv[1:] if argv is None else argv)]
    return {
        "python": sys.executable,
        "script": str(script),
        "cwd": str(Path.cwd()),
        "argv": args,
        "command": [sys.executable, str(script), *args],
    }


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
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def child_cmd(
    args: argparse.Namespace,
    *,
    n_cells: int,
    run_id: str,
    out_dir: Path,
    out_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(CHILD_SCRIPT),
        "--dataset-name",
        args.dataset_name,
        "--rna-10x-h5",
        str(args.rna_10x_h5),
        "--fragments",
        str(args.fragments),
        "--peaks",
        str(args.peaks),
        "--out-dir",
        str(out_dir),
        "--out-json",
        str(out_json),
        "--run-id",
        run_id,
        "--n-cells",
        str(n_cells),
        "--threads",
        str(args.threads),
        "--grn-n-estimators",
        str(args.grn_n_estimators),
        "--grn-max-features",
        str(args.grn_max_features),
        "--topics-n-topics",
        str(args.topics_n_topics),
        "--topics-n-passes",
        str(args.topics_n_passes),
        "--topics-method",
        args.topics_method,
        "--topics-n-iters",
        str(args.topics_n_iters),
        "--topics-n-threads",
        str(args.topics_n_threads),
        "--cistarget-top-frac",
        str(args.cistarget_top_frac),
        "--cistarget-auc-threshold",
        str(args.cistarget_auc_threshold),
        "--cistarget-nes-threshold",
        str(args.cistarget_nes_threshold),
        "--enhancer-max-distance",
        str(args.enhancer_max_distance),
        "--enhancer-min-abs-corr",
        str(args.enhancer_min_abs_corr),
        "--eregulon-min-target-genes",
        str(args.eregulon_min_target_genes),
        "--eregulon-min-enhancer-links",
        str(args.eregulon_min_enhancer_links),
        "--summary-rows",
        str(args.summary_rows),
        "--seed",
        str(args.seed),
        "--quiet",
    ]
    if args.grn_target_block_size is not None:
        cmd.extend(["--grn-target-block-size", str(args.grn_target_block_size)])
    if args.motif_rankings is not None:
        cmd.extend(["--motif-rankings", str(args.motif_rankings)])
    if args.motif_annotations is not None:
        cmd.extend(["--motif-annotations", str(args.motif_annotations)])
    if args.region_motif_rankings is not None:
        cmd.extend(["--region-motif-rankings", str(args.region_motif_rankings)])
    if args.gene_coords is not None:
        cmd.extend(["--gene-coords", str(args.gene_coords)])
    if args.expected_tfs:
        cmd.extend(["--expected-tfs", *args.expected_tfs])
    if args.summary_max_rows is not None:
        cmd.extend(["--summary-max-rows", str(args.summary_max_rows)])
    if args.skip_integrated_adata:
        cmd.append("--skip-integrated-adata")
    if args.require_clean:
        cmd.append("--require-clean")
    return cmd


def run_child(args: argparse.Namespace, *, n_cells: int) -> dict[str, Any]:
    run_id = f"{args.run_prefix}_cells{n_cells}"
    out_dir = args.out_root / f"{run_id}_outputs"
    out_json = args.out_root / f"{run_id}.json"
    _ensure_fresh_child_outputs(out_json, out_dir, force=args.force)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "RAYON_NUM_THREADS": str(args.threads),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    print(f"\n=== full pipeline: n_cells={n_cells:,}, threads={args.threads} ===", flush=True)
    completed = subprocess.run(
        child_cmd(args, n_cells=n_cells, run_id=run_id, out_dir=out_dir, out_json=out_json),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"full-pipeline child failed for n_cells={n_cells}, "
            f"code={completed.returncode}"
        )

    record = json.loads(out_json.read_text())
    failures = validate_record(
        record,
        require_clean=args.require_clean,
        check_output_files=True,
    )
    if failures:
        raise RuntimeError(
            f"child benchmark artefact failed validation for n_cells={n_cells}: "
            + "; ".join(failures)
        )
    return {
        "n_cells_requested": int(n_cells),
        "n_cells_actual": int(record["shapes"]["rna_post_qc"][0]),
        "threads": int(args.threads),
        "params": record["params"],
        "invocation": record["invocation"],
        "json_path": str(out_json),
        "output_dir": str(out_dir),
        "wall_s": record["wall_s"],
        "peak_rss_gb": record["peak_rss_gb"],
        "setup_peak_rss_gb": record["setup_peak_rss_gb"],
        "setup_elapsed_s": record["setup_elapsed_s"],
        "elapsed_per_stage": record["elapsed_per_stage"],
        "io_elapsed_per_stage": record.get("io_elapsed_per_stage", {}),
        "peak_rss_gb_per_stage": record["peak_rss_gb_per_stage"],
        "peak_rss_gb_delta_per_stage": record["peak_rss_gb_delta_per_stage"],
        "output_storage": record.get("output_storage", {}),
        "backend_execution": record["backend_execution"],
        "cell_barcode_filter": record["cell_barcode_filter"],
        "matrix_inputs": record["matrix_inputs"],
        "cistarget_rankings": record["cistarget_rankings"],
        "reference_sources": record["reference_sources"],
        "outputs": record["outputs"],
        "expected_tf_recovery": record.get("expected_tf_recovery"),
        "env": record["env"],
        "write_integrated_adata": record.get("params", {}).get(
            "write_integrated_adata",
            True,
        ),
    }


def _ensure_fresh_child_outputs(out_json: Path, out_dir: Path, *, force: bool) -> None:
    conflicts: list[str] = []
    if out_json.exists():
        conflicts.append(str(out_json))
    if out_dir.exists() and any(out_dir.iterdir()):
        conflicts.append(str(out_dir))
    if conflicts and not force:
        raise RuntimeError(
            "refusing to reuse existing full-pipeline scaling output(s): "
            + ", ".join(conflicts)
            + ". Choose a new --run-prefix or pass --force to overwrite."
        )
    if force:
        if out_json.exists():
            out_json.unlink()
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)


def configure_thread_env(threads: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def aggregate_payload(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    wall_rows = [
        {
            "n_cells": row["n_cells_actual"],
            "end_to_end_wall_s": row["wall_s"]["end_to_end"],
            "pipeline_wall_s": row["wall_s"]["pipeline"],
            "pipeline_compute_stage_wall_s": row["wall_s"].get(
                "pipeline_compute_stages"
            ),
            "pipeline_io_stage_wall_s": row["wall_s"].get(
                "pipeline_io_stages"
            ),
            "pipeline_unattributed_wall_s": row["wall_s"].get(
                "pipeline_unattributed"
            ),
            "peak_rss_gb": row["peak_rss_gb"],
            "output_total_size_gb": row.get("output_storage", {}).get(
                "total_size_gb"
            ),
        }
        for row in runs
    ]
    return {
        "benchmark": "real_multiome_full_pipeline_scaling",
        "dataset_name": args.dataset_name,
        "run_prefix": args.run_prefix,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_state": repo_state(),
        "invocation": getattr(args, "_invocation", invocation_state(None)),
        "runtime_import": runtime_import_state(),
        "backend_capabilities": backend_capabilities(),
        "python_hot_paths": hot_path_state(),
        "rustscenic": version("rustscenic"),
        "params": {
            "cell_counts": args.cell_counts,
            "threads": args.threads,
            "grn_n_estimators": args.grn_n_estimators,
            "grn_max_features": args.grn_max_features,
            "grn_target_block_size": args.grn_target_block_size,
            "topics_n_topics": args.topics_n_topics,
            "topics_n_passes": args.topics_n_passes,
            "topics_method": args.topics_method,
            "topics_n_iters": args.topics_n_iters,
            "topics_n_threads": args.topics_n_threads,
            "cistarget_top_frac": args.cistarget_top_frac,
            "cistarget_auc_threshold": args.cistarget_auc_threshold,
            "cistarget_nes_threshold": args.cistarget_nes_threshold,
            "motif_annotations": str(args.motif_annotations) if args.motif_annotations else None,
            "region_motif_rankings": str(args.region_motif_rankings) if args.region_motif_rankings else None,
            "enhancer_max_distance": args.enhancer_max_distance,
            "enhancer_min_abs_corr": args.enhancer_min_abs_corr,
            "eregulon_min_target_genes": args.eregulon_min_target_genes,
            "eregulon_min_enhancer_links": args.eregulon_min_enhancer_links,
            "write_integrated_adata": not args.skip_integrated_adata,
            "summary_rows": args.summary_rows,
            "summary_max_rows": args.summary_max_rows,
            "expected_tfs": args.expected_tfs,
            "seed": args.seed,
            "force": args.force,
        },
        "runs": runs,
        "scaling": {
            "end_to_end_wall_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "end_to_end_wall_s",
            ),
            "pipeline_wall_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "pipeline_wall_s",
            ),
            "pipeline_compute_stage_wall_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "pipeline_compute_stage_wall_s",
            ),
            "pipeline_io_stage_wall_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "pipeline_io_stage_wall_s",
            ),
            "pipeline_unattributed_wall_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "pipeline_unattributed_wall_s",
            ),
            "peak_rss_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "peak_rss_gb",
            ),
            "output_total_size_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "output_total_size_gb",
            ),
        },
        "stage_scaling": {
            "elapsed_per_stage_slope_vs_cells": _stage_slopes(
                runs,
                field="elapsed_per_stage",
            ),
            "peak_rss_gb_per_stage_slope_vs_cells": _stage_slopes(
                runs,
                field="peak_rss_gb_per_stage",
            ),
            "peak_rss_gb_delta_per_stage_slope_vs_cells": _stage_slopes(
                runs,
                field="peak_rss_gb_delta_per_stage",
            ),
        },
        "env": benchmark_env(),
    }


def _rounded_slope(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    slope = log_log_slope(rows, x_key, y_key)
    return None if slope is None else round(slope, 3)


def _stage_slopes(runs: list[dict[str, Any]], *, field: str) -> dict[str, float | None]:
    stages = sorted(
        {
            stage
            for row in runs
            if isinstance(row.get(field), dict)
            for stage in row[field]
        }
    )
    return {
        stage: _rounded_slope(
            [
                {
                    "n_cells": row.get("n_cells_actual"),
                    "value": row.get(field, {}).get(stage)
                    if isinstance(row.get(field), dict)
                    else None,
                }
                for row in runs
            ],
            "n_cells",
            "value",
        )
        for stage in stages
    }


def coordinator(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    configure_thread_env(args.threads)
    state = repo_state()
    if args.require_clean and state["source_dirty"]:
        raise SystemExit(
            "source files are dirty; commit tracked changes and add or remove "
            "untracked source files before running a publication-grade "
            "benchmark. Use without --require-clean only for explicit "
            "local-build profiling."
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for n_cells in args.cell_counts:
        runs.append(run_child(args, n_cells=n_cells))
        args.out_json.write_text(json.dumps(aggregate_payload(args, runs), indent=2) + "\n")

    payload = aggregate_payload(args, runs)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    failures = validate_record(
        payload,
        require_clean=args.require_clean,
        check_output_files=True,
    )
    if failures:
        raise RuntimeError(
            "aggregate full-pipeline scaling artefact failed validation: "
            + "; ".join(failures)
        )
    return payload


def validate_args(args: argparse.Namespace) -> None:
    if any(n <= 0 for n in args.cell_counts):
        raise SystemExit("--cell-counts must be positive integers")
    if args.cell_counts != sorted(args.cell_counts):
        raise SystemExit("--cell-counts must be sorted ascending")
    if len(set(args.cell_counts)) != len(args.cell_counts):
        raise SystemExit("--cell-counts must not contain duplicates")
    positive_int_fields = {
        "threads": args.threads,
        "grn_n_estimators": args.grn_n_estimators,
        "topics_n_topics": args.topics_n_topics,
        "topics_n_passes": args.topics_n_passes,
        "topics_n_iters": args.topics_n_iters,
        "topics_n_threads": args.topics_n_threads,
        "eregulon_min_target_genes": args.eregulon_min_target_genes,
        "eregulon_min_enhancer_links": args.eregulon_min_enhancer_links,
        "summary_rows": args.summary_rows,
    }
    for name, value in positive_int_fields.items():
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.grn_target_block_size is not None and args.grn_target_block_size <= 0:
        raise SystemExit("--grn-target-block-size must be positive when set")
    if args.summary_max_rows is not None and args.summary_max_rows <= 0:
        raise SystemExit("--summary-max-rows must be positive when set")
    if not (0 < args.grn_max_features <= 1):
        raise SystemExit("--grn-max-features must be in (0, 1]")
    if not (0 < args.cistarget_top_frac <= 1):
        raise SystemExit("--cistarget-top-frac must be in (0, 1]")
    if args.cistarget_auc_threshold < 0:
        raise SystemExit("--cistarget-auc-threshold must be non-negative")
    if args.cistarget_nes_threshold < 0:
        raise SystemExit("--cistarget-nes-threshold must be non-negative")
    if args.enhancer_max_distance < 0:
        raise SystemExit("--enhancer-max-distance must be non-negative")
    if args.enhancer_min_abs_corr < 0:
        raise SystemExit("--enhancer-min-abs-corr must be non-negative")
    for name in (
        "motif_rankings",
        "motif_annotations",
        "region_motif_rankings",
        "gene_coords",
    ):
        path = getattr(args, name)
        if path is not None and not path.exists():
            raise SystemExit(f"missing input: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--rna-10x-h5", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--peaks", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--run-prefix", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cell-counts", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--motif-rankings", type=Path, default=None)
    parser.add_argument("--motif-annotations", type=Path, default=None)
    parser.add_argument("--region-motif-rankings", type=Path, default=None)
    parser.add_argument("--gene-coords", type=Path, default=None)
    parser.add_argument("--expected-tfs", nargs="*", default=[])
    parser.add_argument("--grn-n-estimators", type=int, default=100)
    parser.add_argument("--grn-max-features", type=float, default=0.1)
    parser.add_argument("--grn-target-block-size", type=int, default=None)
    parser.add_argument("--topics-n-topics", type=int, default=10)
    parser.add_argument("--topics-n-passes", type=int, default=3)
    parser.add_argument("--topics-method", choices=["vb", "gibbs"], default="vb")
    parser.add_argument("--topics-n-iters", type=int, default=200)
    parser.add_argument("--topics-n-threads", type=int, default=1)
    parser.add_argument("--cistarget-top-frac", type=float, default=0.05)
    parser.add_argument("--cistarget-auc-threshold", type=float, default=0.05)
    parser.add_argument("--cistarget-nes-threshold", type=float, default=3.0)
    parser.add_argument("--enhancer-max-distance", type=int, default=500_000)
    parser.add_argument("--enhancer-min-abs-corr", type=float, default=0.1)
    parser.add_argument("--eregulon-min-target-genes", type=int, default=2)
    parser.add_argument("--eregulon-min-enhancer-links", type=int, default=1)
    parser.add_argument("--summary-rows", type=int, default=10)
    parser.add_argument("--summary-max-rows", type=int, default=DEFAULT_SUMMARY_MAX_ROWS)
    parser.add_argument("--skip-integrated-adata", action="store_true")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete existing child output JSON/directories for this run-prefix before rerunning",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail if tracked files differ from HEAD; use for publication-grade runs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args._invocation = invocation_state(argv)
    validate_args(args)
    for path in (args.rna_10x_h5, args.fragments, args.peaks):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    payload = coordinator(args)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
