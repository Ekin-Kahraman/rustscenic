"""Collect and validate Minerva RustScenic benchmark artefacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.hpc.minerva.validate_benchmark_artifact import validate_record


KNOWN_BENCHMARKS = {
    "real_multiome_full_pipeline",
    "real_multiome_full_pipeline_scaling",
    "real_pbmc3k_grn_scaling",
}


def _json_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.json")))
        elif item.is_file():
            paths.append(item)
    return [
        path
        for path in paths
        if not path.name.endswith(".preflight.json")
    ]


def _load_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("benchmark") not in KNOWN_BENCHMARKS:
        return None
    return record


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _format_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _dataset(record: dict[str, Any]) -> str:
    dataset = record.get("dataset_name", record.get("dataset", ""))
    if isinstance(dataset, str):
        return dataset
    return ""


def _range(values: list[Any]) -> str:
    clean = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not clean:
        return ""
    if len(clean) == 1 or clean[0] == clean[-1]:
        return _format_number(clean[0])
    return f"{_format_number(clean[0])}..{_format_number(clean[-1])}"


def _cells(record: dict[str, Any]) -> str:
    benchmark = record.get("benchmark")
    if benchmark == "real_multiome_full_pipeline_scaling":
        runs = record.get("runs", [])
        if isinstance(runs, list):
            return _range([
                row.get("n_cells_actual")
                for row in runs
                if isinstance(row, dict)
            ])
    if benchmark == "real_multiome_full_pipeline":
        shapes = record.get("shapes", {})
        if isinstance(shapes, dict):
            rna = shapes.get("rna_post_qc")
            if isinstance(rna, list) and rna:
                return _format_number(rna[0])
    if benchmark == "real_pbmc3k_grn_scaling":
        rows = record.get("subset_scaling", [])
        if isinstance(rows, list):
            return _range([
                row.get("n_cells")
                for row in rows
                if isinstance(row, dict)
            ])
    return ""


def _wall(record: dict[str, Any]) -> str:
    benchmark = record.get("benchmark")
    if benchmark == "real_multiome_full_pipeline_scaling":
        runs = record.get("runs", [])
        if isinstance(runs, list):
            return _range([
                row.get("wall_s", {}).get("end_to_end")
                for row in runs
                if isinstance(row, dict) and isinstance(row.get("wall_s"), dict)
            ])
    if benchmark == "real_pbmc3k_grn_scaling":
        rows = record.get("subset_scaling", [])
        if isinstance(rows, list):
            return _range([
                row.get("grn_wall_s")
                for row in rows
                if isinstance(row, dict)
            ])
    wall_s = record.get("wall_s")
    if isinstance(wall_s, dict):
        return _format_number(wall_s.get("end_to_end"))
    return ""


def _peak_rss(record: dict[str, Any]) -> str:
    benchmark = record.get("benchmark")
    if benchmark == "real_multiome_full_pipeline_scaling":
        runs = record.get("runs", [])
        if isinstance(runs, list):
            return _range([
                row.get("peak_rss_gb")
                for row in runs
                if isinstance(row, dict)
            ])
    if benchmark == "real_pbmc3k_grn_scaling":
        rows = record.get("subset_scaling", [])
        if isinstance(rows, list):
            return _range([
                row.get("peak_rss_gb")
                for row in rows
                if isinstance(row, dict)
            ])
    return _format_number(record.get("peak_rss_gb"))


def _output_summary(record: dict[str, Any]) -> str:
    outputs = record.get("outputs")
    if isinstance(outputs, dict):
        parts = []
        for key in ("grn_edges", "eregulon_rows", "regulons"):
            value = outputs.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                parts.append(f"{key}={value}")
        return ", ".join(parts)
    if record.get("benchmark") == "real_multiome_full_pipeline_scaling":
        runs = record.get("runs", [])
        if isinstance(runs, list) and runs:
            last = runs[-1]
            if isinstance(last, dict) and isinstance(last.get("outputs"), dict):
                return _output_summary({"outputs": last["outputs"]})
    if record.get("benchmark") == "real_pbmc3k_grn_scaling":
        rows = record.get("subset_scaling", [])
        if isinstance(rows, list):
            edges = _range([
                row.get("edges")
                for row in rows
                if isinstance(row, dict)
            ])
            genes = _range([
                row.get("n_genes")
                for row in rows
                if isinstance(row, dict)
            ])
            tfs = _range([
                row.get("n_tfs")
                for row in rows
                if isinstance(row, dict)
            ])
            parts = []
            if edges:
                parts.append(f"edges={edges}")
            if genes:
                parts.append(f"genes={genes}")
            if tfs:
                parts.append(f"tfs={tfs}")
            return ", ".join(parts)
    return ""


def _scaling_summary(record: dict[str, Any]) -> str:
    if record.get("benchmark") == "real_pbmc3k_grn_scaling":
        parts = []
        for key in (
            "subset_wall_slope_vs_cells",
            "subset_memory_slope_vs_cells",
        ):
            if key in record:
                parts.append(f"{key}={_format_number(record[key])}")
        speedups = record.get("thread_speedups", [])
        if isinstance(speedups, list) and speedups:
            best = max(
                (
                    row
                    for row in speedups
                    if isinstance(row, dict)
                    and isinstance(row.get("speedup_vs_baseline"), (int, float))
                    and not isinstance(row.get("speedup_vs_baseline"), bool)
                ),
                key=lambda row: float(row["speedup_vs_baseline"]),
                default=None,
            )
            if best is not None:
                parts.append(
                    "best_thread_speedup="
                    f"{_format_number(best.get('speedup_vs_baseline'))}"
                    f"@{_format_number(best.get('threads'))}t"
                )
        return ", ".join(parts)

    scaling = record.get("scaling")
    if not isinstance(scaling, dict):
        return ""
    keys = (
        "end_to_end_wall_slope_vs_cells",
        "pipeline_wall_slope_vs_cells",
        "peak_rss_slope_vs_cells",
    )
    parts = [
        f"{key}={_format_number(scaling[key])}"
        for key in keys
        if key in scaling
    ]
    return ", ".join(parts)


def summarise_record(
    path: Path,
    record: dict[str, Any],
    *,
    require_clean: bool,
    check_output_files: bool,
) -> dict[str, Any]:
    failures = validate_record(
        record,
        require_clean=require_clean,
        check_output_files=check_output_files,
    )
    repo = record.get("repo_state", {})
    return {
        "path": str(path),
        "modified_time": _mtime(path),
        "valid": not failures,
        "failures": failures,
        "benchmark": record.get("benchmark", ""),
        "dataset": _dataset(record),
        "commit": repo.get("commit", "") if isinstance(repo, dict) else "",
        "tracked_dirty": repo.get("tracked_dirty") if isinstance(repo, dict) else None,
        "cells": _cells(record),
        "end_to_end_s": _wall(record),
        "peak_rss_gb": _peak_rss(record),
        "scaling": _scaling_summary(record),
        "outputs": _output_summary(record),
    }


def _referenced_child_json_paths(records: list[tuple[Path, dict[str, Any]]]) -> set[Path]:
    paths: set[Path] = set()
    for _path, record in records:
        if record.get("benchmark") != "real_multiome_full_pipeline_scaling":
            continue
        runs = record.get("runs", [])
        if not isinstance(runs, list):
            continue
        for row in runs:
            if not isinstance(row, dict):
                continue
            child = row.get("json_path")
            if isinstance(child, str) and child:
                paths.add(Path(child).resolve())
    return paths


def collect(
    inputs: list[Path],
    *,
    require_clean: bool = True,
    check_output_files: bool = False,
    latest_per_benchmark: bool = False,
) -> list[dict[str, Any]]:
    explicit_files = {
        item.resolve()
        for item in inputs
        if item.is_file()
    }
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in _json_paths(inputs):
        record = _load_record(path)
        if record is not None:
            loaded.append((path, record))
    child_json_paths = _referenced_child_json_paths(loaded)

    rows = []
    for path, record in loaded:
        if path.resolve() in child_json_paths and path.resolve() not in explicit_files:
            continue
        rows.append(
            summarise_record(
                path,
                record,
                require_clean=require_clean,
                check_output_files=check_output_files,
            )
        )
    rows.sort(key=lambda row: (str(row["benchmark"]), float(row["modified_time"]), str(row["path"])))
    if not latest_per_benchmark:
        return rows

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row["benchmark"])] = row
    return sorted(latest.values(), key=lambda row: str(row["benchmark"]))


def _markdown_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("valid", "valid"),
        ("benchmark", "benchmark"),
        ("dataset", "dataset"),
        ("cells", "cells"),
        ("end_to_end_s", "wall_s"),
        ("peak_rss_gb", "rss_gb"),
        ("outputs", "outputs"),
        ("scaling", "scaling"),
        ("path", "path"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, "")
            if key == "valid":
                value = "yes" if value else "no"
            values.append(_markdown_escape(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *body])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--check-output-files", action="store_true")
    parser.add_argument("--latest-per-benchmark", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect(
        args.paths,
        require_clean=not args.allow_dirty,
        check_output_files=args.check_output_files,
        latest_per_benchmark=args.latest_per_benchmark,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    print(markdown_table(rows))
    return 1 if any(row["failures"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
