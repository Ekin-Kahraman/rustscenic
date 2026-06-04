"""Validate RustScenic HPC benchmark JSON artefacts.

This is the gate between "an LSF job wrote a JSON file" and "the result can
feed benchmark tables or public claims". It intentionally checks for clean
tracked source, required timing/RSS fields, and non-empty biological outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
import string

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.backend_requirements import REQUIRED_RUST_BACKEND_SYMBOLS


FULL_PIPELINE_STAGES = {
    "load_rna",
    "preproc",
    "topics",
    "grn",
    "candidate_regulons",
    "cistarget",
    "enhancer",
    "eregulons",
    "aucell",
    "integrated_adata",
}
REQUIRED_FULL_PIPELINE_MEMORY_STAGES = FULL_PIPELINE_STAGES
REQUIRED_FULL_PIPELINE_ELAPSED_STAGES = {
    "preproc",
    "topics",
    "grn",
    "cistarget",
    "enhancer",
    "eregulons",
    "aucell",
}
REQUIRED_FULL_PIPELINE_SETUP_STAGES = {
    "load_rna_qc",
    "fragments_to_matrix",
    "subset_shared_cells",
    "motif_rankings",
    "gene_coords",
    "tf_list",
}
REQUIRED_BACKEND_SYMBOLS = {
    stage: set(symbols)
    for stage, symbols in REQUIRED_RUST_BACKEND_SYMBOLS.items()
}
REQUIRED_FULL_PIPELINE_ARTEFACTS = {
    "atac_matrix_path": "file",
    "grn_path": "file",
    "regulons_path": "file",
    "candidate_regulons_path": "file",
    "aucell_path": "file",
    "topics_dir": "dir",
    "cistarget_path": "file",
    "enhancer_links_path": "file",
    "eregulons_path": "file",
    "integrated_adata_path": "file",
}
REQUIRED_FULL_PIPELINE_RUST_EXECUTION = {
    "setup_fragments_to_matrix",
    "pipeline_topics",
    "pipeline_grn",
    "pipeline_candidate_regulons",
    "pipeline_cistarget",
    "pipeline_enhancer",
    "pipeline_eregulon_peak_attribution",
    "pipeline_eregulons",
    "pipeline_aucell",
}
REQUIRED_FULL_PIPELINE_RUST_STAGE_SYMBOLS = {
    "setup_fragments_to_matrix": {
        "all_of": {"preproc_fragments_to_matrix"},
    },
    "pipeline_topics": {
        "any_of": ({"topics_fit"}, {"topics_fit_gibbs"}),
    },
    "pipeline_grn": {
        "all_of": {"gene_duplicate_summary"},
        "any_of": ({"grn_infer"}, {"grn_infer_sparse_csc"}),
    },
    "pipeline_candidate_regulons": {
        "all_of": {"pipeline_candidate_regulons_from_grn"},
    },
    "pipeline_cistarget": {
        "any_of": (
            {"cistarget_enrichment_from_rankings_i16"},
            {"cistarget_enrichment_from_rankings_i32"},
            {"cistarget_enrichment_from_rankings_i64"},
        ),
    },
    "pipeline_cistarget_pruning": {
        "one_from_each": (
            {
                "cistarget_motif_annotation_prune_standard_rows_f32",
                "cistarget_motif_annotation_prune_standard_rows_f64",
                "cistarget_motif_annotation_prune_rows_filtered_f32",
                "cistarget_motif_annotation_prune_rows_filtered_f64",
            },
            {
                "cistarget_prune_regulon_targets_f32",
                "cistarget_prune_regulon_targets_f64",
                "cistarget_prune_regulon_targets_i16",
                "cistarget_prune_regulon_targets_i32",
                "cistarget_prune_regulon_targets_i64",
                "cistarget_prune_regulon_targets_unranked",
            },
        ),
    },
    "pipeline_enhancer": {
        "all_of": {
            "enhancer_align_cell_indices",
            "enhancer_match_gene_coords_to_rna",
            "enhancer_normalise_chrom_codes",
            "enhancer_prepare_gene_order",
            "preproc_peak_coords_for_names",
        },
        "any_of": (
            {"enhancer_link_pearson"},
            {"enhancer_link_pearson_sparse_rna"},
        ),
    },
    "pipeline_eregulon_peak_attribution": {
        "any_of": (
            {"pipeline_attribute_peaks_to_cistarget_rows_f32"},
            {"pipeline_attribute_peaks_to_cistarget_rows_f64"},
            {"pipeline_expand_region_cistarget_rows_f32"},
            {"pipeline_expand_region_cistarget_rows_f64"},
        ),
    },
    "pipeline_eregulons": {
        "any_of": ({"eregulon_assemble"}, {"eregulon_assemble_f32"}),
    },
    "pipeline_aucell": {
        "all_of": {
            "gene_duplicate_summary",
            "stage_prepare_regulon_indices_with_coverage",
        },
        "any_of": ({"aucell_score"}, {"aucell_score_sparse_csr"}),
    },
}


def _positive_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0


def _shape2(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_positive_int(v) for v in value)
    )


def _sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in string.hexdigits for ch in value)
    )


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _repo_failures(record: dict[str, Any], *, require_clean: bool) -> list[str]:
    failures: list[str] = []
    repo = record.get("repo_state")
    if not isinstance(repo, dict):
        return ["missing repo_state"]
    if not repo.get("commit"):
        failures.append("repo_state.commit missing")
    tracked_source_count = repo.get("tracked_source_count")
    if isinstance(tracked_source_count, int) and not isinstance(tracked_source_count, bool):
        if require_clean and tracked_source_count > 0:
            failures.append("repo_state.tracked_source_count must be 0")
    elif require_clean and repo.get("tracked_dirty") is not False:
        failures.append("repo_state.tracked_dirty must be false")
    untracked_source_count = repo.get("untracked_source_count")
    if (
        require_clean
        and isinstance(untracked_source_count, int)
        and not isinstance(untracked_source_count, bool)
        and untracked_source_count > 0
    ):
        failures.append("repo_state.untracked_source_count must be 0")
    return failures


def _runtime_import_failures(record: dict[str, Any], prefix: str) -> list[str]:
    failures: list[str] = []
    runtime = record.get("runtime_import")
    if not isinstance(runtime, dict):
        return [f"{prefix}.runtime_import missing"]
    if runtime.get("extension_error"):
        failures.append(f"{prefix}.runtime_import.extension_error: {runtime['extension_error']}")
    if runtime.get("package_under_repo") is not True:
        failures.append(
            f"{prefix}.runtime_import.package_under_repo must be true"
        )
    if runtime.get("extension_under_repo") is not True:
        failures.append(
            f"{prefix}.runtime_import.extension_under_repo must be true"
        )
    package_version = runtime.get("package_version")
    extension_version = runtime.get("extension_version")
    if not package_version:
        failures.append(f"{prefix}.runtime_import.package_version missing")
    if not extension_version:
        failures.append(f"{prefix}.runtime_import.extension_version missing")
    if package_version and extension_version and package_version != extension_version:
        failures.append(
            f"{prefix}.runtime_import package/extension version mismatch: "
            f"{package_version} != {extension_version}"
        )
    if not runtime.get("package_file"):
        failures.append(f"{prefix}.runtime_import.package_file missing")
    if not runtime.get("extension_file"):
        failures.append(f"{prefix}.runtime_import.extension_file missing")
    return failures


def _backend_failures(record: dict[str, Any], prefix: str) -> list[str]:
    failures: list[str] = []
    backend = record.get("backend_capabilities")
    if not isinstance(backend, dict):
        return [f"{prefix}.backend_capabilities missing"]
    if backend.get("ok") is not True:
        failures.append(f"{prefix}.backend_capabilities.ok must be true")
    if backend.get("extension_error"):
        failures.append(
            f"{prefix}.backend_capabilities.extension_error: {backend['extension_error']}"
        )
    missing_symbols = backend.get("missing_symbols")
    if not isinstance(missing_symbols, list):
        failures.append(f"{prefix}.backend_capabilities.missing_symbols must be a list")
    elif missing_symbols:
        failures.append(
            f"{prefix}.backend_capabilities.missing_symbols must be empty: {missing_symbols}"
        )
    required = backend.get("required_symbols")
    if not isinstance(required, dict):
        failures.append(f"{prefix}.backend_capabilities.required_symbols must be an object")
    else:
        missing_stages = set(REQUIRED_BACKEND_SYMBOLS) - set(required)
        failures.extend(
            f"{prefix}.backend_capabilities.required_symbols.{stage} missing"
            for stage in sorted(missing_stages)
        )
        for stage in sorted(set(REQUIRED_BACKEND_SYMBOLS) & set(required)):
            symbols = required.get(stage)
            if not isinstance(symbols, list) or not symbols:
                failures.append(
                    f"{prefix}.backend_capabilities.required_symbols.{stage} "
                    "must contain at least one symbol"
                )
                continue
            missing = REQUIRED_BACKEND_SYMBOLS[stage] - set(symbols)
            failures.extend(
                f"{prefix}.backend_capabilities.required_symbols.{stage}.{symbol} missing"
                for symbol in sorted(missing)
            )
    return failures


def _python_hot_path_failures(record: dict[str, Any], prefix: str) -> list[str]:
    failures: list[str] = []
    state = record.get("python_hot_paths")
    if not isinstance(state, dict):
        return [f"{prefix}.python_hot_paths missing"]
    if state.get("exists") is not True:
        failures.append(f"{prefix}.python_hot_paths.exists must be true")
    if state.get("ok") is not True:
        failures.append(f"{prefix}.python_hot_paths.ok must be true")
    count = state.get("violation_count")
    if not isinstance(count, int) or isinstance(count, bool):
        failures.append(f"{prefix}.python_hot_paths.violation_count must be an integer")
    elif count != 0:
        failures.append(f"{prefix}.python_hot_paths.violation_count must be 0")
    violations = state.get("violations")
    if not isinstance(violations, list):
        failures.append(f"{prefix}.python_hot_paths.violations must be a list")
    elif violations:
        failures.append(
            f"{prefix}.python_hot_paths.violations must be empty: {violations[:5]}"
        )
    if not _positive_int(state.get("allowed_hit_count")):
        failures.append(f"{prefix}.python_hot_paths.allowed_hit_count must be positive")
    if not _positive_int(state.get("pattern_count")):
        failures.append(f"{prefix}.python_hot_paths.pattern_count must be positive")
    if not _nonempty_str(state.get("package_dir")):
        failures.append(f"{prefix}.python_hot_paths.package_dir must be a non-empty string")
    return failures


def _backend_execution_failures(
    record: dict[str, Any],
    prefix: str,
    required_rust_stages: set[str] | None = None,
) -> list[str]:
    execution = record.get("backend_execution")
    if not isinstance(execution, dict):
        return [f"{prefix}.backend_execution missing"]

    failures: list[str] = []
    for stage in sorted(required_rust_stages or set()):
        state = execution.get(stage)
        stage_prefix = f"{prefix}.backend_execution.{stage}"
        if not isinstance(state, dict):
            failures.append(f"{stage_prefix} must be an object")
            continue
        if state.get("engine") != "rust":
            failures.append(f"{stage_prefix}.engine must be 'rust'")
        symbols = state.get("symbols")
        if (
            not isinstance(symbols, list)
            or not symbols
            or not all(_nonempty_str(symbol) for symbol in symbols)
        ):
            failures.append(f"{stage_prefix}.symbols must be a non-empty string list")
            continue
        failures.extend(_backend_execution_symbol_failures(stage, symbols, stage_prefix))
    for stage, state in sorted(execution.items()):
        if stage in (required_rust_stages or set()):
            continue
        if stage not in REQUIRED_FULL_PIPELINE_RUST_STAGE_SYMBOLS:
            continue
        if not isinstance(state, dict) or state.get("engine") != "rust":
            continue
        symbols = state.get("symbols")
        stage_prefix = f"{prefix}.backend_execution.{stage}"
        if (
            not isinstance(symbols, list)
            or not symbols
            or not all(_nonempty_str(symbol) for symbol in symbols)
        ):
            failures.append(f"{stage_prefix}.symbols must be a non-empty string list")
            continue
        failures.extend(_backend_execution_symbol_failures(stage, symbols, stage_prefix))
    return failures


def _backend_execution_symbol_failures(
    stage: str,
    symbols: list[str],
    stage_prefix: str,
) -> list[str]:
    requirements = REQUIRED_FULL_PIPELINE_RUST_STAGE_SYMBOLS.get(stage)
    if not requirements:
        return []

    failures: list[str] = []
    symbol_set = set(symbols)
    missing_required = set(requirements.get("all_of", set())) - symbol_set
    failures.extend(
        f"{stage_prefix}.symbols missing required Rust symbol {symbol!r}"
        for symbol in sorted(missing_required)
    )

    any_of = requirements.get("any_of", ())
    if any_of and not any(set(option) <= symbol_set for option in any_of):
        options = [
            "{" + ", ".join(sorted(repr(symbol) for symbol in option)) + "}"
            for option in any_of
        ]
        failures.append(
            f"{stage_prefix}.symbols must include at least one Rust symbol set "
            f"from {options}"
        )
    for group in requirements.get("one_from_each", ()):
        group = set(group)
        if not group & symbol_set:
            options = ", ".join(sorted(repr(symbol) for symbol in group))
            failures.append(
                f"{stage_prefix}.symbols must include at least one Rust symbol "
                f"from {{{options}}}"
            )
    return failures


def _require_keys(record: dict[str, Any], keys: set[str], prefix: str) -> list[str]:
    return [f"{prefix}.{key} missing" for key in sorted(keys) if key not in record]


def _reference_fingerprint_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    fingerprints = record.get("reference_fingerprints")
    shapes = record.get("shapes", {})
    if not isinstance(fingerprints, dict):
        return ["reference_fingerprints must be an object"]
    keys = ["motif_rankings", "gene_coords"]
    if "motif_annotations" in fingerprints:
        keys.append("motif_annotations")
    for key in keys:
        fp = fingerprints.get(key)
        if not isinstance(fp, dict):
            failures.append(f"reference_fingerprints.{key} must be an object")
            continue
        if not _shape2(fp.get("shape")):
            failures.append(f"reference_fingerprints.{key}.shape must be [positive_rows, positive_cols]")
        if not _sha256_hex(fp.get("corner_sample_sha256")):
            failures.append(f"reference_fingerprints.{key}.corner_sample_sha256 must be sha256 hex")
        if not isinstance(fp.get("index_sample"), list) or not fp["index_sample"]:
            failures.append(f"reference_fingerprints.{key}.index_sample must be a non-empty list")
        if not isinstance(fp.get("column_sample"), list) or not fp["column_sample"]:
            failures.append(f"reference_fingerprints.{key}.column_sample must be a non-empty list")
        if not isinstance(fp.get("dtype_counts"), dict) or not fp["dtype_counts"]:
            failures.append(f"reference_fingerprints.{key}.dtype_counts must be a non-empty object")

    if isinstance(shapes, dict):
        motif_shape = shapes.get("motif_rankings")
        motif_fp = fingerprints.get("motif_rankings")
        if isinstance(motif_fp, dict) and _shape2(motif_shape) and motif_fp.get("shape") != motif_shape:
            failures.append("reference_fingerprints.motif_rankings.shape must match shapes.motif_rankings")
        annotations_shape = shapes.get("motif_annotations")
        annotations_fp = fingerprints.get("motif_annotations")
        if (
            isinstance(annotations_fp, dict)
            and _shape2(annotations_shape)
            and annotations_fp.get("shape") != annotations_shape
        ):
            failures.append(
                "reference_fingerprints.motif_annotations.shape must match shapes.motif_annotations"
            )
        gene_rows = shapes.get("gene_coords_rows")
        gene_fp = fingerprints.get("gene_coords")
        if isinstance(gene_fp, dict) and _positive_int(gene_rows) and _shape2(gene_fp.get("shape")):
            if gene_fp["shape"][0] != gene_rows:
                failures.append("reference_fingerprints.gene_coords.shape rows must match shapes.gene_coords_rows")
    return failures


def _motif_annotations_supplied(record: dict[str, Any]) -> bool:
    """Return true when a benchmark artefact says annotation pruning was requested."""
    params = record.get("params")
    if isinstance(params, dict):
        motif_annotations = params.get("motif_annotations")
        if isinstance(motif_annotations, str) and motif_annotations.strip():
            return True
    for section in ("reference_fingerprints", "shapes", "setup_elapsed_s"):
        value = record.get(section)
        if isinstance(value, dict) and "motif_annotations" in value:
            return True
    return False


def _motif_annotation_pruning_failures(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:
    if not _motif_annotations_supplied(record):
        return []
    return _backend_execution_failures(
        record,
        prefix,
        {"pipeline_cistarget_pruning"},
    )


def _output_summary_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    summaries = record.get("output_summaries")
    outputs = record.get("outputs", {})
    if not isinstance(summaries, dict):
        return ["output_summaries must be an object"]
    required = {
        "active_regulons_sample": "regulons",
        "top_grn_edges": "grn_edges",
        "top_cistarget_rows": "cistarget_rows",
        "top_enhancer_links": "enhancer_links",
        "top_eregulon_rows": "eregulon_rows",
    }
    for summary_key, output_key in required.items():
        value = summaries.get(summary_key)
        if not isinstance(value, list):
            failures.append(f"output_summaries.{summary_key} must be a list")
            continue
        if _positive_int(outputs.get(output_key)) and not value:
            failures.append(f"output_summaries.{summary_key} must be non-empty")
    for summary_key in (
        "top_grn_edges",
        "top_cistarget_rows",
        "top_enhancer_links",
        "top_eregulon_rows",
    ):
        value = summaries.get(summary_key)
        if isinstance(value, list) and value and not all(isinstance(row, dict) for row in value):
            failures.append(f"output_summaries.{summary_key} rows must be objects")
    return failures


def _expected_tf_recovery_failures(record: dict[str, Any], prefix: str) -> list[str]:
    failures: list[str] = []
    recovery = record.get("expected_tf_recovery")
    if not isinstance(recovery, dict):
        return [f"{prefix}.expected_tf_recovery must be an object"]

    expected = recovery.get("expected_tfs")
    found = recovery.get("found")
    missing = recovery.get("missing")
    fraction = recovery.get("fraction")
    for key, values in (
        ("expected_tfs", expected),
        ("found", found),
        ("missing", missing),
    ):
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            failures.append(f"{prefix}.expected_tf_recovery.{key} must be a list of strings")

    if not all(isinstance(values, list) for values in (expected, found, missing)):
        return failures

    expected_set = set(expected)
    found_set = set(found)
    missing_set = set(missing)
    if len(expected_set) != len(expected):
        failures.append(f"{prefix}.expected_tf_recovery.expected_tfs must not contain duplicates")
    if len(found_set) != len(found):
        failures.append(f"{prefix}.expected_tf_recovery.found must not contain duplicates")
    if len(missing_set) != len(missing):
        failures.append(f"{prefix}.expected_tf_recovery.missing must not contain duplicates")
    if not found_set.issubset(expected_set):
        failures.append(f"{prefix}.expected_tf_recovery.found must be a subset of expected_tfs")
    if not missing_set.issubset(expected_set):
        failures.append(f"{prefix}.expected_tf_recovery.missing must be a subset of expected_tfs")
    if found_set & missing_set:
        failures.append(f"{prefix}.expected_tf_recovery.found and missing must be disjoint")
    if found_set | missing_set != expected_set:
        failures.append(f"{prefix}.expected_tf_recovery.found and missing must cover expected_tfs")

    expected_fraction = None if not expected else round(len(found_set) / len(expected_set), 6)
    if expected_fraction is None:
        if fraction is not None:
            failures.append(f"{prefix}.expected_tf_recovery.fraction must be null when expected_tfs is empty")
    elif not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        failures.append(f"{prefix}.expected_tf_recovery.fraction must be numeric")
    elif round(float(fraction), 6) != expected_fraction:
        failures.append(
            f"{prefix}.expected_tf_recovery.fraction must match found/expected: "
            f"{fraction} != {expected_fraction}"
        )
    return failures


def _thread_budget_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    params_key: str,
) -> list[str]:
    failures: list[str] = []
    params = record.get("params")
    env = record.get("env")
    if not isinstance(params, dict):
        if params_key in record:
            params = record
        else:
            return [f"{prefix}.params must be an object"]

    expected_threads = params.get(params_key)
    if not _positive_int(expected_threads):
        failures.append(f"{prefix}.params.{params_key} must be positive")
        return failures

    if not isinstance(env, dict):
        failures.append(f"{prefix}.env must be an object")
        return failures

    rayon_threads = env.get("rayon_num_threads")
    try:
        actual_threads = int(rayon_threads)
    except (TypeError, ValueError):
        failures.append(f"{prefix}.env.rayon_num_threads must be a positive integer string")
    else:
        if actual_threads <= 0:
            failures.append(f"{prefix}.env.rayon_num_threads must be positive")
        elif actual_threads != expected_threads:
            failures.append(
                f"{prefix}.env.rayon_num_threads must match params.{params_key}: "
                f"{actual_threads} != {expected_threads}"
            )

    for key in ("omp_num_threads", "openblas_num_threads", "mkl_num_threads"):
        if env.get(key) != "1":
            failures.append(f"{prefix}.env.{key} must be '1' for reproducible CPU benchmarking")
    return failures


def _output_inventory_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    inventory = record.get("output_inventory")
    if not isinstance(inventory, dict):
        return ["output_inventory must be an object when --check-output-files is set"]

    for key, expected_type in sorted(REQUIRED_FULL_PIPELINE_ARTEFACTS.items()):
        info = inventory.get(key)
        if not isinstance(info, dict):
            failures.append(f"output_inventory.{key} must be an object")
            continue
        path = info.get("path")
        if not isinstance(path, str) or not path:
            failures.append(f"output_inventory.{key}.path must be a non-empty string")
            continue
        if info.get("exists") is not True:
            failures.append(f"output_inventory.{key} must exist: {path}")
            continue
        if info.get("type") != expected_type:
            failures.append(
                f"output_inventory.{key}.type must be {expected_type!r}"
            )
        actual_path = Path(path)
        if expected_type == "file":
            if not _positive_int(info.get("size_bytes")):
                failures.append(f"output_inventory.{key}.size_bytes must be positive")
            if not actual_path.is_file():
                failures.append(f"output_inventory.{key} path is not a file: {path}")
            else:
                actual_size = actual_path.stat().st_size
                if actual_size <= 0:
                    failures.append(f"output_inventory.{key} file is empty: {path}")
                elif _positive_int(info.get("size_bytes")) and info["size_bytes"] != actual_size:
                    failures.append(
                        f"output_inventory.{key}.size_bytes does not match "
                        f"live file size: {info['size_bytes']} != {actual_size}"
                    )
        else:
            if not _positive_int(info.get("entries")):
                failures.append(f"output_inventory.{key}.entries must be positive")
            if not actual_path.is_dir():
                failures.append(f"output_inventory.{key} path is not a directory: {path}")
            else:
                actual_entries = sum(1 for _ in actual_path.iterdir())
                if actual_entries <= 0:
                    failures.append(f"output_inventory.{key} directory is empty: {path}")
                elif _positive_int(info.get("entries")) and info["entries"] != actual_entries:
                    failures.append(
                        f"output_inventory.{key}.entries does not match "
                        f"live directory entries: {info['entries']} != {actual_entries}"
                    )
    return failures


def _log_log_slope(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
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


def _rounded_slope(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    slope = _log_log_slope(rows, x_key, y_key)
    return None if slope is None else round(slope, 3)


def _full_pipeline_scaling_row_child_failures(
    row: dict[str, Any],
    child: dict[str, Any],
    prefix: str,
) -> list[str]:
    failures: list[str] = []
    shapes = child.get("shapes", {})
    child_cells = shapes.get("rna_post_qc", [None])[0] if isinstance(shapes, dict) else None
    child_requested = child.get("params", {}).get("n_cells_requested")
    checks = {
        "n_cells_actual": child_cells,
        "n_cells_requested": child_requested,
        "wall_s": child.get("wall_s"),
        "peak_rss_gb": child.get("peak_rss_gb"),
        "setup_peak_rss_gb": child.get("setup_peak_rss_gb"),
        "elapsed_per_stage": child.get("elapsed_per_stage"),
        "peak_rss_gb_per_stage": child.get("peak_rss_gb_per_stage"),
        "outputs": child.get("outputs"),
        "expected_tf_recovery": child.get("expected_tf_recovery"),
        "backend_execution": child.get("backend_execution"),
    }
    for key, expected in checks.items():
        if row.get(key) != expected:
            failures.append(f"{prefix}.{key} must match child JSON")

    output_dir = row.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        output_root = Path(output_dir).resolve()
        inventory = child.get("output_inventory", {})
        if isinstance(inventory, dict):
            for key, info in inventory.items():
                if not isinstance(info, dict) or not isinstance(info.get("path"), str):
                    continue
                try:
                    Path(info["path"]).resolve().relative_to(output_root)
                except ValueError:
                    failures.append(
                        f"{prefix}.output_dir must contain child output_inventory.{key}"
                    )
    return failures


def _full_pipeline_scaling_row_failures(
    row: dict[str, Any],
    prefix: str,
    *,
    require_motif_pruning: bool = False,
) -> list[str]:
    failures: list[str] = []
    n_cells = row.get("n_cells_actual")
    if not _positive_int(row.get("n_cells_requested")):
        failures.append(f"{prefix}.n_cells_requested must be positive")

    if not _positive_number(row.get("setup_peak_rss_gb")):
        failures.append(f"{prefix}.setup_peak_rss_gb must be positive")
    elapsed = row.get("elapsed_per_stage")
    if not isinstance(elapsed, dict):
        failures.append(f"{prefix}.elapsed_per_stage must be an object")
    else:
        missing = REQUIRED_FULL_PIPELINE_ELAPSED_STAGES - set(elapsed)
        failures.extend(
            f"{prefix}.elapsed_per_stage.{stage} missing"
            for stage in sorted(missing)
        )
        for stage in sorted(REQUIRED_FULL_PIPELINE_ELAPSED_STAGES & set(elapsed)):
            if not _nonnegative_number(elapsed.get(stage)):
                failures.append(f"{prefix}.elapsed_per_stage.{stage} must be non-negative")

    memory = row.get("peak_rss_gb_per_stage")
    if not isinstance(memory, dict):
        failures.append(f"{prefix}.peak_rss_gb_per_stage must be an object")
    else:
        missing = REQUIRED_FULL_PIPELINE_MEMORY_STAGES - set(memory)
        failures.extend(
            f"{prefix}.peak_rss_gb_per_stage.{stage} missing"
            for stage in sorted(missing)
        )
        for stage in sorted(REQUIRED_FULL_PIPELINE_MEMORY_STAGES & set(memory)):
            if not _positive_number(memory.get(stage)):
                failures.append(f"{prefix}.peak_rss_gb_per_stage.{stage} must be positive")
        unknown = set(memory) - FULL_PIPELINE_STAGES
        failures.extend(
            f"{prefix}.unknown peak_rss_gb_per_stage.{stage}"
            for stage in sorted(unknown)
        )

    outputs = row.get("outputs")
    if not isinstance(outputs, dict):
        failures.append(f"{prefix}.outputs must be an object")
    else:
        if not _positive_int(outputs.get("grn_edges")):
            failures.append(f"{prefix}.outputs.grn_edges must be positive")
        if not _positive_int(outputs.get("regulons")):
            failures.append(f"{prefix}.outputs.regulons must be positive")
        for key in ("cistarget_rows", "enhancer_links", "eregulon_rows", "eregulons"):
            if not _positive_int(outputs.get(key)):
                failures.append(f"{prefix}.outputs.{key} must be positive")
        aucell_shape = outputs.get("aucell_shape")
        if (
            not isinstance(aucell_shape, list)
            or len(aucell_shape) != 2
            or not all(_positive_int(v) for v in aucell_shape)
        ):
            failures.append(f"{prefix}.outputs.aucell_shape must be [positive_cells, positive_regulons]")
        elif _positive_int(n_cells) and aucell_shape[0] != n_cells:
            failures.append(f"{prefix}.outputs.aucell_shape cells must equal n_cells_actual")

    failures.extend(_expected_tf_recovery_failures(row, prefix))
    failures.extend(
        _backend_execution_failures(
            row,
            prefix,
            REQUIRED_FULL_PIPELINE_RUST_EXECUTION,
        )
    )
    if require_motif_pruning:
        failures.extend(
            _backend_execution_failures(
                row,
                prefix,
                {"pipeline_cistarget_pruning"},
            )
        )
    return failures


def validate_full_pipeline(
    record: dict[str, Any],
    *,
    require_clean: bool,
    check_output_files: bool,
) -> list[str]:
    failures = _repo_failures(record, require_clean=require_clean)
    failures.extend(_runtime_import_failures(record, "full_pipeline"))
    failures.extend(_backend_failures(record, "full_pipeline"))
    failures.extend(_python_hot_path_failures(record, "full_pipeline"))
    failures.extend(
        _backend_execution_failures(
            record,
            "full_pipeline",
            REQUIRED_FULL_PIPELINE_RUST_EXECUTION,
        )
    )
    failures.extend(_motif_annotation_pruning_failures(record, "full_pipeline"))
    failures.extend(
        _require_keys(
            record,
            {
                "rustscenic",
                "dataset_name",
                "runtime_import",
                "backend_capabilities",
                "python_hot_paths",
                "backend_execution",
                "input_hashes",
                "reference_fingerprints",
                "params",
                "shapes",
                "wall_s",
                "setup_elapsed_s",
                "setup_peak_rss_gb",
                "peak_rss_gb",
                "elapsed_per_stage",
                "peak_rss_gb_per_stage",
                "outputs",
                "expected_tf_recovery",
                "output_summaries",
                "env",
            },
            "full_pipeline",
        )
    )

    if record.get("benchmark") != "real_multiome_full_pipeline":
        failures.append("benchmark must be real_multiome_full_pipeline")
    if not _nonempty_str(record.get("dataset_name")):
        failures.append("dataset_name must be a non-empty string")
    failures.extend(_reference_fingerprint_failures(record))
    failures.extend(_expected_tf_recovery_failures(record, "full_pipeline"))
    failures.extend(_thread_budget_failures(record, "full_pipeline", params_key="threads"))
    failures.extend(_output_summary_failures(record))

    wall = record.get("wall_s", {})
    if not _positive_number(wall.get("setup")):
        failures.append("wall_s.setup must be positive")
    if not _positive_number(wall.get("pipeline")):
        failures.append("wall_s.pipeline must be positive")
    if not _positive_number(wall.get("end_to_end")):
        failures.append("wall_s.end_to_end must be positive")
    elif _positive_number(wall.get("setup")) and _positive_number(wall.get("pipeline")):
        if wall["end_to_end"] + 0.25 < wall["setup"] + wall["pipeline"]:
            failures.append("wall_s.end_to_end must include setup + pipeline time")
    if not _positive_number(record.get("setup_peak_rss_gb")):
        failures.append("setup_peak_rss_gb must be positive")
    if not _positive_number(record.get("peak_rss_gb")):
        failures.append("peak_rss_gb must be positive")

    setup_elapsed = record.get("setup_elapsed_s", {})
    elapsed = record.get("elapsed_per_stage", {})
    memory = record.get("peak_rss_gb_per_stage", {})
    if not isinstance(setup_elapsed, dict):
        failures.append("setup_elapsed_s must be an object")
    else:
        missing = REQUIRED_FULL_PIPELINE_SETUP_STAGES - set(setup_elapsed)
        failures.extend(f"setup_elapsed_s.{stage} missing" for stage in sorted(missing))
        for stage in sorted(REQUIRED_FULL_PIPELINE_SETUP_STAGES & set(setup_elapsed)):
            if not _nonnegative_number(setup_elapsed.get(stage)):
                failures.append(f"setup_elapsed_s.{stage} must be non-negative")
        if "motif_annotations" in setup_elapsed and not _nonnegative_number(
            setup_elapsed.get("motif_annotations")
        ):
            failures.append("setup_elapsed_s.motif_annotations must be non-negative")
    if not isinstance(elapsed, dict):
        failures.append("elapsed_per_stage must be an object")
    if not isinstance(memory, dict):
        failures.append("peak_rss_gb_per_stage must be an object")
    if isinstance(elapsed, dict):
        missing = REQUIRED_FULL_PIPELINE_ELAPSED_STAGES - set(elapsed)
        failures.extend(f"elapsed_per_stage.{stage} missing" for stage in sorted(missing))
        for stage in sorted(REQUIRED_FULL_PIPELINE_ELAPSED_STAGES & set(elapsed)):
            if not _nonnegative_number(elapsed.get(stage)):
                failures.append(f"elapsed_per_stage.{stage} must be non-negative")
    if isinstance(memory, dict):
        missing = REQUIRED_FULL_PIPELINE_MEMORY_STAGES - set(memory)
        failures.extend(f"peak_rss_gb_per_stage.{stage} missing" for stage in sorted(missing))
        for stage in sorted(REQUIRED_FULL_PIPELINE_MEMORY_STAGES & set(memory)):
            if not _positive_number(memory.get(stage)):
                failures.append(f"peak_rss_gb_per_stage.{stage} must be positive")
        unknown = set(memory) - FULL_PIPELINE_STAGES
        failures.extend(f"unknown peak_rss_gb_per_stage.{stage}" for stage in sorted(unknown))

    outputs = record.get("outputs", {})
    shapes = record.get("shapes", {})
    if not isinstance(shapes, dict):
        failures.append("shapes must be an object")
    else:
        for key in ("rna_post_qc", "atac_shared_cells", "motif_rankings"):
            if not _shape2(shapes.get(key)):
                failures.append(f"shapes.{key} must be [positive_rows, positive_cols]")
        if "motif_annotations" in shapes and not _shape2(shapes.get("motif_annotations")):
            failures.append("shapes.motif_annotations must be [positive_rows, positive_cols]")
        for key in ("gene_coords_rows", "tfs_supplied"):
            if not _positive_int(shapes.get(key)):
                failures.append(f"shapes.{key} must be positive")
        if _shape2(shapes.get("rna_post_qc")) and _shape2(shapes.get("atac_shared_cells")):
            if shapes["rna_post_qc"][0] != shapes["atac_shared_cells"][0]:
                failures.append("shapes.rna_post_qc cells must equal shapes.atac_shared_cells cells")

    if not _positive_int(outputs.get("grn_edges")):
        failures.append("outputs.grn_edges must be positive")
    if not _positive_int(outputs.get("regulons")):
        failures.append("outputs.regulons must be positive")
    for key in ("cistarget_rows", "enhancer_links", "eregulon_rows", "eregulons"):
        if not _positive_int(outputs.get(key)):
            failures.append(f"outputs.{key} must be positive")
    aucell_shape = outputs.get("aucell_shape")
    if (
        not isinstance(aucell_shape, list)
        or len(aucell_shape) != 2
        or not all(_positive_int(v) for v in aucell_shape)
    ):
        failures.append("outputs.aucell_shape must be [positive_cells, positive_regulons]")
    elif isinstance(shapes, dict) and _shape2(shapes.get("rna_post_qc")):
        if aucell_shape[0] != shapes["rna_post_qc"][0]:
            failures.append("outputs.aucell_shape cells must equal shapes.rna_post_qc cells")

    if check_output_files:
        failures.extend(_output_inventory_failures(record))
    return failures


def validate_grn_scaling(record: dict[str, Any], *, require_clean: bool) -> list[str]:
    failures = _repo_failures(record, require_clean=require_clean)
    failures.extend(_runtime_import_failures(record, "grn_scaling"))
    failures.extend(_backend_failures(record, "grn_scaling"))
    failures.extend(_python_hot_path_failures(record, "grn_scaling"))
    failures.extend(
        _require_keys(
            record,
            {
                "runtime_import",
                "backend_capabilities",
                "python_hot_paths",
                "rustscenic",
                "dataset",
                "params",
                "subset_scaling",
                "thread_scaling",
                "thread_speedups",
                "subset_wall_slope_vs_cells",
                "subset_memory_slope_vs_cells",
            },
            "grn_scaling",
        )
    )
    if record.get("benchmark") != "real_pbmc3k_grn_scaling":
        failures.append("benchmark must be real_pbmc3k_grn_scaling")
    if not _nonempty_str(record.get("dataset")):
        failures.append("dataset must be a non-empty string")

    for section in ("subset_scaling", "thread_scaling"):
        rows = record.get(section)
        if not isinstance(rows, list) or not rows:
            failures.append(f"{section} must contain at least one row")
            continue
        for idx, row in enumerate(rows):
            prefix = f"{section}[{idx}]"
            if not _nonempty_str(row.get("dataset")):
                failures.append(f"{prefix}.dataset must be a non-empty string")
            for key in ("n_cells", "n_genes", "n_tfs", "threads", "edges"):
                if not _positive_int(row.get(key)):
                    failures.append(f"{prefix}.{key} must be positive")
            for key in ("grn_wall_s", "peak_rss_gb"):
                if not _positive_number(row.get(key)):
                    failures.append(f"{prefix}.{key} must be positive")
            failures.extend(_thread_budget_failures(row, prefix, params_key="threads"))
            env = row.get("env", {})
            child_repo = env.get("repo_state") if isinstance(env, dict) else None
            child_tracked_source_count = (
                child_repo.get("tracked_source_count")
                if isinstance(child_repo, dict)
                else None
            )
            if (
                require_clean
                and isinstance(child_tracked_source_count, int)
                and not isinstance(child_tracked_source_count, bool)
                and child_tracked_source_count > 0
            ):
                failures.append(f"{prefix}.env.repo_state.tracked_source_count must be 0")
            elif require_clean and isinstance(child_repo, dict) and child_repo.get("tracked_dirty") is not False:
                failures.append(f"{prefix}.env.repo_state.tracked_dirty must be false")
            untracked_source_count = (
                child_repo.get("untracked_source_count")
                if isinstance(child_repo, dict)
                else None
            )
            if (
                require_clean
                and isinstance(untracked_source_count, int)
                and not isinstance(untracked_source_count, bool)
                and untracked_source_count > 0
            ):
                failures.append(f"{prefix}.env.repo_state.untracked_source_count must be 0")
            if isinstance(env, dict):
                failures.extend(_runtime_import_failures(env, f"{prefix}.env"))
                failures.extend(_backend_failures(env, f"{prefix}.env"))
                failures.extend(_python_hot_path_failures(env, f"{prefix}.env"))
            else:
                failures.append(f"{prefix}.env must be an object")
    return failures


def validate_full_pipeline_scaling(
    record: dict[str, Any],
    *,
    require_clean: bool,
    check_output_files: bool,
) -> list[str]:
    failures = _repo_failures(record, require_clean=require_clean)
    failures.extend(_runtime_import_failures(record, "full_pipeline_scaling"))
    failures.extend(_backend_failures(record, "full_pipeline_scaling"))
    failures.extend(_python_hot_path_failures(record, "full_pipeline_scaling"))
    failures.extend(
        _require_keys(
            record,
            {
                "runtime_import",
                "backend_capabilities",
                "python_hot_paths",
                "rustscenic",
                "dataset_name",
                "params",
                "runs",
                "scaling",
                "env",
            },
            "full_pipeline_scaling",
        )
    )
    if record.get("benchmark") != "real_multiome_full_pipeline_scaling":
        failures.append("benchmark must be real_multiome_full_pipeline_scaling")
    dataset_name = record.get("dataset_name")
    if not _nonempty_str(dataset_name):
        failures.append("dataset_name must be a non-empty string")
    failures.extend(_thread_budget_failures(record, "full_pipeline_scaling", params_key="threads"))
    require_motif_pruning = _motif_annotations_supplied(record)

    runs = record.get("runs")
    if not isinstance(runs, list) or not runs:
        failures.append("runs must contain at least one full-pipeline row")
        return failures
    previous_cells = 0
    for idx, row in enumerate(runs):
        prefix = f"runs[{idx}]"
        if not isinstance(row, dict):
            failures.append(f"{prefix} must be an object")
            continue
        n_cells = row.get("n_cells_actual")
        if not _positive_int(n_cells):
            failures.append(f"{prefix}.n_cells_actual must be positive")
        elif n_cells < previous_cells:
            failures.append(f"{prefix}.n_cells_actual must be non-decreasing")
        elif n_cells == previous_cells:
            failures.append(f"{prefix}.n_cells_actual duplicates previous run")
        previous_cells = n_cells if _positive_int(n_cells) else previous_cells

        for key in ("json_path", "output_dir"):
            if not isinstance(row.get(key), str) or not row[key]:
                failures.append(f"{prefix}.{key} must be a non-empty string")
        if not isinstance(row.get("wall_s"), dict):
            failures.append(f"{prefix}.wall_s must be an object")
        else:
            for key in ("setup", "pipeline", "end_to_end"):
                if not _positive_number(row["wall_s"].get(key)):
                    failures.append(f"{prefix}.wall_s.{key} must be positive")
        if not _positive_number(row.get("peak_rss_gb")):
            failures.append(f"{prefix}.peak_rss_gb must be positive")
        failures.extend(
            _full_pipeline_scaling_row_failures(
                row,
                prefix,
                require_motif_pruning=require_motif_pruning,
            )
        )

        if check_output_files and isinstance(row.get("json_path"), str):
            child_path = Path(row["json_path"])
            if not child_path.exists():
                failures.append(f"{prefix}.json_path does not exist: {child_path}")
                continue
            child = json.loads(child_path.read_text())
            child_failures = validate_full_pipeline(
                child,
                require_clean=require_clean,
                check_output_files=True,
            )
            failures.extend(f"{prefix}.{failure}" for failure in child_failures)
            if _nonempty_str(dataset_name) and child.get("dataset_name") != dataset_name:
                failures.append(f"{prefix}.dataset_name must match aggregate dataset_name")
            failures.extend(
                _full_pipeline_scaling_row_child_failures(row, child, prefix)
            )

    scaling = record.get("scaling")
    if not isinstance(scaling, dict):
        failures.append("scaling must be an object")
    elif len(runs) >= 2:
        wall_rows = [
            {
                "n_cells": row.get("n_cells_actual"),
                "end_to_end_wall_s": row.get("wall_s", {}).get("end_to_end")
                if isinstance(row.get("wall_s"), dict) else None,
                "pipeline_wall_s": row.get("wall_s", {}).get("pipeline")
                if isinstance(row.get("wall_s"), dict) else None,
                "peak_rss_gb": row.get("peak_rss_gb"),
            }
            for row in runs
            if isinstance(row, dict)
        ]
        for key in (
            "end_to_end_wall_slope_vs_cells",
            "pipeline_wall_slope_vs_cells",
            "peak_rss_slope_vs_cells",
        ):
            if not _nonnegative_number(scaling.get(key)):
                failures.append(f"scaling.{key} must be non-negative")
        expected_slopes = {
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
            "peak_rss_slope_vs_cells": _rounded_slope(
                wall_rows,
                "n_cells",
                "peak_rss_gb",
            ),
        }
        for key, expected in expected_slopes.items():
            if expected is not None and scaling.get(key) != expected:
                failures.append(
                    f"scaling.{key} must match runs: {scaling.get(key)} != {expected}"
                )
    return failures


def validate_record(
    record: dict[str, Any],
    *,
    require_clean: bool = True,
    check_output_files: bool = False,
) -> list[str]:
    benchmark = record.get("benchmark")
    if benchmark == "real_multiome_full_pipeline":
        return validate_full_pipeline(
            record,
            require_clean=require_clean,
            check_output_files=check_output_files,
        )
    if benchmark == "real_multiome_full_pipeline_scaling":
        return validate_full_pipeline_scaling(
            record,
            require_clean=require_clean,
            check_output_files=check_output_files,
        )
    if benchmark == "real_pbmc3k_grn_scaling":
        return validate_grn_scaling(record, require_clean=require_clean)
    return [f"unknown benchmark: {benchmark!r}"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", type=Path, nargs="+")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--check-output-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    any_failures = False
    for path in args.json_paths:
        record = json.loads(path.read_text())
        failures = validate_record(
            record,
            require_clean=not args.allow_dirty,
            check_output_files=args.check_output_files,
        )
        status = "ok" if not failures else "failed"
        print(json.dumps({"path": str(path), "status": status, "failures": failures}, indent=2))
        any_failures = any_failures or bool(failures)
    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
