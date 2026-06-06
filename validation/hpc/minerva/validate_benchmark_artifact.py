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
from validation.python_hot_paths import ALLOWED_HITS, HOT_PATH_PATTERNS


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
OPTIONAL_FULL_PIPELINE_MEMORY_STAGES = {
    "cistarget_pruning",
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
OPTIONAL_FULL_PIPELINE_ELAPSED_STAGES = {
    "cistarget_pruning",
}
COMPUTE_ELAPSED_BACKEND_STAGES = {
    "preproc": ("setup_fragments_to_matrix",),
    "topics": ("pipeline_topics",),
    "grn": ("pipeline_grn",),
    "cistarget": ("pipeline_cistarget",),
    "cistarget_pruning": ("pipeline_cistarget_pruning",),
    "enhancer": ("pipeline_enhancer",),
    "eregulons": (
        "pipeline_eregulon_peak_attribution",
        "pipeline_eregulons",
    ),
    "aucell": ("pipeline_aucell",),
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
    "manifest_path": "file",
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
REGION_CISTARGET_SCORING_SYMBOLS = {
    "cistarget_enrichment_from_rankings_i16",
    "cistarget_enrichment_from_rankings_i32",
    "cistarget_enrichment_from_rankings_i64",
    "cistarget_enrichment_from_projected_rankings_i16",
    "cistarget_enrichment_from_projected_rankings_i32",
    "cistarget_enrichment_from_projected_rankings_i64",
    "cistarget_region_attribution_i16",
    "cistarget_region_attribution_i32",
    "cistarget_region_attribution_i64",
}
REGION_CISTARGET_PEAK_VALUE_SYMBOLS = {
    "cistarget_region_attribution_peak_values_i16",
    "cistarget_region_attribution_peak_values_i32",
    "cistarget_region_attribution_peak_values_i64",
}
REGION_CISTARGET_EXPAND_SYMBOLS = {
    "pipeline_expand_region_cistarget_rows_f32",
    "pipeline_expand_region_cistarget_rows_f64",
}
FULL_PIPELINE_SCALING_CHILD_PARAM_KEYS = {
    "grn_n_estimators",
    "grn_max_features",
    "grn_target_block_size",
    "topics_n_topics",
    "topics_n_passes",
    "topics_method",
    "topics_n_iters",
    "topics_n_threads",
    "threads",
    "motif_annotations",
    "region_motif_rankings",
    "cistarget_top_frac",
    "cistarget_auc_threshold",
    "cistarget_nes_threshold",
    "enhancer_max_distance",
    "enhancer_min_abs_corr",
    "eregulon_min_target_genes",
    "eregulon_min_enhancer_links",
    "write_integrated_adata",
    "summary_rows",
    "summary_max_rows",
    "expected_tfs",
    "seed",
}
REQUIRED_FULL_PIPELINE_RUST_STAGE_SYMBOLS = {
    "grn": {
        "all_of": {"gene_duplicate_summary"},
        "any_of": ({"grn_infer"}, {"grn_infer_sparse_csc"}),
    },
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
    "pipeline_cistarget_projection_features": {
        "all_of": {"pipeline_unique_regulon_features"},
    },
    "pipeline_cistarget_ranking_projection": {
        "all_of": {"pipeline_project_ranking_columns"},
    },
    "pipeline_cistarget": {
        "any_of": (
            {"cistarget_enrichment_from_rankings_i16"},
            {"cistarget_enrichment_from_rankings_i32"},
            {"cistarget_enrichment_from_rankings_i64"},
            {"cistarget_enrichment_from_projected_rankings_i16"},
            {"cistarget_enrichment_from_projected_rankings_i32"},
            {"cistarget_enrichment_from_projected_rankings_i64"},
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
                "cistarget_prune_regulon_targets_projected_f32",
                "cistarget_prune_regulon_targets_projected_f64",
                "cistarget_prune_regulon_targets_projected_i16",
                "cistarget_prune_regulon_targets_projected_i32",
                "cistarget_prune_regulon_targets_projected_i64",
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
    "pipeline_eregulon_peak_filter": {
        "any_of": (
            {"pipeline_filter_cistarget_peak_rows_f32"},
            {"pipeline_filter_cistarget_peak_rows_f64"},
        ),
    },
    "pipeline_eregulon_peak_regulons": {
        "all_of": {"pipeline_peak_regulons_and_features_from_edges"},
    },
    "pipeline_region_cistarget_ranking_projection": {
        "all_of": {"pipeline_project_ranking_columns"},
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
RANKING_PACKER_SYMBOLS = {
    "pipeline_pack_ranking_columns_i16",
    "pipeline_pack_ranking_columns_i32",
    "pipeline_pack_ranking_columns_i64",
    "pipeline_pack_ranking_columns_f32",
    "pipeline_pack_ranking_columns_f64",
}


def _write_integrated_adata(record: dict[str, Any]) -> bool:
    params = record.get("params")
    if isinstance(params, dict) and params.get("write_integrated_adata") is False:
        return False
    if record.get("write_integrated_adata") is False:
        return False
    return True


def _required_full_pipeline_artefacts(record: dict[str, Any]) -> dict[str, str]:
    required = dict(REQUIRED_FULL_PIPELINE_ARTEFACTS)
    if not _write_integrated_adata(record):
        required.pop("integrated_adata_path", None)
    return required


def _required_full_pipeline_memory_stages(record: dict[str, Any]) -> set[str]:
    required = set(REQUIRED_FULL_PIPELINE_MEMORY_STAGES)
    if not _write_integrated_adata(record):
        required.discard("integrated_adata")
    return required


def _positive_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _shape2(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_positive_int(v) for v in value)
    )


def _matrix_input_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    shape_keys: tuple[str, ...] = ("rna_post_qc", "atac_shared_cells"),
    require_sparse: bool = True,
) -> list[str]:
    matrix_inputs = record.get("matrix_inputs")
    if not isinstance(matrix_inputs, dict):
        return [f"{prefix}.matrix_inputs must be an object"]

    shapes = record.get("shapes", {})
    failures: list[str] = []
    for key in shape_keys:
        stage_prefix = f"{prefix}.matrix_inputs.{key}"
        profile = matrix_inputs.get(key)
        if not isinstance(profile, dict):
            failures.append(f"{stage_prefix} must be an object")
            continue

        expected_shape = shapes.get(key) if isinstance(shapes, dict) else None
        shape = profile.get("shape")
        if not _shape2(shape):
            failures.append(f"{stage_prefix}.shape must be [positive_rows, positive_cols]")
        elif _shape2(expected_shape) and shape != expected_shape:
            failures.append(f"{stage_prefix}.shape must equal {prefix}.shapes.{key}")

        storage = profile.get("storage")
        if storage not in {"sparse", "dense"}:
            failures.append(f"{stage_prefix}.storage must be 'sparse' or 'dense'")
        elif require_sparse and storage != "sparse":
            failures.append(f"{stage_prefix}.storage must be 'sparse'")

        if not _nonempty_str(profile.get("format")):
            failures.append(f"{stage_prefix}.format must be a non-empty string")
        if not _nonempty_str(profile.get("dtype")):
            failures.append(f"{stage_prefix}.dtype must be a non-empty string")

        nnz = profile.get("nnz")
        density = profile.get("density")
        if storage == "sparse":
            if not _positive_int(nnz):
                failures.append(f"{stage_prefix}.nnz must be positive for sparse matrices")
            elif _shape2(shape) and nnz > int(shape[0]) * int(shape[1]):
                failures.append(f"{stage_prefix}.nnz must be <= rows * columns")
            if not _nonnegative_number(density) or density > 1:
                failures.append(f"{stage_prefix}.density must be in [0, 1]")
        elif storage == "dense":
            if nnz is not None:
                failures.append(f"{stage_prefix}.nnz must be null for dense matrices")
            if density is not None:
                failures.append(f"{stage_prefix}.density must be null for dense matrices")
    return failures


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
    allowed_count = state.get("allowed_hit_count")
    expected_allowed = len(ALLOWED_HITS)
    if not isinstance(allowed_count, int) or isinstance(allowed_count, bool):
        failures.append(f"{prefix}.python_hot_paths.allowed_hit_count must be an integer")
    elif allowed_count != expected_allowed:
        failures.append(
            f"{prefix}.python_hot_paths.allowed_hit_count must equal "
            f"{expected_allowed}"
        )
    pattern_count = state.get("pattern_count")
    expected_patterns = len(HOT_PATH_PATTERNS)
    if not isinstance(pattern_count, int) or isinstance(pattern_count, bool):
        failures.append(f"{prefix}.python_hot_paths.pattern_count must be an integer")
    elif pattern_count != expected_patterns:
        failures.append(
            f"{prefix}.python_hot_paths.pattern_count must equal "
            f"{expected_patterns}"
        )
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


def _sparse_rna_backend_failures(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:
    matrix_inputs = record.get("matrix_inputs")
    if not isinstance(matrix_inputs, dict):
        return []
    rna_profile = matrix_inputs.get("rna_post_qc")
    if not isinstance(rna_profile, dict) or rna_profile.get("storage") != "sparse":
        return []

    execution = record.get("backend_execution")
    if not isinstance(execution, dict):
        return []

    required = {
        "pipeline_grn": "grn_infer_sparse_csc",
        "pipeline_enhancer": "enhancer_link_pearson_sparse_rna",
        "pipeline_aucell": "aucell_score_sparse_csr",
    }
    failures: list[str] = []
    for stage, required_symbol in required.items():
        state = execution.get(stage)
        symbols = state.get("symbols") if isinstance(state, dict) else None
        if not isinstance(symbols, list):
            continue
        symbol_set = {symbol for symbol in symbols if isinstance(symbol, str)}
        if required_symbol in symbol_set:
            continue
        failures.append(
            f"{prefix}.backend_execution.{stage}.symbols must include "
            f"{required_symbol!r} when matrix_inputs.rna_post_qc.storage "
            "is 'sparse'"
        )
    return failures


def _sparse_grn_backend_failures(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:
    matrix_inputs = record.get("matrix_inputs")
    if not isinstance(matrix_inputs, dict):
        return []
    rna_profile = matrix_inputs.get("rna_post_qc")
    if not isinstance(rna_profile, dict) or rna_profile.get("storage") != "sparse":
        return []

    execution = record.get("backend_execution")
    state = execution.get("grn") if isinstance(execution, dict) else None
    symbols = state.get("symbols") if isinstance(state, dict) else None
    if not isinstance(symbols, list):
        return []
    if "grn_infer_sparse_csc" in {
        symbol for symbol in symbols if isinstance(symbol, str)
    }:
        return []
    return [
        f"{prefix}.backend_execution.grn.symbols must include "
        "'grn_infer_sparse_csc' when matrix_inputs.rna_post_qc.storage "
        "is 'sparse'"
    ]


def _integrated_adata_execution_failures(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:
    execution = record.get("backend_execution")
    if not isinstance(execution, dict):
        return []

    stage_prefix = f"{prefix}.backend_execution.pipeline_integrated_adata"
    state = execution.get("pipeline_integrated_adata")
    if not isinstance(state, dict):
        return [f"{stage_prefix} must be an object"]

    expected_engine = "python_io" if _write_integrated_adata(record) else "skipped"
    failures: list[str] = []
    if state.get("engine") != expected_engine:
        failures.append(f"{stage_prefix}.engine must be '{expected_engine}'")

    reason = state.get("reason")
    if not _nonempty_str(reason):
        failures.append(f"{stage_prefix}.reason must be a non-empty string")
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
    for optional_key in ("motif_annotations", "region_motif_rankings"):
        if optional_key in fingerprints:
            keys.append(optional_key)
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
        if key in {"motif_rankings", "region_motif_rankings"}:
            failures.extend(_file_backed_ranking_fingerprint_failures(fp, key))

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
        region_shape = shapes.get("region_motif_rankings")
        region_fp = fingerprints.get("region_motif_rankings")
        if (
            isinstance(region_fp, dict)
            and _shape2(region_shape)
            and region_fp.get("shape") != region_shape
        ):
            failures.append(
                "reference_fingerprints.region_motif_rankings.shape must match "
                "shapes.region_motif_rankings"
            )
        gene_rows = shapes.get("gene_coords_rows")
        gene_fp = fingerprints.get("gene_coords")
        if isinstance(gene_fp, dict) and _positive_int(gene_rows) and _shape2(gene_fp.get("shape")):
            if gene_fp["shape"][0] != gene_rows:
                failures.append("reference_fingerprints.gene_coords.shape rows must match shapes.gene_coords_rows")
    return failures


def _reference_source_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    sources = record.get("reference_sources")
    fingerprints = record.get("reference_fingerprints", {})
    fingerprint_keys = fingerprints if isinstance(fingerprints, dict) else {}
    if not isinstance(sources, dict):
        return ["reference_sources must be an object"]
    keys = ["motif_rankings", "gene_coords"]
    if (
        "motif_annotations" in fingerprint_keys
        or _motif_annotations_supplied(record)
    ):
        keys.append("motif_annotations")
    if (
        "region_motif_rankings" in fingerprint_keys
        or _region_motif_rankings_supplied(record)
    ):
        keys.append("region_motif_rankings")

    for key in keys:
        entry = sources.get(key)
        prefix = f"reference_sources.{key}"
        if not isinstance(entry, dict):
            failures.append(f"{prefix} must be an object")
            continue

        source = entry.get("source")
        if source not in {"explicit_path", "default_cache", "default_download"}:
            failures.append(
                f"{prefix}.source must be explicit_path, default_cache, or default_download"
            )
        if not _nonempty_str(entry.get("path")):
            failures.append(f"{prefix}.path must be a non-empty string")
        if not isinstance(entry.get("exists_before"), bool):
            failures.append(f"{prefix}.exists_before must be a boolean")
        if not isinstance(entry.get("exists_after"), bool):
            failures.append(f"{prefix}.exists_after must be a boolean")

        if source == "explicit_path":
            if entry.get("exists_before") is not True or entry.get("exists_after") is not True:
                failures.append(f"{prefix} explicit paths must exist before and after loading")
            if entry.get("cache_exists_before") is not None:
                failures.append(f"{prefix}.cache_exists_before must be null for explicit paths")
            if entry.get("cache_exists_after") is not None:
                failures.append(f"{prefix}.cache_exists_after must be null for explicit paths")
        elif source in {"default_cache", "default_download"}:
            cache_before = entry.get("cache_exists_before")
            cache_after = entry.get("cache_exists_after")
            if not isinstance(cache_before, bool):
                failures.append(f"{prefix}.cache_exists_before must be a boolean")
            if not isinstance(cache_after, bool):
                failures.append(f"{prefix}.cache_exists_after must be a boolean")
            if source == "default_cache" and (
                cache_before is not True or cache_after is not True
            ):
                failures.append(f"{prefix} default_cache must exist before and after loading")
            if source == "default_download" and (
                cache_before is not False or cache_after is not True
            ):
                failures.append(
                    f"{prefix} default_download must be absent before and cached after loading"
                )
    return failures


def _file_backed_ranking_fingerprint_failures(
    fp: dict[str, Any],
    key: str,
) -> list[str]:
    failures: list[str] = []
    prefix = f"reference_fingerprints.{key}"
    if fp.get("file_backed") is not True:
        failures.append(
            f"{prefix}.file_backed must be true"
        )
    if not _nonempty_str(fp.get("path_name")):
        failures.append(
            f"{prefix}.path_name must be a non-empty string"
        )
    if not _positive_int(fp.get("size_bytes")):
        failures.append(
            f"{prefix}.size_bytes must be positive"
        )
    format_value = fp.get("format")
    if format_value not in {"parquet", "feather", "ft"}:
        failures.append(
            f"{prefix}.format must be one of parquet, feather, ft"
        )
    read_columns = fp.get("metadata_read_columns")
    if (
        not isinstance(read_columns, list)
        or len(read_columns) > 1
        or not all(_nonempty_str(column) for column in read_columns)
    ):
        failures.append(
            f"{prefix}.metadata_read_columns must be a string list of length 0 or 1"
        )
    if format_value in {"feather", "ft"} and isinstance(read_columns, list) and not read_columns:
        failures.append(
            f"{prefix}.metadata_read_columns must name the one Feather column "
            "read for row counting"
        )
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


def _region_motif_rankings_supplied(record: dict[str, Any]) -> bool:
    """Return true when a benchmark artefact says region cistarget was requested."""
    params = record.get("params")
    if isinstance(params, dict):
        region_rankings = params.get("region_motif_rankings")
        if isinstance(region_rankings, str) and region_rankings.strip():
            return True
    for section in ("reference_fingerprints", "shapes", "setup_elapsed_s"):
        value = record.get(section)
        if isinstance(value, dict) and "region_motif_rankings" in value:
            return True
    return False


def _cistarget_ranking_metadata_failures(
    record: dict[str, Any],
    prefix: str,
) -> list[str]:
    return _ranking_metadata_failures(
        record,
        prefix,
        field="cistarget_rankings",
        shape_key="motif_rankings",
        projected_stage="pipeline_cistarget",
        projection_stage="pipeline_cistarget_ranking_projection",
        required=True,
    )


def _region_cistarget_ranking_metadata_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    required: bool,
) -> list[str]:
    return _ranking_metadata_failures(
        record,
        prefix,
        field="region_cistarget_rankings",
        shape_key="region_motif_rankings",
        projected_stage="pipeline_eregulon_peak_attribution",
        projection_stage="pipeline_region_cistarget_ranking_projection",
        required=required,
    )


def _ranking_metadata_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    field: str,
    shape_key: str,
    projected_stage: str,
    projection_stage: str,
    required: bool,
) -> list[str]:
    failures: list[str] = []
    meta = record.get(field)
    if not meta and not required:
        return []
    if not isinstance(meta, dict) or not meta:
        return [f"{prefix}.{field} must be an object"]

    mode = meta.get("mode")
    projected = meta.get("projected")
    input_kind = meta.get("input_kind")
    loaded_columns = meta.get("loaded_columns")
    rank_universe_size = meta.get("rank_universe_size")
    requested_features = meta.get("requested_features")
    motifs = meta.get("motifs")

    if mode not in {"projected_file", "full_file", "dataframe"}:
        failures.append(
            f"{prefix}.{field}.mode must be projected_file, "
            "full_file, or dataframe"
        )
    if not _nonempty_str(input_kind):
        failures.append(f"{prefix}.{field}.input_kind must be a non-empty string")
    if not isinstance(projected, bool):
        failures.append(f"{prefix}.{field}.projected must be boolean")
    if not _positive_int(loaded_columns):
        failures.append(f"{prefix}.{field}.loaded_columns must be positive")
    if not _positive_int(rank_universe_size):
        failures.append(f"{prefix}.{field}.rank_universe_size must be positive")
    elif _positive_int(loaded_columns) and rank_universe_size < loaded_columns:
        failures.append(
            f"{prefix}.{field}.rank_universe_size must be >= "
            f"{prefix}.{field}.loaded_columns"
        )
    if requested_features is not None and not _positive_int(requested_features):
        failures.append(
            f"{prefix}.{field}.requested_features must be positive or null"
        )
    elif (
        _positive_int(requested_features)
        and _positive_int(loaded_columns)
        and requested_features < loaded_columns
    ):
        failures.append(
            f"{prefix}.{field}.requested_features must be >= "
            f"{prefix}.{field}.loaded_columns"
        )
    if not _positive_int(motifs):
        failures.append(f"{prefix}.{field}.motifs must be positive")

    if mode == "projected_file":
        if projected is not True:
            failures.append(f"{prefix}.{field}.projected must be true")
        if input_kind == "dataframe":
            failures.append(
                f"{prefix}.{field}.input_kind must not be dataframe "
                "for projected_file mode"
            )
        if _positive_int(rank_universe_size) and _positive_int(loaded_columns):
            if rank_universe_size <= loaded_columns:
                failures.append(
                    f"{prefix}.{field}.rank_universe_size must be "
                    f"> {prefix}.{field}.loaded_columns in "
                    "projected_file mode"
                )
        if not _positive_int(requested_features):
            failures.append(
                f"{prefix}.{field}.requested_features must be "
                "positive in projected_file mode"
            )
        failures.extend(
            _projected_cistarget_symbol_failures(
                record,
                prefix,
                field=field,
                stage=projected_stage,
            )
        )
        failures.extend(
            _ranking_projection_backend_failures(
                record,
                prefix,
                field=field,
                stage=projection_stage,
            )
        )
    elif mode == "full_file":
        if projected is not False:
            failures.append(f"{prefix}.{field}.projected must be false")
        if input_kind == "dataframe":
            failures.append(
                f"{prefix}.{field}.input_kind must not be dataframe "
                "for full_file mode"
            )
        if _positive_int(rank_universe_size) and _positive_int(loaded_columns):
            if rank_universe_size != loaded_columns:
                failures.append(
                    f"{prefix}.{field}.rank_universe_size must equal "
                    f"{prefix}.{field}.loaded_columns in full_file mode"
                )
    elif mode == "dataframe":
        if projected is not False:
            failures.append(f"{prefix}.{field}.projected must be false")
        if input_kind != "dataframe":
            failures.append(
                f"{prefix}.{field}.input_kind must be dataframe "
                "for dataframe mode"
            )
        if requested_features is not None:
            failures.append(
                f"{prefix}.{field}.requested_features must be null "
                "for dataframe mode"
            )

    shapes = record.get("shapes")
    motif_shape = shapes.get(shape_key) if isinstance(shapes, dict) else None
    if _shape2(motif_shape):
        if _positive_int(motifs) and motifs != motif_shape[0]:
            failures.append(
                f"{prefix}.{field}.motifs must match "
                f"{prefix}.shapes.{shape_key} rows"
            )
        if _positive_int(rank_universe_size) and rank_universe_size != motif_shape[1]:
            failures.append(
                f"{prefix}.{field}.rank_universe_size must match "
                f"{prefix}.shapes.{shape_key} columns"
            )
    failures.extend(
        _ranking_reference_format_failures(
            record,
            prefix,
            field=field,
            shape_key=shape_key,
        )
    )

    return failures


def _ranking_reference_format_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    field: str,
    shape_key: str,
) -> list[str]:
    meta = record.get(field)
    fingerprints = record.get("reference_fingerprints")
    if not isinstance(meta, dict) or not isinstance(fingerprints, dict):
        return []
    fp = fingerprints.get(shape_key)
    if not isinstance(fp, dict):
        return []
    input_kind = meta.get("input_kind")
    reference_format = fp.get("format")
    if input_kind == "dataframe" or reference_format is None:
        return []
    if input_kind != reference_format:
        return [
            f"{prefix}.{field}.input_kind must match "
            f"{prefix}.reference_fingerprints.{shape_key}.format"
        ]
    return []


def _projected_cistarget_symbol_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    field: str,
    stage: str,
) -> list[str]:
    execution = record.get("backend_execution")
    state = execution.get(stage) if isinstance(execution, dict) else None
    symbols = state.get("symbols") if isinstance(state, dict) else None
    if not isinstance(symbols, list):
        return []
    if any(
        isinstance(symbol, str)
        and symbol.startswith("cistarget_enrichment_from_projected_rankings")
        for symbol in symbols
    ):
        return []
    return [
        f"{prefix}.backend_execution.{stage}.symbols must include "
        "a cistarget_enrichment_from_projected_rankings* Rust symbol when "
        f"{field}.mode is projected_file"
    ]


def _ranking_projection_backend_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    field: str,
    stage: str,
) -> list[str]:
    failures = _backend_execution_failures(record, prefix, {stage})
    if failures:
        return failures
    execution = record.get("backend_execution")
    state = execution.get(stage) if isinstance(execution, dict) else None
    symbols = state.get("symbols") if isinstance(state, dict) else None
    if not isinstance(symbols, list):
        return []
    symbol_set = {symbol for symbol in symbols if isinstance(symbol, str)}
    if "pipeline_project_ranking_columns" not in symbol_set:
        failures.append(
            f"{prefix}.backend_execution.{stage}.symbols must include "
            f"'pipeline_project_ranking_columns' when {field}.mode is projected_file"
        )
    if not (symbol_set & RANKING_PACKER_SYMBOLS):
        failures.append(
            f"{prefix}.backend_execution.{stage}.symbols must include a "
            f"pipeline_pack_ranking_columns_* Rust symbol when {field}.mode "
            "is projected_file"
        )
    return failures


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


def _region_motif_ranking_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    required: bool = False,
    require_peak_filter: bool = False,
) -> list[str]:
    if not required and not _region_motif_rankings_supplied(record):
        return []
    failures = _backend_execution_failures(
        record,
        prefix,
        {"pipeline_eregulon_peak_regulons", "pipeline_eregulon_peak_attribution"},
    )
    failures.extend(
        _region_cistarget_ranking_metadata_failures(
            record,
            prefix,
            required=True,
        )
    )
    execution = record.get("backend_execution")
    state = execution.get("pipeline_eregulon_peak_attribution") if isinstance(execution, dict) else None
    symbols = state.get("symbols") if isinstance(state, dict) else None
    if not isinstance(symbols, list):
        return failures
    symbol_set = {symbol for symbol in symbols if isinstance(symbol, str)}
    if not (REGION_CISTARGET_SCORING_SYMBOLS & symbol_set):
        failures.append(
            f"{prefix}.backend_execution.pipeline_eregulon_peak_attribution."
            "symbols must include a cistarget enrichment or region-attribution "
            "Rust symbol "
            "when region_motif_rankings is supplied"
        )
    if not (REGION_CISTARGET_PEAK_VALUE_SYMBOLS & symbol_set):
        failures.append(
            f"{prefix}.backend_execution.pipeline_eregulon_peak_attribution."
            "symbols must include a cistarget_region_attribution_peak_values_* "
            "Rust symbol when region_motif_rankings is supplied"
        )
    if not (REGION_CISTARGET_EXPAND_SYMBOLS & symbol_set):
        failures.append(
            f"{prefix}.backend_execution.pipeline_eregulon_peak_attribution."
            "symbols must include a pipeline_expand_region_cistarget_rows_* "
            "Rust symbol when region_motif_rankings is supplied"
        )
    if require_peak_filter or _motif_annotations_supplied(record):
        failures.extend(
            _backend_execution_failures(
                record,
                prefix,
                {"pipeline_eregulon_peak_filter"},
            )
        )
    return failures


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
    summary_max_rows = summaries.get("summary_max_rows")
    if summary_max_rows is not None and not _positive_int(summary_max_rows):
        failures.append("output_summaries.summary_max_rows must be a positive integer or null")

    params = record.get("params")
    if isinstance(params, dict):
        param_summary_max_rows = params.get("summary_max_rows")
        if param_summary_max_rows != summary_max_rows:
            failures.append(
                "output_summaries.summary_max_rows must match params.summary_max_rows"
            )
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
    lsf_cores = env.get("lsf_cores")
    if lsf_cores is not None:
        try:
            actual_cores = int(lsf_cores)
        except (TypeError, ValueError):
            failures.append(f"{prefix}.env.lsf_cores must be a positive integer string")
        else:
            if actual_cores <= 0:
                failures.append(f"{prefix}.env.lsf_cores must be positive")
            elif actual_cores != expected_threads:
                failures.append(
                    f"{prefix}.env.lsf_cores must match params.{params_key}: "
                    f"{actual_cores} != {expected_threads}"
                )
    requested_cores = env.get("lsf_requested_cores")
    if requested_cores is not None:
        try:
            requested = int(requested_cores)
        except (TypeError, ValueError):
            failures.append(f"{prefix}.env.lsf_requested_cores must be a positive integer string")
        else:
            if requested <= 0:
                failures.append(f"{prefix}.env.lsf_requested_cores must be positive")
            elif requested != expected_threads:
                failures.append(
                    f"{prefix}.env.lsf_requested_cores must match params.{params_key}: "
                    f"{requested} != {expected_threads}"
                )
    requested_mem = env.get("lsf_requested_mem_mb")
    if requested_mem is not None:
        try:
            mem_mb = int(requested_mem)
        except (TypeError, ValueError):
            failures.append(f"{prefix}.env.lsf_requested_mem_mb must be a positive integer string")
        else:
            if mem_mb <= 0:
                failures.append(f"{prefix}.env.lsf_requested_mem_mb must be positive")
    for key in ("lsf_project", "lsf_requested_queue", "lsf_requested_walltime"):
        if key in env and not _nonempty_str(env.get(key)):
            failures.append(f"{prefix}.env.{key} must be a non-empty string when present")
    return failures


def _output_inventory_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    inventory = record.get("output_inventory")
    if not isinstance(inventory, dict):
        return ["output_inventory must be an object when --check-output-files is set"]

    for key, expected_type in sorted(_required_full_pipeline_artefacts(record).items()):
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


def _output_storage_failures(record: dict[str, Any]) -> list[str]:
    storage = record.get("output_storage")
    if storage is None:
        return []
    if not isinstance(storage, dict):
        return ["output_storage must be an object when present"]

    failures: list[str] = []
    sizes = storage.get("artifact_size_bytes")
    if not isinstance(sizes, dict):
        failures.append("output_storage.artifact_size_bytes must be an object")
        sizes = {}
    else:
        for key, value in sizes.items():
            if not _nonempty_str(key):
                failures.append("output_storage.artifact_size_bytes keys must be non-empty strings")
            if not _nonnegative_int(value):
                failures.append(
                    f"output_storage.artifact_size_bytes.{key} must be a non-negative integer"
                )

    total = storage.get("total_size_bytes")
    if not _nonnegative_int(total):
        failures.append("output_storage.total_size_bytes must be a non-negative integer")
    elif isinstance(sizes, dict):
        expected_total = sum(
            int(value)
            for value in sizes.values()
            if _nonnegative_int(value)
        )
        if int(total) != expected_total:
            failures.append(
                "output_storage.total_size_bytes must equal artifact_size_bytes sum: "
                f"{total} != {expected_total}"
            )

    total_gb = storage.get("total_size_gb")
    if not _nonnegative_number(total_gb):
        failures.append("output_storage.total_size_gb must be non-negative")
    elif _nonnegative_int(total):
        expected_gb = round(int(total) / (1024 ** 3), 6)
        if round(float(total_gb), 6) != expected_gb:
            failures.append(
                "output_storage.total_size_gb must match total_size_bytes: "
                f"{total_gb} != {expected_gb}"
            )

    inventory = record.get("output_inventory")
    if isinstance(inventory, dict) and isinstance(sizes, dict):
        for key, info in inventory.items():
            if not isinstance(info, dict):
                continue
            size = info.get("size_bytes")
            if _nonnegative_int(size):
                if sizes.get(key) != size:
                    failures.append(
                        f"output_storage.artifact_size_bytes.{key} must match "
                        f"output_inventory.{key}.size_bytes"
                    )
    return failures


def _pipeline_manifest_failures(record: dict[str, Any]) -> list[str]:
    inventory = record.get("output_inventory")
    if not isinstance(inventory, dict):
        return []
    info = inventory.get("manifest_path")
    if not isinstance(info, dict) or not isinstance(info.get("path"), str):
        return []
    manifest_path = Path(info["path"])
    if not manifest_path.is_file():
        return []

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"output_inventory.manifest_path could not be read as JSON: {exc}"]
    if not isinstance(manifest, dict):
        return ["output_inventory.manifest_path JSON must be an object"]

    failures: list[str] = []
    if manifest.get("cell_barcode_filter") != record.get("cell_barcode_filter"):
        failures.append(
            "output_inventory.manifest_path cell_barcode_filter must match benchmark JSON"
        )
    if manifest.get("matrix_inputs") != record.get("matrix_inputs"):
        failures.append(
            "output_inventory.manifest_path matrix_inputs must match benchmark JSON"
        )
    if manifest.get("cistarget_rankings") != record.get("cistarget_rankings"):
        failures.append(
            "output_inventory.manifest_path cistarget_rankings must match benchmark JSON"
        )
    if manifest.get("region_cistarget_rankings") != record.get("region_cistarget_rankings"):
        failures.append(
            "output_inventory.manifest_path region_cistarget_rankings must match benchmark JSON"
        )
    failures.extend(_pipeline_manifest_path_failures(record, manifest))
    failures.extend(_pipeline_manifest_count_failures(record, manifest))
    failures.extend(
        _pipeline_manifest_stage_metric_failures(
            record,
            manifest,
            manifest_key="elapsed",
            record_key="elapsed_per_stage",
        )
    )
    failures.extend(
        _pipeline_manifest_stage_metric_failures(
            record,
            manifest,
            manifest_key="memory",
            record_key="peak_rss_gb_per_stage",
        )
    )
    if "io_elapsed_per_stage" in record:
        failures.extend(
            _pipeline_manifest_stage_metric_failures(
                record,
                manifest,
                manifest_key="io_elapsed",
                record_key="io_elapsed_per_stage",
            )
        )

    manifest_execution = manifest.get("backend_execution")
    benchmark_execution = record.get("backend_execution")
    if not isinstance(manifest_execution, dict):
        failures.append("output_inventory.manifest_path backend_execution must be an object")
    elif isinstance(benchmark_execution, dict):
        for stage, state in manifest_execution.items():
            key = f"pipeline_{stage}"
            if benchmark_execution.get(key) != state:
                failures.append(
                    f"output_inventory.manifest_path backend_execution.{stage} "
                    f"must match benchmark backend_execution.{key}"
                )
    return failures


def _pipeline_manifest_path_failures(
    record: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    inventory = record.get("output_inventory")
    if not isinstance(inventory, dict):
        return failures
    for key in (
        "atac_matrix_path",
        "grn_path",
        "regulons_path",
        "candidate_regulons_path",
        "aucell_path",
        "topics_dir",
        "cistarget_path",
        "enhancer_links_path",
        "eregulons_path",
        "integrated_adata_path",
    ):
        info = inventory.get(key)
        expected = info.get("path") if isinstance(info, dict) else None
        actual = manifest.get(key)
        if expected is None:
            if actual is not None:
                failures.append(
                    f"output_inventory.manifest_path {key} must be null when "
                    f"benchmark output_inventory.{key} is absent"
                )
        elif actual != expected:
            failures.append(
                f"output_inventory.manifest_path {key} must match "
                f"output_inventory.{key}.path"
            )
    return failures


def _pipeline_manifest_count_failures(
    record: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    outputs = record.get("outputs", {})
    shapes = record.get("shapes", {})
    expected = {
        "n_cells": (
            shapes.get("rna_post_qc", [None])[0]
            if isinstance(shapes, dict)
            else None
        ),
        "n_grn_edges": outputs.get("grn_edges") if isinstance(outputs, dict) else None,
        "n_regulons": outputs.get("regulons") if isinstance(outputs, dict) else None,
        "n_candidate_regulons": (
            outputs.get("candidate_regulons") if isinstance(outputs, dict) else None
        ),
        "n_pruned_regulons": (
            outputs.get("pruned_regulons") if isinstance(outputs, dict) else None
        ),
        "n_cistarget_rows": (
            outputs.get("cistarget_rows") if isinstance(outputs, dict) else None
        ),
        "n_enhancer_links": (
            outputs.get("enhancer_links") if isinstance(outputs, dict) else None
        ),
        "n_eregulon_rows": (
            outputs.get("eregulon_rows") if isinstance(outputs, dict) else None
        ),
        "n_eregulons": outputs.get("eregulons") if isinstance(outputs, dict) else None,
        "aucell_shape": outputs.get("aucell_shape") if isinstance(outputs, dict) else None,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(
                f"output_inventory.manifest_path {key} must match benchmark JSON"
            )
    return failures


def _pipeline_manifest_stage_metric_failures(
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_key: str,
    record_key: str,
) -> list[str]:
    failures: list[str] = []
    manifest_values = manifest.get(manifest_key)
    record_values = record.get(record_key)
    if not isinstance(manifest_values, dict):
        return [f"output_inventory.manifest_path {manifest_key} must be an object"]
    if not isinstance(record_values, dict):
        return failures
    for stage, expected in record_values.items():
        actual = manifest_values.get(stage)
        if not _numbers_match_rounded(actual, expected):
            failures.append(
                f"output_inventory.manifest_path {manifest_key}.{stage} "
                f"must match benchmark {record_key}.{stage}"
            )
    return failures


def _numbers_match_rounded(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return round(float(actual), 6) == round(float(expected), 6)
    return actual == expected


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


def _path(prefix: str, field: str) -> str:
    return f"{prefix}.{field}" if prefix else field


def _full_pipeline_wall_breakdown_failures(
    record: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    failures: list[str] = []
    wall = record.get("wall_s")
    elapsed = record.get("elapsed_per_stage")
    if not isinstance(wall, dict) or not isinstance(elapsed, dict):
        return failures

    compute = wall.get("pipeline_compute_stages")
    io = wall.get("pipeline_io_stages", 0.0)
    unattributed = wall.get("pipeline_unattributed")
    if not _nonnegative_number(compute):
        failures.append(
            f"{_path(prefix, 'wall_s.pipeline_compute_stages')} must be non-negative"
        )
    io_elapsed = record.get("io_elapsed_per_stage")
    if io_elapsed is not None and not _nonnegative_number(io):
        failures.append(
            f"{_path(prefix, 'wall_s.pipeline_io_stages')} must be non-negative"
        )
    if not _nonnegative_number(unattributed):
        failures.append(
            f"{_path(prefix, 'wall_s.pipeline_unattributed')} must be non-negative"
        )

    elapsed_values = []
    for value in elapsed.values():
        if not _nonnegative_number(value):
            return failures
        elapsed_values.append(float(value))

    if _nonnegative_number(compute):
        expected_compute = round(sum(elapsed_values), 3)
        if abs(float(compute) - expected_compute) > 0.005:
            failures.append(
                f"{_path(prefix, 'wall_s.pipeline_compute_stages')} must match "
                f"elapsed_per_stage sum: {compute} != {expected_compute}"
            )
        if (
            _positive_number(wall.get("pipeline"))
            and float(compute) > float(wall["pipeline"]) + 0.25
        ):
            failures.append(
                f"{_path(prefix, 'wall_s.pipeline_compute_stages')} must not exceed "
                "wall_s.pipeline"
            )

    if io_elapsed is not None:
        if not isinstance(io_elapsed, dict):
            failures.append(f"{_path(prefix, 'io_elapsed_per_stage')} must be an object")
        else:
            io_values = []
            for stage, value in io_elapsed.items():
                if not _nonempty_str(stage):
                    failures.append(
                        f"{_path(prefix, 'io_elapsed_per_stage')} keys must be non-empty strings"
                    )
                    continue
                if not _nonnegative_number(value):
                    failures.append(
                        f"{_path(prefix, f'io_elapsed_per_stage.{stage}')} must be non-negative"
                    )
                    continue
                io_values.append(float(value))
            if _nonnegative_number(io):
                expected_io = round(sum(io_values), 3)
                if abs(float(io) - expected_io) > 0.005:
                    failures.append(
                        f"{_path(prefix, 'wall_s.pipeline_io_stages')} must match "
                        f"io_elapsed_per_stage sum: {io} != {expected_io}"
                    )
    else:
        io = 0.0

    if (
        _positive_number(wall.get("pipeline"))
        and _nonnegative_number(compute)
        and _nonnegative_number(io)
        and _nonnegative_number(unattributed)
    ):
        expected_unattributed = round(
            max(0.0, float(wall["pipeline"]) - float(compute) - float(io)),
            3,
        )
        if abs(float(unattributed) - expected_unattributed) > 0.01:
            io_note = " and IO stages" if io_elapsed is not None else ""
            failures.append(
                f"{_path(prefix, 'wall_s.pipeline_unattributed')} must match "
                f"pipeline minus compute stages{io_note}: "
                f"{unattributed} != {expected_unattributed}"
            )
    return failures


def _compute_elapsed_backend_failures(
    record: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    elapsed = record.get("elapsed_per_stage")
    if not isinstance(elapsed, dict):
        return []
    execution = record.get("backend_execution")
    if not isinstance(execution, dict):
        return []

    allowed = set(REQUIRED_FULL_PIPELINE_ELAPSED_STAGES) | set(
        OPTIONAL_FULL_PIPELINE_ELAPSED_STAGES
    )
    failures: list[str] = []
    unknown = set(elapsed) - allowed
    failures.extend(
        f"{_path(prefix, 'elapsed_per_stage')}.{stage} is not a recognised Rust compute stage"
        for stage in sorted(unknown)
    )

    for stage in sorted(set(elapsed) & allowed):
        value = elapsed.get(stage)
        if not _nonnegative_number(value) or float(value) == 0.0:
            continue
        for backend_stage in COMPUTE_ELAPSED_BACKEND_STAGES[stage]:
            state = execution.get(backend_stage)
            stage_path = _path(prefix, f"backend_execution.{backend_stage}")
            if not isinstance(state, dict):
                failures.append(
                    f"{_path(prefix, f'elapsed_per_stage.{stage}')} is counted as "
                    f"compute but {stage_path} must be an object"
                )
                continue
            if state.get("engine") != "rust":
                failures.append(
                    f"{_path(prefix, f'elapsed_per_stage.{stage}')} is counted as "
                    f"compute but {stage_path}.engine must be 'rust'"
                )
    return failures


def _cell_barcode_filter_failures(
    record: dict[str, Any],
    prefix: str,
    *,
    expected_requested_cells: int | None = None,
    expected_requested_label: str = "params.n_cells_requested",
    expected_matched_cells: int | None = None,
    expected_matched_label: str = "shapes.rna_post_qc",
) -> list[str]:
    failures: list[str] = []
    barcode_filter = record.get("cell_barcode_filter")
    if not isinstance(barcode_filter, dict):
        return [f"{_path(prefix, 'cell_barcode_filter')} must be an object"]

    requested = barcode_filter.get("requested")
    matched = barcode_filter.get("matched")
    if not _positive_int(requested):
        failures.append(
            f"{_path(prefix, 'cell_barcode_filter.requested')} must be positive"
        )
    elif expected_requested_cells is not None and requested != expected_requested_cells:
        failures.append(
            f"{_path(prefix, 'cell_barcode_filter.requested')} must equal "
            f"{_path(prefix, expected_requested_label)}"
        )
    if not _positive_int(matched):
        failures.append(
            f"{_path(prefix, 'cell_barcode_filter.matched')} must be positive"
        )
    if _positive_int(requested) and _positive_int(matched) and matched > requested:
        failures.append(
            f"{_path(prefix, 'cell_barcode_filter.matched')} must be <= "
            f"{_path(prefix, 'cell_barcode_filter.requested')}"
        )
    if (
        expected_matched_cells is not None
        and _positive_int(matched)
        and matched != expected_matched_cells
    ):
        failures.append(
            f"{_path(prefix, 'cell_barcode_filter.matched')} must equal "
            f"{_path(prefix, expected_matched_label)} cells"
        )
    return failures


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
        "setup_elapsed_s": child.get("setup_elapsed_s"),
        "elapsed_per_stage": child.get("elapsed_per_stage"),
        "io_elapsed_per_stage": child.get("io_elapsed_per_stage"),
        "peak_rss_gb_per_stage": child.get("peak_rss_gb_per_stage"),
        "output_storage": child.get("output_storage"),
        "outputs": child.get("outputs"),
        "expected_tf_recovery": child.get("expected_tf_recovery"),
        "backend_execution": child.get("backend_execution"),
        "cell_barcode_filter": child.get("cell_barcode_filter"),
        "matrix_inputs": child.get("matrix_inputs"),
        "shapes": child.get("shapes"),
        "cistarget_rankings": child.get("cistarget_rankings"),
        "region_cistarget_rankings": child.get("region_cistarget_rankings"),
        "reference_sources": child.get("reference_sources"),
        "params": child.get("params"),
        "write_integrated_adata": child.get("params", {}).get(
            "write_integrated_adata",
            True,
        ),
    }
    optional_checks = {
        "threads": child.get("params", {}).get("threads"),
        "env": child.get("env"),
    }
    for key, expected in checks.items():
        if row.get(key) != expected:
            failures.append(f"{prefix}.{key} must match child JSON")
    for key, expected in optional_checks.items():
        if key in row and row.get(key) != expected:
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


def _full_pipeline_scaling_reference_source_failures(
    row: dict[str, Any],
    prefix: str,
) -> list[str]:
    sources = row.get("reference_sources")
    if not isinstance(sources, dict):
        return [f"{prefix}.reference_sources must be an object"]

    failures: list[str] = []
    for key in ("motif_rankings", "gene_coords"):
        entry = sources.get(key)
        entry_prefix = f"{prefix}.reference_sources.{key}"
        if not isinstance(entry, dict):
            failures.append(f"{entry_prefix} must be an object")
            continue
        source = entry.get("source")
        if source not in {"explicit_path", "default_cache"}:
            failures.append(
                f"{entry_prefix}.source must be explicit_path or default_cache "
                "for full-pipeline scaling"
            )
        if not _nonempty_str(entry.get("path")):
            failures.append(f"{entry_prefix}.path must be a non-empty string")
    return failures


def _full_pipeline_scaling_design_failures(
    record: dict[str, Any],
    runs: list[Any],
) -> list[str]:
    params = record.get("params")
    if not isinstance(params, dict):
        return ["full_pipeline_scaling.params must be an object"]

    failures: list[str] = []
    cell_counts = params.get("cell_counts")
    if (
        not isinstance(cell_counts, list)
        or len(cell_counts) < 2
        or not all(_positive_int(value) for value in cell_counts)
    ):
        failures.append("params.cell_counts must contain at least two positive integers")
    else:
        if cell_counts != sorted(cell_counts):
            failures.append("params.cell_counts must be sorted ascending")
        if len(set(cell_counts)) != len(cell_counts):
            failures.append("params.cell_counts must not contain duplicates")

        requested = [
            row.get("n_cells_requested")
            for row in runs
            if isinstance(row, dict)
        ]
        if requested != cell_counts:
            failures.append("params.cell_counts must match runs[*].n_cells_requested")

    threads = params.get("threads")
    if _positive_int(threads):
        row_threads = [
            row.get("threads")
            for row in runs
            if isinstance(row, dict) and "threads" in row
        ]
        if row_threads and row_threads != [threads] * len(row_threads):
            failures.append("params.threads must match runs[*].threads")

    if "write_integrated_adata" in params:
        write_integrated = params.get("write_integrated_adata")
        if not isinstance(write_integrated, bool):
            failures.append("params.write_integrated_adata must be boolean when present")
        else:
            row_write_flags = [
                row.get("write_integrated_adata")
                for row in runs
                if isinstance(row, dict) and "write_integrated_adata" in row
            ]
            if row_write_flags and row_write_flags != [write_integrated] * len(row_write_flags):
                failures.append(
                    "params.write_integrated_adata must match "
                    "runs[*].write_integrated_adata"
                )

    failures.extend(_full_pipeline_scaling_param_design_failures(params, runs))

    return failures


def _full_pipeline_scaling_param_design_failures(
    params: dict[str, Any],
    runs: list[Any],
) -> list[str]:
    failures: list[str] = []
    for index, row in enumerate(runs):
        if not isinstance(row, dict):
            continue
        prefix = f"runs[{index}]"
        row_params = row.get("params")
        if not isinstance(row_params, dict):
            failures.append(f"{prefix}.params must be an object")
            continue

        if row_params.get("n_cells_requested") != row.get("n_cells_requested"):
            failures.append(
                f"{prefix}.params.n_cells_requested must match "
                f"{prefix}.n_cells_requested"
            )
        if "threads" in row and row_params.get("threads") != row.get("threads"):
            failures.append(f"{prefix}.params.threads must match {prefix}.threads")
        if (
            "write_integrated_adata" in row
            and row_params.get("write_integrated_adata", True)
            != row.get("write_integrated_adata")
        ):
            failures.append(
                f"{prefix}.params.write_integrated_adata must match "
                f"{prefix}.write_integrated_adata"
            )

    for key in sorted(FULL_PIPELINE_SCALING_CHILD_PARAM_KEYS):
        if key not in params:
            failures.append(f"params.{key} missing")
            continue
        mismatches = [
            index
            for index, row in enumerate(runs)
            if isinstance(row, dict)
            and isinstance(row.get("params"), dict)
            and row["params"].get(key) != params.get(key)
        ]
        if mismatches:
            failures.append(f"params.{key} must match runs[*].params.{key}")

    return failures


def _full_pipeline_scaling_row_failures(
    row: dict[str, Any],
    prefix: str,
    *,
    require_motif_pruning: bool = False,
    require_region_cistarget: bool = False,
) -> list[str]:
    failures: list[str] = []
    n_cells = row.get("n_cells_actual")
    if not _positive_int(row.get("n_cells_requested")):
        failures.append(f"{prefix}.n_cells_requested must be positive")
    if "env" in row or "threads" in row:
        if not _positive_int(row.get("threads")):
            failures.append(f"{prefix}.threads must be positive when row env is present")
        else:
            failures.extend(_thread_budget_failures(row, prefix, params_key="threads"))

    if not _positive_number(row.get("setup_peak_rss_gb")):
        failures.append(f"{prefix}.setup_peak_rss_gb must be positive")
    setup_elapsed = row.get("setup_elapsed_s")
    if not isinstance(setup_elapsed, dict):
        failures.append(f"{prefix}.setup_elapsed_s must be an object")
    else:
        missing = REQUIRED_FULL_PIPELINE_SETUP_STAGES - set(setup_elapsed)
        failures.extend(
            f"{prefix}.setup_elapsed_s.{stage} missing"
            for stage in sorted(missing)
        )
        for stage in sorted(REQUIRED_FULL_PIPELINE_SETUP_STAGES & set(setup_elapsed)):
            if not _nonnegative_number(setup_elapsed.get(stage)):
                failures.append(f"{prefix}.setup_elapsed_s.{stage} must be non-negative")
        if "motif_annotations" in setup_elapsed and not _nonnegative_number(
            setup_elapsed.get("motif_annotations")
        ):
            failures.append(f"{prefix}.setup_elapsed_s.motif_annotations must be non-negative")
        if "region_motif_rankings_metadata" in setup_elapsed and not _nonnegative_number(
            setup_elapsed.get("region_motif_rankings_metadata")
        ):
            failures.append(
                f"{prefix}.setup_elapsed_s.region_motif_rankings_metadata must be non-negative"
            )
        if _positive_int(row.get("n_cells_requested")):
            if "subset_requested_cells" not in setup_elapsed:
                failures.append(f"{prefix}.setup_elapsed_s.subset_requested_cells missing")
            elif not _nonnegative_number(setup_elapsed.get("subset_requested_cells")):
                failures.append(
                    f"{prefix}.setup_elapsed_s.subset_requested_cells must be non-negative"
                )
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
        for stage in sorted(OPTIONAL_FULL_PIPELINE_ELAPSED_STAGES & set(elapsed)):
            if not _nonnegative_number(elapsed.get(stage)):
                failures.append(f"{prefix}.elapsed_per_stage.{stage} must be non-negative")

    memory = row.get("peak_rss_gb_per_stage")
    if not isinstance(memory, dict):
        failures.append(f"{prefix}.peak_rss_gb_per_stage must be an object")
    else:
        required_memory = _required_full_pipeline_memory_stages(row)
        missing = required_memory - set(memory)
        failures.extend(
            f"{prefix}.peak_rss_gb_per_stage.{stage} missing"
            for stage in sorted(missing)
        )
        for stage in sorted(required_memory & set(memory)):
            if not _positive_number(memory.get(stage)):
                failures.append(f"{prefix}.peak_rss_gb_per_stage.{stage} must be positive")
        for stage in sorted(OPTIONAL_FULL_PIPELINE_MEMORY_STAGES & set(memory)):
            if not _positive_number(memory.get(stage)):
                failures.append(f"{prefix}.peak_rss_gb_per_stage.{stage} must be positive")
        unknown = set(memory) - FULL_PIPELINE_STAGES - OPTIONAL_FULL_PIPELINE_MEMORY_STAGES
        failures.extend(
            f"{prefix}.unknown peak_rss_gb_per_stage.{stage}"
            for stage in sorted(unknown)
        )
    failures.extend(_full_pipeline_wall_breakdown_failures(row, prefix))
    failures.extend(_compute_elapsed_backend_failures(row, prefix))

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
    failures.extend(
        _cell_barcode_filter_failures(
            row,
            prefix,
            expected_requested_cells=(
                row.get("n_cells_requested")
                if _positive_int(row.get("n_cells_requested"))
                else None
            ),
            expected_requested_label="n_cells_requested",
            expected_matched_cells=n_cells if _positive_int(n_cells) else None,
            expected_matched_label="n_cells_actual",
        )
    )
    failures.extend(_matrix_input_failures(row, prefix))
    failures.extend(_cistarget_ranking_metadata_failures(row, prefix))
    failures.extend(_sparse_rna_backend_failures(row, prefix))
    failures.extend(_full_pipeline_scaling_reference_source_failures(row, prefix))
    failures.extend(_integrated_adata_execution_failures(row, prefix))
    if require_motif_pruning:
        failures.extend(
            _backend_execution_failures(
                row,
                prefix,
                {"pipeline_cistarget_pruning"},
            )
        )
    if require_region_cistarget:
        failures.extend(
            _region_motif_ranking_failures(
                row,
                prefix,
                required=True,
                require_peak_filter=require_motif_pruning,
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
    failures.extend(_integrated_adata_execution_failures(record, "full_pipeline"))
    failures.extend(_motif_annotation_pruning_failures(record, "full_pipeline"))
    failures.extend(_region_motif_ranking_failures(record, "full_pipeline"))
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
                "cell_barcode_filter",
                "cistarget_rankings",
                "region_cistarget_rankings",
                "input_hashes",
                "reference_fingerprints",
                "reference_sources",
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
    failures.extend(_reference_source_failures(record))
    failures.extend(_cistarget_ranking_metadata_failures(record, "full_pipeline"))
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
        if "region_motif_rankings_metadata" in setup_elapsed and not _nonnegative_number(
            setup_elapsed.get("region_motif_rankings_metadata")
        ):
            failures.append("setup_elapsed_s.region_motif_rankings_metadata must be non-negative")
        params_for_subset = record.get("params", {})
        if (
            isinstance(params_for_subset, dict)
            and _positive_int(params_for_subset.get("n_cells_requested"))
        ):
            if "subset_requested_cells" not in setup_elapsed:
                failures.append("setup_elapsed_s.subset_requested_cells missing")
            elif not _nonnegative_number(setup_elapsed.get("subset_requested_cells")):
                failures.append("setup_elapsed_s.subset_requested_cells must be non-negative")
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
        for stage in sorted(OPTIONAL_FULL_PIPELINE_ELAPSED_STAGES & set(elapsed)):
            if not _nonnegative_number(elapsed.get(stage)):
                failures.append(f"elapsed_per_stage.{stage} must be non-negative")
    failures.extend(_full_pipeline_wall_breakdown_failures(record))
    failures.extend(_compute_elapsed_backend_failures(record))
    if isinstance(memory, dict):
        required_memory = _required_full_pipeline_memory_stages(record)
        missing = required_memory - set(memory)
        failures.extend(f"peak_rss_gb_per_stage.{stage} missing" for stage in sorted(missing))
        for stage in sorted(required_memory & set(memory)):
            if not _positive_number(memory.get(stage)):
                failures.append(f"peak_rss_gb_per_stage.{stage} must be positive")
        for stage in sorted(OPTIONAL_FULL_PIPELINE_MEMORY_STAGES & set(memory)):
            if not _positive_number(memory.get(stage)):
                failures.append(f"peak_rss_gb_per_stage.{stage} must be positive")
        unknown = set(memory) - FULL_PIPELINE_STAGES - OPTIONAL_FULL_PIPELINE_MEMORY_STAGES
        failures.extend(f"unknown peak_rss_gb_per_stage.{stage}" for stage in sorted(unknown))

    outputs = record.get("outputs", {})
    shapes = record.get("shapes", {})
    params = record.get("params", {})
    if not isinstance(shapes, dict):
        failures.append("shapes must be an object")
    else:
        for key in ("rna_post_qc", "atac_shared_cells", "motif_rankings"):
            if not _shape2(shapes.get(key)):
                failures.append(f"shapes.{key} must be [positive_rows, positive_cols]")
        if "motif_annotations" in shapes and not _shape2(shapes.get("motif_annotations")):
            failures.append("shapes.motif_annotations must be [positive_rows, positive_cols]")
        if "region_motif_rankings" in shapes and not _shape2(shapes.get("region_motif_rankings")):
            failures.append("shapes.region_motif_rankings must be [positive_rows, positive_cols]")
        for key in ("gene_coords_rows", "tfs_supplied"):
            if not _positive_int(shapes.get(key)):
                failures.append(f"shapes.{key} must be positive")
        if _shape2(shapes.get("rna_post_qc")) and _shape2(shapes.get("atac_shared_cells")):
            if shapes["rna_post_qc"][0] != shapes["atac_shared_cells"][0]:
                failures.append("shapes.rna_post_qc cells must equal shapes.atac_shared_cells cells")
        min_matched_cells = (
            int(shapes["rna_post_qc"][0])
            if _shape2(shapes.get("rna_post_qc"))
            else None
        )
        failures.extend(
            _cell_barcode_filter_failures(
                record,
                "full_pipeline",
                expected_requested_cells=(
                    params.get("n_cells_requested")
                    if isinstance(params, dict)
                    and _positive_int(params.get("n_cells_requested"))
                    else None
                ),
                expected_matched_cells=min_matched_cells,
            )
        )
        failures.extend(_matrix_input_failures(record, "full_pipeline"))
        failures.extend(_sparse_rna_backend_failures(record, "full_pipeline"))

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
        failures.extend(_pipeline_manifest_failures(record))
    failures.extend(_output_storage_failures(record))
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

    subset_rows = record.get("subset_scaling")
    thread_rows = record.get("thread_scaling")
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
            failures.extend(
                _backend_execution_failures(
                    row,
                    prefix,
                    required_rust_stages={"grn"},
                )
            )
            failures.extend(
                _matrix_input_failures(
                    row,
                    prefix,
                    shape_keys=("rna_post_qc",),
                )
            )
            failures.extend(_sparse_grn_backend_failures(row, prefix))
    failures.extend(_grn_scaling_summary_failures(subset_rows, thread_rows, record))
    return failures


def _grn_scaling_summary_failures(
    subset_rows: Any,
    thread_rows: Any,
    record: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if isinstance(subset_rows, list):
        subset_wall = _rounded_slope(
            [row for row in subset_rows if isinstance(row, dict)],
            "n_cells",
            "grn_wall_s",
        )
        subset_memory = _rounded_slope(
            [row for row in subset_rows if isinstance(row, dict)],
            "n_cells",
            "peak_rss_gb",
        )
        for key, expected in (
            ("subset_wall_slope_vs_cells", subset_wall),
            ("subset_memory_slope_vs_cells", subset_memory),
        ):
            actual = record.get(key)
            if expected is None:
                if actual is not None:
                    failures.append(
                        f"{key} must be null when fewer than two subset rows are usable"
                    )
            elif actual != expected:
                failures.append(
                    f"{key} must match subset_scaling: {actual} != {expected}"
                )

    if isinstance(thread_rows, list):
        expected_speedups = _grn_thread_speedups(
            [row for row in thread_rows if isinstance(row, dict)]
        )
        actual_speedups = record.get("thread_speedups")
        if not isinstance(actual_speedups, list):
            failures.append("thread_speedups must be a list")
        elif actual_speedups != expected_speedups:
            failures.append("thread_speedups must match thread_scaling rows")
    return failures


def _grn_thread_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if _positive_int(row.get("threads"))
        and _positive_number(row.get("grn_wall_s"))
    ]
    if not usable:
        return []
    by_threads = sorted(usable, key=lambda row: int(row["threads"]))
    baseline = next((row for row in by_threads if row["threads"] == 1), by_threads[0])
    base_threads = int(baseline["threads"])
    base_wall = float(baseline["grn_wall_s"])
    return [
        {
            "threads": int(row["threads"]),
            "wall_s": row["grn_wall_s"],
            "speedup_vs_baseline": round(base_wall / float(row["grn_wall_s"]), 3),
            "efficiency_vs_baseline": round(
                (base_wall / float(row["grn_wall_s"]))
                / (int(row["threads"]) / base_threads),
                3,
            ),
        }
        for row in by_threads
    ]


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
    require_region_cistarget = _region_motif_rankings_supplied(record)

    runs = record.get("runs")
    if not isinstance(runs, list) or not runs:
        failures.append("runs must contain at least one full-pipeline row")
        return failures
    failures.extend(_full_pipeline_scaling_design_failures(record, runs))
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
                require_region_cistarget=require_region_cistarget,
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
                "pipeline_compute_stage_wall_s": row.get("wall_s", {}).get(
                    "pipeline_compute_stages"
                )
                if isinstance(row.get("wall_s"), dict) else None,
                "pipeline_io_stage_wall_s": row.get("wall_s", {}).get(
                    "pipeline_io_stages"
                )
                if isinstance(row.get("wall_s"), dict) else None,
                "pipeline_unattributed_wall_s": row.get("wall_s", {}).get(
                    "pipeline_unattributed"
                )
                if isinstance(row.get("wall_s"), dict) else None,
                "peak_rss_gb": row.get("peak_rss_gb"),
                "output_total_size_gb": row.get("output_storage", {}).get(
                    "total_size_gb"
                )
                if isinstance(row.get("output_storage"), dict) else None,
            }
            for row in runs
            if isinstance(row, dict)
        ]
        for key in (
            "end_to_end_wall_slope_vs_cells",
            "pipeline_wall_slope_vs_cells",
            "pipeline_compute_stage_wall_slope_vs_cells",
            "pipeline_unattributed_wall_slope_vs_cells",
        ):
            if not _nonnegative_number(scaling.get(key)):
                failures.append(f"scaling.{key} must be non-negative")
        if "pipeline_io_stage_wall_slope_vs_cells" in scaling:
            if not _nonnegative_number(
                scaling.get("pipeline_io_stage_wall_slope_vs_cells")
            ):
                failures.append(
                    "scaling.pipeline_io_stage_wall_slope_vs_cells must be non-negative"
                )
        if not _finite_number(scaling.get("peak_rss_slope_vs_cells")):
            failures.append("scaling.peak_rss_slope_vs_cells must be a finite number")
        if "output_total_size_slope_vs_cells" in scaling:
            if not _finite_number(scaling.get("output_total_size_slope_vs_cells")):
                failures.append(
                    "scaling.output_total_size_slope_vs_cells must be a finite number"
                )
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
        }
        for key, expected in expected_slopes.items():
            if (
                key == "pipeline_io_stage_wall_slope_vs_cells"
                and key not in scaling
                and all(row.get("pipeline_io_stage_wall_s") is None for row in wall_rows)
            ):
                continue
            if (
                key == "output_total_size_slope_vs_cells"
                and key not in scaling
                and all(row.get("output_total_size_gb") is None for row in wall_rows)
            ):
                continue
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
