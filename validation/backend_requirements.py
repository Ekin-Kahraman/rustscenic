"""Shared Rust backend capability contract for validation and HPC runs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


REQUIRED_RUST_BACKEND_SYMBOLS: dict[str, list[str]] = {
    "grn": ["grn_infer", "grn_infer_sparse_csc"],
    "aucell": ["aucell_score", "aucell_score_sparse_csr"],
    "topics": ["topics_fit", "topics_fit_gibbs", "topics_npmi", "topics_cell_assignment"],
    "cistarget": [
        "cistarget_enrichment_from_rankings_i16",
        "cistarget_enrichment_from_rankings_i32",
        "cistarget_enrichment_from_rankings_i64",
        "cistarget_rankings_to_i32_f32",
        "cistarget_rankings_to_i32_f64",
        "cistarget_motif_annotation_prune_rows_filtered_f32",
        "cistarget_motif_annotation_prune_rows_filtered_f64",
        "cistarget_motif_annotation_prune_standard_rows_f32",
        "cistarget_motif_annotation_prune_standard_rows_f64",
        "cistarget_prune_regulon_targets_f32",
        "cistarget_prune_regulon_targets_f64",
        "cistarget_prune_regulon_targets_i16",
        "cistarget_prune_regulon_targets_i32",
        "cistarget_prune_regulon_targets_i64",
        "cistarget_prune_regulon_targets_unranked",
        "cistarget_region_attribution_i16",
        "cistarget_region_attribution_i32",
        "cistarget_region_attribution_i64",
        "cistarget_region_attribution_peak_values_i16",
        "cistarget_region_attribution_peak_values_i32",
        "cistarget_region_attribution_peak_values_i64",
    ],
    "enhancer": [
        "enhancer_align_cell_indices",
        "enhancer_link_pearson",
        "enhancer_link_pearson_sparse_rna",
        "enhancer_match_gene_coords_to_rna",
        "enhancer_normalise_chrom_codes",
        "enhancer_parse_peak_names",
        "enhancer_prepare_gene_order",
    ],
    "eregulon": [
        "eregulon_assemble",
        "eregulon_assemble_f32",
        "eregulon_assemble_summary",
        "eregulon_assemble_summary_f32",
    ],
    "preproc": [
        "preproc_call_peaks",
        "preproc_fragments_to_matrix",
        "preproc_frip",
        "preproc_insert_size_stats",
        "preproc_peak_coords_for_names",
        "preproc_tss_enrichment",
    ],
    "pipeline": [
        "pipeline_candidate_regulons_from_grn",
        "pipeline_attribute_peaks_to_cistarget_rows_f32",
        "pipeline_attribute_peaks_to_cistarget_rows_f64",
        "pipeline_expand_region_cistarget_rows_f32",
        "pipeline_expand_region_cistarget_rows_f64",
        "pipeline_filter_cistarget_peak_rows_f32",
        "pipeline_filter_cistarget_peak_rows_f64",
        "pipeline_match_atac_cell_indices",
        "pipeline_peak_regulons_and_features_from_edges",
        "pipeline_project_ranking_columns",
    ],
    "gene_resolution": [
        "gene_duplicate_summary",
        "stage_prepare_regulon_indices",
        "stage_prepare_regulon_indices_with_coverage",
        "stage_regulon_coverage_counts",
        "gene_dedupe_sparse_csc_f32",
        "gene_dedupe_sparse_csc_f64",
        "gene_dedupe_sparse_csc_i32",
        "gene_dedupe_sparse_csc_i64",
        "gene_dedupe_sparse_csc_u32",
        "gene_dedupe_sparse_csc_u64",
        "gene_dedupe_dense_f32",
        "gene_dedupe_dense_f64",
        "gene_dedupe_dense_i32",
        "gene_dedupe_dense_i64",
        "gene_dedupe_dense_u32",
        "gene_dedupe_dense_u64",
    ],
    "specificity": [
        "specificity_candidate_top_indices",
        "specificity_candidate_top_indices_f32",
        "specificity_group_codes_with_numeric_order",
        "specificity_group_codes_with_order",
        "specificity_rss",
        "specificity_rss_f32",
    ],
}


def missing_backend_symbols(
    extension: Any,
    required: dict[str, list[str]] = REQUIRED_RUST_BACKEND_SYMBOLS,
) -> list[str]:
    """Return ``stage.symbol`` entries missing from a RustScenic extension."""
    return [
        f"{stage}.{symbol}"
        for stage, symbols in required.items()
        for symbol in symbols
        if not hasattr(extension, symbol)
    ]


def backend_capabilities() -> dict[str, Any]:
    """Record Rust extension symbols required by the benchmarked pipeline."""
    try:
        import rustscenic._rustscenic as ext
    except Exception as exc:  # pragma: no cover - benchmark provenance path
        return {
            "ok": False,
            "extension_error": repr(exc),
            "required_symbols": deepcopy(REQUIRED_RUST_BACKEND_SYMBOLS),
            "missing_symbols": [
                f"{stage}.{symbol}"
                for stage, symbols in REQUIRED_RUST_BACKEND_SYMBOLS.items()
                for symbol in symbols
            ],
        }

    missing = missing_backend_symbols(ext)
    return {
        "ok": not missing,
        "extension_error": None,
        "required_symbols": deepcopy(REQUIRED_RUST_BACKEND_SYMBOLS),
        "missing_symbols": missing,
    }
