"""Type stubs for the rustscenic PyO3 extension.

These reflect the real signatures in crates/rustscenic-py/src/lib.rs.
Kept in sync by convention - CI's import-smoke step will complain loudly
if these drift.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

__version__: str


def grn_infer(
    expression: npt.NDArray[np.float32],
    gene_names: list[str],
    tf_names: list[str],
    n_estimators: int = 5000,
    learning_rate: float = 0.01,
    max_features: float = 0.1,
    subsample: float = 0.9,
    max_depth: int = 3,
    early_stop_window: int = 25,
    seed: int = 777,
    target_block_size: int = 0,
    top_targets_per_tf: int | None = None,
    min_importance: float | None = None,
) -> tuple[list[str], list[str], npt.NDArray[np.float32], int, int, list[str], float]:
    """Infer (TF, target, importance) edges from an expression matrix.

    Returns filtered edge arrays, raw fitted edge count, TF overlap metadata,
    and the expression maximum collected during Rust validation.
    """


def grn_infer_sparse_csc(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.float32],
    n_cells: int,
    n_genes: int,
    gene_names: list[str],
    tf_names: list[str],
    n_estimators: int = 5000,
    learning_rate: float = 0.01,
    max_features: float = 0.1,
    subsample: float = 0.9,
    max_depth: int = 3,
    early_stop_window: int = 25,
    seed: int = 777,
    target_block_size: int = 0,
    top_targets_per_tf: int | None = None,
    min_importance: float | None = None,
) -> tuple[list[str], list[str], npt.NDArray[np.float32], int, int, list[str], float]:
    """Infer sparse GRN edges and TF overlap metadata without full densification."""


def pipeline_candidate_regulons_from_grn(
    grn_tfs: list[str],
    grn_targets: list[str],
    importances: npt.NDArray[np.float64],
    top_targets: int,
    min_targets: int,
) -> tuple[list[str], list[list[str]]]:
    """Build top-target candidate regulons directly from GRN edge columns."""


def aucell_score(
    expression: npt.NDArray[np.float32],
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
) -> tuple[npt.NDArray[np.float32], float]:
    """Per-cell regulon recovery AUCs.

    Returns a (n_cells, n_regulons) f32 array of normalised AUCs in [0, 1].
    """


def aucell_score_sparse_csr(
    row_ptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    col_idx: npt.NDArray[np.int32],
    data: npt.NDArray[np.float32],
    n_genes: int,
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
) -> tuple[npt.NDArray[np.float32], float]:
    """Per-cell AUCell for CSR sparse expression without dense materialisation."""


def cistarget_enrichment_from_rankings_i16(
    rankings: npt.NDArray[np.int16],
    motif_names: list[str],
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
    auc_threshold: float = 0.05,
    nes_threshold: float | None = None,
    nes_min_motifs: int = 30,
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    bool,
]:
    """Compute cistarget enrichment rows directly from int16 rank matrices."""


def cistarget_enrichment_from_rankings_i32(
    rankings: npt.NDArray[np.int32],
    motif_names: list[str],
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
    auc_threshold: float = 0.05,
    nes_threshold: float | None = None,
    nes_min_motifs: int = 30,
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    bool,
]:
    """Compute cistarget enrichment rows directly from int32 rank matrices."""


def cistarget_enrichment_from_rankings_i64(
    rankings: npt.NDArray[np.int64],
    motif_names: list[str],
    regulon_names: list[str],
    regulon_gene_indices: list[list[int]],
    top_frac: float = 0.05,
    auc_threshold: float = 0.05,
    nes_threshold: float | None = None,
    nes_min_motifs: int = 30,
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
    bool,
]:
    """Compute cistarget enrichment rows directly from int64 rank matrices."""


def cistarget_rankings_to_i32_f32(
    rankings: npt.NDArray[np.float32],
) -> npt.NDArray[np.int32]:
    """Validate integer-like float32 rankings and convert them to int32 in Rust."""


def cistarget_rankings_to_i32_f64(
    rankings: npt.NDArray[np.float64],
) -> npt.NDArray[np.int32]:
    """Validate integer-like float64 rankings and convert them to int32 in Rust."""


def topics_fit(
    row_ptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    col_idx: npt.NDArray[np.int32],
    counts: npt.NDArray[np.float32],
    n_words: int,
    n_topics: int = 50,
    alpha: float = 0.02,
    eta: float = 0.02,
    tau0: float = 64.0,
    kappa: float = 0.7,
    batch_size: int = 256,
    n_passes: int = 10,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fit Online VB LDA; return (cell_topic, topic_word) probability matrices."""


def topics_fit_gibbs(
    row_ptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    col_idx: npt.NDArray[np.int32],
    counts: npt.NDArray[np.float32],
    n_words: int,
    n_topics: int = 50,
    alpha: float = 0.1,
    eta: float = 0.01,
    n_iters: int = 200,
    seed: int = 42,
    n_threads: int = 1,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fit collapsed-Gibbs LDA; return (cell_topic, topic_word) matrices."""


def topics_npmi(
    topic_word: npt.NDArray[np.float32],
    n_topics: int,
    n_words: int,
    row_ptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    col_idx: npt.NDArray[np.int32],
    top_n: int = 10,
) -> npt.NDArray[np.float32]:
    """Per-topic NPMI coherence over top-ranked words."""


def topics_cell_assignment(
    cell_topic: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int64], int, int]:
    """Return argmax topic index per cell, active topic count and empty cell count."""


def specificity_group_codes_with_order(
    labels: list[str],
    missing: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint64]]:
    """Discover string-sorted RSS groups and encode labels in one Rust pass."""


def specificity_group_codes_with_numeric_order(
    labels: list[str],
    missing: npt.NDArray[np.bool_],
    numeric_values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint64]]:
    """Discover numerically sorted RSS groups and encode labels in one Rust pass."""


def specificity_rss(
    auc: npt.NDArray[np.float64],
    group_codes: npt.NDArray[np.int32],
    n_groups: int,
) -> npt.NDArray[np.float64]:
    """Compute regulon specificity scores from AUCell values and group codes."""


def specificity_rss_f32(
    auc: npt.NDArray[np.float32],
    group_codes: npt.NDArray[np.int32],
    n_groups: int,
) -> npt.NDArray[np.float64]:
    """Compute regulon specificity scores from float32 AUCell values."""


def specificity_candidate_top_indices(
    topic_peak: npt.NDArray[np.float64],
    top_n: int,
) -> npt.NDArray[np.uint64]:
    """Top candidate enhancer indices per topic, sorted by descending weight."""


def specificity_candidate_top_indices_f32(
    topic_peak: npt.NDArray[np.float32],
    top_n: int,
) -> npt.NDArray[np.uint64]:
    """Top candidate enhancer indices per topic from float32 weights."""


def enhancer_align_cell_indices(
    rna_cells: list[str],
    atac_cells: list[str],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Return matched RNA and ATAC cell indices in RNA barcode order."""


def enhancer_normalise_chrom_codes(
    gene_chroms: list[str],
    peak_chroms: list[str],
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Normalise gene/peak chromosomes and return shared integer codes."""


def enhancer_parse_peak_names(
    names: list[str],
) -> tuple[list[str], npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
    """Parse peak names into chromosome, start, and end coordinate arrays."""


def enhancer_match_gene_coords_to_rna(
    rna_gene_names: list[str],
    gene_coord_names: list[str],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Match gene-coordinate rows to RNA feature columns."""


def enhancer_prepare_gene_order(
    peak_chrom_codes: npt.NDArray[np.int32],
    gene_chrom_codes: npt.NDArray[np.int32],
    gene_tss: npt.NDArray[np.int64],
    gene_source_cols: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.uint64], bool, int]:
    """Sort enhancer gene coordinates and return overlap/unique-column metadata."""


def enhancer_link_pearson(
    rna: npt.NDArray[np.float32],
    atac_indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    atac_indices: npt.NDArray[np.int32],
    atac_data: npt.NDArray[np.float32],
    peak_chrom_codes: npt.NDArray[np.int32],
    peak_starts: npt.NDArray[np.int64],
    peak_ends: npt.NDArray[np.int64],
    gene_chrom_codes: npt.NDArray[np.int32],
    gene_tss: npt.NDArray[np.int64],
    gene_rna_cols: npt.NDArray[np.uint32],
    max_distance: int,
    min_abs_corr: float,
    atac_peak_cols: npt.NDArray[np.uint32] | None = None,
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
]:
    """Sparse-ATAC Pearson linker core for enhancer-to-gene links."""


def enhancer_link_pearson_sparse_rna(
    rna_indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    rna_indices: npt.NDArray[np.int32],
    rna_data: npt.NDArray[np.float32],
    n_cells: int,
    n_rna_genes: int,
    atac_indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    atac_indices: npt.NDArray[np.int32],
    atac_data: npt.NDArray[np.float32],
    peak_chrom_codes: npt.NDArray[np.int32],
    peak_starts: npt.NDArray[np.int64],
    peak_ends: npt.NDArray[np.int64],
    gene_chrom_codes: npt.NDArray[np.int32],
    gene_tss: npt.NDArray[np.int64],
    gene_rna_cols: npt.NDArray[np.uint32],
    max_distance: int,
    min_abs_corr: float,
    atac_peak_cols: npt.NDArray[np.uint32] | None = None,
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
]:
    """Sparse-RNA and sparse-ATAC Pearson linker core."""


def eregulon_assemble(
    grn_tfs: list[str],
    grn_targets: list[str],
    cistarget_regulons: list[str],
    cistarget_peaks: list[str],
    cistarget_aucs: npt.NDArray[np.float64],
    enhancer_peaks: list[str],
    enhancer_genes: list[str],
    enhancer_correlations: npt.NDArray[np.float32],
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> tuple[
    list[str],
    list[str],
    list[str],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float64],
    int,
    int,
]:
    """Assemble long-form eRegulon table columns from typed edge columns."""


def eregulon_assemble_f32(
    grn_tfs: list[str],
    grn_targets: list[str],
    cistarget_regulons: list[str],
    cistarget_peaks: list[str],
    cistarget_aucs: npt.NDArray[np.float32],
    enhancer_peaks: list[str],
    enhancer_genes: list[str],
    enhancer_correlations: npt.NDArray[np.float32],
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> tuple[
    list[str],
    list[str],
    list[str],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float64],
    int,
    int,
]:
    """Assemble long-form eRegulon table columns from float32 cistarget AUCs."""


def eregulon_assemble_summary(
    grn_tfs: list[str],
    grn_targets: list[str],
    cistarget_regulons: list[str],
    cistarget_peaks: list[str],
    cistarget_aucs: npt.NDArray[np.float64],
    enhancer_peaks: list[str],
    enhancer_genes: list[str],
    enhancer_correlations: npt.NDArray[np.float32],
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> tuple[
    list[str],
    list[list[str]],
    list[list[str]],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float64],
    list[dict[str, list[str]]],
    int,
    int,
]:
    """Assemble grouped eRegulon summaries from typed edge columns."""


def eregulon_assemble_summary_f32(
    grn_tfs: list[str],
    grn_targets: list[str],
    cistarget_regulons: list[str],
    cistarget_peaks: list[str],
    cistarget_aucs: npt.NDArray[np.float32],
    enhancer_peaks: list[str],
    enhancer_genes: list[str],
    enhancer_correlations: npt.NDArray[np.float32],
    min_target_genes: int = 5,
    min_enhancer_links: int = 2,
    cistarget_auc_threshold: float = 0.05,
    use_grn_intersection: bool = True,
) -> tuple[
    list[str],
    list[list[str]],
    list[list[str]],
    npt.NDArray[np.uint32],
    npt.NDArray[np.float64],
    list[dict[str, list[str]]],
    int,
    int,
]:
    """Assemble grouped eRegulon summaries from float32 cistarget AUCs."""


def cistarget_region_attribution_i16(
    rankings: npt.NDArray[np.int16],
    motif_names: list[str],
    peak_names: list[str],
    peak_regulon_names: list[str],
    peak_regulon_peaks: list[list[str]],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    rank_cutoff: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Attribute enriched region motifs to enriched-row and peak-column indices."""


def cistarget_region_attribution_i32(
    rankings: npt.NDArray[np.int32],
    motif_names: list[str],
    peak_names: list[str],
    peak_regulon_names: list[str],
    peak_regulon_peaks: list[list[str]],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    rank_cutoff: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Attribute enriched region motifs to enriched-row and peak-column indices."""


def cistarget_region_attribution_i64(
    rankings: npt.NDArray[np.int64],
    motif_names: list[str],
    peak_names: list[str],
    peak_regulon_names: list[str],
    peak_regulon_peaks: list[list[str]],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    rank_cutoff: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Attribute enriched region motifs to enriched-row and peak-column indices."""


def cistarget_region_attribution_peak_values_i16(
    rankings: npt.NDArray[np.int16],
    motif_names: list[str],
    peak_names: list[str],
    peak_regulon_names: list[str],
    peak_regulon_peaks: list[list[str]],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    rank_cutoff: int,
) -> tuple[npt.NDArray[np.uint64], list[str]]:
    """Attribute enriched region motifs to enriched-row indices and peak IDs."""


def cistarget_region_attribution_peak_values_i32(
    rankings: npt.NDArray[np.int32],
    motif_names: list[str],
    peak_names: list[str],
    peak_regulon_names: list[str],
    peak_regulon_peaks: list[list[str]],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    rank_cutoff: int,
) -> tuple[npt.NDArray[np.uint64], list[str]]:
    """Attribute enriched region motifs to enriched-row indices and peak IDs."""


def cistarget_region_attribution_peak_values_i64(
    rankings: npt.NDArray[np.int64],
    motif_names: list[str],
    peak_names: list[str],
    peak_regulon_names: list[str],
    peak_regulon_peaks: list[list[str]],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    rank_cutoff: int,
) -> tuple[npt.NDArray[np.uint64], list[str]]:
    """Attribute enriched region motifs to enriched-row indices and peak IDs."""


def pipeline_match_atac_cell_indices(
    rna_cells: list[str],
    atac_cells: list[str],
) -> npt.NDArray[np.uint64]:
    """Return ATAC row indices whose barcodes are present in RNA."""


def pipeline_attribute_peaks_to_cistarget_rows_f32(
    enriched_regulons: list[str],
    enriched_motifs: list[str] | None,
    enriched_aucs: npt.NDArray[np.float32],
    regulon_names: list[str],
    regulon_targets: list[list[str]],
    enhancer_genes: list[str],
    enhancer_peaks: list[str],
) -> tuple[list[str], list[str | None], list[str], npt.NDArray[np.float32]]:
    """Attribute gene-ranking cistarget rows to expanded peak-aware rows."""


def pipeline_attribute_peaks_to_cistarget_rows_f64(
    enriched_regulons: list[str],
    enriched_motifs: list[str] | None,
    enriched_aucs: npt.NDArray[np.float64],
    regulon_names: list[str],
    regulon_targets: list[list[str]],
    enhancer_genes: list[str],
    enhancer_peaks: list[str],
) -> tuple[list[str], list[str | None], list[str], npt.NDArray[np.float64]]:
    """Attribute gene-ranking cistarget rows to expanded peak-aware rows."""


def pipeline_expand_region_cistarget_rows_f32(
    row_indices: npt.NDArray[np.uint64],
    peak_ids: list[str],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    enriched_aucs: npt.NDArray[np.float32],
) -> tuple[list[str], list[str], list[str], npt.NDArray[np.float32]]:
    """Expand region-cistarget attribution indices to peak-aware output rows."""


def pipeline_expand_region_cistarget_rows_f64(
    row_indices: npt.NDArray[np.uint64],
    peak_ids: list[str],
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    enriched_aucs: npt.NDArray[np.float64],
) -> tuple[list[str], list[str], list[str], npt.NDArray[np.float64]]:
    """Expand region-cistarget attribution indices to peak-aware output rows."""


def cistarget_motif_annotation_prune_rows_filtered_f32(
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    enriched_auc: npt.NDArray[np.float32],
    enriched_nes: npt.NDArray[np.float32] | None,
    annotation_motifs: list[str],
    annotation_tfs: list[str],
    auc_threshold: float | None = None,
    nes_threshold: float | None = None,
    case_sensitive: bool = False,
) -> tuple[npt.NDArray[np.uint64], list[str], list[str]]:
    """Prune enriched motif rows with Rust-side AUC/NES threshold filtering."""


def cistarget_motif_annotation_prune_rows_filtered_f64(
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    enriched_auc: npt.NDArray[np.float64],
    enriched_nes: npt.NDArray[np.float64] | None,
    annotation_motifs: list[str],
    annotation_tfs: list[str],
    auc_threshold: float | None = None,
    nes_threshold: float | None = None,
    case_sensitive: bool = False,
) -> tuple[npt.NDArray[np.uint64], list[str], list[str]]:
    """Prune enriched motif rows with Rust-side AUC/NES threshold filtering."""


def cistarget_motif_annotation_prune_standard_rows_f32(
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    enriched_auc: npt.NDArray[np.float32],
    enriched_nes: npt.NDArray[np.float32] | None,
    annotation_motifs: list[str],
    annotation_tfs: list[str],
    auc_threshold: float | None = None,
    nes_threshold: float | None = None,
    case_sensitive: bool = False,
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32] | None,
    list[str],
    list[str],
]:
    """Prune standard enriched motif rows and return final output columns."""


def cistarget_motif_annotation_prune_standard_rows_f64(
    enriched_regulons: list[str],
    enriched_motifs: list[str],
    enriched_auc: npt.NDArray[np.float64],
    enriched_nes: npt.NDArray[np.float64] | None,
    annotation_motifs: list[str],
    annotation_tfs: list[str],
    auc_threshold: float | None = None,
    nes_threshold: float | None = None,
    case_sensitive: bool = False,
) -> tuple[
    list[str],
    list[str],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
    list[str],
    list[str],
]:
    """Prune standard enriched motif rows and return final output columns."""


def pipeline_filter_cistarget_peak_rows_f32(
    row_regulons: list[str],
    row_motifs: list[str],
    row_peaks: list[str],
    row_aucs: npt.NDArray[np.float32],
    keep_regulons: list[str],
    keep_motifs: list[str],
) -> tuple[list[str], list[str], list[str], npt.NDArray[np.float32]]:
    """Return cistarget peak rows kept by regulon/motif pairs."""


def pipeline_filter_cistarget_peak_rows_f64(
    row_regulons: list[str],
    row_motifs: list[str],
    row_peaks: list[str],
    row_aucs: npt.NDArray[np.float64],
    keep_regulons: list[str],
    keep_motifs: list[str],
) -> tuple[list[str], list[str], list[str], npt.NDArray[np.float64]]:
    """Return cistarget peak rows kept by regulon/motif pairs."""


def stage_prepare_regulon_indices(
    gene_names: list[str],
    regulon_names: list[str],
    regulon_genes: list[list[str]],
) -> tuple[list[str], list[list[int]], int]:
    """Map regulon gene names onto a feature axis."""


def stage_prepare_regulon_indices_with_coverage(
    gene_names: list[str],
    regulon_names: list[str],
    regulon_genes: list[list[str]],
) -> tuple[
    list[str],
    list[list[int]],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    int,
]:
    """Map regulon genes and count coverage in one Rust pass."""


def stage_regulon_coverage_counts(
    gene_names: list[str],
    regulon_genes: list[list[str]],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Count per-regulon matched and total genes."""


def cistarget_prune_regulon_targets_i16(
    rankings: npt.NDArray[np.int16],
    motif_names: list[str],
    gene_names: list[str],
    candidate_names: list[str],
    candidate_genes: list[list[str]],
    pruned_regulons: list[str],
    pruned_motifs: list[str],
    rank_cutoff: int,
    min_genes: int,
) -> tuple[list[str], list[list[str]]]:
    """Recover pruned regulon targets from int16 motif rankings."""


def cistarget_prune_regulon_targets_i32(
    rankings: npt.NDArray[np.int32],
    motif_names: list[str],
    gene_names: list[str],
    candidate_names: list[str],
    candidate_genes: list[list[str]],
    pruned_regulons: list[str],
    pruned_motifs: list[str],
    rank_cutoff: int,
    min_genes: int,
) -> tuple[list[str], list[list[str]]]:
    """Recover pruned regulon targets from integer motif rankings."""


def cistarget_prune_regulon_targets_i64(
    rankings: npt.NDArray[np.int64],
    motif_names: list[str],
    gene_names: list[str],
    candidate_names: list[str],
    candidate_genes: list[list[str]],
    pruned_regulons: list[str],
    pruned_motifs: list[str],
    rank_cutoff: int,
    min_genes: int,
) -> tuple[list[str], list[list[str]]]:
    """Recover pruned regulon targets from int64 motif rankings."""


def cistarget_prune_regulon_targets_f32(
    rankings: npt.NDArray[np.float32],
    motif_names: list[str],
    gene_names: list[str],
    candidate_names: list[str],
    candidate_genes: list[list[str]],
    pruned_regulons: list[str],
    pruned_motifs: list[str],
    rank_cutoff: int,
    min_genes: int,
) -> tuple[list[str], list[list[str]]]:
    """Recover pruned regulon targets from float32 motif rankings."""


def cistarget_prune_regulon_targets_f64(
    rankings: npt.NDArray[np.float64],
    motif_names: list[str],
    gene_names: list[str],
    candidate_names: list[str],
    candidate_genes: list[list[str]],
    pruned_regulons: list[str],
    pruned_motifs: list[str],
    rank_cutoff: int,
    min_genes: int,
) -> tuple[list[str], list[list[str]]]:
    """Recover pruned regulon targets from floating-point motif rankings."""


def cistarget_prune_regulon_targets_unranked(
    candidate_names: list[str],
    candidate_genes: list[list[str]],
    pruned_regulons: list[str],
    min_genes: int,
) -> tuple[list[str], list[list[str]]]:
    """Keep annotation-supported regulons when no ranking matrix is supplied."""


def pipeline_peak_regulons_and_features_from_edges(
    grn_tfs: list[str],
    grn_targets: list[str],
    enhancer_genes: list[str],
    enhancer_peaks: list[str],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Build TF peak-regulons plus sorted unique peak IDs for ranking projection."""


def pipeline_project_ranking_columns(
    columns: list[str],
    requested_features: list[str],
    motif_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """Select motif-ranking column names for requested features."""


def preproc_fragments_to_matrix(
    fragments_path: str,
    peaks_path: str,
) -> tuple[
    npt.NDArray[np.uint32],   # data
    npt.NDArray[np.uint32],   # indices
    npt.NDArray[np.uint64],   # indptr
    tuple[int, int],          # shape
    list[str],                # barcodes
    list[str],                # peak ids
    npt.NDArray[np.uint32],   # fragments_per_cell
    npt.NDArray[np.uint32],   # total_counts
    int,                      # total fragment records
    float,                    # median fragments per barcode, NaN when not computed
]:
    """Parse a 10x fragments.tsv[.gz] and peaks BED into CSR cells×peaks."""


def preproc_peak_coords_for_names(
    peaks_path: str,
    peak_names: list[str],
) -> tuple[
    list[str],                # peak ids
    list[str],                # chroms
    npt.NDArray[np.uint32],   # starts
    npt.NDArray[np.uint32],   # ends
]:
    """Read peak BED coordinates aligned to a requested peak-name axis."""


def preproc_insert_size_stats(
    fragments_path: str,
) -> tuple[
    list[str],                # barcodes
    npt.NDArray[np.float32],  # mean
    npt.NDArray[np.float32],  # median
    npt.NDArray[np.uint32],   # n_fragments
    npt.NDArray[np.uint32],   # sub-nucleosomal
    npt.NDArray[np.uint32],   # mono-nucleosomal
    npt.NDArray[np.uint32],   # di-nucleosomal
]:
    """Per-barcode insert-size summary statistics."""


def preproc_frip(
    fragments_path: str,
    peaks_path: str,
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Per-barcode fraction of reads in peaks. Returns (barcodes, frip)."""


def preproc_tss_enrichment(
    fragments_path: str,
    tss_chroms: list[str],
    tss_positions: list[int],
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Per-barcode TSS enrichment (signal-over-background)."""


def preproc_call_peaks(
    fragments_path: str,
    cluster_per_barcode: (
        npt.NDArray[np.int32]
        | npt.NDArray[np.int64]
        | npt.NDArray[np.uint32]
        | npt.NDArray[np.uint64]
    ),
    n_clusters: int | None = None,
    window_size: int = 50,
    min_fragments_per_window: int = 3,
    quantile_threshold: float = 0.95,
    max_gap: int = 250,
    peak_half_width: int = 250,
    cluster_barcodes: list[str] | None = None,
) -> tuple[list[str], npt.NDArray[np.uint32], npt.NDArray[np.uint32], list[str]]:
    """Iterative consensus peak calling. Returns (chroms, starts, ends, names)."""


def gene_duplicate_summary(
    gene_names: list[str],
    max_examples: int = 3,
) -> tuple[int, list[str]]:
    """Count duplicate gene names and return top duplicate examples."""


def gene_dedupe_dense_f32(
    expression: npt.NDArray[np.float32],
    gene_names: list[str],
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Collapse duplicate dense float32 gene columns by summing in Rust."""


def gene_dedupe_dense_f64(
    expression: npt.NDArray[np.float64],
    gene_names: list[str],
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Collapse duplicate dense float64 gene columns by summing in Rust."""


def gene_dedupe_dense_i32(
    expression: npt.NDArray[np.int32],
    gene_names: list[str],
) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Collapse duplicate dense int32 gene columns by summing in Rust."""


def gene_dedupe_dense_i64(
    expression: npt.NDArray[np.int64],
    gene_names: list[str],
) -> tuple[npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate dense int64 gene columns by summing in Rust."""


def gene_dedupe_dense_u32(
    expression: npt.NDArray[np.uint32],
    gene_names: list[str],
) -> tuple[npt.NDArray[np.uint32], list[str]]:
    """Collapse duplicate dense uint32 gene columns by summing in Rust."""


def gene_dedupe_dense_u64(
    expression: npt.NDArray[np.uint64],
    gene_names: list[str],
) -> tuple[npt.NDArray[np.uint64], list[str]]:
    """Collapse duplicate dense uint64 gene columns by summing in Rust."""


def gene_dedupe_sparse_csc_f32(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.float32],
    n_rows: int,
    gene_names: list[str],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate CSC float32 gene columns by summing in Rust."""


def gene_dedupe_sparse_csc_f64(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.float64],
    n_rows: int,
    gene_names: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate CSC float64 gene columns by summing in Rust."""


def gene_dedupe_sparse_csc_i32(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.int32],
    n_rows: int,
    gene_names: list[str],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate CSC int32 gene columns by summing in Rust."""


def gene_dedupe_sparse_csc_i64(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.int64],
    n_rows: int,
    gene_names: list[str],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate CSC int64 gene columns by summing in Rust."""


def gene_dedupe_sparse_csc_u32(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.uint32],
    n_rows: int,
    gene_names: list[str],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32], npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate CSC uint32 gene columns by summing in Rust."""


def gene_dedupe_sparse_csc_u64(
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint64],
    indices: npt.NDArray[np.int32],
    data: npt.NDArray[np.uint64],
    n_rows: int,
    gene_names: list[str],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64], list[str]]:
    """Collapse duplicate CSC uint64 gene columns by summing in Rust."""
