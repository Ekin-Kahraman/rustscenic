//! rustscenic-grn: GRNBoost2 replacement.
//!
//! Matches `arboreto.algo.grnboost2` semantics:
//! - sklearn `GradientBoostingRegressor` with squared-error loss
//! - n_estimators=5000, learning_rate=0.01, max_features=0.1, subsample=0.9,
//!   max_depth=3 (sklearn defaults inherited by arboreto's SGBM_KWARGS)
//! - arboreto's EarlyStopMonitor (window=25) halts when train MSE stalls
//! - Feature importance = sklearn-normalized split gains × n_estimators_fit
//!   (per arboreto/core.py:168)
//!
//! Not bit-identical to sklearn (different RNG tape + histogram quantization).
//!
//! Measured against arboreto on 10x Multiome 3k (n_estimators=5000, 2588 cells
//! × 1457 TFs, deep audit 2026-04-18):
//!     - per-edge Spearman on 816k common edges: 0.58
//!     - per-target TF-rank Spearman: mean 0.57, median 0.60
//!     - top-100 edge Jaccard: 0.10, top-1000: 0.30, top-100k: 0.32
//!
//! Biology still agrees: 94% known edges recovered (PBMC-3k), 8/8 lineage TFs
//! correctly enriched (PBMC-10k). Downstream AUCell is 0.99 per-cell Pearson
//! with pyscenic - fine-edge disagreement does not propagate to regulon
//! activity. See `validation/parity_v0310/grn_parity_pbmc3k_full.json`.

pub mod gbm;
pub mod histogram;
pub mod rng;
pub mod tree;

use ndarray::ArrayView2;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::histogram::BinnedMatrix;

#[derive(Debug, Clone)]
pub struct Adjacency {
    pub tf: String,
    pub target: String,
    pub importance: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexedAdjacency {
    /// Gene-column index of the transcription factor.
    pub tf_idx: usize,
    /// Gene-column index of the target gene.
    pub target_idx: usize,
    pub importance: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TfOverlapSummary {
    /// Number of supplied TF names present in the expression gene list.
    ///
    /// This intentionally counts input names before deduplicating TF columns so
    /// user-facing warnings match the supplied TF list semantics.
    pub present_input_count: usize,
    /// First missing TF names in input order, capped for concise warnings.
    pub missing_examples: Vec<String>,
}

struct ResolvedTfCols {
    cols: Vec<usize>,
    summary: TfOverlapSummary,
}

#[derive(Debug, Clone)]
pub struct GrnConfig {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_features: f32,
    pub subsample: f32,
    pub max_depth: usize,
    pub early_stop_window: usize,
    pub seed: u64,
    /// Retained for API compatibility. No longer affects execution: targets are
    /// fit in a single flat parallel pass over all genes (no blocking).
    pub target_block_size: usize,
}

impl Default for GrnConfig {
    fn default() -> Self {
        Self {
            n_estimators: 5000,
            learning_rate: 0.01,
            max_features: 0.1,
            subsample: 0.9,
            max_depth: 3,
            early_stop_window: 25,
            seed: 777,
            target_block_size: 0,
        }
    }
}

/// Infer a GRN from a dense (n_cells × n_genes) f32 expression matrix.
///
/// Pre-bins the TF matrix once, then processes target genes in cache-friendly
/// blocks while fitting targets inside each block in parallel.
pub fn infer(
    expression: &[f32],
    n_cells: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> Vec<Adjacency> {
    infer_indices(expression, n_cells, gene_names, tf_names, cfg)
        .into_iter()
        .map(|edge| Adjacency {
            tf: gene_names[edge.tf_idx].clone(),
            target: gene_names[edge.target_idx].clone(),
            importance: edge.importance,
        })
        .collect()
}

/// Infer a GRN and return gene-column indices instead of cloned edge strings.
///
/// This is the preferred path for bindings that already own the gene-name
/// table. Large GRNs can contain millions of edges, so carrying two cloned
/// `String`s per edge is a material memory and CPU cost.
pub fn infer_indices(
    expression: &[f32],
    n_cells: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> Vec<IndexedAdjacency> {
    infer_indices_with_overlap(expression, n_cells, gene_names, tf_names, cfg).0
}

/// Infer a GRN and return TF-list overlap metadata from the same Rust-side
/// resolution pass used for fitting.
pub fn infer_indices_with_overlap(
    expression: &[f32],
    n_cells: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> (Vec<IndexedAdjacency>, TfOverlapSummary) {
    let n_genes = gene_names.len();
    assert_eq!(
        expression.len(),
        n_cells * n_genes,
        "expression size mismatch"
    );

    let resolved = resolve_tf_cols_with_summary(gene_names, tf_names);
    let tf_cols = resolved.cols;
    let summary = resolved.summary;
    if tf_cols.is_empty() {
        return (Vec::new(), summary);
    }

    // Bin selected TF columns directly from the row-major expression matrix.
    // Per-target "drop self-TF" is handled by excluding that feature at fit
    // time rather than rebuilding the matrix.
    let binned_all = BinnedMatrix::from_dense_columns(expression, n_cells, n_genes, &tf_cols);

    let gene_to_tf_col: HashMap<usize, usize> = tf_cols
        .iter()
        .enumerate()
        .map(|(tf_col, &gene_idx)| (gene_idx, tf_col))
        .collect();

    let all_edges = fit_all_targets(
        &binned_all,
        n_cells,
        n_genes,
        &tf_cols,
        &gene_to_tf_col,
        cfg,
        |target_idx, buf| {
            // Row-major source: gather this target's response column once.
            for (c, slot) in buf.iter_mut().enumerate() {
                *slot = expression[c * n_genes + target_idx];
            }
        },
    );
    (all_edges, summary)
}

/// Infer a GRN from any dense (cells × genes) f32 matrix view.
///
/// The row-major `infer_indices` path remains the performance default. This
/// view path is for strided NumPy/pandas inputs where copying the whole matrix
/// would dominate peak RSS before Rust can do the useful work.
pub fn infer_indices_view(
    expression: ArrayView2<'_, f32>,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> Vec<IndexedAdjacency> {
    infer_indices_view_with_overlap(expression, gene_names, tf_names, cfg).0
}

/// Infer a GRN from any dense matrix view and return TF overlap metadata.
pub fn infer_indices_view_with_overlap(
    expression: ArrayView2<'_, f32>,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> (Vec<IndexedAdjacency>, TfOverlapSummary) {
    let n_cells = expression.shape()[0];
    let n_genes = expression.shape()[1];
    assert_eq!(gene_names.len(), n_genes, "gene_names size mismatch");

    let resolved = resolve_tf_cols_with_summary(gene_names, tf_names);
    let tf_cols = resolved.cols;
    let summary = resolved.summary;
    if tf_cols.is_empty() {
        return (Vec::new(), summary);
    }

    let binned_all = BinnedMatrix::from_dense_columns_view(expression, &tf_cols);
    let gene_to_tf_col: HashMap<usize, usize> = tf_cols
        .iter()
        .enumerate()
        .map(|(tf_col, &gene_idx)| (gene_idx, tf_col))
        .collect();

    let all_edges = fit_all_targets(
        &binned_all,
        n_cells,
        n_genes,
        &tf_cols,
        &gene_to_tf_col,
        cfg,
        |target_idx, buf| {
            for (c, slot) in buf.iter_mut().enumerate() {
                *slot = expression[(c, target_idx)];
            }
        },
    );
    (all_edges, summary)
}

/// Infer a GRN from a sparse CSC (cells × genes) expression matrix.
///
/// This avoids dense materialisation of the full RNA matrix. TF bins are built
/// directly from selected sparse columns, and only the current target block is
/// expanded to dense column-major response vectors for GBM fitting.
#[allow(clippy::too_many_arguments)]
pub fn infer_indices_sparse_csc(
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    n_cells: usize,
    n_genes: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> Vec<IndexedAdjacency> {
    infer_indices_sparse_csc_with_overlap(
        indptr, indices, data, n_cells, n_genes, gene_names, tf_names, cfg,
    )
    .0
}

/// Infer a GRN from sparse CSC expression and return TF overlap metadata.
#[allow(clippy::too_many_arguments)]
pub fn infer_indices_sparse_csc_with_overlap(
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    n_cells: usize,
    n_genes: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> (Vec<IndexedAdjacency>, TfOverlapSummary) {
    assert_eq!(gene_names.len(), n_genes, "gene_names size mismatch");
    assert_eq!(indptr.len(), n_genes + 1, "CSC indptr size mismatch");
    assert_eq!(indices.len(), data.len(), "CSC indices/data size mismatch");
    assert!(
        indices
            .iter()
            .all(|&row| row >= 0 && (row as usize) < n_cells),
        "CSC row index contains a negative or out-of-range cell index"
    );

    let resolved = resolve_tf_cols_with_summary(gene_names, tf_names);
    let tf_cols = resolved.cols;
    let summary = resolved.summary;
    if tf_cols.is_empty() {
        return (Vec::new(), summary);
    }

    let binned_all =
        BinnedMatrix::from_csc_columns(indptr, indices, data, n_cells, n_genes, &tf_cols);
    let gene_to_tf_col: HashMap<usize, usize> = tf_cols
        .iter()
        .enumerate()
        .map(|(tf_col, &gene_idx)| (gene_idx, tf_col))
        .collect();

    let all_edges = fit_all_targets(
        &binned_all,
        n_cells,
        n_genes,
        &tf_cols,
        &gene_to_tf_col,
        cfg,
        |target_idx, buf| {
            // Sparse CSC column: zero then accumulate nonzeros (preserves the
            // previous += semantics for duplicate row entries in a column).
            buf.fill(0.0);
            for p in indptr[target_idx]..indptr[target_idx + 1] {
                buf[indices[p] as usize] += data[p];
            }
        },
    );
    (all_edges, summary)
}

/// Fit every target gene in a SINGLE flat parallel pass.
///
/// Replaces the previous per-block `into_par_iter` + `collect()` barrier (one
/// hard synchronisation per ~256-gene block). Under early stopping the per-block
/// makespan serialised on the slowest straggler, and that tail grew with core
/// count -- the root cause of the 0.96@4c -> 0.71@16c thread-scaling decay. A
/// flat loop over `0..n_genes` exposes all targets to one Rayon split with no
/// barriers; each task gathers its own response column into a per-worker
/// `n_cells` buffer (O(n_cells), amortised against the GBM compute) instead of a
/// shared 128MB transpose window. Output is identical: targets are independent,
/// the RNG is keyed on `hash_y(y)` (order-independent), and `collect()`
/// preserves `0..n_genes` order exactly as the old block concatenation did.
fn fit_all_targets<F>(
    binned_all: &BinnedMatrix,
    n_cells: usize,
    n_genes: usize,
    tf_cols: &[usize],
    gene_to_tf_col: &HashMap<usize, usize>,
    cfg: &GrnConfig,
    gather_target: F,
) -> Vec<IndexedAdjacency>
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    (0..n_genes)
        .into_par_iter()
        .map_init(
            || {
                (
                    gbm::GbmScratch::new(n_cells, tf_cols.len(), cfg.n_estimators),
                    vec![0.0_f32; n_cells],
                )
            },
            |(gbm_scratch, target_buf), target_idx| {
                gather_target(target_idx, target_buf.as_mut_slice());

                // If this target is itself one of the TFs, drop that column at
                // fit time so the self column cannot absorb the split gain.
                let exclude_self = gene_to_tf_col.get(&target_idx).copied();

                gbm::fit_and_importances_binned_with_scratch(
                    binned_all,
                    target_buf.as_slice(),
                    cfg,
                    exclude_self,
                    gbm_scratch,
                )
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, imp)| *imp > 0.0)
                .map(|(i, imp)| IndexedAdjacency {
                    tf_idx: tf_cols[i],
                    target_idx,
                    importance: imp,
                })
                .collect::<Vec<_>>()
            },
        )
        .flatten_iter()
        .collect()
}

fn resolve_tf_cols_with_summary(gene_names: &[String], tf_names: &[String]) -> ResolvedTfCols {
    let gene_ix: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let mut seen_tf_cols: HashSet<usize> = HashSet::new();
    let mut cols = Vec::new();
    let mut present_input_count = 0;
    let mut missing_examples = Vec::new();
    for tf in tf_names {
        if let Some(&idx) = gene_ix.get(tf.as_str()) {
            present_input_count += 1;
            if seen_tf_cols.insert(idx) {
                cols.push(idx);
            }
        } else if missing_examples.len() < 5 {
            missing_examples.push(tf.clone());
        }
    }

    ResolvedTfCols {
        cols,
        summary: TfOverlapSummary {
            present_input_count,
            missing_examples,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf_overlap_summary_counts_input_names_and_caps_missing_examples() {
        let gene_names = vec!["g0".to_string(), "g1".to_string()];
        let tf_names = vec![
            "g0".to_string(),
            "missing0".to_string(),
            "g0".to_string(),
            "missing1".to_string(),
            "missing2".to_string(),
            "missing3".to_string(),
            "missing4".to_string(),
            "missing5".to_string(),
        ];

        let resolved = resolve_tf_cols_with_summary(&gene_names, &tf_names);

        assert_eq!(resolved.cols, vec![0]);
        assert_eq!(resolved.summary.present_input_count, 2);
        assert_eq!(
            resolved.summary.missing_examples,
            vec!["missing0", "missing1", "missing2", "missing3", "missing4"]
        );
    }

    #[test]
    fn infer_indices_deduplicates_tf_names_and_excludes_self_edges() {
        let n_cells = 80;
        let gene_names = vec!["g0".to_string(), "g1".to_string(), "g2".to_string()];
        let mut expression = Vec::with_capacity(n_cells * gene_names.len());
        for i in 0..n_cells {
            let x = i as f32 / n_cells as f32;
            expression.push(x);
            expression.push(2.0 * x + 0.01);
            expression.push(1.0 - x);
        }
        let tf_names = vec!["g0".to_string(), "g0".to_string()];
        let cfg = GrnConfig {
            n_estimators: 30,
            max_features: 1.0,
            target_block_size: 1,
            ..GrnConfig::default()
        };

        let edges = infer_indices(&expression, n_cells, &gene_names, &tf_names, &cfg);
        assert!(!edges.is_empty());
        assert!(edges.iter().all(|edge| edge.tf_idx == 0));
        assert!(edges.iter().all(|edge| edge.target_idx != edge.tf_idx));

        let mut pairs = HashSet::new();
        for edge in edges {
            assert!(
                pairs.insert((edge.tf_idx, edge.target_idx)),
                "duplicate indexed edge emitted"
            );
        }
    }

    #[test]
    fn sparse_csc_infer_matches_dense_infer() {
        let n_cells = 48;
        let gene_names: Vec<String> = (0..6).map(|i| format!("g{i}")).collect();
        let mut expression = vec![0.0_f32; n_cells * gene_names.len()];
        for c in 0..n_cells {
            let x = c as f32 / n_cells as f32;
            expression[c * 6] = if c % 3 == 0 { x } else { 0.0 };
            expression[c * 6 + 1] = 2.0 * x + 0.1;
            expression[c * 6 + 2] = if c % 2 == 0 { 1.0 - x } else { 0.0 };
            expression[c * 6 + 3] = expression[c * 6] + expression[c * 6 + 2];
            expression[c * 6 + 4] = if c % 5 == 0 { 3.0 } else { 0.0 };
            expression[c * 6 + 5] = 0.5 * x;
        }
        let (indptr, indices, data) = dense_to_csc(&expression, n_cells, gene_names.len());
        let tf_names = vec!["g0".to_string(), "g1".to_string(), "g2".to_string()];
        let cfg = GrnConfig {
            n_estimators: 40,
            max_features: 1.0,
            subsample: 1.0,
            target_block_size: 2,
            seed: 11,
            ..GrnConfig::default()
        };

        let mut dense = infer_indices(&expression, n_cells, &gene_names, &tf_names, &cfg);
        let mut sparse = infer_indices_sparse_csc(
            &indptr,
            &indices,
            &data,
            n_cells,
            gene_names.len(),
            &gene_names,
            &tf_names,
            &cfg,
        );
        dense.sort_by_key(|edge| (edge.tf_idx, edge.target_idx, edge.importance.to_bits()));
        sparse.sort_by_key(|edge| (edge.tf_idx, edge.target_idx, edge.importance.to_bits()));

        assert_eq!(dense.len(), sparse.len());
        for (a, b) in dense.iter().zip(&sparse) {
            assert_eq!(a.tf_idx, b.tf_idx);
            assert_eq!(a.target_idx, b.target_idx);
            assert_eq!(a.importance.to_bits(), b.importance.to_bits());
        }
    }

    #[test]
    fn dense_view_infer_matches_row_major_infer() {
        let n_cells = 48;
        let n_genes = 6;
        let gene_names: Vec<String> = (0..n_genes).map(|i| format!("g{i}")).collect();
        let mut expression = vec![0.0_f32; n_cells * n_genes];
        for c in 0..n_cells {
            let x = c as f32 / n_cells as f32;
            expression[c * n_genes] = x;
            expression[c * n_genes + 1] = 1.0 - x;
            expression[c * n_genes + 2] = 2.0 * x + 0.1;
            expression[c * n_genes + 3] = expression[c * n_genes] + expression[c * n_genes + 1];
            expression[c * n_genes + 4] = 0.5 * x;
            expression[c * n_genes + 5] = 0.25;
        }

        let mut transposed = ndarray::Array2::<f32>::zeros((n_genes, n_cells));
        for c in 0..n_cells {
            for g in 0..n_genes {
                transposed[(g, c)] = expression[c * n_genes + g];
            }
        }
        let view = transposed.t();
        assert!(!view.is_standard_layout());

        let tf_names = vec!["g0".to_string(), "g1".to_string(), "g2".to_string()];
        let cfg = GrnConfig {
            n_estimators: 40,
            max_features: 1.0,
            subsample: 1.0,
            target_block_size: 2,
            seed: 13,
            ..GrnConfig::default()
        };

        let mut row_major = infer_indices(&expression, n_cells, &gene_names, &tf_names, &cfg);
        let mut strided = infer_indices_view(view, &gene_names, &tf_names, &cfg);
        row_major.sort_by_key(|edge| (edge.tf_idx, edge.target_idx, edge.importance.to_bits()));
        strided.sort_by_key(|edge| (edge.tf_idx, edge.target_idx, edge.importance.to_bits()));

        assert_eq!(row_major.len(), strided.len());
        for (a, b) in row_major.iter().zip(&strided) {
            assert_eq!(a.tf_idx, b.tf_idx);
            assert_eq!(a.target_idx, b.target_idx);
            assert_eq!(a.importance.to_bits(), b.importance.to_bits());
        }
    }

    #[test]
    #[should_panic(expected = "CSC row index contains a negative or out-of-range cell index")]
    fn sparse_csc_rejects_negative_row_indices() {
        let gene_names: Vec<String> = (0..3).map(|i| format!("g{i}")).collect();
        let tf_names = vec!["g0".to_string()];
        let cfg = GrnConfig {
            n_estimators: 1,
            max_features: 1.0,
            subsample: 1.0,
            target_block_size: 1,
            ..GrnConfig::default()
        };

        let _ = infer_indices_sparse_csc(
            &[0, 1, 1, 1],
            &[-1],
            &[1.0],
            4,
            gene_names.len(),
            &gene_names,
            &tf_names,
            &cfg,
        );
    }

    fn dense_to_csc(
        expression: &[f32],
        n_cells: usize,
        n_genes: usize,
    ) -> (Vec<usize>, Vec<i32>, Vec<f32>) {
        let mut indptr = Vec::with_capacity(n_genes + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for gene in 0..n_genes {
            for cell in 0..n_cells {
                let value = expression[cell * n_genes + gene];
                if value != 0.0 {
                    indices.push(cell as i32);
                    data.push(value);
                }
            }
            indptr.push(indices.len());
        }
        (indptr, indices, data)
    }
}
