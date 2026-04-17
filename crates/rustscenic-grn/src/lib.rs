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
//! Validated against pyscenic by edge-rank Jaccard ≥0.80, Spearman-on-union
//! ≥0.85 on PBMC-3k.

pub mod gbm;
pub mod histogram;
pub mod importance;
pub mod rng;
pub mod tree;

use rayon::prelude::*;

use crate::histogram::BinnedMatrix;

#[derive(Debug, Clone)]
pub struct Adjacency {
    pub tf: String,
    pub target: String,
    pub importance: f32,
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
        }
    }
}

/// Infer a GRN from a dense (n_cells × n_genes) f32 expression matrix.
///
/// Pre-bins the TF matrix once, then parallelizes across target genes.
pub fn infer(
    expression: &[f32],
    n_cells: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> Vec<Adjacency> {
    let n_genes = gene_names.len();
    assert_eq!(expression.len(), n_cells * n_genes, "expression size mismatch");

    let gene_ix: std::collections::HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let tf_cols: Vec<(String, usize)> = tf_names
        .iter()
        .filter_map(|t| gene_ix.get(t.as_str()).map(|&i| (t.clone(), i)))
        .collect();

    if tf_cols.is_empty() {
        return Vec::new();
    }

    // Build full TF matrix and bin ONCE. Per-target "drop self-TF" is handled by
    // zeroing the self column in residual view rather than rebuilding the matrix.
    let tf_matrix = build_tf_matrix(expression, n_cells, n_genes, &tf_cols);
    let binned_all = BinnedMatrix::from_dense(&tf_matrix, n_cells, tf_cols.len());

    let tf_names_vec: Vec<String> = tf_cols.iter().map(|(n, _)| n.clone()).collect();
    let tf_name_to_idx: std::collections::HashMap<&str, usize> = tf_names_vec
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    (0..n_genes)
        .into_par_iter()
        .flat_map_iter(|target_idx| {
            let target_name = &gene_names[target_idx];
            let target_expr: Vec<f32> = (0..n_cells)
                .map(|c| expression[c * n_genes + target_idx])
                .collect();

            // If this target is itself one of the TFs, drop that column from the
            // feature subset at fit time (not just after). Otherwise the self
            // column is a perfect predictor → absorbs all split gain → every
            // other TF's importance collapses to ~0 for that target.
            let exclude_self = tf_name_to_idx.get(target_name.as_str()).copied();

            let importances =
                gbm::fit_and_importances_binned(&binned_all, &target_expr, cfg, exclude_self);

            importances
                .into_iter()
                .enumerate()
                .filter(|(_, imp)| *imp > 0.0)
                .map(|(i, imp)| Adjacency {
                    tf: tf_names_vec[i].clone(),
                    target: target_name.clone(),
                    importance: imp,
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
        .collect()
}

fn build_tf_matrix(
    expr: &[f32],
    n_cells: usize,
    n_genes: usize,
    tf_cols: &[(String, usize)],
) -> Vec<f32> {
    let n_tfs = tf_cols.len();
    let mut out = vec![0.0_f32; n_cells * n_tfs];
    for (t_i, (_, g_i)) in tf_cols.iter().enumerate() {
        for c in 0..n_cells {
            out[c * n_tfs + t_i] = expr[c * n_genes + g_i];
        }
    }
    out
}
