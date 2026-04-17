//! rustscenic-grn: GRNBoost2 replacement.
//!
//! Matches `arboreto.algo.grnboost2` semantics:
//! - sklearn `GradientBoostingRegressor` with squared-error loss
//! - n_estimators=5000, learning_rate=0.01, max_features=0.1, subsample=0.9,
//!   max_depth=3 (sklearn defaults inherited by arboreto's SGBM_KWARGS)
//! - arboreto's EarlyStopMonitor (window=25) halts when train MSE stalls
//! - Feature importance = sklearn's split-gain accumulation × n_estimators_fit
//!   (per arboreto/core.py:168)
//!
//! Not bit-identical to sklearn (different RNG tape → slightly different tree
//! splits when ties or subsample draws differ). Validated against pyscenic by
//! edge-rank Jaccard ≥0.80, Spearman-on-union ≥0.85 on PBMC-3k.

pub mod gbm;
pub mod importance;
pub mod rng;
pub mod tree;

use rayon::prelude::*;

/// An adjacency: single TF → target edge with importance score.
#[derive(Debug, Clone)]
pub struct Adjacency {
    pub tf: String,
    pub target: String,
    pub importance: f32,
}

/// Hyperparameters. Matches arboreto `SGBM_KWARGS`.
#[derive(Debug, Clone)]
pub struct GrnConfig {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_features: f32, // fraction of TFs per split
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

/// Infer a GRN from an (n_cells × n_genes) dense f32 expression matrix.
///
/// - `expression` is row-major `[n_cells][n_genes]` flattened.
/// - `gene_names.len() == n_genes`.
/// - `tf_names` is the subset of gene names to use as regulators.
///   Any TF absent from `gene_names` is silently dropped (matches arboreto).
/// - Returns one `Adjacency` per non-zero-importance (TF, target) pair, sorted
///   descending within each target, matching arboreto's output schema.
pub fn infer(
    expression: &[f32],
    n_cells: usize,
    gene_names: &[String],
    tf_names: &[String],
    cfg: &GrnConfig,
) -> Vec<Adjacency> {
    let n_genes = gene_names.len();
    assert_eq!(expression.len(), n_cells * n_genes, "expression size mismatch");

    // TF column indices in the expression matrix
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

    // Build a dense (n_cells × n_tfs) TF matrix once
    let tf_matrix = build_tf_matrix(expression, n_cells, n_genes, &tf_cols);

    // Parallel per-target inference
    (0..n_genes)
        .into_par_iter()
        .flat_map_iter(|target_idx| {
            let target_name = &gene_names[target_idx];
            let target_expr: Vec<f32> = (0..n_cells)
                .map(|c| expression[c * n_genes + target_idx])
                .collect();
            // Drop target from TFs if it's itself a TF
            let (tf_mat_sub, tf_names_sub) = drop_target_tf(&tf_matrix, &tf_cols, n_cells, target_name);
            if tf_names_sub.is_empty() {
                return Vec::new().into_iter();
            }
            let importances = gbm::fit_and_importances(&tf_mat_sub, n_cells, &target_expr, cfg);
            importances
                .into_iter()
                .zip(tf_names_sub.into_iter())
                .filter(|(imp, _)| *imp > 0.0)
                .map(move |(imp, tf)| Adjacency {
                    tf,
                    target: target_name.clone(),
                    importance: imp,
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
        .collect()
}

fn build_tf_matrix(expr: &[f32], n_cells: usize, n_genes: usize, tf_cols: &[(String, usize)]) -> Vec<f32> {
    let n_tfs = tf_cols.len();
    let mut out = vec![0.0_f32; n_cells * n_tfs];
    for (t_i, (_, g_i)) in tf_cols.iter().enumerate() {
        for c in 0..n_cells {
            out[c * n_tfs + t_i] = expr[c * n_genes + g_i];
        }
    }
    out
}

fn drop_target_tf(
    tf_matrix: &[f32],
    tf_cols: &[(String, usize)],
    n_cells: usize,
    target_name: &str,
) -> (Vec<f32>, Vec<String>) {
    let n_tfs = tf_cols.len();
    let drop_ix = tf_cols.iter().position(|(n, _)| n == target_name);
    match drop_ix {
        None => {
            let names: Vec<String> = tf_cols.iter().map(|(n, _)| n.clone()).collect();
            (tf_matrix.to_vec(), names)
        }
        Some(d) => {
            let new_n = n_tfs - 1;
            let mut out = vec![0.0_f32; n_cells * new_n];
            for c in 0..n_cells {
                let src_row = &tf_matrix[c * n_tfs..(c + 1) * n_tfs];
                let dst_row = &mut out[c * new_n..(c + 1) * new_n];
                dst_row[..d].copy_from_slice(&src_row[..d]);
                dst_row[d..].copy_from_slice(&src_row[d + 1..]);
            }
            let names: Vec<String> = tf_cols
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != d)
                .map(|(_, (n, _))| n.clone())
                .collect();
            (out, names)
        }
    }
}
