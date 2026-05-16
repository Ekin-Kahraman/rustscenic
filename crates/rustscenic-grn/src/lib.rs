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
//! with pyscenic — fine-edge disagreement does not propagate to regulon
//! activity. See `validation/ours/grn_deep_audit_2026-04-18.md`.

pub mod gbm;
pub mod histogram;
pub mod rng;
pub mod tree;

use rayon::prelude::*;

use crate::histogram::BinnedMatrix;

/// Default target genes materialised together for GRN fitting.
///
/// The expression matrix arrives row-major (cells × genes). Extracting one
/// target at a time is a cache-hostile stride of `n_genes` floats for every
/// target. At atlas shapes, that repeats a TLB/cache miss pattern tens of
/// thousands of times. Blocking targets scans each row once per target window,
/// copies contiguous source values, and fits from compact column-major targets.
const DEFAULT_TARGET_BLOCK_SIZE: usize = 64;

/// Keep each target window under this budget by default. Fixed 64-wide blocks
/// are good for small and mid-sized datasets, but become large enough to add
/// memory-bandwidth pressure once cell counts reach atlas scale.
const TARGET_BLOCK_BYTE_BUDGET: usize = 32 * 1024 * 1024;

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
    /// Target block width for materialising response vectors. `0` uses the
    /// adaptive default, capped at `DEFAULT_TARGET_BLOCK_SIZE`.
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
    let n_genes = gene_names.len();
    assert_eq!(
        expression.len(),
        n_cells * n_genes,
        "expression size mismatch"
    );

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

    let mut all_edges = Vec::new();
    let target_block_size = resolve_target_block_size(cfg, n_cells);
    for block_start in (0..n_genes).step_by(target_block_size) {
        let block_end = (block_start + target_block_size).min(n_genes);
        let block_width = block_end - block_start;
        let mut target_block = vec![0.0_f32; block_width * n_cells];

        // Materialise the target window as column-major:
        // target_block[local_target * n_cells + cell].
        //
        // Source reads are contiguous within each row; downstream GBM reads each
        // target's response vector contiguously. This directly attacks the
        // row-major strided target extraction cliff observed on the 91k atlas.
        for c in 0..n_cells {
            let row_base = c * n_genes + block_start;
            let row = &expression[row_base..row_base + block_width];
            for (local_idx, &v) in row.iter().enumerate() {
                target_block[local_idx * n_cells + c] = v;
            }
        }

        let block_edges: Vec<Adjacency> = (0..block_width)
            .into_par_iter()
            .map_init(
                || gbm::GbmScratch::new(n_cells, tf_cols.len(), cfg.n_estimators),
                |gbm_scratch, local_idx| {
                    let target_idx = block_start + local_idx;
                    let target_name = &gene_names[target_idx];
                    let target_expr = &target_block[local_idx * n_cells..(local_idx + 1) * n_cells];

                    // If this target is itself one of the TFs, drop that column from the
                    // feature subset at fit time (not just after). Otherwise the self
                    // column is a perfect predictor → absorbs all split gain → every
                    // other TF's importance collapses to ~0 for that target.
                    let exclude_self = tf_name_to_idx.get(target_name.as_str()).copied();

                    let importances = gbm::fit_and_importances_binned_with_scratch(
                        &binned_all,
                        target_expr,
                        cfg,
                        exclude_self,
                        gbm_scratch,
                    );

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
                },
            )
            .flatten_iter()
            .collect();
        all_edges.extend(block_edges);
    }
    all_edges
}

fn build_tf_matrix(
    expr: &[f32],
    n_cells: usize,
    n_genes: usize,
    tf_cols: &[(String, usize)],
) -> Vec<f32> {
    let n_tfs = tf_cols.len();
    let mut out = vec![0.0_f32; n_cells * n_tfs];
    for c in 0..n_cells {
        let expr_row = &expr[c * n_genes..(c + 1) * n_genes];
        let out_row = &mut out[c * n_tfs..(c + 1) * n_tfs];
        for (t_i, (_, g_i)) in tf_cols.iter().enumerate() {
            out_row[t_i] = expr_row[*g_i];
        }
    }
    out
}

fn resolve_target_block_size(cfg: &GrnConfig, n_cells: usize) -> usize {
    if cfg.target_block_size > 0 {
        return cfg.target_block_size;
    }
    if n_cells == 0 {
        return DEFAULT_TARGET_BLOCK_SIZE;
    }

    let per_target_bytes = n_cells.saturating_mul(std::mem::size_of::<f32>());
    let mut block_size = DEFAULT_TARGET_BLOCK_SIZE;
    while block_size > 1 && per_target_bytes.saturating_mul(block_size) > TARGET_BLOCK_BYTE_BUDGET {
        block_size /= 2;
    }
    block_size.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_target_block_size_preserves_default_for_mid_sized_inputs() {
        let cfg = GrnConfig::default();
        assert_eq!(resolve_target_block_size(&cfg, 100_000), 64);
    }

    #[test]
    fn adaptive_target_block_size_shrinks_for_high_cell_counts() {
        let cfg = GrnConfig::default();
        assert_eq!(resolve_target_block_size(&cfg, 200_000), 32);
        assert_eq!(resolve_target_block_size(&cfg, 500_000), 16);
    }

    #[test]
    fn explicit_target_block_size_overrides_adaptive_default() {
        let cfg = GrnConfig {
            target_block_size: 7,
            ..GrnConfig::default()
        };
        assert_eq!(resolve_target_block_size(&cfg, 500_000), 7);
    }
}
