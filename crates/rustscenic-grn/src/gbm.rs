//! Gradient boosting with squared-error loss using histogram-binned trees.
//!
//! sklearn-compatible config via arboreto SGBM_KWARGS:
//! n_estimators=5000, learning_rate=0.01, max_features=0.1,
//! subsample=0.9, max_depth=3.
//!
//! Early stopping mirrors arboreto's `EarlyStopMonitor` (window=25):
//! if in-bag MSE[i] >= MSE[i-window], stop. Disabled with window=0.
//!
//! `exclude_feature` is passed through to the tree builder so targets that are
//! themselves TFs don't include their own expression column as a predictor.

use rand::rngs::StdRng;

use crate::histogram::{BinnedMatrix, NodeHist, MAX_BINS};
use crate::rng::{subsample_rows_into, TargetRng};
use crate::tree::{fit_tree_with_scratch, predict_binned, Tree, TreeScratch};
use crate::GrnConfig;

/// Fit a GBM on pre-binned features, return per-TF denormalized importances.
pub fn fit_and_importances_binned(
    binned: &BinnedMatrix,
    y: &[f32],
    cfg: &GrnConfig,
    exclude_feature: Option<usize>,
) -> Vec<f32> {
    let n_samples = binned.n_samples;
    let n_features = binned.n_features;
    let mut importances = vec![0.0_f32; n_features];

    let init: f32 = y.iter().sum::<f32>() / (n_samples as f32);
    let mut predictions = vec![init; n_samples];
    let mut residuals = vec![0.0_f32; n_samples];

    let max_features_per_split = ((cfg.max_features * n_features as f32) as usize).max(1);

    let target_fp = hash_y(y);
    let mut rng_state = TargetRng::new(cfg.seed, target_fp);

    let window = cfg.early_stop_window;
    let mut mse_history: Vec<f32> = Vec::with_capacity(cfg.n_estimators);

    // Reusable buffers — all allocated once per target, re-used across all trees.
    let mut sample_idx: Vec<usize> = Vec::with_capacity(n_samples);
    let mut hist_buf = NodeHist::zeros(MAX_BINS);
    let mut tree = Tree { nodes: Vec::with_capacity(64) };
    let mut gains_buf = vec![0.0_f32; n_features];
    let mut scratch = TreeScratch::new(n_features);

    let mut n_fit = 0usize;

    for i in 0..cfg.n_estimators {
        for k in 0..n_samples {
            residuals[k] = y[k] - predictions[k];
        }

        let mut tree_rng: StdRng = rng_state.for_tree(i);
        sample_idx.clear();
        if (cfg.subsample - 1.0).abs() < 1e-6 {
            sample_idx.extend(0..n_samples);
        } else {
            subsample_rows_into(&mut tree_rng, n_samples, cfg.subsample, &mut sample_idx);
        }
        if sample_idx.is_empty() {
            continue;
        }

        tree.nodes.clear();
        gains_buf.fill(0.0);
        fit_tree_with_scratch(
            binned,
            &residuals,
            &sample_idx,
            cfg.max_depth,
            max_features_per_split,
            exclude_feature,
            &mut tree,
            &mut gains_buf,
            &mut hist_buf,
            &mut scratch,
            &mut tree_rng,
        );

        for (k, p) in predictions.iter_mut().enumerate().take(n_samples) {
            *p += cfg.learning_rate * predict_binned(&tree, binned, k);
        }

        for f in 0..n_features {
            importances[f] += gains_buf[f];
        }

        let mut mse_inbag = 0.0_f32;
        for &k in &sample_idx {
            let d = y[k] - predictions[k];
            mse_inbag += d * d;
        }
        mse_inbag /= sample_idx.len() as f32;
        mse_history.push(mse_inbag);
        n_fit = i + 1;

        if window > 0 && mse_history.len() > window {
            let past = mse_history[mse_history.len() - 1 - window];
            if mse_inbag >= past {
                break;
            }
        }
    }

    // Denormalize per arboreto/core.py:168 — × trained_regressor.estimators_.shape[0]
    let total: f32 = importances.iter().sum();
    if total > 0.0 {
        for v in &mut importances {
            *v /= total;
        }
    }
    for v in &mut importances {
        *v *= n_fit as f32;
    }
    importances
}

fn hash_y(y: &[f32]) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in y.iter().take(64) {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h as usize) & 0x00FF_FFFF_FFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::BinnedMatrix;

    #[test]
    fn recovers_linear_signal() {
        let n = 500;
        let nf = 4;
        let mut x = vec![0.0_f32; n * nf];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a = (i as f32) / n as f32;
            x[i * nf] = a;
            x[i * nf + 1] = (i as f32) * 0.7 % 1.0;
            x[i * nf + 2] = ((i * 13) as f32) * 0.3 % 1.0;
            x[i * nf + 3] = ((i * 7) as f32) * 0.5 % 1.0;
            y[i] = 3.0 * a + 0.1 * x[i * nf + 1];
        }
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let cfg = GrnConfig {
            n_estimators: 500,
            learning_rate: 0.1,
            max_features: 1.0,
            subsample: 0.9,
            max_depth: 3,
            early_stop_window: 25,
            seed: 42,
        };
        let imp = fit_and_importances_binned(&bm, &y, &cfg, None);
        assert!(imp[0] > imp[1] * 3.0, "imp[0]={} imp[1]={} imp[2]={} imp[3]={}", imp[0], imp[1], imp[2], imp[3]);
    }

    #[test]
    fn excludes_self_feature() {
        let n = 300;
        let nf = 3;
        let mut x = vec![0.0_f32; n * nf];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a = (i as f32) / n as f32;
            x[i * nf] = a;               // perfect predictor
            x[i * nf + 1] = ((i * 3) as f32) * 0.37 % 1.0;
            x[i * nf + 2] = ((i * 17) as f32) * 0.13 % 1.0;
            y[i] = 5.0 * a;
        }
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let cfg = GrnConfig {
            n_estimators: 200, learning_rate: 0.1, max_features: 1.0,
            subsample: 1.0, max_depth: 3, early_stop_window: 0, seed: 7,
        };
        let imp = fit_and_importances_binned(&bm, &y, &cfg, Some(0));
        assert_eq!(imp[0], 0.0, "excluded feature must have zero importance");
        // Since feature 0 was excluded, some gain must go elsewhere
        assert!(imp[1] + imp[2] > 0.0);
    }
}
