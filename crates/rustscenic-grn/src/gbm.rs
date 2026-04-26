//! Gradient boosting with squared-error loss using histogram-binned trees.
//!
//! sklearn-compatible config via arboreto SGBM_KWARGS:
//! n_estimators=5000, learning_rate=0.01, max_features=0.1,
//! subsample=0.9, max_depth=3.
//!
//! Early stopping mirrors arboreto's `EarlyStopMonitor` (window=25):
//! if in-bag `MSE[i] >= MSE[i-window]`, stop. Disabled with window=0.
//!
//! `exclude_feature` is passed through to the tree builder so targets that are
//! themselves TFs don't include their own expression column as a predictor.

use rand::rngs::StdRng;

use crate::histogram::{BinnedMatrix, NodeHist, MAX_BINS};
use crate::rng::{subsample_rows_into, TargetRng};
use crate::tree::{fit_tree_with_scratch, predict_binned, Tree, TreeScratch};
use crate::GrnConfig;

/// Worker-local buffers reused across target genes.
///
/// GRN parallelises over targets. At atlas scale, allocating predictions,
/// residuals, sampled-row indices, histograms, and partition buffers once per
/// target becomes allocator/page-fault traffic. Keep these buffers per Rayon
/// worker and resize/clear them for each target instead.
pub struct GbmScratch {
    importances: Vec<f32>,
    predictions: Vec<f32>,
    residuals: Vec<f32>,
    sample_idx: Vec<usize>,
    hist_buf: NodeHist,
    tree: Tree,
    gains_buf: Vec<f32>,
    tree_scratch: TreeScratch,
    mse_history: Vec<f32>,
}

impl GbmScratch {
    pub fn new(n_samples: usize, n_features: usize, n_estimators: usize) -> Self {
        Self {
            importances: vec![0.0; n_features],
            predictions: vec![0.0; n_samples],
            residuals: vec![0.0; n_samples],
            sample_idx: Vec::with_capacity(n_samples),
            hist_buf: NodeHist::zeros(MAX_BINS),
            tree: Tree {
                nodes: Vec::with_capacity(64),
            },
            gains_buf: vec![0.0; n_features],
            tree_scratch: TreeScratch::new(n_features),
            mse_history: Vec::with_capacity(n_estimators),
        }
    }

    fn prepare(&mut self, n_samples: usize, n_features: usize, n_estimators: usize) {
        if self.importances.len() != n_features {
            self.importances.resize(n_features, 0.0);
            self.gains_buf.resize(n_features, 0.0);
            self.tree_scratch = TreeScratch::new(n_features);
        }
        self.importances.fill(0.0);
        self.predictions.resize(n_samples, 0.0);
        self.residuals.resize(n_samples, 0.0);
        self.sample_idx.clear();
        if self.sample_idx.capacity() < n_samples {
            self.sample_idx
                .reserve(n_samples - self.sample_idx.capacity());
        }
        self.tree.nodes.clear();
        self.gains_buf.fill(0.0);
        self.mse_history.clear();
        if self.mse_history.capacity() < n_estimators {
            self.mse_history
                .reserve(n_estimators - self.mse_history.capacity());
        }
    }
}

/// Fit a GBM on pre-binned features, return per-TF denormalized importances.
pub fn fit_and_importances_binned(
    binned: &BinnedMatrix,
    y: &[f32],
    cfg: &GrnConfig,
    exclude_feature: Option<usize>,
) -> Vec<f32> {
    let mut scratch = GbmScratch::new(binned.n_samples, binned.n_features, cfg.n_estimators);
    fit_and_importances_binned_with_scratch(binned, y, cfg, exclude_feature, &mut scratch)
}

/// Same as [`fit_and_importances_binned`], but reuses caller-owned buffers.
pub fn fit_and_importances_binned_with_scratch(
    binned: &BinnedMatrix,
    y: &[f32],
    cfg: &GrnConfig,
    exclude_feature: Option<usize>,
    scratch: &mut GbmScratch,
) -> Vec<f32> {
    let n_samples = binned.n_samples;
    let n_features = binned.n_features;
    scratch.prepare(n_samples, n_features, cfg.n_estimators);

    let init: f32 = y.iter().sum::<f32>() / (n_samples as f32);
    scratch.predictions.fill(init);

    let max_features_per_split = ((cfg.max_features * n_features as f32) as usize).max(1);

    let target_fp = hash_y(y);
    let mut rng_state = TargetRng::new(cfg.seed, target_fp);

    let window = cfg.early_stop_window;

    let mut n_fit = 0usize;

    for i in 0..cfg.n_estimators {
        for (k, yv) in y.iter().enumerate().take(n_samples) {
            scratch.residuals[k] = *yv - scratch.predictions[k];
        }

        let mut tree_rng: StdRng = rng_state.for_tree(i);
        scratch.sample_idx.clear();
        if (cfg.subsample - 1.0).abs() < 1e-6 {
            scratch.sample_idx.extend(0..n_samples);
        } else {
            subsample_rows_into(
                &mut tree_rng,
                n_samples,
                cfg.subsample,
                &mut scratch.sample_idx,
            );
        }
        if scratch.sample_idx.is_empty() {
            continue;
        }

        scratch.tree.nodes.clear();
        scratch.gains_buf.fill(0.0);
        fit_tree_with_scratch(
            binned,
            &scratch.residuals,
            &scratch.sample_idx,
            cfg.max_depth,
            max_features_per_split,
            exclude_feature,
            &mut scratch.tree,
            &mut scratch.gains_buf,
            &mut scratch.hist_buf,
            &mut scratch.tree_scratch,
            &mut tree_rng,
        );

        for (k, p) in scratch.predictions.iter_mut().enumerate().take(n_samples) {
            *p += cfg.learning_rate * predict_binned(&scratch.tree, binned, k);
        }

        for f in 0..n_features {
            scratch.importances[f] += scratch.gains_buf[f];
        }

        let mut mse_inbag = 0.0_f32;
        for &k in &scratch.sample_idx {
            let d = y[k] - scratch.predictions[k];
            mse_inbag += d * d;
        }
        mse_inbag /= scratch.sample_idx.len() as f32;
        scratch.mse_history.push(mse_inbag);
        n_fit = i + 1;

        if window > 0 && scratch.mse_history.len() > window {
            let past = scratch.mse_history[scratch.mse_history.len() - 1 - window];
            if mse_inbag >= past {
                break;
            }
        }
    }

    // Denormalize per arboreto/core.py:168 — × trained_regressor.estimators_.shape[0]
    let total: f32 = scratch.importances.iter().sum();
    if total > 0.0 {
        for v in &mut scratch.importances {
            *v /= total;
        }
    }
    for v in &mut scratch.importances {
        *v *= n_fit as f32;
    }
    scratch.importances.clone()
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
        assert!(
            imp[0] > imp[1] * 3.0,
            "imp[0]={} imp[1]={} imp[2]={} imp[3]={}",
            imp[0],
            imp[1],
            imp[2],
            imp[3]
        );
    }

    #[test]
    fn excludes_self_feature() {
        let n = 300;
        let nf = 3;
        let mut x = vec![0.0_f32; n * nf];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a = (i as f32) / n as f32;
            x[i * nf] = a; // perfect predictor
            x[i * nf + 1] = ((i * 3) as f32) * 0.37 % 1.0;
            x[i * nf + 2] = ((i * 17) as f32) * 0.13 % 1.0;
            y[i] = 5.0 * a;
        }
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let cfg = GrnConfig {
            n_estimators: 200,
            learning_rate: 0.1,
            max_features: 1.0,
            subsample: 1.0,
            max_depth: 3,
            early_stop_window: 0,
            seed: 7,
        };
        let imp = fit_and_importances_binned(&bm, &y, &cfg, Some(0));
        assert_eq!(imp[0], 0.0, "excluded feature must have zero importance");
        // Since feature 0 was excluded, some gain must go elsewhere
        assert!(imp[1] + imp[2] > 0.0);
    }
}
