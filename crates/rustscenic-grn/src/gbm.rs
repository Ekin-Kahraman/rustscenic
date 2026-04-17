//! Gradient boosting with squared-error loss.
//!
//! Matches arboreto's SGBM config: n_estimators=5000, learning_rate=0.01,
//! max_features=0.1 per split, subsample=0.9 per tree, max_depth=3.
//!
//! EarlyStopMonitor (window=25): halt when the current train MSE is not better
//! than the best seen MSE `window` iterations ago. This is arboreto's check,
//! used as a `monitor` callback into sklearn's fit loop.

use rand::rngs::StdRng;

use crate::rng::{subsample_rows, TargetRng};
use crate::tree::{fit_tree, predict, Tree};
use crate::GrnConfig;

/// Fit a GBM and return the per-TF feature importance vector, with
/// sklearn-style denormalization by n_estimators_fit already applied.
///
/// `x`: (n_samples × n_features) row-major.
/// `y`: (n_samples,) target expression.
pub fn fit_and_importances(
    x: &[f32],
    n_samples: usize,
    y: &[f32],
    cfg: &GrnConfig,
) -> Vec<f32> {
    let n_features = x.len() / n_samples;
    let mut importances = vec![0.0_f32; n_features];

    // Initial prediction: mean of y
    let init: f32 = y.iter().sum::<f32>() / (n_samples as f32);
    let mut predictions = vec![init; n_samples];

    let max_features_per_split = ((cfg.max_features * n_features as f32) as usize).max(1);

    // RNG streams — derived deterministically from cfg.seed + target fingerprint
    let target_fp = hash_y(y);
    let mut rng_state = TargetRng::new(cfg.seed, target_fp);

    // Early-stop monitor state: sliding window of train MSE
    let window = cfg.early_stop_window;
    let mut mse_history: Vec<f32> = Vec::with_capacity(cfg.n_estimators);

    let mut n_fit = 0_usize;

    for i in 0..cfg.n_estimators {
        // Residuals for squared-error loss: y - predictions
        let residuals: Vec<f32> = (0..n_samples).map(|k| y[k] - predictions[k]).collect();

        // Subsample rows (stochastic GBM)
        let mut tree_rng: StdRng = rng_state.for_tree(i);
        let sample_idx = if (cfg.subsample - 1.0).abs() < 1e-6 {
            (0..n_samples).collect()
        } else {
            subsample_rows(&mut tree_rng, n_samples, cfg.subsample)
        };
        if sample_idx.is_empty() {
            continue;
        }

        let (tree, gains) = fit_tree(
            x,
            n_features,
            &residuals,
            &sample_idx,
            cfg.max_depth,
            max_features_per_split,
            &mut tree_rng,
        );

        // Update predictions for ALL samples (not just subsample) with learning_rate × tree
        for k in 0..n_samples {
            let row = &x[k * n_features..(k + 1) * n_features];
            predictions[k] += cfg.learning_rate * predict(&tree, row);
        }

        // Accumulate importance (sklearn per-tree sum of split gains)
        for f in 0..n_features {
            importances[f] += gains[f];
        }

        // Compute train MSE
        let mse: f32 = (0..n_samples)
            .map(|k| (y[k] - predictions[k]).powi(2))
            .sum::<f32>() / (n_samples as f32);
        mse_history.push(mse);

        n_fit = i + 1;

        // EarlyStopMonitor check (arboreto): halt if current MSE >= MSE at i-window
        if mse_history.len() > window {
            let past = mse_history[mse_history.len() - 1 - window];
            if mse >= past {
                break;
            }
        }
    }

    // Normalize importances → sum to 1 → then multiply by n_fit (arboreto core.py:168)
    // sklearn's feature_importances_ is normalized; arboreto denormalizes by n_estimators_.
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
    // Cheap deterministic fingerprint to make RNG seed per-target.
    // Enough variance across targets; we don't need cryptographic quality.
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

    #[test]
    fn recovers_linear_signal() {
        // y = 3*x[0] + noise — GBM should assign dominant importance to feature 0
        let n = 300;
        let mut x = vec![0.0_f32; n * 4];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a = (i as f32) / n as f32;
            x[i * 4] = a;
            x[i * 4 + 1] = (i as f32) * 0.7 % 1.0;
            x[i * 4 + 2] = ((i * 13) as f32) * 0.3 % 1.0;
            x[i * 4 + 3] = ((i * 7) as f32) * 0.5 % 1.0;
            y[i] = 3.0 * a + 0.1 * x[i * 4 + 1];
        }
        let cfg = GrnConfig { n_estimators: 200, learning_rate: 0.1, max_features: 1.0,
            subsample: 1.0, max_depth: 3, early_stop_window: 25, seed: 42 };
        let imp = fit_and_importances(&x, n, &y, &cfg);
        assert!(imp[0] > imp[1] * 3.0, "feature 0 should dominate: {:?}", imp);
        assert!(imp[0] > imp[2] * 3.0);
        assert!(imp[0] > imp[3] * 3.0);
    }
}
