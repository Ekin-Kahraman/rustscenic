//! Regression decision tree with MSE (squared-error) splitting.
//!
//! Simplified port of sklearn's tree: given (samples × features) dense f32 data
//! and per-sample targets, grow a tree of max_depth using exhaustive best-split
//! search on a random subset of features per split (`max_features`).
//!
//! Each split evaluation is O(n_samples) along sorted values of one feature.
//! For depth-3 trees this is bounded work per target; we do NOT use histogram
//! binning in v0.1 to keep the first cut simple. Optimization lands in v0.1.1.

use rand::RngCore;

use crate::rng::sample_indices;

#[derive(Debug, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
pub enum Node {
    Leaf { value: f32 },
    Split {
        feature: usize,
        threshold: f32,
        gain: f32,           // MSE reduction weighted by n_samples at this node
        left: usize,
        right: usize,
    },
}

/// Fit a regression tree. `x` is row-major `[n_samples][n_features]`,
/// `y` is `[n_samples]` residuals. `sample_idx` and `feature_idx` let the
/// caller restrict to a subsample / feature subset (boosting + max_features).
pub fn fit_tree(
    x: &[f32],
    n_features: usize,
    y: &[f32],
    sample_idx: &[usize],
    max_depth: usize,
    max_features_per_split: usize,
    rng: &mut impl RngCore,
) -> (Tree, Vec<f32>) {
    // Returns (tree, feature_gain_accumulator_over_n_features)
    let mut nodes: Vec<Node> = Vec::new();
    let mut gains = vec![0.0_f32; n_features];
    let root = build_node(x, n_features, y, sample_idx, 0, max_depth, max_features_per_split, &mut nodes, &mut gains, rng);
    debug_assert_eq!(root, 0);
    (Tree { nodes }, gains)
}

fn build_node(
    x: &[f32],
    n_features: usize,
    y: &[f32],
    sample_idx: &[usize],
    depth: usize,
    max_depth: usize,
    max_features_per_split: usize,
    nodes: &mut Vec<Node>,
    gains: &mut [f32],
    rng: &mut impl RngCore,
) -> usize {
    let leaf_value = mean(y, sample_idx);
    if depth >= max_depth || sample_idx.len() < 2 {
        let idx = nodes.len();
        nodes.push(Node::Leaf { value: leaf_value });
        return idx;
    }

    // Pick random feature subset for this split
    let feat_sub = sample_indices(rng, n_features, max_features_per_split);

    // Find best split across subset
    let parent_var = weighted_variance(y, sample_idx);
    let mut best: Option<(usize, f32, f32, Vec<usize>, Vec<usize>)> = None;

    for &f in &feat_sub {
        if let Some(cand) = best_split_on_feature(x, n_features, y, sample_idx, f, parent_var) {
            match &best {
                None => best = Some(cand),
                Some(b) if cand.2 > b.2 => best = Some(cand),
                _ => {}
            }
        }
    }

    let idx = nodes.len();
    nodes.push(Node::Leaf { value: leaf_value }); // placeholder, overwrite below

    match best {
        None => {} // no valid split → stays a leaf
        Some((feature, threshold, gain, left_samples, right_samples)) => {
            // Record importance contribution (weighted by #samples at node)
            gains[feature] += gain;
            let left = build_node(x, n_features, y, &left_samples, depth + 1, max_depth, max_features_per_split, nodes, gains, rng);
            let right = build_node(x, n_features, y, &right_samples, depth + 1, max_depth, max_features_per_split, nodes, gains, rng);
            nodes[idx] = Node::Split { feature, threshold, gain, left, right };
        }
    }
    idx
}

/// Best split on one feature. Returns `(feature, threshold, gain, left_idx, right_idx)`.
/// Gain is MSE reduction × n_samples (sklearn convention for feature_importance).
fn best_split_on_feature(
    x: &[f32],
    n_features: usize,
    y: &[f32],
    sample_idx: &[usize],
    feature: usize,
    parent_var: f32,
) -> Option<(usize, f32, f32, Vec<usize>, Vec<usize>)> {
    if sample_idx.len() < 2 {
        return None;
    }
    // Collect (value, target) for samples at this node along `feature`
    let mut pairs: Vec<(f32, f32)> = sample_idx
        .iter()
        .map(|&s| (x[s * n_features + feature], y[s]))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = pairs.len() as f32;
    let total_sum: f32 = pairs.iter().map(|p| p.1).sum();
    let total_sq: f32 = pairs.iter().map(|p| p.1 * p.1).sum();

    let mut left_sum = 0.0;
    let mut left_sq = 0.0;
    let mut best_gain = 0.0;
    let mut best_threshold = 0.0;
    let mut best_k: Option<usize> = None;

    for k in 0..pairs.len() - 1 {
        left_sum += pairs[k].1;
        left_sq += pairs[k].1 * pairs[k].1;
        let left_n = (k + 1) as f32;
        let right_n = n - left_n;

        // Skip when values don't change (can't split here)
        if pairs[k].0 == pairs[k + 1].0 {
            continue;
        }

        let left_var = left_sq - (left_sum * left_sum) / left_n;
        let right_sum = total_sum - left_sum;
        let right_sq = total_sq - left_sq;
        let right_var = right_sq - (right_sum * right_sum) / right_n;

        // MSE reduction weighted by n (sklearn's impurity reduction × node weight)
        let gain = parent_var - (left_var + right_var);
        if gain > best_gain {
            best_gain = gain;
            best_threshold = (pairs[k].0 + pairs[k + 1].0) * 0.5;
            best_k = Some(k);
        }
    }

    if best_gain <= 0.0 {
        return None;
    }
    let k = best_k.unwrap();

    // Split sample_idx by the chosen threshold
    let mut left_samples = Vec::with_capacity(k + 1);
    let mut right_samples = Vec::with_capacity(sample_idx.len() - k - 1);
    for &s in sample_idx {
        let v = x[s * n_features + feature];
        if v <= best_threshold {
            left_samples.push(s);
        } else {
            right_samples.push(s);
        }
    }

    Some((feature, best_threshold, best_gain, left_samples, right_samples))
}

fn mean(y: &[f32], idx: &[usize]) -> f32 {
    if idx.is_empty() { return 0.0; }
    let s: f32 = idx.iter().map(|&i| y[i]).sum();
    s / idx.len() as f32
}

/// Returns `sum((y-mean)^2)` not divided by n — matches sklearn impurity × weight.
fn weighted_variance(y: &[f32], idx: &[usize]) -> f32 {
    if idx.is_empty() { return 0.0; }
    let n = idx.len() as f32;
    let sum: f32 = idx.iter().map(|&i| y[i]).sum();
    let sq: f32 = idx.iter().map(|&i| y[i] * y[i]).sum();
    sq - (sum * sum) / n
}

/// Evaluate tree prediction for a single sample at row `x_row` of length `n_features`.
pub fn predict(tree: &Tree, x_row: &[f32]) -> f32 {
    let mut cur = 0;
    loop {
        match &tree.nodes[cur] {
            Node::Leaf { value } => return *value,
            Node::Split { feature, threshold, left, right, .. } => {
                cur = if x_row[*feature] <= *threshold { *left } else { *right };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn single_feature_split_finds_optimal_threshold() {
        // y ~ 2 * x[0]; one feature, perfect split at x=0.5
        let x: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();
        let y: Vec<f32> = x.iter().map(|v| if *v < 0.5 { -1.0 } else { 1.0 }).collect();
        let sample_idx: Vec<usize> = (0..20).collect();
        let mut rng = StdRng::seed_from_u64(0);
        let (tree, gains) = fit_tree(&x, 1, &y, &sample_idx, 1, 1, &mut rng);
        assert!(gains[0] > 0.0);
        // Tree with depth=1 should perfectly fit the step
        let err: f32 = (0..20).map(|i| (predict(&tree, &[x[i]]) - y[i]).abs()).sum();
        assert_abs_diff_eq!(err, 0.0, epsilon = 0.01);
    }

    #[test]
    fn feature_importance_concentrates_on_informative_feature() {
        // x: [informative, noise], y depends only on informative
        let n = 200;
        let mut rng = StdRng::seed_from_u64(42);
        let mut x = vec![0.0_f32; n * 2];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a: f32 = (rng.next_u32() as f32) / (u32::MAX as f32);
            let b: f32 = (rng.next_u32() as f32) / (u32::MAX as f32);
            x[i * 2] = a;
            x[i * 2 + 1] = b;
            y[i] = if a > 0.5 { 1.0 } else { -1.0 };
        }
        let sample_idx: Vec<usize> = (0..n).collect();
        let (_tree, gains) = fit_tree(&x, 2, &y, &sample_idx, 3, 2, &mut rng);
        assert!(gains[0] > gains[1] * 3.0,
            "informative feature should dominate: g[0]={} g[1]={}", gains[0], gains[1]);
    }
}
