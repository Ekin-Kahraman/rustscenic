//! Histogram-based regression decision tree.
//!
//! Uses pre-binned features and per-node histograms to score splits in
//! O(bins + |node_samples|) per feature, not O(n log n). Left/right partitions
//! are local Vecs per recursive call — small allocations (a few KB each) but
//! eliminates the cross-sibling aliasing bug in the prior scratch-stack design.
//!
//! `exclude_feature` lets callers drop a single feature from consideration,
//! used to exclude self-TF when the target gene is itself in the TF set.

use rand::RngCore;

use crate::histogram::{BinnedMatrix, NodeHist};

#[derive(Debug, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
pub enum Node {
    Leaf { value: f32 },
    Split {
        feature: usize,
        bin_threshold: u8,
        gain: f32,
        left: usize,
        right: usize,
    },
}

/// Reusable scratch for tree fitting (held by the boosting loop, one per target).
pub struct TreeScratch {
    pub feat_sub: Vec<usize>,
    pub feat_pool: Vec<usize>,
}

impl TreeScratch {
    pub fn new(n_features: usize) -> Self {
        Self {
            feat_sub: Vec::with_capacity(n_features),
            feat_pool: (0..n_features).collect(),
        }
    }
}

/// Fit a regression tree. Writes into `tree` and `gains` (caller resets).
///
/// The scratch's `feat_pool` is refilled to `0..n_features` on every call to
/// avoid state leakage from a previous tree's exclude_feature swap_remove.
#[allow(clippy::too_many_arguments)]
pub fn fit_tree_with_scratch(
    binned: &BinnedMatrix,
    y: &[f32],
    sample_idx: &[usize],
    max_depth: usize,
    max_features_per_split: usize,
    exclude_feature: Option<usize>,
    tree: &mut Tree,
    gains: &mut [f32],
    hist_buf: &mut NodeHist,
    scratch: &mut TreeScratch,
    rng: &mut impl RngCore,
) {
    // Refill the feature pool to 0..n_features so exclude-feature logic starts fresh
    scratch.feat_pool.clear();
    scratch.feat_pool.extend(0..binned.n_features);
    // Clone incoming sample_idx into an owned Vec so recursion can freely reuse it.
    let root_samples: Vec<usize> = sample_idx.to_vec();
    build_node_rec(
        binned,
        y,
        &root_samples,
        0,
        max_depth,
        max_features_per_split,
        exclude_feature,
        tree,
        gains,
        hist_buf,
        scratch,
        rng,
    );
}

#[allow(clippy::too_many_arguments)]
fn build_node_rec(
    binned: &BinnedMatrix,
    y: &[f32],
    samples: &[usize],
    depth: usize,
    max_depth: usize,
    max_features_per_split: usize,
    exclude_feature: Option<usize>,
    tree: &mut Tree,
    gains: &mut [f32],
    hist_buf: &mut NodeHist,
    scratch: &mut TreeScratch,
    rng: &mut impl RngCore,
) -> usize {
    let leaf_value = mean_at(samples, y);
    let idx = tree.nodes.len();
    tree.nodes.push(Node::Leaf { value: leaf_value });

    if depth >= max_depth || samples.len() < 2 {
        return idx;
    }

    choose_feature_subset(
        rng,
        &mut scratch.feat_pool,
        &mut scratch.feat_sub,
        max_features_per_split,
        exclude_feature,
    );

    let mut best: Option<(usize, u8, f32)> = None;
    for &f in &scratch.feat_sub {
        hist_buf.accumulate(binned, f, y, samples);
        if let Some((bin, gain, _)) = hist_buf.best_split() {
            match best {
                None => best = Some((f, bin as u8, gain)),
                Some((_, _, g)) if gain > g => best = Some((f, bin as u8, gain)),
                _ => {}
            }
        }
    }

    if let Some((feature, bin_threshold, gain)) = best {
        gains[feature] += gain;
        let nf = binned.n_features;

        // Partition into owned left/right Vecs. Small allocations (~few KB each)
        // but avoids any aliasing between sibling subtrees.
        let mut left_samples = Vec::with_capacity(samples.len() / 2);
        let mut right_samples = Vec::with_capacity(samples.len() / 2);
        for &s in samples {
            if binned.bins[s * nf + feature] <= bin_threshold {
                left_samples.push(s);
            } else {
                right_samples.push(s);
            }
        }

        let left = build_node_rec(
            binned, y, &left_samples, depth + 1, max_depth, max_features_per_split,
            exclude_feature, tree, gains, hist_buf, scratch, rng,
        );
        let right = build_node_rec(
            binned, y, &right_samples, depth + 1, max_depth, max_features_per_split,
            exclude_feature, tree, gains, hist_buf, scratch, rng,
        );

        tree.nodes[idx] = Node::Split { feature, bin_threshold, gain, left, right };
    }
    idx
}

fn choose_feature_subset(
    rng: &mut impl RngCore,
    pool: &mut Vec<usize>,
    out: &mut Vec<usize>,
    k: usize,
    exclude_feature: Option<usize>,
) {
    // If this is the first call (or feat_pool has become out of date), ensure it's
    // initialized to 0..n_features minus the excluded index. `pool.len()` is the
    // effective number of features the caller has staged.
    if let Some(excl) = exclude_feature {
        if let Some(pos) = pool.iter().position(|&x| x == excl) {
            pool.swap_remove(pos);
        }
    }
    let n = pool.len();
    if n == 0 {
        out.clear();
        return;
    }
    let k = k.min(n).max(1);
    out.clear();
    for i in 0..k {
        let j = i + (rng.next_u64() as usize) % (n - i);
        pool.swap(i, j);
    }
    out.extend_from_slice(&pool[..k]);
}

fn mean_at(samples: &[usize], y: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut s = 0.0_f32;
    for &i in samples {
        s += y[i];
    }
    s / samples.len() as f32
}

pub fn predict_binned(tree: &Tree, binned: &BinnedMatrix, sample: usize) -> f32 {
    let nf = binned.n_features;
    let mut cur = 0;
    loop {
        match &tree.nodes[cur] {
            Node::Leaf { value } => return *value,
            Node::Split { feature, bin_threshold, left, right, .. } => {
                let b = binned.bins[sample * nf + *feature];
                cur = if b <= *bin_threshold { *left } else { *right };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::BinnedMatrix;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn single_feature_split_finds_optimal_threshold() {
        let x: Vec<f32> = (0..40).map(|i| i as f32 / 20.0).collect();
        let y: Vec<f32> = x.iter().map(|v| if *v < 1.0 { -1.0 } else { 1.0 }).collect();
        let bm = BinnedMatrix::from_dense(&x, 40, 1);
        let sample_idx: Vec<usize> = (0..40).collect();
        let mut rng = StdRng::seed_from_u64(0);
        let mut tree = Tree { nodes: Vec::new() };
        let mut gains = vec![0.0_f32; 1];
        let mut hist = NodeHist::zeros(crate::histogram::MAX_BINS);
        let mut scratch = TreeScratch::new(1);
        fit_tree_with_scratch(&bm, &y, &sample_idx, 1, 1, None, &mut tree, &mut gains, &mut hist, &mut scratch, &mut rng);
        assert!(gains[0] > 0.0);
        let err: f32 = (0..40).map(|i| (predict_binned(&tree, &bm, i) - y[i]).abs()).sum();
        assert!(err < 1.0, "err = {}", err);
    }

    #[test]
    fn informative_feature_dominates() {
        let n = 500;
        let nf = 2;
        let mut rng = StdRng::seed_from_u64(42);
        use rand::RngCore;
        let mut x = vec![0.0_f32; n * nf];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a = (rng.next_u32() as f32) / (u32::MAX as f32);
            let b = (rng.next_u32() as f32) / (u32::MAX as f32);
            x[i * nf] = a;
            x[i * nf + 1] = b;
            y[i] = if a > 0.5 { 1.0 } else { -1.0 };
        }
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let sample_idx: Vec<usize> = (0..n).collect();
        let mut tree = Tree { nodes: Vec::new() };
        let mut gains = vec![0.0_f32; nf];
        let mut hist = NodeHist::zeros(crate::histogram::MAX_BINS);
        let mut scratch = TreeScratch::new(nf);
        fit_tree_with_scratch(&bm, &y, &sample_idx, 3, 2, None, &mut tree, &mut gains, &mut hist, &mut scratch, &mut rng);
        assert!(gains[0] > gains[1] * 2.0, "g[0]={} g[1]={}", gains[0], gains[1]);
    }

    #[test]
    fn right_subtree_uses_correct_samples() {
        // Previously the partition-stack scheme corrupted samples between
        // sibling subtrees. This test catches that regression: construct a
        // tree where the right subtree has a strong secondary signal that
        // would be invisible if its sample set were corrupted.
        let n = 600;
        let nf = 2;
        let mut x = vec![0.0_f32; n * nf];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a: f32 = if i < n / 2 { 0.0 } else { 1.0 };      // primary: feature 0 splits halves
            let b: f32 = (i % 10) as f32 / 10.0;                  // secondary
            x[i * nf] = a;
            x[i * nf + 1] = b;
            // Signal: left half y = noise; right half y strongly depends on b
            y[i] = if i < n / 2 {
                ((i % 3) as f32) * 0.01
            } else {
                3.0 * b
            };
        }
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let mut rng = StdRng::seed_from_u64(1);
        let mut tree = Tree { nodes: Vec::new() };
        let mut gains = vec![0.0_f32; nf];
        let mut hist = NodeHist::zeros(crate::histogram::MAX_BINS);
        let mut scratch = TreeScratch::new(nf);
        fit_tree_with_scratch(&bm, &y, &(0..n).collect::<Vec<_>>(), 3, 2, None, &mut tree, &mut gains, &mut hist, &mut scratch, &mut rng);
        // Secondary feature should accumulate SIGNIFICANT gain via the right subtree
        // (nearly all of it comes from the right half of the data).
        assert!(gains[1] > 10.0, "right-subtree secondary signal should show: g[1]={} g[0]={}", gains[1], gains[0]);
    }

    #[test]
    fn exclude_feature_is_respected() {
        let n = 200;
        let nf = 3;
        let mut x = vec![0.0_f32; n * nf];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let a = (i as f32) / n as f32;
            x[i * nf] = a; // perfect predictor
            x[i * nf + 1] = ((i * 7) as f32) * 0.3 % 1.0;
            x[i * nf + 2] = ((i * 13) as f32) * 0.5 % 1.0;
            y[i] = 5.0 * a;
        }
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let mut rng = StdRng::seed_from_u64(7);
        let mut tree = Tree { nodes: Vec::new() };
        let mut gains = vec![0.0_f32; nf];
        let mut hist = NodeHist::zeros(crate::histogram::MAX_BINS);
        let mut scratch = TreeScratch::new(nf);
        // Exclude the perfect-predictor feature 0; gains must be on others.
        fit_tree_with_scratch(&bm, &y, &(0..n).collect::<Vec<_>>(), 3, 3, Some(0), &mut tree, &mut gains, &mut hist, &mut scratch, &mut rng);
        assert_eq!(gains[0], 0.0, "excluded feature must receive zero gain");
        assert!(gains[1] + gains[2] > 0.0);
    }
}
