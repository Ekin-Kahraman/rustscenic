//! Histogram-based regression decision tree.
//!
//! Uses pre-binned features and per-node histograms to score splits in
//! O(bins + |node_samples|) per feature, not O(n log n). All scratch buffers
//! (per-node sample partitions, feature subsets) are held by the caller and
//! reused across trees to avoid allocation churn in the boosting loop.

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

/// Reusable scratch for tree fitting. One per worker thread.
pub struct TreeScratch {
    /// feature subset buffer (reused each split)
    pub feat_sub: Vec<usize>,
    /// column pool for sample_indices (reused)
    pub feat_pool: Vec<usize>,
    /// stack of sample-partitions per recursion level, pre-sized
    pub partitions: Vec<Vec<usize>>,
}

impl TreeScratch {
    pub fn new(n_features: usize) -> Self {
        Self {
            feat_sub: Vec::with_capacity(n_features),
            feat_pool: (0..n_features).collect(),
            partitions: (0..16).map(|_| Vec::with_capacity(4096)).collect(),
        }
    }
}

/// Fit a regression tree. Writes into `tree` and `gains` (caller resets).
#[allow(clippy::too_many_arguments)]
pub fn fit_tree_with_scratch(
    binned: &BinnedMatrix,
    y: &[f32],
    sample_idx: &[usize],
    max_depth: usize,
    max_features_per_split: usize,
    tree: &mut Tree,
    gains: &mut [f32],
    hist_buf: &mut NodeHist,
    scratch: &mut TreeScratch,
    rng: &mut impl RngCore,
) {
    // Copy sample_idx into first partition buffer
    let part0 = scratch.partitions[0].clone();
    let _ = part0;
    // Actually populate partition 0 with sample_idx
    scratch.partitions[0].clear();
    scratch.partitions[0].extend_from_slice(sample_idx);

    build_node_iter(binned, y, 0, max_depth, max_features_per_split, tree, gains, hist_buf, scratch, rng);
}

#[allow(clippy::too_many_arguments)]
fn build_node_iter(
    binned: &BinnedMatrix,
    y: &[f32],
    level: usize,
    max_depth: usize,
    max_features_per_split: usize,
    tree: &mut Tree,
    gains: &mut [f32],
    hist_buf: &mut NodeHist,
    scratch: &mut TreeScratch,
    rng: &mut impl RngCore,
) -> usize {
    // Work out current-level sample partition (scratch.partitions[level])
    let samples_len = scratch.partitions[level].len();
    if samples_len == 0 {
        let idx = tree.nodes.len();
        tree.nodes.push(Node::Leaf { value: 0.0 });
        return idx;
    }
    let leaf_value = mean_at(&scratch.partitions[level], y);

    let idx = tree.nodes.len();
    tree.nodes.push(Node::Leaf { value: leaf_value });

    if level >= max_depth || samples_len < 2 {
        return idx;
    }

    // Pick random feature subset (avoids alloc using scratch.feat_pool/feat_sub)
    choose_feature_subset(rng, &mut scratch.feat_pool, &mut scratch.feat_sub, max_features_per_split);

    let mut best: Option<(usize, u8, f32)> = None;
    for &f in &scratch.feat_sub {
        hist_buf.accumulate(binned, f, y, &scratch.partitions[level]);
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

        // Partition samples by bin ≤ threshold into scratch.partitions[level+1]
        // Right partition goes into scratch.partitions[level+2]
        // (We use two partitions per depth; 16 preallocated cover depth 7)
        while scratch.partitions.len() <= level + 2 {
            scratch.partitions.push(Vec::with_capacity(4096));
        }
        let nf = binned.n_features;

        // Swap out parent partition so we can borrow new ones mutably
        let parent = std::mem::take(&mut scratch.partitions[level]);
        scratch.partitions[level + 1].clear();
        // borrow right partition via split index trick
        let (pre, post) = scratch.partitions.split_at_mut(level + 2);
        let left_part = &mut pre[level + 1];
        let right_part = &mut post[0];
        right_part.clear();
        for &s in &parent {
            if binned.bins[s * nf + feature] <= bin_threshold {
                left_part.push(s);
            } else {
                right_part.push(s);
            }
        }
        // Put parent back (next sibling at same level would overwrite it if we recurse deeper)
        scratch.partitions[level] = parent;

        let left = build_node_iter(
            binned, y, level + 1, max_depth, max_features_per_split,
            tree, gains, hist_buf, scratch, rng,
        );
        // The recursion may have mutated partitions[level+1] and below;
        // for the RIGHT child we still have scratch.partitions[level+2] intact as we prepared it.
        // Move it into the slot at level+1 for the right subtree.
        let right_partition = std::mem::take(&mut scratch.partitions[level + 2]);
        scratch.partitions[level + 1] = right_partition;

        let right = build_node_iter(
            binned, y, level + 1, max_depth, max_features_per_split,
            tree, gains, hist_buf, scratch, rng,
        );

        tree.nodes[idx] = Node::Split { feature, bin_threshold, gain, left, right };
    }

    idx
}

fn choose_feature_subset(
    rng: &mut impl RngCore,
    pool: &mut [usize],
    out: &mut Vec<usize>,
    k: usize,
) {
    let n = pool.len();
    let k = k.min(n).max(1);
    out.clear();
    // Fisher-Yates partial shuffle on the shared pool (order is randomized across trees)
    for i in 0..k {
        let j = i + (rng.next_u64() as usize) % (n - i);
        pool.swap(i, j);
    }
    out.extend_from_slice(&pool[..k]);
}

fn mean_at(sample_idx: &[usize], y: &[f32]) -> f32 {
    if sample_idx.is_empty() {
        return 0.0;
    }
    let mut s = 0.0_f32;
    for &i in sample_idx {
        s += y[i];
    }
    s / sample_idx.len() as f32
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
        fit_tree_with_scratch(&bm, &y, &sample_idx, 1, 1, &mut tree, &mut gains, &mut hist, &mut scratch, &mut rng);
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
        fit_tree_with_scratch(&bm, &y, &sample_idx, 3, 2, &mut tree, &mut gains, &mut hist, &mut scratch, &mut rng);
        assert!(gains[0] > gains[1] * 2.0, "g[0]={} g[1]={}", gains[0], gains[1]);
    }
}
