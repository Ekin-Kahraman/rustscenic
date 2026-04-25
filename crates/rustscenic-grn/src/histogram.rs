//! Histogram binning for fast split finding.
//!
//! Each feature is pre-binned into at most 255 buckets using quantile cut-points.
//! A node-level histogram of (sum_y, sum_y_sq, count) across bins lets us score
//! every candidate threshold in O(bins) rather than O(n) per feature per split.
//!
//! This is the LightGBM trick. Semantics remain sklearn-compatible up to
//! bin-quantization error; at 255 bins the effect on feature-importance ranks
//! is negligible (validated by the real-time audit against arboreto).

use rand::RngCore;

pub const MAX_BINS: usize = 255; // fits in u8 with 0 reserved as "unassigned"

/// Column-major binned features. `bins[f * n_samples + s]` = bin index of
/// sample s on feature f. Column-major because the dominant op (NodeHist
/// accumulate, tree partition) reads one feature's column at a time —
/// row-major would stride by n_features bytes per access and trash the
/// cache. Switching to column-major fits each column in L1/L2 for the
/// typical (3k–100k samples × 30k features) shape.
pub struct BinnedMatrix {
    pub bins: Vec<u8>,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_bins_per_feature: Vec<u16>, // actual bin count per feature (≤ MAX_BINS)
}

impl BinnedMatrix {
    /// Build bins using quantile cut points per feature.
    /// Cells with the same value can end up in the same bin — we dedup cut points.
    pub fn from_dense(x: &[f32], n_samples: usize, n_features: usize) -> Self {
        assert_eq!(x.len(), n_samples * n_features);
        // Reject NaN up front — silent corruption via NaN in partial_cmp + partition_point.
        // Fail fast with clear message rather than producing plausible-looking wrong bins.
        if x.iter().any(|v| v.is_nan()) {
            panic!(
                "input expression matrix contains NaN values — \
                binarization semantics are undefined. \
                Clean upstream: scanpy.pp.normalize_total + sc.pp.log1p can produce \
                NaN from 0-count cells; filter cells with min_genes first."
            );
        }
        let mut bins = vec![0u8; n_samples * n_features];
        let mut n_bins_per_feature = vec![1u16; n_features];

        for f in 0..n_features {
            let mut col: Vec<f32> = (0..n_samples).map(|s| x[s * n_features + f]).collect();
            // Build cut points: sorted unique values, downsampled to MAX_BINS-1 edges.
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            col.dedup();

            let edges = if col.len() <= MAX_BINS {
                // one bin per unique value pair; edges are midpoints between consecutive unique values
                let mut e = Vec::with_capacity(col.len().saturating_sub(1));
                for w in col.windows(2) {
                    e.push((w[0] + w[1]) * 0.5);
                }
                e
            } else {
                // downsample unique values to MAX_BINS - 1 edges (uniform in sorted-value index)
                let n_edges = MAX_BINS - 1;
                let stride = (col.len() - 1) as f32 / n_edges as f32;
                (0..n_edges)
                    .map(|i| {
                        let idx = ((i + 1) as f32 * stride) as usize;
                        let a = col[idx.min(col.len() - 1)];
                        let b = col[(idx + 1).min(col.len() - 1)];
                        (a + b) * 0.5
                    })
                    .collect()
            };

            n_bins_per_feature[f] = (edges.len() + 1) as u16;

            // Assign bins via binary search on edges. Writes go into the
            // column-major slot `bins[f * n_samples + s]` so each feature's
            // bin column lives contiguously.
            let col_base = f * n_samples;
            for s in 0..n_samples {
                let v = x[s * n_features + f];
                let b = edges.partition_point(|&e| e < v);
                bins[col_base + s] = b.min(MAX_BINS - 1) as u8;
            }
        }

        Self {
            bins,
            n_samples,
            n_features,
            n_bins_per_feature,
        }
    }
}

/// Per-bin accumulator for a single (node, feature) pair.
#[derive(Clone)]
pub struct NodeHist {
    pub sum_y: Vec<f32>,
    pub sum_y_sq: Vec<f32>,
    pub count: Vec<u32>,
}

impl NodeHist {
    pub fn zeros(n_bins: usize) -> Self {
        Self {
            sum_y: vec![0.0; n_bins],
            sum_y_sq: vec![0.0; n_bins],
            count: vec![0; n_bins],
        }
    }

    pub fn clear(&mut self) {
        self.sum_y.fill(0.0);
        self.sum_y_sq.fill(0.0);
        self.count.fill(0);
    }

    pub fn accumulate(
        &mut self,
        binned: &BinnedMatrix,
        feature: usize,
        y: &[f32],
        sample_idx: &[usize],
    ) {
        self.clear();
        // Column-major: feature's bin column lives at [base, base+n_samples).
        // The whole column fits in L1/L2 for typical (≤100k samples × 1B
        // each) shapes, so each `bins[base + s]` access hits cache once.
        let base = feature * binned.n_samples;
        let col = &binned.bins[base..base + binned.n_samples];
        for &s in sample_idx {
            let b = col[s] as usize;
            let yv = y[s];
            self.sum_y[b] += yv;
            self.sum_y_sq[b] += yv * yv;
            self.count[b] += 1;
        }
    }

    /// Scan thresholds left→right, return best split (bin_edge, gain, left_count).
    /// Gain = parent_var - (left_var + right_var), variances as in sklearn
    /// (sum of squared deviations, NOT divided by n).
    pub fn best_split(&self) -> Option<(usize, f32, u32)> {
        let total_n: u32 = self.count.iter().sum();
        if total_n < 2 {
            return None;
        }
        let total_s: f32 = self.sum_y.iter().sum();
        let total_sq: f32 = self.sum_y_sq.iter().sum();
        let parent_var = total_sq - (total_s * total_s) / (total_n as f32);

        let mut left_s = 0.0_f32;
        let mut left_sq = 0.0_f32;
        let mut left_n: u32 = 0;
        let mut best_gain = 0.0_f32;
        let mut best_bin = 0usize;
        let mut best_left_n = 0u32;

        // Candidate split: between bin k and bin k+1
        for k in 0..self.count.len() - 1 {
            left_s += self.sum_y[k];
            left_sq += self.sum_y_sq[k];
            left_n += self.count[k];
            if left_n == 0 {
                continue;
            }
            let right_n = total_n - left_n;
            if right_n == 0 {
                break;
            }
            let l_var = left_sq - (left_s * left_s) / (left_n as f32);
            let r_s = total_s - left_s;
            let r_sq = total_sq - left_sq;
            let r_var = r_sq - (r_s * r_s) / (right_n as f32);
            let gain = parent_var - (l_var + r_var);
            if gain > best_gain {
                best_gain = gain;
                best_bin = k;
                best_left_n = left_n;
            }
        }

        if best_gain <= 0.0 {
            return None;
        }
        Some((best_bin, best_gain, best_left_n))
    }
}

/// Random-subset of feature indices.
pub fn sample_feature_subset(rng: &mut impl RngCore, n_features: usize, k: usize) -> Vec<usize> {
    crate::rng::sample_indices(rng, n_features, k.min(n_features).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_preserves_order() {
        let n = 100;
        let nf = 1;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        // Earlier values → smaller bins
        assert!(bm.bins[0] <= bm.bins[50]);
        assert!(bm.bins[50] <= bm.bins[99]);
    }

    #[test]
    fn histogram_finds_perfect_split() {
        // n samples: first half y=-1, second half y=+1; feature distinguishes them
        let n = 100;
        let nf = 1;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect();
        let bm = BinnedMatrix::from_dense(&x, n, nf);
        let mut h = NodeHist::zeros(MAX_BINS);
        h.accumulate(&bm, 0, &y, &(0..n).collect::<Vec<_>>());
        let (bin, gain, left_n) = h.best_split().expect("split exists");
        // Gain should be full parent variance (~n for ±1 perfectly split halves)
        assert!(gain > 50.0, "gain={}", gain);
        assert_eq!(left_n, 50);
        // Bin boundary should be around the transition
        assert!(bin > 0 && bin < MAX_BINS - 1);
    }
}
