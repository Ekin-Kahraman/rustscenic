//! Histogram binning for fast split finding.
//!
//! Each feature is pre-binned into at most 255 buckets using quantile cut-points.
//! A node-level histogram of (sum_y, sum_y_sq, count) across bins lets us score
//! every candidate threshold in O(bins) rather than O(n) per feature per split.
//!
//! This is the LightGBM trick. Semantics remain sklearn-compatible up to
//! bin-quantization error; at 255 bins the effect on feature-importance ranks
//! is negligible (validated by the real-time audit against arboreto).

use ndarray::ArrayView2;
use rand::RngCore;
use rayon::prelude::*;

pub const MAX_BINS: usize = 255; // fits in u8 with 0 reserved as "unassigned"

/// Column-major binned features. `bins[f * n_samples + s]` = bin index of
/// sample s on feature f. Column-major because the dominant op (NodeHist
/// accumulate, tree partition) reads one feature's column at a time -
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
    /// Cells with the same value can end up in the same bin - we dedup cut points.
    pub fn from_dense(x: &[f32], n_samples: usize, n_features: usize) -> Self {
        Self::from_dense_columns_impl(x, n_samples, n_features, n_features, None)
    }

    /// Build bins for selected columns from a row-major dense matrix.
    ///
    /// This avoids materialising a separate dense TF matrix before binning in
    /// GRN inference. The output feature order follows `feature_cols`.
    pub fn from_dense_columns(
        x: &[f32],
        n_samples: usize,
        n_source_features: usize,
        feature_cols: &[usize],
    ) -> Self {
        Self::from_dense_columns_impl(
            x,
            n_samples,
            n_source_features,
            feature_cols.len(),
            Some(feature_cols),
        )
    }

    /// Build bins for selected columns from any dense matrix view.
    ///
    /// Row-major callers should use `from_dense_columns` for cache-friendly
    /// source reads. This view path is for valid strided arrays from PyO3,
    /// avoiding a full matrix copy before binning.
    pub fn from_dense_columns_view(x: ArrayView2<'_, f32>, feature_cols: &[usize]) -> Self {
        let n_samples = x.shape()[0];
        let n_source_features = x.shape()[1];
        assert!(feature_cols.iter().all(|&col| col < n_source_features));
        reject_nan_view(x);

        let n_output_features = feature_cols.len();
        let mut bins = vec![0u8; n_samples * n_output_features];
        let mut n_bins_per_feature = vec![1u16; n_output_features];

        bins.par_chunks_mut(n_samples)
            .zip(n_bins_per_feature.par_iter_mut())
            .enumerate()
            .for_each(|(f, (bin_col, n_bins_slot))| {
                let source_f = feature_cols[f];
                let values: Vec<f32> = (0..n_samples).map(|s| x[(s, source_f)]).collect();
                assign_bins_from_values(&values, bin_col, n_bins_slot);
            });

        Self {
            bins,
            n_samples,
            n_features: n_output_features,
            n_bins_per_feature,
        }
    }

    /// Build bins for selected columns from a sparse CSC matrix.
    ///
    /// The output feature order follows `feature_cols`. Implicit zeros are
    /// included in quantile/bin assignment, matching dense materialisation.
    pub fn from_csc_columns(
        indptr: &[usize],
        indices: &[i32],
        data: &[f32],
        n_samples: usize,
        n_source_features: usize,
        feature_cols: &[usize],
    ) -> Self {
        assert_eq!(indptr.len(), n_source_features + 1);
        assert_eq!(indices.len(), data.len());
        assert!(feature_cols.iter().all(|&col| col < n_source_features));
        reject_nan(data);

        let n_output_features = feature_cols.len();
        let mut bins = vec![0u8; n_samples * n_output_features];
        let mut n_bins_per_feature = vec![1u16; n_output_features];

        bins.par_chunks_mut(n_samples)
            .zip(n_bins_per_feature.par_iter_mut())
            .enumerate()
            .for_each(|(f, (bin_col, n_bins_slot))| {
                let source_f = feature_cols[f];
                let mut values = vec![0.0_f32; n_samples];
                for p in indptr[source_f]..indptr[source_f + 1] {
                    let row =
                        usize::try_from(indices[p]).expect("CSC indices must be non-negative");
                    assert!(row < n_samples);
                    values[row] += data[p];
                }
                assign_bins_from_values(&values, bin_col, n_bins_slot);
            });

        Self {
            bins,
            n_samples,
            n_features: n_output_features,
            n_bins_per_feature,
        }
    }

    fn from_dense_columns_impl(
        x: &[f32],
        n_samples: usize,
        n_source_features: usize,
        n_output_features: usize,
        feature_cols: Option<&[usize]>,
    ) -> Self {
        assert_eq!(x.len(), n_samples * n_source_features);
        if let Some(cols) = feature_cols {
            assert_eq!(cols.len(), n_output_features);
            assert!(cols.iter().all(|&col| col < n_source_features));
        }
        reject_nan(x);
        let mut bins = vec![0u8; n_samples * n_output_features];
        let mut n_bins_per_feature = vec![1u16; n_output_features];

        bins.par_chunks_mut(n_samples)
            .zip(n_bins_per_feature.par_iter_mut())
            .enumerate()
            .for_each(|(f, (bin_col, n_bins_slot))| {
                let source_f = feature_cols.map_or(f, |cols| cols[f]);
                let values: Vec<f32> = (0..n_samples)
                    .map(|s| x[s * n_source_features + source_f])
                    .collect();
                assign_bins_from_values(&values, bin_col, n_bins_slot);
            });

        Self {
            bins,
            n_samples,
            n_features: n_output_features,
            n_bins_per_feature,
        }
    }
}

fn reject_nan(values: &[f32]) {
    if values.iter().any(|v| v.is_nan()) {
        panic!(
            "input expression matrix contains NaN values - \
            binarization semantics are undefined. \
            Clean upstream: scanpy.pp.normalize_total + sc.pp.log1p can produce \
            NaN from 0-count cells; filter cells with min_genes first."
        );
    }
}

fn reject_nan_view(values: ArrayView2<'_, f32>) {
    if values.iter().any(|v| v.is_nan()) {
        panic!(
            "input expression matrix contains NaN values - \
            binarization semantics are undefined. \
            Clean upstream: scanpy.pp.normalize_total + sc.pp.log1p can produce \
            NaN from 0-count cells; filter cells with min_genes first."
        );
    }
}

fn assign_bins_from_values(values: &[f32], bin_col: &mut [u8], n_bins_slot: &mut u16) {
    // Build cut points: sorted unique values, downsampled to MAX_BINS-1 edges.
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup();

    let edges = if sorted.len() <= MAX_BINS {
        // one bin per unique value pair; edges are midpoints between consecutive unique values
        let mut e = Vec::with_capacity(sorted.len().saturating_sub(1));
        for w in sorted.windows(2) {
            e.push((w[0] + w[1]) * 0.5);
        }
        e
    } else {
        // downsample unique values to MAX_BINS - 1 edges (uniform in sorted-value index)
        let n_edges = MAX_BINS - 1;
        let stride = (sorted.len() - 1) as f32 / n_edges as f32;
        (0..n_edges)
            .map(|i| {
                let idx = ((i + 1) as f32 * stride) as usize;
                let a = sorted[idx.min(sorted.len() - 1)];
                let b = sorted[(idx + 1).min(sorted.len() - 1)];
                (a + b) * 0.5
            })
            .collect()
    };

    *n_bins_slot = (edges.len() + 1) as u16;

    // Assign bins via binary search on edges. Writes go into the
    // column-major slot `bins[f * n_samples + s]` so each feature's
    // bin column lives contiguously.
    for (s, &v) in values.iter().enumerate() {
        let b = edges.partition_point(|&e| e < v);
        bin_col[s] = b.min(MAX_BINS - 1) as u8;
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

    pub fn clear(&mut self, n_bins: usize) {
        let n_bins = n_bins.min(self.count.len());
        self.sum_y[..n_bins].fill(0.0);
        self.sum_y_sq[..n_bins].fill(0.0);
        self.count[..n_bins].fill(0);
    }

    pub fn accumulate(
        &mut self,
        binned: &BinnedMatrix,
        feature: usize,
        y: &[f32],
        sample_idx: &[usize],
    ) {
        self.clear(binned.n_bins_per_feature[feature] as usize);
        // Column-major: feature's bin column lives at [base, base+n_samples).
        // The whole column fits in L1/L2 for typical (≤100k samples × 1B
        // each) shapes, so each `bins[base + s]` access hits cache once.
        let base = feature * binned.n_samples;
        let col = &binned.bins[base..base + binned.n_samples];
        for &s in sample_idx {
            // sample_idx comes from 0..n_samples and child partitions retain
            // those indices, so these checks are redundant in release builds.
            debug_assert!(s < col.len());
            let b = unsafe { *col.get_unchecked(s) as usize };
            let yv = unsafe { *y.get_unchecked(s) };
            debug_assert!(b < self.count.len());
            unsafe {
                *self.sum_y.get_unchecked_mut(b) += yv;
                *self.sum_y_sq.get_unchecked_mut(b) += yv * yv;
                *self.count.get_unchecked_mut(b) += 1;
            }
        }
    }

    /// Scan thresholds left→right, return best split (bin_edge, gain, left_count).
    /// Gain = parent_var - (left_var + right_var), variances as in sklearn
    /// (sum of squared deviations, NOT divided by n).
    pub fn best_split(&self, n_bins: usize) -> Option<(usize, f32, u32)> {
        let n_bins = n_bins.min(self.count.len());
        if n_bins < 2 {
            return None;
        }
        let total_n: u32 = self.count[..n_bins].iter().sum();
        if total_n < 2 {
            return None;
        }
        let total_s: f32 = self.sum_y[..n_bins].iter().sum();
        let total_sq: f32 = self.sum_y_sq[..n_bins].iter().sum();
        let parent_var = total_sq - (total_s * total_s) / (total_n as f32);

        let mut left_s = 0.0_f32;
        let mut left_sq = 0.0_f32;
        let mut left_n: u32 = 0;
        let mut best_gain = 0.0_f32;
        let mut best_bin = 0usize;
        let mut best_left_n = 0u32;

        // Candidate split: between bin k and bin k+1
        for k in 0..n_bins - 1 {
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
    fn selected_column_binning_matches_materialised_matrix() {
        let n = 64;
        let nf = 5;
        let selected = vec![3, 1, 4];
        let mut x = vec![0.0_f32; n * nf];
        for s in 0..n {
            for f in 0..nf {
                x[s * nf + f] = ((s * 17 + f * 11) % 23) as f32;
            }
        }
        let mut materialised = vec![0.0_f32; n * selected.len()];
        for s in 0..n {
            for (out_f, &source_f) in selected.iter().enumerate() {
                materialised[s * selected.len() + out_f] = x[s * nf + source_f];
            }
        }

        let direct = BinnedMatrix::from_dense_columns(&x, n, nf, &selected);
        let copied = BinnedMatrix::from_dense(&materialised, n, selected.len());

        assert_eq!(direct.n_features, copied.n_features);
        assert_eq!(direct.n_bins_per_feature, copied.n_bins_per_feature);
        assert_eq!(direct.bins, copied.bins);
    }

    #[test]
    fn csc_column_binning_matches_dense_with_implicit_zeros() {
        let n = 6;
        let nf = 4;
        let selected = vec![2, 0, 3];
        let dense = vec![
            0.0, 1.0, 0.0, 5.0, //
            2.0, 0.0, 3.0, 0.0, //
            0.0, 4.0, 0.0, 1.0, //
            1.0, 0.0, 7.0, 0.0, //
            0.0, 2.0, 0.0, 6.0, //
            3.0, 0.0, 1.0, 0.0,
        ];
        let indptr = vec![0, 3, 6, 9, 12];
        let indices = vec![1, 3, 5, 0, 2, 4, 1, 3, 5, 0, 2, 4];
        let data = vec![2.0, 1.0, 3.0, 1.0, 4.0, 2.0, 3.0, 7.0, 1.0, 5.0, 1.0, 6.0];

        let direct = BinnedMatrix::from_csc_columns(&indptr, &indices, &data, n, nf, &selected);
        let copied = BinnedMatrix::from_dense_columns(&dense, n, nf, &selected);

        assert_eq!(direct.n_features, copied.n_features);
        assert_eq!(direct.n_bins_per_feature, copied.n_bins_per_feature);
        assert_eq!(direct.bins, copied.bins);
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
        let (bin, gain, left_n) = h
            .best_split(bm.n_bins_per_feature[0] as usize)
            .expect("split exists");
        // Gain should be full parent variance (~n for ±1 perfectly split halves)
        assert!(gain > 50.0, "gain={}", gain);
        assert_eq!(left_n, 50);
        // Bin boundary should be around the transition
        assert!(bin > 0 && bin < MAX_BINS - 1);
    }
}
