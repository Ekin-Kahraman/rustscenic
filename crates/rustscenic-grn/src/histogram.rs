//! Histogram binning for fast split finding.
//!
//! Each feature is pre-binned once. With the default budget every distinct
//! value gets its own bin (up to 65,535 per feature), so the candidate
//! thresholds are exactly scikit-learn's: midpoints between consecutive
//! distinct values. A node-level histogram of (sum_y, count) across bins then
//! scores every candidate threshold in O(bins) rather than O(n log n) per
//! feature per split.
//!
//! A smaller `max_bins` merges distinct values into bins spaced uniformly over
//! the sorted distinct values. RustScenic 0.6 used 255; at tens of thousands of
//! cells that quantisation makes target rankings diverge from arboreto's beyond
//! arboreto's own seed-to-seed variation, which is why the default is exact.

use ndarray::ArrayView2;
use rand::RngCore;
use rayon::prelude::*;

/// Largest bin budget per feature (bin indices are stored as u16). This is the
/// default budget: exact for any feature with at most 65,535 distinct values.
pub const MAX_BINS: usize = 65535;

/// Smallest bin budget. When values are merged the last cut point is the
/// largest value (RustScenic 0.6 behaviour, kept so 255 bins reproduce 0.6),
/// so the top bin stays empty: a budget of 2 would leave one usable bin and no
/// split.
pub const MIN_BINS: usize = 3;

fn check_max_bins(max_bins: usize) {
    assert!(
        (MIN_BINS..=MAX_BINS).contains(&max_bins),
        "max_bins must be between {MIN_BINS} and {MAX_BINS}, got {max_bins}"
    );
}

/// Column-major binned features. `bins[f * n_samples + s]` = bin index of
/// sample s on feature f. Column-major because the dominant op (NodeHist
/// accumulate, tree partition) reads one feature's column at a time -
/// row-major would stride by n_features bytes per access and trash the
/// cache. Switching to column-major fits each column in L1/L2 for the
/// typical (3k–100k samples × 30k features) shape.
pub struct BinnedMatrix {
    pub bins: Vec<u16>,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_bins_per_feature: Vec<u16>, // actual bin count per feature (≤ MAX_BINS)
}

impl BinnedMatrix {
    /// Build bins using quantile cut points per feature.
    /// Cells with the same value can end up in the same bin - we dedup cut points.
    pub fn from_dense(x: &[f32], n_samples: usize, n_features: usize, max_bins: usize) -> Self {
        Self::from_dense_columns_impl(x, n_samples, n_features, n_features, None, max_bins)
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
        max_bins: usize,
    ) -> Self {
        Self::from_dense_columns_impl(
            x,
            n_samples,
            n_source_features,
            feature_cols.len(),
            Some(feature_cols),
            max_bins,
        )
    }

    /// Build bins for selected columns from any dense matrix view.
    ///
    /// Row-major callers should use `from_dense_columns` for cache-friendly
    /// source reads. This view path is for valid strided arrays from PyO3,
    /// avoiding a full matrix copy before binning.
    pub fn from_dense_columns_view(
        x: ArrayView2<'_, f32>,
        feature_cols: &[usize],
        max_bins: usize,
    ) -> Self {
        check_max_bins(max_bins);
        let n_samples = x.shape()[0];
        let n_source_features = x.shape()[1];
        assert!(feature_cols.iter().all(|&col| col < n_source_features));
        reject_nan_view(x);

        let n_output_features = feature_cols.len();
        let mut bins = vec![0u16; n_samples * n_output_features];
        let mut n_bins_per_feature = vec![1u16; n_output_features];

        bins.par_chunks_mut(n_samples)
            .zip(n_bins_per_feature.par_iter_mut())
            .enumerate()
            .for_each(|(f, (bin_col, n_bins_slot))| {
                let source_f = feature_cols[f];
                let values: Vec<f32> = (0..n_samples).map(|s| x[(s, source_f)]).collect();
                assign_bins_from_values(&values, bin_col, n_bins_slot, max_bins);
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
        max_bins: usize,
    ) -> Self {
        check_max_bins(max_bins);
        assert_eq!(indptr.len(), n_source_features + 1);
        assert_eq!(indices.len(), data.len());
        assert!(feature_cols.iter().all(|&col| col < n_source_features));
        reject_nan(data);

        let n_output_features = feature_cols.len();
        let mut bins = vec![0u16; n_samples * n_output_features];
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
                assign_bins_from_values(&values, bin_col, n_bins_slot, max_bins);
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
        max_bins: usize,
    ) -> Self {
        check_max_bins(max_bins);
        assert_eq!(x.len(), n_samples * n_source_features);
        if let Some(cols) = feature_cols {
            assert_eq!(cols.len(), n_output_features);
            assert!(cols.iter().all(|&col| col < n_source_features));
        }
        reject_nan(x);
        let mut bins = vec![0u16; n_samples * n_output_features];
        let mut n_bins_per_feature = vec![1u16; n_output_features];

        bins.par_chunks_mut(n_samples)
            .zip(n_bins_per_feature.par_iter_mut())
            .enumerate()
            .for_each(|(f, (bin_col, n_bins_slot))| {
                let source_f = feature_cols.map_or(f, |cols| cols[f]);
                let values: Vec<f32> = (0..n_samples)
                    .map(|s| x[s * n_source_features + source_f])
                    .collect();
                assign_bins_from_values(&values, bin_col, n_bins_slot, max_bins);
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

fn assign_bins_from_values(
    values: &[f32],
    bin_col: &mut [u16],
    n_bins_slot: &mut u16,
    max_bins: usize,
) {
    // Build cut points: sorted unique values, downsampled to max_bins-1 edges.
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup();

    let edges = if sorted.len() <= max_bins {
        // one bin per unique value pair; edges are midpoints between consecutive unique values
        let mut e = Vec::with_capacity(sorted.len().saturating_sub(1));
        for w in sorted.windows(2) {
            e.push((w[0] + w[1]) * 0.5);
        }
        e
    } else {
        // downsample unique values to max_bins - 1 edges (uniform in sorted-value index)
        let n_edges = max_bins - 1;
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
        bin_col[s] = b.min(max_bins - 1) as u16;
    }
}

/// Per-bin accumulator for a single (node, feature) pair.
#[derive(Clone)]
pub struct NodeHist {
    pub sum_y: Vec<f32>,
    pub count: Vec<u32>,
    occupied: Vec<u32>,
    left_s_arr: Vec<f32>,
    left_n_arr: Vec<u32>,
    gains: Vec<f32>,
    /// Leading bins that may still hold counts. `best_split` zeroes the bins
    /// it reads, so the next `accumulate` only clears after an unconsumed one.
    dirty: usize,
}

impl NodeHist {
    pub fn zeros(n_bins: usize) -> Self {
        Self {
            sum_y: vec![0.0; n_bins],
            count: vec![0; n_bins],
            occupied: vec![0; n_bins],
            left_s_arr: vec![0.0; n_bins],
            left_n_arr: vec![0; n_bins],
            gains: vec![0.0; n_bins],
            dirty: 0,
        }
    }

    pub fn clear(&mut self, n_bins: usize) {
        let n_bins = n_bins.min(self.count.len());
        self.sum_y[..n_bins].fill(0.0);
        self.count[..n_bins].fill(0);
    }

    pub fn accumulate(
        &mut self,
        binned: &BinnedMatrix,
        feature: usize,
        y: &[f32],
        sample_idx: &[u32],
    ) {
        if self.dirty > 0 {
            self.clear(self.dirty);
        }
        let n_bins = binned.n_bins_per_feature[feature] as usize;
        // The writes below are unchecked, so the buffer must hold every bin.
        let capacity = self.count.len().min(self.sum_y.len());
        assert!(
            n_bins <= capacity,
            "histogram buffer holds {capacity} bins, feature has {n_bins}"
        );
        self.dirty = n_bins;
        // Column-major: feature's bin column lives at [base, base+n_samples).
        // The whole column fits in L1/L2 for typical (≤100k samples × 2 B
        // each) shapes, so each `bins[base + s]` access hits cache once.
        let base = feature * binned.n_samples;
        let col = &binned.bins[base..base + binned.n_samples];
        for &sample in sample_idx {
            let s = sample as usize;
            // sample_idx comes from 0..n_samples and child partitions retain
            // those indices, so these checks are redundant in release builds.
            debug_assert!(s < col.len());
            let b = unsafe { *col.get_unchecked(s) as usize };
            let yv = unsafe { *y.get_unchecked(s) };
            debug_assert!(b < self.count.len());
            unsafe {
                *self.sum_y.get_unchecked_mut(b) += yv;
                *self.count.get_unchecked_mut(b) += 1;
            }
        }
    }

    /// Scan thresholds left→right, return best split (bin_edge, gain, left_count).
    /// Gain = parent_var - (left_var + right_var), variances as in sklearn
    /// (sum of squared deviations, NOT divided by n).
    pub fn best_split(&mut self, n_bins: usize) -> Option<(usize, f32, u32)> {
        let n_bins = n_bins.min(self.count.len());
        if n_bins < 2 {
            return None;
        }

        // Pass 1: list the occupied bins in ascending order. With exact binning
        // a feature has up to one bin per distinct value, so below the root
        // most bins are empty, and the later passes touch only occupied bins.
        // Skipping empty bins changes no result: an empty bin adds zero to the
        // prefix sums, and its candidate split repeats the gain of the previous
        // occupied bin, which the lowest-bin tie-break below already prefers.
        // The compaction is branch-free: always write, advance only past an
        // occupied bin.
        let count = &self.count[..n_bins];
        let occupied = &mut self.occupied[..n_bins];
        let mut n_occ = 0usize;
        for (k, &c) in count.iter().enumerate() {
            // n_occ <= k < n_bins == occupied.len().
            unsafe { *occupied.get_unchecked_mut(n_occ) = k as u32 };
            n_occ += usize::from(c != 0);
        }
        let occupied = &occupied[..n_occ];

        // Pass 2: inclusive prefix sums of (sum_y, count) over occupied bins,
        // in bin order, so every partial sum and the totals equal a running sum
        // over all bins exactly. Each bin is zeroed once read, which leaves the
        // histogram clear for the next accumulate without a full-width reset.
        let left_s_arr = &mut self.left_s_arr[..n_occ];
        let left_n_arr = &mut self.left_n_arr[..n_occ];
        let sum_y = &mut self.sum_y[..n_bins];
        let count = &mut self.count[..n_bins];
        let mut acc_s = 0.0_f32;
        let mut acc_n = 0u32;
        for (j, &k) in occupied.iter().enumerate() {
            let k = k as usize;
            acc_s += sum_y[k];
            acc_n += count[k];
            sum_y[k] = 0.0;
            count[k] = 0;
            left_s_arr[j] = acc_s;
            left_n_arr[j] = acc_n;
        }
        if n_bins >= self.dirty {
            self.dirty = 0;
        }
        let (total_s, total_n) = (acc_s, acc_n);
        if total_n < 2 {
            return None;
        }
        // Gain = parent_var - (left_var + right_var). Expanding the
        // sum-of-squared-deviations form, every sum_y_sq term cancels
        // algebraically, leaving this equivalent -- and strictly more stable, no
        // catastrophic cancellation -- reduced form. sum_y_sq is therefore not
        // tracked in the histogram at all (removes one mul + one RMW per sample
        // from the hottest inner loop).
        let total_term = (total_s * total_s) / (total_n as f32);

        // Pass 3: branch-free per-candidate gain. The division is unconditional
        // so it autovectorises (IEEE divide is bit-identical scalar-vs-SIMD, no
        // reassociation). The last occupied bin leaves an empty right child;
        // its NaN or infinite gain is masked to -inf.
        let gains = &mut self.gains[..n_occ];
        for j in 0..n_occ {
            let left_n = left_n_arr[j];
            let right_n = total_n - left_n;
            let left_s = left_s_arr[j];
            let r_s = total_s - left_s;
            let g =
                (left_s * left_s) / (left_n as f32) + (r_s * r_s) / (right_n as f32) - total_term;
            gains[j] = if left_n != 0 && right_n != 0 {
                g
            } else {
                f32::NEG_INFINITY
            };
        }

        // Pass 4: argmax with the original tie-break - strict `>` so the lowest
        // bin wins on ties, best seeded at 0.0 so only positive gains count.
        let mut best_gain = 0.0_f32;
        let mut best_bin = 0usize;
        let mut best_left_n = 0u32;
        for j in 0..n_occ {
            if gains[j] > best_gain {
                best_gain = gains[j];
                best_bin = occupied[j] as usize;
                best_left_n = left_n_arr[j];
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
        let bm = BinnedMatrix::from_dense(&x, n, nf, MAX_BINS);
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

        let direct = BinnedMatrix::from_dense_columns(&x, n, nf, &selected, MAX_BINS);
        let copied = BinnedMatrix::from_dense(&materialised, n, selected.len(), MAX_BINS);

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

        let direct =
            BinnedMatrix::from_csc_columns(&indptr, &indices, &data, n, nf, &selected, MAX_BINS);
        let copied = BinnedMatrix::from_dense_columns(&dense, n, nf, &selected, MAX_BINS);

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
        let bm = BinnedMatrix::from_dense(&x, n, nf, MAX_BINS);
        let mut h = NodeHist::zeros(MAX_BINS);
        h.accumulate(&bm, 0, &y, &(0..n as u32).collect::<Vec<_>>());
        let (bin, gain, left_n) = h
            .best_split(bm.n_bins_per_feature[0] as usize)
            .expect("split exists");
        // Gain should be full parent variance (~n for ±1 perfectly split halves)
        assert!(gain > 50.0, "gain={}", gain);
        assert_eq!(left_n, 50);
        // Bin boundary should be around the transition
        assert!(bin > 0 && bin < MAX_BINS - 1);
    }

    /// The split search before v0.7: every bin, strict `>` so the lowest bin wins.
    fn full_scan_split(sum_y: &[f32], count: &[u32]) -> Option<(usize, f32, u32)> {
        let n_bins = count.len();
        if n_bins < 2 {
            return None;
        }
        let total_n: u32 = count.iter().sum();
        if total_n < 2 {
            return None;
        }
        let total_s: f32 = sum_y.iter().sum();
        let total_term = (total_s * total_s) / (total_n as f32);
        let (mut acc_s, mut acc_n) = (0.0_f32, 0u32);
        let mut best = (0usize, 0.0_f32, 0u32);
        for k in 0..n_bins - 1 {
            acc_s += sum_y[k];
            acc_n += count[k];
            let right_n = total_n - acc_n;
            let r_s = total_s - acc_s;
            let g = (acc_s * acc_s) / (acc_n as f32) + (r_s * r_s) / (right_n as f32) - total_term;
            if acc_n != 0 && right_n != 0 && g > best.1 {
                best = (k, g, acc_n);
            }
        }
        (best.1 > 0.0).then_some(best)
    }

    #[test]
    fn occupied_bin_split_matches_full_scan_across_calls() {
        // Features with all-distinct, zero-inflated, constant and few-level
        // values; targets with ties. One NodeHist is reused across features and
        // nodes, and some histograms are left unconsumed.
        let n = 400;
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut x = vec![0.0_f32; n * 4];
        for s in 0..n {
            x[s * 4] = s as f32;
            x[s * 4 + 1] = if next() % 4 == 0 {
                (next() % 50) as f32
            } else {
                0.0
            };
            x[s * 4 + 2] = 1.0;
            x[s * 4 + 3] = (next() % 7) as f32;
        }
        // Integer targets give exact sums and many tied gains; fractional ones
        // make the f32 summation order matter.
        let y_int: Vec<f32> = (0..n).map(|_| (next() % 5) as f32 - 2.0).collect();
        let y_frac: Vec<f32> = (0..n)
            .map(|_| (next() % 1000) as f32 * 0.0137 - 6.1)
            .collect();
        let bm = BinnedMatrix::from_dense(&x, n, 4, MAX_BINS);
        let mut hist = NodeHist::zeros(MAX_BINS);
        for round in 0..400 {
            let size = 1 + next() % n;
            let samples: Vec<u32> = (0..n as u32).filter(|_| next() % n < size).collect();
            let f = next() % 4;
            let n_bins = bm.n_bins_per_feature[f] as usize;
            let y = if round % 2 == 0 { &y_int } else { &y_frac };
            hist.accumulate(&bm, f, y, &samples);
            if round % 7 == 0 {
                continue;
            }
            let mut sum_y = vec![0.0_f32; n_bins];
            let mut count = vec![0u32; n_bins];
            for &s in &samples {
                let b = bm.bins[f * n + s as usize] as usize;
                sum_y[b] += y[s as usize];
                count[b] += 1;
            }
            let bits = |r: Option<(usize, f32, u32)>| r.map(|(b, g, l)| (b, g.to_bits(), l));
            assert_eq!(
                bits(hist.best_split(n_bins)),
                bits(full_scan_split(&sum_y, &count)),
                "round {round}, feature {f}"
            );
        }
    }

    #[test]
    fn default_budget_gives_one_bin_per_distinct_value() {
        // 300 distinct values, each twice, shuffled: exact binning is the value's rank.
        let n = 600;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7919) % 300) as f32 * 0.5).collect();
        let bm = BinnedMatrix::from_dense(&x, n, 1, MAX_BINS);
        assert_eq!(bm.n_bins_per_feature[0], 300);
        for (s, &v) in x.iter().enumerate() {
            assert_eq!(bm.bins[s] as f32, v * 2.0);
        }
    }

    #[test]
    fn smaller_budget_merges_distinct_values_in_order() {
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let bm = BinnedMatrix::from_dense(&x, n, 1, 255);
        assert_eq!(bm.n_bins_per_feature[0], 255);
        assert_eq!(bm.bins[0], 0);
        // RustScenic 0.6 behaviour, kept so max_bins=255 reproduces 0.6 networks:
        // the last cut point equals the largest value, so the top bin stays empty.
        assert_eq!(bm.bins[n - 1], 253);
        assert!(bm.bins.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    #[should_panic(expected = "max_bins must be between 3 and 65535")]
    fn budget_below_three_is_rejected() {
        BinnedMatrix::from_dense(&[0.0, 1.0], 2, 1, 2);
    }

    #[test]
    #[should_panic(expected = "max_bins must be between 3 and 65535")]
    fn budget_above_storage_is_rejected() {
        BinnedMatrix::from_dense(&[0.0, 1.0], 2, 1, MAX_BINS + 1);
    }
}
