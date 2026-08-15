//! TF-target Pearson correlations used to dichotomise SCENIC regulons.

use ndarray::ArrayView2;
use rayon::prelude::*;

/// Pearson correlations for selected `(TF column, target column)` pairs in a
/// dense, possibly strided, cells-by-genes matrix.
///
/// Work is grouped by TF so each TF expression vector and each target value is
/// read once per TF group rather than once per edge accumulator.  The returned
/// vector preserves `pairs` order regardless of Rayon scheduling.
pub fn correlations_dense_view(
    expression: ArrayView2<'_, f32>,
    pairs: &[(usize, usize)],
    mask_dropouts: bool,
) -> Vec<f64> {
    let n_cells = expression.shape()[0];
    let n_genes = expression.shape()[1];
    assert!(
        pairs
            .iter()
            .all(|&(tf, target)| tf < n_genes && target < n_genes),
        "correlation edge contains an out-of-range gene index"
    );

    let mut groups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_genes];
    for (edge_idx, &(tf_idx, target_idx)) in pairs.iter().enumerate() {
        groups[tf_idx].push((edge_idx, target_idx));
    }

    let grouped: Vec<Vec<(usize, f64)>> = groups
        .into_par_iter()
        .enumerate()
        .filter_map(|(tf_idx, edges)| {
            if edges.is_empty() {
                None
            } else if mask_dropouts {
                let mut stats = vec![PairStats::default(); edges.len()];
                for cell in 0..n_cells {
                    let x = expression[(cell, tf_idx)] as f64;
                    if x == 0.0 {
                        continue;
                    }
                    for (slot, &(_, target_idx)) in stats.iter_mut().zip(&edges) {
                        let y = expression[(cell, target_idx)] as f64;
                        if y != 0.0 {
                            slot.add(x, y);
                        }
                    }
                }
                Some(
                    edges
                        .into_iter()
                        .zip(stats)
                        .map(|((edge_idx, _), stat)| (edge_idx, stat.correlation()))
                        .collect(),
                )
            } else {
                let mut sum_x = 0.0_f64;
                let mut sum_xx = 0.0_f64;
                let mut targets = vec![TargetStats::default(); edges.len()];
                for cell in 0..n_cells {
                    let x = expression[(cell, tf_idx)] as f64;
                    sum_x += x;
                    sum_xx += x * x;
                    for (slot, &(_, target_idx)) in targets.iter_mut().zip(&edges) {
                        slot.add(x, expression[(cell, target_idx)] as f64);
                    }
                }
                Some(
                    edges
                        .into_iter()
                        .zip(targets)
                        .map(|((edge_idx, _), target)| {
                            (
                                edge_idx,
                                pearson_from_sums(
                                    n_cells,
                                    sum_x,
                                    target.sum_y,
                                    sum_xx,
                                    target.sum_yy,
                                    target.sum_xy,
                                ),
                            )
                        })
                        .collect(),
                )
            }
        })
        .collect();

    let mut out = vec![f64::NAN; pairs.len()];
    for group in grouped {
        for (edge_idx, rho) in group {
            out[edge_idx] = rho;
        }
    }
    out
}

/// Pearson correlations for selected pairs in a sorted CSC matrix.
pub fn correlations_sparse_csc(
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    n_cells: usize,
    n_genes: usize,
    pairs: &[(usize, usize)],
    mask_dropouts: bool,
) -> Vec<f64> {
    assert_eq!(indptr.len(), n_genes + 1, "CSC indptr size mismatch");
    assert_eq!(indices.len(), data.len(), "CSC indices/data size mismatch");
    assert!(
        pairs
            .iter()
            .all(|&(tf, target)| tf < n_genes && target < n_genes),
        "correlation edge contains an out-of-range gene index"
    );

    let column_stats: Vec<(f64, f64)> = if mask_dropouts {
        Vec::new()
    } else {
        (0..n_genes)
            .into_par_iter()
            .map(|gene| {
                data[indptr[gene]..indptr[gene + 1]].iter().fold(
                    (0.0_f64, 0.0_f64),
                    |(sum, sum_sq), &value| {
                        let value = value as f64;
                        (sum + value, sum_sq + value * value)
                    },
                )
            })
            .collect()
    };

    pairs
        .par_iter()
        .map(|&(tf_idx, target_idx)| {
            let tf_range = indptr[tf_idx]..indptr[tf_idx + 1];
            let target_range = indptr[target_idx]..indptr[target_idx + 1];
            let mut i = tf_range.start;
            let mut j = target_range.start;
            if mask_dropouts {
                let mut stats = PairStats::default();
                while i < tf_range.end && j < target_range.end {
                    match indices[i].cmp(&indices[j]) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            stats.add(data[i] as f64, data[j] as f64);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                stats.correlation()
            } else {
                let mut sum_xy = 0.0_f64;
                while i < tf_range.end && j < target_range.end {
                    match indices[i].cmp(&indices[j]) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            sum_xy += data[i] as f64 * data[j] as f64;
                            i += 1;
                            j += 1;
                        }
                    }
                }
                let (sum_x, sum_xx) = column_stats[tf_idx];
                let (sum_y, sum_yy) = column_stats[target_idx];
                pearson_from_sums(n_cells, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
            }
        })
        .collect()
}

pub fn regulations_from_correlations(rhos: &[f64], threshold: f64) -> Vec<i8> {
    assert!(threshold.is_finite() && threshold > 0.0);
    rhos.iter()
        .map(|&rho| {
            if rho > threshold {
                1
            } else if rho < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct TargetStats {
    sum_y: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl TargetStats {
    fn add(&mut self, x: f64, y: f64) {
        self.sum_y += y;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }
}

#[derive(Clone, Copy, Default)]
struct PairStats {
    n: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl PairStats {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }

    fn correlation(self) -> f64 {
        pearson_from_sums(
            self.n,
            self.sum_x,
            self.sum_y,
            self.sum_xx,
            self.sum_yy,
            self.sum_xy,
        )
    }
}

fn pearson_from_sums(
    n: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
) -> f64 {
    if n < 2 {
        return f64::NAN;
    }
    let n = n as f64;
    let covariance = n * sum_xy - sum_x * sum_y;
    let variance_x = n * sum_xx - sum_x * sum_x;
    let variance_y = n * sum_yy - sum_y * sum_y;
    if variance_x <= 0.0 || variance_y <= 0.0 {
        return f64::NAN;
    }
    (covariance / (variance_x * variance_y).sqrt()).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_correlations_classify_positive_negative_and_constant_edges() {
        let values = ndarray::arr2(&[
            [0.0_f32, 0.0, 4.0, 1.0],
            [1.0, 2.0, 3.0, 1.0],
            [2.0, 4.0, 2.0, 1.0],
            [3.0, 6.0, 1.0, 1.0],
            [4.0, 8.0, 0.0, 1.0],
        ]);
        let rhos = correlations_dense_view(values.view(), &[(0, 1), (0, 2), (0, 3)], false);
        assert!((rhos[0] - 1.0).abs() < 1e-12);
        assert!((rhos[1] + 1.0).abs() < 1e-12);
        assert!(rhos[2].is_nan());
        assert_eq!(regulations_from_correlations(&rhos, 0.03), vec![1, -1, 0]);
    }

    #[test]
    fn sparse_and_dense_correlations_match_with_and_without_dropout_masking() {
        let values = ndarray::arr2(&[
            [0.0_f32, 0.0, 4.0],
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 2.0],
            [3.0, 0.0, 1.0],
            [4.0, 8.0, 0.0],
        ]);
        let pairs = [(0, 1), (0, 2)];
        let (indptr, indices, data) = dense_to_csc(values.view());
        for mask_dropouts in [false, true] {
            let dense = correlations_dense_view(values.view(), &pairs, mask_dropouts);
            let sparse = correlations_sparse_csc(
                &indptr,
                &indices,
                &data,
                values.nrows(),
                values.ncols(),
                &pairs,
                mask_dropouts,
            );
            for (a, b) in dense.iter().zip(&sparse) {
                if a.is_nan() {
                    assert!(b.is_nan());
                } else {
                    assert!((a - b).abs() < 1e-12, "dense={a} sparse={b}");
                }
            }
        }
    }

    fn dense_to_csc(values: ArrayView2<'_, f32>) -> (Vec<usize>, Vec<i32>, Vec<f32>) {
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for gene in 0..values.ncols() {
            for cell in 0..values.nrows() {
                let value = values[(cell, gene)];
                if value != 0.0 {
                    indices.push(cell as i32);
                    data.push(value);
                }
            }
            indptr.push(indices.len());
        }
        (indptr, indices, data)
    }
}
