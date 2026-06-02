//! AUCell regulon activity scoring (pyscenic aucell replacement).
//!
//! Given an expression matrix and a set of regulons, compute the AUC of each
//! regulon's recovery curve per cell. Canonical Aibar et al. 2017 semantics:
//!
//!   For each cell c:
//!     1. rank all genes by expression (descending; ties broken by gene index)
//!     2. let K = floor(top_frac * n_genes) be the recovery-curve cutoff
//!     3. for each regulon r with gene set G_r:
//!        - let ranks = {rank(g, c) for g in G_r where rank(g, c) < K}
//!        - auc(c, r) = sum over ranks of (K - rank)
//!        - auc_norm(c, r) = auc(c, r) / (K * |G_r|)    (clamped to [0, 1])
//!
//! Output: (n_cells x n_regulons) f32 matrix of normalized recovery AUCs.
//!
//! Matches pyscenic.aucell.aucell at the algorithm level. Runtime:
//! O(n_cells * n_genes * log(n_genes))  for argsort
//! + O(n_cells * sum_r |G_r intersect top-K|)  for AUC aggregation
//!
//! Rayon-parallelized over cells.

use ndarray::ArrayView2;
use rayon::prelude::*;

/// Compute per-cell AUC matrix. Caller supplies gene-index regulons (post gene-name lookup).
///
/// - `expression`: row-major `[n_cells][n_genes]`, f32
/// - `regulons`: list of `(regulon_name, Vec<gene_idx>)`; gene indices must be < n_genes
/// - `top_frac`: fraction of top-ranked genes per cell used as cutoff (default 0.05)
///
/// Returns row-major `[n_cells][n_regulons]` normalized AUC matrix.
pub fn aucell(
    expression: &[f32],
    n_cells: usize,
    n_genes: usize,
    regulons: &[(String, Vec<usize>)],
    top_frac: f32,
) -> Vec<f32> {
    assert_eq!(
        expression.len(),
        n_cells * n_genes,
        "expression size mismatch"
    );
    let expression = ArrayView2::from_shape((n_cells, n_genes), expression)
        .expect("validated expression shape must construct row-major view");
    aucell_view(expression, regulons, top_frac)
}

/// Compute per-cell AUC matrix from any dense 2D view.
///
/// This keeps PyO3 callers from materialising a second C-contiguous copy when
/// pandas/AnnData already expose a valid strided float32 matrix.
pub fn aucell_view(
    expression: ArrayView2<'_, f32>,
    regulons: &[(String, Vec<usize>)],
    top_frac: f32,
) -> Vec<f32> {
    let n_cells = expression.shape()[0];
    let n_genes = expression.shape()[1];
    assert!(
        top_frac > 0.0 && top_frac <= 1.0,
        "top_frac must be in (0, 1]"
    );

    // Reject NaN in expression - silent corruption via partial_cmp tie-breaks.
    if expression.iter().any(|v| v.is_nan()) {
        panic!(
            "expression matrix contains NaN values - AUCell ranking is undefined. \
            Filter upstream (scanpy.pp.normalize_total + sc.pp.log1p on raw count \
            matrices, or drop rows with all-zero expression first)."
        );
    }

    // ctxcore.recovery.derive_rank_cutoff semantics (preserves R-SCENIC values):
    //   rank_cutoff = round(auc_threshold * n_genes) - 1
    // This is a 1-off-by-one that the Aertslab ctxcore bakes in for R
    // compatibility. Matching it keeps per-cell AUCs numerically equal to
    // pyscenic's output (verified against ctxcore.recovery.aucs).
    let rank_cutoff_raw = (top_frac * n_genes as f32).round() as i64;
    let rank_cutoff = (rank_cutoff_raw - 1).max(0) as usize;
    let n_regulons = regulons.len();
    if n_regulons == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; n_cells * n_regulons];

    out.par_chunks_mut(n_regulons)
        .enumerate()
        .for_each(|(cell_idx, cell_out)| {
            let row = expression.row(cell_idx);
            // Argsort descending by expression. Tie-break: lower gene index first
            // (deterministic across runs).
            let mut order: Vec<u32> = (0..n_genes as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                let va = row[a as usize];
                let vb = row[b as usize];
                // descending: b.cmp(a)
                match vb.partial_cmp(&va) {
                    Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
                    Some(ord) => ord,
                }
            });
            // Build rank array: rank[gene] = position (0-based) in sorted order
            let mut rank = vec![u32::MAX; n_genes];
            for (pos, &g) in order.iter().enumerate() {
                rank[g as usize] = pos as u32;
            }

            let rank_cutoff_u32 = rank_cutoff as u32;
            for (r_idx, (_, gene_set)) in regulons.iter().enumerate() {
                // ctxcore.recovery.weighted_auc1d: filter ranks < rank_cutoff,
                // then Riemann-sum the staircase recovery curve up to rank_cutoff.
                // For unit weights this equals sum of (rank_cutoff - r) for r < rank_cutoff,
                // which is algebraically the same as the "right Riemann" form.
                let mut auc_sum: u64 = 0;
                for &g in gene_set {
                    let r = rank[g];
                    if r < rank_cutoff_u32 {
                        auc_sum += (rank_cutoff_u32 - r) as u64;
                    }
                }
                let g_len = gene_set.len() as u64;
                if g_len == 0 {
                    cell_out[r_idx] = 0.0;
                    continue;
                }
                // ctxcore.recovery.aucs: maxauc = (rank_cutoff + 1) * y_max where
                // y_max = sum of weights (= |G| for unit weights).
                let max_auc = (rank_cutoff as u64 + 1) * g_len;
                let norm = (auc_sum as f64) / (max_auc as f64);
                cell_out[r_idx] = norm.clamp(0.0, 1.0) as f32;
            }
        });

    out
}

/// Sparse CSR AUCell for non-negative scRNA matrices without densifying.
///
/// This preserves the dense rank semantics, including gene-index tie breaks
/// for implicit zeros. It only ranks the genes that can contribute below the
/// recovery cutoff, avoiding an O(n_genes log n_genes) dense argsort per cell
/// on atlas-scale sparse matrices.
pub fn aucell_sparse_csr(
    row_ptr: &[usize],
    col_idx: &[i32],
    data: &[f32],
    n_cells: usize,
    n_genes: usize,
    regulons: &[(String, Vec<usize>)],
    top_frac: f32,
) -> Vec<f32> {
    assert_eq!(row_ptr.len(), n_cells + 1, "row_ptr size mismatch");
    assert_eq!(col_idx.len(), data.len(), "CSR index/data size mismatch");
    assert!(
        col_idx.iter().all(|&g| g >= 0 && (g as usize) < n_genes),
        "CSR column index contains a negative or out-of-range gene index"
    );
    assert!(
        top_frac > 0.0 && top_frac <= 1.0,
        "top_frac must be in (0, 1]"
    );
    if data.iter().any(|v| v.is_nan()) {
        panic!("expression matrix contains NaN values - AUCell ranking is undefined.");
    }

    let rank_cutoff_raw = (top_frac * n_genes as f32).round() as i64;
    let rank_cutoff = (rank_cutoff_raw - 1).max(0) as usize;
    let n_regulons = regulons.len();
    if n_regulons == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; n_cells * n_regulons];
    if rank_cutoff == 0 {
        return out;
    }

    let memberships = gene_regulon_memberships(n_genes, regulons);
    let regulon_lens: Vec<u64> = regulons
        .iter()
        .map(|(_, genes)| genes.len() as u64)
        .collect();
    let rank_cutoff_u32 = rank_cutoff as u32;
    let max_aucs: Vec<u64> = regulon_lens
        .iter()
        .map(|&len| (rank_cutoff as u64 + 1) * len)
        .collect();

    out.par_chunks_mut(n_regulons)
        .enumerate()
        .for_each(|(cell_idx, cell_out)| {
            let start = row_ptr[cell_idx];
            let end = row_ptr[cell_idx + 1];
            let mut positives: Vec<(u32, f32)> = Vec::new();
            let mut negatives: Vec<(u32, f32)> = Vec::new();
            let mut nonzero_genes: Vec<u32> = Vec::new();

            for k in start..end {
                let gene = col_idx[k] as u32;
                let value = data[k];
                if value > 0.0 {
                    positives.push((gene, value));
                    nonzero_genes.push(gene);
                } else if value < 0.0 {
                    negatives.push((gene, value));
                    nonzero_genes.push(gene);
                }
            }
            positives.sort_unstable_by(compare_sparse_entry_desc);

            let mut auc_sums = vec![0_u64; n_regulons];
            let mut rank = 0_u32;

            for &(gene, _) in positives.iter().take(rank_cutoff) {
                add_sparse_auc_contribution(
                    gene as usize,
                    rank,
                    rank_cutoff_u32,
                    &memberships,
                    &mut auc_sums,
                );
                rank += 1;
            }

            if rank < rank_cutoff_u32 {
                nonzero_genes.sort_unstable();
                nonzero_genes.dedup();
                let mut nz_pos = 0usize;
                for gene in 0..n_genes as u32 {
                    while nz_pos < nonzero_genes.len() && nonzero_genes[nz_pos] < gene {
                        nz_pos += 1;
                    }
                    if nz_pos < nonzero_genes.len() && nonzero_genes[nz_pos] == gene {
                        continue;
                    }
                    add_sparse_auc_contribution(
                        gene as usize,
                        rank,
                        rank_cutoff_u32,
                        &memberships,
                        &mut auc_sums,
                    );
                    rank += 1;
                    if rank >= rank_cutoff_u32 {
                        break;
                    }
                }
            }

            if rank < rank_cutoff_u32 {
                negatives.sort_unstable_by(compare_sparse_entry_desc);
                for &(gene, _) in negatives.iter().take((rank_cutoff_u32 - rank) as usize) {
                    add_sparse_auc_contribution(
                        gene as usize,
                        rank,
                        rank_cutoff_u32,
                        &memberships,
                        &mut auc_sums,
                    );
                    rank += 1;
                    if rank >= rank_cutoff_u32 {
                        break;
                    }
                }
            }

            for r_idx in 0..n_regulons {
                let max_auc = max_aucs[r_idx];
                if max_auc == 0 {
                    cell_out[r_idx] = 0.0;
                } else {
                    let norm = (auc_sums[r_idx] as f64) / (max_auc as f64);
                    cell_out[r_idx] = norm.clamp(0.0, 1.0) as f32;
                }
            }
        });

    out
}

fn gene_regulon_memberships(n_genes: usize, regulons: &[(String, Vec<usize>)]) -> Vec<Vec<usize>> {
    let mut memberships = vec![Vec::new(); n_genes];
    for (regulon_idx, (_, genes)) in regulons.iter().enumerate() {
        for &gene in genes {
            if gene < n_genes {
                memberships[gene].push(regulon_idx);
            }
        }
    }
    memberships
}

fn compare_sparse_entry_desc(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    match b.1.partial_cmp(&a.1) {
        Some(std::cmp::Ordering::Equal) | None => a.0.cmp(&b.0),
        Some(ord) => ord,
    }
}

fn add_sparse_auc_contribution(
    gene: usize,
    rank: u32,
    rank_cutoff: u32,
    memberships: &[Vec<usize>],
    auc_sums: &mut [u64],
) {
    for &regulon_idx in &memberships[gene] {
        auc_sums[regulon_idx] += (rank_cutoff - rank) as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn empty_regulons_returns_empty_matrix() {
        let expr = vec![1.0_f32; 10 * 5];
        let out = aucell(&expr, 10, 5, &[], 0.2);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "top_frac must be in (0, 1]")]
    fn rejects_zero_top_frac() {
        let expr = vec![1.0_f32; 5];
        let regs = vec![("R".to_string(), vec![0])];
        let _ = aucell(&expr, 1, 5, &regs, 0.0);
    }

    #[test]
    fn single_cell_top_ranked_regulon_matches_ctxcore() {
        // 5 genes, 1 cell, gene 0 is expressed highest; regulon contains gene 0.
        // top_frac=0.4 -> rank_cutoff = round(0.4*5)-1 = 1 (ctxcore R-compat).
        // Only gene 0 has rank 0 < 1, auc_sum = (1 - 0) * 1 = 1.
        // max_auc = (rank_cutoff+1) * |G| = 2 * 1 = 2.
        // Expected: 1 / 2 = 0.5 - verified equal to ctxcore.recovery.aucs output.
        let expr = vec![10.0_f32, 1.0, 1.0, 1.0, 1.0];
        let regs = vec![("R".to_string(), vec![0])];
        let out = aucell(&expr, 1, 5, &regs, 0.4);
        assert_abs_diff_eq!(out[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn regulon_gene_outside_top_k_contributes_zero() {
        // Gene 4 is lowest-ranked; top 2 excludes it
        let expr = vec![10.0_f32, 9.0, 8.0, 7.0, 1.0];
        let regs = vec![("R".to_string(), vec![4])];
        let out = aucell(&expr, 1, 5, &regs, 0.4);
        assert_abs_diff_eq!(out[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn regulon_is_not_affected_by_unrelated_genes() {
        // Two cells; regulon = {gene 0}. Cell 1 has gene 0 highest, cell 2 has gene 4 highest.
        let expr = vec![
            10.0, 1.0, 1.0, 1.0, 1.0, // cell 0
            1.0, 1.0, 1.0, 1.0, 10.0, // cell 1
        ];
        let regs = vec![("R".to_string(), vec![0])];
        let out = aucell(&expr, 2, 5, &regs, 0.4);
        assert!(
            out[0] > out[1],
            "cell 0 should have higher auc for reg {{0}}, got {} vs {}",
            out[0],
            out[1]
        );
    }

    #[test]
    fn many_cells_deterministic() {
        // run twice, same seed/data -> same output (no stochasticity in AUCell)
        let expr: Vec<f32> = (0..100 * 20).map(|i| ((i * 37 + 1) % 23) as f32).collect();
        let regs = vec![
            ("R1".to_string(), vec![0, 3, 7, 11]),
            ("R2".to_string(), vec![2, 5, 9]),
        ];
        let o1 = aucell(&expr, 100, 20, &regs, 0.25);
        let o2 = aucell(&expr, 100, 20, &regs, 0.25);
        assert_eq!(o1, o2);
    }

    #[test]
    fn sparse_csr_matches_dense_with_implicit_zero_ties() {
        let expr = vec![
            5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let regs = vec![
            ("early_zero".to_string(), vec![1, 2]),
            ("late_zero".to_string(), vec![8, 9]),
            ("positive".to_string(), vec![0, 2]),
        ];
        let dense = aucell(&expr, 3, 10, &regs, 0.5);
        let sparse = aucell_sparse_csr(
            &[0, 1, 1, 3],
            &[0, 0, 2],
            &[5.0, 3.0, 2.0],
            3,
            10,
            &regs,
            0.5,
        );
        assert_eq!(sparse, dense);
    }

    #[test]
    fn sparse_csr_matches_dense_with_negative_values() {
        let expr = vec![0.0, -1.0, 0.0, 2.0, -0.5, 0.0];
        let regs = vec![
            ("zeros".to_string(), vec![0, 2, 5]),
            ("negative".to_string(), vec![1, 4]),
            ("positive".to_string(), vec![3]),
        ];
        let dense = aucell(&expr, 1, 6, &regs, 1.0);
        let sparse = aucell_sparse_csr(&[0, 3], &[1, 3, 4], &[-1.0, 2.0, -0.5], 1, 6, &regs, 1.0);
        assert_eq!(sparse, dense);
    }

    #[test]
    #[should_panic(expected = "CSR column index contains a negative or out-of-range gene index")]
    fn sparse_csr_rejects_negative_col_idx() {
        let regs = vec![("R".to_string(), vec![0])];
        let _ = aucell_sparse_csr(&[0, 1], &[-1], &[1.0], 1, 3, &regs, 0.5);
    }
}
