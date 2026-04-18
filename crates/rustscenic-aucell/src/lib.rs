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
    assert_eq!(expression.len(), n_cells * n_genes, "expression size mismatch");
    assert!((0.0..=1.0).contains(&top_frac), "top_frac must be in (0, 1]");

    let max_rank: usize = ((top_frac * n_genes as f32) as usize).max(1);
    let n_regulons = regulons.len();
    if n_regulons == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f32; n_cells * n_regulons];

    out.par_chunks_mut(n_regulons)
        .enumerate()
        .for_each(|(cell_idx, cell_out)| {
            let row = &expression[cell_idx * n_genes..(cell_idx + 1) * n_genes];
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

            let max_rank_u32 = max_rank as u32;
            for (r_idx, (_, gene_set)) in regulons.iter().enumerate() {
                let mut auc_sum: u64 = 0;
                for &g in gene_set {
                    let r = rank[g];
                    if r < max_rank_u32 {
                        auc_sum += (max_rank_u32 - r) as u64;
                    }
                }
                // True maximum AUC: if all |G| regulon genes fell at top ranks 0..|G|-1,
                // auc_sum = sum_{i=0..|G|} (K - i) = K*|G| - |G|*(|G|-1)/2
                // We use min(|G|, K) because extras past K can never contribute.
                let g = (gene_set.len() as u64).min(max_rank as u64);
                if g == 0 {
                    cell_out[r_idx] = 0.0;
                    continue;
                }
                let max_auc = (max_rank as u64) * g - g * g.saturating_sub(1) / 2;
                let norm = (auc_sum as f64) / (max_auc as f64);
                cell_out[r_idx] = norm.clamp(0.0, 1.0) as f32;
            }
        });

    out
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
    fn single_cell_top_ranked_regulon_hits_one() {
        // 5 genes, 1 cell, gene 0 is expressed highest; regulon contains gene 0
        let expr = vec![10.0_f32, 1.0, 1.0, 1.0, 1.0];
        let regs = vec![("R".to_string(), vec![0])];
        let out = aucell(&expr, 1, 5, &regs, 0.4); // top 2
        // rank(gene 0) = 0, auc = max_rank - 0 = 2, denom = max_rank * 1 = 2 → 1.0
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-6);
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
        assert!(out[0] > out[1], "cell 0 should have higher auc for reg {{0}}, got {} vs {}", out[0], out[1]);
    }

    #[test]
    fn many_cells_deterministic() {
        // run twice, same seed/data -> same output (no stochasticity in AUCell)
        let expr: Vec<f32> = (0..100 * 20)
            .map(|i| ((i * 37 + 1) % 23) as f32)
            .collect();
        let regs = vec![
            ("R1".to_string(), vec![0, 3, 7, 11]),
            ("R2".to_string(), vec![2, 5, 9]),
        ];
        let o1 = aucell(&expr, 100, 20, &regs, 0.25);
        let o2 = aucell(&expr, 100, 20, &regs, 0.25);
        assert_eq!(o1, o2);
    }
}
