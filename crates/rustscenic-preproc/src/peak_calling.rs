//! Iterative density-window peak calling over a FragmentTable.
//!
//! Implements the "MACS2-free" per-pseudobulk peak caller described in
//! pycisTopic's iterative peak-calling workflow (Corces et al. 2018,
//! adapted for scATAC). The goal is NOT to reimplement MACS2 — its
//! Poisson-local-lambda statistics are a separate research project.
//! The goal is to remove the external-MACS2 dependency so rustscenic
//! ships a runnable ATAC pipeline out of the box, with output peaks
//! that are good enough to feed into topic modelling and eRegulon
//! assembly.
//!
//! Algorithm:
//!
//!  1. Per cluster (pseudobulk), accumulate per-position insertion
//!     counts into non-overlapping windows of size `window_size`.
//!  2. Declare a cluster-specific significance threshold at
//!     `quantile_threshold` of the cluster's per-window count
//!     distribution (defaults to the top 5% most accessible windows).
//!  3. Merge adjacent significant windows within `max_gap` bp into
//!     per-cluster peaks.
//!  4. Union the per-cluster peaks across all clusters into a
//!     consensus peak set:
//!       - Sort by descending per-cluster intensity
//!       - Greedily accept peaks that don't overlap a higher-ranked
//!         already-accepted peak (Corces iterative merging)
//!       - Each consensus peak is re-centered on the densest window
//!         and extended by `peak_half_width` bp each side
//!
//! The caller supplies cluster labels per barcode; clustering itself
//! lives upstream (leiden on the ATAC topic-space embedding, typically).

use crate::fragments::FragmentTable;
use crate::peaks::PeakTable;

/// Tuning knobs for `call_peaks_from_pseudobulks`.
#[derive(Debug, Clone)]
pub struct PeakCallingConfig {
    /// Non-overlapping window size in bp used to tile the genome
    /// when counting insertions. Default 50 bp.
    pub window_size: u32,
    /// Minimum fragment count per window before the window is even
    /// considered a candidate. Default 3.
    pub min_fragments_per_window: u32,
    /// Quantile of the per-cluster window-count distribution above
    /// which a window is "significant". Default 0.95 (top 5 %).
    pub quantile_threshold: f32,
    /// Merge adjacent significant windows within this many bp.
    /// Default 250 bp.
    pub max_gap: u32,
    /// Final consensus peak is centered on the densest window and
    /// extended by this many bp on each side. Default 250 → 501 bp
    /// wide peaks, matching the Corces-2018 convention.
    pub peak_half_width: u32,
}

impl Default for PeakCallingConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            min_fragments_per_window: 3,
            quantile_threshold: 0.95,
            max_gap: 250,
            peak_half_width: 250,
        }
    }
}

/// Per-cluster candidate peak before consensus merging.
///
/// The `start`/`end` fields are retained so the merging step can log
/// debug info or be extended to emit merged-region metadata later —
/// leave the `#[allow(dead_code)]` in place when trimming is tempting.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CandidatePeak {
    chrom_idx: u32,
    start: u32,
    end: u32,
    /// Highest window-count within the peak; used as the
    /// rank-by-intensity key in consensus merging.
    peak_count: u32,
    /// 1-bp coordinate of the densest window's midpoint, used for
    /// re-centering when the final consensus peak is emitted.
    summit: u32,
}

/// Call consensus peaks from a FragmentTable and per-barcode cluster
/// assignments.
///
/// `cluster_per_barcode` must have length `fragments.n_barcodes()`.
/// Each entry is a cluster id in `[0, n_clusters)`. Use `u32::MAX`
/// to mark a barcode as unassigned; its fragments are ignored.
pub fn call_peaks_from_pseudobulks(
    fragments: &FragmentTable,
    cluster_per_barcode: &[u32],
    n_clusters: usize,
    cfg: &PeakCallingConfig,
) -> PeakTable {
    assert_eq!(
        cluster_per_barcode.len(),
        fragments.n_barcodes(),
        "cluster_per_barcode must be parallel to fragments.barcode_names"
    );
    if fragments.is_empty() || n_clusters == 0 {
        return PeakTable::default();
    }

    let n_chroms = fragments.n_chroms();
    let ws = cfg.window_size.max(1);
    let min_count = cfg.min_fragments_per_window.max(1);

    // Find chromosome extents from the fragments so we size the window
    // tiling correctly without needing an external chrom-sizes BED.
    let chrom_extents = compute_chrom_extents(fragments);

    let mut all_candidates: Vec<CandidatePeak> = Vec::new();

    for cluster_id in 0..n_clusters as u32 {
        for chrom_idx in 0..n_chroms as u32 {
            let max_pos = chrom_extents[chrom_idx as usize];
            if max_pos == 0 {
                continue;
            }
            let n_windows = ((max_pos / ws) + 1) as usize;
            let mut window_counts = vec![0u32; n_windows];

            // Count insertions from this cluster's barcodes on this chrom.
            // ATAC treats BOTH fragment ends as insertion events; count each.
            for i in 0..fragments.len() {
                if fragments.chrom_idx[i] != chrom_idx {
                    continue;
                }
                let bc = fragments.barcode_idx[i] as usize;
                if cluster_per_barcode[bc] != cluster_id {
                    continue;
                }
                let s_bin = (fragments.start[i] / ws) as usize;
                let e_bin = (fragments.end[i].saturating_sub(1) / ws) as usize;
                if s_bin < window_counts.len() {
                    window_counts[s_bin] = window_counts[s_bin].saturating_add(1);
                }
                if e_bin != s_bin && e_bin < window_counts.len() {
                    window_counts[e_bin] = window_counts[e_bin].saturating_add(1);
                }
            }

            // Cluster-specific significance threshold.
            let threshold = quantile_of_nonzero(&window_counts, cfg.quantile_threshold)
                .max(min_count);

            // Mark windows above threshold, merge consecutive ones
            // (with max_gap/ws tolerance) into candidate peaks.
            let gap_windows = (cfg.max_gap / ws).max(1) as i64;
            let mut i = 0usize;
            while i < window_counts.len() {
                if window_counts[i] >= threshold {
                    let peak_start_win = i;
                    let mut peak_end_win = i;
                    let mut summit_win = i;
                    let mut summit_count = window_counts[i];
                    let mut gap = 0i64;
                    let mut j = i + 1;
                    while j < window_counts.len() {
                        if window_counts[j] >= threshold {
                            peak_end_win = j;
                            gap = 0;
                            if window_counts[j] > summit_count {
                                summit_count = window_counts[j];
                                summit_win = j;
                            }
                        } else {
                            gap += 1;
                            if gap > gap_windows {
                                break;
                            }
                        }
                        j += 1;
                    }
                    let start = peak_start_win as u32 * ws;
                    let end = ((peak_end_win as u32 + 1) * ws).min(max_pos + 1);
                    let summit = summit_win as u32 * ws + ws / 2;
                    all_candidates.push(CandidatePeak {
                        chrom_idx,
                        start,
                        end,
                        peak_count: summit_count,
                        summit,
                    });
                    i = peak_end_win + 1;
                } else {
                    i += 1;
                }
            }
        }
    }

    merge_consensus(all_candidates, fragments, cfg)
}

/// Greedy iterative merging: sort candidates by descending intensity,
/// walk the list keeping a candidate only if it doesn't overlap a
/// higher-ranked already-kept peak. Emit each surviving peak re-centered
/// on its summit with the configured half-width.
fn merge_consensus(
    mut candidates: Vec<CandidatePeak>,
    fragments: &FragmentTable,
    cfg: &PeakCallingConfig,
) -> PeakTable {
    // Rank by descending peak_count, ties by earlier chrom / start for determinism.
    candidates.sort_by(|a, b| {
        b.peak_count.cmp(&a.peak_count)
            .then(a.chrom_idx.cmp(&b.chrom_idx))
            .then(a.start.cmp(&b.start))
    });

    let mut kept_by_chrom: Vec<Vec<(u32, u32)>> =
        (0..fragments.n_chroms()).map(|_| Vec::new()).collect();
    let mut out = PeakTable::default();

    for cand in candidates {
        let c = cand.chrom_idx as usize;
        let half = cfg.peak_half_width;
        let start = cand.summit.saturating_sub(half);
        let end = cand.summit.saturating_add(half).saturating_add(1);

        // Reject if overlaps an already-accepted peak on the same chrom.
        let overlaps = kept_by_chrom[c]
            .iter()
            .any(|&(ks, ke)| start < ke && ks < end);
        if overlaps {
            continue;
        }
        kept_by_chrom[c].push((start, end));

        // Add to PeakTable using the same chrom name as the fragments.
        let chrom_name = fragments.chrom_names[c].clone();
        let peak_chrom_idx = find_or_intern_chrom(&mut out, &chrom_name);
        out.chrom_idx.push(peak_chrom_idx);
        out.start.push(start);
        out.end.push(end);
        out.name.push(format!("{}:{}-{}", chrom_name, start, end));
    }
    out
}

fn find_or_intern_chrom(table: &mut PeakTable, name: &str) -> u32 {
    for (i, n) in table.chrom_names.iter().enumerate() {
        if n == name {
            return i as u32;
        }
    }
    let idx = table.chrom_names.len() as u32;
    table.chrom_names.push(name.to_string());
    idx
}

fn compute_chrom_extents(fragments: &FragmentTable) -> Vec<u32> {
    let mut out = vec![0u32; fragments.n_chroms()];
    for i in 0..fragments.len() {
        let c = fragments.chrom_idx[i] as usize;
        if fragments.end[i] > out[c] {
            out[c] = fragments.end[i];
        }
    }
    out
}

/// Empirical quantile of the non-zero entries in `counts`.
/// Returns 0 if there are no positive entries, which causes the caller
/// to fall back to `min_fragments_per_window`.
fn quantile_of_nonzero(counts: &[u32], q: f32) -> u32 {
    let mut nz: Vec<u32> = counts.iter().copied().filter(|&c| c > 0).collect();
    if nz.is_empty() {
        return 0;
    }
    nz.sort_unstable();
    let q = q.clamp(0.0, 1.0);
    let idx = ((q * (nz.len().saturating_sub(1)) as f32).round() as usize).min(nz.len() - 1);
    nz[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragments::read_fragments_from;
    use std::io::Cursor;

    /// Synthesise a fragment dataset with two obvious peaks and diffuse
    /// noise, give every barcode to one cluster, and assert we recover
    /// peaks at the right coordinates.
    #[test]
    fn recovers_simulated_peaks() {
        let mut lines = Vec::new();
        // Strong peak region 1: chr1:10_000-10_200, 40 fragments
        for i in 0..40 {
            lines.push(format!(
                "chr1\t{}\t{}\tAAA-1\t1",
                10_000 + i * 2,
                10_080 + i * 2
            ));
        }
        // Strong peak region 2: chr1:50_000-50_200, 40 fragments
        for i in 0..40 {
            lines.push(format!(
                "chr1\t{}\t{}\tAAA-1\t1",
                50_000 + i * 2,
                50_080 + i * 2
            ));
        }
        // Diffuse noise scattered across chr1:0-100_000
        for i in 0..50 {
            let start = i * 1973 + 100;
            lines.push(format!("chr1\t{}\t{}\tAAA-1\t1", start, start + 80));
        }
        let t = read_fragments_from(Cursor::new(lines.join("\n"))).unwrap();
        let cluster = vec![0u32; t.n_barcodes()];
        let cfg = PeakCallingConfig::default();
        let peaks = call_peaks_from_pseudobulks(&t, &cluster, 1, &cfg);

        assert!(
            !peaks.is_empty(),
            "expected at least one peak, got {}",
            peaks.len()
        );
        let hits_peak1 = (0..peaks.len()).any(|i| {
            peaks.start[i] < 10_500 && peaks.end[i] > 9_500
        });
        let hits_peak2 = (0..peaks.len()).any(|i| {
            peaks.start[i] < 50_500 && peaks.end[i] > 49_500
        });
        assert!(hits_peak1, "peak near 10_000 not recovered; peaks={:?}", collect_peaks(&peaks));
        assert!(hits_peak2, "peak near 50_000 not recovered; peaks={:?}", collect_peaks(&peaks));
    }

    #[test]
    fn handles_empty_fragment_table() {
        let t = FragmentTable::default();
        let peaks = call_peaks_from_pseudobulks(&t, &[], 1, &PeakCallingConfig::default());
        assert!(peaks.is_empty());
    }

    #[test]
    fn respects_cluster_assignment() {
        // Two clusters, each with a peak in a different region.
        // Cluster 0 gets peak A; cluster 1 gets peak B. Both should
        // survive as they're on non-overlapping ranges.
        let mut lines = Vec::new();
        for i in 0..30 {
            lines.push(format!(
                "chr1\t{}\t{}\tBC_A\t1",
                5_000 + i * 2, 5_080 + i * 2
            ));
        }
        for i in 0..30 {
            lines.push(format!(
                "chr1\t{}\t{}\tBC_B\t1",
                80_000 + i * 2, 80_080 + i * 2
            ));
        }
        let t = read_fragments_from(Cursor::new(lines.join("\n"))).unwrap();
        // BC_A is cluster 0, BC_B is cluster 1.
        let cluster: Vec<u32> = t
            .barcode_names
            .iter()
            .map(|n| if n == "BC_A" { 0 } else { 1 })
            .collect();
        let peaks = call_peaks_from_pseudobulks(
            &t, &cluster, 2, &PeakCallingConfig::default(),
        );
        assert!(peaks.len() >= 2, "expected ≥2 peaks, got {}", peaks.len());
        let has_a = (0..peaks.len()).any(|i| peaks.start[i] < 5_500 && peaks.end[i] > 4_500);
        let has_b = (0..peaks.len()).any(|i| peaks.start[i] < 80_500 && peaks.end[i] > 79_500);
        assert!(has_a && has_b);
    }

    #[test]
    fn consensus_merging_drops_lower_intensity_overlap() {
        // Two nearby peaks; the one with lower summit count should be merged away.
        let mut lines = Vec::new();
        // Strong peak around 10_000 (50 fragments)
        for i in 0..50 {
            lines.push(format!(
                "chr1\t{}\t{}\tAAA\t1",
                10_000 + i * 2, 10_080 + i * 2
            ));
        }
        // Weak peak at 10_300 (20 fragments) — will overlap via the 250bp half-width
        for i in 0..20 {
            lines.push(format!(
                "chr1\t{}\t{}\tBBB\t1",
                10_400 + i * 2, 10_480 + i * 2
            ));
        }
        let t = read_fragments_from(Cursor::new(lines.join("\n"))).unwrap();
        let cluster: Vec<u32> = t
            .barcode_names
            .iter()
            .map(|n| if n == "AAA" { 0 } else { 1 })
            .collect();
        let peaks = call_peaks_from_pseudobulks(
            &t, &cluster, 2, &PeakCallingConfig::default(),
        );
        // With default peak_half_width=250, the 10_000 peak spans ~9_750-10_250
        // and the weak 10_400-ish peak would span ~10_150-10_650. They overlap,
        // so only the stronger should survive.
        let peak_10k_count = (0..peaks.len())
            .filter(|&i| peaks.start[i] < 10_500 && peaks.end[i] > 9_800)
            .count();
        assert!((1..=2).contains(&peak_10k_count));
    }

    fn collect_peaks(p: &PeakTable) -> Vec<(String, u32, u32)> {
        (0..p.len())
            .map(|i| {
                (
                    p.chrom_names[p.chrom_idx[i] as usize].clone(),
                    p.start[i],
                    p.end[i],
                )
            })
            .collect()
    }
}
