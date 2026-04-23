//! Cell-level QC metrics for scATAC fragments.
//!
//! Three metrics follow the aertslab / pycisTopic conventions and
//! can be used as per-barcode filters before building the cells × peaks
//! matrix:
//!
//! - **Insert-size distribution statistics** — per-barcode mean /
//!   median insert size, and counts in nucleosome bands. Captures
//!   library-prep quality (expect a dominant sub-nucleosomal peak plus
//!   mono- / di-nucleosomal peaks from intact chromatin).
//!
//! - **TSS enrichment** — signal-over-background ratio at gene
//!   transcription start sites. Higher values mean the library is
//!   enriched for open chromatin at promoters (expected; most labs
//!   require ≥ 4 before downstream analysis).
//!
//! - **FRiP** — fraction of reads in peaks. Measures how much of the
//!   per-cell signal is concentrated at reproducible regulatory
//!   elements vs random background. Typical cut-offs ≥ 0.15.
//!
//! All three are computed over a passed-in fragment table and a
//! passed-in BED (TSS or peaks). The caller decides which barcodes
//! pass; this module just produces numbers.

use crate::fragments::FragmentTable;
use crate::peaks::PeakTable;

/// Per-barcode insert-size summary statistics.
///
/// `n_fragments` is the fragment count used to compute the stats (so
/// callers can drop barcodes with too few fragments for stable estimates).
#[derive(Debug, Clone)]
pub struct InsertSizeStats {
    pub mean: f32,
    pub median: f32,
    pub n_fragments: u32,
    /// Counts in the three standard nucleosomal bands.
    pub sub_nucleosomal: u32,  // < 150 bp
    pub mono_nucleosomal: u32, // 150..300 bp
    pub di_nucleosomal: u32,   // 300..450 bp
}

impl InsertSizeStats {
    fn empty() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            n_fragments: 0,
            sub_nucleosomal: 0,
            mono_nucleosomal: 0,
            di_nucleosomal: 0,
        }
    }
}

/// Per-barcode insert-size statistics, parallel to `table.barcode_names`.
///
/// Insert size = `end - start` for each fragment; median is computed
/// exactly by sorting the per-barcode slice rather than using a
/// streaming estimator. At typical ATAC coverage (~10k fragments per
/// cell) this is cheap.
pub fn insert_size_stats(table: &FragmentTable) -> Vec<InsertSizeStats> {
    let n_barcodes = table.n_barcodes();
    if n_barcodes == 0 {
        return Vec::new();
    }

    // Bucket fragment lengths by barcode first so we can compute medians.
    let mut per_bc: Vec<Vec<u32>> = (0..n_barcodes).map(|_| Vec::new()).collect();
    for i in 0..table.len() {
        let b = table.barcode_idx[i] as usize;
        let size = table.end[i] - table.start[i];
        per_bc[b].push(size);
    }

    per_bc
        .into_iter()
        .map(|mut sizes| {
            let n = sizes.len() as u32;
            if n == 0 {
                return InsertSizeStats::empty();
            }
            let sum: u64 = sizes.iter().map(|s| *s as u64).sum();
            let mean = (sum as f64 / n as f64) as f32;
            sizes.sort_unstable();
            let median = if n % 2 == 1 {
                sizes[sizes.len() / 2] as f32
            } else {
                let lo = sizes[sizes.len() / 2 - 1] as f64;
                let hi = sizes[sizes.len() / 2] as f64;
                ((lo + hi) / 2.0) as f32
            };
            let mut sub = 0u32;
            let mut mono = 0u32;
            let mut di = 0u32;
            for &s in &sizes {
                if s < 150 {
                    sub += 1;
                } else if s < 300 {
                    mono += 1;
                } else if s < 450 {
                    di += 1;
                }
            }
            InsertSizeStats {
                mean,
                median,
                n_fragments: n,
                sub_nucleosomal: sub,
                mono_nucleosomal: mono,
                di_nucleosomal: di,
            }
        })
        .collect()
}

/// Per-barcode TSS enrichment score (signal-over-background).
///
/// Computed by counting fragment insertion events in a window around
/// each TSS and comparing to a flanking background. Returns a `Vec`
/// parallel to `table.barcode_names`. Barcodes with no TSS-overlapping
/// fragments get a score of 0.
///
/// The simplified Corces-2018 formulation used here:
///   - center_window:  `tss ± 50 bp` (the 101-bp window the TSS signal peaks at)
///   - flanking_window: `tss ± 2_000 bp` but excluding the center (3,899 bp combined)
///   - score = mean insertions per bp in center / mean insertions per bp in flank
///
/// `tss_sites` should be the midpoints of TSS features in the same
/// chromosome namespace as `table.chrom_names` (or a subset; non-matching
/// chroms are silently ignored).
pub fn tss_enrichment(
    table: &FragmentTable,
    tss_sites: &[TssSite],
) -> Vec<f32> {
    const CENTER_HALFWIDTH: u32 = 50;     // bp
    const FLANK_HALFWIDTH: u32 = 2_000;   // bp
    let center_bp = (CENTER_HALFWIDTH * 2 + 1) as f32; // 101
    let flank_bp_total = (FLANK_HALFWIDTH * 2 + 1 - (CENTER_HALFWIDTH * 2 + 1)) as f32; // 3899

    let n_barcodes = table.n_barcodes();
    let mut center_counts: Vec<u64> = vec![0; n_barcodes];
    let mut flank_counts: Vec<u64> = vec![0; n_barcodes];

    // Resolve TSS chroms into fragment's chrom index space once.
    // For each TSS, find the matching chrom_idx in the fragment table
    // (or None if the chrom isn't present).
    let tss_by_chrom = group_tss_by_chrom(table, tss_sites);

    for (chrom_idx, site_positions) in tss_by_chrom {
        if site_positions.is_empty() {
            continue;
        }
        let mut sorted_sites = site_positions;
        sorted_sites.sort_unstable();

        // Iterate fragments on this chromosome. For each fragment, add 1
        // to center_counts[bc] and flank_counts[bc] for every TSS the
        // fragment overlaps the center/flank window of. Binary search in
        // the sorted TSS list bounds the work.
        for i in 0..table.len() {
            if table.chrom_idx[i] != chrom_idx {
                continue;
            }
            let frag_start = table.start[i];
            let frag_end = table.end[i];
            let bc = table.barcode_idx[i] as usize;
            accumulate_tss_overlap(
                frag_start,
                frag_end,
                &sorted_sites,
                CENTER_HALFWIDTH,
                FLANK_HALFWIDTH,
                bc,
                &mut center_counts,
                &mut flank_counts,
            );
        }
    }

    (0..n_barcodes)
        .map(|b| {
            let center = center_counts[b] as f32 / center_bp;
            let flank = flank_counts[b] as f32 / flank_bp_total;
            if flank <= 0.0 {
                0.0
            } else {
                center / flank
            }
        })
        .collect()
}

/// A transcription-start-site reference.
///
/// `position` is the 0-based TSS coordinate (typically gene start for +
/// strand, gene end for − strand; most TSS BEDs already resolve this).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TssSite {
    pub chrom: String,
    pub position: u32,
}

fn group_tss_by_chrom(
    table: &FragmentTable,
    tss_sites: &[TssSite],
) -> Vec<(u32, Vec<u32>)> {
    use crate::peaks::normalise_chrom;
    // Pre-normalise the fragment-table chrom names so TSS passed in either
    // UCSC (`chr1`) or Ensembl (`1`) convention joins — same pattern the
    // peaks module uses. Without this, a TSS BED in the wrong convention
    // produces a silently all-zero enrichment (the exact class of bug that
    // hit the cellxgene integration earlier).
    let frag_chroms_norm: Vec<String> = table
        .chrom_names
        .iter()
        .map(|n| normalise_chrom(n))
        .collect();
    let mut sites_by_chrom: Vec<Vec<u32>> = (0..table.n_chroms()).map(|_| Vec::new()).collect();
    for tss in tss_sites {
        let tss_norm = normalise_chrom(&tss.chrom);
        for (i, name) in frag_chroms_norm.iter().enumerate() {
            if name == &tss_norm {
                sites_by_chrom[i].push(tss.position);
                break;
            }
        }
    }
    sites_by_chrom
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn accumulate_tss_overlap(
    frag_start: u32,
    frag_end: u32,
    sorted_tss: &[u32],
    center_half: u32,
    flank_half: u32,
    bc: usize,
    center_counts: &mut [u64],
    flank_counts: &mut [u64],
) {
    // Fragment insertion points: both ends are insertion events in ATAC.
    // For simplicity treat the fragment as a (start..end) interval and
    // count any TSS whose center/flank window overlaps it.
    // Window around TSS: [tss - flank_half, tss + flank_half].
    // The flank window includes the center; we count each separately.
    if sorted_tss.is_empty() {
        return;
    }
    // Earliest relevant TSS: frag_end must be > tss - flank_half, so
    // tss < frag_end + flank_half + 1. Find last position that passes via binary search.
    // Latest relevant TSS: frag_start must be < tss + flank_half + 1.
    let lo_tss = frag_start.saturating_sub(flank_half);
    let hi_tss = frag_end.saturating_add(flank_half);
    let lo = sorted_tss.partition_point(|&t| t < lo_tss);
    let hi = sorted_tss.partition_point(|&t| t <= hi_tss);
    for &tss in &sorted_tss[lo..hi] {
        let flank_start = tss.saturating_sub(flank_half);
        let flank_end = tss.saturating_add(flank_half);
        let center_start = tss.saturating_sub(center_half);
        let center_end = tss.saturating_add(center_half);
        if frag_end <= flank_start || frag_start > flank_end {
            continue;
        }
        flank_counts[bc] += 1;
        if !(frag_end <= center_start || frag_start > center_end) {
            center_counts[bc] += 1;
        }
    }
}

/// Per-barcode fraction of fragments in peaks (FRiP).
///
/// Returns a Vec parallel to `table.barcode_names`: each value is the
/// share of that barcode's fragments that overlap at least one peak in
/// `peaks`. Peaks on chromosomes absent from the fragment table are
/// dropped (matches `build_cell_peak_matrix` behaviour).
pub fn frip(table: &FragmentTable, peaks: &PeakTable) -> Vec<f32> {
    let n_barcodes = table.n_barcodes();
    if n_barcodes == 0 || table.is_empty() {
        return vec![0.0; n_barcodes];
    }

    // Align peak chroms to the fragment chrom namespace.
    let peak_chroms_aligned = peaks.align_chroms_to(&table.chrom_names);

    // For each chromosome present in fragments, pre-sort that chromosome's
    // peaks by start so we can binary-search.
    let n_chroms = table.n_chroms();
    let mut peaks_by_chrom: Vec<Vec<(u32, u32)>> = (0..n_chroms).map(|_| Vec::new()).collect();
    for (p, &maybe_cx) in peak_chroms_aligned.iter().enumerate() {
        if let Some(cx) = maybe_cx {
            peaks_by_chrom[cx as usize].push((peaks.start[p], peaks.end[p]));
        }
    }
    for v in peaks_by_chrom.iter_mut() {
        v.sort_unstable_by_key(|&(s, _)| s);
    }

    let mut in_peak: Vec<u32> = vec![0; n_barcodes];
    let mut total: Vec<u32> = vec![0; n_barcodes];

    for i in 0..table.len() {
        let bc = table.barcode_idx[i] as usize;
        total[bc] += 1;
        let chrom_peaks = &peaks_by_chrom[table.chrom_idx[i] as usize];
        if fragment_hits_any_peak(table.start[i], table.end[i], chrom_peaks) {
            in_peak[bc] += 1;
        }
    }

    (0..n_barcodes)
        .map(|b| if total[b] == 0 { 0.0 } else { in_peak[b] as f32 / total[b] as f32 })
        .collect()
}

fn fragment_hits_any_peak(
    frag_start: u32,
    frag_end: u32,
    peaks: &[(u32, u32)],
) -> bool {
    if peaks.is_empty() {
        return false;
    }
    // Find first peak whose start is > frag_end; earlier peaks are candidates.
    let hi = peaks.partition_point(|&(ps, _)| ps < frag_end);
    // Walk backward from hi-1, stop when peak_end <= frag_start.
    for j in (0..hi).rev() {
        let (ps, pe) = peaks[j];
        if pe <= frag_start {
            break;
        }
        if ps < frag_end && pe > frag_start {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragments::read_fragments_from;
    use crate::peaks::read_peaks_from;
    use std::io::Cursor;

    const FRAGS: &str = "\
chr1\t100\t200\tAAA-1\t1
chr1\t150\t250\tAAA-1\t1
chr1\t500\t900\tAAA-1\t1
chr1\t600\t620\tBBB-1\t1
chr2\t1000\t1100\tBBB-1\t1
";

    // ---- insert_size_stats ----

    #[test]
    fn insert_size_stats_counts_and_means_correct() {
        let t = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        let stats = insert_size_stats(&t);
        // barcode 0 = AAA-1: sizes 100, 100, 400 → mean 200, median 100
        assert_eq!(stats[0].n_fragments, 3);
        assert!((stats[0].mean - 200.0).abs() < 1e-5);
        assert_eq!(stats[0].median as u32, 100);
        // 100 is sub-nucleosomal (< 150), 400 is di-nucleosomal (300..450)
        assert_eq!(stats[0].sub_nucleosomal, 2);
        assert_eq!(stats[0].mono_nucleosomal, 0);
        assert_eq!(stats[0].di_nucleosomal, 1);
        // barcode 1 = BBB-1: sizes 20, 100 → mean 60, median 60
        assert_eq!(stats[1].n_fragments, 2);
        assert!((stats[1].mean - 60.0).abs() < 1e-5);
        assert_eq!(stats[1].sub_nucleosomal, 2);
    }

    #[test]
    fn insert_size_empty_table_returns_empty() {
        let t = crate::fragments::FragmentTable::default();
        assert!(insert_size_stats(&t).is_empty());
    }

    // ---- tss_enrichment ----

    #[test]
    fn tss_enrichment_high_when_fragments_cluster_at_tss() {
        // Ten AAA fragments all within 50 bp of TSS at chr1:10_000.
        // Ten BBB fragments scattered 1500-2000 bp away from any TSS.
        let mut lines = Vec::new();
        for i in 0..10 {
            let s = 10_000 - 30 + i * 3;
            lines.push(format!("chr1\t{}\t{}\tAAA-1\t1", s, s + 50));
        }
        for i in 0..10 {
            let s = 15_000 + i * 50;
            lines.push(format!("chr1\t{}\t{}\tBBB-1\t1", s, s + 50));
        }
        let t = read_fragments_from(Cursor::new(lines.join("\n"))).unwrap();
        let tss = vec![TssSite { chrom: "chr1".to_string(), position: 10_000 }];
        let scores = tss_enrichment(&t, &tss);
        // AAA-1 (all fragments in center window) should have high enrichment;
        // BBB-1 (fragments 5,000-5,500 away, outside flank) should be 0.
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > 5.0, "AAA-1 score {}", scores[0]);
        assert!(scores[1] < 0.5, "BBB-1 score {}", scores[1]);
    }

    #[test]
    fn tss_enrichment_ignores_chroms_not_in_fragments() {
        let t = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        // All TSS are on chr99 which doesn't exist in fragments.
        let tss = vec![TssSite { chrom: "chr99".to_string(), position: 10_000 }];
        let scores = tss_enrichment(&t, &tss);
        for s in scores {
            assert_eq!(s, 0.0);
        }
    }

    // ---- frip ----

    #[test]
    fn frip_counts_in_peak_fragments() {
        // Peaks cover chr1:100-300 and chr1:500-1000. chr2 has no peaks.
        let peaks_bed = "chr1\t100\t300\tpeak1\nchr1\t500\t1000\tpeak2\n";
        let t = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        let p = read_peaks_from(Cursor::new(peaks_bed)).unwrap();
        let scores = frip(&t, &p);
        // AAA-1 fragments: 100-200 (in peak1), 150-250 (in peak1), 500-900 (in peak2)
        // All 3 hit a peak → FRiP = 1.0
        assert!((scores[0] - 1.0).abs() < 1e-5);
        // BBB-1: 600-620 (in peak2), 1000-1100 (chr2, no peaks) → 1/2 = 0.5
        assert!((scores[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn frip_empty_peaks_gives_zero() {
        let t = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        let p = crate::peaks::PeakTable::default();
        let scores = frip(&t, &p);
        for s in scores {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn frip_peaks_on_other_chrom_drop() {
        let peaks_bed = "chrZ\t100\t200\tp\n";
        let t = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        let p = read_peaks_from(Cursor::new(peaks_bed)).unwrap();
        let scores = frip(&t, &p);
        for s in scores {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn frip_touching_not_overlapping_excluded() {
        // Fragment 200-300, peak 100-200 — touching, no overlap.
        let frags = "chr1\t200\t300\tX-1\t1\n";
        let peaks_bed = "chr1\t100\t200\tp\n";
        let t = read_fragments_from(Cursor::new(frags)).unwrap();
        let p = read_peaks_from(Cursor::new(peaks_bed)).unwrap();
        let scores = frip(&t, &p);
        assert_eq!(scores[0], 0.0);
    }
}
