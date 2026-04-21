//! Build a cells × peaks sparse count matrix from fragments + peaks.
//!
//! Algorithm (sorted-sweep, O((F + P) log(F + P))):
//!
//! 1. Group fragments and peaks by chromosome.
//! 2. For each chromosome, sort both by `start`.
//! 3. Sweep peaks and fragments in parallel; for each peak, find all
//!    fragments whose `[start, end)` overlaps the peak's `[start, end)`.
//! 4. Increment the count at `(barcode, peak)`.
//!
//! Output is CSR-like: `(data, indices, indptr)` where rows are barcodes
//! and columns are peaks. This is the format scipy.sparse.csr_matrix
//! accepts directly, and what rustscenic's Topics stage expects.

use crate::fragments::FragmentTable;
use crate::peaks::PeakTable;

/// Sparse matrix in CSR layout.
///
/// `data[k]` is the count in row `indices` lookup… standard CSR:
/// - `indptr` has length `n_rows + 1`
/// - row `r`'s non-zero entries live in `data[indptr[r]..indptr[r+1]]`
///   with column indices `indices[indptr[r]..indptr[r+1]]`
pub struct CsrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<u32>,
    pub indices: Vec<u32>,
    pub indptr: Vec<u64>,
}

impl CsrMatrix {
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Build the cells × peaks matrix.
///
/// Returns:
/// - CSR matrix shape `(n_barcodes, n_peaks)` with `u32` counts
/// - unchanged barcode names (from `fragments.barcode_names`)
/// - peak names in peak-table row order
pub fn build_cell_peak_matrix(
    fragments: &FragmentTable,
    peaks: &PeakTable,
) -> (CsrMatrix, Vec<String>, Vec<String>) {
    let n_barcodes = fragments.n_barcodes();
    let n_peaks = peaks.len();

    // 1. Align peaks' chrom indices into fragments' chrom space.
    //    Peaks on chroms not in fragments are dropped (mapped to None).
    let peak_chrom_aligned = peaks.align_chroms_to(&fragments.chrom_names);

    // 2. For each chromosome in the fragments' name space, collect
    //    (start, end, barcode_idx) fragment triples sorted by start,
    //    and (start, end, peak_row) peak triples sorted by start.
    //    Then sweep.
    //
    // We accumulate counts in a HashMap keyed by (barcode_idx, peak_row)
    // for each chromosome, then fold into CSR at the end.
    use std::collections::HashMap;
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();

    let n_chroms = fragments.n_chroms();
    for c in 0..n_chroms as u32 {
        let frag_rows: Vec<usize> = fragments
            .chrom_idx
            .iter()
            .enumerate()
            .filter_map(|(i, &cc)| (cc == c).then_some(i))
            .collect();
        let peak_rows: Vec<usize> = peak_chrom_aligned
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| (m == Some(c)).then_some(i))
            .collect();
        if frag_rows.is_empty() || peak_rows.is_empty() {
            continue;
        }

        // Sort by start
        let mut frags_sorted = frag_rows.clone();
        frags_sorted.sort_by_key(|&i| fragments.start[i]);
        let mut peaks_sorted = peak_rows.clone();
        peaks_sorted.sort_by_key(|&i| peaks.start[i]);

        // Sorted-sweep: for each peak, find overlapping fragments.
        // Because we sorted both by start, we can keep a "first candidate
        // fragment" pointer that only moves forward as peaks advance.
        let mut first_candidate = 0_usize;
        for &p in &peaks_sorted {
            let p_start = peaks.start[p];
            let p_end = peaks.end[p];

            // Advance first_candidate past fragments that end before peak starts.
            while first_candidate < frags_sorted.len()
                && fragments.end[frags_sorted[first_candidate]] <= p_start
            {
                first_candidate += 1;
            }

            // Walk forward from first_candidate, stop when fragment starts
            // are past peak end.
            for &f in frags_sorted.iter().skip(first_candidate) {
                let f_start = fragments.start[f];
                if f_start >= p_end {
                    break;
                }
                // Overlap check: f_end > p_start is already satisfied by
                // the first_candidate advance. Also need f_start < p_end,
                // satisfied by the loop condition above.
                let barcode = fragments.barcode_idx[f];
                *counts.entry((barcode, p as u32)).or_insert(0) += 1;
            }
        }
    }

    // 3. Fold into CSR.
    // Group by barcode row, sort each row's column indices ascending.
    let mut rows: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_barcodes];
    for ((b, p), cnt) in counts {
        rows[b as usize].push((p, cnt));
    }
    let mut indptr = Vec::with_capacity(n_barcodes + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0_u64);
    for row in &mut rows {
        row.sort_by_key(|(p, _)| *p);
        for &(p, c) in row.iter() {
            indices.push(p);
            data.push(c);
        }
        indptr.push(data.len() as u64);
    }

    let csr = CsrMatrix {
        n_rows: n_barcodes,
        n_cols: n_peaks,
        data,
        indices,
        indptr,
    };

    (csr, fragments.barcode_names.clone(), peaks.name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragments::read_fragments_from;
    use crate::peaks::read_peaks_from;
    use std::io::Cursor;

    // Fragments: AAA has 2 overlapping peak1, 1 overlapping peak2. BBB has 1 overlapping peak2.
    const FRAGS: &str = "\
chr1\t150\t250\tAAA-1\t1
chr1\t180\t290\tAAA-1\t1
chr1\t600\t700\tAAA-1\t1
chr1\t620\t680\tBBB-1\t1
chr2\t50\t100\tAAA-1\t1
";

    const PEAKS: &str = "\
chr1\t100\t300\tpeak1
chr1\t500\t800\tpeak2
chr2\t0\t30\tpeak3
chr3\t1\t100\tnowhere
";

    #[test]
    fn builds_correct_matrix() {
        let f = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        let p = read_peaks_from(Cursor::new(PEAKS)).unwrap();
        let (mtx, bnames, pnames) = build_cell_peak_matrix(&f, &p);

        assert_eq!(mtx.n_rows, 2); // AAA-1, BBB-1
        assert_eq!(mtx.n_cols, 4); // peak1, peak2, peak3, nowhere
        assert_eq!(bnames, vec!["AAA-1".to_string(), "BBB-1".to_string()]);
        assert_eq!(pnames.len(), 4);

        // Row 0 = AAA-1: peak1=2, peak2=1, peak3=0 (chr2 50-100 vs peak3 0-30, no overlap)
        // Row 1 = BBB-1: peak2=1
        let row0_start = mtx.indptr[0] as usize;
        let row0_end = mtx.indptr[1] as usize;
        let row0_cols: Vec<u32> = mtx.indices[row0_start..row0_end].to_vec();
        let row0_data: Vec<u32> = mtx.data[row0_start..row0_end].to_vec();
        assert_eq!(row0_cols, vec![0, 1]);
        assert_eq!(row0_data, vec![2, 1]);

        let row1_start = mtx.indptr[1] as usize;
        let row1_end = mtx.indptr[2] as usize;
        let row1_cols: Vec<u32> = mtx.indices[row1_start..row1_end].to_vec();
        let row1_data: Vec<u32> = mtx.data[row1_start..row1_end].to_vec();
        assert_eq!(row1_cols, vec![1]);
        assert_eq!(row1_data, vec![1]);
    }

    #[test]
    fn drops_peaks_on_chroms_not_in_fragments() {
        let f = read_fragments_from(Cursor::new(FRAGS)).unwrap();
        let p = read_peaks_from(Cursor::new(PEAKS)).unwrap();
        let (mtx, _, _) = build_cell_peak_matrix(&f, &p);
        // peak on chr3 should contribute 0 counts — check no row has a non-zero at col 3 (nowhere)
        for row in 0..mtx.n_rows {
            let s = mtx.indptr[row] as usize;
            let e = mtx.indptr[row + 1] as usize;
            assert!(!mtx.indices[s..e].contains(&3));
        }
    }

    #[test]
    fn empty_fragments_returns_empty_matrix() {
        let f = FragmentTable::default();
        let p = read_peaks_from(Cursor::new(PEAKS)).unwrap();
        let (mtx, _, _) = build_cell_peak_matrix(&f, &p);
        assert_eq!(mtx.n_rows, 0);
        assert_eq!(mtx.nnz(), 0);
    }

    #[test]
    fn touching_but_not_overlapping_excluded() {
        // Peak: 100..200. Fragment: 200..300. Half-open, no overlap.
        let frags = "chr1\t200\t300\tA-1\t1\n";
        let peaks = "chr1\t100\t200\tp\n";
        let f = read_fragments_from(Cursor::new(frags)).unwrap();
        let p = read_peaks_from(Cursor::new(peaks)).unwrap();
        let (mtx, _, _) = build_cell_peak_matrix(&f, &p);
        assert_eq!(mtx.nnz(), 0);
    }
}
