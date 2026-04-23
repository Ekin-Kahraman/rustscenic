//! Peak BED parser + interval operations.
//!
//! Consensus peak BEDs from pycisTopic / MACS2 have 3–10 columns. We
//! only care about the first three (chrom, start, end) plus an optional
//! name column. Everything else is ignored.
//!
//! Peak IDs are assigned by row order, so downstream matrix columns
//! match the input BED's row order exactly.

use anyhow::{anyhow, Context, Result};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A genomic peak (region).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Peak {
    pub chrom: String,
    pub start: u32,
    pub end: u32,
    pub name: String,
}

/// Columnar peak table — one Vec per column.
///
/// `chrom` interned into `chrom_names` so peak–fragment joins can
/// be done by `u32` equality rather than string compare.
#[derive(Debug, Default)]
pub struct PeakTable {
    pub chrom_idx: Vec<u32>,
    pub chrom_names: Vec<String>,
    pub start: Vec<u32>,
    pub end: Vec<u32>,
    pub name: Vec<String>,
}

impl PeakTable {
    pub fn len(&self) -> usize {
        self.start.len()
    }

    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    pub fn n_chroms(&self) -> usize {
        self.chrom_names.len()
    }

    fn intern_chrom(&mut self, name: &str) -> u32 {
        for (i, n) in self.chrom_names.iter().enumerate() {
            if n == name {
                return i as u32;
            }
        }
        let idx = self.chrom_names.len() as u32;
        self.chrom_names.push(name.to_string());
        idx
    }

    /// Re-intern chromosome indices into the order of a reference
    /// chrom-name list (e.g. the one from a `FragmentTable`). Returns
    /// a parallel Vec where each element is the new chrom index, or
    /// `None` if the peak's chromosome isn't present in the reference.
    ///
    /// Uses normalised matching so that `chr1` and `1` are treated as
    /// the same chromosome (UCSC vs Ensembl convention), along with
    /// the `chrM` / `chrMT` / `MT` mitochondrial aliases. Without this
    /// normalisation, a peak BED with Ensembl chrom names against a
    /// fragments file with UCSC chrom names silently drops every peak
    /// — the same silent-zero failure class as the cellxgene ENSEMBL
    /// var_names bug, just in a different layer.
    ///
    /// Use this before joining peaks to fragments so chromosome
    /// comparisons reduce to `u32` equality.
    pub fn align_chroms_to(&self, reference_chrom_names: &[String]) -> Vec<Option<u32>> {
        let normalised_refs: Vec<String> = reference_chrom_names
            .iter()
            .map(|n| normalise_chrom(n))
            .collect();
        let mapping: Vec<Option<u32>> = self
            .chrom_names
            .iter()
            .map(|cn| {
                let cn_norm = normalise_chrom(cn);
                normalised_refs
                    .iter()
                    .position(|r| r == &cn_norm)
                    .map(|i| i as u32)
            })
            .collect();
        self.chrom_idx.iter().map(|&c| mapping[c as usize]).collect()
    }
}

/// Normalise a chromosome name to a canonical form for alignment.
///
/// - Strip a leading `chr` prefix (so `chr1` and `1` match)
/// - Collapse mitochondrial aliases `M` and `MT` to `MT`
/// - Uppercase the remainder so `chrX` and `chrx` match
///
/// Designed for the small set of conventions real data uses (UCSC,
/// Ensembl, NCBI RefSeq). Returns a new owned String so callers can
/// reuse it as a hashmap key.
pub fn normalise_chrom(name: &str) -> String {
    let trimmed = name.trim();
    // Strip leading "chr" (case-insensitive) if present
    let no_prefix = if trimmed.len() >= 3
        && trimmed[..3].eq_ignore_ascii_case("chr")
    {
        &trimmed[3..]
    } else {
        trimmed
    };
    let upper = no_prefix.to_ascii_uppercase();
    // Canonicalise mitochondrial name
    if upper == "M" {
        "MT".to_string()
    } else {
        upper
    }
}

/// Parse a BED file (plain or gzipped). Uses the first 3 columns for
/// coordinates; column 4 (if present) is stored as `name`, otherwise
/// `name` is set to `chrom:start-end`.
pub fn read_peaks<P: AsRef<Path>>(path: P) -> Result<PeakTable> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader: Box<dyn Read> = if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
    {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    read_peaks_from(reader)
}

/// Parse peaks from any `Read` source.
pub fn read_peaks_from<R: Read>(reader: R) -> Result<PeakTable> {
    let buffered = BufReader::with_capacity(1 << 20, reader);
    let mut table = PeakTable::default();

    for (line_no, line) in buffered.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_no + 1))?;
        if line.is_empty() || line.starts_with('#') || line.starts_with("track") {
            continue;
        }

        let mut fields = line.split('\t');
        let chrom = fields
            .next()
            .ok_or_else(|| anyhow!("line {}: missing chrom", line_no + 1))?;
        let start_s = fields
            .next()
            .ok_or_else(|| anyhow!("line {}: missing start", line_no + 1))?;
        let end_s = fields
            .next()
            .ok_or_else(|| anyhow!("line {}: missing end", line_no + 1))?;
        let name_field = fields.next();

        let start: u32 = start_s
            .parse()
            .with_context(|| format!("line {}: invalid start '{}'", line_no + 1, start_s))?;
        let end: u32 = end_s
            .parse()
            .with_context(|| format!("line {}: invalid end '{}'", line_no + 1, end_s))?;

        if start >= end {
            return Err(anyhow!(
                "line {}: start ({}) must be < end ({})",
                line_no + 1,
                start,
                end
            ));
        }

        let chrom_idx = table.intern_chrom(chrom);
        let name = match name_field {
            Some(n) if !n.is_empty() && n != "." => n.to_string(),
            _ => format!("{}:{}-{}", chrom, start, end),
        };

        table.chrom_idx.push(chrom_idx);
        table.start.push(start);
        table.end.push(end);
        table.name.push(name);
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE_BED: &str = "\
track name=consensus
# header
chr1\t100\t200\tpeak_1
chr1\t500\t700\tpeak_2
chr2\t10\t50
chr2\t100\t200\t.
";

    #[test]
    fn parses_bed_with_names() {
        let t = read_peaks_from(Cursor::new(SAMPLE_BED)).unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(t.n_chroms(), 2);
        assert_eq!(t.chrom_names, vec!["chr1", "chr2"]);
        assert_eq!(t.name, vec![
            "peak_1".to_string(),
            "peak_2".to_string(),
            "chr2:10-50".to_string(),      // no name
            "chr2:100-200".to_string(),    // dot placeholder
        ]);
    }

    #[test]
    fn rejects_zero_length_peak() {
        let bad = "chr1\t100\t100\tz\n";
        let err = read_peaks_from(Cursor::new(bad)).unwrap_err();
        assert!(err.to_string().contains("must be <"));
    }

    #[test]
    fn align_chroms_to_maps_correctly() {
        let t = read_peaks_from(Cursor::new(SAMPLE_BED)).unwrap();
        // Reference has chroms in different order than peaks
        let reference = vec!["chr2".to_string(), "chrX".to_string(), "chr1".to_string()];
        let aligned = t.align_chroms_to(&reference);
        // Peaks: chr1, chr1, chr2, chr2
        assert_eq!(aligned, vec![Some(2), Some(2), Some(0), Some(0)]);
    }

    #[test]
    fn align_chroms_to_returns_none_for_missing() {
        let t = read_peaks_from(Cursor::new(SAMPLE_BED)).unwrap();
        let reference = vec!["chr3".to_string()];
        let aligned = t.align_chroms_to(&reference);
        assert_eq!(aligned, vec![None, None, None, None]);
    }

    #[test]
    fn align_chroms_handles_ucsc_vs_ensembl_convention() {
        // BED has `chr1`/`chr2` (UCSC); reference uses Ensembl `1`/`2`.
        let t = read_peaks_from(Cursor::new(SAMPLE_BED)).unwrap();
        let reference = vec!["1".to_string(), "2".to_string()];
        let aligned = t.align_chroms_to(&reference);
        // Peaks: chr1, chr1, chr2, chr2 → 0, 0, 1, 1 after normalisation
        assert_eq!(aligned, vec![Some(0), Some(0), Some(1), Some(1)]);
    }

    #[test]
    fn align_chroms_handles_reverse_convention() {
        // BED has Ensembl `1`/`2`; reference has UCSC `chr1`/`chr2`.
        let bed = "1\t100\t200\n2\t300\t400\n";
        let t = read_peaks_from(Cursor::new(bed)).unwrap();
        let reference = vec!["chr1".to_string(), "chr2".to_string()];
        let aligned = t.align_chroms_to(&reference);
        assert_eq!(aligned, vec![Some(0), Some(1)]);
    }

    #[test]
    fn align_chroms_collapses_mitochondrial_aliases() {
        let bed = "chrM\t1\t100\n";
        let t = read_peaks_from(Cursor::new(bed)).unwrap();
        let reference = vec!["MT".to_string()];
        let aligned = t.align_chroms_to(&reference);
        assert_eq!(aligned, vec![Some(0)]);
    }

    #[test]
    fn normalise_chrom_strips_prefix_and_canonicalises_mt() {
        use super::normalise_chrom;
        assert_eq!(normalise_chrom("chr1"), "1");
        assert_eq!(normalise_chrom("1"), "1");
        assert_eq!(normalise_chrom("CHR2"), "2");
        assert_eq!(normalise_chrom("chrX"), "X");
        assert_eq!(normalise_chrom("chrM"), "MT");
        assert_eq!(normalise_chrom("chrMT"), "MT");
        assert_eq!(normalise_chrom("MT"), "MT");
        assert_eq!(normalise_chrom(" chr1 "), "1");
    }

    #[test]
    fn skips_track_and_comment_lines() {
        let input = "track type=bed\n# hello\n\nchr1\t1\t2\n";
        let t = read_peaks_from(Cursor::new(input)).unwrap();
        assert_eq!(t.len(), 1);
    }
}
