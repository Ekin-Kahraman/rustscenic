//! Fragment-file parser.
//!
//! 10x cellranger emits `fragments.tsv.gz` with tab-separated columns:
//!
//! ```text
//! chrom  start  end  barcode  count
//! ```
//!
//! Comments (`#`) and empty lines are skipped. `chrom` is a string,
//! `start`/`end` are 0-based half-open integers, `barcode` is the
//! cell barcode (often `ATCG.*-1`), `count` is the PCR duplicate count
//! for the fragment.
//!
//! We parse into a columnar `FragmentTable` so downstream passes
//! (per-barcode counts, peak intersection) stay cache-friendly.

use anyhow::{anyhow, Context, Result};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A single ATAC fragment record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    pub chrom: String,
    pub start: u32,
    pub end: u32,
    pub barcode: String,
    pub count: u32,
}

/// Columnar fragment table — one Vec per column.
///
/// `chrom` is stored as an interned index into `chrom_names`, so
/// downstream joins don't re-compare strings per row.
#[derive(Debug, Default)]
pub struct FragmentTable {
    pub chrom_idx: Vec<u32>,
    pub chrom_names: Vec<String>,
    pub start: Vec<u32>,
    pub end: Vec<u32>,
    pub barcode_idx: Vec<u32>,
    pub barcode_names: Vec<String>,
    pub count: Vec<u32>,
}

impl FragmentTable {
    /// Number of fragment records.
    pub fn len(&self) -> usize {
        self.start.len()
    }

    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    /// Number of distinct chromosomes observed.
    pub fn n_chroms(&self) -> usize {
        self.chrom_names.len()
    }

    /// Number of distinct barcodes observed.
    pub fn n_barcodes(&self) -> usize {
        self.barcode_names.len()
    }

    fn intern_chrom(&mut self, name: &str) -> u32 {
        // Linear scan — fine for the ~25 chroms we see in practice.
        for (i, n) in self.chrom_names.iter().enumerate() {
            if n == name {
                return i as u32;
            }
        }
        let idx = self.chrom_names.len() as u32;
        self.chrom_names.push(name.to_string());
        idx
    }

    fn intern_barcode(
        &mut self,
        name: &str,
        lookup: &mut std::collections::HashMap<String, u32>,
    ) -> u32 {
        if let Some(&idx) = lookup.get(name) {
            return idx;
        }
        let idx = self.barcode_names.len() as u32;
        self.barcode_names.push(name.to_string());
        lookup.insert(name.to_string(), idx);
        idx
    }
}

/// Parse a 10x fragments BED file (plain or gzipped) into a `FragmentTable`.
///
/// Accepts `.tsv`, `.tsv.gz`, `.bed`, `.bed.gz`. Detection by extension.
pub fn read_fragments<P: AsRef<Path>>(path: P) -> Result<FragmentTable> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
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
    read_fragments_from(reader)
}

/// Parse from any `Read`. Separated for testability.
pub fn read_fragments_from<R: Read>(reader: R) -> Result<FragmentTable> {
    let buffered = BufReader::with_capacity(1 << 20, reader);
    let mut table = FragmentTable::default();
    let mut barcode_lookup = std::collections::HashMap::with_capacity(1 << 16);

    for (line_no, line) in buffered.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_no + 1))?;
        if line.is_empty() || line.starts_with('#') {
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
        let barcode = fields
            .next()
            .ok_or_else(|| anyhow!("line {}: missing barcode", line_no + 1))?;
        let count_s = fields.next().unwrap_or("1");

        let start: u32 = start_s
            .parse()
            .with_context(|| format!("line {}: invalid start '{}'", line_no + 1, start_s))?;
        let end: u32 = end_s
            .parse()
            .with_context(|| format!("line {}: invalid end '{}'", line_no + 1, end_s))?;
        let count: u32 = count_s
            .parse()
            .with_context(|| format!("line {}: invalid count '{}'", line_no + 1, count_s))?;

        if start >= end {
            return Err(anyhow!(
                "line {}: start ({}) must be < end ({})",
                line_no + 1,
                start,
                end
            ));
        }

        let chrom_idx = table.intern_chrom(chrom);
        let barcode_idx = table.intern_barcode(barcode, &mut barcode_lookup);

        table.chrom_idx.push(chrom_idx);
        table.start.push(start);
        table.end.push(end);
        table.barcode_idx.push(barcode_idx);
        table.count.push(count);
    }

    Ok(table)
}

/// Fragments per barcode (unique-fragment count, not summed `count` column).
///
/// Returns a Vec parallel to `table.barcode_names`.
pub fn fragments_per_barcode(table: &FragmentTable) -> Vec<u32> {
    let mut out = vec![0_u32; table.n_barcodes()];
    for &b in &table.barcode_idx {
        out[b as usize] += 1;
    }
    out
}

/// Summed PCR-duplicate counts per barcode (total fragments including dups).
pub fn total_counts_per_barcode(table: &FragmentTable) -> Vec<u32> {
    let mut out = vec![0_u32; table.n_barcodes()];
    for (i, &b) in table.barcode_idx.iter().enumerate() {
        out[b as usize] += table.count[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SAMPLE: &str = "\
#comment
chr1\t100\t200\tAAA-1\t1
chr1\t150\t260\tAAA-1\t2
chr1\t500\t600\tBBB-1\t1
chr2\t10\t50\tAAA-1\t1
";

    #[test]
    fn parses_basic_bed() {
        let t = read_fragments_from(Cursor::new(SAMPLE)).unwrap();
        assert_eq!(t.len(), 4);
        assert_eq!(t.n_chroms(), 2);
        assert_eq!(t.n_barcodes(), 2);
        assert_eq!(t.chrom_names, vec!["chr1", "chr2"]);
        assert_eq!(t.barcode_names, vec!["AAA-1", "BBB-1"]);
        assert_eq!(t.start, vec![100, 150, 500, 10]);
        assert_eq!(t.end, vec![200, 260, 600, 50]);
        assert_eq!(t.count, vec![1, 2, 1, 1]);
    }

    #[test]
    fn rejects_start_ge_end() {
        let bad = "chr1\t200\t100\tA-1\t1\n";
        let err = read_fragments_from(Cursor::new(bad)).unwrap_err();
        assert!(err.to_string().contains("must be <"));
    }

    #[test]
    fn rejects_bad_integer() {
        let bad = "chr1\tabc\t100\tA-1\t1\n";
        let err = read_fragments_from(Cursor::new(bad)).unwrap_err();
        assert!(err.to_string().contains("invalid start"));
    }

    #[test]
    fn skips_comments_and_blank_lines() {
        let input = "# header\n\nchr1\t1\t2\tA-1\t1\n# mid-comment\nchr1\t3\t4\tA-1\t1\n";
        let t = read_fragments_from(Cursor::new(input)).unwrap();
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn fragments_per_barcode_counts_rows() {
        let t = read_fragments_from(Cursor::new(SAMPLE)).unwrap();
        let fpb = fragments_per_barcode(&t);
        // AAA-1 = 3 rows, BBB-1 = 1 row
        assert_eq!(fpb, vec![3, 1]);
    }

    #[test]
    fn total_counts_sums_dup_column() {
        let t = read_fragments_from(Cursor::new(SAMPLE)).unwrap();
        let totals = total_counts_per_barcode(&t);
        // AAA-1: 1+2+1=4; BBB-1: 1
        assert_eq!(totals, vec![4, 1]);
    }

    #[test]
    fn missing_count_column_defaults_to_one() {
        let input = "chr1\t1\t2\tA-1\n";
        let t = read_fragments_from(Cursor::new(input)).unwrap();
        assert_eq!(t.count, vec![1]);
    }
}
