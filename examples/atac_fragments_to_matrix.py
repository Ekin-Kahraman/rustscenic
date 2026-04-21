"""End-to-end example: fragments.tsv.gz + peaks.bed -> cells × peaks AnnData.

Demonstrates rustscenic.preproc on synthetic data you can verify by eye.
No external downloads, no paths to configure — runs standalone.

Workflow mirrors what you'd do on real 10x multiome output:

    1. Write tiny fragments.tsv.gz + peaks.bed to a temp dir
    2. Call rustscenic.preproc.fragments_to_matrix
    3. Verify shape, per-cell QC, and the expected counts

Runtime: <1 second.
"""
from __future__ import annotations

import gzip
import tempfile
import time
from pathlib import Path


def write_toy_data(tmp: Path) -> tuple[Path, Path]:
    """Emit a minimal fragments.tsv.gz and peaks.bed with known ground truth.

    3 cells × 3 peaks, hand-computed expected matrix:

        cell         peak1_chr1  peak2_chr1  peak3_chr2
        CELL_A            2           1           0
        CELL_B            0           1           0
        CELL_C            0           0           1
    """
    fragments = tmp / "fragments.tsv.gz"
    peaks = tmp / "peaks.bed"
    lines = [
        # chrom  start  end  barcode   count
        "chr1\t150\t250\tCELL_A-1\t1",  # overlaps peak1 (100-300)
        "chr1\t180\t290\tCELL_A-1\t1",  # overlaps peak1 (100-300)
        "chr1\t600\t700\tCELL_A-1\t1",  # overlaps peak2 (500-800)
        "chr1\t620\t680\tCELL_B-1\t1",  # overlaps peak2 (500-800)
        "chr2\t50\t150\tCELL_C-1\t1",   # overlaps peak3 (0-200)
        "chr3\t1\t100\tCELL_C-1\t1",    # chr3 not in peaks — no overlap
    ]
    with gzip.open(fragments, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    peaks.write_text(
        "chr1\t100\t300\tpeak1\n"
        "chr1\t500\t800\tpeak2\n"
        "chr2\t0\t200\tpeak3\n"
    )
    return fragments, peaks


def main() -> int:
    import rustscenic.preproc

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fragments_path, peaks_path = write_toy_data(tmp)
        print(f"toy fragments: {fragments_path.name} (gzipped)")
        print(f"toy peaks:     {peaks_path.name}")

        t0 = time.perf_counter()
        adata = rustscenic.preproc.fragments_to_matrix(
            fragments_path, peaks_path
        )
        elapsed = time.perf_counter() - t0
        print(f"\nbuilt matrix in {elapsed*1e3:.1f} ms")

        print(f"\nshape:        {adata.shape}  (cells x peaks)")
        print(f"cells:        {list(adata.obs_names)}")
        print(f"peaks:        {list(adata.var_names)}")
        print(f"\nper-cell QC (.obs):")
        print(adata.obs.to_string())
        print(f"\ncount matrix (dense view):")
        print(adata.X.toarray())

        # Ground truth from the docstring
        expected = [
            [2, 1, 0],  # CELL_A: peak1 x2, peak2 x1
            [0, 1, 0],  # CELL_B: peak2 x1
            [0, 0, 1],  # CELL_C: peak3 x1
        ]
        actual = adata.X.toarray().tolist()
        # Cell ordering follows insertion order in fragments file,
        # so CELL_A first row, CELL_B second, CELL_C third.
        assert actual == expected, (
            f"matrix mismatch\nexpected: {expected}\nactual:   {actual}"
        )
        # CELL_C has 2 fragments total (chr2 peak3 + chr3 nowhere);
        # only the first overlaps a peak, but both count toward QC.
        assert adata.obs.loc["CELL_C-1", "fragments_per_cell"] == 2
        print("\nok — matrix and QC metrics match ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
