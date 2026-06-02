from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "rustscenic"


HOT_PATH_PATTERNS = (
    ".groupby(",
    ".merge(",
    "pd.concat(",
    ".concat(",
    ".assign(",
    ".intersection(",
    ".sort_values(",
    ".nlargest(",
    ".todense(",
    ".toarray(",
    ".min()",
    ".max()",
    ".sum()",
    ".sum(axis=",
    "np.where(",
    "np.median(",
    "np.unique(",
    "len(set(",
    "Counter(",
    "numeric_rest = df.drop(",
    "peak_centers =",
    "row_idx = np.asarray(row_idx)",
    "row_idx = np.asarray(row_idx, dtype=np.intp)",
    "[row_idx]",
    "idx = np.asarray(row_ix)",
    "idx = np.asarray(row_ix, dtype=np.intp)",
    "np.asarray(row_ix)",
    "np.asarray(row_ix, dtype=np.intp)",
    "iloc[np.asarray(row_ix)]",
    "top_indices = np.asarray(top_indices_raw, dtype=np.intp)",
    "top_indices = np.asarray(top_indices_kernel(weights_arg, int(top_n)), dtype=np.intp)",
    "_pipeline_filter_cistarget_peak_rows(",
    "_pipeline_attribute_peak_values",
    "pipeline_attribute_peaks_to_cistarget_peak_values",
    "_sort_links_by_abs_corr",
    "cell_groups ==",
    "tfs_present =",
    "warn_if_likely_unnormalized(",
    " @ ",
    "sp.csr_matrix((data, (rows, cols))",
)

ALLOWED_HITS = {
    ("quickstart.py", ".nlargest("): "demo display only",
    ("quickstart.py", ".sum(axis="): "demo normalisation only",
    ("_gene_resolution.py", ".max()"): "public diagnostic helper only",
    ("_gene_resolution.py", "warn_if_likely_unnormalized("): "public diagnostic helper only",
    ("cistarget.py", "idx = np.asarray(row_ix)"): "extra-column public fallback after Rust pruning",
    ("cistarget.py", "np.asarray(row_ix)"): "extra-column public fallback after Rust pruning",
}


def test_package_hot_paths_do_not_reintroduce_pandas_table_work():
    """Keep scale-sensitive table operations out of Python package code.

    RustScenic can still use pandas for public input/output tables. The
    expensive joins, groupbys, concats, sorts and sparse densification paths
    should not creep back into package compute code now that the core stages
    are Rust-backed.
    """
    violations: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern in HOT_PATH_PATTERNS:
                if pattern not in stripped:
                    continue
                if (rel, pattern) in ALLOWED_HITS:
                    continue
                violations.append(f"{rel}:{line_no}: {stripped}")

    assert not violations, (
        "Python hot-path table work detected. Move the operation to Rust or "
        "add a narrowly justified allowlist entry:\n" + "\n".join(violations)
    )
