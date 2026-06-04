"""Static guardrails for Python compute hot paths in the package."""
from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "python" / "rustscenic"

HOT_PATH_PATTERNS = (
    ".groupby(",
    ".merge(",
    "pd.concat(",
    ".concat(",
    ".assign(",
    ".intersection(",
    ".sort_values(",
    ".nlargest(",
    ".iterrows(",
    ".itertuples(",
    ".apply(",
    ".agg(",
    ".pivot(",
    ".melt(",
    ".explode(",
    ".drop_duplicates(",
    ".duplicated(",
    ".value_counts(",
    ".notna(",
    ".todense(",
    ".toarray(",
    ".where(",
    ".min()",
    ".max()",
    ".sum()",
    ".sum(axis=",
    "np.where(",
    "np.argsort(",
    "np.argpartition(",
    "np.bincount(",
    "np.searchsorted(",
    "np.corrcoef(",
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
    ("pipeline.py", "pd.concat("): "final AnnData obs attachment and h5ad IO only",
    ("pipeline.py", ".concat("): "final AnnData obs attachment and h5ad IO only",
    ("_gene_resolution.py", ".max()"): "public diagnostic helper only",
    ("_gene_resolution.py", "warn_if_likely_unnormalized("): "public diagnostic helper only",
}


def scan_python_hot_paths(package_dir: Path = DEFAULT_PACKAGE) -> list[str]:
    """Return package lines that look like scale-sensitive Python compute."""
    if not package_dir.exists():
        return [f"missing package directory: {package_dir}"]

    violations: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        rel = path.relative_to(package_dir).as_posix()
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
                break
    return violations


def hot_path_state(package_dir: Path = DEFAULT_PACKAGE) -> dict[str, Any]:
    """Return a JSON-serialisable scan result for benchmark preflight."""
    violations = scan_python_hot_paths(package_dir)
    return {
        "package_dir": str(package_dir),
        "exists": package_dir.exists(),
        "ok": not violations,
        "violation_count": len(violations),
        "violations": violations,
        "allowed_hit_count": len(ALLOWED_HITS),
        "pattern_count": len(HOT_PATH_PATTERNS),
    }
