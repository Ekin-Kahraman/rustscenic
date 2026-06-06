"""Static guardrails for Python compute hot paths in the package."""
from __future__ import annotations

import argparse
import json
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
    ".take(",
    ".loc[",
    ".iloc[",
    ".todense(",
    ".toarray(",
    ".to_numpy(",
    ".where(",
    ".min()",
    ".max()",
    ".sum()",
    ".sum(axis=",
    "np.where(",
    "np.argsort(",
    "np.argpartition(",
    "np.bincount(",
    "np.column_stack(",
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
    (
        "quickstart.py",
        ".nlargest(",
        'print(grn.nlargest(10, "importance").to_string(index=False))',
    ): "demo display only",
    (
        "quickstart.py",
        ".sum(axis=",
        "libsize = X.sum(axis=1, keepdims=True)",
    ): "demo normalisation only",
    (
        "pipeline.py",
        "pd.concat(",
        "adata_rna.obs = pd.concat([obs, auc_obs], axis=1, copy=False)",
    ): "final AnnData obs attachment and h5ad IO only",
    (
        "cistarget.py",
        ".take(",
        "out = enriched.take(row_ix, axis=0).reset_index(drop=True)",
    ): "Rust-returned row-index projection boundary",
    (
        "enhancer.py",
        ".take(",
        "out = peak_coords.take(row_ix, axis=0)",
    ): "Rust-returned peak-coordinate projection boundary",
    (
        "enhancer.py",
        ".iloc[",
        "genes_in_rna = genes.iloc[matched_gene_rows].reset_index(drop=True)",
    ): "Rust-returned gene-coordinate projection boundary",
    (
        "pipeline.py",
        ".loc[",
        "df = df.loc[:, keep]",
    ): "caller-supplied DataFrame ranking projection after Rust column selection; file-backed HPC projection reads selected columns directly",
    (
        "pipeline.py",
        ".iloc[",
        "return all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes.iloc[1:])",
    ): "metadata-only dtype scan for first-column motif export detection",
    (
        "_gene_resolution.py",
        ".max()",
        "max_val = float(X.max())",
    ): "public diagnostic helper only",
    (
        "_gene_resolution.py",
        ".max()",
        "max_val = float(arr.max())",
    ): "public diagnostic helper only",
    (
        "_gene_resolution.py",
        "warn_if_likely_unnormalized(",
        "Plus ``warn_if_likely_unnormalized(X)`` - flags raw-count input that",
    ): "public diagnostic helper docs only",
    (
        "_gene_resolution.py",
        "warn_if_likely_unnormalized(",
        "def warn_if_likely_unnormalized(X, *, max_threshold: float = 50.0, stacklevel: int = 2) -> None:",
    ): "public diagnostic helper only",
    (
        "_stage_utils.py",
        ".to_numpy(",
        "arr = values.to_numpy(copy=False)",
    ): "central PyO3 boundary conversion helper",
    (
        "cistarget.py",
        ".to_numpy(",
        "rankings.to_numpy(copy=False),",
    ): "ranking matrix handoff to Rust cistarget/pruning kernels",
    (
        "cistarget.py",
        ".to_numpy(",
        "ranking_values=rankings.to_numpy(copy=False),",
    ): "ranking matrix handoff to Rust motif-pruning kernel",
    (
        "cistarget.py",
        ".to_numpy(",
        "auc_values = auc.to_numpy(copy=False)",
    ): "numeric cistarget score handoff to Rust motif-pruning kernel",
    (
        "cistarget.py",
        ".to_numpy(",
        "nes_values = None if nes is None else nes.to_numpy(copy=False)",
    ): "numeric cistarget score handoff to Rust motif-pruning kernel",
    (
        "cistarget.py",
        ".to_numpy(",
        "values = rankings.to_numpy(copy=False)",
    ): "ranking matrix dtype dispatch before Rust cistarget kernel",
    (
        "enhancer.py",
        ".to_numpy(",
        'peak_starts = peaks["start"].to_numpy(dtype=np.int64, copy=False)',
    ): "peak-coordinate vector handoff to Rust enhancer kernel",
    (
        "enhancer.py",
        ".to_numpy(",
        'peak_ends = peaks["end"].to_numpy(dtype=np.int64, copy=False)',
    ): "peak-coordinate vector handoff to Rust enhancer kernel",
    (
        "enhancer.py",
        ".to_numpy(",
        'gene_tss_raw = genes_in_rna["tss"].to_numpy(dtype=np.int64)',
    ): "gene-coordinate vector handoff to Rust enhancer kernel",
    (
        "enhancer.py",
        ".to_numpy(",
        'gene_names = genes_in_rna["gene"].to_numpy(dtype=object)[gene_order]',
    ): "Rust-returned gene-order projection boundary",
    (
        "eregulon.py",
        ".to_numpy(",
        'enhancer_links["correlation"].to_numpy(dtype=np.float32, copy=False),',
    ): "enhancer-link score vector handoff to Rust eRegulon kernel",
    (
        "eregulon.py",
        ".to_numpy(",
        "arr = values.to_numpy(copy=False)",
    ): "numeric cistarget score handoff to Rust eRegulon kernel",
    (
        "pipeline.py",
        ".to_numpy(",
        "ranking_values = rankings_df.to_numpy(copy=False)",
    ): "ranking matrix dtype dispatch before Rust cistarget kernel",
    (
        "pipeline.py",
        ".to_numpy(",
        'grn["importance"].to_numpy(dtype=np.float64, copy=False),',
    ): "GRN importance vector handoff to Rust candidate-regulon kernel",
    (
        "pipeline.py",
        ".to_numpy(",
        "table.column(str(col)).combine_chunks().to_numpy(zero_copy_only=False)",
    ): "file-backed ranking column materialisation after Rust column projection",
    (
        "pipeline.py",
        ".to_numpy(",
        "ranking_values = region_rankings.to_numpy(copy=False)",
    ): "region ranking matrix handoff to Rust cistarget kernel",
    (
        "pipeline.py",
        ".to_numpy(",
        "arr = values.to_numpy(copy=False)",
    ): "numeric cistarget score handoff to Rust region/eRegulon kernels",
    (
        "pipeline.py",
        ".to_numpy(",
        "rankings.to_numpy(copy=False)",
    ): "ranking matrix handoff to Rust motif-pruning kernel",
    (
        "pipeline.py",
        ".to_numpy(",
        "auc.to_numpy(copy=False),",
    ): "final AUCell output attachment boundary",
    (
        "preproc.py",
        ".to_numpy(",
        "clusters = cluster_per_barcode.to_numpy(dtype=np.int64)",
    ): "cluster label vector handoff to Rust aggregation kernel",
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
                if (rel, pattern, stripped) in ALLOWED_HITS:
                    break
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that scale-sensitive RustScenic package paths remain "
            "outside Python."
        )
    )
    parser.add_argument(
        "package_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PACKAGE,
        help="Package directory to scan. Defaults to python/rustscenic.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full scan state as JSON.",
    )
    args = parser.parse_args(argv)

    state = hot_path_state(args.package_dir)
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True))
    elif state["ok"]:
        print(
            f"ok: no Python hot-path violations in {state['package_dir']} "
            f"({state['pattern_count']} patterns)"
        )
    else:
        print(
            f"fail: {state['violation_count']} Python hot-path violations in "
            f"{state['package_dir']}"
        )
        for violation in state["violations"]:
            print(violation)
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
