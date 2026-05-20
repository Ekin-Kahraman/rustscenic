"""Compare stored RustScenic and SCENIC+ benchmark output signatures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    signature = record.get("output_signature")
    if not signature:
        raise ValueError(f"{path} has no output_signature; rerun with --save-signatures")
    return record


def _keys(signature: dict[str, Any], field: str) -> set[str]:
    return {str(row["key"]) for row in signature.get(field, [])}


def _jaccard(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    return round(len(a & b) / len(a | b), 6)


def _overlap(a: set[str], b: set[str]) -> dict[str, Any]:
    return {
        "rustscenic_count": len(a),
        "scenicplus_count": len(b),
        "intersection": len(a & b),
        "union": len(a | b),
        "jaccard": _jaccard(a, b),
        "rustscenic_recall": round(len(a & b) / len(a), 6) if a else None,
        "scenicplus_recall": round(len(a & b) / len(b), 6) if b else None,
    }


def _cell_index(cells: list[str]) -> dict[str, int]:
    return {cell: i for i, cell in enumerate(cells)}


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    x = x[ok]
    y = y[ok]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 6)


def _auc_correlations(
    rust_sig: dict[str, Any],
    scenic_sig: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    rust_cells = [str(c) for c in rust_sig.get("cells", [])]
    scenic_cells = [str(c) for c in scenic_sig.get("cells", [])]
    common_cells = sorted(set(rust_cells) & set(scenic_cells))
    rust_by_tf = rust_sig.get(field, {})
    scenic_by_tf = scenic_sig.get(field, {})
    common_tfs = sorted(set(rust_by_tf) & set(scenic_by_tf))
    if not common_cells or not common_tfs:
        return {
            "common_tfs": len(common_tfs),
            "common_cells": len(common_cells),
            "per_tf": {},
            "mean_pearson": None,
            "median_pearson": None,
        }

    rust_idx = _cell_index(rust_cells)
    scenic_idx = _cell_index(scenic_cells)
    rust_positions = [rust_idx[c] for c in common_cells]
    scenic_positions = [scenic_idx[c] for c in common_cells]

    per_tf: dict[str, float | None] = {}
    valid = []
    for tf in common_tfs:
        rust_values = np.asarray(rust_by_tf[tf], dtype=np.float64)[rust_positions]
        scenic_values = np.asarray(scenic_by_tf[tf], dtype=np.float64)[scenic_positions]
        corr = _pearson(rust_values, scenic_values)
        per_tf[tf] = corr
        if corr is not None:
            valid.append(corr)
    return {
        "common_tfs": len(common_tfs),
        "common_cells": len(common_cells),
        "per_tf": per_tf,
        "mean_pearson": round(float(np.mean(valid)), 6) if valid else None,
        "median_pearson": round(float(np.median(valid)), 6) if valid else None,
    }


def compare(rust_record: dict[str, Any], scenic_record: dict[str, Any]) -> dict[str, Any]:
    rust_sig = rust_record["output_signature"]
    scenic_sig = scenic_record["output_signature"]
    rust_tfs = set(rust_sig.get("eregulon_tfs", []))
    scenic_tfs = set(scenic_sig.get("eregulon_tfs", []))
    return {
        "benchmark": rust_record.get("benchmark"),
        "rust_label": rust_record.get("label"),
        "scenicplus_label": scenic_record.get("label"),
        "dataset": rust_record.get("settings", {}).get("dataset"),
        "settings": {
            key: rust_record.get("settings", {}).get(key)
            for key in [
                "n_cells",
                "n_genes",
                "n_peaks",
                "n_tfs",
                "search_space_pairs",
                "seed",
                "signature_top_n",
            ]
        },
        "tf_to_gene_top_overlap": _overlap(
            _keys(rust_sig, "tf_to_gene_top_edges"),
            _keys(scenic_sig, "tf_to_gene_top_edges"),
        ),
        "region_to_gene_top_overlap": _overlap(
            _keys(rust_sig, "region_to_gene_top_edges"),
            _keys(scenic_sig, "region_to_gene_top_edges"),
        ),
        "eregulon_tf_overlap": _overlap(rust_tfs, scenic_tfs),
        "eregulon_edge_overlap": _overlap(
            _keys(rust_sig, "eregulon_edges"),
            _keys(scenic_sig, "eregulon_edges"),
        ),
        "gene_auc_by_tf_pearson": _auc_correlations(
            rust_sig, scenic_sig, "gene_auc_by_tf"
        ),
        "region_auc_by_tf_pearson": _auc_correlations(
            rust_sig, scenic_sig, "region_auc_by_tf"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--scenicplus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = compare(_load(args.rust), _load(args.scenicplus))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
