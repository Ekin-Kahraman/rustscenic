"""Re-run the issue #69 GRN and signed-regulon path on its public 10x data.

The issue did not include its post-QC H5AD, cell list, QC thresholds, or TF
list.  This runner therefore starts from the byte-identifiable 10x filtered
feature matrix and applies a fully recorded, conventional QC transform.  It
is a direct same-source-dataset validation, not a claim to reproduce the
unavailable 14,039 x 27,178 derived matrix byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc

import rustscenic
import rustscenic.aucell
import rustscenic.grn


OFFICIAL_10X_MD5 = "e0edcb64500137a3fdd4ab83ab791640"
EXPECTED_B_CELL_TFS = ("POU2F2", "PAX5", "MEF2B", "SPIB", "EBF1", "BCL11A")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_gb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return peak / divisor


def _stage(timings: dict[str, float], name: str, function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    timings[name] = time.perf_counter() - start
    return result


def _prepare_rna(path: Path) -> tuple[Any, dict[str, Any]]:
    rna = sc.read_10x_h5(path, gex_only=True)
    raw_shape = [int(value) for value in rna.shape]
    # These thresholds are explicit because the issue's own transform was not
    # supplied. Duplicate symbols are deliberately retained here: RustScenic
    # sums them deterministically at the GRN and correlation boundaries.
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    qc_shape = [int(value) for value in rna.shape]
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    return rna, {
        "raw_rna_shape": raw_shape,
        "qc_rna_shape": qc_shape,
        "cell_filter": {"min_genes": 200},
        "gene_filter": {"min_cells": 3},
        "normalisation": {"method": "total_log1p", "target_sum": 10000.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("tf_list", type=Path)
    parser.add_argument("output_parquet", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--early-stop-mode", choices=("arboreto", "legacy_inbag"), default="arboreto"
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    input_md5 = _digest(args.input_h5, "md5")
    if input_md5 != OFFICIAL_10X_MD5:
        raise SystemExit(
            "input is not the official issue #69 10x filtered matrix: "
            f"expected MD5 {OFFICIAL_10X_MD5}, got {input_md5}"
        )

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    rna, transform = _stage(timings, "data_preparation", _prepare_rna, args.input_h5)
    tf_names = [
        value.strip()
        for value in args.tf_list.read_text().splitlines()
        if value.strip()
    ]

    grn = _stage(
        timings,
        "grn",
        rustscenic.grn.infer,
        rna,
        tf_names=tf_names,
        n_estimators=args.n_estimators,
        seed=args.seed,
        early_stop_mode=args.early_stop_mode,
        verbose=True,
    )
    signed = _stage(
        timings,
        "correlation",
        rustscenic.grn.add_correlation,
        grn,
        rna,
        rho_threshold=0.03,
        mask_dropouts=False,
    )
    regulons = _stage(
        timings,
        "regulons",
        rustscenic.grn.build_regulons,
        signed,
        top_targets_per_tf=50,
        min_targets=10,
        include_repressors=True,
    )
    auc = _stage(timings, "aucell", rustscenic.aucell.score, rna, regulons)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    signed.to_parquet(args.output_parquet, index=False)
    output_sha256 = _digest(args.output_parquet, "sha256")
    names = set(regulons)
    found_b_cell = {
        tf: sorted(name for name in names if name.startswith(f"{tf}_"))
        for tf in EXPECTED_B_CELL_TFS
    }
    result = {
        "schema_version": 1,
        "validation": "issue_69_same_10x_source_signed_grn",
        "status": "ok",
        "source": {
            "dataset": "10x fresh-frozen lymph node with B-cell lymphoma 14k",
            "cell_ranger_arc_version": "1.0.0",
            "input_h5_md5": input_md5,
            "input_h5_sha256": _digest(args.input_h5, "sha256"),
            "exact_issue_post_qc_matrix_available": False,
            "reason": "issue omitted its H5AD, cell list, QC thresholds, and TF list",
        },
        "transform": transform,
        "configuration": {
            "tf_list_sha256": _digest(args.tf_list, "sha256"),
            "tf_list_entries": len(tf_names),
            "n_estimators": args.n_estimators,
            "early_stop_mode": args.early_stop_mode,
            "early_stop_window": 25,
            "seed": args.seed,
            "rho_threshold": 0.03,
            "mask_dropouts": False,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "outputs": {
            "grn_edges": int(len(grn)),
            "signed_edges": int(len(signed)),
            "signed_parquet_sha256": output_sha256,
            "polarity": dict(signed.attrs["correlation"]),
            "regulons": len(regulons),
            "activator_regulons": sum(name.endswith("_activator") for name in names),
            "repressor_regulons": sum(name.endswith("_repressor") for name in names),
            "aucell_shape": [int(value) for value in auc.shape],
            "expected_b_cell_tfs": found_b_cell,
        },
        "grn_fit": dict(grn.attrs["grn_fit"]),
        "timings_seconds": {
            **{name: round(value, 6) for name, value in timings.items()},
            "total": round(time.perf_counter() - total_start, 6),
        },
        "peak_rss_gb": round(_peak_rss_gb(), 6),
        "environment": {
            "rustscenic": rustscenic.__version__,
            "python": platform.python_version(),
            "scanpy": version("scanpy"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
