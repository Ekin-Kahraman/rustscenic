"""Run rustscenic.grn.infer on the SAME pbmc3k.h5ad + allTFs_hg38.txt that
the reference Docker container uses, with the SAME seed. Output is a
parquet identical in schema to arboreto's grnboost2 output.

Argv: <h5ad_path> <tf_list_path> <output_parquet> <output_meta_json>
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import resource
from pathlib import Path
import anndata as ad
import pandas as pd
import numpy as np
import rustscenic
import rustscenic.grn


def main(
    h5ad: str,
    tfs: str,
    out_parquet: str,
    out_meta: str,
    *,
    early_stop_mode: str = "arboreto",
) -> int:
    adata = ad.read_h5ad(h5ad)
    tf_list = [t.strip() for t in Path(tfs).read_text().splitlines() if t.strip()]
    tf_list = [t for t in tf_list if t in adata.var_names]

    print(f"adata shape: {adata.shape}, TFs in expression: {len(tf_list)}", flush=True)

    t0 = time.monotonic()
    grn = rustscenic.grn.infer(
        adata,
        tf_names=tf_list,
        n_estimators=5000,           # MATCH reference (arboreto default)
        early_stop_mode=early_stop_mode,
        seed=777,
        verbose=False,
    )
    wall = time.monotonic() - t0

    grn.to_parquet(out_parquet, index=False)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / (1024 ** 3) if peak_kb > 1e6 else peak_kb / (1024 ** 2)

    meta = {
        "rustscenic_version": rustscenic.__version__,
        "wall_clock_s": round(wall, 2),
        "peak_rss_gb": round(peak_gb, 2),
        "n_edges": int(len(grn)),
        "n_tfs_used": len(tf_list),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_estimators": 5000,
        "early_stop_mode": early_stop_mode,
        "grn_fit": dict(grn.attrs.get("grn_fit", {})),
        "seed": 777,
        "input_h5ad": Path(h5ad).name,
        "input_tfs": Path(tfs).name,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad")
    parser.add_argument("tfs")
    parser.add_argument("out_parquet")
    parser.add_argument("out_meta")
    parser.add_argument(
        "--early-stop-mode",
        choices=("arboreto", "legacy_inbag"),
        default="arboreto",
    )
    cli_args = parser.parse_args()
    sys.exit(
        main(
            cli_args.h5ad,
            cli_args.tfs,
            cli_args.out_parquet,
            cli_args.out_meta,
            early_stop_mode=cli_args.early_stop_mode,
        )
    )
