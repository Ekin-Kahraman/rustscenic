"""Run rustscenic.grn.infer on the SAME pbmc3k.h5ad + allTFs_hg38.txt that
the reference Docker container uses, with the SAME seed. Output is a
parquet identical in schema to arboreto's grnboost2 output.

Argv: <h5ad_path> <tf_list_path> <output_parquet> <output_meta_json>
"""
from __future__ import annotations
import json
import sys
import time
import resource
from pathlib import Path
import anndata as ad
import pandas as pd
import numpy as np
import rustscenic
import rustscenic.grn


def main(h5ad: str, tfs: str, out_parquet: str, out_meta: str) -> int:
    adata = ad.read_h5ad(h5ad)
    tf_list = [t.strip() for t in Path(tfs).read_text().splitlines() if t.strip()]
    tf_list = [t for t in tf_list if t in adata.var_names]

    print(f"adata shape: {adata.shape}, TFs in expression: {len(tf_list)}", flush=True)

    t0 = time.monotonic()
    grn = rustscenic.grn.infer(
        adata,
        tf_names=tf_list,
        n_estimators=5000,           # MATCH reference (arboreto default)
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
        "seed": 777,
        "input_h5ad": h5ad,
        "input_tfs": tfs,
    }
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("usage: run_rustscenic_grn_pbmc3k.py <h5ad> <tfs> <out_parquet> <out_meta>")
    sys.exit(main(*sys.argv[1:]))
