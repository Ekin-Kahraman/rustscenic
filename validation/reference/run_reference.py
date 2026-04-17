"""Generate reference baseline outputs from pinned pyscenic.

Uses sync execution (client_or_address=None) — arboreto's Dask cluster path
crashes on every modern dask version, per our profiling audit 2026-04-16.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from arboreto.algo import grnboost2


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_grn(expression_path: str, tfs_path: str, output: str, seed: int) -> None:
    adata = ad.read_h5ad(expression_path)
    tf_names = Path(tfs_path).read_text().strip().splitlines()
    tf_names = [t for t in tf_names if t in adata.var_names]

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    t0 = time.monotonic()
    adjacencies = grnboost2(
        expression_data=expr,
        tf_names=tf_names,
        seed=seed,
        client_or_address=None,   # force sync path — Dask path is broken
        verbose=False,
    )
    wall = time.monotonic() - t0

    adjacencies.to_parquet(output)
    meta = {
        "wall_clock_s": wall,
        "n_edges": int(len(adjacencies)),
        "n_tfs_used": len(tf_names),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "seed": seed,
        "dataset_sha256": _sha256(Path(expression_path)),
        "reference_stack": {
            "pyscenic": "0.12.1",
            "arboreto": "0.1.6",
            "dask": "2024.1.1",
            "numpy": "1.26.4",
            "pandas": "2.1.4",
            "lightgbm": "4.6.0",
            "scanpy": "1.11.5",
        },
    }
    Path(output).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["grn", "aucell", "topics", "cistarget"])
    p.add_argument("--expression", default="/data/pbmc3k.h5ad")
    p.add_argument("--tfs", default="/data/allTFs_hg38.txt")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()

    if args.stage == "grn":
        run_grn(args.expression, args.tfs, args.output, args.seed)
    else:
        sys.exit(f"stage {args.stage} reference runner pending")
