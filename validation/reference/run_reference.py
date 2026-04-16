"""Generate reference baseline outputs from pinned pyscenic."""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import pandas as pd
from arboreto.algo import grnboost2


def run_grn(expression_path: str, tfs_path: str, output: str, seed: int = 777) -> None:
    adata = ad.read_h5ad(expression_path)
    tf_names = Path(tfs_path).read_text().strip().splitlines()
    tf_names = [t for t in tf_names if t in adata.var_names]

    expr = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    t0 = time.monotonic()
    adjacencies = grnboost2(expression_data=expr, tf_names=tf_names, seed=seed, verbose=False)
    wall = time.monotonic() - t0

    adjacencies.to_parquet(output)
    meta = {"wall_clock_s": wall, "n_edges": len(adjacencies), "n_tfs_used": len(tf_names),
            "n_cells": adata.n_obs, "n_genes": adata.n_vars, "seed": seed}
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
