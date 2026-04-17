"""rustscenic CLI entry point.

Usage:
    rustscenic grn --expression data.h5ad --tfs tfs.txt --output grn.parquet [--seed 777]
    rustscenic --version

The CLI is a thin Python wrapper around `rustscenic.grn.infer`. Input formats
mirror arboreto's conventions so existing pipelines can swap one call for the
other. Output parquet matches the arboreto.grnboost2 schema exactly:
    (TF: str, target: str, importance: f32)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_grn(args: argparse.Namespace) -> int:
    import anndata as ad  # lazy import to keep --help fast
    import pandas as pd
    from . import grn as rs_grn
    from . import __version__

    expr_path = Path(args.expression)
    if not expr_path.exists():
        print(f"error: expression file not found: {expr_path}", file=sys.stderr)
        return 2

    suffix = expr_path.suffix.lower()
    if suffix == ".h5ad":
        adata = ad.read_h5ad(expr_path)
        gene_names = list(adata.var_names)
        expression = adata
    elif suffix in (".tsv", ".csv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(expr_path, sep=sep, index_col=0)
        gene_names = list(df.columns)
        expression = df
    else:
        print(f"error: unsupported expression format {suffix}. Use .h5ad, .tsv, or .csv",
              file=sys.stderr)
        return 2

    tfs_path = Path(args.tfs)
    if not tfs_path.exists():
        print(f"error: tf list not found: {tfs_path}", file=sys.stderr)
        return 2
    tfs = rs_grn.load_tfs(tfs_path)
    tfs_in = [t for t in tfs if t in set(gene_names)]
    if len(tfs_in) == 0:
        print(f"error: none of the {len(tfs)} TFs in {tfs_path} found in expression data",
              file=sys.stderr)
        return 2

    n_cells = adata.n_obs if suffix == ".h5ad" else len(df)
    print(f"rustscenic {__version__}  grn  cells={n_cells}  genes={len(gene_names)}  tfs={len(tfs_in)}  seed={args.seed}",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    out = rs_grn.infer(
        expression,
        tfs_in,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_features=args.max_features,
        subsample=args.subsample,
        max_depth=args.max_depth,
        early_stop_window=args.early_stop_window,
        seed=args.seed,
    )
    wall = time.monotonic() - t0

    output_path = Path(args.output)
    if output_path.suffix.lower() == ".parquet":
        out.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() in (".tsv", ".txt"):
        out.to_csv(output_path, sep="\t", index=False)
    elif output_path.suffix.lower() == ".csv":
        out.to_csv(output_path, index=False)
    else:
        # default to parquet if extension is unfamiliar
        out.to_parquet(output_path.with_suffix(".parquet"), index=False)

    meta = {
        "wall_clock_s": round(wall, 2),
        "n_edges": int(len(out)),
        "n_cells": int(n_cells),
        "n_genes": int(len(gene_names)),
        "n_tfs_used": len(tfs_in),
        "seed": args.seed,
        "rustscenic_version": __version__,
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"wrote {output_path}  ({len(out)} edges, wall {wall:.1f}s)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    from . import __version__
    p = argparse.ArgumentParser(prog="rustscenic", description="Fast SCENIC+ stage replacements")
    p.add_argument("--version", action="version", version=f"rustscenic {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("grn", help="Infer gene regulatory network (GRNBoost2 replacement)")
    pg.add_argument("--expression", required=True, help="Expression matrix: .h5ad, .tsv, or .csv")
    pg.add_argument("--tfs", required=True, help="TF list: one gene symbol per line")
    pg.add_argument("--output", required=True, help="Output path (.parquet, .tsv, .csv)")
    pg.add_argument("--seed", type=int, default=777)
    pg.add_argument("--n-estimators", type=int, default=5000)
    pg.add_argument("--learning-rate", type=float, default=0.01)
    pg.add_argument("--max-features", type=float, default=0.1)
    pg.add_argument("--subsample", type=float, default=0.9)
    pg.add_argument("--max-depth", type=int, default=3)
    pg.add_argument("--early-stop-window", type=int, default=25,
                    help="EarlyStopMonitor window; 0 disables")
    pg.set_defaults(func=cmd_grn)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
