"""Local head-to-head stage benchmark harness.

This is intentionally small and local-only. It generates deterministic
synthetic expression data, runs one stage in one process, and writes a JSON
record with wall time, peak RSS, settings, and output counts.

Examples:
  python validation/head_to_head/bench_stage.py \
    --tool rustscenic --stage grn --n-cells 100 --out results.json

  validation/head_to_head/.venv-scenicplus-py3118/bin/python \
    validation/head_to_head/bench_stage.py \
    --tool reference --stage grn --n-cells 100 --out results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 3)
    return rss / (1024 ** 2)


def synthetic_expression(
    n_cells: int,
    n_genes: int,
    n_tfs: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str], list[tuple[str, list[str]]]]:
    rng = np.random.default_rng(seed)
    n_programmes = max(1, min(n_tfs, 8))
    programmes = rng.integers(0, n_programmes, size=n_cells)
    latent = rng.normal(size=(n_cells, n_programmes)).astype(np.float32)
    latent += np.eye(n_programmes, dtype=np.float32)[programmes] * 2.0

    gene_names = [f"G{i:04d}" for i in range(n_genes)]
    tf_names = gene_names[:n_tfs]
    X = rng.gamma(shape=1.5, scale=0.25, size=(n_cells, n_genes)).astype(np.float32)

    targets_per_programme = max(5, min(30, (n_genes - n_tfs) // n_programmes))
    planted: list[tuple[str, list[str]]] = []
    for p in range(n_programmes):
        tf = tf_names[p % len(tf_names)]
        start = n_tfs + p * targets_per_programme
        stop = min(n_genes, start + targets_per_programme)
        targets = gene_names[start:stop]
        if not targets:
            continue
        X[:, start:stop] += latent[:, [p]] * rng.uniform(
            0.8, 1.4, size=(1, stop - start)
        ).astype(np.float32)
        planted.append((f"{tf}_regulon", targets))

    np.clip(X, 0.0, None, out=X)
    X = np.log1p(X)
    expr = pd.DataFrame(X, columns=gene_names, index=[f"cell_{i}" for i in range(n_cells)])
    return expr, tf_names, planted


def synthetic_regulons(
    gene_names: list[str],
    n_regulons: int,
    regulon_size: int,
    seed: int,
) -> list[tuple[str, list[str]]]:
    rng = np.random.default_rng(seed + 10_000)
    if n_regulons < 1:
        raise ValueError("n_regulons must be >= 1")
    if regulon_size < 1:
        raise ValueError("regulon_size must be >= 1")
    size = min(regulon_size, len(gene_names))
    regulons = []
    genes = np.asarray(gene_names)
    for i in range(n_regulons):
        idx = rng.choice(len(genes), size=size, replace=False)
        regulons.append((f"REG_{i:04d}", genes[idx].tolist()))
    return regulons


def dataframe_checksum(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def run_rustscenic_grn(expr: pd.DataFrame, tf_names: list[str], args) -> dict:
    import rustscenic.grn

    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        expr,
        tf_names=tf_names,
        n_estimators=args.n_estimators,
        seed=args.seed,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "output_counts": {
            "edges": int(len(grn)),
            "tfs": int(grn["TF"].nunique()) if not grn.empty else 0,
            "targets": int(grn["target"].nunique()) if not grn.empty else 0,
        },
    }


def run_reference_grn(expr: pd.DataFrame, tf_names: list[str], args) -> dict:
    from arboreto.algo import grnboost2

    t0 = time.perf_counter()
    grn = grnboost2(
        expression_data=expr,
        tf_names=tf_names,
        seed=args.seed,
        client_or_address=None,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    tf_col = "TF" if "TF" in grn.columns else "regulator"
    target_col = "target"
    return {
        "wall_s": wall,
        "output_counts": {
            "edges": int(len(grn)),
            "tfs": int(grn[tf_col].nunique()) if not grn.empty else 0,
            "targets": int(grn[target_col].nunique()) if not grn.empty else 0,
        },
    }


def run_rustscenic_aucell(
    expr: pd.DataFrame,
    regulons: list[tuple[str, list[str]]],
    args,
) -> dict:
    import rustscenic.aucell

    t0 = time.perf_counter()
    auc = rustscenic.aucell.score(expr, regulons, top_frac=args.top_frac)
    wall = time.perf_counter() - t0
    values = auc.to_numpy()
    return {
        "wall_s": wall,
        "output_counts": {
            "cells": int(auc.shape[0]),
            "regulons": int(auc.shape[1]),
            "nonzero": int((values > 0).sum()),
        },
    }


def run_reference_aucell(
    expr: pd.DataFrame,
    regulons: list[tuple[str, list[str]]],
    args,
) -> dict:
    from ctxcore.genesig import GeneSignature
    from pyscenic.aucell import aucell

    signatures = [
        GeneSignature(name=name, gene2weight={gene: 1.0 for gene in genes})
        for name, genes in regulons
    ]
    t0 = time.perf_counter()
    auc = aucell(
        expr,
        signatures,
        auc_threshold=args.top_frac,
        noweights=True,
        normalize=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    wall = time.perf_counter() - t0
    values = auc.to_numpy()
    return {
        "wall_s": wall,
        "output_counts": {
            "cells": int(auc.shape[0]),
            "regulons": int(auc.shape[1]),
            "nonzero": int((values > 0).sum()),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["rustscenic", "reference"], required=True)
    p.add_argument("--stage", choices=["grn", "aucell"], required=True)
    p.add_argument("--n-cells", type=int, required=True)
    p.add_argument("--n-genes", type=int, default=300)
    p.add_argument("--n-tfs", type=int, default=20)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--n-regulons", type=int, default=0)
    p.add_argument("--regulon-size", type=int, default=30)
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    expr, tf_names, regulons = synthetic_expression(
        args.n_cells, args.n_genes, args.n_tfs, args.seed
    )
    if args.stage == "aucell" and args.n_regulons:
        regulons = synthetic_regulons(
            list(expr.columns), args.n_regulons, args.regulon_size, args.seed
        )
    start_rss = peak_rss_gb()
    status = "ok"
    error = None
    result: dict
    t0 = time.perf_counter()
    try:
        if args.tool == "rustscenic" and args.stage == "grn":
            result = run_rustscenic_grn(expr, tf_names, args)
        elif args.tool == "reference" and args.stage == "grn":
            result = run_reference_grn(expr, tf_names, args)
        elif args.tool == "rustscenic" and args.stage == "aucell":
            result = run_rustscenic_aucell(expr, regulons, args)
        elif args.tool == "reference" and args.stage == "aucell":
            result = run_reference_aucell(expr, regulons, args)
        else:
            raise AssertionError("unreachable tool/stage combination")
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        result = {"wall_s": time.perf_counter() - t0, "output_counts": {}}

    record = {
        "label": args.label,
        "tool": args.tool,
        "stage": args.stage,
        "status": status,
        "error": error,
        "wall_s": round(float(result["wall_s"]), 6),
        "peak_rss_gb": round(float(peak_rss_gb()), 6),
        "start_rss_gb": round(float(start_rss), 6),
        "output_counts": result["output_counts"],
        "settings": {
            "n_cells": args.n_cells,
            "n_genes": args.n_genes,
            "n_tfs": args.n_tfs,
            "n_estimators": args.n_estimators if args.tool == "rustscenic" else None,
            "n_regulons": args.n_regulons if args.stage == "aucell" else None,
            "regulon_size": args.regulon_size if args.stage == "aucell" else None,
            "top_frac": args.top_frac,
            "num_workers": args.num_workers if args.stage == "aucell" else None,
            "seed": args.seed,
        },
        "input_checksum": dataframe_checksum(expr),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
