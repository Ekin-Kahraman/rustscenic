"""rustscenic vs current pyscenic GRN parity on PBMC 3k.

Runs both pipelines on the SAME preprocessed PBMC 3k AnnData with the SAME
seed and TF list, then computes:
  - top-K edge Jaccard
  - per-TF top-N target overlap
  - importance-rank Spearman correlation on shared edges
  - wall time + n_edges comparison

Inputs (must already exist):
  ARG 1: rustscenic GRN parquet (TF, target, importance)
  ARG 2: pyscenic / arboreto GRN parquet (same schema)
  ARG 3: output JSON path

Both inputs must come from the same preprocessed AnnData and same seed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def edge_set(df: pd.DataFrame, k: int) -> set[tuple[str, str]]:
    return set(zip(*df.sort_values("importance", ascending=False).head(k)[["TF", "target"]].values.T))


def per_tf_topn(df: pd.DataFrame, n: int) -> dict[str, list[str]]:
    out = {}
    for tf, sub in df.groupby("TF"):
        out[tf] = sub.sort_values("importance", ascending=False).head(n)["target"].tolist()
    return out


def main(rust_path: str, pyscenic_path: str, out_path: str) -> int:
    rust = pd.read_parquet(rust_path)
    pys = pd.read_parquet(pyscenic_path)

    # Some pyscenic outputs use 'tf' lowercase; normalise
    if "tf" in pys.columns and "TF" not in pys.columns:
        pys = pys.rename(columns={"tf": "TF"})
    assert {"TF", "target", "importance"}.issubset(rust.columns), rust.columns
    assert {"TF", "target", "importance"}.issubset(pys.columns), pys.columns

    rust_n = len(rust)
    pys_n = len(pys)
    rust_tfs = set(rust["TF"].unique())
    pys_tfs = set(pys["TF"].unique())
    shared_tfs = rust_tfs & pys_tfs

    # Top-K Jaccard at multiple K
    jaccard = {}
    for k in [1_000, 5_000, 10_000, 50_000]:
        re_set = edge_set(rust, k)
        pe_set = edge_set(pys, k)
        if re_set and pe_set:
            j = len(re_set & pe_set) / len(re_set | pe_set)
            jaccard[k] = round(j, 4)

    # Per-TF top-N target overlap (mean across shared TFs)
    per_tf_overlap = {}
    for n in [10, 20, 50]:
        rust_top = per_tf_topn(rust, n)
        pys_top = per_tf_topn(pys, n)
        ratios = []
        for tf in shared_tfs:
            rt = set(rust_top.get(tf, []))
            pt = set(pys_top.get(tf, []))
            if rt and pt:
                ratios.append(len(rt & pt) / max(len(rt | pt), 1))
        per_tf_overlap[n] = {
            "mean_jaccard": round(float(np.mean(ratios)), 4) if ratios else None,
            "median_jaccard": round(float(np.median(ratios)), 4) if ratios else None,
            "n_tfs": len(ratios),
        }

    # Importance-rank Spearman on shared (TF, target) edges
    rust_idx = rust.set_index(["TF", "target"])["importance"]
    pys_idx = pys.set_index(["TF", "target"])["importance"]
    shared = rust_idx.index.intersection(pys_idx.index)
    rho_per_edge = None
    if len(shared) > 100:
        rho, _ = spearmanr(rust_idx.loc[shared].values, pys_idx.loc[shared].values)
        rho_per_edge = round(float(rho), 4)

    # Per-TF importance-rank Spearman (within-TF, across targets)
    rho_within_tf_list = []
    for tf in sorted(shared_tfs):
        rt = rust[rust["TF"] == tf].set_index("target")["importance"]
        pt = pys[pys["TF"] == tf].set_index("target")["importance"]
        common_t = rt.index.intersection(pt.index)
        if len(common_t) >= 20:
            r, _ = spearmanr(rt.loc[common_t].values, pt.loc[common_t].values)
            if not np.isnan(r):
                rho_within_tf_list.append(r)
    rho_within_tf = (
        {
            "mean": round(float(np.mean(rho_within_tf_list)), 4),
            "median": round(float(np.median(rho_within_tf_list)), 4),
            "n_tfs_compared": len(rho_within_tf_list),
        }
        if rho_within_tf_list else None
    )

    out = {
        "rustscenic": {
            "path": str(Path(rust_path).resolve()),
            "n_edges": rust_n,
            "n_tfs": len(rust_tfs),
        },
        "pyscenic": {
            "path": str(Path(pyscenic_path).resolve()),
            "n_edges": pys_n,
            "n_tfs": len(pys_tfs),
        },
        "shared_tfs": len(shared_tfs),
        "shared_edges": len(shared),
        "topk_jaccard": jaccard,
        "per_tf_topn_jaccard": per_tf_overlap,
        "spearman_per_edge_on_shared": rho_per_edge,
        "spearman_within_tf": rho_within_tf,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: grn_parity_v0310.py <rustscenic_parquet> <pyscenic_parquet> <output_json>")
    sys.exit(main(*sys.argv[1:]))
