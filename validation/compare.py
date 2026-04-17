"""Diff our stage output against the reference baseline.

Exits non-zero (via --fail-below-threshold) if any metric is below its gate.
Metrics are designed to penalize divergence, not hide it:
  - Jaccard of top-10k edge sets: catches cases where our tool produces different
    top edges (inner-join Spearman misses this).
  - Spearman on importance ranks across the UNION of both top-10k sets: penalizes
    missing edges (rank = bottom) and out-of-order edges simultaneously.
  - Per-TF top-100 target overlap: average across all TFs present in both outputs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


THRESHOLDS = {
    "grn": {
        "jaccard_top10k": 0.80,
        "spearman_union_top10k": 0.85,
        "per_tf_top100_mean": 0.70,
    },
    "aucell": {"auc_correlation": 0.99},
    "topics": {"topic_ari": 0.85},
    "cistarget": {"motif_auc_corr": 0.95},
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = {"tf", "target", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: expected {required}, got {set(df.columns)}")
    return df[["tf", "target", "importance"]]


def compare_grn(ours_path: str, reference_path: str) -> dict:
    ours = _normalize(pd.read_parquet(ours_path))
    ref = _normalize(pd.read_parquet(reference_path))

    top_k = 10_000
    ours_top = ours.sort_values("importance", ascending=False).head(top_k)
    ref_top = ref.sort_values("importance", ascending=False).head(top_k)

    ours_edges = set(zip(ours_top["tf"], ours_top["target"]))
    ref_edges = set(zip(ref_top["tf"], ref_top["target"]))

    intersection = ours_edges & ref_edges
    union = ours_edges | ref_edges
    jaccard = len(intersection) / max(len(union), 1)

    # Spearman across the UNION — missing edges in either set get a rank of 0
    # (importance 0 equivalent) so their presence/absence is penalized.
    edge_to_ours = dict(zip(zip(ours["tf"], ours["target"]), ours["importance"]))
    edge_to_ref = dict(zip(zip(ref["tf"], ref["target"]), ref["importance"]))

    ours_imp = np.array([edge_to_ours.get(e, 0.0) for e in union])
    ref_imp = np.array([edge_to_ref.get(e, 0.0) for e in union])
    spearman_union = float(spearmanr(ours_imp, ref_imp).statistic)

    # Per-TF top-100 target overlap across ALL shared TFs
    shared_tfs = set(ours["tf"]) & set(ref["tf"])
    per_tf = []
    for tf in shared_tfs:
        a = ours[ours["tf"] == tf].nlargest(100, "importance")["target"].tolist()
        b = ref[ref["tf"] == tf].nlargest(100, "importance")["target"].tolist()
        if not a or not b:
            continue
        per_tf.append(len(set(a) & set(b)) / max(len(a), len(b)))
    per_tf_mean = float(np.mean(per_tf)) if per_tf else 0.0

    return {
        "jaccard_top10k": jaccard,
        "spearman_union_top10k": spearman_union,
        "per_tf_top100_mean": per_tf_mean,
        "n_edges_ours": len(ours),
        "n_edges_ref": len(ref),
        "n_tfs_shared": len(shared_tfs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(THRESHOLDS.keys()))
    p.add_argument("--ours", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--fail-below-threshold", action="store_true")
    p.add_argument("--json-out", help="optional: write metrics JSON here")
    args = p.parse_args()

    if args.stage == "grn":
        metrics = compare_grn(args.ours, args.reference)
    else:
        sys.exit(f"stage {args.stage} comparator pending")

    out = json.dumps(metrics, indent=2)
    print(out)
    if args.json_out:
        Path(args.json_out).write_text(out)

    if args.fail_below_threshold:
        failed = []
        for k, v in metrics.items():
            thresh = THRESHOLDS[args.stage].get(k)
            if thresh is not None and v < thresh:
                failed.append(f"{k}={v:.4f} below threshold {thresh}")
        if failed:
            sys.exit("FAIL: " + "; ".join(failed))
        print("PASS")


if __name__ == "__main__":
    main()
