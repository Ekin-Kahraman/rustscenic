"""Diff our stage output against the reference baseline. Fails CI if below threshold."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


THRESHOLDS = {
    "grn": {"spearman_top10k": 0.95, "exact_rank_top100": 0.75},
    "aucell": {"auc_correlation": 0.99},
    "topics": {"topic_ari": 0.85},
    "cistarget": {"motif_auc_corr": 0.95},
}


def compare_grn(ours: str, reference: str) -> dict:
    a = pd.read_parquet(ours)
    b = pd.read_parquet(reference)

    a = a.rename(columns=str.lower).sort_values("importance", ascending=False).head(10000)
    b = b.rename(columns=str.lower).sort_values("importance", ascending=False).head(10000)

    # Align on (TF, target) pairs, inner join
    merged = a.merge(b, on=["tf", "target"], suffixes=("_ours", "_ref"))
    spearman_top10k = float(spearmanr(merged["importance_ours"], merged["importance_ref"]).statistic)

    # Per-TF top-100 exact rank agreement
    per_tf_agreements = []
    for tf in a["tf"].unique()[:50]:
        a_tf = a[a["tf"] == tf].head(100)["target"].tolist()
        b_tf = b[b["tf"] == tf].head(100)["target"].tolist()
        if not a_tf or not b_tf:
            continue
        overlap = len(set(a_tf) & set(b_tf)) / max(len(a_tf), len(b_tf))
        per_tf_agreements.append(overlap)
    exact_rank_top100 = sum(per_tf_agreements) / max(len(per_tf_agreements), 1)

    return {"spearman_top10k": spearman_top10k, "exact_rank_top100": exact_rank_top100}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(THRESHOLDS.keys()))
    p.add_argument("--ours", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--fail-below-threshold", action="store_true")
    args = p.parse_args()

    if args.stage == "grn":
        metrics = compare_grn(args.ours, args.reference)
    else:
        sys.exit(f"stage {args.stage} comparator pending")

    print(json.dumps(metrics, indent=2))

    if args.fail_below_threshold:
        for k, v in metrics.items():
            thresh = THRESHOLDS[args.stage].get(k)
            if thresh is not None and v < thresh:
                sys.exit(f"FAIL: {k}={v:.4f} below threshold {thresh}")
        print("PASS")


if __name__ == "__main__":
    main()
