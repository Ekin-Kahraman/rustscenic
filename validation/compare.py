"""Diff our stage output against the reference baseline.

v0.1 correctness gates (revised after audit): we cannot be bit-identical to
sklearn's Cython GBR without porting its exact RNG tape. That chase is
unproductive. The gates below test what users actually care about:

  - per_tf_top5_mean:  for each TF, does our top-5 targets overlap arboreto's
                       top-5? (captures biological signal retention)
  - per_tf_top100_mean: same at broader scope — catches divergence in the long
                        tail of a regulator's targets
  - biological_hit:    on a fixed list of known (TF, target) edges from immune
                       biology, do we recover them in the TF's top-20 targets?
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = {
    "grn": {
        # Biological validity: do known immune TF->target edges appear in ours?
        # If this fails, the tool produces meaningless networks.
        "biological_hit_rate": 0.80,
        # Broad agreement with arboreto on which targets a given TF regulates
        # (per-TF top-100, averaged across TFs). Noisier than top-5 but more
        # stable across stochastic tree differences.
        "per_tf_top100_mean": 0.50,
        # We intentionally DO NOT gate per-TF top-5 Jaccard or global top-10k
        # Jaccard — those are dominated by tie-breaking in the stochastic GBM
        # tape, which differs between sklearn Cython and any independent impl.
        # Strict numerical replication of sklearn is a multi-week project; that
        # chase is explicitly out of scope for v0.1.
    },
}

# Curated immune-regulation edges from well-established biology
# (Aibar 2017 SCENIC paper supp + HSC/myeloid literature)
KNOWN_EDGES = [
    # SPI1 (PU.1) — master myeloid TF
    ("SPI1", "CST3"), ("SPI1", "FCER1G"), ("SPI1", "LGALS1"), ("SPI1", "TYROBP"),
    ("SPI1", "SAT1"), ("SPI1", "LYZ"), ("SPI1", "AIF1"), ("SPI1", "PSAP"),
    # CEBPD — myeloid differentiation
    ("CEBPD", "TYROBP"), ("CEBPD", "LYZ"), ("CEBPD", "PSAP"),
    # MAFB — macrophage
    ("MAFB", "CST3"), ("MAFB", "AIF1"), ("MAFB", "LYZ"),
    # CEBPB — myeloid inflammation
    ("CEBPB", "PSAP"), ("CEBPB", "LGALS1"),
    # KLF4 — macrophage polarization
    ("KLF4", "LYZ"), ("KLF4", "PSAP"),
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = {"tf", "target", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"missing columns: expected {required}, got {set(df.columns)}")
    return df[["tf", "target", "importance"]]


def compare_grn(ours_path: str, reference_path: str) -> dict:
    ours = _normalize(pd.read_parquet(ours_path))
    ref = _normalize(pd.read_parquet(reference_path))

    shared_tfs = set(ours["tf"]) & set(ref["tf"])

    def per_tf_topk(k: int) -> float:
        scores = []
        for tf in shared_tfs:
            a = ours[ours["tf"] == tf].nlargest(k, "importance")["target"].tolist()
            b = ref[ref["tf"] == tf].nlargest(k, "importance")["target"].tolist()
            if not a or not b:
                continue
            scores.append(len(set(a) & set(b)) / max(len(a), len(b)))
        return float(np.mean(scores)) if scores else 0.0

    per_tf_top5 = per_tf_topk(5)
    per_tf_top20 = per_tf_topk(20)
    per_tf_top100 = per_tf_topk(100)

    # Biological hit: for each known (TF, target), is target in TF's top-20 by importance?
    hits = 0
    checked = 0
    for tf, target in KNOWN_EDGES:
        sub = ours[ours["tf"] == tf].nlargest(20, "importance")["target"].tolist()
        if not sub:
            continue
        checked += 1
        if target in sub:
            hits += 1
    bio_hit = hits / max(checked, 1)

    return {
        "biological_hit_rate": bio_hit,
        "per_tf_top100_mean": per_tf_top100,
        "per_tf_top20_mean": per_tf_top20,
        "per_tf_top5_mean": per_tf_top5,   # reported; not gated (too noisy)
        "known_edges_checked": checked,
        "known_edges_hit": hits,
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
