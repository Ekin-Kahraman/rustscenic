"""Run our stage, measure wall-clock + peak RSS, append to results.csv."""
import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RESULTS_CSV = Path(__file__).parent / "results.csv"
HEADER = ["commit_sha", "stage", "dataset", "metric_name", "value", "wall_clock_s", "peak_rss_mb", "date"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_and_measure(cmd: list[str]) -> tuple[float, float]:
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    wall = time.monotonic() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # macOS reports bytes, linux reports KB — normalize to MB
    rss_mb = rss_kb / (1024 * 1024) if sys.platform == "darwin" else rss_kb / 1024
    return wall, rss_mb


def append(rows: list[dict]) -> None:
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--dataset", default="pbmc3k")
    p.add_argument("--metrics-json", help="path to compare.py output")
    p.add_argument("--cmd", nargs="+", required=True, help="command to run and measure")
    args = p.parse_args()

    wall, rss_mb = run_and_measure(args.cmd)
    sha = git_sha()
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    rows.append({"commit_sha": sha, "stage": args.stage, "dataset": args.dataset,
                 "metric_name": "wall_clock_s", "value": wall,
                 "wall_clock_s": wall, "peak_rss_mb": rss_mb, "date": date})
    rows.append({"commit_sha": sha, "stage": args.stage, "dataset": args.dataset,
                 "metric_name": "peak_rss_mb", "value": rss_mb,
                 "wall_clock_s": wall, "peak_rss_mb": rss_mb, "date": date})

    if args.metrics_json and Path(args.metrics_json).exists():
        m = json.loads(Path(args.metrics_json).read_text())
        for k, v in m.items():
            rows.append({"commit_sha": sha, "stage": args.stage, "dataset": args.dataset,
                         "metric_name": k, "value": v,
                         "wall_clock_s": wall, "peak_rss_mb": rss_mb, "date": date})

    append(rows)
    print(f"logged {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
