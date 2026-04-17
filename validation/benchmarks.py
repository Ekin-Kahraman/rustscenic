"""Run a command, measure wall-clock + peak RSS, append to results.csv.

Uses psutil for per-process RSS sampling — resource.getrusage(RUSAGE_CHILDREN)
accumulates max RSS across ALL children in the session, which pollutes
sequential measurements of different stages.
"""
import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


RESULTS_CSV = Path(__file__).parent / "results.csv"
HEADER = ["commit_sha", "stage", "dataset", "metric_name", "value", "wall_clock_s", "peak_rss_mb", "date"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_and_measure(cmd: list[str], poll_interval: float = 0.05) -> tuple[float, float]:
    """Run cmd, sample its RSS every poll_interval seconds, return (wall_s, peak_rss_mb)."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)
    peak_rss = 0
    try:
        while proc.poll() is None:
            try:
                total = ps.memory_info().rss + sum(
                    child.memory_info().rss for child in ps.children(recursive=True)
                )
                peak_rss = max(peak_rss, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(poll_interval)
    finally:
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"subprocess failed: rc={rc} cmd={cmd}")
    wall = time.monotonic() - t0
    return wall, peak_rss / (1024 * 1024)


def append(rows: list[dict]) -> None:
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
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
    p.add_argument("--cmd", nargs=argparse.REMAINDER, required=True, help="-- <command...>")
    args = p.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    wall, rss_mb = run_and_measure(cmd)
    sha = git_sha()
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")

    base = {"commit_sha": sha, "stage": args.stage, "dataset": args.dataset,
            "wall_clock_s": wall, "peak_rss_mb": rss_mb, "date": date}

    rows = [
        {**base, "metric_name": "wall_clock_s", "value": wall},
        {**base, "metric_name": "peak_rss_mb", "value": rss_mb},
    ]

    if args.metrics_json and Path(args.metrics_json).exists():
        for k, v in json.loads(Path(args.metrics_json).read_text()).items():
            if isinstance(v, (int, float)):
                rows.append({**base, "metric_name": k, "value": v})

    append(rows)
    print(f"logged {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
