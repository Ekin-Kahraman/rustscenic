#!/usr/bin/env python3
"""Run rustscenic validation scenarios in fresh Docker containers.

The harness is intentionally deterministic. It archives git HEAD into a Docker
build context, runs one named scenario in the container, and emits a structured
report plus the full log. Dirty worktree state is recorded but not copied.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCENARIO_DIR = HERE / "scenarios"
DEFAULT_REPORTS_DIR = HERE / "reports"
SCHEMA_VERSION = "1.0"
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def run_cmd(
    args: list[str],
    *,
    cwd: Path | None = None,
    input_stream: Any | None = None,
    capture: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        input=input_stream,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


def require_success(proc: subprocess.CompletedProcess[str], what: str) -> None:
    if proc.returncode == 0:
        return
    detail = (proc.stdout or "") + (proc.stderr or "")
    raise RuntimeError(f"{what} failed with exit code {proc.returncode}\n{detail}")


def repo_root() -> Path:
    proc = run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=HERE)
    require_success(proc, "git rev-parse")
    return Path(proc.stdout.strip()).resolve()


def git_text(repo: Path, args: list[str]) -> str:
    proc = run_cmd(["git", *args], cwd=repo)
    require_success(proc, "git " + " ".join(args))
    return proc.stdout.strip()


def git_status(repo: Path) -> list[str]:
    proc = run_cmd(["git", "status", "--porcelain"], cwd=repo)
    require_success(proc, "git status")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        scenario = json.load(fh)
    required = {
        "id",
        "title",
        "agent_persona",
        "goal",
        "base_image",
        "timeout_seconds",
        "tags",
        "commands",
    }
    missing = required.difference(scenario)
    if missing:
        raise ValueError(f"{path} missing required field(s): {', '.join(sorted(missing))}")
    if not isinstance(scenario["commands"], list) or not scenario["commands"]:
        raise ValueError(f"{path} must define at least one command")
    for command in scenario["commands"]:
        if not isinstance(command, dict) or {"name", "run"}.difference(command):
            raise ValueError(f"{path} has an invalid command entry: {command!r}")
    return scenario


def discover_scenarios() -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        scenario = load_scenario(path)
        sid = scenario["id"]
        if sid in scenarios:
            raise ValueError(f"duplicate scenario id: {sid}")
        scenarios[sid] = scenario
    return scenarios


def print_scenarios(scenarios: dict[str, dict[str, Any]]) -> None:
    for sid, scenario in scenarios.items():
        tags = ", ".join(scenario.get("tags", []))
        print(f"{sid:28} {scenario['title']} [{tags}]")


def archive_head(repo: Path, context_dir: Path) -> None:
    repo_dir = context_dir / "repo"
    repo_dir.mkdir(parents=True)
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        text=False,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        ["tar", "-xf", "-", "-C", str(repo_dir)],
        stdin=archive.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    archive.stdout.close()
    archive_code = archive.wait()
    if archive_code != 0:
        raise RuntimeError(f"git archive failed with exit code {archive_code}")
    require_success(extract, "tar extract")


def write_dockerfile(context_dir: Path, base_image: str) -> Path:
    dockerfile = context_dir / "Dockerfile"
    dockerfile.write_text(
        textwrap.dedent(
            f"""\
            FROM {base_image}

            ENV DEBIAN_FRONTEND=noninteractive
            RUN apt-get update \\
                && apt-get install -y --no-install-recommends \\
                    bash \\
                    build-essential \\
                    ca-certificates \\
                    curl \\
                    git \\
                    pkg-config \\
                    python3 \\
                    python3-pip \\
                    python3-venv \\
                    time \\
                && rm -rf /var/lib/apt/lists/*

            ENV VIRTUAL_ENV=/opt/rustscenic-agentic-venv
            RUN python3 -m venv "$VIRTUAL_ENV"
            ENV PATH=/opt/rustscenic-agentic-venv/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            RUN python -m pip install --upgrade pip

            WORKDIR /repo
            COPY repo/ /repo/
            """
        )
    )
    return dockerfile


def render_script(scenario: dict[str, Any]) -> str:
    lines = [
        "set -euo pipefail",
        "export PYTHONUNBUFFERED=1",
        f"echo {shlex.quote('scenario: ' + scenario['id'])}",
        f"echo {shlex.quote('goal: ' + scenario['goal'])}",
        "echo '--- environment ---'",
        "python --version",
        "rustc --version",
        "cargo --version",
        "echo '--- commands ---'",
    ]
    for idx, command in enumerate(scenario["commands"], start=1):
        name = command["name"]
        lines.extend(
            [
                f"echo {shlex.quote(f'[{idx}/{len(scenario['commands'])}] {name}')}",
                command["run"],
            ]
        )
    return "\n\n".join(lines) + "\n"


def docker_run_script(image: str, script: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    wrapped = "\n".join(
        [
            "cat > /tmp/rustscenic-agentic-scenario.sh <<'RUSTSCENIC_AGENTIC_EOF'",
            script.rstrip(),
            "RUSTSCENIC_AGENTIC_EOF",
            "/usr/bin/time -v bash /tmp/rustscenic-agentic-scenario.sh",
        ]
    )
    return run_cmd(
        ["docker", "run", "--rm", image, "bash", "-c", wrapped],
        capture=True,
        timeout=timeout_seconds,
    )


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def write_report(
    *,
    scenario: dict[str, Any],
    repo: Path,
    reports_dir: Path,
    docker_image: str,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    duration_seconds: float,
    proc: subprocess.CompletedProcess[str],
    build_log: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}_{safe_id(scenario['id'])}"
    log_path = reports_dir / f"{base}.log"
    report_path = reports_dir / f"{base}.json"

    combined_log = (
        "=== docker build ===\n"
        + build_log
        + "\n=== scenario run ===\n"
        + (proc.stdout or "")
        + (proc.stderr or "")
    )
    log_path.write_text(combined_log)

    rss_match = RSS_RE.search(combined_log)
    status = git_status(repo)
    notes: list[str] = []
    if status:
        notes.append(
            "Worktree was dirty; Docker image was built from git HEAD only, not uncommitted changes."
        )
    if proc.returncode != 0:
        notes.append("Scenario command failed; inspect log_path for stdout and stderr.")

    remote_proc = run_cmd(["git", "config", "--get", "remote.origin.url"], cwd=repo)
    remote_url = remote_proc.stdout.strip() if remote_proc.returncode == 0 else ""

    report = {
        "schema_version": SCHEMA_VERSION,
        "scenario": {
            "id": scenario["id"],
            "title": scenario["title"],
            "agent_persona": scenario["agent_persona"],
            "goal": scenario["goal"],
            "tags": scenario.get("tags", []),
        },
        "repo": {
            "head_sha": git_text(repo, ["rev-parse", "HEAD"]),
            "remote_url": remote_url,
            "dirty": bool(status),
            "status": status,
        },
        "environment": {
            "host_platform": platform.platform(),
            "base_image": scenario["base_image"],
            "docker_image": docker_image,
        },
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": round(duration_seconds, 3),
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "peak_rss_kb": int(rss_match.group(1)) if rss_match else None,
        "commands": scenario["commands"],
        "log_path": str(log_path.relative_to(repo) if log_path.is_relative_to(repo) else log_path),
        "notes": notes,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


def run_scenario(
    scenario: dict[str, Any],
    *,
    repo: Path,
    reports_dir: Path,
    no_cache: bool,
    pull: bool,
    dry_run: bool,
    keep_context: bool,
) -> bool:
    script = render_script(scenario)
    if dry_run:
        print(f"\n# {scenario['id']}\n{script}")
        return True

    started_at = utc_now()
    t0 = time.perf_counter()
    build_log = ""

    temp_manager = tempfile.TemporaryDirectory(prefix=f"rustscenic-agentic-{scenario['id']}-")
    context_path = Path(temp_manager.name)
    try:
        archive_head(repo, context_path)
        write_dockerfile(context_path, scenario["base_image"])

        image = f"rustscenic-agentic-{safe_id(scenario['id'])}:{git_text(repo, ['rev-parse', '--short', 'HEAD'])}"
        build_cmd = ["docker", "build", "-t", image]
        if pull:
            build_cmd.append("--pull")
        if no_cache:
            build_cmd.append("--no-cache")
        build_cmd.append(str(context_path))

        print(f"building {image} from {scenario['base_image']}...")
        build = run_cmd(build_cmd, capture=True)
        build_log = (build.stdout or "") + (build.stderr or "")
        if build.returncode != 0:
            proc = subprocess.CompletedProcess(build_cmd, build.returncode, "", build_log)
        else:
            print(f"running scenario {scenario['id']}...")
            proc = docker_run_script(image, script, int(scenario["timeout_seconds"]))
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(
            exc.cmd,
            124,
            exc.stdout or "",
            (exc.stderr or "") + f"\nTimed out after {scenario['timeout_seconds']} seconds.\n",
        )
    finally:
        finished_at = utc_now()
        duration = time.perf_counter() - t0

    report = write_report(
        scenario=scenario,
        repo=repo,
        reports_dir=reports_dir,
        docker_image=(
            f"rustscenic-agentic-{safe_id(scenario['id'])}:"
            f"{git_text(repo, ['rev-parse', '--short', 'HEAD'])}"
        ),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        proc=proc,
        build_log=build_log,
    )

    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"{status}: {scenario['id']} -> {report.relative_to(repo)}")
    if keep_context:
        print(f"kept Docker context at {context_path}")
    else:
        temp_manager.cleanup()
    return proc.returncode == 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="scenario id to run; can be passed multiple times",
    )
    parser.add_argument("--all", action="store_true", help="run all scenarios")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="directory for JSON reports and logs",
    )
    parser.add_argument("--dry-run", action="store_true", help="print rendered scripts")
    parser.add_argument("--no-cache", action="store_true", help="pass --no-cache to docker build")
    parser.add_argument("--pull", action="store_true", help="pass --pull to docker build")
    parser.add_argument(
        "--keep-context",
        action="store_true",
        help="keep the temporary Docker build context for debugging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo = repo_root()
    scenarios = discover_scenarios()

    if args.list or (not args.scenario and not args.all):
        print_scenarios(scenarios)
        return 0

    selected = list(scenarios) if args.all else args.scenario
    unknown = [sid for sid in selected if sid not in scenarios]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    if not args.dry_run:
        docker = run_cmd(["docker", "version", "--format", "{{.Server.Version}}"])
        require_success(docker, "docker version")

    ok = True
    for sid in selected:
        ok = (
            run_scenario(
                scenarios[sid],
                repo=repo,
                reports_dir=args.reports_dir,
                no_cache=args.no_cache,
                pull=args.pull,
                dry_run=args.dry_run,
                keep_context=args.keep_context,
            )
            and ok
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
