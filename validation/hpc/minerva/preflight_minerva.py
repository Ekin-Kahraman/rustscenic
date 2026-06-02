"""Preflight checks for RustScenic Minerva benchmark jobs.

Run this from the Minerva checkout before submitting or inside an LSF launcher.
It checks the repo, Python environment, required real-data files, git state, and
active rustscenic import before the benchmark spends allocation time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.backend_requirements import REQUIRED_RUST_BACKEND_SYMBOLS
from validation.repo_cleanliness import repo_state_from_git_outputs


DEFAULT_PROJECT = Path("/sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic")
DEFAULT_ENV = Path("/sc/arion/work/kahrae01/rustscenic/envs/rustscenic-v047")
DEFAULT_REPO = DEFAULT_PROJECT / "repo"
REQUIRED_DATA_FILES = (
    "pbmc_3k_filtered_feature_bc_matrix.h5",
    "pbmc_3k_atac_fragments.tsv.gz",
    "pbmc_3k_atac_peaks.bed",
)


def _run(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _path_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        return {"path": str(path), "exists": True, "type": "dir"}
    return {
        "path": str(path),
        "exists": True,
        "type": "file",
        "size_bytes": path.stat().st_size,
    }


def _git_state(repo: Path) -> dict[str, Any]:
    commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo), "status", "--short", "--untracked-files=no"])
    untracked_status = _run(["git", "-C", str(repo), "status", "--short", "--untracked-files=all"])
    diff = _run(["git", "-C", str(repo), "diff", "HEAD", "--binary", "--no-ext-diff"])
    state = repo_state_from_git_outputs(
        commit=commit.stdout.strip() if commit.returncode == 0 else None,
        tracked_status=status.stdout if status.returncode == 0 else "",
        untracked_status=(
            untracked_status.stdout if untracked_status.returncode == 0 else ""
        ),
        tracked_diff=diff.stdout if diff.returncode == 0 else "",
    )
    state.update({
        "commit_error": commit.stderr.strip() if commit.returncode != 0 else None,
        "status_error": status.stderr.strip() if status.returncode != 0 else None,
        "untracked_error": (
            untracked_status.stderr.strip()
            if untracked_status.returncode != 0
            else None
        ),
        "diff_error": diff.stderr.strip() if diff.returncode != 0 else None,
    })
    return state


def _path_under(path: str | None, root: Path) -> bool | None:
    if not path:
        return None
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _import_state(python: Path, repo: Path) -> dict[str, Any]:
    code = "\n".join(
        [
            "import json",
            "import sys",
            "import rustscenic",
            "package_version = getattr(rustscenic, '__version__', None)",
            "try:",
            "    import rustscenic._rustscenic as ext",
            "except Exception as exc:",
            "    extension_file = None",
            "    extension_version = None",
            "    extension_error = repr(exc)",
            "else:",
            "    extension_file = getattr(ext, '__file__', None)",
            "    extension_version = getattr(ext, '__version__', None)",
            "    extension_error = None",
            "print(json.dumps({",
            "    'python': sys.executable,",
            "    'rustscenic_version': package_version,",
            "    'package_version': package_version,",
            "    'extension_version': extension_version,",
            "    'package_file': getattr(rustscenic, '__file__', None),",
            "    'extension_file': extension_file,",
            "    'extension_error': extension_error,",
            "}))",
        ]
    )
    proc = _run([str(python), "-c", code], cwd=repo)
    payload: dict[str, Any] = {}
    parse_error = None
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        try:
            payload = json.loads(lines[-1]) if lines else {}
        except (IndexError, json.JSONDecodeError) as exc:
            parse_error = repr(exc)

    package_file = payload.get("package_file")
    extension_file = payload.get("extension_file")
    extension_error = payload.get("extension_error")
    return {
        "ok": proc.returncode == 0 and parse_error is None and extension_error is None,
        "python": payload.get("python", str(python)),
        "rustscenic_version": payload.get("rustscenic_version"),
        "package_version": payload.get("package_version"),
        "extension_version": payload.get("extension_version"),
        "package_file": package_file,
        "package_under_repo": _path_under(package_file, repo),
        "extension_file": extension_file,
        "extension_under_repo": _path_under(extension_file, repo),
        "extension_error": extension_error,
        "parse_error": parse_error,
        "stderr": proc.stderr.strip(),
    }


def _backend_state(python: Path, repo: Path) -> dict[str, Any]:
    required = json.dumps(REQUIRED_RUST_BACKEND_SYMBOLS, sort_keys=True)
    code = "\n".join(
        [
            "import json",
            f"required = json.loads({required!r})",
            "try:",
            "    import rustscenic._rustscenic as ext",
            "except Exception as exc:",
            "    print(json.dumps({",
            "        'ok': False,",
            "        'extension_error': repr(exc),",
            "        'required_symbols': required,",
            "        'missing_symbols': [",
            "            f'{stage}.{symbol}'",
            "            for stage, symbols in required.items()",
            "            for symbol in symbols",
            "        ],",
            "    }))",
            "else:",
            "    missing = [",
            "        f'{stage}.{symbol}'",
            "        for stage, symbols in required.items()",
            "        for symbol in symbols",
            "        if not hasattr(ext, symbol)",
            "    ]",
            "    print(json.dumps({",
            "        'ok': not missing,",
            "        'extension_error': None,",
            "        'required_symbols': required,",
            "        'missing_symbols': missing,",
            "    }))",
        ]
    )
    proc = _run([str(python), "-c", code], cwd=repo)
    payload: dict[str, Any] = {}
    parse_error = None
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        try:
            payload = json.loads(lines[-1]) if lines else {}
        except (IndexError, json.JSONDecodeError) as exc:
            parse_error = repr(exc)
    if parse_error is not None:
        payload = {
            "ok": False,
            "extension_error": None,
            "required_symbols": REQUIRED_RUST_BACKEND_SYMBOLS,
            "missing_symbols": [],
        }
    payload["parse_error"] = parse_error
    payload["stderr"] = proc.stderr.strip()
    return payload


def _thread_env_state() -> dict[str, Any]:
    keys = (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LSB_DJOB_NUMPROC",
    )
    return {key.lower(): os.environ.get(key) for key in keys}


def _thread_env_failures(thread_env: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("omp_num_threads", "openblas_num_threads", "mkl_num_threads"):
        if thread_env.get(key) != "1":
            failures.append(f"{key.upper()} must be 1")

    rayon = thread_env.get("rayon_num_threads")
    if rayon is None:
        failures.append("RAYON_NUM_THREADS must be set")
    else:
        try:
            rayon_threads = int(rayon)
        except ValueError:
            failures.append("RAYON_NUM_THREADS must be a positive integer")
        else:
            if rayon_threads <= 0:
                failures.append("RAYON_NUM_THREADS must be positive")
            lsf_cores = thread_env.get("lsb_djob_numproc")
            if lsf_cores is not None:
                try:
                    lsf_threads = int(lsf_cores)
                except ValueError:
                    failures.append("LSB_DJOB_NUMPROC must be a positive integer when set")
                else:
                    if lsf_threads <= 0:
                        failures.append("LSB_DJOB_NUMPROC must be positive when set")
                    elif lsf_threads != rayon_threads:
                        failures.append(
                            "RAYON_NUM_THREADS must match LSB_DJOB_NUMPROC: "
                            f"{rayon_threads} != {lsf_threads}"
                        )
    return failures


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    env = args.env.resolve()
    data_dir = args.data_dir.resolve()
    python = args.python.resolve() if args.python else env / "bin" / "python"

    checks: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": _path_status(repo),
        "env": _path_status(env),
        "python": _path_status(python),
        "data_dir": _path_status(data_dir),
        "data_files": {
            name: _path_status(data_dir / name)
            for name in REQUIRED_DATA_FILES
        },
        "benchmark_scripts": {
            "full_pipeline": _path_status(repo / "validation/scaling/bench_real_multiome_pipeline.py"),
            "full_pipeline_scaling": _path_status(repo / "validation/scaling/bench_real_multiome_pipeline_scaling.py"),
            "grn_scaling": _path_status(repo / "validation/scaling/bench_real_pbmc3k_grn_scaling.py"),
        },
        "lsf_scripts": {
            "full_pipeline": _path_status(repo / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline.lsf"),
            "full_pipeline_scaling": _path_status(repo / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline_scaling.lsf"),
            "grn_scaling": _path_status(repo / "validation/hpc/minerva/run_real_pbmc3k_grn_scaling.lsf"),
        },
        "hpc_tools": {
            "prepare_data": _path_status(repo / "validation/hpc/minerva/prepare_real_pbmc3k_data.py"),
            "collector": _path_status(repo / "validation/hpc/minerva/collect_benchmark_results.py"),
            "validator": _path_status(repo / "validation/hpc/minerva/validate_benchmark_artifact.py"),
        },
        "thread_env": _thread_env_state(),
    }

    failures: list[str] = []
    if args.require_thread_pins:
        failures.extend(_thread_env_failures(checks["thread_env"]))
    for key in ("repo", "env", "python", "data_dir"):
        if not checks[key]["exists"]:
            failures.append(f"missing {key}: {checks[key]['path']}")
    for name, status in checks["data_files"].items():
        if not status["exists"]:
            failures.append(f"missing data file {name}: {status['path']}")
    for group in ("benchmark_scripts", "lsf_scripts", "hpc_tools"):
        for name, status in checks[group].items():
            if not status["exists"]:
                failures.append(f"missing {group}.{name}: {status['path']}")

    if repo.exists():
        checks["git"] = _git_state(repo)
        if checks["git"]["status_error"]:
            failures.append(f"git status failed: {checks['git']['status_error']}")
        if checks["git"].get("untracked_error"):
            failures.append(f"git untracked status failed: {checks['git']['untracked_error']}")
        if args.require_clean and checks["git"].get("tracked_source_count", 0):
            sample = ", ".join(checks["git"].get("tracked_source_sample", [])[:5])
            failures.append(f"tracked source files are dirty: {sample}")
        if args.require_clean and checks["git"].get("untracked_source_count", 0):
            sample = ", ".join(checks["git"].get("untracked_source_sample", [])[:5])
            failures.append(f"untracked source files are present: {sample}")

    if python.exists() and repo.exists():
        checks["import"] = _import_state(python, repo)
        if not checks["import"]["ok"]:
            import_error = (
                checks["import"]["stderr"]
                or checks["import"]["extension_error"]
                or checks["import"]["parse_error"]
                or "unknown import error"
            )
            failures.append(f"rustscenic import failed: {import_error}")
        if args.require_repo_import and checks["import"]["package_under_repo"] is not True:
            failures.append(
                "rustscenic package is not imported from repo: "
                f"{checks['import']['package_file']}"
            )
        if args.require_repo_import and checks["import"]["extension_under_repo"] is not True:
            failures.append(
                "rustscenic extension is not imported from repo: "
                f"{checks['import']['extension_file']}"
            )
        package_version = checks["import"].get("package_version")
        extension_version = checks["import"].get("extension_version")
        if not package_version:
            failures.append("rustscenic package version missing")
        if not extension_version:
            failures.append("rustscenic extension version missing")
        if package_version and extension_version and package_version != extension_version:
            failures.append(
                "rustscenic package/extension version mismatch: "
                f"{package_version} != {extension_version}"
            )
        checks["backend"] = _backend_state(python, repo)
        if not checks["backend"].get("ok"):
            backend_error = (
                checks["backend"].get("extension_error")
                or checks["backend"].get("parse_error")
                or checks["backend"].get("stderr")
                or "missing Rust backend symbols"
            )
            failures.append(f"rustscenic Rust backend incomplete: {backend_error}")
        missing_symbols = checks["backend"].get("missing_symbols") or []
        if missing_symbols:
            failures.append(
                "missing Rust backend symbols: "
                + ", ".join(str(symbol) for symbol in missing_symbols)
            )

    checks["ok"] = not failures
    checks["failures"] = failures
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_REPO / "validation/real_multiome_v036")
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--require-thread-pins",
        action="store_true",
        help=(
            "Fail unless Rayon is pinned to the LSF core count and BLAS/OpenMP "
            "libraries are single-threaded."
        ),
    )
    parser.add_argument(
        "--require-repo-import",
        action="store_true",
        help=(
            "Fail unless rustscenic.__file__ resolves under --repo. Use this for "
            "benchmark jobs to avoid stale site-packages imports."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = preflight(args)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload)
    print(payload, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
