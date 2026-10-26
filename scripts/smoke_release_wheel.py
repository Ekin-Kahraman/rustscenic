#!/usr/bin/env python3
"""Qualify a built wheel in an isolated environment without importing checkout code."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import venv


QUALIFICATION = r"""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import rustscenic
from rustscenic import _rustscenic, aucell, grn

expected_version, environment = sys.argv[1:]
environment = Path(environment).resolve()
for module in [rustscenic, _rustscenic, aucell, grn]:
    assert Path(module.__file__).resolve().is_relative_to(environment), module.__file__
assert importlib.metadata.version("rustscenic") == expected_version
assert rustscenic.__version__ == _rustscenic.__version__ == expected_version

X = np.random.default_rng(7).random((64, 16), dtype=np.float32)
X[X < 0.4] = 0
genes = [f"g{i}" for i in range(X.shape[1])]
regulons = {"R": ["g0", "g1", "g2"]}
dense = aucell.score((X, genes), regulons, top_frac=0.2)
sparse_result = aucell.score((sparse.csr_matrix(X), genes), regulons, top_frac=0.2)
np.testing.assert_array_equal(dense.to_numpy(), sparse_result.to_numpy())
assert dense.shape == (64, 1) and np.isfinite(dense.to_numpy()).all()
assert ((dense.to_numpy() >= 0) & (dense.to_numpy() <= 1)).all()
assert (dense.to_numpy() > 0).any()
assert dense.attrs["regulon_coverage"] == {"R": (3, 3)}
for matrix in [X, sparse.csr_matrix(X), sparse.csc_matrix(X)]:
    try:
        aucell.score((matrix, genes[:-1]), regulons, top_frac=0.2)
    except ValueError as exc:
        assert "gene_names length" in str(exc), str(exc)
    else:
        raise AssertionError("built wheel accepted malformed AUCell gene axis")
try:
    aucell.score((X[0], genes), regulons, top_frac=0.2)
except ValueError as exc:
    assert "two-dimensional" in str(exc), str(exc)
else:
    raise AssertionError("built wheel accepted a one-dimensional AUCell matrix")

edges = grn.infer((X, genes), ["g0", "g1"], n_estimators=3,
                  early_stop_window=0, seed=777, learning_rate=0.05,
                  max_features=0.75, max_depth=2, top_targets_per_tf=2,
                  target_block_size=32, verbose=False)
settings = edges.attrs["grn_fit"]
required = {"seed", "learning_rate", "max_features", "max_depth",
            "top_targets_per_tf", "min_importance", "target_block_size_requested",
            "target_block_size_effective"}
assert required <= settings.keys(), settings
assert settings["seed"] == 777 and settings["max_depth"] == 2
assert settings["target_block_size_requested"] == 32
assert settings["target_block_size_effective"] is None
assert settings["target_count"] == len(genes)
assert settings["fitted_trees_min"] == settings["fitted_trees_max"] == 3
json.dumps(settings, allow_nan=False)
print(json.dumps({"version": expected_version, "python": sys.version.split()[0],
                  "installed_package": rustscenic.__file__,
                  "installed_native": _rustscenic.__file__,
                  "dense_sparse_scores_equal": True, "invalid_axes_rejected": True,
                  "activity_nonzero_values": int(np.count_nonzero(dense.to_numpy())),
                  "grn_edges": len(edges), "grn_provenance_keys": sorted(required)}))
"""


def wheel_version(wheel: Path) -> str:
    match = re.fullmatch(
        r"rustscenic-([0-9]+\.[0-9]+\.[0-9]+)-[^-]+-[^-]+-[^-]+\.whl", wheel.name
    )
    if not match:
        raise ValueError(f"unexpected RustScenic wheel filename: {wheel.name}")
    return match.group(1)


def select_linux_wheel(dist: Path) -> Path:
    wheels = [
        path
        for path in dist.rglob("rustscenic-*.whl")
        if "manylinux" in path.name.rsplit("-", 1)[-1]
        and "x86_64" in path.name.rsplit("-", 1)[-1]
    ]
    if len(wheels) != 1:
        raise ValueError(f"expected one Linux x86_64 wheel, found {len(wheels)}")
    return wheels[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dist", type=Path, help="Select the Linux x86_64 release wheel"
    )
    source.add_argument(
        "--wheel", type=Path, help="Explicit compatible wheel for local qualification"
    )
    args = parser.parse_args(argv)
    try:
        wheel = (
            args.wheel if args.wheel is not None else select_linux_wheel(args.dist)
        ).resolve()
        version = wheel_version(wheel)
        with tempfile.TemporaryDirectory(prefix="rustscenic-wheel-smoke-") as directory:
            environment = Path(directory) / "venv"
            # Match `python -m venv`: symlink on POSIX, copy on Windows. This
            # also preserves relocatable Python runtimes' library resolution.
            venv.EnvBuilder(with_pip=True, symlinks=sys.platform != "win32").create(
                environment
            )
            python = environment / (
                "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
            )
            subprocess.run(
                [
                    str(python),
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    str(wheel),
                ],
                cwd=directory,
                check=True,
            )
            subprocess.run(
                [str(python), "-I", "-c", QUALIFICATION, version, str(environment)],
                cwd=directory,
                check=True,
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"wheel qualification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
