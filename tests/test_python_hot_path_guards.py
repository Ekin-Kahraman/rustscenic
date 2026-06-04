from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.python_hot_paths import scan_python_hot_paths


def test_package_hot_paths_do_not_reintroduce_pandas_table_work():
    """Keep scale-sensitive table operations out of Python package code.

    RustScenic can still use pandas for public input/output tables. The
    expensive joins, groupbys, concats, sorts and sparse densification paths
    should not creep back into package compute code now that the core stages
    are Rust-backed.
    """
    violations = scan_python_hot_paths()

    assert not violations, (
        "Python hot-path table work detected. Move the operation to Rust or "
        "add a narrowly justified allowlist entry:\n" + "\n".join(violations)
    )


def test_hot_path_scan_checks_nested_package_modules(tmp_path):
    package = tmp_path / "rustscenic"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (nested / "__init__.py").write_text("")
    (nested / "stage.py").write_text(
        "def bad(left, right):\n"
        "    return left.merge(right, on='gene')\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == [
        "nested/stage.py:2: return left.merge(right, on='gene')"
    ]
