from __future__ import annotations

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
