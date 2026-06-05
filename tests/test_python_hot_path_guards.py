from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import python_hot_paths
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


def test_hot_path_scan_rejects_python_row_iteration_and_apply(tmp_path):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "stage.py").write_text(
        "def bad(df):\n"
        "    rows = [row for _, row in df.iterrows()]\n"
        "    return df.apply(lambda col: col)\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == [
        "stage.py:2: rows = [row for _, row in df.iterrows()]",
        "stage.py:3: return df.apply(lambda col: col)",
    ]


def test_hot_path_scan_rejects_concat_outside_allowlist(tmp_path):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "stage.py").write_text(
        "def bad(frames):\n"
        "    return pd.concat(frames, ignore_index=True)\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == [
        "stage.py:2: return pd.concat(frames, ignore_index=True)"
    ]


def test_hot_path_scan_pipeline_allowlist_is_exact_line_only(tmp_path):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "pipeline.py").write_text(
        "def attach(obs, auc_obs, extra):\n"
        "    adata_rna.obs = pd.concat([obs, auc_obs], axis=1, copy=False)\n"
        "    return pd.concat(extra, ignore_index=True)\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == [
        "pipeline.py:3: return pd.concat(extra, ignore_index=True)"
    ]


def test_hot_path_scan_rejects_unapproved_dataframe_take(tmp_path):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "stage.py").write_text(
        "def bad(df, row_ix):\n"
        "    return df.take(row_ix, axis=0)\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == [
        "stage.py:2: return df.take(row_ix, axis=0)"
    ]


def test_hot_path_scan_rejects_unapproved_loc_and_iloc_projections(tmp_path):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "stage.py").write_text(
        "def bad(df, keep, rows):\n"
        "    projected = df.loc[:, keep]\n"
        "    return projected.iloc[rows]\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == [
        "stage.py:2: projected = df.loc[:, keep]",
        "stage.py:3: return projected.iloc[rows]",
    ]


def test_hot_path_scan_allows_rust_index_projection_boundaries(tmp_path):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "cistarget.py").write_text(
        "def ok(enriched, row_ix):\n"
        "    out = enriched.take(row_ix, axis=0).reset_index(drop=True)\n"
        "    return out\n"
    )
    (package / "enhancer.py").write_text(
        "def ok(peak_coords, row_ix):\n"
        "    out = peak_coords.take(row_ix, axis=0)\n"
        "    genes_in_rna = genes.iloc[matched_gene_rows].reset_index(drop=True)\n"
        "    return out\n"
    )
    (package / "pipeline.py").write_text(
        "def ok(df, enriched_with_peaks, keep):\n"
        "    df = df.loc[:, keep]\n"
        "    return all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes.iloc[1:])\n"
    )

    violations = scan_python_hot_paths(package)

    assert violations == []


def test_hot_path_cli_prints_json_state_for_hpc_preflight(tmp_path, capsys):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")

    rc = python_hot_paths.main([str(package), "--json"])

    out = capsys.readouterr().out
    assert rc == 0
    assert '"ok": true' in out
    assert '"violation_count": 0' in out


def test_hot_path_cli_fails_on_violations(tmp_path, capsys):
    package = tmp_path / "rustscenic"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "bad.py").write_text(
        "def bad(left, right):\n"
        "    return left.merge(right, on='gene')\n"
    )

    rc = python_hot_paths.main([str(package)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "fail: 1 Python hot-path violations" in out
    assert "bad.py:2: return left.merge(right, on='gene')" in out
