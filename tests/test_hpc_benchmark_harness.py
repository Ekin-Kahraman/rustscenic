from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import types
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.backend_requirements import (
    REQUIRED_RUST_BACKEND_SYMBOLS,
    backend_capabilities as shared_backend_capabilities,
)
from validation.repo_cleanliness import (
    git_status_paths,
    repo_state_from_git_outputs,
    untracked_source_paths,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_multiome_harness_sets_motif_index_for_feather(tmp_path):
    module = _load_module(
        "bench_real_multiome_pipeline",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    path = tmp_path / "rankings.feather"
    pd.DataFrame(
        {
            "motifs": ["m1", "m2"],
            "GATA3": [0, 1],
            "SPI1": [1, 0],
        }
    ).to_feather(path)

    loaded = module.load_optional_table(path)

    assert list(loaded.index) == ["m1", "m2"]
    assert list(loaded.columns) == ["GATA3", "SPI1"]


def test_real_multiome_harness_strips_common_regulon_suffixes():
    module = _load_module(
        "bench_real_multiome_pipeline",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )

    assert module.strip_regulon_name("SPI1_regulon") == "SPI1"
    assert module.strip_regulon_name("PAX5_extended(+)") == "PAX5"
    assert module.strip_regulon_name("IRF8_repressor(-)") == "IRF8"


def test_real_multiome_harness_fingerprints_reference_dataframes():
    module = _load_module(
        "bench_real_multiome_fingerprint",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    df = pd.DataFrame(
        np.arange(36, dtype=np.int32).reshape(6, 6),
        index=[f"m{i}" for i in range(6)],
        columns=[f"g{i}" for i in range(6)],
    )

    first = module.dataframe_fingerprint(df, sample=2)
    second = module.dataframe_fingerprint(df.copy(), sample=2)
    changed = module.dataframe_fingerprint(df.rename(columns={"g0": "changed"}), sample=2)

    assert first == second
    assert first["shape"] == [6, 6]
    assert first["index_sample"] == ["m0", "m1"]
    assert first["column_sample"] == ["g0", "g1"]
    assert first["dtype_counts"] == {"int32": 6}
    assert len(first["corner_sample_sha256"]) == 64
    assert changed["corner_sample_sha256"] != first["corner_sample_sha256"]


def test_real_multiome_harness_builds_compact_output_summaries(tmp_path):
    module = _load_module(
        "bench_real_multiome_output_summaries",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    grn_path = tmp_path / "grn.parquet"
    cistarget_path = tmp_path / "cistarget.parquet"
    enhancer_path = tmp_path / "enhancer.parquet"
    eregulon_path = tmp_path / "eregulons.parquet"
    regulons_path = tmp_path / "regulons.json"
    pd.DataFrame(
        {
            "TF": ["TF2", "TF1"],
            "target": ["G2", "G1"],
            "importance": [0.2, 0.9],
        }
    ).to_parquet(grn_path)
    pd.DataFrame(
        {
            "regulon": ["TF1_regulon", "TF2_regulon"],
            "motif": ["m1", "m2"],
            "auc": [0.5, 0.8],
            "nes": [3.1, 4.2],
        }
    ).to_parquet(cistarget_path)
    pd.DataFrame(
        {
            "peak_id": ["p1", "p2"],
            "gene": ["G1", "G2"],
            "correlation": [-0.9, 0.1],
            "distance": [10, 20],
        }
    ).to_parquet(enhancer_path)
    pd.DataFrame(
        {
            "tf": ["TF1", "TF2"],
            "enhancer": ["p1", "p2"],
            "target_gene": ["G1", "G2"],
            "n_enhancer_links": [1, 3],
            "motif_auc": [0.9, 0.7],
        }
    ).to_parquet(eregulon_path)
    regulons_path.write_text(json.dumps({"TF1_regulon": ["G1"], "TF2_regulon": ["G2"]}))
    result = SimpleNamespace(
        grn_path=grn_path,
        cistarget_path=cistarget_path,
        enhancer_links_path=enhancer_path,
        eregulons_path=eregulon_path,
        regulons_path=regulons_path,
    )

    summaries = module.output_summaries(result, n=1)

    assert summaries["active_regulons_sample"] == ["TF1_regulon"]
    assert summaries["top_grn_edges"] == [{"TF": "TF1", "target": "G1", "importance": 0.9}]
    assert summaries["top_cistarget_rows"][0]["motif"] == "m2"
    assert summaries["top_enhancer_links"][0]["peak_id"] == "p1"
    assert summaries["top_enhancer_links"][0]["abs_correlation"] == 0.9
    assert summaries["top_eregulon_rows"][0]["tf"] == "TF1"


def test_real_multiome_harness_does_not_reread_pipeline_outputs_for_counts():
    source = (ROOT / "validation/scaling/bench_real_multiome_pipeline.py").read_text()

    assert "read_parquet_if_present" not in source
    assert "pd.read_parquet(result." not in source
    assert "result.n_grn_edges" in source
    assert "result.aucell_shape" in source
    assert '"end_to_end": round(end_to_end_wall, 3)' in source
    assert '"setup_elapsed_s"' in source


def test_real_multiome_harness_configures_cpu_thread_env(monkeypatch):
    module = _load_module(
        "bench_real_multiome_thread_env",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)

    module.configure_thread_env(6)

    assert os.environ["RAYON_NUM_THREADS"] == "6"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_benchmark_thread_env_overrides_inherited_blas_settings(monkeypatch):
    modules = [
        _load_module(
            "bench_real_multiome_thread_env_override",
            ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
        ),
        _load_module(
            "bench_real_multiome_scaling_thread_env_override",
            ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
        ),
        _load_module(
            "bench_real_pbmc3k_grn_thread_env_override",
            ROOT / "validation/scaling/bench_real_pbmc3k_grn_scaling.py",
        ),
    ]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "8")

    for module in modules:
        module.configure_thread_env(3)
        assert os.environ["RAYON_NUM_THREADS"] == "3"
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            monkeypatch.setenv(key, "8")


def test_real_multiome_harness_rejects_invalid_args(tmp_path):
    module = _load_module(
        "bench_real_multiome_args",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    common = [
        "--dataset-name",
        "pbmc",
        "--rna-10x-h5",
        str(tmp_path / "rna.h5"),
        "--fragments",
        str(tmp_path / "fragments.tsv.gz"),
        "--peaks",
        str(tmp_path / "peaks.bed"),
        "--out-dir",
        str(tmp_path / "outputs"),
        "--out-json",
        str(tmp_path / "run.json"),
    ]

    for extra, message in (
        (["--dataset-name", ""], "--dataset-name must be non-empty"),
        (["--n-cells", "0"], "--n-cells must be positive"),
        (["--threads", "0"], "--threads must be positive"),
        (["--grn-target-block-size", "0"], "--grn-target-block-size must be positive"),
        (["--grn-max-features", "1.5"], "--grn-max-features must be in"),
        (["--cistarget-top-frac", "0"], "--cistarget-top-frac must be in"),
        (["--cistarget-auc-threshold", "-1"], "--cistarget-auc-threshold must be non-negative"),
        (["--enhancer-max-distance", "-1"], "--enhancer-max-distance must be non-negative"),
        (["--summary-rows", "0"], "--summary-rows must be positive"),
    ):
        args = module.parse_args([*common, *extra])
        try:
            module.validate_args(args)
        except SystemExit as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected invalid args to fail: {extra}")


def test_real_multiome_scaling_child_cmd_runs_full_pipeline_in_child(tmp_path):
    module = _load_module(
        "bench_real_multiome_pipeline_scaling",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    args = module.parse_args(
        [
            "--dataset-name",
            "pbmc",
            "--rna-10x-h5",
            str(tmp_path / "rna.h5"),
            "--fragments",
            str(tmp_path / "fragments.tsv.gz"),
            "--peaks",
            str(tmp_path / "peaks.bed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--out-json",
            str(tmp_path / "aggregate.json"),
            "--run-prefix",
            "fixture",
            "--cell-counts",
            "100",
            "200",
            "--threads",
            "4",
            "--expected-tfs",
            "SPI1",
            "PAX5",
            "--require-clean",
        ]
    )

    cmd = module.child_cmd(
        args,
        n_cells=100,
        run_id="fixture_cells100",
        out_dir=tmp_path / "runs" / "fixture_cells100_outputs",
        out_json=tmp_path / "runs" / "fixture_cells100.json",
    )

    assert cmd[0] == sys.executable
    assert cmd[1].endswith("bench_real_multiome_pipeline.py")
    assert "--n-cells" in cmd
    assert cmd[cmd.index("--n-cells") + 1] == "100"
    assert "--expected-tfs" in cmd
    assert "SPI1" in cmd
    assert "--require-clean" in cmd


def test_real_multiome_scaling_rejects_ambiguous_cell_counts(tmp_path):
    module = _load_module(
        "bench_real_multiome_pipeline_scaling_args",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    common = [
        "--dataset-name",
        "pbmc",
        "--rna-10x-h5",
        str(tmp_path / "rna.h5"),
        "--fragments",
        str(tmp_path / "fragments.tsv.gz"),
        "--peaks",
        str(tmp_path / "peaks.bed"),
        "--out-root",
        str(tmp_path / "runs"),
        "--out-json",
        str(tmp_path / "aggregate.json"),
    ]

    for extra, message in (
        (["--cell-counts", "200", "100"], "sorted ascending"),
        (["--cell-counts", "100", "100"], "duplicates"),
        (["--cell-counts", "0", "100"], "positive integers"),
        (["--threads", "0"], "--threads must be positive"),
        (["--grn-max-features", "1.5"], "--grn-max-features must be in"),
    ):
        args = module.parse_args([*common, *extra])
        try:
            module.validate_args(args)
        except SystemExit as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected invalid args to fail: {extra}")


def test_real_multiome_scaling_refuses_stale_child_outputs(tmp_path):
    module = _load_module(
        "bench_real_multiome_pipeline_scaling_stale",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    out_json = tmp_path / "fixture_cells100.json"
    out_dir = tmp_path / "fixture_cells100_outputs"
    out_dir.mkdir()
    out_json.write_text("{}")
    (out_dir / "stale.parquet").write_text("old\n")

    try:
        module._ensure_fresh_child_outputs(out_json, out_dir, force=False)
    except RuntimeError as exc:
        assert "refusing to reuse existing" in str(exc)
        assert str(out_json) in str(exc)
        assert str(out_dir) in str(exc)
    else:
        raise AssertionError("expected stale outputs to fail without force")

    module._ensure_fresh_child_outputs(out_json, out_dir, force=True)

    assert not out_json.exists()
    assert not out_dir.exists()


def test_real_multiome_scaling_aggregate_payload_has_slopes(tmp_path, monkeypatch):
    module = _load_module(
        "bench_real_multiome_pipeline_scaling_payload",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    args = module.parse_args(
        [
            "--dataset-name",
            "pbmc",
            "--rna-10x-h5",
            str(tmp_path / "rna.h5"),
            "--fragments",
            str(tmp_path / "fragments.tsv.gz"),
            "--peaks",
            str(tmp_path / "peaks.bed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--out-json",
            str(tmp_path / "aggregate.json"),
            "--cell-counts",
            "100",
            "200",
            "--expected-tfs",
            "SPI1",
            "PAX5",
        ]
    )
    monkeypatch.setattr(module, "repo_state", _clean_repo_state)
    monkeypatch.setattr(module, "runtime_import_state", _runtime_import_state)
    monkeypatch.setattr(module, "backend_capabilities", _backend_capabilities)
    runs = [
        {
            "n_cells_requested": 100,
            "n_cells_actual": 100,
            "json_path": str(tmp_path / "one.json"),
            "output_dir": str(tmp_path / "one"),
            "wall_s": {"setup": 1.0, "pipeline": 2.0, "end_to_end": 3.0},
            "peak_rss_gb": 1.0,
            "setup_peak_rss_gb": 0.8,
            "elapsed_per_stage": {},
            "peak_rss_gb_per_stage": {},
            "outputs": {"grn_edges": 1},
        },
        {
            "n_cells_requested": 200,
            "n_cells_actual": 200,
            "json_path": str(tmp_path / "two.json"),
            "output_dir": str(tmp_path / "two"),
            "wall_s": {"setup": 1.0, "pipeline": 4.0, "end_to_end": 6.0},
            "peak_rss_gb": 1.5,
            "setup_peak_rss_gb": 1.0,
            "elapsed_per_stage": {},
            "peak_rss_gb_per_stage": {},
            "outputs": {"grn_edges": 1},
        },
    ]

    payload = module.aggregate_payload(args, runs)

    assert payload["benchmark"] == "real_multiome_full_pipeline_scaling"
    assert payload["runs"] == runs
    assert payload["params"]["expected_tfs"] == ["SPI1", "PAX5"]
    assert payload["scaling"]["pipeline_wall_slope_vs_cells"] == 1.0
    assert payload["scaling"]["end_to_end_wall_slope_vs_cells"] == 1.0
    assert payload["scaling"]["peak_rss_slope_vs_cells"] > 0


def test_real_multiome_scaling_coordinator_validates_final_aggregate(tmp_path, monkeypatch):
    module = _load_module(
        "bench_real_multiome_pipeline_scaling_coordinator",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    args = module.parse_args(
        [
            "--dataset-name",
            "pbmc",
            "--rna-10x-h5",
            str(tmp_path / "rna.h5"),
            "--fragments",
            str(tmp_path / "fragments.tsv.gz"),
            "--peaks",
            str(tmp_path / "peaks.bed"),
            "--out-root",
            str(tmp_path / "runs"),
            "--out-json",
            str(tmp_path / "aggregate.json"),
            "--cell-counts",
            "100",
        ]
    )
    monkeypatch.setattr(module, "repo_state", _clean_repo_state)
    monkeypatch.setattr(module, "runtime_import_state", _runtime_import_state)
    monkeypatch.setattr(module, "backend_capabilities", _backend_capabilities)
    monkeypatch.setattr(
        module,
        "run_child",
        lambda _args, *, n_cells: {
            "n_cells_requested": n_cells,
            "n_cells_actual": n_cells,
            "json_path": str(tmp_path / "child.json"),
            "output_dir": str(tmp_path / "child_outputs"),
            "wall_s": {"setup": 1.0, "pipeline": 2.0, "end_to_end": 3.0},
            "peak_rss_gb": 1.0,
            "setup_peak_rss_gb": 0.8,
            "elapsed_per_stage": {"grn": 1.0},
            "peak_rss_gb_per_stage": {"grn": 1.0},
            "outputs": {"grn_edges": 1},
        },
    )
    calls = []

    def fake_validate_record(payload, *, require_clean, check_output_files):
        calls.append(
            {
                "benchmark": payload["benchmark"],
                "require_clean": require_clean,
                "check_output_files": check_output_files,
                "runs": len(payload["runs"]),
            }
        )
        return []

    monkeypatch.setattr(module, "validate_record", fake_validate_record)

    payload = module.coordinator(args)

    assert payload["benchmark"] == "real_multiome_full_pipeline_scaling"
    assert json.loads(args.out_json.read_text())["benchmark"] == payload["benchmark"]
    assert calls == [
        {
            "benchmark": "real_multiome_full_pipeline_scaling",
            "require_clean": False,
            "check_output_files": True,
            "runs": 1,
        }
    ]


def test_benchmark_harness_records_rust_backend_capabilities():
    full = _load_module(
        "bench_real_multiome_backend",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    full_scaling = _load_module(
        "bench_real_multiome_scaling_backend",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    grn = _load_module(
        "bench_real_pbmc3k_grn_backend",
        ROOT / "validation/scaling/bench_real_pbmc3k_grn_scaling.py",
    )

    full_backend = full.backend_capabilities()
    full_scaling_backend = full_scaling.backend_capabilities()
    grn_backend = grn.backend_capabilities()

    for backend in (full_backend, full_scaling_backend, grn_backend):
        assert backend["ok"] is True
        assert backend["missing_symbols"] == []
        assert "pipeline_project_ranking_columns" in backend["required_symbols"]["pipeline"]
        assert "enhancer_link_pearson_sparse_rna" in backend["required_symbols"]["enhancer"]


def test_backend_requirement_contract_is_shared_across_hpc_scripts():
    full = _load_module(
        "bench_real_multiome_shared_backend",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    full_scaling = _load_module(
        "bench_real_multiome_scaling_shared_backend",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )
    grn = _load_module(
        "bench_real_pbmc3k_grn_shared_backend",
        ROOT / "validation/scaling/bench_real_pbmc3k_grn_scaling.py",
    )
    preflight = _load_module(
        "preflight_minerva_shared_backend",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    validator = _load_module(
        "validate_benchmark_shared_backend",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )

    assert full.backend_capabilities is shared_backend_capabilities
    assert full_scaling.backend_capabilities is shared_backend_capabilities
    assert grn.backend_capabilities is shared_backend_capabilities
    assert preflight.REQUIRED_RUST_BACKEND_SYMBOLS is REQUIRED_RUST_BACKEND_SYMBOLS
    assert validator.REQUIRED_BACKEND_SYMBOLS == {
        stage: set(symbols)
        for stage, symbols in REQUIRED_RUST_BACKEND_SYMBOLS.items()
    }


def test_backend_requirement_contract_covers_package_rust_imports():
    imported_symbols: set[str] = set()
    for path in (ROOT / "python/rustscenic").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "rustscenic._rustscenic":
                continue
            imported_symbols.update(
                alias.name
                for alias in node.names
                if alias.name != "__version__"
            )

    required_symbols = {
        symbol
        for symbols in REQUIRED_RUST_BACKEND_SYMBOLS.values()
        for symbol in symbols
    }

    assert imported_symbols == required_symbols


def test_repo_state_marks_tracked_dirty_without_untracked_noise(monkeypatch):
    module = _load_module(
        "bench_real_multiome_pipeline",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )

    def fake_git(args):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["status", "--short", "--untracked-files=no"]:
            return " M python/rustscenic/grn.py"
        if args == ["status", "--short", "--untracked-files=all"]:
            return " M python/rustscenic/grn.py\n?? scratch.json"
        if args == ["diff", "HEAD", "--binary", "--no-ext-diff"]:
            return "diff --git a/python/rustscenic/grn.py b/python/rustscenic/grn.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git_output", fake_git)

    state = module.repo_state()

    assert state["commit"] == "abc123"
    assert state["tracked_dirty"] is True
    assert state["tracked_source_dirty"] is True
    assert state["source_dirty"] is True
    assert state["tracked_status_short"] == [" M python/rustscenic/grn.py"]
    assert state["tracked_diff_sha256"]
    assert state["tracked_diff_bytes"] > 0
    assert state["tracked_source_count"] == 1
    assert state["tracked_source_sample"] == ["python/rustscenic/grn.py"]
    assert state["untracked_count"] == 1
    assert state["untracked_sample"] == ["scratch.json"]
    assert state["untracked_source_count"] == 0
    assert state["untracked_source_sample"] == []


def test_repo_state_marks_untracked_source_dirty(monkeypatch):
    module = _load_module(
        "bench_real_multiome_pipeline_untracked_source",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )

    def fake_git(args):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["status", "--short", "--untracked-files=no"]:
            return ""
        if args == ["status", "--short", "--untracked-files=all"]:
            return "\n".join(
                [
                    "?? scratch.json",
                    "?? validation/hpc/minerva/preflight_minerva.py",
                    "?? outreach/local_note.md",
                ]
            )
        if args == ["diff", "HEAD", "--binary", "--no-ext-diff"]:
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git_output", fake_git)

    state = module.repo_state()

    assert state["tracked_dirty"] is False
    assert state["tracked_source_dirty"] is False
    assert state["source_dirty"] is True
    assert state["tracked_source_count"] == 0
    assert state["untracked_count"] == 3
    assert state["untracked_source_count"] == 1
    assert state["untracked_source_sample"] == [
        "validation/hpc/minerva/preflight_minerva.py"
    ]


def test_repo_state_ignores_tracked_non_source_docs_for_clean_benchmark(monkeypatch):
    module = _load_module(
        "bench_real_multiome_pipeline_tracked_docs",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )

    def fake_git(args):
        if args == ["rev-parse", "HEAD"]:
            return "abc123"
        if args == ["status", "--short", "--untracked-files=no"]:
            return " M manuscript/rustscenic_preprint.md"
        if args == ["status", "--short", "--untracked-files=all"]:
            return " M manuscript/rustscenic_preprint.md\n?? outreach/local_note.md"
        if args == ["diff", "HEAD", "--binary", "--no-ext-diff"]:
            return "diff --git a/manuscript/rustscenic_preprint.md b/manuscript/rustscenic_preprint.md\n"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git_output", fake_git)

    state = module.repo_state()

    assert state["tracked_dirty"] is True
    assert state["tracked_source_dirty"] is False
    assert state["source_dirty"] is False
    assert state["tracked_source_count"] == 0
    assert state["untracked_source_count"] == 0


def test_source_classifier_ignores_local_outreach_and_scratch():
    assert untracked_source_paths(
        [
            "scratch.json",
            "outreach/local_note.md",
            "validation/hpc/minerva/preflight_minerva.py",
            "python/rustscenic/_stage_utils.py",
            "crates/rustscenic-py/src/lib.rs",
        ]
    ) == [
        "validation/hpc/minerva/preflight_minerva.py",
        "python/rustscenic/_stage_utils.py",
        "crates/rustscenic-py/src/lib.rs",
    ]


def test_git_status_paths_handles_renames():
    assert git_status_paths(
        [
            "R  old.py -> python/rustscenic/new.py",
            " M validation/hpc/minerva/preflight_minerva.py",
        ]
    ) == [
        "old.py",
        "python/rustscenic/new.py",
        "validation/hpc/minerva/preflight_minerva.py",
    ]


def test_repo_cleanliness_state_builder_classifies_source_dirty():
    state = repo_state_from_git_outputs(
        commit="abc123",
        tracked_status="\n".join(
            [
                " M manuscript/rustscenic_preprint.md",
                " M python/rustscenic/grn.py",
            ]
        ),
        untracked_status="\n".join(
            [
                "?? scratch.json",
                "?? validation/hpc/minerva/preflight_minerva.py",
            ]
        ),
        tracked_diff="diff --git a/python/rustscenic/grn.py b/python/rustscenic/grn.py\n",
    )

    assert state["tracked_dirty"] is True
    assert state["tracked_source_dirty"] is True
    assert state["source_dirty"] is True
    assert state["tracked_source_sample"] == ["python/rustscenic/grn.py"]
    assert state["untracked_source_sample"] == [
        "validation/hpc/minerva/preflight_minerva.py"
    ]
    assert state["tracked_diff_sha256"]


def test_grn_scaling_rss_units_handle_darwin_and_linux(monkeypatch):
    module = _load_module(
        "bench_real_pbmc3k_grn_scaling",
        ROOT / "validation/scaling/bench_real_pbmc3k_grn_scaling.py",
    )

    monkeypatch.setattr(
        module.resource,
        "getrusage",
        lambda _who: types.SimpleNamespace(ru_maxrss=1024**3),
    )
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert module.peak_rss_gb() == 1.0

    monkeypatch.setattr(
        module.resource,
        "getrusage",
        lambda _who: types.SimpleNamespace(ru_maxrss=1024**2),
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert module.peak_rss_gb() == 1.0


def test_grn_scaling_rejects_ambiguous_or_invalid_args(tmp_path):
    module = _load_module(
        "bench_real_pbmc3k_grn_scaling_args",
        ROOT / "validation/scaling/bench_real_pbmc3k_grn_scaling.py",
    )
    common = [
        "--data-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "out.json"),
    ]

    for extra, message in (
        (["--subset-sizes", "200", "100"], "--subset-sizes must be sorted ascending"),
        (["--subset-sizes", "100", "100"], "--subset-sizes must not contain duplicates"),
        (["--subset-sizes", "0", "100"], "--subset-sizes must be positive integers"),
        (["--thread-counts", "4", "2"], "--thread-counts must be sorted ascending"),
        (["--thread-counts", "4", "4"], "--thread-counts must not contain duplicates"),
        (["--subset-threads", "0"], "--subset-threads must be positive"),
        (["--thread-cells", "0"], "--thread-cells must be positive"),
        (["--n-estimators", "0"], "--n-estimators must be positive"),
        (["--max-depth", "0"], "--max-depth must be positive"),
        (["--early-stop-window", "0"], "--early-stop-window must be positive"),
        (["--target-block-size", "0"], "--target-block-size must be positive"),
        (["--learning-rate", "0"], "--learning-rate must be positive"),
        (["--max-features", "1.5"], "--max-features must be in (0, 1]"),
        (["--subsample", "0"], "--subsample must be in (0, 1]"),
        (["--run-one", "--n-cells", "0"], "--n-cells must be positive"),
        (["--run-one", "--threads", "0"], "--threads must be positive"),
    ):
        args = module.parse_args([*common, *extra])
        try:
            module.validate_args(args)
        except SystemExit as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected invalid args to fail: {extra}")


def test_grn_scaling_require_clean_rejects_dirty_state(monkeypatch, tmp_path):
    module = _load_module(
        "bench_real_pbmc3k_grn_scaling",
        ROOT / "validation/scaling/bench_real_pbmc3k_grn_scaling.py",
    )
    args = module.parse_args(
        [
            "--data-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.json"),
            "--require-clean",
        ]
    )
    monkeypatch.setattr(
        module,
        "repo_state",
        lambda: {
            "commit": "abc123",
            "tracked_dirty": True,
            "tracked_source_dirty": True,
            "tracked_source_count": 1,
            "tracked_source_sample": ["python/rustscenic/grn.py"],
            "source_dirty": True,
            "tracked_status_short": [" M python/rustscenic/grn.py"],
            "untracked_count": 0,
            "untracked_sample": [],
            "untracked_source_count": 0,
            "untracked_source_sample": [],
        },
    )

    try:
        module.coordinator(args)
    except SystemExit as exc:
        assert "source files are dirty" in str(exc)
    else:
        raise AssertionError("dirty benchmark should fail under --require-clean")


def _write_minerva_preflight_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    data = repo / "validation" / "real_multiome_v036"
    hpc = repo / "validation" / "hpc" / "minerva"
    scaling = repo / "validation" / "scaling"
    env = tmp_path / "env"
    for path in (data, hpc, scaling, env / "bin"):
        path.mkdir(parents=True, exist_ok=True)
    for name in (
        "pbmc_3k_filtered_feature_bc_matrix.h5",
        "pbmc_3k_atac_fragments.tsv.gz",
        "pbmc_3k_atac_peaks.bed",
    ):
        (data / name).write_text("fixture\n")
    (scaling / "bench_real_multiome_pipeline.py").write_text("# fixture\n")
    (scaling / "bench_real_multiome_pipeline_scaling.py").write_text("# fixture\n")
    (scaling / "bench_real_pbmc3k_grn_scaling.py").write_text("# fixture\n")
    (hpc / "run_real_pbmc3k_full_pipeline.lsf").write_text("# fixture\n")
    (hpc / "run_real_pbmc3k_full_pipeline_scaling.lsf").write_text("# fixture\n")
    (hpc / "run_real_pbmc3k_grn_scaling.lsf").write_text("# fixture\n")
    (hpc / "prepare_real_pbmc3k_data.py").write_text("# fixture\n")
    (hpc / "collect_benchmark_results.py").write_text("# fixture\n")
    (hpc / "validate_benchmark_artifact.py").write_text("# fixture\n")
    (env / "bin" / "python").write_text("# fixture\n")
    return repo, env, data


def _preflight_backend_state():
    return {
        "ok": True,
        "extension_error": None,
        "required_symbols": _backend_capabilities()["required_symbols"],
        "missing_symbols": [],
        "parse_error": None,
        "stderr": "",
    }


def test_prepare_real_pbmc3k_data_skips_valid_existing_file(tmp_path):
    from validation.hpc.minerva import prepare_real_pbmc3k_data as module

    dest = tmp_path / "tiny.txt"
    dest.write_text("fixture\n")
    spec = module.DataFile(
        filename=dest.name,
        url="https://example.invalid/tiny.txt",
        size_bytes=dest.stat().st_size,
        sha256=module.sha256_file(dest),
    )

    result = module.download_file(spec, dest, force=False, timeout=0.01)

    assert result == {
        "filename": "tiny.txt",
        "status": "present",
        "path": str(dest),
    }


def test_prepare_real_pbmc3k_data_rejects_existing_hash_mismatch(tmp_path):
    from validation.hpc.minerva import prepare_real_pbmc3k_data as module

    dest = tmp_path / "tiny.txt"
    dest.write_text("wrong\n")
    spec = module.DataFile(
        filename=dest.name,
        url="https://example.invalid/tiny.txt",
        size_bytes=999,
        sha256="0" * 64,
    )

    try:
        module.download_file(spec, dest, force=False, timeout=0.01)
    except RuntimeError as exc:
        assert "does not match expected PBMC3k hash" in str(exc)
    else:
        raise AssertionError("mismatched existing file should fail")


def test_minerva_preflight_accepts_ready_clean_checkout(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
            "tracked_source_count": 0,
            "tracked_source_sample": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    monkeypatch.setattr(
        module,
        "_python_hot_path_state",
        lambda _repo: {
            "ok": True,
            "violation_count": 0,
            "violations": [],
            "package_dir": str(repo / "python" / "rustscenic"),
        },
    )
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-repo-import",
            "--require-clean",
            "--require-rust-hot-paths",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is True
    assert result["failures"] == []
    assert result["git"]["commit"] == "abc123"
    assert result["import"]["rustscenic_version"] == "0.4.7"
    assert result["backend"]["ok"] is True
    assert result["python_hot_paths"]["ok"] is True
    assert result["hpc_tools"]["prepare_data"]["exists"] is True
    assert result["hpc_tools"]["collector"]["exists"] is True
    assert result["hpc_tools"]["validator"]["exists"] is True


def test_minerva_preflight_rejects_python_hot_path_regression(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_hot_paths",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
            "tracked_source_count": 0,
            "tracked_source_sample": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    monkeypatch.setattr(
        module,
        "_python_hot_path_state",
        lambda _repo: {
            "ok": False,
            "violation_count": 1,
            "violations": ["pipeline.py:999: left.merge(right, on='gene')"],
            "package_dir": str(repo / "python" / "rustscenic"),
        },
    )
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-rust-hot-paths",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert (
        "Python hot-path table work detected: "
        "pipeline.py:999: left.merge(right, on='gene')"
        in result["failures"]
    )


def test_minerva_preflight_rejects_unpinned_thread_env(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_thread_pins",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
            "tracked_source_count": 0,
            "tracked_source_sample": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-thread-pins",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert result["thread_env"]["rayon_num_threads"] == "4"
    assert "OMP_NUM_THREADS must be 1" in result["failures"]
    assert "MKL_NUM_THREADS must be 1" in result["failures"]
    assert (
        "RAYON_NUM_THREADS must match LSB_DJOB_NUMPROC: 4 != 8"
        in result["failures"]
    )


def test_minerva_preflight_rejects_data_hash_mismatch(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_data_hashes",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
            "tracked_source_count": 0,
            "tracked_source_sample": [],
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-data-hashes",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert any(
        failure.startswith("data file hash mismatch pbmc_3k_filtered_feature_bc_matrix.h5")
        for failure in result["failures"]
    )


def test_minerva_preflight_rejects_dirty_tracked_checkout(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_dirty",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [" M python/rustscenic/grn.py"],
            "status_error": None,
            "tracked_dirty": True,
            "tracked_source_count": 1,
            "tracked_source_sample": ["python/rustscenic/grn.py"],
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-clean",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert "tracked source files are dirty: python/rustscenic/grn.py" in result["failures"]


def test_minerva_preflight_rejects_untracked_source_checkout(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_untracked_source",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
            "untracked_source_count": 1,
            "untracked_source_sample": ["validation/hpc/minerva/new_launcher.lsf"],
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-clean",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert (
        "untracked source files are present: "
        "validation/hpc/minerva/new_launcher.lsf"
        in result["failures"]
    )


def test_minerva_preflight_rejects_missing_rust_backend_symbol(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_missing_backend",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.7",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        module,
        "_backend_state",
        lambda _python, _repo: {
            "ok": False,
            "extension_error": None,
            "required_symbols": _backend_capabilities()["required_symbols"],
            "missing_symbols": ["pipeline.pipeline_project_ranking_columns"],
            "parse_error": None,
            "stderr": "",
        },
    )
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-repo-import",
            "--require-clean",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert "rustscenic Rust backend incomplete: missing Rust backend symbols" in result["failures"]
    assert (
        "missing Rust backend symbols: pipeline.pipeline_project_ranking_columns"
        in result["failures"]
    )


def test_minerva_preflight_rejects_package_extension_version_mismatch(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_version_mismatch",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.7",
            "package_version": "0.4.7",
            "extension_version": "0.4.6",
            "package_file": str(repo / "python" / "rustscenic" / "__init__.py"),
            "package_under_repo": True,
            "extension_file": str(repo / "python" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": True,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-repo-import",
            "--require-clean",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert (
        "rustscenic package/extension version mismatch: 0.4.7 != 0.4.6"
        in result["failures"]
    )


def test_minerva_preflight_rejects_stale_installed_package(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_stale_import",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_git_state",
        lambda _repo: {
            "commit": "abc123",
            "commit_error": None,
            "tracked_status_short": [],
            "status_error": None,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        module,
        "_import_state",
        lambda _python, _repo: {
            "ok": True,
            "python": str(env / "bin" / "python"),
            "rustscenic_version": "0.4.6",
            "package_version": "0.4.6",
            "extension_version": "0.4.6",
            "package_file": str(tmp_path / "site-packages" / "rustscenic" / "__init__.py"),
            "package_under_repo": False,
            "extension_file": str(tmp_path / "site-packages" / "rustscenic" / "_rustscenic.so"),
            "extension_under_repo": False,
            "extension_error": None,
            "parse_error": None,
            "stderr": "",
        },
    )
    monkeypatch.setattr(module, "_backend_state", lambda _python, _repo: _preflight_backend_state())
    args = module.parse_args(
        [
            "--repo", str(repo),
            "--env", str(env),
            "--data-dir", str(data),
            "--require-repo-import",
            "--require-clean",
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert result["failures"] == [
        "rustscenic package is not imported from repo: "
        f"{tmp_path / 'site-packages' / 'rustscenic' / '__init__.py'}",
        "rustscenic extension is not imported from repo: "
        f"{tmp_path / 'site-packages' / 'rustscenic' / '_rustscenic.so'}",
    ]


def _clean_repo_state():
    return {
        "commit": "abc123",
        "tracked_dirty": False,
        "tracked_source_dirty": False,
        "tracked_source_count": 0,
        "tracked_source_sample": [],
        "source_dirty": False,
        "tracked_status_short": [],
        "tracked_diff_sha256": None,
        "tracked_diff_bytes": 0,
        "untracked_count": 0,
        "untracked_sample": [],
        "untracked_source_count": 0,
        "untracked_source_sample": [],
    }


def _runtime_import_state():
    return {
        "package_version": "0.4.7",
        "extension_version": "0.4.7",
        "package_file": str(ROOT / "python" / "rustscenic" / "__init__.py"),
        "package_under_repo": True,
        "extension_file": str(ROOT / "python" / "rustscenic" / "_rustscenic.so"),
        "extension_under_repo": True,
        "extension_error": None,
    }


def _backend_capabilities():
    return {
        "ok": True,
        "extension_error": None,
        "required_symbols": deepcopy(REQUIRED_RUST_BACKEND_SYMBOLS),
        "missing_symbols": [],
    }


def _python_hot_paths_state():
    return {
        "package_dir": str(ROOT / "python" / "rustscenic"),
        "exists": True,
        "ok": True,
        "violation_count": 0,
        "violations": [],
        "allowed_hit_count": 6,
        "pattern_count": 40,
    }


def _backend_execution_state():
    return {
        "setup_fragments_to_matrix": {
            "engine": "rust",
            "symbols": ["preproc_fragments_to_matrix"],
        },
        "pipeline_topics": {"engine": "rust", "symbols": ["topics_fit"]},
        "pipeline_grn": {
            "engine": "rust",
            "symbols": ["gene_duplicate_summary", "grn_infer"],
        },
        "pipeline_candidate_regulons": {
            "engine": "rust",
            "symbols": ["pipeline_candidate_regulons_from_grn"],
        },
        "pipeline_cistarget": {
            "engine": "rust",
            "symbols": ["cistarget_enrichment_from_rankings_i32"],
        },
        "pipeline_enhancer": {
            "engine": "rust",
            "symbols": [
                "enhancer_align_cell_indices",
                "enhancer_match_gene_coords_to_rna",
                "enhancer_normalise_chrom_codes",
                "enhancer_prepare_gene_order",
                "enhancer_link_pearson_sparse_rna",
            ],
        },
        "pipeline_eregulon_peak_attribution": {
            "engine": "rust",
            "symbols": ["pipeline_attribute_peaks_to_cistarget_rows_f32"],
        },
        "pipeline_eregulons": {
            "engine": "rust",
            "symbols": ["eregulon_assemble_f32"],
        },
        "pipeline_aucell": {
            "engine": "rust",
            "symbols": [
                "gene_duplicate_summary",
                "stage_prepare_regulon_indices_with_coverage",
                "aucell_score_sparse_csr",
            ],
        },
    }


def _full_pipeline_record(tmp_path: Path):
    artefact_names = {
        "atac_matrix_path": "atac_cells_by_peaks.h5ad",
        "grn_path": "grn.parquet",
        "regulons_path": "regulons.json",
        "candidate_regulons_path": "candidate_regulons.json",
        "aucell_path": "aucell.parquet",
        "cistarget_path": "cistarget_enriched.parquet",
        "enhancer_links_path": "enhancer_links.parquet",
        "eregulons_path": "eregulons.parquet",
        "integrated_adata_path": "rna_with_regulons.h5ad",
    }
    output_inventory = {}
    for key, filename in artefact_names.items():
        path = tmp_path / filename
        path.write_text(f"{key}\n")
        output_inventory[key] = {
            "path": str(path),
            "exists": True,
            "type": "file",
            "size_bytes": path.stat().st_size,
        }
    topics_dir = tmp_path / "topics"
    topics_dir.mkdir(exist_ok=True)
    (topics_dir / "cell_topic.parquet").write_text("fixture\n")
    output_inventory["topics_dir"] = {
        "path": str(topics_dir),
        "exists": True,
        "type": "dir",
        "entries": len(list(topics_dir.iterdir())),
    }
    return {
        "benchmark": "real_multiome_full_pipeline",
        "dataset_name": "pbmc3k",
        "repo_state": _clean_repo_state(),
        "runtime_import": _runtime_import_state(),
        "backend_capabilities": _backend_capabilities(),
        "python_hot_paths": _python_hot_paths_state(),
        "backend_execution": _backend_execution_state(),
        "rustscenic": "0.4.7",
        "input_hashes": {
            "rna_10x_h5_md5": "a",
            "fragments_first_8mb_md5": "b",
            "peaks_bed_md5": "c",
        },
        "reference_fingerprints": {
            "motif_rankings": {
                "shape": [50, 2000],
                "index_name": None,
                "index_sample": ["motif_a"],
                "column_sample": ["gene_a"],
                "dtype_counts": {"int32": 2000},
                "corner_sample_sha256": "a" * 64,
            },
            "gene_coords": {
                "shape": [1000, 3],
                "index_name": None,
                "index_sample": ["0"],
                "column_sample": ["gene"],
                "dtype_counts": {"object": 2, "int64": 1},
                "corner_sample_sha256": "b" * 64,
            },
        },
        "params": {
            "seed": 777,
            "n_cells_requested": 100,
            "threads": 4,
            "rayon_num_threads": 4,
        },
        "shapes": {
            "rna_post_qc": [100, 1000],
            "atac_shared_cells": [100, 2000],
            "motif_rankings": [50, 2000],
            "gene_coords_rows": 1000,
            "tfs_supplied": 100,
        },
        "wall_s": {"setup": 1.0, "pipeline": 2.0, "end_to_end": 3.0},
        "setup_elapsed_s": {
            "load_rna_qc": 0.2,
            "fragments_to_matrix": 0.3,
            "subset_shared_cells": 0.1,
            "motif_rankings": 0.15,
            "gene_coords": 0.2,
            "tf_list": 0.05,
        },
        "setup_peak_rss_gb": 1.0,
        "peak_rss_gb": 1.25,
        "elapsed_per_stage": {
            "preproc": 0.0,
            "topics": 0.4,
            "grn": 0.5,
            "cistarget": 0.3,
            "enhancer": 0.2,
            "eregulons": 0.1,
            "aucell": 0.25,
        },
        "peak_rss_gb_per_stage": {
            "load_rna": 0.5,
            "preproc": 0.6,
            "topics": 0.7,
            "grn": 0.8,
            "candidate_regulons": 0.9,
            "cistarget": 0.95,
            "enhancer": 1.0,
            "eregulons": 1.1,
            "aucell": 1.0,
            "integrated_adata": 1.25,
        },
        "outputs": {
            "grn_edges": 10,
            "candidate_regulons": 3,
            "regulons": 2,
            "cistarget_rows": 4,
            "enhancer_links": 5,
            "eregulon_rows": 6,
            "eregulons": 1,
            "aucell_shape": [100, 2],
        },
        "expected_tf_recovery": {
            "expected_tfs": ["TF1", "TF2"],
            "found": ["TF1"],
            "missing": ["TF2"],
            "fraction": 0.5,
        },
        "output_summaries": {
            "active_regulons_sample": ["TF1_regulon"],
            "top_grn_edges": [{"TF": "TF1", "target": "G1", "importance": 0.9}],
            "top_cistarget_rows": [{"regulon": "TF1_regulon", "motif": "m1", "auc": 0.5}],
            "top_enhancer_links": [{"peak_id": "p1", "gene": "G1", "correlation": 0.7}],
            "top_eregulon_rows": [{"tf": "TF1", "enhancer": "p1", "target_gene": "G1"}],
        },
        "output_inventory": output_inventory,
        "env": {
            "python": "3.13.0",
            "host": "minerva",
            "rayon_num_threads": "4",
            "omp_num_threads": "1",
            "openblas_num_threads": "1",
            "mkl_num_threads": "1",
        },
    }


def _full_pipeline_scaling_record(tmp_path: Path):
    runs = []
    for label, n_cells, end_to_end, peak_rss in (
        ("cells100", 100, 3.0, 1.25),
        ("cells200", 200, 6.0, 1.75),
    ):
        child_dir = tmp_path / label
        child_dir.mkdir()
        child = _full_pipeline_record(child_dir)
        child["run_id"] = label
        child["shapes"]["rna_post_qc"] = [n_cells, 1000]
        child["shapes"]["atac_shared_cells"] = [n_cells, 2000]
        child["params"]["n_cells_requested"] = n_cells
        child["outputs"]["aucell_shape"] = [n_cells, 2]
        child["wall_s"]["end_to_end"] = end_to_end
        child["wall_s"]["pipeline"] = end_to_end - 1.0
        child["peak_rss_gb"] = peak_rss
        child_path = tmp_path / f"{label}.json"
        child_path.write_text(json.dumps(child))
        runs.append(
            {
                "n_cells_requested": n_cells,
                "n_cells_actual": n_cells,
                "json_path": str(child_path),
                "output_dir": str(child_dir),
                "wall_s": child["wall_s"],
                "peak_rss_gb": child["peak_rss_gb"],
                "setup_peak_rss_gb": child["setup_peak_rss_gb"],
                "elapsed_per_stage": child["elapsed_per_stage"],
                "peak_rss_gb_per_stage": child["peak_rss_gb_per_stage"],
                "backend_execution": child["backend_execution"],
                "outputs": child["outputs"],
                "expected_tf_recovery": child.get("expected_tf_recovery"),
            }
        )
    return {
        "benchmark": "real_multiome_full_pipeline_scaling",
        "dataset_name": "pbmc3k",
        "run_prefix": "fixture",
        "repo_state": _clean_repo_state(),
        "runtime_import": _runtime_import_state(),
        "backend_capabilities": _backend_capabilities(),
        "python_hot_paths": _python_hot_paths_state(),
        "rustscenic": "0.4.7",
        "params": {"cell_counts": [100, 200], "threads": 4, "seed": 777},
        "runs": runs,
        "scaling": {
            "end_to_end_wall_slope_vs_cells": 1.0,
            "pipeline_wall_slope_vs_cells": 1.322,
            "peak_rss_slope_vs_cells": 0.485,
        },
        "env": {
            "python": "3.13.0",
            "host": "minerva",
            "rayon_num_threads": "4",
            "omp_num_threads": "1",
            "openblas_num_threads": "1",
            "mkl_num_threads": "1",
        },
    }


def _grn_scaling_record():
    row = {
        "dataset": "10x_pbmc_unsorted_3k",
        "run_kind": "subset_scaling",
        "n_cells": 100,
        "n_genes": 1000,
        "n_tfs": 50,
        "threads": 4,
        "edges": 1000,
        "grn_wall_s": 1.5,
        "peak_rss_gb": 0.75,
        "env": {
            "repo_state": _clean_repo_state(),
            "runtime_import": _runtime_import_state(),
            "backend_capabilities": _backend_capabilities(),
            "python_hot_paths": _python_hot_paths_state(),
            "rayon_num_threads": "4",
            "omp_num_threads": "1",
            "openblas_num_threads": "1",
            "mkl_num_threads": "1",
        },
    }
    return {
        "benchmark": "real_pbmc3k_grn_scaling",
        "dataset": "10x PBMC unsorted 3k multiome RNA post-QC",
        "repo_state": _clean_repo_state(),
        "runtime_import": _runtime_import_state(),
        "backend_capabilities": _backend_capabilities(),
        "python_hot_paths": _python_hot_paths_state(),
        "rustscenic": "0.4.7",
        "params": {
            "subset_sizes": [100],
            "subset_threads": 4,
            "thread_cells": 100,
            "thread_counts": [4],
            "seed": 777,
        },
        "subset_scaling": [row],
        "thread_scaling": [{**row, "run_kind": "thread_scaling"}],
        "thread_speedups": [
            {
                "threads": 4,
                "wall_s": 1.5,
                "speedup_vs_baseline": 1.0,
                "efficiency_vs_baseline": 1.0,
            }
        ],
        "subset_wall_slope_vs_cells": 1.0,
        "subset_memory_slope_vs_cells": 0.5,
    }


def test_benchmark_artifact_validator_accepts_full_pipeline_record(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )

    failures = module.validate_record(
        _full_pipeline_record(tmp_path),
        require_clean=True,
        check_output_files=True,
    )

    assert failures == []


def test_benchmark_artifact_validator_rejects_bad_expected_tf_recovery(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_bad_expected_tf_recovery",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["expected_tf_recovery"]["found"] = ["TF1", "NOT_EXPECTED"]
    record["expected_tf_recovery"]["missing"] = []
    record["expected_tf_recovery"]["fraction"] = 0.25

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "full_pipeline.expected_tf_recovery.found must be a subset of expected_tfs" in failures
    assert "full_pipeline.expected_tf_recovery.found and missing must cover expected_tfs" in failures
    assert any(
        failure.startswith("full_pipeline.expected_tf_recovery.fraction must match found/expected:")
        for failure in failures
    )


def test_benchmark_artifact_validator_accepts_full_pipeline_scaling_record(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )

    failures = module.validate_record(
        _full_pipeline_scaling_record(tmp_path),
        require_clean=True,
        check_output_files=True,
    )

    assert failures == []


def test_benchmark_artifact_validator_rejects_missing_scaling_child(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_missing_child",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["json_path"] = str(tmp_path / "missing.json")

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert f"runs[0].json_path does not exist: {tmp_path / 'missing.json'}" in failures


def test_benchmark_artifact_validator_rejects_scaling_child_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["wall_s"]["pipeline"] = 999.0
    record["scaling"]["pipeline_wall_slope_vs_cells"] = 0.0

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "runs[0].wall_s must match child JSON" in failures
    assert any(
        failure.startswith("scaling.pipeline_wall_slope_vs_cells must match runs:")
        for failure in failures
    )


def test_benchmark_artifact_validator_rejects_scaling_tf_recovery_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_tf_recovery_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["expected_tf_recovery"]["fraction"] = 0.0

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "runs[0].expected_tf_recovery must match child JSON" in failures


def test_benchmark_artifact_validator_rejects_incomplete_scaling_row_without_child_check(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_incomplete_row",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    row = record["runs"][0]
    row["setup_peak_rss_gb"] = 0.0
    del row["elapsed_per_stage"]["enhancer"]
    del row["peak_rss_gb_per_stage"]["aucell"]
    row["peak_rss_gb_per_stage"]["unexpected"] = 1.0
    row["outputs"]["grn_edges"] = 0
    row["outputs"]["aucell_shape"] = [999, 2]
    del row["expected_tf_recovery"]

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=False,
    )

    assert "runs[0].setup_peak_rss_gb must be positive" in failures
    assert "runs[0].elapsed_per_stage.enhancer missing" in failures
    assert "runs[0].peak_rss_gb_per_stage.aucell missing" in failures
    assert "runs[0].unknown peak_rss_gb_per_stage.unexpected" in failures
    assert "runs[0].outputs.grn_edges must be positive" in failures
    assert "runs[0].outputs.aucell_shape cells must equal n_cells_actual" in failures
    assert "runs[0].expected_tf_recovery must be an object" in failures


def test_benchmark_artifact_validator_rejects_scaling_child_dataset_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_dataset_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    child_path = Path(record["runs"][0]["json_path"])
    child = json.loads(child_path.read_text())
    child["dataset_name"] = "wrong_dataset"
    child_path.write_text(json.dumps(child))

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "runs[0].dataset_name must match aggregate dataset_name" in failures


def test_benchmark_artifact_validator_rejects_incomplete_output_inventory(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_inventory",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    del record["output_inventory"]["grn_path"]
    record["output_inventory"]["aucell_path"]["size_bytes"] = 0
    record["output_inventory"]["topics_dir"]["entries"] = 0
    cistarget_path = Path(record["output_inventory"]["cistarget_path"]["path"])
    cistarget_path.write_text("")

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "output_inventory.grn_path must be an object" in failures
    assert "output_inventory.aucell_path.size_bytes must be positive" in failures
    assert "output_inventory.topics_dir.entries must be positive" in failures
    assert (
        f"output_inventory.cistarget_path file is empty: {cistarget_path}"
        in failures
    )


def test_benchmark_artifact_validator_rejects_dirty_full_pipeline(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_dirty",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["repo_state"]["tracked_dirty"] = True
    record["repo_state"]["tracked_source_dirty"] = True
    record["repo_state"]["tracked_source_count"] = 1
    record["repo_state"]["tracked_source_sample"] = ["python/rustscenic/grn.py"]
    record["repo_state"]["source_dirty"] = True

    failures = module.validate_record(record, require_clean=True)

    assert "repo_state.tracked_source_count must be 0" in failures
    assert module.validate_record(record, require_clean=False) == []


def test_benchmark_artifact_validator_allows_tracked_non_source_dirty_full_pipeline(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_tracked_doc_dirty",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["repo_state"]["tracked_dirty"] = True
    record["repo_state"]["tracked_status_short"] = [" M manuscript/rustscenic_preprint.md"]
    record["repo_state"]["tracked_source_dirty"] = False
    record["repo_state"]["tracked_source_count"] = 0
    record["repo_state"]["tracked_source_sample"] = []
    record["repo_state"]["source_dirty"] = False

    assert module.validate_record(record, require_clean=True) == []


def test_benchmark_artifact_validator_rejects_untracked_source_full_pipeline(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_untracked_source",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["repo_state"]["untracked_source_count"] = 1
    record["repo_state"]["untracked_source_sample"] = ["python/rustscenic/new_stage.py"]
    record["repo_state"]["source_dirty"] = True

    failures = module.validate_record(record, require_clean=True)

    assert "repo_state.untracked_source_count must be 0" in failures
    assert module.validate_record(record, require_clean=False) == []


def test_benchmark_artifact_validator_rejects_missing_dataset_provenance(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_missing_dataset",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _full_pipeline_record(full_dir)
    full["dataset_name"] = ""
    grn = _grn_scaling_record()
    del grn["dataset"]
    del grn["subset_scaling"][0]["dataset"]

    full_failures = module.validate_record(full, require_clean=True)
    grn_failures = module.validate_record(grn, require_clean=True)

    assert "dataset_name must be a non-empty string" in full_failures
    assert "grn_scaling.dataset missing" in grn_failures
    assert "dataset must be a non-empty string" in grn_failures
    assert "subset_scaling[0].dataset must be a non-empty string" in grn_failures


def test_benchmark_artifact_validator_rejects_missing_backend_symbols(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_missing_backend",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["backend_capabilities"]["ok"] = False
    record["backend_capabilities"]["missing_symbols"] = ["enhancer.enhancer_link_pearson"]
    record["backend_capabilities"]["required_symbols"]["enhancer"] = []

    failures = module.validate_record(record, require_clean=True)

    assert "full_pipeline.backend_capabilities.ok must be true" in failures
    assert any("missing_symbols must be empty" in failure for failure in failures)
    assert (
        "full_pipeline.backend_capabilities.required_symbols.enhancer "
        "must contain at least one symbol"
    ) in failures


def test_benchmark_artifact_validator_rejects_python_hot_path_regression(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_hot_path_regression",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["python_hot_paths"]["ok"] = False
    record["python_hot_paths"]["violation_count"] = 1
    record["python_hot_paths"]["violations"] = [
        "pipeline.py:999: merged = left.merge(right)"
    ]

    failures = module.validate_record(record, require_clean=True)

    assert "full_pipeline.python_hot_paths.ok must be true" in failures
    assert "full_pipeline.python_hot_paths.violation_count must be 0" in failures
    assert any(
        failure.startswith("full_pipeline.python_hot_paths.violations must be empty:")
        for failure in failures
    )


def test_benchmark_artifact_validator_rejects_child_grn_python_hot_path_regression():
    module = _load_module(
        "validate_benchmark_artifact_child_hot_path_regression",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _grn_scaling_record()
    record["subset_scaling"][0]["env"]["python_hot_paths"]["ok"] = False
    record["subset_scaling"][0]["env"]["python_hot_paths"]["violation_count"] = 1
    record["subset_scaling"][0]["env"]["python_hot_paths"]["violations"] = [
        "eregulon.py:999: grouped = df.groupby('tf')"
    ]

    failures = module.validate_record(record, require_clean=True)

    assert "subset_scaling[0].env.python_hot_paths.ok must be true" in failures
    assert (
        "subset_scaling[0].env.python_hot_paths.violation_count must be 0"
        in failures
    )
    assert any(
        failure.startswith(
            "subset_scaling[0].env.python_hot_paths.violations must be empty:"
        )
        for failure in failures
    )


def test_benchmark_artifact_validator_rejects_non_rust_full_pipeline_stage(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_backend_execution_regression",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["backend_execution"]["pipeline_enhancer"] = {
        "engine": "python",
        "symbols": [],
    }

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.backend_execution.pipeline_enhancer.engine must be 'rust'"
        in failures
    )
    assert (
        "full_pipeline.backend_execution.pipeline_enhancer.symbols must be a non-empty string list"
        in failures
    )


def test_benchmark_artifact_validator_rejects_incomplete_rust_stage_symbols(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_incomplete_stage_symbols",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["backend_execution"]["pipeline_enhancer"] = {
        "engine": "rust",
        "symbols": ["enhancer_link_pearson_sparse_rna"],
    }
    record["backend_execution"]["pipeline_aucell"] = {
        "engine": "rust",
        "symbols": ["aucell_score_sparse_csr"],
    }

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.backend_execution.pipeline_enhancer.symbols missing "
        "required Rust symbol 'enhancer_align_cell_indices'"
    ) in failures
    assert (
        "full_pipeline.backend_execution.pipeline_aucell.symbols missing "
        "required Rust symbol 'stage_prepare_regulon_indices_with_coverage'"
    ) in failures


def test_benchmark_artifact_validator_rejects_scaling_row_without_backend_execution(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_scaling_backend_execution_regression",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    del record["runs"][0]["backend_execution"]["pipeline_eregulons"]

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=False,
    )

    assert (
        "runs[0].backend_execution.pipeline_eregulons must be an object"
        in failures
    )


def test_benchmark_artifact_validator_rejects_package_extension_version_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_version_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["runtime_import"]["extension_version"] = "0.4.6"

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.runtime_import package/extension version mismatch: 0.4.7 != 0.4.6"
        in failures
    )


def test_benchmark_artifact_validator_rejects_thread_budget_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_thread_budget_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["params"]["threads"] = 8
    record["env"]["openblas_num_threads"] = "4"

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.env.rayon_num_threads must match params.threads: 4 != 8"
        in failures
    )
    assert (
        "full_pipeline.env.openblas_num_threads must be '1' for reproducible CPU benchmarking"
        in failures
    )


def test_benchmark_artifact_validator_rejects_incomplete_full_pipeline(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_incomplete",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["outputs"]["grn_edges"] = 0
    record["outputs"]["cistarget_rows"] = 0
    record["outputs"]["enhancer_links"] = 0
    record["outputs"]["eregulon_rows"] = 0
    record["outputs"]["eregulons"] = 0
    del record["elapsed_per_stage"]["enhancer"]
    del record["peak_rss_gb_per_stage"]["integrated_adata"]
    record["peak_rss_gb_per_stage"]["unexpected"] = 1.0
    del record["setup_elapsed_s"]["fragments_to_matrix"]
    record["wall_s"]["end_to_end"] = 1.0

    failures = module.validate_record(record, require_clean=True)

    assert "outputs.grn_edges must be positive" in failures
    assert "outputs.cistarget_rows must be positive" in failures
    assert "outputs.enhancer_links must be positive" in failures
    assert "outputs.eregulon_rows must be positive" in failures
    assert "outputs.eregulons must be positive" in failures
    assert "elapsed_per_stage.enhancer missing" in failures
    assert "peak_rss_gb_per_stage.integrated_adata missing" in failures
    assert "unknown peak_rss_gb_per_stage.unexpected" in failures
    assert "setup_elapsed_s.fragments_to_matrix missing" in failures
    assert "wall_s.end_to_end must include setup + pipeline time" in failures


def test_benchmark_artifact_validator_rejects_invalid_full_pipeline_shapes(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_bad_shapes",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["shapes"]["rna_post_qc"] = [0, 1000]
    record["shapes"]["atac_shared_cells"] = [99, 2000]
    record["shapes"]["motif_rankings"] = [50]
    record["shapes"]["gene_coords_rows"] = 0
    record["shapes"]["tfs_supplied"] = 0
    record["outputs"]["aucell_shape"] = [101, 2]

    failures = module.validate_record(record, require_clean=True)

    assert "shapes.rna_post_qc must be [positive_rows, positive_cols]" in failures
    assert "shapes.motif_rankings must be [positive_rows, positive_cols]" in failures
    assert "shapes.gene_coords_rows must be positive" in failures
    assert "shapes.tfs_supplied must be positive" in failures

    record = _full_pipeline_record(tmp_path)
    record["shapes"]["atac_shared_cells"] = [99, 2000]
    record["outputs"]["aucell_shape"] = [99, 2]
    failures = module.validate_record(record, require_clean=True)

    assert "shapes.rna_post_qc cells must equal shapes.atac_shared_cells cells" in failures
    assert "outputs.aucell_shape cells must equal shapes.rna_post_qc cells" in failures


def test_benchmark_artifact_validator_rejects_missing_reference_fingerprints(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_reference_fingerprints",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    del record["reference_fingerprints"]["motif_rankings"]["corner_sample_sha256"]
    record["reference_fingerprints"]["gene_coords"]["shape"] = [999, 3]

    failures = module.validate_record(record, require_clean=True)

    assert (
        "reference_fingerprints.motif_rankings.corner_sample_sha256 must be sha256 hex"
        in failures
    )
    assert (
        "reference_fingerprints.gene_coords.shape rows must match shapes.gene_coords_rows"
        in failures
    )


def test_benchmark_artifact_validator_rejects_missing_output_summaries(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_output_summaries",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["output_summaries"]["top_grn_edges"] = []
    record["output_summaries"]["top_enhancer_links"] = ["not-a-row"]

    failures = module.validate_record(record, require_clean=True)

    assert "output_summaries.top_grn_edges must be non-empty" in failures
    assert "output_summaries.top_enhancer_links rows must be objects" in failures


def test_benchmark_artifact_validator_accepts_grn_scaling_record():
    module = _load_module(
        "validate_benchmark_artifact_grn",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )

    failures = module.validate_record(_grn_scaling_record(), require_clean=True)

    assert failures == []


def test_benchmark_artifact_validator_rejects_dirty_child_grn_record():
    module = _load_module(
        "validate_benchmark_artifact_child_dirty",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _grn_scaling_record()
    record["subset_scaling"][0]["env"]["repo_state"]["tracked_dirty"] = True

    failures = module.validate_record(record, require_clean=True)

    assert "subset_scaling[0].env.repo_state.tracked_dirty must be false" in failures


def test_benchmark_artifact_validator_rejects_grn_thread_budget_mismatch():
    module = _load_module(
        "validate_benchmark_artifact_grn_thread_budget_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _grn_scaling_record()
    record["subset_scaling"][0]["env"]["rayon_num_threads"] = "8"
    record["thread_scaling"][0]["env"]["mkl_num_threads"] = "2"

    failures = module.validate_record(record, require_clean=True)

    assert (
        "subset_scaling[0].env.rayon_num_threads must match params.threads: 8 != 4"
        in failures
    )
    assert (
        "thread_scaling[0].env.mkl_num_threads must be '1' for reproducible CPU benchmarking"
        in failures
    )


def test_minerva_result_collector_discovers_latest_valid_benchmarks(tmp_path):
    module = _load_module(
        "collect_benchmark_results_latest",
        ROOT / "validation/hpc/minerva/collect_benchmark_results.py",
    )
    old_dir = tmp_path / "old_full_outputs"
    new_dir = tmp_path / "new_full_outputs"
    scaling_dir = tmp_path / "scaling_outputs"
    for path in (old_dir, new_dir, scaling_dir):
        path.mkdir()

    old_record = _full_pipeline_record(old_dir)
    new_record = _full_pipeline_record(new_dir)
    new_record["shapes"]["rna_post_qc"] = [200, 1000]
    new_record["shapes"]["atac_shared_cells"] = [200, 2000]
    new_record["outputs"]["aucell_shape"] = [200, 2]
    scaling_record = _full_pipeline_scaling_record(scaling_dir)

    old_path = tmp_path / "pbmc_full_old.json"
    new_path = tmp_path / "pbmc_full_new.json"
    scaling_path = tmp_path / "pbmc_full_scaling.json"
    old_path.write_text(json.dumps(old_record))
    new_path.write_text(json.dumps(new_record))
    scaling_path.write_text(json.dumps(scaling_record))
    (tmp_path / "pbmc_full_new.preflight.json").write_text(json.dumps({"ok": True}))

    os.utime(old_path, (1_700_000_000, 1_700_000_000))
    os.utime(new_path, (1_700_000_100, 1_700_000_100))
    os.utime(scaling_path, (1_700_000_050, 1_700_000_050))

    rows = module.collect(
        [tmp_path],
        require_clean=True,
        check_output_files=True,
        latest_per_benchmark=True,
    )

    assert [row["benchmark"] for row in rows] == [
        "real_multiome_full_pipeline",
        "real_multiome_full_pipeline_scaling",
    ]
    full_row = rows[0]
    assert full_row["path"] == str(new_path)
    assert full_row["valid"] is True
    assert full_row["cells"] == "200"
    assert full_row["outputs"] == "grn_edges=10, eregulon_rows=6, regulons=2"
    assert rows[1]["cells"] == "100..200"
    assert rows[1]["peak_rss_gb"] == "1.25..1.75"


def test_minerva_result_collector_reports_validation_failures(tmp_path):
    module = _load_module(
        "collect_benchmark_results_failures",
        ROOT / "validation/hpc/minerva/collect_benchmark_results.py",
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    record = _full_pipeline_record(output_dir)
    record["repo_state"]["tracked_dirty"] = True
    record["repo_state"]["tracked_source_dirty"] = True
    record["repo_state"]["tracked_source_count"] = 1
    record["repo_state"]["tracked_source_sample"] = ["python/rustscenic/grn.py"]
    record["repo_state"]["source_dirty"] = True
    path = tmp_path / "dirty_full_pipeline.json"
    path.write_text(json.dumps(record))

    rows = module.collect(
        [path],
        require_clean=True,
        check_output_files=False,
        latest_per_benchmark=False,
    )

    assert len(rows) == 1
    assert rows[0]["valid"] is False
    assert rows[0]["failures"] == ["repo_state.tracked_source_count must be 0"]


def test_minerva_result_collector_summarises_grn_scaling(tmp_path):
    module = _load_module(
        "collect_benchmark_results_grn",
        ROOT / "validation/hpc/minerva/collect_benchmark_results.py",
    )
    record = _grn_scaling_record()
    record["subset_scaling"].append(
        {
            **record["subset_scaling"][0],
            "n_cells": 200,
            "edges": 2000,
            "grn_wall_s": 3.0,
            "peak_rss_gb": 1.25,
        }
    )
    record["subset_wall_slope_vs_cells"] = 1.0
    record["subset_memory_slope_vs_cells"] = 0.5
    record["thread_speedups"].append(
        {
            "threads": 8,
            "wall_s": 0.75,
            "speedup_vs_baseline": 2.0,
            "efficiency_vs_baseline": 0.25,
        }
    )
    path = tmp_path / "grn_scaling.json"
    path.write_text(json.dumps(record))

    rows = module.collect(
        [path],
        require_clean=True,
        check_output_files=False,
        latest_per_benchmark=False,
    )

    assert len(rows) == 1
    assert rows[0]["valid"] is True
    assert rows[0]["dataset"] == "10x PBMC unsorted 3k multiome RNA post-QC"
    assert rows[0]["cells"] == "100..200"
    assert rows[0]["end_to_end_s"] == "1.5..3"
    assert rows[0]["peak_rss_gb"] == "0.75..1.25"
    assert rows[0]["outputs"] == "edges=1000..2000, genes=1000, tfs=50"
    assert (
        rows[0]["scaling"]
        == "subset_wall_slope_vs_cells=1, subset_memory_slope_vs_cells=0.5, "
        "best_thread_speedup=2@8t"
    )


def test_minerva_result_collector_markdown_table_is_compact(tmp_path):
    module = _load_module(
        "collect_benchmark_results_markdown",
        ROOT / "validation/hpc/minerva/collect_benchmark_results.py",
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    record = _full_pipeline_record(output_dir)
    path = tmp_path / "full_pipeline.json"
    path.write_text(json.dumps(record))
    rows = module.collect(
        [path],
        require_clean=True,
        check_output_files=True,
        latest_per_benchmark=False,
    )

    table = module.markdown_table(rows)

    assert table.startswith("| valid | benchmark | dataset | cells |")
    assert "| yes | real_multiome_full_pipeline | pbmc3k | 100 |" in table
    assert "grn_edges=10, eregulon_rows=6, regulons=2" in table


def test_minerva_launchers_validate_benchmark_artifacts_after_run():
    full = (ROOT / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline.lsf").read_text()
    full_scaling = (
        ROOT / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline_scaling.lsf"
    ).read_text()
    grn = (ROOT / "validation/hpc/minerva/run_real_pbmc3k_grn_scaling.lsf").read_text()

    assert "validation/hpc/minerva/prepare_real_pbmc3k_data.py" in full
    assert "validation/hpc/minerva/validate_benchmark_artifact.py" in full
    assert "validation/hpc/minerva/collect_benchmark_results.py" in full
    assert "--check-output-files" in full
    assert "--require-repo-import" in full
    assert "--require-thread-pins" in full
    assert "--require-data-hashes" in full
    assert "--require-rust-hot-paths" in full
    assert '--threads "${RAYON_NUM_THREADS}"' in full
    assert "validation/hpc/minerva/prepare_real_pbmc3k_data.py" in full_scaling
    assert "validation/scaling/bench_real_multiome_pipeline_scaling.py" in full_scaling
    assert "validation/hpc/minerva/validate_benchmark_artifact.py" in full_scaling
    assert "validation/hpc/minerva/collect_benchmark_results.py" in full_scaling
    assert "--check-output-files" in full_scaling
    assert "--require-repo-import" in full_scaling
    assert "--require-thread-pins" in full_scaling
    assert "--require-data-hashes" in full_scaling
    assert "--require-rust-hot-paths" in full_scaling
    assert "validation/hpc/minerva/prepare_real_pbmc3k_data.py" in grn
    assert "validation/hpc/minerva/validate_benchmark_artifact.py" in grn
    assert "validation/hpc/minerva/collect_benchmark_results.py" in grn
    assert "--require-repo-import" in grn
    assert "--require-thread-pins" in grn
    assert "--require-data-hashes" in grn
    assert "--require-rust-hot-paths" in grn
    assert 'export RAYON_NUM_THREADS="${LSB_DJOB_NUMPROC:-16}"' in grn


def test_minerva_project_path_matches_lab_shared_folder():
    preflight = _load_module(
        "preflight_minerva_project_path",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    expected = "/sc/arion/projects/DiseaseGeneCell/Huang_lab_projects/rustscenic"
    files = [
        ROOT / "validation/hpc/minerva/README.md",
        ROOT / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline.lsf",
        ROOT / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline_scaling.lsf",
        ROOT / "validation/hpc/minerva/run_real_pbmc3k_grn_scaling.lsf",
    ]

    assert str(preflight.DEFAULT_PROJECT) == expected
    for path in files:
        text = path.read_text()
        assert expected in text
        assert "Huang_lab_project/rustscenic" not in text
