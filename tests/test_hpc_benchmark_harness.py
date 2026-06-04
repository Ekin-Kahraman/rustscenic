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
from validation.python_hot_paths import ALLOWED_HITS, HOT_PATH_PATTERNS
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


def test_real_multiome_harness_subsets_requested_cells_before_fragments():
    module = _load_module(
        "bench_real_multiome_cell_subset",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    import anndata as ad

    cells = [f"c{i}" for i in range(6)]
    adata = ad.AnnData(
        X=np.zeros((6, 3), dtype=np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )

    subset = module.subset_requested_cells(adata, n_cells=3, seed=7)
    expected = sorted(
        np.random.default_rng(7).choice(
            np.asarray(sorted(cells), dtype=object),
            size=3,
            replace=False,
        )
    )

    assert list(subset.obs_names) == expected
    assert module.subset_requested_cells(adata, n_cells=None, seed=7) is adata
    assert module.subset_requested_cells(adata, n_cells=6, seed=7) is adata


def test_real_multiome_harness_records_sparse_matrix_profile():
    module = _load_module(
        "bench_real_multiome_matrix_profile",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    import anndata as ad
    import scipy.sparse as sp

    adata = ad.AnnData(
        X=sp.csr_matrix(
            np.asarray(
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0],
                ],
                dtype=np.float32,
            )
        ),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )

    assert module.matrix_profile(adata) == {
        "shape": [2, 3],
        "storage": "sparse",
        "format": "csr",
        "dtype": "float32",
        "nnz": 3,
        "density": 0.5,
    }


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


def test_real_multiome_harness_fingerprints_feather_without_full_table_read(
    monkeypatch,
    tmp_path,
):
    module = _load_module(
        "bench_real_multiome_file_backed_fingerprint",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    path = tmp_path / "regions_vs_motifs.rankings.feather"
    pd.DataFrame(
        {
            "motifs": ["m1", "m2", "m3"],
            "peak_1": np.asarray([1, 2, 3], dtype=np.int32),
            "peak_2": np.asarray([3, 2, 1], dtype=np.int32),
        }
    ).to_feather(path)

    import pyarrow.feather as feather

    real_read_table = feather.read_table
    read_columns = []

    def recording_read_table(*args, **kwargs):
        read_columns.append(kwargs.get("columns"))
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(feather, "read_table", recording_read_table)

    fp = module.file_backed_table_fingerprint(path)

    assert fp["shape"] == [3, 3]
    assert fp["file_backed"] is True
    assert fp["format"] == "feather"
    assert fp["metadata_read_columns"] == ["motifs"]
    assert fp["path_name"] == path.name
    assert fp["size_bytes"] == path.stat().st_size
    assert read_columns == [["motifs"]]


def test_real_multiome_harness_builds_compact_output_summaries(monkeypatch, tmp_path):
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
    assert summaries["summary_max_rows"] is None

    def fail_full_parquet_read(*_args, **_kwargs):
        raise AssertionError("bounded summaries must not call pd.read_parquet")

    monkeypatch.setattr(module.pd, "read_parquet", fail_full_parquet_read)
    bounded = module.output_summaries(result, n=1, max_rows=1)

    assert bounded["summary_max_rows"] == 1
    assert bounded["top_grn_edges"] == [{"TF": "TF2", "target": "G2", "importance": 0.2}]
    assert bounded["top_cistarget_rows"][0]["motif"] == "m1"


def test_real_multiome_harness_defaults_to_bounded_output_summaries():
    module = _load_module(
        "bench_real_multiome_default_summary_bound",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    scaling = _load_module(
        "bench_real_multiome_scaling_default_summary_bound",
        ROOT / "validation/scaling/bench_real_multiome_pipeline_scaling.py",
    )

    assert module.DEFAULT_SUMMARY_MAX_ROWS == 1000
    assert module.parse_args([
        "--dataset-name", "pbmc",
        "--rna-10x-h5", "rna.h5",
        "--fragments", "fragments.tsv.gz",
        "--peaks", "peaks.bed",
        "--out-dir", "out",
        "--out-json", "out.json",
    ]).summary_max_rows == 1000
    assert scaling.parse_args([
        "--dataset-name", "pbmc",
        "--rna-10x-h5", "rna.h5",
        "--fragments", "fragments.tsv.gz",
        "--peaks", "peaks.bed",
        "--out-root", "out",
        "--out-json", "out.json",
    ]).summary_max_rows == 1000


def test_real_multiome_harness_does_not_reread_pipeline_outputs_for_counts():
    source = (ROOT / "validation/scaling/bench_real_multiome_pipeline.py").read_text()

    assert "read_parquet_if_present" not in source
    assert "pd.read_parquet(result." not in source
    assert "result.n_grn_edges" in source
    assert "result.aucell_shape" in source
    assert '"pipeline_compute_stages": round(pipeline_compute_stage_wall, 3)' in source
    assert '"pipeline_unattributed": round(pipeline_unattributed_wall, 3)' in source
    assert '"end_to_end": round(end_to_end_wall, 3)' in source
    assert '"setup_elapsed_s"' in source


def test_real_multiome_harness_passes_peak_bed_to_pipeline():
    source = (ROOT / "validation/scaling/bench_real_multiome_pipeline.py").read_text()

    assert "adata_atac=atac," in source
    assert "peaks=args.peaks," in source


def test_real_multiome_harness_prefixes_pipeline_backend_execution():
    module = _load_module(
        "bench_real_multiome_backend_execution",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    result = SimpleNamespace(
        backend_execution={
            "grn": {
                "engine": "rust",
                "symbols": ["gene_duplicate_summary", "grn_infer"],
            },
            "integrated_adata": {
                "engine": "python_io",
                "reason": "AnnData obs attachment and h5ad write",
            },
        }
    )

    execution = module.backend_execution_for_benchmark(result)

    assert execution == {
        "setup_fragments_to_matrix": {
            "engine": "rust",
            "symbols": ["preproc_fragments_to_matrix"],
        },
        "pipeline_grn": {
            "engine": "rust",
            "symbols": ["gene_duplicate_summary", "grn_infer"],
        },
        "pipeline_integrated_adata": {
            "engine": "python_io",
            "reason": "AnnData obs attachment and h5ad write",
        },
    }


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


def test_real_multiome_harness_omits_empty_lsf_env(monkeypatch):
    module = _load_module(
        "bench_real_multiome_benchmark_env",
        ROOT / "validation/scaling/bench_real_multiome_pipeline.py",
    )
    for key in (
        "LSB_JOBID",
        "LSB_QUEUE",
        "LSB_DJOB_NUMPROC",
        "RUSTSCENIC_LSF_PROJECT",
        "RUSTSCENIC_LSF_QUEUE",
        "RUSTSCENIC_LSF_CORES",
        "RUSTSCENIC_LSF_MEM_MB",
        "RUSTSCENIC_LSF_WALLTIME",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")

    env = module.benchmark_env()

    assert env["rayon_num_threads"] == "4"
    assert env["omp_num_threads"] == "1"
    assert "lsf_project" not in env
    assert "lsf_requested_queue" not in env
    assert "lsf_requested_walltime" not in env

    monkeypatch.setenv("RUSTSCENIC_LSF_PROJECT", "acc_DiseaseGeneCell")
    monkeypatch.setenv("RUSTSCENIC_LSF_QUEUE", "express")
    monkeypatch.setenv("RUSTSCENIC_LSF_WALLTIME", "08:00")

    env = module.benchmark_env()

    assert env["lsf_project"] == "acc_DiseaseGeneCell"
    assert env["lsf_requested_queue"] == "express"
    assert env["lsf_requested_walltime"] == "08:00"


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
        (["--summary-max-rows", "0"], "--summary-max-rows must be positive"),
        (["--motif-rankings", str(tmp_path / "missing.parquet")], "missing input:"),
        (["--motif-annotations", str(tmp_path / "missing.tsv")], "missing input:"),
        (["--region-motif-rankings", str(tmp_path / "missing.feather")], "missing input:"),
        (["--gene-coords", str(tmp_path / "missing_genes.parquet")], "missing input:"),
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
            "--motif-rankings",
            str(tmp_path / "motifs.parquet"),
            "--motif-annotations",
            str(tmp_path / "annotations.tsv"),
            "--region-motif-rankings",
            str(tmp_path / "region_rankings.feather"),
            "--gene-coords",
            str(tmp_path / "genes.parquet"),
            "--expected-tfs",
            "SPI1",
            "PAX5",
            "--summary-max-rows",
            "25",
            "--skip-integrated-adata",
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
    assert "--motif-rankings" in cmd
    assert cmd[cmd.index("--motif-rankings") + 1].endswith("motifs.parquet")
    assert "--motif-annotations" in cmd
    assert cmd[cmd.index("--motif-annotations") + 1].endswith("annotations.tsv")
    assert "--region-motif-rankings" in cmd
    assert cmd[cmd.index("--region-motif-rankings") + 1].endswith("region_rankings.feather")
    assert "--gene-coords" in cmd
    assert cmd[cmd.index("--gene-coords") + 1].endswith("genes.parquet")
    assert "--expected-tfs" in cmd
    assert "SPI1" in cmd
    assert "--summary-max-rows" in cmd
    assert cmd[cmd.index("--summary-max-rows") + 1] == "25"
    assert "--skip-integrated-adata" in cmd
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
        (["--motif-rankings", str(tmp_path / "missing.parquet")], "missing input:"),
        (["--motif-annotations", str(tmp_path / "missing.tsv")], "missing input:"),
        (["--region-motif-rankings", str(tmp_path / "missing.feather")], "missing input:"),
        (["--gene-coords", str(tmp_path / "missing_genes.parquet")], "missing input:"),
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
            "wall_s": {
                "setup": 1.0,
                "pipeline": 2.0,
                "pipeline_compute_stages": 1.0,
                "pipeline_unattributed": 1.0,
                "end_to_end": 3.0,
            },
            "peak_rss_gb": 1.0,
            "setup_peak_rss_gb": 0.8,
            "setup_elapsed_s": {"load_rna_qc": 0.1, "fragments_to_matrix": 0.9},
            "elapsed_per_stage": {},
            "peak_rss_gb_per_stage": {},
            "outputs": {"grn_edges": 1},
        },
        {
            "n_cells_requested": 200,
            "n_cells_actual": 200,
            "json_path": str(tmp_path / "two.json"),
            "output_dir": str(tmp_path / "two"),
            "wall_s": {
                "setup": 1.0,
                "pipeline": 4.0,
                "pipeline_compute_stages": 2.0,
                "pipeline_unattributed": 2.0,
                "end_to_end": 6.0,
            },
            "peak_rss_gb": 1.5,
            "setup_peak_rss_gb": 1.0,
            "setup_elapsed_s": {"load_rna_qc": 0.2, "fragments_to_matrix": 1.8},
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
    assert payload["scaling"]["pipeline_compute_stage_wall_slope_vs_cells"] == 1.0
    assert payload["scaling"]["pipeline_unattributed_wall_slope_vs_cells"] == 1.0
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
            "wall_s": {
                "setup": 1.0,
                "pipeline": 2.0,
                "pipeline_compute_stages": 1.0,
                "pipeline_unattributed": 1.0,
                "end_to_end": 3.0,
            },
            "peak_rss_gb": 1.0,
            "setup_peak_rss_gb": 0.8,
            "setup_elapsed_s": {"load_rna_qc": 0.1, "fragments_to_matrix": 0.9},
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
        assert "gene_dedupe_sparse_csc_f32" in backend["required_symbols"]["gene_resolution"]
        assert "specificity_rss_f32" in backend["required_symbols"]["specificity"]


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
    from validation import process_memory

    fake_resource = types.SimpleNamespace(RUSAGE_SELF=object())
    monkeypatch.setattr(process_memory, "resource", fake_resource)
    monkeypatch.setattr(
        fake_resource,
        "getrusage",
        lambda _who: types.SimpleNamespace(ru_maxrss=1024**3),
        raising=False,
    )
    monkeypatch.setattr(process_memory.sys, "platform", "darwin")
    assert process_memory.peak_rss_gb() == 1.0

    monkeypatch.setattr(
        fake_resource,
        "getrusage",
        lambda _who: types.SimpleNamespace(ru_maxrss=1024**2),
        raising=False,
    )
    monkeypatch.setattr(process_memory.sys, "platform", "linux")
    assert process_memory.peak_rss_gb() == 1.0


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


def test_minerva_backend_state_uses_package_self_check():
    module = _load_module(
        "preflight_minerva_backend_self_check",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )

    state = module._backend_state(Path(sys.executable), ROOT)

    assert state["ok"] is True
    assert state["extension_error"] is None
    assert state["missing_symbols"] == []
    assert "pipeline_project_ranking_columns" in state["required_symbols"]["pipeline"]


def test_minerva_backend_state_rejects_package_without_self_check(tmp_path):
    module = _load_module(
        "preflight_minerva_backend_stale_package",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    stale_pkg = tmp_path / "rustscenic"
    stale_pkg.mkdir()
    (stale_pkg / "__init__.py").write_text("__version__ = '0.4.6'\n")

    state = module._backend_state(Path(sys.executable), tmp_path)

    assert state["ok"] is False
    assert "backend_capabilities" in state["extension_error"]
    assert state["required_symbols"] == module.REQUIRED_RUST_BACKEND_SYMBOLS


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


def test_minerva_preflight_checks_optional_reference_tables(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_reference_tables",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    motif_rankings = tmp_path / "motifs.parquet"
    motif_annotations = tmp_path / "motif_annotations.tsv"
    region_motif_rankings = tmp_path / "region_motifs.feather"
    gene_coords = tmp_path / "gene_coords.parquet"
    for path in (motif_rankings, motif_annotations, region_motif_rankings, gene_coords):
        path.write_text("fixture\n")
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
            "--motif-rankings", str(motif_rankings),
            "--motif-annotations", str(motif_annotations),
            "--region-motif-rankings", str(region_motif_rankings),
            "--gene-coords", str(gene_coords),
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is True
    assert result["reference_tables"]["motif_rankings"]["exists"] is True
    assert result["reference_tables"]["motif_annotations"]["exists"] is True
    assert result["reference_tables"]["region_motif_rankings"]["exists"] is True
    assert result["reference_tables"]["gene_coords"]["exists"] is True


def test_minerva_preflight_rejects_missing_optional_reference_table(monkeypatch, tmp_path):
    module = _load_module(
        "preflight_minerva_missing_reference_table",
        ROOT / "validation/hpc/minerva/preflight_minerva.py",
    )
    repo, env, data = _write_minerva_preflight_fixture(tmp_path)
    missing = tmp_path / "missing_annotations.tsv"
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
            "--motif-annotations", str(missing),
        ]
    )

    result = module.preflight(args)

    assert result["ok"] is False
    assert result["reference_tables"]["motif_annotations"]["exists"] is False
    assert f"missing reference table motif_annotations: {missing}" in result["failures"]


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
        "allowed_hit_count": len(ALLOWED_HITS),
        "pattern_count": len(HOT_PATH_PATTERNS),
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
                "preproc_peak_coords_for_names",
                "enhancer_match_peak_coords_to_atac",
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
        "pipeline_integrated_adata": {
            "engine": "python_io",
            "reason": "AnnData obs attachment and h5ad write",
        },
    }


def _matrix_inputs_state(n_cells: int = 100):
    return {
        "rna_post_qc": {
            "shape": [n_cells, 1000],
            "storage": "sparse",
            "format": "csr",
            "dtype": "float32",
            "nnz": n_cells * 100,
            "density": 0.1,
        },
        "atac_shared_cells": {
            "shape": [n_cells, 2000],
            "storage": "sparse",
            "format": "csr",
            "dtype": "float32",
            "nnz": n_cells * 50,
            "density": 0.025,
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
    backend_execution = _backend_execution_state()
    cell_barcode_filter = {"requested": 100, "matched": 100}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    output_inventory["manifest_path"] = {
        "path": str(manifest_path),
        "exists": True,
        "type": "file",
        "size_bytes": manifest_path.stat().st_size,
    }
    record = {
        "benchmark": "real_multiome_full_pipeline",
        "dataset_name": "pbmc3k",
        "repo_state": _clean_repo_state(),
        "runtime_import": _runtime_import_state(),
        "backend_capabilities": _backend_capabilities(),
        "python_hot_paths": _python_hot_paths_state(),
        "backend_execution": backend_execution,
        "cell_barcode_filter": cell_barcode_filter,
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
        "matrix_inputs": _matrix_inputs_state(),
        "wall_s": {
            "setup": 1.0,
            "pipeline": 2.0,
            "pipeline_compute_stages": 1.75,
            "pipeline_unattributed": 0.25,
            "end_to_end": 3.0,
        },
        "setup_elapsed_s": {
            "load_rna_qc": 0.2,
            "subset_requested_cells": 0.01,
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
    _sync_full_pipeline_manifest(record)
    return record


def _sync_full_pipeline_manifest(record: dict) -> None:
    info = record["output_inventory"]["manifest_path"]
    path = Path(info["path"])
    manifest_backend = {
        key.removeprefix("pipeline_"): value
        for key, value in record["backend_execution"].items()
        if key.startswith("pipeline_")
    }
    inventory = record["output_inventory"]
    outputs = record["outputs"]
    shapes = record["shapes"]
    path.write_text(
        json.dumps(
            {
                "output_dir": str(path.parent),
                "atac_matrix_path": inventory["atac_matrix_path"]["path"],
                "grn_path": inventory["grn_path"]["path"],
                "regulons_path": inventory["regulons_path"]["path"],
                "candidate_regulons_path": inventory["candidate_regulons_path"]["path"],
                "aucell_path": inventory["aucell_path"]["path"],
                "topics_dir": inventory["topics_dir"]["path"],
                "cistarget_path": inventory["cistarget_path"]["path"],
                "enhancer_links_path": inventory["enhancer_links_path"]["path"],
                "eregulons_path": inventory["eregulons_path"]["path"],
                "integrated_adata_path": (
                    inventory.get("integrated_adata_path", {}).get("path")
                ),
                "elapsed": record["elapsed_per_stage"],
                "memory": record["peak_rss_gb_per_stage"],
                "n_cells": shapes["rna_post_qc"][0],
                "n_grn_edges": outputs["grn_edges"],
                "n_regulons": outputs["regulons"],
                "n_candidate_regulons": outputs["candidate_regulons"],
                "n_pruned_regulons": outputs.get("pruned_regulons"),
                "n_cistarget_rows": outputs["cistarget_rows"],
                "n_enhancer_links": outputs["enhancer_links"],
                "n_eregulon_rows": outputs["eregulon_rows"],
                "n_eregulons": outputs["eregulons"],
                "aucell_shape": outputs["aucell_shape"],
                "regulon_source": "candidate_grn_top_targets",
                "backend_execution": manifest_backend,
                "cell_barcode_filter": record.get("cell_barcode_filter"),
            }
        )
    )
    info["size_bytes"] = path.stat().st_size


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
        child["matrix_inputs"] = _matrix_inputs_state(n_cells)
        child["params"]["n_cells_requested"] = n_cells
        child["cell_barcode_filter"] = {"requested": n_cells, "matched": n_cells}
        child["outputs"]["aucell_shape"] = [n_cells, 2]
        child["wall_s"]["end_to_end"] = end_to_end
        child["wall_s"]["pipeline"] = end_to_end - 1.0
        child["wall_s"]["pipeline_compute_stages"] = round(
            sum(float(v) for v in child["elapsed_per_stage"].values()),
            3,
        )
        child["wall_s"]["pipeline_unattributed"] = round(
            max(
                0.0,
                child["wall_s"]["pipeline"] - child["wall_s"]["pipeline_compute_stages"],
            ),
            3,
        )
        child["peak_rss_gb"] = peak_rss
        _sync_full_pipeline_manifest(child)
        child_path = tmp_path / f"{label}.json"
        child_path.write_text(json.dumps(child))
        runs.append(
            {
                "n_cells_requested": n_cells,
                "n_cells_actual": n_cells,
                "threads": child["params"]["threads"],
                "json_path": str(child_path),
                "output_dir": str(child_dir),
                "wall_s": child["wall_s"],
                "peak_rss_gb": child["peak_rss_gb"],
                "setup_peak_rss_gb": child["setup_peak_rss_gb"],
                "setup_elapsed_s": child["setup_elapsed_s"],
                "elapsed_per_stage": child["elapsed_per_stage"],
                "peak_rss_gb_per_stage": child["peak_rss_gb_per_stage"],
                "backend_execution": child["backend_execution"],
                "cell_barcode_filter": child["cell_barcode_filter"],
                "matrix_inputs": child["matrix_inputs"],
                "outputs": child["outputs"],
                "expected_tf_recovery": child.get("expected_tf_recovery"),
                "env": child["env"],
                "write_integrated_adata": child["params"].get(
                    "write_integrated_adata",
                    True,
                ),
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
            "pipeline_compute_stage_wall_slope_vs_cells": 0.0,
            "pipeline_unattributed_wall_slope_vs_cells": 3.7,
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
        "backend_execution": {
            "grn": {
                "engine": "rust",
                "symbols": ["gene_duplicate_summary", "grn_infer_sparse_csc"],
            }
        },
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
        "thread_scaling": [{**deepcopy(row), "run_kind": "thread_scaling"}],
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


def test_benchmark_artifact_validator_rejects_bad_full_pipeline_wall_breakdown(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_bad_full_wall_breakdown",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["wall_s"]["pipeline_compute_stages"] = 0.5
    record["wall_s"]["pipeline_unattributed"] = 0.1

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert (
        "wall_s.pipeline_compute_stages must match elapsed_per_stage sum: "
        "0.5 != 1.75"
    ) in failures
    assert (
        "wall_s.pipeline_unattributed must match pipeline minus compute stages: "
        "0.1 != 1.5"
    ) in failures


def test_benchmark_artifact_validator_accepts_compute_profile_without_integrated_h5ad(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_no_integrated",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["params"]["write_integrated_adata"] = False
    record["output_inventory"].pop("integrated_adata_path")
    record["peak_rss_gb_per_stage"].pop("integrated_adata")
    record["backend_execution"]["pipeline_integrated_adata"] = {
        "engine": "skipped",
        "reason": "write_integrated_adata=False",
    }
    _sync_full_pipeline_manifest(record)

    failures = module.validate_record(
        record,
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


def test_benchmark_artifact_validator_accepts_negative_peak_rss_slope(tmp_path):
    """Peak RSS is measured in independent child processes, so allocator and
    OS effects can make the larger subset report a lower maximum RSS. The
    validator should require provenance consistency, not monotonic memory."""
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_negative_rss",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    for row, peak_rss in zip(record["runs"], (1.9, 1.7), strict=True):
        row["peak_rss_gb"] = peak_rss
        child_path = Path(row["json_path"])
        child = json.loads(child_path.read_text())
        child["peak_rss_gb"] = peak_rss
        child_path.write_text(json.dumps(child))
    record["scaling"]["peak_rss_slope_vs_cells"] = module._rounded_slope(
        [
            {"n_cells": row["n_cells_actual"], "peak_rss_gb": row["peak_rss_gb"]}
            for row in record["runs"]
        ],
        "n_cells",
        "peak_rss_gb",
    )

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert record["scaling"]["peak_rss_slope_vs_cells"] < 0
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


def test_benchmark_artifact_validator_rejects_scaling_cell_barcode_filter_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_cell_barcode_filter_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["cell_barcode_filter"] = {"requested": 101, "matched": 100}

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "runs[0].cell_barcode_filter must match child JSON" in failures


def test_benchmark_artifact_validator_rejects_scaling_row_env_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_row_env_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["env"]["rayon_num_threads"] = "2"

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "runs[0].env.rayon_num_threads must match params.threads: 2 != 4" in failures
    assert "runs[0].env must match child JSON" in failures


def test_benchmark_artifact_validator_accepts_legacy_scaling_row_without_env(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_legacy_row_env",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    for row in record["runs"]:
        row.pop("threads")
        row.pop("env")

    assert module.validate_record(
        record,
        require_clean=True,
        check_output_files=False,
    ) == []


def test_benchmark_artifact_validator_rejects_scaling_matrix_inputs_child_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_matrix_inputs_child_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["matrix_inputs"]["rna_post_qc"]["nnz"] += 1

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert "runs[0].matrix_inputs must match child JSON" in failures


def test_benchmark_artifact_validator_rejects_incomplete_scaling_row_without_child_check(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_full_scaling_incomplete_row",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    row = record["runs"][0]
    row["setup_peak_rss_gb"] = 0.0
    del row["setup_elapsed_s"]
    del row["elapsed_per_stage"]["enhancer"]
    del row["peak_rss_gb_per_stage"]["aucell"]
    row["peak_rss_gb_per_stage"]["unexpected"] = 1.0
    row["outputs"]["grn_edges"] = 0
    row["outputs"]["aucell_shape"] = [999, 2]
    del row["expected_tf_recovery"]
    del row["cell_barcode_filter"]

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=False,
    )

    assert "runs[0].setup_peak_rss_gb must be positive" in failures
    assert "runs[0].setup_elapsed_s must be an object" in failures
    assert "runs[0].elapsed_per_stage.enhancer missing" in failures
    assert "runs[0].peak_rss_gb_per_stage.aucell missing" in failures
    assert "runs[0].unknown peak_rss_gb_per_stage.unexpected" in failures
    assert "runs[0].outputs.grn_edges must be positive" in failures
    assert "runs[0].outputs.aucell_shape cells must equal n_cells_actual" in failures
    assert "runs[0].expected_tf_recovery must be an object" in failures
    assert "runs[0].cell_barcode_filter must be an object" in failures


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


def test_benchmark_artifact_validator_rejects_manifest_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_manifest_mismatch",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    manifest_info = record["output_inventory"]["manifest_path"]
    manifest_path = Path(manifest_info["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["cell_barcode_filter"] = {"requested": 99, "matched": 99}
    manifest["grn_path"] = str(tmp_path / "wrong_grn.parquet")
    manifest["n_grn_edges"] = 999
    manifest["elapsed"]["grn"] = 999.0
    manifest["memory"]["grn"] = 999.0
    manifest["backend_execution"]["enhancer"] = {
        "engine": "rust",
        "symbols": ["enhancer_link_pearson"],
    }
    manifest_path.write_text(json.dumps(manifest))
    manifest_info["size_bytes"] = manifest_path.stat().st_size

    failures = module.validate_record(
        record,
        require_clean=True,
        check_output_files=True,
    )

    assert (
        "output_inventory.manifest_path cell_barcode_filter must match benchmark JSON"
        in failures
    )
    assert (
        "output_inventory.manifest_path grn_path must match "
        "output_inventory.grn_path.path"
    ) in failures
    assert (
        "output_inventory.manifest_path n_grn_edges must match benchmark JSON"
        in failures
    )
    assert (
        "output_inventory.manifest_path elapsed.grn must match "
        "benchmark elapsed_per_stage.grn"
    ) in failures
    assert (
        "output_inventory.manifest_path memory.grn must match "
        "benchmark peak_rss_gb_per_stage.grn"
    ) in failures
    assert (
        "output_inventory.manifest_path backend_execution.enhancer must match "
        "benchmark backend_execution.pipeline_enhancer"
    ) in failures


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


def test_benchmark_artifact_validator_rejects_weakened_hot_path_scan(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_hot_path_scan_coverage",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["python_hot_paths"]["pattern_count"] = len(HOT_PATH_PATTERNS) - 1
    record["python_hot_paths"]["allowed_hit_count"] = len(ALLOWED_HITS) + 1

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.python_hot_paths.pattern_count must equal "
        f"{len(HOT_PATH_PATTERNS)}"
    ) in failures
    assert (
        "full_pipeline.python_hot_paths.allowed_hit_count must equal "
        f"{len(ALLOWED_HITS)}"
    ) in failures


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


def test_benchmark_artifact_validator_requires_integrated_adata_io_provenance(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_integrated_adata_io",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    missing_record = _full_pipeline_record(tmp_path)
    del missing_record["backend_execution"]["pipeline_integrated_adata"]

    missing_failures = module.validate_record(missing_record, require_clean=True)

    assert (
        "full_pipeline.backend_execution.pipeline_integrated_adata must be an object"
        in missing_failures
    )

    wrong_engine_record = _full_pipeline_record(tmp_path)
    wrong_engine_record["backend_execution"]["pipeline_integrated_adata"] = {
        "engine": "rust",
        "symbols": ["some_rust_symbol"],
    }

    wrong_engine_failures = module.validate_record(
        wrong_engine_record,
        require_clean=True,
    )

    assert (
        "full_pipeline.backend_execution.pipeline_integrated_adata.engine "
        "must be 'python_io'"
    ) in wrong_engine_failures
    assert (
        "full_pipeline.backend_execution.pipeline_integrated_adata.reason "
        "must be a non-empty string"
    ) in wrong_engine_failures


def test_benchmark_artifact_validator_accepts_annotation_pruning_metadata(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_annotation_pruning",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["setup_elapsed_s"]["motif_annotations"] = 0.05
    record["reference_fingerprints"]["motif_annotations"] = {
        "shape": [4, 2],
        "index_name": None,
        "index_sample": ["0"],
        "column_sample": ["motif"],
        "dtype_counts": {"object": 2},
        "corner_sample_sha256": "c" * 64,
    }
    record["shapes"]["motif_annotations"] = [4, 2]
    record["outputs"]["pruned_regulons"] = 1
    record["backend_execution"]["pipeline_cistarget_pruning"] = {
        "engine": "rust",
        "symbols": [
            "cistarget_motif_annotation_prune_standard_rows_f32",
            "cistarget_prune_regulon_targets_i32",
        ],
    }

    assert module.validate_record(record, require_clean=True) == []

    record["backend_execution"]["pipeline_cistarget_pruning"]["symbols"] = [
        "cistarget_motif_annotation_prune_standard_rows_f32"
    ]
    failures = module.validate_record(record, require_clean=True)

    assert any(
        failure.startswith(
            "full_pipeline.backend_execution.pipeline_cistarget_pruning.symbols "
            "must include at least one Rust symbol from"
        )
        for failure in failures
    )


def test_benchmark_artifact_validator_accepts_region_motif_rankings_metadata(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_region_rankings",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["params"]["region_motif_rankings"] = "region_rankings.feather"
    record["setup_elapsed_s"]["region_motif_rankings_metadata"] = 0.02
    record["reference_fingerprints"]["region_motif_rankings"] = {
        "shape": [8, 2000],
        "index_name": None,
        "index_sample": ["file-backed:not-loaded"],
        "column_sample": ["motif", "peak_1"],
        "dtype_counts": {"int32": 2000},
        "corner_sample_sha256": "d" * 64,
        "file_backed": True,
        "format": "feather",
        "metadata_read_columns": ["motifs"],
        "path_name": "region_rankings.feather",
        "size_bytes": 1024,
    }
    record["shapes"]["region_motif_rankings"] = [8, 2000]
    record["backend_execution"]["pipeline_eregulon_peak_regulons"] = {
        "engine": "rust",
        "symbols": ["pipeline_peak_regulons_and_features_from_edges"],
    }
    record["backend_execution"]["pipeline_eregulon_peak_attribution"] = {
        "engine": "rust",
        "symbols": [
            "cistarget_region_attribution_i32",
            "cistarget_region_attribution_peak_values_i32",
            "pipeline_expand_region_cistarget_rows_f32",
        ],
    }

    assert module.validate_record(record, require_clean=True) == []


def test_benchmark_artifact_validator_requires_region_cistarget_symbols(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_requires_region_rankings",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["params"]["region_motif_rankings"] = "region_rankings.feather"
    record["setup_elapsed_s"]["region_motif_rankings_metadata"] = 0.02
    record["reference_fingerprints"]["region_motif_rankings"] = {
        "shape": [8, 2000],
        "index_name": None,
        "index_sample": ["file-backed:not-loaded"],
        "column_sample": ["motif", "peak_1"],
        "dtype_counts": {"int32": 2000},
        "corner_sample_sha256": "d" * 64,
    }
    record["shapes"]["region_motif_rankings"] = [8, 2000]

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.backend_execution.pipeline_eregulon_peak_regulons must be an object"
        in failures
    )
    assert (
        "full_pipeline.backend_execution.pipeline_eregulon_peak_attribution."
        "symbols must include a cistarget_region_attribution_* Rust symbol "
        "when region_motif_rankings is supplied"
    ) in failures
    assert (
        "reference_fingerprints.region_motif_rankings.file_backed must be true"
        in failures
    )


def test_benchmark_artifact_validator_requires_region_peak_filter_with_annotations(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_requires_region_peak_filter",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["params"]["motif_annotations"] = "motif_annotations.tsv"
    record["params"]["region_motif_rankings"] = "region_rankings.feather"
    record["setup_elapsed_s"]["motif_annotations"] = 0.05
    record["setup_elapsed_s"]["region_motif_rankings_metadata"] = 0.02
    record["reference_fingerprints"]["motif_annotations"] = {
        "shape": [4, 2],
        "index_name": None,
        "index_sample": ["0"],
        "column_sample": ["motif"],
        "dtype_counts": {"object": 2},
        "corner_sample_sha256": "c" * 64,
    }
    record["reference_fingerprints"]["region_motif_rankings"] = {
        "shape": [8, 2000],
        "index_name": None,
        "index_sample": ["file-backed:not-loaded"],
        "column_sample": ["motif", "peak_1"],
        "dtype_counts": {"int32": 2000},
        "corner_sample_sha256": "d" * 64,
        "file_backed": True,
        "format": "feather",
        "metadata_read_columns": ["motifs"],
        "path_name": "region_rankings.feather",
        "size_bytes": 1024,
    }
    record["shapes"]["motif_annotations"] = [4, 2]
    record["shapes"]["region_motif_rankings"] = [8, 2000]
    record["backend_execution"]["pipeline_cistarget_pruning"] = {
        "engine": "rust",
        "symbols": [
            "cistarget_motif_annotation_prune_standard_rows_f32",
            "cistarget_prune_regulon_targets_i32",
        ],
    }
    record["backend_execution"]["pipeline_eregulon_peak_regulons"] = {
        "engine": "rust",
        "symbols": ["pipeline_peak_regulons_and_features_from_edges"],
    }
    record["backend_execution"]["pipeline_eregulon_peak_attribution"] = {
        "engine": "rust",
        "symbols": [
            "cistarget_region_attribution_i32",
            "cistarget_region_attribution_peak_values_i32",
            "pipeline_expand_region_cistarget_rows_f32",
        ],
    }

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.backend_execution.pipeline_eregulon_peak_filter must be an object"
        in failures
    )

    record["backend_execution"]["pipeline_eregulon_peak_filter"] = {
        "engine": "rust",
        "symbols": ["pipeline_filter_cistarget_peak_rows_f32"],
    }

    assert module.validate_record(record, require_clean=True) == []


def test_benchmark_artifact_validator_requires_pruning_when_annotations_supplied(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_requires_annotation_pruning",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["params"]["motif_annotations"] = "motif_annotations.tsv"
    record["setup_elapsed_s"]["motif_annotations"] = 0.05
    record["reference_fingerprints"]["motif_annotations"] = {
        "shape": [4, 2],
        "index_name": None,
        "index_sample": ["0"],
        "column_sample": ["motif"],
        "dtype_counts": {"object": 2},
        "corner_sample_sha256": "c" * 64,
    }
    record["shapes"]["motif_annotations"] = [4, 2]

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.backend_execution.pipeline_cistarget_pruning must be an object"
        in failures
    )


def test_benchmark_artifact_validator_requires_scaling_pruning_when_annotations_supplied(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_requires_scaling_annotation_pruning",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["params"]["motif_annotations"] = "motif_annotations.tsv"

    failures = module.validate_record(record, require_clean=True)

    assert (
        "runs[0].backend_execution.pipeline_cistarget_pruning must be an object"
        in failures
    )


def test_benchmark_artifact_validator_requires_scaling_region_peak_filter_with_annotations(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_requires_scaling_region_peak_filter",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["params"]["motif_annotations"] = "motif_annotations.tsv"
    record["params"]["region_motif_rankings"] = "region_rankings.feather"
    for row in record["runs"]:
        row["backend_execution"]["pipeline_cistarget_pruning"] = {
            "engine": "rust",
            "symbols": [
                "cistarget_motif_annotation_prune_standard_rows_f32",
                "cistarget_prune_regulon_targets_i32",
            ],
        }
        row["backend_execution"]["pipeline_eregulon_peak_regulons"] = {
            "engine": "rust",
            "symbols": ["pipeline_peak_regulons_and_features_from_edges"],
        }
        row["backend_execution"]["pipeline_eregulon_peak_attribution"] = {
            "engine": "rust",
            "symbols": [
                "cistarget_region_attribution_i32",
                "cistarget_region_attribution_peak_values_i32",
                "pipeline_expand_region_cistarget_rows_f32",
            ],
        }

    failures = module.validate_record(record, require_clean=True)

    assert (
        "runs[0].backend_execution.pipeline_eregulon_peak_filter must be an object"
        in failures
    )

    for row in record["runs"]:
        row["backend_execution"]["pipeline_eregulon_peak_filter"] = {
            "engine": "rust",
            "symbols": ["pipeline_filter_cistarget_peak_rows_f32"],
        }

    assert module.validate_record(record, require_clean=True) == []


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
    record["env"]["lsf_cores"] = "4"
    record["env"]["lsf_requested_cores"] = "4"
    record["env"]["lsf_requested_mem_mb"] = "0"
    record["env"]["lsf_requested_queue"] = ""

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.env.rayon_num_threads must match params.threads: 4 != 8"
        in failures
    )
    assert (
        "full_pipeline.env.openblas_num_threads must be '1' for reproducible CPU benchmarking"
        in failures
    )
    assert (
        "full_pipeline.env.lsf_cores must match params.threads: 4 != 8"
        in failures
    )
    assert (
        "full_pipeline.env.lsf_requested_cores must match params.threads: 4 != 8"
        in failures
    )
    assert "full_pipeline.env.lsf_requested_mem_mb must be positive" in failures
    assert (
        "full_pipeline.env.lsf_requested_queue must be a non-empty string when present"
        in failures
    )


def test_benchmark_artifact_validator_rejects_missing_cell_barcode_filter(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_cell_barcode_filter_missing",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    del record["cell_barcode_filter"]

    failures = module.validate_record(record, require_clean=True)

    assert "full_pipeline.cell_barcode_filter missing" in failures
    assert "full_pipeline.cell_barcode_filter must be an object" in failures


def test_benchmark_artifact_validator_rejects_invalid_cell_barcode_filter(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_cell_barcode_filter_invalid",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_record(tmp_path)
    record["cell_barcode_filter"] = {"requested": 50, "matched": 50}

    failures = module.validate_record(record, require_clean=True)

    assert (
        "full_pipeline.cell_barcode_filter.requested must equal "
        "full_pipeline.params.n_cells_requested"
    ) in failures
    assert (
        "full_pipeline.cell_barcode_filter.matched must equal "
        "full_pipeline.shapes.rna_post_qc cells"
    ) in failures


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
    del record["setup_elapsed_s"]["subset_requested_cells"]
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
    assert "setup_elapsed_s.subset_requested_cells missing" in failures
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


def test_benchmark_artifact_validator_rejects_invalid_matrix_inputs(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_matrix_inputs",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    missing_record = _full_pipeline_record(tmp_path)
    del missing_record["matrix_inputs"]

    missing_failures = module.validate_record(missing_record, require_clean=True)

    assert "full_pipeline.matrix_inputs must be an object" in missing_failures

    dense_record = _full_pipeline_record(tmp_path)
    dense_record["matrix_inputs"]["rna_post_qc"] = {
        "shape": [100, 999],
        "storage": "dense",
        "format": "ndarray",
        "dtype": "float32",
        "nnz": 100,
        "density": 0.1,
    }
    dense_record["matrix_inputs"]["atac_shared_cells"]["density"] = 1.5

    dense_failures = module.validate_record(dense_record, require_clean=True)

    assert (
        "full_pipeline.matrix_inputs.rna_post_qc.shape must equal "
        "full_pipeline.shapes.rna_post_qc"
    ) in dense_failures
    assert (
        "full_pipeline.matrix_inputs.rna_post_qc.storage must be 'sparse'"
    ) in dense_failures
    assert (
        "full_pipeline.matrix_inputs.rna_post_qc.nnz must be null for dense matrices"
    ) in dense_failures
    assert (
        "full_pipeline.matrix_inputs.rna_post_qc.density must be null for dense matrices"
    ) in dense_failures
    assert (
        "full_pipeline.matrix_inputs.atac_shared_cells.density must be in [0, 1]"
    ) in dense_failures


def test_benchmark_artifact_validator_rejects_scaling_row_matrix_mismatch(tmp_path):
    module = _load_module(
        "validate_benchmark_artifact_scaling_matrix_inputs",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _full_pipeline_scaling_record(tmp_path)
    record["runs"][0]["matrix_inputs"]["rna_post_qc"]["storage"] = "dense"
    record["runs"][0]["matrix_inputs"]["rna_post_qc"]["nnz"] = None
    record["runs"][0]["matrix_inputs"]["rna_post_qc"]["density"] = None

    failures = module.validate_record(record, require_clean=True)

    assert "runs[0].matrix_inputs.rna_post_qc.storage must be 'sparse'" in failures


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


def test_benchmark_artifact_validator_rejects_grn_row_without_rust_execution():
    module = _load_module(
        "validate_benchmark_artifact_grn_backend_execution",
        ROOT / "validation/hpc/minerva/validate_benchmark_artifact.py",
    )
    record = _grn_scaling_record()
    record["thread_scaling"][0]["backend_execution"] = {
        "grn": {
            "engine": "python",
            "symbols": [],
        }
    }
    del record["subset_scaling"][0]["backend_execution"]["grn"]

    failures = module.validate_record(record, require_clean=True)

    assert "subset_scaling[0].backend_execution.grn must be an object" in failures
    assert "thread_scaling[0].backend_execution.grn.engine must be 'rust'" in failures
    assert (
        "thread_scaling[0].backend_execution.grn.symbols must be a non-empty string list"
        in failures
    )


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
    record["subset_scaling"][0]["env"]["lsf_cores"] = "8"
    record["thread_scaling"][0]["env"]["mkl_num_threads"] = "2"

    failures = module.validate_record(record, require_clean=True)

    assert (
        "subset_scaling[0].env.rayon_num_threads must match params.threads: 8 != 4"
        in failures
    )
    assert (
        "subset_scaling[0].env.lsf_cores must match params.threads: 8 != 4"
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
    new_record["matrix_inputs"] = _matrix_inputs_state(200)
    new_record["params"]["n_cells_requested"] = 200
    new_record["cell_barcode_filter"] = {"requested": 200, "matched": 200}
    new_record["outputs"]["aucell_shape"] = [200, 2]
    _sync_full_pipeline_manifest(new_record)
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
    assert full_row["cell_barcode_filter"] == "200/200"
    assert full_row["outputs"] == "grn_edges=10, eregulon_rows=6, regulons=2"
    assert rows[1]["cells"] == "100..200"
    assert rows[1]["cell_barcode_filter"] == "100/100..200/200"
    assert rows[1]["peak_rss_gb"] == "1.25..1.75"
    assert rows[1]["scaling"] == (
        "end_to_end_wall_slope_vs_cells=1, "
        "pipeline_wall_slope_vs_cells=1.322, "
        "pipeline_compute_stage_wall_slope_vs_cells=0, "
        "pipeline_unattributed_wall_slope_vs_cells=3.7, "
        "peak_rss_slope_vs_cells=0.485"
    )


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

    assert table.startswith("| valid | benchmark | dataset | cells | barcode_filter |")
    assert "| yes | real_multiome_full_pipeline | pbmc3k | 100 | 100/100 |" in table
    assert "grn_edges=10, eregulon_rows=6, regulons=2" in table


def test_minerva_launchers_validate_benchmark_artifacts_after_run():
    full = (ROOT / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline.lsf").read_text()
    full_scaling = (
        ROOT / "validation/hpc/minerva/run_real_pbmc3k_full_pipeline_scaling.lsf"
    ).read_text()
    grn = (ROOT / "validation/hpc/minerva/run_real_pbmc3k_grn_scaling.lsf").read_text()
    readme = (ROOT / "validation/hpc/minerva/README.md").read_text()

    for launcher in (full, full_scaling, grn):
        assert "rustscenic_doctor_json_begin" in launcher
        assert "python -m rustscenic doctor --json" in launcher
        assert "rustscenic_doctor_json_end" in launcher
        assert "python_hot_paths_json_begin" in launcher
        assert "python validation/python_hot_paths.py python/rustscenic --json" in launcher
        assert "python_hot_paths_json_end" in launcher
        assert launcher.index("python -m rustscenic doctor --json") < launcher.index(
            "python validation/python_hot_paths.py python/rustscenic --json"
        )
        assert launcher.index(
            "python validation/python_hot_paths.py python/rustscenic --json"
        ) < launcher.index(
            "validation/hpc/minerva/preflight_minerva.py"
        )

    assert "validation/hpc/minerva/prepare_real_pbmc3k_data.py" in full
    assert "validation/hpc/minerva/validate_benchmark_artifact.py" in full
    assert "validation/hpc/minerva/collect_benchmark_results.py" in full
    for required_flag in (
        "--require-repo-import",
        "--require-thread-pins",
        "--require-data-hashes",
        "--require-rust-hot-paths",
        "--require-clean",
    ):
        assert required_flag in readme
        assert required_flag in full
        assert required_flag in full_scaling
        assert required_flag in grn
    assert "--check-output-files" in full
    assert '--threads "${RAYON_NUM_THREADS}"' in full
    assert 'MOTIF_RANKINGS="${MOTIF_RANKINGS:-}"' in full
    assert 'MOTIF_ANNOTATIONS="${MOTIF_ANNOTATIONS:-}"' in full
    assert 'REGION_MOTIF_RANKINGS="${REGION_MOTIF_RANKINGS:-}"' in full
    assert 'GENE_COORDS="${GENE_COORDS:-}"' in full
    assert 'SUMMARY_MAX_ROWS="${SUMMARY_MAX_ROWS:-1000}"' in full
    assert 'SKIP_INTEGRATED_ADATA="${SKIP_INTEGRATED_ADATA:-0}"' in full
    assert "--region-motif-rankings" in full
    assert "--summary-max-rows" in full
    assert "--skip-integrated-adata" in full
    assert "benchmark_args=${BENCHMARK_ARGS[*]}" in full
    assert "export RUSTSCENIC_LSF_PROJECT=acc_DiseaseGeneCell" in full
    assert "export RUSTSCENIC_LSF_QUEUE=express" in full
    assert "export RUSTSCENIC_LSF_CORES=4" in full
    assert "export RUSTSCENIC_LSF_MEM_MB=8000" in full
    assert "export RUSTSCENIC_LSF_WALLTIME=03:00" in full
    assert "lsf_request=project:${RUSTSCENIC_LSF_PROJECT}" in full
    assert full.count('"${REFERENCE_TABLE_ARGS[@]}"') == 2
    assert full.count('"${BENCHMARK_ARGS[@]}"') == 1
    assert "validation/hpc/minerva/prepare_real_pbmc3k_data.py" in full_scaling
    assert "validation/scaling/bench_real_multiome_pipeline_scaling.py" in full_scaling
    assert "validation/hpc/minerva/validate_benchmark_artifact.py" in full_scaling
    assert "validation/hpc/minerva/collect_benchmark_results.py" in full_scaling
    assert "--check-output-files" in full_scaling
    assert 'MOTIF_RANKINGS="${MOTIF_RANKINGS:-}"' in full_scaling
    assert 'MOTIF_ANNOTATIONS="${MOTIF_ANNOTATIONS:-}"' in full_scaling
    assert 'REGION_MOTIF_RANKINGS="${REGION_MOTIF_RANKINGS:-}"' in full_scaling
    assert 'GENE_COORDS="${GENE_COORDS:-}"' in full_scaling
    assert 'SUMMARY_MAX_ROWS="${SUMMARY_MAX_ROWS:-1000}"' in full_scaling
    assert 'SKIP_INTEGRATED_ADATA="${SKIP_INTEGRATED_ADATA:-0}"' in full_scaling
    assert 'CELL_COUNTS="${CELL_COUNTS:-500 1000 2000 2767}"' in full_scaling
    assert 'read -r -a CELL_COUNT_ARGS <<< "${CELL_COUNTS}"' in full_scaling
    assert 'cell_counts=${CELL_COUNT_ARGS[*]}' in full_scaling
    assert "--region-motif-rankings" in full_scaling
    assert "--summary-max-rows" in full_scaling
    assert "--skip-integrated-adata" in full_scaling
    assert "benchmark_args=${BENCHMARK_ARGS[*]}" in full_scaling
    assert "export RUSTSCENIC_LSF_PROJECT=acc_DiseaseGeneCell" in full_scaling
    assert "export RUSTSCENIC_LSF_QUEUE=express" in full_scaling
    assert "export RUSTSCENIC_LSF_CORES=4" in full_scaling
    assert "export RUSTSCENIC_LSF_MEM_MB=8000" in full_scaling
    assert "export RUSTSCENIC_LSF_WALLTIME=08:00" in full_scaling
    assert "lsf_request=project:${RUSTSCENIC_LSF_PROJECT}" in full_scaling
    assert '--cell-counts "${CELL_COUNT_ARGS[@]}"' in full_scaling
    assert full_scaling.count('"${REFERENCE_TABLE_ARGS[@]}"') == 2
    assert full_scaling.count('"${BENCHMARK_ARGS[@]}"') == 1
    assert "validation/hpc/minerva/prepare_real_pbmc3k_data.py" in grn
    assert "validation/hpc/minerva/validate_benchmark_artifact.py" in grn
    assert "validation/hpc/minerva/collect_benchmark_results.py" in grn
    assert 'export RAYON_NUM_THREADS="${LSB_DJOB_NUMPROC:-16}"' in grn
    assert "export RUSTSCENIC_LSF_PROJECT=acc_DiseaseGeneCell" in grn
    assert "export RUSTSCENIC_LSF_QUEUE=express" in grn
    assert "export RUSTSCENIC_LSF_CORES=16" in grn
    assert "export RUSTSCENIC_LSF_MEM_MB=4000" in grn
    assert "export RUSTSCENIC_LSF_WALLTIME=03:00" in grn
    assert "lsf_request=project:${RUSTSCENIC_LSF_PROJECT}" in grn


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

    assert str(preflight.DEFAULT_PROJECT).replace("\\", "/") == expected
    for path in files:
        text = path.read_text()
        assert expected in text
        assert "Huang_lab_project/rustscenic" not in text
