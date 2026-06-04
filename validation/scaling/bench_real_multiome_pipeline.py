"""Run and record a real multiome RustScenic full-pipeline benchmark.

This is the reusable benchmark harness for HPC batch jobs. It runs
``rustscenic.pipeline.run`` on a real 10x-style multiome input, records
runtime, peak RSS, dataset shape, output counts and environment metadata,
and writes a single JSON artefact that can later feed benchmark docs.

Example:
    python validation/scaling/bench_real_multiome_pipeline.py \
        --dataset-name pbmc3k \
        --rna-10x-h5 validation/real_multiome_v036/pbmc_3k_filtered_feature_bc_matrix.h5 \
        --fragments validation/real_multiome_v036/pbmc_3k_atac_fragments.tsv.gz \
        --peaks validation/real_multiome_v036/pbmc_3k_atac_peaks.bed \
        --out-dir /tmp/rustscenic_pbmc3k_outputs \
        --out-json /tmp/rustscenic_pbmc3k.json \
        --expected-tfs SPI1 PAX5 GATA3 TBX21 EBF1 IRF8 TCF7 RUNX3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.backend_requirements import backend_capabilities
from validation.python_hot_paths import hot_path_state
from validation.process_memory import peak_rss_gb
from validation.repo_cleanliness import repo_state_from_git_outputs


DEFAULT_SUMMARY_MAX_ROWS = 1000


def configure_thread_env(threads: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def env_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_first(path: Path, n_bytes: int = 8 * 1024 * 1024) -> str:
    with path.open("rb") as handle:
        return hashlib.md5(handle.read(n_bytes)).hexdigest()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def repo_state() -> dict[str, Any]:
    """Return enough git state to decide whether a benchmark is release-grade."""
    commit = _git_output(["rev-parse", "HEAD"])
    tracked_status = _git_output(["status", "--short", "--untracked-files=no"]) or ""
    untracked_status = _git_output(["status", "--short", "--untracked-files=all"]) or ""
    tracked_diff = _git_output(["diff", "HEAD", "--binary", "--no-ext-diff"]) or ""
    return repo_state_from_git_outputs(
        commit=commit,
        tracked_status=tracked_status,
        untracked_status=untracked_status,
        tracked_diff=tracked_diff,
    )


def _path_under(path: str | None, root: Path) -> bool | None:
    if not path:
        return None
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def runtime_import_state() -> dict[str, Any]:
    """Record the actual Python package and extension imported by this run."""
    import rustscenic

    package_version = getattr(rustscenic, "__version__", None)
    try:
        import rustscenic._rustscenic as ext
    except Exception as exc:  # pragma: no cover - benchmark provenance path
        extension_file = None
        extension_version = None
        extension_error = repr(exc)
    else:
        extension_file = getattr(ext, "__file__", None)
        extension_version = getattr(ext, "__version__", None)
        extension_error = None

    package_file = getattr(rustscenic, "__file__", None)
    return {
        "package_version": package_version,
        "extension_version": extension_version,
        "package_file": package_file,
        "package_under_repo": _path_under(package_file, REPO_ROOT),
        "extension_file": extension_file,
        "extension_under_repo": _path_under(extension_file, REPO_ROOT),
        "extension_error": extension_error,
    }


def file_info(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if p.is_dir():
        return {
            "path": str(p),
            "exists": True,
            "type": "dir",
            "entries": len(list(p.iterdir())),
        }
    return {
        "path": str(p),
        "exists": True,
        "type": "file",
        "size_bytes": p.stat().st_size,
    }


def explicit_reference_source(path: Path) -> dict[str, Any]:
    return {
        "source": "explicit_path",
        "path": str(path),
        "exists_before": path.exists(),
        "exists_after": path.exists(),
        "cache_exists_before": None,
        "cache_exists_after": None,
    }


def default_reference_source(
    path: Path,
    *,
    cache_exists_before: bool,
) -> dict[str, Any]:
    cache_exists_after = path.exists()
    return {
        "source": "default_cache" if cache_exists_before else "default_download",
        "path": str(path),
        "exists_before": cache_exists_before,
        "exists_after": cache_exists_after,
        "cache_exists_before": cache_exists_before,
        "cache_exists_after": cache_exists_after,
    }


def backend_execution_for_benchmark(result) -> dict[str, Any]:
    execution = {
        "setup_fragments_to_matrix": {
            "engine": "rust",
            "symbols": ["preproc_fragments_to_matrix"],
        }
    }
    for stage, state in result.backend_execution.items():
        execution[f"pipeline_{stage}"] = state
    return execution


def _jsonable_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(float(value), 6)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _records(df: pd.DataFrame, *, n: int) -> list[dict[str, Any]]:
    return [
        {str(col): _jsonable_value(value) for col, value in row.items()}
        for row in df.head(n).to_dict("records")
    ]


def _read_parquet_columns(
    path: str | Path | None,
    columns: list[str],
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=columns)
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=columns)
    if max_rows is not None:
        return _read_parquet_columns_bounded(p, columns, max_rows=max_rows)
    try:
        df = pd.read_parquet(p, columns=columns)
    except Exception:
        df = pd.read_parquet(p)
        df = df[[col for col in columns if col in df.columns]]
    return df


def _read_parquet_columns_bounded(
    path: Path,
    columns: list[str],
    *,
    max_rows: int,
) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.DataFrame(columns=columns)
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    read_columns = [col for col in columns if col in available]
    if not read_columns:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    remaining = int(max_rows)
    for row_group in range(parquet.num_row_groups):
        if remaining <= 0:
            break
        table = parquet.read_row_group(row_group, columns=read_columns)
        if table.num_rows > remaining:
            table = table.slice(0, remaining)
        frame = table.to_pandas()
        frames.append(frame)
        remaining -= len(frame)
    if not frames:
        return pd.DataFrame(columns=read_columns)
    return pd.concat(frames, ignore_index=True)


def output_summaries(
    result: Any,
    *,
    n: int = 10,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Read compact biological summaries from output artefacts.

    Counts come from ``PipelineResult``. This post-run summary is deliberately
    small and human-auditable, so benchmark JSON can be inspected without
    loading full parquet outputs.
    """
    grn = _read_parquet_columns(
        result.grn_path,
        ["TF", "target", "importance"],
        max_rows=max_rows,
    )
    if {"TF", "target", "importance"}.issubset(grn.columns):
        grn = grn.sort_values("importance", ascending=False, kind="mergesort")

    cistarget = _read_parquet_columns(
        result.cistarget_path,
        ["regulon", "motif", "auc", "nes"],
        max_rows=max_rows,
    )
    cistarget_sort = [col for col in ("nes", "auc") if col in cistarget.columns]
    if cistarget_sort:
        cistarget = cistarget.sort_values(cistarget_sort, ascending=False, kind="mergesort")

    enhancer = _read_parquet_columns(
        result.enhancer_links_path,
        ["peak_id", "gene", "correlation", "distance"],
        max_rows=max_rows,
    )
    if "correlation" in enhancer.columns:
        enhancer = enhancer.assign(abs_correlation=enhancer["correlation"].abs())
        enhancer = enhancer.sort_values("abs_correlation", ascending=False, kind="mergesort")

    eregulons = _read_parquet_columns(
        result.eregulons_path,
        ["tf", "enhancer", "target_gene", "n_enhancer_links", "motif_auc"],
        max_rows=max_rows,
    )
    eregulon_sort = [col for col in ("motif_auc", "n_enhancer_links") if col in eregulons.columns]
    if eregulon_sort:
        eregulons = eregulons.sort_values(eregulon_sort, ascending=False, kind="mergesort")

    regulons_sample: list[str] = []
    if result.regulons_path and Path(result.regulons_path).exists():
        payload = json.loads(Path(result.regulons_path).read_text())
        regulons_sample = [str(name) for name in list(payload)[:n]]

    return {
        "active_regulons_sample": regulons_sample,
        "top_grn_edges": _records(grn, n=n),
        "top_cistarget_rows": _records(cistarget, n=n),
        "top_enhancer_links": _records(enhancer, n=n),
        "top_eregulon_rows": _records(eregulons, n=n),
        "summary_max_rows": max_rows,
    }


def load_and_qc_rna(path: Path):
    import scanpy as sc

    rna = sc.read_10x_h5(path)
    rna.var_names_make_unique()
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, inplace=True)
    rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    return rna


def subset_shared_cells(rna, atac, *, n_cells: int | None, seed: int):
    shared = np.asarray(sorted(set(rna.obs_names) & set(atac.obs_names)), dtype=object)
    if n_cells is not None and n_cells < len(shared):
        rng = np.random.default_rng(seed)
        shared = np.asarray(sorted(rng.choice(shared, size=n_cells, replace=False)), dtype=object)
    return rna[shared].copy(), atac[shared].copy()


def subset_requested_cells(adata, *, n_cells: int | None, seed: int):
    """Apply requested cell-count scaling before fragment-matrix construction."""
    if n_cells is None or n_cells >= adata.n_obs:
        return adata
    rng = np.random.default_rng(seed)
    cells = np.asarray(sorted(adata.obs_names), dtype=object)
    selected = np.asarray(
        sorted(rng.choice(cells, size=n_cells, replace=False)),
        dtype=object,
    )
    return adata[selected].copy()


def load_optional_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".feather", ".ft"}:
        df = pd.read_feather(path)
        for col in ("motifs", "features", "#motif_id"):
            if col in df.columns:
                return df.set_index(col)
        return df
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"unsupported table format for {path}")


def file_backed_table_fingerprint(path: Path, *, sample: int = 8) -> dict[str, Any]:
    """Return metadata for a large reference table without loading it.

    Region motif-ranking databases can be tens of GB wide. Full DataFrame
    loading would defeat the benchmark's memory model, so record schema and a
    small byte sample while letting ``pipeline.run`` project columns on read.
    """
    suffix = path.suffix.lower()
    n_rows: int | None = None
    columns: list[str] = []
    dtype_counts: dict[str, int] = {}
    metadata_read_columns: list[str] = []
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        n_rows = int(parquet.metadata.num_rows)
        arrow_schema = parquet.schema_arrow
        columns = [str(name) for name in arrow_schema.names]
        for field in arrow_schema:
            dtype = str(field.type)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    elif suffix in {".feather", ".ft"}:
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc

        with ipc.open_file(str(path)) as reader:
            schema = reader.schema
            columns = [str(name) for name in schema.names]
            for field in schema:
                dtype = str(field.type)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        count_col = _file_backed_row_count_column(columns, path)
        metadata_read_columns = [count_col]
        n_rows = int(feather.read_table(path, columns=[count_col]).num_rows)
    else:
        raise ValueError(f"unsupported file-backed table format for {path}")

    digest = hashlib.sha256()
    digest.update(str(path.name).encode())
    digest.update(str(path.stat().st_size).encode())
    digest.update(json.dumps(columns[:sample] + columns[-sample:], sort_keys=True).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "shape": [int(n_rows or 0), len(columns)],
        "index_name": None,
        "index_sample": ["file-backed:not-loaded"],
        "column_sample": columns[: min(sample, len(columns))],
        "dtype_counts": dtype_counts,
        "corner_sample_sha256": digest.hexdigest(),
        "file_backed": True,
        "format": suffix.removeprefix("."),
        "metadata_read_columns": metadata_read_columns,
        "path_name": path.name,
        "size_bytes": path.stat().st_size,
    }


def _file_backed_row_count_column(columns: list[str], path: Path) -> str:
    if not columns:
        raise ValueError(f"file-backed table has no columns: {path}")
    for candidate in ("motifs", path.stem, "motif", "motif_id", "features", "feature"):
        if candidate in columns:
            return candidate
    return columns[0]


def dataframe_fingerprint(df: pd.DataFrame, *, sample: int = 8) -> dict[str, Any]:
    """Return a cheap deterministic fingerprint for large reference tables.

    Full byte hashing of motif ranking databases can dominate setup on HPC.
    This records the shape, labels, dtype mix, and a stable hash of the
    dataframe corners so benchmark artefacts still identify the reference
    tables that were actually used.
    """
    n_rows, n_cols = df.shape
    row_positions = sorted(
        {
            *range(min(sample, n_rows)),
            *range(max(0, n_rows - sample), n_rows),
        }
    )
    col_positions = sorted(
        {
            *range(min(sample, n_cols)),
            *range(max(0, n_cols - sample), n_cols),
        }
    )
    corner = df.iloc[row_positions, col_positions] if row_positions and col_positions else df.iloc[[]]
    dtype_counts: dict[str, int] = {}
    for dtype in df.dtypes.astype(str):
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    hash_values = pd.util.hash_pandas_object(corner, index=True).to_numpy(dtype=np.uint64)
    digest_payload = {
        "index": [str(v) for v in corner.index.tolist()],
        "columns": [str(v) for v in corner.columns.tolist()],
        "dtype_counts": dtype_counts,
    }
    digest = hashlib.sha256()
    digest.update(hash_values.tobytes())
    digest.update(json.dumps(digest_payload, sort_keys=True).encode())
    return {
        "shape": [int(n_rows), int(n_cols)],
        "index_name": None if df.index.name is None else str(df.index.name),
        "index_sample": [str(v) for v in df.index[: min(sample, n_rows)].tolist()],
        "column_sample": [str(v) for v in df.columns[: min(sample, n_cols)].tolist()],
        "dtype_counts": dtype_counts,
        "corner_sample_sha256": digest.hexdigest(),
    }


def matrix_profile(adata) -> dict[str, Any]:
    """Record sparse/dense matrix provenance without scanning dense payloads."""
    import scipy.sparse as sp

    rows, cols = int(adata.n_obs), int(adata.n_vars)
    shape = [rows, cols]
    matrix = adata.X
    dtype = str(getattr(matrix, "dtype", "unknown"))
    if sp.issparse(matrix):
        nnz = int(matrix.nnz)
        total = rows * cols
        density = 0.0 if total == 0 else round(nnz / total, 8)
        return {
            "shape": shape,
            "storage": "sparse",
            "format": matrix.getformat(),
            "dtype": dtype,
            "nnz": nnz,
            "density": density,
        }
    return {
        "shape": shape,
        "storage": "dense",
        "format": type(matrix).__name__,
        "dtype": dtype,
        "nnz": None,
        "density": None,
    }


def benchmark_env() -> dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    optional_env = {
        "lsf_jobid": os.environ.get("LSB_JOBID"),
        "lsf_queue": os.environ.get("LSB_QUEUE"),
        "lsf_cores": os.environ.get("LSB_DJOB_NUMPROC"),
        "lsf_project": os.environ.get("RUSTSCENIC_LSF_PROJECT"),
        "lsf_requested_queue": os.environ.get("RUSTSCENIC_LSF_QUEUE"),
        "lsf_requested_cores": os.environ.get("RUSTSCENIC_LSF_CORES"),
        "lsf_requested_mem_mb": os.environ.get("RUSTSCENIC_LSF_MEM_MB"),
        "lsf_requested_walltime": os.environ.get("RUSTSCENIC_LSF_WALLTIME"),
    }
    env.update({key: value for key, value in optional_env.items() if value})
    return env


def strip_regulon_name(name: str) -> str:
    value = str(name)
    for suffix in ("(+)", "(-)"):
        if value.endswith(suffix):
            value = value[:-3].strip()
    for suffix in ("_regulon", "_extended", "_activator", "_repressor"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    import rustscenic.data
    import rustscenic.pipeline
    import rustscenic.preproc

    state = repo_state()
    if args.require_clean and state["source_dirty"]:
        raise SystemExit(
            "source files are dirty; commit tracked changes and add or remove "
            "untracked source files before running a publication-grade "
            "benchmark. Use without --require-clean only for explicit "
            "local-build profiling."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    t_end_to_end = time.perf_counter()
    t_setup = t_end_to_end
    setup_elapsed: dict[str, float] = {}

    print(f"[setup] loading RNA from {args.rna_10x_h5}", flush=True)
    t_step = time.perf_counter()
    rna = load_and_qc_rna(args.rna_10x_h5)
    setup_elapsed["load_rna_qc"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    rna = subset_requested_cells(rna, n_cells=args.n_cells, seed=args.seed)
    setup_elapsed["subset_requested_cells"] = time.perf_counter() - t_step

    print(f"[setup] building ATAC matrix from {args.fragments}", flush=True)
    t_step = time.perf_counter()
    atac = rustscenic.preproc.fragments_to_matrix(
        args.fragments,
        args.peaks,
        cell_barcodes=[str(cell) for cell in rna.obs_names],
    )
    cell_barcode_filter = {
        str(key): int(value)
        for key, value in dict(atac.uns.get("cell_barcode_filter", {})).items()
    }
    setup_elapsed["fragments_to_matrix"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    rna, atac = subset_shared_cells(rna, atac, n_cells=args.n_cells, seed=args.seed)
    setup_elapsed["subset_shared_cells"] = time.perf_counter() - t_step

    reference_sources: dict[str, dict[str, Any]] = {}

    t_step = time.perf_counter()
    motif_rankings = load_optional_table(args.motif_rankings)
    motif_rankings_default_path = None
    motif_rankings_cache_exists_before = None
    if motif_rankings is None:
        motif_rankings_default_path = rustscenic.data._motif_rankings_cache_path(
            species=args.motif_species,
        )
        motif_rankings_cache_exists_before = motif_rankings_default_path.exists()
        print("[setup] downloading or loading cached motif rankings", flush=True)
        motif_rankings = rustscenic.data.download_motif_rankings(
            species=args.motif_species,
            verbose=not args.quiet,
        )
        reference_sources["motif_rankings"] = default_reference_source(
            motif_rankings_default_path,
            cache_exists_before=motif_rankings_cache_exists_before,
        )
    else:
        reference_sources["motif_rankings"] = explicit_reference_source(
            args.motif_rankings
        )
    setup_elapsed["motif_rankings"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    motif_annotations = load_optional_table(args.motif_annotations)
    if args.motif_annotations is not None:
        reference_sources["motif_annotations"] = explicit_reference_source(
            args.motif_annotations
        )
        setup_elapsed["motif_annotations"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    region_motif_rankings = args.region_motif_rankings
    region_motif_rankings_fingerprint = None
    if region_motif_rankings is not None:
        reference_sources["region_motif_rankings"] = explicit_reference_source(
            region_motif_rankings
        )
        region_motif_rankings_fingerprint = file_backed_table_fingerprint(
            region_motif_rankings
        )
        setup_elapsed["region_motif_rankings_metadata"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    gene_coords = load_optional_table(args.gene_coords)
    gene_coords_default_path = None
    gene_coords_cache_exists_before = None
    if gene_coords is None:
        gene_coords_default_path = Path(
            rustscenic.data._gene_coords_cache_paths(
                species=args.gene_species,
            )["parquet_path"]
        )
        gene_coords_cache_exists_before = gene_coords_default_path.exists()
        print("[setup] downloading or loading cached gene coordinates", flush=True)
        gene_coords = rustscenic.data.download_gene_coords(
            species=args.gene_species,
            verbose=not args.quiet,
        )
        reference_sources["gene_coords"] = default_reference_source(
            gene_coords_default_path,
            cache_exists_before=gene_coords_cache_exists_before,
        )
    else:
        reference_sources["gene_coords"] = explicit_reference_source(
            args.gene_coords
        )
    setup_elapsed["gene_coords"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    tfs = rustscenic.data.tfs(species=args.tf_species)
    setup_elapsed["tf_list"] = time.perf_counter() - t_step
    setup_wall = time.perf_counter() - t_setup
    setup_peak_rss_gb = peak_rss_gb()

    print(
        "[pipeline] running full pipeline "
        f"cells={rna.n_obs:,}, genes={rna.n_vars:,}, peaks={atac.n_vars:,}",
        flush=True,
    )
    t_pipeline = time.perf_counter()
    result = rustscenic.pipeline.run(
        rna=rna,
        output_dir=args.out_dir,
        adata_atac=atac,
        peaks=args.peaks,
        motif_rankings=motif_rankings,
        motif_annotations=motif_annotations,
        region_motif_rankings=region_motif_rankings,
        gene_coords=gene_coords,
        tfs=tfs,
        grn_n_estimators=args.grn_n_estimators,
        grn_max_features=args.grn_max_features,
        grn_target_block_size=args.grn_target_block_size,
        topics_n_topics=args.topics_n_topics,
        topics_n_passes=args.topics_n_passes,
        topics_method=args.topics_method,
        topics_n_iters=args.topics_n_iters,
        topics_n_threads=args.topics_n_threads,
        cistarget_top_frac=args.cistarget_top_frac,
        cistarget_auc_threshold=args.cistarget_auc_threshold,
        cistarget_nes_threshold=args.cistarget_nes_threshold,
        enhancer_max_distance=args.enhancer_max_distance,
        enhancer_min_abs_corr=args.enhancer_min_abs_corr,
        eregulon_min_target_genes=args.eregulon_min_target_genes,
        eregulon_min_enhancer_links=args.eregulon_min_enhancer_links,
        write_integrated_adata=not args.skip_integrated_adata,
        seed=args.seed,
        verbose=not args.quiet,
    )
    pipeline_wall = time.perf_counter() - t_pipeline
    end_to_end_wall = time.perf_counter() - t_end_to_end
    pipeline_compute_stage_wall = sum(float(v) for v in result.elapsed.values())
    pipeline_unattributed_wall = max(
        0.0,
        pipeline_wall - pipeline_compute_stage_wall,
    )

    regulon_names: set[str] = set()
    if result.regulons_path and Path(result.regulons_path).exists():
        regulons_payload = json.loads(Path(result.regulons_path).read_text())
        regulon_names = {strip_regulon_name(name) for name in regulons_payload}

    expected = list(args.expected_tfs)
    found = sorted(tf for tf in expected if tf in regulon_names)

    artefact_paths = {
        "atac_matrix_path": result.atac_matrix_path,
        "grn_path": result.grn_path,
        "regulons_path": result.regulons_path,
        "candidate_regulons_path": result.candidate_regulons_path,
        "aucell_path": result.aucell_path,
        "topics_dir": result.topics_dir,
        "cistarget_path": result.cistarget_path,
        "enhancer_links_path": result.enhancer_links_path,
        "eregulons_path": result.eregulons_path,
        "integrated_adata_path": result.integrated_adata_path,
        "manifest_path": result.output_dir / "manifest.json",
    }

    reference_fingerprints = {
        "motif_rankings": dataframe_fingerprint(motif_rankings),
        "gene_coords": dataframe_fingerprint(gene_coords),
    }
    if motif_annotations is not None:
        reference_fingerprints["motif_annotations"] = dataframe_fingerprint(
            motif_annotations
        )
    if region_motif_rankings_fingerprint is not None:
        reference_fingerprints["region_motif_rankings"] = (
            region_motif_rankings_fingerprint
        )

    shapes = {
        "rna_post_qc": [int(rna.n_obs), int(rna.n_vars)],
        "atac_shared_cells": [int(atac.n_obs), int(atac.n_vars)],
        "motif_rankings": list(motif_rankings.shape),
        "gene_coords_rows": int(len(gene_coords)),
        "tfs_supplied": int(len(tfs)),
    }
    if motif_annotations is not None:
        shapes["motif_annotations"] = list(motif_annotations.shape)
    if region_motif_rankings_fingerprint is not None:
        shapes["region_motif_rankings"] = region_motif_rankings_fingerprint["shape"]
    matrix_inputs = {
        "rna_post_qc": matrix_profile(rna),
        "atac_shared_cells": matrix_profile(atac),
    }

    outputs = {
        "grn_edges": int(result.n_grn_edges or 0),
        "candidate_regulons": int(result.n_candidate_regulons or 0),
        "regulons": int(result.n_regulons or 0),
        "cistarget_rows": int(result.n_cistarget_rows or 0),
        "enhancer_links": int(result.n_enhancer_links or 0),
        "eregulon_rows": int(result.n_eregulon_rows or 0),
        "eregulons": int(result.n_eregulons or 0),
        "aucell_shape": list(result.aucell_shape or [0, 0]),
    }
    if result.n_pruned_regulons is not None:
        outputs["pruned_regulons"] = int(result.n_pruned_regulons)

    record = {
        "benchmark": "real_multiome_full_pipeline",
        "dataset_name": args.dataset_name,
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_state": state,
        "runtime_import": runtime_import_state(),
        "backend_capabilities": backend_capabilities(),
        "python_hot_paths": hot_path_state(),
        "backend_execution": backend_execution_for_benchmark(result),
        "cell_barcode_filter": cell_barcode_filter,
        "rustscenic": version("rustscenic"),
        "input_hashes": {
            "rna_10x_h5_md5": md5(args.rna_10x_h5),
            "fragments_first_8mb_md5": md5_first(args.fragments),
            "peaks_bed_md5": md5(args.peaks),
        },
        "reference_fingerprints": reference_fingerprints,
        "reference_sources": reference_sources,
        "params": {
            "n_cells_requested": args.n_cells,
            "grn_n_estimators": args.grn_n_estimators,
            "grn_max_features": args.grn_max_features,
            "grn_target_block_size": args.grn_target_block_size,
            "topics_n_topics": args.topics_n_topics,
            "topics_n_passes": args.topics_n_passes,
            "topics_method": args.topics_method,
            "topics_n_iters": args.topics_n_iters,
            "topics_n_threads": args.topics_n_threads,
            "threads": args.threads,
            "rayon_num_threads": env_positive_int("RAYON_NUM_THREADS"),
            "motif_annotations": str(args.motif_annotations) if args.motif_annotations else None,
            "region_motif_rankings": str(args.region_motif_rankings) if args.region_motif_rankings else None,
            "cistarget_nes_threshold": args.cistarget_nes_threshold,
            "enhancer_max_distance": args.enhancer_max_distance,
            "enhancer_min_abs_corr": args.enhancer_min_abs_corr,
            "eregulon_min_target_genes": args.eregulon_min_target_genes,
            "eregulon_min_enhancer_links": args.eregulon_min_enhancer_links,
            "write_integrated_adata": not args.skip_integrated_adata,
            "summary_max_rows": args.summary_max_rows,
            "seed": args.seed,
        },
        "shapes": shapes,
        "matrix_inputs": matrix_inputs,
        "wall_s": {
            "setup": round(setup_wall, 3),
            "pipeline": round(pipeline_wall, 3),
            "pipeline_compute_stages": round(pipeline_compute_stage_wall, 3),
            "pipeline_unattributed": round(pipeline_unattributed_wall, 3),
            "end_to_end": round(end_to_end_wall, 3),
        },
        "setup_elapsed_s": {
            k: round(float(v), 6) for k, v in setup_elapsed.items()
        },
        "setup_peak_rss_gb": round(setup_peak_rss_gb, 6),
        "peak_rss_gb": round(peak_rss_gb(), 6),
        "elapsed_per_stage": {k: round(float(v), 6) for k, v in result.elapsed.items()},
        "peak_rss_gb_per_stage": {
            k: round(float(v), 6) for k, v in result.memory.items()
        },
        "outputs": outputs,
        "expected_tf_recovery": {
            "expected_tfs": expected,
            "found": found,
            "missing": sorted(set(expected) - set(found)),
            "fraction": None if not expected else round(len(found) / len(expected), 6),
        },
        "output_summaries": output_summaries(
            result,
            n=args.summary_rows,
            max_rows=args.summary_max_rows,
        ),
        "output_inventory": {k: file_info(v) for k, v in artefact_paths.items()},
        "env": benchmark_env(),
    }
    args.out_json.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[done] wrote {args.out_json}", flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return record


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.dataset_name).strip():
        raise SystemExit("--dataset-name must be non-empty")
    if args.n_cells is not None and args.n_cells <= 0:
        raise SystemExit("--n-cells must be positive when set")
    positive_int_fields = {
        "threads": args.threads,
        "grn_n_estimators": args.grn_n_estimators,
        "topics_n_topics": args.topics_n_topics,
        "topics_n_passes": args.topics_n_passes,
        "topics_n_iters": args.topics_n_iters,
        "topics_n_threads": args.topics_n_threads,
        "eregulon_min_target_genes": args.eregulon_min_target_genes,
        "eregulon_min_enhancer_links": args.eregulon_min_enhancer_links,
        "summary_rows": args.summary_rows,
    }
    for name, value in positive_int_fields.items():
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.grn_target_block_size is not None and args.grn_target_block_size <= 0:
        raise SystemExit("--grn-target-block-size must be positive when set")
    if args.summary_max_rows is not None and args.summary_max_rows <= 0:
        raise SystemExit("--summary-max-rows must be positive when set")
    if not (0 < args.grn_max_features <= 1):
        raise SystemExit("--grn-max-features must be in (0, 1]")
    if not (0 < args.cistarget_top_frac <= 1):
        raise SystemExit("--cistarget-top-frac must be in (0, 1]")
    if args.cistarget_auc_threshold < 0:
        raise SystemExit("--cistarget-auc-threshold must be non-negative")
    if args.cistarget_nes_threshold < 0:
        raise SystemExit("--cistarget-nes-threshold must be non-negative")
    if args.enhancer_max_distance < 0:
        raise SystemExit("--enhancer-max-distance must be non-negative")
    if args.enhancer_min_abs_corr < 0:
        raise SystemExit("--enhancer-min-abs-corr must be non-negative")
    for name in (
        "motif_rankings",
        "motif_annotations",
        "region_motif_rankings",
        "gene_coords",
    ):
        path = getattr(args, name)
        if path is not None and not path.exists():
            raise SystemExit(f"missing input: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--rna-10x-h5", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--peaks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--n-cells", type=int, default=None)
    parser.add_argument("--motif-rankings", type=Path, default=None)
    parser.add_argument("--motif-annotations", type=Path, default=None)
    parser.add_argument("--region-motif-rankings", type=Path, default=None)
    parser.add_argument("--gene-coords", type=Path, default=None)
    parser.add_argument("--motif-species", default="human")
    parser.add_argument("--gene-species", default="hs")
    parser.add_argument("--tf-species", default="hs")
    parser.add_argument("--expected-tfs", nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grn-n-estimators", type=int, default=100)
    parser.add_argument("--grn-max-features", type=float, default=0.1)
    parser.add_argument("--grn-target-block-size", type=int, default=None)
    parser.add_argument("--topics-n-topics", type=int, default=10)
    parser.add_argument("--topics-n-passes", type=int, default=3)
    parser.add_argument("--topics-method", choices=["vb", "gibbs"], default="vb")
    parser.add_argument("--topics-n-iters", type=int, default=200)
    parser.add_argument("--topics-n-threads", type=int, default=1)
    parser.add_argument("--cistarget-top-frac", type=float, default=0.05)
    parser.add_argument("--cistarget-auc-threshold", type=float, default=0.05)
    parser.add_argument("--cistarget-nes-threshold", type=float, default=3.0)
    parser.add_argument("--enhancer-max-distance", type=int, default=500_000)
    parser.add_argument("--enhancer-min-abs-corr", type=float, default=0.1)
    parser.add_argument("--eregulon-min-target-genes", type=int, default=2)
    parser.add_argument("--eregulon-min-enhancer-links", type=int, default=1)
    parser.add_argument("--skip-integrated-adata", action="store_true")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--summary-rows", type=int, default=10)
    parser.add_argument("--summary-max-rows", type=int, default=DEFAULT_SUMMARY_MAX_ROWS)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail if tracked files differ from HEAD; use for publication-grade runs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    configure_thread_env(args.threads)
    for path in (args.rna_10x_h5, args.fragments, args.peaks):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
