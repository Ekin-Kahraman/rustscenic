"""Prepare and benchmark a real 1.3M-cell 10x RNA GRN workload.

The public 10x E18 mouse-brain matrix is old Cell Ranger v1 HDF5 (genes x
cells, CSC). ``prepare`` performs one fixed, recorded feature selection from
the separately published 20k-cell sample, log-normalises the full matrix and
writes a row-shuffled cells x features CSR HDF5. Every benchmark point is then
a strict prefix of exactly the same real cells and feature universe.

Examples
--------
Prepare once (the measured 1.3M input requires a node with at least 96 GB RAM)::

    python validation/scaling/bench_real_rna_grn_scaling.py prepare \
      --source 1M_neurons_filtered_gene_bc_matrices_h5.h5 \
      --feature-sample 1M_neurons_neuron20k.h5 \
      --out 1M_neurons_rustscenic_grn.h5

Run production-parameter GRN checkpoints in fresh processes::

    python validation/scaling/bench_real_rna_grn_scaling.py scale \
      --prepared 1M_neurons_rustscenic_grn.h5 \
      --sizes 50000 100000 200000 400000 800000 1306127 \
      --n-hvg 2000 --n-tfs 256 --threads 16 --out real_rna_grn.json

This is execution and numerical-invariant evidence. It is not a substitute
for cell-type-aware biological validation on the collaborator's dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


DATASET_NAME = "10x 1.3 Million Brain Cells from E18 Mice"
DATASET_URL = (
    "https://www.10xgenomics.com/datasets/"
    "1-3-million-brain-cells-from-e-18-mice-2-standard-1-3-0"
)
DATASET_LICENSE = "CC BY 4.0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2) if sys.platform == "darwin" else rss / 1024


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _tenx_group(handle):
    """Return a group containing one old- or new-schema 10x matrix."""
    if "matrix" in handle and "data" in handle["matrix"]:
        return handle["matrix"]
    for value in handle.values():
        if hasattr(value, "keys") and {"data", "indices", "indptr", "shape"} <= set(
            value.keys()
        ):
            return value
    if {"data", "indices", "indptr", "shape"} <= set(handle.keys()):
        return handle
    raise ValueError("HDF5 does not contain a recognised 10x sparse matrix")


def load_10x_gene_cell_csc(path: Path) -> tuple[sp.csc_matrix, list[str], list[str]]:
    """Load a 10x HDF5 matrix as its native genes x cells CSC orientation."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = _tenx_group(handle)
        shape = tuple(int(value) for value in np.asarray(group["shape"]))
        # SciPy needs 64-bit compressed indices once nnz exceeds i32::MAX.
        # Read directly into the final width: casting this 2.624B-nnz atlas to
        # int32 and then letting SciPy upcast would create a 10+ GB temporary.
        index_dtype = (
            np.int64
            if int(group["data"].shape[0]) > np.iinfo(np.int32).max
            else np.int32
        )
        matrix = sp.csc_matrix(
            (
                np.asarray(group["data"]),
                np.asarray(group["indices"], dtype=index_dtype),
                np.asarray(group["indptr"], dtype=index_dtype),
            ),
            shape=shape,
        )
        if "gene_names" in group:
            genes = _decode(np.asarray(group["gene_names"]))
        elif "features" in group and "name" in group["features"]:
            genes = _decode(np.asarray(group["features"]["name"]))
        else:
            raise ValueError("10x matrix has no gene_names or features/name dataset")
        barcodes = _decode(np.asarray(group["barcodes"]))

    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError(
            f"10x shape {matrix.shape} disagrees with {len(genes)} genes and "
            f"{len(barcodes)} barcodes"
        )
    return matrix, genes, barcodes


def normalise_log1p_csr(matrix: sp.csr_matrix, totals: np.ndarray) -> None:
    """In-place library-size normalisation without an nnz-sized scale array."""
    totals = np.asarray(totals, dtype=np.float64)
    factors = np.divide(
        10_000.0,
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    ).astype(np.float32)
    block_rows = 65_536
    for start in range(0, matrix.shape[0], block_rows):
        stop = min(start + block_rows, matrix.shape[0])
        left = int(matrix.indptr[start])
        right = int(matrix.indptr[stop])
        counts = np.diff(matrix.indptr[start : stop + 1])
        matrix.data[left:right] *= np.repeat(factors[start:stop], counts)
    np.log1p(matrix.data, out=matrix.data)


def collapse_duplicate_gene_rows(
    matrix: sp.csc_matrix, genes: list[str]
) -> tuple[sp.csc_matrix, list[str], np.ndarray]:
    """Sum duplicate symbols while preserving first-appearance gene order."""
    unique_genes: list[str] = []
    gene_to_unique: dict[str, int] = {}
    inverse = np.empty(len(genes), dtype=np.int32)
    first_source_index: list[int] = []
    for source_index, gene in enumerate(genes):
        unique_index = gene_to_unique.get(gene)
        if unique_index is None:
            unique_index = len(unique_genes)
            gene_to_unique[gene] = unique_index
            unique_genes.append(gene)
            first_source_index.append(source_index)
        inverse[source_index] = unique_index
    if len(unique_genes) == len(genes):
        return matrix, genes, np.arange(len(genes), dtype=np.int32)
    projection = sp.csr_matrix(
        (
            np.ones(len(genes), dtype=np.uint8),
            (inverse, np.arange(len(genes), dtype=np.int32)),
        ),
        shape=(len(unique_genes), len(genes)),
    )
    collapsed = (projection @ matrix).tocsc()
    collapsed.sum_duplicates()
    return collapsed, unique_genes, np.asarray(first_source_index, dtype=np.int32)


def select_collapsed_gene_rows(
    matrix: sp.csc_matrix,
    source_genes: list[str],
    selected_genes: list[str],
) -> sp.csc_matrix:
    """Select named rows and sum only selected duplicate-symbol groups."""
    selected_lookup = {gene: index for index, gene in enumerate(selected_genes)}
    source_indices = np.asarray(
        [index for index, gene in enumerate(source_genes) if gene in selected_lookup],
        dtype=np.int32,
    )
    if not len(source_indices):
        raise AssertionError("selected gene set has no overlap with source")
    target_rows = np.asarray(
        [selected_lookup[source_genes[index]] for index in source_indices],
        dtype=np.int32,
    )
    projection = sp.csr_matrix(
        (
            np.ones(len(source_indices), dtype=np.uint8),
            (target_rows, np.arange(len(source_indices), dtype=np.int32)),
        ),
        shape=(len(selected_genes), len(source_indices)),
    )
    collapsed = (projection @ matrix[source_indices, :]).tocsc()
    collapsed.sum_duplicates()
    return collapsed


def feature_ranks(
    sample_gene_cell: sp.csc_matrix,
    genes: list[str],
    candidate_tfs: list[str],
    *,
    n_hvg: int,
    min_detected_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rank HVGs and expressed TFs once on the published 20k sample."""
    if n_hvg < 1:
        raise ValueError("n_hvg must be positive")
    if min_detected_cells < 1:
        raise ValueError("min_detected_cells must be positive")
    if len(genes) != len(set(genes)):
        duplicates = len(genes) - len(set(genes))
        raise ValueError(
            f"feature selection requires unique gene symbols; found {duplicates} duplicates"
        )

    cells_genes = sample_gene_cell.T.tocsr().astype(np.float32)
    totals = np.asarray(cells_genes.sum(axis=1)).ravel()
    normalise_log1p_csr(cells_genes, totals)
    detected = np.asarray((cells_genes != 0).sum(axis=0)).ravel().astype(np.int64)
    mean = np.asarray(cells_genes.mean(axis=0)).ravel().astype(np.float64)
    mean_square = np.asarray(cells_genes.power(2).mean(axis=0)).ravel().astype(
        np.float64
    )
    variance = np.maximum(mean_square - mean * mean, 0.0)
    dispersion = variance / np.maximum(mean, 1e-12)

    allowed = detected >= min_detected_cells
    nuisance = np.asarray(
        [
            gene.lower().startswith("mt-")
            or gene.startswith("Rpl")
            or gene.startswith("Rps")
            for gene in genes
        ],
        dtype=bool,
    )
    hvg_candidates = np.flatnonzero(allowed & ~nuisance)
    order = np.lexsort(
        (np.asarray(genes, dtype=object)[hvg_candidates], -dispersion[hvg_candidates])
    )
    hvg_indices = hvg_candidates[order[: min(n_hvg, len(order))]]
    hvg_rank = np.full(len(genes), -1, dtype=np.int32)
    hvg_rank[hvg_indices] = np.arange(len(hvg_indices), dtype=np.int32)

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    tf_indices = np.asarray(
        sorted(
            {
                gene_to_index[tf]
                for tf in candidate_tfs
                if tf in gene_to_index and allowed[gene_to_index[tf]]
            },
            key=lambda index: (-dispersion[index], genes[index]),
        ),
        dtype=np.int32,
    )
    tf_rank = np.full(len(genes), -1, dtype=np.int32)
    tf_rank[tf_indices] = np.arange(len(tf_indices), dtype=np.int32)

    selected = np.flatnonzero((hvg_rank >= 0) | (tf_rank >= 0)).astype(np.int32)
    summary = {
        "sample_cells": int(sample_gene_cell.shape[1]),
        "source_genes": int(len(genes)),
        "eligible_genes": int(np.count_nonzero(allowed)),
        "excluded_mito_ribosomal_from_hvg": int(np.count_nonzero(allowed & nuisance)),
        "selected_hvg": int(len(hvg_indices)),
        "expressed_candidate_tfs": int(len(tf_indices)),
        "selected_union": int(len(selected)),
        "min_detected_cells": int(min_detected_cells),
    }
    return selected, hvg_rank, tf_rank, summary


def prepare(args: argparse.Namespace) -> int:
    import h5py
    import rustscenic.data

    started = time.perf_counter()
    sample, raw_sample_genes, _ = load_10x_gene_cell_csc(args.feature_sample)
    sample, sample_genes, first_source_indices = collapse_duplicate_gene_rows(
        sample, raw_sample_genes
    )
    tfs = rustscenic.data.tfs("mouse")
    selected, hvg_rank, tf_rank, feature_summary = feature_ranks(
        sample,
        sample_genes,
        tfs,
        n_hvg=args.prepare_hvg,
        min_detected_cells=args.min_detected_cells,
    )
    del sample

    source, raw_source_genes, barcodes = load_10x_gene_cell_csc(args.source)
    if raw_source_genes != raw_sample_genes:
        raise AssertionError("full matrix and 20k feature sample have different genes")
    if source.shape[1] < 2:
        raise AssertionError("source must contain at least two cells")

    totals = np.asarray(source.sum(axis=0)).ravel()
    rng = np.random.default_rng(args.cell_order_seed)
    cell_order = rng.permutation(source.shape[1]).astype(np.int64)
    selected_genes = [sample_genes[index] for index in selected]
    selected_gene_cell = select_collapsed_gene_rows(
        source, raw_source_genes, selected_genes
    )[:, cell_order]
    cells_features = selected_gene_cell.T.tocsr().astype(np.float32)
    cells_features.sum_duplicates()
    cells_features.sort_indices()
    normalise_log1p_csr(cells_features, totals[cell_order])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    source_sha = sha256_file(args.source)
    sample_sha = sha256_file(args.feature_sample)
    with h5py.File(args.out, "w") as handle:
        x_group = handle.create_group("X")
        x_group.create_dataset(
            "data", data=cells_features.data, compression="lzf", chunks=True
        )
        x_group.create_dataset(
            "indices", data=cells_features.indices.astype(np.int32), compression="lzf"
        )
        x_group.create_dataset("indptr", data=cells_features.indptr.astype(np.int64))
        x_group.attrs["shape"] = cells_features.shape

        var = handle.create_group("var")
        var.create_dataset(
            "gene_name",
            data=np.asarray(selected_genes, dtype=object),
            dtype=string_dtype,
        )
        var.create_dataset("source_index", data=first_source_indices[selected])
        var.create_dataset("hvg_rank", data=hvg_rank[selected])
        var.create_dataset("tf_rank", data=tf_rank[selected])

        obs = handle.create_group("obs")
        obs.create_dataset("source_cell_index", data=cell_order)
        obs.create_dataset(
            "barcode",
            data=np.asarray([barcodes[index] for index in cell_order], dtype=object),
            dtype=string_dtype,
            compression="lzf",
        )

        metadata = {
            "format": "rustscenic-real-rna-grn-csr-v1",
            "dataset": DATASET_NAME,
            "dataset_url": DATASET_URL,
            "dataset_license": DATASET_LICENSE,
            "source_file": args.source.name,
            "source_sha256": source_sha,
            "feature_sample_file": args.feature_sample.name,
            "feature_sample_sha256": sample_sha,
            "original_cells": int(source.shape[1]),
            "original_genes": int(source.shape[0]),
            "unique_gene_symbols": int(len(sample_genes)),
            "duplicate_gene_rows_collapsed": int(
                len(raw_sample_genes) - len(sample_genes)
            ),
            "selected_features": int(cells_features.shape[1]),
            "selected_nnz": int(cells_features.nnz),
            "normalisation": "library_size_10000_then_log1p",
            "cell_order": "numpy_PCG64_permutation",
            "cell_order_seed": int(args.cell_order_seed),
            "feature_selection": feature_summary,
        }
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

    elapsed = time.perf_counter() - started
    result = {
        **metadata,
        "prepared_file": args.out.name,
        "prepared_sha256": sha256_file(args.out),
        "prepare_wall_s": round(elapsed, 3),
        "prepare_peak_rss_mb": round(peak_rss_mb(), 1),
    }
    print("__PREPARED__ " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def prepared_metadata(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        return json.loads(handle.attrs["metadata_json"])


def load_prepared_prefix(
    path: Path,
    *,
    n_cells: int,
    n_hvg: int,
    n_tfs: int,
) -> tuple[sp.csr_matrix, list[str], list[str], np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        shape = tuple(int(value) for value in handle["X"].attrs["shape"])
        if not 1 <= n_cells <= shape[0]:
            raise ValueError(f"n_cells must be in [1, {shape[0]}], got {n_cells}")
        genes = _decode(np.asarray(handle["var/gene_name"]))
        hvg_rank = np.asarray(handle["var/hvg_rank"], dtype=np.int32)
        tf_rank = np.asarray(handle["var/tf_rank"], dtype=np.int32)
        keep = ((hvg_rank >= 0) & (hvg_rank < n_hvg)) | (
            (tf_rank >= 0) & (tf_rank < n_tfs)
        )
        if not np.any(keep):
            raise ValueError("requested feature profile selected no genes")
        selected_tfs = [
            genes[index]
            for index in np.flatnonzero((tf_rank >= 0) & (tf_rank < n_tfs))
        ]
        end = int(handle["X/indptr"][n_cells])
        matrix = sp.csr_matrix(
            (
                np.asarray(handle["X/data"][:end], dtype=np.float32),
                np.asarray(handle["X/indices"][:end], dtype=np.int32),
                np.asarray(handle["X/indptr"][: n_cells + 1], dtype=np.int64),
            ),
            shape=(n_cells, shape[1]),
        )
        source_indices = np.asarray(
            handle["obs/source_cell_index"][:n_cells], dtype=np.int64
        )
    matrix = matrix[:, keep].tocsr()
    kept_genes = [genes[index] for index in np.flatnonzero(keep)]
    return matrix, kept_genes, selected_tfs, source_indices


def canonical_grn_sha256(frame) -> str:
    canonical = frame[["TF", "target", "importance"]].copy()
    canonical["importance"] = canonical["importance"].round(7)
    canonical = canonical.sort_values(["TF", "target"], kind="stable")
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    import rustscenic.aucell
    import rustscenic.grn

    load_started = time.perf_counter()
    matrix, genes, tfs, source_indices = load_prepared_prefix(
        args.prepared,
        n_cells=args.run_one,
        n_hvg=args.n_hvg,
        n_tfs=args.n_tfs,
    )
    load_wall = time.perf_counter() - load_started
    if matrix.nnz == 0 or not np.isfinite(matrix.data).all():
        raise AssertionError("prepared expression prefix must be finite and non-empty")
    if not tfs:
        raise AssertionError("feature profile must include at least one TF")

    grn_started = time.perf_counter()
    adjacencies = rustscenic.grn.infer(
        (matrix, genes),
        tf_names=tfs,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_features=args.max_features,
        subsample=args.subsample,
        max_depth=args.max_depth,
        early_stop_window=args.early_stop_window,
        early_stop_mode=args.early_stop_mode,
        top_targets_per_tf=args.top_targets_per_tf,
        seed=args.seed,
        verbose=True,
    )
    grn_wall = time.perf_counter() - grn_started
    if adjacencies.empty or not np.isfinite(adjacencies["importance"]).all():
        raise AssertionError("GRN must contain finite non-empty importances")

    correlation_started = time.perf_counter()
    signed = rustscenic.grn.add_correlation(
        adjacencies,
        (matrix, genes),
        rho_threshold=args.rho_threshold,
        mask_dropouts=args.mask_dropouts,
    )
    regulons = rustscenic.grn.build_regulons(
        signed,
        top_targets_per_tf=args.regulon_top_targets,
        min_targets=args.regulon_min_targets,
        include_repressors=True,
    )
    correlation_wall = time.perf_counter() - correlation_started

    aucell_started = time.perf_counter()
    auc = rustscenic.aucell.score((matrix, genes), regulons, top_frac=args.top_frac)
    aucell_wall = time.perf_counter() - aucell_started
    auc_values = auc.to_numpy()
    if auc.shape != (args.run_one, len(regulons)) or not np.isfinite(auc_values).all():
        raise AssertionError("AUCell output shape or finite-value check failed")

    fit = dict(adjacencies.attrs.get("grn_fit", {}))
    correlation = dict(signed.attrs.get("correlation", {}))
    polarities = {
        "activator": int(sum(name.endswith("_activator") for name in regulons)),
        "repressor": int(sum(name.endswith("_repressor") for name in regulons)),
    }
    return {
        "n_cells": int(args.run_one),
        "n_genes": int(matrix.shape[1]),
        "n_tfs": int(len(tfs)),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "threads": int(args.threads),
        "load_wall_s": round(load_wall, 3),
        "grn_wall_s": round(grn_wall, 3),
        "correlation_regulon_wall_s": round(correlation_wall, 3),
        "aucell_wall_s": round(aucell_wall, 3),
        "edges": int(len(adjacencies)),
        "output_sha256": canonical_grn_sha256(adjacencies),
        "grn_fit": fit,
        "correlation": correlation,
        "regulons": int(len(regulons)),
        "regulon_polarities": polarities,
        "aucell_shape": [int(value) for value in auc.shape],
        "aucell_nonzero": int(np.count_nonzero(auc_values)),
        "cell_prefix_sha256": hashlib.sha256(source_indices.tobytes()).hexdigest(),
        "gene_profile_sha256": hashlib.sha256(
            "\n".join(genes).encode("utf-8")
        ).hexdigest(),
        "backend_execution": dict(adjacencies.attrs.get("rust_backend", {})),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }


def log_log_slope(rows: list[dict[str, Any]], key: str) -> float:
    xs = np.log([float(row["n_cells"]) for row in rows])
    ys = np.log([float(row[key]) for row in rows])
    return float(np.polyfit(xs, ys, 1)[0])


def source_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--short"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "source_dirty": None}
    return {"commit": commit, "source_dirty": bool(status)}


def _child_command(args: argparse.Namespace, size: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-one",
        "--prepared",
        str(args.prepared),
        "--run-one",
        str(size),
        "--n-hvg",
        str(args.n_hvg),
        "--n-tfs",
        str(args.n_tfs),
        "--threads",
        str(args.threads),
        "--n-estimators",
        str(args.n_estimators),
        "--learning-rate",
        str(args.learning_rate),
        "--max-features",
        str(args.max_features),
        "--subsample",
        str(args.subsample),
        "--max-depth",
        str(args.max_depth),
        "--early-stop-window",
        str(args.early_stop_window),
        "--early-stop-mode",
        args.early_stop_mode,
        "--top-targets-per-tf",
        str(args.top_targets_per_tf),
        "--rho-threshold",
        str(args.rho_threshold),
        "--regulon-top-targets",
        str(args.regulon_top_targets),
        "--regulon-min-targets",
        str(args.regulon_min_targets),
        "--top-frac",
        str(args.top_frac),
        "--seed",
        str(args.seed),
    ] + (["--mask-dropouts"] if args.mask_dropouts else [])


def scale(args: argparse.Namespace) -> int:
    metadata = prepared_metadata(args.prepared)
    if len(args.sizes) < 2:
        raise SystemExit("--sizes must contain at least two checkpoints")
    if args.sizes != sorted(set(args.sizes)):
        raise SystemExit("--sizes must be unique and strictly increasing")
    if args.sizes[-1] > int(metadata["original_cells"]):
        raise SystemExit("largest checkpoint exceeds prepared cell count")

    env = dict(os.environ)
    env.update(
        {
            "RAYON_NUM_THREADS": str(args.threads),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    rows: list[dict[str, Any]] = []
    for size in args.sizes:
        print(f"\n=== real RNA n_cells={size:,} ===", flush=True)
        process = subprocess.Popen(
            _child_command(args, size),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        result = None
        for line in process.stdout:
            print(f"  child: {line}", end="", flush=True)
            if line.startswith("__RESULT__ "):
                result = json.loads(line.split(" ", 1)[1])
        returncode = process.wait()
        if returncode != 0 or result is None:
            return returncode or 1
        rows.append(result)

    grn_slope = log_log_slope(rows, "grn_wall_s")
    memory_slope = log_log_slope(rows, "peak_rss_mb")
    gene_profiles = {row["gene_profile_sha256"] for row in rows}
    checks = {
        "all_outputs_non_empty": all(row["edges"] > 0 for row in rows),
        "all_rust_backend": all(
            row["backend_execution"].get("engine") == "rust" for row in rows
        ),
        "fixed_gene_profile": len(gene_profiles) == 1,
        "aucell_shapes_valid": all(
            row["aucell_shape"] == [row["n_cells"], row["regulons"]]
            for row in rows
        ),
        "signed_semantics_exercised": all(
            row["correlation"].get("activating_edges", 0) > 0
            and row["correlation"].get("repressing_edges", 0) > 0
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"real RNA scaling checks failed: {checks}")

    payload = {
        "benchmark": "real_rna_grn_scaling",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": (
            "Real-cell RNA GRN scaling on one fixed selected-gene profile; "
            "not a complete SCENIC+ biological-equivalence claim."
        ),
        "dataset": {
            key: metadata[key]
            for key in (
                "dataset",
                "dataset_url",
                "dataset_license",
                "source_file",
                "source_sha256",
                "feature_sample_file",
                "feature_sample_sha256",
                "original_cells",
                "original_genes",
                "normalisation",
                "cell_order",
                "cell_order_seed",
                "feature_selection",
            )
        },
        "rustscenic": {"version": version("rustscenic"), **source_state()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "thread_policy": {"rayon": args.threads, "blas_openmp": 1},
        },
        "path_policy": "portable",
        "params": {
            "sizes": args.sizes,
            "n_hvg": args.n_hvg,
            "n_tfs": args.n_tfs,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_features": args.max_features,
            "subsample": args.subsample,
            "max_depth": args.max_depth,
            "early_stop_window": args.early_stop_window,
            "early_stop_mode": args.early_stop_mode,
            "top_targets_per_tf": args.top_targets_per_tf,
            "rho_threshold": args.rho_threshold,
            "mask_dropouts": args.mask_dropouts,
            "regulon_top_targets": args.regulon_top_targets,
            "regulon_min_targets": args.regulon_min_targets,
            "top_frac": args.top_frac,
            "seed": args.seed,
        },
        "results": rows,
        "grn_wall_slope": round(grn_slope, 3),
        "peak_rss_slope": round(memory_slope, 3),
        "correctness_checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nGRN wall-time log-log slope: {grn_slope:.3f}")
    print(f"peak-RSS log-log slope: {memory_slope:.3f}")
    print(f"wrote {args.out}")
    return 0


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--n-hvg", type=int, default=2_000)
    parser.add_argument("--n-tfs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-features", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--early-stop-window", type=int, default=25)
    parser.add_argument(
        "--early-stop-mode", choices=("arboreto", "legacy_inbag"), default="arboreto"
    )
    parser.add_argument("--top-targets-per-tf", type=int, default=50)
    parser.add_argument("--rho-threshold", type=float, default=0.03)
    parser.add_argument("--mask-dropouts", action="store_true")
    parser.add_argument("--regulon-top-targets", type=int, default=50)
    parser.add_argument("--regulon-min-targets", type=int, default=10)
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=777)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--feature-sample", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--prepare-hvg", type=int, default=5_000)
    prepare_parser.add_argument("--min-detected-cells", type=int, default=20)
    prepare_parser.add_argument("--cell-order-seed", type=int, default=20260828)

    scale_parser = subparsers.add_parser("scale")
    add_run_arguments(scale_parser)
    scale_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 400_000, 800_000, 1_306_127],
    )
    scale_parser.add_argument("--out", type=Path, required=True)

    one_parser = subparsers.add_parser("run-one")
    add_run_arguments(one_parser)
    one_parser.add_argument("--run-one", type=int, required=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        return
    if args.n_hvg < 1 or args.n_tfs < 1 or args.threads < 1:
        raise SystemExit("--n-hvg, --n-tfs and --threads must be positive")
    if args.n_estimators < 1 or args.max_depth < 1 or args.early_stop_window < 0:
        raise SystemExit("invalid estimator/depth/early-stop configuration")
    if not 0 < args.max_features <= 1 or not 0 < args.subsample <= 1:
        raise SystemExit("--max-features and --subsample must be in (0, 1]")
    if args.learning_rate <= 0 or args.rho_threshold <= 0:
        raise SystemExit("--learning-rate and --rho-threshold must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "run-one":
        print("__RESULT__ " + json.dumps(run_one(args), sort_keys=True), flush=True)
        return 0
    return scale(args)


if __name__ == "__main__":
    raise SystemExit(main())
