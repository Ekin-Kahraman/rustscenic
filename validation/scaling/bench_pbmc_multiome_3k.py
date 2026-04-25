"""Baseline benchmark on the 10x public PBMC 3k Multiome.

Runs the same pipeline.run path Codex measured (multiome 3k = 10.4 min total)
on the real dataset, with stage-by-stage wall-clock so we can target perf
fixes at the dominant cost.

Inputs (downloaded once into validation/real_multiome/):
    rna.h5              — 10x filtered_feature_bc_matrix.h5 (RNA + ATAC counts)
    fragments.tsv.gz    — 10x ATAC fragments
    peaks.bed           — 10x ATAC peaks

The h5 carries both RNA (Gene Expression feature_type) and ATAC (Peaks
feature_type). We split, log-normalise the RNA, and feed everything
through pipeline.run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Data lives under validation/real_multiome/ (gitignored — file sizes > 100MB).
# Run the curl downloads below first; the bench reads from there.
HERE = Path(__file__).parent.parent / "real_multiome"
HERE.mkdir(exist_ok=True)


def load_10x_multiome_h5(h5_path: Path):
    """Split a 10x multiome h5 into (rna, atac) AnnDatas."""
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = m["shape"][:]
        barcodes = m["barcodes"][:].astype(str)
        feat = m["features"]
        feat_id = feat["id"][:].astype(str)
        feat_name = feat["name"][:].astype(str)
        feat_type = feat["feature_type"][:].astype(str)

    full = sp.csc_matrix((data, indices, indptr), shape=tuple(shape)).T  # cells × features
    is_rna = feat_type == "Gene Expression"

    rna = ad.AnnData(
        X=full[:, is_rna].tocsr().astype(np.float32),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(
            {"feature_name": feat_name[is_rna]},
            index=feat_id[is_rna],
        ),
    )
    return rna


def main() -> int:
    rna_path = HERE / "rna.h5"
    frag_path = HERE / "fragments.tsv.gz"
    peaks_path = HERE / "peaks.bed"

    if not all(p.exists() for p in (rna_path, frag_path, peaks_path)):
        print("missing inputs — run the curl downloads first", file=sys.stderr)
        return 1

    print(f"loading 10x multiome from {rna_path.name}...")
    t0 = time.monotonic()
    rna = load_10x_multiome_h5(rna_path)
    print(f"  RNA shape: {rna.shape} (loaded in {time.monotonic()-t0:.1f}s)")

    # Light QC + log normalize so warn_if_likely_unnormalized doesn't trip
    print("normalising RNA (target_sum=1e4, log1p)...")
    counts = rna.X.sum(axis=1)
    counts = np.asarray(counts).flatten()
    counts[counts == 0] = 1.0
    rna.X = rna.X.multiply(1e4 / counts[:, None]).tocsr()
    rna.X.data = np.log1p(rna.X.data)
    rna.X = rna.X.astype(np.float32)
    print(f"  X dtype: {rna.X.dtype}, max value: {rna.X.max():.2f}")

    # Use the bundled human TF list — full pyscenic-equivalent regulator set.
    import rustscenic.data
    tfs = rustscenic.data.tfs("human")
    tfs_present = [t for t in tfs if t in rna.var["feature_name"].values]
    print(f"  TFs in data: {len(tfs_present)} of {len(tfs)} bundled")

    # Run pipeline.run RNA-only path with full TF list (no ATAC for now —
    # baseline GRN+AUCell is the dominant cost we want to attack first).
    # Persist output dir under HERE so stage artifacts survive the run.
    import warnings
    import rustscenic.pipeline
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.monotonic()
        result = rustscenic.pipeline.run(
            rna,
            out_dir,
            tfs=tfs_present,
            grn_n_estimators=100,
            grn_top_targets=50,
            aucell_top_frac=0.05,
            verbose=True,
        )
        total = time.monotonic() - t0

    print()
    print("=== STAGE WALL-CLOCK ===")
    for k, v in result.elapsed.items():
        print(f"  {k:12s} {v:7.1f}s")
    print(f"  {'TOTAL':12s} {total:7.1f}s")
    print()
    print(f"GRN edges: {pd.read_parquet(result.grn_path).shape[0]:,}")
    print(f"regulons: {result.n_regulons}")

    # Save the timing record for cross-run comparison.
    import json
    record = {
        "n_cells": rna.n_obs,
        "n_genes": rna.n_vars,
        "n_tfs": len(tfs_present),
        "elapsed": result.elapsed,
        "total": total,
    }
    (HERE / "bench_record.json").write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {HERE / 'bench_record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
