"""End-to-end pipeline.run with `topics_method='gibbs'` on real PBMC.

Mirrors `bench_full_pipeline_pbmc_multiome.py` (which uses Online VB for
topics) but exercises the new collapsed-Gibbs path. Validates that the
parallel AD-LDA sampler integrates cleanly with the rest of the
orchestrator on real ATAC + RNA data.

Inputs: same as the VB E2E bench (validation/real_multiome/, gitignored).

Run:
  python validation/scaling/bench_full_pipeline_gibbs.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent.parent / "real_multiome"


def load_10x_h5(path: Path) -> ad.AnnData:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = m["shape"][:]
        bcs = m["barcodes"][:].astype(str)
        feat = m["features"]
        is_rna = feat["feature_type"][:].astype(str) == "Gene Expression"
        feat_id = feat["id"][:].astype(str)
        feat_name = feat["name"][:].astype(str)
    full = sp.csc_matrix((data, indices, indptr), shape=tuple(shape)).T
    return ad.AnnData(
        X=full[:, is_rna].tocsr().astype(np.float32),
        obs=pd.DataFrame(index=bcs),
        var=pd.DataFrame({"feature_name": feat_name[is_rna]}, index=feat_id[is_rna]),
    )


def main() -> int:
    rna_path = HERE / "rna.h5"
    frag_path = HERE / "fragments.tsv.gz"
    peaks_path = HERE / "peaks.bed"
    if not all(p.exists() for p in (rna_path, frag_path, peaks_path)):
        sys.exit("missing real_multiome/ inputs — see docstring")

    print("loading RNA + normalising...", flush=True)
    rna = load_10x_h5(rna_path)
    counts = np.asarray(rna.X.sum(axis=1)).flatten()
    counts[counts == 0] = 1.0
    rna.X = rna.X.multiply(1e4 / counts[:, None]).tocsr()
    rna.X.data = np.log1p(rna.X.data)
    rna.X = rna.X.astype(np.float32)
    print(f"  RNA: {rna.shape}", flush=True)

    rng = np.random.default_rng(0)
    gene_coords = pd.DataFrame({
        "gene": rna.var["feature_name"].astype(str).values,
        "chrom": ["chr1"] * rna.n_vars,
        "tss": rng.integers(0, 250_000_000, size=rna.n_vars),
    })

    import rustscenic.data
    all_tfs = rustscenic.data.tfs("human")
    present = set(rna.var["feature_name"].astype(str))
    tfs = [t for t in all_tfs if t in present][:30]
    print(f"  {len(tfs)} TFs intersected", flush=True)

    out_dir = HERE / "out_full_e2e_gibbs"
    out_dir.mkdir(exist_ok=True)

    import rustscenic.pipeline
    print("\n=== running pipeline.run with topics_method='gibbs' ===", flush=True)
    t_total = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rustscenic.pipeline.run(
            rna, out_dir,
            fragments=str(frag_path), peaks=str(peaks_path),
            tfs=tfs, gene_coords=gene_coords,
            grn_n_estimators=30, grn_top_targets=30,
            topics_n_topics=10,
            topics_method="gibbs", topics_n_iters=100, topics_n_threads=4,
            verbose=True,
        )
    total = time.monotonic() - t_total

    print("\n=== STAGE WALL-CLOCK ===", flush=True)
    for k, v in result.elapsed.items():
        print(f"  {k:12s} {v:7.1f}s", flush=True)
    print(f"  {'TOTAL':12s} {total:7.1f}s", flush=True)
    print(f"  GRN edges: {pd.read_parquet(result.grn_path).shape[0]:,}", flush=True)
    print(f"  AUCell shape: {pd.read_parquet(result.aucell_path).shape}", flush=True)

    record = {
        "n_cells": rna.n_obs,
        "n_genes": rna.n_vars,
        "n_tfs": len(tfs),
        "topics_method": "gibbs",
        "topics_n_threads": 4,
        "elapsed": result.elapsed,
        "total": total,
    }
    out_file = Path(__file__).parent / "real_multiome_gibbs_e2e.json"
    out_file.write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
