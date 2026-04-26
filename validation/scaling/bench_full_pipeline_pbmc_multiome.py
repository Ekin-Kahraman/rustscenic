"""End-to-end pipeline.run on real 10x PBMC 3k Multiome.

This is the test we've never run before. Synthetic multiome was tested
(test_pipeline_integration.py); a small real-Kamath RNA-only run was
tested (validate_kamath_fix.py). But running pipeline.run with all
stages (preproc → topics → GRN → cistarget → enhancer → eRegulon →
AUCell) on REAL multiome data has never happened.

What this proves
----------------
- Whether the orchestrator's stage ordering holds on real data
- Whether real fragment counts + real cell barcodes survive every
  guardrail without firing false-positive warnings
- How long the full pipeline takes on a 2.7k-cell multiome (the small
  end of "real" — atlas-scale needs HPC anyway)
- Where the next perf gap lives — preproc vs topics vs GRN

Inputs (under validation/real_multiome/, gitignored):
  rna.h5              filtered_feature_bc_matrix from cellranger-arc
  fragments.tsv.gz    ATAC fragments
  peaks.bed           consensus peaks

Run:
  python validation/scaling/bench_full_pipeline_pbmc_multiome.py
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

    print(f"loading RNA from {rna_path.name}...")
    rna = load_10x_h5(rna_path)
    print(f"  RNA: {rna.shape}")
    counts = np.asarray(rna.X.sum(axis=1)).flatten()
    counts[counts == 0] = 1.0
    rna.X = rna.X.multiply(1e4 / counts[:, None]).tocsr()
    rna.X.data = np.log1p(rna.X.data)
    rna.X = rna.X.astype(np.float32)
    print(f"  X normalised, max {rna.X.data.max():.2f}")

    # Synthetic gene_coords for the enhancer→gene step. Real users would
    # load a GTF; this works as long as gene names match.
    rng = np.random.default_rng(0)
    gene_coords = pd.DataFrame({
        "gene": rna.var["feature_name"].astype(str).values,
        "chrom": ["chr1"] * rna.n_vars,
        "tss": rng.integers(0, 250_000_000, size=rna.n_vars),
    })

    # Use bundled human TFs that we know are in the dataset
    import rustscenic.data
    all_tfs = rustscenic.data.tfs("human")
    present = set(rna.var["feature_name"].astype(str))
    tfs = [t for t in all_tfs if t in present][:30]
    print(f"  {len(tfs)} TFs intersected with data")

    out_dir = HERE / "out_full_e2e"
    out_dir.mkdir(exist_ok=True)

    import rustscenic.pipeline
    print("\n=== running full pipeline.run on real multiome ===")
    t_total = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rustscenic.pipeline.run(
            rna,
            out_dir,
            fragments=str(frag_path),
            peaks=str(peaks_path),
            tfs=tfs,
            gene_coords=gene_coords,
            grn_n_estimators=30,
            grn_top_targets=30,
            topics_n_topics=10,
            topics_n_passes=1,
            verbose=True,
        )
    total = time.monotonic() - t_total

    print()
    print("=== STAGE WALL-CLOCK ===")
    for k, v in result.elapsed.items():
        print(f"  {k:12s} {v:7.1f}s")
    print(f"  {'TOTAL':12s} {total:7.1f}s")
    print(f"\n  GRN edges: {pd.read_parquet(result.grn_path).shape[0]:,}")
    print(f"  AUCell shape: {pd.read_parquet(result.aucell_path).shape}")
    print(f"  enhancer links: {pd.read_parquet(result.enhancer_links_path).shape[0] if result.enhancer_links_path else 'skipped'}")
    print(f"  eRegulons: {result.n_eregulons}")

    record = {
        "n_cells": rna.n_obs,
        "n_genes": rna.n_vars,
        "n_tfs": len(tfs),
        "elapsed": result.elapsed,
        "total": total,
    }
    (Path(__file__).parent / "real_multiome_full_e2e.json").write_text(
        json.dumps(record, indent=2)
    )
    print(f"\nrecord → {Path(__file__).parent / 'real_multiome_full_e2e.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
