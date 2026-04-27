"""Real 10x PBMC 3k Multiome — full 8-stage E2E with motif cistarget.

The existing `bench_full_pipeline_pbmc_multiome.py` runs preproc → topics
→ GRN → enhancer → AUCell on real data. It deliberately skips the
cistarget step because no motif rankings were provided. This bench
provides them (cached aertslab hg38 v10 feather) so the cistarget +
eRegulon path actually executes on real PBMC data — the most complete
real-data validation rustscenic has shipped.

Run:
  python validation/scaling/bench_real_pbmc_full_e2e.py
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
RANKINGS = (
    Path(__file__).parent.parent.parent
    / "data"
    / "cistarget"
    / "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
)


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
    if not RANKINGS.exists():
        sys.exit(f"missing motif rankings: {RANKINGS}")
    rna_path = HERE / "rna.h5"
    frag_path = HERE / "fragments.tsv.gz"
    peaks_path = HERE / "peaks.bed"
    if not all(p.exists() for p in (rna_path, frag_path, peaks_path)):
        sys.exit("missing real_multiome/ inputs")

    print("Loading RNA + normalising...", flush=True)
    rna = load_10x_h5(rna_path)
    counts = np.asarray(rna.X.sum(axis=1)).flatten()
    counts[counts == 0] = 1.0
    rna.X = rna.X.multiply(1e4 / counts[:, None]).tocsr()
    rna.X.data = np.log1p(rna.X.data)
    rna.X = rna.X.astype(np.float32)
    print(f"  RNA: {rna.shape}", flush=True)

    print("Loading aertslab hg38 motif rankings...", flush=True)
    t0 = time.monotonic()
    motif_rankings = pd.read_feather(RANKINGS)
    # aertslab v10 feather: motifs are rows under a `motifs` column at the
    # end; genes are columns. Set the motif column as the index so cistarget
    # gets a (n_motifs × n_genes) numeric DataFrame indexed by motif name.
    if "motifs" in motif_rankings.columns:
        motif_rankings = motif_rankings.set_index("motifs")
    else:
        motif_rankings = motif_rankings.set_index(motif_rankings.columns[0])
    print(f"  rankings: {motif_rankings.shape} (motifs × genes) in "
          f"{time.monotonic()-t0:.1f}s", flush=True)

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
    print(f"  {len(tfs)} TFs", flush=True)

    out_dir = HERE / "out_full_e2e_with_cistarget"
    out_dir.mkdir(exist_ok=True)

    import rustscenic.pipeline
    print("\n=== running pipeline.run with motif_rankings + Gibbs topics ===",
          flush=True)
    t_total = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rustscenic.pipeline.run(
            rna, out_dir,
            fragments=str(frag_path), peaks=str(peaks_path),
            tfs=tfs, gene_coords=gene_coords,
            motif_rankings=motif_rankings,
            grn_n_estimators=30, grn_top_targets=30,
            topics_n_topics=10,
            topics_method="gibbs", topics_n_iters=100, topics_n_threads=4,
            cistarget_top_frac=0.05, cistarget_auc_threshold=0.05,
            verbose=True,
        )
    total = time.monotonic() - t_total

    print("\n=== STAGE WALL-CLOCK ===", flush=True)
    for k, v in result.elapsed.items():
        print(f"  {k:12s} {v:7.1f}s", flush=True)
    print(f"  {'TOTAL':12s} {total:7.1f}s", flush=True)
    print(f"  GRN edges:        {pd.read_parquet(result.grn_path).shape[0]:,}",
          flush=True)
    print(f"  AUCell shape:     {pd.read_parquet(result.aucell_path).shape}",
          flush=True)
    if result.cistarget_path:
        ct = pd.read_parquet(result.cistarget_path)
        print(f"  Cistarget hits:   {len(ct):,}", flush=True)
    if result.enhancer_links_path:
        links = pd.read_parquet(result.enhancer_links_path)
        print(f"  Enhancer links:   {len(links):,}", flush=True)
    print(f"  eRegulons:        {result.n_eregulons}", flush=True)

    record = {
        "n_cells": rna.n_obs,
        "n_genes": rna.n_vars,
        "n_tfs": len(tfs),
        "topics_method": "gibbs",
        "topics_n_threads": 4,
        "elapsed": result.elapsed,
        "total": total,
        "n_eregulons": result.n_eregulons,
    }
    out_file = Path(__file__).parent / "real_pbmc_full_e2e.json"
    out_file.write_text(json.dumps(record, indent=2))
    print(f"\nrecord → {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
