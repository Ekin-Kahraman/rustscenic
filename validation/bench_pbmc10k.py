"""Benchmark rustscenic vs arboreto on PBMC-10k."""
import json
import os
import resource
import sys
import time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd
import anndata as ad

ADATA_PATH = Path("/Users/ekin/rustscenic/validation/reference/data/pbmc10k.h5ad")
TFS_FILE = Path("/Users/ekin/rustscenic/validation/reference/data/allTFs_hg38.txt")

TOOL = sys.argv[1]  # "rustscenic" or "arboreto"
OUT_DIR = Path("/Users/ekin/rustscenic/validation/ours")
OUT_DIR.mkdir(exist_ok=True, parents=True)

adata = ad.read_h5ad(ADATA_PATH)
tfs = [t for t in TFS_FILE.read_text().strip().splitlines() if t in set(adata.var_names)]
print(f"PBMC-10k: cells={adata.n_obs} genes={adata.n_vars} tfs={len(tfs)}")

rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

if TOOL == "rustscenic":
    import rustscenic
    import rustscenic.grn
    t0 = time.monotonic()
    df = rustscenic.grn.infer(adata, tfs, seed=777, n_estimators=300, early_stop_window=25)
    wall = time.monotonic() - t0
    out_pq = OUT_DIR / "pbmc10k_grn_ours.parquet"
    df.to_parquet(out_pq, index=False)
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"rustscenic PBMC-10k: wall={wall:.1f}s  edges={len(df)}  rss_delta_mb={(rss_peak-rss_start)/1e6:.0f}")
elif TOOL == "arboreto":
    from arboreto.core import SGBM_KWARGS, to_tf_matrix, infer_partial_network
    gene_names = list(adata.var_names)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(X, gene_names, tfs)

    global _X_, _TF_, _TFN_, _GN_
    _X_ = X; _TF_ = tf_matrix; _TFN_ = tf_matrix_gene_names; _GN_ = gene_names

    def init_w(X, tf_matrix, tf_names, gn):
        global _X, _TF, _TFN, _GN
        _X = X; _TF = tf_matrix; _TFN = tf_names; _GN = gn

    def one(tgt_idx):
        tgt = _GN[tgt_idx]
        return infer_partial_network(
            regressor_type="GBM", regressor_kwargs=SGBM_KWARGS,
            tf_matrix=_TF, tf_matrix_gene_names=_TFN,
            target_gene_name=tgt, target_gene_expression=_X[:, tgt_idx],
            include_meta=False, early_stop_window_length=25, seed=777,
        )

    t0 = time.monotonic()
    ctx = get_context("fork")
    with ctx.Pool(8, initializer=init_w, initargs=(X, tf_matrix, tf_matrix_gene_names, gene_names)) as pool:
        results = []
        for i, df in enumerate(pool.imap_unordered(one, range(len(gene_names)), chunksize=16)):
            results.append(df)
            if (i + 1) % 2000 == 0:
                el = time.monotonic() - t0
                print(f"  {i+1}/{len(gene_names)}  ({el:.0f}s, {(i+1)/el:.1f}/s)", flush=True)
    wall = time.monotonic() - t0
    adj = pd.concat(results, ignore_index=True)
    out_pq = OUT_DIR / "pbmc10k_grn_arboreto.parquet"
    adj.to_parquet(out_pq, index=False)
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"arboreto PBMC-10k: wall={wall:.1f}s  edges={len(adj)}  rss_delta_mb={(rss_peak-rss_start)/1e6:.0f}")

meta = {
    "tool": TOOL,
    "dataset": "pbmc10k",
    "n_cells": int(adata.n_obs),
    "n_genes": int(adata.n_vars),
    "n_tfs": len(tfs),
    "wall_clock_s": wall,
    "n_edges": int(out_pq.stat().st_size),  # just file size as proxy
}
(OUT_DIR / f"pbmc10k_{TOOL}_meta.json").write_text(json.dumps(meta, indent=2))
