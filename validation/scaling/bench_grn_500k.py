"""GRN scaling at 500k synthetic cells — extends the v0.3.1 200k story."""
from __future__ import annotations
import json, resource, sys, time
from pathlib import Path
import numpy as np
import anndata as ad
import pandas as pd
import warnings

import rustscenic.grn

n_cells = 500_000
n_genes = 5_000  # smaller n_genes to keep dense RNA tractable: 500k × 5k = 10 GB
n_tfs = 50

print(f"Synthesising {n_cells:,} × {n_genes:,} RNA + {n_tfs} TFs...", flush=True)
t0 = time.monotonic()
rng = np.random.default_rng(42)
# Plant 30 cell-state programmes; first 50 genes = candidate TFs
n_programmes = 30
cluster = rng.integers(0, n_programmes, size=n_cells, dtype=np.int32)

# Allocate once with noise, overlay programme signal
rna = (0.1 * rng.normal(size=(n_cells, n_genes))).astype(np.float32)
for c in range(n_cells):
    prog = int(cluster[c])
    prog_genes = list(range(prog * (n_genes // n_programmes), (prog + 1) * (n_genes // n_programmes)))
    rna[c, prog_genes[:30]] += (1.5 + 0.3 * rng.normal(size=30).astype(np.float32))
np.clip(rna, 0.0, None, out=rna)
rna += 0.05

cells = [f"c{i}" for i in range(n_cells)]
genes = [f"G{i:05d}" for i in range(n_genes)]
adata = ad.AnnData(X=rna, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))
tfs = genes[:n_tfs]  # use first 50 genes as candidate TFs
build_t = time.monotonic() - t0
print(f"  built in {build_t:.0f}s", flush=True)

print("\nfitting GRN (n_estimators=20)...", flush=True)
t0 = time.monotonic()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grn = rustscenic.grn.infer(adata, tf_names=tfs, n_estimators=20, seed=42, verbose=False)
wall = time.monotonic() - t0
rss_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_gb = rss_b / (1024**3) if sys.platform == "darwin" else rss_b / (1024**2)

result = {
    "n_cells": n_cells, "n_genes": n_genes, "n_tfs": n_tfs,
    "n_estimators": 20,
    "build_s": round(build_t, 1),
    "grn_s": round(wall, 1),
    "grn_min": round(wall/60, 1),
    "n_edges": int(len(grn)),
    "peak_rss_gb": round(rss_gb, 2),
}
print(json.dumps(result, indent=2), flush=True)
Path("/Users/ekin/rustscenic/validation/scaling/grn_500k.json").write_text(json.dumps(result, indent=2))
