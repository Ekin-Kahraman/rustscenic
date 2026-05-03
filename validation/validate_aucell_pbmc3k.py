"""Validate rustscenic.aucell.score against pyscenic.aucell on PBMC-3k."""
import time
import numpy as np
import pandas as pd
import anndata as ad

# pyscenic reference
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell as pyscenic_aucell
from ctxcore.genesig import Regulon

import rustscenic
import rustscenic.aucell
import rustscenic.grn

ADATA = ad.read_h5ad("/Users/ekin/projects/bio/rustscenic/validation/reference/data/pbmc3k.h5ad")
our_grn = pd.read_parquet("/Users/ekin/projects/bio/rustscenic/validation/ours/pbmc3k_grn.parquet").rename(columns={"tf": "TF"})
ex_mtx = ADATA.to_df()

# Build regulons once
print("building regulons via pyscenic.modules_from_adjacencies...")
t0 = time.monotonic()
modules = list(modules_from_adjacencies(
    our_grn, ex_mtx, thresholds=(), top_n_targets=(50,),
    top_n_regulators=(), min_genes=10, rho_mask_dropouts=False,
))
seen = set()
regs_for_pyscenic = []
regs_for_rustscenic = []
for m in modules:
    tf = m.transcription_factor
    if tf in seen:
        continue
    seen.add(tf)
    name = f"{tf}_regulon"
    regs_for_pyscenic.append(Regulon(
        name=name, gene2weight=m.gene2weight, transcription_factor=tf, gene2occurrence={},
    ))
    regs_for_rustscenic.append((name, list(m.gene2weight.keys())))
print(f"  {len(regs_for_pyscenic)} regulons  ({time.monotonic()-t0:.1f}s)")

# --- pyscenic.aucell ---
print("\nrunning pyscenic.aucell (single worker, to avoid mp quirks)...")
t0 = time.monotonic()
py_auc = pyscenic_aucell(ex_mtx, regs_for_pyscenic, num_workers=1)
py_wall = time.monotonic() - t0
print(f"  pyscenic: {py_wall:.1f}s  shape={py_auc.shape}")

# --- rustscenic.aucell ---
print("\nrunning rustscenic.aucell.score...")
t0 = time.monotonic()
our_auc = rustscenic.aucell.score(ADATA, regs_for_rustscenic, top_frac=0.05)
our_wall = time.monotonic() - t0
print(f"  rustscenic: {our_wall:.1f}s  shape={our_auc.shape}")
print(f"  speedup: {py_wall/our_wall:.1f}x")

# --- correctness: correlate per-regulon activity vectors ---
# pyscenic column names "SPI1_regulon" match ours
common = sorted(set(py_auc.columns) & set(our_auc.columns))
print(f"\ncommon regulons: {len(common)}")
rhos = []
for r in common:
    py_vec = py_auc[r].values
    our_vec = our_auc[r].values
    if np.std(py_vec) < 1e-12 or np.std(our_vec) < 1e-12:
        continue
    rho = float(np.corrcoef(py_vec, our_vec)[0, 1])
    rhos.append(rho)

import numpy as np
rhos_arr = np.asarray(rhos)
print(f"per-regulon Pearson correlation (ours vs pyscenic):")
print(f"  mean = {np.mean(rhos_arr):.4f}")
print(f"  median = {np.median(rhos_arr):.4f}")
print(f"  quantiles = [min {np.min(rhos_arr):.4f}, 10% {np.quantile(rhos_arr, 0.1):.4f}, 90% {np.quantile(rhos_arr, 0.9):.4f}, max {np.max(rhos_arr):.4f}]")
print(f"  >0.99: {np.sum(rhos_arr > 0.99)}/{len(rhos_arr)}  >0.95: {np.sum(rhos_arr > 0.95)}/{len(rhos_arr)}")
