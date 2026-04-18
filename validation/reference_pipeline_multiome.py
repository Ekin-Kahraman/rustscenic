"""Reference pipeline: arboreto.grnboost2 → pyscenic.aucell → tomotopy LDA
on same 10x Multiome PBMC data we already ran rustscenic on.

This is the head-to-head integration test:
  Ours:    rustscenic grn → rustscenic aucell → rustscenic topics  (548s)
  Ref:     arboreto grnboost2 → pyscenic aucell → tomotopy LDA     (this script)
Compare total wall-clock AND output equivalence (top regulons per cluster).
"""
import time
import json
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# Reference (sync path — arboreto Dask is broken on modern Python)
from arboreto.core import SGBM_KWARGS, to_tf_matrix, infer_partial_network
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell as py_aucell
from ctxcore.genesig import Regulon
import tomotopy as tp

RNA = Path("/Users/ekin/rustscenic/validation/reference/data/multiome3k/rna.h5ad")
ATAC = Path("/Users/ekin/rustscenic/validation/reference/data/multiome3k/atac_binarized.h5ad")
TFS_FILE = Path("/Users/ekin/rustscenic/validation/reference/data/allTFs_hg38.txt")
OUT = Path("/Users/ekin/rustscenic/validation/ours/reference_pipeline_multiome.json")

rna = ad.read_h5ad(RNA)
atac = ad.read_h5ad(ATAC)
shared = sorted(set(rna.obs_names) & set(atac.obs_names))
rna = rna[shared].copy()
atac = atac[shared].copy()
print(f"RNA {rna.shape}   ATAC {atac.shape}   shared cells {len(shared)}")

tfs = [t for t in TFS_FILE.read_text().strip().splitlines() if t in set(rna.var_names)]
print(f"TFs: {len(tfs)}")

# ---------------- Stage 1: arboreto.grnboost2 via sync per-target fork --------
print("\n=== Stage 1: arboreto.grnboost2 (sync, 8 workers) ===")
gene_names = list(rna.var_names)
X = rna.X.toarray() if hasattr(rna.X, "toarray") else np.asarray(rna.X)
X = np.ascontiguousarray(X, dtype=np.float32)
tf_matrix, tf_matrix_gene_names = to_tf_matrix(X, gene_names, tfs)

_X = _TF = _TFN = _GN = None
def init_w(Xg, tfm, tfnames, gn):
    global _X, _TF, _TFN, _GN
    _X = Xg; _TF = tfm; _TFN = tfnames; _GN = gn
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
    n = len(gene_names)
    for i, df in enumerate(pool.imap_unordered(one, range(n), chunksize=16)):
        results.append(df)
        if (i+1) % 2000 == 0:
            el = time.monotonic() - t0
            print(f"  {i+1}/{n}  ({el:.0f}s, {(i+1)/el:.1f}/s)", flush=True)
t_grn = time.monotonic() - t0
adj = pd.concat(results, ignore_index=True)
print(f"  done: {t_grn:.1f}s   edges {len(adj)}")

# ---------------- Stage 2: pyscenic.aucell ---------------------------------
print("\n=== Stage 2: pyscenic.aucell (num_workers=1 to avoid mp issue) ===")
ex_mtx = rna.to_df()
t0 = time.monotonic()
modules = list(modules_from_adjacencies(adj, ex_mtx, thresholds=(), top_n_targets=(50,),
                                        top_n_regulators=(), min_genes=10, rho_mask_dropouts=False))
seen = set(); regs_py = []
for m in modules:
    if m.transcription_factor in seen:
        continue
    seen.add(m.transcription_factor)
    regs_py.append(Regulon(name=f"{m.transcription_factor}_regulon",
                           gene2weight=m.gene2weight,
                           transcription_factor=m.transcription_factor,
                           gene2occurrence={}))
print(f"  {len(regs_py)} regulons")
py_auc = py_aucell(ex_mtx, regs_py, num_workers=1)
t_aucell = time.monotonic() - t0
print(f"  done: {t_aucell:.1f}s   shape {py_auc.shape}")

# ---------------- Stage 3: tomotopy LDA on ATAC ---------------------------
print("\n=== Stage 3: tomotopy LDA (K=10, 500 Gibbs iters, 8 workers) ===")
K = 10
peak_names = list(atac.var_names)
t0 = time.monotonic()
mdl = tp.LDAModel(k=K, alpha=1.0/K, eta=1.0/K, seed=777)
Xa = atac.X.tocsr() if sp.issparse(atac.X) else sp.csr_matrix(atac.X)
for c in range(Xa.shape[0]):
    row = Xa.getrow(c); nz = row.indices
    if len(nz) == 0: continue
    mdl.add_doc(words=[peak_names[p] for p in nz])
mdl.train(500, workers=8)
tomo_wall = time.monotonic() - t0
tomo_assign = np.zeros(Xa.shape[0], dtype=int)
for i, doc in enumerate(mdl.docs):
    tomo_assign[i] = int(np.argmax(doc.get_topic_dist()))
print(f"  done: {tomo_wall:.1f}s   {len(set(tomo_assign))} unique top topics")

# ---------------- Summary ---------------------------
total = t_grn + t_aucell + tomo_wall
summary = {
    "pipeline": "reference (arboreto + pyscenic.aucell + tomotopy)",
    "dataset": "10x Multiome PBMC 3k (cell-arc/2.0.0)",
    "n_cells": int(len(shared)),
    "n_rna_genes": int(rna.n_vars),
    "n_atac_peaks": int(atac.n_vars),
    "stage_1_grn_s": round(t_grn, 1),
    "stage_2_aucell_s": round(t_aucell, 1),
    "stage_3_topics_s": round(tomo_wall, 1),
    "total_s": round(total, 1),
    "total_min": round(total/60, 1),
    "n_regulons": len(regs_py),
    "n_topics": K,
    "n_adjacencies": int(len(adj)),
}
print(f"\n=== TOTAL REFERENCE PIPELINE: {total:.1f}s ({total/60:.1f} min) ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

OUT.write_text(json.dumps(summary, indent=2))
# also save the adj + auc for comparison
adj.to_parquet(OUT.with_suffix(".adj.parquet"), index=False)
py_auc.to_parquet(OUT.with_suffix(".auc.parquet"))
np.save(str(OUT.with_suffix(".tomo_assign.npy")), tomo_assign)
print(f"wrote {OUT}")
