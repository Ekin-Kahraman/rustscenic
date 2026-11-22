"""Bit-identity check of the PUBLISHED rustscenic 0.7.0 wheel under the original rule (early_stop_mode="paired_oob",
max_bins=255) on the bone-marrow input, against the returned HPC network bmmc-777/native_same. Added 22 November 2026;
otherwise identical to bitcheck.py.

Input preparation and settings copied from diagnosis/fit_native_bins.py (bmmc branch).
Writes only into this script's directory.
"""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
M = Path('<local working copy>/RustScenic-Stopping-Rule-Minerva-20261113')  # path redacted for publication
PKG = M / 'rustscenic-stopping-rule'
SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)
REFS = {'paired255': M / 'return-20261118/audit/extracted/jobs/bmmc-777/native_same/network.parquet'}
which = 'paired255'

panel = json.loads((PKG / 'inputs/bmmc/panel.json').read_text())
x = sparse.load_npz(PKG / 'inputs/bmmc/training.npz').tocsr()
genes, tfs = panel['genes'], panel['tfs']
cells = pd.read_csv(PKG / 'inputs/bmmc/training-cells.tsv', sep='\t', dtype=str).barcode.tolist()
x = x.tocsc()
mean = np.asarray(x.mean(axis=0)).ravel()
varying = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2 > 0
genes = [g for g, keep in zip(genes, varying) if keep]
x = x[:, np.flatnonzero(varying)].tocsr()
tfs = [t for t in tfs if t in set(genes)]
sha = lambda v: hashlib.sha256('\n'.join(v).encode()).hexdigest()

import rustscenic
from rustscenic import grn

kw = {'max_bins': 255}
t = time.perf_counter()
net = grn.infer((x, genes), tf_names=tfs, seed=777, verbose=False, early_stop_mode='paired_oob', **SETTINGS, **kw)
secs = time.perf_counter() - t
fit = dict(net.attrs.get('grn_fit', {}))

# Same column selection and float32 cast as fit_native_bins.py, then the categorical round trip it writes.
a = net[['TF', 'target', 'importance']].assign(importance=lambda d: d.importance.astype(np.float32))
a.astype({'TF': 'category', 'target': 'category'}).to_parquet(HERE / f'net-{which}.parquet', index=False)
a = a.astype({'TF': str, 'target': str}).reset_index(drop=True)
b = pd.read_parquet(REFS[which], columns=['TF', 'target', 'importance'])
ref_dtypes = {c: str(t) for c, t in b.dtypes.items()}
b = b.assign(importance=b.importance.astype(np.float32)).astype({'TF': str, 'target': str}).reset_index(drop=True)

equals_ordered = a.equals(b)
key = ['target', 'TF']
a_s = a.sort_values(key).reset_index(drop=True)
b_s = b.sort_values(key).reset_index(drop=True)
equals_sorted = a_s.equals(b_s)
m = a.merge(b, on=['TF', 'target'], how='outer', indicator=True)
diff = float((m.importance_x - m.importance_y).abs().max())
bits_a = a_s.importance.to_numpy().view(np.uint32)
bits_b = b_s.importance.to_numpy().view(np.uint32) if len(a_s) == len(b_s) else None
row = dict(which=which, rustscenic=rustscenic.__version__, module=rustscenic.__file__, grn_fit=fit,
           cells_sha256=sha(cells), genes_sha256=sha(genes), n_genes=len(genes), n_tfs=len(tfs), n_cells=x.shape[0],
           inference_s=secs, maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           trees_mean=float(net.groupby('target', observed=True).importance.sum().round().astype(int).mean()),
           edges=len(a), reference=str(REFS[which]), reference_edges=len(b), reference_dtypes=ref_dtypes,
           equals_ordered=bool(equals_ordered), equals_sorted=bool(equals_sorted),
           unmatched=int((m._merge != 'both').sum()), max_abs_diff=diff,
           float32_bits_identical=bool(bits_b is not None and np.array_equal(bits_a, bits_b)))
(HERE / f'result-{which}.json').write_text(json.dumps(row, indent=2, default=str) + '\n')
print(json.dumps(row, default=str), flush=True)
