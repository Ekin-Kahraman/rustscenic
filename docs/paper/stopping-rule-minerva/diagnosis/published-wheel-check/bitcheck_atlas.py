"""Atlas 20,000 bit-identity check of the PUBLISHED rustscenic 0.7.0 wheel (as verify_default.py, atlas rows).
Writes only into this script's directory."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
M = Path('<local working copy>/RustScenic-Stopping-Rule-Minerva-20261113')  # path redacted for publication
D = M / 'diagnosis'
sys.path.append(str(M / 'rustscenic-stopping-rule'))  # appended, so site-packages rustscenic wins
from data_io import load_matrix  # noqa: E402
import rustscenic  # noqa: E402
from rustscenic import grn  # noqa: E402

SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)
x, genes, _ = load_matrix(D / 'atlas-rebuilt/rna.h5', 20000)
tfs = (D / 'atlas-rebuilt/tfs.txt').read_text().split()
x = x.tocsc()
mean = np.asarray(x.mean(axis=0)).ravel()
varying = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2 > 0
genes = [g for g, keep in zip(genes, varying) if keep]
x = x[:, np.flatnonzero(varying)].tocsr()
tfs = [t for t in tfs if t in set(genes)]

rows = []
for max_bins, ref in [(None, 'atlas20000-cap65535-arboreto-777'), (255, 'atlas20000-cap255-arboreto-777')]:
    kw = {} if max_bins is None else {'max_bins': max_bins}
    t = time.perf_counter()
    net = grn.infer((x, genes), tf_names=tfs, seed=777, verbose=False, early_stop_mode='arboreto', **SETTINGS, **kw)
    secs = time.perf_counter() - t
    a = net[['TF', 'target', 'importance']].assign(importance=lambda d: d.importance.astype(np.float32))
    a = a.astype({'TF': str, 'target': str}).reset_index(drop=True)
    b = pd.read_parquet(D / 'native-bins' / ref / 'network.parquet', columns=['TF', 'target', 'importance'])
    b = b.assign(importance=b.importance.astype(np.float32)).astype({'TF': str, 'target': str}).reset_index(drop=True)
    m = a.merge(b, on=['TF', 'target'], how='outer', indicator=True)
    rows.append(dict(rustscenic=rustscenic.__version__, max_bins=net.attrs['grn_fit']['max_bins'], reference=ref,
                     edges=len(a), reference_edges=len(b), equals_ordered=bool(a.equals(b)),
                     unmatched=int((m._merge != 'both').sum()),
                     max_abs_diff=float((m.importance_x - m.importance_y).abs().max()), inference_s=secs))
    print(rows[-1], flush=True)
(HERE / 'result-atlas20000.json').write_text(json.dumps(rows, indent=2) + '\n')
