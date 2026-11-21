"""Post hoc: refit the bone-marrow RustScenic arm exactly as the package's fit.py does, with the
finer-bins prototype (RUSTSCENIC_MAX_BINS sets the cut-point budget; 255 is the released behaviour)."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

PKG, OUT, SEED, MODE = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
DATA = sys.argv[5] if len(sys.argv) > 5 else 'bmmc'  # or atlasN, read from PREPARED (rebuild_atlas.py output)
PREPARED = Path(sys.argv[6]) if len(sys.argv) > 6 else None
SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)

if DATA == 'bmmc':
    panel = json.loads((PKG / 'inputs/bmmc/panel.json').read_text())
    x = sparse.load_npz(PKG / 'inputs/bmmc/training.npz').tocsr()
    genes, tfs = panel['genes'], panel['tfs']
    cells = pd.read_csv(PKG / 'inputs/bmmc/training-cells.tsv', sep='\t', dtype=str).barcode.tolist()
else:
    sys.path.insert(0, str(PKG))
    from data_io import load_matrix
    x, genes, cells = load_matrix(PREPARED / 'rna.h5', int(DATA.removeprefix('atlas')))
    tfs = (PREPARED / 'tfs.txt').read_text().split()
x = x.tocsc()
mean = np.asarray(x.mean(axis=0)).ravel()
varying = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2 > 0
genes = [g for g, keep in zip(genes, varying) if keep]
x = x[:, np.flatnonzero(varying)].tocsr()
tfs = [t for t in tfs if t in set(genes)]

from rustscenic import grn
t = time.perf_counter()
net = grn.infer((x, genes), tf_names=tfs, seed=SEED, verbose=False, early_stop_mode=MODE, **SETTINGS)
secs = time.perf_counter() - t
FIT = dict(net.attrs.get('grn_fit', {}))  # finished builds record max_bins; the prototype used the environment
OUT.mkdir(parents=True, exist_ok=True)
net = net[['TF', 'target', 'importance']].assign(importance=lambda d: d.importance.astype(np.float32))
net.astype({'TF': 'category', 'target': 'category'}).to_parquet(OUT / 'network.parquet', index=False)
trees = net.groupby('target', observed=True).importance.sum().round().astype(int)
sha = lambda v: hashlib.sha256('\n'.join(v).encode()).hexdigest()
(OUT / 'info.json').write_text(json.dumps({'dataset': DATA, 'cells_sha256': sha(cells), 'genes_sha256': sha(genes), 'max_bins': str(net_attrs_bins) if (net_attrs_bins := FIT.get('max_bins')) else os.environ.get('RUSTSCENIC_MAX_BINS', '255'), 'seed': SEED, 'mode': MODE,
                                            'inference_s': secs, 'trees_mean': float(trees.mean()), 'edges': len(net)}))
print(OUT.name, round(secs, 1), 's', float(trees.mean()), len(net), flush=True)
