"""Post hoc: check the finished max_bins implementation against earlier networks, bit for bit.

- default (exact bins) must equal the prototype's RUSTSCENIC_MAX_BINS=65535 networks;
- max_bins=255 must equal the 255-bin networks (bone marrow: Minerva's returned native_cross).
Run with the finished build's Python. Writes verify-default.json next to this script.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

D = Path(__file__).resolve().parent
PKG = D.parent / 'rustscenic-stopping-rule'
sys.path.insert(0, str(PKG))
from data_io import load_matrix  # noqa: E402
from rustscenic import grn  # noqa: E402

SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)


def prepare(x, genes, tfs):
    x = x.tocsc()
    mean = np.asarray(x.mean(axis=0)).ravel()
    varying = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2 > 0
    genes = [g for g, keep in zip(genes, varying) if keep]
    x = x[:, np.flatnonzero(varying)].tocsr()
    return x, genes, [t for t in tfs if t in set(genes)]


def load(name):
    if name == 'bmmc':
        panel = json.loads((PKG / 'inputs/bmmc/panel.json').read_text())
        return prepare(sparse.load_npz(PKG / 'inputs/bmmc/training.npz').tocsr(), panel['genes'], panel['tfs'])
    x, genes, _ = load_matrix(D / 'atlas-rebuilt/rna.h5', int(name.removeprefix('atlas')))
    return prepare(x, genes, (D / 'atlas-rebuilt/tfs.txt').read_text().split())


def same(net, reference):
    a = net[['TF', 'target', 'importance']].astype({'TF': str, 'target': str})
    a = a.assign(importance=a.importance.astype(np.float32))
    b = pd.read_parquet(reference, columns=['TF', 'target', 'importance']).astype({'TF': str, 'target': str})
    b = b.assign(importance=b.importance.astype(np.float32))
    m = a.merge(b, on=['TF', 'target'], how='outer', indicator=True)
    diff = (m.importance_x - m.importance_y).abs().max()
    return dict(edges=len(a), reference_edges=len(b), unmatched=int((m._merge != 'both').sum()), max_abs_diff=float(diff),
                identical=bool((m._merge == 'both').all() and diff == 0))


checks = [
    ('atlas20000', None, D / 'native-bins/atlas20000-cap65535-arboreto-777/network.parquet'),
    ('atlas20000', 255, D / 'native-bins/atlas20000-cap255-arboreto-777/network.parquet'),
    ('bmmc', None, D / 'native-bins/cap65535-arboreto-777/network.parquet'),
    ('bmmc', 255, D.parent / 'return-20261118/audit/extracted/jobs/bmmc-777/native_cross/network.parquet'),
]
results = []
for name, max_bins, reference in checks:
    x, genes, tfs = load(name)
    kwargs = {} if max_bins is None else {'max_bins': max_bins}
    net = grn.infer((x, genes), tf_names=tfs, seed=777, verbose=False, early_stop_mode='arboreto', **SETTINGS, **kwargs)
    row = dict(dataset=name, max_bins=net.attrs['grn_fit']['max_bins'], reference=str(reference.relative_to(D.parent)), **same(net, reference))
    print(row, flush=True)
    results.append(row)
(D / 'verify-default.json').write_text(json.dumps(results, indent=2) + '\n')
print('ALL IDENTICAL' if all(r['identical'] for r in results) else 'MISMATCH', flush=True)
