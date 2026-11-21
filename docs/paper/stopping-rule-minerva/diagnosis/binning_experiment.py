"""Post hoc causal test: does RustScenic's 255-bin feature quantisation explain the bone-marrow fidelity gap?

Fits arboreto 0.1.6 / scikit-learn 1.3.2 (current rule, the Minerva reference_cross settings) on a random
subset of bone-marrow targets twice: on the raw TF matrix (must reproduce the returned Minerva networks
exactly) and on RustScenic's bin indices for each TF (identical cut points, re-implemented from
histogram.rs). Writes one parquet per (variant, seed)."""
import json
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

PKG, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N_TARGETS = int(sys.argv[3])
VARIANTS = sys.argv[4].split(',')
SEEDS = [int(s) for s in sys.argv[5].split(',')]
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
KW = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3)
MAX_BINS = 255


def rust_bins(values, max_bins=MAX_BINS):
    """Exact port of histogram.rs assign_bins_from_values (f32 arithmetic); max_bins=255 is RustScenic."""
    MAX_BINS = max_bins
    u = np.unique(values.astype(np.float32))
    if len(u) <= MAX_BINS:
        edges = ((u[:-1] + u[1:]) * np.float32(.5)).astype(np.float32)
    else:
        stride = np.float32(len(u) - 1) / np.float32(MAX_BINS - 1)
        idx = np.array([int(np.float32(i + 1) * stride) for i in range(MAX_BINS - 1)])
        a, b = u[np.minimum(idx, len(u) - 1)], u[np.minimum(idx + 1, len(u) - 1)]
        edges = ((a + b) * np.float32(.5)).astype(np.float32)
    return np.minimum(np.searchsorted(edges, values.astype(np.float32), side='left'), MAX_BINS - 1).astype(np.float32)


def load():
    panel = json.loads((PKG / 'inputs/bmmc/panel.json').read_text())
    x = sparse.load_npz(PKG / 'inputs/bmmc/training.npz').tocsc()
    genes, tfs = panel['genes'], panel['tfs']
    mean = np.asarray(x.mean(axis=0)).ravel()
    varying = np.asarray(x.power(2).mean(axis=0)).ravel() - mean ** 2 > 0
    genes = [g for g, k in zip(genes, varying) if k]
    x = x[:, np.flatnonzero(varying)].tocsc()
    kept = set(genes)
    return x, genes, [t for t in tfs if t in kept]


STATE = {}


def one(args):
    from arboreto.core import infer_partial_network
    variant, seed, gi = args
    x, genes, tf_matrix, tf_names = STATE['x'], STATE['genes'], STATE[variant], STATE['tf_names']
    links, meta = infer_partial_network('GBM', KW, tf_matrix, tf_names, genes[gi], x[:, gi],
                                        include_meta=True, early_stop_window_length=25, seed=seed)
    return variant, seed, links, meta


if __name__ == '__main__':
    from arboreto.core import to_tf_matrix
    x, genes, tfs = load()
    rng = np.random.default_rng(20261119)
    targets = sorted(rng.choice(len(genes), size=N_TARGETS, replace=False).tolist())
    tf_matrix, tf_names = to_tf_matrix(x, genes, tfs)
    STATE.update(x=x, genes=genes, tf_names=tf_names, raw=tf_matrix)
    for v in VARIANTS:
        if v.startswith('binned'):  # 'binned' = RustScenic's 255; 'binned1023' etc. = same scheme, more cut points
            cap = int(v[6:] or MAX_BINS)
            dense = tf_matrix.toarray() if sparse.issparse(tf_matrix) else np.asarray(tf_matrix)
            STATE[v] = np.column_stack([rust_bins(dense[:, j], cap) for j in range(dense.shape[1])])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'targets.json').write_text(json.dumps([genes[i] for i in targets]))
    jobs = [(v, s, gi) for v in VARIANTS for s in SEEDS for gi in targets]
    got = {}
    with get_context('fork').Pool(THREADS) as pool:
        for k, (v, s, links, meta) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            got.setdefault((v, s), ([], []))
            got[v, s][0].append(links)
            got[v, s][1].append(meta)
            if (k + 1) % 200 == 0:
                print(k + 1, 'of', len(jobs), flush=True)
    for (v, s), (links, meta) in got.items():
        pd.concat(links, ignore_index=True).to_parquet(OUT / f'{v}-{s}.parquet', index=False)
        pd.concat(meta, ignore_index=True).to_parquet(OUT / f'{v}-{s}-trees.parquet', index=False)
    print('done', flush=True)
