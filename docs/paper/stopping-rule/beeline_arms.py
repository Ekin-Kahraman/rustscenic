"""Fit one GRN arm on BEELINE panels. One arm per process, so monitor patches never leak.

Arms
  native_cross     RustScenic, early_stop_mode="arboreto"   (scikit-learn >= 1.3 semantics)
  native_same      RustScenic, early_stop_mode="paired_oob" (same-mask, GRNBoost2 as published)
  reference_cross  arboreto 0.1.6 on the installed scikit-learn (>= 1.3)
  reference_same   arboreto 0.1.6 with the same-mask monitor (reference_paired.py)
  native_trees<N>  RustScenic, fixed N trees, no early stopping (dose-response only)
All arms share the package GRN settings: 5,000-tree ceiling, learning rate 0.01,
max_features 0.1, subsample 0.9, depth 3, window 25.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
SETTINGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3, early_stop_window=25)


def run(arm, x, genes, tfs, seed, threads):
    start = time.perf_counter()
    if arm.startswith('native'):
        import anndata as ad
        from rustscenic import grn
        extra = dict(early_stop_mode='arboreto')
        if arm == 'native_same':
            extra = dict(early_stop_mode='paired_oob')
        elif arm.startswith('native_trees'):
            extra = dict(early_stop_mode='arboreto', n_estimators=int(arm.removeprefix('native_trees')), early_stop_window=0)
        elif arm != 'native_cross':
            raise ValueError(arm)
        net = grn.infer(ad.AnnData(x, var=pd.DataFrame(index=genes)), tf_names=tfs, seed=seed, verbose=False,
                        **{**SETTINGS, **extra})
    else:
        sys.path.insert(0, str(PACKAGE))
        from grn_reference import infer_multiprocessing
        if arm == 'reference_same':
            import reference_paired
            reference_paired.install()
        elif arm != 'reference_cross':
            raise ValueError(arm)
        net, _, _, _, _ = infer_multiprocessing(x, genes, tfs, SETTINGS, seed, threads)
    seconds = time.perf_counter() - start
    net = net[['TF', 'target', 'importance']]
    trees = net.groupby('target').importance.sum()
    return net, {'arm': arm, 'seed': seed, 'seconds': seconds, 'edges': len(net), 'trees_mean': float(trees.mean()),
                 'trees_median': float(trees.median()), 'trees_max': float(trees.max())}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inputs', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('arm')
    p.add_argument('--datasets', required=True)
    p.add_argument('--seeds', default='777,778,779')
    p.add_argument('--threads', type=int, default=10)
    a = p.parse_args()
    import sklearn
    for name in a.datasets.split(','):
        info = json.loads((a.inputs / name / 'input.json').read_text())
        x = sparse.load_npz(a.inputs / name / 'matrix.npz').tocsr()
        dest = a.out / name
        dest.mkdir(parents=True, exist_ok=True)
        for seed in map(int, a.seeds.split(',')):
            target = dest / f'{a.arm}-{seed}.parquet'
            if target.exists():
                continue
            net, meta = run(a.arm, x, info['genes'], info['tfs'], seed, a.threads)
            meta.update(dataset=name, cells=x.shape[0], genes=x.shape[1], tfs=len(info['tfs']),
                        scikit_learn=sklearn.__version__, threads=a.threads)
            net.to_parquet(target, index=False)
            target.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
            print({k: meta[k] for k in ('dataset', 'arm', 'seed', 'seconds', 'trees_mean', 'trees_median', 'edges')}, flush=True)
