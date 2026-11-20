"""Does the arboreto early-stopping rule limit GRN quality?

arboreto stops boosting when the mean of the last 25 out-of-bag improvements is
negative. On every dataset inspected so far this fires at about 25-35 trees of
learning rate 0.01, for both pySCENIC and RustScenic. This sweep varies only the
boosting budget on the three BEELINE panels used in the paper and scores the raw
network against the packaged experimental references with the paper's metric.
Development screen on inspected data.
"""
import itertools
import json
import sys
import time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from rustscenic import grn

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'RustScenic-Paper-Validation-20261106/rustscenic-paper-validation'
sys.path.insert(0, str(PKG))
from reference_metrics import reference_metrics, positive_top_sets  # noqa: E402

BASE = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9, max_depth=3,
            early_stop_window=25, early_stop_mode='arboreto')
VARIANTS = {
    'default_early_stop': {},
    'trees100': dict(n_estimators=100, early_stop_window=0),
    'trees500': dict(n_estimators=500, early_stop_window=0),
    'trees2000': dict(n_estimators=2000, early_stop_window=0),
    'lr0.1_early_stop': dict(learning_rate=.1),
    'lr0.1_trees200': dict(learning_rate=.1, n_estimators=200, early_stop_window=0),
    'paired_oob': dict(early_stop_mode='paired_oob'),
}
if len(sys.argv) > 1:  # e.g. "default_early_stop,paired_oob" tag
    VARIANTS = {k: VARIANTS[k] for k in sys.argv[1].split(',')}
TAG = sys.argv[2] if len(sys.argv) > 2 else 'all'
SEEDS = (777, 778, 779)


def pearson_network(x, genes, tfs):
    dense = np.asarray(x.todense(), dtype=np.float64)
    z = (dense - dense.mean(0)) / np.where(dense.std(0) > 0, dense.std(0), 1)
    rho = z.T @ z / len(z)
    idx = {g: i for i, g in enumerate(genes)}
    rows = [(tf, g, abs(rho[idx[tf], idx[g]])) for tf in tfs for g in genes if tf != g]
    return pd.DataFrame(rows, columns=['TF', 'target', 'importance'])


def stability(networks, tfs):
    values = []
    for a, b in itertools.combinations(networks, 2):
        x, y = positive_top_sets(a, tfs), positive_top_sets(b, tfs)
        values.append(np.mean([len(x[t] & y[t]) / len(x[t] | y[t]) for t in tfs if x[t] | y[t]]))
    return float(np.mean(values))


rows = []
for name in ('hESC', 'mDC', 'mESC'):
    info = json.loads((PKG / f'rna-inputs/{name}/input.json').read_text())
    x = sparse.load_npz(PKG / f'rna-inputs/{name}/matrix.npz').tocsr()
    genes, tfs = info['genes'], info['tfs']
    adata = ad.AnnData(x, var=pd.DataFrame(index=genes))
    refs = {r['file']: pd.read_csv(PKG / f'rna-inputs/{name}' / r['file'], sep='\t') for r in info['references']}
    candidates = {'pearson': ([pearson_network(x, genes, tfs)], np.nan, 0.)}
    for variant, change in VARIANTS.items():
        nets, start = [], time.perf_counter()
        for seed in SEEDS:
            nets.append(grn.infer(adata, tf_names=tfs, seed=seed, verbose=False, **{**BASE, **change}))
        seconds = (time.perf_counter() - start) / len(SEEDS)
        trees = float(np.mean([n.groupby('target').importance.sum().mean() for n in nets]))
        candidates[variant] = (nets, trees, seconds)
    # Seed ensemble of the default: mean importance over the three default runs.
    if 'default_early_stop' in candidates:
        merged = pd.concat(candidates['default_early_stop'][0]).groupby(['TF', 'target'], as_index=False).importance.sum()
        candidates['default_3seed_sum'] = ([merged], np.nan, candidates['default_early_stop'][2] * 3)
    for variant, (nets, trees, seconds) in candidates.items():
        support_tfs = sorted(set(tfs))
        for ref_name, truth in refs.items():
            metrics = [reference_metrics(n, truth, genes, tfs) for n in nets]
            rows.append({'dataset': name, 'reference': ref_name.replace('-network.tsv', ''), 'variant': variant,
                         'mean_trees_per_target': trees, 'seconds_per_fit': seconds,
                         'AP': np.mean([m['average_precision'] for m in metrics]),
                         'AP_over_prevalence': np.mean([m['ap_over_prevalence'] for m in metrics]),
                         'top50_fold_over_uniform': np.mean([m['top50_fold_over_uniform'] for m in metrics]),
                         'seed_top50_jaccard': stability(nets, support_tfs) if len(nets) > 1 else np.nan})
        print(name, variant, 'done', flush=True)

frame = pd.DataFrame(rows)
(HERE / 'diagnosis').mkdir(exist_ok=True)
frame.to_csv(HERE / f'diagnosis/tree-budget-beeline-{TAG}.csv', index=False)
pd.set_option('display.width', 220, 'display.max_rows', 200)
print(frame.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
