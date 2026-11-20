"""Post hoc diagnostic: per-TF agreement (TF-side top-50) against how RustScenic bins that TF.
RustScenic keeps at most 254 cut points, spaced evenly over a TF's sorted unique values. When a TF has
more than 255 unique values the first cut point is not the one between zero and the smallest positive
value, so some expressing cells share bin 0 with non-expressing cells. Not a protocol check."""
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

E, PKG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEEDS = (777, 778, 779)
MAX_BINS = 255


def first_edge_info(values):
    u = np.unique(values.astype(np.float32))
    n_pos = int((values > 0).sum())
    if len(u) <= MAX_BINS:
        return len(u), 0, n_pos
    stride = np.float32(len(u) - 1) / np.float32(MAX_BINS - 1)
    idx = int(np.float32(1) * stride)
    edge = (u[idx] + u[min(idx + 1, len(u) - 1)]) * np.float32(.5)
    merged = int(((values > 0) & (values < edge)).sum())  # expressing cells binned with zeros
    return len(u), merged, n_pos


def tf_tops(path):
    n = pd.read_parquet(path, columns=['TF', 'target', 'importance']).astype({'TF': str, 'target': str})
    n = n[(n.importance > 0) & (n.TF != n.target)]
    r = n.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable').groupby('TF', sort=False).head(50)
    return r.groupby('TF').target.agg(frozenset).to_dict()


def j(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


import json
panel = json.loads((PKG / 'inputs/bmmc/panel.json').read_text())
x = sparse.load_npz(PKG / 'inputs/bmmc/training.npz').tocsc()
genes = panel['genes']
col = {g: i for i, g in enumerate(genes)}
fitted = json.loads((E / 'jobs/bmmc-777/native_cross/panel.json').read_text())
info = {t: first_edge_info(x[:, col[t]].toarray().ravel()) for t in fitted['tfs']}

rows = []
for rule, (nat, ref) in {'current': ('native_cross', 'reference_cross'), 'original': ('native_same', 'reference_original')}.items():
    R = {s: tf_tops(E / f'jobs/bmmc-{s}/{nat}/network.parquet') for s in SEEDS}
    P = {s: tf_tops(E / f'jobs/bmmc-{s}/{ref}/network.parquet') for s in SEEDS}
    for tf in fitted['tfs']:
        g = lambda d, s: d[s].get(tf, frozenset())
        wp = np.nanmean([j(g(P, a), g(P, b)) for a, b in combinations(SEEDS, 2)])
        wr = np.nanmean([j(g(R, a), g(R, b)) for a, b in combinations(SEEDS, 2)])
        bt = np.nanmean([j(g(R, a), g(P, b)) for a, b in product(SEEDS, SEEDS)])
        u, merged, n_pos = info[tf]
        rows.append(dict(rule=rule, TF=tf, unique_values=u, expressing_cells=n_pos, expressing_cells_in_zero_bin=merged,
                         share_merged=merged / n_pos if n_pos else np.nan, within_pySCENIC=wp, within_RustScenic=wr, between=bt))
d = pd.DataFrame(rows)
d.to_csv(OUT, index=False)
d['group'] = pd.cut(d.share_merged, [-1e-9, 0, .05, .2, 1.0], labels=['exact (<=255 values)', '0-5% merged', '5-20% merged', '>20% merged'])
summary = d.groupby(['rule', 'group'], observed=True).agg(TFs=('TF', 'size'), within_pySCENIC=('within_pySCENIC', 'mean'),
                                                           within_RustScenic=('within_RustScenic', 'mean'), between=('between', 'mean'))
summary['ratio'] = summary.between / summary.within_pySCENIC
print(summary.round(3).to_string())
print(d.groupby('rule')[['unique_values', 'share_merged']].describe().round(3).T.to_string())
