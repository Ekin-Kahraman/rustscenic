"""Post hoc diagnostic: how much of the TF-side disagreement comes from per-target tree counts.
(1) agreement of per-target tree counts within and between tools; (2) the protocol's TF-side
top-50 ratio recomputed after dividing each target's importances by its tree count. Not a protocol check."""
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd

E, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (777, 778, 779)


def load(ds, seed, arm):
    d = E / f'jobs/{ds}-{seed}/{arm}'
    n = pd.read_parquet(d / 'network.parquet', columns=['TF', 'target', 'importance']).astype({'TF': str, 'target': str})
    t = pd.read_csv(d / 'trees.csv').set_index('target').trees
    return n[(n.importance > 0) & (n.TF != n.target)], t


def tf_tops(n, col):
    r = n.sort_values(['TF', col, 'target'], ascending=[True, False, True], kind='stable').groupby('TF', sort=False).head(50)
    return r.groupby('TF').target.agg(frozenset).to_dict()


def mean_j(a, b):
    keys = set(a) | set(b)
    return np.mean([len(a.get(k, frozenset()) & b.get(k, frozenset())) / max(1, len(a.get(k, frozenset()) | b.get(k, frozenset()))) for k in keys])


rows = []
for ds in ('bmmc', 'atlas20000'):
    for rule, (nat, ref) in {'current': ('native_cross', 'reference_cross'), 'original': ('native_same', 'reference_original')}.items():
        R = {s: load(ds, s, nat) for s in SEEDS}
        P = {s: load(ds, s, ref) for s in SEEDS}
        sp = lambda a, b: a.corr(b.reindex(a.index), method='spearman')
        row = dict(dataset=ds, rule=rule,
                   trees_rho_within_pySCENIC=np.mean([sp(P[a][1], P[b][1]) for a, b in combinations(SEEDS, 2)]),
                   trees_rho_within_RustScenic=np.mean([sp(R[a][1], R[b][1]) for a, b in combinations(SEEDS, 2)]),
                   trees_rho_between=np.mean([sp(R[a][1], P[b][1]) for a, b in product(SEEDS, SEEDS)]),
                   trees_mean_RustScenic=np.mean([R[s][1].mean() for s in SEEDS]), trees_mean_pySCENIC=np.mean([P[s][1].mean() for s in SEEDS]))
        for label, col in (('raw', 'importance'), ('per_tree', 'share')):
            tops = {}
            for tool, nets in (('R', R), ('P', P)):
                for s, (n, t) in nets.items():
                    n = n.assign(share=n.importance / n.target.map(t).astype(float))
                    tops[tool, s] = tf_tops(n, col)
            wp = np.mean([mean_j(tops['P', a], tops['P', b]) for a, b in combinations(SEEDS, 2)])
            wr = np.mean([mean_j(tops['R', a], tops['R', b]) for a, b in combinations(SEEDS, 2)])
            bt = np.mean([mean_j(tops['R', a], tops['P', b]) for a, b in product(SEEDS, SEEDS)])
            row.update({f'{label}_within_pySCENIC': wp, f'{label}_within_RustScenic': wr, f'{label}_between': bt, f'{label}_ratio': bt / wp})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
