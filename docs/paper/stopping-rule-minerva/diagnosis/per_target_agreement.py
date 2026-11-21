"""Post hoc diagnostic: agreement measured from the regulator side (top-k TFs per target)
on the returned Minerva networks, beside the protocol's TF-side metric. Not a protocol check."""
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd

E = Path(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEEDS = (777, 778, 779)


def tops(path, k, by):
    n = pd.read_parquet(path, columns=['TF', 'target', 'importance'])
    n = n[(n.importance > 0) & (n.TF.astype(str) != n.target.astype(str))].astype({'TF': str, 'target': str})
    other = 'TF' if by == 'target' else 'target'
    r = n.sort_values([by, 'importance', other], ascending=[True, False, True], kind='stable').groupby(by, sort=False).head(k)
    return r.groupby(by)[other].agg(frozenset).to_dict()


def mean_j(a, b):
    keys = set(a) | set(b)
    return np.mean([len(a.get(t, frozenset()) & b.get(t, frozenset())) / max(1, len(a.get(t, frozenset()) | b.get(t, frozenset()))) for t in keys])


rows = []
for ds in ('bmmc', 'atlas20000'):
    for rule, (nat, ref) in {'current': ('native_cross', 'reference_cross'), 'original': ('native_same', 'reference_original')}.items():
        for by, k in (('target', K), ('TF', 50)):
            R = {s: tops(E / f'jobs/{ds}-{s}/{nat}/network.parquet', k, by) for s in SEEDS}
            P = {s: tops(E / f'jobs/{ds}-{s}/{ref}/network.parquet', k, by) for s in SEEDS}
            within_p = np.mean([mean_j(P[a], P[b]) for a, b in combinations(SEEDS, 2)])
            within_r = np.mean([mean_j(R[a], R[b]) for a, b in combinations(SEEDS, 2)])
            between = np.mean([mean_j(R[a], P[b]) for a, b in product(SEEDS, SEEDS)])
            rows.append(dict(dataset=ds, rule=rule, side=f'top{k} per {by}', within_pySCENIC=within_p, within_RustScenic=within_r,
                             between=between, ratio=between / within_p))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(Path(sys.argv[3]) if len(sys.argv) > 3 else 'per-target-agreement.csv', index=False)
