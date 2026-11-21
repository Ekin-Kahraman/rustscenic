"""Post hoc: the protocol's fidelity metric (TF-side top-50 Jaccard, positive importance, ties by target
name, single precision) for locally refitted RustScenic networks against the returned Minerva pySCENIC
networks of the same rule. Also by whether the TF has at most 255 distinct values. Not a protocol check."""
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd

E, NATIVE, PATTERN, RULE, BIN = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
DATASET = sys.argv[6] if len(sys.argv) > 6 else 'bmmc'  # BIN '-' skips the by-binning split
REF = {'current': 'reference_cross', 'original': 'reference_original'}[RULE]
SEEDS = (777, 778, 779)
exact = set(pd.read_csv(BIN).query('rule == "current" and unique_values <= 255').TF) if BIN != '-' else None


def tops(path):
    n = pd.read_parquet(path, columns=['TF', 'target', 'importance']).astype({'TF': str, 'target': str})
    n = n[(n.importance.astype(np.float32) > 0) & (n.TF != n.target)].assign(importance=lambda d: d.importance.astype(np.float32))
    r = n.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable').groupby('TF', sort=False).head(50)
    return r.groupby('TF').target.agg(frozenset).to_dict()


def mj(a, b, keep):
    keys = [k for k in set(a) | set(b) if keep(k)]
    return np.mean([len(a.get(k, frozenset()) & b.get(k, frozenset())) / max(1, len(a.get(k, frozenset()) | b.get(k, frozenset()))) for k in keys])


R = {s: tops(NATIVE / PATTERN.format(seed=s) / 'network.parquet') for s in SEEDS}
P = {s: tops(E / f'jobs/{DATASET}-{s}/{REF}/network.parquet') for s in SEEDS}
rows = []
groups = [('all TFs', lambda k: True)]
if exact is not None:
    groups += [('exact (<=255 values)', lambda k: k in exact), ('compressed (>255 values)', lambda k: k not in exact)]
for group, keep in groups:
    wp = np.mean([mj(P[a], P[b], keep) for a, b in combinations(SEEDS, 2)])
    wr = np.mean([mj(R[a], R[b], keep) for a, b in combinations(SEEDS, 2)])
    bt = np.mean([mj(R[a], P[b], keep) for a, b in product(SEEDS, SEEDS)])
    rows.append(dict(dataset=DATASET, networks=PATTERN, rule=RULE, TFs=group, within_pySCENIC=wp, within_RustScenic=wr, between=bt, fidelity=bt / wp))
print(pd.DataFrame(rows).round(3).to_string(index=False))
