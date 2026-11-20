"""Post hoc: agreement among RustScenic (Minerva), arboreto on raw values (Minerva) and arboreto on
RustScenic's bins (local), all restricted to the same random targets, current rule. TF-side top-k as in
the protocol, overall and by whether RustScenic bins the TF exactly. Not a protocol check."""
import json
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd

E, LOC, BIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
K = int(sys.argv[5]) if len(sys.argv) > 5 else 50
SEEDS = (777, 778, 779)
targets = set(json.loads((LOC / 'targets.json').read_text()))
exact = set(pd.read_csv(BIN).query('rule == "current" and unique_values <= 255').TF)


def tops(path):
    n = pd.read_parquet(path, columns=['TF', 'target', 'importance']).astype({'TF': str, 'target': str})
    n = n[(n.importance > 0) & (n.TF != n.target) & n.target.isin(targets)]
    n = n.assign(importance=n.importance.astype(np.float32))
    r = n.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable').groupby('TF', sort=False).head(K)
    return r.groupby('TF').target.agg(frozenset).to_dict()


nets = {'RustScenic': {s: tops(E / f'jobs/bmmc-{s}/native_cross/network.parquet') for s in SEEDS},
        'arboreto_raw': {s: tops(E / f'jobs/bmmc-{s}/reference_cross/network.parquet') for s in SEEDS},
        'arboreto_binned': {s: tops(LOC / f'binned-{s}.parquet') for s in SEEDS}}


def mj(a, b, keep):
    keys = [k for k in set(a) | set(b) if keep(k)]
    return np.mean([len(a.get(k, frozenset()) & b.get(k, frozenset())) / max(1, len(a.get(k, frozenset()) | b.get(k, frozenset()))) for k in keys])


rows = []
for group, keep in (('all TFs', lambda k: True), ('exact (<=255 values)', lambda k: k in exact), ('compressed (>255 values)', lambda k: k not in exact)):
    within = {t: np.mean([mj(nets[t][a], nets[t][b], keep) for a, b in combinations(SEEDS, 2)]) for t in nets}
    for a, b in (('RustScenic', 'arboreto_raw'), ('arboreto_binned', 'arboreto_raw'), ('RustScenic', 'arboreto_binned')):
        between = np.mean([mj(nets[a][x], nets[b][y], keep) for x, y in product(SEEDS, SEEDS)])
        rows.append(dict(TFs=group, pair=f'{a} vs {b}', within_first=within[a], within_second=within[b], between=between,
                         ratio_to_second=between / within[b]))
d = pd.DataFrame(rows)
d.to_csv(OUT, index=False)
print(d.round(3).to_string(index=False))
