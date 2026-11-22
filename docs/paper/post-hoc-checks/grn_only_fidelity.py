"""Protocol fidelity metric on the 26 Oct GRN-only atlas networks (0.5.0 engine vs arboreto), written separately."""
import sys
from itertools import combinations, product
from pathlib import Path
import numpy as np, pandas as pd
R = Path(sys.argv[1]); SEEDS = (777, 778, 779)
def path(seed, tool):
    return next((R / f'atlas-n1306127-seed{seed}').glob(f'*/{tool}/network.parquet'))
def tops(p):
    n = pd.read_parquet(p, columns=['TF', 'target', 'importance']).astype({'TF': str, 'target': str})
    n['importance'] = n.importance.astype(np.float32)
    n = n[(n.importance > 0) & (n.TF != n.target)]
    n = n.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable').groupby('TF', sort=False).head(50)
    return {k: frozenset(v) for k, v in n.groupby('TF').target}
def jac(a, b):
    ks = set(a) | set(b)
    return float(np.mean([len(a.get(k, frozenset()) & b.get(k, frozenset())) / max(1, len(a.get(k, frozenset()) | b.get(k, frozenset()))) for k in ks]))
for ref in ('arboreto_mp_sparse', 'arboreto_sparse'):
    A = {s: tops(path(s, ref)) for s in SEEDS}; N = {s: tops(path(s, 'rustscenic')) for s in SEEDS}
    wa = np.mean([jac(A[a], A[b]) for a, b in combinations(SEEDS, 2)])
    wn = np.mean([jac(N[a], N[b]) for a, b in combinations(SEEDS, 2)])
    bt = np.mean([jac(N[a], A[b]) for a, b in product(SEEDS, SEEDS)])
    print(f'{ref}: within_arboreto={wa:.4f} within_rustscenic={wn:.4f} between={bt:.4f} fidelity={bt/wa:.4f} TFs={len(A[777])}')
