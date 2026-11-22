"""Post hoc, written separately for the R4 audit: (1) tree-count predictability; (2) cross-rule regulon concordance
and lineage AUROC by rule, BMMC 20,000 cells, from the returned archive. Reads only."""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
E = Path(sys.argv[1]); SEEDS = (777, 778, 779)
ARMS = {('pySCENIC', 'current'): ('reference_cross', 'regulons-reference.json'),
        ('pySCENIC', 'original'): ('reference_original', 'regulons-reference.json'),
        ('RustScenic-div', 'current'): ('native_cross', 'regulons-diverse_importance.json'),
        ('RustScenic-div', 'original'): ('native_same', 'regulons-diverse_importance.json'),
        ('RustScenic-top50', 'current'): ('native_cross', 'regulons-original.json'),
        ('RustScenic-top50', 'original'): ('native_same', 'regulons-original.json')}
print('# (1) Spearman: trees per target vs max |rho| over TFs (seed 777); seed-to-seed Spearman of tree counts')
for ds in ('bmmc', 'atlas20000'):
    p = pd.read_parquet(E / f'{ds}-777/pearson/network.parquet', columns=['TF', 'target', 'rho'])
    p = p[p.TF.astype(str) != p.target.astype(str)]
    pred = p.assign(a=p.rho.abs()).groupby('target', observed=True).a.max()
    pred.index = pred.index.astype(str)
    for arm in ('reference_cross', 'reference_original', 'native_cross', 'native_same'):
        t = {s: pd.read_csv(E / f'{ds}-{s}/{arm}/trees.csv').astype({'target': str}).set_index('target').trees for s in SEEDS}
        j = pd.concat([t[777], pred], axis=1, join='inner')
        r_pred = spearmanr(j.iloc[:, 0], j.iloc[:, 1]).statistic
        r_seed = [spearmanr(*pd.concat([t[a], t[b]], axis=1, join='inner').T.values).statistic for a, b in combinations(SEEDS, 2)]
        at25 = np.mean([(t[s] == 25).mean() for s in SEEDS])
        print(f'{ds:11s} {arm:19s} rho_pred={r_pred:.4f} rho_seeds={min(r_seed):.4f}-{max(r_seed):.4f} share_at_25={at25:.4f} n={len(j)}')
print('# (2) regulons, BMMC')
def load(tool, rule, seed):
    arm, f = ARMS[(tool, rule)]
    return {k: frozenset(v) for k, v in json.loads((E / f'bmmc-{seed}/{arm}/{f}').read_text()).items()}
J = lambda a, b: len(a & b) / max(1, len(a | b))
for tool in ('pySCENIC', 'RustScenic-div', 'RustScenic-top50'):
    R = {(r, s): load(tool, r, s) for r in ('current', 'original') for s in SEEDS}
    cons = {r: set.intersection(*[set(R[(r, s)]) for s in SEEDS]) for r in ('current', 'original')}
    shared = cons['current'] & cons['original']
    within = {r: np.mean([J(set(R[(r, a)]), set(R[(r, b)])) for a, b in combinations(SEEDS, 2)]) for r in ('current', 'original')}
    across = np.mean([J(set(R[('current', s)]), set(R[('original', s)])) for s in SEEDS])
    def tgt(r1, s1, r2, s2):
        A, B = R[(r1, s1)], R[(r2, s2)]; ks = set(A) & set(B)
        return np.mean([J(A[k], B[k]) for k in ks])
    tw = {r: np.mean([tgt(r, a, r, b) for a, b in combinations(SEEDS, 2)]) for r in ('current', 'original')}
    ta = np.mean([tgt('current', s, 'original', s) for s in SEEDS])
    print(f'{tool:17s} consensus current={len(cons["current"])} original={len(cons["original"])} shared={len(shared)} '
          f'retained={len(shared)/len(cons["current"]):.3f} | identity J within cur={within["current"]:.3f} orig={within["original"]:.3f} across(same seed)={across:.3f} '
          f'| target J within cur={tw["current"]:.3f} orig={tw["original"]:.3f} across={ta:.3f}')
print('# (3) lineage activity AUROC by rule, BMMC (regulator present under both rules, same seed)')
LIN = {'pySCENIC': ('reference_cross', 'reference_original', None), 'RustScenic': ('native_cross', 'native_same', None)}
for tool, (c, o, _) in LIN.items():
    rows = []
    for s in SEEDS:
        a = pd.read_csv(E / f'bmmc-{s}/{c}/lineage.csv'); b = pd.read_csv(E / f'bmmc-{s}/{o}/lineage.csv')
        for rec in sorted(set(a.regulons)):
            m = a[a.regulons == rec].merge(b[b.regulons == rec], on=['TF', 'lineage'], suffixes=('_c', '_o'))
            m = m[m.regulon_present_c & m.regulon_present_o]
            for _, r in m.iterrows():
                rows.append(dict(recipe=rec, seed=s, TF=r.TF, tier=r.tier_c, c=r.activity_auroc_c, o=r.activity_auroc_o))
    d = pd.DataFrame(rows)
    for rec, g in d.groupby('recipe'):
        diff = g.o - g.c
        print(f'{tool:10s} {rec:22s} n={len(g)} median_cur={g.c.median():.3f} median_orig={g.o.median():.3f} median_diff={diff.median():+.4f} improved={(diff > 0).mean():.2f}')
