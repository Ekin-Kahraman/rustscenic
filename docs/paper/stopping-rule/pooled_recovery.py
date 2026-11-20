"""Post hoc, descriptive: AP difference (original minus current rule) averaged over the
four confirmation datasets, per reference type, with an interval from the same
per-dataset TF-bootstrap draws used by beeline_score.py (seed 20261112, 2,000 draws).

Not part of the confirmation protocol, which judged each dataset on its own. The
mHSC datasets share one study, so a family-weighted mean (hHep 1/2, mHSC 1/6 each)
is reported beside the plain mean.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beeline_score as bs  # noqa: E402

INPUTS, RUNS = HERE / 'inputs/beeline-confirmation', HERE / 'runs/beeline-confirmation'
DATASETS = ('hHep', 'mHSC-E', 'mHSC-GM', 'mHSC-L')
FAMILY_WEIGHT = {'hHep': .5, 'mHSC-E': 1 / 6, 'mHSC-GM': 1 / 6, 'mHSC-L': 1 / 6}
SEEDS = (777, 778, 779)
COMPARISONS = (('native_same', 'native_cross'), ('reference_same', 'reference_cross'))


def kind(ref_type, label):
    if 'string' in label.lower():
        return 'STRING'
    return 'cell-type ChIP-seq' if 'non' not in label.lower() else 'non-specific ChIP-seq'


rows, draws = [], {}
for name in DATASETS:
    info = json.loads((INPUTS / name / 'input.json').read_text())
    genes, tfs = set(info['genes']), set(info['tfs'])
    nets = {(arm, s): pd.read_parquet(RUNS / name / f'{arm}-{s}.parquet') for arm in ('native_same', 'native_cross', 'reference_same', 'reference_cross')
            for s in SEEDS}
    for ref in info['references']:
        label = ref['file'].replace('-network.tsv', '')
        if kind(ref['reference_type'], label) != 'cell-type ChIP-seq':
            continue  # secondary references already have per-dataset intervals in ap-differences.csv
        truth = pd.read_csv(INPUTS / name / ref['file'], sep='\t')
        blocks = {k: bs.tf_blocks(n, truth, genes, tfs) for k, n in nets.items()}
        supported = sorted(next(iter(blocks.values())))
        rng = np.random.default_rng(bs.SEED)
        sample = [list(rng.choice(supported, size=len(supported), replace=True)) for _ in range(bs.BOOT)]
        point = {k: bs.pooled_ap(b, supported) for k, b in blocks.items()}
        table = {k: np.array([bs.pooled_ap(b, d) for d in sample]) for k, b in blocks.items()}
        for a, b in COMPARISONS:
            diff = np.mean([table[a, s] for s in SEEDS], 0) - np.mean([table[b, s] for s in SEEDS], 0)
            est = np.mean([point[a, s] for s in SEEDS]) - np.mean([point[b, s] for s in SEEDS])
            draws[name, kind(ref['reference_type'], label), a] = (est, diff)
        print('done', name, label, flush=True)

for (a, _b) in COMPARISONS:
    for k in ('cell-type ChIP-seq', 'non-specific ChIP-seq', 'STRING'):
        got = {d: draws[d, k, a] for d in DATASETS if (d, k, a) in draws}
        if len(got) < 4:
            continue
        for label, w in (('equal', {d: .25 for d in got}), ('family', FAMILY_WEIGHT)):
            est = sum(w[d] * got[d][0] for d in got)
            dist = sum(w[d] * got[d][1] for d in got)
            lo, hi = np.quantile(dist, [.025, .975])
            rows.append({'tool': 'RustScenic' if a.startswith('native') else 'arboreto', 'reference': k, 'weighting': label,
                         'AP_difference': est, 'low': lo, 'high': hi, 'datasets': len(got), 'draws': bs.BOOT})
out = pd.DataFrame(rows)
out.to_csv(HERE / 'results/beeline-confirmation-all-intervals/pooled-ap-differences.csv', index=False)
print(out.to_string(index=False))
