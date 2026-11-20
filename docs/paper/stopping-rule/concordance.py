"""Seed-noise-calibrated agreement between implementations.

For any set of runs labelled (method, seed) this compares
  within-reference   (pySCENIC seed a vs pySCENIC seed b)
  within-native      (RustScenic seed a vs RustScenic seed b)
  between            (RustScenic seed a vs pySCENIC seed b)
on the raw network (top-k targets per TF, importance rank correlation) and on
the final regulons (TF identity, target membership). If `between` matches
`within-reference`, swapping the implementation changes the result no more than
re-running the reference with another seed.
"""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def top_sets(net, k):
    net = net[net.TF != net.target]
    ranked = net.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True])
    return {tf: set(g.target.head(k)) for tf, g in ranked.groupby('TF', sort=False)}


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else np.nan


def network_agreement(a, b, k=50):
    ta, tb = top_sets(a, k), top_sets(b, k)
    tfs = sorted(set(ta) & set(tb))
    top = np.nanmean([jaccard(ta[t], tb[t]) for t in tfs])
    merged = a.merge(b, on=['TF', 'target'], suffixes=('_a', '_b'))
    rho = spearmanr(merged.importance_a, merged.importance_b).statistic
    return {'top50_target_jaccard': top, 'importance_spearman_shared_edges': rho,
            'shared_edge_fraction': len(merged) / min(len(a), len(b))}


def regulon_agreement(a, b):
    tf_a, tf_b = set(a), set(b)
    shared = sorted(tf_a & tf_b)
    target = np.nanmean([jaccard(set(a[t]), set(b[t])) for t in shared]) if shared else np.nan
    return {'regulon_identity_jaccard': jaccard(tf_a, tf_b), 'shared_regulons': len(shared),
            'target_jaccard_shared_regulons': target}


def compare(runs):
    """runs: {(method, seed): {'network': DataFrame, 'regulons': dict}}"""
    rows = []
    for (ka, a), (kb, b) in itertools.combinations(sorted(runs.items()), 2):
        if ka[0] == kb[0]:
            kind = f'within-{ka[0]}'
        else:
            kind = 'between'
        rows.append({'comparison': kind, 'a': f'{ka[0]}-{ka[1]}', 'b': f'{kb[0]}-{kb[1]}',
                     **network_agreement(a['network'], b['network']),
                     **regulon_agreement(a['regulons'], b['regulons'])})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent / 'RustScenic-Biology-Confirm-Results-Audit-20261112/returned/jobs'
    out = Path(__file__).resolve().parent / 'diagnosis'
    frames = []
    for dataset in ('k562', 'rpe1'):
        runs = {}
        for seed in (777, 778, 779):
            job = root / f'{dataset}-{seed}'
            runs[('native', seed)] = {'network': pd.read_parquet(job / 'native/network.parquet'),
                                      'regulons': json.loads((job / 'native/diverse_importance.json').read_text())}
            runs[('reference', seed)] = {'network': pd.read_parquet(job / 'reference/network.parquet'),
                                         'regulons': json.loads((job / 'reference/reference.json').read_text())}
        frame = compare(runs)
        frame.insert(0, 'dataset', dataset)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out / 'replogle-control-concordance-pairs.csv', index=False)
    summary = result.groupby(['dataset', 'comparison']).mean(numeric_only=True)
    summary.to_csv(out / 'replogle-control-concordance-summary.csv')
    pd.set_option('display.width', 220)
    print(summary.to_string(float_format=lambda v: f'{v:.4f}'))
