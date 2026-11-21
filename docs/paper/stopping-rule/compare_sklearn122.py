"""Is the emulated original-rule arm the same as arboreto on a real scikit-learn 1.2.2?

Emulated: arboreto 0.1.6 on scikit-learn 1.3.2 with the same-mask monitor
(reference_same, runs/beeline-dev and runs/beeline-confirmation).
Real: arboreto 0.1.6 with scikit-learn 1.2.2 first on the import path and no
patch (runs/beeline-sklearn122; the arm file is named reference_cross because
beeline_arms.py was run unmodified, and each JSON records scikit_learn 1.2.2).
Same data, settings and seeds. Added on 13 November, after the frozen protocol.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SETS = {'beeline-dev': ['hESC', 'mDC', 'mESC'], 'beeline-confirmation': ['hHep', 'mHSC-E', 'mHSC-GM', 'mHSC-L']}


def top50(net):
    net = net[(net.importance > 0) & (net.TF != net.target)]
    ranked = net.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable')
    return {tf: set(g.target.head(50)) for tf, g in ranked.groupby('TF', sort=False)}


rows = []
for folder, datasets in SETS.items():
    for dataset in datasets:
        for seed in (777, 778, 779):
            real_path = HERE / f'runs/beeline-sklearn122/{dataset}/reference_cross-{seed}'
            emulated_path = HERE / f'runs/{folder}/{dataset}/reference_same-{seed}'
            if not real_path.with_suffix('.parquet').exists():
                continue
            real_meta, emulated_meta = (json.loads(p.with_suffix('.json').read_text()) for p in (real_path, emulated_path))
            assert real_meta['scikit_learn'] == '1.2.2' and emulated_meta['scikit_learn'] == '1.3.2'
            real, emulated = (pd.read_parquet(p.with_suffix('.parquet')) for p in (real_path, emulated_path))
            merged = real.merge(emulated, on=['TF', 'target'], how='outer', suffixes=('_real', '_emulated')).fillna(0.)
            trees_real, trees_emulated = (n.groupby('target').importance.sum().round() for n in (real, emulated))
            a, b = top50(real), top50(emulated)
            overlap = [len(a.get(t, set()) & b.get(t, set())) / len(a.get(t, set()) | b.get(t, set())) for t in set(a) | set(b)]
            rows.append({'dataset': dataset, 'seed': seed, 'trees_mean_real_1.2.2': real_meta['trees_mean'],
                         'trees_mean_emulated': emulated_meta['trees_mean'],
                         'targets_with_identical_tree_count': float((trees_real.reindex(trees_emulated.index).fillna(-1) == trees_emulated).mean()),
                         'edges_real': len(real), 'edges_emulated': len(emulated),
                         'max_abs_importance_difference': float((merged.importance_real - merged.importance_emulated).abs().max()),
                         'max_relative_importance_difference': float(((merged.importance_real - merged.importance_emulated).abs()
                                                                      / merged[['importance_real', 'importance_emulated']].max(axis=1).clip(lower=1e-12)).max()),
                         'top50_jaccard_real_vs_emulated': float(np.mean(overlap))})
frame = pd.DataFrame(rows)
out = HERE / 'results/sklearn122-vs-emulated.csv'
frame.to_csv(out, index=False)
pd.set_option('display.width', 250)
print(frame.to_string(index=False))
print('written', out)
