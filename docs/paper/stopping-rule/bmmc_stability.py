"""Seed stability of bone-marrow networks and regulons under each stopping rule (development)."""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'lineage_regulators.json').read_text())
LINEAGE = list(SPEC['primary']) + list(SPEC['extended'])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


def top_sets(net, k=50):
    net = net[(net.importance > 0) & (net.TF != net.target)]
    ranked = net.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable')
    return {tf: set(g.target.head(k)) for tf, g in ranked.groupby('TF', sort=False)}


def main(prefix, modes, seeds, recipe='diverse_importance'):
    rows, presence = [], []
    for mode in modes:
        regs, tops, info = {}, {}, {}
        for seed in seeds:
            d = HERE / 'fits' / f'{prefix}-{mode}-{seed}'
            if not (d / f'{recipe}.json').exists():
                continue
            regs[seed] = json.loads((d / f'{recipe}.json').read_text())
            tops[seed] = top_sets(pd.read_parquet(d / 'network.parquet', columns=['TF', 'target', 'importance']))
            info[seed] = json.loads((d / 'variant.json').read_text())
        pairs = list(itertools.combinations(sorted(regs), 2))
        ident = [jaccard(set(regs[a]), set(regs[b])) for a, b in pairs]
        target = [np.mean([jaccard(set(regs[a][k]), set(regs[b][k])) for k in set(regs[a]) & set(regs[b])]) for a, b in pairs]
        raw = [np.mean([jaccard(tops[a][t], tops[b][t]) for t in set(tops[a]) & set(tops[b])]) for a, b in pairs]
        rows.append({'mode': mode, 'seeds': len(regs), 'trees_mean': np.mean([i['trees_mean'] for i in info.values()]),
                     'inference_s': np.mean([i['inference_s'] for i in info.values()]),
                     'regulons': np.mean([len(r) for r in regs.values()]),
                     'raw_top50_jaccard': np.mean(raw) if raw else np.nan,
                     'regulon_identity_jaccard': np.mean(ident) if ident else np.nan,
                     'target_jaccard_shared': np.mean(target) if target else np.nan,
                     'regulons_in_all_seeds': len(set.intersection(*[set(r) for r in regs.values()])) if regs else 0})
        for tf in LINEAGE:
            n = sum(len(set(r.get(tf + '_activator', [])) - {tf}) >= 5 for r in regs.values())
            presence.append({'mode': mode, 'TF': tf, 'seeds_with_activating_regulon': n, 'seeds': len(regs)})
    return pd.DataFrame(rows), pd.DataFrame(presence)


if __name__ == '__main__':
    prefix = sys.argv[1]
    modes = sys.argv[2].split(',')
    seeds = [int(s) for s in sys.argv[3].split(',')]
    summary, presence = main(prefix, modes, seeds)
    out = HERE / 'results'
    summary.to_csv(out / f'{prefix}-stability.csv', index=False)
    presence.to_csv(out / f'{prefix}-lineage-presence.csv', index=False)
    pd.set_option('display.width', 220)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    wide = presence.pivot(index='TF', columns='mode', values='seeds_with_activating_regulon').loc[LINEAGE]
    print(wide.T.to_string())
    print('lineage regulators with an activating regulon in every seed:', (wide == presence.seeds.max()).sum().to_dict())
