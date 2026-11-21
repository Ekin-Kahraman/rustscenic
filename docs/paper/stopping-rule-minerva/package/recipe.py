"""Frozen candidate-module recipes; inference remains the official 0.6.1 kernel."""
from time import perf_counter
import numpy as np
import pandas as pd
from rustscenic import cistarget, grn

VARIANTS = ('original', 'diverse_importance', 'diverse_pearson', 'diverse_consensus')


def weighted_edges(network, variant):
    edge = network.loc[network.importance.gt(0) & network.TF.ne(network.target)].copy()
    if not np.isfinite(edge[['importance','rho']].to_numpy()).all():
        raise ValueError('Nonfinite network weights')
    edge['regulation'] = (edge.rho.gt(.03).astype(int) - edge.rho.lt(-.03).astype(int))
    edge['original_importance'] = edge.importance
    if variant == 'diverse_pearson':
        edge['importance'] = edge.rho.abs()
    elif variant == 'diverse_consensus':
        group = edge.groupby(['TF','regulation'], sort=False)
        a = group.original_importance.rank(method='average', pct=True)
        edge['absolute_rho'] = edge.rho.abs()
        b = edge.groupby(['TF','regulation'], sort=False).absolute_rho.rank(method='average', pct=True)
        edge['importance'] = (a+b)/2
    elif variant not in ('original','diverse_importance'):
        raise ValueError(variant)
    return edge


def module_families(edge, variant):
    if variant == 'original':
        return {'top50': grn.build_regulons(edge, top_targets_per_tf=50, min_targets=10, include_repressors=True)}
    thresholds = edge.importance.quantile([.75,.90]).tolist()
    families = {name:{} for name in ('q75','q90','top50','reg5','reg10','reg50')}
    for sign, suffix in ((1,'activator'),(-1,'repressor')):
        part = edge.loc[edge.regulation.eq(sign)]
        ranked = part.sort_values(['TF','importance','target'], ascending=[True,False,True])
        per_target = part.sort_values(['target','importance','TF'], ascending=[True,False,True])
        selections = {'q75':part[part.importance.gt(thresholds[0])],
                      'q90':part[part.importance.gt(thresholds[1])],
                      'top50':ranked.groupby('TF',sort=False).head(50)}
        selections.update({f'reg{n}':per_target.groupby('target',sort=False).head(n) for n in (5,10,50)})
        for family, frame in selections.items():
            for tf, group in frame.groupby('TF',sort=True):
                genes = sorted(set(group.target) | {tf})
                if len(genes)>=20: families[family][tf+'_'+suffix]=genes
    return families


def refine(network, rankings, annotations, universe, variant):
    start=perf_counter()
    edges=weighted_edges(network,variant)
    families=module_families(edges,variant)
    membership={}; evidence=[]; stats=[]
    for family,candidates in families.items():
        t=perf_counter()
        enriched=cistarget.enrich(rankings,candidates,top_frac=.05,auc_threshold=0.,nes_threshold=3.,rank_universe_size=universe)
        kept=cistarget.prune_regulons(enriched,candidates,annotations,rankings=rankings,top_frac=.05,auc_threshold=0.,nes_threshold=3.,min_genes=5,rank_universe_size=universe)
        for name, genes in kept.items():membership.setdefault(name,set()).update(genes)
        if len(enriched):evidence.append(enriched.assign(family=family))
        stats.append({'family':family,'candidate_modules':len(candidates),'retained_modules':len(kept),'seconds':perf_counter()-t})
    regulons={name:sorted(genes) for name,genes in sorted(membership.items())}
    return regulons, edges, {'variant':variant,'wall_s':perf_counter()-start,'families':stats,'regulons':len(regulons)}, pd.concat(evidence,ignore_index=True) if evidence else pd.DataFrame()
