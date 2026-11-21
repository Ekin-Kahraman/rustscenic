"""Endpoint tables and the pre-specified checks of PROTOCOL.md, from whichever arms completed.

Reads only files that are returned in SEND_BACK (SUMMARY.json, monitor.json,
panel.json, network.parquet, top100.parquet, recovery.csv, regulons-*.json,
lineage.csv), so every number can be recomputed from the archive.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_json, write_json

DATASETS = ('bmmc', 'atlas20000', 'atlas100000', 'atlas500000', 'atlas1306127', 'hdma817740')
SEEDS = (777, 778, 779)
RULES = {'current': ('native_cross', 'reference_cross'), 'original': ('native_same', 'reference_original')}
ARMS = tuple(arm for pair in zip(*RULES.values()) for arm in pair)
REGULON_SETS = (('native_cross', 'original'), ('native_same', 'original'), ('native_cross', 'diverse_importance'),
                ('native_same', 'diverse_importance'), ('reference_cross', 'reference'), ('reference_original', 'reference'))
CONFIRMATORY = ('bmmc', 'atlas20000')
TREES_BAR, STABILITY_BAR, FIDELITY_BAR = 3., 1.5, .90


def top50(path):
    frame = pd.read_parquet(path, columns=['TF', 'target'])  # already ordered by TF, importance, target
    return {tf: set(group.target.head(50)) for tf, group in frame.groupby('TF', sort=False)}


def mean_jaccard(a, b):
    """Mean per-TF Jaccard over TFs with a target in either network (a TF absent from one scores 0)."""
    empty = set()
    values = [len(a.get(t, empty) & b.get(t, empty)) / len(a.get(t, empty) | b.get(t, empty)) for t in set(a) | set(b)]
    return float(np.mean(values)) if values else np.nan


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else np.nan


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def value(frame, column, **where):
    if not len(frame):
        return np.nan
    rows = frame.loc[np.logical_and.reduce([frame[k].eq(v) for k, v in where.items()])]
    return float(rows[column].iloc[0]) if len(rows) else np.nan


def recovery_intervals(done, out):
    """TF-bootstrap intervals for the two confirmatory settings; a failure here never blocks the other tables."""
    from recovery import intervals
    frames = []
    for dataset in CONFIRMATORY:
        folders = {key[1:]: folder for key, folder in done.items() if key[0] == dataset and (folder / 'network.parquet').exists()}
        if not folders:
            continue
        panel = read_json(next(iter(folders.values())) / 'panel.json')
        networks = {key: (lambda path=folder / 'network.parquet': pd.read_parquet(path, columns=['TF', 'target', 'importance']))
                    for key, folder in folders.items()}
        frames.append(intervals(networks, panel['genes'], panel['tfs'], dataset))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(out / 'recovery-differences.csv', index=False)
    return result


def summarise(work, expected_arms=None):
    out = work / 'summary'
    out.mkdir(exist_ok=True)
    done = {}
    for dataset, arm, seed in product(DATASETS, (*ARMS, 'pearson'), SEEDS):
        folder = work / f'jobs/{dataset}-{seed}/{arm}'
        if (folder / 'SUMMARY.json').exists() and read_json(folder / 'SUMMARY.json').get('status') == 'complete':
            done[dataset, arm, seed] = folder
    fitted = {key: folder for key, folder in done.items() if key[1] != 'pearson'}

    runs, stage_errors = [], {}
    for (dataset, arm, seed), folder in done.items():
        s, m = read_json(folder / 'SUMMARY.json'), read_json(folder / 'monitor.json') if (folder / 'monitor.json').exists() else {}
        if s.get('stage_errors'):
            stage_errors[f'{dataset}-{seed}/{arm}'] = sorted(s['stage_errors'])
        if arm != 'pearson':
            runs.append({'dataset': dataset, 'arm': arm, 'seed': seed, 'cells': s['cells'], 'genes': s['genes'], 'TFs': s['TFs'],
                         'stopping_rule': s['stopping_rule'], 'same_mask_monitor_emulated': s.get('same_mask_monitor_emulated', False),
                         'inference_s': s['inference_s'], 'wall_s': s['wall_s'], 'trees_mean': s['trees_mean'],
                         'trees_median': s['trees_median'], 'trees_p90': s['trees_p90'], 'trees_max': s['trees_max'],
                         'share_at_ceiling': s['share_at_ceiling'], 'edges': s['edges'],
                         'peak_pss_mb': m.get('peak_tree_pss_mb'), 'inference_peak_pss_mb': m.get('analysis_peak_pss_mb')})
    runs = pd.DataFrame(runs)
    runs.to_csv(out / 'cost-per-run.csv', index=False)
    cost = (runs.groupby(['dataset', 'arm'], sort=False).agg(
        seeds=('seed', 'size'), cells=('cells', 'first'), genes=('genes', 'first'), TFs=('TFs', 'first'),
        trees_mean=('trees_mean', 'mean'), trees_median=('trees_median', 'mean'), trees_p90=('trees_p90', 'mean'),
        share_at_ceiling=('share_at_ceiling', 'mean'), inference_s=('inference_s', 'mean'),
        peak_pss_mb=('peak_pss_mb', 'mean')).reset_index() if len(runs) else pd.DataFrame())
    cost.to_csv(out / 'cost.csv', index=False)

    tops = {key: top50(folder / 'top100.parquet') for key, folder in fitted.items()}
    stability, fidelity = [], []
    for dataset in DATASETS:
        seeds = {arm: [s for s in SEEDS if (dataset, arm, s) in tops] for arm in ARMS}
        within = {}
        for arm in ARMS:
            pairs = [mean_jaccard(tops[dataset, arm, a], tops[dataset, arm, b]) for a, b in combinations(seeds[arm], 2)]
            within[arm] = mean_or_nan(pairs)
            if seeds[arm]:
                stability.append({'dataset': dataset, 'arm': arm, 'seeds': len(seeds[arm]), 'seed_pairs': len(pairs),
                                  'top50_jaccard': within[arm], 'min_pair': min(pairs, default=np.nan), 'max_pair': max(pairs, default=np.nan)})
        for rule, (native, reference) in RULES.items():
            between = [mean_jaccard(tops[dataset, native, a], tops[dataset, reference, b])
                       for a, b in product(seeds[native], seeds[reference])]
            if between:
                fidelity.append({'dataset': dataset, 'rule': rule, 'pair': f'{native} vs {reference}', 'pairs': len(between),
                                 'between_implementations': mean_or_nan(between), 'within_reference_seeds': within[reference],
                                 'within_native_seeds': within[native], 'ratio': ratio(mean_or_nan(between), within[reference])})
    stability, fidelity = pd.DataFrame(stability), pd.DataFrame(fidelity)
    stability.to_csv(out / 'stability.csv', index=False)
    fidelity.to_csv(out / 'fidelity.csv', index=False)

    recovered = [pd.read_csv(folder / 'recovery.csv').assign(dataset=dataset, arm=arm, seed=seed)
                 for (dataset, arm, seed), folder in done.items() if (folder / 'recovery.csv').exists()]
    if recovered:
        recovered = pd.concat(recovered, ignore_index=True)
        recovered.to_csv(out / 'recovery-per-run.csv', index=False)
        recovered.groupby(['dataset', 'reference', 'arm'], sort=False).agg(
            seeds=('seed', 'size'), AP=('AP', 'mean'), prevalence=('prevalence', 'first'), AP_over_prevalence=('AP_over_prevalence', 'mean'),
            top50_fold_over_uniform=('top50_fold_over_uniform', 'mean'), reference_TFs=('reference_TFs', 'first'),
            eligible_links=('eligible_links', 'first')).reset_index().to_csv(out / 'recovery.csv', index=False)

    regulon_rows, lineage = [], []
    for arm, name in REGULON_SETS:
        sets = {seed: read_json(done['bmmc', arm, seed] / f'regulons-{name}.json') for seed in SEEDS
                if ('bmmc', arm, seed) in done and (done['bmmc', arm, seed] / f'regulons-{name}.json').exists()}
        if not sets:
            continue
        pairs = list(combinations(sets, 2))
        identity = [len(set(sets[a]) & set(sets[b])) / max(1, len(set(sets[a]) | set(sets[b]))) for a, b in pairs]
        shared = [mean_or_nan([len(set(sets[a][k]) & set(sets[b][k])) / len(set(sets[a][k]) | set(sets[b][k]))
                               for k in set(sets[a]) & set(sets[b])]) for a, b in pairs]
        regulon_rows.append({'arm': arm, 'regulons': name, 'seeds': len(sets), 'regulons_mean': mean_or_nan([len(r) for r in sets.values()]),
                             'regulons_in_all_seeds': len(set.intersection(*[set(r) for r in sets.values()])),
                             'identity_jaccard': mean_or_nan(identity), 'target_jaccard_shared': mean_or_nan(shared)})
    for (dataset, arm, seed), folder in fitted.items():
        if dataset == 'bmmc' and (folder / 'lineage.csv').exists():
            lineage.append(pd.read_csv(folder / 'lineage.csv').assign(arm=arm, seed=seed))
    pd.DataFrame(regulon_rows).to_csv(out / 'regulon-stability.csv', index=False)
    per_tf = pd.DataFrame()
    if lineage:
        lineage = pd.concat(lineage, ignore_index=True)
        lineage.to_csv(out / 'lineage-per-run.csv', index=False)
        per_tf = lineage.groupby(['arm', 'regulons', 'tier', 'TF', 'lineage'], sort=False).agg(
            seeds=('seed', 'size'), seeds_with_regulon=('regulon_present', 'sum'), targets=('targets', 'mean'),
            activity_auroc=('activity_auroc', 'mean'), expression_auroc=('expression_auroc', 'mean'),
            random_auroc=('random_auroc', 'mean')).reset_index()
        per_tf.to_csv(out / 'lineage-per-regulator.csv', index=False)
        per_tf.assign(every_seed=per_tf.seeds_with_regulon.eq(per_tf.seeds)).groupby(['arm', 'regulons', 'tier'], sort=False).agg(
            regulators=('TF', 'size'), with_regulon_every_seed=('every_seed', 'sum'),
            median_activity_auroc=('activity_auroc', 'median'), median_expression_auroc=('expression_auroc', 'median'),
            median_random_auroc=('random_auroc', 'median')).reset_index().to_csv(out / 'lineage-summary.csv', index=False)

    checks = []

    def check(endpoint, dataset, subject, observed, bar, confirmatory=True):
        met = None if not np.isfinite(observed) else bool(observed >= bar) if confirmatory else None
        checks.append({'endpoint': endpoint, 'dataset': dataset, 'subject': subject,
                       'observed': None if not np.isfinite(observed) else observed, 'bar': bar if confirmatory else None,
                       'confirmatory': confirmatory, 'met': met})
    for dataset in DATASETS:
        confirm = dataset in CONFIRMATORY
        for tool, (cross, same) in (('RustScenic', ('native_cross', 'native_same')), ('pySCENIC', ('reference_cross', 'reference_original'))):
            if tool == 'pySCENIC' and not confirm and dataset != 'atlas100000':
                continue
            check('1 training length: trees per target, original over current rule', dataset, tool,
                  ratio(value(cost, 'trees_mean', dataset=dataset, arm=same), value(cost, 'trees_mean', dataset=dataset, arm=cross)),
                  TREES_BAR, confirm)
            if dataset in ('bmmc', 'atlas20000') or (dataset == 'atlas100000' and tool == 'RustScenic'):
                check('2 seed stability: top-50 overlap, original over current rule', dataset, tool,
                      ratio(value(stability, 'top50_jaccard', dataset=dataset, arm=same),
                            value(stability, 'top50_jaccard', dataset=dataset, arm=cross)), STABILITY_BAR, confirm)
        if confirm:
            for rule in RULES:
                check('3 fidelity: between implementations over within pySCENIC seeds', dataset, rule + ' rule',
                      value(fidelity, 'ratio', dataset=dataset, rule=rule), FIDELITY_BAR)
    required = [c for c in checks if c['confirmatory']]
    emulated = bool(len(runs) and runs.loc[runs.arm.eq('reference_original'), 'same_mask_monitor_emulated'].any())
    primary = [{'arm': r.arm, 'TF': r.TF, 'seeds': int(r.seeds), 'seeds_with_regulon': int(r.seeds_with_regulon),
                'activity_auroc': float(r.activity_auroc) if np.isfinite(r.activity_auroc) else None}
               for r in per_tf.itertuples() if r.tier == 'primary' and r.regulons in ('diverse_importance', 'reference')]
    verdict = {
        'status': 'complete' if all(c['met'] is not None for c in required) else 'partial',
        'all_confirmatory_checks_met': all(c['met'] for c in required) if all(c['met'] is not None for c in required) else None,
        'reference_original_implementation': 'same-mask monitor on scikit-learn 1.3.2 (fallback)' if emulated else 'scikit-learn 1.2.2',
        'completed_arms': len(done), 'expected_arms': expected_arms, 'stage_errors': stage_errors, 'checks': checks,
        'reference_recovery': {'status': 'not computed'}, 'primary_lineage_regulators': primary,
        'scope': 'Checks follow PROTOCOL.md. The twelve confirmatory checks concern training length, reproducibility and '
                 'fidelity. Recovery of broad reference links is judged only by the two rules stated in the protocol.'}
    write_json(out / 'VERDICT.json', verdict)  # written before the bootstrap, which is the slowest and least essential step

    differences = recovery_intervals(done, out)
    rows = differences.loc[differences.comparison.eq('native_same minus native_cross')] if len(differences) else differences
    pairs = [{'dataset': r.dataset, 'reference': r.reference, 'AP_difference': r.AP_difference, 'low': r.low, 'high': r.high}
             for r in rows.itertuples()]
    full = len(pairs) == 4 and all(r.seeds_first == 3 and r.seeds_second == 3 for r in rows.itertuples())
    verdict['reference_recovery'] = {
        'status': 'complete' if full else 'partial', 'comparison': 'RustScenic, original rule minus current rule, pooled AP',
        'pairs': pairs, 'no_loss': bool(all(p['high'] >= 0 for p in pairs)) if full else None,
        'improves': bool(all(p['low'] > 0 for p in pairs)) if full else None}
    write_json(out / 'VERDICT.json', verdict)


if __name__ == '__main__':
    import sys
    summarise(Path(sys.argv[1]).resolve())
