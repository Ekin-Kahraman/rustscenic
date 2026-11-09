"""Report numerical agreement, biological gaps and paired scoring costs."""
import csv
import statistics
import numpy as np
import pandas as pd
from common import ROOT, read_json, write_json
from utility import paired_summary


def summarise(work, allow_smoke=False):
    from run import ARMS, valid, equal_scores
    required = [work/'biology'/arm for arm in ARMS]
    required += [work/'atlas'/str(seed)/arm for seed in (777,778,779) for arm in ARMS]
    required += [work/'probe'/str(seed)/arm for seed in (777,778,779) for arm in ('seeded','reference')]
    missing = [str(p.relative_to(work)) for p in required if not valid(p,allow_smoke)]
    numerical, performance, biological = [], [], {}
    for arm in ARMS:
        out = work/'biology'/arm
        if valid(out,allow_smoke):
            obj = read_json(out/'verification.json')
            numerical.append({'test':'external-'+arm,'passed':len(obj['records'])==12 and obj['numerical_pass']})
    out = work/'biology/seeded'
    if valid(out,allow_smoke):
        records = read_json(out/'verification.json')['records']
        for n in (5000,20000):
            seeded = [r['result'] for r in records if r['training_cells']==n and r['mode']=='signed']
            union = [r['result'] for r in records if r['training_cells']==n and r['mode']=='union']
            original = [read_json(ROOT/'expected'/f'{n}-{seed}'/'assessment.json')['results'] for seed in (777,778,779)]
            biological[str(n)] = {
                'seeded_vs_original_native_signed':paired_summary(seeded,[o['rustscenic-signed-native-common'] for o in original]),
                'seeded_vs_same_network_shared_scorer':paired_summary(seeded,[o['rustscenic-signed-common'] for o in original]),
                'seeded_vs_scenicplus_networks':paired_summary(seeded,[o['reference-signed-common'] for o in original]),
                'signed_vs_union':paired_summary(seeded,union),
            }
        converged = all(r['classifier_converged'] for r in records)
    else:
        converged = False
    for seed in (777,778,779):
        probe = work/'probe'/str(seed)
        if all(valid(probe/arm,allow_smoke) for arm in ('seeded','reference')):
            numerical.append({'test':f'atlas-probe-{seed}',**equal_scores(probe/'seeded/full-scores.h5',probe/'reference/full-scores.h5',1e-7)})
        atlas = work/'atlas'/str(seed)
        if all(valid(atlas/arm,allow_smoke) for arm in ARMS):
            numerical.append({'test':f'whole-atlas-legacy-{seed}',**equal_scores(atlas/'baseline/full-scores.h5',atlas/'unseeded/full-scores.h5',0.)})
            if valid(probe/'reference',allow_smoke):
                left = pd.read_parquet(atlas/'seeded/sampled-scores.parquet')
                right = pd.read_parquet(probe/'reference/sampled-scores.parquet')
                if not left.index.equals(right.index) or not left.columns.equals(right.columns):
                    raise ValueError('Atlas reference sample axes differ')
                error = float(np.max(np.abs(left.to_numpy()-right.to_numpy())))
                numerical.append({'test':f'full-run-sample-{seed}','max_absolute_error':error,'passed':bool(np.isfinite(error) and error<=1e-7)})
        for arm in ARMS:
            out = atlas/arm
            if not valid(out,allow_smoke):
                continue
            score = read_json(out/'score-summary.json')
            monitor = read_json(out/'monitor.json')
            env = read_json(out/'environment.json')
            performance.append({'seed':seed,'arm':arm,'host':env['host'],'cells':score['cells'],
                'input_signatures':score['input_signatures'],'features':score['tf_features'],
                'scoring_seconds':score['timings']['score_seconds']+score['timings']['combine_seconds'],
                'stage_elapsed_seconds':score['stage_elapsed_seconds'],
                **{k:monitor[k] for k in ('process_wall_s','peak_tree_pss_mb','peak_summed_rss_mb','pss_incomplete_samples',
                  'analysis_peak_pss_mb','analysis_memory_samples','analysis_pss_incomplete_samples')}})
    write_json(work/'numerical-checks.json',numerical)
    write_json(work/'biological-summary.json',biological)
    if performance:
        with (work/'scoring-performance.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(performance[0]))
            writer.writeheader()
            writer.writerows(performance)
    ratios = {}
    memory_valid = len(performance)==9 and all(r['peak_tree_pss_mb'] and r['analysis_peak_pss_mb'] and
        not r['pss_incomplete_samples'] and not r['analysis_pss_incomplete_samples'] and r['analysis_memory_samples']>=3 for r in performance)
    same_node = len({r['host'] for r in performance})==1
    if len(performance)==9 and memory_valid and same_node:
        for baseline in ('baseline','unseeded'):
            pairs = []
            for seed in (777,778,779):
                old = next(r for r in performance if r['seed']==seed and r['arm']==baseline)
                new = next(r for r in performance if r['seed']==seed and r['arm']=='seeded')
                if any(old[k]!=new[k] for k in ('cells','input_signatures','features')):
                    raise ValueError('Unmatched performance comparison')
                pairs.append({'seed':seed,'scoring_time_ratio':new['scoring_seconds']/old['scoring_seconds'],
                              'process_pss_ratio':new['peak_tree_pss_mb']/old['peak_tree_pss_mb'],
                              'analysis_pss_ratio':new['analysis_peak_pss_mb']/old['analysis_peak_pss_mb']})
            ratios[baseline] = {'pairs':pairs,**{f'median_{k}':statistics.median(r[k] for r in pairs)
                for k in ('scoring_time_ratio','process_pss_ratio','analysis_pss_ratio')}}
    regression = any(r[k]>1.1 for r in ratios.values() for k in ('median_scoring_time_ratio','median_process_pss_ratio'))
    numerical_pass = len(numerical)==12 and all(r['passed'] for r in numerical)
    write_json(work/'SUMMARY.json',{'execution_complete':not missing,'missing':missing,'numerical_agreement_pass':numerical_pass,
        'classifier_convergence_pass':converged,'memory_measurements_complete':memory_valid,'same_node':same_node,
        'paired_candidate_over_baseline_ratios':ratios,'regression_review_flag_over_10_percent':regression if ratios else None,
        'full_atlas_cells':817740,'smoke':allow_smoke,
        'scope':'Scoring-only confirmation. No new network inference or full-pipeline speedup claim.',
        'biology':'Compatibility confirmation on reused data; report remaining method differences. Not evidence of biological equivalence.',
        'performance':'Same frozen signatures, 16 CPUs, fresh processes. Three network seeds; each arm occupies each order position once. Ratios above 1 mean higher cost. 10% is an operational review flag, not a statistical test.'})
