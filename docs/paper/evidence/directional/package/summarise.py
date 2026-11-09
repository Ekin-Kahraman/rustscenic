"""Complete results and adverse outcomes retained; no biological pass threshold."""
import csv
import statistics
from common import ROOT, read_json, write_json
from utility import paired_summary


def summarise(work, allow_smoke=False):
    from run import valid
    missing = []
    biological = {}
    records = []
    if valid(work/'biology',allow_smoke):
        records = read_json(work/'biology/verification.json')['records']
        for n in (5000,20000):
            by_method = {method:[r for r in records if r['training_cells']==n and r['method']==method]
                         for method in ('rustscenic','reference')}
            for method, rows in by_method.items():
                if sorted(r['seed'] for r in rows)!=[777,778,779]:
                    raise ValueError('Missing or duplicated biology seeds: '+method)
            rust = [r['results']['directional'] for r in by_method['rustscenic']]
            old = [r['results']['contrast'] for r in by_method['rustscenic']]
            reference = [r['results']['directional'] for r in by_method['reference']]
            biological[str(n)] = dict(directional_vs_contrast=paired_summary(rust,old),
                directional_vs_scenicplus=paired_summary(rust,reference),
                seed_means=[dict(seed=r['seed'],method=r['method'],mode=mode,
                    macro_f1=statistics.mean(f['macro_f1'] for f in result['folds']))
                    for method in by_method for r in by_method[method] for mode,result in r['results'].items()])
    else: missing.append('biology')
    performance = []
    numerical = []
    for seed in (777,778,779):
        for arm in ('union','directional'):
            out = work/'atlas'/str(seed)/arm
            if not valid(out,allow_smoke):
                missing.append(str(out.relative_to(work)))
                continue
            score = read_json(out/'score-summary.json')
            mon = read_json(out/'monitor.json')
            env = read_json(out/'environment.json')
            network = read_json(out/'network.json')
            performance.append(dict(seed=seed,arm=arm,host=env['host'],cells=score['cells'],
                genes=score['genes'],features=score['features'],network_sha256=network['sha256'],
                signature_sha256=score['signature_sha256'],output_array_bytes=score['output_array_bytes'],
                full_scores_bytes=score['full_scores_bytes'],scoring_seconds=score['timings']['score_seconds'],
                stage_elapsed_seconds=score['stage_elapsed_seconds'],
                **{k:mon[k] for k in ('process_wall_s','peak_tree_pss_mb','peak_summed_rss_mb',
                    'analysis_peak_pss_mb','pss_incomplete_samples','analysis_pss_incomplete_samples','analysis_memory_samples')}))
            if arm=='directional' and not allow_smoke:
                numerical.append(dict(seed=seed,**read_json(out/'prior-sample-check.json')))
    write_json(work/'biological-summary.json',biological)
    if performance:
        with (work/'scoring-performance.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(performance[0]))
            writer.writeheader(); writer.writerows(performance)
    same_node = len(performance)==6 and len({r['host'] for r in performance})==1
    memory_valid = len(performance)==6 and all(r['peak_tree_pss_mb'] and r['analysis_peak_pss_mb']
        and not r['pss_incomplete_samples'] and not r['analysis_pss_incomplete_samples']
        and r['analysis_memory_samples']>=3 for r in performance)
    pairs = []
    if same_node and memory_valid:
        for seed in (777,778,779):
            old = next(r for r in performance if r['seed']==seed and r['arm']=='union')
            new = next(r for r in performance if r['seed']==seed and r['arm']=='directional')
            if any(old[k]!=new[k] for k in ('cells','genes','network_sha256')):
                raise ValueError('Unmatched performance inputs')
            pairs.append(dict(seed=seed,**{k+'_ratio':new[k]/old[k] for k in
                ('scoring_seconds','stage_elapsed_seconds','peak_tree_pss_mb','analysis_peak_pss_mb','output_array_bytes')}))
    medians = {k:statistics.median(r[k] for r in pairs) for k in pairs[0] if k!='seed'} if pairs else {}
    good_folds = len(records)==12 and all(r['classifier_converged'] for r in records)
    write_json(work/'SUMMARY.json',dict(execution_complete=not missing,missing=missing,smoke=allow_smoke,
        numerical_agreement_pass=len(records)==12 and all(r['passed'] for r in records)
            and (allow_smoke or len(numerical)==3 and all(r['passed'] for r in numerical)),
        classifier_convergence_pass=good_folds,atlas_numerical_checks=numerical,
        memory_measurements_complete=memory_valid,same_node=same_node,
        pairs=pairs,median_ratios=medians,
        cost_review_flag_over_10_percent=(any(medians[k]>1.1 for k in
            ('scoring_seconds_ratio','stage_elapsed_seconds_ratio','peak_tree_pss_mb_ratio')) if medians else None),
        scope='Independent execution on reused networks/cohorts; gene-activity scoring on RNA only. No inference, ATAC imputation, motif fitting or new full-workflow benchmark.',
        biology='Report both endpoints, all seeds/donors/classes and intervals. No retrospective biological equivalence or superiority pass threshold.',
        performance='Three paired network seeds, two fresh-process modes, alternating order, fixed16CPUs. Ratios>1 mean increased cost;10% is an operational review flag, not a statistical test.'))
