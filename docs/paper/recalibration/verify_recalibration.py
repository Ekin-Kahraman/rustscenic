"""Check the deliverable from saved pair tables, independent R q-values and TF rows."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from analysis_paths import BASE, PACKAGE, AUDIT
checks = []


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def record(name, **details):
    checks.append(dict(check=name, status='passed', **details))


def near(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13, equal_nan=True)


def independent_ap(y, score):
    """Integrate precision at each distinct threshold, without sklearn/metrics.py."""
    thresholds, inverse = np.unique(score, return_inverse=True)
    positives = np.bincount(inverse, weights=y, minlength=len(thresholds))[::-1]
    sizes = np.bincount(inverse, minlength=len(thresholds))[::-1]
    precision = positives.cumsum()/sizes.cumsum()
    return float(np.dot(precision, positives)/y.sum())


labels = pd.read_csv(BASE/'summary/label-coverage.csv')
rankings = pd.read_csv(BASE/'summary/rankings-per-TF.csv')
coverage = pd.read_csv(BASE/'summary/coverage-per-TF.csv')
macros = pd.read_csv(BASE/'summary/macro-metrics.csv')
deltas = pd.read_csv(BASE/'summary/paired-differences.csv')

for dataset in ('k562', 'rpe1'):
    folder = BASE/'results'/dataset
    meta = json.loads((BASE/'inputs'/dataset/'metadata.json').read_text())
    assert sha(BASE/'RECALIBRATION-PLAN.md') == meta['plan_sha256']
    for name, expected in meta['files'].items():
        assert sha(BASE/'inputs'/dataset/name) == expected, name
    assert meta['source_aggregate_exact'] and meta['training_cells_disjoint']
    record(dataset+' input integrity', files=len(meta['files']))

    frames = [pd.read_csv(folder/f'calibration-{name}.csv')
              for name in ('median-a', 'validation', 'single', 'upper')]
    controls = pd.concat(frames, ignore_index=True)
    assert len(controls) == 26000
    assert not controls[['grna_target', 'response_id']].duplicated().any()
    assert controls.pass_qc.all() and np.isfinite(controls.p_value).all()
    assert not controls.significant.any()
    assert sha(folder/'calibration-median-a.csv') == sha(folder/'calibration-median-b.csv')
    summaries = pd.concat([pd.read_csv(folder/'calibration-summary.csv').query("run != 'median-b'"),
                           pd.read_csv(folder/'validation-summary.csv')])
    assert summaries['pass'].all() and summaries.n.sum() == 26000
    record(dataset+' calibration', distinct_pairs=len(controls), BH_discoveries=0)

    original = pd.read_csv(folder/'discovery.csv')
    corrected = pd.read_parquet(folder/'corrected-pair-results.parquet')
    pd.testing.assert_frame_equal(original, corrected[original.columns])
    q = np.fromfile(BASE/'verification'/f'{dataset}-r-bh.bin', dtype='<f8')
    near(corrected.q_value, q)
    assert not corrected[['grna_target', 'response_id']].duplicated().any()
    assert not corrected.grna_target.eq(corrected.response_id).any()
    eligible = corrected.pass_qc & np.isfinite(corrected.p_value) & np.isfinite(corrected.log_2_fold_change)
    assert (eligible == corrected.eligible).all()
    assert (corrected.response == (eligible & (q <= .05) & corrected.log_2_fold_change.abs().ge(.5))).all()
    assert not corrected.loc[~eligible, 'response'].any()
    primary = set((BASE/'inputs'/dataset/'primary-tfs.txt').read_text().splitlines())
    old_qc = json.loads((AUDIT/'returned/jobs'/f'{dataset}-777/evaluation/contrast-qc.json').read_text())
    metadata_roster = {entry['TF'] for entry in old_qc if len(entry['halves']) == 2 and
                      all(len(half['captures']) >= 3 and half['target_cells'] >= 25 for half in entry['halves'])}
    assert primary == metadata_roster  # Original roster is also recoverable using only metadata.
    for row in labels[labels.dataset.eq(dataset)].itertuples():
        a = corrected if row.roster == 'all' else corrected[corrected.grna_target.isin(primary)]
        response = a.eligible & a.q_value.le(.05) & a.log_2_fold_change.abs().ge(row.threshold)
        assert row.planned_pairs == len(a)
        assert row.eligible_pairs == a.eligible.sum()
        assert row.response_pairs == response.sum()
        assert row.significance_only_pairs == a.q_value.le(.05).sum()
        assert row.response_up + row.response_down == row.response_pairs
    record(dataset+' response labels', rows=len(corrected), R_BH_agreement=True)

    ranks = rankings[rankings.dataset.eq(dataset)]
    cov = coverage[coverage.dataset.eq(dataset)]
    assert set(ranks.seed) == {777, 778, 779}
    assert set(ranks.threshold) == {.25, .5, 1.}
    assert set(ranks.method) == {'original', 'diverse_importance', 'reference', 'native_raw',
                                 'reference_raw', 'full_pearson', 'uniform'}
    assert len(cov) == meta['all_target_TFs'] * 3 * 3 * 3
    assert (cov.groupby(['seed', 'threshold', 'method']).primary_roster.sum() == len(primary)).all()
    groups = ranks.groupby(['seed', 'threshold', 'TF'])
    assert groups.method.nunique().eq(7).all()
    assert groups.eligible_genes.nunique().eq(1).all() and groups.response_genes.nunique().eq(1).all()
    u = ranks[ranks.method.eq('uniform')]
    near(u.average_precision, u.response_genes/u.eligible_genes)
    assert u.predictions_at_50.eq(0).all() and u.emitted_hits_at_50.eq(0).all()
    assert (ranks.emitted_hits_at_50 <= ranks.predictions_at_50 + 1e-12).all()
    record(dataset+' scoring coverage and invariants', ranking_rows=len(ranks), coverage_rows=len(cov))

    # Fixed lexical selection, independent of the observed outcomes. Verify real
    # source networks with a separate AP implementation, including absent TFs.
    panel = json.loads((PACKAGE/'inputs'/dataset/'panel.json').read_text())
    genes = np.asarray(panel['genes'])
    selected = sorted(set(corrected.grna_target))[::max(1, len(set(corrected.grna_target))//16)]
    checked = 0
    for seed in [777, 778, 779]:
        job = AUDIT/'returned/jobs'/f'{dataset}-{seed}'
        networks = {name:pd.read_parquet(job/f'{name}/network.parquet') for name in ['native', 'reference']}
        regulons = {name:json.loads((job/path).read_text()) for name, path in [
            ('original', 'native/original.json'), ('diverse_importance', 'native/diverse_importance.json'),
            ('reference', 'reference/reference.json')]}
        with np.load(job/'native/pearson.npz') as p:
            rho = dict(zip(p['TFs'], p['rho']))
        for tf in selected:
            pair = corrected[corrected.grna_target.eq(tf)].set_index('response_id').reindex(genes)
            keep = pair.eligible.fillna(False).to_numpy(bool)
            weights = {name:frame[frame.TF.eq(tf)].set_index('target').importance.reindex(genes, fill_value=0).to_numpy()[keep]
                       for name, frame in networks.items()}
            scores = {'native_raw':weights['native'], 'reference_raw':weights['reference'],
                      'full_pearson':np.abs(rho[tf][keep]), 'uniform':np.zeros(keep.sum())}
            for name, regs in regulons.items():
                members = set(regs.get(tf+'_activator', [])) | set(regs.get(tf+'_repressor', []))
                member = np.asarray([gene in members and gene != tf for gene in genes[keep]])
                raw = weights['reference' if name == 'reference' else 'native']
                minimum = raw[raw > 0].min()/2 if (raw > 0).any() else 1.
                scores[name] = np.where(member, np.maximum(raw, np.where(raw > 0, raw, minimum)), 0.)
            for threshold in [.25, .5, 1.]:
                y = (pair.q_value.le(.05) & pair.log_2_fold_change.abs().ge(threshold)).to_numpy()[keep]
                if not 0 < y.sum() < len(y):
                    continue
                for method, score in scores.items():
                    row = ranks[ranks.seed.eq(seed) & ranks.TF.eq(tf) & ranks.threshold.eq(threshold) & ranks.method.eq(method)]
                    assert len(row) == 1
                    near(independent_ap(y, score), row.average_precision.iloc[0])
                    checked += 1
    record(dataset+' independently recomputed real-network AP sample', rows=checked, TFs=len(selected), selection='fixed lexical stride')

for row in macros.itertuples():
    a = rankings[rankings.dataset.eq(row.dataset) & rankings.threshold.eq(row.threshold)
                 & rankings.method.eq(row.method)]
    if row.roster != 'all':
        a = a[a.primary_roster]
    if row.roster == 'primary_adequate_knockdown':
        a = a[a.adequate_knockdown]
    tf = a.groupby('TF')
    assert len(tf) == row.TFs and tf.size().eq(3).all()
    near(row.average_precision, tf.average_precision.mean().mean())
    near(row.response_prevalence, tf.response_prevalence.mean().mean())
    near(row.emitted_hits50, tf.emitted_hits_at_50.mean().mean())
    near(row.expected_hits50, tf.expected_hits_at_50.mean().mean())
    near(row.predictions50, tf.predictions_at_50.mean().mean())
    precision = a.emitted_hits_at_50.sum()/a.predictions_at_50.sum() if a.predictions_at_50.sum() else np.nan
    near(row.pooled_emitted_precision, precision)
record('All macro summaries reconstructed', rows=len(macros))

for row in deltas.itertuples():
    a = rankings[rankings.dataset.eq(row.dataset) & rankings.threshold.eq(row.threshold)]
    if row.roster != 'all':
        a = a[a.primary_roster]
    if row.roster == 'primary_adequate_knockdown':
        a = a[a.adequate_knockdown]
    method, baseline = row.comparison.split(' minus ')
    x = a[a.method.eq(method)].groupby('TF').average_precision.mean()
    y = a[a.method.eq(baseline)].groupby('TF').average_precision.mean()
    assert x.index.equals(y.index) and len(x) == row.TFs
    difference = x.to_numpy() - y.to_numpy()
    near(row.mean_AP_difference, difference.mean())
    # Sequential draws independently reproduce the vectorised implementation.
    rng = np.random.default_rng(20261112)
    replicates = [difference[rng.integers(len(difference), size=len(difference))].mean() for _ in range(5000)]
    near([row.TF_bootstrap_low, row.TF_bootstrap_high], np.quantile(replicates, [.025, .975]))
record('All paired differences and bootstrap intervals reconstructed', rows=len(deltas))

# Reuse the independently reproduced activity rows; validate their saved artefacts.
replay = json.loads((AUDIT/'REPLAY-RECEIPT.json').read_text())
assert replay['status'] == 'passed' and replay['summary_tables_reproduced']
for job in replay['jobs']:
    a = pd.read_csv(AUDIT/'replay/jobs'/job['job']/'evaluation/activity.csv')
    b = pd.read_csv(AUDIT/'returned/jobs'/job['job']/'evaluation/activity.csv')
    pd.testing.assert_frame_equal(a, b, rtol=1e-10, atol=1e-12)
record('Prior activity replay retained', rows=2457, scope=replay['activity_scope'])

receipt = dict(status='passed', checks=checks, model_refits=0,
               scripts={name:sha(BASE/name) for name in ['analysis_paths.py', 'prepare_counts.py', 'recalibrate.R',
                   'validate_distinct.R', 'corrected_evaluation.py', 'score_saved_networks.py',
                   'verify_bh.R', 'verify_recalibration.py']})
(BASE/'VERIFICATION.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
