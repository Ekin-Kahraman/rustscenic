"""Independent audit and analysis of the stopping-rule SEND_BACK from Minerva.

Follows the return-audit plan fixed before the run: integrity and completeness first, then every
confirmatory check and the recovery rule recomputed from the raw per-arm files
with code written separately from the package's summarise.py and recovery.py,
then the descriptive endpoints, a figure and a short report.

Usage (from any directory; never from inside the extracted data):
    python3 -I analyse_return.py SEND_BACK.tar.gz OUT_DIR      # or a folder of .part files + SEND_BACK.parts.json
    python3 -I analyse_return.py --extracted DIR OUT_DIR        # already extracted and verified
"""
import argparse
import hashlib
import json
import tarfile
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PACKAGE = Path(__file__).resolve().parents[1] / 'rustscenic-stopping-rule'
SEEDS = (777, 778, 779)
FITTED = ('native_cross', 'native_same', 'reference_cross', 'reference_original')
RULE_PAIRS = {'RustScenic': ('native_cross', 'native_same'), 'pySCENIC': ('reference_cross', 'reference_original')}
CONFIRMATORY = ('bmmc', 'atlas20000')
EXPECTED_JOBS = ['bmmc-777', 'bmmc-778', 'bmmc-779', 'atlas20000-777', 'atlas20000-778', 'atlas20000-779', 'atlas100000-777',
                 'atlas100000-778', 'atlas100000-779', 'atlas500000-777', 'atlas1306127-777', 'hdma817740-777']
EXPECTED_ARMS = 44
BARS = {'trees': 3.0, 'stability': 1.5, 'fidelity': 0.90}
BOOT_DRAWS, BOOT_SEED = 2000, 20261113
REFERENCES = {'human': ('human-nonspecific-chip', 'human-string'), 'mouse': ('mouse-nonspecific-chip', 'mouse-string')}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- integrity
def receive(source, out):
    """Verify transport hashes and extract into a new, empty folder."""
    source, notes = Path(source), {}
    if source.is_dir():
        manifest = read_json(source / 'SEND_BACK.parts.json')
        parts = [source / p['name'] for p in manifest['parts']]
        notes['parts'] = [{'name': p.name, 'ok': sha256(p) == e['sha256']} for p, e in zip(parts, manifest['parts'])]
        archive = out / 'SEND_BACK.tar.gz'
        with archive.open('wb') as w:
            for p in parts:
                w.write(p.read_bytes())
        expected = manifest['archive_sha256']
    else:
        archive = source
        sidecar = source.parent / 'SEND_BACK.sha256'
        expected = sidecar.read_text().split()[0] if sidecar.exists() else None
    notes['archive_sha256'] = sha256(archive)
    notes['transport_ok'] = expected is None or notes['archive_sha256'] == expected
    if expected is None:
        notes['transport_ok'] = 'no .sha256 found beside the archive'
    target = out / 'extracted'
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if m.name.startswith(('/', '..')) or '/../' in m.name or not (m.isfile() or m.isdir()):
                raise SystemExit(f'refusing unsafe archive member {m.name}')
        t.extractall(target, filter='data')
    return target, notes


def integrity(root):
    report = {}
    inventory = read_json(root / 'OUTPUT-INVENTORY.json')
    bad = [e['path'] for e in inventory if not (root / e['path']).exists() or sha256(root / e['path']) != e['sha256']]
    report['inventory'] = {'files': len(inventory), 'mismatched_or_missing': bad}
    sent = dict(line.split()[::-1] for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines() if line.strip())
    back = dict(line.split()[::-1] for line in (root / 'SHA256SUMS').read_text().splitlines() if line.strip()) \
        if (root / 'SHA256SUMS').exists() else {}
    report['package_identical_to_sent'] = back == sent
    report['package_files_differing'] = sorted(k for k in set(sent) | set(back) if sent.get(k) != back.get(k))
    report['protocol_sha256'] = sha256(root / 'PROTOCOL.md') if (root / 'PROTOCOL.md').exists() else None
    report['protocol_matches_sent'] = report['protocol_sha256'] == sha256(PACKAGE / 'PROTOCOL.md')
    arm_hash_failures = []
    for summary in sorted(root.glob('jobs/*/*/SUMMARY.json')):
        for name, digest in read_json(summary).get('outputs', {}).items():
            path = summary.parent / name
            if path.exists() and sha256(path) != digest:
                arm_hash_failures.append(str(path.relative_to(root)))
    report['arm_output_hash_failures'] = arm_hash_failures
    return report


def completeness(root):
    status = read_json(root / 'RUN-STATUS.json') if (root / 'RUN-STATUS.json').exists() else {}
    jobs = {j: status.get('jobs', {}).get(j, {}).get('status', 'missing') for j in EXPECTED_JOBS}
    arms, errors = {}, {}
    for summary in sorted(root.glob('jobs/*/*/SUMMARY.json')):
        s = read_json(summary)
        key = f'{summary.parent.parent.name}/{summary.parent.name}'
        arms[key] = s.get('status')
        if s.get('stage_errors'):
            errors[key] = s['stage_errors']
    pre = read_json(root / 'preflight.json') if (root / 'preflight.json').exists() else {}
    return {'jobs': jobs, 'jobs_complete': sum(v == 'complete' for v in jobs.values()), 'arms_complete': sum(v == 'complete' for v in arms.values()),
            'arms_expected': EXPECTED_ARMS, 'arm_status': arms, 'stage_errors': errors,
            'reference_original_used': pre.get('reference_original'), 'preflight_native_ok': pre.get('native', {}).get('ok')}


# ---------------------------------------------------------------- loading
def arms_available(root):
    found = {}
    for summary in root.glob('jobs/*/*/SUMMARY.json'):
        s = read_json(summary)
        if s.get('status') == 'complete' and (summary.parent / 'network.parquet').exists():
            found[s['dataset'], s['arm'], int(s['seed'])] = summary.parent
    return found


def top50_positive(folder):
    """Top 50 targets per TF by positive importance, ties broken by target name, from the full network."""
    net = pd.read_parquet(folder / 'network.parquet', columns=['TF', 'target', 'importance'])
    net = net.loc[net.importance > 0].astype({'TF': str, 'target': str})
    net = net.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True])
    return {tf: frozenset(g.target.iloc[:50]) for tf, g in net.groupby('TF', sort=False)}


def jaccard_mean(a, b):
    tfs = set(a) | set(b)
    if not tfs:
        return np.nan
    return float(np.mean([len(a.get(t, frozenset()) & b.get(t, frozenset())) / len(a.get(t, frozenset()) | b.get(t, frozenset()))
                          for t in tfs]))


# ---------------------------------------------------------------- checks
def recompute_checks(found):
    trees = {k: pd.read_csv(f / 'trees.csv').trees.mean() for k, f in found.items() if (f / 'trees.csv').exists()}
    tops = {k: top50_positive(f) for k, f in found.items() if k[1] in FITTED}
    rows, scale = [], []
    datasets = sorted({k[0] for k in found})
    for d in datasets:
        def seeds(arm):
            return [s for s in SEEDS if (d, arm, s) in tops]

        within = {arm: np.mean([jaccard_mean(tops[d, arm, a], tops[d, arm, b]) for a, b in combinations(seeds(arm), 2)])
                  if len(seeds(arm)) >= 2 else np.nan for arm in FITTED}
        mean_trees = {arm: np.mean([trees[d, arm, s] for s in SEEDS if (d, arm, s) in trees]) if any((d, arm, s) in trees for s in SEEDS)
                      else np.nan for arm in FITTED}
        for tool, (cross, same) in RULE_PAIRS.items():
            rows.append({'dataset': d, 'check': 'trees', 'subject': tool, 'observed': mean_trees[same] / mean_trees[cross],
                         'current': mean_trees[cross], 'original': mean_trees[same]})
            rows.append({'dataset': d, 'check': 'stability', 'subject': tool, 'observed': within[same] / within[cross],
                         'current': within[cross], 'original': within[same]})
        for rule, (native, ref) in {'current': ('native_cross', 'reference_cross'), 'original': ('native_same', 'reference_original')}.items():
            between = [jaccard_mean(tops[d, native, a], tops[d, ref, b]) for a, b in product(seeds(native), seeds(ref))]
            rows.append({'dataset': d, 'check': 'fidelity', 'subject': f'{rule} rule',
                         'observed': np.mean(between) / within[ref] if between and np.isfinite(within[ref]) else np.nan,
                         'current': np.mean(between) if between else np.nan, 'original': within[ref]})
        for arm in FITTED:
            for s in SEEDS:
                if (d, arm, s) in found:
                    sm = read_json(found[d, arm, s] / 'SUMMARY.json')
                    mon = read_json(found[d, arm, s] / 'monitor.json') if (found[d, arm, s] / 'monitor.json').exists() else {}
                    scale.append({'dataset': d, 'arm': arm, 'seed': s, 'cells': sm['cells'], 'genes': sm['genes'], 'TFs': sm['TFs'],
                                  'trees_mean': trees.get((d, arm, s)), 'trees_median': sm.get('trees_median'),
                                  'share_at_ceiling': sm.get('share_at_ceiling'), 'inference_s': sm['inference_s'],
                                  'peak_pss_mb': mon.get('peak_tree_pss_mb'), 'seed_stability': within[arm]})
    checks = pd.DataFrame(rows)
    checks['confirmatory'] = checks.dataset.isin(CONFIRMATORY)
    checks['bar'] = checks.check.map(BARS)
    checks['result'] = np.where(~checks.observed.apply(np.isfinite), 'not evaluable',
                                np.where(checks.observed >= checks.bar, 'met', 'not met'))
    checks.loc[~checks.confirmatory, 'result'] = 'descriptive'
    return checks, pd.DataFrame(scale)


# ---------------------------------------------------------------- recovery
def universe(genes, tfs, ref):
    names = pd.Index([g.upper() for g in genes])
    keep = names[~names.duplicated(keep=False)]
    col = pd.Series(np.arange(len(keep)), index=keep)
    panel_tfs = set(t.upper() for t in tfs) & set(keep)
    ref = ref.assign(TF=ref.TF.str.upper(), target=ref.target.str.upper())
    links = ref.loc[ref.TF.isin(panel_tfs) & ref.target.isin(keep) & (ref.TF != ref.target)].drop_duplicates()
    rtf = sorted(links.TF.unique())
    row = pd.Series(np.arange(len(rtf)), index=rtf)
    y = np.zeros((len(rtf), len(keep)), bool)
    y[row[links.TF].to_numpy(), col[links.target].to_numpy()] = True
    mask = np.ones_like(y)
    mask[np.arange(len(rtf)), col[rtf].to_numpy()] = False
    return row, col, y, mask


def score_matrix(net, row, col, shape):
    tf, tg = net.TF.astype(str).str.upper(), net.target.astype(str).str.upper()
    ok = tf.isin(row.index) & tg.isin(col.index)
    m = np.zeros(shape)
    m[row[tf[ok]].to_numpy(), col[tg[ok]].to_numpy()] = net.importance.to_numpy(float)[ok.to_numpy()]
    return m


def weighted_ap(group, pos_w, all_w):
    """AP over tie groups: sum_g (positive weight in g / all positive weight) * cumulative precision at g."""
    p = np.bincount(group, weights=pos_w)
    a = np.bincount(group, weights=all_w)
    cp, ca = np.cumsum(p), np.cumsum(a)
    prec = np.divide(cp, ca, out=np.zeros_like(cp), where=ca > 0)
    return float(np.sum(p / cp[-1] * prec))


def recovery(found, dataset, draws=BOOT_DRAWS):
    keys = [k for k in found if k[0] == dataset]
    if not keys:
        return pd.DataFrame(), pd.DataFrame()
    panel = read_json(found[keys[0]] / 'panel.json')
    species = 'mouse' if dataset.startswith(('atlas',)) else 'human'
    points, diffs = [], []
    for name in REFERENCES[species]:
        ref = pd.read_csv(PACKAGE / 'resources/references' / f'{name}.tsv.gz', sep='\t', dtype=str)
        row, col, y, mask = universe(panel['genes'], panel['tfs'], ref)
        if len(row) < 2 or y.sum() < 20:
            continue
        flat_y, flat_tf = y[mask], np.broadcast_to(np.arange(len(row))[:, None], y.shape)[mask]
        prepared = {}
        for k in sorted(keys):
            s = score_matrix(pd.read_parquet(found[k] / 'network.parquet', columns=['TF', 'target', 'importance']), row, col, y.shape)[mask]
            ap = average_precision_score(flat_y, s) if flat_y.any() else np.nan
            points.append({'dataset': dataset, 'reference': name, 'arm': k[1], 'seed': k[2], 'AP': ap,
                           'prevalence': flat_y.mean(), 'reference_TFs': len(row), 'eligible_links': int(flat_y.sum())})
            uniq, group = np.unique(-s, return_inverse=True)
            prepared[k] = (group, flat_y.astype(float), flat_tf)
        rng = np.random.default_rng(BOOT_SEED)
        boot = {k: np.empty(draws) for k in prepared}
        for d in range(draws):
            counts = np.bincount(rng.integers(0, len(row), size=len(row)), minlength=len(row)).astype(float)
            for k, (group, yy, tf) in prepared.items():
                w = counts[tf]
                boot[k][d] = weighted_ap(group, yy * w, w)
        pt = pd.DataFrame(points)
        pt = pt.loc[pt.reference.eq(name)]
        for a, b in (('native_same', 'native_cross'), ('reference_original', 'reference_cross'), ('native_same', 'pearson'),
                     ('native_cross', 'pearson'), ('native_same', 'reference_original'), ('native_cross', 'reference_cross')):
            ka, kb = [k for k in prepared if k[1] == a], [k for k in prepared if k[1] == b]
            if not ka or not kb:
                continue
            dist = np.mean([boot[k] for k in ka], 0) - np.mean([boot[k] for k in kb], 0)
            lo, hi = np.quantile(dist, [.025, .975])
            diffs.append({'dataset': dataset, 'reference': name, 'comparison': f'{a} minus {b}',
                          'AP_difference': pt.loc[pt.arm.eq(a), 'AP'].mean() - pt.loc[pt.arm.eq(b), 'AP'].mean(),
                          'low': lo, 'high': hi, 'seeds_first': len(ka), 'seeds_second': len(kb), 'draws': draws})
    return pd.DataFrame(points), pd.DataFrame(diffs)


# ---------------------------------------------------------------- comparison with returned tables
def compare_with_returned(root, checks, diffs):
    verdict_path = root / 'summary/VERDICT.json'
    if not verdict_path.exists():
        return {'returned_verdict': 'missing'}
    v = read_json(verdict_path)
    name = {'1': 'trees', '2': 'stability', '3': 'fidelity'}
    out = []
    for c in v['checks']:
        mine = checks.loc[checks.dataset.eq(c['dataset']) & checks.check.eq(name[c['endpoint'][0]]) & checks.subject.eq(c['subject'])]
        if len(mine):
            o = mine.observed.iloc[0]
            out.append({'dataset': c['dataset'], 'check': name[c['endpoint'][0]], 'subject': c['subject'], 'returned': c['observed'],
                        'recomputed': o, 'abs_difference': None if c['observed'] is None else abs(c['observed'] - o)})
    rec = []
    returned = pd.read_csv(root / 'summary/recovery-differences.csv') if (root / 'summary/recovery-differences.csv').exists() else pd.DataFrame()
    for r in diffs.itertuples():
        m = returned.loc[returned.dataset.eq(r.dataset) & returned.reference.eq(r.reference) & returned.comparison.eq(r.comparison)] if len(returned) else []
        if len(m):
            rec.append({'dataset': r.dataset, 'reference': r.reference, 'comparison': r.comparison,
                        'max_abs_difference': float(np.max(np.abs(m[['AP_difference', 'low', 'high']].to_numpy()[0] - [r.AP_difference, r.low, r.high])))})
    return {'checks': out, 'recovery': rec, 'returned_all_met': v.get('all_confirmatory_checks_met'),
            'returned_reference_recovery': v.get('reference_recovery')}


# ---------------------------------------------------------------- figure
def figure(scale, checks, diffs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colour = {'native_cross': '#9aa5b1', 'native_same': '#1f5f99', 'reference_cross': '#d9a066', 'reference_original': '#b5531b'}
    label = {'native_cross': 'RustScenic, current rule', 'native_same': 'RustScenic, original rule',
             'reference_cross': 'pySCENIC, current rule', 'reference_original': 'pySCENIC, original rule'}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    per = scale.groupby(['dataset', 'arm']).agg(cells=('cells', 'first'), trees=('trees_mean', 'mean'), seconds=('inference_s', 'mean'),
                                                stability=('seed_stability', 'first')).reset_index()
    atlas = per.loc[per.dataset.str.startswith('atlas')].sort_values('cells')
    for arm in FITTED:
        a = atlas.loc[atlas.arm.eq(arm)]
        if len(a):
            ax[0].plot(a.cells, a.trees, 'o-', color=colour[arm], label=label[arm], lw=2, ms=6)
            ax[1].plot(a.cells, a.seconds / 3600, 'o-', color=colour[arm], lw=2, ms=6)
    for i, title, ylab in ((0, 'Training length (mouse brain atlas)', 'Mean trees per target'),
                           (1, 'Gene-network inference time (16 CPUs)', 'Hours')):
        ax[i].set_xscale('log'); ax[i].set_yscale('log'); ax[i].set_xlabel('Cells'); ax[i].set_ylabel(ylab); ax[i].set_title(title, loc='left', fontsize=10)
        ax[i].grid(alpha=.25, lw=.5)
    ax[0].legend(frameon=False, fontsize=8)
    rec = diffs.loc[diffs.comparison.eq('native_same minus native_cross') & diffs.dataset.isin(CONFIRMATORY)]
    for j, r in enumerate(rec.itertuples()):
        ax[2].errorbar(r.AP_difference, j, xerr=[[r.AP_difference - r.low], [r.high - r.AP_difference]], fmt='o', color='#1f5f99', capsize=3)
    ax[2].axvline(0, color='#555', lw=1)
    ax[2].set_yticks(range(len(rec)), [f'{r.dataset}, {r.reference.split("-", 1)[1]}' for r in rec.itertuples()], fontsize=8)
    ax[2].set_xlabel('AP difference, original minus current rule')
    ax[2].set_title('Broad reference recovery, RustScenic (95% CI)', loc='left', fontsize=10)
    fig.savefig(path, dpi=200)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', nargs='?')
    p.add_argument('out', type=Path)
    p.add_argument('--extracted', type=Path)
    p.add_argument('--draws', type=int, default=BOOT_DRAWS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    if a.extracted:
        root = a.extracted
    else:
        root, report['transport'] = receive(a.source, a.out)
    report['integrity'] = integrity(root)
    report['completeness'] = completeness(root)
    found = arms_available(root)
    checks, scale = recompute_checks(found)
    checks.to_csv(a.out / 'checks-recomputed.csv', index=False)
    scale.to_csv(a.out / 'scale-recomputed.csv', index=False)
    points, diffs = [], []
    for d in sorted({k[0] for k in found}):
        pt, df = recovery(found, d, draws=a.draws if d in CONFIRMATORY else 200)
        points.append(pt); diffs.append(df)
    points = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    diffs = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame()
    points.to_csv(a.out / 'recovery-points-recomputed.csv', index=False)
    diffs.to_csv(a.out / 'recovery-differences-recomputed.csv', index=False)
    rule = diffs.loc[diffs.comparison.eq('native_same minus native_cross') & diffs.dataset.isin(CONFIRMATORY)]
    full = len(rule) == 4 and (rule.seeds_first == 3).all() and (rule.seeds_second == 3).all()
    confirm = checks.loc[checks.confirmatory]
    report['confirmatory'] = {
        'checks': len(confirm), 'met': int(confirm.result.eq('met').sum()), 'not_met': int(confirm.result.eq('not met').sum()),
        'not_evaluable': int(confirm.result.eq('not evaluable').sum()),
        'no_loss_rule': bool((rule.high >= 0).all()) if full else 'not evaluable',
        'improves_rule': bool((rule.low > 0).all()) if full else 'not evaluable'}
    report['agreement_with_returned_summary'] = compare_with_returned(root, checks, diffs)
    (a.out / 'AUDIT.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    if len(scale):
        figure(scale, checks, diffs, a.out / 'stopping-rule-at-scale.png')
    print(json.dumps({k: report[k] for k in ('confirmatory',)}, indent=2, default=float))
    print(checks.loc[checks.confirmatory, ['dataset', 'check', 'subject', 'observed', 'result']].to_string(index=False))


if __name__ == '__main__':
    main()
