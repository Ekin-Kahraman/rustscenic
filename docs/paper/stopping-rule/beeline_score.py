"""Score BEELINE arms: stability, reference recovery, cross-implementation fidelity.

Metrics are the ones already used in the manuscript (reference_metrics.py from the
6 November package): fixed candidate universe, pooled average precision, tie-aware
top-50 hits. Uncertainty for AP differences resamples reference TFs.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from multiprocessing import get_context

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'RustScenic-Paper-Validation-20261106/rustscenic-paper-validation'))
from reference_metrics import reference_metrics  # noqa: E402

BOOT = 2000
SEED = 20261112


def top_sets(net, tfs, k=50):
    net = net[(net.importance > 0) & (net.TF != net.target)]
    ranked = net.sort_values(['TF', 'importance', 'target'], ascending=[True, False, True], kind='stable')
    found = {tf: set(g.target.head(k)) for tf, g in ranked.groupby('TF', sort=False)}
    return {tf: found.get(tf, set()) for tf in tfs}


def mean_jaccard(a, b, tfs):
    values = [len(a[t] & b[t]) / len(a[t] | b[t]) for t in tfs if a[t] | b[t]]
    return float(np.mean(values)) if values else np.nan


def pearson_network(x, genes, tfs):
    dense = np.asarray(x.todense(), dtype=np.float64)
    sd = dense.std(0)
    z = (dense - dense.mean(0)) / np.where(sd > 0, sd, 1)
    rho = np.abs(z.T @ z / len(z))
    idx = {g: i for i, g in enumerate(genes)}
    rows = [(tf, g, rho[idx[tf], idx[g]]) for tf in tfs for g in genes if tf != g]
    return pd.DataFrame(rows, columns=['TF', 'target', 'importance'])


def tf_blocks(network, truth, genes, tfs):
    """Per-reference-TF (labels, scores) so pooled AP can be bootstrapped over TFs."""
    known = {(a, b) for a, b in truth[['TF', 'target']].itertuples(index=False, name=None) if a in tfs and b in genes and a != b}
    supported = sorted({tf for tf, _ in known})
    scores = network.set_index(['TF', 'target']).importance
    blocks = {}
    for tf in supported:
        targets = [g for g in genes if g != tf]
        s = scores.reindex(pd.MultiIndex.from_product([[tf], targets]), fill_value=0).to_numpy(float)
        blocks[tf] = (np.array([(tf, g) in known for g in targets], dtype=np.int8), s)
    return blocks


def fast_ap(y, s):
    """average_precision_score (tie-aware step integral) without sklearn's overhead."""
    order = np.argsort(-s, kind='stable')
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each tied score group
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def pooled_ap(blocks, order):
    return fast_ap(np.concatenate([blocks[t][0] for t in order]), np.concatenate([blocks[t][1] for t in order]))


_BLOCKS = None


def _draw_aps(job):
    keys, draw = job
    return [pooled_ap(_BLOCKS[k], draw) for k in keys]


def main(inputs, runs, datasets, arms, seeds, out, bootstrap_references=99, jobs=8):
    out.mkdir(parents=True, exist_ok=True)
    recovery, stability, fidelity, cost, differences = [], [], [], [], []
    for name in datasets:
        info = json.loads((inputs / name / 'input.json').read_text())
        genes, tfs = info['genes'], info['tfs']
        geneset, tfset = set(genes), set(tfs)
        x = sparse.load_npz(inputs / name / 'matrix.npz').tocsr()
        nets = {(arm, seed): pd.read_parquet(runs / name / f'{arm}-{seed}.parquet') for arm in arms for seed in seeds
                if (runs / name / f'{arm}-{seed}.parquet').exists()}
        present = sorted({arm for arm, _ in nets})
        for (arm, seed) in nets:
            cost.append(json.loads((runs / name / f'{arm}-{seed}.json').read_text()))
        tops = {key: top_sets(net, tfs) for key, net in nets.items()}
        for arm in present:
            pairs = [mean_jaccard(tops[(arm, a)], tops[(arm, b)], tfs) for a, b in itertools.combinations(seeds, 2)
                     if (arm, a) in tops and (arm, b) in tops]
            stability.append({'dataset': name, 'arm': arm, 'seed_pairs': len(pairs), 'top50_jaccard': float(np.mean(pairs)) if pairs else np.nan,
                              'min_pair': float(np.min(pairs)) if pairs else np.nan, 'max_pair': float(np.max(pairs)) if pairs else np.nan})
        for native, reference in (('native_cross', 'reference_cross'), ('native_same', 'reference_same')):
            if native in present and reference in present:
                between = [mean_jaccard(tops[(native, a)], tops[(reference, b)], tfs) for a in seeds for b in seeds
                           if (native, a) in tops and (reference, b) in tops]
                within = [mean_jaccard(tops[(reference, a)], tops[(reference, b)], tfs) for a, b in itertools.combinations(seeds, 2)
                          if (reference, a) in tops and (reference, b) in tops]
                fidelity.append({'dataset': name, 'pair': f'{native} vs {reference}', 'between_implementations': float(np.mean(between)),
                                 'within_reference_seeds': float(np.mean(within)), 'ratio': float(np.mean(between) / np.mean(within))})
        candidates = dict(nets)
        candidates[('pearson', 0)] = pearson_network(x, genes, tfs)
        done_refs = []
        for ref in info['references']:
            truth = pd.read_csv(inputs / name / ref['file'], sep='\t')
            label = ref['file'].replace('-network.tsv', '')
            blocks = {key: tf_blocks(net, truth, geneset, tfset) for key, net in candidates.items()}
            for key, net in candidates.items():
                m = reference_metrics(net, truth, genes, tfs)
                recovery.append({'dataset': name, 'reference': label, 'reference_type': ref['reference_type'], 'arm': key[0], 'seed': key[1],
                                 'AP': m['average_precision'], 'prevalence': m['reference_prevalence'],
                                 'AP_over_prevalence': m['ap_over_prevalence'], 'top50_fold_over_uniform': m['top50_fold_over_uniform'],
                                 'reference_TFs': m['reference_covered_tfs'], 'eligible_links': m['eligible_reference_links']})
            if len(done_refs) >= bootstrap_references:
                continue
            done_refs.append(label)
            rng = np.random.default_rng(SEED)
            supported = sorted(next(iter(blocks.values())))
            draws = [list(rng.choice(supported, size=len(supported), replace=True)) for _ in range(BOOT)]
            global _BLOCKS
            _BLOCKS = blocks
            keys = sorted(blocks)
            with get_context('fork').Pool(jobs) as pool:  # blocks are inherited, never pickled
                table = np.array(pool.map(_draw_aps, [(keys, d) for d in draws], chunksize=25))
            column = {k: i for i, k in enumerate(keys)}
            point_ap = {k: pooled_ap(blocks[k], supported) for k in keys}
            for a, b in (('native_same', 'native_cross'), ('reference_same', 'reference_cross'), ('native_same', 'pearson'),
                         ('native_cross', 'pearson'), ('native_same', 'reference_same'), ('native_cross', 'reference_cross')):
                ka = [k for k in keys if k[0] == a]
                kb = [k for k in keys if k[0] == b]
                if not ka or not kb:
                    continue
                point = np.mean([point_ap[k] for k in ka]) - np.mean([point_ap[k] for k in kb])
                boot = table[:, [column[k] for k in ka]].mean(1) - table[:, [column[k] for k in kb]].mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                differences.append({'dataset': name, 'reference': label, 'comparison': f'{a} minus {b}', 'AP_difference': float(point),
                                    'low': float(lo), 'high': float(hi), 'reference_TFs': len(supported)})
        print('scored', name, flush=True)
    rec = pd.DataFrame(recovery)
    rec.to_csv(out / 'recovery-per-seed.csv', index=False)
    summary = rec.groupby(['dataset', 'reference', 'reference_type', 'arm'], sort=False).agg(
        AP=('AP', 'mean'), AP_over_prevalence=('AP_over_prevalence', 'mean'), top50_fold_over_uniform=('top50_fold_over_uniform', 'mean'),
        reference_TFs=('reference_TFs', 'first'), eligible_links=('eligible_links', 'first')).reset_index()
    summary.to_csv(out / 'recovery.csv', index=False)
    pd.DataFrame(stability).to_csv(out / 'stability.csv', index=False)
    pd.DataFrame(fidelity).to_csv(out / 'fidelity.csv', index=False)
    pd.DataFrame(differences).to_csv(out / 'ap-differences.csv', index=False)
    c = pd.DataFrame(cost)
    c.to_csv(out / 'cost-per-run.csv', index=False)
    if len(c):
        c.groupby(['dataset', 'arm']).agg(seconds=('seconds', 'mean'), trees_mean=('trees_mean', 'mean'), trees_median=('trees_median', 'mean'),
                                          edges=('edges', 'mean')).reset_index().to_csv(out / 'cost.csv', index=False)
    return summary, pd.DataFrame(stability), pd.DataFrame(fidelity), pd.DataFrame(differences)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inputs', type=Path)
    p.add_argument('runs', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--datasets', required=True)
    p.add_argument('--arms', default='native_cross,native_same,reference_cross,reference_same')
    p.add_argument('--seeds', default='777,778,779')
    p.add_argument('--bootstrap-references', type=int, default=99, help='bootstrap only the first N references per dataset')
    p.add_argument('--jobs', type=int, default=8)
    a = p.parse_args()
    frames = main(a.inputs, a.runs, a.datasets.split(','), a.arms.split(','), [int(s) for s in a.seeds.split(',')], a.out,
                  a.bootstrap_references, a.jobs)
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    for frame in frames:
        print(frame.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print()
