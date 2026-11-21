"""Score fitted networks against the saved calibrated SCEPTRE labels.

Reports, per training population:
  * the original endpoint (all responses, equal-TF macro AP);
  * the same endpoint on TF-specific responses (genes responding to more than
    `cap` of the OTHER knockdowns are removed from the positives and universe);
  * final-regulon coverage, emitted precision and enrichment over prevalence.
Development analysis on inspected K562/RPE1 data.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECAL = ROOT / 'RustScenic-Biology-Recalibration-20261112'
PACKAGE = ROOT / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
RETURNED = ROOT / 'RustScenic-Biology-Confirm-Results-Audit-20261112/returned'
CAPS = (None, 0.05)


def labels(dataset):
    panel = json.loads((PACKAGE / f'inputs/{dataset}/panel.json').read_text())
    genes = np.array(panel['genes'])
    primary = (RECAL / f'inputs/{dataset}/primary-tfs.txt').read_text().split()
    pairs = pd.read_parquet(RECAL / f'results/{dataset}/corrected-pair-results.parquet')
    pairs = pairs[pairs.grna_target.isin(primary)]
    tfs = np.array(sorted(pairs.grna_target.unique()))
    wide = lambda column, fill: (pairs.pivot(index='grna_target', columns='response_id', values=column)
                                 .reindex(index=tfs, columns=genes).fillna(fill).to_numpy())
    eligible = wide('eligible', False).astype(bool)
    response = wide('response', False).astype(bool) & eligible
    direction = wide('response_direction', 0).astype(int)
    loo = (response.sum(0, keepdims=True) - response) / np.maximum(1, eligible.sum(0, keepdims=True) - eligible)
    return dict(genes=genes, tfs=tfs, eligible=eligible, response=response, direction=direction, loo=loo, panel=panel)


def view(lab, cap):
    if cap is None:
        return lab['response'], lab['eligible']
    specific = lab['response'] & (lab['loo'] <= cap)
    return specific, lab['eligible'] & ~(lab['response'] & ~specific)


def per_tf_ap(y, eligible, score):
    out = np.full(y.shape[0], np.nan)
    for i in range(y.shape[0]):
        s = eligible[i]
        if 0 < y[i, s].sum() < s.sum():
            out[i] = average_precision_score(y[i, s], score[i, s])
    return out


def load_fit(job, lab):
    genes, tfs = lab['genes'], lab['tfs']
    job = Path(job)
    native = job / 'native' if (job / 'native').exists() else job
    net = pd.read_parquet(native / 'network.parquet')
    net = net[net.TF.isin(tfs)]
    raw = net.pivot(index='TF', columns='target', values='importance').reindex(index=tfs, columns=genes).fillna(0).to_numpy()
    with np.load(native / 'pearson.npz') as z:
        rho = pd.DataFrame(z['rho'], index=z['TFs'], columns=z['genes']).reindex(index=tfs, columns=genes).to_numpy()
    regs = {name: json.loads((native / f'{name}.json').read_text()) for name in ('original', 'diverse_importance')
            if (native / f'{name}.json').exists()}
    if (job / 'reference/reference.json').exists():
        regs['reference'] = json.loads((job / 'reference/reference.json').read_text())
    member = {}
    for name, reg in regs.items():
        m = np.zeros(raw.shape, bool)
        for i, tf in enumerate(tfs):
            targets = (set(reg.get(tf + '_activator', [])) | set(reg.get(tf + '_repressor', []))) - {tf}
            m[i] = np.isin(genes, list(targets))
        member[name] = m
    return raw, rho, member


def score(dataset, job, lab=None):
    lab = lab or labels(dataset)
    raw, rho, member = load_fit(job, lab)
    rows = []
    for cap in CAPS:
        y, eligible = view(lab, cap)
        evaluable = np.array([0 < y[i, eligible[i]].sum() < eligible[i].sum() for i in range(len(y))])
        prevalence = np.array([y[i, eligible[i]].mean() if evaluable[i] else np.nan for i in range(len(y))])
        row = {'labels': 'all' if cap is None else f'specific<={cap}', 'TFs': int(evaluable.sum()),
               'prevalence': np.nanmean(prevalence),
               'AP_generic_LOO': np.nanmean(per_tf_ap(y, eligible, lab['loo'])),
               'AP_pearson': np.nanmean(per_tf_ap(y, eligible, np.abs(rho))),
               'AP_raw_importance': np.nanmean(per_tf_ap(y, eligible, raw))}
        # top-50 per TF by each raw score: precision and enrichment over prevalence
        for name, s in (('pearson', np.abs(rho)), ('raw_importance', raw)):
            hits = total = 0
            for i in np.flatnonzero(evaluable):
                sel = np.flatnonzero(eligible[i] & (s[i] > 0))
                top = sel[np.argsort(-s[i, sel], kind='stable')[:50]]
                hits += y[i, top].sum(); total += len(top)
            row[f'top50_precision_{name}'] = hits / max(1, total)
        for name, m in member.items():
            m = m & eligible
            with_reg = m.any(1)
            hits = (m & y)[evaluable].sum(); total = m[evaluable].sum()
            # matched expectation: prevalence of the same TFs, weighted by emitted targets
            expected = np.nansum(prevalence[evaluable] * m[evaluable].sum(1))
            row[f'{name}_TFs_with_regulon'] = int((with_reg & evaluable).sum())
            row[f'{name}_targets'] = int(total)
            row[f'{name}_precision'] = hits / total if total else np.nan
            row[f'{name}_enrichment'] = hits / expected if expected else np.nan
            final = np.where(m, np.where(raw > 0, raw, 1e-12), 0)
            row[f'AP_{name}'] = np.nanmean(per_tf_ap(y, eligible, final))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    dataset = sys.argv[1]
    lab = labels(dataset)
    frames = []
    for job in sys.argv[2:]:
        f = score(dataset, job, lab)
        f.insert(0, 'fit', Path(job).name)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    pd.set_option('display.width', 250, 'display.max_columns', 50)
    print(out.set_index(['fit', 'labels']).T.to_string(float_format=lambda v: f'{v:.4g}'))
