"""Held-out-donor lineage activity for frozen regulons (BMMC, development data).

For each prespecified regulator -> lineage pair (lineage_regulators.json):
  * is a motif-supported activating regulon produced at all (coverage);
  * AUROC of its AUCell activity for the expected lineage versus all other
    cells, per held-out donor, with the regulator's own gene removed;
  * the same AUROC for the regulator's expression alone and for size-matched
    random gene sets (controls).
Networks never see the held-out donors. One common seeded scorer for all methods.
"""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from rustscenic import aucell, validated

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'lineage_regulators.json').read_text())


def donor_auroc(score, positive, donors):
    values = []
    for d in np.unique(donors):
        m = donors == d
        if 0 < positive[m].sum() < m.sum():
            values.append(roc_auc_score(positive[m], score[m]))
    return float(np.mean(values)) if values else np.nan, len(values)


def evaluate(inputs, regulon_files, n_random=20, seed=20261112):
    inputs = Path(inputs)
    panel = json.loads((inputs / 'panel.json').read_text())
    genes = np.array(panel['genes'])
    x = sparse.load_npz(inputs / 'evaluation-expression.npz').tocsr()
    meta = pd.read_csv(inputs / 'evaluation-cells.tsv', sep='\t', dtype=str)
    adata = ad.AnnData(x, obs=meta.set_index('barcode'), var=pd.DataFrame(index=genes))
    donors = meta.donor.to_numpy()
    cell_type = meta.cell_type.to_numpy()
    pairs = [(tf, g, 'primary') for tf, g in SPEC['primary'].items()] + [(tf, g, 'extended') for tf, g in SPEC['extended'].items()]
    lookup = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)
    rows = []
    for label, path in regulon_files.items():
        regulons = json.loads(Path(path).read_text())
        signatures = {}
        for tf, _, _ in pairs:
            members = sorted(set(regulons.get(tf + '_activator', [])) - {tf})
            if len(members) >= 5:
                signatures[tf] = members
        randoms = {f'{tf}#r{i}': list(rng.choice(genes, size=len(m), replace=False))
                   for tf, m in signatures.items() for i in range(n_random)}
        scores = aucell.score(adata, {**signatures, **randoms}, top_frac=.05, seed=validated.SCORING_SEED) if signatures else pd.DataFrame()
        for tf, group, tier in pairs:
            positive = np.isin(cell_type, SPEC['groups'][group])
            row = {'method': label, 'TF': tf, 'lineage': group, 'tier': tier, 'in_panel': tf in lookup,
                   'regulon_present': tf in signatures, 'targets': len(signatures.get(tf, [])),
                   'all_regulons': len(regulons)}
            if tf in lookup:
                row['expression_auroc'], _ = donor_auroc(np.asarray(x[:, lookup[tf]].todense()).ravel(), positive, donors)
            if tf in signatures:
                row['activity_auroc'], row['donors'] = donor_auroc(scores[tf].to_numpy(), positive, donors)
                row['random_auroc'] = float(np.mean([donor_auroc(scores[f'{tf}#r{i}'].to_numpy(), positive, donors)[0] for i in range(n_random)]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(frame):
    out = []
    for (method, tier), g in frame.groupby(['method', 'tier'], sort=False):
        present = g[g.regulon_present]
        out.append({'method': method, 'tier': tier, 'regulators': len(g), 'with_regulon': len(present),
                    'median_activity_auroc': present.activity_auroc.median(),
                    'auroc>=0.9': int((present.activity_auroc >= .9).sum()),
                    'auroc>=0.8': int((present.activity_auroc >= .8).sum()),
                    'median_random_auroc': present.random_auroc.median(),
                    'all_regulons': int(g.all_regulons.iloc[0])})
    return pd.DataFrame(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs')
    parser.add_argument('out')
    parser.add_argument('regulons', nargs='+', help='label=path.json')
    args = parser.parse_args()
    files = dict(item.split('=', 1) for item in args.regulons)
    frame = evaluate(args.inputs, files)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    print(summarise(frame).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(frame[frame.tier.eq('primary')].drop(columns=['tier', 'in_panel', 'all_regulons']).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
