"""Held-out-donor lineage activity for frozen regulons (bone marrow, GSE194122).

For each prespecified regulator and lineage (lineage_regulators.json, written
12 November before any standard-configuration result existed):
  * whether a motif-supported activating regulon exists (coverage);
  * AUROC of its AUCell activity for the expected lineage against all other
    cells, per held-out donor, with the regulator's own gene removed;
  * the same AUROC for the regulator's expression alone and for 20 size-matched
    random gene sets (controls).
Networks never see the three held-out donors. One seeded scorer for every arm.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

SPEC = json.loads((Path(__file__).resolve().parent / 'lineage_regulators.json').read_text())
PAIRS = [(tf, group, tier) for tier in ('primary', 'extended') for tf, group in SPEC[tier].items()]


def auroc(score, positive):
    """Tie-aware area under the ROC curve (Mann-Whitney), identical to scikit-learn's."""
    ranks = rankdata(score)
    n = int(positive.sum())
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * (len(score) - n)))


def donor_auroc(score, positive, donors):
    values = [auroc(score[donors == d], positive[donors == d]) for d in np.unique(donors)
              if 0 < positive[donors == d].sum() < (donors == d).sum()]
    return (float(np.mean(values)) if values else np.nan), len(values)


def evaluate(inputs, regulon_sets, n_random=20, seed=20261112):
    from rustscenic import aucell, validated
    inputs = Path(inputs)
    genes = np.array(json.loads((inputs / 'panel.json').read_text())['genes'])
    x = sparse.load_npz(inputs / 'evaluation-expression.npz').tocsr()
    meta = pd.read_csv(inputs / 'evaluation-cells.tsv', sep='\t', dtype=str)
    adata = ad.AnnData(x, obs=meta.set_index('barcode'), var=pd.DataFrame(index=genes))
    donors, cell_type = meta.donor.to_numpy(), meta.cell_type.to_numpy()
    lookup = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)
    rows = []
    for label, regulons in regulon_sets.items():
        signatures = {tf: members for tf, _, _ in PAIRS
                      if len(members := sorted(set(regulons.get(tf + '_activator', [])) - {tf})) >= 5}
        randoms = {f'{tf}#r{i}': list(rng.choice(genes, size=len(m), replace=False))
                   for tf, m in signatures.items() for i in range(n_random)}
        scores = (aucell.score(adata, {**signatures, **randoms}, top_frac=.05, seed=validated.SCORING_SEED)
                  if signatures else pd.DataFrame())
        for tf, group, tier in PAIRS:
            positive = np.isin(cell_type, SPEC['groups'][group])
            row = {'regulons': label, 'TF': tf, 'lineage': group, 'tier': tier, 'in_panel': tf in lookup,
                   'regulon_present': tf in signatures, 'targets': len(signatures.get(tf, [])), 'all_regulons': len(regulons)}
            if tf in lookup:
                row['expression_auroc'], _ = donor_auroc(np.asarray(x[:, lookup[tf]].todense()).ravel(), positive, donors)
            if tf in signatures:
                row['activity_auroc'], row['donors'] = donor_auroc(scores[tf].to_numpy(), positive, donors)
                row['random_auroc'] = float(np.mean([donor_auroc(scores[f'{tf}#r{i}'].to_numpy(), positive, donors)[0]
                                                     for i in range(n_random)]))
            rows.append(row)
    return pd.DataFrame(rows)
