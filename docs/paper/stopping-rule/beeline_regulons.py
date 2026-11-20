"""Motif-supported regulons from each BEELINE network, and their seed stability.

One common downstream path for every arm (so only the GRN stage differs): the
12 November package's standard SCENIC module families and native cisTarget
pruning (recipe.refine, 'diverse_importance'), with unmodified ranks, the full
motif collection and the original rank universe.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
sys.path.insert(0, str(PACKAGE))
from recipe import refine  # noqa: E402
from rustscenic import grn  # noqa: E402

CACHE = Path.home() / '.cache/rustscenic/cistarget'
RANKINGS = {'human': CACHE / 'hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather',
            'mouse': CACHE / 'mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'}
ANNOTATIONS = {s: ROOT / f'RustScenic-Final-Scaling-20261028/rustscenic-final-scaling/inputs/motifs/{s}-gene-annotations.tbl'
               for s in ('human', 'mouse')}


def resources(species, genes):
    """Project ranks onto the panel by unique uppercase name; never re-rank."""
    schema = pa.ipc.open_file(pa.memory_map(str(RANKINGS[species]), 'r')).schema
    by_upper = {}
    for name in schema.names:
        if name != 'motifs':
            by_upper.setdefault(name.upper(), []).append(name)
    unique = {u: v[0] for u, v in by_upper.items() if len(v) == 1}
    keep = [g for g in genes if g in unique]
    table = feather.read_table(RANKINGS[species], columns=[unique[g] for g in keep] + ['motifs']).rename_columns(keep + ['motifs'])
    rankings = table.to_pandas().set_index('motifs')
    annot = pd.read_csv(ANNOTATIONS[species], sep='\t')
    annot = annot[(annot.motif_similarity_qvalue <= .001) & (annot.orthologous_identity >= 0)]
    annot = pd.DataFrame({'motif': annot['#motif_id'], 'TF': annot.gene_name.str.upper()}).drop_duplicates()
    return rankings, annot, len(schema.names) - 1


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


def agreement(a, b):
    shared = sorted(set(a) & set(b))
    return jaccard(set(a), set(b)), (float(np.mean([jaccard(set(a[k]), set(b[k])) for k in shared])) if shared else np.nan), len(shared)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inputs', type=Path)
    p.add_argument('runs', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--datasets', required=True)
    p.add_argument('--arms', default='native_cross,native_same,reference_cross,reference_same')
    p.add_argument('--seeds', default='777,778,779')
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    a.out.mkdir(parents=True, exist_ok=True)
    rows, counts = [], []
    for name in a.datasets.split(','):
        info = json.loads((a.inputs / name / 'input.json').read_text())
        genes = info['genes']
        x = sparse.load_npz(a.inputs / name / 'matrix.npz').tocsr()
        rankings, annot, universe = resources(info['species'], genes)
        regs = {}
        for arm in a.arms.split(','):
            for seed in seeds:
                path = a.runs / name / f'{arm}-{seed}.parquet'
                if not path.exists():
                    continue
                saved = a.runs / name / f'{arm}-{seed}.regulons.json'
                if saved.exists():
                    regs[(arm, seed)] = json.loads(saved.read_text())
                else:
                    net = grn.add_correlation(pd.read_parquet(path), (x, genes), rho_threshold=.03)
                    regulons, _, _, _ = refine(net, rankings, annot, universe, 'diverse_importance')
                    saved.write_text(json.dumps(regulons) + '\n')
                    regs[(arm, seed)] = regulons
                r = regs[(arm, seed)]
                counts.append({'dataset': name, 'arm': arm, 'seed': seed, 'regulons': len(r),
                               'TFs': len({k.rsplit('_', 1)[0] for k in r}), 'median_targets': float(np.median([len(v) for v in r.values()])) if r else np.nan})
        arms = sorted({arm for arm, _ in regs})
        for arm in arms:
            vals = [agreement(regs[(arm, s)], regs[(arm, t)]) for s, t in itertools.combinations(seeds, 2) if (arm, s) in regs and (arm, t) in regs]
            if vals:
                rows.append({'dataset': name, 'comparison': f'within {arm}', 'regulon_identity_jaccard': np.nanmean([v[0] for v in vals]),
                             'target_jaccard_shared': np.nanmean([v[1] for v in vals]), 'shared_regulons': np.mean([v[2] for v in vals])})
        for native, reference in (('native_cross', 'reference_cross'), ('native_same', 'reference_same')):
            vals = [agreement(regs[(native, s)], regs[(reference, t)]) for s in seeds for t in seeds if (native, s) in regs and (reference, t) in regs]
            if vals:
                rows.append({'dataset': name, 'comparison': f'{native} vs {reference}', 'regulon_identity_jaccard': np.nanmean([v[0] for v in vals]),
                             'target_jaccard_shared': np.nanmean([v[1] for v in vals]), 'shared_regulons': np.mean([v[2] for v in vals])})
        print('regulons', name, flush=True)
    c = pd.DataFrame(counts)
    c.to_csv(a.out / 'regulon-counts-per-seed.csv', index=False)
    summary = c.groupby(['dataset', 'arm'], sort=False).mean(numeric_only=True).drop(columns='seed').reset_index()
    summary.to_csv(a.out / 'regulon-counts.csv', index=False)
    stab = pd.DataFrame(rows)
    stab.to_csv(a.out / 'regulon-stability.csv', index=False)
    pd.set_option('display.width', 220, 'display.max_rows', 300)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(stab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
