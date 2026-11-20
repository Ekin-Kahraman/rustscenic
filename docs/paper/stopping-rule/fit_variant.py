"""Run the frozen package's native fit with a different GRN stopping rule.

Everything except the GRN settings is the unmodified 12 November package code
(correlation, module families, motif pruning, Pearson). Needs a RustScenic build
that exposes early_stop_mode="paired_oob".
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
sys.path.insert(0, str(PACKAGE))


def subset_inputs(inputs, cells, seed=20261112):
    """Seeded random training-cell subset in a sibling inputs directory."""
    out = inputs.with_name(f'{inputs.name}-c{cells}')
    if not (out / 'training.npz').exists():
        out.mkdir(parents=True, exist_ok=True)
        x = sparse.load_npz(inputs / 'training.npz').tocsr()
        rows = np.sort(np.random.default_rng(seed).choice(x.shape[0], size=cells, replace=False))
        sparse.save_npz(out / 'training.npz', x[rows])
        np.savetxt(out / 'training-rows.txt', rows, fmt='%d')
        for name in ('panel.json', 'motifs.feather', 'motifs.json', 'provenance.json',
                     'evaluation-expression.npz', 'evaluation-cells.tsv'):
            if (inputs / name).exists() and not (out / name).exists():
                os.symlink((inputs / name).resolve(), out / name)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inputs', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--mode', default='arboreto')
    p.add_argument('--window', type=int, default=25)
    p.add_argument('--trees', type=int, default=5000)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--cells', type=int)
    p.add_argument('--threads', type=int, default=10)
    a = p.parse_args()
    from rustscenic import validated
    validated.GRN_SETTINGS = {**validated.GRN_SETTINGS, 'early_stop_mode': a.mode, 'early_stop_window': a.window,
                              'n_estimators': a.trees, 'learning_rate': a.lr}
    import fit
    inputs = subset_inputs(a.inputs.resolve(), a.cells) if a.cells else a.inputs.resolve()
    fit.fit('bmmc', a.seed, 'native', a.out.resolve(), a.threads, inputs)
    import json
    import pandas as pd
    net = pd.read_parquet(a.out / 'network.parquet')
    trees = net.groupby('target').importance.sum()
    frozen = json.loads((a.out / 'PREDICTIONS-FROZEN.json').read_text())
    summary = {'settings': validated.GRN_SETTINGS, 'cells': frozen['cells'], 'inference_s': frozen['inference_s'],
               'trees_mean': float(trees.mean()), 'trees_median': float(trees.median()),
               'trees_p90': float(trees.quantile(.9)), 'trees_max': float(trees.max()),
               'share_at_ceiling': float((trees >= a.trees - .5).mean()), 'edges': len(net)}
    (a.out / 'variant.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(summary)
