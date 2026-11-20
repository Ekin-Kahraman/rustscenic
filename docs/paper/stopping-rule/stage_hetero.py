"""Stage alternative TRAINING populations on the frozen 5,000-gene panels.

The evaluation cells (held-out controls and every single-TF-knockdown cell) are
never read. Candidates are single-perturbation cells whose target is NOT one of
the evaluated regulators, so training and evaluation stay disjoint and the saved
SCEPTRE labels remain valid. Sampling is a deterministic barcode-hash order, so
smaller sets are nested in larger ones.

Development on inspected K562/RPE1 data, not confirmation.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
SOURCE = ROOT / 'RustScenic-Biology-Confirm-20261112/source-cache'
FILES = {'k562': 'ReplogleWeissman2022_K562_essential.h5ad', 'rpe1': 'ReplogleWeissman2022_rpe1.h5ad'}
sys.path.insert(0, str(PACKAGE))
from data_io import normalise  # noqa: E402


def stage(dataset, sizes):
    frozen = PACKAGE / 'inputs' / dataset
    panel = json.loads((frozen / 'panel.json').read_text())
    data = ad.read_h5ad(SOURCE / FILES[dataset], backed='r')
    obs = data.obs
    evaluation = set(pd.read_csv(frozen / 'evaluation-cells.tsv.gz', sep='\t').barcode)
    control_training = set(pd.read_csv(frozen / 'training-cells.tsv', sep='\t').barcode)
    targets = set(panel['target_tfs'])
    candidate = (obs.nperts.eq(1) & ~obs.perturbation.isin(targets) & obs.perturbation.ne('control')).to_numpy()
    barcodes = obs.index.to_numpy(dtype=str)
    assert not (set(barcodes[candidate]) & evaluation) and not (set(barcodes[candidate]) & control_training)
    ids = np.flatnonzero(candidate)
    key = [hashlib.sha256(f'20261112:{dataset}:hetero:{barcodes[i]}'.encode()).digest() for i in ids]
    ids = ids[np.argsort(np.array(key, dtype='S32'), kind='stable')]
    cols = data.var_names.get_indexer(panel['genes'])
    assert (cols >= 0).all()
    largest = max(sizes)
    chosen = ids[:largest]
    order = np.argsort(chosen)
    x = sparse.csr_matrix(data.X[chosen[order]])
    assert np.all(x.data == np.floor(x.data)) and np.all(x.data >= 0)
    totals = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    assert np.allclose(totals, obs.ncounts.to_numpy()[chosen[order]])
    normal = normalise(x, totals)[:, cols].tocsr()
    inverse = np.empty(largest, int)
    inverse[order] = np.arange(largest)  # row of the i-th cell in hash order
    for size in sizes:
        out = HERE / 'inputs' / f'{dataset}-hetero{size}'
        out.mkdir(parents=True, exist_ok=True)
        rows = np.sort(inverse[:size])
        sparse.save_npz(out / 'training.npz', normal[rows])
        cells = obs.iloc[chosen[order][rows]]
        pd.DataFrame({'barcode': cells.index.astype(str), 'batch': cells.batch.astype(int).to_numpy(),
                      'perturbation': cells.perturbation.astype(str).to_numpy()}).to_csv(out / 'training-cells.tsv', sep='\t', index=False)
        for name in ('panel.json', 'motifs.feather', 'motifs.json', 'provenance.json'):
            link = out / name
            if not link.exists():
                os.symlink(frozen / name, link)
        print(dataset, size, 'cells', len(rows), 'distinct perturbations', cells.perturbation.nunique(), flush=True)
    data.file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=FILES)
    parser.add_argument('sizes', type=int, nargs='+')
    args = parser.parse_args()
    stage(args.dataset, sorted(args.sizes))
