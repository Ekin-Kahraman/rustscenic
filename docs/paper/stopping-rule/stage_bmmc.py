"""Stage BMMC (GSE194122) RNA for a standard-configuration SCENIC run.

Same donor split as the 6 November paper package: networks see training donors
only; three donors are held out for activity evaluation. Output directories use
the frozen package's input layout, so its unmodified fit.py can run on them.

Tiers
  panel5000 : all expressed listed TFs + highest-variance genes to 5,000
              (the restricted panel style used so far)
  full      : every gene detected in >=1% of training cells (standard practice)

BMMC outcomes were inspected in earlier work: this is development data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / 'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
SOURCE = ROOT / 'RustScenic-Final-Scaling-20261028/source-cache/bmmc-multiome.h5ad'
RANKINGS = Path.home() / '.cache/rustscenic/cistarget/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather'
TRAIN = ['13272', '28483', '18303', '11466', '19593', '15078', '28045']
TEST = ['10886', '12710', '16710']
sys.path.insert(0, str(PACKAGE))
from data_io import normalise  # noqa: E402


def hash_order(barcodes, prefix):
    keys = np.array([hashlib.sha256((prefix + b).encode()).digest() for b in barcodes], dtype='S32')
    return np.argsort(keys, kind='stable')


def main(train_cells, test_cells):
    data = ad.read_h5ad(SOURCE, backed='r')
    obs = data.obs
    gex = np.flatnonzero((data.var.feature_types == 'GEX').to_numpy())
    genes = data.var_names[gex].to_numpy(dtype=str)
    assert len(set(genes)) == len(genes)
    donor = obs.DonorID.astype(str).to_numpy()
    barcodes = obs.index.to_numpy(dtype=str)

    def pick(donors, n, tag):
        ids = np.flatnonzero(np.isin(donor, donors))
        ids = ids[hash_order(barcodes[ids], f'20261112:bmmc:{tag}:')][:n]
        return np.sort(ids)

    train, test = pick(TRAIN, train_cells, 'train'), pick(TEST, test_cells, 'test')
    assert not set(donor[train]) & set(donor[test])

    def counts(ids):
        x = sparse.csr_matrix(data.layers['counts'][ids])[:, gex].tocsr()
        assert np.all(x.data == np.floor(x.data)) and x.data.min() >= 0
        return x

    tx, ex = counts(train), counts(test)
    tn, en = normalise(tx), normalise(ex)
    detection = np.asarray(tx.getnnz(axis=0)).ravel() / tx.shape[0]
    mean = np.asarray(tn.mean(axis=0)).ravel()
    var = np.asarray(tn.power(2).mean(axis=0)).ravel() - mean ** 2
    valid = (detection >= .01) & (var > 0)
    tf_list = set((PACKAGE / 'resources/human-tfs.txt').read_text().split())
    schema = pa.ipc.open_file(pa.memory_map(str(RANKINGS), 'r')).schema
    ranked = set(schema.names) - {'motifs'}
    universe = len(schema.names) - 1
    tfs = sorted(set(genes[valid]) & tf_list)
    order = np.lexsort((genes, -var))
    panels = {
        'panel5000': tfs + [str(genes[i]) for i in order if valid[i] and genes[i] not in tf_list][:5000 - len(tfs)],
        'full': sorted(str(g) for g in genes[valid]),
    }
    index = {g: i for i, g in enumerate(genes)}
    meta = obs.iloc[test][['cell_type', 'DonorID', 'batch']].astype(str).rename(columns={'DonorID': 'donor'})
    meta.insert(0, 'barcode', barcodes[test])
    for tier, panel in panels.items():
        out = HERE / 'inputs' / f'bmmc-{tier}'
        out.mkdir(parents=True, exist_ok=True)
        cols = [index[g] for g in panel]
        sparse.save_npz(out / 'training.npz', tn[:, cols].tocsr())
        sparse.save_npz(out / 'evaluation-expression.npz', en[:, cols].tocsr())
        meta.to_csv(out / 'evaluation-cells.tsv', sep='\t', index=False)
        obs.iloc[train][['cell_type', 'DonorID']].astype(str).assign(barcode=barcodes[train]).to_csv(out / 'training-cells.tsv', sep='\t', index=False)
        in_db = [g for g in panel if g in ranked]
        table = feather.read_table(RANKINGS, columns=in_db + ['motifs'])
        feather.write_feather(table, out / 'motifs.feather', compression='zstd')
        (out / 'motifs.json').write_text(json.dumps({'original_rank_universe': universe, 'ranks_reranked_or_rescaled': False,
                                                     'source': RANKINGS.name}, indent=2) + '\n')
        (out / 'panel.json').write_text(json.dumps({'genes': panel, 'tfs': tfs, 'target_tfs': tfs}) + '\n')
        (out / 'provenance.json').write_text(json.dumps({'dataset': 'bmmc', 'tier': tier, 'source': 'GSE194122',
            'train_donors': TRAIN, 'test_donors': TEST, 'training_cells': len(train), 'evaluation_cells': len(test),
            'genes': len(panel), 'genes_with_motif_ranks': len(in_db), 'TF_predictors': len(tfs),
            'rank_universe': universe}, indent=2) + '\n')
        print(tier, 'genes', len(panel), 'ranked', len(in_db), 'TFs', len(tfs), 'train', len(train), 'test', len(test), flush=True)
    data.file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-cells', type=int, default=20000)
    parser.add_argument('--test-cells', type=int, default=10000)
    args = parser.parse_args()
    main(args.train_cells, args.test_cells)
