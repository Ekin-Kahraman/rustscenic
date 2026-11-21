"""Post hoc: rebuild the first N cells of the Minerva atlas input locally.

Follows prepare_atlas.prepare from the final-scaling package (same source, panel, per-cell
normalisation and cell order seed 20261028) but keeps only cells whose rank in that order
is below N, so the 1.3M-cell matrix is never held in memory. The result is read with
data_io.load_matrix exactly as fit.py reads recovery/work/prepared/atlas/rna.h5.

Usage: python rebuild_atlas.py FINAL_SCALING_PACKAGE SOURCE_H5 OUT_DIR N
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

PKG, SOURCE, OUT, N = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, str(PKG))
from data_io import strings, normalise, save_matrix  # noqa: E402

panel = json.loads((PKG / 'inputs/atlas-panel.json').read_text())
genes = panel['genes']
OUT.mkdir(parents=True, exist_ok=True)
with h5py.File(SOURCE) as f:
    group = next(g for g in f.values() if isinstance(g, h5py.Group) and 'indptr' in g)
    source_names = strings(group['gene_names'][:])
    n_genes, n_cells = map(int, group['shape'][:])
    barcodes = strings(group['barcodes'][:n_cells])
    gene_index = {g: i for i, g in enumerate(genes)}
    rows = [i for i, g in enumerate(source_names) if g in gene_index]
    projection = sparse.csr_matrix((np.ones(len(rows)), ([gene_index[source_names[i]] for i in rows], rows)),
                                   shape=(len(genes), n_genes))
    order = np.random.default_rng(20261028).permutation(n_cells)
    rank = np.empty(n_cells, dtype=np.int64)
    rank[order] = np.arange(n_cells)
    blocks, ranks = [], []
    for start in range(0, n_cells, 4096):
        stop = min(start + 4096, n_cells)
        keep = np.flatnonzero(rank[start:stop] < N)
        if len(keep) == 0:
            continue
        ptr = group['indptr'][start:stop + 1]
        lo, hi = int(ptr[0]), int(ptr[-1])
        block = sparse.csc_matrix((group['data'][lo:hi], group['indices'][lo:hi], ptr - lo), shape=(n_genes, stop - start))
        totals = np.asarray(block.sum(axis=0)).ravel()
        blocks.append(normalise((projection @ block).T.tocsr(), totals)[keep])
        ranks.append(rank[start:stop][keep])
        if start % 131072 == 0:
            print('Read source cells', stop, '/', n_cells, flush=True)
matrix = sparse.vstack(blocks, format='csr')
del blocks
r = np.concatenate(ranks)
o = np.argsort(r)
assert np.array_equal(r[o], np.arange(N))
save_matrix(OUT / 'rna.h5', matrix[o], genes, [barcodes[i] for i in order[:N]])
(OUT / 'tfs.txt').write_text('\n'.join(panel['tfs']) + '\n')
print('Wrote', N, 'cells x', len(genes), 'genes to', OUT / 'rna.h5', flush=True)
