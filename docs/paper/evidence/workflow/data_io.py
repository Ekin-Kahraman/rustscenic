"""Bounded sparse I/O; real cells are sampled once, never duplicated."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def strings(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def h5_column(group, name):
    values = group[name][:]
    if '__categories' in group and name in group['__categories']:
        cats = np.array(strings(group['__categories'][name][:]), dtype=object)
        return np.array([cats[int(v)] if int(v) >= 0 else '' for v in values], dtype=object)
    if values.dtype.kind in 'SO':
        return np.array(strings(values), dtype=object)
    return values


def csr_slice(group, start, stop):
    shape = tuple(group.attrs['shape'])
    pointers = group['indptr'][start:stop + 1]
    lo, hi = int(pointers[0]), int(pointers[-1])
    return sparse.csr_matrix((group['data'][lo:hi], group['indices'][lo:hi], pointers - lo),
                             shape=(stop - start, shape[1]))


def normalise(x, totals=None):
    x = x.astype(np.float32).tocsr(copy=True)
    if totals is None:
        totals = np.asarray(x.sum(axis=1)).ravel()
    factors = np.divide(10000., totals, out=np.zeros(len(totals)), where=np.asarray(totals) > 0)
    x = sparse.diags(factors.astype(np.float32)).dot(x).tocsr()
    np.log1p(x.data, out=x.data)
    return x


def save_matrix(path, x, genes, cells):
    x = x.tocsr()
    with h5py.File(path, 'w') as f:
        f.attrs['shape'] = x.shape
        f.create_dataset('data', data=x.data, compression='gzip', compression_opts=1)
        f.create_dataset('indices', data=x.indices, compression='gzip', compression_opts=1)
        f.create_dataset('indptr', data=x.indptr)
        f.create_dataset('features', data=np.array(genes, dtype=h5py.string_dtype()))
        f.create_dataset('cells', data=np.array(cells, dtype=h5py.string_dtype()))


def load_matrix(path, count):
    with h5py.File(path) as f:
        if not 1 <= count <= int(f.attrs['shape'][0]):
            raise ValueError('Requested cell count exceeds real input')
        x = csr_slice(f, 0, count)
        return x, strings(f['features'][:]), strings(f['cells'][:count])


def gene_coordinates(gtf):
    import gzip
    import re
    rows = []
    with gzip.open(gtf, 'rt') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            if fields[2] != 'gene' or fields[0] not in ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']:
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]+)"', fields[8]))
            if attrs.get('gene_type') == 'protein_coding':
                rows.append({'ensembl': attrs['gene_id'].split('.')[0], 'gene': attrs['gene_name'],
                             'chrom': fields[0], 'tss': int(fields[3]) - 1 if fields[6] == '+' else int(fields[4]) - 1,
                             'strand': fields[6]})
    return pd.DataFrame(rows).drop_duplicates('ensembl')
