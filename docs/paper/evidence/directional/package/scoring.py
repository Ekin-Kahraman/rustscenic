"""Seeded native activity; bounded input/output, one selected representation."""
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from rustscenic.aucell import score
from common import digest, fingerprint, write_json
from matrix_io import csr_slice, validate_matrix
from polarity import signed_gene_sets

SEED = 20261106


def signatures(edges, method, mode):
    if mode == 'directional':
        return signed_gene_sets(edges, method)
    if mode == 'union':
        return {str(tf): sorted(set(group.target)) for tf, group in edges.groupby('tf', sort=True)}
    raise ValueError('Unknown activity mode')


def coverage(sets, genes):
    available = set(genes)
    return {name: len(set(targets) & available)/len(targets) for name, targets in sets.items()}


def score_block(x, genes, cells, sets):
    cov = coverage(sets, genes)
    eligible = {name: targets for name, targets in sets.items() if cov[name] >= .8}
    frame = score((x, genes), eligible, top_frac=.05, seed=SEED) if eligible else pd.DataFrame(index=range(len(cells)))
    frame.index = cells
    frame = frame.reindex(columns=list(sets), fill_value=0.).astype(np.float32)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Nonfinite activity')
    return frame


def score_file(matrix, edges, mode, out, block_size=4096):
    out = Path(out)
    genes, cells = validate_matrix(matrix)
    sets = signatures(edges, 'rustscenic', mode)
    names = sorted(sets)
    sets = {name: sets[name] for name in names}
    if not sets:
        raise ValueError('No signatures')
    cov = coverage(sets, genes)
    write_json(out/'signatures.json', sets)
    write_json(out/'coverage.json', cov)
    rows = np.sort(np.random.default_rng(20261028).choice(len(cells), min(10000,len(cells)), replace=False))
    samples = []
    nonzero = np.zeros(len(names), np.int64)
    sums = np.zeros(len(names), np.float64)
    times = {k: 0. for k in ('read_seconds','score_seconds','write_seconds')}
    final = out/'full-scores.h5'
    temporary = out/'full-scores.h5.tmp'
    write_json(out/'phase.json', {'phase':'analysis'})
    started = time.perf_counter()
    with h5py.File(matrix) as src, h5py.File(temporary,'w') as dst:
        target = dst.create_dataset('scores', shape=(len(cells),len(names)), dtype='f4',
                                    chunks=(min(block_size,len(cells)),len(names)))
        for key, values in (('cells',cells),('features',names)):
            dst.create_dataset(key,data=np.asarray(values,dtype=object),dtype=h5py.string_dtype())
        for start in range(0,len(cells),block_size):
            end = min(start+block_size,len(cells))
            t = time.perf_counter(); x = csr_slice(src,start,end)
            times['read_seconds'] += time.perf_counter()-t
            t = time.perf_counter(); frame = score_block(x,genes,cells[start:end],sets)
            times['score_seconds'] += time.perf_counter()-t
            values = frame.to_numpy(copy=False)
            t = time.perf_counter(); target[start:end] = values
            times['write_seconds'] += time.perf_counter()-t
            nonzero += np.count_nonzero(values,axis=0)
            sums += values.sum(axis=0,dtype=np.float64)
            take = rows[(rows>=start)&(rows<end)]-start
            if len(take): samples.append(frame.iloc[take])
    temporary.replace(final)
    elapsed = time.perf_counter()-started
    write_json(out/'phase.json', {'phase':'export'})
    pd.concat(samples).to_parquet(out/'sampled-scores.parquet')
    write_json(out/'score-summary.json', dict(mode=mode,cells=len(cells),genes=len(genes),
        features=len(names),input_signatures=len(sets),signature_sha256=fingerprint(sets),
        block_size=block_size,stage_elapsed_seconds=elapsed,timings=times,
        output_array_bytes=len(cells)*len(names)*4,full_scores_bytes=final.stat().st_size,
        full_scores_sha256=digest(final),
        per_signature={name:dict(nonzero=int(nonzero[i]),mean=float(sums[i]/len(cells))) for i,name in enumerate(names)}))
