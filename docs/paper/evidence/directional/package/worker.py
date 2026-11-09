"""One independent biology check or one atlas scoring arm; no network fitting."""
import argparse
import os
from pathlib import Path
import traceback
import h5py
import numpy as np
import pandas as pd
from common import ROOT, digest, fingerprint, read_json, write_json
from matrix_io import csr_slice, strings
from monitoring import environment
from polarity import signed_contrast
from scoring import score_block, score_file, signatures, coverage
from utility import evaluate


def verify_engine():
    import rustscenic
    expected = read_json(ROOT/'build.json')['wheels']['candidate']
    module = Path(rustscenic.__file__).parent
    if (module.resolve() != (ROOT/'work/engines/candidate/rustscenic').resolve()
            or digest(module/Path(expected['extension']).name) != expected['extension_sha256']):
        raise RuntimeError('Wrong native engine loaded: '+str(module))


def difference(frame, expected):
    if not frame.index.equals(expected.index) or not frame.columns.equals(expected.columns):
        raise ValueError('Activity cell or signature axes changed')
    delta = np.abs(frame.to_numpy(dtype=np.float64)-expected.to_numpy(dtype=np.float64))
    if not np.isfinite(delta).all(): raise ValueError('Nonfinite comparison')
    return float(delta.max(initial=0))


def biology(out):
    with h5py.File(ROOT/'external/rna.h5') as h:
        genes,cells = strings(h['features'][:]),strings(h['cells'][:])
        x = csr_slice(h,0,len(cells))
    meta = pd.read_csv(ROOT/'external/cell-metadata.tsv',sep='\t',dtype={'donor':str}).set_index('cell').loc[cells]
    records = []
    for n in (5000,20000):
        for seed in (777,778,779):
            pair = ROOT/'networks'/f'bmmc-train{n}-seed{seed}'
            common = read_json(pair/'common-tfs.json')
            names = [tf+suffix for tf in common for suffix in ('_activator','_repressor')]
            for method in ('rustscenic','reference'):
                edge_path = pair/f'{method}.parquet'
                sets = signatures(pd.read_parquet(edge_path),method,'directional')
                sets = {name:sets[name] for name in names if name in sets}
                scored = pd.concat([score_block(x[i:i+2048],genes,cells[i:i+2048],sets)
                                    for i in range(0,len(cells),2048)])
                scored = scored.reindex(columns=names,fill_value=0.).astype(np.float32)
                expected = pd.read_parquet(ROOT/'expected'/f'{n}-{seed}'/f'{method}-directional.parquet')
                error = difference(scored,expected)
                contrast = signed_contrast(scored)[common]
                old_error = None
                if method == 'rustscenic':
                    old = pd.read_parquet(ROOT/'expected'/f'{n}-{seed}'/'rustscenic-hpc-contrast.parquet')[common]
                    old_error = difference(contrast,old)
                results = {mode:evaluate(frame,meta,method+'_'+mode)
                           for mode,frame in (('contrast',contrast),('directional',scored))}
                record = dict(training_cells=n,seed=seed,method=method,common_tfs=common,
                    network_sha256=digest(edge_path),signature_sha256=fingerprint(sets),coverage=coverage(sets,genes),
                    padded_columns=len(names),nonconstant_features=int(np.count_nonzero(np.ptp(scored.to_numpy(),axis=0))),
                    max_absolute_error=error,previous_hpc_contrast_error=old_error,
                    passed=error<=1e-7 and (old_error is None or old_error<=1e-7),results=results,
                    classifier_converged=all(row['status']=='complete' and row['converged']
                        for result in results.values() for row in result['folds']))
                records.append(record)
                scored.to_parquet(out/f'{n}-{seed}-{method}-directional.parquet')
                write_json(out/'verification.json',dict(records=records,numerical_pass=all(r['passed'] for r in records)))


def main(a):
    verify_engine()
    a.out.mkdir(parents=True,exist_ok=True)
    write_json(a.out/'environment.json',environment())
    write_json(a.out/'phase.json',{'phase':'analysis'})
    if a.kind == 'biology':
        biology(a.out)
    else:
        path = ROOT/'networks'/f'hdma-n817740-seed{a.seed}'/'rustscenic.parquet'
        score_file(a.matrix,pd.read_parquet(path),a.arm,a.out)
        write_json(a.out/'network.json',{'sha256':digest(path),'inference_seed':a.seed})
        if a.arm == 'directional' and not a.smoke:
            sample = signed_contrast(pd.read_parquet(a.out/'sampled-scores.parquet'))
            previous = pd.read_parquet(ROOT/'expected'/f'atlas-{a.seed}'/'contrast-sample.parquet')
            error = difference(sample[previous.columns],previous)
            write_json(a.out/'prior-sample-check.json',{'max_absolute_error':error,'passed':error<=1e-7})
            if error>1e-7: raise ValueError('Atlas sample differs from previously confirmed scoring')
    outputs = {p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in a.out.iterdir()
        if p.is_file() and p.suffix in ('.json','.h5','.parquet') and p.name not in ('complete.json','monitor.json','phase.json')}
    write_json(a.out/'complete.json',dict(status='complete',smoke=a.smoke,
        package=digest(ROOT/'SHA256SUMS'),outputs=outputs))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind',choices=['biology','atlas'],required=True)
    p.add_argument('--arm',choices=['union','directional'],required=True)
    p.add_argument('--seed',type=int,default=777)
    p.add_argument('--matrix',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a = p.parse_args()
    if not a.smoke and not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Use start.sh on the login node')
    try: main(a)
    except BaseException:
        write_json(a.out/'failure.json',{'traceback':traceback.format_exc()})
        raise
